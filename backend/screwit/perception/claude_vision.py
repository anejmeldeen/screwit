"""One structured Claude call with images, validated by a Pydantic model (SPEC 6.2 steps 2, 5, 6).

Used by the labeller and the scene builder when SCREWIT_LLM names a real provider. The reply must be
a single JSON object; it is validated, and one repair round is attempted with the validation error.
Anything else raises ``VisionCallFailed`` so the caller can fall back to its geometry-only answer.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

MAX_IMAGE_SIDE = 768
MAX_TOKENS = 1500


class VisionCallFailed(RuntimeError):
    pass


def image_block_from_path(path: Path) -> dict:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return _block("image/jpeg", buf.getvalue())


def image_block_from_png(data: bytes) -> dict:
    return _block("image/png", data)


def _block(media_type: str, data: bytes) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": base64.b64encode(data).decode("ascii")}}


def evenly_spaced(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def ask[M: BaseModel](model_cls: type[M], system: str, content: list[dict], *, client=None,
                      model: str | None = None) -> M:
    """Ask Claude for one JSON object matching ``model_cls``; one repair round on a bad reply.

    Validated replies are cached on disk when SCREWIT_LLM_CACHE=1 (run.sh sets it), keyed by the
    model, the reply schema, the system prompt and every content block (the images included), so
    re-running a scan of the same video reuses the answers: about 35-45 s of Claude calls skipped,
    and the same parts every time."""
    from screwit.llm import client as make_client
    from screwit.llm import config

    model = model or config().model
    schema = json.dumps(model_cls.model_json_schema())
    key = _cache_key(model, schema, system, content)
    cached = _cache_get(key, model_cls)
    if cached is not None:
        return cached
    cli = client or make_client()
    system = (f"{system}\n\nReply with exactly one JSON object and nothing else. It must validate "
              f"against this JSON schema:\n{schema}")
    messages = [{"role": "user", "content": content}]
    last_error = ""
    for _ in range(2):
        try:
            resp = cli.messages.create(model=model, max_tokens=MAX_TOKENS, system=system,
                                       messages=messages)
        except Exception as e:  # noqa: BLE001 - provider/network failure
            raise VisionCallFailed(f"{type(e).__name__}: {e}") from e
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", None) == "text")
        try:
            reply = model_cls.model_validate(_extract_json(text))
            _cache_put(key, reply)
            return reply
        except (ValueError, ValidationError) as e:
            last_error = str(e)[:800]
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": f"That reply did not validate: {last_error}\n"
                                            "Send the corrected JSON object only."}]
    raise VisionCallFailed(f"reply did not validate after a repair round: {last_error}")


# ------------------------------------------------------------------------------- reply cache

def _cache_dir() -> Path | None:
    """The reply cache directory, or None when caching is off. Opt-in (SCREWIT_LLM_CACHE=1, which
    run.sh sets) so tests and ad-hoc runs always exercise the provider path."""
    import os

    if os.environ.get("SCREWIT_LLM_CACHE", "0").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    base = os.environ.get("SCREWIT_DATA_DIR")
    if not base:
        return None
    d = Path(base) / "llm-cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def _cache_key(model: str, schema: str, system: str, content: list[dict]) -> str:
    import hashlib

    h = hashlib.sha256()
    for part in (model, schema, system, json.dumps(content, sort_keys=True)):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:40]


def _cache_get[M: BaseModel](key: str, model_cls: type[M]) -> M | None:
    d = _cache_dir()
    if d is None:
        return None
    p = d / f"{key}.json"
    try:
        return model_cls.model_validate_json(p.read_text()) if p.exists() else None
    except (OSError, ValueError, ValidationError):
        return None   # a stale or corrupt entry is just a miss


def _cache_put(key: str, reply: BaseModel) -> None:
    d = _cache_dir()
    if d is None:
        return
    try:
        tmp = d / f"{key}.json.tmp"
        tmp.write_text(reply.model_dump_json())
        tmp.replace(d / f"{key}.json")   # atomic: never a half-written entry
    except OSError:
        pass
