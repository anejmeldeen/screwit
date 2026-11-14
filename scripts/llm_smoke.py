"""Check the configured LLM provider end to end with three small calls (costs a few cents).

    SCREWIT_LLM=openrouter SCREWIT_MODEL=anthropic/claude-sonnet-4.5 OPENROUTER_API_KEY=... \
        PYTHONPATH=backend backend/.venv/bin/python scripts/llm_smoke.py

1. a plain text reply; 2. an image plus a JSON reply validated like the perception calls;
3. a streamed tool call, the way the chat agent uses it. Never prints the key.
"""

from __future__ import annotations

import io
import sys
import time

from PIL import Image, ImageDraw
from pydantic import BaseModel

from screwit.llm import client, config
from screwit.perception import claude_vision as cv


class Shape(BaseModel):
    shape: str
    colour: str


def main() -> int:
    cfg = config()
    print(f"provider={cfg.provider} model={cfg.model}")
    cli = client()
    ok = True

    t = time.time()
    r = cli.messages.create(model=cfg.model, max_tokens=20,
                            messages=[{"role": "user", "content": "Reply with the word ready."}])
    text = "".join(getattr(b, "text", "") for b in r.content)
    print(f"1 text     {time.time() - t:5.1f}s  {text.strip()[:40]!r}")
    ok &= "ready" in text.lower()

    img = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(img).ellipse((40, 40, 160, 160), fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    t = time.time()
    try:
        s = cv.ask(Shape, "You describe images.", [cv.image_block_from_png(buf.getvalue()),
                                                   {"type": "text", "text": "Shape and colour?"}],
                   client=cli)
        print(f"2 image    {time.time() - t:5.1f}s  {s.model_dump()}")
        ok &= "red" in s.colour.lower()
    except cv.VisionCallFailed as e:
        print(f"2 image    FAILED {e}")
        ok = False

    tool = {"name": "measure", "description": "Measure a part by id.",
            "input_schema": {"type": "object", "properties": {"part_id": {"type": "string"}},
                             "required": ["part_id"]}}
    t = time.time()
    first = None
    with cli.messages.stream(model=cfg.model, max_tokens=200, tools=[tool],
                             messages=[{"role": "user", "content": "Measure part p3."}]) as st:
        for _ev in st:
            first = first or time.time() - t
        final = st.get_final_message()
    calls = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
    got = calls[0].input if calls else None
    print(f"3 stream   {time.time() - t:5.1f}s  first event {first or 0:.1f}s  tool_use={got}")
    ok &= bool(calls) and got.get("part_id") == "p3"

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
