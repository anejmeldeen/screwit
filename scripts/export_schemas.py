"""Export the Pydantic contracts as JSON Schema for the web app's generated TypeScript types.

    backend/.venv/bin/python scripts/export_schemas.py   # writes fixtures/schema/*.json
    (cd web && npm run gen:types)                        # schema -> web/lib/contracts.gen.ts

tests/test_fixtures.py fails if the committed schemas are out of date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import TypeAdapter  # noqa: E402

from screwit.contracts import SCHEMA_ROOTS, ChatEvent  # noqa: E402

OUT = ROOT / "fixtures" / "schema"


def build() -> dict[Path, str]:
    out = {}
    for name, model in SCHEMA_ROOTS.items():
        out[OUT / f"{name}.json"] = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
    chat = TypeAdapter(ChatEvent).json_schema()
    chat["title"] = "ChatEvent"
    out[OUT / "ChatEvent.json"] = json.dumps(chat, indent=2, sort_keys=True) + "\n"
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in build().items():
        path.write_text(text)
    print(f"wrote {len(build())} schemas to {OUT.relative_to(ROOT)}")
