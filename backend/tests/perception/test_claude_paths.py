"""Real-provider perception paths with a stubbed Messages client (no network).

With SCREWIT_LLM set to a real provider, naming, labelling and the scene graph go to Claude; a bad or
failed reply must fall back to the generic answer instead of failing the scan.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from screwit.perception.labels import Labeller, PartLabel, PartNames


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)])


@pytest.fixture
def frames(tmp_path):
    paths = []
    for i in range(10):
        p = tmp_path / f"{i:04d}.jpg"
        Image.fromarray(np.full((60, 80, 3), i * 20, np.uint8)).save(p)
        paths.append(p)
    return paths


def _use(monkeypatch, stub):
    monkeypatch.setenv("SCREWIT_LLM", "openrouter")
    monkeypatch.setenv("SCREWIT_MODEL", "anthropic/claude-test")
    monkeypatch.setattr("screwit.llm.client", lambda: stub)


def test_name_parts_sends_six_frames_and_validates(monkeypatch, frames):
    stub = StubClient([json.dumps({"names": ["faucet spout", "handle"]})])
    _use(monkeypatch, stub)
    names = Labeller("openrouter").name_parts("scn_x", frames)
    assert names == PartNames(names=["faucet spout", "handle"])
    blocks = stub.calls[0]["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in blocks) == 6
    assert stub.calls[0]["model"] == "anthropic/claude-test"


def test_bad_reply_gets_one_repair_round(monkeypatch, frames):
    stub = StubClient(["sure! here are the parts", json.dumps({"names": ["lid"]})])
    _use(monkeypatch, stub)
    assert Labeller("openrouter").name_parts("scn_x", frames).names == ["lid"]
    assert len(stub.calls) == 2
    assert "did not validate" in stub.calls[1]["messages"][-1]["content"]


def test_provider_failure_falls_back_instead_of_failing_the_scan(monkeypatch, frames):
    stub = StubClient([RuntimeError("401 bad key"), RuntimeError("401 bad key")])
    _use(monkeypatch, stub)
    lab = Labeller("openrouter")
    assert lab.name_parts("scn_x", frames).names == ["part"]
    label = lab.label_part("scn_x", "hinge", b"\x89PNG fake")
    assert label.label == "hinge" and label.label_confidence == 0.5


def test_label_part_uses_the_thumbnail(monkeypatch):
    reply = PartLabel(label="hinge", label_confidence=0.8, description="lets the lid swing",
                      material="steel", material_confidence=0.6)
    stub = StubClient([reply.model_dump_json()])
    _use(monkeypatch, stub)
    assert Labeller("openrouter").label_part("scn_x", "hinge", b"PNGDATA") == reply
    img = stub.calls[0]["messages"][0]["content"][0]
    assert img["source"]["media_type"] == "image/png"


def _parts():
    from pathlib import Path

    from screwit.contracts import Part

    root = Path(__file__).resolve().parents[3] / "fixtures" / "scans" / "scn_trap" / "parts.json"
    return [Part.model_validate(p) for p in json.loads(root.read_text())]


def test_scene_graph_via_claude_filters_the_reply(monkeypatch):
    from screwit.perception import relations as R

    parts = _parts()
    ids = [p.part_id for p in parts]
    pairs = [R.PairGeometry(a=ids[0], b=ids[1], surface_gap_mm=0.5, coaxial=True)]
    monkeypatch.setattr(R, "candidate_pairs", lambda *_a, **_k: pairs)
    reply = {
        "object": {"category": "drain", "summary": "a drain"},
        "function": "carries water away",
        "relations": [
            {"a": ids[0], "b": ids[1], "type": "threaded", "confidence": 0.7},
            {"a": ids[0], "b": "p99", "type": "glued", "confidence": 0.9},          # unknown id
            {"a": ids[2], "b": ids[3], "type": "seals", "confidence": 0.9},         # not touching
        ],
        "failure_hypotheses": [
            {"part_id": ids[1], "mode": "cracked", "evidence": ["drip"], "likelihood": 0.3},
            {"part_id": ids[0], "mode": "loose", "evidence": ["gap"], "likelihood": 0.6},
            {"part_id": "p99", "mode": "x", "evidence": [], "likelihood": 0.9},
        ],
    }
    stub = StubClient([json.dumps(reply)])
    _use(monkeypatch, stub)
    graph = R.SceneBuilder("openrouter").build("scn_trap", parts, {})
    assert graph.object.category == "drain"
    assert [(r.a, r.b) for r in graph.relations] == [(ids[0], ids[1])]
    assert [f.mode for f in graph.failure_hypotheses] == ["loose", "cracked"]


def test_scene_graph_falls_back_to_geometry(monkeypatch):
    from screwit.perception import relations as R

    parts = _parts()
    monkeypatch.setattr(R, "candidate_pairs", lambda *_a, **_k: [])
    stub = StubClient([RuntimeError("timeout"), RuntimeError("timeout")])
    _use(monkeypatch, stub)
    graph = R.SceneBuilder("openrouter").build("scn_unknown", parts, {})
    assert graph.object.category == "object" and graph.failure_hypotheses == []


def test_openrouter_client_targets_openrouter_with_bearer_key(monkeypatch):
    import screwit.llm as L

    monkeypatch.setenv("SCREWIT_LLM", "openrouter")
    monkeypatch.setenv("SCREWIT_MODEL", "anthropic/claude-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-used")
    c = L.client()
    assert str(c.base_url).rstrip("/") == "https://openrouter.ai/api"
    assert c.auth_token == "sk-or-test"
    assert c.api_key is None
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(L.LLMNotConfigured):
        L.client()
