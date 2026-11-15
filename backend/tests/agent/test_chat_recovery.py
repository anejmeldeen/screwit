from screwit.agent.fake_llm import FakeDriver, ScriptTurn
from screwit.agent.loop import LlmChatAgent, _handle_result
from screwit.agent.tools import ToolContext
from screwit.contracts import TokenEvent
from screwit.storage import LocalStorage


def test_fixture_render_with_empty_live_storage(registry, tmp_path):
    registry.storage = LocalStorage(tmp_path / "empty-live")
    result = ToolContext(registry, "scn_trap").dispatch(
        "render_view", {"part_ids": ["p1"], "angle": 45})
    assert result["image_base64"].startswith("iVBOR")
    _, content = _handle_result(result)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"


def test_tool_failure_can_continue_to_answer(registry, monkeypatch):
    def fail(*args):
        raise FileNotFoundError("missing render")

    monkeypatch.setattr(ToolContext, "dispatch", fail)
    driver = FakeDriver([
        ScriptTurn(text="Let me look.", tool_calls=[("render_view", {"part_ids": ["p1"]})]),
        ScriptTurn(text="I cannot render it right now. Which part do you want to modify?"),
    ])
    events = list(LlmChatAgent(registry, driver).reply("scn_trap", "modify this", []))
    assert not any(e.event == "error" for e in events)
    assert "Which part" in "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert events[-1].event == "done"
