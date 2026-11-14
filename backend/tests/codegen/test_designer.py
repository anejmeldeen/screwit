"""The designer loop with the fake LLM picks a passing candidate and records the repair (X7 brief).

Docker-dependent (the loop runs each candidate in the real sandbox); skips cleanly without Docker.
"""

from __future__ import annotations

from screwit.codegen import design as design_entry
from screwit.codegen.designer import design as design_loop
from screwit.codegen.llm import DesignBrief, FakeCodegenLLM


def _brief() -> DesignBrief:
    return DesignBrief(
        request="a small rectangular spacer",
        required_dims_mm={"height": 10.0},
        build_volume_mm=(250.0, 210.0, 220.0),
        min_wall_mm=1.2,
        max_overhang_deg=50.0,
    )


def test_fake_llm_env_selects_fake(monkeypatch):
    monkeypatch.setenv("SCREWIT_LLM", "fake")
    from screwit.codegen.llm import build_codegen_llm

    assert isinstance(build_codegen_llm(), FakeCodegenLLM)


def test_loop_picks_passing_candidate_and_records_repair(docker_or_skip, tmp_path):
    llm = FakeCodegenLLM()
    # k=3 -> the third fake candidate (_FAKE_BROKEN) fails first and is repaired.
    outcome = design_loop(_brief(), llm, out_root=tmp_path, params={}, k=3, image=docker_or_skip)

    assert outcome.winner is not None
    assert outcome.winner.built
    assert outcome.winner.final.result.watertight is True

    # The broken candidate (index 2) failed on its first attempt, then a repair attempt succeeded.
    broken = outcome.candidates[2]
    assert broken.repaired
    assert len(broken.attempts) >= 2
    assert broken.attempts[0].result.ok is False
    assert broken.attempts[-1].result.ok is True
    assert broken.attempts[-1].repaired_from_error is not None
    assert llm.repair_calls >= 1

    # Rationale names the winner and whether it was repaired.
    assert "Chose candidate" in outcome.rationale


def test_design_entry_persists_files(docker_or_skip, tmp_path, fake_storage, fake_profile):
    llm = FakeCodegenLLM()
    files, rationale, attempts = design_entry(
        request="a small rectangular spacer",
        part=None,
        params_hint={"height": 10.0},
        profile=fake_profile,
        storage=fake_storage,
        out_prefix="scans/scn_x/cad/cad_cg01",
        llm=llm,
        k=2,
        image=docker_or_skip,
    )
    assert files is not None
    for key in ("stl", "step", "source", "preview"):
        assert files[key] in fake_storage.objects
        assert len(fake_storage.objects[files[key]]) > 0
    # The persisted source is runnable CadQuery (it defines build) and is the winning program.
    assert b"def build(params)" in fake_storage.objects[files["source"]]
    assert isinstance(attempts, list) and attempts
    assert "Chose candidate" in rationale


def test_design_entry_reports_failure_when_nothing_builds(docker_or_skip, tmp_path, fake_storage,
                                                          fake_profile):
    class AllBrokenLLM(FakeCodegenLLM):
        def propose(self, brief, k):
            bad = ("def build(params):\n    import cadquery as cq\n"
                   "    return cq.Workplane('XY').nope()\n")
            return [bad for _ in range(k)]

        def repair(self, brief, source, traceback):
            self.repair_calls += 1
            return source  # never fixes it

    files, rationale, attempts = design_entry(
        request="an impossible part", part=None, params_hint={}, profile=fake_profile,
        storage=fake_storage, out_prefix="scans/scn_x/cad/cad_cg02", llm=AllBrokenLLM(), k=2,
        image=docker_or_skip,
    )
    assert files is None
    assert "No candidate built" in rationale
    assert all(a["built"] is False for a in attempts)
