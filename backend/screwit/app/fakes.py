"""Fixture-backed fakes of every seam the API consumes (SPEC 8, interfaces.py).

The other four workstreams build the real Storage, MediaValidator, ReconPipeline, PerceptionService
and CadService in parallel. The API codes against the Protocols in screwit.interfaces; these fakes let
P4 build and test the whole platform and run mock mode with no network, GPU or credentials.

Everything reads from the fixtures tree (SCREWIT_FIXTURES_DIR): fixtures/scans/<id>/*.json and the
scan assets under fixtures/scans/<id>/... served through the asset route.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from screwit.contracts import (
    CadFiles,
    CadRequest,
    CadResult,
    Check,
    InputRevisions,
    MeasureRequest,
    MeasureResponse,
    MergeRequest,
    Part,
    PartPatch,
    PresignedPost,
    PrintSettings,
    Reconstruction,
    Scan,
    SceneGraph,
)
from screwit.interfaces import FrameSet, MediaInfo

_ASSET_SIG = "fixture"  # marks URLs issued by the mock fixture store; not access control (see verify_signed).

_CONTENT_TYPES = {".glb": "model/gltf-binary", ".spz": "application/octet-stream", ".ply": "application/octet-stream",
                  ".stl": "model/stl", ".step": "model/step", ".png": "image/png", ".jpg": "image/jpeg",
                  ".json": "application/json", ".py": "text/plain"}


def _content_type(p: Path) -> str:
    return _CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream")


class FixtureStorage:
    """Storage (P5) backed by the read-only fixtures tree. Writes are refused: mock mode is read-only."""

    def __init__(self, fixtures_dir: Path, generated_dir: Path | None = None) -> None:
        self._root = fixtures_dir
        self._generated = generated_dir

    def _path(self, key: str) -> Path:
        p = (self._root / key).resolve()
        if not p.is_relative_to(self._root.resolve()):
            raise ValueError("key escapes the fixtures tree")
        if self._generated is not None:
            generated = (self._generated / key).resolve()
            if not generated.is_relative_to(self._generated.resolve()):
                raise ValueError("key escapes generated assets")
            if generated.is_file():
                return generated
        return p

    def presign_upload(self, key: str, content_types: list[str], max_bytes: int,
                       expires_s: int = 600) -> PresignedPost:
        return PresignedPost(
            url="/api/storage/upload",
            fields={"key": key, "content_type": ",".join(content_types), "max_bytes": str(max_bytes)},
            key=key,
        )

    def signed_url(self, key: str, expires_s: int = 900) -> str:
        return f"/api/assets/{key}?exp=0&sig={_ASSET_SIG}"

    def verify_signed(self, key: str, exp: str, sig: str) -> bool:
        """Mock mode serves the read-only demo fixtures behind the loopback gate (AGENTS.md decision 1).
        The constant signature only marks URLs this store issued; it is not access control."""
        try:
            self._path(key)
        except ValueError:
            return False
        return sig == _ASSET_SIG and exp == "0"

    def head(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return {"size": p.stat().st_size, "content_type": _content_type(p)}

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        raise NotImplementedError("mock storage is read-only")

    def local_path(self, key: str, scratch: Path) -> Path:
        return self._path(key)

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        raise NotImplementedError("mock storage is read-only")


class FixtureData:
    """Loads and validates the per-scan fixture JSON. Shared by the fakes and mock routes."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._root = fixtures_dir

    def _scan_dir(self, scan_id: str) -> Path:
        return self._root / "scans" / scan_id

    def _load(self, scan_id: str, name: str):
        p = self._scan_dir(scan_id) / name
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def scan(self, scan_id: str) -> Scan | None:
        raw = self._load(scan_id, "scan.json")
        return Scan.model_validate(raw) if raw is not None else None

    def reconstruction(self, scan_id: str) -> Reconstruction | None:
        raw = self._load(scan_id, "reconstruction.json")
        return Reconstruction.model_validate(raw) if raw is not None else None

    def parts(self, scan_id: str) -> list[Part]:
        raw = self._load(scan_id, "parts.json")
        return [Part.model_validate(p) for p in raw] if raw is not None else []

    def scene_graph(self, scan_id: str) -> SceneGraph | None:
        raw = self._load(scan_id, "scene_graph.json")
        return SceneGraph.model_validate(raw) if raw is not None else None

    def cad_result(self, scan_id: str) -> CadResult | None:
        raw = self._load(scan_id, "cad_result.json")
        return CadResult.model_validate(raw) if raw is not None else None

    def cad_request(self, scan_id: str) -> CadRequest | None:
        raw = self._load(scan_id, "cad_request.json")
        return CadRequest.model_validate(raw) if raw is not None else None

    def chat_script_raw(self, scan_id: str) -> dict | None:
        return self._load(scan_id, "chat_script.json")


class FakeMediaValidator:
    """MediaValidator (P5): accepts the fixture scan's declared input without touching media."""

    def validate(self, scan: Scan, storage) -> MediaInfo:
        inp = scan.input
        return MediaInfo(
            kind="mesh" if inp.type == "mesh" else ("photos" if inp.type == "photos" else "video"),
            duration_s=inp.duration_s,
            container=inp.container,
            codec=inp.codec,
            width=None,
            height=None,
            hdr=inp.hdr,
        )


class FakeReconPipeline:
    """ReconPipeline (P2): mock mode runs nothing (fixtures are already 'ready'); rescale is analytic."""

    def __init__(self, data: FixtureData) -> None:
        self._data = data

    def run(self, scan, storage, perception, progress) -> None:
        for stage in scan.stages:
            progress(stage.name, stage.state, stage.progress, stage.message)

    def rescale(self, scan_id, req, storage, perception) -> Reconstruction:
        recon = self._data.reconstruction(scan_id)
        if recon is None:
            raise FileNotFoundError(scan_id)
        return recon


class FakePerceptionService:
    """PerceptionService (P3): serves fixture parts/scene graph; measure is computed; patch mutates
    the in-memory part copy so the API's revision and stale rules are exercised in mock mode."""

    def __init__(self, data: FixtureData) -> None:
        self._data = data

    def object_mask(self, frames: FrameSet, object_hint, stored_prompts) -> list[np.ndarray]:
        return [np.ones((4, 4), dtype=bool) for _ in frames.images]

    def parts(self, scan_id, frames, masks, mesh_path, reconstruction_revision, storage):
        return self._data.parts(scan_id), self._data.scene_graph(scan_id)

    def patch_part(self, scan_id, part_id, patch: PartPatch, storage) -> Part:
        part = next((p for p in self._data.parts(scan_id) if p.part_id == part_id), None)
        if part is None:
            raise KeyError(part_id)
        return _apply_patch(part, patch)

    def merge_parts(self, scan_id, req: MergeRequest, storage) -> Part:
        parts = {p.part_id: p for p in self._data.parts(scan_id)}
        first = parts[req.part_ids[0]]
        merged = first.model_copy(update={
            "color_source_id": first.color_source_id or first.part_id,
            "part_id": "merged_" + "_".join(req.part_ids),
            "label": req.label or first.label,
            "measurement_revision": 1,
            "source": "user_edited",
        })
        return merged

    def measure(self, scan_id, req: MeasureRequest, storage) -> MeasureResponse:
        a = self._point(scan_id, req.a)
        b = self._point(scan_id, req.b)
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
        return MeasureResponse(distance_mm=dist, a_mm=a, b_mm=b)

    def rescale_parts(self, scan_id, s, reconstruction_revision, storage) -> list[Part]:
        return self._data.parts(scan_id)

    def render_view(self, scan_id, part_ids, angle_deg, storage) -> bytes:
        return b"\x89PNG\r\n\x1a\n"  # a PNG magic-byte stub; real renders come from P3.

    def _point(self, scan_id: str, ref):
        if isinstance(ref, str):
            part = next((p for p in self._data.parts(scan_id) if p.part_id == ref), None)
            if part is None:
                raise KeyError(ref)
            return part.centroid_mm
        return tuple(float(x) for x in ref)


class FakeCadService:
    """CadService (P5): returns the fixture cad_result, with input revisions taken from the request."""

    def __init__(self, data: FixtureData) -> None:
        self._data = data

    def run(self, req: CadRequest, parts: dict[str, Part], storage) -> CadResult:
        result = self._data.cad_result(req.scan_id)
        if result is not None:
            return result.model_copy(update={
                "cad_id": req.cad_id,
                "input_revisions": result.input_revisions.model_copy(update={
                    "reconstruction_revision": req.reconstruction_revision,
                    "part_measurement_revisions": dict(req.part_measurement_revisions),
                }),
            })
        # No fixture cad_result for this scan (e.g. a freshly created or synthetic scan): synthesize a
        # minimal contract-valid result so the CAD flow still produces a card. The real CadService (P5)
        # computes true checks; this fake just confirms the template built and the required checks pass.
        return _synth_cad_result(req)


    def templates(self) -> dict[str, dict]:
        from screwit.agent.cad import TEMPLATE_SCHEMAS

        return TEMPLATE_SCHEMAS


def _apply_patch(part: Part, patch: PartPatch) -> Part:
    """Apply a PartPatch to a Part, returning a new validated Part (mirrors SPEC 5 PATCH rules)."""
    data = part.model_dump()
    bump = False
    if patch.label is not None:
        data["label"] = patch.label
    if patch.description is not None:
        data["description"] = patch.description
    for edit in patch.measurements:
        feat = next((f for f in data["features"] if f["feature_id"] == edit.feature_id), None)
        if feat is None or edit.field not in feat:
            raise ValueError(f"unknown measurement {edit.feature_id}.{edit.field}")
        meas = feat[edit.field]
        if not isinstance(meas, dict):
            raise ValueError(f"{edit.field} is not a measurement")
        if edit.caliper_mm is not None:
            meas["caliper_mm"] = edit.caliper_mm
        if edit.user_confirmed is not None:
            meas["user_confirmed"] = edit.user_confirmed
        meas["preferred"] = _expected_preferred(meas)
        bump = True
    if bump:
        data["measurement_revision"] = part.measurement_revision + 1
        data["source"] = "user_edited"
    return Part.model_validate(data)


def _expected_preferred(meas: dict) -> str:
    if meas.get("caliper_mm") is not None:
        return "caliper"
    if meas.get("snapped_mm") is not None:
        return "snapped"
    return "scan"


def mock_match(turns: list[dict], message: str) -> dict:
    """First turn whose regex matches the message (SPEC 8); the script's last turn matches '.*'."""
    for turn in turns:
        if re.search(turn["match"], message):
            return turn
    return turns[-1]


def _synth_cad_result(req: CadRequest) -> CadResult:
    """A minimal contract-valid CAD result for a scan with no fixture cad_result. Required checks pass
    (the template built; mating inputs are user_confirmed by the time create_cad runs), advisories are
    reported as not_computed. The real CadService (P5) replaces this with computed checks."""
    base = f"scans/{req.scan_id}/cad/{req.cad_id}"
    checks = [
        Check(name="watertight", kind="required", result="pass"),
        Check(name="fits_build_volume", kind="required", result="pass"),
    ]
    for mate in req.mates_with:
        checks.append(Check(name=f"mate_{mate.feature_id}_diameter_clearance_mm", kind="required",
                            result="pass", value=mate.radial_clearance_mm, limit=mate.radial_clearance_mm,
                            input_source="user_confirmed"))
    checks += [
        Check(name="min_wall_thickness_mm", kind="advisory", result="not_computed",
              reason="synthesized fake result; P5 computes this"),
        Check(name="strength_estimate", kind="advisory", result="not_computed",
              reason="not computed by the fake CAD service"),
    ]
    return CadResult(
        cad_id=req.cad_id, status="ok",
        input_revisions=InputRevisions(reconstruction_revision=req.reconstruction_revision,
                                       part_measurement_revisions=dict(req.part_measurement_revisions)),
        stale=False,
        files=CadFiles(stl=f"{base}/model.stl", step=f"{base}/model.step",
                       source=f"{base}/model.py", preview=f"{base}/preview.png"),
        checks=checks,
        print_settings=PrintSettings(material=req.material, layer_height_mm=0.2, infill_pct=25,
                                     orientation="auto", supports=False, est_time_min=30,
                                     est_material_g=10, estimate_method="volume_formula"),
        notes="Synthesized by the fake CAD service (no fixture cad_result for this scan).",
        error=None,
    )
