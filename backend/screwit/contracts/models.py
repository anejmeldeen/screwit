"""Data contracts shared by every stage (SPEC section 4).

Conventions, enforced here and relied on everywhere else:
- All lengths are millimetres, all coordinates are in the scan frame (right-handed, Z up).
- Rotations are unit quaternions [x, y, z, w]; never Euler angles.
- 4x4 matrices are row-major lists of rows.
- extra="forbid" on every model, so a fixture or producer that drifts from the contract fails loudly.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # [x, y, z, w]
Row4 = tuple[float, float, float, float]
Matrix4 = tuple[Row4, Row4, Row4, Row4]
Row3 = tuple[float, float, float]
Matrix3 = tuple[Row3, Row3, Row3]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unit_quat(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if not 0.999 <= n <= 1.001:
        raise ValueError(f"quaternion must be unit length, got norm {n:.4f}")
    return q


def _unit_vec(v: Vec3) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if not 0.999 <= n <= 1.001:
        raise ValueError(f"direction must be unit length, got norm {n:.4f}")
    return v


# --------------------------------------------------------------------------- 4.1 Scan

InputType = Literal["video", "photos", "mesh"]
ScanStatus = Literal["uploaded", "validating", "processing", "ready", "failed"]
StageName = Literal["frames", "geometry", "scale", "object_mask", "splat", "mesh", "parts"]
StageState = Literal["pending", "running", "done", "failed", "skipped"]
STAGE_ORDER: tuple[StageName, ...] = ("frames", "geometry", "scale", "object_mask", "splat", "mesh", "parts")
MESH_INPUT_SKIPPED: frozenset[StageName] = frozenset({"frames", "geometry", "object_mask", "splat"})


class ScanInput(Contract):
    type: InputType
    uri: str
    duration_s: float | None = None
    container: str | None = None
    codec: str | None = None
    hdr: bool | None = None
    mesh_units: Literal["metres", "millimetres"] | None = None


class Stage(Contract):
    name: StageName
    state: StageState
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str | None = None


class Scan(Contract):
    scan_id: str
    created_at: datetime
    input: ScanInput
    status: ScanStatus
    reconstruction_revision: int = Field(ge=0)
    stages: list[Stage]

    @model_validator(mode="after")
    def _stages_complete_and_ordered(self) -> Scan:
        names = tuple(s.name for s in self.stages)
        if names != STAGE_ORDER:
            raise ValueError(f"stages must be exactly {STAGE_ORDER} in order, got {names}")
        if self.input.type == "mesh":
            for s in self.stages:
                if s.name in MESH_INPUT_SKIPPED and s.state != "skipped":
                    raise ValueError(f"mesh input must skip stage {s.name}")
        return self

    def stage(self, name: StageName) -> Stage:
        return next(s for s in self.stages if s.name == name)


# --------------------------------------------------------------------------- 4.2 Reconstruction

class ScaleInfo(Contract):
    method: Literal["aruco", "manual", "learned"]
    board: str | None = None
    corner_views: int | None = None
    residual_mm: float | None = None
    board_tilt_deg: float | None = None
    learned_scale_ratio: float | None = None


class ScanToWeb(Contract):
    matrix: Matrix4
    unit_scale: float = Field(gt=0)


SCAN_TO_WEB_DEFAULT = ScanToWeb(
    matrix=((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)),  # Rx(-90 deg), row-major
    unit_scale=0.001,
)


class BBox(Contract):
    min: Vec3
    max: Vec3


class ReconQuality(Contract):
    frames_used: int
    total_frames: int
    mean_depth_confidence: float
    gaussians: int | None = None


class Reconstruction(Contract):
    scan_id: str
    reconstruction_revision: int = Field(ge=0)
    splat_uri: str | None
    mesh_uri: str
    cameras_uri: str
    keyframes: list[str]
    scale: ScaleInfo
    frame_origin: Literal["board", "object_bottom_center"]
    scan_to_web: ScanToWeb
    object_bbox_mm: BBox
    quality: ReconQuality
    # Phone-sized variants (AGENTS.md "Performance budgets"); null until produced, clients fall back to the full ones.
    splat_mobile_uri: str | None = None
    mesh_mobile_uri: str | None = None


class CameraFrame(Contract):
    """One entry of geometry/cameras.json (SPEC 3.1 camera contract)."""

    frame: str
    width: int
    height: int
    world_to_camera: Matrix4  # row-major, OpenCV axes (X right, Y down, Z forward), scan frame, mm
    K: Matrix3


class Cameras(Contract):
    scan_id: str
    reconstruction_revision: int
    frames: list[CameraFrame]


# --------------------------------------------------------------------------- 4.3 Part

MeasurementSource = Literal["scan", "snapped", "caliper"]
ParamSource = Literal["scan", "snapped", "caliper", "user_confirmed"]


class Measurement(Contract):
    scan_mm: float | None
    snapped_mm: float | None = None
    snap_rule: str | None = None
    caliper_mm: float | None = None
    user_confirmed: bool = False
    preferred: MeasurementSource

    @model_validator(mode="after")
    def _preferred_is_consistent(self) -> Measurement:
        expected = expected_preferred(self.scan_mm, self.snapped_mm, self.caliper_mm)
        if self.preferred != expected:
            raise ValueError(f"preferred must be {expected!r} for these values, got {self.preferred!r}")
        if (self.snapped_mm is None) != (self.snap_rule is None):
            raise ValueError("snapped_mm and snap_rule must be set together")
        return self

    @property
    def value_mm(self) -> float:
        v = {"caliper": self.caliper_mm, "snapped": self.snapped_mm, "scan": self.scan_mm}[self.preferred]
        assert v is not None
        return v

    @property
    def param_source(self) -> ParamSource:
        """Source reported in a CAD request's param_sources (SPEC 4.6)."""
        if self.preferred == "caliper":
            return "caliper"
        return "user_confirmed" if self.user_confirmed else self.preferred


def expected_preferred(scan_mm: float | None, snapped_mm: float | None,
                       caliper_mm: float | None) -> MeasurementSource:
    if caliper_mm is not None:
        return "caliper"
    if snapped_mm is not None:
        return "snapped"
    if scan_mm is None:
        raise ValueError("a Measurement needs at least a scan_mm or a caliper_mm")
    return "scan"


def measurement(scan_mm: float | None, *, snapped_mm: float | None = None, snap_rule: str | None = None,
                caliper_mm: float | None = None, user_confirmed: bool = False) -> Measurement:
    return Measurement(scan_mm=scan_mm, snapped_mm=snapped_mm, snap_rule=snap_rule, caliper_mm=caliper_mm,
                       user_confirmed=user_confirmed,
                       preferred=expected_preferred(scan_mm, snapped_mm, caliper_mm))


class _AxisFeature(Contract):
    feature_id: str
    axis_point_mm: Vec3
    axis_dir: Vec3
    fit_rmse_mm: float
    inlier_fraction: float = Field(ge=0, le=1)
    radius: Measurement
    length: Measurement

    @model_validator(mode="after")
    def _unit_axis(self) -> _AxisFeature:
        _unit_vec(self.axis_dir)
        return self


class CylinderFeature(_AxisFeature):
    type: Literal["cylinder"] = "cylinder"


class BoreFeature(_AxisFeature):
    type: Literal["bore"] = "bore"


class DShaftFeature(_AxisFeature):
    type: Literal["d_shaft"] = "d_shaft"
    flat_normal: Vec3
    flat_depth: Measurement

    @model_validator(mode="after")
    def _unit_flat(self) -> DShaftFeature:
        _unit_vec(self.flat_normal)
        return self


class PlaneFeature(Contract):
    type: Literal["plane"] = "plane"
    feature_id: str
    point_mm: Vec3
    normal: Vec3
    fit_rmse_mm: float
    inlier_fraction: float = Field(ge=0, le=1)
    extent_u: Measurement
    extent_v: Measurement


class HoleFeature(Contract):
    type: Literal["hole"] = "hole"
    feature_id: str
    center_mm: Vec3
    normal: Vec3
    fit_rmse_mm: float
    radius: Measurement
    depth: Measurement


class ThreadSuspectedFeature(Contract):
    type: Literal["thread_suspected"] = "thread_suspected"
    feature_id: str
    nominal: str
    confidence: float = Field(ge=0, le=1)


Feature = Annotated[
    CylinderFeature | BoreFeature | DShaftFeature | PlaneFeature | HoleFeature | ThreadSuspectedFeature,
    Field(discriminator="type"),
]

# Measurement fields per feature type (SPEC 4.3 table); PATCH may only target these.
FEATURE_MEASUREMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "cylinder": ("radius", "length"),
    "bore": ("radius", "length"),
    "d_shaft": ("radius", "length", "flat_depth"),
    "plane": ("extent_u", "extent_v"),
    "hole": ("radius", "depth"),
    "thread_suspected": (),
}


class OwnDimension(Contract):
    name: str
    value: Measurement


class DerivedDimension(Contract):
    """Read-only view of a feature Measurement, so one physical quantity is never stored twice."""

    name: str
    derived_from: str = Field(pattern=r"^[A-Za-z0-9_]+\.[a-z_]+$")  # "<feature_id>.<field>"
    factor: float = 1.0


Dimension = OwnDimension | DerivedDimension


class OBB(Contract):
    center: Vec3
    extent: Vec3
    rotation_quat: Quat

    @model_validator(mode="after")
    def _unit(self) -> OBB:
        _unit_quat(self.rotation_quat)
        return self


class Guess(Contract):
    value: str
    confidence: float = Field(ge=0, le=1)


class Part(Contract):
    part_id: str
    reconstruction_revision: int = Field(ge=0)
    measurement_revision: int = Field(ge=1)
    label: str
    label_confidence: float = Field(ge=0, le=1)
    description: str
    mesh_uri: str
    thumbnail_uri: str
    centroid_mm: Vec3
    obb: OBB
    dimensions: list[Dimension]
    features: list[Feature]
    material_guess: Guess
    source: Literal["auto", "user_edited"]

    @model_validator(mode="after")
    def _ids_and_derivations_resolve(self) -> Part:
        ids = [f.feature_id for f in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate feature_id in part {self.part_id}")
        for d in self.dimensions:
            if isinstance(d, DerivedDimension):
                fid, field = d.derived_from.split(".")
                feat = self.feature(fid)
                if feat is None or field not in FEATURE_MEASUREMENT_FIELDS[feat.type]:
                    raise ValueError(f"dimension {d.name} derives from unknown {d.derived_from}")
        return self

    def feature(self, feature_id: str):
        return next((f for f in self.features if f.feature_id == feature_id), None)

    def dimension_value_mm(self, name: str) -> float:
        d = next(d for d in self.dimensions if d.name == name)
        if isinstance(d, OwnDimension):
            return d.value.value_mm
        fid, field = d.derived_from.split(".")
        return getattr(self.feature(fid), field).value_mm * d.factor


# --------------------------------------------------------------------------- 4.4 Scene graph

RelationType = Literal["threaded", "press_fit", "fastened", "hinged", "sliding", "seals", "glued",
                       "rests_on", "unknown"]


class SceneObject(Contract):
    category: str
    summary: str


class Relation(Contract):
    a: str
    b: str
    type: RelationType
    confidence: float = Field(ge=0, le=1)
    note: str | None = None


class FailureHypothesis(Contract):
    part_id: str
    mode: str
    evidence: list[str]
    likelihood: float = Field(ge=0, le=1)


class SceneGraph(Contract):
    scan_id: str
    object: SceneObject
    parts: list[str]
    relations: list[Relation]
    function: str
    failure_hypotheses: list[FailureHypothesis]

    @model_validator(mode="after")
    def _refs_resolve(self) -> SceneGraph:
        known = set(self.parts)
        for r in self.relations:
            if r.a not in known or r.b not in known:
                raise ValueError(f"relation references unknown part: {r.a}-{r.b}")
        for h in self.failure_hypotheses:
            if h.part_id not in known:
                raise ValueError(f"failure hypothesis references unknown part {h.part_id}")
        return self


# --------------------------------------------------------------------------- 4.5 Repair plan

class Diagnosis(Contract):
    summary: str
    part_ids: list[str]
    confidence: float = Field(ge=0, le=1)


class Step(Contract):
    n: int = Field(ge=1)
    text: str
    part_ids: list[str] = []


class BuyLink(Contract):
    vendor: str
    url: str


class BomLine(Contract):
    name: str
    spec: str
    qty: int = Field(ge=1)
    est_price_usd: float = Field(ge=0)
    buy_links: list[BuyLink]


class RepairPlan(Contract):
    plan_id: str
    intent: str
    diagnosis: Diagnosis
    safety: list[str]
    tools: list[str]
    steps: list[Step]
    bom: list[BomLine]
    printable: list[str]  # cad_ids of printable parts proposed with this plan
    followups: list[str]

    @model_validator(mode="after")
    def _steps_numbered(self) -> RepairPlan:
        if [s.n for s in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("steps must be numbered 1..N in order")
        return self


# --------------------------------------------------------------------------- 4.6 CAD request

FitType = Literal["snug", "sliding", "press"]


class Mate(Contract):
    part_id: str
    feature_id: str
    fit: FitType
    radial_clearance_mm: float


class Placement(Contract):
    translate_mm: Vec3
    rotation_quat: Quat

    @model_validator(mode="after")
    def _unit(self) -> Placement:
        _unit_quat(self.rotation_quat)
        return self


class CadRequest(Contract):
    cad_id: str
    scan_id: str
    reconstruction_revision: int
    part_measurement_revisions: dict[str, int]
    mode: Literal["template", "codegen"]
    template: str | None
    params: dict[str, float]
    param_sources: dict[str, ParamSource]
    codegen_prompt: str | None
    mates_with: list[Mate]
    # Optional hint. The CAD service derives the placement from the mated feature (axis point and
    # direction) whenever there is a mate, and reports the placement it used in CadResult.placement.
    placement: Placement | None = None
    material: str
    printer_profile: str

    @model_validator(mode="after")
    def _consistent(self) -> CadRequest:
        if self.mode == "template" and not self.template:
            raise ValueError("template mode needs a template name")
        if self.mode == "codegen" and not self.codegen_prompt:
            raise ValueError("codegen mode needs a codegen_prompt")
        missing = set(self.params) - set(self.param_sources)
        if missing:
            raise ValueError(f"param_sources missing for {sorted(missing)}")
        for m in self.mates_with:
            if m.part_id not in self.part_measurement_revisions:
                raise ValueError(f"mate part {m.part_id} needs an entry in part_measurement_revisions")
        return self


# --------------------------------------------------------------------------- 4.7 CAD result

class Check(Contract):
    name: str
    kind: Literal["required", "advisory"]
    result: Literal["pass", "fail", "not_computed"]
    value: float | None = None
    limit: float | None = None
    fix: str | None = None
    reason: str | None = None
    input_source: ParamSource | None = None
    note: str | None = None


class CadFiles(Contract):
    stl: str
    step: str
    source: str
    preview: str | None


class InputRevisions(Contract):
    reconstruction_revision: int
    part_measurement_revisions: dict[str, int]


class PrintSettings(Contract):
    material: str
    layer_height_mm: float
    infill_pct: int
    orientation: str
    supports: bool
    est_time_min: int
    est_material_g: int
    estimate_method: Literal["volume_formula", "slicer"]


class CadResult(Contract):
    cad_id: str
    status: Literal["ok", "failed_checks", "error"]
    input_revisions: InputRevisions
    stale: bool
    files: CadFiles | None
    placement: Placement | None = None  # where the part sits in the scan frame (the viewer overlays it here)
    checks: list[Check]
    print_settings: PrintSettings | None
    notes: str
    error: str | None = None

    @model_validator(mode="after")
    def _status_matches_checks(self) -> CadResult:
        if self.status == "error":
            return self
        required_ok = all(c.result == "pass" for c in self.checks if c.kind == "required")
        if (self.status == "ok") != required_ok:
            raise ValueError("status must be ok exactly when every required check passes")
        return self


# --------------------------------------------------------------------------- 4.8 Chat stream

class ChatRequest(Contract):
    message: str = Field(min_length=1, max_length=4000)
    image_uris: list[str] = []


class TokenEvent(Contract):
    event: Literal["token"] = "token"
    text: str


class ToolCallEvent(Contract):
    event: Literal["tool_call"] = "tool_call"
    name: str
    args: dict


class HighlightEvent(Contract):
    event: Literal["highlight"] = "highlight"
    part_ids: list[str]


class RepairPlanCard(Contract):
    type: Literal["repair_plan"] = "repair_plan"
    payload: RepairPlan


class CadResultCard(Contract):
    type: Literal["cad_result"] = "cad_result"
    payload: CadResult


class PartsToBuyCard(Contract):
    type: Literal["parts_to_buy"] = "parts_to_buy"
    payload: list[BomLine]


Card = Annotated[RepairPlanCard | CadResultCard | PartsToBuyCard, Field(discriminator="type")]


class CardEvent(Contract):
    event: Literal["card"] = "card"
    card: Card


class DoneEvent(Contract):
    event: Literal["done"] = "done"
    message_id: str


class ErrorEvent(Contract):
    event: Literal["error"] = "error"
    message: str


ChatEvent = Annotated[TokenEvent | ToolCallEvent | HighlightEvent | CardEvent | DoneEvent | ErrorEvent,
                      Field(discriminator="event")]


class ChatMessage(Contract):
    message_id: str
    role: Literal["user", "assistant"]
    text: str
    cards: list[Card] = []
    created_at: datetime


class MockChatTurn(Contract):
    """One scripted reply for mock mode: the first turn whose regex matches the user message is replayed."""

    match: str
    events: list[ChatEvent]


class MockChatScript(Contract):
    scan_id: str
    turns: list[MockChatTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def _last_turn_is_catch_all(self) -> MockChatScript:
        if self.turns[-1].match != ".*":
            raise ValueError("the last mock turn must match '.*' so every message gets a reply")
        for t in self.turns:
            if not t.events or t.events[-1].event != "done":
                raise ValueError("every mock turn must end with a done event")
        return self


# --------------------------------------------------------------------------- Section 5 request bodies

class CreateScanRequest(Contract):
    input_type: InputType
    mesh_units: Literal["metres", "millimetres"] | None = None
    photo_count: int | None = Field(default=None, ge=1, le=60)

    @model_validator(mode="after")
    def _mesh_units_only_for_mesh(self) -> CreateScanRequest:
        if (self.input_type == "mesh") != (self.mesh_units is not None):
            raise ValueError("mesh_units is required for mesh input and not allowed otherwise")
        if self.input_type == "photos" and self.photo_count is None:
            raise ValueError("photo_count is required for photo input")
        return self


class PresignedPost(Contract):
    url: str
    fields: dict[str, str]
    key: str


class CreateScanResponse(Contract):
    scan_id: str
    uploads: list[PresignedPost]


class ScaleRequest(Contract):
    point_a_mm: Vec3
    point_b_mm: Vec3
    known_length_mm: float = Field(gt=0)
    expected_revision: int = Field(ge=0)


class MeasurementEdit(Contract):
    feature_id: str
    field: str
    caliper_mm: float | None = Field(default=None, gt=0)
    user_confirmed: bool | None = None

    @model_validator(mode="after")
    def _something_to_do(self) -> MeasurementEdit:
        if self.caliper_mm is None and self.user_confirmed is None:
            raise ValueError("set caliper_mm or user_confirmed")
        return self


class PartPatch(Contract):
    label: str | None = None
    description: str | None = None
    measurements: list[MeasurementEdit] = []


class MergeRequest(Contract):
    part_ids: list[str] = Field(min_length=2)
    label: str | None = None


class MeasureRequest(Contract):
    a: Vec3 | str  # a point in mm, or a part_id
    b: Vec3 | str


class MeasureResponse(Contract):
    distance_mm: float
    a_mm: Vec3
    b_mm: Vec3


class DemoScan(Contract):
    scan_id: str
    title: str
    scenario: Literal["A", "B"]
    thumbnail_uri: str


# --------------------------------------------------------------------------- Printer profile (6.4)

class PrinterProfile(Contract):
    name: str
    material: str
    build_volume_mm: Vec3
    nozzle_mm: float
    perimeters: int
    min_wall_mm: float
    max_overhang_deg: float
    min_feature_mm: float
    radial_clearance_mm: dict[FitType, float]
    effective_strength_mpa: dict[str, float] = {}  # by print orientation; empty means not_computed
    density_g_cm3: float
    layer_height_mm: float
    infill_pct: int
    volumetric_speed_mm3_s: float
