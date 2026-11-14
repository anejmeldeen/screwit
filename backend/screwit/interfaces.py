"""Seams between the five workstreams (SPEC section 7).

Each Protocol is implemented by exactly one owner and consumed by others. The API layer (P4) wires
implementations through screwit.app.registry; mock mode wires fixture-backed fakes instead.
Everything here works in scan-frame millimetres (SPEC 3.1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from screwit.contracts import (
    CadRequest,
    CadResult,
    Cameras,
    MeasureRequest,
    MeasureResponse,
    MergeRequest,
    Part,
    PartPatch,
    PresignedPost,
    Reconstruction,
    ScaleRequest,
    Scan,
    SceneGraph,
    StageName,
    StageState,
)

# ------------------------------------------------------------------ storage (owner: P5)

class Storage(Protocol):
    """Object store keyed by 'scans/<scan_id>/...'. Local disk in dev, private S3 when deployed."""

    def presign_upload(self, key: str, content_types: list[str], max_bytes: int,
                       expires_s: int = 600) -> PresignedPost: ...

    def signed_url(self, key: str, expires_s: int = 900) -> str: ...

    def head(self, key: str) -> dict | None:
        """{'size': int, 'content_type': str} or None if missing. Never reads the body."""

    def get_bytes(self, key: str) -> bytes: ...

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    def local_path(self, key: str, scratch: Path) -> Path:
        """Materialise an object as a local file (download to `scratch` for S3)."""

    def put_file(self, key: str, path: Path, content_type: str) -> None: ...


@dataclass(frozen=True)
class MediaInfo:
    kind: str  # "video" | "photos" | "mesh"
    duration_s: float | None
    container: str | None
    codec: str | None
    width: int | None
    height: int | None
    hdr: bool | None


class MediaValidator(Protocol):
    """Contained worker-side validation job (SPEC 3.3, owner: P5). Raises ValueError with a
    user-facing message when the upload is unacceptable."""

    def validate(self, scan: Scan, storage: Storage) -> MediaInfo: ...


# ------------------------------------------------------------------ pipeline (owner: P2)

ProgressFn = Callable[[StageName, StageState, float | None, str | None], None]


class ReconPipeline(Protocol):
    """Runs frames -> geometry -> scale -> object_mask -> splat -> mesh -> parts on the GPU worker,
    reporting every stage transition through `progress`. Calls the perception service for the
    object_mask and parts stages."""

    def run(self, scan: Scan, storage: Storage, perception: PerceptionService,
            progress: ProgressFn) -> None: ...

    def rescale(self, scan_id: str, req: ScaleRequest, storage: Storage,
                perception: PerceptionService) -> Reconstruction:
        """Analytic uniform rescale (SPEC 5 /scale). Raises StaleRevision on expected_revision mismatch."""


class StaleRevision(Exception):
    pass


# ------------------------------------------------------------------ perception (owner: P3)

@dataclass
class FrameSet:
    """Model-grid keyframes with their cameras and depth, all in the scan frame."""

    images: list[np.ndarray]        # HxWx3 uint8, model grid
    depth_mm: list[np.ndarray]      # HxW float32, 0 = invalid
    confidence: list[np.ndarray]    # HxW float32
    cameras: Cameras
    full_res_paths: list[Path]      # for the agent's close-ups


class PerceptionService(Protocol):
    def object_mask(self, frames: FrameSet, object_hint: str | None,
                    stored_prompts: dict | None) -> list[np.ndarray]:
        """One HxW bool mask per keyframe; the board is excluded."""

    def parts(self, scan_id: str, frames: FrameSet, masks: list[np.ndarray], mesh_path: Path,
              reconstruction_revision: int, storage: Storage) -> tuple[list[Part], SceneGraph]:
        """Segment, lift, measure, label, relate. Writes part GLBs and renders to storage."""

    def patch_part(self, scan_id: str, part_id: str, patch: PartPatch, storage: Storage) -> Part: ...

    def merge_parts(self, scan_id: str, req: MergeRequest, storage: Storage) -> Part: ...

    def measure(self, scan_id: str, req: MeasureRequest, storage: Storage) -> MeasureResponse: ...

    def rescale_parts(self, scan_id: str, s: float, reconstruction_revision: int,
                      storage: Storage) -> list[Part]:
        """Analytic part rescale for /scale: keeps ids, multiplies scan_mm, re-evaluates snaps."""

    def render_view(self, scan_id: str, part_ids: list[str], angle_deg: float,
                    storage: Storage) -> bytes:
        """PNG rendered on the CPU (never queued on the GPU worker)."""


# ------------------------------------------------------------------ CAD (owner: P5)

class CadService(Protocol):
    def run(self, req: CadRequest, parts: dict[str, Part], storage: Storage) -> CadResult:
        """Build the template, run every check, write STL/STEP/source/preview to storage."""

    def templates(self) -> dict[str, dict]:
        """Template name -> JSON Schema of its params (for the agent's tool description)."""


# ------------------------------------------------------------------ agent (owner: P4)

class ChatAgent(Protocol):
    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        """Yields contracts ChatEvent objects, ending with a DoneEvent."""
