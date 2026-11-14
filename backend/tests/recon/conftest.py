"""Shared fakes for the recon tests: a tmpdir-backed Storage and a CPU PerceptionService.

These implement the Storage and PerceptionService Protocols from screwit.interfaces well enough to
drive the pipeline end to end on the CPU. They do NOT model S3 semantics; the real ones are P5's/P3's.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from screwit.contracts import PresignedPost


class TmpStorage:
    """Local-disk Storage keyed by 'scans/<id>/...', rooted at a tmp directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def presign_upload(self, key, content_types, max_bytes, expires_s=600) -> PresignedPost:
        return PresignedPost(url=f"file://{self._p(key)}", fields={}, key=key)

    def signed_url(self, key, expires_s=900) -> str:
        return f"file://{self._p(key)}"

    def head(self, key):
        p = self.root / key
        if not p.exists():
            return None
        return {"size": p.stat().st_size, "content_type": "application/octet-stream"}

    def get_bytes(self, key) -> bytes:
        return (self.root / key).read_bytes()

    def put_bytes(self, key, data, content_type) -> None:
        self._p(key).write_bytes(data)

    def local_path(self, key, scratch) -> Path:
        return self.root / key

    def put_file(self, key, path, content_type) -> None:
        dst = self._p(key)
        shutil.copyfile(path, dst)

    def exists(self, key) -> bool:
        return (self.root / key).exists()


class FakePerception:
    """A CPU PerceptionService: object_mask returns a centred box on the object; parts writes a
    trivial part GLB and returns one Part + a SceneGraph; rescale_parts is a no-op list."""

    def __init__(self) -> None:
        self.rescale_calls: list[tuple[str, float, int]] = []

    def object_mask(self, frames, object_hint, stored_prompts):
        # Mask the region where the object (grey, value ~120) was rendered, falling back to a centred
        # box so depth back-projection yields an object point cloud.
        masks = []
        for img in frames.images:
            h, w = img.shape[:2]
            grey = img.mean(axis=2)
            m = (grey > 90) & (grey < 160)
            if m.sum() < 50:
                m = np.zeros((h, w), bool)
                m[h // 3:2 * h // 3, w // 3:2 * w // 3] = True
            masks.append(m)
        return masks

    def parts(self, scan_id, frames, masks, mesh_path, reconstruction_revision, storage):
        from screwit.contracts import (
            OBB,
            Guess,
            Part,
            SceneGraph,
            SceneObject,
            measurement,
        )

        part = Part(
            part_id="p1", reconstruction_revision=reconstruction_revision, measurement_revision=1,
            label="object", label_confidence=0.9, description="synthetic test object",
            mesh_uri=f"scans/{scan_id}/parts/p1.glb",
            thumbnail_uri=f"scans/{scan_id}/renders/p1.png",
            centroid_mm=(210.0, 105.0, 50.0),
            obb=OBB(center=(210.0, 105.0, 50.0), extent=(80.0, 80.0, 100.0),
                    rotation_quat=(0.0, 0.0, 0.0, 1.0)),
            dimensions=[],
            features=[],
            material_guess=Guess(value="plastic", confidence=0.5),
            source="auto",
        )
        _ = measurement
        sg = SceneGraph(scan_id=scan_id, object=SceneObject(category="test", summary="test object"),
                        parts=["p1"], relations=[], function="test", failure_hypotheses=[])
        return [part], sg

    def rescale_parts(self, scan_id, s, reconstruction_revision, storage):
        self.rescale_calls.append((scan_id, s, reconstruction_revision))
        return []

    # Unused-by-recon members of the Protocol (present for completeness).
    def patch_part(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def merge_parts(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def measure(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def render_view(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def storage(tmp_path):
    return TmpStorage(tmp_path / "data")


@pytest.fixture
def perception():
    return FakePerception()


class ProgressRecorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def __call__(self, stage, state, progress, message):
        self.events.append((stage, state, progress, message))


@pytest.fixture
def progress():
    return ProgressRecorder()
