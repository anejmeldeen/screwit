"""Real-adapter code paths checked with stubbed model libraries (no torch, no weights).

These exercise the exact argument shapes and names the real SAM 2 / Grounding DINO APIs expect,
as pinned down by a real CPU run (cpu-verify/FINDINGS.md section 5, items 13-16). Every heavy
dependency (torch, sam2, transformers, PIL is real) is replaced through sys.modules so the checks
run on any machine, but they still fail loudly if the adapter passes the wrong device, label dtype,
video layout, result key or point count.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from screwit.perception import adapters

# --------------------------------------------------------------------------- resolve_device (13)

def test_resolve_device_env_wins(monkeypatch):
    monkeypatch.setenv("SCREWIT_DEVICE", "cpu")
    assert adapters.resolve_device() == "cpu"
    monkeypatch.setenv("SCREWIT_DEVICE", "cuda:1")
    assert adapters.resolve_device() == "cuda:1"


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("SCREWIT_DEVICE", raising=False)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert adapters.resolve_device() == "cpu"


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("SCREWIT_DEVICE", raising=False)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert adapters.resolve_device() == "cuda"


# --------------------------------------------------------------------------- shared stubs

class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_torch():
    return types.SimpleNamespace(
        inference_mode=lambda: _InferenceMode(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )


# --------------------------------------------------------------------------- SAM 2 video (13, 14)

class _FakeLogits:
    """Stand-in for a torch tensor of mask logits: indexable, with .cpu().numpy()."""

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, i):
        return _FakeLogits(self.arr[i])

    def __gt__(self, other):
        return _FakeLogits(self.arr > other)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeVideoPredictor:
    def __init__(self, calls):
        self.calls = calls

    def init_state(self, video_dir):
        self.calls["init_state_arg"] = video_dir
        # The directory must exist and hold integer-named JPEGs (finding 14).
        self.calls["jpeg_names"] = sorted(os.listdir(video_dir))
        return {"state": True}

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, points, labels, box):
        self.calls["labels"] = labels
        self.calls["frame_idx"] = frame_idx
        self.calls["obj_id"] = obj_id
        return (frame_idx, [obj_id], None)

    def propagate_in_video(self, state):
        # Two frames, one object; logits shape (n_obj, 1, H, W); mask = logits > 0.
        for fi in range(2):
            arr = np.where(np.arange(4 * 4).reshape(1, 1, 4, 4) % 2 == 0, 1.0, -1.0)
            yield fi, [1], _FakeLogits(arr)


def test_sam2_video_passes_device_int_labels_and_jpeg_dir(monkeypatch):
    calls = {}

    def build_sam2_video_predictor(cfg, ckpt, device=None):
        calls["cfg"] = cfg
        calls["ckpt"] = ckpt
        calls["device"] = device  # finding 13: a device must be passed
        return _FakeVideoPredictor(calls)

    sam2_build = types.ModuleType("sam2.build_sam")
    sam2_build.build_sam2_video_predictor = build_sam2_video_predictor
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", sam2_build)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    adapter = adapters.Sam2VideoAdapter("ckpt.pt", "cfg.yaml", device="cpu")
    out = adapter.segment(frames, points=np.array([[2.0, 2.0]]), labels=[1], ann_frame_idx=0,
                          obj_id=1)

    assert calls["device"] == "cpu"
    assert calls["cfg"] == "cfg.yaml" and calls["ckpt"] == "ckpt.pt"
    # finding 14: labels arrive as an int32 numpy array.
    assert isinstance(calls["labels"], np.ndarray) and calls["labels"].dtype == np.int32
    assert list(calls["labels"]) == [1]
    # finding 14: the video dir holds integer-named JPEGs.
    assert calls["jpeg_names"] == ["00000.jpg", "00001.jpg"]
    # Two per-frame boolean masks of shape (H, W) came back.
    assert set(out) == {0, 1}
    assert out[0].shape == (4, 4) and out[0].dtype == bool


def test_sam2_video_defaults_device_from_resolver(monkeypatch):
    calls = {}
    sam2_build = types.ModuleType("sam2.build_sam")
    sam2_build.build_sam2_video_predictor = lambda cfg, ckpt, device=None: (
        calls.__setitem__("device", device) or _FakeVideoPredictor(calls))
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", sam2_build)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())
    monkeypatch.setenv("SCREWIT_DEVICE", "cpu")

    adapters.Sam2VideoAdapter("c", "m").segment([np.zeros((4, 4, 3), np.uint8)], labels=[1])
    assert calls["device"] == "cpu"


# --------------------------------------------------------------------------- Grounding DINO (15)

class _Tensor:
    def __init__(self, vals):
        self.vals = vals

    def tolist(self):
        return self.vals


class _Inputs(dict):
    def __init__(self, input_ids="ids"):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        self["device"] = device
        return self


class _FakeProcessor:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, images, text, return_tensors):
        self.calls["prompt"] = text
        return _Inputs()

    def post_process_grounded_object_detection(self, outputs, input_ids, *, box_threshold,
                                               text_threshold, target_sizes):
        # Mirror transformers >= 4.51: labels are ints, text_labels are the phrase strings.
        return [{
            "boxes": [_Tensor([1.0, 2.0, 3.0, 4.0])],
            "scores": [0.9],
            "labels": [7],                 # int -> must NOT be used by the adapter
            "text_labels": ["a mug"],      # finding 15: adapter must read this
        }]


def test_grounding_dino_reads_text_labels_and_moves_to_device(monkeypatch):
    calls = {}

    class _FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def __call__(self, **inputs):
            return {"logits": None}

    def _from_pretrained(model_id):
        calls["model_id"] = model_id
        return _FakeModel()

    tf = types.ModuleType("transformers")
    tf.AutoModelForZeroShotObjectDetection = types.SimpleNamespace(from_pretrained=_from_pretrained)
    tf.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda mid: _FakeProcessor(calls))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())

    adapter = adapters.GroundingDinoAdapter(device="cpu")
    boxes = adapter.detect(np.zeros((8, 8, 3), np.uint8), ["mug", "bracket"], box_threshold=0.3,
                           text_threshold=0.2)

    assert calls["device"] == "cpu"
    assert calls["prompt"] == "mug. bracket."  # lower-case, period-separated
    assert len(boxes) == 1
    assert boxes[0].phrase == "a mug"          # from text_labels, not the int label
    assert boxes[0].xyxy == (1.0, 2.0, 3.0, 4.0)
    assert boxes[0].score == pytest.approx(0.9)


# --------------------------------------------------------------------------- automatic masks (13, 16)

def test_auto_masks_points_per_side_by_device(monkeypatch):
    seen = {}

    class _Gen:
        def generate(self, image):
            return [{"segmentation": np.ones((2, 2), bool)}]

    def build_sam2(cfg, ckpt, device=None):
        seen["device"] = device
        return object()

    class _AutoGen:
        def __init__(self, model, points_per_side=None):
            seen["points_per_side"] = points_per_side

        def generate(self, image):
            return _Gen().generate(image)

    amg = types.ModuleType("sam2.automatic_mask_generator")
    amg.SAM2AutomaticMaskGenerator = _AutoGen
    bsm = types.ModuleType("sam2.build_sam")
    bsm.build_sam2 = build_sam2
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", amg)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", bsm)

    cpu = adapters.Sam2AutomaticMasksAdapter("c", "m", device="cpu")
    masks = cpu.generate(np.zeros((4, 4, 3), np.uint8))
    assert seen["device"] == "cpu"
    assert seen["points_per_side"] == 16      # finding 16: 16 on CPU
    assert len(masks) == 1 and masks[0].shape == (2, 2)

    gpu = adapters.Sam2AutomaticMasksAdapter("c", "m", device="cuda")
    gpu.generate(np.zeros((4, 4, 3), np.uint8))
    assert seen["points_per_side"] == 32      # 32 on GPU

    override = adapters.Sam2AutomaticMasksAdapter("c", "m", device="cpu", points_per_side=8)
    override.generate(np.zeros((4, 4, 3), np.uint8))
    assert seen["points_per_side"] == 8
