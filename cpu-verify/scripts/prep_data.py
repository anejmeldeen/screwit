"""Build the real-image sets used by every model (all taken from the cloned repos)."""
import shutil, pathlib
from PIL import Image
from common import ROOT
src = ROOT / "src"
scene = ROOT / "data" / "scene"; scene.mkdir(parents=True, exist_ok=True)
bed = src / "sam2/notebooks/videos/bedroom"
# 8 real frames of a moving-camera video (SAM2 repo notebook clip), every 25th frame
for i, n in enumerate(range(0, 200, 25)):
    shutil.copy(bed / f"{n:05d}.jpg", scene / f"{i:02d}.jpg")
hd = ROOT / "data" / "hd"; hd.mkdir(parents=True, exist_ok=True)
for i in range(2):  # 1920x1080 upscale of two real frames, only to test the model-grid arithmetic
    Image.open(scene / f"{i:02d}.jpg").convert("RGB").resize((1920, 1080), Image.BICUBIC).save(hd / f"{i:02d}.jpg")
print(sorted(p.name for p in scene.iterdir()))
