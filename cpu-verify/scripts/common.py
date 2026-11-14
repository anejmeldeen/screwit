"""Shared helpers: cache env, timing, peak RSS."""
import os, time, resource, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

class Timer:
    def __init__(self, name, log):
        self.name, self.log = name, log
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.log[self.name] = {"wall_s": round(time.time() - self.t, 2), "peak_rss_gb_so_far": round(peak_rss_gb(), 2)}
        print(self.name, self.log[self.name], flush=True)

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))
