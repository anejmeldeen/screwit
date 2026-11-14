"""Find which gaussian slab matches which view and whether gaussians live in the unscaled (scale_factor-divided) frame."""
import glob, json
import numpy as np
from common import ROOT, save
from depth_anything_3.api import DepthAnything3
from geom_check import to44
imgs = sorted(glob.glob(str(ROOT / "data/scene/*.jpg")))[:8]
m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cpu")
p = m.inference(imgs, infer_gs=True, ref_view_strategy="middle")
sf = p.scale_factor; N, H, W = p.depth.shape; n0 = H * W
means = p.gaussians.means[0].numpy()
E = np.stack([to44(e) for e in p.extrinsics])
out = {"scale_factor": sf, "slabs": []}
def z(X, e, tdiv):
    e = e.copy(); e[:3, 3] /= tdiv
    return (e @ np.vstack([X.T, np.ones(len(X))]))[2].reshape(H, W)
for k in range(N):
    X = means[k * n0:(k + 1) * n0]
    best = None
    for j in range(N):
        for name, tdiv, dd in (("as_is", 1.0, p.depth[j]), ("unscaled", sf, p.depth[j] / sf)):
            r = dd / z(X, E[j], tdiv)
            s = float(np.std(np.log(np.clip(r, 1e-3, 1e3))))
            if best is None or s < best[0]:
                best = (s, j, name, float(np.median(r)))
    out["slabs"].append({"slab": k, "best_view": best[1], "hypothesis": best[2], "log_ratio_std": best[0], "median_ratio": best[3]})
    print(out["slabs"][-1], flush=True)
save("da3_nested_gs_analysis2.json", out)
