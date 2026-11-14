"""Verify the SPEC 3.1 nested-model quirk and the gs_ply layout (one fresh in-memory run plus the exported PLY)."""
import glob, json
import numpy as np
from common import ROOT, OUT, save
from plyfile import PlyData
from scipy.spatial import cKDTree
from depth_anything_3.api import DepthAnything3
from geom_check import to44

log = {}
imgs = sorted(glob.glob(str(ROOT / "data/scene/*.jpg")))[:8]
m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cpu")
p = m.inference(imgs, infer_gs=True, ref_view_strategy="middle")
sf = p.scale_factor
log["scale_factor"] = sf
E = np.stack([to44(e) for e in p.extrinsics])
N, H, W = p.depth.shape
g = p.gaussians
means = g.means[0].numpy()
n0 = H * W
Xw0 = means[:n0]
zd = p.depth[0]


def cam_z(X):
    return (E[0] @ np.vstack([X.T, np.ones(len(X))]))[2].reshape(H, W)


zg = cam_z(Xw0)
r = zd / zg
log["view0_depth_over_gaussian_z_median"] = float(np.median(r))
log["view0_depth_over_gaussian_z_p10_p90"] = [float(np.percentile(r, 10)), float(np.percentile(r, 90))]
log["gaussian_view0_z_median"] = float(np.median(zg))
log["depth_view0_median"] = float(np.median(zd))
log["gaussians_per_view"] = n0
log["after_spec_correction_ratio_median"] = float(np.median(zd / cam_z(Xw0 * sf)))
log["gaussian_scale_median"] = float(np.median(g.scales[0].numpy()))

f = glob.glob(str(OUT / "*gs_export/gs_ply/0000.ply"))[0]
v = PlyData.read(f)["vertex"]
log["ply_fields"] = list(v.data.dtype.names)
log["ply_count"] = int(v.count)
q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
log["ply_rot_norm_mean"] = float(np.linalg.norm(q, axis=1).mean())
pm = np.stack([v["x"], v["y"], v["z"]], 1)
d, idx = cKDTree(means).query(pm[:5000])
mem = g.rotations[0].numpy()[idx]
log["ply_vs_memory_mean_dist_max"] = float(d.max())
log["rot_col_i_vs_memory_col_j_maxabs"] = {
    f"rot_{i}~mem_{j}": float(np.abs(q[:5000, i] - mem[:, j]).max()) for i in range(4) for j in range(4)
}
sc = np.stack([v[f"scale_{i}"] for i in range(3)], 1)[:5000]
log["ply_exp_scale_over_memory_scale"] = float(np.median(np.exp(sc) / g.scales[0].numpy()[idx]))
op = g.opacities[0].numpy()[idx]
log["ply_opacity_sigmoid_vs_memory_maxabs"] = float(np.abs(1 / (1 + np.exp(-v["opacity"][:5000])) - op).max())
log["ply_sh_fields"] = [x for x in v.data.dtype.names if x.startswith("f_")]
save("da3_nested_gs_analysis.json", log)
print(json.dumps(log, indent=1, default=str))
