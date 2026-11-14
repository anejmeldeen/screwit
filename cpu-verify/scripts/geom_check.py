"""Reprojection check: does depth of view A, warped with the given extrinsics, land on the same image content in view B?"""
import numpy as np

def to44(E):
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 4):
        E = np.vstack([E, [0, 0, 0, 1]])
    return E

def photo_error(img_a, img_b, depth_a, K_a, K_b, A_to_B, conf=None, stride=4):
    """A_to_B: 4x4 mapping camera-A coordinates to camera-B coordinates. Returns (mean abs gray diff, frac in-bounds)."""
    H, W = depth_a.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_a[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    if conf is not None:
        ok &= conf[ys, xs] >= np.percentile(conf, 50)
    pix = np.stack([xs[ok], ys[ok], np.ones(ok.sum())], 0).astype(np.float64)
    Xa = (np.linalg.inv(K_a) @ pix) * z[ok]
    Xb = A_to_B[:3, :3] @ Xa + A_to_B[:3, 3:4]
    front = Xb[2] > 1e-3
    p = K_b @ Xb
    u = p[0] / np.where(front, p[2], 1); v = p[1] / np.where(front, p[2], 1)
    inb = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    if inb.sum() < 50:
        return float("nan"), float(inb.mean())
    ga = img_a.astype(np.float64).mean(-1); gb = img_b.astype(np.float64).mean(-1)
    va = ga[pix[1, inb].astype(int), pix[0, inb].astype(int)]
    vb = gb[np.clip(np.round(v[inb]).astype(int), 0, H - 1), np.clip(np.round(u[inb]).astype(int), 0, W - 1)]
    return float(np.abs(va - vb).mean()), float(inb.mean())

def convention_report(images, depth, conf, E, K):
    """Compare hypotheses: E is world->camera (R,t) vs E is camera->world, over all ordered neighbouring pairs."""
    res = {"w2c": [], "c2w_hyp": [], "identity_control": []}
    n = len(images)
    for a in range(n):
        for b in range(n):
            if abs(a - b) not in (1, 2):
                continue
            Ea, Eb = to44(E[a]), to44(E[b])
            hyp = {
                "w2c": Eb @ np.linalg.inv(Ea),
                "c2w_hyp": np.linalg.inv(Eb) @ Ea,
                "identity_control": np.eye(4),
            }
            for k, M in hyp.items():
                e, f = photo_error(images[a], images[b], depth[a], K[a], K[b], M, conf[a])
                res[k].append(e)
    return {k: float(np.nanmean(v)) for k, v in res.items()}

def camera_centres(E):
    out = []
    for e in E:
        e = to44(e); R, t = e[:3, :3], e[:3, 3]
        out.append((-R.T @ t).tolist())
    return out
