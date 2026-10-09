"""Estimate the focal length from the iPhone box lid seen in the first frame of each clip.

The lid is a rectangle. Its rim corners start from rough positions picked by eye
(scripts/phone/annotations/box_corners_rough.json) and are refined by fitting a line to the
strongest edge along each side and intersecting neighbouring lines. Focal length and the
lid's aspect ratio are then fit jointly over all clips by minimising the reprojection
error of the corners, with the principal point taken from the per-frame optical centre
the iPhone records. Clips marked unreliable are left out of the fit.

    python scripts/phone/calibrate_from_box.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
CAM = ROOT / "data" / "processed" / "camera"
LONG_SIDE_MM = 164.0   # user-measured iPhone 12 Pro box; only sets the scale, the fit finds the aspect


def first_frame(clip):
    cap = cv2.VideoCapture(str(ROOT / "data" / "raw" / "sdr" / f"{clip}.mp4"))
    ok, frame = cap.read()
    cap.release()
    return frame


def refine(quad, gray, search):
    """Move each side onto the strongest nearby edge and return the four intersections."""
    g = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(float)
    gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    h, w = gray.shape
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        offs = np.arange(-search, search + 0.01, 0.5)
        pts = []
        for s in np.linspace(0.15, 0.85, 50):
            q = a + s * (b - a) + offs[:, None] * n
            xs = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            pts.append(q[np.argmax(np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1]))])
        pts = np.array(pts)
        for _ in range(4):   # refit without outliers
            c = pts.mean(0)
            dirn = np.linalg.svd(pts - c)[2][0]
            r = (pts - c) @ np.array([-dirn[1], dirn[0]])
            mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            keep = np.abs(r - np.median(r)) < max(1.0, 2.5 * mad)
            if keep.sum() < 8:
                break
            pts = pts[keep]
        c = pts.mean(0)
        lines.append((c, np.linalg.svd(pts - c)[2][0]))
    out = []
    for i in range(4):
        (c1, d1), (c2, d2) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.array([d1, -d2]).T, c2 - c1)
        out.append(c1 + t[0] * d1)
    return np.array(out)


def reprojection_sse(params, obs):
    f, aspect = params
    obj = np.array([[0, 0, 0], [LONG_SIDE_MM, 0, 0], [LONG_SIDE_MM, LONG_SIDE_MM / aspect, 0],
                    [0, LONG_SIDE_MM / aspect, 0]])
    sse = 0.0
    for q, c in obs:
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
        _, rvec, tvec = cv2.solvePnP(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        sse += ((proj.reshape(-1, 2) - q) ** 2).sum()
    return sse


def main():
    rough = json.loads((Path(__file__).resolve().parent / "annotations" / "box_corners_rough.json").read_text())
    corners, obs = {}, []
    for clip, entry in rough.items():
        if clip.startswith("_"):
            continue
        gray = cv2.cvtColor(first_frame(clip), cv2.COLOR_BGR2GRAY)
        quad = refine(np.array(entry["rough"], float), gray, search=14)
        for i, xy in entry.get("fix", {}).items():
            quad[int(i)] = xy
        quad = refine(quad, gray, search=6)
        corners[clip] = {"corners": quad.round(2).tolist(), "unreliable": entry.get("unreliable")}
        if not entry.get("unreliable"):
            cam = json.loads((CAM / f"{clip}.json").read_text())
            obs.append((quad, np.array(cam["optical_center_normalized"][0]) * [cam["width"], cam["height"]]))

    fit = minimize(reprojection_sse, [1500.0, 1.8], args=(obs,), method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-4})
    f, aspect = fit.x
    rmse = np.sqrt(fit.fun / (4 * len(obs)))
    loo = [minimize(reprojection_sse, fit.x, args=(obs[:k] + obs[k + 1:],), method="Nelder-Mead",
                    options={"xatol": 0.5, "fatol": 1e-4}).x[0] for k in range(len(obs))]
    jack = np.sqrt((len(loo) - 1) / len(loo) * ((np.array(loo) - np.mean(loo)) ** 2).sum())

    (CAM / "box_corners_f0.json").write_text(json.dumps(corners, indent=1))
    result = {
        "fx": round(f, 1), "fy": round(f, 1), "width": 1920, "height": 1080,
        "principal_point": "per frame: optical_center_normalized in <clip>.json times (width, height)",
        "focal_jackknife_std_px": round(jack, 1), "corner_rmse_px": round(rmse, 2),
        "box_aspect": round(aspect, 3), "box_mm": [LONG_SIDE_MM, round(LONG_SIDE_MM / aspect, 1)],
        "clips_used": len(obs),
    }
    (CAM / "intrinsics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
