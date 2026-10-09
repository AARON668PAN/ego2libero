"""Turn each clip into a 3D pinch trajectory in the box frame, with grasp and release times.

Per frame:
  * Depth Anything V2 gives relative inverse depth. Its scale and offset are fitted, with
    RANSAC, to the true inverse depth of the desk, which is known from the calibrated camera
    pose. Off-desk things such as the keyboard drop out as outliers.
  * The pinch point is the midpoint of the thumb and index tips (MediaPipe, 2D). Its depth is
    the median depth over the two tips and the two joints behind them, which keeps the sample
    on the fingers rather than on whatever is between them.
Grasp and release come from the AirPods case track: the case starts moving when it is lifted
and stops once it rests in the box. The gripper closes at the slowest hand moment just before
lift-off and opens when the case comes to rest.

Writes data/processed/traj/<clip>.npz and outputs/traj/<clip>.png.

    python scripts/phone/extract_trajectory.py data/raw/sdr/*.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ego2libero.geometry import BOX_HEIGHT_MM, CAM, CASE_HALF_HEIGHT_MM, ROOT, K_for, backproject_to_plane  # noqa: E402

DEPTH_CKPT = "depth-anything/Depth-Anything-V2-Large-hf"
DESK_Z = -BOX_HEIGHT_MM
FPS = 30.0


def depth_maps(frames, model, proc, batch=8):
    out = []
    for i in range(0, len(frames), batch):
        chunk = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i:i + batch]]
        inputs = proc(images=chunk, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            pred = model(**inputs).predicted_depth.float()
        pred = torch.nn.functional.interpolate(pred[:, None], size=(540, 960), mode="bilinear", align_corners=False)[:, 0]
        out.append(pred.cpu().numpy())
    return np.concatenate(out)          # relative inverse depth at 960x540


def desk_fit(disp, K, R, t, exclude):
    """Fit true inverse depth of the desk = a * disparity + b, robustly."""
    ys, xs = np.mgrid[300:540:6, 0:960:6]
    keep = ~exclude[ys, xs]
    px = np.c_[xs[keep], ys[keep]] * 2.0                       # full-res pixels
    pts = backproject_to_plane(px, K, R, t, DESK_Z)
    zc = (pts @ R.T + t)[:, 2]
    ok = zc > 0
    d, inv = disp[ys[keep], xs[keep]][ok], 1.0 / zc[ok]
    best, rng = None, np.random.default_rng(0)
    for _ in range(200):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        a = (inv[i] - inv[j]) / (d[i] - d[j]); b = inv[i] - a * d[i]
        inl = np.abs(a * d + b - inv) < 0.03 * inv
        if best is None or inl.sum() > best.sum():
            best = inl
    A = np.c_[d[best], np.ones(best.sum())]
    a, b = np.linalg.lstsq(A, inv[best], rcond=None)[0]
    return a, b, best.mean()


def exclusion_mask(hand_img, case_q, box_px):
    m = np.zeros((540, 960), bool)
    if hand_img is not None:
        x0, y0 = hand_img[:, :2].min(0) / 2; x1, y1 = hand_img[:, :2].max(0) / 2
        pad = 0.6 * max(x1 - x0, y1 - y0)
        m[int(max(0, y0 - pad)):, int(max(0, x0 - pad)):] = True
    m |= cv2.dilate(cv2.resize(case_q.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST), np.ones((15, 15), np.uint8)) > 0
    box = np.zeros((540, 960), np.uint8)
    cv2.fillConvexPoly(box, cv2.convexHull((box_px / 2).astype(np.int32)), 1)
    m |= cv2.dilate(box, np.ones((21, 21), np.uint8)) > 0
    return m


def sample(disp, px, r=3):
    x, y = int(round(px[0] / 2)), int(round(px[1] / 2))
    win = disp[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return np.median(win) if win.size else np.nan


def plot(clip, t, P, close, open_, first, case_start, case_end, L, W, flip, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c, o = close - first, open_ - first
    for seg, col, lab in ((slice(0, c + 1), "tab:blue", "approach"), (slice(c, o + 1), "tab:red", "carry"), (slice(o, None), "tab:gray", "retreat")):
        ax[0].plot(P[seg, 0], P[seg, 1], color=col, label=lab)
    box = np.array([[0, 0], [L, 0], [L, W * flip], [0, W * flip], [0, 0]])
    ax[0].plot(box[:, 0], box[:, 1], "k-", lw=1, label="box rim")
    ax[0].plot(*case_start[:2], "go", label="case start"); ax[0].plot(*case_end[:2], "ms", label="case end")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x mm"); ax[0].set_ylabel("y mm"); ax[0].legend(fontsize=7); ax[0].set_title(f"{clip} top view, box frame")
    ax[1].plot(t, P[:, 2] - DESK_Z, "k")
    for e, col, lab in ((close, "g", "close"), (open_, "r", "open")):
        ax[1].axvline(e / FPS, color=col, ls="--", label=lab)
    ax[1].axhline(BOX_HEIGHT_MM, color="0.6", lw=1, label="box rim")
    ax[1].set_xlabel("s"); ax[1].set_ylabel("pinch height above desk, mm"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    args = p.parse_args()
    proc = AutoImageProcessor.from_pretrained(DEPTH_CKPT)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_CKPT, torch_dtype=torch.float16).to("cuda").eval()
    (ROOT / "data/processed/traj").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/traj").mkdir(parents=True, exist_ok=True)
    intr = json.loads((CAM / "intrinsics.json").read_text())
    L, W = intr["box_mm"]

    for video in sorted(map(Path, args.videos)):
        clip = video.stem
        cam = json.loads((CAM / f"{clip}.json").read_text())
        poses = np.load(CAM / f"{clip}_poses.npz")
        hands = np.load(ROOT / "data/processed/hands" / f"{clip}.npz")
        obj = np.load(ROOT / "data/processed/object" / f"{clip}.npz")
        masks = np.unpackbits(obj["mask_quarter"], axis=-1)[..., :obj["mask_shape"][2]].astype(bool)
        flip = int(poses["y_flip"])
        box3d = np.array([[x, y * flip, z] for z in (0, DESK_Z) for x, y in ((0, 0), (L, 0), (L, W), (0, W))])

        cap = cv2.VideoCapture(str(video)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        n = len(frames)
        disp = depth_maps(frames, model, proc)

        pinch = np.full((n, 3), np.nan); thumb = np.full((n, 3), np.nan); index = np.full((n, 3), np.nan)
        fit_inliers = np.full(n, np.nan)
        for k in range(n):
            if not hands["found"][k]:
                continue
            K, R, t = K_for(clip, k, cam), poses["R"][k], poses["t"][k]
            img = hands["img"][k][:, :2].astype(float)
            box_px, _ = cv2.projectPoints(box3d, cv2.Rodrigues(R)[0], t, K, None)
            a, b, frac = desk_fit(disp[k], K, R, t, exclusion_mask(img, masks[k], box_px.reshape(-1, 2)))
            fit_inliers[k] = frac
            zs = [1.0 / (a * sample(disp[k], img[j]) + b) for j in (3, 4, 7, 8)]
            z = np.median(zs)
            if not np.isfinite(z) or z <= 0:
                continue
            for j, out in ((4, thumb), (8, index)):
                ray = np.linalg.solve(K, [img[j][0], img[j][1], 1.0])
                out[k] = R.T @ (ray * z - t)          # camera -> box frame
            pinch[k] = (thumb[k] + index[k]) / 2

        # Case motion: warp each case mask back into frame 0 and measure how much of it still sits
        # on the starting (or final) footprint. Partial occlusion by the fingers keeps the visible
        # part on the footprint; lifting the case moves it off.
        Sq = np.diag([masks.shape[2] / 1920.0, masks.shape[1] / 1080.0, 1.0])
        stab = np.array([cv2.warpPerspective(masks[k].astype(np.uint8), Sq @ np.linalg.inv(poses["H"][k]) @ np.linalg.inv(Sq),
                                             (masks.shape[2], masks.shape[1]), flags=cv2.INTER_NEAREST) > 0 for k in range(n)])
        area = np.maximum(stab.sum((1, 2)), 1)
        on_start = (stab & stab[0]).sum((1, 2)) / area
        on_end = (stab & stab[-1]).sum((1, 2)) / area
        lift = next(k for k in range(1, n - 5) if (on_start[k:k + 5] < 0.5).all())
        rest = next(k for k in range(lift, n - 5) if (on_end[k:k + 5] > 0.8).all())
        c0 = obj["centroid"][0]

        # Hand span: frames where the hand is fully inside the image.
        inside = hands["found"] & np.all((hands["img"][:, :, 0] > 10) & (hands["img"][:, :, 0] < 1910) &
                                         (hands["img"][:, :, 1] > 10) & (hands["img"][:, :, 1] < 1070), axis=1)
        valid = inside & np.isfinite(pinch).all(1)
        first, last = np.flatnonzero(valid)[[0, -1]]
        tt = np.arange(n) / FPS
        span = np.arange(first, last + 1)
        P = np.array([np.interp(span, np.flatnonzero(valid), pinch[valid, d]) for d in range(3)]).T
        P = savgol_filter(P, 15, 2, axis=0)
        v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)] * FPS
        lo = max(first, lift - int(1.0 * FPS))
        close = lo + int(np.argmin(v[lo - first:lift - 3 - first + 1])) if lift - 3 > lo else lo
        open_ = rest
        grip = ((span >= close) & (span < open_)).astype(np.float32)

        # Case position at rest before and after, in the box frame.
        K0, R0, t0 = K_for(clip, 0, cam), poses["R"][0], poses["t"][0]
        case_start = backproject_to_plane(c0, K0, R0, t0, DESK_Z + CASE_HALF_HEIGHT_MM)[0]
        ke = n - 1
        case_end = backproject_to_plane(obj["centroid"][ke], K_for(clip, ke, cam), poses["R"][ke], poses["t"][ke],
                                        DESK_Z + 1.0 + CASE_HALF_HEIGHT_MM)[0]
        plot(clip, tt[span], P, close, open_, first, case_start, case_end, L, W, flip, ROOT / "outputs/traj" / f"{clip}.png")
        h_close = P[close - first, 2] - DESK_Z
        np.savez_compressed(ROOT / "data/processed/traj" / f"{clip}.npz", t=tt[span], frame=span, pos=P, grip=grip, close=close, open=open_, lift=lift, case_start=case_start, case_end=case_end,
                            raw_pinch=pinch, fit_inliers=fit_inliers, y_flip=flip, on_start=on_start, on_end=on_end)
        print(f"{clip}: frames {first}-{last}, close {close} ({tt[close]:.1f}s), lift {lift}, open {open_} ({tt[open_]:.1f}s); "
              f"pinch height at close {h_close:.0f} mm, max height {P[:, 2].max() - DESK_Z:.0f} mm, "
              f"pinch-to-case at close {np.linalg.norm(P[close - first, :2] - case_start[:2]):.0f} mm, "
              f"desk fit inliers {np.nanmedian(fit_inliers):.0%}")


if __name__ == "__main__":
    main()
