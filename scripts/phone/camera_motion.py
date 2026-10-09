"""Estimate the camera pose in the box frame for every frame of each clip.

The phone moves a little while recording. Background features of each frame are matched
to frame 0, with the hand, forearm and AirPods case masked out, and a RANSAC homography
gives the image motion. Frame 0's pose comes from the box corners (scripts/calibrate_from_box.py);
virtual points on the rim plane are pushed through each homography and a PnP solve gives that
frame's pose.

Writes data/processed/camera/<clip>_poses.npz: R (N,3,3) and t (N,3) mapping box-frame mm to
camera coordinates, plus the homographies and their inlier counts.

    python scripts/phone/camera_motion.py data/raw/sdr/*.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ego2libero.geometry import CAM, ROOT, K_for, box_pose  # noqa: E402

S = 0.5   # features are computed at half resolution


def feature_mask(shape_full, hand_img, case_small):
    h, w = int(shape_full[0] * S), int(shape_full[1] * S)
    m = np.full((h, w), 255, np.uint8)
    if hand_img is not None and not np.isnan(hand_img).any():
        x0, y0 = hand_img[:, :2].min(0) * S
        x1, y1 = hand_img[:, :2].max(0) * S
        pad = 0.6 * max(x1 - x0, y1 - y0)
        m[int(max(0, y0 - pad)):, int(max(0, x0 - pad)):] = 0   # hand plus the forearm, which enters bottom-right
    if case_small is not None:
        cm = cv2.resize(case_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        m[cv2.dilate(cm, np.ones((25, 25), np.uint8)) > 0] = 0
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    args = p.parse_args()
    corners = json.loads((CAM / "box_corners_f0.json").read_text())
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for video in sorted(map(Path, args.videos)):
        clip = video.stem
        cam = json.loads((CAM / f"{clip}.json").read_text())
        hands = np.load(ROOT / "data/processed/hands" / f"{clip}.npz")
        obj = np.load(ROOT / "data/processed/object" / f"{clip}.npz")
        masks = np.unpackbits(obj["mask_quarter"], axis=-1)[..., :obj["mask_shape"][2]].astype(bool)

        cap = cv2.VideoCapture(str(video))
        Hs, inliers, k0, d0 = [], [], None, None
        prev_H, prev_kd = np.eye(3), None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(frame, None, fx=S, fy=S, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            hand = hands["img"][i] if hands["found"][i] else None
            k, d = orb.detectAndCompute(g, feature_mask(frame.shape, hand, masks[i]))
            if i == 0:
                k0, d0 = k, d
                H, n_in = np.eye(3), len(k)
            else:
                H, n_in = None, 0
                m = bf.match(d0, d) if d is not None else []
                if len(m) >= 30:
                    p0 = np.float32([k0[x.queryIdx].pt for x in m]) / S
                    p1 = np.float32([k[x.trainIdx].pt for x in m]) / S
                    H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 6.0)
                    n_in = int(inl.sum()) if H is not None else 0
                if H is None or n_in < 60:      # fall back to chaining through the previous frame
                    kp, dp = prev_kd
                    m = bf.match(dp, d) if d is not None else []
                    pp = np.float32([kp[x.queryIdx].pt for x in m]) / S
                    p1 = np.float32([k[x.trainIdx].pt for x in m]) / S
                    Hstep, inl = cv2.findHomography(pp, p1, cv2.RANSAC, 6.0)
                    H, n_in = Hstep @ prev_H, -int(inl.sum())   # negative count marks a chained estimate
            Hs.append(H / H[2, 2]); inliers.append(n_in)
            prev_H, prev_kd = H, (k, d)
            i += 1
        cap.release()

        # Frame 0 pose from the box, then every other frame through its homography.
        K0 = K_for(clip, 0, cam)
        R0, t0, flip = box_pose(corners[clip]["corners"], K0)
        gx, gy = np.meshgrid(np.linspace(-150, 300, 10), np.linspace(-150, 250, 9) * flip)
        grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
        px0, _ = cv2.projectPoints(grid, cv2.Rodrigues(R0)[0], t0, K0, None)
        Rs, ts = [], []
        for f, H in enumerate(Hs):
            pxf = cv2.perspectiveTransform(px0, H).reshape(-1, 2)
            _, rvec, tvec = cv2.solvePnP(grid, pxf, K_for(clip, f, cam), None, flags=cv2.SOLVEPNP_IPPE)
            Rs.append(cv2.Rodrigues(rvec)[0]); ts.append(tvec.ravel())
        Rs, ts = np.array(Rs), np.array(ts)
        centres = -np.einsum("nji,nj->ni", Rs, ts)
        np.savez_compressed(CAM / f"{clip}_poses.npz", R=Rs, t=ts, H=np.array(Hs), inliers=np.array(inliers), y_flip=flip)
        ins = np.array(inliers)
        print(f"{clip}: {len(Hs)} frames, direct matches {np.mean(ins > 0):.0%} (median inliers {int(np.median(ins[ins > 0]))}), "
              f"camera height above rim {centres[0, 2]:.0f} mm, moved up to {np.linalg.norm(centres - centres[0], axis=1).max():.0f} mm")


if __name__ == "__main__":
    main()
