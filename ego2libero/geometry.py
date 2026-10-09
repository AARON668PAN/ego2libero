"""Camera and box geometry shared by the data scripts.

Box frame: origin at rim corner 0 of the upside-down iPhone box lid, x along the long rim
edge, y along the short one, z up. The rim is the lid's open edge, BOX_HEIGHT_MM above the
desk, so the desk is the plane z = -BOX_HEIGHT_MM. All lengths are in millimetres.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "data" / "processed" / "camera"
BOX_HEIGHT_MM = 29.0      # user-measured iPhone 12 Pro box height
CASE_HALF_HEIGHT_MM = 10.85   # AirPods Pro 2 case lying flat: 21.7 mm tall


def intrinsics():
    return json.loads((CAM / "intrinsics.json").read_text())


def K_for(clip, frame, cam=None):
    """Pinhole matrix for one frame, with the optical centre the iPhone recorded for it."""
    cam = cam or json.loads((CAM / f"{clip}.json").read_text())
    f = intrinsics()["fx"]
    cx, cy = np.array(cam["optical_center_normalized"][min(frame, len(cam["optical_center_normalized"]) - 1)])
    return np.array([[f, 0, cx * cam["width"]], [0, f, cy * cam["height"]], [0, 0, 1.0]])


def box_object_points():
    intr = intrinsics()
    L, W = intr["box_mm"]
    return np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], float)


def box_pose(corners_px, K):
    """Rotation and translation taking box-frame points to camera coordinates, z pointing up."""
    obj = box_object_points()
    for flip in (1, -1):
        o = obj * [1, flip, 1]
        _, rvec, tvec = cv2.solvePnP(o, np.asarray(corners_px, float), K, None, flags=cv2.SOLVEPNP_IPPE)
        R, _ = cv2.Rodrigues(rvec)
        centre = -R.T @ tvec.ravel()
        if centre[2] > 0:   # camera above the rim plane
            return R, tvec.ravel(), flip
    raise RuntimeError("could not find a box pose with the camera above the box")


def backproject_to_plane(px, K, R, t, z):
    """Intersect the camera ray through pixel(s) px with the box-frame plane at height z."""
    px = np.atleast_2d(px).astype(float)
    rays_cam = np.linalg.solve(K, np.c_[px, np.ones(len(px))].T).T
    rays = rays_cam @ R              # rotate rays into the box frame (R^T applied row-wise)
    centre = -R.T @ t
    s = (z - centre[2]) / rays[:, 2]
    return centre + s[:, None] * rays
