"""Map a human pinch trajectory onto a LIBERO pick-and-place scene.

The human trajectory lives in the box frame (mm). Two anchors tie it to the scene: the pinch
point when the hand closes on the case, and the pinch point when it lets go. A planar
similarity (rotation plus uniform scale) sends the human carry, anchor to anchor, onto the
line from the simulated object to the basket. The approach and the retreat are only
rotated, so their size stays metric. Heights keep the human profile around each anchor, and
the carry blends linearly from the grasp height to the release height so the object is let go
above the basket.
"""
import numpy as np

SIM_HZ = 20.0
GRASP_Z = 0.051          # eef height at grasp, median of the official LIBERO demos for this task (m)
RELEASE_Z = 0.16         # eef height at release, same source
MIN_Z = 0.015            # never command the gripper lower than this
CARRY_APEX = 0.25        # the human lift is scaled so its peak reaches this eef height (demos peak near 0.28 m)
CLEAR_Z = 0.11           # eef height that keeps the open fingers above the can's top (0.076 m)


V_MAX = 0.010             # m per control step; the teleoperated demos peak near 0.012
GRASP_DWELL = 8          # control steps to hold still after closing the gripper


def retime(path, grip, v_max=V_MAX):
    """Stretch time wherever the path is faster than the arm can follow, keeping its shape."""
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    stretch = np.maximum(1.0, d / v_max)
    t_old = np.r_[0.0, np.cumsum(stretch)]             # new time of each original sample
    t_new = np.arange(0.0, t_old[-1] + 1e-9, 1.0)
    path_n = np.array([np.interp(t_new, t_old, path[:, j]) for j in range(3)]).T
    idx = np.clip(np.searchsorted(t_old, t_new, side="right") - 1, 0, len(grip) - 1)
    return path_n, grip[idx]


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def retarget(traj, obj_pos, basket_pos, eef_start, speed=1.0, approach_radius=0.15, retreat_s=0.5, bridge_steps=20):
    """Return per-step eef targets (T, 3) in metres and gripper commands (T,) in {-1, +1}.
    The gripper keeps the robot's starting orientation."""
    frames = traj["frame"]; P = traj["pos"] / 1000.0
    c = int(traj["close"]) - int(frames[0]); o = int(traj["open"]) - int(frames[0])
    Gh, Rh = P[c], P[o]
    Gs = np.array([obj_pos[0], obj_pos[1], GRASP_Z])
    Rs = np.array([basket_pos[0], basket_pos[1], RELEASE_Z])

    dh, ds = Rh[:2] - Gh[:2], Rs[:2] - Gs[:2]
    theta = np.arctan2(ds[1], ds[0]) - np.arctan2(dh[1], dh[0])
    k = np.linalg.norm(ds) / max(np.linalg.norm(dh), 1e-6)
    Rm = rot2(theta)

    out = np.zeros_like(P)
    out[:c + 1, :2] = Gs[:2] + (P[:c + 1, :2] - Gh[:2]) @ Rm.T
    out[:c + 1, 2] = Gs[2] + (P[:c + 1, 2] - Gh[2])
    u = np.linspace(0, 1, o - c + 1)
    # Scale the human lift so its peak clears the basket the way the teleoperated demos do;
    # the shape and timing of the lift stay the human's.
    rel = P[c:o + 1, 2] - Gh[2] - u * (Rh[2] - Gh[2])
    lift_gain = max(1.0, (CARRY_APEX - Gs[2]) / max(rel.max(), 1e-3))
    # Along the carry direction the human path is stretched to the scene's distance and made
    # monotone (no back-and-forth); across it the human deviation keeps its real size, so a
    # small adjustment near the box does not become a detour of tens of centimetres.
    e = dh / max(np.linalg.norm(dh), 1e-6); nrm = np.array([-e[1], e[0]])
    rel = P[c:o + 1, :2] - Gh[:2]
    along = np.maximum.accumulate(np.clip(rel @ e, 0, None)) * k
    across = rel @ nrm
    es, ns = Rm @ e, Rm @ nrm
    out[c:o + 1, :2] = Gs[:2] + along[:, None] * es + across[:, None] * ns
    out[c:o + 1, 2] = Gs[2] + lift_gain * (P[c:o + 1, 2] - Gh[2] - u * (Rh[2] - Gh[2])) + u * (Rs[2] - Gs[2])
    out[o:, :2] = Rs[:2] + (P[o:, :2] - Rh[:2]) @ Rm.T
    out[o:, 2] = Rs[2] + (P[o:, 2] - Rh[2])
    # Fixed grasp ending: from 1.4 s to 0.6 s before the grasp, blend the human path into the point
    # above the object at CLEAR_Z; in the last 0.6 s go straight down. Earlier, keep the human
    # path but above CLEAR_Z. After the release, rise straight up by 6 cm in 0.5 s.
    a, pre = max(0, c - 42), max(0, c - 18)
    out[:pre, 2] = np.maximum(out[:pre, 2], CLEAR_Z)
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pre - a)) if pre > a else np.zeros(0)
    out[a:pre, :2] = (1 - w[:, None]) * out[a:pre, :2] + w[:, None] * Gs[:2]
    out[pre:c + 1, :2] = Gs[:2]
    v = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, c + 1 - pre))
    out[pre:c + 1, 2] = CLEAR_Z + v * (Gs[2] - CLEAR_Z)
    r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, min(15, len(P) - o)))
    out[o:o + len(r), :2] = Rs[:2]
    out[o:o + len(r), 2] = Rs[2] + 0.06 * r
    out[o + len(r):, :2] = Rs[:2]
    out[o + len(r):, 2] = Rs[2] + 0.06
    out[:, 2] = np.maximum(out[:, 2], MIN_Z)

    # Start the approach where the hand comes within approach_radius of the grasp point,
    # and stop retreat_s after the release.
    d = np.linalg.norm(out[:c + 1, :2] - Gs[:2], axis=1)
    start = int(np.argmax(d < approach_radius)) if (d < approach_radius).any() else 0
    end = min(len(P) - 1, o + int(retreat_s * 30))
    t = np.arange(len(P)) / 30.0
    t_sim = np.arange(t[start], t[end], speed / SIM_HZ)
    path = np.array([np.interp(t_sim, t[start:end + 1], out[start:end + 1, j]) for j in range(3)]).T
    grip = np.where((t_sim >= t[c]) & (t_sim < t[o]), 1.0, -1.0)

    # Bridge from the robot's starting pose to the first target with a smooth ease-in-out.
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, bridge_steps))
    bridge = eef_start + s[:, None] * (path[0] - eef_start)
    grip = np.r_[-np.ones(bridge_steps), grip]
    path, grip = retime(np.vstack([bridge, path]), grip)
    # Hold still while the fingers close: a human lifts as the pinch closes, the Panda gripper
    # needs about 8 control steps to shut.
    ci = int(np.argmax(grip > 0))
    path = np.vstack([path[:ci + 1], np.repeat(path[ci:ci + 1], GRASP_DWELL, 0), path[ci + 1:]])
    grip = np.r_[grip[:ci + 1], np.ones(GRASP_DWELL), grip[ci + 1:]]
    oi = ci + int(np.argmax(grip[ci:] < 0))
    return path, grip, {"theta": theta, "scale": k, "lift_gain": lift_gain,
                        "close_idx": ci, "open_idx": oi, "G": Gs, "R": Rs, "funnel": True}


def scripted(obj_pos, basket_pos, eef_start, carry_z=0.25, hover_z=0.15):
    """Straight-line pick and place through the same two anchors, for comparison."""
    Gs = np.array([obj_pos[0], obj_pos[1], GRASP_Z]); Rs = np.array([basket_pos[0], basket_pos[1], RELEASE_Z])
    keys = [(eef_start, -1, 0), (np.r_[Gs[:2], hover_z], -1, 30), (Gs, -1, 20), (Gs, 1, 12),
            (np.r_[Gs[:2], carry_z], 1, 20), (np.r_[Rs[:2], carry_z], 1, 35), (Rs, 1, 15), (Rs, -1, 10),
            (np.r_[Rs[:2], carry_z], -1, 15)]
    path, grip = [], []
    for (a, _, _), (b, g, n) in zip(keys[:-1], keys[1:]):
        s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, endpoint=False))
        path.append(np.asarray(a) + s[:, None] * (np.asarray(b) - np.asarray(a))); grip.append(np.full(n, g, float))
    path, grip = np.vstack(path), np.concatenate(grip)
    path, grip = retime(path, grip)
    ci = int(np.argmax(grip > 0)); oi = ci + int(np.argmax(grip[ci:] < 0))
    return path, grip, {"close_idx": ci, "open_idx": oi, "G": Gs, "R": Rs}


def flatten_rhythm(traj):
    """The same human path at constant speed within each segment (approach, carry, retreat): the rhythm goes,
    the shape, the segment durations and the grasp and release frames stay. For the rhythm ablation."""
    out = dict(traj)
    P = np.asarray(traj["pos"], float).copy()
    f0 = int(traj["frame"][0]); c = int(traj["close"]) - f0; o = int(traj["open"]) - f0
    for a, b in ((0, c), (c, o), (o, len(P) - 1)):
        seg = P[a:b + 1]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        if b - a < 2 or s[-1] < 1e-9:
            continue
        u = np.linspace(0.0, s[-1], len(seg))
        P[a:b + 1] = np.array([np.interp(u, s, seg[:, j]) for j in range(3)]).T
    out["pos"] = P
    return out


def straighten_path(traj, approach_radius_mm=150.0, retreat_frames=15):
    """The same human timing and heights, but the horizontal path of the parts retarget() uses (the approach from where
    the hand comes within approach_radius of the grasp point, the carry, and the first retreat_frames after the release)
    replaced by straight lines between their ends, walked with the hand's own horizontal progress. The direction the
    hand comes from stays the human one. For the shape ablation."""
    out = dict(traj)
    P = np.asarray(traj["pos"], float).copy()
    f0 = int(traj["frame"][0]); c = int(traj["close"]) - f0; o = int(traj["open"]) - f0
    d = np.linalg.norm(P[:c + 1, :2] - P[c, :2], axis=1)
    start = int(np.argmax(d < approach_radius_mm)) if (d < approach_radius_mm).any() else 0
    for a, b in ((start, c), (c, o), (o, min(len(P) - 1, o + retreat_frames))):
        seg = P[a:b + 1, :2]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        if b - a < 2 or s[-1] < 1e-9:
            continue
        P[a:b + 1, :2] = seg[0] + (s / s[-1])[:, None] * (seg[-1] - seg[0])
    out["pos"] = P
    return out
