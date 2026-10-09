"""Fit the probe that reads the gripper and can positions off a 64 px front-camera frame (ego2libero/probe.py).

Training frames come from the simulator with the true positions: moved-can replays of the clips and of the
script, plus some of the same episodes with their actions edited and run open loop again (gripper never
closes, approach pushed sideways, gripper opened mid-carry). The edited episodes matter. In successes alone the
can's height follows the gripper's after the grasp, and a probe could read the gripper instead of the can.
Clips IMG_3564 and IMG_3565 are held out and give the error.

    python scripts/wm2_probe_fit.py          # -> outputs/wm2/probe.pt, about 10 minutes
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ego2libero.probe import LIFT_Z, RES, train_probe  # noqa: E402

SEED_BASE = 100000          # scene seeds of scripts/replay_libero.py
SPEED_PER_ACTION = 0.0129   # m/step of gripper motion for an action of 1
SHIFT_CM = 6                # the replays used here were made with the can moved U(0, 6) cm
HELD_OUT = ("IMG_3564", "IMG_3565")
EDITS = ("no_grasp", "miss", "drop")


def scene_seed(stem):
    """The reset seed replay_libero.py used for episode file <clip>_<init> with the can moved."""
    clip, init = stem.rsplit("_", 1)
    digits = "".join(c for c in clip if c.isdigit())
    return SEED_BASE + int(init) + 1000 * (int(digits) if digits else 0)


def edit_actions(act, edit, rng):
    a = act.copy()
    closing = a[:, 6] > 0
    close = int(np.argmax(closing)) if closing.any() else len(a) // 3
    reopen = close + int(np.argmax(a[close:, 6] < 0)) if (a[close:, 6] < 0).any() else len(a)
    if edit == "no_grasp":
        a[:, 6] = -1.0
    elif edit == "miss":
        ang, size, n = rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 0.09), 15
        s0 = max(0, close - 35)
        a[s0:s0 + n, :2] += size / n / SPEED_PER_ACTION * np.array([np.cos(ang), np.sin(ang)])
    elif edit == "drop":
        a[close + int(rng.uniform(0.3, 0.7) * (reopen - close)):, 6] = -1.0
    elif edit != "replay":
        raise ValueError(edit)
    a[:, :6] = np.clip(a[:, :6], -1, 1)
    return a


def load_episode(f):
    """Recorded replay -> 64 px frames, gripper xyz, can xyz, actions."""
    with np.load(f) as d:
        img, st, obj, act = d["image"], d["state"], d["obj_pos"], d["action"]
    frames = np.stack([cv2.resize(x, (RES, RES), interpolation=cv2.INTER_AREA) for x in img])
    return frames, st[:, :3].astype(np.float32), obj.astype(np.float32), act.astype(np.float32)


_ENV = None


def rollout(job):
    """Run actions open loop in the scene of `seed`; returns 64 px frames, gripper xyz, can xyz per step."""
    global _ENV
    seed, actions = job
    if _ENV is None:
        from libero.libero import benchmark
        from ego2libero.shift_env import ShiftedLiberoEnv
        _ENV = ShiftedLiberoEnv(benchmark.get_benchmark_dict()["libero_object"](), 0, "libero_object",
                                obs_type="pixels_agent_pos", init_states=False, shift_cm=SHIFT_CM, shift_uniform=True)
    obs, _ = _ENV.reset(seed=seed)
    sim = _ENV._env.env
    obj = sim.obj_of_interest[0]
    frames, eef, pos = [], [], []
    for a in actions:
        frames.append(cv2.resize(obs["pixels"]["image"][::-1, ::-1], (RES, RES), interpolation=cv2.INTER_AREA))
        eef.append(obs["robot_state"]["eef"]["pos"].copy())
        pos.append(sim._get_observations()[f"{obj}_pos"].copy())
        try:
            obs, _, _, _, _ = _ENV.step(a)
        except ValueError:          # robosuite horizon reached
            break
    return np.array(frames), np.array(eef, np.float32), np.array(pos, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-per-clip", type=int, default=20)
    p.add_argument("--scripted", type=int, default=80)
    p.add_argument("--edited", type=int, default=60, help="episodes that also run with each of the edits")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "outputs/wm2/probe.pt"))
    a = p.parse_args()
    rng = np.random.default_rng(0)
    R = ROOT / "data/processed/replay"
    files = []
    for clip in sorted({f.stem.rsplit("_", 1)[0] for f in (R / "human_v3_fixed_shift").glob("*.npz")}):
        fs = sorted((R / "human_v3_fixed_shift").glob(f"{clip}_*.npz"))
        files += list(rng.choice(fs, min(a.scenes_per_clip if clip not in HELD_OUT else 10, len(fs)), replace=False))
    files += list(rng.choice(sorted((R / "scripted_v3_shift").glob("*.npz")), a.scripted, replace=False))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        eps = pool.map(load_episode, files, chunksize=4)
    held = np.array([f.stem.rsplit("_", 1)[0] in HELD_OUT for f in files])
    print(f"{len(files)} recorded episodes ({held.sum()} held out), {sum(len(e[0]) for e in eps)} frames, {time.time() - t0:.0f}s", flush=True)

    # edited episodes, run again in the same scenes; a few unedited replays check that the scene is the same
    pick = [i for i in rng.permutation(len(files)) if not held[i]][:a.edited] + [i for i in range(len(files)) if held[i]][:10]
    jobs, kinds = [], []
    for i in pick[:4]:
        jobs.append((scene_seed(files[i].stem), eps[i][3])); kinds.append((i, "replay"))
    for i in pick:
        for e in EDITS:
            jobs.append((scene_seed(files[i].stem), edit_actions(eps[i][3], e, rng))); kinds.append((i, e))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        runs = pool.map(rollout, jobs, chunksize=1)
    print(f"{len(runs)} edited episodes in {time.time() - t0:.0f}s", flush=True)
    for (i, e), (_, _, pos) in zip(kinds, runs):
        if e == "replay":
            n = min(len(pos), len(eps[i][2]))
            print(f"  scene check {files[i].stem}: can position differs by at most {np.abs(pos[:n] - eps[i][2][:n]).max() * 1000:.2f} mm", flush=True)

    tr_f, tr_y, te_f, te_y, te_edit = [], [], [], [], []
    for (fr, eef, pos, _), h in zip(eps, held):
        (te_f if h else tr_f).append(fr); (te_y if h else tr_y).append(np.concatenate([eef, pos], 1))
    for (i, e), (fr, eef, pos) in zip(kinds, runs):
        if e == "replay":
            continue
        y = np.concatenate([eef, pos], 1)
        if held[i]:
            te_f.append(fr); te_y.append(y); te_edit.append(np.full(len(fr), True))
        else:
            tr_f.append(fr); tr_y.append(y)
    tr_f, tr_y = np.concatenate(tr_f), np.concatenate(tr_y)
    te_f, te_y = np.concatenate(te_f), np.concatenate(te_y)
    te_edit = np.concatenate([np.full(sum(len(f) for f, h in zip([e[0] for e in eps], held) if h), False)] + te_edit)
    print(f"training frames {len(tr_f)}, held-out frames {len(te_f)} ({te_edit.sum()} from edited episodes)", flush=True)
    read = train_probe(tr_f, tr_y, a.out)
    x = torch.from_numpy(te_f).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
    pred = read(x).cpu().numpy()
    err = np.linalg.norm(pred.reshape(-1, 2, 3) - te_y.reshape(-1, 2, 3), axis=2) * 100
    lifted_ok = ((pred[:, 5] > LIFT_Z) == (te_y[:, 5] > LIFT_Z))
    print(f"held out: gripper error median {np.median(err[:, 0]):.2f} cm, can error median {np.median(err[:, 1]):.2f} cm; "
          f"can lifted or not read right on {lifted_ok.mean():.1%} of frames, {lifted_ok[te_edit].mean():.1%} on edited episodes -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
