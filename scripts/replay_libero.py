"""Replay retargeted human trajectories in LIBERO and keep the successful episodes.

For every (clip, scene) pair LIBERO samples a layout from the scene's seed (far from the test seeds),
the object and basket positions are read from the simulator, the clip is retargeted
onto them, and the robot is driven by a proportional controller on the end-effector position
with the gripper held pointing down. Episodes that end in task success are written to
data/processed/replay/<variant>/<clip>_<init>.npz in the format of the official LeRobot LIBERO
dataset: both images rotated by 180 degrees, 8-d state, 7-d action.

    python scripts/replay_libero.py --variant human --inits 0-49
    python scripts/replay_libero.py --variant scripted --inits 0-49

--shift-cm R moves the can U(0, R) cm in a random direction after each reset (ego2libero/shift_env.py),
so the demos grasp it in many places instead of one; the direction differs per clip and scene.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUITE, TASK_ID = "libero_object", 0
LOOKAHEAD = 2
MAX_STEPS = 300
GATE_TOL = {1.0: 0.012, -1.0: 0.02}   # position error allowed before closing / opening the gripper (m)
MAX_WAIT = 40
# Rigid-gripper clearances (m), from the can (6.3 cm wide, 7.6 cm tall) and basket (rim at 0.141 m)
# collision boxes. They act on the measured eef position, so they hold even when the arm lags.
APPROACH_CLEAR_Z, APPROACH_CENTRED = 0.11, 0.006   # gripper opens 8 cm, can is 6.3 cm: 8.5 mm spare per side
CARRY_Z, LEFT_GRASP, OVER_BASKET = 0.21, 0.03, 0.04
RETREAT_Z = 0.22


def shape_target(target, eef, tau, info, obj_now):
    """Clamp a trajectory target so a parallel gripper does not hit the can or the basket."""
    t = target.copy()
    dG = np.linalg.norm(eef[:2] - info["G"][:2]); dR = np.linalg.norm(eef[:2] - info["R"][:2])
    if tau <= info["close_idx"]:                       # approach: follow the object if it was nudged,
        t[:2] += obj_now[:2] - info["obj0"][:2]        # and descend only once centred over it
        dG = np.linalg.norm(eef[:2] - (info["G"][:2] + obj_now[:2] - info["obj0"][:2]))
        centred = 0.012 if info.get("funnel") else APPROACH_CENTRED   # the funnel path already keeps clear
        if dG > centred:
            t[2] = max(t[2], APPROACH_CLEAR_Z)
    elif tau <= info["open_idx"]:                      # carry: travel above the basket rim
        if dG > LEFT_GRASP and dR > OVER_BASKET:
            t[2] = max(t[2], CARRY_Z)
            if eef[2] < CARRY_Z - 0.03:
                t[:2] = eef[:2]                        # rise before moving sideways
    else:                                              # retreat: clear the basket vertically first
        t[2] = max(t[2], RETREAT_Z)
        if eef[2] < RETREAT_Z - 0.03:
            t[:2] = eef[:2]
    return t
ROT_SCALE = 0.5                     # OSC_POSE rotation limit for an action of 1
SPEED_PER_ACTION = 0.0129           # measured steady-state eef speed (m/step) for an action of 1, same on x, y, z
KP = 0.4                            # feedback gain on the position error, per step


def quat2axisangle(q):
    w = np.clip(q[3], -1.0, 1.0); den = np.sqrt(max(1.0 - w * w, 0.0))
    return np.zeros(3) if den < 1e-10 else q[:3] / den * 2.0 * np.arccos(w)


def state8(obs):
    rs = obs["robot_state"]
    return np.r_[rs["eef"]["pos"], quat2axisangle(rs["eef"]["quat"]), rs["gripper"]["qpos"]].astype(np.float32)


_ENV = None
SEED_BASE = 100000   # training scenes are sampled with seeds far from anything the benchmark uses


def raw_pos(sim, name):
    return sim._get_observations()[f"{name}_pos"].copy()


def get_env(shift_cm=0.0):
    """One env per worker; layouts are sampled from the reset seed."""
    global _ENV
    if _ENV is None:
        from libero.libero import benchmark
        from ego2libero.shift_env import ShiftedLiberoEnv
        suite = benchmark.get_benchmark_dict()[SUITE]()
        _ENV = ShiftedLiberoEnv(suite, TASK_ID, SUITE, obs_type="pixels_agent_pos", init_states=False,
                                shift_cm=shift_cm, shift_uniform=True)
    return _ENV


def run_one(job):
    from ego2libero.retarget import flatten_rhythm, retarget, scripted, straighten_path
    variant, clip, init_id, out_dir, kw = job
    env = get_env(kw["shift_cm"])
    # with a shift, give every clip its own directions (the seed sets layout and shift)
    clip_no = int("".join(c for c in clip if c.isdigit()) or 0) if kw["shift_cm"] > 0 else 0
    obs, reset_info = env.reset(seed=SEED_BASE + init_id + 1000 * clip_no)
    sim = env._env.env
    raw = sim._get_observations()
    obj_name, tgt_name = sim.obj_of_interest
    obj_pos, basket_pos = raw[f"{obj_name}_pos"], raw[f"{tgt_name}_pos"]
    eef0, q_ref = obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy()
    R0 = Rotation.from_quat(q_ref)                     # the gripper holds this orientation throughout
    if variant == "scripted":
        path, grip, info = scripted(obj_pos, basket_pos, eef0)
    else:
        traj = dict(np.load(ROOT / "data/processed/traj" / f"{clip}.npz"))
        if kw["rhythm"] == "flat":
            traj = flatten_rhythm(traj)
        if kw["shape"] == "straight":
            traj = straighten_path(traj)
        path, grip, info = retarget(traj, obj_pos, basket_pos, eef0, speed=kw["speed"])
    info["obj0"] = obj_pos.copy()
    info["shift_cm"] = float(np.linalg.norm(reset_info["shift"]) * 100)

    imgs, wrists, states, actions, objs = [], [], [], [], []
    success, steps, tau, waited, total_wait = False, 0, 0, 0, 0
    clamp_dev = []
    n = len(path)
    for k in range(MAX_STEPS):
        i = min(tau, n - 1)
        g = grip[i] if tau < n else -1.0
        rs = obs["robot_state"]
        # Wait at a gripper change until the arm has caught up with the trajectory.
        changing = tau < n - 1 and grip[i + 1] != grip[i]
        goal = path[i].copy()
        if tau <= info["close_idx"]:
            goal[:2] += raw_pos(sim, obj_name)[:2] - info["obj0"][:2]
        if changing and np.linalg.norm(goal - rs["eef"]["pos"]) > GATE_TOL[grip[i + 1]] and waited < MAX_WAIT:
            target, waited, total_wait = path[i], waited + 1, total_wait + 1
        else:
            target, waited, tau = path[min(i + LOOKAHEAD, n - 1)], 0, tau + 1
        ff = path[min(i + LOOKAHEAD + 1, n - 1)] - path[min(i + LOOKAHEAD, n - 1)] if waited == 0 else np.zeros(3)
        rot_err = (R0 * Rotation.from_quat(rs["eef"]["quat"]).inv()).as_rotvec()
        shaped = shape_target(target, rs["eef"]["pos"], tau, info, raw_pos(sim, obj_name))
        phase = 0 if tau <= info["close_idx"] else (1 if tau <= info["open_idx"] else 2)
        clamp_dev.append((phase, np.linalg.norm(shaped - target)))
        if np.linalg.norm(shaped[:2] - target[:2]) > 1e-6:   # a clearance rule overrode the path: drop its velocity
            ff[:2] = 0.0
        if abs(shaped[2] - target[2]) > 1e-6:
            ff[2] = 0.0
        target = shaped
        a = np.zeros(7, np.float32)
        # feed-forward on the path velocity plus a gentle correction, so commands stay below the limit
        a[:3] = np.clip((ff + KP * (target - rs["eef"]["pos"])) / SPEED_PER_ACTION, -1, 1)
        a[3:6] = np.clip(rot_err / ROT_SCALE, -1, 1)
        a[6] = g
        imgs.append(obs["pixels"]["image"][::-1, ::-1].copy()); wrists.append(obs["pixels"]["image2"][::-1, ::-1].copy())
        states.append(state8(obs)); actions.append(a); objs.append(raw_pos(sim, obj_name))
        obs, _, terminated, _, step_info = env.step(a)
        steps = k + 1
        if step_info["is_success"]:
            success = True
            break
        if tau >= n + 30:
            break
    info["waited"] = total_wait
    ph, dev = (np.array([c[0] for c in clamp_dev]), np.array([c[1] for c in clamp_dev])) if clamp_dev else (np.zeros(0), np.zeros(0))
    info["clamp_frac"] = float((dev > 0.002).mean()) if len(dev) else 0.0
    info["clamp_dev_mm"] = float(dev[dev > 0.002].mean() * 1000) if (dev > 0.002).any() else 0.0
    for j, name in enumerate(("approach", "carry", "retreat")):
        sel = ph == j
        info[f"steps_{name}"] = int(sel.sum())
        info[f"clamp_frac_{name}"] = float((dev[sel] > 0.002).mean()) if sel.any() else 0.0
    if success:
        np.savez_compressed(Path(out_dir) / f"{clip}_{init_id:02d}.npz", image=np.array(imgs), image2=np.array(wrists),
                            state=np.array(states), action=np.array(actions), task=env.task_description,
                            obj_pos=np.array(objs, np.float32), basket_pos=np.asarray(basket_pos, np.float32))
    return {"clip": clip, "init": init_id, "success": success, "steps": steps, "planned": len(path),
            **{k: float(v) for k, v in info.items() if np.ndim(v) == 0}}



def parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["human", "scripted"], default="human")
    p.add_argument("--rhythm", choices=["human", "flat"], default="human",
                   help="ablation, human variant: flat keeps the path but moves along each segment at constant speed")
    p.add_argument("--shape", choices=["human", "straight"], default="human",
                   help="ablation, human variant: straight keeps the timing but makes each segment a straight line")
    p.add_argument("--inits", default="0-49", help="scene indices; each one is a sampled layout seed")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--shift-cm", type=float, default=0.0, help="move the can U(0, this) cm after each reset")
    args = p.parse_args()
    clips = sorted(f.stem for f in (ROOT / "data/processed/traj").glob("IMG_*.npz"))
    if args.variant == "scripted":
        clips = ["scripted"]
    name = args.variant + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "data/processed/replay" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"speed": args.speed, "shift_cm": args.shift_cm, "rhythm": args.rhythm, "shape": args.shape}
    jobs = [(args.variant, c, i, str(out_dir), kw) for c in clips for i in parse_range(args.inits)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=0))
    by = {}
    for r in results:
        by.setdefault(r["clip"], []).append(r)
    for c, rs in by.items():
        ok = [r for r in rs if r["success"]]
        print(f"{c}: {len(ok)}/{len(rs)} succeeded, mean steps of successes {np.mean([r['steps'] for r in ok]) if ok else float('nan'):.0f}")
    print(f"total {sum(r['success'] for r in results)}/{len(results)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
