"""SmolVLA with world model v2 choosing its next action chunk, in the real simulator.

At every decision point (every 10 control steps) the policy samples K action chunks (different
flow-matching noise). World model v2 starts from the real history (both cameras encoded by the VAE,
robot state, past actions), imagines the first H steps of each chunk, and the value head scores where
each one ends up (an imagined step on which the success head fires scores 1). The best chunk's first
10 actions run in the simulator, then the policy looks again. --mode policy runs one sample as usual,
so both arms share this code, the 50 benchmark layouts and the can shifts of shift_eval.sh.

    python scripts/wm2_guided_eval.py --policy outputs/train/X/checkpoints/last/pretrained_model \
        --name phone_shift_base --rename '{...}' --mode wm --shifts 0 3 --tag pilot
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ego2libero.world_model2 import (ActionValueHead, Denoiser2, History, StateModel, SuccessHead, ValueHead,  # noqa: E402
                                     encode, load_vae, sample, state_to_robot)
from rollout_policy import make_vec_env  # noqa: E402

IMGS = ("observation.images.image", "observation.images.image2")


def branch(hist, n):
    """Copy a History n times per row (row-major, like repeat_interleave)."""
    h = History.__new__(History)
    h.z, h.r, h.a = (x.repeat_interleave(n, 0) for x in (hist.z, hist.r, hist.a))
    return h


@torch.no_grad()
def score_chunks(wm, hist, r_now, chunks, K, H, M, final=False):
    """chunks (n*K, 50, 7) in env units -> scores (n, K): mean over M imagined futures of each chunk."""
    den, sd, sm, head, tau, value = wm
    h = branch(hist, K * M)
    r = r_now.repeat_interleave(K * M, 0)
    acts = chunks.repeat_interleave(M, 0)
    done = torch.zeros(len(r), dtype=torch.bool, device=r.device)
    for s in range(H):
        h.push_action(acts[:, s])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zn = sample(den, h.context(), h.a, r, sd).float()
        rn = sm(r, h.r[:, -2], h.a[:, -4:])
        h.push_step(zn, rn); r = rn
        done |= torch.sigmoid(head(zn)) > tau
    v = torch.sigmoid(value(zn, r))
    sc = torch.where(done, torch.ones_like(v), v).reshape(-1, K, M).mean(-1)
    if final:                                   # imagined last step of each candidate's first sample
        return sc, zn.reshape(-1, K, M, *zn.shape[1:])[:, :, 0], r.reshape(-1, K, M, r.shape[-1])[:, :, 0]
    return sc


@torch.no_grad()
def imagine_checkpoints(wm, hist, r_now, chunks, K, M, cps):
    """Scores of M imagined futures of every chunk at each step count in cps: (n, K, M, C), and the predicted
    robot state there: (n, K, M, C, 9)."""
    den, sd, sm, head, tau, value = wm
    h = branch(hist, K * M)
    r = r_now.repeat_interleave(K * M, 0)
    acts = chunks.repeat_interleave(M, 0)
    done = torch.zeros(len(r), dtype=torch.bool, device=r.device)
    sc, rob = [], []
    for s in range(cps[-1]):
        h.push_action(acts[:, s])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zn = sample(den, h.context(), h.a, r, sd).float()
        rn = sm(r, h.r[:, -2], h.a[:, -4:])
        h.push_step(zn, rn); r = rn
        done |= torch.sigmoid(head(zn)) > tau
        if s + 1 in cps:
            v = torch.sigmoid(value(zn, r))
            sc.append(torch.where(done, torch.ones_like(v), v)); rob.append(r)
    sc = torch.stack(sc, -1).reshape(-1, K, M, len(cps))
    return sc, torch.stack(rob, 1).reshape(-1, K, M, len(cps), r.shape[-1])


@torch.no_grad()
def sim_checkpoints(env, env_pre, value, vae, chunks, K, cps, task):
    """Perfect foresight at each step count in cps: scores (n, K, C) and robot states (n, K, C, 9)."""
    n = len(task)
    plans = chunks.reshape(n, K, -1, 7)[:, :, :cps[-1]].cpu().numpy().astype(np.float64)
    res = env.call("preview_checkpoints", plans, list(cps))
    o = preprocess_observation(stack([c for obs, _ in res for cand in obs for c in cand]))
    o["task"] = [t for t in task for _ in range(K * len(cps))]; o = env_pre(o)
    N = len(o[IMGS[0]])                                                       # encode in slices: n*K*C images
    z = torch.cat([torch.cat([encode(vae, o[k][j:j + 64]) for j in range(0, N, 64)]) for k in IMGS], 1)
    r = state_to_robot(o["observation.state"].double().cuda()).float()
    v = torch.sigmoid(value(z, r))
    done = torch.tensor([d for _, ds in res for cand in ds for d in cand], device=v.device)
    sc = torch.where(done, torch.ones_like(v), v).reshape(n, K, len(cps))
    return sc, r.reshape(n, K, len(cps), -1)


@torch.no_grad()
def state_only_scores(wm, hist, r_now, z_now, chunks, K, H):
    """The world model without its image prediction: the value head sees the current real frame together with
    the robot state the state model predicts after H steps of each chunk. (n, K)"""
    den, sd, sm, head, tau, value = wm
    rh, ah = hist.r.repeat_interleave(K, 0), hist.a.repeat_interleave(K, 0)
    r = r_now.repeat_interleave(K, 0)
    for s in range(H):
        ah = torch.cat([ah[:, 1:], chunks[:, s][:, None]], 1)
        rn = sm(r, rh[:, -2], ah[:, -4:])
        rh = torch.cat([rh[:, 1:], rn[:, None]], 1); r = rn
    return torch.sigmoid(value(z_now.repeat_interleave(K, 0), r)).reshape(-1, K)


def stack(items):
    """List of (nested) observation dicts -> one dict of arrays with a leading batch axis."""
    if isinstance(items[0], dict):
        return {k: stack([d[k] for d in items]) for k in items[0]}
    return np.stack(items)


@torch.no_grad()
def score_in_sim(env, env_pre, value, vae, chunks, K, H, task, final=False):
    """Same score as score_chunks, but each candidate's first H steps run in the simulator."""
    n = len(task)
    plans = chunks.reshape(n, K, -1, 7)[:, :, :H].cpu().numpy().astype(np.float64)
    res = env.call("preview", plans, H)                                    # per env: (K observations, K done flags)
    o = preprocess_observation(stack([ob for obs, _ in res for ob in obs]))
    o["task"] = [t for t in task for _ in range(K)]; o = env_pre(o)
    z = torch.cat([encode(vae, o[k]) for k in IMGS], 1)
    r = state_to_robot(o["observation.state"].double().cuda()).float()
    v = torch.sigmoid(value(z, r))
    done = torch.tensor([d for _, ds in res for d in ds], device=v.device)
    sc = torch.where(done, torch.ones_like(v), v).reshape(n, K)
    return (sc, z.reshape(n, K, *z.shape[1:]), r.reshape(n, K, -1)) if final else sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--name", required=True, help="shift_eval.sh name of this policy, for the reference numbers")
    p.add_argument("--rename", default=None)
    p.add_argument("--mode", choices=["policy", "wm", "sim", "q", "compare", "compare2"], required=True,
                   help="sim: perfect foresight, each candidate is run in the simulator itself and the state restored; "
                        "q: no world model, a head scores the current state and the candidate's actions directly; "
                        "compare: run the policy's own first sample, log all three scores of every candidate; "
                        "compare2: the same with --samples imagined futures per chunk, scored at every --checkpoints step")
    p.add_argument("--k", type=int, default=8, help="candidate chunks per decision")
    p.add_argument("--horizon", type=int, default=20, help="imagined steps per candidate")
    p.add_argument("--samples", type=int, default=1, help="imagined futures per candidate, averaged")
    p.add_argument("--shifts", nargs="+", type=float, default=[0, 2, 3, 4, 5])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="torch seed for the policy's sampling noise")
    p.add_argument("--wm", default=str(ROOT / "outputs/wm2"))
    p.add_argument("--vae", default=str(ROOT / "models/vae/sd-vae-ft-mse"))
    p.add_argument("--tag", required=True)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[5, 10, 20, 40], help="compare2: steps ahead to score")
    p.add_argument("--seed-base", type=int, default=1000, help="1000 = the layouts and shifts of shift_eval.sh")
    a = p.parse_args()
    rename = json.loads(a.rename) if a.rename else {}
    out = ROOT / "outputs/wm2_guided"; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    env_cfg = LiberoEnv(task="libero_object", task_ids=[0], init_states=True, observation_height=256, observation_width=256)
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.pretrained_path = a.policy; cfg.n_action_steps = a.chunk; cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename); policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.policy, preprocessor_overrides={
        "device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": rename}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    K = 1 if a.mode == "policy" else a.k
    wd = Path(a.wm)
    if a.mode != "policy":
        vk = torch.load(wd / "value_head.pt", map_location="cuda")
        value = ValueHead().cuda(); value.load_state_dict(vk["head"]); value.eval()
        vae = load_vae(a.vae)
    if a.mode in ("wm", "compare", "compare2"):
        ck = torch.load(wd / "denoiser.pt", map_location="cuda")
        den = Denoiser2().cuda(); den.load_state_dict(ck["ema"]); den.eval()
        sm = StateModel().cuda(); sm.load_state_dict(torch.load(wd / "state_model.pt", map_location="cuda")); sm.eval()
        hk = torch.load(wd / "success_head.pt", map_location="cuda")
        head = SuccessHead().cuda(); head.load_state_dict(hk["head"]); head.eval()
        wm = (den, ck["sd"], sm, head, hk["tau"], value)
    if a.mode in ("q", "compare", "compare2"):
        qk = torch.load(wd / "qhead.pt", map_location="cuda")
        qhead = ActionValueHead(qk["horizon"]).cuda(); qhead.load_state_dict(qk["head"]); qhead.eval()

    for cm in a.shifts:
        env = make_vec_env(env_cfg, a.batch, cm, False)
        max_steps = env.call("_max_episode_steps")[0]
        n, results, decisions, t0 = a.batch, [], [], time.time()
        cmp_log = []
        for b in range(a.episodes // n):
            seeds = [a.seed_base + b * n + i for i in range(n)]
            obs, _ = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
            policy.reset()
            task = list(env.call("task_description"))
            done = np.zeros(n, bool); success = np.zeros(n, bool); steps = np.zeros(n, int)
            hist = None
            grip = [[] for _ in range(n)]                                     # executed gripper command per step
            for t in range(max_steps):
                o = preprocess_observation(obs); o["task"] = task; o = env_pre(o)
                if a.mode in ("wm", "q", "compare", "compare2"):
                    z = torch.cat([encode(vae, o[k]) for k in IMGS], 1)                    # (n,8,32,32)
                    r_now = state_to_robot(o["observation.state"].double().cuda()).float()
                    if hist is None:
                        hist = History(z, r_now)
                    else:
                        hist.push_step(z, r_now)
                if t % a.chunk == 0:
                    batch = pre(o)
                    rep = {k: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n else v)
                           for k, v in batch.items()}
                    with torch.inference_mode():
                        chunks = policy.predict_action_chunk(rep)                          # (n*K, 50, 7) normalised
                    chunks = env_post({ACTION: post(chunks)})[ACTION].float().cuda()       # env units
                    choice = np.zeros(n, int)
                    if a.mode != "policy":
                        if a.mode == "wm":
                            sc = score_chunks(wm, hist, r_now, chunks, K, a.horizon, a.samples).cpu().numpy()
                        elif a.mode == "compare2":
                            cps = sorted(a.checkpoints)
                            sw, rw = imagine_checkpoints(wm, hist, r_now, chunks, K, a.samples, cps)
                            ss, rs = sim_checkpoints(env, env_pre, value, vae, chunks, K, cps, task)
                            with torch.no_grad():
                                sq = torch.sigmoid(qhead(z.repeat_interleave(K, 0), r_now.repeat_interleave(K, 0),
                                                         chunks[:, :qhead.horizon].contiguous())).reshape(n, K)
                            eef = ((rw[:, :, 0, :, :3] - rs[..., :3]).norm(dim=-1) * 100).cpu().numpy()      # (n, K, C) cm
                            sw, ss, sq = (x.cpu().numpy() for x in (sw, ss, sq))
                            for i in np.flatnonzero(~done):
                                cmp_log.append({"episode": int(b * n + i), "t": int(t), "checkpoints": cps,
                                                "wm": np.round(sw[i], 4).tolist(), "sim": np.round(ss[i], 4).tolist(),
                                                "q": np.round(sq[i], 4).tolist(), "eef_err_cm": np.round(eef[i], 3).tolist()})
                            sc = np.zeros((n, K))                                     # run the policy's own first sample
                        elif a.mode == "compare":
                            sw, zw, rw = score_chunks(wm, hist, r_now, chunks, K, a.horizon, 1, final=True)
                            so = state_only_scores(wm, hist, r_now, z, chunks, K, a.horizon).cpu().numpy()
                            ss, zs, rs = score_in_sim(env, env_pre, value, vae, chunks, K, a.horizon, task, final=True)
                            with torch.no_grad():
                                sq = torch.sigmoid(qhead(z.repeat_interleave(K, 0), r_now.repeat_interleave(K, 0),
                                                         chunks[:, :qhead.horizon].contiguous())).reshape(n, K)
                            eef = ((rw[..., :3] - rs[..., :3]).norm(dim=-1) * 100).cpu().numpy()             # cm
                            fing = ((rw[..., 7:9] - rs[..., 7:9]).abs().sum(-1) * 1000).cpu().numpy()       # mm
                            lat = ((zw - zs) ** 2).mean((2, 3, 4)).cpu().numpy() / ck["sd"] ** 2
                            sw, ss, sq = (x.cpu().numpy() for x in (sw, ss, sq))
                            for i in np.flatnonzero(~done):
                                cmp_log.append({"episode": int(b * n + i), "t": int(t), "wm": np.round(sw[i], 4).tolist(),
                                                "sim": np.round(ss[i], 4).tolist(), "q": np.round(sq[i], 4).tolist(),
                                                "eef_err_cm": np.round(eef[i], 3).tolist(), "finger_err_mm": np.round(fing[i], 2).tolist(),
                                                "latent_err": np.round(lat[i], 4).tolist(), "state_only": np.round(so[i], 4).tolist(),
                                                "eef_z": float(r_now[i, 2]), "fingers": float(r_now[i, 7] - r_now[i, 8])})
                            sc = np.zeros((n, K))                                     # run the policy's own first sample
                        elif a.mode == "q":
                            with torch.no_grad():
                                sc = torch.sigmoid(qhead(z.repeat_interleave(K, 0), r_now.repeat_interleave(K, 0),
                                                         chunks[:, :qhead.horizon].contiguous())).reshape(n, K).cpu().numpy()
                        else:
                            sc = score_in_sim(env, env_pre, value, vae, chunks, K, a.horizon, task).cpu().numpy()
                        choice = sc.argmax(1)
                        for i in np.flatnonzero(~done):
                            decisions.append({"episode": int(b * n + i), "t": int(t), "scores": np.round(sc[i], 4).tolist(),
                                              "choice": int(choice[i])})
                    plan = chunks.reshape(n, K, -1, 7)[np.arange(n), choice]
                act = plan[:, t % a.chunk]
                if a.mode in ("wm", "q", "compare", "compare2"):
                    hist.push_action(act)
                for i in np.flatnonzero(~done):
                    grip[i].append(float(act[i, 6]))
                obs, _, term, trunc, info = env.step(act.cpu().numpy())
                succ = np.asarray(info.get("is_success", np.zeros(n)), bool)
                newly = ~done & (term | trunc | succ)
                success |= newly & succ; steps[newly] = t + 1; done |= newly
                if done.all():
                    break
            for i in range(n):
                g = np.array(grip[i]); closed = np.flatnonzero(g > 0)
                close_t = int(closed[0]) if len(closed) else -1
                reopen = np.flatnonzero(g[close_t:] < 0) if close_t >= 0 else []
                results.append({"episode": b * n + i, "seed": seeds[i], "success": bool(success[i]), "steps": int(steps[i]),
                                "close_t": close_t, "reopen_t": int(close_t + reopen[0]) if len(reopen) else -1})
            print(f"r{cm:g} batch {b}: {success.sum()}/{n}, running {np.mean([r['success'] for r in results]):.0%}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        env.close()
        ref = ROOT / "outputs/rollouts" / f"shift_{a.name}_r{cm:g}.json"
        ref_s = json.loads(ref.read_text())["success"] if ref.exists() else None
        summ = {"policy": a.policy, "name": a.name, "mode": a.mode, "shift_cm": cm, "k": K, "horizon": a.horizon,
                "samples": a.samples, "episodes": len(results), "success": float(np.mean([r["success"] for r in results])),
                "shift_eval_success": ref_s, "seconds": round(time.time() - t0)}
        (out / f"{a.tag}_{a.name}_{a.mode}_r{cm:g}.json").write_text(
            json.dumps({"summary": summ, "episodes": results, "decisions": decisions, "compare": cmp_log}, indent=1))
        print(f"{a.name} {a.mode} r{cm:g}: {summ['success']:.0%} (shift_eval {ref_s}), {summ['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
