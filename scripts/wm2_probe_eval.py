"""Does world model v2 follow unusual actions, and where? At every decision of the moved-can policy's own
rollouts, the policy's chunk and six edited versions of it run for one second, imagined by the world model and
in the simulator from the same saved state:
  own      the policy's chunk
  freeze   no motion, gripper held as it is
  open     the policy's chunk with the gripper kept open (never closes, or lets go)
  close    the policy's chunk with the gripper kept closed (closes early, or never lets go)
  shifted  the policy's chunk drifting about 3 cm sideways
  other    another episode's chunk from the same step
  reverse  the policy's chunk with the motion reversed
Logged: value scores, the gripper position, and the can position, which the simulator gives exactly and is read
off imagined front-camera frames by the probe of ego2libero/probe.py (also run on the simulator's own frames, to
separate the probe's error from the world model's).

    python scripts/wm2_probe_eval.py --policy outputs/train/X/checkpoints/last/pretrained_model --rename '{...}' \
        --shifts 0 3 5 --episodes 30 --tag probe
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
from ego2libero.world_model2 import (Denoiser2, History, StateModel, SuccessHead, ValueHead, decode, encode,  # noqa: E402
                                     load_vae, state_to_robot)
from rollout_policy import make_vec_env  # noqa: E402
from wm2_guided_eval import IMGS, score_chunks, stack  # noqa: E402
from ego2libero.probe import frame64, load_probe  # noqa: E402

PROBES = ["own", "freeze", "open", "close", "shifted", "other", "reverse"]


def edits(own, last_grip, rng):
    """own (n, 50, 7) env-unit chunks -> (n, 7, 50, 7) in the order of PROBES."""
    n = own.shape[0]
    fr = torch.zeros_like(own); fr[..., 6] = last_grip[:, None]
    op = own.clone(); op[..., 6] = -1.0
    cl = own.clone(); cl[..., 6] = 1.0
    sh = own.clone()
    ang = torch.tensor(rng.uniform(0, 2 * np.pi, n), dtype=own.dtype, device=own.device)
    sh[:, :20, 0] += 0.12 * torch.cos(ang)[:, None]; sh[:, :20, 1] += 0.12 * torch.sin(ang)[:, None]   # ~3 cm over 1 s
    ot = own.roll(1, 0)
    rv = own.clone(); rv[..., :3] *= -1
    return torch.stack([own, fr, op, cl, sh, ot, rv], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--rename", default=None)
    p.add_argument("--shifts", nargs="+", type=float, default=[0, 3, 5])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--wm", default=str(ROOT / "outputs/wm2"))
    p.add_argument("--vae", default=str(ROOT / "models/vae/sd-vae-ft-mse-libero"))
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    rename = json.loads(a.rename) if a.rename else {}
    torch.manual_seed(0); rng = np.random.default_rng(0)
    env_cfg = LiberoEnv(task="libero_object", task_ids=[0], init_states=True, observation_height=256, observation_width=256)
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.pretrained_path = a.policy; cfg.n_action_steps = 10; cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename); policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.policy, preprocessor_overrides={
        "device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": rename}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    wd = Path(a.wm)
    ck = torch.load(wd / "denoiser.pt", map_location="cuda")
    den = Denoiser2().cuda(); den.load_state_dict(ck["ema"]); den.eval()
    sm = StateModel().cuda(); sm.load_state_dict(torch.load(wd / "state_model.pt", map_location="cuda")); sm.eval()
    hk = torch.load(wd / "success_head.pt", map_location="cuda")
    head = SuccessHead().cuda(); head.load_state_dict(hk["head"]); head.eval()
    value = ValueHead().cuda(); value.load_state_dict(torch.load(wd / "value_head.pt", map_location="cuda")["head"]); value.eval()
    wm = (den, ck["sd"], sm, head, hk["tau"], value)
    vae = load_vae(a.vae)
    read = load_probe(ROOT / "outputs/wm2/probe.pt")
    def can_from(img):                                    # (B,3,256,256) in [0,1] -> probe's can xyz (B,3)
        f = torch.from_numpy(np.stack([frame64(x) for x in img])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
        return read(f)[:, 3:].cpu().numpy()
    K, n, H = len(PROBES), a.batch, a.horizon
    out = ROOT / "outputs/wm2_guided"; out.mkdir(parents=True, exist_ok=True)
    for cm in a.shifts:
        env = make_vec_env(env_cfg, n, cm, False)
        max_steps = env.call("_max_episode_steps")[0]
        log, results, t0 = [], [], time.time()
        for b in range(a.episodes // n):
            seeds = [1000 + b * n + i for i in range(n)]
            obs, _ = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
            policy.reset()
            task = list(env.call("task_description"))
            done = np.zeros(n, bool); success = np.zeros(n, bool); hist = None
            last_grip = torch.full((n,), -1.0, device="cuda"); grip_hist = [[] for _ in range(n)]
            for t in range(max_steps):
                o = preprocess_observation(obs); o["task"] = task; o = env_pre(o)
                z = torch.cat([encode(vae, o[k]) for k in IMGS], 1)
                r_now = state_to_robot(o["observation.state"].double().cuda()).float()
                hist = History(z, r_now) if hist is None else (hist.push_step(z, r_now) or hist)
                if t % 10 == 0:
                    with torch.inference_mode():
                        own = policy.predict_action_chunk(pre(o))
                    own = env_post({ACTION: post(own)})[ACTION].float().cuda()                  # (n, 50, 7)
                    ch = edits(own, last_grip, rng)                                             # (n, K, 50, 7)
                    flat = ch.reshape(n * K, -1, 7)
                    sw, zw, rw = score_chunks(wm, hist, r_now, flat, K, H, 1, final=True)      # (n,K) (n,K,8,32,32) (n,K,9)
                    img_w = decode(vae, zw.reshape(n * K, *zw.shape[2:])[:, :4])
                    can_w = can_from(img_w).reshape(n, K, 3)
                    can_now_probe = can_from(o[IMGS[0]])
                    res = env.call("preview_probe", flat.reshape(n, K, -1, 7)[:, :, :H].cpu().numpy().astype(np.float64), H)
                    so = preprocess_observation(stack([ob for obs_, _, _, _ in res for ob in obs_]))
                    so["task"] = [tk for tk in task for _ in range(K)]; so = env_pre(so)
                    zs = torch.cat([torch.cat([encode(vae, so[k][j:j + 64]) for j in range(0, n * K, 64)]) for k in IMGS], 1)
                    rs = state_to_robot(so["observation.state"].double().cuda()).float()
                    with torch.no_grad():
                        vs = torch.sigmoid(value(zs, rs)).reshape(n, K).cpu().numpy()
                    d_s = np.array([d for _, ds, _, _ in res for d in ds]).reshape(n, K)
                    vs = np.where(d_s, 1.0, vs)
                    can_s = np.array([c for _, _, cs, _ in res for c in cs]).reshape(n, K, 3)
                    can_now = np.array([cn for _, _, _, cn in res])
                    can_pr = can_from(so[IMGS[0]]).reshape(n, K, 3)
                    eef_w = rw[..., :3].cpu().numpy(); eef_s = rs[:, :3].reshape(n, K, 3).cpu().numpy()
                    sw = sw.cpu().numpy()
                    for i in np.flatnonzero(~done):
                        log.append({"episode": int(b * n + i), "t": int(t), "wm": np.round(sw[i], 4).tolist(),
                                    "sim": np.round(vs[i], 4).tolist(), "done": d_s[i].tolist(),
                                    "can_now": np.round(can_now[i], 4).tolist(), "can_now_probe": np.round(can_now_probe[i], 4).tolist(),
                                    "can_sim": np.round(can_s[i], 4).tolist(), "can_wm": np.round(can_w[i], 4).tolist(),
                                    "can_probe_on_sim": np.round(can_pr[i], 4).tolist(),
                                    "eef_wm": np.round(eef_w[i], 4).tolist(), "eef_sim": np.round(eef_s[i], 4).tolist()})
                    plan = own
                act = plan[:, t % 10]
                hist.push_action(act); last_grip = act[:, 6].clone()
                for i in np.flatnonzero(~done):
                    grip_hist[i].append(float(act[i, 6]))
                obs, _, term, trunc, info = env.step(act.cpu().numpy())
                succ = np.asarray(info.get("is_success", np.zeros(n)), bool)
                newly = ~done & (term | trunc | succ); success |= newly & succ; done |= newly
                if done.all():
                    break
            for i in range(n):
                g = np.array(grip_hist[i]); closed = np.flatnonzero(g > 0)
                ct = int(closed[0]) if len(closed) else -1
                ro = np.flatnonzero(g[ct:] < 0) if ct >= 0 else []
                results.append({"episode": b * n + i, "success": bool(success[i]), "close_t": ct,
                                "reopen_t": int(ct + ro[0]) if len(ro) else -1})
            print(f"r{cm:g} batch {b}: {success.sum()}/{n}, {time.time() - t0:.0f}s", flush=True)
        env.close()
        (out / f"{a.tag}_r{cm:g}.json").write_text(json.dumps({"shift_cm": cm, "probes": PROBES, "episodes": results,
                                                                "decisions": log}))
        print(f"PROBEDONE r{cm:g} {len(log)} decisions", flush=True)


if __name__ == "__main__":
    main()
