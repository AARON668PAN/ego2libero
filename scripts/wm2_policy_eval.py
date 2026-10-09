"""Evaluate a policy inside world model v2 instead of the simulator.

The simulator is used once per episode, for the first observation: the same 50 benchmark layouts and
the same can shifts as scripts/shift_eval.sh. From there SmolVLA sees only decoded imagined frames and
the predicted robot state, its actions drive the world model, and the success head decides when the
can is in the basket. The estimate is compared with the simulator's result for the same policy, layout
and shift (outputs/rollouts/shift_<name>_r<cm>.json).

    python scripts/wm2_policy_eval.py --policy outputs/train/X/checkpoints/last/pretrained_model \
        --name phone_base --rename '{...}' --shifts 0 2 3 4 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
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
from ego2libero.world_model2 import (Denoiser2, History, StateModel, SuccessHead, decode, encode,  # noqa: E402
                                     load_vae, robot_to_state, sample, state_to_robot)
from rollout_policy import make_vec_env  # noqa: E402

IMGS = ("observation.images.image", "observation.images.image2")


def first_observations(env_cfg, shift_cm, episodes, batch, seed_base):
    """The observation each benchmark episode starts from, exactly as shift_eval.sh sees it."""
    env = make_vec_env(env_cfg, batch, shift_cm, False)
    obs_all = []
    for b in range(episodes // batch):
        seeds = [seed_base + b * batch + i for i in range(batch)]
        obs, _ = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
        obs_all.append(preprocess_observation(obs))
    task = list(env.call("task_description"))[0]
    env.close()
    return obs_all, task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--name", required=True, help="name used by shift_eval.sh for this policy")
    p.add_argument("--rename", default=None)
    p.add_argument("--shifts", nargs="+", type=float, default=[0, 2, 3, 4, 5])
    p.add_argument("--wm", default=str(ROOT / "outputs/wm2"))
    p.add_argument("--vae", default=str(ROOT / "models/vae/sd-vae-ft-mse"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--denoise-steps", type=int, default=3)
    p.add_argument("--videos", type=int, default=4, help="episodes per shift to save as video")
    p.add_argument("--out", default=str(ROOT / "outputs/wm2_eval"))
    a = p.parse_args()
    rename = json.loads(a.rename) if a.rename else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    wm = Path(a.wm)
    ck = torch.load(wm / "denoiser.pt", map_location="cuda")
    den = Denoiser2().cuda(); den.load_state_dict(ck["ema"]); den.eval(); sd = ck["sd"]
    sm = StateModel().cuda(); sm.load_state_dict(torch.load(wm / "state_model.pt", map_location="cuda")); sm.eval()
    hk = torch.load(wm / "success_head.pt", map_location="cuda")
    head = SuccessHead().cuda(); head.load_state_dict(hk["head"]); head.eval(); tau = hk["tau"]
    vae = load_vae(a.vae)

    env_cfg = LiberoEnv(task="libero_object", task_ids=[0], init_states=True, observation_height=256, observation_width=256)
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.pretrained_path = a.policy; cfg.n_action_steps = a.chunk; cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename); policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.policy, preprocessor_overrides={
        "device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": rename}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    for cm in a.shifts:
        t0 = time.time()
        firsts, task = first_observations(env_cfg, cm, a.episodes, 10, 1000)
        o0 = [env_pre(dict(o, task=[task] * len(o[IMGS[0]]))) for o in firsts]
        img = {k: torch.cat([o[k] for o in o0]) for k in IMGS}                     # (E,3,256,256) in [0,1]
        state = torch.cat([o["observation.state"] for o in o0]).double()
        E = len(state)
        z = torch.cat([encode(vae, img[k].cuda()) for k in IMGS], 1)               # (E,8,32,32)
        r = state_to_robot(state.cuda()).float()
        hist = History(z, r)
        policy.reset()
        alive = torch.ones(E, dtype=torch.bool, device="cuda"); won = torch.zeros_like(alive)
        steps_taken = torch.full((E,), a.max_steps, device="cuda")
        frames = []
        for t in range(a.max_steps):
            queue = getattr(policy, "_queues", {}).get(ACTION, []) if hasattr(policy, "_queues") else []
            if t == 0 or len(queue) == 0:                                           # the policy looks: show it the imagined frames
                if t > 0:
                    img = {k: decode(vae, hist.z[:, -1, 4 * j:4 * j + 4]).cpu() for j, k in enumerate(IMGS)}
            o = {IMGS[0]: img[IMGS[0]], IMGS[1]: img[IMGS[1]],
                 "observation.state": robot_to_state(r.double()).float().cpu(), "task": [task] * E}
            with torch.inference_mode():
                act = post(policy.select_action(pre(o)))
            act = env_post({ACTION: act})[ACTION].to("cuda").float()
            hist.push_action(act)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zn = sample(den, hist.context(), hist.a, r, sd, steps=a.denoise_steps).float()
            rn = sm(r, hist.r[:, -2], hist.a[:, -4:])
            hist.push_step(zn, rn); r = rn
            hit = alive & (torch.sigmoid(head(zn)) > tau)
            won |= hit; steps_taken[hit] = t + 1; alive &= ~hit
            if a.videos:
                frames.append(torch.cat([decode(vae, zn[:a.videos, 4 * j:4 * j + 4]) for j in range(2)], 3).cpu())
            if not alive.any():
                break
        truth_f = ROOT / "outputs/rollouts" / f"shift_{a.name}_r{cm:g}.json"
        truth = json.load(open(truth_f)) if truth_f.exists() else None
        sim = [e["success"] for e in truth["per_episode"]] if truth else None
        w = won.cpu().numpy()
        res = {"policy": a.policy, "name": a.name, "shift_cm": cm, "episodes": E, "wm_success": float(w.mean()),
               "sim_success": float(np.mean(sim)) if sim else None,
               "episode_agreement": float(np.mean(np.array(sim) == w)) if sim else None,
               "wm_steps_median": float(steps_taken[won].median()) if won.any() else None,
               "per_episode_wm": w.tolist(), "seconds": round(time.time() - t0)}
        json.dump(res, open(out / f"{a.name}_r{cm:g}.json", "w"), indent=1)
        print(f"{a.name} r{cm:g}: world model {w.mean():.0%}, simulator {res['sim_success']}, "
              f"same outcome on {res['episode_agreement']} of episodes, {res['seconds']}s", flush=True)
        if a.videos and frames:
            v = (torch.stack(frames, 1) * 255).byte().permute(0, 1, 3, 4, 2).numpy()   # (V,T,256,512,3)
            vw = cv2.VideoWriter(str(out / f"{a.name}_r{cm:g}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512 * 2, 256 * 2))
            for f in range(v.shape[1]):
                grid = np.concatenate([np.concatenate(list(v[i:i + 2, f]), 1) for i in range(0, min(4, len(v)), 2)], 0)
                if grid.shape[0] < 512:
                    grid = np.concatenate([grid, np.zeros((512 - grid.shape[0], *grid.shape[1:]), np.uint8)], 0)
                vw.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            vw.release()


if __name__ == "__main__":
    main()
