"""Do the world model's picks raise the chance that the episode ends in success?

The policy runs its own first sample as usual. At a few fixed decision points the simulator is saved and
each of the K candidate chunks is played out to the end of the episode R times: its first 10 actions, then
the policy as usual. That gives every candidate a measured success rate to compare with the world model's
score (imagined second + value head), the scorer without a world model, and the one-second value of the
simulator itself. The episode then resumes from the saved state.

    python scripts/wm2_outcome_check.py --policy outputs/train/X/checkpoints/last/pretrained_model \
        --rename '{...}' --shifts 3 --episodes 60 --tag outcome
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
                                     encode, load_vae, state_to_robot)
from rollout_policy import make_vec_env  # noqa: E402
from wm2_guided_eval import IMGS, imagine_checkpoints, sim_checkpoints, stack  # noqa: E402


def play_out(env, policy, pre, post, env_pre, env_post, task, first, steps_left):
    """Run first (n, 10, 7) and then the policy until each sub-env succeeds or runs out of steps."""
    n = len(task)
    won = np.zeros(n, bool)
    res = None
    for j in range(first.shape[1]):
        res = env.call("raw_step", first[:, j])
        won |= np.array([d for _, d in res])
    policy.reset()
    for _ in range(steps_left - first.shape[1]):
        if won.all():
            break
        o = preprocess_observation(stack([ob for ob, _ in res])); o["task"] = task; o = env_pre(o)
        with torch.inference_mode():
            act = post(policy.select_action(pre(o)))
        act = env_post({ACTION: act})[ACTION].cpu().numpy()
        res = env.call("raw_step", act)
        won |= np.array([d for _, d in res])
    return won


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--rename", default=None)
    p.add_argument("--shifts", nargs="+", type=float, default=[3])
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--repeats", type=int, default=2, help="play-outs per candidate")
    p.add_argument("--at", nargs="+", type=int, default=[40, 80, 120], help="decision steps that get play-outs")
    p.add_argument("--samples", type=int, default=3, help="imagined futures per candidate")
    p.add_argument("--wm", default=str(ROOT / "outputs/wm2"))
    p.add_argument("--vae", default=str(ROOT / "models/vae/sd-vae-ft-mse"))
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    rename = json.loads(a.rename) if a.rename else {}
    torch.manual_seed(0)
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
    qk = torch.load(wd / "qhead.pt", map_location="cuda")
    qhead = ActionValueHead(qk["horizon"]).cuda(); qhead.load_state_dict(qk["head"]); qhead.eval()
    wm = (den, ck["sd"], sm, head, hk["tau"], value)
    vae = load_vae(a.vae)
    K, n = a.k, a.batch
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
            for t in range(max_steps):
                o = preprocess_observation(obs); o["task"] = task; o = env_pre(o)
                z = torch.cat([encode(vae, o[k]) for k in IMGS], 1)
                r_now = state_to_robot(o["observation.state"].double().cuda()).float()
                if hist is None:
                    hist = History(z, r_now)
                else:
                    hist.push_step(z, r_now)
                if t % 10 == 0:
                    batch = pre(o)
                    rep = {k: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n else v)
                           for k, v in batch.items()}
                    with torch.inference_mode():
                        chunks = policy.predict_action_chunk(rep)
                    chunks = env_post({ACTION: post(chunks)})[ACTION].float().cuda()            # (n*K, 50, 7)
                    if t in a.at and (~done).any():
                        sw, _ = imagine_checkpoints(wm, hist, r_now, chunks, K, a.samples, [20])
                        ss, _ = sim_checkpoints(env, env_pre, value, vae, chunks, K, [20], task)
                        with torch.no_grad():
                            sq = torch.sigmoid(qhead(z.repeat_interleave(K, 0), r_now.repeat_interleave(K, 0),
                                                     chunks[:, :qhead.horizon].contiguous())).reshape(n, K)
                        ch = chunks.reshape(n, K, -1, 7)[:, :, :10].cpu().numpy()
                        env.call("snapshot")
                        wins = np.zeros((n, K, a.repeats))
                        for k in range(K):
                            for rr in range(a.repeats):
                                env.call("restore_snapshot")
                                wins[:, k, rr] = play_out(env, policy, pre, post, env_pre, env_post, task, ch[:, k], max_steps - t)
                        env.call("restore_snapshot"); policy.reset()
                        sw, ss, sq = sw[..., 0].cpu().numpy(), ss[..., 0].cpu().numpy(), sq.cpu().numpy()
                        for i in np.flatnonzero(~done):
                            log.append({"episode": int(b * n + i), "t": int(t), "wm": np.round(sw[i], 4).tolist(),
                                        "sim1s": np.round(ss[i], 4).tolist(), "q": np.round(sq[i], 4).tolist(),
                                        "success": wins[i].tolist()})
                        print(f"  batch {b} step {t}: play-outs done, {time.time() - t0:.0f}s", flush=True)
                    plan = chunks.reshape(n, K, -1, 7)[:, 0]                                    # the policy's own sample
                act = plan[:, t % 10]
                hist.push_action(act)
                obs, _, term, trunc, info = env.step(act.cpu().numpy())
                succ = np.asarray(info.get("is_success", np.zeros(n)), bool)
                newly = ~done & (term | trunc | succ); success |= newly & succ; done |= newly
                if done.all():
                    break
            results += [{"episode": b * n + i, "success": bool(success[i])} for i in range(n)]
            print(f"r{cm:g} batch {b}: {success.sum()}/{n}, {time.time() - t0:.0f}s", flush=True)
        env.close()
        (out / f"{a.tag}_r{cm:g}.json").write_text(json.dumps({"shift_cm": cm, "episodes": results, "decisions": log}))
        print(f"OUTCOMEDONE r{cm:g} {len(log)} decisions", flush=True)


if __name__ == "__main__":
    main()
