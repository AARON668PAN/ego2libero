"""Roll out a trained policy in LIBERO and keep its episodes.

Observations go through the same pipeline lerobot-eval uses. Successful episodes are saved in the format of
our generated data (images rotated 180 degrees, 8-d state, 7-d action), so build_dataset.py can pack them.

    python scripts/rollout_policy.py --policy outputs/train/X/checkpoints/last/pretrained_model \
        --episodes 200 --tag example

With --shift-cm the soup can is moved that far from its usual spot after every reset (direction
from the seed, see ego2libero/shift_env.py). --benchmark uses LIBERO's 50 fixed layouts and
--no-save only scores the policy, so the same script is also the shifted-can evaluation.

For world model v2, --vae DIR --save-latents keeps every episode, failures included, as VAE latents
under data/processed/wm2/rollouts.
"""
import argparse
import json
import os
import sys
import time
from functools import partial
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
sys.path.insert(0, str(ROOT))


def to_uint8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def make_vec_env(env_cfg, n, shift_cm, shift_uniform):
    """The vector env make_env builds, with ShiftedLiberoEnv as the sub-env (identical when shift_cm is 0)."""
    from libero.libero import benchmark
    from lerobot.envs.utils import _LazyAsyncVectorEnv
    from ego2libero.shift_env import ShiftedLiberoEnv
    kw = dict(env_cfg.gym_kwargs, task_suite=benchmark.get_benchmark_dict()[env_cfg.task](), task_id=0,
              task_suite_name=env_cfg.task, camera_name=env_cfg.camera_name, init_states=env_cfg.init_states,
              n_envs=n, control_mode=env_cfg.control_mode, shift_cm=shift_cm, shift_uniform=shift_uniform)
    kw.pop("task_ids", None)
    return _LazyAsyncVectorEnv([partial(ShiftedLiberoEnv, episode_index=i, **kw) for i in range(n)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=500000)   # far from the eval seed and from data generation
    p.add_argument("--tag", required=True)
    p.add_argument("--rename", default=None, help="JSON rename map if the policy uses other camera names")
    p.add_argument("--shift-cm", type=float, default=0.0, help="move the can this far after each reset")
    p.add_argument("--shift-uniform", action="store_true", help="draw the distance from U(0, shift-cm)")
    p.add_argument("--benchmark", action="store_true", help="LIBERO's 50 fixed layouts instead of sampled ones")
    p.add_argument("--no-save", action="store_true", help="only score the policy, keep no episodes")
    p.add_argument("--vae", default=None, help="VAE directory for --save-latents")
    p.add_argument("--save-latents", action="store_true", help="keep all episodes as VAE latents")
    args = p.parse_args()
    vae = None
    if args.vae:
        from ego2libero.world_model2 import encode, load_vae
        vae = load_vae(args.vae)
    IMGS = ("observation.images.image", "observation.images.image2")
    lat_out = ROOT / "data/processed/wm2/rollouts" / args.tag
    if args.save_latents:
        args.no_save = True
        lat_out.mkdir(parents=True, exist_ok=True)
        for f in lat_out.glob("*.npz"):
            f.unlink()
    rename = json.loads(args.rename) if args.rename else {}
    out = ROOT / "data/processed/replay" / f"policy_{args.tag}"
    if not args.no_save:
        if out.exists():                  # never mix with files from an earlier run
            for f in out.glob("*.npz"):
                f.unlink()
        out.mkdir(parents=True, exist_ok=True)

    env_cfg = LiberoEnv(task="libero_object", task_ids=[0], init_states=args.benchmark,
                        observation_height=256, observation_width=256)
    env = make_vec_env(env_cfg, args.batch, args.shift_cm, args.shift_uniform)
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy
    cfg.n_action_steps = args.chunk
    cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.policy, preprocessor_overrides={
        "device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": rename}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    max_steps = env.call("_max_episode_steps")[0]

    kept, total, t0, lengths, episodes = 0, 0, time.time(), [], []
    for b in range(args.episodes // args.batch):
        seeds = [args.seed_base + b * args.batch + i for i in range(args.batch)]
        # LeRobot freezes a finished sub-env and replays its last transition until a reset carries
        # NEW_ROLLOUT_OPTION; without it every batch after the first starts on the previous ending.
        obs, reset_info = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
        shifts = np.asarray(reset_info.get("shift", np.zeros((args.batch, 2))))
        policy.reset()
        done = np.zeros(args.batch, bool); success = np.zeros(args.batch, bool)
        rec = [{"image": [], "image2": [], "state": [], "action": []} for _ in range(args.batch)]
        task = list(env.call("task_description"))
        for _ in range(max_steps):
            o = preprocess_observation(obs)
            o["task"] = task
            o = env_pre(o)
            if vae is not None:
                z = torch.stack([encode(vae, o[k]) for k in IMGS], 1)            # (B,2,4,32,32)
                if args.save_latents:
                    for i in np.flatnonzero(~done):
                        rec[i].setdefault("z", []).append(z[i].half().cpu().numpy())
                        rec[i].setdefault("st", []).append(o["observation.state"][i].cpu().numpy().astype(np.float32))
            with torch.inference_mode():
                act = post(policy.select_action(pre(o)))
            act = env_post({ACTION: act})[ACTION].to("cpu").numpy()
            for i in np.flatnonzero(~done):
                rec[i]["n"] = rec[i].get("n", 0) + 1
                if args.save_latents:
                    rec[i].setdefault("act", []).append(act[i].astype(np.float32))
                if args.no_save:
                    continue
                rec[i]["image"].append(to_uint8(o["observation.images.image"][i]))
                rec[i]["image2"].append(to_uint8(o["observation.images.image2"][i]))
                rec[i]["state"].append(o["observation.state"][i].cpu().numpy().astype(np.float32))
                rec[i]["action"].append(act[i].astype(np.float32))
            obs, _, term, trunc, info = env.step(act)
            succ = np.asarray(info.get("is_success", np.zeros(args.batch)), bool)
            newly = ~done & (term | trunc | succ)
            success |= newly & succ
            done |= newly
            if done.all():
                break
        for i in range(args.batch):
            total += 1
            n = rec[i].pop("n", 0)
            lengths.append(n)
            episodes.append({"seed": seeds[i], "success": bool(success[i]), "steps": n,
                             "shift_cm": np.round(shifts[i] * 100, 2).tolist()})
            if success[i] and n < 10:
                raise RuntimeError(f"seed {seeds[i]}: success after {n} steps, the env did not reset")
            if args.save_latents:
                np.savez(lat_out / f"ep_{seeds[i]}.npz", z=np.stack(rec[i]["z"]), state=np.stack(rec[i]["st"]),
                         action=np.stack(rec[i]["act"]), success=bool(success[i]), shift=shifts[i])
            if success[i] and not args.no_save:
                kept += 1
                np.savez_compressed(out / f"rl_{seeds[i]}.npz", task=task[i],
                                    **{k: np.array(v) for k, v in rec[i].items()})
        print(f"batch {b}: {success.sum()}/{args.batch} succeeded, total kept {kept}/{total}, {time.time() - t0:.0f}s", flush=True)
    env.close()
    n_ok = sum(e["success"] for e in episodes)
    summary = ROOT / "outputs/rollouts" / f"{args.tag}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "policy": args.policy, "chunk": args.chunk, "shift_cm": args.shift_cm, "shift_uniform": args.shift_uniform,
        "benchmark": args.benchmark, "seed_base": args.seed_base, "episodes": total, "successes": n_ok,
        "success": n_ok / total, "per_episode": episodes}, indent=1))
    print(f"{n_ok} of {total} episodes succeeded ({n_ok / total:.0%}), episode length min {min(lengths)} "
          f"median {int(np.median(lengths))}; kept {kept}; summary {summary}")


if __name__ == "__main__":
    main()
