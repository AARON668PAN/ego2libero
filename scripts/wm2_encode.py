"""Pack episodes for world model v2: replays saved as images are encoded with the VAE, rollouts saved
as latents (rollout_policy.py --save-latents) are copied. One shard per source:
data/processed/wm2/shards/<name>.npz with z (N,2,4,32,32) fp16, state (N,8), action (N,7) and per
episode start, length, success and a validation flag (every 20th episode is held out).

    python scripts/wm2_encode.py --vae models/vae/sd-vae-ft-mse \
        --replay human_v3_fixed human_v3_fixed_shift scripted_v3_shift --rollouts wm2_phone_base wm2_teleop_base
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ego2libero.world_model2 import encode, load_vae  # noqa: E402

OUT = ROOT / "data/processed/wm2/shards"
torch.multiprocessing.set_sharing_strategy("file_system")   # whole episodes per item: the default runs out of file descriptors


class Episodes(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            return {k: d[k] for k in ("image", "image2", "state", "action")}


def write(name, eps):
    z = np.concatenate([e["z"] for e in eps]); lens = np.array([len(e["z"]) for e in eps])
    np.savez(OUT / f"{name}.npz", z=z, state=np.concatenate([e["state"] for e in eps]).astype(np.float32),
             action=np.concatenate([e["action"] for e in eps]).astype(np.float32),
             ep_start=np.r_[0, np.cumsum(lens)[:-1]], ep_len=lens, ep_success=np.array([e["success"] for e in eps]),
             ep_val=np.arange(len(eps)) % 20 == 7)
    print(f"{name}: {len(eps)} episodes, {len(z)} steps, {np.mean([e['success'] for e in eps]):.0%} successes", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True)
    p.add_argument("--replay", nargs="*", default=[], help="dirs under data/processed/replay with image episodes")
    p.add_argument("--rollouts", nargs="*", default=[], help="tags under data/processed/wm2/rollouts")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vae = load_vae(a.vae)
    for name in a.replay:
        files = sorted((ROOT / "data/processed/replay" / name).glob("*.npz"))
        dl = torch.utils.data.DataLoader(Episodes(files), batch_size=None, num_workers=8)
        eps = []
        for d in dl:
            img = torch.stack([torch.as_tensor(d["image"]), torch.as_tensor(d["image2"])], 1)   # (T,2,H,W,3) uint8
            T = img.shape[0]
            x = img.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
            z = torch.cat([encode(vae, x[i:i + 256]) for i in range(0, len(x), 256)]).reshape(T, 2, 4, 32, 32)
            eps.append({"z": z.half().cpu().numpy(), "state": np.asarray(d["state"]), "action": np.asarray(d["action"]),
                        "success": True})                                   # replays are kept only when they succeed
        write(name, eps)
    for tag in a.rollouts:
        eps = []
        for f in sorted((ROOT / "data/processed/wm2/rollouts" / tag).glob("*.npz")):
            with np.load(f) as d:
                eps.append({"z": d["z"], "state": d["state"], "action": d["action"], "success": bool(d["success"])})
        write(tag, eps)


if __name__ == "__main__":
    main()
