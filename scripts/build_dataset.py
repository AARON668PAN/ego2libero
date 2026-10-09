"""Pack replayed episodes into a LeRobot dataset with the same schema as lerobot/libero.

    python scripts/build_dataset.py --variant human --name ego2libero_human
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="human")
    p.add_argument("--name", required=True)
    p.add_argument("--max-len", type=int, default=None, help="skip episodes longer than this (eval allows 280 steps)")
    args = p.parse_args()
    files = sorted((ROOT / "data/processed/replay" / args.variant).glob("*.npz"))
    out = ROOT / "data/lerobot" / args.name
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=f"local/{args.name}", fps=10, features=FEATURES, root=out,
                               robot_type="panda", use_videos=True, image_writer_threads=8)
    kept = []
    for f in files:
        with np.load(f) as d:
            if args.max_len and len(d["action"]) > args.max_len:
                continue   # read each array once; indexing an NpzFile re-decompresses it every time
            if len(d["action"]) < 10:
                raise ValueError(f"{f}: only {len(d['action'])} steps, not a real episode")
            img, img2, state, action, task = d["image"], d["image2"], d["state"], d["action"], str(d["task"])
        for k in range(len(action)):
            ds.add_frame({"observation.images.image": img[k], "observation.images.image2": img2[k],
                          "observation.state": state[k], "action": action[k], "task": task})
        ds.save_episode()
        kept.append(f)
    ds.finalize()
    import json
    (out / "meta" / "ego2libero_episodes.json").write_text(json.dumps(
        [{"episode_index": i, "source": f.resolve().parent.name + "/" + f.stem} for i, f in enumerate(kept)]))
    print(f"kept {len(kept)} of {len(files)} episodes")
    print(f"{args.name}: {len(kept)} episodes, {ds.meta.total_frames} frames -> {out}")


if __name__ == "__main__":
    main()
