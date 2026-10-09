"""Give a dataset the normalisation statistics the pretrained SmolVLA LIBERO policy was trained with.

Fine-tuning normally recomputes mean/std from the new data. Our demonstrations hold the gripper
pointing straight down, so the orientation part of the state barely varies (std up to 300x
smaller than in the official data), and that orientation sits where the axis-angle encoding
jumps between +pi and -pi. Dividing by such a small std turns tiny pose errors, or one jump,
into inputs thousands of standard deviations out, and the fine-tuned policy fails. Keeping the
pretrained statistics avoids that and keeps the inputs on the scale the model already knows.

    python scripts/use_pretrained_stats.py data/lerobot/ego2libero_human_v3_fixed
"""
import json
import shutil
import sys
from pathlib import Path

OFFICIAL = Path.home() / ".cache/huggingface/lerobot/lerobot/libero/meta/stats.json"

for root in map(Path, sys.argv[1:]):
    f = root / "meta" / "stats.json"
    if not (root / "meta" / "stats.own.json").exists():
        shutil.copy(f, root / "meta" / "stats.own.json")
    stats, official = json.loads(f.read_text()), json.loads(OFFICIAL.read_text())
    for key in ("observation.state", "action"):
        stats[key] = official[key]
    f.write_text(json.dumps(stats, indent=1))
    print(f"{root.name}: state and action statistics replaced with the pretrained ones (own copy kept in stats.own.json)")
