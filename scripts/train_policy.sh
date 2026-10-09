#!/usr/bin/env bash
# Fine-tune SmolVLA from lerobot/smolvla_base (never trained on LIBERO) for 15k steps at batch 32, then score it
# with scripts/eval_policy.sh. DATASET=teleop trains on LIBERO's own 44 teleoperated demos of task 0 instead.
# SEED=2000 repeats a run with another training seed (output directory gets an _s2000 suffix).
# Usage: bash scripts/train_policy.sh DATASET NAME
#   e.g. bash scripts/train_policy.sh ego2libero_human_v3_fixed_shift phone_shift_base
#        bash scripts/train_policy.sh teleop teleop_base
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=$1; NAME=$2; STEPS=${STEPS:-15000}; SEED=${SEED:-1000}
REN='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
export MUJOCO_GL=egl
if [ "$DATA" = teleop ]; then
  ROOT_DS=$HOME/.cache/huggingface/lerobot/lerobot/libero
  EPS=$(python - "$ROOT_DS" <<'PY'
import glob, sys
import pandas as pd
root = sys.argv[1]
ti = int(pd.read_parquet(f"{root}/meta/tasks.parquet").loc["pick up the alphabet soup and place it in the basket", "task_index"])
d = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
print(sorted(d[d.task_index == ti].episode_index.unique().tolist()))
PY
)
  DS=(--dataset.repo_id=lerobot/libero --dataset.root="$ROOT_DS" --dataset.episodes="$EPS")
  OUT=outputs/train/libero_official_task0_frombase_$STEPS
else
  DS=(--dataset.repo_id="local/$DATA" --dataset.root="data/lerobot/$DATA")
  OUT=outputs/train/${DATA}_frombase_$STEPS
fi
[ "$SEED" = 1000 ] || OUT=${OUT}_s$SEED
lerobot-train --policy.path=lerobot/smolvla_base "${DS[@]}" \
  --batch_size=32 --steps="$STEPS" --save_freq=2500 --log_freq=100 --num_workers=8 --seed="$SEED" \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --output_dir="$OUT" --rename_map="$REN"
bash scripts/eval_policy.sh "$OUT/checkpoints/last/pretrained_model" "$NAME" "$REN"
