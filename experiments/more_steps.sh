#!/usr/bin/env bash
# Policies A and B trained 5000 more steps on their own replays, two held-out policies for the world model.
#   more_base   from phone_base on human_v3_fixed
#   shift_more  from phone_shift_base on human_v3_fixed_shift
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
REN='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
more() {  # FROM DATASET OUT NAME
  lerobot-train --policy.path="outputs/train/$1/checkpoints/last/pretrained_model" --dataset.repo_id="local/$2" \
    --dataset.root="data/lerobot/$2" --batch_size=32 --steps=5000 --save_freq=2500 --log_freq=100 --num_workers=8 --seed=1000 \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false --output_dir="outputs/train/$3" --rename_map="$REN"
  bash scripts/shift_eval.sh "outputs/train/$3/checkpoints/last/pretrained_model" "$4" "$REN"
}
more ego2libero_human_v3_fixed_frombase_15000 ego2libero_human_v3_fixed ego2libero_human_v3_fixed_frombase_more_5000 more_base
more ego2libero_human_v3_fixed_shift_frombase_15000 ego2libero_human_v3_fixed_shift ego2libero_shift_more_5000 shift_more
