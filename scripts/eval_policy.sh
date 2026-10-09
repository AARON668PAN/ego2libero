#!/usr/bin/env bash
# Score a policy on LIBERO-Object task 0: the 50 benchmark layouts at 256 px with 10-step chunks (lerobot-eval),
# then the moved-can curve (scripts/shift_eval.sh, results in outputs/rollouts/shift_<NAME>_r<cm>.json).
# Usage: bash scripts/eval_policy.sh CHECKPOINT NAME [RENAME_JSON]
#   e.g. bash scripts/eval_policy.sh HuggingFaceVLA/smolvla_libero smolvla_libero
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=$1; NAME=$2; RENAME=${3:-}
if [ "$CKPT" = lerobot/smolvla_base ] && [ -z "$RENAME" ]; then   # the base model's camera names
  RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
fi
RARG=(); [ -n "$RENAME" ] && RARG=(--rename_map="$RENAME")
export MUJOCO_GL=egl
lerobot-eval --policy.path="$CKPT" --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_object --env.task_ids="[0]" --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=50 --eval.batch_size=10 --seed=1000 --output_dir="outputs/eval/${NAME}_task0_r256" "${RARG[@]}"
bash scripts/shift_eval.sh "$CKPT" "$NAME" "$RENAME"
