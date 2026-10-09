#!/usr/bin/env bash
# Shifted-can robustness: success over LIBERO's 50 fixed layouts of task 0 with the can moved
# 0, 2, 3, 4 and 5 cm from its usual spot, 10-step action chunks.
# Usage: bash scripts/shift_eval.sh CHECKPOINT NAME [RENAME_JSON]   (results: outputs/rollouts/shift_NAME_r*.json)
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=$1; NAME=$2; RENAME=${3:-}
RARG=(); [ -n "$RENAME" ] && RARG=(--rename "$RENAME")
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
for CM in ${RADII:-0 2 3 4 5}; do
  python scripts/rollout_policy.py --policy "$CKPT" --episodes 50 --chunk 10 --benchmark --no-save \
    --shift-cm "$CM" --seed-base 1000 --tag "shift_${NAME}_r$CM" "${RARG[@]}"
done
