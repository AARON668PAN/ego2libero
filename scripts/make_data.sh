#!/usr/bin/env bash
# Robot episodes from the hand trajectories in data/processed/traj, packed as LeRobot datasets that carry the
# pretrained normalisation statistics (data/lerobot/ego2libero_<name>):
#   human_v3_fixed         my ten clips, 50 sampled scenes each
#   human_v3_fixed_shift   the same with the can moved up to 6 cm before every replay
#   scripted_v3            straight lines through the same grasp and release points, 50 scenes
#   scripted_v3_shift      the same with the can moved, 500 scenes
# Usage: bash scripts/make_data.sh [SCENES_PER_CLIP] [SCRIPTED_SCENES]      (defaults 0-49 and 0-499)
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
SCENES=${1:-0-49}; SCRIPTED=${2:-0-499}
HUMAN=(--variant human --speed 1.5 --workers 16)
pack() {
  python scripts/build_dataset.py --variant "$1" --name "ego2libero_$1" --max-len 260
  python scripts/use_pretrained_stats.py "data/lerobot/ego2libero_$1"
}
python scripts/replay_libero.py "${HUMAN[@]}" --inits "$SCENES" --tag v3_fixed && pack human_v3_fixed
python scripts/replay_libero.py "${HUMAN[@]}" --inits "$SCENES" --shift-cm 6 --tag v3_fixed_shift && pack human_v3_fixed_shift
python scripts/replay_libero.py --variant scripted --workers 16 --inits 0-49 --tag v3 && pack scripted_v3
python scripts/replay_libero.py --variant scripted --workers 16 --inits "$SCRIPTED" --shift-cm 6 --tag v3_shift && pack scripted_v3_shift
