#!/usr/bin/env bash
# The moved-can and the unmoved-can replays together (1,000 episodes), same from-base recipe. Scored lower than
# the moved-can replays alone with the can moved; kept as one of the world model's held-out policies (phone_mix_base).
set -euo pipefail
cd "$(dirname "$0")/.."
MIX=data/processed/replay/human_v3_fixed_mix
rm -rf $MIX && mkdir -p $MIX
for f in data/processed/replay/human_v3_fixed_shift/*.npz; do ln -s "$PWD/$f" "$MIX/shift_$(basename "$f")"; done
for f in data/processed/replay/human_v3_fixed/*.npz; do ln -s "$PWD/$f" "$MIX/fixed_$(basename "$f")"; done
python scripts/build_dataset.py --variant human_v3_fixed_mix --name ego2libero_human_v3_fixed_mix --max-len 260
python scripts/use_pretrained_stats.py data/lerobot/ego2libero_human_v3_fixed_mix
bash scripts/train_policy.sh ego2libero_human_v3_fixed_mix phone_mix_base
