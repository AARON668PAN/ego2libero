#!/usr/bin/env bash
# World model v2 from scratch. Needs the datasets of scripts/make_data.sh and four policies from
# scripts/train_policy.sh (phone_base, scripted_base, teleop_base, phone_shift_base).
#   1. roll five policies out 200 times each with the can moved up to 6 cm, failures kept, as VAE latents, and
#      and keep the successes of 300 rollouts of phone_base with the can moved up to 5 cm
#   2. pack those and the replays into shards (data/processed/wm2/shards)
#   3. train the denoiser, the robot-state model and the success head, and check them open loop (outputs/wm2)
#   4. fine-tune the VAE decoder on LIBERO frames, encoder frozen (models/vae/sd-vae-ft-mse-libero)
#   5. train the value head and the scorer that imagines nothing
#   6. fit the probe that reads gripper and can positions off 64 px frames, for the edit test (outputs/wm2/probe.pt)
# Usage: bash scripts/world_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
REN='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
VAE=models/vae/sd-vae-ft-mse
ck() { echo "outputs/train/$1/checkpoints/last/pretrained_model"; }
POLICIES="phone_base:$(ck ego2libero_human_v3_fixed_frombase_15000):r
scripted_base:$(ck ego2libero_scripted_v3_frombase_15000):r
teleop_base:$(ck libero_official_task0_frombase_15000):r
phone_shift_base:$(ck ego2libero_human_v3_fixed_shift_frombase_15000):r
smolvla_libero:HuggingFaceVLA/smolvla_libero:-"
rarg() { [ "$1" = r ] && echo "$REN" || echo ""; }
k=0
while IFS=: read -r name path r; do
  k=$((k + 1))
  python scripts/rollout_policy.py --policy "$path" --episodes 200 --chunk 10 --shift-cm 6 --shift-uniform \
    --seed-base $((800000 + k * 10000)) --tag "wm2_$name" --rename "$(rarg "$r")" --vae $VAE --save-latents < /dev/null
done <<< "$POLICIES"
python scripts/rollout_policy.py --policy "$(ck ego2libero_human_v3_fixed_frombase_15000)" --episodes 300 --chunk 10 \
  --shift-cm 5 --shift-uniform --tag phone_base_moved --seed-base 710000 --rename "$REN" < /dev/null
ROLL=$(echo "$POLICIES" | cut -d: -f1 | sed 's/^/wm2_/' | tr '\n' ' ')
SHARDS="human_v3_fixed human_v3_fixed_shift scripted_v3_shift scripted_v3 policy_phone_base_moved"
python scripts/wm2_encode.py --vae $VAE --replay $SHARDS --rollouts $ROLL
python scripts/wm2_train.py --shards $SHARDS $ROLL --out outputs/wm2 --hours 2 --steps 90000 --batch 128
python scripts/wm2_check.py
python scripts/vae_finetune_decoder.py --out models/vae/sd-vae-ft-mse-libero
python scripts/wm2_value_fit.py
python scripts/wm2_qhead_fit.py
python scripts/wm2_probe_fit.py
