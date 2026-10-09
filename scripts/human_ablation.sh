#!/usr/bin/env bash
# What in the human motion matters? Two knock-outs of the moved-can phone data, each removing one property and
# keeping everything else (same clips, layouts, can shifts, robot constraints, training recipe):
#   flat      the rhythm: every segment walked at constant speed in the same time
#   straight  the path shape: every used segment a straight line, walked with the hand's own timing
# Compare with phone_shift_base (183/250) and the straight-line control scripted_shift_base (137/250).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
REN='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
replay() {  # FLAG VALUE TAG
  python scripts/replay_libero.py --variant human --inits 0-49 --workers 16 --speed 1.5 \
    --shift-cm 6 $1 $2 --tag $3 < /dev/null
}
pack() {  # TAG NAME
  python scripts/build_dataset.py --variant human_$1 --name $2 --max-len 260 < /dev/null && python scripts/use_pretrained_stats.py data/lerobot/$2
}
train() {  # NAME
  lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=local/$1 --dataset.root=data/lerobot/$1 \
    --batch_size=32 --steps=15000 --save_freq=2500 --log_freq=100 --num_workers=8 --seed=1000 --policy.device=cuda \
    --policy.push_to_hub=false --wandb.enable=false --output_dir=outputs/train/$1_frombase_15000 --rename_map="$REN" < /dev/null
}
echo "== replay $(date +%H:%M)"
replay --rhythm flat v3_fixed_shift_flat
replay --shape straight v3_fixed_shift_straight
echo "== pack flat $(date +%H:%M)"
pack v3_fixed_shift_flat ego2libero_human_flat_shift || { echo "PACKFAILED"; exit 1; }
echo "== train flat, pack straight alongside $(date +%H:%M)"
( pack v3_fixed_shift_straight ego2libero_human_straight_shift > logs/ablation_pack_straight.log 2>&1 ) &
PACK=$!
rm -rf outputs/train/ego2libero_human_flat_shift_frombase_15000
train ego2libero_human_flat_shift || { echo "TRAINFAILED flat"; exit 1; }
bash scripts/shift_eval.sh outputs/train/ego2libero_human_flat_shift_frombase_15000/checkpoints/last/pretrained_model phone_flat_shift "$REN"
wait $PACK
echo "== train straight $(date +%H:%M)"
rm -rf outputs/train/ego2libero_human_straight_shift_frombase_15000
train ego2libero_human_straight_shift || { echo "TRAINFAILED straight"; exit 1; }
bash scripts/shift_eval.sh outputs/train/ego2libero_human_straight_shift_frombase_15000/checkpoints/last/pretrained_model phone_straight_shift "$REN"
echo "ABLATIONDONE $(date +%H:%M)"
