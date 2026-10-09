#!/usr/bin/env bash
# Every world-model result in the README, after scripts/world_model.sh.
#   1. nine policies run inside the world model from the first frame, against the simulator (outputs/wm2_eval_dec2);
#      four are held out, three of them from experiments/ (skipped if their checkpoints are missing)
#   2. picking chunks for the moved-can policy: alone, scorer without a world model, world model, perfect foresight
#   3. where to trust it: same-state comparison, look-ahead and self-confidence, edited actions, play-outs to the end
# Usage: bash scripts/world_model_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
REN='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
VAE=models/vae/sd-vae-ft-mse-libero
ck() { echo "outputs/train/$1/checkpoints/last/pretrained_model"; }
rarg() { [ "$1" = r ] && echo "$REN" || echo ""; }
SEEN="phone_base:$(ck ego2libero_human_v3_fixed_frombase_15000):r
scripted_base:$(ck ego2libero_scripted_v3_frombase_15000):r
teleop_base:$(ck libero_official_task0_frombase_15000):r
phone_shift_base:$(ck ego2libero_human_v3_fixed_shift_frombase_15000):r
smolvla_libero:HuggingFaceVLA/smolvla_libero:-"
HELD="scripted_shift_base:$(ck ego2libero_scripted_v3_shift_frombase_15000):r
more_base:$(ck ego2libero_human_v3_fixed_frombase_more_5000):r
phone_mix_base:$(ck ego2libero_human_v3_fixed_mix_frombase_15000):r
shift_more:$(ck ego2libero_shift_more_5000):r"
while IFS=: read -r name path r; do
  [ "$path" != HuggingFaceVLA/smolvla_libero ] && [ ! -e "$path" ] && { echo "skip $name: no checkpoint"; continue; }
  [ -f "outputs/rollouts/shift_${name}_r5.json" ] || bash scripts/shift_eval.sh "$path" "$name" "$(rarg "$r")"
  python scripts/wm2_policy_eval.py --policy "$path" --name "$name" --rename "$(rarg "$r")" --shifts 0 2 3 4 5 \
    --vae $VAE --out outputs/wm2_eval_dec2 < /dev/null
done <<< "$SEEN
$HELD"
python scripts/wm2_report.py --eval-dir outputs/wm2_eval_dec2 --held-out $(echo "$HELD" | cut -d: -f1 | tr '\n' ' ')

P=$(ck ego2libero_human_v3_fixed_shift_frombase_15000)
G=(--policy "$P" --name phone_shift_base --rename "$REN")
for mode in policy q wm sim; do
  python scripts/wm2_guided_eval.py "${G[@]}" --mode $mode --shifts 0 2 3 4 5 --tag v1 < /dev/null
done
python scripts/wm2_guided_report.py --tag v1

python scripts/wm2_guided_eval.py "${G[@]}" --mode compare --shifts 0 3 5 --episodes 50 --tag fid2 < /dev/null
python scripts/wm2_fidelity_report.py --tag fid2
python scripts/wm2_guided_eval.py "${G[@]}" --mode compare2 --samples 3 --shifts 0 3 5 --episodes 50 --tag trust < /dev/null
python scripts/wm2_trust_report.py --tag trust
python scripts/wm2_probe_eval.py --policy "$P" --rename "$REN" --shifts 0 3 5 --episodes 30 --tag probe < /dev/null
python scripts/wm2_probe_report.py --tag probe
python scripts/wm2_outcome_check.py --policy "$P" --rename "$REN" --shifts 0 3 5 --episodes 30 --at 40 80 120 --repeats 2 \
  --tag outcome < /dev/null
python scripts/wm2_outcome_report.py --tag outcome
