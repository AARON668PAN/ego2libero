#!/usr/bin/env bash
# Run the phone-video steps on the one clip in the repo (IMG_3562, 10 s) and compare the trajectory with the
# one shipped in data/processed/traj, which the robot data was made from.
#
# The clip is already the SDR output of prepare_videos.py (that step needs the iPhone original, which is not
# public). The focal length comes from data/processed/camera/intrinsics.json: calibrate_from_box.py fits it
# jointly over all clips. Needs a GPU; SAM 2 and Depth Anything V2 (about 2 GB) download on first use.
#
#   bash scripts/phone/run_sample.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLIP=IMG_3562
V=data/raw/sdr/$CLIP.mp4
OUT=outputs/phone_sample
mkdir -p $OUT
cp data/processed/traj/$CLIP.npz $OUT/${CLIP}_shipped.npz
python scripts/phone/extract_hands.py $V        # MediaPipe landmarks
python scripts/phone/track_object.py $V         # SAM 2 track of the case from one click
python scripts/phone/camera_motion.py $V        # camera pose per frame
python scripts/phone/extract_trajectory.py $V   # metric pinch trajectory, grasp and release
cp data/processed/traj/$CLIP.npz $OUT/${CLIP}_rerun.npz
cp $OUT/${CLIP}_shipped.npz data/processed/traj/$CLIP.npz        # leave the shipped trajectory untouched
cp outputs/traj/$CLIP.png $OUT/ 2>/dev/null || true
python - "$OUT" "$CLIP" <<'PY'
import sys
import numpy as np
out, clip = sys.argv[1:]
a, b = np.load(f"{out}/{clip}_shipped.npz"), np.load(f"{out}/{clip}_rerun.npz")
print(f"grasp frame {int(a['close'])} -> {int(b['close'])}, release frame {int(a['open'])} -> {int(b['open'])}")
if len(a["pos"]) == len(b["pos"]):
    d = np.linalg.norm(a["pos"] - b["pos"], axis=1)
    print(f"pinch trajectory, {len(d)} frames: largest difference {d.max():.2f} mm, mean {d.mean():.2f} mm")
else:
    print(f"trajectory length {len(a['pos'])} -> {len(b['pos'])} frames")
print(f"plot: {out}/{clip}.png")
PY
