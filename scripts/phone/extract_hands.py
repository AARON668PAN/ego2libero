"""Run MediaPipe HandLandmarker on every frame of the recorded videos.

Writes one .npz per video to data/processed/hands/ holding, per frame:
  t        timestamp in seconds
  found    whether a hand was detected
  score    handedness confidence
  img      21x3 image landmarks (x, y in pixels, z relative depth from MediaPipe)
  world    21x3 world landmarks in metres, centred on the hand

    python scripts/phone/extract_hands.py data/raw/sdr/*.mp4
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "hand_landmarker.task"


def extract(video, landmarker):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t, found, score, img, world = [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(rgb, int(round(i * 1000 / fps)))
        t.append(i / fps)
        if res.hand_landmarks:
            found.append(True)
            score.append(res.handedness[0][0].score)
            img.append([[lm.x * w, lm.y * h, lm.z * w] for lm in res.hand_landmarks[0]])
            world.append([[lm.x, lm.y, lm.z] for lm in res.hand_world_landmarks[0]])
        else:
            found.append(False)
            score.append(0.0)
            img.append(np.full((21, 3), np.nan))
            world.append(np.full((21, 3), np.nan))
        i += 1
    cap.release()
    return dict(t=np.array(t), found=np.array(found), score=np.array(score),
                img=np.array(img, dtype=np.float32), world=np.array(world, dtype=np.float32),
                fps=fps, size=np.array([w, h]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "hands"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for video in sorted(args.videos):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            d = extract(video, landmarker)
        np.savez_compressed(out / (Path(video).stem + ".npz"), **d)
        idx = np.flatnonzero(d["found"])
        inside = d["found"][idx[0]:idx[-1] + 1] if len(idx) else np.array([])
        print(f"{Path(video).name}: {len(d['found'])} frames, hand in {d['found'].mean():.0%}, "
              f"missing inside the hand's span: {(~inside).sum() if len(inside) else '-'}")


if __name__ == "__main__":
    main()
