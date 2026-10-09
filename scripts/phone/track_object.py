"""Track the AirPods case through each clip with SAM 2.

One click on the case in frame 0 (scripts/phone/annotations/case_points.json) is enough.
Writes data/processed/object/<clip>.npz holding, per frame, the mask area, centroid and
bounding box in full-resolution pixels, plus the mask itself at quarter resolution.

    python scripts/phone/track_object.py data/raw/sdr/*.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

ROOT = Path(__file__).resolve().parents[2]
CKPT = "facebook/sam2.1-hiera-large"
SCALE = 0.5   # frames go to SAM 2 at half resolution; it resizes to 1024 internally anyway


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    args = p.parse_args()
    prompts = json.loads((Path(__file__).resolve().parent / "annotations" / "case_points.json").read_text())
    out_dir = ROOT / "data/processed/object"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Sam2VideoModel.from_pretrained(CKPT).to("cuda", dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(CKPT)

    for video in sorted(map(Path, args.videos)):
        frames = read_frames(video)
        h, w = frames[0].shape[:2]
        session = processor.init_video_session(video=frames, inference_device="cuda",
                                               video_storage_device="cpu", dtype=torch.bfloat16)
        x, y = prompts[video.stem]["point"]
        processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=1,
                                                  input_points=[[[[x * SCALE, y * SCALE]]]], input_labels=[[[1]]])
        model(inference_session=session, frame_idx=0)

        n = len(frames)
        area = np.zeros(n); cent = np.full((n, 2), np.nan); bbox = np.full((n, 4), np.nan)
        small = np.zeros((n, h // 2, w // 2), bool)
        with torch.inference_mode():
            for o in model.propagate_in_video_iterator(session):
                m = processor.post_process_masks([o.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                m = m[0, 0].cpu().numpy().astype(bool)
                k = o.frame_idx
                small[k] = cv2.resize(m.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
                ys, xs = np.nonzero(m)
                area[k] = len(xs) / SCALE**2
                if len(xs):
                    cent[k] = [xs.mean() / SCALE, ys.mean() / SCALE]
                    bbox[k] = [xs.min() / SCALE, ys.min() / SCALE, xs.max() / SCALE, ys.max() / SCALE]
        np.savez_compressed(out_dir / f"{video.stem}.npz", area=area, centroid=cent, bbox=bbox,
                            mask_quarter=np.packbits(small, axis=-1), mask_shape=np.array(small.shape))
        print(f"{video.stem}: {n} frames, case visible in {np.mean(area > 0):.0%}, "
              f"area frame0 {area[0]:.0f} px, median {np.median(area[area > 0]):.0f} px")


if __name__ == "__main__":
    main()
