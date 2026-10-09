"""Turn the iPhone originals into clean SDR clips plus camera metadata.

For each videos/IMG_xxxx.MOV this writes
  data/raw/sdr/IMG_xxxx.mp4          tone-mapped to SDR BT.709, no audio, no metadata (GPS included)
  data/processed/camera/IMG_xxxx.json lens, 35 mm equivalent focal length, per-frame optical centre

The iPhone records Dolby Vision / HLG. Decoded naively it looks washed out, and the
hand and segmentation models expect SDR, so it is tone-mapped first.
Needs ffmpeg built with zscale, and exiftool ($EXIFTOOL, on PATH, or tools/exiftool from setup_env.sh).

    python scripts/phone/prepare_videos.py videos/*.MOV
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TONEMAP = ("zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def to_sdr(src, dst):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-map_metadata", "-1", "-map_chapters", "-1", "-vf", TONEMAP,
           "-c:v", "libx264", "-crf", "16", "-preset", "medium",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def camera_info(src, exiftool):
    """Lens, focal length and the optical centre the iPhone writes for every frame."""
    out = subprocess.run(exiftool + ["-j", "-ee", "-G3", "-n", "-ImageWidth", "-ImageHeight",
                                     "-VideoFrameRate", "-LensModel", "-FocalLengthIn35mmFormat",
                                     "-SampleTime", "-OpticalCenter", str(src)],
                         check=True, capture_output=True, text=True).stdout
    tags = json.loads(out)[0]
    get = lambda name: next((v for k, v in tags.items() if k.split(":")[-1] == name), None)
    centres = []
    for key, value in tags.items():
        doc, name = key.split(":", 1) if ":" in key else ("", key)
        if name == "OpticalCenter":
            t = tags.get(f"{doc}:SampleTime", 0.0)
            centres.append((float(t), [float(x) for x in str(value).split()]))
    centres.sort(key=lambda c: c[0])
    return {
        "lens_model": get("LensModel"),
        "focal_length_35mm": get("FocalLengthIn35mmFormat"),
        "width": get("ImageWidth"),
        "height": get("ImageHeight"),
        "fps": get("VideoFrameRate"),
        "optical_center_normalized": [c for _, c in centres],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    args = p.parse_args()
    exiftool = shlex.split(os.environ.get("EXIFTOOL", ""))
    if not exiftool and shutil.which("exiftool"):
        exiftool = [shutil.which("exiftool")]
    if not exiftool and (ROOT / "tools" / "exiftool" / "exiftool").exists():
        exiftool = ["perl", str(ROOT / "tools" / "exiftool" / "exiftool")]
    sdr_dir = ROOT / "data" / "raw" / "sdr"
    cam_dir = ROOT / "data" / "processed" / "camera"
    sdr_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(map(Path, args.videos)):
        to_sdr(src, sdr_dir / f"{src.stem}.mp4")
        if exiftool:
            info = camera_info(src, exiftool)
            (cam_dir / f"{src.stem}.json").write_text(json.dumps(info))
            print(f"{src.name}: {info['lens_model']}, {info['focal_length_35mm']} mm eq, "
                  f"{len(info['optical_center_normalized'])} optical centres")
        else:
            print(f"{src.name}: converted, exiftool not found so no camera metadata")


if __name__ == "__main__":
    main()
