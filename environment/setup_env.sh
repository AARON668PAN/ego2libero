#!/usr/bin/env bash
# Build the ego2libero conda env and fetch the files the code expects on disk.
# Behind a slow network, point pip and Hugging Face at mirrors first, e.g.
#   PIP_INDEX_URL=https://mirrors.bfsu.edu.cn/pypi/web/simple \
#   HF_ENDPOINT=https://hf-mirror.com bash environment/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME=${ENV_NAME:-ego2libero}

conda create -y -n "$ENV_NAME" python=3.12
BIN="$(conda info --base)/envs/$ENV_NAME/bin"

# LeRobot with SmolVLA, LIBERO and the training extras.
"$BIN/pip" install "lerobot[smolvla,libero,training]==0.6.1"

# MediaPipe for hand landmarks. It brings opencv-contrib-python, while LeRobot and
# LIBERO bring opencv-python-headless and opencv-python. All three install into the
# same cv2 folder, so pin one version and install the contrib build last.
"$BIN/pip" install mediapipe==1.0.1 opencv-python==4.13.0.92 opencv-contrib-python==4.13.0.92
"$BIN/pip" install --force-reinstall --no-deps opencv-contrib-python==4.13.0.92

# LIBERO asks on first import where to keep its datasets. Take the default.
echo N | "$BIN/python" -c "import libero.libero" > /dev/null

# LIBERO downloads its scene and object assets on first use. Fetch them now.
"$BIN/hf" download lerobot/libero-assets --repo-type dataset --local-dir ~/.cache/libero/assets

# MediaPipe hand landmark model.
mkdir -p models
curl -fL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# World model v2: diffusers for the Stable Diffusion VAE, and the VAE itself.
"$BIN/pip" install diffusers==0.41.0 importlib-metadata==9.0.1
"$BIN/hf" download stabilityai/sd-vae-ft-mse --local-dir models/vae/sd-vae-ft-mse
# The phone-video steps (scripts/phone/: track_object.py, extract_trajectory.py) download facebook/sam2.1-hiera-large and
# depth-anything/Depth-Anything-V2-Large-hf on first use, e.g. for scripts/phone/run_sample.sh; nothing from replay_libero.py on needs them.
