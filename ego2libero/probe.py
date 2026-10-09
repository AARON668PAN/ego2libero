"""A small CNN that reads the gripper and can positions off one 64 px front-camera frame.

It turns imagined frames into physical questions, is the can lifted and where did it go, for the edit test of
scripts/wm2_probe_eval.py; run on the simulator's own frames it gives its own floor. scripts/wm2_probe_fit.py
fits it on simulator frames with the true positions.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 64
LIFT_Z = 0.07      # can centre above this height counts as lifted (it rests at 0.038)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 128]
        self.f = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU())])
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256), nn.SiLU(), nn.Linear(256, 6))

    def forward(self, x):                                           # (B,3,64,64) in [-1,1] -> (B,6) normalised
        return self.h(self.f(x))


def frame64(img):
    """Policy-side image (3, 256, 256) in [0, 1], already rotated like the dataset -> uint8 64x64, the probe's input."""
    import cv2
    u8 = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.resize(u8, (RES, RES), interpolation=cv2.INTER_AREA)


def train_probe(frames, labels, path, steps=8000, batch=256):
    """frames uint8 (N,64,64,3), labels (N,6) = gripper xyz, can xyz in metres. Saves to path, returns read()."""
    frames = torch.from_numpy(frames).cuda(); y = torch.from_numpy(labels).float().cuda()
    mu, sd = y.mean(0), y.std(0)
    probe = Probe().cuda()
    opt = torch.optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(len(frames), (batch,), device="cuda")
        x = frames[i].permute(0, 3, 1, 2).float() / 127.5 - 1
        x = x + 0.03 * torch.randn_like(x)                          # imagined frames are not clean
        loss = F.mse_loss(probe(x), (y[i] - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": probe.state_dict(), "mu": mu.cpu(), "sd": sd.cpu()}, path)
    return load_probe(path)


def load_probe(path):
    """Returns read(x): (N,3,64,64) in [-1,1] -> (N,6) gripper xyz and can xyz in metres."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: fit the probe first with scripts/wm2_probe_fit.py")
    ck = torch.load(path, map_location="cuda")
    probe = Probe().cuda(); probe.load_state_dict(ck["model"]); probe.eval()
    mu, sd = ck["mu"].cuda(), ck["sd"].cuda()

    @torch.no_grad()
    def read(x, chunk=2048):
        return torch.cat([probe(x[i:i + chunk]) * sd + mu for i in range(0, len(x), chunk)])
    return read
