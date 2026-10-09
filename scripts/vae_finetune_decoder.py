"""Fine-tune the decoder of the Stable Diffusion VAE on LIBERO frames.

Off the shelf it shifts colours on these renders (the soup can's red and green label turns teal), and
a policy looking at decoded frames loses the can. The encoder is frozen, so the latent space, and any
world model trained in it, stays valid; only what the policy sees gets better.

    python scripts/vae_finetune_decoder.py --out models/vae/sd-vae-ft-mse-libero
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[1]


def load_frames(dirs, n_eps, per_ep, rng):
    files = []
    for d in dirs:
        fs = sorted((ROOT / "data/processed/replay" / d).glob("*.npz"))
        files += list(rng.choice(fs, min(n_eps, len(fs)), replace=False))
    out = []
    for f in files:
        with np.load(f) as d:
            img, img2 = d["image"], d["image2"]          # each access decompresses the whole array: load once
            for t in rng.choice(len(img), min(per_ep, len(img)), replace=False):
                out += [img[t].copy(), img2[t].copy()]   # a view would keep the whole episode in memory
    return torch.from_numpy(np.stack(out))                       # (N,256,256,3) uint8


def grad_l1(a, b):
    dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    return (dx(a) - dx(b)).abs().mean() + (dy(a) - dy(b)).abs().mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default=str(ROOT / "models/vae/sd-vae-ft-mse"))
    p.add_argument("--out", default=str(ROOT / "models/vae/sd-vae-ft-mse-libero"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    dirs = ["human_v3_fixed_shift", "scripted_v3_shift", "human_v3_fixed"]
    train = load_frames(dirs, 120, 40, rng)
    val = load_frames(dirs, 10, 10, np.random.default_rng(1))
    print(f"train frames {len(train)}, val frames {len(val)}", flush=True)
    vae = AutoencoderKL.from_pretrained(a.vae).cuda()
    vae.encoder.requires_grad_(False); vae.quant_conv.requires_grad_(False)
    params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    to_x = lambda u8: u8.cuda().permute(0, 3, 1, 2).float() / 127.5 - 1

    @torch.no_grad()
    def psnr(frames):
        vae.eval(); out = []
        for i in range(0, len(frames), 32):
            x = to_x(frames[i:i + 32])
            r = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)
            out.append(10 * torch.log10(4 / ((r - x) ** 2).flatten(1).mean(1)))
        vae.train(); p = torch.cat(out)
        return float(p[0::2].mean()), float(p[1::2].mean())

    print("before: PSNR front %.1f dB, wrist %.1f dB" % psnr(val), flush=True)
    vae.train(); vae.encoder.eval()
    for s in range(1, a.steps + 1):
        x = to_x(train[torch.randint(len(train), (a.batch,))])
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = vae.decode(z).sample
        r = r.float()
        loss = (r - x).abs().mean() + grad_l1(r, x)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 1000 == 0:
            print(f"step {s} loss {loss.item():.4f} val PSNR front %.1f dB, wrist %.1f dB" % psnr(val), flush=True)
    vae.eval()
    print("after: PSNR front %.1f dB, wrist %.1f dB" % psnr(val), flush=True)
    vae.save_pretrained(a.out)


if __name__ == "__main__":
    main()
