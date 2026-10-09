"""Fit the robot-state model of world model v2 so that it stays stable when it feeds on its own
predictions for a whole episode, and compare candidates by open-loop error on held-out episodes
(true actions, no images). Runs on the CPU.

    python scripts/wm2_state_fit.py --out outputs/wm2/state_model.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ego2libero.world_model2 import StateModel, state_to_robot  # noqa: E402

SHARDS = ("human_v3_fixed human_v3_fixed_shift scripted_v3_shift scripted_v3 policy_phone_base_moved wm2_phone_base "
          "wm2_scripted_base wm2_teleop_base wm2_phone_shift_base wm2_smolvla_libero").split()


def load():
    tr, va = [], []
    for n in SHARDS:
        d = np.load(ROOT / "data/processed/wm2/shards" / f"{n}.npz")
        st, ac = d["state"], d["action"]
        for e, (s0, L) in enumerate(zip(d["ep_start"], d["ep_len"])):
            R = state_to_robot(torch.from_numpy(st[s0:s0 + L]).double()).float()
            (va if d["ep_val"][e] else tr).append((R, torch.from_numpy(ac[s0:s0 + L])))
    return tr, va


def windows(eps, H):
    """All length-H+1 windows (state) with the H actions inside them, padded at episode start."""
    Rs, As = [], []
    for R, A in eps:
        L = len(R)
        if L < H + 1:
            continue
        Ap = torch.cat([torch.zeros(3, 7), A])                    # 3 zero actions before the start
        for s in range(0, L - H, 2):
            Rs.append(R[s:s + H + 1]); As.append(Ap[s:s + H + 3])
    return torch.stack(Rs), torch.stack(As)


def rollout(model, R0, A, H):
    """R0 (B,9) start state, A (B,H+3,7) with the 3 previous actions first. Returns (B,H,9)."""
    r, out = R0, []
    for t in range(H):
        r = model(r, None, A[:, t:t + 4])
        out.append(r)
    return torch.stack(out, 1)


W = torch.tensor([50.0] * 3 + [20.0] * 4 + [100.0] * 2)


def evaluate(model, eps, steps=(10, 50, 100, 150)):
    errs = {k: [] for k in steps}
    with torch.no_grad():
        for R, A in eps:
            H = min(len(R) - 1, max(steps))
            Ap = torch.cat([torch.zeros(3, 7), A])[None]
            pred = rollout(model, R[:1], Ap, H)[0]
            for k in steps:
                if k <= H:
                    errs[k].append(float((pred[k - 1, :3] - R[k, :3]).norm() * 100))
    return {k: (float(np.median(v)), float(np.percentile(v, 90)), float(np.max(v))) for k, v in errs.items() if v}


def fit(out, steps=3000, horizon=16):
    torch.set_num_threads(16); torch.manual_seed(0)
    tr, va = load()
    Rw, Aw = windows(tr, horizon)
    print(f"train episodes {len(tr)}, windows {len(Rw)}, held-out episodes {len(va)}", flush=True)
    model = StateModel()
    model.fit_linear(Rw, Aw)
    print("linear part only:", evaluate(model, va), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(1, steps + 1):
        i = torch.randint(len(Rw), (512,))
        R, A = Rw[i], Aw[i]
        pred = rollout(model, R[:, 0], A, horizon)
        tgt = R[:, 1:].clone()
        loss = (((pred - tgt) * W) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 500 == 0:
            print(f"step {s} loss {loss.item():.5f}", flush=True)
    res = evaluate(model, va)
    print("linear + residual, unrolled training:", res, flush=True)
    torch.save(model.state_dict(), out)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "outputs/wm2/state_model.pt"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--horizon", type=int, default=16)
    a = p.parse_args()
    fit(a.out, a.steps, a.horizon)


if __name__ == "__main__":
    main()
