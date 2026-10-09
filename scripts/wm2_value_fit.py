"""Value head for world model v2: from one step's latents and robot state, predict gamma ** (steps
left) if the episode ends in success and 0 if it fails. Trained on the same shards as the world model
(replays and policy rollouts, failures included), half of each batch from failed episodes, with noise
on the latents because it will score imagined steps. Held-out episodes report how well it predicts the
final outcome from early, middle and late steps (AUC).

    python scripts/wm2_value_fit.py            # writes outputs/wm2/value_head.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ego2libero.world_model2 import ValueHead  # noqa: E402
from wm2_train import Data  # noqa: E402

SHARDS = ["human_v3_fixed", "human_v3_fixed_shift", "scripted_v3_shift", "scripted_v3", "policy_phone_base_moved",
          "wm2_phone_base", "wm2_scripted_base", "wm2_teleop_base", "wm2_phone_shift_base", "wm2_smolvla_libero"]


def auc(score, label):
    """Probability that a random positive outranks a random negative (ties count half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                      # average ranks of ties
        m = score == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos = label.sum()
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * (~label).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--out", default=str(ROOT / "outputs/wm2/value_head.pt"))
    a = p.parse_args()
    torch.manual_seed(0)
    data = Data(SHARDS)
    ep = data.step_ep.cpu().numpy(); pos = data.pos.cpu().numpy(); L = data.ep_len[ep]; s = data.ep_succ[ep]
    y = torch.from_numpy(np.where(s, a.gamma ** (L - 1 - pos), 0.0)).float().cuda()
    train = ~data.ep_val[ep]
    pos_i = torch.from_numpy(np.flatnonzero(train & s)).cuda(); neg_i = torch.from_numpy(np.flatnonzero(train & ~s)).cuda()
    print(f"train steps: {len(pos_i)} from successes, {len(neg_i)} from failures", flush=True)
    head = ValueHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for st in range(1, a.steps + 1):
        i = torch.cat([pos_i[torch.randint(len(pos_i), (a.batch // 2,), device="cuda")],
                       neg_i[torch.randint(len(neg_i), (a.batch // 2,), device="cuda")]])
        z = data.z[i].float()
        z = z + torch.rand(len(i), 1, 1, 1, device="cuda") * 0.3 * data.sd * torch.randn_like(z)   # imagined latents are not clean
        loss = F.binary_cross_entropy_with_logits(head(z, data.robot[i]), y[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if st % 2000 == 0:
            print(f"value head step {st} loss {loss.item():.4f}", flush=True)
    head.eval()
    # held-out episodes: does the value at step t predict the final outcome? (only sources with failures)
    val = np.flatnonzero(data.ep_val & np.isin(data.ep_src, [i for i, n in enumerate(SHARDS) if n.startswith("wm2_")]))
    rep = {"held_out_episodes": int(len(val)), "held_out_successes": int(data.ep_succ[val].sum())}
    with torch.no_grad():
        vals = []
        for e in val:
            s0, n = int(data.ep_start[e]), int(data.ep_len[e])
            vals.append(torch.sigmoid(head(data.z[s0:s0 + n].float(), data.robot[s0:s0 + n])).cpu().numpy())
    lab = data.ep_succ[val]
    for frac in (0.1, 0.25, 0.5, 0.75):
        rep[f"auc_at_{int(frac * 100)}pct"] = auc([v[int(frac * (len(v) - 1))] for v in vals], lab)
    for t in (20, 50, 100):
        rep[f"auc_at_step_{t}"] = auc([v[min(t, len(v) - 1)] for v in vals], lab)
    rep["mean_value_successes_first_step"] = float(np.mean([v[0] for v, l in zip(vals, lab) if l]))
    rep["mean_value_failures_first_step"] = float(np.mean([v[0] for v, l in zip(vals, lab) if not l]))
    print(json.dumps(rep, indent=1), flush=True)
    torch.save({"head": head.state_dict(), "gamma": a.gamma, "report": rep}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
