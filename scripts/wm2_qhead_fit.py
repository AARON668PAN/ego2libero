"""No-world-model control for chunk selection: predict the value H steps from now (same target as the
value head, gamma ** (steps left) on successes, 0 on failures) from the current latents, robot state
and the next H actions, with no imagination. Trained on the world model's shards; held-out episodes
report AUC and, more to the point, whether swapping in another episode's actions changes the output.

    python scripts/wm2_qhead_fit.py            # writes outputs/wm2/qhead.pt
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
from ego2libero.world_model2 import ActionValueHead  # noqa: E402
from wm2_train import Data  # noqa: E402
from wm2_value_fit import SHARDS, auc  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--out", default=str(ROOT / "outputs/wm2/qhead.pt"))
    a = p.parse_args()
    torch.manual_seed(0); H = a.horizon
    data = Data(SHARDS)
    ep = data.step_ep.cpu().numpy(); pos = data.pos.cpu().numpy(); L = data.ep_len[ep]; s = data.ep_succ[ep]
    ok = pos + H <= L - 1                                                     # the H actions stay inside the episode
    y = torch.from_numpy(np.where(s, a.gamma ** np.maximum(L - 1 - pos - H, 0), 0.0)).float().cuda()
    train = ~data.ep_val[ep] & ok
    pos_i = torch.from_numpy(np.flatnonzero(train & s)).cuda(); neg_i = torch.from_numpy(np.flatnonzero(train & ~s)).cuda()
    ar = torch.arange(H, device="cuda")
    head = ActionValueHead(H).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for st in range(1, a.steps + 1):
        i = torch.cat([pos_i[torch.randint(len(pos_i), (a.batch // 2,), device="cuda")],
                       neg_i[torch.randint(len(neg_i), (a.batch // 2,), device="cuda")]])
        loss = F.binary_cross_entropy_with_logits(head(data.z[i].float(), data.robot[i], data.act[i[:, None] + ar]), y[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if st % 2000 == 0:
            print(f"q head step {st} loss {loss.item():.4f}", flush=True)
    head.eval()
    val = np.flatnonzero(data.ep_val & np.isin(data.ep_src, [k for k, n in enumerate(SHARDS) if n.startswith("wm2_")]))
    rep = {"held_out_episodes": int(len(val))}
    with torch.no_grad():
        for t in (20, 50, 100):
            idx = [int(data.ep_start[e]) + t for e in val if t + H <= data.ep_len[e] - 1]
            lab = [bool(data.ep_succ[e]) for e in val if t + H <= data.ep_len[e] - 1]
            ii = torch.tensor(idx, device="cuda")
            q = torch.sigmoid(head(data.z[ii].float(), data.robot[ii], data.act[ii[:, None] + ar]))
            q_sw = torch.sigmoid(head(data.z[ii].float(), data.robot[ii], data.act[ii.roll(1)[:, None] + ar]))   # someone else's actions
            rep[f"auc_at_step_{t}"] = auc(q.cpu().numpy(), lab)
            rep[f"mean_abs_change_with_other_actions_step_{t}"] = float((q - q_sw).abs().mean())
    print(json.dumps(rep, indent=1), flush=True)
    torch.save({"head": head.state_dict(), "gamma": a.gamma, "horizon": H, "report": rep}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
