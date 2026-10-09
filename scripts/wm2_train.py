"""Train world model v2 on the packed shards: the latent denoiser, then the robot-state model, then the
success head, whose threshold is calibrated on held-out episodes. Ends with an open-loop check: from
the first step of held-out episodes the model imagines the whole episode under the recorded actions,
and the success head on the imagined last steps is compared with what the simulator said.

    python scripts/wm2_train.py --shards human_v3_fixed human_v3_fixed_shift ... --out outputs/wm2 --hours 2
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ego2libero.world_model2 import (N_ACT, OFFSETS, Denoiser2, History, StateModel, SuccessHead,  # noqa: E402
                                     edm_loss, sample, state_to_robot)


class Data:
    def __init__(self, names, device="cuda"):
        ds = [np.load(ROOT / "data/processed/wm2/shards" / f"{n}.npz") for n in names]
        off = np.cumsum([0] + [int(d["ep_len"].sum()) for d in ds[:-1]])
        self.z = torch.from_numpy(np.concatenate([d["z"] for d in ds])).to(device).flatten(1, 2)     # (N,8,32,32) fp16
        self.robot = state_to_robot(torch.from_numpy(np.concatenate([d["state"] for d in ds])).double()).float().to(device)
        self.act = torch.from_numpy(np.concatenate([d["action"] for d in ds])).to(device)
        self.ep_start = np.concatenate([d["ep_start"] + o for d, o in zip(ds, off)])
        self.ep_len = np.concatenate([d["ep_len"] for d in ds])
        self.ep_succ = np.concatenate([d["ep_success"] for d in ds])
        self.ep_val = np.concatenate([d["ep_val"] for d in ds])
        self.ep_src = np.concatenate([np.full(len(d["ep_len"]), i) for i, d in enumerate(ds)])
        ep = np.concatenate([np.full(l, i) for i, l in enumerate(self.ep_len)])
        self.step_ep = torch.from_numpy(ep).to(device)
        self.step_start = torch.from_numpy(self.ep_start[ep]).to(device)
        pos = np.arange(len(ep)) - self.ep_start[ep]
        self.pos = torch.from_numpy(pos).to(device)
        last = pos == self.ep_len[ep] - 1
        train = ~self.ep_val[ep]
        # transitions t -> t+1 inside training episodes; half of each batch from successful episodes
        ok = np.flatnonzero(train & ~last)
        succ = self.ep_succ[ep[ok]]
        w = np.where(succ, 0.5 / max(succ.sum(), 1), 0.5 / max((~succ).sum(), 1)) if (~succ).any() else np.ones(len(ok))
        self.idx = torch.from_numpy(ok).to(device)
        self.prob = torch.from_numpy(w / w.sum()).float().to(device)
        self.sd = float(self.z[torch.randint(len(self.z), (4096,), device=device)].float().std())
        print(f"steps {len(ep)}, episodes {len(self.ep_len)} ({self.ep_succ.mean():.0%} successes, {self.ep_val.sum()} held out), "
              f"latent std {self.sd:.3f}", flush=True)

    def context(self, t):
        k = t[:, None] - torch.tensor(OFFSETS, device=t.device)[None]
        return self.z[torch.maximum(k, self.step_start[t][:, None])].float()            # repeat the first step

    def actions(self, t, n=N_ACT):
        k = t[:, None] - torch.arange(n - 1, -1, -1, device=t.device)[None]                 # t-n+1 .. t
        valid = k >= self.step_start[t][:, None]
        return self.act[torch.maximum(k, self.step_start[t][:, None])] * valid[..., None]

    def batch(self, n, ctx_noise=0.1):
        t = self.idx[torch.multinomial(self.prob, n, replacement=True)]
        ctx = self.context(t)
        sig = torch.rand(n, device=t.device) * ctx_noise
        ctx = ctx + sig[:, None, None, None, None] * torch.randn_like(ctx)
        return self.z[t + 1].float(), ctx, self.actions(t), self.robot[t], sig


def train_denoiser(data, out, steps, hours, batch=64, lr=2e-4):
    model = Denoiser2().cuda()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    print(f"denoiser params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    def step_fn():
        x, ctx, act, robot, sig = data.batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = edm_loss(model, x, ctx, act, robot, sig, data.sd)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1e-3)
        return loss.item()

    t0 = time.time()
    for _ in range(50):                                   # time a few steps to fit the budget
        step_fn()
    torch.cuda.synchronize(); per = (time.time() - t0) / 50
    steps = min(steps, int(hours * 3600 / per))
    print(f"{per * 1000:.0f} ms/step -> {steps} steps", flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    run, t0 = None, time.time()
    for s in range(1, steps + 1):
        l = step_fn(); sched.step()
        run = l if run is None else 0.98 * run + 0.02 * l
        if s % 500 == 0:
            print(f"step {s} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / s * 1000:.0f} ms/step", flush=True)
        if s % 5000 == 0 or s == steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "sd": data.sd, "step": s}, out / "denoiser.pt")
    return ema


def success_labels(data):
    """1 on the last two recorded steps of a successful episode, 0 on failed episodes and on steps more
    than 12 before the end of a successful one; the steps in between are left out."""
    ep = data.step_ep.cpu().numpy(); pos = data.pos.cpu().numpy(); L = data.ep_len[ep]; s = data.ep_succ[ep]
    y = np.full(len(ep), -1, np.int8)
    y[~s] = 0
    y[s & (pos < L - 12)] = 0
    y[s & (pos >= L - 2)] = 1
    return torch.from_numpy(y).cuda()


def train_success(data, out, steps=6000, batch=512):
    head = SuccessHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    y = success_labels(data)
    train = torch.from_numpy(~data.ep_val[data.step_ep.cpu().numpy()]).cuda()
    pos_i = torch.nonzero((y == 1) & train)[:, 0]; neg_i = torch.nonzero((y == 0) & train)[:, 0]
    for s in range(1, steps + 1):
        i = torch.cat([pos_i[torch.randint(len(pos_i), (batch // 2,), device="cuda")], neg_i[torch.randint(len(neg_i), (batch // 2,), device="cuda")]])
        z = data.z[i].float()
        z = z + torch.rand(len(i), 1, 1, 1, device="cuda") * 0.2 * data.sd * torch.randn_like(z)   # imagined latents are not clean
        loss = F.binary_cross_entropy_with_logits(head(z), y[i].float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"success head step {s} loss {loss.item():.4f}", flush=True)
    head.eval()
    # threshold: an episode counts as a success when the head fires on any step; pick the threshold that
    # best separates the held-out episodes
    val_eps = np.flatnonzero(data.ep_val)
    mx = []
    with torch.no_grad():
        for e in val_eps:
            s0, n = data.ep_start[e], data.ep_len[e]
            mx.append(torch.sigmoid(head(data.z[s0:s0 + n].float())).max().item())
    mx, lab = np.array(mx), data.ep_succ[val_eps]
    cands = np.linspace(0.05, 0.95, 91)
    acc = [((mx > c) == lab).mean() for c in cands]
    tau = float(cands[int(np.argmax(acc))])
    print(f"success head on {len(val_eps)} held-out episodes: accuracy {max(acc):.1%} at threshold {tau:.2f}", flush=True)
    torch.save({"head": head.state_dict(), "tau": tau}, out / "success_head.pt")
    return head, tau, max(acc)


@torch.no_grad()
def open_loop(data, den, sm, head, tau, out, n_eps=40, steps=3):
    """Imagine held-out episodes from their first step under the recorded actions; did the success head
    fire on the imagined episode as the simulator says it should?"""
    val = np.flatnonzero(data.ep_val)
    rng = np.random.default_rng(0)
    pick = np.r_[rng.permutation(val[data.ep_succ[val]])[:n_eps // 2], rng.permutation(val[~data.ep_succ[val]])[:n_eps // 2]]
    T = int(data.ep_len[pick].max())
    s0 = torch.from_numpy(data.ep_start[pick]).cuda(); L = torch.from_numpy(data.ep_len[pick]).cuda()
    hist = History(data.z[s0].float(), data.robot[s0])
    r = data.robot[s0]; fired = torch.zeros(len(pick), dtype=torch.bool, device="cuda")
    err = []
    for t in range(T - 1):
        tt = torch.minimum(s0 + t, s0 + L - 2)
        a = data.act[tt]
        hist.push_action(a)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = sample(den, hist.context(), hist.a, r, data.sd, steps=steps).float()
        r_new = sm(r, hist.r[:, -2], hist.a[:, -4:])
        hist.push_step(z, r_new); r = r_new
        alive = t + 1 < L
        fired |= alive & (torch.sigmoid(head(z)) > tau)
        if t + 1 in (10, 50, 100, 150):
            m = alive.clone()
            err.append((t + 1, float(((z - data.z[s0 + t + 1].float()) ** 2).mean(dim=(1, 2, 3))[m].mean() / data.sd ** 2),
                        float((r[:, :3] - data.robot[s0 + t + 1][:, :3]).norm(dim=1)[m].mean() * 100)))
    truth = torch.from_numpy(data.ep_succ[pick]).cuda()
    res = {"episodes": len(pick), "outcome_accuracy": float((fired == truth).float().mean()),
           "fired_on_successes": float(fired[truth].float().mean()), "fired_on_failures": float(fired[~truth].float().mean()),
           "latent_mse_rel": {e[0]: e[1] for e in err}, "eef_err_cm": {e[0]: e[2] for e in err}}
    print("open loop:", json.dumps(res), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "outputs/wm2"))
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    data = Data(a.shards)
    den = train_denoiser(data, out, a.steps, a.hours, batch=a.batch)
    from wm2_state_fit import fit as fit_state                    # stable state model, fitted on the CPU
    fit_state(out / "state_model.pt")
    sm = StateModel().cuda(); sm.load_state_dict(torch.load(out / "state_model.pt", map_location="cuda")); sm.eval()
    head, tau, acc = train_success(data, out)
    res = open_loop(data, den, sm, head, tau, out)
    json.dump({"shards": a.shards, "latent_std": data.sd, "success_threshold": tau, "success_head_val_accuracy": acc,
               "open_loop": res}, open(out / "train_summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
