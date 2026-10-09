"""Open-loop check of world model v2 on the held-out episodes: imagine each one from its first step
under the recorded actions and ask the success head, at every threshold, whether the imagined episode
ends in success; compare with the simulator. Also refits the success head on imagined latents (first
half of the held-out episodes) when that separates better on the second half.

    python scripts/wm2_check.py --wm outputs/wm2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ego2libero.world_model2 import Denoiser2, History, StateModel, SuccessHead, sample, state_to_robot  # noqa: E402

SHARDS = ("human_v3_fixed human_v3_fixed_shift scripted_v3_shift scripted_v3 policy_phone_base_moved wm2_phone_base "
          "wm2_scripted_base wm2_teleop_base wm2_phone_shift_base wm2_smolvla_libero").split()


def load_val():
    eps = []
    for n in SHARDS:
        d = np.load(ROOT / "data/processed/wm2/shards" / f"{n}.npz")
        for e in np.flatnonzero(d["ep_val"]):
            s0, L = d["ep_start"][e], d["ep_len"][e]
            eps.append({"z": torch.from_numpy(d["z"][s0:s0 + L].astype(np.float32)).flatten(1, 2),
                        "robot": state_to_robot(torch.from_numpy(d["state"][s0:s0 + L]).double()).float(),
                        "act": torch.from_numpy(d["action"][s0:s0 + L]), "success": bool(d["ep_success"][e]), "src": n})
    return eps


@torch.no_grad()
def imagine_all(eps, den, sm, sd, steps=3):
    L = torch.tensor([len(e["z"]) for e in eps]); T = int(L.max())
    pad = lambda k, n: torch.stack([torch.cat([e[k], e[k][-1:].repeat(n - len(e[k]), *[1] * (e[k].dim() - 1))]) for e in eps])
    Z, A = pad("z", T).cuda(), pad("act", T).cuda()
    R0 = torch.stack([e["robot"][0] for e in eps]).cuda()
    hist = History(Z[:, 0], R0); r = R0
    out = [Z[:, 0]]
    for t in range(T - 1):
        hist.push_action(A[:, t])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = sample(den, hist.context(), hist.a, r, sd, steps=steps).float()
        rn = sm(r, None, hist.a[:, -4:]); hist.push_step(z, rn); r = rn
        out.append(z)
    return torch.stack(out, 1), L.cuda(), Z                              # (B,T,8,32,32)


def max_prob(head, Zi, L):
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(head(Zi[:, t])) for t in range(Zi.shape[1])], 1)
    mask = torch.arange(Zi.shape[1], device=L.device)[None] < L[:, None]
    return (p * mask).max(1).values


def best_threshold(mx, y):
    c = np.linspace(0.02, 0.98, 97)
    acc = [((mx > t) == y).mean() for t in c]
    return float(c[int(np.argmax(acc))]), float(max(acc))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default=str(ROOT / "outputs/wm2"))
    a = p.parse_args()
    wm = Path(a.wm); torch.manual_seed(0)
    ck = torch.load(wm / "denoiser.pt", map_location="cuda")
    den = Denoiser2().cuda(); den.load_state_dict(ck["ema"]); den.eval(); sd = ck["sd"]
    sm = StateModel().cuda(); sm.load_state_dict(torch.load(wm / "state_model.pt", map_location="cuda")); sm.eval()
    hk = torch.load(wm / "success_head.pt", map_location="cuda")
    head = SuccessHead().cuda(); head.load_state_dict(hk["head"]); head.eval()
    eps = load_val()
    y = np.array([e["success"] for e in eps])
    Zi, L, Zr = imagine_all(eps, den, sm, sd)
    mx_real = max_prob(head, Zr, L).cpu().numpy(); mx_im = max_prob(head, Zi, L).cpu().numpy()
    res = {"episodes": len(eps), "successes": int(y.sum()), "tau": hk["tau"],
           "real_acc_at_tau": float(((mx_real > hk["tau"]) == y).mean()), "imagined_acc_at_tau": float(((mx_im > hk["tau"]) == y).mean()),
           "imagined_best": best_threshold(mx_im, y)}
    rel = ((Zi - Zr) ** 2).mean(dim=(2, 3, 4)) / sd ** 2
    res["latent_mse_rel"] = {t: float(rel[:, t][L > t].mean()) for t in (10, 50, 100, 150)}
    # refit the head on imagined latents of the first half, test on the second half
    idx = np.random.default_rng(0).permutation(len(eps)); fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
    head2 = SuccessHead().cuda(); head2.load_state_dict(hk["head"]); head2.train()
    opt = torch.optim.AdamW(head2.parameters(), lr=3e-4)
    Lc = L.cpu().numpy()
    pos = [(i, t) for i in fit if y[i] for t in range(max(0, Lc[i] - 2), Lc[i])]
    neg = [(i, t) for i in fit for t in range(Lc[i]) if (not y[i]) or t < Lc[i] - 12]
    for s in range(2000):
        bp = [pos[k] for k in np.random.randint(len(pos), size=128)]; bn = [neg[k] for k in np.random.randint(len(neg), size=128)]
        zb = torch.stack([Zi[i, t] for i, t in bp + bn]); yb = torch.cat([torch.ones(128), torch.zeros(128)]).cuda()
        loss = F.binary_cross_entropy_with_logits(head2(zb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    head2.eval()
    mx2 = max_prob(head2, Zi, L).cpu().numpy()
    t_fit, _ = best_threshold(mx2[fit], y[fit]); t_old, _ = best_threshold(mx_im[fit], y[fit])
    res["test_half"] = {"old_head_imagined_acc": float(((mx_im[test] > t_old) == y[test]).mean()), "old_tau": t_old,
                        "refit_head_imagined_acc": float(((mx2[test] > t_fit) == y[test]).mean()), "refit_tau": t_fit}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(wm / "check.json", "w"), indent=1)
    torch.save({"head": head2.state_dict(), "tau": t_fit}, wm / "success_head_imagined.pt")


if __name__ == "__main__":
    main()
