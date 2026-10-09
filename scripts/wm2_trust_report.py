"""When can world model v2 be trusted, from --mode compare2 runs of wm2_guided_eval.py: at every decision the
same K candidate chunks were imagined M times by the world model and run in the simulator, and both were
scored 5, 10, 20 and 40 steps ahead. Reports
  1. how the world model's choice holds up with the look-ahead,
  2. whether its own disagreement between imagined futures warns when its choice is wrong,
  3. what a rule that only follows the world model when it is sure would gain, offline, in true one-second value.
Intervals are 95% bootstrap intervals over episodes.

    python scripts/wm2_trust_report.py --tag trust
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def boot(values, groups, stat=np.mean, n=2000):
    """Point estimate and 95% interval, resampling whole episodes."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ids = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ids}
    est = stat(values)
    bs = [stat(values[np.concatenate([idx[g] for g in RNG.choice(ids, len(ids))])]) for _ in range(n)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(e, pct=True):
    v, lo, hi = e
    return f"{v:.0%} ({lo:.0%}-{hi:.0%})" if pct else f"{v:.2f} ({lo:.2f}-{hi:.2f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="trust")
    p.add_argument("--min-spread", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=0.02, help="a pick is right within this of the true best")
    a = p.parse_args()
    W, S, Q, E, G, PH = [], [], [], [], [], []
    for f in sorted(glob.glob(str(ROOT / f"outputs/wm2_guided/{a.tag}_*_compare2_r*.json"))):
        d = json.loads(Path(f).read_text()); cm = d["summary"]["shift_cm"]
        ep = {e["episode"]: e for e in d["episodes"]}
        for x in d["compare"]:
            W.append(x["wm"]); S.append(x["sim"]); Q.append(x["q"]); E.append(x["eef_err_cm"]); G.append(f"{cm}_{x['episode']}")
            e = ep[x["episode"]]; t, ct, rt = x["t"], e["close_t"], e["reopen_t"]
            PH.append("reach" if ct < 0 or t < ct - 20 else "grasp" if t < ct + 20 else
                      "carry" if rt < 0 or t < rt - 10 else "release")
            cps = x["checkpoints"]
    W, S, Q, E, G, PH = np.array(W), np.array(S), np.array(Q), np.array(E), np.array(G), np.array(PH)   # W (N,K,M,C)  S (N,K,C)
    N, K, M, C = W.shape
    out = [f"{N} decisions, {len(np.unique(G))} episodes, {K} candidates, {M} imagined futures each.", ""]

    out += ["## 1. Look-ahead", "",
            "| Look-ahead | Candidates differ | World model picks right | Without world model | Chance | Gripper error |",
            "|---|---|---|---|---|---|"]
    for c, h in enumerate(cps):
        s = S[:, :, c]; m = (s.max(1) - s.min(1)) >= a.min_spread
        best = s.max(1, keepdims=True)
        right_w = s[np.arange(N), W[:, :, :, c].mean(2).argmax(1)] >= best[:, 0] - a.tol
        right_q = s[np.arange(N), Q.argmax(1)] >= best[:, 0] - a.tol
        chance = (s >= best - a.tol).mean(1)
        out.append(f"| {h / 20:.2f} s | {m.mean():.0%} | {fmt(boot(right_w[m], G[m]))} | {fmt(boot(right_q[m], G[m]))} "
                   f"| {chance[m].mean():.0%} | {np.median(E[:, :, c]):.2f} cm |")

    c = cps.index(20) if 20 in cps else C - 1
    s = S[:, :, c]; w = W[:, :, :, c]; mu, sd = w.mean(2), w.std(2, ddof=1)
    m = (s.max(1) - s.min(1)) >= a.min_spread
    order = np.argsort(-mu, 1); b1, b2 = order[:, 0], order[:, 1]
    z = (mu[np.arange(N), b1] - mu[np.arange(N), b2]) / np.sqrt(sd[np.arange(N), b1] ** 2 + sd[np.arange(N), b2] ** 2 + 1e-6)
    right = s[np.arange(N), b1] >= s.max(1) - a.tol
    out += ["", f"## 2. Does the world model know when it is wrong? ({cps[c] / 20:.0f} s ahead, decisions where candidates differ)", "",
            "Confidence is the gap between its best and second-best candidate in units of the spread of its own imagined futures.", "",
            "| Confidence | Decisions | Picks right |", "|---|---|---|"]
    qs = np.quantile(z[m], [0, 1 / 3, 2 / 3, 1])
    for lo, hi, name in zip(qs[:-1], qs[1:], ("low third", "middle third", "high third")):
        sel = m & (z >= lo) & (z <= hi)
        out.append(f"| {name} ({lo:.1f} to {hi:.1f}) | {sel.sum()} | {fmt(boot(right[sel], G[sel]))} |")
    sp = sd.mean(1)
    qs2 = np.quantile(sp[m], [0, 1 / 3, 2 / 3, 1])
    out += ["", "| Spread of its imagined futures | Decisions | Picks right |", "|---|---|---|"]
    for lo, hi, name in zip(qs2[:-1], qs2[1:], ("low third", "middle third", "high third")):
        sel = m & (sp >= lo) & (sp <= hi)
        out.append(f"| {name} ({lo:.3f} to {hi:.3f}) | {sel.sum()} | {fmt(boot(right[sel], G[sel]))} |")

    out += ["", f"## 3. Following the world model only when it is sure (offline, true value {cps[c] / 20:.0f} s ahead, all decisions)", "",
            "| Rule | Mean true value of the run chunk | Share of the oracle's gain | Decisions overridden |", "|---|---|---|---|"]
    base = s[:, 0]; orc = s.max(1)
    def row(name, pick):
        v = s[np.arange(N), pick]
        gain = (v - base).sum() / max((orc - base).sum(), 1e-9)
        out.append(f"| {name} | {fmt(boot(v, G), pct=False)} | {gain:.0%} | {(pick != 0).mean():.0%} |")
    row("policy's own first sample", np.zeros(N, int))
    row("world model, one imagined future", W[:, :, 0, c].argmax(1))
    row(f"world model, mean of {M}", b1)
    z0 = (mu[np.arange(N), b1] - mu[:, 0]) / np.sqrt(sd[np.arange(N), b1] ** 2 + sd[:, 0] ** 2 + 1e-6)
    for thr in (0.5, 1, 2, 4):
        row(f"world model if its best beats the first sample by > {thr} spreads", np.where(z0 > thr, b1, 0))
    row(f"world model, mean of {M}, except while letting go", np.where(PH == "release", 0, b1))
    row("perfect foresight", s.argmax(1))
    out += ["", "## 4. By phase, 1 s ahead, world model with the mean of its imagined futures", "",
            "| Phase | Decisions | Candidates differ | Picks right | Chance |", "|---|---|---|---|---|"]
    for ph in ("reach", "grasp", "carry", "release"):
        sel = PH == ph; mm = sel & m
        out.append(f"| {ph} | {sel.sum()} | {m[sel].mean():.0%} | {fmt(boot(right[mm], G[mm]))} "
                   f"| {(s[mm] >= s[mm].max(1, keepdims=True) - a.tol).mean():.0%} |")
    # 5. picking among more candidates: true value of the pick and how much the pick is over-scored
    out += ["", "## 5. More candidates, more over-scoring (1 s ahead, all decisions)", "",
            "| Candidates | True value of the pick | Pick's score minus its true value | Average over all candidates |", "|---|---|---|---|"]
    err = mu - s
    for kk in (1, 2, 4, 8):
        tv, gap = [], []
        for _ in range(20 if kk < K else 1):
            sub = np.stack([RNG.choice(K, kk, replace=False) for _ in range(N)])
            pk = sub[np.arange(N), mu[np.arange(N)[:, None], sub].argmax(1)]
            tv.append(s[np.arange(N), pk].mean()); gap.append(err[np.arange(N), pk].mean())
        out.append(f"| {kk} | {np.mean(tv):.3f} | {np.mean(gap):+.3f} | {err.mean():+.3f} |")

    # 6. gain weighted by what is at stake
    out += ["", "## 6. Gain captured, weighted by stakes (1 s ahead, all decisions)", "",
            "Sum over decisions of (true value of the pick - average candidate) / sum of (best - average).", "",
            "| Scorer | Stakes-weighted gain |", "|---|---|"]
    den_ = (s.max(1) - s.mean(1)).sum()
    for name, pk in [("world model, mean of 3", b1), ("world model, one imagined future", W[:, :, 0, c].argmax(1)),
                     ("scorer without world model", Q.argmax(1)), ("policy's own first sample", np.zeros(N, int))]:
        out.append(f"| {name} | {(s[np.arange(N), pk] - s.mean(1)).sum() / den_:.0%} |")

    # 7. risk-coverage with thresholds fit on the other half of the episodes
    out += ["", "## 7. Acting only when confident, thresholds fit on the other half of the episodes (1 s ahead, all decisions)", "",
            "Regret = true value of the best candidate minus that of the pick. Confidence as in section 2, over all decisions.", "",
            "| Act on the most confident | Regret where it acts | Regret where it holds back | Picks right where it acts |", "|---|---|---|---|"]
    regret = s.max(1) - s[np.arange(N), b1]; right_all = s[np.arange(N), b1] >= s.max(1) - a.tol
    half = np.array([hash(g) % 2 for g in G])
    for cov in (1.0, 0.75, 0.5, 0.25):
        acts, holds, rights = [], [], []
        for fit, test in ((0, 1), (1, 0)):
            thr = np.quantile(z[half == fit], 1 - cov) if cov < 1 else -np.inf
            t_ = half == test; act = t_ & (z >= thr); hold = t_ & (z < thr)
            acts.append(regret[act]); holds.append(regret[hold]); rights.append(right_all[act])
        A_, H_, R_ = np.concatenate(acts), np.concatenate(holds), np.concatenate(rights)
        out.append(f"| {cov:.0%} | {A_.mean():.3f} | {H_.mean() if len(H_) else float('nan'):.3f} | {R_.mean():.0%} |")
    out += ["", "| Scoring of the imagined futures | True value of the pick |", "|---|---|"]
    for name, sc_ in [("mean", mu), ("lowest of 3", w.min(2)), ("mean minus one spread", mu - sd)]:
        out.append(f"| {name} | {s[np.arange(N), sc_.argmax(1)].mean():.3f} |")
    txt = "\n".join(out)
    print(txt)
    (ROOT / f"outputs/wm2_guided/trust_{a.tag}.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
