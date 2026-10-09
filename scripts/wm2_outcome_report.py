"""Score the candidates' measured final outcomes (scripts/wm2_outcome_check.py): does picking by the world
model raise the chance that the episode succeeds, and how well does the one-second value predict it?
Intervals are 95% bootstrap intervals over episodes.

    python scripts/wm2_outcome_report.py --tag outcome
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def boot(v, g, n=2000):
    v, g = np.asarray(v, float), np.asarray(g)
    ids = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in ids}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(ids, len(ids))])].mean() for _ in range(n)]
    return f"{v.mean():.0%} ({np.percentile(bs, 2.5):.0%}-{np.percentile(bs, 97.5):.0%})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="outcome")
    a = p.parse_args()
    P, W, S1, Q, G, T = [], [], [], [], [], []
    R = 0
    for f in sorted(glob.glob(str(ROOT / f"outputs/wm2_guided/{a.tag}_r*.json"))):
        d = json.loads(Path(f).read_text())
        for x in d["decisions"]:
            R = len(x["success"][0])
            P.append(np.mean(x["success"], 1)); W.append(np.mean(x["wm"], 1)); S1.append(x["sim1s"]); Q.append(x["q"])
            G.append(f"{d['shift_cm']}_{x['episode']}"); T.append(x["t"])
    P, W, S1, Q, G, T = (np.array(v) for v in (P, W, S1, Q, G, T))
    N, K = P.shape
    rows = np.arange(N)
    out = [f"{N} decisions from {len(np.unique(G))} episodes; every candidate played out to the end {R} times.", "",
           "| Chunk run at the decision | Episode succeeds |", "|---|---|"]
    for name, pick in [("the policy's own sample", np.zeros(N, int)), ("a random candidate", None),
                       ("scorer without world model", Q.argmax(1)), ("world model", W.argmax(1)),
                       ("simulator's one-second value", S1.argmax(1))]:
        v = P.mean(1) if pick is None else P[rows, pick]
        out.append(f"| {name} | {boot(v, G)} |")
    out.append(f"| best candidate in hindsight (optimistic, noisy) | {boot(P.max(1), G)} |")
    vary = P.max(1) > P.min(1)
    out += ["", f"Final outcomes differ between candidates at {vary.mean():.0%} of decisions. Rank correlation with the "
            "measured success rate across the candidates of those decisions:", "",
            "| Score | Spearman with final success |", "|---|---|"]
    for name, X in [("world model", W), ("simulator's one-second value", S1), ("scorer without world model", Q)]:
        r = [spearmanr(x, pp).statistic for x, pp in zip(X[vary], P[vary])]
        out.append(f"| {name} | {np.nanmean(r):.2f} |")
    low = S1 < 0.1
    out += ["", f"Candidates whose one-second value is below 0.1 (looking like failures): {low.mean():.0%} of all; "
            f"they still end in success {P[low].mean():.0%} of the time, against {P[~low].mean():.0%} for the rest.", "",
            "| Decision step | Decisions | Own sample | World model pick | Random candidate |", "|---|---|---|---|---|"]
    for t in np.unique(T):
        m = T == t
        out.append(f"| {t} | {m.sum()} | {P[m, 0].mean():.0%} | {P[m][np.arange(m.sum()), W[m].argmax(1)].mean():.0%} "
                   f"| {P[m].mean():.0%} |")
    txt = "\n".join(out)
    print(txt)
    (ROOT / f"outputs/wm2_guided/outcome_{a.tag}.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
