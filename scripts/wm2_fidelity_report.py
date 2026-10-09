"""Where can world model v2 be trusted when it picks chunks? From --mode compare runs of
wm2_guided_eval.py: at every decision the same K candidate chunks were scored by the world model
(imagined second + value head), by the scorer without a world model, and by the simulator itself
(perfect foresight, same value head). Decisions are grouped by task phase, from when the executed
gripper command first closed and later reopened.

    python scripts/wm2_fidelity_report.py --tag fid
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["reach", "grasp", "carry", "release"]


def phase(t, close_t, reopen_t):
    if close_t < 0 or t < close_t - 20:
        return "reach"
    if t < close_t + 20:
        return "grasp"
    if reopen_t < 0 or t < reopen_t - 10:
        return "carry"
    return "release"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="fid")
    p.add_argument("--min-spread", type=float, default=0.1, help="candidates differ: best minus worst true score")
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"outputs/wm2_guided/{a.tag}_*_compare_r*.json"))):
        d = json.loads(Path(f).read_text())
        ep = {e["episode"]: e for e in d["episodes"]}
        for x in d["compare"]:
            e = ep[x["episode"]]
            rows.append(dict(x, phase=phase(x["t"], e["close_t"], e["reopen_t"]), shift=d["summary"]["shift_cm"]))
    out = [f"{len(rows)} decisions from {len({(r['shift'], r['episode']) for r in rows})} episodes; candidates count as "
           f"different when the true scores spread at least {a.min_spread}.", "",
           "| Phase | Decisions | Candidates differ | Rank agreement, world model | Rank agreement, no world model "
           "| Picks the best, world model | Picks the best, state model only | Picks the best, no world model | Chance "
           "| Gain captured, world model | Gain captured, state model only | Gain captured, no world model "
           "| Gripper error after 1 s | Finger error | Score bias |",
           "|---|" + "---|" * 14]
    for ph in PHASES + ["all"]:
        R = [r for r in rows if ph == "all" or r["phase"] == ph]
        if not R:
            continue
        W, S, Q = (np.array([r[k] for r in R]) for k in ("wm", "sim", "q"))
        SO = np.array([r.get("state_only", [np.nan] * len(r["wm"])) for r in R])
        spread = S.max(1) - S.min(1); m = spread >= a.min_spread
        def rank(X):
            return np.nanmean([spearmanr(x, s).statistic for x, s in zip(X[m], S[m])]) if m.any() else np.nan
        def best(X):                      # the pick is within 0.02 of the true best
            return np.mean(S[m][np.arange(m.sum()), X[m].argmax(1)] >= S[m].max(1) - 0.02) if m.any() else np.nan
        def captured(X):                  # share of (best - average) that the pick realises
            s = S[m]; pick = s[np.arange(len(s)), X[m].argmax(1)]
            return np.mean((pick - s.mean(1)) / (s.max(1) - s.mean(1))) if m.any() else np.nan
        chance = np.mean((S[m] >= S[m].max(1, keepdims=True) - 0.02).mean(1)) if m.any() else np.nan
        eef = np.median([v for r in R for v in r["eef_err_cm"]]); fing = np.median([v for r in R for v in r["finger_err_mm"]])
        bias = np.mean(W - S)
        so_b = best(SO) if not np.isnan(SO).all() else np.nan; so_c = captured(SO) if not np.isnan(SO).all() else np.nan
        out.append(f"| {ph} | {len(R)} | {m.mean():.0%} | {rank(W):.2f} | {rank(Q):.2f} | {best(W):.0%} | {so_b:.0%} | {best(Q):.0%} "
                   f"| {chance:.0%} | {captured(W):.0%} | {so_c:.0%} | {captured(Q):.0%} | {eef:.2f} cm | {fing:.1f} mm | {bias:+.2f} |")
    txt = "\n".join(out)
    print(txt)
    (ROOT / f"outputs/wm2_guided/fidelity_{a.tag}.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
