"""Compare success rates estimated inside world model v2 with the simulator's, for every policy and can
shift evaluated by wm2_policy_eval.py. Writes outputs/wm2_eval/report.md and report.png.

    python scripts/wm2_report.py --held-out scripted_shift_base more_base phone_mix_base shift_more
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--held-out", nargs="*", default=[], help="policies whose rollouts were not in the world model's data")
    p.add_argument("--eval-dir", default=str(ROOT / "outputs/wm2_eval"))
    a = p.parse_args()
    EVAL = Path(a.eval_dir)
    rows = []
    for f in sorted(EVAL.glob("*_r*.json")):
        d = json.load(open(f))
        if d["sim_success"] is None:
            continue
        rows.append(d)
    if not rows:
        print("no results"); return
    names = sorted({r["name"] for r in rows})
    lines = ["| Policy | Seen by the world model | " + " | ".join(f"{c:g} cm" for c in sorted({r['shift_cm'] for r in rows})) + " |",
             "|---|---|" + "---|" * len({r['shift_cm'] for r in rows})]
    for n in names:
        rs = sorted([r for r in rows if r["name"] == n], key=lambda r: r["shift_cm"])
        cells = [f"{r['wm_success']:.0%} / {r['sim_success']:.0%}" for r in rs]
        lines.append(f"| {n} | {'no' if n in a.held_out else 'yes'} | " + " | ".join(cells) + " |")
    out = ["Cells: success estimated in the world model / measured in the simulator, 50 episodes each, same layouts and shifts.", ""] + lines + [""]
    fig, ax = plt.subplots(figsize=(5, 5))
    for group, mark in (("seen", "o"), ("held out", "^")):
        sel = [r for r in rows if (r["name"] in a.held_out) == (group == "held out")]
        if not sel:
            continue
        x = np.array([r["sim_success"] for r in sel]); y = np.array([r["wm_success"] for r in sel])
        n_pol = len({r["name"] for r in sel})
        word = "policies whose rollouts trained the world model" if group == "seen" else "policies it never saw"
        ax.scatter(x, y, marker=mark, label=f"{n_pol} {word} ({len(sel)} points)", alpha=0.8)
        if len(sel) > 2:
            out.append(f"- {group}: {len(sel)} policy-shift pairs, Pearson r = {pearsonr(x, y)[0]:.2f}, "
                       f"Spearman rho = {spearmanr(x, y)[0]:.2f}, mean |error| = {np.abs(x - y).mean():.1%}, "
                       f"same outcome per episode {np.mean([r['episode_agreement'] for r in sel]):.0%}")
    x = np.array([r["sim_success"] for r in rows]); y = np.array([r["wm_success"] for r in rows])
    out.append(f"- all: {len(rows)} pairs, Pearson r = {pearsonr(x, y)[0]:.2f}, Spearman rho = {spearmanr(x, y)[0]:.2f}, "
               f"mean |error| = {np.abs(x - y).mean():.1%}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.text(0.66, 0.60, "perfect agreement", rotation=45, fontsize=8, color="0.3", ha="left", va="top", rotation_mode="anchor")
    ax.set_xlabel("success rate measured in the simulator"); ax.set_ylabel("success rate estimated by the world model")
    ax.set_title("one point = one policy at one can distance", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(EVAL / "report.png", dpi=150)
    (EVAL / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
