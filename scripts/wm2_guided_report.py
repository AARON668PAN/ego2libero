"""Table of the guided-selection runs: success per can shift for the policy alone (same code, and the
shift_eval.sh numbers), with world model v2 picking chunks, and with perfect foresight (simulator).

    python scripts/wm2_guided_report.py [--tag v1]
"""
import argparse
import json
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    a = p.parse_args()
    runs = {}
    for f in sorted((ROOT / "outputs/wm2_guided").glob(f"{a.tag}_*.json")):
        s = json.loads(f.read_text())["summary"]
        runs.setdefault(s["name"], {}).setdefault(s["mode"], {})[s["shift_cm"]] = (round(s["success"] * s["episodes"]), s["episodes"])
        ref = s.get("shift_eval_success")
        if ref is not None:
            runs[s["name"]].setdefault("shift_eval", {})[s["shift_cm"]] = (round(ref * 50), 50)
    cols = [("shift_eval", "alone (shift_eval)"), ("policy", "alone (same code)"), ("q", "scorer without world model picks"), ("wm", "world model picks"), ("sim", "simulator picks")]
    for name, modes in runs.items():
        shifts = sorted({c for m in modes.values() for c in m})
        print(f"\n### {name}\n")
        print("| Can moved | " + " | ".join(t for k, t in cols if k in modes) + " |")
        print("|---|" + "---|" * sum(k in modes for k, _ in cols))
        tot = {k: [0, 0] for k, _ in cols}
        for c in shifts:
            row = []
            for k, _ in cols:
                if k not in modes:
                    continue
                if c in modes[k]:
                    s, n = modes[k][c]; tot[k][0] += s; tot[k][1] += n
                    row.append(f"{s}/{n}")
                else:
                    row.append("-")
            print(f"| {c:g} cm | " + " | ".join(row) + " |")
        print("| total | " + " | ".join(f"{tot[k][0]}/{tot[k][1]}" for k, _ in cols if k in modes) + " |")
        alone = [tot["policy"][0] + tot["shift_eval"][0], tot["policy"][1] + tot["shift_eval"][1]]
        for k in ("q", "wm", "sim"):
            if k in modes and alone[1]:
                s, n = tot[k]
                _, pv = fisher_exact([[s, n - s], [alone[0], alone[1] - alone[0]]])
                print(f"\n{k}: {s}/{n} = {s / n:.0%} against the policy alone {alone[0]}/{alone[1]} = {alone[0] / alone[1]:.0%} "
                      f"(both baselines pooled, same shifts only if complete), Fisher p = {pv:.3g}")


if __name__ == "__main__":
    main()
