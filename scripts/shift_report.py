"""Moved-can results as a markdown table: one column per policy, 95% Wilson intervals, the total over the five
distances, and a Fisher exact test of each total against the first column's.

    python scripts/shift_report.py phone_shift_base phone_flat_shift phone_straight_shift scripted_shift_base
"""
import json
import sys
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RADII = (0, 2, 3, 4, 5)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def cell(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({lo:.0f}-{hi:.0f})"


def main():
    names = sys.argv[1:]
    res = {}
    for name in names:
        for r in RADII:
            d = json.loads((ROOT / f"outputs/rollouts/shift_{name}_r{r}.json").read_text())
            res[name, r] = (d["successes"], d["episodes"])
    tot = {n: tuple(sum(res[n, r][i] for r in RADII) for i in (0, 1)) for n in names}
    print("| Can moved | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for r in RADII:
        print(f"| {r} cm | " + " | ".join(cell(*res[n, r]) for n in names) + " |")
    print("| all five | " + " | ".join(f"{cell(*tot[n])}, {tot[n][0]}/{tot[n][1]}" for n in names) + " |")
    k0, n0 = tot[names[0]]
    for n in names[1:]:
        k, m = tot[n]
        print(f"{n} against {names[0]}: Fisher p = {fisher_exact([[k, m - k], [k0, n0 - k0]])[1]:.2g}")


if __name__ == "__main__":
    main()
