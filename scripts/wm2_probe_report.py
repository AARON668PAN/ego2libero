"""Tables for scripts/wm2_probe_eval.py: does the world model follow edited actions, per edit and task phase?

  direction  the world model and the simulator agree on whether an edit makes things better or worse than the
             policy's own chunk (among cases where the simulator's value moves by at least 0.05; chance 50%)
  can moves  the can's vertical motion over the second (up, still, down by 1.5 cm) read off the imagined front
             frame agrees with the truth; the probe reading the simulator's own frame is the ceiling
  can error  imagined can position against the probe on the simulator's frame (same probe, so its bias cancels)

    python scripts/wm2_probe_report.py --tag probe
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["reach", "grasp", "carry", "release"]


def phase(t, ct, rt):
    return "reach" if ct < 0 or t < ct - 20 else "grasp" if t < ct + 20 else "carry" if rt < 0 or t < rt - 10 else "release"


def cat(dz, thr=0.015):
    return np.where(dz > thr, 1, np.where(dz < -thr, -1, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="probe")
    a = p.parse_args()
    rows, probes = [], None
    for f in sorted(glob.glob(str(ROOT / f"outputs/wm2_guided/{a.tag}_r*.json"))):
        d = json.loads(Path(f).read_text()); probes = d["probes"]
        ep = {e["episode"]: e for e in d["episodes"]}
        for x in d["decisions"]:
            e = ep[x["episode"]]
            rows.append(dict(x, phase=phase(x["t"], e["close_t"], e["reopen_t"])))
    W = np.array([r["wm"] for r in rows]); S = np.array([r["sim"] for r in rows])
    cs, cw, cp = (np.array([r[k] for r in rows]) for k in ("can_sim", "can_wm", "can_probe_on_sim"))
    cn, cnp = np.array([r["can_now"] for r in rows]), np.array([r["can_now_probe"] for r in rows])
    ew, es = np.array([r["eef_wm"] for r in rows]), np.array([r["eef_sim"] for r in rows])
    PH = np.array([r["phase"] for r in rows])
    true_c = cat(cs[..., 2] - cn[:, None, 2]); wm_c = cat(cw[..., 2] - cnp[:, None, 2]); pr_c = cat(cp[..., 2] - cnp[:, None, 2])
    dS, dW = S - S[:, :1], W - W[:, :1]
    out = [f"{len(rows)} decisions, {len(probes)} chunks each.", "",
           "| Chunk | Value bias (imagined - true) | Direction right | Can motion right, world model | Can motion right, probe ceiling "
           "| Moving can, world model | Moving can, ceiling | Can error | Gripper error |", "|---|" + "---|" * 8]
    for k, name in enumerate(probes):
        bias = np.mean(W[:, k] - S[:, k])
        m = np.abs(dS[:, k]) >= 0.05
        direc = f"{np.mean(np.sign(dW[m, k]) == np.sign(dS[m, k])):.0%} ({m.sum()})" if k > 0 and m.any() else "-"
        mv = true_c[:, k] != 0
        out.append(f"| {name} | {bias:+.3f} | {direc} | {np.mean(wm_c[:, k] == true_c[:, k]):.0%} | {np.mean(pr_c[:, k] == true_c[:, k]):.0%} "
                   f"| {np.mean(wm_c[mv, k] == true_c[mv, k]):.0%} ({mv.sum()}) | {np.mean(pr_c[mv, k] == true_c[mv, k]):.0%} "
                   f"| {np.median(np.linalg.norm(cw[:, k] - cp[:, k], axis=1)) * 100:.1f} cm "
                   f"| {np.median(np.linalg.norm(ew[:, k] - es[:, k], axis=1)) * 100:.1f} cm |")
    out += ["", "Direction right, by edit and phase (cases where the simulator's value moves by at least 0.05):", "",
            "| Chunk | " + " | ".join(PHASES) + " |", "|---|" + "---|" * len(PHASES)]
    for k, name in enumerate(probes[1:], 1):
        cells = []
        for ph in PHASES:
            m = (PH == ph) & (np.abs(dS[:, k]) >= 0.05)
            cells.append(f"{np.mean(np.sign(dW[m, k]) == np.sign(dS[m, k])):.0%} ({m.sum()})" if m.sum() >= 10 else f"- ({m.sum()})")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "Value bias (imagined - true), by edit and phase:", "", "| Chunk | " + " | ".join(PHASES) + " |", "|---|" + "---|" * len(PHASES)]
    for k, name in enumerate(probes):
        out.append(f"| {name} | " + " | ".join(f"{np.mean(W[PH == ph, k] - S[PH == ph, k]):+.2f}" for ph in PHASES) + " |")
    txt = "\n".join(out)
    print(txt)
    (ROOT / f"outputs/wm2_guided/probe_{a.tag}.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
