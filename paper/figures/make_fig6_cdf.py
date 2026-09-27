#!/usr/bin/env python3
"""Paired per-query translation-error CDF, ours vs ACE on identical frames.

    python3 make_fig6_cdf.py [results/final_seed0]      (run from paper/figures)
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OURS = sys.argv[1] if len(sys.argv) > 1 else "results/final_seed0"
SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
oe, ae = [], []
for s in SC:
    o = json.load(open(f"{ROOT}/{OURS}/7scenes_{s}.json"))
    a = json.load(open(f"{ROOT}/results/ace_dsacstar_100/7scenes_{s}.json"))
    O = {(r["seq"], r["frame_id"]): 100 * r["t_err"] for r in o["records"]}
    A = {(r["seq"], r["frame_id"]): 100 * r["t_err"] for r in a["records"]}
    for k in set(O) & set(A):
        oe.append(O[k]); ae.append(A[k])
oe, ae = np.sort(oe), np.sort(ae); y = np.arange(1, len(oe) + 1) / len(oe)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.semilogx(oe, y, color="#1B9E77", lw=1.9, label="USGLoc (ours)")
ax.semilogx(ae, y, color="#7570B3", lw=1.6, label="ACE (re-run)")
for x in (1, 5):
    ax.axvline(x, color="0.75", lw=0.6, zorder=0)
ax.set_xlabel("translation error (cm)"); ax.set_ylabel("fraction of queries $\\leq$ error")
ax.set_xlim(0.1, 100); ax.set_ylim(0, 1); ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(os.path.dirname(__file__), "fig6_cdf.pdf"), bbox_inches="tight")
print(f"fig6 ({OURS}): n={len(oe)} | <=0.5cm ours {100*(oe<=0.5).mean():.0f}% ACE {100*(ae<=0.5).mean():.0f}% | "
      f"<=1cm ours {100*(oe<=1).mean():.0f}% ACE {100*(ae<=1).mean():.0f}% | >5cm ours {100*(oe>5).mean():.0f}% ACE {100*(ae>5).mean():.0f}%")
