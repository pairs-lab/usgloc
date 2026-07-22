#!/usr/bin/env python3
# =============================================================================
# Fig. (calibration) TEMPLATE for the USGLoc RAL paper.
#
#   Panel (a): reliability via the normalized-squared-error histogram vs. the
#              ideal chi-square curve (cf. KSE-Net Fig. 3). If the covariance is
#              calibrated, the histogram of d^2 = e^T Sigma^{-1} e matches chi^2.
#   Panel (b): sparsification curve for AUSE. Rank queries by *predicted*
#              uncertainty and remove the worst first; a calibrated predictor
#              tracks the oracle (rank by *true* error). AUSE = area between.
#
#   THE DATA BELOW IS SYNTHETIC. Replace the marked block with your measured
#   arrays and re-run:  python3 fig3_calibration_template.py
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

def chi2_pdf(x, k):
    return x**(k/2 - 1) * np.exp(-x/2) / (2**(k/2) * gamma(k/2))

rng = np.random.default_rng(0)
N, K = 4000, 3   # K = dof (e.g., 3 for translation in R^3; set per your error)

# ============ REPLACE THIS BLOCK WITH YOUR MEASURED RESULTS ==================
# d2_*: per-query normalized squared error  e^T Sigma^{-1} e  (one scalar each)
z = rng.standard_normal((N, K))
d2_ours = np.sum(z**2, axis=1)                 # calibrated  -> ~ chi^2_K
d2_base = np.sum((1.7*z)**2, axis=1) * 0.9     # overconfident softmax -> heavy tail
# per-query error (e.g., translation in cm) and the predictor's reported uncertainty
err      = np.abs(rng.standard_normal(N)) * 5 + 1.0
unc_ours = err + rng.standard_normal(N) * 1.5  # well-correlated with error (good)
unc_rand = rng.standard_normal(N)              # uninformative ranking
# ============================================================================

def sparsification(error, ranking, n_steps=20, max_frac=0.9):
    """Mean remaining error after removing the highest-`ranking` queries first."""
    order = np.argsort(ranking)                 # ascending; keep smallest, drop largest
    fr = np.linspace(0.0, max_frac, n_steps)
    n = len(error)
    means = [error[order[:max(1, int(n*(1-f)))]].mean() for f in fr]
    return fr, np.asarray(means)

fr, m_oracle = sparsification(err, err)         # rank by TRUE error  (oracle / lower bound)
_,  m_ours   = sparsification(err, unc_ours)    # rank by predicted uncertainty (Ours)
_,  m_rand   = sparsification(err, unc_rand)    # random ranking

_d = m_ours - m_oracle
ause = float(np.sum((_d[:-1] + _d[1:]) * np.diff(fr) / 2.0))  # AUSE

# --------------------------------- plot -------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

# (a) reliability histogram
bins = np.linspace(0, 16, 41)
ax1.hist(d2_ours, bins=bins, density=True, alpha=0.55,
         color="#1B9E77", label="Ours")
ax1.hist(d2_base, bins=bins, density=True, alpha=0.45,
         color="#D95F02", label="Softmax (overconf.)")
xs = np.linspace(0.01, 16, 400)
ax1.plot(xs, chi2_pdf(xs, K), "k--", lw=1.4, label=fr"ideal $\chi^2_{{{K}}}$")
ax1.set_xlabel(r"$d^2 = \mathbf{e}^\top \Sigma^{-1}\mathbf{e}$")
ax1.set_ylabel("density")
ax1.set_xlim(0, 16)
ax1.set_title("(a) Reliability", fontsize=9)
ax1.legend(fontsize=7, frameon=False)

# (b) sparsification / AUSE
ax2.fill_between(fr, m_oracle, m_ours, color="#1B9E77", alpha=0.18,
                 label=fr"AUSE $=$ {ause:.2f}")
ax2.plot(fr, m_ours,   color="#1B9E77", lw=1.8, label="Ours (pred. unc.)")
ax2.plot(fr, m_oracle, color="#273746", lw=1.4, ls="--", label="Oracle (true err.)")
ax2.plot(fr, m_rand,   color="#888888", lw=1.4, ls=":",  label="Random")
ax2.set_xlabel("fraction of queries removed")
ax2.set_ylabel("mean trans. error (cm)")
ax2.set_title("(b) Sparsification", fontsize=9)
ax2.legend(fontsize=7, frameon=False)

# Unmistakable "this is synthetic" banner -- DELETE once real data is in.
fig.suptitle("ILLUSTRATIVE TEMPLATE — synthetic data; replace before use",
             color="#C0392B", fontsize=9, fontweight="bold", y=1.02)

fig.tight_layout()
fig.savefig("fig3_calibration_template.pdf", bbox_inches="tight")
fig.savefig("fig3_calibration_template.png", dpi=150, bbox_inches="tight")
print("wrote fig3_calibration_template.{pdf,png};  AUSE(synthetic) =", round(ause, 3))
