#!/usr/bin/env python3
"""Fig. 1 (teaser) from MEASURED results. No hand-drawn numbers.

Usage:  python3 make_fig1_teaser.py figures/fig1_teaser.pdf

Panel (a) -- THE CLAIM. Queries are bucketed into deciles of each method's OWN
reported confidence (ours: tempered sigma_t; ACE and SplatHLoc: inlier count,
their only signal). We plot the median actual error per decile. An informative
signal rises monotonically from its confident decile to its uncertain one; a
flat curve means the signal does not know which predictions are bad.

Panel (b) -- THE CAPABILITY. Reported 1-sigma translation uncertainty against
actual error, per query, for our method. Ours is the only one of the three that
emits a metric SE(3) covariance at all, so this panel has no counterpart: the
baselines have nothing with centimetre units to plot on the x axis.

All three methods are scored on the SAME 100 frames per scene over the five
scenes where we reproduced SplatHLoc (chess, fire, office, redkitchen, stairs).

NOTE ON AGGREGATION: rho printed in panel (a) is POOLED over all 500 queries,
which is higher than the per-scene mean (ours 0.61, SplatHLoc 0.40, ACE 0.39)
because between-scene error spread adds rankable signal. Both are honest; they
are different statistics and the caption must say which. The 3-sigma reference
lines in (b) are the translation marginal only -- they are NOT the 6-DoF
Mahalanobis coverage of Sec. V, which is over-confident at high levels.
"""
import json, glob, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
SC = ['chess', 'fire', 'office', 'redkitchen', 'stairs']

def ours(sc):
    d = json.load(open(f'results/m3u8_seed0/7scenes_{sc}.json'))
    tau = d['tau']
    return {(x['seq'], x['frame_id']): (x['t_err']*100, x['sigma_t'], tau) for x in d['records']}

def ace(sc):
    r = json.load(open(f'results/ace_dsacstar_100/7scenes_{sc}.json'))['records']
    return {(x['seq'], x['frame_id']): (x['t_err']*100, float(x['inliers'])) for x in r}

def splathloc(sc):
    f = sorted(glob.glob(f'SplatHLoc/results/verify-map_extracted_featgs_7scenes_{sc}-*/results.json'))[-1]
    o = {}
    for x in json.load(open(f)):
        m = re.match(r'(seq-\d+)/frame-(\d+)\.color\.png', x['image_name'])
        if m:
            o[(m.group(1), int(m.group(2)))] = (x['refined_TE'], float(x['refined'][-1]['inliers']))
    return o

E, U = {k: [] for k in ('ours', 'ace', 'sh')}, {k: [] for k in ('ours', 'ace', 'sh')}
SIG = []
for sc in SC:
    a, b, c = ours(sc), ace(sc), splathloc(sc)
    keys = sorted(set(a) & set(b) & set(c))
    for k in keys:
        E['ours'].append(a[k][0]); U['ours'].append(a[k][1])      # sigma_t: higher = worse
        E['ace'].append(b[k][0]);  U['ace'].append(-b[k][1])      # negate: higher = worse
        E['sh'].append(c[k][0]);   U['sh'].append(-c[k][1])
        SIG.append((a[k][1] * np.sqrt(a[k][2])) * 100)            # tempered 1-sigma, cm
n = len(E['ours'])

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.45))

STYLE = {'ours': ('USGLoc (ours): $\\sigma_t$', 'o', '-', '#1f4e79'),
         'sh':   ('SplatHLoc: inliers',         's', '--', '#c0504d'),
         'ace':  ('ACE: inliers',               '^', ':',  '#7f7f7f')}
for key in ('ours', 'sh', 'ace'):
    e, u = np.array(E[key]), np.array(U[key])
    order = np.argsort(u)                      # most confident first
    dec = np.array_split(order, 10)
    y = np.array([np.median(e[d]) for d in dec])
    y = y / y[0]                      # relative to the method's OWN confident decile
    lbl, mk, ls, col = STYLE[key]
    ax[0].plot(np.arange(1, 11), y, marker=mk, ls=ls, color=col, ms=3.5, lw=1.4,
               label=f"{lbl}  ({y[-1]:.0f}$\\times$)")
ax[0].set_xlabel('decile of the method\'s own reported confidence\n(1 = most confident)')
ax[0].set_ylabel('median error, relative to the\nmethod\'s most-confident decile')
ax[0].set_yscale('log')
ax[0].set_xticks(range(1, 11))
ax[0].legend(frameon=False, fontsize=7, loc='upper left')
ax[0].set_title('(a) how well does each method rank its own errors?', fontsize=8.5)

sig, err = np.array(SIG), np.array(E['ours'])
ax[1].scatter(sig, err, s=5, alpha=0.35, color='#1f4e79', edgecolors='none')
lo = 10**np.floor(np.log10(np.percentile(np.r_[sig, err], 1)))
hi = 10**np.ceil(np.log10(np.percentile(np.r_[sig, err], 99)))
clipped = int(np.sum((sig > hi) | (err > hi)))
g = np.logspace(np.log10(lo), np.log10(hi), 50)
for m, ls, lb in [(1, '-', r'error $=1\sigma$'), (3, '--', r'$3\sigma$')]:
    ax[1].plot(g, m*g, ls, color='k', lw=0.9, alpha=0.7, label=lb)
inside = float(np.mean(err <= 3*sig))
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
ax[1].set_xlabel(r'reported $1\sigma$ translation uncertainty (cm)')
ax[1].set_ylabel('actual error (cm)')
ax[1].legend(frameon=False, fontsize=7, loc='upper left')
ax[1].set_title('(b) ours is a metric covariance (cm), not a unitless score', fontsize=8.5)

fig.tight_layout(pad=0.4)
out = sys.argv[1] if len(sys.argv) > 1 else 'paper/figures/fig1_teaser.pdf'
fig.savefig(out, bbox_inches='tight')
print(f"wrote {out}  (n={n} queries over {len(SC)} scenes, identical frames)")
for key in ('ours', 'sh', 'ace'):
    e, u = np.array(E[key]), np.array(U[key])
    o = np.argsort(u); dec = np.array_split(o, 10)
    y = [np.median(e[d]) for d in dec]
    print(f"  {key:5s}: decile-1 median {y[0]:6.3f} cm -> decile-10 {y[-1]:7.3f} cm "
          f"(ratio {y[-1]/y[0]:5.1f}x), rho={abs(spearmanr(u,e).statistic):.3f}")
print(f"  ours: {100*inside:.1f}% of queries within 3 sigma (translation marginal)")
print(f"  panel (b) axis clipped to [{lo:g}, {hi:g}] cm; {clipped}/{n} queries fall outside")
