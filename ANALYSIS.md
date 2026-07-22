# Is USGLoc a good RA-L submission? — Analysis (updated 2026-06-11)

**Short answer: yes — it is the stronger of the two drafts in this workspace,
with one positioning risk and three technical issues to fix before submission
(one of them is a real bug in Eq. 5's story, found while implementing).**

> **2026-06-11 update — SOTA survey.** A 100+-method web survey with
> adversarial verification (see [docs/sota_survey_2026-06.md](docs/sota_survey_2026-06.md))
> confirms the gap **survives in scoped form**: no published or preprint work
> outputs a *calibrated* SE(3) pose *distribution* from a 3DGS map. The
> closest competitor is **UGS-Loc (CVPR 2026, arXiv:2603.16538)** — Fisher-information
> uncertainty maps + Monte Carlo pose sampling on a 3DGS map — but it uses
> uncertainty only internally and emits a point estimate. Loc-NeRF/MCLoc
> particle localizers, VBGS-SLAM, CityLoc, and CROSS maintain uncalibrated
> beliefs in non-GS or SLAM settings; Chen et al. (ICRA 2024) bring EDL to
> scene-coordinate regression. The draft's claim, related work, positioning
> figure caption, and Table I have been rewritten accordingly, and verified
> published numbers pre-filled (UGS-Loc 0.64 cm/0.21°, STDLoc 0.76/0.24,
> GSplatLoc 1.40/0.37 per-scene, ACE 1.1/0.34, DSAC* 1.1/0.35 on 7Scenes).
> Reviewer-facing caveat: 7Scenes numbers mix DSLAM vs SfM pseudo-GT across
> papers — the table now flags this.
>
> **Gap-fill follow-up (same day, [docs/sota_gapfill_2026-06.md](docs/sota_gapfill_2026-06.md)):**
> - **Protocol decision made:** Table I is now committed to *SfM pseudo-GT*
>   (the convention of every modern 3DGS localizer) with full per-scene
>   pre-fill. Accuracy bar to beat or match: SplatHLoc 0.55 cm/0.17° <
>   UGS-Loc 0.64/0.21 < STDLoc/LoGS 0.76/0.24 < ACE+GS-CPR 0.8/0.25 (all
>   CVPR/ICLR/ICRA 2025–26, all point estimators).
> - **iComMa and SplatLoc publish no 7Scenes results at all** (despite many
>   papers tabulating them) — removed from Table I; iComMa is covered by the
>   in-framework ablation, SplatLoc belongs on 12Scenes.
> - **SGLoc's headline (0.5 cm prior-free) uses an easier protocol**: 4/7
>   scenes, DSLAM GT, omits the three hardest scenes, code is a stub repo.
>   Treated separately in the text, not as a table row.
> - **Chu & Kim full text is paywalled with no mirror** — obtaining the IEEE
>   PDF (document 11305116) is a REQUIRED action item before adopting their
>   semi-dynamic per-scene numbers. Their documented LLM robust-object
>   failure case (their Fig. 13) is now cited as motivation for our
>   continuous vacuity weighting.
> - **Claim-narrowing confirmed twice over:** uncertainty-gated matching
>   alone is not novel (ACE-G prefilters by coordinate uncertainty; UGS-Loc
>   gates RANSAC by Fisher uncertainty), and "first evidential semantic
>   Gaussian map" would be falsified by E²-BKI. The surviving claim — used
>   verbatim in the revised draft — is evidential vacuity-gated matching
>   with a *calibrated SE(3) posterior output* from a 3DGS map.
> - **Good news for the companion vMF draft (pairs_uags):** no 2024–2026
>   work maintains directional (vMF) posteriors over semantic embeddings in
>   a map; nearest is LatentBKI's Gaussian latents (RA-L 2025).

## 1. Fit and novelty

The claimed gap is real as of the draft's related work: semantic 3DGS
localizers (SGLoc, IROS 2025; Chu & Kim, RA-L 2026) output point poses;
uncertainty-aware evidential mapping (EBS, E²-BKI) does not localize; KSE-Net
models a measurement-error distribution but only for 2D image retrieval. A
localizer that returns a *calibrated* SE(3) distribution consumable by an
EKF/UKF is exactly the kind of system-with-principled-uncertainty contribution
RA-L favors, and the calibration-centric protocol (coverage vs. χ², AUSE, NLL,
ECE) is itself a contribution — no prior 3DGS localizer reports any of these.

The MAV/filter consumption story is the right framing: it converts
"uncertainty" from an add-on into the deliverable. Keep it front and center;
if an on-board EKF experiment (even a small one) is feasible, it would
preempt the most likely reviewer complaint (next point).

## 2. Main risk: integrative novelty

Every block is individually known: EDL/DST from EBS/E²-BKI, retrieval from
SGLoc, refinement from iComMa, Laplace covariance is textbook
nonlinear-least-squares. A reviewer can frame this as "known pieces, new
glue." The genuinely new pieces to emphasize:

* **Eq. 8** — alpha-blending primitive *vacuity* exactly like color, giving a
  per-pixel uncertainty weight for free inside the rasterizer (and a graded
  replacement for Chu & Kim's hard robust-object mask);
* the **evidential gate on correspondences** (Eq. 6) lifting map-side
  uncertainty into the matching stage;
* the **end-to-end propagation** map → match → refine → SE(3) mixture →
  condensed Gaussian, evaluated for calibration rather than accuracy alone.

## 3. Technical issues found while implementing (fix in the paper)

### 3.1 Dempster's rule does not produce the vacuity the method needs (bug)

Sec. IV-A claims cluster-level fusion via Eq. 5 yields a primitive "whose
vacuity reflects genuine semantic ambiguity." Numerically it does the
opposite for exactly the case the paper cares about. Two confident,
contradictory opinions (3 classes, evidence 15 on different classes,
u₁ = u₂ ≈ 0.16) fuse under Dempster's rule to **u ≈ 0.09 with conflict
η ≈ 0.69** (Yager's rule gives u ≈ 0.72 from the same inputs) — the conflict
is renormalized away (Zadeh's paradox) and a moved object ends up looking
*more* certain than its members. Verified in
`tests/test_evidential.py::test_dempster_vs_yager_on_conflict`.

Fixes (both implemented in `usgloc/evidential.py` / `primitives.py`):
1. record the accumulated pairwise conflict η per primitive and use
   `effective_vacuity = min(1, u + η)` downstream — keeps Eq. 5 intact and
   adds one line to the paper;
2. or switch the cluster fusion to **Yager's rule** (intersection masses
   unchanged, conflict mass assigned to vacuity), which is arguably the
   more principled choice for this use.

The paper must adopt one of these explicitly; DST-literate reviewers will
catch it. (E²-BKI avoids the issue because its fusion is kernel-weighted
evidence accumulation, not a Dempster product.)

### 3.2 Eq. 9's weights must be treated as constants during optimization

If gradients flow through `w(x) = exp(-γ U_r(x))`, the optimizer is rewarded
for *increasing* rendered vacuity over high-residual pixels — it can reduce
the loss by hiding misalignment instead of fixing it. The implementation
detaches `w` per iteration (IRLS-style); the paper should say so (one
sentence after Eq. 9).

### 3.3 Photometric Laplace covariances will be overconfident in practice

`σ̂²(JᵀWJ)⁻¹` reflects the local basin, but photometric residuals on real
scenes are heavy-tailed and spatially correlated (rendering artifacts,
exposure), which the i.i.d. assumption ignores; raw coverage will likely
undershoot the nominal levels on real data even when AUSE is good. Plan for a
**single scalar inflation factor (temperature) calibrated on a validation
split**, reported honestly as part of the method. This also gives the paper a
clean ablation (raw vs. tempered) and strengthens, not weakens, the
calibration story. The mixture weights (retrieval × PSNR softmax) are
likewise heuristic — fine, but state it.

## 4. Relation to the other draft (`pairs_uags`, vMF/BKI mapping)

Complementary, not competing: the vMF draft quantifies *what to trust in the
map's semantics* (mapping-side, open-vocabulary posterior); USGLoc quantifies
*how sure the robot is about where it is* (localization-side pose posterior).
If both are submitted, separate the stories cleanly and cross-cite. USGLoc is
the better near-term RA-L bet: its evaluation needs no new conjugacy theory,
the baseline field (point-pose localizers) is mature and beatable on a metric
they do not even report, and the filter-integration story is concrete
robotics. The vMF draft's core update is closer to prior art (CSS explicitly
lists an open-dictionary conjugate prior as future work), so its bar for
execution is higher.

## 5. Experimental feasibility

* 7Scenes/12Scenes/ScanNet + the Chu & Kim semi-dynamic protocol are standard;
  ACE, Marepo, DSAC*, iComMa, SplatLoc, GS-CPR, STDLoc have public code.
  SGLoc and Chu & Kim may not — budget reimplementation time or report their
  published numbers with a footnote.
* Calibration has no external baseline (nobody reports it) — the softmax
  self-ablation in the draft is the right comparison; add the tempered/raw
  Laplace ablation from §3.3.
* Known draft bugs: the 12Scenes citation points at the CLIP entry (already
  TODO-flagged). The Fig. 1/2 TikZ sources and the Fig. 3 generator script
  already live in `paper/`.

## 6. What the synthetic benchmark already shows

On the 16-query semi-dynamic room (`scripts/run_demo_synthetic.py`, CUDA):

* the evidential map separates content cleanly — mean splat vacuity **0.67
  (dynamic) vs 0.16 (static)** with no semantic thresholds;
* the full pipeline localizes to **~5–6 cm / <1° median** with 3/16 queries
  failing into a wrong retrieval basin (70–130 cm) — exactly the multimodal,
  non-Gaussian error KSE-Net describes, which is why the GMM posterior (not a
  single Gaussian) is the right output;
* the **raw Laplace covariance is overconfident** (coverage 0 %, ECE 0.50,
  huge NLL), and a scalar temperature fit on a held-out half (τ ≈ 27) recovers
  ECE ≈ 0.16 and sane NLL — direct empirical support for §3.3: temperature
  scaling must be part of the method, not an afterthought;
* per-query predicted uncertainty ranks errors well (low AUSE), i.e. the
  covariance is *informative* even before tempering — the paper's
  AUSE-vs-coverage distinction in action.

## 7. What the code in this repo gives you

Validated math (33 passing tests): SE(3) exp/log; EDL/DST including the
conflict behavior above; differentiable splatting with the vacuity channel;
refinement that provably shrinks pose error; Laplace covariances that are SPD
and grow under image degradation; sampling-consistent coverage/AUSE/ECE/NLL;
and an end-to-end synthetic semi-dynamic benchmark
(`scripts/run_demo_synthetic.py`) exercising Algorithm 1 with the paper's
ablations. The remaining work for the paper is real-data plumbing: a trained
open-vocabulary 3DGS map (Gaussian Grouping / OpenGaussian), an EDL head
trained on a segmentation set, FastSAM+CLIP weights (front end already in
`usgloc/query.py`), and the baseline runs.
