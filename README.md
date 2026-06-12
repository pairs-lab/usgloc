# USGLoc — Uncertainty-Aware Semantic Visual Localization from 3D Gaussian Splatting

Reference implementation of the RA-L draft in [`paper/usgloc_ral_draft.tex`](paper/usgloc_ral_draft.tex):
a semantic 3DGS localizer that outputs a **calibrated distribution** over camera
pose (a Gaussian mixture on SE(3)) instead of a point estimate, by propagating
evidential (Dempster–Shafer) semantic uncertainty end-to-end from the map to
the pose.

## Layout

| Module | Paper section | Contents |
|---|---|---|
| `usgloc/evidential.py` | IV-A, Eq. 3–5, 7 | EDL evidence → (belief, vacuity), Dempster & Yager fusion, EDL loss, shared evidential head |
| `usgloc/primitives.py` | IV-A, Eq. 2 | Evidential primitives `(mu, Sigma, f, p, u)`, clustering, DST fusion per object, submaps |
| `usgloc/descriptors.py` | IV-B, Eq. 6 | Instance descriptors (CLIP ⊕ geometry MLPs), SGLoc-style attention scene aggregation with evidential pooling gate, InfoNCE |
| `usgloc/matching.py` | IV-C, Eq. 6 | Vacuity-gated 2D–3D similarity, submap retrieval → weighted pose hypotheses (pose mixture) |
| `usgloc/rendering.py` | IV-D, Eq. 8 | Differentiable splatting with the **vacuity channel alpha-blended like color**; pure-torch backend + `gsplat` backend |
| `usgloc/refinement.py` | IV-D, Eq. 9–10 | Tangent-space pose refinement with per-pixel weight `exp(-gamma * U_r)`, Laplace covariance from the weighted-residual Jacobian |
| `usgloc/posterior.py` | IV-D, Eq. 11–12 | SE(3) Gaussian mixture, NLL, moment-matched condensation for an EKF/UKF |
| `usgloc/metrics.py` | V | Median errors, success rate, coverage vs. χ², AUSE, coverage-ECE |
| `usgloc/calibration.py` | IV-D, V | Scalar covariance temperature: `fit_temperature` (median/MLE), `temper(_mixture)`, `CalibrationReport` |
| `usgloc/baselines.py` | V | In-framework baselines: iComMa-style photometric refinement, SGLoc-style top-1 point, random anchor, fixed-covariance wrapper |
| `usgloc/datasets.py` | V (setup) | 7Scenes/12Scenes loader + `load_gaussian_map` bridge format for externally trained evidential 3DGS maps |
| `usgloc/pipeline.py` | Alg. 1 | End-to-end `localize(image, instances) → posterior` |
| `usgloc/query.py` | IV-B | Real-data front end: FastSAM masks + CLIP + the shared EDL head (optional deps) |
| `usgloc/synthetic.py` | V (protocol) | Semi-dynamic synthetic rooms, simulated evidential observations, retrieval training |

## Install & test

```bash
cd usgloc
pip install -e .[dev]          # torch, numpy, scipy, pytest, matplotlib
python3 -m pytest tests/ -q    # 33 tests: SE(3), EDL/DST, rendering, Laplace, calibration, end-to-end
```

`gsplat` is optional (`pip install -e .[render]`) — the pure-torch renderer is
used by default and is differentiable in both reverse and forward mode (the
Laplace Jacobian uses forward-mode AD with a finite-difference fallback).

## Quick start (synthetic semi-dynamic benchmark)

```bash
python3 scripts/run_demo_synthetic.py --queries 16
```

Builds a room of splat objects (25 % of which move after mapping), constructs
the evidential map, trains the retrieval encoders, localizes the queries and
prints accuracy **and calibration** (coverage @ 68.3/95.4/99.7 %, ECE, AUSE,
NLL) for:

* `full` / `full+temp` — the method as in the paper, raw and temperature-calibrated,
* `no-weight` / `nw+temp` — ablation, `gamma = 0` (no vacuity down-weighting in Eq. 9),
* `fixed-cov` — same poses, one shared covariance (the fixed noise model a
  filter must assume when a localizer reports no uncertainty),
* `top1-point` — K = 1 ablation (top-1 retrieval, single hypothesis, point estimate),
* `icomma` — retrieval + photometric-only refinement (iComMa-style point baseline).

For real data, `scripts/run_7scenes.py` runs the same protocol on 7Scenes given
an externally trained evidential 3DGS map (`usgloc.datasets.load_gaussian_map`
documents the bridge format) and trained retrieval encoders.

`paper/fig3_calibration_template.py` regenerates the calibration figure
referenced in the draft (`paper/fig3_calibration_template.pdf`); replace its
synthetic block with arrays from `results/demo_results.json` or your real runs.

## Using a real scene

1. Train an open-vocabulary semantic 3DGS map (Gaussian Grouping / OpenGaussian)
   and export per-Gaussian means/covariances/colors/opacities, instance ids,
   CLIP features, and per-Gaussian evidential opinions from your EDL head.
2. `build_evidential_map(...)` + `make_submaps(...)` (see `usgloc/primitives.py`).
3. Encode queries with `usgloc.query.RealQueryEncoder` (FastSAM + CLIP + the
   **same** EDL head as the map — required for comparable vacuities).
4. Swap `TorchSplatRenderer` for `GsplatRenderer` and run
   `USGLocPipeline.localize`.

## Implementation notes that matter for the paper

* **Dempster's rule suppresses conflict** (Zadeh's paradox): two confident,
  contradictory opinions fuse to a *low*-vacuity compromise, so Eq. 5 alone
  will not flag moved objects. The map builder therefore records the mean
  pairwise conflict `eta` per primitive and exposes
  `effective_vacuity = min(1, u + eta)`; alternatively use `fuse_rule="yager"`,
  which routes conflict mass into vacuity. See `tests/test_evidential.py::
  test_dempster_vs_yager_on_conflict` and the discussion in `ANALYSIS.md`.
* The per-pixel weights `w(x)` are **detached** during refinement (IRLS-style);
  differentiating through them would reward moving into high-vacuity regions.
* The Laplace covariance uses `sigma_hat^2 (J^T W J)^{-1}` with `sigma_hat^2`
  estimated from the residuals, so degraded queries get honestly larger
  covariances (`tests/test_refinement.py::test_covariance_grows_when_image_is_corrupted`).
