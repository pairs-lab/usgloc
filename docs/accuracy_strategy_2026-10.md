# Accuracy strategy for USGLoc on 7Scenes: decision memo (2026-10-09)

**Question.** "On 7Scenes accuracy we still lose too much. Do we give up, or try a different approach?"

**Inputs.**
- The per-scene gap diagnosis.
- Four candidate strategies: *fix-the-floor*, *change-paradigm*, *uncertainty-layer* and *reposition*.
- Three independent judges, scoring for expected gain, feasibility and reviewer outlook.

**Sourcing conventions.**
- Code citations were re-checked against HEAD `6692488` on 2026-10-09.
- Line numbers in `usgloc_full.tex` come from the diagnosis unless marked otherwise.
- Numbers from other papers or repos are marked **[ext]**. We have not reproduced them.
- Per-query result JSONs exist only on the author's machine. In this container `results/` holds just `demo_results.json`.

---

## 1. Verdict

**Continue, but reposition. Stop competing on accuracy, and let the paper stand or fall on the calibrated SE(3) posterior. First, spend one week checking that the posterior's advantage holds up against fair baselines.**

- **Every route lands between about 0.72 and 0.89 cm.** The judges' corrected central estimates are 0.88 cm for the cheap fixes inside our pipeline, 0.83 cm for a map rebuild we build ourselves, and 0.74 cm only if we adopt STDLoc's map and engine. SplatHLoc is at 0.55 cm [ext]. Our own evidential and semantic components each move accuracy by less than 0.1 cm, so no route makes accuracy our contribution.
- **Giving up the paper is not justified yet. Neither is continuing without a test.** The uncertainty claim has not been checked against fair baselines:
  - ACE's raw inlier count beats our σ_t on failure detection: ROC AUC 0.958 vs 0.952 (`usgloc_full.tex:1322`, checked).
  - We have never run a PnP-covariance baseline.
  - Our AUSE win (0.063 vs 0.166) compares two very different error distributions: 96/700 queries beyond 5 cm for us against 20/700 for ACE.
- **Stage 0 settles this in about 2 days, mostly on CPU.** If σ_t does not beat (a) an inlier count computed on *our own* poses and (b) a tempered ACE PnP covariance, by 10–20%, do not submit an RA-L method paper in its current form.
- **If it does beat them,** submit uncertainty-first at about 0.87 cm. The alternative is about 0.72–0.78 cm, but only if STDLoc's map and engine port in week 1 and at least 6 weeks of runway remain.

---

## 2. Where the gap comes from

### 2.1 Per scene (median cm)

| Scene | Ours (cascade, s0) | LoFTR floor from GT | STDLoc paper / repo [ext] | ULF-Loc [ext] | SplatHLoc [ext] | Gap vs SplatHLoc | Gap vs STDLoc paper | Main attributed cause (inferred) |
|---|---|---|---|---|---|---|---|---|
| chess | 0.43 | 0.47 | 0.46 / 0.43 | 0.42 | 0.28 | +0.15 | −0.03 | map 0.03–0.08, field gain 0.03–0.08 |
| fire | 0.48 | 0.59 | 0.57 / 0.49 | 0.44 | 0.44 | +0.04 | −0.09 | at parity |
| heads | 0.47 | 0.37 | 0.45 / 0.41 | 0.36 | 0.29 | +0.18 | +0.02 | map 0.05–0.10, matcher 0.03–0.06 |
| office | 1.08 | 1.30 | 0.86 / 0.72 | 0.73 | 0.62 | +0.46 | +0.22 | map 0.10–0.25, matcher 0.05–0.15, back end 0.03–0.15, tail ~0.10 |
| pumpkin | 0.93 | 1.17 | 0.93 / 0.91 | 0.92 | 0.70 | +0.23 | 0.00 | map 0.10–0.20 |
| kitchen | 0.81 | 0.74 | 0.63 / 0.59 | 0.65 | 0.49 | +0.32 | +0.18 | map 0.15–0.25 |
| stairs | 2.22 | 1.16 (73% reg.) | 1.42 / 1.19 | 1.38 | 1.03 | **+1.19** | **+0.80** | front end 0.60–0.90: the correct tread is not among the hypotheses |
| **avg** | **0.92** | 0.83 | 0.76 / 0.67 | 0.70 | 0.55 | **+0.37** | **+0.16** | |

**Reliability.**
- SR@5 cm is 88% for us, against 97–99.5% for the prior-free SOTA [ext].
- Beyond 5 cm we have 96/700 queries; ACE has 20/700.

**Stairs dominates the gap.** It accounts for 46% of the gap to SplatHLoc and about 70% of the gap to STDLoc.

**Contribution of each cause to the 0.37 cm average gap.** These are inferred; each range names the test that would pin it down.

| Cause | Contribution (cm) | Notes |
|---|---|---|
| Map | ~0.11 (0.08–0.18) | |
| Matcher gain | ~0.07 (0.04–0.10) | |
| Solver / back end | ~0.03 (0.01–0.05) | |
| Front end / failure tail | ~0.14 (0.12–0.16) | Stairs ≈ 0.10 of this |
| Protocol | ±0.03 | |

The tail estimate assumes a per-scene log-error σ of 0.5–0.7. A failure fraction f = 0.10 inflates a median by about 9%; f = 0.35 inflates it by about 55%.

### 2.2 Two readings of our own data that must be corrected

1. **"6/7 scenes are at their floor, so the map is the limit" is wrong as stated.** That floor was measured with LoFTR. The cascade already beats it on Chess, Fire, Office and Pumpkin. The letter's sentence "map-limited: no amount of work on retrieval or selection would move them" (`usgloc_ral_draft.tex:267`, checked) has to go.
2. **The field floors do not prove the map supports SOTA precision.** They are 0.169 cm on Chess and 0.320 cm on Office (`usgloc_full.tex:1351–1355`).
   - They were measured on the `_r1` map, on two scenes only.
   - The field removes only about 25–30% of a 1–2 cm offset: 0.267 cm from GT, 0.773 from a 1 cm start, 1.580 from a 2 cm start (`:1357–1361`).
   - A matcher that moves so little from its start gives an optimistic floor when started from GT. These floors are lower bounds.

### 2.3 What SOTA does that we don't (mechanisms)

| # | SOTA | Ours (verified in code where cited) |
|---|---|---|
| 1 | **Reference map recipe:** SfM-point initialisation, all frames, 30k iterations at 640, L1 + 0.2 D-SSIM, decaying means LR, densification to 15k, SH3, antialiasing. STDLoc's README credits antialiasing + `align_corners` with 0.76 → 0.67 [ext]. | **Depth initialisation:** raw Kinect depth is back-projected with the SfM colour focal (`--init-intrinsics` default `sfm`, `train_map.py:55`). The `depth_camera` docstring claims depth is registered to colour (`datasets.py:246–249`); it is not. The result is a ×1.11 lateral stretch of the seeds.<br>**Fit settings:** `scene_scale` is half the bounding-box diagonal of the initial cloud (`mapping.py:274`). `ssim_lambda=0`. `lr_means_final=None`. No `rasterize_mode` or antialiasing anywhere in `usgloc/` rendering.<br>**Schedule:** every 10th frame at 320×240, with densification at 500–4000 inside a 7k fit. |
| 2 | **Jointly trained 256-d feature field.** | A 64-d PCA field trained for 4k iterations on frozen geometry. The fine stage takes an integer argmax inside co-located 8×8 windows (`locfeat.py:308–358`). This causes the gain deficit above. |
| 3 | **Pose-free initial pose:** STDLoc and ULF-Loc use sparse landmarks; SplatHLoc uses top-50 retrieval + SuperPoint/LightGlue + virtual views [ext]. | CLIP top-8 anchors. On Stairs the nearest anchor is 27 cm away, about one tread, so the field's ~1 cm basin is never reached. |
| 4 | **Learned sub-pixel fine matching on the RGB render**, seeded by field coarse matches (SplatHLoc) [ext]. | None. |
| 5 | **PoseLib LO-RANSAC at 8–12 px + robust bundle, no photometric back end.** | cv2 RANSAC at 2 px + LM. Every snap then gets 60 Adam iterations at lr 3e-3 (`pipeline.py:59`), and an accepted polish gets 15 more (`pipeline.py:64`). Adam's first step is about ±3 mm / ±0.17° whatever the gradient. |

**What we have that they lack** is the posterior. Nothing else.

### 2.4 Confirmed bug and protocol risk

**Bug: the MASt3R warm-up gets a grayscale query.**
- The cascade runs `--corr-matcher stage --corr-stage-warm mast3r`, but the runner switches to RGB only for `mast3r`/`roma` (`run_7scenes.py:1013–1014`, checked).
- The luma tensor therefore reaches the main correction (`pipeline.py:576`) and the polish (`pipeline.py:860`).
- There `correction.py:1243–1245` replicates luma into 3 channels for MASt3R, while the render is in colour.
- The 0.68 cm Stairs basin that justified the cascade, and MASt3R-only Stairs at 1.99, were both measured with RGB.
- **Expected value of the fix is 0 to −0.03 cm on the average. Treat −0.03 as an upper bound.** Commit `ce7e09f` already attributes the Stairs 1.99 → 2.22 regression to the field ("the field still costs 0.23 cm").

**Protocol risk: render resolution.** `--width/--height` default to 160×120 (`run_7scenes.py:236–238`), and they set the camera used for refinement and for the Laplace covariance. Confirm that every paper run passed 640×480.

---

## 3. Already ruled out (do not repeat)

### 3.1 Ruled out by end-to-end evidence

| Idea | Evidence |
|---|---|
| Hypothesis selection on Stairs | About 20 rules tried. The photometric arbiter picks the right candidate only 30–35% of the time. |
| Wider retrieval | Shortlist 30 → 3.04, K=6 → 3.12, spread escalation → 5.65 (Stairs). |
| Denser anchors | Stride 20/10/5 → Office 1.09/1.11/1.09. |
| Virtual-view prescreen | Stairs 3.59. |
| STDLoc-style landmark pose **on our maps** | As an extra hypothesis: Stairs 2.24 vs 2.39 (48/49 paired). As a retrieval ranker: a wash. |
| LoFTR → field stage / pooled LoFTR + field | Stairs 3.37 / 3.21. |
| Pooling correction rounds | The error is a bias, not noise (`diag_round_indep`). |
| Choosing the matcher per scene | Validation and test rank matchers in opposite orders (Stairs field 0.92 vs 1.27 on validation; 3.37 vs 2.39 on test). |
| RoMa; LoFTR conf 0.05; per-round inlier-ratio gate; truncated warm-up; 12 rounds | Gate: Chess 0.47 → 0.77. Truncated warm-up: 0.70 vs 0.34 (perturbed Chess). 12 rounds: no gain, plus CUDA timeouts. |
| Prior refiner with the LoFTR correction | ACE + ours 1.01 vs ACE + GS-CPR 0.79 [ext]; Marepo + ours 1.24 vs 0.89 [ext]. |
| Evidential front-end parts as accuracy levers | Each moves accuracy by less than 0.1 cm (tab:ablation-full). |
| EKF fusion beyond Chess | Kitchen −20%, Stairs −107%. |
| K=1 posterior, i.e. Laplace alone | AUSE: Chess 0.058 → 0.373, Stairs 0.297 → 0.656. This constrains any plan: **the spread between hypotheses is what makes σ informative.** |

### 3.2 Ruled out only under confounds (these do not count against the map or solver routes)

- **r1 recipe** (20k iterations, all frames, D-SSIM, decayed LR): Stairs 2.93 vs 2.39, and 4.02 with a rebuilt field. Scored only on Stairs, with misregistered seeds.
- **Gentle densification:** Office 1.11 vs 1.09.
- **40k/3 refit:** 16.5 dB, 45% registration.
- **480×360 refit.** This and the two above perturbed the same depth-seeded recipe that is broken.
- **SH:** Office 1.09 → 1.08, on frozen geometry.
- **"Geometric" ablation:** train-side, LoFTR only, and only after accepted polishes (`:718–720`).
- **Depth supervision:** raw unregistered depth with the wrong focal; no test number.
- **Back-end bypass, with conflicting evidence:**
  - *For:* with the ACE prior, the correction alone gives 1.01 and the full pipeline 1.31 (`:1181–1189`).
  - *Against:* Stairs staged without photometric refinement gives 3.83 vs 3.37.

---

## 4. Recommended plan

The plan has three parts:
- **Stage 0:** a 2-day gate on whether the posterior is a contribution, plus correctness fixes.
- **Stage 1:** about 5 days to answer whether the map or the engine is the lever, and to screen cheap knobs.
- **Decision D1** at about day 8, then one branch.

| Stage | GPU-h | Calendar |
|---|---|---|
| 0 | ≤7 | days 1–2 |
| 1 | 20–26 | days 3–7 |
| Branch B | ~30 | ~3 weeks |
| Branch A | 60–80 | 6–8 weeks |

### Stage 0 (days 1–2, decisive)

#### S0-A. Does the posterior beat fair baselines? (CPU; ≤2 GPU-h for the ACE dump)

**Inputs** (on the author's machine):
- `results/v2_seed0` (the LoFTR config the letter's calibration uses);
- the cascade run dir from `ce7e09f`;
- `results/ace_dsacstar_100` (ACE T and inliers on the same 700 frames).

**Code.** A new `scripts/analysis/uncertainty_fair.py` reusing `load`, `auroc`, `fpr_at_recall` and `max_coverage` from `scripts/analysis/uncertainty_benefit.py:39–112`. Temperature uses `usgloc/calibration.py:63` (`fit_temperature`).

**Computations.** Pooled and per scene, with a paired bootstrap 95% CI (1000 resamples, stratified by scene):
1. **Is the win just bimodality?** On our own poses, compare σ_t against two internal baselines: σ = c/√inliers and σ = c/√max_inliers (both already in the run records). Metrics:
   - AUSE;
   - E-AURC at 1, 2 and 5 cm and on continuous error;
   - Spearman on the queries below 5 cm.
2. **Fair external baseline.** Extend `scripts/analysis/run_ace_baseline.py` to dump ACE scene coordinates and the inlier mask at ACE's final pose through its cv2 `solve_pose` path. From those, form the ACE PnP covariance σ̂²(JᵀJ)⁻¹, tempered with *our* rule τ = med(d²)/χ²₆,₀.₅. Rank ACE's own errors with four signals: ACE inliers, ACE PnP covariance, plain disagreement ‖t_ACE − t_ours‖, and the d² of T_ACE under our condensed Gaussian.
3. **Calibration repair.** Fit and compare a leave-one-scene-out (LOSO) τ, LOSO split-conformal on d², and a 2-parameter defensive mixture (1−ε)N(μ,τΣ) + εN(μ,Σ_wide). Report coverage at 68/95/99.7 and the condensed and full-GMM NLL. Run items 1 and 3 for the cascade config too.

**Success (the posterior is a contribution).** All three must hold:
- σ_t beats inlier-count-on-our-poses by ≥20% on pooled AUSE and on E-AURC(2 cm).
- σ_t beats ACE + PnP-cov + τ on pooled NLL, and on E-AURC on ≥5/7 scenes.
- The best LOSO scheme puts pooled 95% coverage within about 5 points of nominal.

**Kill → branch C.** Either of these:
- inlier-count-on-our-poses comes within 10% of σ_t on both AUSE and E-AURC;
- ACE + PnP-cov + τ comes within 10% on NLL and E-AURC and has better coverage.

#### S0-B. Correctness fixes and audits (CPU about 1 h, then GPU 3–4 h overnight)

**1. Audits.**
- Read width and height from every Table I run log or JSON header. If any paper run refined at 160×120, stop and rerun the baseline before anything else.
- Print `scene_scale` (`mapping.py:274`) against 1.1 × the maximum camera-centre distance, per scene.
- Fit an Umeyama Sim(3) between the SfM pseudo-GT and DSLAM camera centres. This should explain the "map ≈ 0.9× metric" note (`mapping.py:225–227`), which is suspiciously close to 526/585.
- Check that train and test focals are identical.

**2. Grayscale fix,** at `run_7scenes.py:1013`:
```python
if corrector is not None and (args.corr_matcher in ("mast3r", "roma")
        or (args.corr_matcher == "stage" and args.corr_stage_warm == "mast3r")):
    gray_full = gf          # RGB to MASt3R warm-up and polish
```
- `EvidentialMatchCorrector.correct` already accepts (H,W,3) and derives luma itself for LoFTR and the field (`correction.py:1234–1238`).
- The landmark corrector (`pipeline.py:506`) receives the same tensor. Check that it accepts RGB before combining this with `--landmark-init`.

**3. Rerun the paired Table I cascade,** 7 scenes, seed 0:
```bash
BASE="<copy verbatim from the ce7e09f run log: --map/--encoders/--corr-matcher stage --corr-stage-warm mast3r --corr-feats ... --top-k ...>"
for s in chess fire heads office pumpkin redkitchen stairs; do
  python scripts/run_7scenes.py $BASE --scene $s --queries 100 --width 640 --height 480 --seed 0 --out results/casc_rgbfix
done
```

**Success:** Stairs ≤2.05, average ≤0.91, and no scene worse by more than 0.03 (paired).
**Kill:** none. The fix stays because it is a correctness fix. If Stairs does not move (±0.05), the regression comes from the field, as `ce7e09f` says.

### Stage 1 (days 3–7)

#### S1-A. Floors from GT on all 7 scenes, and the Stairs overlap (GPU about 5 h)

**Floors.** On the deployed maps, for the 100 test frames per scene, run the correction from GT with MASt3R, the field alone, and the cascade:
- `scripts/analysis/exp_floor_matcher.py --split test --map-suffix _ft --feats-suffix ...`
- If it has no `stage` matcher, use `run_7scenes.py --prior-json <T_gt json> --top-k 1` instead.

**Other measurements.**
- The full pipeline with GT as the prior (`--prior-json`, `run_7scenes.py:415`), compared with the correction alone from GT.
- The Stairs overlap: per query, the end-to-end failures beyond 5 cm against the correction-from-GT failures.
- A displacement-slope test (about 1 h): `--perturb 0.005 0.5`, `0.01 0.5` and `0.02 0.5` on Chess, Office and Stairs. Regress the recovered displacement against the applied one.

**Decision rules.**

| Rule | Measurement | Reading |
|---|---|---|
| (a) | Cascade floor ≥0.6 cm averaged over Office, Pumpkin and Kitchen | The map is binding, which raises the stakes of S1-C. |
| (b) | Full-pipeline-from-GT exceeds correction-from-GT by ≥0.1 cm | The back end costs accuracy: bypass it in S1-B. |
| (c) | ≥50% of Stairs end-to-end failures also fail from GT | Stairs is partly a map problem. |
| (d) | Field slope <0.6 | Gain deficit confirmed. Field knobs go into S1-B; no separate fine-stage project. |

#### S1-B. Screening Tier-0 knobs on validation (GPU 10–12 h, plus one 4 h test run)

**Setup.**
- Validation uses train-side frames with `--val-exclude-seq` (`run_7scenes.py:404`), on Chess, Office, Stairs and Kitchen.
- Every variant is compared, paired, against S0-B.

**Variants.**

| Variant | Change |
|---|---|
| V1 | `--corr-refine-iters 0 --report-geometric`, polish refinement off, Laplace at the PnP pose, τ refit. `corr_polish` and `polish_refine_iters` are `PipelineConfig` fields (`pipeline.py:62–64`) with no CLI flag; add two flags. Prior against V1: Stairs without photometric refinement gave 3.83 vs 3.37. |
| V2 | `--corr-refine-all --corr-refine lm+huber`. Train-side perturbed floor: Office 1.41 → 0.94, Stairs 2.17 → 1.80. Never run end to end. |
| V3 | `--corr-solver poselib --corr-reproj-px 8 --corr-refine none`. |
| V4 | More field rounds with `--corr-stop-dt 5e-4`; `feat_gain` 1.5/2.0, `feat_subpix` 0.1, `feat_fine=mnn`. The last three are constructor arguments in `correction.py:250–301` with no CLI flag; add flags. |

**Selection.** Freeze ONE configuration for all scenes, then run it once on the 7-scene test.

**Success:**
- the paired validation median improves on ≥2 of the 3 gap scenes;
- no scene is worse by more than 0.05;
- the test average is ≤0.88 with SR not lower.

**Kill:** drop any knob that gains less than 0.02 cm. If none survives, the S0-B configuration is final.

#### S1-C. STDLoc map oracle and engine factorial (hard timebox of 2 person-days for the port; GPU 6–10 h)

This one experiment serves as fix-the-floor E5, change-paradigm E0/E1, reposition X7 and uncertainty-layer E3.

**1. Port STDLoc (MIT)** onto the existing torch 2.11 / cu128 gsplat stack. The released environment pins torch 2.4.1 and py3.8 [ext]. Run it on our 700 frames. Its README reports 0.67 cm on the full split [ext].

**2. Write `usgloc/io_featgs.py`**, converting their PLY into an EvidentialMap:
- Read xyz, f_dc, f_rest, opacity, scale, rot and loc_feature.
- Store scales and quaternions directly, avoiding the float32 eigh re-decomposition.
- Feed SH through `--use-sh`.
- Write the 256-d field as a `--corr-feats` sidecar with no PCA.
- Add a flag for `rasterize_mode='antialiased'` on the `rasterization()` call in `rendering.py` (about lines 266–280) and on the field render.

**3. Run on their maps,** with gates off (`--no-corr-gate`):
- (a) LoFTR and cascade floors from GT;
- (b) our fixed cascade as a prior refiner from *our* current cascade poses (`--prior-json`), which needs no retrieval encoders on their map;
- (c) an STDLoc-mode preset: `--landmark-init --landmark-sole-inliers N --corr-matcher feat --corr-rounds 4 --corr-solver poselib --corr-reproj-px 8 --corr-refine none --corr-refine-iters 0 --report-geometric`.

**Success:**
- STDLoc reproduces at ≤0.75 cm on our frames;
- on their maps, our cascade floor is ≤0.75× our-map floor on Office, Pumpkin and Kitchen;
- (b) improves those scenes by ≥0.1 cm;
- (c) is within 0.05 cm of STDLoc's own engine on Chess, Office and Stairs.

**Kill:**
- The port does not run after 2 person-days, or STDLoc is above 0.80 on our frames: drop the paradigm route.
- Their maps change our floors by less than 10%: the map is not the lever, so do no map work at all.

### Decision D1 (about day 8)

| Branch | Condition | Action | Expected average |
|---|---|---|---|
| **A** | S0-A passes AND S1-C passes AND ≥6 weeks of runway | Adopt STDLoc's trainer and engine and keep our posterior layer. | 0.72–0.78 cm, SR ≈97%+ [inferred], accuracy credited to STDLoc |
| **B** | S0-A passes AND (S1-C fails, OR their maps buy <0.1 cm, OR <6 weeks of runway) | Reposition: Tier-0 configuration, uncertainty-first letter. | 0.86–0.90 cm, SR ≈88% |
| **C** | S0-A fails | Do not submit the RA-L method paper in this form. | n/a |

### Stage 2, Branch B (about 3 weeks, about 30 GPU-h)

- **B1. Final runs.** The frozen configuration, 3 seeds, every 10th test frame (about 1,700 queries; about 18 GPU-h), replacing the 100-frame subsample. Recompute *all* accuracy and calibration numbers for the cascade configuration.
- **B2. Does "evidential" earn its place?** On the frozen configuration, run four variants: the full method, the softmax head on the matched map, γ=0, and both gates off. Use Chess and Office with seed 0 and Stairs with 3 seeds (about 10 GPU-h).
  - Keep "evidential" only if the EDL head or the pixel weight moves Stairs SR by ≥5 points, or NLL/E-AURC by ≥15%, beyond the seed spread.
  - Today the evidence points both ways. γ=0 *improves* AUSE (Chess 0.045 vs 0.058, Stairs 0.249 vs 0.297). But softmax vs evidential on Stairs gave 2.83 vs 11.60 cm, single seed.
- **B3. EKF.** In `scripts/eval_fusion.py`, add a χ² innovation gate to both arms. Tune the fixed covariance on the other sequences. Report ANEES (CPU only). If per-query Σ loses on 2 of 3 sequences again, take fusion out of the abstract.
- **B4. 12Scenes.** Fill Table II only if the maps already exist. Otherwise cut the table; do not ship `\res` placeholders.

### Stage 2, Branch A (6–8 weeks, 60–80 GPU-h)

- **A1. Own maps with STDLoc `train.py`.**
  - Initialise from SfM points re-triangulated from **training images only**, with poses fixed (COLMAP `point_triangulator`).
  - All frames, 30k iterations at 640, D-SSIM, SH3, antialiasing, joint 256-d field. Fall back to SplatHLoc's 64-d field plus decoder if 16 GB runs out.
  - Office and Stairs first.
  - Success: within 0.08 cm of the released-map result. Kill: more than 0.15 cm worse after one recipe fix.
- **A2. Evidential channel on the frozen geometry.** Run `scripts/analysis/gx_distill.py`, `scripts/reapply_head.py` and `scripts/train_encoders.py`, with the occlusion test on rendered depth. The gates must be accuracy-neutral (≤0.02 cm).
- **A3. Does the posterior survive the engine change?** This decides whether the contribution survives.
  - Compute the covariance at the PnP pose: `refinement.laplace_covariance` (`refinement.py:172`) with 0 steps, and an evidential reprojection Hessian with an n_eff correction.
  - Compare against a plain PnP Hessian + τ on the *same* correspondences.
  - Kill: the plain baseline comes within 10% on AUSE and NLL.
- **A4. Multi-source mixture.** Restore the spread between hypotheses that K=1 shows is essential: add CLIP anchors pushed through the dense rounds, plus sequential-RANSAC models.
  - Success: 95% coverage ≥88 and median within ±0.02 cm.
  - Kill: a gain of less than 3 points.
- **A5. Final runs:** 1/10 split, 3 seeds.

### Branch C options

1. A workshop or benchmark paper on pose-uncertainty evaluation for 3DGS localizers: the protocol, the fair baselines and the negative results.
2. A thesis chapter.
3. Branch A on a longer clock, as engineering parity with no unique claim. This is not an RA-L contribution by itself.

### Not in this plan, on any branch

All three judges agree on dropping these:
- the self-built reference-recipe rebuild of `mapping.py` (fix-the-floor B/E6; if the map is the lever, use STDLoc's trainer);
- the JamMa / sub-pixel fine-stage project;
- the online host-anchored uncertainty layer (unless S0-A shows our d² beats plain disagreement by a wide margin);
- further Stairs selection or retrieval variants;
- full 17k-frame runs before the final.

---

## 5. How the paper changes

### 5.1 Fix now, whatever the outcome

**Claims to remove or fix:**
- **Floor claims.** Remove "map-limited: no amount of work... would move them" (`usgloc_ral_draft.tex:267`) and "the floor is the accuracy ceiling" (`usgloc_full.tex:533–535`). The cascade contradicts both.
- **Failure-detection AUC.** Report ACE's 0.958 AUC and 13% FPR next to our 0.952 and 19%. The letter reports only ours (`usgloc_ral_draft.tex:388`).
- **AUSE 0.063 vs 0.166.** Replace it with the same-pose comparison from S0-A; as it stands it ranks two different error distributions.
- **Calibration numbers.** They come from the LoFTR 1.05 cm configuration, while the headline row is the 0.92 cm cascade. Recompute them. The Implementation paragraph (`usgloc_ral_draft.tex:262`) also describes the LoFTR configuration.
- **"Calibrated."** Keep the word only once LOSO coverage is within about 5 points of nominal. Today it is 61/74/80 pooled; Office is 48/64/75 and Stairs 45/51/55.
- **Semi-dynamic robustness.** Drop the claim: the pixel weight uses map-side vacuity only (`refinement.py:89`).

**Inconsistencies to reconcile:**
- The field floors were measured on `_r1`.
- The field's GT-start error is quoted as both 0.169 and 0.267.
- The MASt3R Stairs floor is quoted as both 0.60 and 0.68.
- "SR 85 → 88" compares against LoFTR. Against MASt3R alone it is 89 → 88.
- The prior-refinement numbers differ between documents. The letter has ACE 1.05 → 1.05 and Marepo 2.87 → 1.40 (1.16 with refinement); the report summary has 1.01 and 1.24.

### 5.2 By outcome

| | Branch A (STDLoc engine) | Branch B (reposition) | Branch C (gate fails) |
|---|---|---|---|
| **Title** | Calibrated SE(3) posteriors on evidential Feature-GS maps. Drop "evidential semantic" if A3 or B2 kills it. | Calibrated multi-hypothesis SE(3) posterior from (semantic) 3DGS maps. "Evidential" stays only if B2 passes. | No RA-L method paper. |
| **Table I** | "Ours" ≈0.72–0.78 cm. State explicitly that the accuracy is inherited from STDLoc; cite their field, trainer and PoseLib. Add SR@5/5, SR@2/2 and an "outputs Σ" column. | Keep every SOTA row. "Ours" = frozen cascade, 3 seeds on the 1/10 split (≈0.87). Add SR@5/5, SR@2/2 and an "outputs Σ" column. No selective medians in Table I. | — |
| **Accuracy sentence** | "On par with STDLoc/ULF-Loc; accuracy is the engine's, not our contribution." | "Ahead of ACE/DSAC*, within ~0.1 cm of STDLoc's paper numbers, ~0.3 cm behind SplatHLoc; the only method in the table reporting a calibrated covariance." | — |
| **Uncertainty section** | A3 table: ours vs plain PnP Hessian + τ and inlier count on the same correspondences. Coverage, NLL, AUSE, E-AURC at 1/2/5 cm. LOSO calibration. | Fair-baseline table: σ_t vs inlier count on own poses, ACE + PnP-cov + τ, and disagreement. A risk-coverage figure with the SOTA methods as points at 100% availability. LOSO calibration disclosed next to train-side τ. | A workshop or benchmark paper built on the S0-A protocol and the negative results. |
| **Diagnostics** | Headroom, floors and the Stairs grid move to the extended report as "why anchor-based render-and-match could not reach SOTA". | A short honest diagnosis: the floor depends on the matcher; Stairs is confidently wrong for every signal, ours included (σ ratio 1.3×). | A thesis chapter. |
| **EKF / fusion** | Only if NEES/ANEES pass. | Only if B3 passes; otherwise a stated limitation. | — |

---

## 6. Risks and where the judges disagreed

### 6.1 Judge scores (1–10) and corrected averages

| Proposal | Expected gain | Feasibility | Reviewer | Mean | Proposal's own central | Judges' corrected central |
|---|---|---|---|---|---|---|
| reposition | 6.5 | 7.5 | 7.5 | **7.2** | 0.86 | 0.87–0.89 |
| change-paradigm | 7 | 5.5 | 6.5 | 6.3 | 0.68 | 0.72–0.74 (≈0.78 weighted by probability) |
| uncertainty-layer | 4.5 | 4.5 | 5 | 4.7 | host's number | +0.00 by construction; standalone 0.88–0.92 |
| fix-the-floor | 5 | 4 | 4.5 | 4.5 | 0.74 | 0.82–0.83 |

### 6.2 Where the judges agreed

- Drop the accuracy race against SplatHLoc.
- Run the fair-baseline gate before anything heavy.
- The STDLoc released-map experiment (S1-C) is the most informative GPU experiment available.
- If the map is the lever, use STDLoc's trainer rather than re-engineering ours.
- Drop the fine-stage work and the online uncertainty layer.

### 6.3 Where they disagreed

1. **Paradigm switch vs repositioning.**
   - The expected-gain judge ranks change-paradigm first (7 vs 6.5) and gives it a 0.60–0.65 chance of full success.
   - The feasibility judge expects 8 weeks or more, outside the horizon.
   - The reviewer puts the chance of the route being killed at about 40%.
   - The crux on the uncertainty side: at SR ≈99% the mixture spread rarely fires, and K=1 AUSE collapses (Chess 0.058 → 0.373). So the engine change can remove the very thing that makes our σ informative.
2. **Value of the grayscale fix.** The proposals assumed −0.03 cm. The judges say 0 to −0.03, because of `ce7e09f`.
3. **Bypassing the back end.** The sign is uncertain: 1.01 vs 1.31 with the ACE prior argues for it; Stairs 3.83 vs 3.37 without photometric refinement argues against.
4. **Whether the evidential parts matter.** γ=0 and gates-off improve AUSE, but softmax vs evidential on Stairs was 2.83 vs 11.60 (single seed). B2 settles this with 3 seeds.
5. **The fix-the-floor projections.** Its Stairs 2.22 → 1.70 and Kitchen → 0.62 ignore the fidelity-robustness trade-off in our own data: Chess SR 97 → 86% with the 640 map, Kitchen-640 registering 65%, and Stairs snap rate only 37 → 40% while the floor went 1.77 → 1.41.

### 6.4 Risks

- **The most likely failure is that S0-A fails.** ACE's inlier count already wins on AUC (0.958) and on Stairs ranking (Spearman 0.56 vs 0.36). If a generic PnP covariance + τ matches us, we are left with "we evaluated it", which is not an RA-L method contribution.
- **A pass does not guarantee acceptance.** Branch B ships at about 0.87 cm with SR 88% against 97–99.5% [ext]. Some reviewers will reject on reliability whatever the framing.
- **LOSO calibration uses other scenes' test labels.** Disclose it and report it next to the train-side τ.
- **Calibration can shift under the accuracy changes.** Reporting the PnP pose or switching maps moves the point where the Laplace covariance is evaluated, so τ and coverage change. Re-verify before adopting any accuracy change.
- **Noise and selection.** The 100-frame subsample adds ±0.03–0.05 cm, the same size as the per-knob gains, and validation and test have disagreed before. Mitigations: held-out-sequence validation, one frozen configuration, 3 seeds and the 1/10 split.
- **Porting to the RTX 5070 Ti (sm_120).** STDLoc's CUDA extensions, a 256-d field render alongside MASt3R (≈9 GB) on 16 GB, and mamba for JamMa (excluded anyway).
- **Protocol.** SfM-point initialisation from a reference model that probably contains test-image points (inferred). Competitors share this; we disclose it and use training-only re-triangulation.
- **Novelty.** RIC-Loc (arXiv 2607.04722), UGS-Loc (arXiv 2603.16538, outputs a point pose) and the evidential SCR line (Chen ICRA'24, SCREP) narrow the niche [ext, not re-checked here].
- **Venue timing.** RA-L submission is rolling. The ICRA 2027 deadline (normally mid-September) has probably passed, and the next major slot is IROS, around March [ext].
- **Sunk-cost pressure.** It could push toward submitting without passing the gate. The gate exists to stop that.
