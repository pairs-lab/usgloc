# USGLoc — SOTA Survey & Gap Analysis for RA-L Revision (compiled 2026-06-10)

Note: Section (C) of the input was empty; gap-threat analysis below is synthesized from the GAP-THREAT-ADJACENT entries embedded in (B) plus the verified entries in (A). Entries from (B) are **unverified** unless their inline notes say otherwise and are flagged where load-bearing.

---

## 1. Gap verdict

**The novelty claim survives, but only in a precisely scoped form.** No published or preprint work was found that outputs a **calibrated** SE(3) pose **distribution** from a **3D Gaussian Splatting map** for single-query relocalization, nor any that builds an **evidential (EDL + Dempster-Shafer) semantic 3DGS map** and propagates that uncertainty to the pose posterior. However, several works force a rewording of the claim:

**Direct threats (must cite and differentiate explicitly):**

1. **UGS-Loc (Kong et al., CVPR 2026, arXiv:2603.16538)** — verified high. THE closest competitor: 3DGS pose refinement with Monte Carlo pose-particle sampling + Fisher-information geometric uncertainty maps biasing PnP-RANSAC correspondence sampling. It kills any claim of "first uncertainty-aware localization on a 3DGS map." **Survival path:** UGS-Loc uses uncertainty only *internally* and outputs a point estimate (highest-weight particle or weighted mean — verified from paper text); no semantics, no evidential map, no calibration, no SE(3) distribution output. USGLoc must claim novelty at the *output* and *map* levels, not the "uses uncertainty" level.
2. **VBGS-SLAM (arXiv:2604.02696, Apr 2026, unverified-B)** — variational Bayesian GS-SLAM maintaining posterior uncertainty over camera poses AND Gaussian map parameters. It is sequential SLAM tracking, not single-query relocalization in a prebuilt map, and reports no calibration. **Survival path:** scope the claim to *map-based (re)localization* (single-query, prebuilt map).
3. **HAL-NeRF v2 (IEEE Access 2026, DOI 10.1109/ACCESS.2026.3659994, unverified-B)** — particle-filter pose belief, partially GS-rendered training views; outputs only a MAP point estimate, no covariance/calibration. Cite and differentiate.
4. **CityLoc (arXiv:2501.08982, unverified-B)** — generates 6DoF pose *distributions* (diffusion) with 3DGS-based re-ranking. Text-conditioned localization, not image-based relocalization; distribution is an uncalibrated sampler. Cite to protect the "first *calibrated* distribution from a 3DGS map" wording.
5. **CROSS (arXiv:2605.02227, May 2026, unverified-B)** — maintains *Gaussian-mixture pose beliefs* under furniture rearrangement, but over a topological keyframe map, not 3DGS, and uncalibrated. Cite to scope "Gaussian-mixture posterior" novelty to GS-map relocalization.
6. **Evidential-localization prior art:** Chen et al. (ICRA 2024, arXiv:2310.06984, EDL scene-coordinate regression with NeRF-derived uncertainty) and SCREP (arXiv:2507.07467, EDL SCR + IMU fusion) mean USGLoc **cannot claim "first evidential learning for localization."** Differentiate via: 3DGS map, DS evidence combination, propagation through differentiable rendering, calibrated SE(3) GMM output.
7. **Calibration wording collision:** Horseshoe Splatting (ICLR 2026 submission, unverified-B) uses "calibrated" for *map/rendering* uncertainty in 3DGS. Phrase USGLoc's claim as "calibrated *pose-distribution*" calibration (e.g., via ECE/NLL on pose error), and check that paper's acceptance status before submission.
8. **Theory adjacency:** the differentiable-rendering Cramér-Rao bound paper (arXiv:2510.21785, unverified-B, venue claim unverified) derives pose covariance from a differentiable renderer — methodologically adjacent to USGLoc's Laplace covariance. It is an analysis/bound, not a localizer; cite it as theoretical support.

**Recommended claim wording:**
> "To our knowledge, USGLoc is the first visual relocalization system that (i) builds an evidential (EDL + Dempster-Shafer) semantic 3DGS map, (ii) propagates map-level semantic and geometric uncertainty through retrieval, matching, and differentiable-rendering refinement, and (iii) outputs a *calibrated* Gaussian-mixture pose posterior on SE(3). Prior 3DGS localizers either ignore uncertainty entirely [GSplatLoc, STDLoc, SplatHLoc, GS-CPR, ...], use it only internally and emit a point estimate [UGS-Loc, Hi2-GSLoc, GeomGS], or maintain uncalibrated pose beliefs over non-GS or SLAM settings [Loc-NeRF, NuRF, MCLoc, HAL-NeRF, VBGS-SLAM, CROSS]."

---

## 2. Baselines to ADD to comparison tables (prioritized)

| # | Method | Venue | Why add | Code | Published 7Scenes numbers (median trans/rot) |
|---|--------|-------|---------|------|---------------------------------------------|
| 1 | **UGS-Loc** | CVPR 2026 (arXiv:2603.16538) | Mandatory: closest uncertainty-aware 3DGS competitor | Yes — https://github.com/kmk97/UGSLoc (pretrained models pending) | **Avg 0.64 cm / 0.21°** (vs GS-CPR 0.8 cm / 0.25°); 12Scenes 0.4 cm / 0.18°, 99.1% @2cm/2°; ~30% trans. error reduction vs GS-CPR on Cambridge |
| 2 | **GSplatLoc (Sidorov et al.)** | IROS 2025 (arXiv:2409.16502) | Feature-attached-to-Gaussians coarse-matching baseline, no uncertainty | Yes — https://github.com/haksorus/gsplatloc | Fine (v3 Table I) **avg 1.40 cm / 0.37°**: Chess 0.39/0.13, Fire 0.91/0.29, Heads 0.94/0.50, Office 1.41/0.32, Pumpkin 1.41/0.26, Redkitchen 1.32/0.29, Stairs 3.44/0.82. (Base variant 2.53/0.68 — do not mix.) Cambridge Fine avg 16 cm / 0.48° |
| 3 | **STDLoc** (already cited — pre-fill table) | CVPR 2025 (arXiv:2503.19358) | Strongest published GS-map relocalizer numbers | Yes — https://github.com/zju3dv/STDLoc | **Avg 0.76 cm / 0.24°** (vs DSAC* 1.07/0.35, ACE 1.10/0.34, NeRFMatch 2.71/0.73); Cambridge avg 10.1 cm / 0.14° |
| 4 | **SplatHLoc** | CVPR 2026 (arXiv:2603.29185) | Concurrent SOTA feature-GS relocalizer (must cite as concurrent) | Yes — https://github.com/HqiTao/SplatHLoc (MIT) | Claims new SOTA incl. 7Scenes; **exact medians not extracted — pull from PDF tables before submission (low confidence)** |
| 5 | **GS-SMC** | arXiv-only 2025 (2508.17876) | Newest refinement competitor to GS-CPR-style methods | No | Marepo+GS-SMC **avg 1.13 cm / 0.34°**: Chess 0.64/0.20, Fire 0.88/0.36, Heads 0.70/0.43, Office 1.4/0.37, Pumpkin 1.4/0.28, RedKitchen 1.0/0.26, Stairs 1.9/0.54; Cambridge (DFNet init) avg 0.16 m / 0.29°. Note: 53.3%/56.9% reductions are vs NeFeS, not vs initializer |
| 6 | **GSFFs** | CVPR 2025 (arXiv:2507.23569) | Closest feature-field-on-3DGS relative; segmentation-rendering refinement parallels USGLoc's semantic refinement | No | SfM-GT, feature variant: Chess 0.4/0.19, Fire 0.6/0.26, Heads 0.5/0.36, Office 1.0/0.31, Pumpkin 1.3/0.38, Redkitchen 0.6/0.23, **Stairs 25.1/0.63 (failure case — do not quote sub-cm averages without it)** |
| 7 | **HGSLoc** | ICRA 2025 (arXiv:2409.10925, DOI 10.1109/ICRA55743.2025.11127431) | Plug-and-play heuristic GS refinement on ACE/Marepo — direct refinement-stage baseline | Yes — https://github.com/anchang699/HGSLoc | Reports improved ratios @1cm/1° when refining ACE/Marepo (exact medians in paper tables; Deep Blending exps refine HLoc, not ACE/Marepo) |
| 8 | **LoGS** | ICRA 2025 (arXiv:2410.11505, IEEE 11127759) | Hierarchical GS pipeline incl. few-shot mapping angle | No | SfM-GT full: **0.4–1.6 cm / 0.10–0.43°** (chess 0.4/0.10 … stairs 1.6/0.43); DSLAM-GT full: 1.0–4.0 cm / 0.62–1.13°; Cambridge full 4.1–14.6 cm / 0.09–0.31°. No 12Scenes (claim in earlier draft was wrong) |

**Honorable mention for the uncertainty table specifically:** HR-APR (ICRA 2024, scalar uncertainty gate, 27.4% refinement-overhead cut on 7Scenes, code: https://github.com/lck666666/HR-APR) and MCLoc (CVPR 2024, particle render-and-compare, code: https://github.com/ga1i13o/mcloc_poseref) as the uncertainty-adjacent / multi-hypothesis baselines.

Caveat for all tables: 7Scenes numbers mix DSLAM vs SfM pseudo-GT across papers (GRLoc, LoGS, GSFFs report SfM-pGT; older baselines DSLAM-GT). Add a GT-convention column or footnote — reviewers check this.

---

## 3. Related-work additions (ready-to-insert prose)

### Visual Localization (retrieval / regression / matching)
"Beyond classical APR and SCR pipelines, recent work regresses geometric representations rather than poses directly: GRLoc (arXiv 2025) predicts raymaps and pointmaps as the inverse of novel-view synthesis, reporting state-of-the-art APR results on 7-Scenes (SfM pseudo-GT), while ACE-G (arXiv 2025) augments scene-coordinate regression with per-coordinate uncertainty for generalization. At the object level, SceneGraphLoc (ECCV 2024) localizes queries in multi-modal 3D scene graphs and AirLoc (arXiv 2023) performs room-level object-based relocalization, but neither recovers a metric 6-DoF pose distribution. None of these methods reports calibrated pose uncertainty."

### Rendering-based localization (NeRF / 3DGS maps)
"Localization against 3DGS maps has expanded rapidly: descriptor-distilled matching (GSplatLoc, IROS 2025; STDLoc, CVPR 2025; SplatHLoc, CVPR 2026; GSVisLoc, ICCVW 2025), render-and-compare refinement (HGSLoc, ICRA 2025; LoGS, ICRA 2025; GS-CPR, ICLR 2025; GS-SMC, arXiv 2025; GauLoc, Pacific Graphics/CGF 2024; GSFeatLoc, arXiv 2025), feature-field alignment (GSFFs, CVPR 2025), and domain-specific systems for driving, UAVs, and event cameras (3DGS-ReLoc; Hi2-GSLoc; LSGS-Loc; GS-EVT, ICRA 2025; Splat-Nav, T-RO 2025). Most relevant, UGS-Loc (CVPR 2026) refines poses in a 3DGS map via Monte Carlo pose sampling guided by Fisher-information geometric uncertainty — yet, like all of the above, it ultimately emits a single point estimate. Particle-based localizers over radiance fields (Loc-NeRF, ICRA 2023; MCLoc, CVPR 2024; Fast Loc-NeRF, arXiv 2024; NuRF, IEEE TCDS 2025; HAL-NeRF, IEEE Access 2026) maintain nonparametric pose beliefs but report only MAP estimates without calibration, and none operates on an evidential semantic GS map."

### Uncertainty in localization and neural maps
"Pose-level uncertainty has been pursued via scalar gates (HR-APR, ICRA 2024), variational and generative posteriors (Zangeneh et al., ICRA 2023, IROS 2024, and SCIA 2025; CVAE pose regression), probabilistic PnP layers with SE(3) densities (EPro-PnP, CVPR 2022 / TPAMI 2024), and statistically guaranteed pose sets (conformal PURSE, CVPR 2023; CLOSURE, RSS 2024; SLUE, arXiv 2025) — all either map-free, object-centric, or uncalibrated at the pose-distribution level. Evidential learning has reached localization through EDL scene-coordinate regression (Chen et al., ICRA 2024; SCREP, arXiv 2025) and semantic-feature error models (Wu & Campbell, ICRA 2025), and uncertainty-aware semantic mapping through the BKI line and its evidential successors (EvidMTL, arXiv 2025; E2-BKI, RA-L 2026; Continuous Semantic Splatting, ICRA 2025). On the map side, FisherRF (ECCV 2024), Bayes' Rays (CVPR 2024), and recent 3DGS-uncertainty works (PRIMU; VarSplat; view-dependent and predictive photometric uncertainty, arXiv 2025-2026) quantify rendering or per-primitive uncertainty, and WildGS-SLAM (CVPR 2025) uses per-pixel uncertainty to weight rendering losses — but in all cases the uncertainty remains internal to mapping or tracking and is never propagated into a calibrated SE(3) pose posterior, which is the gap USGLoc fills."

---

## 4. Citation corrections for existing draft entries

| Entry | Correction |
|---|---|
| **SGLoc** | Correct as cited. Full record: B. Xu, S. Zhu, H. Wang (SJTU), "SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation," IROS 2025, pp. 8872-8878, DOI 10.1109/IROS60139.2025.11245850; arXiv:2507.12027; code https://github.com/IRMVLab/SGLoc. |
| **Chu & Kim** | Update from "RA-L 2026" placeholder to: B.-T.-H. Chu and G.-W. Kim, "Semantic-Augmented 3D Gaussian Splatting for Visual Localization in Complex Indoor Environments," IEEE RA-L, vol. 11, no. 3, pp. 3852-3859, Mar. 2026, DOI 10.1109/LRA.2025.3643336. (DOI year prefix is 2025 although the issue is 2026 — keep DOI as-is.) |
| **E2-BKI** | J. Kim, M. Jeon, J. Min, K. Kwak, J. Seo, IEEE RA-L, vol. 11, no. 3, pp. 2378-2385, 2026, DOI 10.1109/LRA.2026.3653367; arXiv:2509.11964. |
| **KSE-Net** | **Title is wrong in draft.** Actual title: "Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles," Q. Wu and M. Campbell (Cornell), ICRA 2025, pp. 8034-8040, DOI 10.1109/ICRA55743.2025.11127943; arXiv:2506.15851. KSE-Net is only the internal model name — cite by full title, mention KSE-Net in text only. |
| **12Scenes** | Cite as: J. Valentin, A. Dai, M. Niessner, P. Kohli, P. Torr, S. Izadi, C. Keskin, "Learning to Navigate the Energy Landscape," 3DV 2016, pp. 323-332, DOI 10.1109/3DV.2016.41; arXiv:1603.05772. |
| **iComMa** | **Not ICRA/IROS.** arXiv-only preprint: arXiv:2312.09031 (v1 Dec 2023, v2 Mar 2024), Sun et al. No proceedings record exists — cite as preprint. |
| **GS-CPR vs GSLoc** | **Duplicate-citation hazard.** GS-CPR (ICLR 2025, arXiv:2408.11085) *is* the paper formerly titled "GSLoc" (v1, Aug 2024) — same authors (Liu, Chen, Bhalgat, et al., HKUST+Oxford). If the draft cites both "GSLoc (2408.11085)" and "GS-CPR," merge them. If a separate GSLoc baseline is intended, point it to the *distinct* GSLoc by Botashev, Pyatov, Ferrer, Lefkimmiatis, IROS 2024, pp. 5664-5671, DOI 10.1109/IROS58592.2024.10801919, arXiv:2410.06165. Audit every "GS*Loc" bib entry — further collisions exist (GSplatLoc x2, GSFeatLoc, GSVisLoc, GS-ReLoc, ReLoc-3DGS, 3DGS-Loc ASME). |
| **STDLoc** | Correct attribution: Z. Huang, H. Yu, Y. Shentu, J. Yuan, G. Zhang, "From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting," CVPR 2025; arXiv:2503.19358; code https://github.com/zju3dv/STDLoc. (Note: NOT Hong Zhang's group — SplatHLoc, which builds on STDLoc, is from Hong Zhang's SUSTech group; do not describe them as same-group successors.) |

---

## 5. Full method matrix

Legend: U? = outputs/uses pose-relevant uncertainty (out = outputs distribution/covariance; int = internal only). Entries marked * are from the unverified context list (B).

| Name | Venue / Year | Category | U? | 7Scenes (median) | Code |
|---|---|---|---|---|---|
| UGS-Loc | CVPR 2026 | uncertainty-loc (3DGS) | int only, point output | 0.64 cm / 0.21° avg | github.com/kmk97/UGSLoc |
| SplatHLoc | CVPR 2026 | 3dgs-loc | no | SOTA claimed; medians TBD (low conf) | github.com/HqiTao/SplatHLoc |
| GSplatLoc (Sidorov) | IROS 2025 | 3dgs-loc | no | 1.40 cm / 0.37° (Fine) | github.com/haksorus/gsplatloc |
| GSplatLoc (Zeller & Wu) | arXiv 2024 | 3dgs-loc | no | — (Replica/TUM, ATE-style metrics) | github.com/AtticusZeller/GsplatLoc |
| HGSLoc | ICRA 2025 | 3dgs-loc | no | ratios @1cm/1° over ACE/Marepo | github.com/anchang699/HGSLoc |
| LoGS | ICRA 2025 | 3dgs-loc | no | 0.4–1.6 cm / 0.10–0.43° (SfM-GT full) | no |
| STDLoc | CVPR 2025 | 3dgs-loc | no | 0.76 cm / 0.24° avg | github.com/zju3dv/STDLoc |
| GS-CPR (ex-GSLoc) | ICLR 2025 | 3dgs-loc | no | refines ACE/DFNet (see paper tables) | xrim-lab.github.io/GS-CPR |
| GSLoc (Botashev) | IROS 2024 | 3dgs-loc | no | — | no |
| GSVisLoc | ICCVW 2025 (CALIPOSE) | 3dgs-loc | no | evaluated; Cambridge avg 17 cm / 0.36° (7S medians not extracted) | no — "coming soon" (project page only) |
| GSFFs | CVPR 2025 | 3dgs-loc (feature field) | no | 0.4–1.3 cm / 0.19–0.38° (6 scenes), Stairs 25.1 cm | no |
| GSFeatLoc | arXiv 2025 | 3dgs-loc | no | — (Synthetic NeRF / Mip-NeRF360 / T&T) | github.com/jongwonjlee/gsfeatloc |
| GS-SMC | arXiv 2025 | 3dgs-loc | no | 1.13 cm / 0.34° (Marepo init) | no |
| GauLoc | CGF / Pacific Graphics 2024 | 3dgs-loc | no | in paper Table 1 (not extracted) | repo placeholder only (xinzhe11/GauLoc) |
| 3DGS-ReLoc | arXiv 2024 | 3dgs-loc | no | — (KITTI-360) | no |
| 3DGS-Loc (ASME) | ASME JAVS 5(3):031004, 2025 | 3dgs-loc | no | — (KITTI-360 only; journal ver. of 2403.11367) | no |
| Hi2-GSLoc | arXiv 2025 | 3dgs-loc | binary reliability filter only | — (Cambridge ~8.4 cm / 0.15°; remote sensing) | no |
| LSGS-Loc | arXiv 2026 (RA-L under review) | 3dgs-loc | no | — (GauUscene UAV) | github.com/xzhang-z/LSGS-Loc |
| CurvLoc | IJCV 134(3), 2026 | 3dgs-loc | no | in Springer table pages (not extracted here) | no |
| R3GS | arXiv 2025 (ICRA claim unverified — cite as preprint) | 3dgs-loc | no | — | no |
| LEGS | MDPI J. Imaging 12(2):84, 2026 | 3dgs-loc (training-data aug.) | no | — (Cambridge: KC 23/0.5, Shop 5.4/0.3, StMary 17/0.5, OldHosp 27/0.5) | no |
| SplatPose (DARS-Net) | IROS 2025 | 3dgs-loc | no | — (12Scenes; name collisions: CVPR'24 anomaly, ACM MM'25) | github.com/Bellakira7/SplatPose |
| GS-EVT | ICRA 2025 | 3dgs-loc (event) | no | — | github.com/ChillTerry/GS-EVT |
| Splat-Nav / Splat-Loc | T-RO 41:2765-2784, 2025 | 3dgs-loc (robot) | no | — | github.com/chengine/splatnav |
| SGLoc | IROS 2025 | semantic-3dgs-loc | no | 7Scenes+12Scenes (medians in paper) | github.com/IRMVLab/SGLoc |
| Chu & Kim | RA-L 11(3):3852-3859, 2026 | semantic-3dgs-loc | no | not extracted (paywalled) | no |
| GRLoc | arXiv 2025 (v2 May 2026) | apr | no | 2.75 cm / 0.75° avg (SfM-pGT — not DSLAM-comparable) | no |
| HR-APR | ICRA 2024 | uncertainty-loc | scalar gate | SOTA APR maintained; -27.4% refinement cost | github.com/lck666666/HR-APR |
| Zangeneh et al. (epistemic APR) | SCIA 2025, LNCS 15726:180-195 | uncertainty-loc | out (likelihood/epistemic) | evaluated (numbers not extracted) | no |
| KSE-Net paper (Wu & Campbell) | ICRA 2025 | uncertainty-loc | out (2D GMM error model) | — (Ithaca365) | no |
| MCLoc* | CVPR 2024 | nerf-loc (particles) | int, point output | evaluated (see paper) | github.com/ga1i13o/mcloc_poseref |
| Loc-NeRF* | ICRA 2023 | nerf-loc | out (particle belief, uncal.) | — | github.com/MIT-SPARK/Loc-NeRF |
| Fast Loc-NeRF* | arXiv 2024 | nerf-loc | out (uncal.) | — | github.com/kmk97/fast-loc-nerf |
| NuRF* | IEEE TCDS 2025 | nerf-loc | out (particle, uncal.) | — | no |
| HAL-NeRF (+v2)* | arXiv 2025 / IEEE Access 2026 | nerf-loc / uncertainty-loc | particle belief, MAP output | 2.5 cm / 0.59° avg (v1 abstract) | no |
| CaLDiff* | arXiv 2023 | nerf-loc | implicit (diffusion) | — | no |
| CityLoc* | arXiv 2025 | 3dgs-loc (text) | out (uncal. diffusion) | — | no |
| VBGS-SLAM* | arXiv 2026 | GS-SLAM | out (variational, uncal.) | — | no |
| CROSS* | arXiv 2026 | topo-loc | out (GMM belief, uncal.) | — | no |
| EPro-PnP / v2* | CVPR 2022 / TPAMI 2024 | uncertainty-loc (object) | out (SE(3) density) | — | github.com/tjiiv-cprg/EPro-PnP-v2 |
| PURSE / Conformal Keypoints* | CVPR 2023 | uncertainty-loc (object) | out (guaranteed sets) | — | github.com/NVlabs/ConformalKeypoint |
| CLOSURE* | RSS 2024 | uncertainty-loc | out (set bounds) | — | yes |
| SLUE* | arXiv 2025 | uncertainty-loc (object) | out (ellipsoid bounds) | — | github.com/MIT-SPARK/PoseUncertaintySets |
| SCREP* | arXiv 2025 | scr (evidential) | out (EDL) | — | no |
| Chen et al. EDL-SCR* | ICRA 2024 (arXiv 2310.06984) | scr (evidential) | out (EDL) | — | no |
| NVINS* | IROS 2024 | uncertainty-loc | out (MC-dropout/ensemble, fused) | — | no |
| FisherRF* | ECCV 2024 Oral | uncertainty-mapping | map-side | — | github.com/JiangWenPL/FisherRF |
| Bayes' Rays* | CVPR 2024 | uncertainty-mapping | map-side (Laplace) | — | bayesrays.github.io |
| WildGS-SLAM* | CVPR 2025 | GS-SLAM | int per-pixel | — | github.com/GradientSpaces/WildGS-SLAM |
| E2-BKI | RA-L 11(3):2378-2385, 2026 | uncertainty-mapping | map-side (EDL+BKI) | — | project page (kjyoung.github.io) |
| Continuous Semantic Splatting* | ICRA 2025 | uncertainty-mapping | map-side (conjugate semantics) | — | no |
| EvidMTL* | arXiv 2025 | uncertainty-mapping | map-side (EDL) | — | no |
| Diff.-Rendering Cramér-Rao bound* | arXiv 2025 (venue claims unverified) | uncertainty theory | out (CRLB covariance) | — | no |

---

## 6. Sources

Core competitors / baselines:
- https://arxiv.org/abs/2603.16538 ; https://kmk97.github.io/UGSLoc/ ; https://github.com/kmk97/UGSLoc (UGS-Loc)
- https://arxiv.org/abs/2409.16502 ; https://gsplatloc.github.io/ ; https://github.com/haksorus/gsplatloc (GSplatLoc, IROS 2025)
- https://arxiv.org/abs/2412.20056 ; https://github.com/AtticusZeller/GsplatLoc (GSplatLoc, Zeller)
- https://arxiv.org/abs/2603.29185 ; https://hqitao.github.io/SplatHLoc ; https://github.com/HqiTao/SplatHLoc (SplatHLoc)
- https://arxiv.org/abs/2409.10925 ; https://doi.org/10.1109/ICRA55743.2025.11127431 ; https://github.com/anchang699/HGSLoc (HGSLoc)
- https://arxiv.org/abs/2410.11505 ; https://ieeexplore.ieee.org/document/11127759/ (LoGS)
- https://arxiv.org/abs/2503.19358 ; https://zju3dv.github.io/STDLoc/ ; https://github.com/zju3dv/STDLoc (STDLoc)
- https://arxiv.org/abs/2408.11085 ; https://xrim-lab.github.io/GS-CPR/ (GS-CPR)
- https://arxiv.org/abs/2410.06165 ; https://api.crossref.org/works/10.1109/iros58592.2024.10801919 (GSLoc, Botashev)
- https://arxiv.org/abs/2508.18242 ; https://gsvisloc.github.io/ ; https://ieeexplore.ieee.org/document/11375573/ (GSVisLoc)
- https://arxiv.org/abs/2507.23569 ; https://openaccess.thecvf.com/content/CVPR2025/html/Pietrantoni_Gaussian_Splatting_Feature_Fields_for_Privacy-Preserving_Visual_Localization_CVPR_2025_paper.html (GSFFs)
- https://arxiv.org/abs/2504.20379 ; https://github.com/jongwonjlee/gsfeatloc (GSFeatLoc)
- https://arxiv.org/abs/2508.17876 (GS-SMC)
- https://onlinelibrary.wiley.com/doi/10.1111/cgf.15256 ; http://chenming-wu.github.io/pdfs/gauloc_cgf.pdf (GauLoc)
- https://arxiv.org/abs/2403.11367 ; https://doi.org/10.1115/1.4069346 (3DGS-ReLoc / ASME 3DGS-Loc)
- https://arxiv.org/abs/2507.15683 (Hi2-GSLoc) ; https://arxiv.org/abs/2604.05402 ; https://github.com/xzhang-z/LSGS-Loc (LSGS-Loc)
- https://link.springer.com/article/10.1007/s11263-026-02748-9 (CurvLoc) ; https://arxiv.org/abs/2505.15294 (R3GS)
- https://www.mdpi.com/2313-433X/12/2/84 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12941419/ (LEGS)
- https://arxiv.org/abs/2503.05174 ; https://github.com/Bellakira7/SplatPose (SplatPose) ; https://arxiv.org/abs/2409.19228 ; https://github.com/ChillTerry/GS-EVT (GS-EVT)
- https://arxiv.org/abs/2403.02751 ; https://dl.acm.org/doi/10.1109/TRO.2025.3552348 ; https://github.com/chengine/splatnav (Splat-Nav)
- https://arxiv.org/abs/2511.13864 (GRLoc)

Semantics / citations to correct:
- https://arxiv.org/abs/2507.12027 ; https://api.crossref.org/works/10.1109/iros60139.2025.11245850 ; https://github.com/IRMVLab/SGLoc (SGLoc)
- https://ieeexplore.ieee.org/document/11305116/ ; https://api.crossref.org/works/10.1109/lra.2025.3643336 (Chu & Kim)
- https://arxiv.org/abs/2509.11964 ; https://api.crossref.org/works/10.1109/LRA.2026.3653367 (E2-BKI)
- https://arxiv.org/abs/2506.15851 ; https://api.crossref.org/works/10.1109/icra55743.2025.11127943 (Wu & Campbell)
- https://arxiv.org/abs/1603.05772 ; https://api.crossref.org/works/10.1109/3dv.2016.41 (12Scenes)
- https://arxiv.org/abs/2312.09031 (iComMa)

Uncertainty / posterior / gap-threat works:
- https://arxiv.org/abs/2402.14371 ; https://github.com/lck666666/HR-APR (HR-APR)
- https://arxiv.org/abs/2504.07260 ; https://doi.org/10.1007/978-3-031-95918-9_13 (Zangeneh SCIA 2025)
- https://arxiv.org/abs/2410.04989 ; https://arxiv.org/abs/2301.02086 (Zangeneh CVAE/ICRA 2023)
- https://arxiv.org/abs/2303.12787 ; https://github.com/tjiiv-cprg/EPro-PnP-v2 (EPro-PnP)
- https://arxiv.org/abs/2303.12246 ; https://github.com/NVlabs/ConformalKeypoint (PURSE) ; https://arxiv.org/abs/2403.09990 (CLOSURE) ; https://arxiv.org/html/2511.21666v1 (SLUE) ; https://arxiv.org/pdf/2309.11018 ; https://arxiv.org/pdf/2303.02207 (conformal pose/VO)
- https://arxiv.org/abs/2404.01400 (NVINS) ; https://arxiv.org/abs/2209.09050 (Loc-NeRF) ; https://arxiv.org/abs/2406.12202 (Fast Loc-NeRF) ; https://arxiv.org/abs/2406.00312 (NuRF) ; https://arxiv.org/abs/2404.10438 (MCLoc) ; https://arxiv.org/abs/2504.08901 (HAL-NeRF) ; https://arxiv.org/abs/2312.15242 (CaLDiff) ; https://arxiv.org/abs/2501.08982 (CityLoc)
- https://arxiv.org/abs/2604.02696 (VBGS-SLAM) ; https://arxiv.org/abs/2605.02227 (CROSS) ; https://arxiv.org/abs/2510.21785 (CRLB) ; https://arxiv.org/abs/2507.07467 (SCREP) ; https://arxiv.org/abs/2310.06984 (Chen et al. EDL-SCR)
- https://arxiv.org/abs/2311.17874 (FisherRF) ; https://arxiv.org/abs/2309.03185 (Bayes' Rays) ; https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.pdf (WildGS-SLAM) ; https://arxiv.org/pdf/2508.02443 (PRIMU) ; https://arxiv.org/abs/2603.09673 (VarSplat) ; https://openreview.net/forum?id=NHuyk9KsG6 (Horseshoe Splatting) ; https://arxiv.org/pdf/2503.04441 (EvidMTL) ; https://www.amazon.science/publications/modeling-uncertainty-in-3d-gaussian-splatting-through-continuous-semantic-splatting (Continuous Semantic Splatting) ; https://arxiv.org/html/2503.14665v1 (These Magic Moments) ; https://aaltoml.github.io/uncertainty-nerf-gs/ (Sources of Uncertainty)

**Low-confidence flags:** SplatHLoc 7Scenes medians (not extracted); HAL-NeRF v2 IEEE Access record (from B, unverified); CRLB paper's ICCV/IROS association (unverified); R3GS ICRA 2025 journal_ref (contradicted by DBLP — cite as preprint); LTGS "CVPR 2026" and Horseshoe Splatting ICLR 2026 acceptance (verify before camera-ready); GS-ReLoc IEEE Access 11039628 (bot-blocked, abstract-only); CurvLoc and GauLoc 7Scenes medians (retrievable but not extracted here); all entries marked * in the matrix inherit unverified status from list (B).
