# USGLoc Revision — Merged Gap-Fill Research Report (compiled 2026-06-11)

## 0. Critical protocol decision before any table is filled: 7Scenes ground-truth convention

7Scenes has two incompatible GT conventions: original DSLAM/KinectFusion poses vs SfM pseudo-GT (Brachmann et al., ICCV 2021). The same method differs wildly between them (ACE: 80.8% @5cm/5° DSLAM vs 97.1% SfM, https://ar5iv.labs.arxiv.org/html/2305.14059; DSAC* Chess: 1.9 cm/1.11° DSLAM vs 0.5 cm/0.17° SfM). The entire modern 3DGS-refinement line (GS-CPR, STDLoc, UGS-Loc, SplatHLoc, LoGS, RAP) reports **SfM pseudo-GT**; SGLoc (IROS 2025) reports DSLAM-consistent baselines; Marepo/DSAC*/PoseNet originals are DSLAM.

**Recommendation for USGLoc Table I:** evaluate and report under **SfM pseudo-GT** (matches all direct 3DGS competitors and 3DGS map quality), state the convention explicitly in the table caption, and footnote any DSLAM-sourced row. Mixing the columns silently is the most common citation error in this space.

---

## 1. Pre-filled baseline number block — Table I (7Scenes, median translation cm / rotation deg)

Scene order: Chess | Fire | Heads | Office | Pumpkin | RedKitchen | Stairs | **Avg**.

### Primary block — SfM pseudo-GT (recommended protocol)

| Method | Chess | Fire | Heads | Office | Pumpkin | RedKitchen | Stairs | Avg | Source |
|---|---|---|---|---|---|---|---|---|---|
| PoseNet (PyTorch reimpl., SfM re-eval)¹ | 10/4.02 | 27/10.0 | 18/13.0 | 17/5.97 | 19/4.67 | 22/5.91 | 35/10.5 | 21/7.74 | NeFeS Tab.2 https://arxiv.org/html/2303.10087v2 ; GS-CPR Tab.1 https://arxiv.org/html/2408.11085v4 |
| Marepo (SfM-retrained by GS-CPR)² | 1.9/0.83 | 2.3/0.92 | 2.1/1.24 | 2.9/0.93 | 2.5/0.88 | 2.9/0.98 | 5.9/1.48 | 2.9/1.04 | GS-CPR Tab.1 https://arxiv.org/html/2408.11085v4 |
| ACE³ | 0.5/0.18 | 0.8/0.33 | 0.5/0.33 | 1.0/0.29 | 1.0/0.22 | 0.8/0.20 | 2.9/0.81 | 1.1/0.34 | GS-CPR Tab.1 https://arxiv.org/html/2408.11085v4 |
| DSAC*³ | 0.5/0.17 | 0.8/0.28 | 0.5/0.34 | 1.2/0.34 | 1.2/0.28 | 0.7/0.21 | 2.7/0.78 | 1.1/0.34 | GS-CPR Tab.1; corroborated NeRFMatch Tab.2 https://arxiv.org/html/2403.09577v2 |
| iComMa | — | — | — | — | — | — | — | — | **GAP: no 7Scenes results in any arXiv version** (relative-pose AUC on NeRF-Synthetic/LLFF/Mip-NeRF360/Deep Blending only) https://arxiv.org/html/2312.09031v2 |
| SplatLoc (TVCG'25) | — | — | — | — | — | — | — | — | **GAP: no 7Scenes results; Replica + 12Scenes only** https://arxiv.org/html/2409.14067v1 |
| ACE + GS-CPR (ICLR'25) | 0.5/0.15 | 0.6/0.25 | 0.4/0.28 | 0.9/0.26 | 1.0/0.23 | 0.7/0.17 | 1.4/0.42 | 0.8/0.25 | GS-CPR Tab.1 https://arxiv.org/html/2408.11085v4 |
| STDLoc (CVPR'25)⁴ | 0.46/0.15 | 0.57/0.24 | 0.45/0.26 | 0.86/0.24 | 0.93/0.21 | 0.63/0.19 | 1.42/0.41 | 0.76/0.24 | identical rows in UGS-Loc Tab.1 https://arxiv.org/html/2603.16538v1 and SplatHLoc Tab.1 https://arxiv.org/html/2603.29185v1 |
| SGLoc (IROS'25)⁵ | — | — | — | — | — | — | — | — | **cannot be placed in SfM block; see DSLAM block** |
| ACE + UGS-Loc (CVPR'26) | 0.37/0.12 | 0.47/0.20 | 0.36/0.25 | 0.77/0.22 | 0.79/0.18 | 0.58/0.15 | 1.11/0.33 | 0.64/0.21 | https://arxiv.org/html/2603.16538v1 (Tab.1) |
| SplatHLoc (CVPR'26) | 0.28/0.09 | 0.44/0.19 | 0.29/0.18 | 0.62/0.18 | 0.70/0.15 | 0.49/0.12 | 1.03/0.30 | 0.55/0.17 | https://arxiv.org/html/2603.29185v1 (Tab.1) |

Optional extra SfM rows (same tables): GLACE 1.2/0.36 avg (GS-CPR Tab.1); MCLoc 4.1/1.43; DFNet+NeFeS50 2.4/0.79; HR-APR 2.4/0.85; NeRFMatch 2.8/0.7; LoGS 0.76/0.24; RAP_ref 0.61/0.20 (SplatHLoc Tab.1). Recall rows: ACE+GS-CPR 100% @5cm/5° / 93.1% @2cm/2°; ACE+UGS-Loc 100 / 95.6; SplatHLoc RR@[2cm,2°] on 12Scenes 97.3%.

Footnotes:
1. NOT the ICCV 2015 PoseNet numbers; original Kendall et al. (DSLAM GT, meters) avg ~0.44 m/10.4° (https://ar5iv.labs.arxiv.org/html/1505.07427, Fig. 5 table).
2. NOT from the Marepo paper. The original Marepo (CVPR 2024) row is DSLAM GT: 2.6/1.35 | 2.5/1.42 | 2.3/2.21 | 3.6/1.44 | 4.2/1.55 | 5.1/1.99 | 6.7/1.83 | avg 3.9/1.68 (https://arxiv.org/html/2404.09884v1, Tab.1). GS-CPR footnote 1 confirms they retrained Marepo's ACE head on SfM GT.
3. Neither the ACE nor the DSAC* paper publishes per-scene 7Scenes medians; ACE/DSAC* per-scene medians are secondary (SfM: GS-CPR/NeRFMatch; DSLAM: Marepo Tab.1 — ACE avg 2.8/0.93, DSAC* avg 2.7/1.41, https://arxiv.org/html/2404.09884v1).
4. STDLoc primary paper covered by prior survey; the per-scene rows above are reprinted identically by two independent CVPR 2026 papers, both stating SfM pGT (GS-CPR protocol).
5. **SGLoc protocol incompatibility:** evaluates only 4 of 7 scenes (Chess, Heads, Office, Redkitchen) on DSLAM-consistent GT (its DSAC*/ACE+GSLoc baseline rows match DSLAM tables): 0.14/0.05 | 0.14/0.06 | 0.43/0.22 | 1.3/0.26 | avg 0.5/0.15 (https://arxiv.org/html/2507.12027v1, Tab. II). Putting SGLoc in the SfM table or comparing its avg against 7-scene averages is invalid; footnote it as "4/7 scenes, DSLAM GT, code unreleased (stub repo https://github.com/IRMVLab/SGLoc)".

### Secondary block — DSLAM GT (only if a DSLAM table is kept)

| Method | Per-scene (Chess…Stairs) | Avg | Source |
|---|---|---|---|
| PoseNet (orig., m) | 0.32/8.12, 0.47/14.4, 0.29/12.0, 0.48/7.68, 0.47/8.42, 0.59/8.64, 0.47/13.8 | ~0.44 m/10.4° | https://ar5iv.labs.arxiv.org/html/1505.07427 |
| MS-Transformer (orig., m) | 0.11/4.66, 0.24/9.6, 0.14/12.19, 0.17/5.66, 0.18/4.44, 0.17/5.94, 0.26/8.45 | 0.18/7.28 | https://ar5iv.labs.arxiv.org/html/2103.11468 |
| DSAC* | 1.9/1.11, 1.9/1.24, 1.1/1.82, 2.6/1.18, 4.2/1.41, 3.0/1.70, 4.2/1.42 | 2.7/1.41 | Marepo Tab.1 https://arxiv.org/html/2404.09884v1 |
| ACE | 1.9/0.7, 1.9/0.9, 0.9/0.6, 2.7/0.8, 4.2/1.1, 4.2/1.3, 3.9/1.1 | 2.8/0.93 | Marepo Tab.1 |
| Marepo | 2.6/1.35, 2.5/1.42, 2.3/2.21, 3.6/1.44, 4.2/1.55, 5.1/1.99, 6.7/1.83 | 3.9/1.68 | https://arxiv.org/html/2404.09884v1 |
| SGLoc (4 scenes) | Chess 0.14/0.05, Heads 0.14/0.06, Office 0.43/0.22, Redkitchen 1.3/0.26 | 0.5/0.15 | https://arxiv.org/html/2507.12027v1 |

### 12Scenes block (median cm/°, for Table II if used)

- ACE+GS-CPR 0.5/0.21 (100% @5cm/5°, 98.7% @2cm/2°); Marepo+GS-CPR 0.7/0.28 (98.9/90.9); DSAC* 0.5/0.25 (99.8/96.7); ACE 0.7/0.26 (100/97.2); GLACE 0.7/0.25 (100/97.5); Marepo 2.1/1.04 (95/50.4) — all SfM pGT, averages only (https://arxiv.org/html/2408.11085v4, Tab.4).
- ACE+UGS-Loc 0.4/0.18, 99.1% @[2cm,2°] (averages only, no per-scene table) — https://arxiv.org/html/2603.16538v1 (Tab.3).
- SplatHLoc 0.3/0.14 avg, RR@[2cm,2°] 97.3%; per-scene in suppl. Tab. V — https://arxiv.org/html/2603.29185v1.
- SplatLoc per-scene (apt1/kitchen … office2/5b): 0.8/0.4, 1.1/0.4, 1.2/0.5, 1.0/0.5, 1.2/0.5, 1.5/0.6, 1.1/0.5, 1.2/0.5, 1.6/0.5, 1.1/0.5, 1.4/0.6, 1.5/0.5 — dataset-native RGB-D GT, NOT SfM pGT (https://arxiv.org/html/2409.14067v1, Tab.2).
- SGLoc: only 3 of 12 scenes (apt2/bed 0.5/0.4, apt2/kitchen 0.1/0.1, office1/lounge 0.3/0.1, avg 0.3/0.2) — https://arxiv.org/html/2507.12027v1, Tab. I. Footnote required.

---

## 2. SGLoc + Chu&Kim protocol summary (for our semi-dynamic experimental design)

**SGLoc (Xu, Zhu, Wang, SJTU; IROS 2025, arXiv:2507.12027).** Prior-free two-stage localization against a Gaussian-Grouping semantic 3DGS map: (1) "SGLoc1" coarse stage — CLIP 2D + PointNet++ 3D semantic descriptors retrieve k=5 cubic submaps sampled along training trajectories (similarity threshold τ=0.1); (2) "SGLoc2" — iterative render-and-compare refinement minimizing λ·L_match + (1−λ)·L_pixel with ALIKED-style keypoint matching. No uncertainty output of any kind. Protocol quirks to respect when comparing: only 4/7 7Scenes and 3/12 12Scenes scenes; DSLAM-consistent GT; ablations show the coarse stage alone is at 3.97 cm/2.38° (7Scenes avg) and that swapping in ACE/GLACE/Marepo initializations breaks their refinement entirely (avg errors 1.7–3 m, Tabs. III–IV), suggesting refinement is tightly coupled to their own retrieval. Code is a README-only stub (https://github.com/IRMVLab/SGLoc), so their 0.14 cm Chess number (3.5x better than DSAC*) is unverifiable; printed typos exist (ACE avg "0.7/0.5"; "SpaltLoc"). Sources: https://arxiv.org/abs/2507.12027, https://arxiv.org/html/2507.12027v1.

**Chu & Kim (RA-L vol. 11 no. 3, pp. 3852–3859, March 2026, DOI 10.1109/LRA.2025.3643336; IEEE article 11305116).** "Semantic-Augmented 3D Gaussian Splatting for Visual Localization in Complex Indoor Environments", Intelligent Robotics Lab, Chungbuk National Univ. Pipeline (from abstract + public figure/footnote metadata): OpenGaussian semantic 3DGS map; an LLM (refs include LLaVA, Phi-3, GPT-4V) labels objects as robust vs movable to build a "robust map" of scene-change-insensitive objects (Fig. 13 documents an LLM robust-object failure case); coarse-to-fine matching generates candidate viewpoints over the Gaussian map, renders, selects and refines the best pose (components: CLIP, FastSAM, ALIKED+LightGlue, P3P). **Semi-dynamic protocol we adopt:** three evaluation sets — 7Scenes (Tabs. 1–2: median error / accuracy), ScanNet (Tabs. 3–4), and a custom scene-change dataset with rearranged objects (Figs. 8–10; Tabs. 5–6), plus an ablation (Tab. 7) and runtime (Tab. 8, Fig. 11); GS-based baselines are stress-tested under both high-quality and low-quality GS map inputs (table footnotes "(*)¹/(*)²"). Baselines: Marepo, ACE, PNeRFLoc, CrossFire, GS-CPR, STDLoc, iComMa, SplatLoc. No uncertainty modeling anywhere. **Full text is paywalled with no mirror as of 2026-06-11** — per-scene numbers and the exact rearranged-object counts require one IEEE Xplore institutional download (https://ieeexplore.ieee.org/document/11305116). 7Scenes GT convention unconfirmed (cites the original Glocker et al. 2013 release, suggesting DSLAM).

**Implication for USGLoc:** if we adopt their semi-dynamic protocol, we must (a) obtain the PDF for their Tables 5–6 numbers, (b) replicate the high-/low-quality GS-map split, and (c) note that their robustness mechanism is discrete LLM-based object gating, whereas ours is continuous evidential vacuity — their documented LLM failure case (Fig. 13) is a direct motivation we can cite.

---

## 3. New related-work sentences (drop-in drafts, each claim sourced)

**SCR/APR (feed-forward wave).** "Scene coordinate regression remains the accuracy reference for scene-specific relocalization (DSAC* [arXiv:2002.12324], ACE [arXiv:2305.14059], GLACE [arXiv:2406.04340]), and has recently been extended to self-supervised mapping (ACE0 [arXiv:2404.14351]), large-scale robustness (R-SCoRe [arXiv:2501.01421]), and scene-agnostic query pre-training with per-pixel coordinate uncertainty used to prefilter correspondences before PnP-RANSAC (ACE-G [arXiv:2510.11605]). In parallel, feed-forward pointmap and relative-pose models (DUSt3R [arXiv:2312.14132], MASt3R [arXiv:2406.09756], Reloc3r [arXiv:2412.08376], FastForward [arXiv:2510.00978], Reloc-VGGT [arXiv:2512.21883]) localize in unseen scenes using internal per-correspondence confidences; none of these methods, however, outputs a calibrated pose posterior — uncertainty, where present (ACE-G, FastForward, DUSt3R/MASt3R confidences, or the variational APR likelihood of Zangeneh et al. [arXiv:2504.07260]), is consumed internally for filtering rather than exposed as a distribution over SE(3)."

**Semantic localization.** "Semantic cues have been exploited for relocalization at the object level (GOReloc, RA-L 2024 [arXiv:2408.07917]; CLIP-Loc, ICRA 2024 [arXiv:2402.06092]), via scene graphs (SceneGraphLoc, ECCV 2024 [arXiv:2404.00469]), semantic ray regression (DIMM, ICCV 2025 [CVF]), and floorplan or panoramic semantic matching (SemRayLoc [arXiv:2507.09291]; SPVLoc, ECCV 2024 [arXiv:2404.10527]). Explicit pose uncertainty in this line appears only in Monte-Carlo formulations whose semantic likelihoods weight a particle posterior (OMCL [arXiv:2512.15557]; ShelfAware, RA-L 2026 [arXiv:2512.09065]; VLM-GLoc [arXiv:2605.30506]) or in Gaussian-NLL position confidences from VLMs (Loc3R-VLM [arXiv:2603.18002]); none provides a calibrated SE(3) posterior, and none models semantic evidence evidentially."

**Evidential / uncertainty-aware semantic mapping.** "Evidential deep learning has entered semantic mapping through Dirichlet-based segmentation fused into voxel maps (EvSemMap, IROS 2024 [arXiv:2403.14138]) and multi-task evidential depth+semantics for TSDF mapping (EvidMTL/EvidKimera, IROS 2025 [arXiv:2503.04441]); on Gaussian primitives, Continuous Semantic Splatting maintains per-Gaussian Dirichlet posteriors with closed-form variance (ICRA 2025 [arXiv:2411.02547]), PhysGS performs Bayesian per-splat belief updates over physical properties (CVPR 2026 [arXiv:2511.18570]), and E2-BKI — the closest prior work to ours — fuses EDL belief masses into ellipsoidal primitives via iterated Dempster combination with plain (1−η) conflict normalization (RA-L [arXiv:2509.11964]). None of these maps is used for camera localization, none calibrates a downstream pose posterior, and only learned-discounting [MDPI Electronics 14(23):4602] and reduced-combination [arXiv:2405.06265] variants address Dempster's conflict (Zadeh) pathology that our discounted fusion targets; no prior work maintains directional (vMF) semantic posteriors in a map (nearest: Gaussian-latent LatentBKI, RA-L 2025 [arXiv:2410.11783])."

---

## 4. UGS-Loc differentiation paragraph (draft)

"UGS-Loc (CVPR 2026) [arXiv:2603.16538] is the most closely related uncertainty-aware 3DGS localizer: it refines an APR/SCR pose prior on a Scaffold-GS map by spawning Monte-Carlo pose particles whose importance weights combine matcher confidence with (1 − U), where U is a rendered per-pixel geometric uncertainty obtained from a diagonal-Laplace Fisher-information approximation over anchor parameters, and it biases PnP-RANSAC sampling toward low-uncertainty correspondences (s_i = exp(−β·Ū(r_i)) + ε). Its uncertainty is therefore post-hoc, frequentist, and purely geometric — derived from the photometric training objective of a fixed map — whereas USGLoc's vacuity arises from evidential (EDL + Dempster-Shafer) semantic fusion at map-construction time and quantifies what the map does not know, not merely where rendering gradients are flat. Moreover, UGS-Loc consumes uncertainty only internally and outputs a single pose (the argmax-weight particle or an SE(3) weighted mean), with no covariance, no calibration analysis, and no semantics; USGLoc instead exposes a calibrated Gaussian-mixture posterior on SE(3) whose quality is itself evaluated. Empirically UGS-Loc reaches 0.64 cm/0.21° on 7Scenes (SfM pGT) at roughly 2.2 s/query (8 particles, 2 iterations, MASt3R matching) [arXiv:2603.16538], which positions it as our strongest non-evidential uncertainty-weighted baseline."

---

## 5. Corrections and contradictions relative to the earlier survey

1. **"Uncertainty-gated 2D-3D matching" is no longer unique as a mechanism class.** ACE-G (ICCV 2025) prefilters correspondences by per-pixel coordinate uncertainty quantile before PnP (https://arxiv.org/abs/2510.11605), and UGS-Loc gates RANSAC sampling by rendered Fisher uncertainty (https://arxiv.org/html/2603.16538v1). USGLoc's claim must be narrowed to *evidential vacuity-gated* matching with a *calibrated posterior output* — that combination remains unclaimed by anyone found.
2. **"First evidential semantic Gaussian map" would be falsified by E2-BKI** (RA-L, arXiv:2509.11964): it already fuses EDL Dirichlet evidence into anisotropic Gaussian/ellipsoidal primitives via DST. Safe claim: first evidential semantic **3DGS rendering** map used for **localization**, with conflict-discounted fusion (E2-BKI uses un-discounted (1−η) Dempster normalization and never localizes).
3. **CSS venue verified:** "Continuous Semantic Splatting" (arXiv:2411.02547) is ICRA 2025 per the Amazon Science publication page (https://www.amazon.science/publications/modeling-uncertainty-in-3d-gaussian-splatting-through-continuous-semantic-splatting); arXiv lists no venue. No code released. PhysGS is CVPR 2026 per project page (https://samchopra2003.github.io/physgs). EvidMTL/EvidKimera is IROS 2025, DOI 10.1109/IROS60139.2025.11247528, code at https://github.com/HumanoidsBonn/evidential_mapping.
4. **If the survey tabulated iComMa or 6DGS on 7Scenes, those numbers were third-party re-runs.** Neither primary source contains any 7Scenes (or cm/°) results (https://arxiv.org/html/2312.09031v2; https://ar5iv.labs.arxiv.org/html/2407.15484). iComMa also still has no peer-reviewed venue as of June 2026.
5. **SplatLoc has no 7Scenes results** — Replica + 12Scenes only, on dataset-native (not SfM) GT (https://arxiv.org/html/2409.14067v1). Any "SplatLoc 7Scenes" row in a survey table is mis-sourced (SGLoc's Table I cites it on 12Scenes, misspelled "SpaltLoc").
6. **Marepo numbers commonly mis-attributed:** the 1.9/0.83 (Chess) … 2.9/1.04 avg row is GS-CPR's SfM-retrained Marepo (https://arxiv.org/html/2408.11085v4, footnote 1), not the CVPR 2024 paper's DSLAM numbers (3.9/1.68 avg). Likewise per-scene "DSAC*/ACE medians" are secondary (Marepo Tab.1 for DSLAM, GS-CPR/NeRFMatch for SfM) — the primary papers publish only accuracy percentages.
7. **SOTA ladder update (SfM pGT 7Scenes):** SplatHLoc 0.55/0.17 < RAP_ref 0.61/0.20 < UGS-Loc 0.64/0.21 < LoGS/STDLoc 0.76/0.24 < ACE+GS-CPR 0.78/0.25. SplatHLoc and UGS-Loc are both **accepted at CVPR 2026** (https://arxiv.org/abs/2603.29185, https://github.com/HqiTao/SplatHLoc; https://arxiv.org/abs/2603.16538, https://kmk97.github.io/UGSLoc/) — if the survey listed them as preprints, update venue. USGLoc's accuracy must be contextualized against SplatHLoc, not GS-CPR.
8. **SGLoc's headline (0.5 cm/0.15° "prior-free") is on an easier protocol** (4/7 scenes, DSLAM GT, omits Fire/Pumpkin/Stairs — the three hardest), with unreleased code and printed typos; do not treat it as a 7-scene SOTA datapoint.
9. **ObjLoc was withdrawn from ICLR 2026** (https://openreview.net/forum?id=KrhLCdXmh9) — citable only as a preprint, if at all.
10. **MASt3R won the ECCV 2024 Map-free Visual Relocalization challenge** (https://europe.naverlabs.com/updates/mast3r-wins-map-free-visual-relocalization-halleng/) — strengthen any "feed-forward matching is a serious relocalization baseline" framing; FastForward (ICLR 2026, arXiv:2510.00978) reaches 73.6% @5cm/5° on 7Scenes with zero scene-specific training.
11. **Novelty gaps confirmed, not contradicted:** (a) no 2024–2026 work maintains vMF/directional posteriors over semantic embeddings in a map (nearest: LatentBKI's Gaussian latents, arXiv:2410.11783; vMF-Contact is grasping-only, arXiv:2411.03591); (b) no SCR/APR/3DGS/semantic localizer outputs a calibrated SE(3) pose posterior — particle filters (OMCL/ShelfAware/VLM-GLoc) and the variational APR work (arXiv:2504.07260) are uncalibrated; (c) conformal map-level sets exist (arXiv:2509.25124) but complement rather than overlap pose-level calibration.
12. **Chu & Kim full text remains paywalled** (no arXiv/RG/archive mirror, lab page not yet listing it) — the semi-dynamic per-scene numbers for our comparison table require institutional access to https://ieeexplore.ieee.org/document/11305116 before the camera-ready.
