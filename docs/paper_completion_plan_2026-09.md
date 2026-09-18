# USGLoc — Paper Completion Plan (updated 2026-09-18, evening)

## Do we need to train the EDL model? (asked 2026-09-18)

**Yes — and it is already trained.** `ckpts/edl_head.pt` exists (created by
`train_map.py`). The facts:

- **No pretrained EDL model exists to download.** The head is our own tiny
  2-layer MLP (512 → 256 → 25, ~550 KB) mapping CLIP features to Dirichlet
  evidence for OUR 25-class indoor vocabulary
  (`usgloc.mapping.INDOOR_VOCABULARY`). Nobody publishes that.
- **What stays pretrained and frozen:** FastSAM (masks) and CLIP ViT-B-16
  (embeddings). We never fine-tune them.
- **No manual labels needed.** Training data is CLIP zero-shot pseudo-labels
  (argmax against text prototypes, filtered by a top1–top2 margin) — fully
  automatic.
- **Cost: seconds.** 40 epochs over ~14k features on the 5070 Ti.
- **Trained once, shared everywhere** — the same head serves the map side and
  the query side (Sec. IV-B requires this so vacuities are comparable), and
  is reused across scenes. `train_map.py` loads `--head-ckpt` if it exists,
  else trains and saves it. To force a retrain: `rm ckpts/edl_head.pt`.
- **Two findings from today (keep for the paper's implementation section):**
  1. Classification accuracy converges in ~10 epochs, but the *uncertainty
     ordering* (ambiguous features → higher vacuity) only emerges at 40–60
     epochs. Default is now 40; `tests/test_mapping.py` pins the property.
  2. The head MUST train on **both input domains**: raw mask-crop features
     (query side) AND per-Gaussian mask-averaged features (map side).
     Mask-only training collapsed map evidence (median vacuity 0.997 despite
     95% coverage); dual-domain training
     (`mapping.build_head_training_set`) restored a full spread (chess:
     u_eff 0.001–1.0, 402/1916 primitives < 0.1).
- **[ ] The head-freeze decision is step 2.5b below and GATES 2.6 and 3.1** —
  every downstream artefact (maps, encoders, tuning) embeds the head, so
  retraining it later silently invalidates all of them.

## Phase 1 — Machine + data (DONE 2026-09-18)

- [x] **1.1 Disk**: 2 GB → 113 GB free (28 GB stale VNC log deleted, pip/ov
      caches purged, approved docker prune ~80 GB; texlive images and
      hesfm_dataset kept).
- [x] **1.2 Environment**: torch 2.11.0+cu128 + torchvision, gsplat (CUDA
      backend JIT-builds on sm_120), open_clip_torch, ultralytics, scipy;
      CUDA 12.8 toolchain at `~/.local/cuda-12.8` (export
      `CUDA_HOME=$HOME/.local/cuda-12.8` for gsplat rebuilds). Rebuild
      script: scratchpad `env_rebuild.sh` + `uags/scripts/setup_cuda_toolchain.sh`.
- [x] **1.3 Tests**: 61 original tests pass on the new stack; synthetic demo
      runs end-to-end (temperature scaling NLL 5216 → −12).
- [x] **1.4 Data**: all 7 scenes at `usgloc/data/7scenes_source/` (21 GB,
      26k train / 17k test frames) via `scripts/download_7scenes.sh`
      (tolerates the 7Scenes zips' benign unzip exit-1); Brachmann SfM
      pseudo-GT cloned at `usgloc/data/visloc_pseudo_gt_limitations/`.
- [x] **1.5 SfM pseudo-GT loader**: `SevenScenes(..., sfm_pose_dir=...)`
      drops unregistered frames (0 on all 7 scenes), exposes per-scene SfM
      focal (525.14–527.75) via `scene_camera()`; `--sfm-pose-dir` flag in
      run_7scenes.py. Validated: DSLAM-vs-SfM pose gap on the same chess
      frame = 0.95 cm > the SOTA ladder spacing — the flag is mandatory for
      Table I numbers. `quat_to_rotmat` added to se3.py. +5 tests.
- [x] **1.6 Front-end weights**: CLIP ViT-B-16 laion2b cached; FastSAM-s.pt
      in the usgloc/ repo root (scripts resolve it from there by default).

## Phase 2 — Training pipeline (BUILT 2026-09-18; final validation running)

New module `usgloc/mapping.py` + two CLIs; 73 tests pass.

- [x] **2.1 `scripts/train_map.py`** — per scene, four stages: depth
      back-projected init (voxel-deduplicated) → photometric 3DGS fit
      (gsplat `rasterization` + `DefaultStrategy` densification, plain RGB,
      L1, 7000 iters @ 320×240) → FastSAM+CLIP mask distillation onto
      Gaussians (depth-occlusion tested; observation counts kept) → shared
      EDL head + observation-scaled evidence (n=0 → vacuity exactly 1) →
      voxel×class instance ids → export for `load_gaussian_map`.
      Chess: 330k Gaussians, 1916 primitives, **94 s total**. Command:
      ```
      CUDA_HOME=$HOME/.local/cuda-12.8 python3 scripts/train_map.py \
        --root data/7scenes_source --scene chess \
        --sfm-pose-dir data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes \
        --out maps/chess.pt --head-ckpt ckpts/edl_head.pt
      ```
- [x] **2.2 EDL head** trained (see the EDL section above): `ckpts/edl_head.pt`.
- [~] **2.3 `scripts/train_encoders.py`** — real-data InfoNCE: query variants
      are FastSAM+CLIP+EDL encodings of frames at ±{0,4,8} around each
      submap anchor; reference side via `submap_reference_inputs`; saves
      `{descriptor, aggregator, edl_head}`. **Finding #3 (2026-09-18): submaps
      MUST be frustum-culled.** With radius-only submaps in a room smaller
      than the radius, every submap held 768–1529 of 1916 primitives, all
      reference descriptors collapsed (pairwise cosine 0.9996–1.0) and
      InfoNCE pinned at chance (ln 24 = 3.178) for 400 iters. `make_submaps`
      now takes `camera=` and keeps only primitives projecting into the
      anchor view → 154–340 primitives/submap, Jaccard 0.40 between
      neighbors. Retraining now. Command:
      ```
      CUDA_HOME=$HOME/.local/cuda-12.8 python3 scripts/train_encoders.py \
        --root data/7scenes_source --scene chess \
        --sfm-pose-dir data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes \
        --map maps/chess.pt --head-ckpt ckpts/edl_head.pt \
        --out ckpts/chess_encoders.pt
      ```
- [x] **2.4 gsplat renderer wiring**: `run_7scenes.py --renderer gsplat`.
- [x] **2.5 End-to-end smoke on chess — PASSED mechanically (2026-09-18
      evening).** All stages ran on 10 real test queries (one fix needed:
      cusolver rejects batched eigh at 64k+; GsplatRenderer now decomposes
      covariances once on CPU and caches). Untuned accuracy: median
      46.8 cm — but the follow-up diagnostics pinpointed why, giving 2.6 its
      priorities: (a) **refinement works and gsplat pose gradients are
      correct** — from perturbed GT it converges 5 cm → 0.85 cm/0.18°
      (PSNR 24 dB) at 320×240, but its convergence basin ends between 10
      and 20 cm; (b) **retrieval is good on validation views** (best-of-top-3
      anchor median 7.8 cm, 9/10 within 50 cm) but anchors at stride 20 sit
      8–30 cm away — at the basin's edge — and test viewpoints sometimes
      retrieve far anchors (the 1–2 m failures). Original command:
      ```
      CUDA_HOME=$HOME/.local/cuda-12.8 python3 scripts/run_7scenes.py \
        --root data/7scenes_source --scene chess \
        --sfm-pose-dir data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes \
        --map maps/chess.pt --encoders ckpts/chess_encoders.pt \
        --renderer gsplat --width 320 --height 240 --queries 10
      ```
      This also exercises the untested path: gradients through
      `gsplat.rasterization` w.r.t. the pose during refinement. If pose
      gradients are broken, fall back to `--renderer torch` (works, smaller
      resolution) and file the gsplat issue.
- [x] **2.5b HEAD FROZEN (2026-09-18 night) — the contingency path fired.**
      The chess-only head did NOT transfer: 8.9% pseudo-label agreement on
      fire and fire's map evidence collapsed (median u 0.994). New
      `scripts/train_head.py` pools BOTH domains across scenes (mask banks
      from all 7 scenes — banks need no maps — plus gaussian domains from
      existing maps; 40,731 feats). Pooled head: 70–86% agreement on every
      scene, vacuity ordering holds everywhere → frozen. Maps now also
      export `obs_count` and cache `<scene>_bank.pt`, so future head swaps
      never re-distill. chess/fire maps + chess encoders rebuilt with the
      frozen head. Finding #4 for the paper: like the dual-domain lesson,
      the shared head must also be trained across SCENES, not just domains.
- [ ] **2.6 Tune on chess — on a VALIDATION split, never the test split**
      (tuning ~8 knobs on test frames is test-set leakage; no Table I
      baseline did that and a reviewer will check). Carve validation from
      train-side frames disjoint from the map-fit/distillation frames (e.g.
      every 40th train frame offset by 5), localize those, and iterate:
      knobs prioritized by the 2.5 diagnostics: (1) **denser anchors**
      (`--anchor-stride` 5–10) so the best hypothesis starts INSIDE the
      ~10 cm refinement basin; (2) **coarse-to-fine refinement** (start
      160×120 for a wider basin, finish 320×240 or full res — the 0.85 cm
      floor at 320×240 suggests a final full-res polish to reach the
      0.55–0.7 cm ladder); (3) retrieval generalization to test views
      (more/wider encoder variants, longer InfoNCE); then `--top-k`,
      `--gamma`, `--submap-radius`, map-side `--instance-voxel` /
      `--distill-stride`. Freeze the best config as the paper defaults,
      write them into Sec. V-A (fills the hyperparameter TODOs), state the
      validation protocol there in one sentence, and only then touch the
      test split (once, in 3.2).

## Phase 3 — Experiment campaign (compute is cheap now: ~2 min/scene maps)

- [ ] **3.1 Build all 7 maps + encoders** (after 2.6 freezes settings):
      ```
      for s in chess fire heads office pumpkin redkitchen stairs; do
        python3 scripts/train_map.py --root data/7scenes_source --scene $s \
          --sfm-pose-dir ... --out maps/$s.pt --head-ckpt ckpts/edl_head.pt
        python3 scripts/train_encoders.py --root ... --scene $s \
          --map maps/$s.pt --out ckpts/${s}_encoders.pt
      done
      ```
      (~1–2 h wall-clock total, encoder FastSAM encoding dominates.)
      Precondition: head frozen (2.5b).
- [ ] **3.2 Full benchmark, 3 seeds** — with a MANDATORY code fix first.
      **The harness's current temperature protocol would produce invalid
      paper numbers**: it fits τ on the first `--temperature-split` (0.5)
      fraction of TEST queries and reports metrics only on the remainder —
      calibration fitted on test data (circular V-C claims) AND Table I
      accuracy on half the standard test split (incomparable to every
      baseline). Fix before running: add a train-side calibration path to
      run_7scenes.py (e.g. `--temp-calib-frames`: localize N train-side
      frames disjoint from map-fit/distill/validation frames, fit τ on their
      d², then evaluate the FULL test split with `--temperature-split 0`).
      Then run `--queries 0 --seed {0,1,2}` per scene → 21 runs; JSON in
      `results/`; aggregate median-of-runs into Table I's Ours row + the
      calibration table (raw vs tempered both reported).
- [ ] **3.3 Ablation grid** (one or two scenes, e.g. chess + stairs):
      supported today — Dempster vs Yager (`load_gaussian_map` fuse_rule;
      add a `--fuse-rule` flag to run_7scenes.py, 3 lines), K=1 vs 3
      (`--top-k`), no pixel weighting (`--gamma 0`), raw vs tempered (both
      already reported). **Code to write:** (a) softmax-head variant (replace
      EDL head with softmax + entropy-derived pseudo-vacuity — needed for the
      central calibration comparison); (b) Bingham rotation posterior OR
      soften the Sec. IV-D promise; (c) no-gate variant (λ_u = 0) via
      PipelineConfig.
- [ ] **3.4 Robustness sweeps** (Sec. V's three axes, currently
      evidence-free). All three map to existing train_map.py flags:
      map budget (`--fit-stride 10 vs 40 vs 80` ≈ 600/150/75 images),
      input sparsity (`--distill-stride 20 vs 80 vs 200`), degraded
      semantics (`--label-margin` up, or add mask-dropout). Plot error AND
      coverage/NLL vs degradation → new robustness figure; the claim to
      verify: accuracy degrades but calibration holds (covariance grows
      honestly).
- [ ] **3.5 Runtime table**: add `torch.cuda.Event` timing around
      retrieval / refinement / Laplace+condensation in `pipeline.localize`
      (small patch), report per-stage ms on the 5070 Ti; context: UGS-Loc
      ≈ 2.2 s/query. Update the draft's hardware sentence (see 0.5).
- [ ] **3.6 Rebuttal-proofing extras** (cheap now, impossible in the 30-day
      R&R): per-scene coverage table (falls out of 3.2), bootstrap CIs on
      coverage/AUSE/NLL (small metrics.py addition), fixed-global-scaled
      covariance baseline row (exists in the synthetic demo — port to the
      harness) showing the covariance STRUCTURE is learned, not just the
      scale via τ.
- [ ] **3.7 RESULTS GATE — decide the framing before writing Phase 4.**
      Compare the median-of-seeds 7-scene average against the ladder
      (SplatHLoc 0.55 / UGS-Loc 0.64 / ULF-Loc 0.70 / STDLoc 0.76 cm; watch
      stairs, the classic failure scene). Competitive → write V-B as
      planned. Not competitive → either loop back to 2.6 on the failing
      scene, or lead the abstract/V-B with the calibrated-distribution
      novelty (the actual first-of-kind claim) and treat accuracy as
      secondary. Decide HERE, not while drafting.

## Phase 4 — Figures + writing (overlaps Phase 3)

- [ ] **4.0 Port to `ieeeconf` FIRST and measure pages.** RA-L initial
      submissions use the conference class, and the 6-page limit INCLUDES
      references — porting after writing risks a rewrite cycle. Port, compile
      (Overleaf, or re-download tectonic), measure, and take the
      6-vs-paid-8-page decision now so 4.1–4.7 are written into the real
      budget. (5.1 then keeps only the `\ifanonymous` toggle work.)
- [ ] **4.1 Fig. 3 (calibration)**: regenerate from real records via
      `python3 paper/figures/fig3_calibration_template.py` fed by
      `results/7scenes_*.json`; strip the red caption note.
- [ ] **4.2 New Fig. 4 (vacuity heatmaps)**: query RGB | rendered vacuity |
      per-pixel weight, 2–3 queries. Add a `--save-visuals` flag to
      run_7scenes.py dumping `U_r` maps during refinement. The core
      mechanism currently has zero visual evidence in the paper.
- [ ] **4.3 Robustness figure** from 3.4. Optional if pages allow: pose-error
      CDFs, refinement-convergence plot, GMM-hypotheses view (else keep for
      rebuttal).
- [ ] **4.4 Fill all 13 draft TODOs**: abstract headline (line ~75),
      contribution 4 (~96), implementation hyperparameters (~250, from 2.6),
      V-B accuracy discussion (~253) — the SGLoc-protocol "supplementary"
      the letter references NOW EXISTS: `paper/usgloc_full.tex` Sec. VIII
      (point the sentence at the extended report / arXiv version), V-C
      calibration claims (~282), robustness discussion (~310), ablation
      summary (~313), runtime (~337), conclusion (~341). Delete the optional
      MAV TODO (~244) per 0.1.
- [ ] **4.8 Keep `paper/usgloc_full.tex` (Extended Report) in sync** — the
      4-page companion created 2026-09-18: EDL analysis with today's real
      measured numbers (findings 1–3 as tables), full hyperparameter table,
      map statistics, protocol details; compiles clean with the scratchpad
      tectonic. As Phase 3 results land, fill its `\res`/TODO tables
      (per-scene map stats, per-scene calibration, full ablations,
      robustness, runtime, SGLoc DSLAM comparison, converged InfoNCE) —
      it doubles as the arXiv extended version and the reviewer supplement.
- [ ] **4.5 Add a Limitations paragraph** (none exists): Dempster/Yager
      conflict behavior, local-unimodal Laplace vs the GMM argument,
      FastSAM/CLIP dependence, per-scene training, indoor-only evaluation.
- [ ] **4.6 Table I surgery**: add **ULF-Loc row** (CVPR 2026 highlight,
      arXiv:2605.04730, 7Scenes 0.70 cm/0.20° — beats STDLoc/GS-CPR rungs;
      also one Related-Work sentence + bib entry); print two decimals for
      UGS-Loc (0.64) and SplatHLoc (0.55) avgs so table matches text; add
      the GT-protocol footnote (0.2); fill the Ours row from 3.2; remove all
      red caption notes and the `\res`/`\TODO` macros (final check counts
      OCCURRENCES, not lines:
      `grep -o 'TODO{\|\\res{\|textcolor{red' usgloc_ral_draft.tex | wc -l` → 0).
- [ ] **4.7 Author block fixes**: Xiem HoangVan currently duplicates Thanh's
      ORCID; Thanh's own `\orcidlink` holds a full URL; `\thanks` lists
      non-authors (pham_n_quang, van-ho) and "tongvan" for Bang Nguyen Van;
      `\markboth` still says "Month, Year".

## Phase 5 — Submission package (RA-L rules verified Sept 2026)

- [ ] **5.1 `\ifanonymous` toggle** (the ieeeconf port itself happened in
      4.0): anonymized author block + `\thanks` (including the JST grant)
      stripped for submission, restored at camera-ready, from one source.
- [ ] **5.2 Final page check** under ieeeconf: 6 pages including references,
      or the paid-page decision from 4.0 confirmed.
- [ ] **5.3 Reproducibility**: one sentence in Sec. V-A committing to code
      release (the draft criticizes SGLoc for unreleased code); anonymized
      mirror (anonymous.4open.science) with per-table regeneration commands;
      exclude raw 7Scenes frames (Microsoft research-only license — ship the
      download script instead); credit the pseudo-GT license.
- [ ] **5.4 Video**: one ≤50 MB zip (video + ASCII ReadMe.txt) — localization
      runs + vacuity visualization; submitted with the paper.
- [ ] **5.5 Chu & Kim PDF** via JAIST library (IEEE doc 11305116, no public
      preprint) — verifies the Sec. II-B failure-case claim even with the
      semi-dynamic protocol scoped out.
- [ ] **5.6 Mechanics**: PaperCept PINs for all 4 authors
      (https://ras.papercept.net/journals/ral), early PDF margin-compliance
      test, 2–5 RA-L keywords, co-author sign-off (0.4), submit. Keep the
      experiment stack runnable for the 30-day no-extension R&R (~3 months
      out).
- [ ] **5.7 Full double-anonymous checklist** (a desk reject on anonymity
      wastes the whole cycle; the author block alone is not enough):
      self-citations in third person; grants/acknowledgments behind the
      `\ifanonymous` toggle; PDF metadata clean (`pdfinfo`/`exiftool` show
      no Author, no JAIST paths); the video zip + ReadMe.txt reviewed for
      names, machine paths, lab shots, voiceover; the submitted PDF links
      ONLY the anonymous.4open.science mirror, never the named repo.

---

## Timeline (revised — the GPU changed the math)

Maps+encoders for all 7 scenes are ~1–2 h of compute, not the week the audit
assumed. The critical path is now **tuning quality on chess (2.5–2.6)**, then:
3.1–3.2 (1 day) → 3.3–3.6 (2–3 days incl. small code) → Phase 4 writing
(2–3 days) → Phase 5 packaging (2 days). **≈ 1.5–2 weeks to submission** if
chess tunes well; the open risk is accuracy tuning (2.6), not infrastructure.
Conference pairing: ICRA 2027 transfer needs acceptance by Dec 31, 2026
(unrealistic); **IROS 2027** window (acceptance by Apr 30, 2027) is the target.

## Artefacts on disk (2026-09-18)

| Path | What |
|---|---|
| `maps/chess.pt` | evidential chess map (330k gaussians, 735 MB) |
| `ckpts/edl_head.pt` | shared EDL head (trained, dual-domain) |
| `ckpts/chess_encoders.pt` | retrieval encoders — appears when train_encoders.py finishes; 2.5's command fails on `--encoders` until then |
| `data/7scenes_source/` | all 7 scenes (21 GB) |
| `data/visloc_pseudo_gt_limitations/` | SfM pseudo-GT poses |
| `FastSAM-s.pt` (repo root) | FastSAM weights |
| `results/` | run JSONs (currently demo only) |
| `paper/usgloc_full.tex` | Extended Report (EDL analysis + extended tables; compiles, 4 pp) |
| scratchpad `tectonic` | re-downloaded 2026-09-18; compiles both papers locally |
