#!/usr/bin/env python3
"""USGLoc real-data harness for 7Scenes (Sec. V of the RA-L draft).

This is the benchmark counterpart of ``scripts/run_demo_synthetic.py``: same
pipeline, same metric table (median errors, SR@5cm/5deg, coverage @
68.3/95.4/99.7, ECE, AUSE, NLL), but driven by the standard 7Scenes release
(the directory conventions also cover identically organized 12Scenes mirrors).

The harness does NOT train anything. You supply:

* ``--root``      the 7Scenes root (root/<scene>/TrainSplit.txt, seq-XX/...).
* ``--map``       an evidential 3DGS map exported by your open-vocabulary
                  3DGS training pipeline (e.g. Gaussian Grouping + an EDL
                  head) as a torch.save'd tensor dict -- see
                  ``usgloc.datasets.load_gaussian_map`` for the exact format.
* ``--encoders``  a torch.save'd checkpoint with the trained retrieval
                  encoder state_dicts: {"descriptor": ..., "aggregator": ...,
                  optionally "edl_head": ...}.  Without it the encoders are
                  randomly initialized and retrieval is RANDOM -- results are
                  then meaningless except as a smoke test.

Query instances come from the FastSAM+CLIP+EDL front end
(``usgloc.query.RealQueryEncoder``; pip install usgloc[realdata]).

Submaps are anchored on the TRAIN split trajectory (every ``--anchor-stride``
frames); test frames are localized at ``--width x --height`` (the pure-PyTorch
renderer is O(N_gaussians x N_pixels) -- keep it small, or wire in the gsplat
backend).  The covariance temperature (draft Sec. IV-D) is fitted on the first
``--temperature-split`` fraction of queries via ``usgloc.calibration.
fit_temperature`` when that module is available, and falls back to tau=1
otherwise.

Usage:
    python3 scripts/run_7scenes.py --root /data/7scenes --scene chess \\
        --map maps/chess_evidential.pt --encoders ckpts/chess_encoders.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc import se3
from usgloc.datasets import SevenScenes, TwelveScenes, load_gaussian_map
from usgloc.posterior import _project_pd
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator
from usgloc.metrics import (
    ause,
    coverage,
    coverage_ece,
    mahalanobis_sq,
    pose_errors,
    success_rate,
)
from usgloc.pipeline import PipelineConfig, USGLocPipeline
from usgloc.primitives import make_submaps
from usgloc.rendering import TorchSplatRenderer

# usgloc.calibration is an optional sibling module; code only against
# fit_temperature(d2, dof=6, method='median') -> float and degrade gracefully.
try:
    from usgloc.calibration import fit_temperature
except ImportError:
    fit_temperature = None


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def move_map(emap, device):
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        v = getattr(emap, f)
        if v is not None:
            setattr(emap, f, v.to(device))
    for g in emap.primitives:
        for f in dataclasses.fields(g):
            setattr(g, f.name, getattr(g, f.name).to(device))
    for sm in emap.submaps:
        sm.anchor_pose = sm.anchor_pose.to(device)
        if sm.descriptor is not None:
            sm.descriptor = sm.descriptor.to(device)
    return emap


def move_instances(instances, device):
    return [
        dataclasses.replace(
            q,
            clip_feat=q.clip_feat.to(device), centroid=q.centroid.to(device),
            area=q.area.to(device), belief=q.belief.to(device),
            vacuity=q.vacuity.to(device),
        )
        for q in instances
    ]


def resize_image(img: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """(H, W, 3) float -> (height, width, 3) float, bilinear."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    x = img.permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(
        x, size=(height, width), mode="bilinear", align_corners=False
    )
    return x[0].permute(1, 2, 0).contiguous()


def build_front_end(emap, clip_dim: int, encoders_ckpt, device, opinion_mode="edl",
                    head_ckpt_path=None):
    """FastSAM+CLIP+EDL query encoder; actionable error if deps are absent."""
    from usgloc.evidential import EvidentialHead
    from usgloc.query import RealQueryEncoder

    num_classes = emap.primitives[0].belief.shape[0]
    head = EvidentialHead(in_dim=clip_dim, num_classes=num_classes)
    if head_ckpt_path is not None:
        head.load_state_dict(torch.load(head_ckpt_path, map_location="cpu",
                                        weights_only=True))
        print(f"      front-end head overridden from {head_ckpt_path}")
    elif encoders_ckpt is not None and "edl_head" in encoders_ckpt:
        head.load_state_dict(encoders_ckpt["edl_head"])
    else:
        print("warning: no 'edl_head' state_dict in --encoders; query vacuities "
              "come from an UNTRAINED evidential head.")
    try:
        return RealQueryEncoder(edl_head=head, device=str(device),
                                opinion_mode=opinion_mode)
    except ImportError as e:
        die(f"{e}\nThe real-data front end needs the optional dependencies: "
            "pip install usgloc[realdata] (ultralytics + open_clip_torch).")


def rescale_records(records, tau: float, dof: int = 6):
    """Apply the covariance temperature Sigma <- tau * Sigma to metric records."""
    out = []
    for r in records:
        rr = dict(r)
        rr["d2"] = r["d2"] / tau
        rr["sigma_t"] = r["sigma_t"] * tau ** 0.5
        rr["nll"] = 0.5 * (r["d2"] / tau + r["logdet"] + dof * math.log(tau)
                           + dof * math.log(2 * math.pi))
        out.append(rr)
    return out


def evaluate(tag, records, dof=6):
    t_errs = [r["t_err"] for r in records]
    r_errs = [r["r_err"] for r in records]
    d2 = [r["d2"] for r in records]
    return {
        "method": tag,
        "median_t_cm": 100 * float(torch.median(torch.tensor(t_errs))),
        "median_r_deg": float(torch.median(torch.tensor(r_errs))),
        "sr_5cm_5deg": success_rate(t_errs, r_errs, 0.05, 5.0),
        "coverage": coverage(d2, dof=dof),
        "ece": coverage_ece(d2, dof=dof),
        "ause_t": ause([r["sigma_t"] for r in records], t_errs),
        "mean_nll": float(torch.tensor([r["nll"] for r in records]).mean()),
    }


def fmt(m):
    cov = m["coverage"]
    return (
        f"| {m['method']:<10} | {m['median_t_cm']:6.2f} | {m['median_r_deg']:6.2f} "
        f"| {100*m['sr_5cm_5deg']:5.1f} | {100*cov[0.683]:5.1f} / {100*cov[0.954]:5.1f} / {100*cov[0.997]:5.1f} "
        f"| {m['ece']:.3f} | {m['ause_t']:.3f} | {m['mean_nll']:7.2f} |"
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True, help="scene name, e.g. chess")
    ap.add_argument("--sfm-pose-dir", default=None,
                    help="pgt/sfm/7scenes directory of the visloc_pseudo_gt_limitations "
                         "release: evaluate against SfM pseudo-GT (the protocol Table I "
                         "commits to) instead of the original DSLAM pose.txt files")
    ap.add_argument("--map", required=True,
                    help="torch.save'd evidential 3DGS map (see usgloc.datasets.load_gaussian_map)")
    ap.add_argument("--encoders", default=None,
                    help="torch.save'd dict {'descriptor', 'aggregator'[, 'edl_head']} of "
                         "trained state_dicts; WITHOUT THIS, RETRIEVAL IS RANDOM")
    ap.add_argument("--queries", type=int, default=50,
                    help="number of test frames to localize (evenly subsampled; 0 = all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top-k", type=int, default=3, help="pose hypotheses from retrieval")
    ap.add_argument("--iters", type=int, default=120, help="refinement iterations")
    ap.add_argument("--temperature-split", type=float, default=0.5,
                    help="LEGACY: fraction of eval queries used to fit the covariance "
                         "temperature (fits tau on test-side data and reports only the "
                         "remainder). For paper numbers use --temp-calib-frames instead.")
    ap.add_argument("--temp-calib-frames", type=int, default=0,
                    help="fit tau on this many TRAIN-side frames (offset --calib-offset) "
                         "instead of test queries; the FULL eval split is then reported "
                         "(the protocol of Sec. V: calibration never touches test data)")
    ap.add_argument("--calib-offset", type=int, default=11,
                    help="frame offset of the train-side calibration frames (chosen to be "
                         "disjoint from map-fit/distill strides and --val-offset)")
    ap.add_argument("--seq-mode", default=None,
                    help="evaluate CONSECUTIVE frames of one test sequence "
                         "(e.g. seq-03) instead of a spread subsample, for "
                         "trajectory-level / filtering experiments")
    ap.add_argument("--seq-stride", type=int, default=5)
    ap.add_argument("--val-stride", type=int, default=0,
                    help="VALIDATION MODE: evaluate on every K-th TRAIN frame (offset "
                         "--val-offset) instead of the test split -- for hyperparameter "
                         "tuning without test-set leakage; never for paper numbers")
    ap.add_argument("--val-offset", type=int, default=5)
    ap.add_argument("--save-visuals", default=None,
                    help="directory to dump per-query qualitative panels (query RGB, "
                         "render at the coarse anchor pose, render at the refined pose, "
                         "rendered vacuity U_r, per-pixel weight e^{-gamma U_r}) -- the "
                         "UGS-Loc-style qualitative figure inputs and video material")
    ap.add_argument("--anchor-stride", type=int, default=20,
                    help="subsample the TRAIN trajectory every K frames as submap anchors")
    ap.add_argument("--submap-radius", type=float, default=3.0,
                    help="submap radius around each anchor camera centre (m)")
    ap.add_argument("--renderer", choices=("torch", "gsplat"), default="torch",
                    help="refinement backend: 'torch' (pure PyTorch, O(N x pixels), "
                         "keep --width/--height small) or 'gsplat' (CUDA rasterizer, "
                         "full resolution feasible)")
    ap.add_argument("--width", type=int, default=160,
                    help="refinement render width (TorchSplatRenderer is O(N x pixels))")
    ap.add_argument("--height", type=int, default=120, help="refinement render height")
    ap.add_argument("--fuse-rule", choices=("yager", "dempster"), default="yager",
                    help="DST fusion rule for cluster opinions (ablation: dempster)")
    ap.add_argument("--lambda-u", type=float, default=2.0,
                    help="evidential matching-gate sharpness (ablation: 0 = no gate)")
    ap.add_argument("--shortlist", type=int, default=10,
                    help="hierarchical retrieval: CLIP whole-image shortlist size "
                         "reranked by the evidential gated similarity (0 = rank all "
                         "submaps with the gated similarity alone)")
    ap.add_argument("--front-head-ckpt", default=None,
                    help="override the front-end head state_dict (needed for the "
                         "softmax ablation, whose CE-trained head is not the one "
                         "bundled in --encoders)")
    ap.add_argument("--opinion-mode", choices=("edl", "softmax"), default="edl",
                    help="query-side opinions: evidential head or the CE/softmax "
                         "ablation (pair with a --map built with --head-mode softmax)")
    ap.add_argument("--corr-refine-iters", type=int, default=60,
                    help="photometric refinement iterations for a SNAPPED "
                         "hypothesis (0 = trust the correction, compute only "
                         "the covariance)")
    ap.add_argument("--correction", action="store_true",
                    help="evidential matching correction (LoFTR + vacuity-gated "
                         "PnP against the rendered map); rescues hypotheses "
                         "outside the photometric basin and polishes inside it")
    ap.add_argument("--weight-retrieval", action="store_true",
                    help="ablation: weight mixture components by the retrieval "
                         "score x render fidelity (the pre-fix rule) instead of "
                         "by the correction inlier support")
    ap.add_argument("--report-geometric", action="store_true",
                    help="report the polished PnP pose instead of the short-refined one")
    ap.add_argument("--support-power", type=float, default=0.0,
                    help="mixture weight *= anchor-agreement count ** power")
    ap.add_argument("--geo-verify-refine", action="store_true",
                    help="report the polish PnP pose when the short refinement loses polish inliers")
    ap.add_argument("--escalate-inliers", type=int, default=0,
                    help="support-gated retrieval escalation (0 = off): when no "
                         "corrected candidate reaches this many PnP inliers, correct "
                         "the NEXT tier of retrieval candidates too. Needs --shortlist "
                         ">= max(top_k,8) + tiers*batch to have candidates to escalate to")
    ap.add_argument("--escalate-ratio", type=float, default=0.0,
                    help="scene-independent form of --escalate-inliers: escalate while "
                         "no corrected candidate reaches this inlier/match ratio "
                         "(0.90-0.92 at the ground-truth pose on chess/office/stairs)")
    ap.add_argument("--escalate-tiers", type=int, default=2,
                    help="extra candidate tiers for --escalate-inliers")
    ap.add_argument("--escalate-batch", type=int, default=8,
                    help="candidates per extra tier for --escalate-inliers")
    ap.add_argument("--escalate-tier1", type=int, default=0,
                    help="restrict the FIRST tier to this many shortlist entries and "
                         "escalate into the rest; set it to the shortlist size the run "
                         "WITHOUT escalation used, so the two differ only by escalation")
    ap.add_argument("--escalate-order", default="gated", choices=["gated", "shortlist", "spread"],
                    help="order the escalation tiers are consumed in")
    ap.add_argument("--escalate-min-dist", type=float, default=0.0,
                    help="skip an escalation candidate whose anchor centre is within "
                         "this many metres of an already-corrected anchor")
    # --- adaptive virtual-view search (SplatHLoc's mechanism; default OFF) ---
    ap.add_argument("--vsearch-n", type=int, default=0,
                    help="adaptive virtual-view search (0/1 = off): when no corrected "
                         "candidate clears the support trigger, render+match this many "
                         "virtual views around the best candidate (sample 0 is the "
                         "unperturbed pose) and keep the one with the largest "
                         "vacuity-weighted inlier mass. Needs a trigger: --vsearch-inliers "
                         "and/or --vsearch-ratio and/or --vsearch-psnr")
    ap.add_argument("--vsearch-inliers", type=int, default=0,
                    help="trigger: search while no corrected candidate has this many PnP inliers")
    ap.add_argument("--vsearch-ratio", type=float, default=0.0,
                    help="trigger: ... this inlier/fed-match ratio")
    ap.add_argument("--vsearch-psnr", type=float, default=0.0,
                    help="trigger: ... this render PSNR (dB)")
    ap.add_argument("--vsearch-sigma-t", type=float, default=0.25,
                    help="camera-centre sampling scale (m). With --vsearch-dist normal it "
                         "is the PER-AXIS std, so the displacement magnitude averages "
                         "1.6x it; with uniform/shell it is the ball radius / shell radius")
    ap.add_argument("--vsearch-sigma-r", type=float, default=5.0,
                    help="rotation sampling scale (deg), same convention")
    ap.add_argument("--vsearch-dist", default="normal",
                    choices=["normal", "uniform", "shell"],
                    help="virtual-view sampling law")
    ap.add_argument("--vsearch-rounds", type=int, default=1,
                    help="correction rounds spent evaluating ONE virtual view (the winner "
                         "is then iterated with --corr-rounds)")
    ap.add_argument("--vsearch-seed", type=int, default=0,
                    help="seed of the virtual-view sampler (its RNG is private)")
    ap.add_argument("--vsearch-score", default="wmass",
                    choices=["wmass", "inliers", "ratio", "rms"],
                    help="how a virtual view is ranked (default: the vacuity-weighted "
                         "inlier mass sum_i exp(-gamma U_r/alpha))")
    ap.add_argument("--vsearch-prescreen", type=int, default=0,
                    help=">0: two-stage search. Render all --vsearch-n views, rank "
                         "them by a coarse global appearance descriptor, and correct "
                         "only this many. A render costs ~10x less than a "
                         "render-match-PnP round, so the same budget buys a much wider "
                         "sample -- the thing a self-similar scene needs, since no "
                         "scoring rule can pick a pose the sampling never drew.")
    ap.add_argument("--vsearch-around", default="snap", choices=["snap", "anchor"],
                    help="centre the sampling on the best corrected pose or on the top anchor")
    ap.add_argument("--vsearch-no-verify", action="store_true",
                    help="skip the render-fidelity check on the search winner")
    ap.add_argument("--vsearch-gain", type=float, default=1.0,
                    help="accept the winner only if its score is >= gain x the unperturbed "
                         "view's score")
    ap.add_argument("--vsearch-stop", type=float, default=0.0,
                    help=">0: stop sampling views once one scores this high (cost control)")
    ap.add_argument("--vsearch-no-priority", action="store_true",
                    help="do NOT give the accepted search pose the first mixture slot")
    ap.add_argument("--vsearch-support", type=int, default=1,
                    help="anchor-agreement count credited to the accepted search pose "
                         "(>1 stops several anchors agreeing on one wrong pose from "
                         "outvoting it under --support-power)")
    # --- prior-free landmark initialisation (usgloc.landmarks) -------------
    ap.add_argument("--landmark-init", action="store_true",
                    help="ADD a prior-free landmark hypothesis to the retrieval "
                         "shortlist: query SuperPoint keypoints are matched "
                         "straight against the per-Gaussian descriptor field and "
                         "solved by PnP-RANSAC (no anchor, no render, no prior, "
                         "~0.05 s). Needs --landmark-feats (a train_locfeat.py "
                         "--feats-only sidecar built for THIS map). Everything "
                         "downstream is unchanged: the fidelity check, the "
                         "de-duplication, the inlier weighting and the "
                         "anchor-agreement term still decide.")
    ap.add_argument("--landmark-feats", default=None,
                    help="descriptor-field sidecar .pt; default "
                         "<map dir>/<scene>_lfinit.pt")
    ap.add_argument("--landmark-score", default=None,
                    help="match-score sidecar .pt (train_landmarks.py); only "
                         "needed for --landmark-sample mo/top; default "
                         "<map dir>/<scene>_lmsc_init.pt when it exists")
    ap.add_argument("--landmark-sample", default="all",
                    choices=["all", "mo", "top", "det", "random"],
                    help="landmark set: 'all' every observed Gaussian (measured "
                         "best), 'mo' STDLoc's matching-oriented sampling")
    ap.add_argument("--landmark-num", type=int, default=16384,
                    help="landmark budget for the sampled modes")
    ap.add_argument("--landmark-k", type=int, default=32,
                    help="kNN width of the matching-oriented sampling")
    ap.add_argument("--landmark-vac", default="gate",
                    choices=["none", "gate", "score", "both"],
                    help="how the map's own per-Gaussian vacuity enters the "
                         "landmark set (pose-free, no render)")
    ap.add_argument("--landmark-u-max", type=float, default=0.5,
                    help="vacuity gate: drop landmarks with u > this")
    ap.add_argument("--landmark-detect", type=int, default=2048,
                    help="SuperPoint keypoints per query")
    ap.add_argument("--landmark-topk", type=int, default=1,
                    help="landmarks kept per keypoint (1 = one-way NN, STDLoc)")
    ap.add_argument("--landmark-mnn", action="store_true",
                    help="mutual nearest neighbour instead of the one-way top-k")
    ap.add_argument("--landmark-reproj-px", type=float, default=12.0,
                    help="PnP-RANSAC reprojection threshold of the sparse stage")
    ap.add_argument("--landmark-ransac-iters", type=int, default=100000)
    ap.add_argument("--landmark-solver", default="poselib",
                    choices=["poselib", "cv2_epnp", "cv2_iter"])
    ap.add_argument("--landmark-refine", default="none",
                    help="post-RANSAC refinement of the sparse pose "
                         "(none | lm | huber | cauchy | lm+huber | lm+cauchy)")
    ap.add_argument("--landmark-min-inliers", type=int, default=8)
    ap.add_argument("--landmark-sole-inliers", type=int, default=0,
                    help=">0: once the landmark solve reaches this many inliers "
                         "it becomes the SOLE hypothesis and no retrieval anchor "
                         "is corrected (the cheap variant)")
    ap.add_argument("--landmark-weight", default="max",
                    choices=["max", "mean", "one"],
                    help="retrieval weight handed to the landmark hypothesis")
    ap.add_argument("--val-exclude-anchors", type=int, default=0,
                    help="validation mode: drop shortlist anchors within this many train frames of the query (test-like anchor distance)")
    ap.add_argument("--val-exclude-seq", action="store_true",
                    help="validation mode: ALSO drop every shortlist anchor from the "
                         "query's own sequence (leave-one-sequence-out). --val-exclude-"
                         "anchors alone still leaves near-duplicate anchors from the "
                         "other mapping traversals, which is why train-side validation "
                         "is far easier than the test split on repetitive scenes "
                         "(Stairs: 5%% of queries beyond 5 cm with --val-exclude-anchors "
                         "40 alone, 42%% with this flag, 43%% on the test split)")
    ap.add_argument("--report-mean", action="store_true",
                    help="ablation: report the moment-matched mixture mean "
                         "instead of the highest-weight mode")
    ap.add_argument("--prior-json", default=None,
                    help="evaluate in the PRIOR-REFINEMENT setting used by "
                         "GS-CPR/STDLoc/UGS-Loc: read per-frame poses from "
                         "this results JSON (key 'T', camera-from-world) "
                         "and refine them instead of running retrieval")
    ap.add_argument("--corr-conf-min", type=float, default=0.2,
                    help="matcher coarse-confidence threshold; pushed INTO "
                         "kornia LoFTR (its own default is 0.2, which would "
                         "otherwise override any smaller value)")
    ap.add_argument("--corr-rounds", type=int, default=5,
                    help="render/match/PnP rounds per hypothesis; 5 is better "
                         "than 3 from a perturbed initialization, which is the "
                         "operating regime of anchor-based retrieval")
    ap.add_argument("--no-corr-gate", action="store_true",
                    help="ablation: disable the vacuity gate on correspondences")
    ap.add_argument("--corr-solver", default="cv2_iter",
                    choices=["cv2_iter", "cv2_epnp", "poselib"],
                    help="PnP-RANSAC backend of the correction: OpenCV "
                         "iterative/EPnP RANSAC (default) or poselib's "
                         "LO-RANSAC + robust bundle, the estimator STDLoc / "
                         "ULF-Loc / SplatHLoc use.  NOTE poselib's bundle only "
                         "survives with --corr-refine none")
    ap.add_argument("--corr-reproj-px", type=float, default=2.0,
                    help="RANSAC reprojection threshold of the correction, px "
                         "(default 2; STDLoc runs 8-12)")
    ap.add_argument("--corr-refine", default="lm",
                    choices=["lm", "none", "huber", "cauchy", "lm+huber",
                             "lm+cauchy", "l2", "plb", "lm+plb", "plb+cauchy"],
                    help="final refinement over the RANSAC inliers: 'lm' (the "
                         "historical cv2.solvePnPRefineLM, which OVERWRITES a "
                         "poselib bundle), 'none', an IRLS Huber/Cauchy, or "
                         "'plb' (poselib's own robust bundle)")
    ap.add_argument("--corr-refine-sigma", type=float, default=1.0,
                    help="robust scale of the IRLS refinement, px")
    ap.add_argument("--corr-refine-all", action="store_true",
                    help="run the final refinement over EVERY fed match instead of "
                         "only the RANSAC inliers. Measured on the perturbed "
                         "train-side floor (results/estimator_lift/floor): with "
                         "--corr-refine lm+huber it takes Office 1.41 -> 0.94 and "
                         "Stairs 2.17 -> 1.80 cm, because a hard inlier set throws "
                         "away correspondences that a robust loss can still use.")
    ap.add_argument("--corr-lift", default="depth",
                    choices=["depth", "xyz", "maxw", "median", "band",
                             "maxw_d", "median_d", "band_d"],
                    help="how a render-side keypoint becomes a 3D point. 'depth' "
                         "(default) uses the rendered EXPECTED depth, which blends "
                         "fore- and background into a phantom surface at every "
                         "silhouette. 'median' recomputes the compositing weight "
                         "profile along the ray and takes its median-weight depth; "
                         "'xyz' reads the alpha-normalised blended world position "
                         "off the rasteriser instead, with no camera-model "
                         "inversion. See usgloc/correction.py.")
    ap.add_argument("--corr-pl-iters", type=int, default=10000,
                    help="poselib RANSAC max_iterations (poselib's own default "
                         "is 100000, which is what STDLoc runs)")
    ap.add_argument("--corr-wvac", type=float, default=0.0,
                    help=">0: weight each correspondence by its rendered "
                         "vacuity weight exp(-gamma*U_r/alpha) ** wvac inside "
                         "the IRLS refinement (the evidence used continuously "
                         "instead of only as the u_max gate)")
    ap.add_argument("--corr-wvac-thr", type=float, default=0.0,
                    help=">0: scale each correspondence's reprojection gate by "
                         "that same weight (soft u_max inside the geometric "
                         "verification)")
    ap.add_argument("--corr-matcher", default="loftr",
                    choices=["loftr", "mast3r", "roma", "feat", "pooled", "stage",
                             "jamma"],
                    help="correction matcher: kornia LoFTR indoor (default), "
                         "MASt3R reciprocal-NN matches (needs the MASt3R "
                         "sources + checkpoint; see usgloc/correction.py), "
                         "RoMa dense warp samples (pip package romatch), or the "
                         "map's own rasterised descriptor field -- 'feat' alone, "
                         "'pooled' with LoFTR in one PnP, 'stage' after "
                         "--corr-feat-stage LoFTR rounds. The field paths need "
                         "--corr-feats.")
    ap.add_argument("--landmark-retrieval", type=int, default=0,
                    help=">0: SELECT the retrieval shortlist with the pose-free "
                         "landmark stage instead of CLIP. For each query the "
                         "landmark solve gives an absolute pose with no anchor, "
                         "and the N anchors geometrically NEAREST that pose become "
                         "the shortlist. Motivation: the CLIP shortlist never "
                         "offers an anchor closer than 13 cm on any scene (closest "
                         "of top-8: heads 12.9, chess 25.4, stairs 27.0, office "
                         "29.1, pumpkin 30.5, kitchen 31.7, fire 34.3 cm), and "
                         "initialisation distance is the dominant per-query "
                         "predictor of final error. Needs --landmark-init's field "
                         "(the corrector is shared). Distinct from --landmark-init, "
                         "which merely ADDS the landmark pose as one more "
                         "hypothesis and was measured to be a wash.")
    ap.add_argument("--use-sh", action="store_true",
                    help="render VIEW-DEPENDENT colour: substitute the map's "
                         "sh_coeffs (N,K,3) for its diffuse colors (N,3). Our "
                         "maps store one RGB per Gaussian, so they cannot express "
                         "how a surface looks from a direction the fit never saw; "
                         "optimising colour alone against test views recovers "
                         "+4.3 dB on Office and +3.5 dB on Chess, which is why "
                         "this exists. Needs a map built with "
                         "finetune_map.py --sh-degree N --save-sh.")
    ap.add_argument("--corr-feats", default=None,
                    help="per-Gaussian descriptor field for --corr-matcher "
                         "feat/pooled/stage: a train_locfeat.py --feats-only "
                         "sidecar built for THIS map (maps/<scene>_lf64.pt). "
                         "Merged into the map in memory; the map file on disk "
                         "and every RGB-matching path are untouched.")
    ap.add_argument("--corr-feat-stage", type=int, default=5,
                    help="--corr-matcher stage: leading LoFTR rounds before the "
                         "field takes over. The warm-up must be long enough for "
                         "LoFTR to CONVERGE -- truncating it hands the field a "
                         "hypothesis outside its basin, measured as 0.70 cm vs "
                         "0.34 cm on perturbed Chess.")
    ap.add_argument("--corr-stop-dt", type=float, default=0.0,
                    help=">0: stop the correction rounds once one moves the "
                         "camera centre less than this (metres) and less than "
                         "--corr-stop-dr degrees. The field iteration converges "
                         "monotonically but slowly, so its useful round budget "
                         "is large while most frames plateau well before it; "
                         "the gate is on the UPDATE, so frames still moving "
                         "keep their rounds.")
    ap.add_argument("--corr-stop-dr", type=float, default=0.0,
                    help="companion rotation threshold for --corr-stop-dt (deg)")
    ap.add_argument("--fuse-starts", action="store_true",
                    help="pool the correspondences of EVERY corrected hypothesis "
                         "into one PnP and let the result compete as a further "
                         "candidate. Each correction's inliers are 3D world points "
                         "against pixels of the same query image, so they all "
                         "constrain the identical pose; the pipeline otherwise "
                         "keeps only the winning hypothesis and discards the rest. "
                         "Motivation: on Office the field reaches a 0.32 cm floor "
                         "from ground truth on test frames but the pipeline reports "
                         "1.08, so the loss is convergence from a real anchor, not "
                         "map quality -- and three anchors see the scene from "
                         "geometrically different places.")
    ap.add_argument("--corr-feat-subpix", type=float, default=0.0,
                    help="soft-argmax temperature for SUB-PIXEL localisation of the "
                         "descriptor-field correspondences. The fine stage "
                         "(locfeat.refine_match_windows) returns INTEGER pixels, which "
                         "quantises every correspondence and puts a floor under the pose; "
                         "0 keeps that behaviour, >0 localises the render-side position "
                         "by a soft-argmax over the 3x3 neighbourhood. Try 0.1.")
    ap.add_argument("--corr-stage-warm", default="loftr",
                    choices=["loftr", "mast3r", "jamma"],
                    help="--corr-matcher stage: which matcher runs the WARM-UP "
                         "rounds. The warm-up only has to land inside the field's "
                         "~1 cm basin, so its BASIN matters, not its precision. On "
                         "the perturbed train-side floor MASt3R is far better at "
                         "that than LoFTR (Stairs 0.68 vs 2.17, Office 0.91 vs "
                         "1.41 cm); LoFTR leaves Stairs around 2.2 cm, outside the "
                         "basin, which is why staged LoFTR loses on that scene.")
    ap.add_argument("--corr-arbiter", default="none",
                    choices=["none", "stage", "rounds"],
                    help="how the returned pose is chosen among the rounds: "
                         "'none' (default) the inlier rule, or photometric "
                         "re-rendering. The arbiter is a measured NEGATIVE "
                         "result (picks the best candidate 30-35%% of the time "
                         "against the inlier rule's oracle agreement); kept for "
                         "the ablation, not recommended.")
    ap.add_argument("--corr-min-inlier-ratio", type=float, default=None,
                    help="success needs inliers >= ratio*matches (default 0 for loftr, 0.5 otherwise)")
    ap.add_argument("--corr-roma-model", default="indoor", choices=["indoor", "outdoor"],
                    help="RoMa weights when --corr-matcher roma")
    ap.add_argument("--corr-roma-num", type=int, default=5000,
                    help="matches sampled from RoMa's dense warp per round")
    ap.add_argument("--gamma", type=float, default=3.0,
                    help="per-pixel vacuity weight sharpness (Eq. 9)")
    ap.add_argument("--oom-retries", type=int, default=0,
                    help="on a shared GPU: retry a query this many times after a CUDA "
                         "OOM (waiting --oom-wait seconds and emptying the cache) "
                         "instead of aborting the run. 0 = abort, the historical "
                         "behaviour; retries change no computed quantity")
    ap.add_argument("--oom-wait", type=float, default=60.0,
                    help="seconds to wait between --oom-retries attempts")
    ap.add_argument("--snap-rank", default="psnr",
                    choices=["psnr", "inliers", "inl_cov"],
                    help="what decides which corrected candidates take the "
                         "top-k mixture slots (default psnr: render fidelity, "
                         "the historical behaviour)")
    ap.add_argument("--snap-overflow", default="anchor", choices=["anchor", "snap"],
                    help="a corrected candidate beyond the last mixture slot "
                         "falls back to the photometric path from its anchor "
                         "(default) or from the corrected pose")
    ap.add_argument("--coverage-power", type=float, default=0.0,
                    help="mixture weight *= (fraction of an 8x8 image grid "
                         "holding a PnP inlier) ** p (default 0 = off)")
    ap.add_argument("--cov-w-power", type=float, default=0.0,
                    help="as --coverage-power, with the evenness-softened "
                         "coverage (normalized perplexity of the inlier "
                         "distribution over the grid)")
    ap.add_argument("--ratio-power", type=float, default=0.0,
                    help="mixture weight *= (inliers / fed matches) ** p")
    ap.add_argument("--rms-gamma", type=float, default=0.0,
                    help="mixture weight *= exp(-gamma * (inlier RMS px - best))")
    ap.add_argument("--debug-json", default=None,
                    help="dump every query's FULL hypothesis set (anchor pose, "
                         "corrected pose, inliers/coverage/RMS per candidate, "
                         "kept components with their weights) plus the ground "
                         "truth to this JSON. Diagnosis only -- the pipeline's "
                         "debug capture changes no computed quantity, and the "
                         "reported metrics are bit-for-bit the same.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ----- input validation: this is a harness, users supply the artefacts.
    if not Path(args.root).is_dir():
        die(f"--root {args.root!r} is not a directory.\n"
            "Download 7Scenes (https://www.microsoft.com/en-us/research/project/"
            "rgb-d-dataset-7-scenes/) and pass its root, which must contain "
            f"{args.scene}/TrainSplit.txt, TestSplit.txt and seq-XX/ folders.")
    if not Path(args.map).is_file():
        die(f"--map {args.map!r} does not exist.\n"
            "Export your trained open-vocabulary evidential 3DGS map as a torch.save'd "
            "dict with keys means, covs, colors, opacities, clip_feats, beliefs, "
            "vacuities (optional: instance_ids); see usgloc.datasets.load_gaussian_map.")

    print(f"[1/5] loading 7Scenes scene '{args.scene}' ({args.device})")
    try:
        DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
        train_ds = DS(args.root, args.scene, split="train",
                      sfm_pose_dir=args.sfm_pose_dir)
        test_ds = DS(args.root, args.scene, split="test",
                     sfm_pose_dir=args.sfm_pose_dir)
    except (FileNotFoundError, ValueError) as e:
        die(f"failed to parse the dataset: {e}")
    if args.sfm_pose_dir is not None:
        print(f"      ground truth: SfM pseudo-GT (focal {test_ds.focal:.2f}; "
              f"unregistered frames dropped: train {train_ds.n_unregistered}, "
              f"test {test_ds.n_unregistered})")
    else:
        print("      ground truth: original DSLAM pose.txt "
              "(NOTE: Table I of the draft uses the SfM pseudo-GT protocol; "
              "pass --sfm-pose-dir for paper numbers)")
    print(f"      train {len(train_ds)} frames / test {len(test_ds)} frames")

    print(f"[2/5] loading evidential map from {args.map}")
    try:
        emap = load_gaussian_map(args.map, fuse_rule=args.fuse_rule)
    except (ValueError, FileNotFoundError) as e:
        die(f"failed to load --map: {e}")
    anchor_poses = [train_ds[i].pose_cw for i in range(0, len(train_ds), args.anchor_stride)]
    make_submaps(emap, anchor_poses, radius=args.submap_radius,
                 camera=train_ds.scene_camera(train_ds.color_width,
                                             train_ds.color_height))
    if not emap.submaps:
        die(f"no submaps: no primitives within --submap-radius {args.submap_radius} m of "
            "any train anchor. Check that the map and the dataset share a world frame.")
    print(f"      {emap.means.shape[0]} splats, {len(emap.primitives)} primitives, "
          f"{len(emap.submaps)} submaps (anchor stride {args.anchor_stride})")
    emap = move_map(emap, device)
    if args.use_sh:
        # View-dependent colour: hand the renderer the (N, K, 3) SH in place of
        # the (N, 3) diffuse RGB.  GsplatRenderer.render accepts either shape and
        # evaluates the SH per Gaussian inside the SAME single pass, so every
        # downstream call site is untouched.
        if getattr(emap, "sh_coeffs", None) is None:
            die(f"--use-sh: {args.map} carries no 'sh_coeffs'; build it with "
                "scripts/finetune_map.py --sh-degree N --save-sh")
        emap.colors = emap.sh_coeffs.to(device, emap.means.dtype)
        K = emap.colors.shape[1]
        print(f"      view-dependent colour ON: SH degree {int(round(K ** 0.5)) - 1} "
              f"({K} coefficients per Gaussian)")

    print("[3/5] building pipeline + query front end")
    clip_dim = emap.primitives[0].clip_feat.shape[0]
    desc = InstanceDescriptorNet(clip_dim=clip_dim)
    agg = SceneAggregator()
    ckpt = None
    if args.encoders is not None:
        if not Path(args.encoders).is_file():
            die(f"--encoders {args.encoders!r} does not exist.")
        ckpt = torch.load(args.encoders, map_location="cpu", weights_only=True)
        try:
            desc.load_state_dict(ckpt["descriptor"])
            agg.load_state_dict(ckpt["aggregator"])
        except (KeyError, RuntimeError) as e:
            die(f"--encoders checkpoint must hold state_dicts under keys "
                f"'descriptor' and 'aggregator' matching InstanceDescriptorNet"
                f"(clip_dim={clip_dim}) / SceneAggregator: {e}")
        print(f"      loaded trained encoders from {args.encoders}")
    else:
        print("      WARNING: no --encoders checkpoint given. The retrieval encoders are "
              "untrained, so submap retrieval is RANDOM; treat this run as a smoke test only.")
    desc, agg = desc.to(device).eval(), agg.to(device).eval()

    front_end = build_front_end(emap, clip_dim, ckpt, device,
                                opinion_mode=args.opinion_mode,
                                head_ckpt_path=args.front_head_ckpt)
    cam = test_ds.scene_camera(args.width, args.height)
    cfg = PipelineConfig(top_k=args.top_k, refine_iters=args.iters, gamma=args.gamma,
                         corr_refine_iters=args.corr_refine_iters,
                         weight_by_inliers=not args.weight_retrieval,
                         retrieval_prior_power=1.0 if args.weight_retrieval else 0.0,
                         report_mode=not args.report_mean,
                         report_geometric=args.report_geometric,
                         support_power=args.support_power,
                         snap_rank=args.snap_rank,
                         snap_overflow=args.snap_overflow,
                         coverage_power=args.coverage_power,
                         cov_w_power=args.cov_w_power,
                         ratio_power=args.ratio_power,
                         rms_gamma=args.rms_gamma,
                         geo_verify_refine=args.geo_verify_refine,
                         escalate_inliers=args.escalate_inliers,
                         escalate_ratio=args.escalate_ratio,
                         escalate_tiers=args.escalate_tiers,
                         escalate_batch=args.escalate_batch,
                         escalate_tier1=args.escalate_tier1,
                         escalate_order=args.escalate_order,
                         escalate_min_dist=args.escalate_min_dist,
                         vsearch_n=args.vsearch_n,
                         vsearch_inliers=args.vsearch_inliers,
                         vsearch_ratio=args.vsearch_ratio,
                         vsearch_psnr=args.vsearch_psnr,
                         vsearch_sigma_t=args.vsearch_sigma_t,
                         vsearch_sigma_r=args.vsearch_sigma_r,
                         vsearch_dist=args.vsearch_dist,
                         vsearch_rounds=args.vsearch_rounds,
                         vsearch_seed=args.vsearch_seed,
                         vsearch_score=args.vsearch_score,
                         vsearch_prescreen=args.vsearch_prescreen,
                         fuse_starts=args.fuse_starts,
                         vsearch_around=args.vsearch_around,
                         vsearch_verify=not args.vsearch_no_verify,
                         vsearch_gain=args.vsearch_gain,
                         vsearch_stop=args.vsearch_stop,
                         vsearch_priority=not args.vsearch_no_priority,
                         vsearch_support=args.vsearch_support,
                         landmark_init=args.landmark_init,
                         landmark_sole_inliers=args.landmark_sole_inliers,
                         landmark_weight=args.landmark_weight,
                         lambda_u=args.lambda_u)
    if args.renderer == "gsplat":
        from usgloc.rendering import GsplatRenderer

        renderer = GsplatRenderer()
    else:
        renderer = TorchSplatRenderer()
    corrector = None
    if args.correction:
        if args.renderer != "gsplat":
            die("--correction needs the gsplat renderer (rendered depth).")
        from usgloc.correction import EvidentialMatchCorrector

        corr_cam = test_ds.scene_camera(640, 480)   # LoFTR's native ScanNet size
        if args.corr_matcher in ("feat", "pooled", "stage"):
            if not args.corr_feats:
                die(f"--corr-matcher {args.corr_matcher} needs --corr-feats")
            from usgloc import locfeat
            locfeat.attach_loc_feats(emap, args.corr_feats, device=emap.means.device)
        corrector = EvidentialMatchCorrector(
            emap, renderer, corr_cam, gamma=args.gamma, rounds=args.corr_rounds,
            conf_min=args.corr_conf_min,
            gate_vacuity=not args.no_corr_gate,
            matcher=args.corr_matcher,
            # The field paths keep LoFTR's 0.0: they feed ~2.4x fewer
            # correspondences, so the same ratio means something different, and
            # the floor harness measured them without a ratio gate.
            min_inlier_ratio=(args.corr_min_inlier_ratio if args.corr_min_inlier_ratio is not None
                              else (0.3 if args.corr_matcher in ("mast3r", "roma") else 0.0)),
            roma_model=args.corr_roma_model, roma_num=args.corr_roma_num,
            feat_subpix=args.corr_feat_subpix,
            feat_stage=args.corr_feat_stage, arbiter=args.corr_arbiter,
            stage_warm=args.corr_stage_warm,
            stop_dt=args.corr_stop_dt, stop_dr=args.corr_stop_dr,
            solver=args.corr_solver, reproj_thresh_px=args.corr_reproj_px,
            refine=args.corr_refine, refine_sigma=args.corr_refine_sigma,
            refine_all=args.corr_refine_all, lift=args.corr_lift,
            pl_max_iters=args.corr_pl_iters,
            wvac=args.corr_wvac, wvac_thr=args.corr_wvac_thr,
        )
        _mname = {"mast3r": "MASt3R", "roma": f"RoMa {args.corr_roma_model}",
                  "jamma": "JamMa",
                  "feat": "descriptor field", "pooled": "LoFTR + field pooled",
                  "stage": (f"{ {'mast3r': 'MASt3R', 'loftr': 'LoFTR', 'jamma': 'JamMa'}.get(args.corr_stage_warm, 'LoFTR') }"
                            f" x{args.corr_feat_stage} then field")}.get(
            args.corr_matcher, "LoFTR indoor")
        print(f"      matching correction ON ({_mname}, 640x480, "
              f"vacuity gate {'off' if args.no_corr_gate else 'on'}, "
              f"solver {args.corr_solver} @ {args.corr_reproj_px:g} px, "
              f"refine {args.corr_refine}"
              + (f", wvac {args.corr_wvac:g}" if args.corr_wvac > 0 else "")
              + (f", wthr {args.corr_wvac_thr:g}" if args.corr_wvac_thr > 0 else "")
              + ")")
    lm_corrector = None
    # --landmark-retrieval needs the same corrector, so build it for either flag.
    if args.landmark_init or args.landmark_retrieval > 0:
        from usgloc import locfeat as _locfeat
        from usgloc.landmarks import LandmarkCorrector

        mdir, mstem = Path(args.map).parent, Path(args.map).stem
        feats = args.landmark_feats
        if feats is None:
            # <map stem>_lfinit.pt first (unambiguous when several maps of one
            # scene carry fields), then the plain <scene>_lfinit.pt. Either way
            # attach_loc_feats() refuses a field built for a different map.
            for c in (mdir / f"{mstem}_lfinit.pt", mdir / f"{args.scene}_lfinit.pt"):
                if c.is_file():
                    feats = str(c)
                    break
            feats = feats or str(mdir / f"{args.scene}_lfinit.pt")
        if not Path(feats).is_file():
            die(f"--landmark-init needs a per-Gaussian descriptor field; "
                f"{feats!r} does not exist. Build it with:\n"
                f"  python3 scripts/analysis/train_locfeat.py --scene {args.scene} "
                f"--map {args.map} --out {feats} --feats-only --iters 0")
        try:
            _locfeat.attach_loc_feats(emap, feats, device=device)
        except ValueError as e:
            die(f"--landmark-feats does not match --map: {e}")
        score = args.landmark_score
        if score is None:
            for c in (mdir / f"{mstem}_lmsc_init.pt",
                      mdir / f"{args.scene}_lmsc_init.pt"):
                if c.is_file():
                    score = str(c)
                    break
        if score is None and args.landmark_sample in ("mo", "top"):
            die(f"--landmark-sample {args.landmark_sample} needs the match-score "
                f"field; build it with scripts/analysis/train_landmarks.py and "
                f"pass --landmark-score")
        lm_cam = test_ds.scene_camera(640, 480)   # SuperPoint's fitting resolution
        lm_corrector = LandmarkCorrector(
            emap, None, lm_cam,
            landmark_score=(score if args.landmark_sample in ("mo", "top", "det")
                            else None),
            landmark_num=args.landmark_num, landmark_k=args.landmark_k,
            landmark_sample=args.landmark_sample,
            landmark_vac=args.landmark_vac, landmark_u_max=args.landmark_u_max,
            landmark_detect=args.landmark_detect, landmark_topk=args.landmark_topk,
            landmark_mnn=args.landmark_mnn,
            solver=args.landmark_solver, reproj_thresh_px=args.landmark_reproj_px,
            ransac_iters=args.landmark_ransac_iters,
            min_inliers=args.landmark_min_inliers,
            refine=args.landmark_refine, gamma=args.gamma,
        )
        _li = lm_corrector.landmarks.info
        print(f"      landmark init ON (field {feats}, {_li['n_landmarks']} "
              f"landmarks / {_li['n_candidates']} candidates, sample "
              f"{args.landmark_sample}, vac {args.landmark_vac}"
              + (f" u<={args.landmark_u_max:g}" if args.landmark_vac in
                 ("gate", "both") else "")
              + f", {args.landmark_detect} keypoints, {args.landmark_solver} @ "
                f"{args.landmark_reproj_px:g} px"
              + (f", SOLE above {args.landmark_sole_inliers} inliers"
                 if args.landmark_sole_inliers > 0 else "") + ")")
    pipe = USGLocPipeline(emap, renderer, cam, desc, agg, cfg, corrector=corrector,
                          landmark_corrector=lm_corrector)
    if args.debug_json:
        pipe.debug = True
        print(f"      hypothesis-set dump ON -> {args.debug_json}")
    pipe.encode_map()
    debug_rows = []

    import time

    def _retry(fn, label):
        """Run fn(), retrying failures a saturated SHARED GPU produces.

        Retries change no computed quantity: the same call is simply made
        again once memory frees. --oom-retries 0 (the default) re-raises
        immediately, which is the historical behaviour.
        """
        for _attempt in range(args.oom_retries + 1):
            try:
                return fn()
            except RuntimeError as e:
                m = str(e).lower()
                transient = (isinstance(e, torch.OutOfMemoryError)
                             or "out of memory" in m or "cusolver" in m
                             or "cublas" in m or "launch timed out" in m)
                if not transient or _attempt == args.oom_retries:
                    raise
                torch.cuda.empty_cache()
                print(f"      transient CUDA failure on {label} (attempt "
                      f"{_attempt + 1}/{args.oom_retries}); waiting "
                      f"{args.oom_wait:.0f}s: {str(e).splitlines()[0][:90]}",
                      flush=True)
                time.sleep(args.oom_wait)

    anchor_globals = None
    if args.shortlist > 0:
        # Re-associate each surviving submap with its anchor frame (empty
        # anchors were dropped in order), then embed those train images with
        # the front end's CLIP for the viewpoint-robust global shortlist.
        anchor_ids = list(range(0, len(train_ds), args.anchor_stride))
        kept_frames, k = [], 0
        for i in anchor_ids:
            if k < len(emap.submaps) and torch.allclose(
                emap.submaps[k].anchor_pose, train_ds[i].pose_cw.to(device)
            ):
                kept_frames.append(i)
                k += 1
        assert len(kept_frames) == len(emap.submaps)
        print(f"      embedding {len(kept_frames)} anchor images for the "
              f"global shortlist (top-{args.shortlist})")
        gfeats = []
        for i in kept_frames:
            bgr_a = (train_ds.load_image(i).numpy() * 255).astype("uint8")[:, :, ::-1]
            gfeats.append(_retry(lambda: front_end.encode_global(bgr_a),
                                 f"anchor {i} global"))
        anchor_globals = torch.stack(gfeats)
        anchor_seqs = [train_ds[i].seq for i in kept_frames]

    def save_png(arr, path):
        """(H, W[, 3]) float tensor in [0, 1] -> 8-bit PNG."""
        from PIL import Image

        a = (arr.detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(a).save(path)

    def save_visual_panels(res, cond, img, rec_ds, tag):
        vis = Path(args.save_visuals)
        vis.mkdir(parents=True, exist_ok=True)
        stem = f"{tag}_{rec_ds.seq}_frame-{rec_ds.frame_id:06d}"
        k = max(range(len(res.psnrs)), key=lambda j: res.psnrs[j])
        with torch.no_grad():
            out_init = pipe.renderer.render(
                emap.means, emap.covs, emap.colors, emap.opacities,
                res.hypotheses[k].T0, cam, vacuity=emap.gaussian_vacuity)
            out_ref = pipe.renderer.render(
                emap.means, emap.covs, emap.colors, emap.opacities,
                cond.T_mode, cam, vacuity=emap.gaussian_vacuity)
        save_png(img, vis / f"{stem}_query.png")
        save_png(out_init["rgb"], vis / f"{stem}_init.png")
        save_png(out_ref["rgb"], vis / f"{stem}_refined.png")
        save_png(out_ref["vacuity"], vis / f"{stem}_vacuity.png")
        save_png(torch.exp(-args.gamma * out_ref["vacuity"]), vis / f"{stem}_weight.png")
        import json as _json
        with open(vis / f"{stem}_poses.json", "w") as f:   # for the third-person map view
            _json.dump({"T0": res.hypotheses[k].T0.detach().cpu().tolist(),
                        "T_mode": cond.T_mode.detach().cpu().tolist()}, f)

    prior_poses = None
    if args.prior_json:
        import json as _json
        _pj = _json.load(open(args.prior_json))
        prior_poses = {(r["seq"], r["frame_id"]): torch.tensor(r["T"], dtype=torch.float32)
                       for r in _pj["records"] if r.get("T") is not None}
        print(f"      PRIOR-REFINEMENT mode: {len(prior_poses)} prior poses "
              f"from {args.prior_json} (retrieval bypassed)")

    # MASt3R / RoMa consume the RGB frame, and so does the MASt3R warm-up of
    # the stage cascade; every other consumer (LoFTR, the descriptor field,
    # landmarks, vsearch) converts RGB with the bit-identical luma formula, so
    # passing RGB is a no-op for them.  Recorded in the output JSON because
    # runs before this fix fed the cascade's MASt3R warm-up a luma frame.
    corr_query_rgb = corrector is not None and (
        args.corr_matcher in ("mast3r", "roma")
        or (args.corr_matcher == "stage" and args.corr_stage_warm == "mast3r"))

    def localize_frames(ds_obj, idx, tag):
        recs = []
        for i in idx:
            rec_ds = ds_obj[i]
            img_full = ds_obj.load_image(i)
            img = resize_image(img_full, args.width, args.height).to(device)
            bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
            _lbl = f"{rec_ds.seq}/frame-{rec_ds.frame_id:06d}"
            t0 = time.perf_counter()
            instances = _retry(lambda: move_instances(front_end.encode(bgr), device),
                               _lbl + " encode")
            cand = None
            if args.landmark_retrieval > 0 and lm_corrector is not None:
                # Landmark-guided retrieval: the pose-free landmark solve gives an
                # absolute pose with NO anchor, and the anchors geometrically
                # nearest that pose become the shortlist.  CLIP ranks anchors by
                # whole-image appearance, which on every scene leaves the nearest
                # anchor outside the top-10; this ranks them by the quantity that
                # actually predicts the final error.
                _gf = resize_image(img_full, 640, 480).to(device)
                gq = (0.299 * _gf[..., 0] + 0.587 * _gf[..., 1]
                      + 0.114 * _gf[..., 2]).clamp(0, 1)   # as gray_full below
                _lm = lm_corrector.correct(gq, emap.submaps[0].anchor_pose)
                if _lm.success:
                    Clm = -(_lm.T[:3, :3].transpose(0, 1) @ _lm.T[:3, 3])
                    Ca = torch.stack([
                        -(sm.anchor_pose[:3, :3].transpose(0, 1) @ sm.anchor_pose[:3, 3])
                        for sm in emap.submaps])
                    d = (Ca - Clm[None]).norm(dim=-1)
                    if args.val_stride > 0 and (args.val_exclude_anchors > 0
                                                or args.val_exclude_seq):
                        far = torch.tensor(
                            [abs(f - i) >= args.val_exclude_anchors for f in kept_frames],
                            device=d.device)
                        if args.val_exclude_seq:
                            far &= torch.tensor([sq != rec_ds.seq for sq in anchor_seqs],
                                                device=d.device)
                        d = torch.where(far, d, torch.full_like(d, float("inf")))
                    k = int(min(args.landmark_retrieval, int(torch.isfinite(d).sum())))
                    if k > 0:
                        cand = d.topk(k, largest=False).indices.tolist()
            if cand is None and anchor_globals is not None:
                qg = _retry(lambda: front_end.encode_global(bgr), _lbl + " global")
                sims = anchor_globals @ qg
                n_vis = anchor_globals.shape[0]
                if args.val_stride > 0 and (args.val_exclude_anchors > 0
                                            or args.val_exclude_seq):
                    # test-like conditions on train-side frames: anchors
                    # adjacent to the query frame are hidden
                    far = torch.tensor([abs(f - i) >= args.val_exclude_anchors
                                        for f in kept_frames], device=sims.device)
                    if args.val_exclude_seq:
                        far &= torch.tensor([s != rec_ds.seq for s in anchor_seqs],
                                            device=sims.device)
                    sims = torch.where(far, sims, torch.full_like(sims, -float('inf')))
                    n_vis = int(far.sum())
                cand = sims.topk(min(args.shortlist, n_vis)).indices.tolist()
            t1 = time.perf_counter()
            if len(instances) < 2:
                print(f"      {rec_ds.seq}/frame-{rec_ds.frame_id:06d}: "
                      "<2 instances, skipped")
                continue
            T_gt = rec_ds.pose_cw.to(device)
            gray_full = None
            if corrector is not None or lm_corrector is not None:
                gf = resize_image(img_full, 640, 480).to(device)
                gray_full = (0.299 * gf[..., 0] + 0.587 * gf[..., 1]
                             + 0.114 * gf[..., 2]).clamp(0, 1)
                if corr_query_rgb:
                    gray_full = gf          # see corr_query_rgb above
            prior_T = None
            if prior_poses is not None:
                pk = (rec_ds.seq, rec_ds.frame_id)
                if pk not in prior_poses:
                    continue
                prior_T = prior_poses[pk].to(device)
            res = _retry(lambda: pipe.localize(img, instances, candidates=cand,
                                               gray_full=gray_full, prior=prior_T),
                         _lbl + " localize")
            t2 = time.perf_counter()
            cond = res.condensed
            t_err, r_err = pose_errors(cond.T_mode, T_gt)
            # float64 + PD projection: a float32 covariance whose smallest
            # eigenvalue is ~1e-9 of the largest can solve to a negative d2
            # or a NaN log-determinant (about one query in 700).
            _Sig = _project_pd(cond.Sigma.double())
            rec_d2 = mahalanobis_sq(cond.T_mode.double(), _Sig, T_gt.double())
            rec_logdet = float(torch.logdet(_Sig))
            # The dominant mixture component is what the reported pose
            # reflects; attribute snapped/inliers to it, keep aggregates.
            k_dom = int(torch.argmax(res.posterior.weights))
            recs.append({
                "seq": rec_ds.seq, "frame_id": rec_ds.frame_id,
                "t_err": t_err, "r_err": r_err,
                "d2": rec_d2,
                "sigma_t": float(cond.translation_sigma),
                "logdet": rec_logdet,
                # NLL of the condensed Gaussian: the same model that d2 and
                # logdet describe, so the raw and tempered rows compare the
                # same quantity (the mixture NLL is kept separately).
                "nll": 0.5 * (rec_d2 + rec_logdet + 6 * math.log(2 * math.pi)),
                "nll_mixture": res.posterior.nll(T_gt),
                "psnr": max(res.psnrs),
                "t_encode_s": t1 - t0, "t_localize_s": t2 - t1,
                "snapped": bool(res.snapped[k_dom]) if res.snapped else False,
                "inliers": int(res.inliers[k_dom]) if res.inliers else 0,
                # pose + covariance, so downstream fusion can be evaluated
                # offline without re-running localization
                "T": cond.T_mode.detach().cpu().tolist(),
                "Sigma": cond.Sigma.detach().cpu().tolist(),
                "T_gt": T_gt.detach().cpu().tolist(),
                "any_snapped": bool(any(res.snapped)),
                "max_inliers": int(max(res.inliers) if res.inliers else 0),
                "n_corrections": int(res.n_corrections),
                "n_escalations": int(res.n_escalations),
                "n_vviews": int(res.n_vviews),
                "vsearch_fired": bool(res.vsearch_fired),
                "vsearch_used": bool(res.vsearch_used),
                "landmark_ok": bool(res.landmark_ok),
                "landmark_inliers": int(res.landmark_inliers),
                "landmark_sole": bool(res.landmark_sole),
                "landmark_kept": bool(res.landmark_kept),
                "landmark_sec": float(res.landmark_sec),
            })
            if args.debug_json and pipe.last_debug is not None:
                dbg = pipe.last_debug

                def _l(x):
                    return x.detach().cpu().tolist() if torch.is_tensor(x) else x

                debug_rows.append({
                    "tag": tag, "frame": int(i), "seq": rec_ds.seq,
                    "frame_id": int(rec_ds.frame_id),
                    "T_gt": T_gt.detach().cpu().tolist(),
                    "T_rep": cond.T_mode.detach().cpu().tolist(),
                    "sigma_t": float(cond.translation_sigma),
                    "d2": rec_d2, "logdet": rec_logdet,
                    "n_cand": (len(cand) if cand is not None else 0),
                    "n_snapped": int(dbg.get("n_snapped", 0)),
                    "snaps": [{k: _l(v) for k, v in s.items()}
                              for s in dbg.get("snaps", [])],
                    "kept": [{k: _l(v) for k, v in c.items()}
                             for c in dbg.get("kept", [])],
                })
            if args.save_visuals:
                save_visual_panels(res, cond, img, rec_ds, tag)
            print(f"      [{tag}] {rec_ds.seq}/frame-{rec_ds.frame_id:06d}: "
                  f"{100*t_err:7.2f} cm / {r_err:6.2f} deg "
                  f"| psnr {max(res.psnrs):5.1f} dB | {t2 - t1:4.1f}s")
        return recs

    if args.val_stride > 0:
        eval_ds = train_ds
        idx = list(range(args.val_offset, len(train_ds), args.val_stride))
        if args.queries > 0:
            sel = torch.linspace(0, len(idx) - 1, args.queries).round().long().unique()
            idx = [idx[k] for k in sel.tolist()]
        print(f"[4/5] VALIDATION MODE: localizing {len(idx)} TRAIN-side frames "
              "(tuning only -- never paper numbers)")
    else:
        eval_ds = test_ds
        if args.seq_mode:
            idx = [i for i in range(len(test_ds))
                   if test_ds[i].seq == args.seq_mode][::args.seq_stride]
            if args.queries > 0:
                idx = idx[: args.queries]
            print(f"      sequence mode: {len(idx)} consecutive frames of "
                  f"{args.seq_mode} (stride {args.seq_stride})")
        else:
            n_q = len(test_ds) if args.queries <= 0 else min(args.queries, len(test_ds))
            idx = torch.linspace(0, len(test_ds) - 1, n_q).round().long().unique().tolist()
        print(f"[4/5] localizing {len(idx)} test frames")
    records = localize_frames(eval_ds, idx, "val" if args.val_stride > 0 else "test")
    if len(records) < 4:
        die(f"only {len(records)} frames produced >=2 query instances; "
            "not enough to report metrics.")

    print("[5/5] results")
    if args.temp_calib_frames > 0:
        # Paper protocol: tau from TRAIN-side frames, full eval split reported.
        stride = max(1, len(train_ds) // args.temp_calib_frames)
        # Exclude anchor frames explicitly: for some (stride, offset) pairs
        # the arithmetic progression collides with the anchor grid.
        calib_idx = [i for i in range(args.calib_offset, len(train_ds), stride)
                     if i % args.anchor_stride != 0][: args.temp_calib_frames]
        print(f"      fitting tau on {len(calib_idx)} train-side calibration frames "
              f"(offset {args.calib_offset})")
        calib = localize_frames(train_ds, calib_idx, "calib")
        test_recs = records
    else:
        n_cal = int(round(args.temperature_split * len(records)))
        calib, test_recs = records[:n_cal], records[n_cal:]
        if not test_recs:
            die("--temperature-split leaves no test queries; use a value < 1.")
    if fit_temperature is not None and calib:
        tau = fit_temperature([r["d2"] for r in calib], dof=6, method="median")
        print(f"      covariance temperature (fit on {len(calib)} "
              f"{'train-side' if args.temp_calib_frames > 0 else 'eval'} queries, "
              f"Sec. IV-D): tau={tau:.2f}")
    else:
        tau = 1.0
        if not calib:
            print("      warning: empty calibration split; tau=1 (raw Laplace covariance).")
        else:
            print("      warning: usgloc.calibration not available; tau=1 "
                  "(raw Laplace covariance, expect overconfident coverage).")
    results = [
        evaluate("raw", test_recs),
        evaluate("tempered", rescale_records(test_recs, tau)),
    ]
    header = ("| method     | med t (cm) | med R (deg) | SR@5/5 | cov 68/95/99.7 (%) "
              "| ECE | AUSE_t | NLL |")
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for m in results:
        print(fmt(m))
    med_enc = float(torch.median(torch.tensor([r["t_encode_s"] for r in test_recs])))
    med_loc = float(torch.median(torch.tensor([r["t_localize_s"] for r in test_recs])))
    print(f"      median per-query time: encode {med_enc:.2f}s + localize {med_loc:.2f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.scene.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"args": vars(args), "tau": tau, "metrics": results, "records": records,
             "corrector_query": ("rgb" if corr_query_rgb else "luma")
             if corrector is not None else "none"},
            f, indent=2, default=str,
        )
    print(f"saved {out_path}")
    if args.debug_json:
        dbg_path = Path(args.debug_json)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dbg_path, "w") as f:
            json.dump({"args": vars(args), "rows": debug_rows}, f, default=str)
        print(f"saved {dbg_path} ({len(debug_rows)} queries)")


if __name__ == "__main__":
    main()
