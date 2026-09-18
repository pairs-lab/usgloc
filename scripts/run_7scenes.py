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


def build_front_end(emap, clip_dim: int, encoders_ckpt, device):
    """FastSAM+CLIP+EDL query encoder; actionable error if deps are absent."""
    from usgloc.evidential import EvidentialHead
    from usgloc.query import RealQueryEncoder

    num_classes = emap.primitives[0].belief.shape[0]
    head = EvidentialHead(in_dim=clip_dim, num_classes=num_classes)
    if encoders_ckpt is not None and "edl_head" in encoders_ckpt:
        head.load_state_dict(encoders_ckpt["edl_head"])
    else:
        print("warning: no 'edl_head' state_dict in --encoders; query vacuities "
              "come from an UNTRAINED evidential head.")
    try:
        return RealQueryEncoder(edl_head=head, device=str(device))
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
    ap.add_argument("--gamma", type=float, default=3.0,
                    help="per-pixel vacuity weight sharpness (Eq. 9)")
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
        emap = load_gaussian_map(args.map)
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

    front_end = build_front_end(emap, clip_dim, ckpt, device)
    cam = test_ds.scene_camera(args.width, args.height)
    cfg = PipelineConfig(top_k=args.top_k, refine_iters=args.iters, gamma=args.gamma)
    if args.renderer == "gsplat":
        from usgloc.rendering import GsplatRenderer

        renderer = GsplatRenderer()
    else:
        renderer = TorchSplatRenderer()
    pipe = USGLocPipeline(emap, renderer, cam, desc, agg, cfg)
    pipe.encode_map()

    import time

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

    def localize_frames(ds_obj, idx, tag):
        recs = []
        for i in idx:
            rec_ds = ds_obj[i]
            img_full = ds_obj.load_image(i)
            img = resize_image(img_full, args.width, args.height).to(device)
            bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
            t0 = time.perf_counter()
            instances = move_instances(front_end.encode(bgr), device)
            t1 = time.perf_counter()
            if len(instances) < 2:
                print(f"      {rec_ds.seq}/frame-{rec_ds.frame_id:06d}: "
                      "<2 instances, skipped")
                continue
            T_gt = rec_ds.pose_cw.to(device)
            res = pipe.localize(img, instances)
            t2 = time.perf_counter()
            cond = res.condensed
            t_err, r_err = pose_errors(cond.T_mode, T_gt)
            recs.append({
                "seq": rec_ds.seq, "frame_id": rec_ds.frame_id,
                "t_err": t_err, "r_err": r_err,
                "d2": mahalanobis_sq(cond.T_mode, cond.Sigma, T_gt),
                "sigma_t": float(cond.translation_sigma),
                "logdet": float(torch.logdet(cond.Sigma)),
                "nll": res.posterior.nll(T_gt),
                "psnr": max(res.psnrs),
                "t_encode_s": t1 - t0, "t_localize_s": t2 - t1,
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
        calib_idx = list(range(args.calib_offset, len(train_ds), stride))[
            : args.temp_calib_frames]
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
    out_path = out_dir / f"7scenes_{args.scene}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"args": vars(args), "tau": tau, "metrics": results, "records": records},
            f, indent=2, default=str,
        )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
