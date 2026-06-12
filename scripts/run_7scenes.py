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
from usgloc.datasets import SevenScenes, load_gaussian_map
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
    ap.add_argument("--root", required=True, help="7Scenes dataset root")
    ap.add_argument("--scene", required=True, help="scene name, e.g. chess")
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
                    help="fraction of queries (the first ones) used to fit the covariance "
                         "temperature; the rest are the reported test queries")
    ap.add_argument("--anchor-stride", type=int, default=20,
                    help="subsample the TRAIN trajectory every K frames as submap anchors")
    ap.add_argument("--submap-radius", type=float, default=3.0,
                    help="submap radius around each anchor camera centre (m)")
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
        train_ds = SevenScenes(args.root, args.scene, split="train")
        test_ds = SevenScenes(args.root, args.scene, split="test")
    except (FileNotFoundError, ValueError) as e:
        die(f"failed to parse the dataset: {e}")
    print(f"      train {len(train_ds)} frames / test {len(test_ds)} frames")

    print(f"[2/5] loading evidential map from {args.map}")
    try:
        emap = load_gaussian_map(args.map)
    except (ValueError, FileNotFoundError) as e:
        die(f"failed to load --map: {e}")
    anchor_poses = [train_ds[i].pose_cw for i in range(0, len(train_ds), args.anchor_stride)]
    make_submaps(emap, anchor_poses, radius=args.submap_radius)
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
    cam = SevenScenes.camera(args.width, args.height)
    cfg = PipelineConfig(top_k=args.top_k, refine_iters=args.iters, gamma=args.gamma)
    pipe = USGLocPipeline(emap, TorchSplatRenderer(), cam, desc, agg, cfg)
    pipe.encode_map()

    n_q = len(test_ds) if args.queries <= 0 else min(args.queries, len(test_ds))
    idx = torch.linspace(0, len(test_ds) - 1, n_q).round().long().unique().tolist()
    print(f"[4/5] localizing {len(idx)} test frames")
    records = []
    for i in idx:
        rec_ds = test_ds[i]
        img_full = test_ds.load_image(i)
        img = resize_image(img_full, args.width, args.height).to(device)
        bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
        instances = move_instances(front_end.encode(bgr), device)
        if len(instances) < 2:
            print(f"      {rec_ds.seq}/frame-{rec_ds.frame_id:06d}: <2 instances, skipped")
            continue
        T_gt = rec_ds.pose_cw.to(device)
        res = pipe.localize(img, instances)
        cond = res.condensed
        t_err, r_err = pose_errors(cond.T_mode, T_gt)
        records.append({
            "seq": rec_ds.seq, "frame_id": rec_ds.frame_id,
            "t_err": t_err, "r_err": r_err,
            "d2": mahalanobis_sq(cond.T_mode, cond.Sigma, T_gt),
            "sigma_t": float(cond.translation_sigma),
            "logdet": float(torch.logdet(cond.Sigma)),
            "nll": res.posterior.nll(T_gt),
            "psnr": max(res.psnrs),
        })
        print(f"      {rec_ds.seq}/frame-{rec_ds.frame_id:06d}: "
              f"{100*t_err:7.2f} cm / {r_err:6.2f} deg | psnr {max(res.psnrs):5.1f} dB")
    if len(records) < 4:
        die(f"only {len(records)} frames produced >=2 query instances; "
            "not enough to report metrics.")

    print("[5/5] results")
    n_cal = int(round(args.temperature_split * len(records)))
    calib, test_recs = records[:n_cal], records[n_cal:]
    if not test_recs:
        die("--temperature-split leaves no test queries; use a value < 1.")
    if fit_temperature is not None and calib:
        tau = fit_temperature([r["d2"] for r in calib], dof=6, method="median")
        print(f"      covariance temperature (fit on the first {len(calib)} queries, "
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
