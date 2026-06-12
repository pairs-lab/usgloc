#!/usr/bin/env python3
"""End-to-end USGLoc demo on a synthetic semi-dynamic room.

Builds an evidential map, trains the retrieval encoders, localizes a set of
query images whose dynamic objects moved after mapping, and reports the
paper's metrics -- accuracy (median errors, success rate) and calibration
(coverage @ 68.3/95.4/99.7, ECE, AUSE, NLL) -- for the full method and two
ablations:

* ``no-weight``  : gamma = 0, i.e. no per-pixel vacuity down-weighting (Eq. 9)
* ``fixed-cov``  : point estimate with one shared covariance (the "fixed noise
                   model" a filter must assume when the localizer reports none)
* ``top1-point`` : K = 1 ablation -- top-1 retrieval, single refined hypothesis,
                   point estimate (usgloc.baselines.top1_point)
* ``icomma``     : retrieval + photometric-only refinement (gamma = 0), point
                   estimate -- the iComMa-style render-and-compare baseline

Point baselines get a fixed diagonal covariance (usgloc.baselines.as_pose_gaussian)
for the calibration columns. Temperatures are fitted with usgloc.calibration.

Usage: python3 scripts/run_demo_synthetic.py [--queries 16] [--device cuda]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc import baselines, se3, synthetic
from usgloc.calibration import fit_temperature
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator
from usgloc.metrics import ause, coverage, coverage_ece, mahalanobis_sq, pose_errors, success_rate
from usgloc.pipeline import PipelineConfig, USGLocPipeline
from usgloc.posterior import PoseGaussian
from usgloc.rendering import TorchSplatRenderer


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


def temper(records, dof=6):
    """Fit a scalar covariance temperature on even-indexed queries (calibration
    split) and return re-scaled odd-indexed records (test split), per the
    draft's Sec. IV-D: Sigma <- tau * Sigma."""
    calib, test = records[::2], records[1::2]
    tau = fit_temperature([r["d2"] for r in calib], dof=dof, method="median")
    out = []
    for r in test:
        rr = dict(r)
        rr["d2"] = r["d2"] / tau
        rr["sigma_t"] = r["sigma_t"] * tau**0.5
        # NLL of the condensed Gaussian with the tempered covariance.
        rr["nll"] = 0.5 * (r["d2"] / tau + r["logdet"] + dof * np.log(tau)
                           + dof * np.log(2 * np.pi))
        out.append(rr)
    return out, tau


def evaluate(tag, records, dof=6):
    t_errs = [r["t_err"] for r in records]
    r_errs = [r["r_err"] for r in records]
    d2 = [r["d2"] for r in records]
    out = {
        "method": tag,
        "median_t_cm": 100 * float(torch.median(torch.tensor(t_errs))),
        "median_r_deg": float(torch.median(torch.tensor(r_errs))),
        "sr_5cm_5deg": success_rate(t_errs, r_errs, 0.05, 5.0),
        "coverage": coverage(d2, dof=dof),
        "ece": coverage_ece(d2, dof=dof),
        "ause_t": ause([r["sigma_t"] for r in records], t_errs),
        "mean_nll": float(torch.tensor([r["nll"] for r in records]).mean()),
    }
    return out


def fmt(m):
    cov = m["coverage"]
    return (
        f"| {m['method']:<10} | {m['median_t_cm']:6.2f} | {m['median_r_deg']:6.2f} "
        f"| {100*m['sr_5cm_5deg']:5.1f} | {100*cov[0.683]:5.1f} / {100*cov[0.954]:5.1f} / {100*cov[0.997]:5.1f} "
        f"| {m['ece']:.3f} | {m['ause_t']:.3f} | {m['mean_nll']:7.2f} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[1/4] building synthetic semi-dynamic scene + evidential map ({args.device})")
    scene = synthetic.make_scene(num_objects=args.objects, dynamic_fraction=0.25, seed=args.seed)
    emap = synthetic.build_map_from_scene(scene, n_anchors=args.anchors)
    n_dyn = int(scene.is_dynamic.sum())
    print(f"      {scene.means.shape[0]} splats, {len(emap.primitives)} primitives "
          f"({n_dyn} dynamic splats), {len(emap.submaps)} submaps")
    vac_dyn = emap.gaussian_vacuity[scene.is_dynamic].mean()
    vac_sta = emap.gaussian_vacuity[~scene.is_dynamic].mean()
    print(f"      mean map vacuity: dynamic {vac_dyn:.2f} vs static {vac_sta:.2f}")

    cam = synthetic.default_camera(80, 60)
    renderer = TorchSplatRenderer()
    desc = InstanceDescriptorNet(clip_dim=scene.clip_feats.shape[1], out_dim=64)
    agg = SceneAggregator(dim=64, heads=4)

    print("[2/4] training retrieval encoders (InfoNCE)")
    loss = synthetic.train_retrieval(desc, agg, scene, emap, cam, iters=300, seed=args.seed)
    print(f"      final loss {loss:.4f}")

    emap = move_map(emap, device)
    desc, agg = desc.to(device), agg.to(device)

    cfg_full = PipelineConfig(top_k=3, refine_iters=args.iters, refine_lr=5e-3,
                              gamma=3.0, jacobian_pixels=2048)
    cfg_nw = dataclasses.replace(cfg_full, gamma=0.0)
    pipe_full = USGLocPipeline(emap, renderer, cam, desc, agg, cfg_full)
    pipe_nw = USGLocPipeline(emap, renderer, cam, desc, agg, cfg_nw)
    pipe_full.encode_map()

    print(f"[3/4] localizing {args.queries} semi-dynamic queries (full + no-weight)")
    g = torch.Generator().manual_seed(args.seed + 1)
    recs = {"full": [], "no-weight": [], "top1-point": [], "icomma": []}
    sigmas_full, errors_full = [], []
    for q in range(args.queries):
        anchor = emap.submaps[int(torch.randint(len(emap.submaps), (1,), generator=g))].anchor_pose
        xi = torch.cat([0.06 * torch.randn(3, generator=g),
                        0.03 * torch.randn(3, generator=g)]).to(device)
        T_gt = se3.apply_tangent(anchor, xi)
        img = synthetic.render_query_image(scene, renderer, cam, T_gt.cpu()).to(device)
        inst = move_instances(
            synthetic.make_query_instances(scene, T_gt.cpu(), cam, seed=args.seed + 10 + q), device
        )
        if len(inst) < 2:
            continue
        for tag, pipe in (("full", pipe_full), ("no-weight", pipe_nw)):
            res = pipe.localize(img, inst)
            cond = res.condensed
            t_err, r_err = pose_errors(cond.T_mode, T_gt)
            rec = {
                "t_err": t_err, "r_err": r_err,
                "d2": mahalanobis_sq(cond.T_mode, cond.Sigma, T_gt),
                "sigma_t": float(cond.translation_sigma),
                "logdet": float(torch.logdet(cond.Sigma)),
                "nll": res.posterior.nll(T_gt),
            }
            recs[tag].append(rec)
            if tag == "full":
                sigmas_full.append(cond.Sigma.cpu())
                errors_full.append(se3.relative_tangent(cond.T_mode, T_gt).cpu())
        # Point-estimate baselines: K=1 ablation (full config) and the
        # iComMa-style photometric-only variant (gamma = 0 via pipe_nw).
        for tag, pipe in (("top1-point", pipe_full), ("icomma", pipe_nw)):
            b = baselines.top1_point(pipe, img, inst)
            pg = baselines.as_pose_gaussian(b, sigma_t=0.05, sigma_r=0.05)
            t_err, r_err = pose_errors(b.T_est, T_gt)
            recs[tag].append({
                "t_err": t_err, "r_err": r_err,
                "d2": float(pg.mahalanobis_sq(T_gt)),
                "sigma_t": float(pg.translation_sigma),
                "logdet": float(torch.logdet(pg.Sigma)),
                "nll": float(-pg.log_prob(T_gt)),
            })
        print(f"      query {q:02d}: full {100*recs['full'][-1]['t_err']:6.2f} cm / "
              f"{recs['full'][-1]['r_err']:5.2f} deg | no-weight "
              f"{100*recs['no-weight'][-1]['t_err']:6.2f} cm / {recs['no-weight'][-1]['r_err']:5.2f} deg | "
              f"top1 {100*recs['top1-point'][-1]['t_err']:6.2f} cm")

    # Ablation: same refined poses as `full`, but one shared (average)
    # covariance -- the fixed noise model a downstream filter must assume when
    # the localizer reports no covariance.
    Sigma_fixed = torch.stack(sigmas_full).mean(dim=0)
    fixed_gauss = PoseGaussian(T_mode=torch.eye(4), Sigma=Sigma_fixed)
    recs["fixed-cov"] = []
    for rec_full, e in zip(recs["full"], errors_full):
        r = dict(rec_full)
        r["d2"] = float(e @ torch.linalg.solve(Sigma_fixed, e))
        r["nll"] = float(-fixed_gauss.log_prob(se3.se3_exp(e)))
        r["sigma_t"] = float(fixed_gauss.translation_sigma)
        recs["fixed-cov"].append(r)

    print("[4/4] results")
    tempered, tau = temper(recs["full"])
    tempered_nw, tau_nw = temper(recs["no-weight"])
    print(f"      covariance temperature (fit on even queries): "
          f"full tau={tau:.1f}, no-weight tau={tau_nw:.1f}")
    results = [
        evaluate("full", recs["full"]),
        evaluate("full+temp", tempered),
        evaluate("no-weight", recs["no-weight"]),
        evaluate("nw+temp", tempered_nw),
        evaluate("fixed-cov", recs["fixed-cov"]),
        evaluate("top1-point", recs["top1-point"]),
        evaluate("icomma", recs["icomma"]),
    ]
    header = ("| method     | med t (cm) | med R (deg) | SR@5/5 | cov 68/95/99.7 (%) "
              "| ECE | AUSE_t | NLL |")
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for m in results:
        print(fmt(m))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved {out_dir / 'demo_results.json'}")


if __name__ == "__main__":
    main()
