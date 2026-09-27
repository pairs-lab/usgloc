#!/usr/bin/env python3
"""Is the correction floor still limited by the GAUSSIAN SET?

The _r1 recipe (20k iters, every train frame, D-SSIM 0.2, decayed means lr)
improved the floor with the Gaussian set FROZEN.  This driver asks whether
densification -- cloning/splitting where the 2D gradient is large, which is
where reference 3DGS gets its quality -- buys more, under that same stabilised
recipe.  Every variant is scored with the SAME protocol as
scripts/analysis/exp_floor_matcher.py (FLOOR_FRAMES train frames, LoFTR,
rounds=5), both started from the ground-truth pose (the floor) and from a
20 cm / 10 deg perturbation (the number that predicts end-to-end behaviour).

Variants whose Gaussian count changes have their distilled semantics
NN-transferred from the donor map (scripts/analysis/gx_nn_transfer.py) purely
so the floor is measurable; such a map needs train_map.py's distillation
stages + reapply_head.py + train_encoders.py before end-to-end use.

    FLOOR_FRAMES=60 python3 scripts/analysis/exp_densify.py \
        --scenes chess office --variants r1 d1 sc30k --out results/densify

Candidate maps are deleted after scoring unless --keep-maps (the host disk has
only a few GB free); maps/<scene>_r1.pt and maps/<scene>_ft.pt are never
touched -- the r1/ft variants only measure them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
GPU_LOCK = os.environ.get(
    "USGLOC_GPU_LOCK",
    "/tmp/claude-1005/-home-thanhnc-gauss-ws-src/"
    "887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad/.gpu.lock")

# The stabilised _r1 recipe, as rolled out to maps/<scene>_r1.pt.
R1 = dict(iters=20000, fit_stride=1, ssim_lambda=0.2,
          lr_means=1.6e-5, lr_means_final=1.6e-7)

# variant -> (build spec | None to measure an existing map, suffix)
#   'measure': an existing map suffix, no build
#   otherwise: kwargs for scripts/analysis/gx_fit.py
VARIANTS = {
    # (a) baselines: no build, just measure at this FLOOR_FRAMES
    "ft":      dict(measure="_ft"),
    "r1":      dict(measure="_r1"),

    # (b) the stabilised recipe WITH a densification window, warm-started
    "d1":      dict(**R1, warm=True, densify=True,
                    refine_start_iter=500, refine_stop_iter=10000, refine_every=100),
    "d1long":  dict(**R1, warm=True, densify=True,
                    refine_start_iter=500, refine_stop_iter=15000, refine_every=100),
    # more aggressive growth (lower gradient threshold) and less pruning
    "d1grow":  dict(**R1, warm=True, densify=True,
                    refine_start_iter=500, refine_stop_iter=10000, refine_every=100,
                    grow_grad2d=1e-4),
    "d1grow4": dict(**R1, warm=True, densify=True,
                    refine_start_iter=500, refine_stop_iter=10000, refine_every=100,
                    grow_grad2d=5e-5),
    # densify with a from-scratch-sized means lr so the new splats can move
    "d1lrm":   dict(**{**R1, "lr_means": 1.6e-4, "lr_means_final": 1.6e-6},
                    warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=10000, refine_every=100),
    # DefaultStrategy's periodic opacity reset (every 3000 iters) is meant for
    # a from-scratch fit; on an already converged map it throws the opacities
    # away.  These two switch it off.
    "d1nr":    dict(**R1, warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=10000, refine_every=100,
                    reset_every=10_000_000),
    "d1grownr": dict(**R1, warm=True, densify=True, refine_start_iter=500,
                     refine_stop_iter=10000, refine_every=100,
                     reset_every=10_000_000, grow_grad2d=1e-4),
    # A gentler window: the default schedule roughly DOUBLES the splat count
    # and costs >1 dB, so grow briefly, with no opacity reset, and give the new
    # splats the rest of the run to settle.
    "d1soft":  dict(**R1, warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=3000, refine_every=200,
                    reset_every=10_000_000),
    # ... and the same window with a 4x stricter growth threshold, so only the
    # very worst-reconstructed regions are split.
    "d1tiny":  dict(**R1, warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=3000, refine_every=200,
                    reset_every=10_000_000, grow_grad2d=8e-4),
    # a 4x looser threshold and a longer soft window: how far does the dose go
    # before it turns into the over-densification that costs registration?
    "d1med":   dict(**R1, warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=6000, refine_every=200,
                    reset_every=10_000_000),
    # the two winners together: the gentle window plus the depth term
    "d1softdep": dict(**R1, warm=True, densify=True, refine_start_iter=500,
                      refine_stop_iter=3000, refine_every=200,
                      reset_every=10_000_000, depth_weight=0.2),
    "d1grownrdep": dict(**R1, warm=True, densify=True, refine_start_iter=500,
                        refine_stop_iter=10000, refine_every=100,
                        reset_every=10_000_000, grow_grad2d=1e-4,
                        depth_weight=0.2),

    # (c) reference 3DGS from scratch at the correction resolution
    "sc30k":   dict(iters=30000, fit_stride=1, ssim_lambda=0.2, lr_means=1.6e-4,
                    lr_means_final=1.6e-6, warm=False, densify=True,
                    refine_start_iter=500, refine_stop_iter=15000, refine_every=100),
    "sc30kg":  dict(iters=30000, fit_stride=1, ssim_lambda=0.2, lr_means=1.6e-4,
                    lr_means_final=1.6e-6, warm=False, densify=True,
                    refine_start_iter=500, refine_stop_iter=15000, refine_every=100,
                    grow_grad2d=1e-4),
    # 30k iterations over every train frame is only ~7 gradient steps per view
    # (reference 3DGS gives each of its ~200 images 100+), so the plain sc30k
    # is badly underfit.  These two restore the per-view budget, one by using
    # every 5th frame (800 views, ~37 steps each), one by tripling the budget.
    "sc30ks5": dict(iters=30000, fit_stride=5, ssim_lambda=0.2, lr_means=1.6e-4,
                    lr_means_final=1.6e-6, warm=False, densify=True,
                    refine_start_iter=500, refine_stop_iter=15000, refine_every=100),
    "sc90k":   dict(iters=90000, fit_stride=1, ssim_lambda=0.2, lr_means=1.6e-4,
                    lr_means_final=1.6e-6, warm=False, densify=True,
                    refine_start_iter=500, refine_stop_iter=45000, refine_every=100),
    # 3x denser depth seeding: more primitives from the start instead of from
    # densification (the depth maps, not the gradient, choose where they go)
    "sc30kdi": dict(iters=30000, fit_stride=1, ssim_lambda=0.2, lr_means=1.6e-4,
                    lr_means_final=1.6e-6, warm=False, densify=True,
                    refine_start_iter=500, refine_stop_iter=15000, refine_every=100,
                    init_stride=10, max_init_points=600_000),

    # (d) orthogonal control: the same window with the depth term that
    # constrains the rendered depth the correction lifts through
    "d1dep":   dict(**R1, warm=True, densify=True, refine_start_iter=500,
                    refine_stop_iter=10000, refine_every=100, depth_weight=0.2),
    "r1dep":   dict(**R1, warm=True, densify=False, depth_weight=0.2),
    "r1dep5":  dict(**R1, warm=True, densify=False, depth_weight=0.5),
    "r1dep1":  dict(**R1, warm=True, densify=False, depth_weight=1.0),
    "sc30kdep": dict(iters=30000, fit_stride=1, ssim_lambda=0.2, lr_means=1.6e-4,
                     lr_means_final=1.6e-6, warm=False, densify=True,
                     refine_start_iter=500, refine_stop_iter=15000,
                     refine_every=100, depth_weight=0.2),
}

FLAG = {"iters": "--iters", "fit_stride": "--fit-stride", "ssim_lambda": "--ssim-lambda",
        "lr_means": "--lr-means", "lr_means_final": "--lr-means-final",
        "lr_quats": "--lr-quats", "lr_scales": "--lr-scales",
        "refine_start_iter": "--refine-start-iter",
        "refine_stop_iter": "--refine-stop-iter", "refine_every": "--refine-every",
        "reset_every": "--reset-every", "grow_grad2d": "--grow-grad2d",
        "grow_scale3d": "--grow-scale3d", "grow_scale2d": "--grow-scale2d",
        "prune_opa": "--prune-opa", "depth_weight": "--depth-weight",
        "pause_refine_after_reset": "--pause-refine-after-reset",
        "init_stride": "--init-stride", "max_init_points": "--max-init-points"}


def gpu_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().split("\n")[0])


def disk_free_gb(path: Path = ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _q(cmd) -> str:
    import shlex
    return " ".join(shlex.quote(str(c)) for c in cmd)


def sh(cmd, log: Path, retries: int = 20, need_mib: int = 5000,
       lock_wait: int = 14400, guard_file: Path | None = None,
       then: list | None = None) -> str:
    """Run under the shared advisory GPU lock, waiting for free memory inside
    the lock; retry on CUDA OOM / launch timeouts.

    ``then`` runs a SECOND command inside the SAME lock acquisition (the card
    is shared with several other queues, and one acquisition can wait tens of
    minutes, so a build and the scoring of what it built are fused into one).
    ``guard_file`` makes the first command conditional on that file not
    existing, so a retry after a transient failure in ``then`` does not redo
    an expensive build that already succeeded."""
    wait = ("for i in $(seq 60); do "
            "f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits "
            "| head -1); "
            f"[ \"$f\" -ge {need_mib} ] && break; sleep 30; done\n")
    first = _q(cmd)
    if guard_file is not None:
        first = f"if [ ! -f {_q([guard_file])} ]; then {first}; fi"
    script = "set -e\n" + wait + first + "\n"
    if then is not None:
        script += _q(then) + "\n"
    label = _q(cmd) + (f" && {_q(then)}" if then else "")
    for attempt in range(retries):
        print(f"    $ {label}", flush=True)
        full = ["flock", "-w", str(lock_wait), GPU_LOCK, "bash", "-c", script]
        p = subprocess.run(full, cwd=ROOT, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        with open(log, "a") as f:
            f.write(f"\n$ {label}\n{txt}")
        if p.returncode == 0:
            return txt
        low = txt.lower()
        transient = ("out of memory" in low or "launch timed out" in low
                     or "unspecified launch failure" in low
                     or "no space left" in low)
        print(f"    !! exit {p.returncode}{' (transient)' if transient else ''}; tail:\n"
              f"{txt[-2000:]}", flush=True)
        if not transient:
            raise RuntimeError(f"command failed: {label}")
        time.sleep(90)
    raise RuntimeError(f"command kept failing: {label}")


def score_cmd(scene: str, suffix: str, out_dir: Path, frames: int,
              matcher: str) -> list:
    return [sys.executable, "scripts/analysis/gx_score.py", scene,
            "--map-suffix", suffix, "--matcher", matcher, "--frames", str(frames),
            "--out", str(out_dir)]


def read_score(scene: str, suffix: str, out_dir: Path, matcher: str) -> dict:
    tag = matcher.replace(":", "_").replace("=", "")
    stag = "" if suffix == "_ft" else f"_{suffix.lstrip('_')}"
    res = {}
    for name, ptag in (("gt", ""), ("pert", "_p0.2m10deg")):
        res[name] = json.loads(
            (out_dir / f"floor_{scene}_{tag}{stag}{ptag}.json").read_text())
    return res


def build_cmd(scene: str, v: str, cfg: dict, dst: Path, donor: Path,
              half_clip_over: int = 800_000) -> list:
    cmd = [sys.executable, "scripts/analysis/gx_fit.py", "--scene", scene,
           "--out", str(dst), "--donor", str(donor),
           "--fit-width", "640", "--fit-height", "480",
           "--half-clip-over", str(half_clip_over),
           "--views-uint8", "--psnr-frames", "20"]
    if cfg.get("warm"):
        cmd += ["--init-map", str(donor)]
    if cfg.get("densify"):
        cmd += ["--densify"]
    for k, flag in FLAG.items():
        if k in cfg:
            cmd += [flag, str(cfg[k])]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["chess", "office"])
    ap.add_argument("--variants", nargs="+", default=["r1", "d1", "sc30k"])
    ap.add_argument("--donor-suffix", default="_ft",
                    help="map the warm starts begin from and the semantics are "
                         "copied from (the map _r1 itself was fine-tuned from)")
    ap.add_argument("--tag-prefix", default="gx")
    ap.add_argument("--frames", type=int, default=int(os.environ.get("FLOOR_FRAMES", 60)))
    ap.add_argument("--matcher", default="loftr")
    ap.add_argument("--out", default="results/densify")
    ap.add_argument("--keep-maps", action="store_true")
    ap.add_argument("--half-clip-over", type=int, default=800_000,
                    help="store the NN-transferred clip_feats as float16 above this "
                         "Gaussian count: the corrector never reads them, and a "
                         "densified candidate is otherwise several GB of disk")
    ap.add_argument("--min-disk-gb", type=float, default=4.0,
                    help="refuse to build a candidate with less free disk than this")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    for scene in args.scenes:
        donor = Path(f"maps/{scene}{args.donor_suffix}.pt")
        for v in args.variants:
            if v not in VARIANTS:
                raise SystemExit(f"unknown variant {v!r}; known: {sorted(VARIANTS)}")
            cfg = dict(VARIANTS[v])
            res_path = out / f"{scene}_{v}.json"
            if res_path.is_file() and not args.redo:
                print(f"== {scene}/{v}: already done", flush=True)
                continue
            measure = cfg.pop("measure", None)
            build_s = None
            n_gauss_in = None
            dst = None
            if measure:
                suffix = measure
                if not Path(f"maps/{scene}{suffix}.pt").is_file():
                    print(f"== {scene}/{v}: maps/{scene}{suffix}.pt missing, skipping",
                          flush=True)
                    continue
            else:
                tag = f"{args.tag_prefix}{v}"
                suffix = f"_{tag}"
                dst = Path(f"maps/{scene}_{tag}.pt")
                if not donor.is_file():
                    raise SystemExit(f"donor {donor} missing")
                free = disk_free_gb()
                if free < args.min_disk_gb and not dst.is_file():
                    raise SystemExit(f"only {free:.1f} GB free on disk; refusing to "
                                     f"build {dst} (need >= {args.min_disk_gb})")
                print(f"== {scene}/{v}: build+score {dst} ({free:.1f} GB disk free)",
                      flush=True)
            sc = score_cmd(scene, suffix, out, args.frames, args.matcher)
            t0 = time.time()
            try:
                if dst is None:
                    txt = sh(sc, log)
                else:
                    # One lock acquisition for the fit AND its scoring: the card
                    # is shared with several other queues.
                    txt = sh(build_cmd(scene, v, cfg, dst, donor,
                                       args.half_clip_over),
                             log, guard_file=dst, then=sc)
                    for line in txt.splitlines():
                        if line.startswith("done: ") and "fit " in line:
                            try:
                                build_s = float(line.split("fit ")[1].split("s")[0])
                            except (IndexError, ValueError):
                                build_s = time.time() - t0
                    if build_s is None:
                        build_s = time.time() - t0
                res = read_score(scene, suffix, out, args.matcher)
                n_gauss = int(res["gt"]["n_gaussians"])
                fit_s = build_s
                nn_done = "nn-transferred semantics" in txt
                psnr_fit = None
                for line in txt.splitlines():
                    if "fit PSNR on" in line:
                        try:
                            psnr_fit = float(line.split(":")[1].split("dB")[0])
                        except (IndexError, ValueError):
                            pass
                    if line.startswith("      fitted ") and "(was " in line:
                        try:
                            n_gauss_in = int(line.split("(was ")[1].split(")")[0])
                        except (IndexError, ValueError):
                            pass
                if dst is None:
                    # An existing map: its own fit PSNR is stored in the file.
                    ck = torch.load(f"maps/{scene}{suffix}.pt", map_location="cpu",
                                    weights_only=False)
                    psnr_fit = ck.get("psnr_fit")
                    del ck
            finally:
                if dst is not None and not args.keep_maps and dst.is_file():
                    dst.unlink()
                    print(f"   removed {dst}", flush=True)
            g, p = res["gt"], res["pert"]
            rec = dict(scene=scene, variant=v, suffix=suffix, cfg=cfg,
                       map=f"maps/{scene}{suffix}.pt", n_gaussians=n_gauss,
                       n_gaussians_in=n_gauss_in, build_s=build_s, fit_s=fit_s,
                       psnr_fit=psnr_fit, nn_transferred=nn_done,
                       frames=args.frames, matcher=args.matcher,
                       gt={k: val for k, val in g.items() if k != "frames"},
                       pert={k: val for k, val in p.items() if k != "frames"},
                       gt_frames=g["frames"], pert_frames=p["frames"])
            res_path.write_text(json.dumps(rec, indent=1))
            print(f"== {scene:<9}{v:<9} floor(GT) {g['floor_cm']:.3f}  "
                  f"floor(pert) {p['floor_cm']:.3f}  reg {100*g['reg_rate']:.0f}/"
                  f"{100*p['reg_rate']:.0f}%  inl {g['median_inliers']:.0f}  "
                  f"PSNR@GT {g['psnr_gt']:.2f}  {n_gauss} splats"
                  + (f"  build {build_s:.0f}s" if build_s else ""), flush=True)


if __name__ == "__main__":
    main()
