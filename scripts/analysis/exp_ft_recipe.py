#!/usr/bin/env python3
"""Sweep the photometric fine-tune recipe and score every variant by the
train-side correction floor.

The adopted fine-tune (scripts/finetune_map.py: 5000 iters, every 5th train
frame, 640x480, lr_means 1.6e-5, densification off) was the first recipe that
worked, not a tuned one.  This driver rebuilds maps/<scene>_sel.pt under a
one-factor-at-a-time variation of that recipe, scores each resulting map with
the SAME protocol as scripts/analysis/exp_floor_matcher.py -- 20 train frames,
LoFTR, rounds=5, started from the ground-truth pose (the floor) and from a
20 cm / 10 deg perturbation (the honest, non-optimistic number) -- records the
render PSNR at the ground-truth pose and the build time, and then deletes the
candidate map again (they are ~1 GB each).

    python3 scripts/analysis/exp_ft_recipe.py --scenes office stairs \
        --variants base i15k s2 --out results/ft_recipe

Variants marked ``densify`` re-enable gsplat's DefaultStrategy for a short
window.  If that changes the Gaussian count the map's distilled semantics no
longer line up; for the purposes of this measurement only, the per-Gaussian
semantic tensors are carried over by a nearest-neighbour lookup from the
source map (``--nn-transfer``).  A map like that could NOT be used end to end
without re-running scripts/train_map.py's distillation stage, then
scripts/reapply_head.py and scripts/train_encoders.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# One factor at a time from the adopted recipe (``base``).
VARIANTS = {
    "base":   dict(),                                  # 5000 it, stride 5, lr_means 1.6e-5
    "i15k":   dict(iters=15000),
    "i30k":   dict(iters=30000),
    "s2":     dict(fit_stride=2),
    "s1":     dict(fit_stride=1),
    "lrm3x":  dict(lr_means=4.8e-5),
    "lrm3rd": dict(lr_means=5.3e-6),
    "dens":   dict(densify_window=(100, 2000, 200)),
    # (e) residual ideas. The floor is set by the accuracy of the RENDERED
    # DEPTH, which the photometric loss only constrains indirectly through the
    # alpha profile, so give the geometry parameters (rotations and scales) a
    # larger share of the fine-tune than the colours get.
    "geo3x":  dict(lr_quats=3e-3, lr_scales=1.5e-2),
    "geo10x": dict(lr_quats=1e-2, lr_scales=5e-2),
    "col0":   dict(lr_colors=0.0),          # geometry only: is the gain photometric?
    "best":   dict(),                       # filled in from the sweep's winners
}


def gpu_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return int(out.strip().split()[0])


def wait_gpu(need_mib: int = 4500, timeout_s: int = 7200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        f = gpu_free_mib()
        if f >= need_mib:
            return
        print(f"    [gpu] {f} MiB free < {need_mib}; waiting 60 s", flush=True)
        time.sleep(60)
    raise RuntimeError(f"GPU never had {need_mib} MiB free within {timeout_s} s")


# The GPU is shared with other long jobs; every GPU command runs under this
# advisory lock so only one of them is resident at a time.
GPU_LOCK = os.environ.get(
    "USGLOC_GPU_LOCK",
    "/tmp/claude-1005/-home-thanhnc-gauss-ws-src/"
    "887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad/.gpu.lock")


def sh(cmd, log: Path, retries: int = 25, need_mib: int = 5000) -> str:
    """Run a command, echo+append its output to ``log``, retry on CUDA OOM.

    The command runs under the shared advisory GPU lock, and the wait for free
    memory happens INSIDE the lock (so other lock users queue behind us instead
    of racing us for the memory that just came free), with a cap so we never
    sit on the lock indefinitely.
    """
    inner = ("for i in $(seq 40); do "
             f"f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | awk '{{print $1}}'); "
             f"[ \"$f\" -ge {need_mib} ] && break; sleep 30; done; "
             'exec "$@"')
    for attempt in range(retries):
        wait_gpu(need_mib)
        print(f"    $ {' '.join(cmd)}", flush=True)
        full = (["flock", "-w", "10800", GPU_LOCK, "bash", "-c", inner, "_"] + cmd
                if GPU_LOCK else cmd)
        p = subprocess.run(full, cwd=ROOT, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n{txt}")
        if p.returncode == 0:
            return txt
        transient = ("out of memory" in txt.lower() or "launch timed out" in txt.lower()
                     or "unspecified launch failure" in txt.lower())
        print(f"    !! exit {p.returncode}{' (transient)' if transient else ''}; "
              f"tail:\n{txt[-1500:]}", flush=True)
        if not transient:
            raise RuntimeError(f"command failed: {' '.join(cmd)}")
        time.sleep(60)
    raise RuntimeError(f"command kept failing: {' '.join(cmd)}")


def nn_transfer(src_path: Path, dst_path: Path, chunk: int = 256) -> dict:
    """Copy the per-Gaussian semantic tensors of ``src`` onto ``dst``'s
    Gaussians by nearest neighbour in 3D (measurement scaffold only)."""
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    dst = torch.load(dst_path, map_location="cpu", weights_only=False)
    ref = src["means"].cuda()
    q = dst["means"].cuda()
    idx = torch.empty(q.shape[0], dtype=torch.long)
    for a in range(0, q.shape[0], chunk):
        b = min(a + chunk, q.shape[0])
        idx[a:b] = torch.cdist(q[a:b], ref).argmin(1).cpu()
    del ref, q
    torch.cuda.empty_cache()
    moved = []
    for k in ("clip_feats", "beliefs", "vacuities", "instance_ids", "obs_count"):
        if k in src and torch.is_tensor(src[k]) and src[k].shape[0] == src["means"].shape[0]:
            dst[k] = src[k][idx].contiguous()
            moved.append(k)
    dst["semantics_nn_transferred"] = True
    dst.pop("semantics_invalid", None)
    torch.save(dst, dst_path)
    return dict(moved=moved, n_src=int(src["means"].shape[0]), n_dst=int(dst["means"].shape[0]))


def score(scene: str, tag: str, out_dir: Path, log: Path) -> dict:
    """Both floor runs (from GT, and from the 20 cm / 10 deg perturbation)."""
    res = {}
    for name, extra in (("gt", []), ("pert", ["--perturb", "0.2", "10"])):
        sh([sys.executable, "scripts/analysis/exp_floor_matcher.py", scene,
            "--matchers", "loftr", "--map-suffix", f"_{tag}", "--psnr",
            "--out", str(out_dir)] + extra, log)
        ptag = "" if name == "gt" else "_p0.2m10deg"
        f = out_dir / f"floor_{scene}_loftr_{tag}{ptag}.json"
        d = json.loads(f.read_text())
        res[name] = {k: v for k, v in d.items() if k != "frames"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["office", "stairs", "pumpkin", "chess"])
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--src-suffix", default="_sel", help="map the fine-tune starts from")
    ap.add_argument("--out", default="results/ft_recipe")
    ap.add_argument("--keep-maps", action="store_true")
    ap.add_argument("--iters", type=int, default=5000, help="the base recipe's iterations")
    ap.add_argument("--fit-stride", type=int, default=5)
    ap.add_argument("--lr-means", type=float, default=1.6e-5)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra finetune_map.py flags appended to every variant "
                         "(used for the 'best' combination variant)")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    for scene in args.scenes:
        for v in args.variants:
            if v not in VARIANTS:
                raise SystemExit(f"unknown variant {v!r}; known: {sorted(VARIANTS)}")
            cfg = dict(VARIANTS[v])
            tag = f"rc{v}"
            dst = Path(f"maps/{scene}_{tag}.pt")
            src = Path(f"maps/{scene}{args.src_suffix}.pt")
            res_path = out / f"{scene}_{v}.json"
            if res_path.is_file():
                print(f"== {scene}/{v}: already done, skipping", flush=True)
                continue
            cmd = [sys.executable, "scripts/finetune_map.py", "--scene", scene,
                   "--map", str(src), "--out", str(dst),
                   "--iters", str(cfg.get("iters", args.iters)),
                   "--fit-width", "640", "--fit-height", "480",
                   "--fit-stride", str(cfg.get("fit_stride", args.fit_stride)),
                   "--lr-means", str(cfg.get("lr_means", args.lr_means)),
                   "--views-uint8"]
            for k in ("lr_quats", "lr_scales", "lr_opacities", "lr_colors"):
                if k in cfg:
                    cmd += [f"--{k.replace('_', '-')}", str(cfg[k])]
            if "densify_window" in cfg:
                cmd += ["--densify-window"] + [str(x) for x in cfg["densify_window"]]
            cmd += list(args.extra)
            print(f"== {scene}/{v}: building {dst}", flush=True)
            t0 = time.time()
            txt = sh(cmd, log)
            wall_s = time.time() - t0
            # finetune_map.py prints "..., <n>s"; that is the fit alone, without
            # the time this job spent queued behind the shared GPU lock.
            build_s = wall_s
            for line in txt.splitlines():
                if line.startswith("saved ") and line.rstrip().endswith("s"):
                    try:
                        build_s = float(line.rstrip().rsplit(", ", 1)[1][:-1])
                    except (IndexError, ValueError):
                        pass
            print(f"   built in {build_s:.0f}s (wall incl. GPU queue {wall_s:.0f}s)", flush=True)
            n_src = int(torch.load(src, map_location="cpu", weights_only=False)["means"].shape[0]) \
                if "densify_window" in cfg else None
            transfer = None
            if "densify_window" in cfg:
                d = torch.load(dst, map_location="cpu", weights_only=False)
                if d.get("semantics_invalid"):
                    del d
                    print("   gaussian count changed -> NN semantic transfer (scaffold)",
                          flush=True)
                    wait_gpu()
                    transfer = nn_transfer(src, dst)
                    print(f"   transferred {transfer}", flush=True)
                else:
                    del d
            try:
                res = score(scene, tag, out, log)
            finally:
                if not args.keep_maps and dst.is_file():
                    dst.unlink()
            rec = dict(scene=scene, variant=v, cmd=" ".join(cmd), build_s=build_s,
                       wall_s=wall_s,
                       src_map=str(src), n_src_gaussians=n_src,
                       nn_transfer=transfer, **res)
            res_path.write_text(json.dumps(rec, indent=1))
            g = res["gt"]
            p = res["pert"]
            print(f"== {scene:<9}{v:<8} floor(GT) {g['floor_cm']:.3f} cm  "
                  f"floor(pert) {p['floor_cm']:.3f} cm  reg {100*g['reg_rate']:.0f}/"
                  f"{100*p['reg_rate']:.0f}%  PSNR {g['psnr_gt']:.2f}  "
                  f"{g['n_gaussians']} splats  build {build_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
