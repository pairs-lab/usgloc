#!/usr/bin/env python3
"""GSplatLoc baseline (Sidorov et al., IROS 2025, arXiv:2409.16502) on our 100 test queries, SfM pGT.

Runs the authors' released code (github.com/be2rlab/gsplatloc, cloned at third_party/gsplatloc and left
unmodified on disk). This script only
  prepare   (CPU) writes the scene in GSplatLoc's split layout: train frames + SfM-pGT poses + focal, our
            100 test queries, and a link to the COLMAP reference model its loader initialises from;
  check     (CPU) reads the prepared data back through GSplatLoc's own readers: counts, intrinsics, pose
            convention vs our T_gt, init point cloud, and a dry run of the results writer;
  train     (GPU) the main block of its train.py (same parser and defaults), calling train.training();
  localize  (GPU) its loc_inference.localize_set() one query at a time, then writes
            results/gsplatloc_100/<dataset>_<scene>.json (T world->camera in the SfM-pGT frame, t_err/r_err
            against our T_gt, exactly the 100 query keys of results/v2_<dataset>_seed0/<dataset>_<scene>.json);
  all       prepare, check, train, localize, each stage a subprocess with its own log under <work>/logs/.

Run with the GSplatLoc venv (torch 2.11+cu128 shared from ~/venvs/uags through a .pth file):
  cd ~/gauss_ws/src/usgloc && export PYTHONPATH=
  ~/venvs/gsplatloc/bin/python scripts/analysis/run_gsplatloc.py all 7scenes stairs
  ~/venvs/gsplatloc/bin/python scripts/analysis/run_gsplatloc.py all 12scenes apt2/kitchen

Settings (paper v3 where it states a value, else the released code's default):
  training   15,000 iterations (paper; code default 2,000, README example 7,000), every other
             optimisation parameter = arguments/__init__.py defaults; --data_device cpu (a README option:
             images stay in host RAM, no numerical change); --test_iterations -1 (no test-image rendering).
  localize   1,000 XFeat keypoints, 20,000 PnP-RANSAC iterations (paper = code), warp lr 1e-3 (paper; code
             default 5e-4), 251 warp iterations (code; paper "approximately 250").
  variant    the released loc_inference.py = coarse PnP + photometric warp refinement, i.e. the paper's
             "Base". The paper's "Fine" (an extra feature-based refinement) is NOT in the released code.

In-memory patches (source text of one function replaced and re-executed in its module; files on disk
untouched; each replacement must match exactly once or the run stops):
  P1 12Scenes only: scene.dataset_readers.readSplitInfo accepts paths with '12scenes' in its '7scenes'
     branch (the released reader raises "Unknown dataset" otherwise). The branch derives the COLMAP model
     path from the folder name; `prepare` puts a link to the chosen model at exactly that path.
  P2 only with --features-on-cpu: XFeat target maps kept in host RAM (memory fallback, no numerical change).
  P3 localize_set: query image moved to the GPU (images live on the CPU with --data_device cpu; the
     released code assumes the default cuda device). No numerical change.
  P4 localize_set: records the coarse (PnP) and final pose of each view; nothing else changes.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                       # usgloc project root
GSL_REPO = Path("/home/thanhnc/gauss_ws/src/third_party/gsplatloc")
WORK = Path("/home/thanhnc/gauss_ws/src/third_party/gsplatloc_work")
PGT = ROOT / "data/visloc_pseudo_gt_limitations/pgt/sfm"
SCALE = {"7scenes": 1.0, "12scenes": 0.5}   # 12Scenes 1296x968 -> 648x484 (exact half, square pixels)
NATIVE = {"7scenes": (640, 480), "12scenes": (1296, 968)}


# ----------------------------------------------------------------------------------------------- paths
def key_of(ds, scene):
    return f"{ds}_{scene.replace('/', '_')}"                         # 7scenes_stairs, 12scenes_apt2_kitchen


def paths(a):
    k = key_of(a.dataset, a.scene)
    frames = WORK / "data/frames" / f"pgt_{k}"                       # shared rgb/poses/calibration
    src = WORK / "data" / a.init_model / f"pgt_{k}"                  # GSplatLoc -s (source_path)
    model = WORK / "output" / a.init_model / k                       # GSplatLoc -m (model_path)
    if a.dataset == "7scenes":
        ref = WORK / "downloads/7scenes_ref/7scenes_reference_models" / a.scene / a.init_model
    else:
        ref = WORK / "downloads/12scenes_ref/12scenes_reference_models" / a.scene / a.init_model
    ours = ROOT / f"results/v2_{a.dataset}_seed0/{k}.json"
    out = (ROOT / a.out if not os.path.isabs(a.out) else Path(a.out)) / f"{k}.json"
    return dict(key=k, frames=frames, src=src, model=model, ref=ref, ours=ours, out=out,
                logs=WORK / "logs" / a.init_model)


def gsl_sfm_path(source_path: Path) -> Path:
    """The COLMAP model folder GSplatLoc's readSplitInfo derives from -s (same expression as its code)."""
    path = os.path.abspath(str(source_path))
    scene_name = path.split("_")[-1]
    return Path(os.path.join(f"{path.split(scene_name)[0].replace('pgt_', '')}reference_models",
                             scene_name, "old_gt_refined"))


def gsl_import_path():
    if str(GSL_REPO) not in sys.path:
        sys.path.insert(0, str(GSL_REPO))


def git_info(repo):
    def run(*c):
        return subprocess.run(["git", "-C", str(repo), *c], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def frame_name(ds, seq, fid):
    return f"{seq}-frame-{fid:06d}" if ds == "7scenes" else f"frame-{fid:06d}"


def parse_image_name(ds, name):
    m = re.match(r"(?:(seq-\d+)-)?frame-(\d+)\.color", name)
    if not m:
        raise ValueError(f"unexpected GSplatLoc image name {name!r}")
    return (m.group(1) if ds == "7scenes" else "data"), int(m.group(2))


def load_ours(p):
    d = json.load(open(p["ours"]))
    return d["records"]


# --------------------------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    p = paths(a)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(GSL_REPO / "datasets"))
    import numpy as np
    from PIL import Image
    import dataset_util as dutil                                  # GSplatLoc's own pose reader/writers
    from usgloc.datasets import SevenScenes, TwelveScenes

    DS = SevenScenes if a.dataset == "7scenes" else TwelveScenes
    root = ROOT / f"data/{a.dataset}_source"
    pgt_dir = PGT / a.dataset
    flat = a.scene.replace("/", "_")
    s = SCALE[a.dataset]
    W, H = (round(NATIVE[a.dataset][0] * s), round(NATIVE[a.dataset][1] * s))
    if not p["ref"].is_dir():
        sys.exit(f"missing COLMAP reference model {p['ref']} (download: see the README of "
                 "visloc_pseudo_gt_limitations, 'Full Reconstructions')")

    train_ds = DS(str(root), a.scene, split="train", sfm_pose_dir=str(pgt_dir))
    test_ds = DS(str(root), a.scene, split="test", sfm_pose_dir=str(pgt_dir))
    pgt = {}
    for split in ("train", "test"):
        pgt[split] = dutil.read_pose_data(str(pgt_dir / f"{flat}_{split}.txt"))  # c2w (float64), focal
    focals = {f for d in pgt.values() for _, f in d.values()}
    assert len(focals) == 1, f"expected one SfM focal per scene, got {focals}"
    f_sfm = focals.pop()
    focal = f_sfm * s

    queries = [(r["seq"], r["frame_id"]) for r in load_ours(p)]
    test_idx = {(fr.seq, fr.frame_id): i for i, fr in enumerate(test_ds.frames)}
    missing = [q for q in queries if q not in test_idx]
    assert not missing, f"queries not in the SfM test split: {missing[:5]}"

    def write_split(split, frames):
        d = p["frames"] / split
        for sub in ("rgb", "poses", "calibration"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        max_dev = 0.0
        for fr in frames:
            name = frame_name(a.dataset, fr.seq, fr.frame_id)
            c2w, f = pgt[split][f"{fr.seq}/frame-{fr.frame_id:06d}"]
            max_dev = max(max_dev, float(np.abs(c2w - fr.pose_c2w.double().numpy()).max()))
            rgb = d / "rgb" / f"{name}.color.png"
            if s == 1.0:
                if rgb.is_symlink() or rgb.exists():
                    rgb.unlink()
                rgb.symlink_to(fr.rgb_path.resolve())
            elif not rgb.exists():
                with Image.open(fr.rgb_path) as im:
                    im.convert("RGB").resize((W, H), Image.Resampling.BICUBIC).save(rgb)
            dutil.write_cam_pose(str(d / "poses" / f"{name}.pose.txt"), c2w)
            dutil.write_focal_length(str(d / "calibration" / f"{name}.calibration.txt"), f * s)
        return max_dev

    dev_tr = write_split("train", train_ds.frames)
    dev_te = write_split("test", [test_ds.frames[test_idx[q]] for q in queries])
    # the GSplatLoc source folder of this init variant: links to the shared frames + the COLMAP model
    p["src"].mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        ln = p["src"] / split
        if ln.is_symlink():
            ln.unlink()
        ln.symlink_to(p["frames"] / split, target_is_directory=True)
    sfm = gsl_sfm_path(p["src"])
    sfm.parent.mkdir(parents=True, exist_ok=True)
    if sfm.is_symlink():
        sfm.unlink()
    sfm.symlink_to(p["ref"], target_is_directory=True)
    manifest = dict(dataset=a.dataset, scene=a.scene, init_model=a.init_model, created=datetime.now().isoformat(),
                    n_train=len(train_ds), n_train_unregistered=train_ds.n_unregistered, n_test=len(queries),
                    image_size=[W, H], focal_sfm_native=f_sfm, focal=focal, principal_point=[W / 2, H / 2],
                    source_path=str(p["src"]), colmap_model_link=str(sfm), colmap_model=str(p["ref"]),
                    pose_source=str(pgt_dir), max_abs_c2w_diff_vs_usgloc_loader=[dev_tr, dev_te],
                    resize=None if s == 1.0 else f"PIL BICUBIC {NATIVE[a.dataset]} -> {(W, H)}")
    json.dump(manifest, open(p["frames"] / f"manifest_{a.init_model}.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


# ----------------------------------------------------------------------------------------------- check
def cmd_check(a):
    """CPU only: read the prepared folder through GSplatLoc's readers and compare with our data."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    p = paths(a)
    gsl_import_path()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from PIL import Image
    from scene.dataset_readers import readSplit_cams_params
    from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
    from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
    from usgloc.datasets import SevenScenes, TwelveScenes
    from usgloc.metrics import pose_errors

    ok = True
    src = p["src"]
    for split in ("train", "test"):
        names = {sub: sorted(os.listdir(src / split / sub)) for sub in ("rgb", "poses", "calibration")}
        stems = {sub: [n.split(".")[0] for n in v] for sub, v in names.items()}
        same = stems["rgb"] == stems["poses"] == stems["calibration"]
        ok &= same
        w, h = Image.open(src / split / "rgb" / names["rgb"][0]).size
        print(f"[{split}] rgb {len(names['rgb'])}  poses {len(names['poses'])}  calib {len(names['calibration'])}"
              f"  sorted stems pairwise equal: {same}  first image {names['rgb'][0]} {w}x{h}")
    K_tr, w2c_tr = readSplit_cams_params(str(src / "train/calibration"), str(src / "train/poses"))
    K_te, w2c_te = readSplit_cams_params(str(src / "test/calibration"), str(src / "test/poses"))
    W, H = Image.open(src / "train/rgb" / sorted(os.listdir(src / "train/rgb"))[0]).size
    fovx, fovy = focal2fov(K_tr, W), focal2fov(K_tr, H)
    fx_loc = fov2focal(fovx, W)                                     # what loc_inference puts in K
    DS = SevenScenes if a.dataset == "7scenes" else TwelveScenes
    ds = DS(str(ROOT / f"data/{a.dataset}_source"), a.scene, split="test", sfm_pose_dir=str(PGT / a.dataset))
    cam = ds.scene_camera(W, H)
    print(f"[intrinsics] GSplatLoc: K={K_tr} (test {K_te}) W x H={W}x{H} FoVx={np.degrees(fovx):.4f} deg "
          f"FoVy={np.degrees(fovy):.4f} deg; loc K: fx=fy={fx_loc:.4f} cx={W/2} cy={H/2}")
    print(f"[intrinsics] usgloc scene_camera({W},{H}): fx={cam.fx:.4f} fy={cam.fy:.4f} cx={cam.cx} cy={cam.cy}")
    ok &= abs(fx_loc - cam.fx) < 1e-6 and abs(cam.fx - cam.fy) < 1e-6 and cam.cx == W / 2 and cam.cy == H / 2

    # pose convention: GSplatLoc camera of each test frame vs our T_gt (world->camera)
    recs = load_ours(p)
    test_names = sorted(os.listdir(src / "test/rgb"))
    by_key = {parse_image_name(a.dataset, n.split(".png")[0]): i for i, n in enumerate(test_names)}
    dmax, cmax, rmax, show = 0.0, 0.0, 0.0, []
    for r in recs:
        i = by_key[(r["seq"], r["frame_id"])]
        w2c = w2c_te[i]
        R, T = w2c[:3, :3].T, w2c[:3, 3]                            # what readSplitInfo stores
        V = getWorld2View2(R, T).astype(np.float64)                 # the camera the rasterizer uses
        G = np.array(r["T_gt"], float)
        dmax = max(dmax, float(np.abs(V - G).max()))
        te, re_ = pose_errors(torch.tensor(V), torch.tensor(G))
        cmax, rmax = max(cmax, te), max(rmax, re_)
        if len(show) < 2:
            show.append((r["seq"], r["frame_id"], G, V))
    print(f"[poses] {len(recs)} queries: max |GSplatLoc world->view - our T_gt| = {dmax:.2e}; "
          f"centre diff max {100*cmax:.2e} cm; rot diff max {rmax:.2e} deg")
    for seq, fid, G, V in show:
        np.set_printoptions(precision=6, suppress=True)
        print(f"  {seq}/{fid}: our T_gt (world->camera)\n{G}\n  GSplatLoc getWorld2View2(R, T)\n{V}")
    ok &= dmax < 1e-5

    # init point cloud at the exact path the released reader derives
    sfm = gsl_sfm_path(src)
    xyz, rgb, _ = read_points3D_binary(str(sfm / "points3D.bin"))
    imgs = read_extrinsics_binary(str(sfm / "images.bin"))
    pg = {}
    for split in ("train",):
        for ln in open(PGT / a.dataset / f"{a.scene.replace('/', '_')}_{split}.txt"):
            q = ln.split()
            pg[q[0]] = (qvec2rotmat(np.array(q[1:5], float)), np.array(q[5:8], float))
    dc, dr = [], []
    for im in imgs.values():
        if im.name in pg:
            Rm, tm = qvec2rotmat(im.qvec), im.tvec
            Rp, tp = pg[im.name]
            dc.append(np.linalg.norm(-Rm.T @ tm + Rp.T @ tp))
            dr.append(np.degrees(np.arccos(np.clip((np.trace(Rm @ Rp.T) - 1) / 2, -1, 1))))
    print(f"[init] {sfm} -> {os.path.realpath(sfm)}: {len(xyz)} points; its train cameras vs SfM pGT: "
          f"centre diff median {100*np.median(dc):.2f} cm (max {100*np.max(dc):.2f}), rot diff median "
          f"{np.median(dr):.3f} deg (max {np.max(dr):.3f}), n={len(dc)}")

    # dry run of the results writer: estimate = T_gt moved by a known (3 cm, 1 deg) offset
    def perturb(G):
        th = np.radians(1.0)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        c = -G[:3, :3].T @ G[:3, 3] + np.array([0.03, 0, 0])
        R = Rz @ G[:3, :3]
        return R, -R @ c
    est = {}
    for r in recs:
        R, t = perturb(np.array(r["T_gt"], float))
        est[frame_name(a.dataset, r["seq"], r["frame_id"]) + ".color"] = dict(R_prior=R, t_prior=t.reshape(3, 1),
                                                                             R_est=R, t_est=t, sec=0.0)
    fake = WORK / "check" / f"{p['key']}_writer_dryrun.json"
    fake.parent.mkdir(parents=True, exist_ok=True)
    write_results(a, p, est, {}, {"dry_run": True}, fake)
    d = json.load(open(fake))
    keys = [(r["seq"], r["frame_id"]) for r in d["records"]]
    gap = max(abs(100 * np.linalg.norm(_centre(r["T"]) - _centre(r["T_gt"])) - 100 * r["t_err"]) for r in d["records"])
    te = np.array([r["t_err"] for r in d["records"]]); re_ = np.array([r["r_err"] for r in d["records"]])
    print(f"[writer] dry run {fake}: {len(keys)} records, keys == ours: {keys == [(r['seq'], r['frame_id']) for r in recs]}; "
          f"render_footage centre check gap {gap:.2e} cm (< 0.01 needed); t_err {100*te.min():.4f}..{100*te.max():.4f} cm "
          f"(3 expected), r_err {re_.min():.4f}..{re_.max():.4f} deg (1 expected)")
    ok &= gap < 0.01 and abs(te - 0.03).max() < 1e-6 and abs(re_ - 1.0).max() < 1e-4
    print("CHECK", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def _centre(T):
    import numpy as np
    T = np.asarray(T, float)
    return -T[:3, :3].T @ T[:3, 3]


def _rot_deg(Ra, Rb):
    import numpy as np
    M = Ra @ Rb.T
    s = np.linalg.norm([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2
    return float(np.degrees(np.arctan2(s, (np.trace(M) - 1) / 2)))


def _nearest_rotation(R):
    import numpy as np
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ----------------------------------------------------------------------------------------------- write
def write_results(a, p, est, failures, run_args, out_path):
    """est: GSplatLoc image_name -> dict(R_prior, t_prior, R_est, t_est, sec). One record per query of ours."""
    import numpy as np
    recs_out, failed = [], []
    for r in load_ours(p):
        name = frame_name(a.dataset, r["seq"], r["frame_id"]) + ".color"
        G = np.array(r["T_gt"], float)
        rec = {"seq": r["seq"], "frame_id": r["frame_id"]}
        e = est.get(name)
        good = e is not None and all(np.isfinite(np.asarray(e[k], float)).all() for k in ("R_est", "t_est"))
        if good:
            R_raw = np.asarray(e["R_est"], float)
            T = np.eye(4); T[:3, :3] = _nearest_rotation(R_raw); T[:3, 3] = np.asarray(e["t_est"], float).reshape(3)
            Tc = np.eye(4); Tc[:3, :3] = np.asarray(e["R_prior"], float); Tc[:3, 3] = np.asarray(e["t_prior"], float).reshape(3)
            rec["rot_orth_err"] = float(np.abs(R_raw @ R_raw.T - np.eye(3)).max())
            rec["sec"] = float(e["sec"])
            if "gsl_final" in e:               # GSplatLoc's own printed errors: [deg, cm of ||t_gt - t_est||]
                rec["gsl_internal_final"] = [float(x) for x in e["gsl_final"]]
                rec["gsl_internal_coarse"] = [float(x) for x in e["gsl_prior"]]
        else:                                  # failure (exception or non-finite pose): identity, flagged
            T, Tc = np.eye(4), np.eye(4)
            rec["failed"] = failures.get(name, "non-finite pose" if e is not None else "no estimate")
            failed.append([r["seq"], r["frame_id"], rec["failed"]])
        T_l, G_l, Tc_l = T.tolist(), G.tolist(), Tc.tolist()
        Tn, Tcn = np.array(T_l), np.array(Tc_l)              # exactly what a consumer loads back
        rec.update(T=T_l, T_gt=G_l,
                   t_err=float(np.linalg.norm(_centre(Tn) - _centre(G))), r_err=_rot_deg(Tn[:3, :3], G[:3, :3]),
                   T_coarse=Tc_l, t_err_coarse=float(np.linalg.norm(_centre(Tcn) - _centre(G))),
                   r_err_coarse=_rot_deg(Tcn[:3, :3], G[:3, :3]))
        recs_out.append(rec)
    te = np.array([r["t_err"] for r in recs_out]); re_ = np.array([r["r_err"] for r in recs_out])
    tc = np.array([r["t_err_coarse"] for r in recs_out]); rc = np.array([r["r_err_coarse"] for r in recs_out])
    metrics = {"n": len(recs_out), "n_failed": len(failed),
               "median_t_cm": 100 * float(np.median(te)), "median_r_deg": float(np.median(re_)),
               "sr_5cm_5deg": float(np.mean((te <= 0.05) & (re_ <= 5))),
               "sr_2cm_2deg": float(np.mean((te <= 0.02) & (re_ <= 2))),
               "sr_1cm_1deg": float(np.mean((te <= 0.01) & (re_ <= 1))),
               "coarse_median_t_cm": 100 * float(np.median(tc)), "coarse_median_r_deg": float(np.median(rc))}
    out = {"args": run_args, "metrics": metrics, "failed": failed, "records": recs_out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, out_path)
    return metrics


# --------------------------------------------------------------------------------------------- patches
# (old text, new text, expected number of matches) in the released function's source
P1 = ("if '7scenes' in path:", "if '7scenes' in path or '12scenes' in path:", 1)                 # readSplitInfo
P2 = ("semantic_feature = model.get_descriptors(tensor_image)[0]\n",
      "semantic_feature = model.get_descriptors(tensor_image)[0].cpu()\n", 2)                    # readSplitInfo
P3 = ("gt_im = view.original_image[0:3, :, :]", "gt_im = view.original_image[0:3, :, :].cuda()", 1)  # localize_set
P4 = ("            rErrs.append(rotError)\n",
      "            rErrs.append(rotError)\n"
      "            _USGLOC_RECORDS[view.image_name] = dict(R_prior=R, t_prior=t, R_est=R_est, t_est=t_est, "
      "sec=time.time() - start, gsl_prior=(prior_rErr[-1], prior_tErr[-1]), gsl_final=(rotError, transError))\n",
      1)                                                                                           # localize_set


def patch_function(module, name, replacements):
    """Replace text in one function's source (each (old, new, count) must match exactly `count` times) and
    re-execute it in its module (original line numbers kept), so the module's globals use the patched one.
    Returns (released source, patched source)."""
    lines, first = inspect.getsourcelines(getattr(module, name))
    orig = src = "".join(lines)
    for old, new, count in replacements:
        n = src.count(old)
        if n != count:
            raise RuntimeError(f"patch {module.__name__}.{name}: expected {count} match(es) of {old!r}, found {n}")
        src = src.replace(old, new)
    exec(compile("\n" * (first - 1) + src, module.__file__, "exec"), module.__dict__)
    return orig, src


def apply_reader_patches(a):
    import scene.dataset_readers as dr
    reps = ([P1] if a.dataset == "12scenes" else []) + ([P2] if a.features_on_cpu else [])
    if reps:
        patch_function(dr, "readSplitInfo", reps)
        dr.sceneLoadTypeCallbacks["Split"] = dr.readSplitInfo
    return [r[1].strip() for r in reps]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


# ----------------------------------------------------------------------------------------------- train
def cmd_train(a):
    p = paths(a)
    gsl_import_path()
    import torch
    import train as gsl_train                                       # GSplatLoc train.py (main is guarded)
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from utils.general_utils import safe_state
    if (p["model"] / "point_cloud" / f"iteration_{a.iterations}").exists() and not a.force:
        print(f"trained model exists: {p['model']} (use --force to retrain)"); return
    patches = apply_reader_patches(a)
    # == train.py __main__, with our paths
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    argv = ["-s", str(p["src"]), "-m", str(p["model"]), "--iterations", str(a.iterations),
            "--data_device", a.data_device, "--test_iterations", "-1",
            "--save_iterations", str(a.iterations), "--port", str(free_port())]
    args = parser.parse_args(argv)
    args.save_iterations.append(args.iterations)
    print("GSplatLoc train.py args:", " ".join(argv), "| patches:", patches)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    gsl_train.network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pipe = pp.extract(args)
    pipe.debug = a.rasterizer_debug       # released default True = CPU copy of all inputs per call; no numerical effect
    t0 = time.time()
    gsl_train.training(lp.extract(args), op.extract(args), pipe, args.test_iterations, args.save_iterations,
                       args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    wall = time.time() - t0
    from plyfile import PlyData
    ply = p["model"] / "point_cloud" / f"iteration_{a.iterations}" / "point_cloud.ply"
    n_g = PlyData.read(str(ply))["vertex"].count
    meta = dict(argv=argv, patches=patches, rasterizer_debug=a.rasterizer_debug, wall_s=wall, n_gaussians=n_g,
                peak_mem_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_mem_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                gpu=torch.cuda.get_device_name(0), finished=datetime.now().isoformat())
    json.dump(meta, open(p["model"] / "usgloc_train_meta.json", "w"), indent=1)
    print("\nTraining complete.", json.dumps(meta))


# -------------------------------------------------------------------------------------------- localize
def cmd_localize(a):
    p = paths(a)
    if p["out"].exists() and not a.force:
        sys.exit(f"{p['out']} exists (use --force)")
    gsl_import_path()
    import numpy as np
    import torch
    import cv2
    import kornia
    import loc_inference as li                                      # GSplatLoc loc_inference.py
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import GaussianModel
    from utils.general_utils import safe_state
    patches = apply_reader_patches(a)
    patch_function(li, "localize_set", [P3, P4])
    patches += [P3[1].strip(), P4[1].strip()]
    li._USGLOC_RECORDS = {}
    collected = {"prior": ([], []), "warp": ([], [])}
    real_log_errors = li.log_errors
    li.log_errors = lambda mp, name, r, t, tag: (collected[tag][0].extend(r), collected[tag][1].extend(t))
    # == loc_inference.py __main__, with our paths
    parser = argparse.ArgumentParser(description="Testing script parameters")
    model, pipeline = ModelParams(parser, sentinel=True), PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=1_000, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    argv = ["-m", str(p["model"]), "--iteration", str(a.iterations), "--top_k", str(a.top_k),
            "--ransac_iters", str(a.ransac_iters), "--warp_lr", str(a.warp_lr), "--warp_iters", str(a.warp_iters)]
    sys.argv = ["loc_inference.py"] + argv
    args = get_combined_args(parser)
    print("GSplatLoc loc_inference.py args:", " ".join(argv), "| patches:", patches)
    safe_state(args.quiet)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    pipe.debug = a.rasterizer_debug
    t0 = time.time()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    views = scene.getTestCameras()
    t_load = time.time() - t0
    failures = {}
    for view in views:                                              # one view per call: a failure stays local
        try:
            li.localize_set(dataset.model_path, "test", [view], gaussians, pipe, background, args)
        except Exception as e:                                      # noqa: BLE001 - recorded, run continues
            failures[view.image_name] = f"{type(e).__name__}: {e}"
            print(f"FAILED {view.image_name}: {failures[view.image_name]}")
    t_loc = time.time() - t0 - t_load
    for tag in ("prior", "warp"):                                   # GSplatLoc's own summary files (error_logs/)
        if collected[tag][0]:
            real_log_errors(dataset.model_path, "test", *collected[tag], tag)
    tm = json.load(open(p["model"] / "usgloc_train_meta.json")) if (p["model"] / "usgloc_train_meta.json").exists() else {}
    man = json.load(open(p["frames"] / f"manifest_{a.init_model}.json"))
    run_args = {
        "method": "GSplatLoc",
        "variant": "Base = released loc_inference.py: XFeat top-k 2D-3D matching to the Gaussian descriptor "
                   "field + PnP-RANSAC (coarse), then photometric warp refinement. The paper's 'Fine' "
                   "(extra feature-based refinement) is not in the released code.",
        "paper": "Sidorov et al., GSplatLoc, IROS 2025, arXiv:2409.16502v3",
        "code": "https://github.com/be2rlab/gsplatloc", **{f"code_{k}": v for k, v in git_info(GSL_REPO).items()},
        "runtime_patches": patches, "dataset": a.dataset, "scene": a.scene,
        "pose_gt": "SfM pGT (visloc_pseudo_gt_limitations pgt/sfm), train and test",
        "init_point_cloud": f"COLMAP model {a.init_model} ({man['colmap_model']})",
        "n_train": man["n_train"], "n_queries": len(views), "image_size": man["image_size"], "focal": man["focal"],
        "principal_point": man["principal_point"], "resize": man["resize"],
        "train": {"iterations": a.iterations, "data_device": a.data_device, "features_on_cpu": a.features_on_cpu,
                  "other_params": "GSplatLoc arguments/__init__.py defaults", **{k: tm.get(k) for k in
                  ("wall_s", "n_gaussians", "peak_mem_allocated_gb", "peak_mem_reserved_gb", "gpu")}},
        "localize": {"iteration": a.iterations, "top_k": a.top_k, "ransac_iters": a.ransac_iters,
                     "warp_lr": a.warp_lr, "warp_iters": a.warp_iters, "load_s": t_load, "wall_s": t_loc,
                     "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2**30},
        "rasterizer_debug": a.rasterizer_debug,
        "pose_convention": "T world->camera (x_cam = R x_world + t), SfM-pGT world frame; R = nearest rotation "
                           "(SVD) to GSplatLoc's final quaternion matrix (rot_orth_err = its max |R R^T - I|)",
        "t_err": "camera-centre distance to our T_gt [m]", "r_err": "geodesic angle of R R_gt^T [deg]",
        "gsplatloc_internal_metric": "loc_inference prints ||t_gt - t_est|| of world->camera translations (not the "
                                     "camera centre) and its log_errors drops NaN queries; see error_logs/ in the model dir",
        "env": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                "numpy": np.__version__, "opencv": cv2.__version__, "kornia": kornia.__version__,
                "gpu": torch.cuda.get_device_name(0)},
        "model_path": str(p["model"]), "source_path": str(p["src"]), "queries_from": str(p["ours"]),
        "date": datetime.now().isoformat(),
    }
    n_bad = len(load_ours(p)) - sum(1 for e in li._USGLOC_RECORDS.values()
                                    if np.isfinite(np.asarray(e["R_est"], float)).all() and np.isfinite(np.asarray(e["t_est"], float)).all())
    out = p["out"] if n_bad <= a.max_failures else p["out"].with_suffix(".failed.json")
    m = write_results(a, p, li._USGLOC_RECORDS, failures, run_args, out)
    print(f"\nwrote {out}: {json.dumps(m)}")
    if out != p["out"]:
        sys.exit(f"{n_bad} queries failed (> --max-failures {a.max_failures}); results kept out of {p['out']}")


# ------------------------------------------------------------------------------------------------- all
def cmd_all(a):
    p = paths(a)
    p["logs"].mkdir(parents=True, exist_ok=True)
    opts = ["--init-model", a.init_model, "--iterations", str(a.iterations), "--data-device", a.data_device,
            "--top-k", str(a.top_k), "--ransac-iters", str(a.ransac_iters), "--warp-lr", str(a.warp_lr),
            "--warp-iters", str(a.warp_iters), "--out", a.out, "--max-failures", str(a.max_failures)]
    opts += ["--features-on-cpu"] * a.features_on_cpu + ["--rasterizer-debug"] * a.rasterizer_debug + ["--force"] * a.force
    for stage in ("prepare", "check", "train", "localize"):
        log = p["logs"] / f"{p['key']}_{stage}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), stage, a.dataset, a.scene] + opts
        print(f"[{datetime.now():%H:%M:%S}] {stage}: {' '.join(cmd)}  > {log}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="")).returncode
        print(f"[{datetime.now():%H:%M:%S}] {stage}: exit {rc}", flush=True)
        if rc:
            sys.exit(f"stage {stage} failed; see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "check", "train", "localize", "all"])
    ap.add_argument("dataset", choices=["7scenes", "12scenes"])
    ap.add_argument("scene", help="e.g. stairs, apt2/kitchen")
    ap.add_argument("--init-model", default="old_gt_refined", choices=["old_gt_refined", "sfm_gt"],
                    help="COLMAP model whose points3D.bin initialises the Gaussians. old_gt_refined = the "
                         "released code's choice; sfm_gt = the model the SfM pGT poses come from")
    ap.add_argument("--iterations", type=int, default=15000, help="paper: 15,000")
    ap.add_argument("--data-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--features-on-cpu", action="store_true", help="memory fallback (patch P2)")
    ap.add_argument("--rasterizer-debug", action="store_true", help="released default (slow, same numbers)")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--ransac-iters", type=int, default=20000)
    ap.add_argument("--warp-lr", type=float, default=0.001, help="paper: 0.001; code default 0.0005")
    ap.add_argument("--warp-iters", type=int, default=251)
    ap.add_argument("--out", default="results/gsplatloc_100", help="results dir (relative to the usgloc root)")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="more failed queries than this: results go to <key>.failed.json and the run exits 1")
    ap.add_argument("--force", action="store_true")
    a, rest = ap.parse_known_args()
    if rest:
        ap.error(f"unknown arguments {rest}")
    if a.command in ("prepare", "check"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # CPU-only stages never touch the GPU
    if a.command == "all":
        cmd_all(a)
    else:
        {"prepare": cmd_prepare, "check": cmd_check, "train": cmd_train, "localize": cmd_localize}[a.command](a)


if __name__ == "__main__":
    main()
