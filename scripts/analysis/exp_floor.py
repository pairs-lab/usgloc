#!/usr/bin/env python3
"""Optimize the correction FLOOR: correct from ground truth and see how
close the stage can get. Isolates the correction from retrieval/refinement,
so it is fast and directly measures the accuracy ceiling of the system.
"""
import os, sys, itertools
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")
import torch
from usgloc.datasets import SevenScenes, load_gaussian_map
from usgloc.rendering import GsplatRenderer
from usgloc.correction import EvidentialMatchCorrector, _to_gray
from usgloc import se3

dev = "cuda"; PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SCENE = sys.argv[1] if len(sys.argv) > 1 else "chess"
MAP = sys.argv[2] if len(sys.argv) > 2 else f"maps/{SCENE}.pt"

def err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    c = ((d[0,0]+d[1,1]+d[2,2]-1)/2).clamp(-1,1)
    return float(d[:3,3].norm())*100, float(torch.rad2deg(torch.arccos(c)))

def perturb(T, tm, rd, seed):
    g = torch.Generator().manual_seed(seed)
    dt = torch.randn(3, generator=g); dt = dt/dt.norm()*tm
    dr = torch.randn(3, generator=g); dr = dr/dr.norm()*(rd*3.14159265/180)
    return se3.apply_tangent(T.to(dev, torch.float32),
                             torch.cat([dt, dr]).to(dev, torch.float32))

ds = SevenScenes("data/7scenes_source", SCENE, split="test", sfm_pose_dir=PGT)
emap = load_gaussian_map(MAP)
for f in ("means","covs","colors","opacities","gaussian_vacuity"):
    setattr(emap, f, getattr(emap, f).to(dev))
cam = ds.scene_camera(640, 480); r = GsplatRenderer()
idx = list(range(0, len(ds), max(1, len(ds)//25)))[:25]
grays = {i: _to_gray(ds.load_image(i).to(dev)) for i in idx}
gts = {i: ds[i].pose_cw.to(dev, torch.float32) for i in idx}

print(f"=== {SCENE} map={MAP}  {len(idx)} frames")
print(f"{'reproj':>7} {'rounds':>7} {'conf':>6} | {'floor cm':>9} {'rot':>6} {'ok':>6} | "
      f"{'from 20cm':>10} {'rot':>6} {'ok':>6}")
for reproj, rounds, conf in itertools.product([2.0, 1.0, 0.5], [3, 5], [0.15]):
    corr = EvidentialMatchCorrector(emap, r, cam, rounds=rounds,
                                    reproj_thresh_px=reproj, conf_min=conf)
    fl, fr, n_ok = [], [], 0
    p20, p20r, n20 = [], [], 0
    for i in idx:
        res = corr.correct(grays[i], gts[i])
        if res.success:
            e = err(res.T, gts[i]); fl.append(e[0]); fr.append(e[1]); n_ok += 1
        rp = corr.correct(grays[i], perturb(gts[i], 0.20, 10, 7*i))
        if rp.success:
            e = err(rp.T, gts[i]); p20.append(e[0]); p20r.append(e[1]); n20 += 1
    t = torch.tensor(fl) if fl else torch.tensor([float('nan')])
    tr = torch.tensor(fr) if fr else torch.tensor([float('nan')])
    q = torch.tensor(p20) if p20 else torch.tensor([float('nan')])
    qr = torch.tensor(p20r) if p20r else torch.tensor([float('nan')])
    print(f"{reproj:7.1f} {rounds:7d} {conf:6.2f} | {float(t.median()):9.3f} "
          f"{float(tr.median()):6.3f} {n_ok:3d}/{len(idx)} | {float(q.median()):10.3f} "
          f"{float(qr.median()):6.3f} {n20:3d}/{len(idx)}")
