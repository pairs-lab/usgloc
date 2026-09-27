#!/usr/bin/env python3
"""PnP solver sweep, adopting the settings GS-CPR and STDLoc actually use."""
import os, sys, itertools
sys.path.insert(0,"/home/thanhnc/gauss_ws/src/usgloc"); os.chdir("/home/thanhnc/gauss_ws/src/usgloc")
import torch, numpy as np
from usgloc.datasets import SevenScenes, load_gaussian_map
from usgloc.rendering import GsplatRenderer
from usgloc.correction import EvidentialMatchCorrector, _to_gray
from usgloc import se3
dev="cuda"; PGT="data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
def er(Ta,Tb):
    d=Ta.double()@torch.linalg.inv(Tb.double()); return float(d[:3,3].norm())*100
def perturb(T,tm,rd,seed):
    g=torch.Generator().manual_seed(seed)
    dt=torch.randn(3,generator=g); dt=dt/dt.norm()*tm
    dr=torch.randn(3,generator=g); dr=dr/dr.norm()*(rd*3.14159265/180)
    return se3.apply_tangent(T.to(dev,torch.float32),torch.cat([dt,dr]).to(dev,torch.float32))
r=GsplatRenderer()
for scene in sys.argv[1:] or ["chess"]:
    ds=SevenScenes("data/7scenes_source",scene,split="test",sfm_pose_dir=PGT)
    idx=torch.linspace(0,len(ds)-1,40).round().long().unique().tolist()
    emap=load_gaussian_map(f"maps/{scene}_sel.pt")
    for f in ("means","covs","colors","opacities","gaussian_vacuity"):
        setattr(emap,f,getattr(emap,f).to(dev))
    cam=ds.scene_camera(640,480)
    grays={i:_to_gray(ds.load_image(i).to(dev)) for i in idx}
    gts={i:ds[i].pose_cw.to(dev,torch.float32) for i in idx}
    print(f"\n=== {scene} ({len(idx)} frames)   [ours today: cv2_iter, 2.0px, no guess]")
    print(f"{'solver':<12}{'reproj':>7}{'guess':>7} | {'floor':>7}{'ok':>6} | {'from 20cm':>10}{'ok':>6}")
    for solver,reproj,guess in itertools.product(["cv2_iter","cv2_epnp","poselib"],[2.0,1.0],[False,True]):
        if solver=="poselib" and guess: continue
        try:
            c=EvidentialMatchCorrector(emap,r,cam,rounds=5,reproj_thresh_px=reproj,
                                       solver=solver,use_guess=guess)
        except Exception as e:
            print(f"{solver:<12}{reproj:7.1f}{str(guess):>7} | ERROR {e}"); continue
        fl,p20=[],[]
        for i in idx:
            a=c.correct(grays[i],gts[i])
            if a.success: fl.append(er(a.T,gts[i]))
            b=c.correct(grays[i],perturb(gts[i],0.20,10,7*i))
            if b.success: p20.append(er(b.T,gts[i]))
        f1=np.median(fl) if fl else float('nan'); f2=np.median(p20) if p20 else float('nan')
        print(f"{solver:<12}{reproj:7.1f}{str(guess):>7} | {f1:7.3f}{100*len(fl)/len(idx):5.0f}% | "
              f"{f2:10.3f}{100*len(p20)/len(idx):5.0f}%")
