"""Render, for the chosen fire query: the query image and the map at
GT / USGLoc / ACE / Marepo poses, plus a free-camera view of the fire map."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from usgloc.datasets import SevenScenes, load_gaussian_map
from usgloc.rendering import Camera, GsplatRenderer
PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SCENE, SEQ, FRAME = "fire", "seq-04", 131
OUT = ROOT/"paper/figures/fig1_src"; OUT.mkdir(exist_ok=True)

o = json.load(open(f"results/m3u8_seed0/7scenes_{SCENE}.json")); tau = o["tau"]
rec = next(r for r in o["records"] if r["seq"] == SEQ and r["frame_id"] == FRAME)
ace = next(r for r in json.load(open(f"results/ace_dsacstar_100/7scenes_{SCENE}.json"))["records"]
           if r["seq"] == SEQ and r["frame_id"] == FRAME)
mar = next(r for r in json.load(open(f"results/marepo_100/7scenes_{SCENE}.json"))["records"]
           if r["seq"] == SEQ and r["frame_id"] == FRAME)
meta = dict(ours=100*rec["t_err"], ace=100*ace["t_err"], marepo=100*mar["t_err"],
            sigma=100*rec["sigma_t"]*tau**0.5, seq=SEQ, frame=FRAME, scene=SCENE,
            T=rec["T"], T_gt=rec["T_gt"], T_ace=ace["T"], T_mar=mar["T"],
            Sigma=rec["Sigma"], tau=tau)
json.dump(meta, open(OUT/"meta.json", "w"))

ds = SevenScenes("data/7scenes_source", SCENE, split="test", sfm_pose_dir=PGT)
idx = {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))}[(SEQ, FRAME)]
img = ds.load_image(idx)
if img.shape[0] != 480:
    img = torch.nn.functional.interpolate(img.permute(2,0,1)[None], size=(480,640),
            mode="bilinear", align_corners=False)[0].permute(1,2,0)
np.save(OUT/"query.npy", img.numpy())

emap = load_gaussian_map(f"maps/{SCENE}_r1.pt")
for f in ("means","covs","colors","opacities","gaussian_vacuity"):
    setattr(emap, f, getattr(emap, f).to("cuda"))
rend, cam = GsplatRenderer(), ds.scene_camera(640, 480)
with torch.no_grad():
    for tag, T_ in [("gt", rec["T_gt"]), ("ours", rec["T"]), ("ace", ace["T"]), ("marepo", mar["T"])]:
        T = torch.tensor(T_, dtype=torch.float32, device="cuda")
        r = rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                        vacuity=emap.gaussian_vacuity)["rgb"].clamp(0,1).cpu().numpy()
        np.save(OUT/f"view_{tag}.npy", r)
        print("rendered", tag)

# free-camera view of the whole fire map
def look_at(eye, tgt, up):
    f = tgt-eye; f/=np.linalg.norm(f); r=np.cross(up,f); r/=np.linalg.norm(r)
    return np.stack([r, np.cross(r,f), f])
def centre(T):
    T=np.asarray(T,float); return -T[:3,:3].T@T[:3,3]
C = np.array([centre(r["T_gt"]) for r in o["records"]])
tgt = C.mean(0); UP = np.array([0.,-1.,0.])
a, e = np.radians(295.0), np.radians(44.0)
eye = tgt + 2.9*np.array([np.cos(a)*np.cos(e), -np.sin(e), np.sin(a)*np.cos(e)])
R = look_at(eye, tgt, UP)
W, H = 1500, 1000
pc = (C-eye)@R.T; xn, yn = pc[:,0]/pc[:,2], pc[:,1]/pc[:,2]
fx = 0.60*W/(xn.max()-xn.min())
cx = 0.50*W - fx*0.5*(xn.min()+xn.max()); cy = 0.50*H - fx*0.5*(yn.min()+yn.max())
m = torch.load(f"maps/{SCENE}_r1.pt", map_location="cpu", weights_only=False)
mu = m["means"].numpy()
smax = np.sqrt(np.clip(np.linalg.eigvalsh(m["covs"].numpy())[:,-1],0,None))
lo, hi = C.min(0), C.max(0)
box = (np.all(mu > lo-np.array([1.6,1.2,1.6]),1) & np.all(mu < hi+np.array([1.6,1.2,1.6]),1))
keep = torch.from_numpy(box & (m["opacities"].numpy() > 0.12) & (smax < 0.08))
print("bg gaussians", int(keep.sum()), "/", len(mu))
T = np.eye(4); T[:3,:3]=R; T[:3,3]=-R@eye
with torch.no_grad():
    bg = rend.render(m["means"][keep].cuda(), m["covs"][keep].cuda(), m["colors"][keep].cuda(),
                     m["opacities"][keep].cuda(), torch.tensor(T,dtype=torch.float32,device="cuda"),
                     Camera(fx=fx, fy=fx, cx=cx, cy=cy, width=W, height=H))["rgb"].clamp(0,1).cpu().numpy()
np.savez_compressed(OUT/"bg.npz", rgb=bg, eye=eye, R=R, fx=fx, cx=cx, cy=cy, W=W, H=H)
print("bg rendered")
