#!/usr/bin/env python3
"""PoseNet baseline, retrained by us per scene under the SfM pseudo-GT, tested on our 100 queries.

No released PoseNet model exists for 12Scenes (or under the SfM pseudo-GT), so the tables' PoseNet rows
come from this. Recipe: the PoseNet of MapNet (Brahmbhatt et al., CVPR 2018), the usual PyTorch
re-implementation of Kendall et al.: ImageNet ResNet34, global average pool, FC 2048 + ReLU + dropout 0.5,
heads for position and log-quaternion; L1 losses weighted by learned beta/gamma (Kendall & Cipolla, CVPR
2017; beta0 = 0, gamma0 = -3); Adam lr 1e-4, weight decay 5e-4, batch 64, 300 epochs; images resized to
height 256, colour jitter 0.7/0.7/0.7/0.5 when training. Positions are centred on the training mean.
No epoch or parameter is picked on the test queries: the last epoch is evaluated once.

    python scripts/analysis/train_posenet.py --dataset 12scenes --scene apt1/kitchen
        # -> results/posenet_ours_100/12scenes_apt1_kitchen.json
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PGT = ROOT / "data/visloc_pseudo_gt_limitations/pgt/sfm"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_pgt(fn):  # name -> cam-to-world (4, 4)
    out = {}
    for line in open(fn):
        s = line.split()
        if len(s) < 8:
            continue
        q = np.array(s[1:5], float); T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = np.array(s[5:8], float)
        out[s[0]] = np.linalg.inv(T)
    return out


def qlog(R):  # rotation matrices (n, 3, 3) -> log quaternion (n, 3), MapNet's convention (w >= 0)
    q = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    q *= np.sign(q[:, :1] + 1e-12)
    n = np.linalg.norm(q[:, 1:], axis=1, keepdims=True)
    return np.where(n > 1e-8, q[:, 1:] / np.maximum(n, 1e-12) * np.arccos(np.clip(q[:, :1], -1, 1)), 0.0)


def qexp(v):  # log quaternion (n, 3) -> rotation matrices (n, 3, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    q = np.concatenate([np.cos(n), np.sinc(n / np.pi) * v], axis=1)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def load_images(files, h=256):
    ims = []
    for f in files:
        im = Image.open(f).convert("RGB")
        w = round(im.width * h / im.height)
        ims.append(torch.from_numpy(np.asarray(im.resize((w, h), Image.BILINEAR))).permute(2, 0, 1))
    return torch.stack(ims)                                   # (n, 3, h, w) uint8


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        r.fc = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.net, self.xyz, self.wpqr = r, nn.Linear(2048, 3), nn.Linear(2048, 3)
        for m in (r.fc[0], self.xyz, self.wpqr):
            nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
        self.beta, self.gamma = nn.Parameter(torch.zeros(())), nn.Parameter(torch.full((), -3.0))

    def forward(self, x):
        f = self.net(x)
        return self.xyz(f), self.wpqr(f)


def jitter(x, g):  # per-sample colour jitter 0.7/0.7/0.7/0.5 on a float (b, 3, h, w) batch in [0, 1]
    b = x.shape[0]
    u = lambda lo, hi: torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi, generator=g)
    x = x * u(0.3, 1.7)
    gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).clamp(0, 1)
    x = (x - gray.mean((2, 3), keepdim=True)) * u(0.3, 1.7) + gray.mean((2, 3), keepdim=True)
    x = x.clamp(0, 1); gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]
    x = ((x - gray) * u(0.3, 1.7) + gray).clamp(0, 1)
    hue = torch.empty(b, device=x.device).uniform_(-0.5, 0.5, generator=g)
    return torch.stack([TF.adjust_hue(x[i], float(hue[i])) for i in range(b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["7scenes", "12scenes"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=0, help="debug: use only the first n training frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/posenet_ours_100")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, f = a.dataset, a.scene.replace("/", "_")
    ext = "png" if ds == "7scenes" else "jpg"
    src = ROOT / f"data/{ds}_source/{a.scene}"
    tr = read_pgt(PGT / f"{ds}/{f}_train.txt"); te = read_pgt(PGT / f"{ds}/{f}_test.txt")
    # the same 100 queries as every other row (12Scenes: the full list, incl. the one our run skips)
    ref = (ROOT / f"results/v2_7scenes_seed0/7scenes_{f}.json" if ds == "7scenes"
           else ROOT / f"results/ace_dsacstar_12scenes_100/12scenes_{f}.json")
    q_names = [f"{r['seq']}/frame-{r['frame_id']:06d}.color.{ext}" for r in json.load(open(ref))["records"]]
    tr_names = sorted(tr)[: a.max_train or None]
    t0 = time.time()
    X = load_images([src / n for n in tr_names]); Xq = load_images([src / n for n in q_names])
    G = np.stack([tr[n] for n in tr_names]); Gq = np.stack([te[n] for n in q_names])
    t_mean = G[:, :3, 3].mean(0)
    Y = torch.tensor(np.concatenate([G[:, :3, 3] - t_mean, qlog(G[:, :3, :3])], 1), dtype=torch.float32)
    print(f"{ds} {a.scene}: {len(X)} train, {len(Xq)} queries, images {tuple(X.shape[2:])}, "
          f"load {time.time() - t0:.0f}s", flush=True)

    net = PoseNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    g = torch.Generator(device=dev).manual_seed(a.seed)
    X, Y, mean, std = X.to(dev), Y.to(dev), MEAN.to(dev), STD.to(dev)
    n, steps = len(X), 0
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n, device=dev, generator=g); tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            x = (jitter(X[idx].float() / 255, g) - mean) / std; y = Y[idx]
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                pt, pq = net(x)
            lt = F.l1_loss(pt.float(), y[:, :3]); lq = F.l1_loss(pq.float(), y[:, 3:])
            L = lt * torch.exp(-net.beta) + net.beta + lq * torch.exp(-net.gamma) + net.gamma
            opt.zero_grad(set_to_none=True); scaler.scale(L).backward(); scaler.step(opt); scaler.update()
            tot += float(L.detach()) * len(idx); steps += 1
        if ep % 25 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep}/{a.epochs}  loss {tot / n:.4f}  beta {float(net.beta):.2f} "
                  f"gamma {float(net.gamma):.2f}  {time.time() - t0:.0f}s", flush=True)

    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(Xq), 50):
            x = (Xq[i:i + 50].to(dev).float() / 255 - mean) / std
            pt, pq = net(x); P.append(torch.cat([pt, pq], 1).float().cpu())
    P = torch.cat(P).numpy().astype(float)
    Rq, tq = qexp(P[:, 3:]), P[:, :3] + t_mean
    recs = []
    for k, nme in enumerate(q_names):
        T = np.eye(4); T[:3, :3], T[:3, 3] = Rq[k], tq[k]
        terr = float(np.linalg.norm(Gq[k, :3, 3] - tq[k]))
        rerr = math.degrees(Rotation.from_matrix(Rq[k] @ Gq[k, :3, :3].T).magnitude())
        seq, fr = nme.split("/"); recs.append({"seq": seq, "frame_id": int(fr[6:12]), "T": T.tolist(),
                                               "t_err": terr, "r_err": rerr})
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": ds, "scene": a.scene, "n": len(recs), "n_train": n, "epochs": a.epochs, "steps": steps,
               "frames_from": str(ref), "seconds": time.time() - t0, "records": recs},
              open(out / f"{ds}_{f}.json", "w"))
    t = np.array([100 * r["t_err"] for r in recs]); r = np.array([r["r_err"] for r in recs])
    print(f"{ds} {a.scene}: n={len(recs)} {time.time() - t0:.0f}s | median {np.median(t):.2f}/{np.median(r):.2f}"
          f" SR5 {100 * np.mean((t < 5) & (r < 5)):.0f} SR2 {100 * np.mean((t < 2) & (r < 2)):.0f}", flush=True)


if __name__ == "__main__":
    main()
