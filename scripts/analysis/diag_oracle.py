#!/usr/bin/env python3
"""How much headroom is there in hypothesis SELECTION?

For each query we run the real pipeline but record the error of EVERY
mixture component, not just the reported one. Comparing the reported error
with the best-of-k error separates two very different failures:

  reported >> best-of-k  -> the right pose IS among our hypotheses and the
                            selection criterion picks the wrong one. Fixable
                            by a better criterion.
  reported ~= best-of-k  -> every hypothesis is wrong; selection cannot help
                            and the correction itself lands in the wrong
                            place (e.g. a repeated structure).

    python3 diag_oracle.py stairs [n_queries]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.datasets import SevenScenes, load_gaussian_map        # noqa: E402
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator  # noqa: E402
from usgloc.pipeline import USGLocPipeline, PipelineConfig        # noqa: E402
from usgloc.primitives import make_submaps                        # noqa: E402
from usgloc.rendering import GsplatRenderer                       # noqa: E402
from usgloc.correction import EvidentialMatchCorrector            # noqa: E402

sys.path.insert(0, "scripts")
import importlib.util                                             # noqa: E402
spec = importlib.util.spec_from_file_location("r7", "scripts/run_7scenes.py")
r7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r7)

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def pose_err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    c = ((d[0, 0] + d[1, 1] + d[2, 2] - 1) / 2).clamp(-1, 1)
    return float(d[:3, 3].norm()) * 100, float(torch.rad2deg(torch.arccos(c)))


def main():
    scene = sys.argv[1]
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    dev = "cuda"

    tr = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
    te = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
    emap = load_gaussian_map(f"maps/{scene}_sel.pt")
    make_submaps(emap, [tr[i].pose_cw for i in range(0, len(tr), 20)], radius=3.0,
                 camera=tr.scene_camera(tr.color_width, tr.color_height))
    emap = r7.move_map(emap, dev)

    ck = torch.load(f"ckpts/{scene}_sel_encoders.pt", map_location="cpu", weights_only=True)
    clip_dim = emap.primitives[0].clip_feat.shape[0]
    desc = InstanceDescriptorNet(clip_dim=clip_dim); desc.load_state_dict(ck["descriptor"])
    agg = SceneAggregator(); agg.load_state_dict(ck["aggregator"])
    desc, agg = desc.to(dev).eval(), agg.to(dev).eval()
    front = r7.build_front_end(emap, clip_dim, ck, dev, opinion_mode="edl",
                               head_ckpt_path=None)

    cam = te.scene_camera(640, 480)
    rend = GsplatRenderer()
    corr = EvidentialMatchCorrector(emap, rend, te.scene_camera(640, 480),
                                    gamma=3.0, rounds=5)
    cfg = PipelineConfig(top_k=top_k, refine_iters=60, gamma=3.0,
                         corr_refine_iters=60, lambda_u=2.0)
    pipe = USGLocPipeline(emap, rend, cam, desc, agg, cfg, corrector=corr)
    pipe.encode_map()

    # global shortlist over the surviving submaps' anchor frames
    anchor_ids = list(range(0, len(tr), 20))
    kept, k = [], 0
    for i in anchor_ids:
        if k < len(emap.submaps) and torch.allclose(
                emap.submaps[k].anchor_pose, tr[i].pose_cw.to(dev)):
            kept.append(i); k += 1
    gf = []
    for i in kept:
        bgr_a = (tr.load_image(i).numpy() * 255).astype("uint8")[:, :, ::-1]
        gf.append(front.encode_global(bgr_a))
    anchor_globals = torch.stack(gf) if gf else None

    idx = torch.linspace(0, len(te) - 1, nq).round().long().unique().tolist()
    rep, best, nhyp, mapc = [], [], [], []
    SIG = []
    for i in idx:
        img_full = te.load_image(i)
        img = r7.resize_image(img_full, 640, 480).to(dev)
        bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
        inst = r7.move_instances(front.encode(bgr), dev)
        if len(inst) < 2:
            continue
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).clamp(0, 1)
        cand = None
        if anchor_globals is not None:
            qg = front.encode_global(bgr)
            cand = (anchor_globals @ qg).topk(
                min(10, anchor_globals.shape[0])).indices.tolist()
        res = pipe.localize(img, inst, candidates=cand, gray_full=gray)
        T_gt = te[i].pose_cw.to(dev)
        rep.append(pose_err(res.condensed.T_mode, T_gt)[0])
        errs = [pose_err(c.T_mode, T_gt)[0] for c in res.posterior.components]
        best.append(min(errs)); nhyp.append(len(errs))
        kbest = int(torch.argmax(res.posterior.weights))
        mapc.append(errs[kbest])
        SIG.append(dict(err=errs,
                        psnr=list(res.psnrs),
                        inl=[float(x) for x in res.inliers],
                        w=[float(x) for x in res.posterior.weights],
                        rw=[float(h.weight) for h in res.hypotheses]))

    rep, best = np.array(rep), np.array(best)
    print(f"\n{scene}  top_k={top_k}  n={len(rep)}  mean #components {np.mean(nhyp):.1f}")
    print(f"  reported     median {np.median(rep):7.2f} cm   >5cm {100*(rep>5).mean():3.0f}%")
    print(f"  best-of-k    median {np.median(best):7.2f} cm   >5cm {100*(best>5).mean():3.0f}%")
    mapc = np.array(mapc)
    print(f"  MAP component median {np.median(mapc):7.2f} cm   >5cm {100*(mapc>5).mean():3.0f}%"
          f"   <-- report the mode, not the moment-matched mean")
    # which available signal picks the best component?
    import itertools
    print("  --- component-selection signals (pick argmax; lower median err is better)")
    for name, key, sign in [("retrieval weight","rw",+1), ("render PSNR","psnr",+1),
                            ("PnP inliers","inl",+1), ("mixture weight","w",+1)]:
        picked=[]
        for d in SIG:
            v=d[key]
            if not v or len(v)!=len(d["err"]): continue
            j=int(np.argmax(np.array(v)*sign)); picked.append(d["err"][j])
        if picked:
            picked=np.array(picked)
            print(f"      {name:<18} median {np.median(picked):7.2f} cm   >5cm {100*(picked>5).mean():3.0f}%")
    orc=np.array([min(d["err"]) for d in SIG])
    print(f"      {'ORACLE':<18} median {np.median(orc):7.2f} cm   >5cm {100*(orc>5).mean():3.0f}%")
    _dump(SIG, scene)
    gap = rep > best * 1.5
    print(f"  SELECTION headroom: on {100*gap.mean():3.0f}% of queries a better "
          f"component existed (reported {np.median(rep[gap]) if gap.any() else float('nan'):.1f} "
          f"vs best {np.median(best[gap]) if gap.any() else float('nan'):.1f} cm)")


def _dump(SIG, scene):
    import json
    with open(f"/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad/sig_{scene}.json","w") as f:
        json.dump(SIG, f)
    print(f"  dumped {len(SIG)} queries -> sig_{scene}.json")


if __name__ == "__main__":
    main()
