#!/usr/bin/env python3
"""Fig. 5: qualitative comparison against the Table I baselines, identical queries.

Baselines are the Table I methods with per-frame poses on this machine: ACE
(authors' models + their dsacstar solver) and Marepo. Ours is the reported
configuration (results/m3u8_seed0), 100 frames per scene. One row per scene:
the query with the largest margin over both baselines among our sub-centimetre
results (rule in pick()), so these are favourable examples by design.

    python3 make_fig_qual3.py                 # render panels (GPU) + compose
    python3 make_fig_qual3.py --compose-only  # re-lay out the cached panels (CPU)
"""
import json, glob, os, re, sys, functools
print = functools.partial(print, flush=True)
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.rendering import GsplatRenderer                  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SC = ["chess", "fire", "office", "redkitchen", "stairs"]
OUT = Path("paper/figures/fig_qual3")

def gather():
    rows = []
    for s in SC:
        o = json.load(open(f"results/m3u8_seed0/7scenes_{s}.json")); tau = o["tau"]
        a = {(r["seq"], r["frame_id"]): r
             for r in json.load(open(f"results/ace_dsacstar_100/7scenes_{s}.json"))["records"]}
        mp = {(r["seq"], r["frame_id"]): r
              for r in json.load(open(f"results/marepo_100/7scenes_{s}.json"))["records"]}
        for r in o["records"]:
            k = (r["seq"], r["frame_id"])
            if "T" not in r or k not in a or "T" not in a[k] or k not in mp or "T" not in mp[k]:
                continue
            rows.append(dict(
                scene=s, seq=k[0], frame=k[1],
                e_ours=100*r["t_err"], sig=100*r["sigma_t"]*tau**0.5,
                e_ace=100*a[k]["t_err"], e_mp=100*mp[k]["t_err"],
                T_ours=r["T"], T_ace=a[k]["T"], T_gt=r["T_gt"], T_mp=mp[k]["T"]))
    sig = np.array([r["sig"] for r in rows])
    for r in rows:
        r["sig_pct"] = 100*float(np.mean(sig <= r["sig"]))
    return rows

def pick(rows):
    """Our best cases: largest margin over BOTH baselines, one row per scene.

    A case qualifies only if USGLoc is under 1 cm (so it is genuinely good, not
    merely relatively good) and its reported sigma stays within 20x the error
    (so the panel does not advertise a wildly over-conservative covariance).
    These are favourable examples chosen on purpose; the tables carry the
    distribution.
    """
    ok = [r for r in rows
          if r["e_ours"] < 1.0 and r["sig"] <= 20.0 * max(r["e_ours"], 0.05)]
    ok.sort(key=lambda r: -min(r["e_ace"], r["e_mp"]) / max(r["e_ours"], 1e-6))
    sel, seen = [], set()
    for r in ok:
        if r["scene"] in seen:
            continue
        seen.add(r["scene"])
        r["why"] = ""          # per-panel xlabels already carry the numbers
        sel.append(r)
        if len(sel) == 4:
            break
    return sel


def look_at(eye, tgt, up):
    f = np.asarray(tgt,float)-np.asarray(eye,float); f/=np.linalg.norm(f)
    r = np.cross(up,f); r/=np.linalg.norm(r)
    return np.stack([r, np.cross(r,f), f])


def centre(T):
    T=np.asarray(T,float); return -T[:3,:3].T@T[:3,3]


def mapview_cam(Cs, W=900, H=760):
    """Free camera framed on the estimates (a pure function of the poses)."""
    UP = np.array([0., -1., 0.])
    tgt = np.mean(Cs, 0)
    a, e = np.radians(295.0), np.radians(42.0)
    eye = tgt + 1.9*np.array([np.cos(a)*np.cos(e), -np.sin(e), np.sin(a)*np.cos(e)])
    return dict(eye=eye, R=look_at(eye, tgt, UP), fx=0.70*W, cx=0.5*W, cy=0.5*H, W=W, H=H)


def mapview(scene, emap, rend, Cs):
    """Free-camera render of the scene map, framed on the estimates."""
    from usgloc.rendering import Camera
    import torch as _t
    cam = mapview_cam(Cs)
    T = np.eye(4); T[:3,:3] = cam["R"]; T[:3,3] = -cam["R"]@cam["eye"]
    mu = emap.means.detach().cpu().numpy()
    op = emap.opacities.detach().cpu().numpy()
    lo, hi = np.min(Cs,0)-1.5, np.max(Cs,0)+1.5
    keep = _t.from_numpy(np.all(mu>lo,1) & np.all(mu<hi,1) & (op>0.12)).to(emap.means.device)
    with _t.no_grad():
        rgb = rend.render(emap.means[keep], emap.covs[keep], emap.colors[keep],
                          emap.opacities[keep],
                          _t.tensor(T, dtype=_t.float32, device=emap.means.device),
                          Camera(fx=cam["fx"], fy=cam["fx"], cx=cam["cx"], cy=cam["cy"],
                                 width=cam["W"], height=cam["H"])
                          )["rgb"].clamp(0,1).cpu().numpy()
    return rgb, cam


def proj(P, cam):
    pc = (np.atleast_2d(P)-cam["eye"]) @ cam["R"].T
    z = np.maximum(pc[:,2], 1e-6)
    return cam["fx"]*pc[:,0]/z + cam["cx"], cam["fx"]*pc[:,1]/z + cam["cy"]


def render(sel):
    """GPU: render every panel of the selected rows into OUT/ as PNG."""
    from PIL import Image
    dev, rend, cache = "cuda", GsplatRenderer(), {}
    for i, r in enumerate(sel):
        s = r["scene"]
        if s not in cache:
            ds = SevenScenes("data/7scenes_source", s, split="test", sfm_pose_dir=PGT)
            emap = load_gaussian_map(f"maps/{s}_sel.pt")
            for fl in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
                setattr(emap, fl, getattr(emap, fl).to(dev))
            cache[s] = (ds, emap, {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))})
        ds, emap, idx = cache[s]
        img = ds.load_image(idx[(r["seq"], r["frame"])])
        if img.shape[0] != 480:
            img = torch.nn.functional.interpolate(img.permute(2,0,1)[None], size=(480,640),
                     mode="bilinear", align_corners=False)[0].permute(1,2,0)
        outs = {"query": img.numpy()}
        with torch.no_grad():
            for key in ("T_ours", "T_ace", "T_mp", "T_gt"):
                T = torch.tensor(r[key], dtype=torch.float32, device=dev)
                outs[key] = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                        T, ds.scene_camera(640,480),
                                        vacuity=emap.gaussian_vacuity)["rgb"].clamp(0,1).cpu().numpy()
        Cs = np.array([centre(r[k]) for k in ("T_ours","T_ace","T_mp","T_gt")])
        outs["mapview"] = mapview(s, emap, rend, Cs)[0]
        for key, im in outs.items():
            Image.fromarray((im*255).astype("uint8")).save(OUT/f"{i}_{key}.png")


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"


def compose(sel, out="paper/figures/fig_qual3.pdf"):
    """CPU: exact grid, every panel 4:3 and the same size, Times like the paper."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    C = dict(T_ours="#1B9E77", T_ace="#7570B3", T_mp="#555555")   # = Fig. 4 colours
    cols = [("query", "query"), ("T_gt", "ground truth"), ("T_ace", "ACE"),
            ("T_mp", "Marepo"), ("T_ours", r"$\mathbf{USGLoc}$ (ours)"), ("mapview", "3DGS map view")]
    W, L, R, TOP, BOT = 7.16, 0.15, 0.01, 0.17, 0.01      # inches; W = IEEE text width
    g, gm, gr = 0.022, 0.07, 0.022                        # column gap, gap before map, row gap
    w = (W - L - R - 4*g - gm) / 6; h = 0.75*w
    H = TOP + len(sel)*h + (len(sel)-1)*gr + BOT
    fig = plt.figure(figsize=(W, H))
    for i, r in enumerate(sel):
        y = H - TOP - (i+1)*h - i*gr
        for j, (key, title) in enumerate(cols):
            x = L + j*(w+g) + (gm - g if j == 5 else 0)
            ax = fig.add_axes([x/W, y/H, w/W, h/H]); ax.set_axis_off()
            im = np.asarray(Image.open(OUT/f"{i}_{key}.png"))
            if key == "mapview":                           # 900x760 -> 4:3, centred on the poses
                cam = mapview_cam(np.array([centre(r[k]) for k in ("T_ours","T_ace","T_mp","T_gt")]))
                ch = int(round(im.shape[1]*0.75))
                gu, gv = proj(centre(r["T_gt"]), cam)
                v0 = int(np.clip(gv[0] - ch/2, 0, im.shape[0]-ch))
                ax.imshow(im[v0:v0+ch], interpolation="lanczos")
                for k, mk in (("T_ace", "s"), ("T_mp", "D"), ("T_ours", "^")):
                    cu, cv = proj(centre(r[k]), cam)
                    ax.plot(cu, cv-v0, mk, color=C[k], ms=5.0, mec="white", mew=0.7, zorder=5)
                ax.plot(gu, gv-v0, "o", mfc="none", mec="white", mew=1.6, ms=10.5, zorder=6)
                ax.plot(gu, gv-v0, "o", mfc="none", mec="black", mew=0.5, ms=12.2, zorder=6)
                ax.set_xlim(0, im.shape[1]); ax.set_ylim(ch, 0)
            else:
                ax.imshow(im, interpolation="lanczos")
            if i == 0:
                ax.text(0.5, 1.0 + 0.04/h, title, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8)
            if key in C:
                txt = f"{r['e_' + {'T_ours':'ours','T_ace':'ace','T_mp':'mp'}[key]]:.2f} cm"
                if key == "T_ours":
                    txt += rf"$\;\cdot\;\sigma_t$ {ordinal(int(round(r['sig_pct'])))} pct"
                ax.text(0.03, 0.045, txt, transform=ax.transAxes, fontsize=6.8, color="white",
                        va="bottom", ha="left", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc=C[key], alpha=0.92, lw=0))
        fig.text((L-0.02)/W, (y+h/2)/H, r["scene"], rotation=90, ha="right", va="center", fontsize=7.5)
    fig.savefig(out, dpi=300)
    print(f"saved {out}  ({W:.2f} x {H:.2f} in)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = gather(); sel = pick(rows)
    print(f"{len(rows)} frames with all three poses")
    for r in sel:
        print(f"  {r['scene']:11s} {r['seq']}/{r['frame']:06d}  ours {r['e_ours']:6.2f} cm"
              f" (sig pct {r['sig_pct']:3.0f})  ACE {r['e_ace']:6.2f}  Marepo {r['e_mp']:6.2f}")
    if "--compose-only" not in sys.argv:
        render(sel)
    compose(sel)
    json.dump([{k:v for k,v in r.items() if not k.startswith("T_")} for r in sel],
              open(OUT/"selection.json","w"), indent=1)


if __name__ == "__main__":
    main()
