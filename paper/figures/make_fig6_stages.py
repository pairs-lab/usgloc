#!/usr/bin/env python3
"""Fig. 6: the stages on two Office test queries, with a third-person map view.

Panels come from a re-run of the LoFTR configuration (results/final_seed0 args)
with --save-visuals, which also dumps T0 (retrieved anchor of the reported
hypothesis) and the reported pose. Rows by a disclosed rule: final error < 1 cm,
largest coarse error, one row per sequence -- favourable by design.

    python3 paper/figures/make_fig6_stages.py <vis_dir>                 # render map views (GPU) + compose
    python3 paper/figures/make_fig6_stages.py <vis_dir> --compose-only  # CPU
"""
import glob, json, os, shutil, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
SCENE, MAP = "office", "maps/office_ft.pt"
PGT = f"data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes/{SCENE}_test.txt"
OUT = Path("paper/figures/fig6")
C = dict(T0="#D95F02", ours="#1B9E77", gt="white")
W, H = 640, 480


def centre(T):
    T = np.asarray(T, float); return -T[:3, :3].T @ T[:3, 3]


def gt_poses():
    P = {}
    for l in open(PGT):
        p = l.split(); q = [float(x) for x in p[1:5]]
        T = np.eye(4); T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = [float(x) for x in p[5:8]]; P[p[0]] = T
    return P


def select(vis):
    G, rows = gt_poses(), []
    for f in sorted(glob.glob(f"{vis}/test_*_poses.json")):
        stem = os.path.basename(f)[:-len("_poses.json")]
        seq, fid = stem.split("_")[1], int(stem.split("frame-")[1])
        P = json.load(open(f)); Tg = G[f"{seq}/frame-{fid:06d}.color.png"]
        rows.append(dict(stem=stem, seq=seq, frame=fid, T0=P["T0"], T=P["T_mode"], T_gt=Tg.tolist(),
                         e0=100 * np.linalg.norm(centre(P["T0"]) - centre(Tg)),
                         e=100 * np.linalg.norm(centre(P["T_mode"]) - centre(Tg))))
    sel, seen = [], set()
    for r in sorted([r for r in rows if r["e"] < 1.0], key=lambda r: -r["e0"]):
        if r["seq"] not in seen:
            seen.add(r["seq"]); sel.append(r)
        if len(sel) == 2:
            break
    return sel


def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), f])


UP = np.array([0., -1., 0.])                              # world up is -y in the SfM frame


def view_cam(r):
    """Tilted top-down view (65 deg) from above and behind the true camera: no wall in the way."""
    c0, cg = centre(r["T0"]), centre(r["T_gt"])
    fwd = np.asarray(r["T_gt"])[2, :3]; fwd = fwd - np.dot(fwd, UP) * UP; fwd /= np.linalg.norm(fwd)
    tgt = 0.5 * (c0 + cg); dist = 1.1 * np.linalg.norm(c0 - cg) + 1.2
    e = np.radians(65)
    eye = tgt + dist * (np.sin(e) * UP - np.cos(e) * fwd)
    return dict(eye=eye, R=look_at(eye, tgt, fwd), fx=0.62 * W)


def proj(P, cam):
    pc = (np.atleast_2d(P) - cam["eye"]) @ cam["R"].T
    z = np.maximum(pc[:, 2], 1e-6)
    return cam["fx"] * pc[:, 0] / z + W / 2, cam["fx"] * pc[:, 1] / z + H / 2


def frustum(T, d=0.30):
    T = np.asarray(T, float); R, t = T[:3, :3], T[:3, 3]
    k = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]) / 525.0
    return np.vstack([centre(T), (np.c_[k * d, np.full(4, d)] - t) @ R])


def render_views(sel):
    import torch
    from PIL import Image
    from usgloc.datasets import load_gaussian_map
    from usgloc.rendering import GsplatRenderer, Camera
    emap = load_gaussian_map(MAP)
    for f in ("means", "covs", "colors", "opacities"):
        setattr(emap, f, getattr(emap, f).to("cuda"))
    mu = emap.means.cpu().numpy(); op = emap.opacities.cpu().numpy(); rend = GsplatRenderer()
    for i, r in enumerate(sel):
        cam = view_cam(r); tgt = 0.5 * (centre(r["T0"]) + centre(r["T_gt"]))
        top = max(np.dot(centre(r["T0"]), UP), np.dot(centre(r["T_gt"]), UP)) + 0.4   # drop the ceiling
        keep = torch.from_numpy(np.all(np.abs(mu - tgt) < 2.5, 1) & (mu @ UP < top) & (op > 0.12)).cuda()
        T = np.eye(4); T[:3, :3] = cam["R"]; T[:3, 3] = -cam["R"] @ cam["eye"]
        with torch.no_grad():
            rgb = rend.render(emap.means[keep], emap.covs[keep], emap.colors[keep], emap.opacities[keep],
                              torch.tensor(T, dtype=torch.float32, device="cuda"),
                              Camera(fx=cam["fx"], fy=cam["fx"], cx=W / 2, cy=H / 2, width=W, height=H)
                              )["rgb"].clamp(0, 1).cpu().numpy()
        Image.fromarray((rgb * 255).astype("uint8")).save(OUT / f"{i}_mapview.png")


def compose(sel, out="paper/figures/fig6_stages.pdf"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    cols = [("query", "query"), ("init", r"render at $T^0$"), ("refined", r"render at $\hat{T}$"),
            ("vacuity", r"vacuity $U_r$"), ("weight", r"weight $w$"), ("mapview", "3DGS map view")]
    Wf, L, R, TOP, BOT = 7.16, 0.15, 0.01, 0.21, 0.01
    g, gm, gr = 0.022, 0.07, 0.022
    w = (Wf - L - R - 4 * g - gm) / 6; h = 0.75 * w
    Hf = TOP + len(sel) * h + (len(sel) - 1) * gr + BOT
    fig = plt.figure(figsize=(Wf, Hf))
    for i, r in enumerate(sel):
        y = Hf - TOP - (i + 1) * h - i * gr
        for j, (key, title) in enumerate(cols):
            x = L + j * (w + g) + (gm - g if j == 5 else 0)
            ax = fig.add_axes([x / Wf, y / Hf, w / Wf, h / Hf]); ax.set_axis_off()
            im = np.asarray(Image.open(OUT / f"{i}_{key}.png").convert("RGB"))
            ax.imshow(im, interpolation="lanczos"); ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0)
            if key == "mapview":
                cam = view_cam(r)
                (u0, v0), (ug, vg) = [np.array(proj(centre(r[k]), cam)).ravel() for k in ("T0", "T_gt")]
                ax.annotate("", xy=(ug, vg), xytext=(u0, v0),
                            arrowprops=dict(arrowstyle="-|>", ls="--", lw=1.0, color="white", mutation_scale=8))
                for k, T in (("gt", r["T_gt"]), ("T0", r["T0"]), ("ours", r["T"])):
                    u, v = proj(frustum(T), cam)
                    for m in range(1, 5):
                        ax.plot([u[0], u[m]], [v[0], v[m]], color=C[k], lw=1.1)
                    ax.plot(np.r_[u[1:], u[1]], np.r_[v[1:], v[1]], color=C[k], lw=1.1)
            if i == 0:
                ax.text(0.5, 1.0 + 0.04 / h, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
            stamp = {"init": (f"{r['e0']:.1f} cm", C["T0"]), "refined": (f"{r['e']:.2f} cm", C["ours"])}.get(key)
            if stamp:
                ax.text(0.03, 0.045, stamp[0], transform=ax.transAxes, fontsize=6.8, color="white",
                        va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc=stamp[1], alpha=0.92, lw=0))
        fig.text((L - 0.02) / Wf, (y + h / 2) / Hf, f"{r['seq']}/{r['frame']}", rotation=90,
                 ha="right", va="center", fontsize=7)
    fig.savefig(out, dpi=300)
    print(f"saved {out}  ({Wf:.2f} x {Hf:.2f} in)")


def main():
    vis = sys.argv[1]; OUT.mkdir(parents=True, exist_ok=True)
    sel = select(vis)
    for i, r in enumerate(sel):
        print(f"  row {i}: {r['seq']}/{r['frame']:06d}  coarse {r['e0']:.1f} cm -> final {r['e']:.2f} cm")
        for k in ("query", "init", "refined", "vacuity", "weight"):
            shutil.copy(f"{vis}/{r['stem']}_{k}.png", OUT / f"{i}_{k}.png")
    if "--compose-only" not in sys.argv:
        render_views(sel)
    compose(sel)
    json.dump([{k: v for k, v in r.items()} for r in sel], open(OUT / "selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
