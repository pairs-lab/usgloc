#!/usr/bin/env python3
"""Join the per-spec floor JSONs of a solver sweep into one table.

exp_floor_matcher.py writes one floor_<scene>_<tag>.json per (scene, spec,
start condition).  This collects them, pairs each spec's from-GT run with its
perturbed-start run, and reports the per-frame better/worse split against a
baseline spec (default: the plain ``loftr`` cell of the same scene and start
condition), which is what tells a real win from a shifted median.

    python3 scripts/analysis/solver_table.py results/solver_sweep
    python3 scripts/analysis/solver_table.py results/solver_sweep --base loftr
"""
import argparse
import json
from pathlib import Path


def load(d: Path):
    """-> {(scene, spec, pert): summary}, pert in {'gt', 'pert'}"""
    out = {}
    for f in sorted(d.glob("floor_*.json")):
        s = json.load(open(f))
        pert = "gt" if s.get("perturb") is None else "pert"
        out[(s["scene"], s["matcher"], pert)] = s
    return out


def split(a, b, tol=0.0):
    """Per-frame better/worse/equal of run a vs run b (translation error, cm).

    A frame only counts when both runs registered it; frames one run registers
    and the other does not are counted separately -- losing a frame is not a
    'worse' error, it is a failure.
    """
    fa = {r["frame"]: r for r in a["frames"]}
    fb = {r["frame"]: r for r in b["frames"]}
    better = worse = same = 0
    only_a = only_b = 0
    for k in sorted(set(fa) | set(fb)):
        ra, rb = fa.get(k), fb.get(k)
        oka = ra is not None and ra["success"]
        okb = rb is not None and rb["success"]
        if oka and not okb:
            only_a += 1
        elif okb and not oka:
            only_b += 1
        elif oka and okb:
            d = ra["t_cm"] - rb["t_cm"]
            if d < -tol:
                better += 1
            elif d > tol:
                worse += 1
            else:
                same += 1
    return better, worse, same, only_a, only_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base", default="loftr", help="baseline spec for the frame split")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    runs = {}
    for d in args.dirs:
        runs.update(load(Path(d)))
    scenes = args.scenes or sorted({k[0] for k in runs})
    specs = sorted({k[1] for k in runs})

    hdr = (f"{'spec':<44}{'GTcm':>8}{'GTdeg':>7}{'reg':>5}{'inl':>7}"
           f"{'B/W':>9}{'Pcm':>8}{'Pdeg':>7}{'reg':>5}{'inl':>7}{'B/W':>9}{'s':>6}")
    for sc in scenes:
        print(f"\n=== {sc}   (base = {args.base})")
        print(hdr)
        rows = []
        for sp in specs:
            g = runs.get((sc, sp, "gt"))
            p = runs.get((sc, sp, "pert"))
            if g is None and p is None:
                continue
            bg = runs.get((sc, args.base, "gt"))
            bp = runs.get((sc, args.base, "pert"))
            def cells(r, b):
                if r is None:
                    return ("     -  ", "     - ", "   - ", "     - ", "      - ")
                s = ""
                if b is not None and b is not r:
                    bw = split(r, b)
                    s = f"{bw[0]}/{bw[1]}" + (f"+{bw[3]}-{bw[4]}" if bw[3] or bw[4] else "")
                return (f"{r['floor_cm']:8.3f}", f"{r['floor_deg']:7.3f}",
                        f"{100*r['reg_rate']:4.0f}%", f"{r['median_inliers']:7.0f}",
                        f"{s:>9}")
            cg, cp = cells(g, bg), cells(p, bp)
            sec = (g or p)["sec_median"]
            rows.append((g["floor_cm"] if g else 9e9, sp,
                         f"{sp:<44}" + "".join(cg) + "".join(cp) + f"{sec:6.2f}"))
        for _, _, line in sorted(rows):
            print(line)
        g = runs.get((sc, args.base, "gt"))
        if g and g.get("psnr_gt"):
            print(f"    map {g['map']}  {g['n_gaussians']} gaussians  "
                  f"PSNR@GT {g['psnr_gt']:.2f} dB  {g['n_frames']} frames")

    # scene-mean ranking over the specs measured on every scene
    print("\n=== mean over scenes measured in every cell")
    print(f"{'spec':<44}{'GTcm':>8}{'Pcm':>8}{'GTdeg':>8}{'Pdeg':>8}")
    agg = []
    for sp in specs:
        gs = [runs[(sc, sp, "gt")] for sc in scenes if (sc, sp, "gt") in runs]
        ps = [runs[(sc, sp, "pert")] for sc in scenes if (sc, sp, "pert") in runs]
        if len(gs) != len(scenes):
            continue
        mg = sum(r["floor_cm"] for r in gs) / len(gs)
        mgd = sum(r["floor_deg"] for r in gs) / len(gs)
        mp = sum(r["floor_cm"] for r in ps) / len(ps) if len(ps) == len(scenes) else float("nan")
        mpd = sum(r["floor_deg"] for r in ps) / len(ps) if len(ps) == len(scenes) else float("nan")
        agg.append((mp if mp == mp else mg, f"{sp:<44}{mg:8.3f}{mp:8.3f}{mgd:8.3f}{mpd:8.3f}"))
    for _, line in sorted(agg):
        print(line)


if __name__ == "__main__":
    main()
