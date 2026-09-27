#!/usr/bin/env python3
"""Collect the Marepo runs into the project results folder.

results/marepo_100/7scenes_<scene>.json        primary config (100 linspace frames)
results/marepo_100/<variant>/7scenes_<scene>.json   other configs, same schema
results/marepo_100/full_split_medians.json     full-split medians for every config
"""
import json
import shutil
from pathlib import Path

SP = Path("/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad")
RUNS = SP / "marepo_runs"
OUT = Path("/home/thanhnc/gauss_ws/src/usgloc/results/marepo_100")
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
PRIMARY = "pgt_marepo_acepgt"
CONFIGS = {
    "pgt_marepo_acepgt": "released scene-agnostic marepo.pt + official ACE SfM-pGT head, evaluated vs SfM pGT",
    "pgt_marepo_s_acepgt": "released per-scene marepo_s (fine-tuned on dSLAM-head maps) + official ACE SfM-pGT head, evaluated vs SfM pGT",
    "paper_protocol_marepo_s": "paper protocol: marepo_s + marepo's own dSLAM head, evaluated vs calibrated dSLAM GT (Table 1 check)",
    "paper_protocol_marepo": "paper protocol: marepo.pt + marepo's own dSLAM head, evaluated vs calibrated dSLAM GT (Table 1 check)",
    "pgt_marepo_s_dslamhead": "MISMATCH demo (chess only): marepo_s + dSLAM head evaluated vs SfM pGT",
}
PAPER_TABLE1 = {  # Chen et al. CVPR 2024, Table 1 (dSLAM GT), median cm / deg
    "marepo":   {"chess": (2.6, 1.35), "fire": (2.5, 1.42), "heads": (2.3, 2.21), "office": (3.6, 1.44),
                 "pumpkin": (4.2, 1.55), "redkitchen": (5.1, 1.99), "stairs": (6.7, 1.83)},
    "marepo_s": {"chess": (2.1, 1.24), "fire": (2.3, 1.39), "heads": (1.8, 2.03), "office": (2.8, 1.26),
                 "pumpkin": (3.5, 1.48), "redkitchen": (4.2, 1.71), "stairs": (5.6, 1.67)},
}

OUT.mkdir(parents=True, exist_ok=True)
medians = {"note": "median translation (cm) / rotation (deg); 'n100' = the 100 linspace query frames, "
                   "'full' = whole test split. Paper Table 1 values are the published dSLAM-GT numbers.",
           "configs": CONFIGS, "primary": PRIMARY, "scenes": {}}
for scene in SCENES:
    medians["scenes"][scene] = {}
    for cfg in CONFIGS:
        f = RUNS / cfg / f"7scenes_{scene}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        entry = {"n100": d["stats"], "full": d["full_split_stats"]}
        if cfg.startswith("paper_protocol"):
            reg = "marepo_s" if cfg.endswith("_s") else "marepo"
            entry["paper_table1_dslam"] = {"median_t_cm": PAPER_TABLE1[reg][scene][0],
                                           "median_r_deg": PAPER_TABLE1[reg][scene][1]}
        medians["scenes"][scene][cfg] = entry
        dst_dir = OUT if cfg == PRIMARY else OUT / cfg
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst_dir / f"7scenes_{scene}.json")
json.dump(medians, open(OUT / "full_split_medians.json", "w"), indent=1)

# console table
print(f"{'scene':11s} | {'C marepo+pgtHead n100 / full':30s} | {'D marepo_s+pgtHead n100 / full':30s} | "
      f"{'A marepo_s paper-proto full (Tab.1)':36s} | {'B marepo paper-proto full (Tab.1)':36s}")
for scene in SCENES:
    s = medians["scenes"][scene]
    def fmt(cfg, key):
        if cfg not in s: return "   --   "
        st = s[cfg][key]; return f"{st['median_t_cm']:.2f}/{st['median_r_deg']:.2f}"
    def pap(cfg):
        if cfg not in s: return ""
        p = s[cfg]["paper_table1_dslam"]; return f"({p['median_t_cm']:.1f}/{p['median_r_deg']:.2f})"
    print(f"{scene:11s} | {fmt(PRIMARY,'n100'):>13s} {fmt(PRIMARY,'full'):>13s}   | "
          f"{fmt('pgt_marepo_s_acepgt','n100'):>13s} {fmt('pgt_marepo_s_acepgt','full'):>13s}   | "
          f"{fmt('paper_protocol_marepo_s','full'):>13s} {pap('paper_protocol_marepo_s'):>12s}          | "
          f"{fmt('paper_protocol_marepo','full'):>13s} {pap('paper_protocol_marepo'):>12s}")
