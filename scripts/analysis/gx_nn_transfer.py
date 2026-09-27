#!/usr/bin/env python3
"""Copy a donor map's per-Gaussian distilled semantics onto a candidate map
whose Gaussian set changed (densification), by nearest neighbour in 3D.

MEASUREMENT SCAFFOLD ONLY.  A map patched this way renders and corrects, so
the correction floor can be measured, but its CLIP features / beliefs /
vacuities / instance ids are the donor's, not this Gaussian set's.  To use
such a map end to end the distillation has to be re-run (train_map.py stages
4-5, then reapply_head.py and train_encoders.py).

    python3 scripts/analysis/gx_nn_transfer.py --map maps/chess_gxd1.pt \
        --donor maps/chess_ft.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

SEM_KEYS = ("clip_feats", "beliefs", "vacuities", "instance_ids", "obs_count")


def nn_index(query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour index of every ``query`` row in ``ref`` (CPU KD-tree:
    exact, and far cheaper than a chunked cdist for millions of queries)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(ref.double().numpy())
    _, idx = tree.query(query.double().numpy(), k=1, workers=-1)
    return torch.as_tensor(idx, dtype=torch.long)


def transfer(map_path: Path, donor_path: Path, half_clip_over: int = 0) -> dict:
    dst = torch.load(map_path, map_location="cpu", weights_only=False)
    src = torch.load(donor_path, map_location="cpu", weights_only=False)
    n_dst = int(dst["means"].shape[0])
    n_src = int(src["means"].shape[0])
    t0 = time.time()
    idx = nn_index(dst["means"], src["means"])
    moved = []
    for k in SEM_KEYS:
        v = src.get(k)
        if torch.is_tensor(v) and v.shape[0] == n_src:
            w = v[idx].contiguous()
            if (k == "clip_feats" and half_clip_over and n_dst > half_clip_over):
                w = w.half()          # disk economy on very large candidates
            dst[k] = w
            moved.append(k)
    dst["semantics_nn_transferred"] = True
    dst["semantics_donor"] = str(donor_path)
    dst.pop("semantics_invalid", None)
    torch.save(dst, map_path)
    return dict(moved=moved, n_src=n_src, n_dst=n_dst, sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--half-clip-over", type=int, default=0,
                    help="store clip_feats in float16 when the candidate has more "
                         "than this many Gaussians (0 = always float32)")
    args = ap.parse_args()
    info = transfer(Path(args.map), Path(args.donor), args.half_clip_over)
    print(f"nn-transfer {args.donor} -> {args.map}: {info}")


if __name__ == "__main__":
    main()
