"""Build a STALE map: delete whole object instances from a trained map.

Simulates map staleness -- content that is present in the query image but has
been removed from (or moved out of) the map.  The query images are untouched,
so the only thing that changes is the map's agreement with reality.

The per-Gaussian descriptor field lives in a SEPARATE file that is indexed by
the same row order, so it must be masked with the SAME mask or Gaussian i ends
up with Gaussian j's descriptor.  This is a row SUBSET of a fixed map, not a
re-fit, so it does not hit the "field must be rebuilt per map variant" problem
(that applies to re-fitting the Gaussians, which we do not do here).

Usage:
  make_stale_map.py --map maps/chess_r1.pt --feats maps/chess_lf64.pt \
      --frac 0.12 --seed 0 --out-tag stale12
"""
import argparse, json
from pathlib import Path
import torch


def mask_dict(d, keep, n):
    """Mask every per-Gaussian tensor (dim0 == n); pass everything else through."""
    out, masked, kept = {}, [], []
    for k, v in d.items():
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[keep].clone()
            masked.append(k)
        else:
            out[k] = v
            kept.append(k)
    return out, masked, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--feats', default=None)
    ap.add_argument('--frac', type=float, required=True,
                    help='target fraction of Gaussians to delete')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-inst-frac', type=float, default=5e-4,
                    help='only delete instances at least this big (avoids dust)')
    ap.add_argument('--out-tag', required=True)
    a = ap.parse_args()

    mp = Path(a.map)
    m = torch.load(mp, map_location='cpu', weights_only=False)
    n = m['means'].shape[0]
    iid = m['instance_ids']
    cnt = torch.bincount(iid.clamp_min(0), minlength=int(iid.max()) + 1)

    # candidate instances: big enough to be a real object, not dust
    cand = torch.nonzero(cnt >= a.min_inst_frac * n).flatten()
    g = torch.Generator().manual_seed(a.seed)
    order = cand[torch.randperm(len(cand), generator=g)]

    # greedily take instances until the target fraction is reached
    removed, tot = [], 0
    for i in order.tolist():
        if tot >= a.frac * n:
            break
        removed.append(i)
        tot += int(cnt[i])
    removed_t = torch.tensor(sorted(removed))
    drop = torch.isin(iid, removed_t)
    keep = ~drop
    actual = float(drop.sum()) / n

    out_map, masked, passed = mask_dict(m, keep, n)
    out_map['stale_removed_instances'] = removed_t
    out_map['stale_frac_requested'] = a.frac
    out_map['stale_frac_actual'] = actual
    out_map['stale_seed'] = a.seed
    out_map['stale_source_map'] = str(mp)
    dst = mp.with_name(f"{mp.stem.split('_')[0]}_{a.out_tag}.pt")
    torch.save(out_map, dst)

    rec = {'map': str(dst), 'n_before': n, 'n_after': int(keep.sum()),
           'frac_requested': a.frac, 'frac_actual': round(actual, 5),
           'instances_removed': len(removed), 'seed': a.seed,
           'map_tensors_masked': masked, 'map_keys_passed_through': passed}

    if a.feats:
        fp = Path(a.feats)
        f = torch.load(fp, map_location='cpu', weights_only=False)
        out_f, fmasked, fpassed = mask_dict(f, keep, n)
        assert 'loc_feats' in fmasked, 'loc_feats was NOT masked -- row count mismatch'
        # loc_feat_n / loc_feat_base are SCALARS, so mask_dict passes them through
        # unchanged -- but attach_loc_feats() validates the map against loc_feat_n,
        # not against the tensor length, so they must be restamped for the subset.
        out_f['loc_feat_n'] = int(keep.sum())
        out_f['loc_feat_base'] = str(dst)
        assert out_f['loc_feat_n'] == out_f['loc_feats'].shape[0]
        fdst = fp.with_name(f"{fp.stem.split('_')[0]}_{a.out_tag}_lf64.pt")
        torch.save(out_f, fdst)
        rec.update({'feats': str(fdst), 'feat_tensors_masked': fmasked,
                    'feat_keys_passed_through': fpassed})

    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
