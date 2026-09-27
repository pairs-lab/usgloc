"""Per-ray depth statistics from the rasterizer's own tile/intersection lists.

The correction lifts a render-side keypoint to 3D through the rendered depth.
gsplat's ``render_mode="RGB+ED"`` gives the ALPHA-BLENDED (expected) depth,

    d_exp(p) = sum_i w_i(p) z_i / sum_i w_i(p),   w_i = alpha_i prod_{j<i}(1-alpha_j),

which mixes every Gaussian along the ray.  Two things go wrong with it as a
surface estimate:

* at a depth discontinuity it interpolates foreground and background into a
  surface that is not there;
* even on a clean surface it is biased towards the camera, because the blend
  integrates through the front half of every Gaussian before the weight mass
  is exhausted.

This module recomputes, for a handful of chosen pixels only, the *exact* same
front-to-back compositing the CUDA kernel performs -- same tile lists, same
depth order, same ``alpha < 1/255`` skip, same ``T <= 1e-4`` early exit -- and
returns per-ray depth statistics that do not average across surfaces:

``exp``     the alpha-blended depth (reproduces the ``ED`` channel; a check)
``maxw``    the depth of the single largest-weight Gaussian (the mode)
``median``  the depth at which the accumulated weight crosses half its total
``band``    the weighted mean restricted to a relative depth band around the
            median-depth surface

It needs the ``meta`` dict ``gsplat.rasterization`` already returns, so it
costs one extra pass over ~10^3 pixels and no extra rendering.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

_ALPHA_MIN = 1.0 / 255.0
_ALPHA_MAX = 0.999
_T_MIN = 1e-4

_STATS: Sequence[str] = ("exp", "maxw", "median", "band")


@torch.no_grad()
def ray_depths(
    meta: Dict,
    px: torch.Tensor,          # (M,) float, pixel-centre x (i.e. col + 0.5)
    py: torch.Tensor,          # (M,) float, pixel-centre y (i.e. row + 0.5)
    band: float = 0.02,        # relative half-width of the 'band' statistic
    chunk_pairs: int = 4_000_000,
) -> Dict[str, torch.Tensor]:
    """Exact per-ray depth statistics at the given pixel centres.

    ``px``/``py`` are the centres the rasterizer itself evaluates, so they
    must be ``col + 0.5`` / ``row + 0.5`` for integer pixels; any other value
    evaluates the same compositing at an off-grid sample of the *tile's*
    Gaussian list, which is valid as long as the sample stays inside the tile
    the index was taken from (the caller passes integer-pixel centres).

    Returns a dict of (M,) tensors: ``exp``, ``maxw``, ``median``, ``band``,
    plus ``w_sum`` (= the rendered alpha) and ``n_hit`` (contributing
    Gaussians).  Rays with no contributor get 0 everywhere.
    """
    dev = px.device
    means2d = meta["means2d"]
    if means2d.dim() == 3:                      # unpacked: (I, N, 2)
        means2d = means2d[0]
    conics = meta["conics"]
    conics = conics[0] if conics.dim() == 3 else conics
    opac = meta["opacities"]
    opac = opac[0] if opac.dim() == 2 else opac
    depths = meta["depths"]
    depths = depths[0] if depths.dim() == 2 else depths
    flat = meta["flatten_ids"].long()
    offs = meta["isect_offsets"].reshape(-1).long()     # (tile_h * tile_w,)
    tw, ts = int(meta["tile_width"]), int(meta["tile_size"])
    n_isects = int(flat.numel())

    M = px.shape[0]
    tile = (py.long().div(ts, rounding_mode="floor") * tw
            + px.long().div(ts, rounding_mode="floor"))
    start = offs[tile]
    end = torch.where(tile + 1 < offs.numel(), offs[(tile + 1).clamp_max(offs.numel() - 1)],
                      torch.full_like(start, n_isects))
    end = torch.where(tile + 1 >= offs.numel(), torch.full_like(start, n_isects), end)
    cnt = (end - start).clamp_min(0)

    out = {k: torch.zeros(M, device=dev, dtype=torch.float32) for k in _STATS}
    out["w_sum"] = torch.zeros(M, device=dev, dtype=torch.float32)
    out["n_hit"] = torch.zeros(M, device=dev, dtype=torch.float32)
    if M == 0:
        return out

    order = torch.argsort(cnt)                  # group similar tile sizes together
    pos = 0
    while pos < M:
        # grow the chunk while (rows * max_count) stays under the pair budget
        hi = pos + 1
        while hi < M:
            L = int(cnt[order[hi]].item())
            if (hi + 1 - pos) * max(L, 1) > chunk_pairs:
                break
            hi += 1
        sel = order[pos:hi]
        pos = hi
        L = int(cnt[sel].max().item())
        if L == 0:
            continue
        ar = torch.arange(L, device=dev)
        idx = start[sel][:, None] + ar[None]                     # (C, L)
        ok = ar[None] < cnt[sel][:, None]
        g = flat[idx.clamp(0, n_isects - 1)]                     # (C, L) nnz ids
        dx = means2d[g, 0] - px[sel][:, None]
        dy = means2d[g, 1] - py[sel][:, None]
        c = conics[g]
        sigma = 0.5 * (c[..., 0] * dx * dx + c[..., 2] * dy * dy) + c[..., 1] * dx * dy
        alpha = torch.clamp(opac[g] * torch.exp(-sigma), max=_ALPHA_MAX)
        live = ok & (sigma >= 0) & (alpha >= _ALPHA_MIN)
        alpha = torch.where(live, alpha, torch.zeros_like(alpha))
        # front-to-back transmittance (the list is already depth-sorted)
        one_m = 1.0 - alpha
        T_incl = torch.cumprod(one_m, dim=1)
        T_excl = T_incl / one_m.clamp_min(1e-20)
        # the kernel STOPS (exclusive) at the first Gaussian whose next_T <= 1e-4
        killed = torch.cummax((T_incl <= _T_MIN).float(), dim=1).values > 0
        w = torch.where(killed, torch.zeros_like(alpha), alpha * T_excl)
        z = depths[g]

        w_sum = w.sum(1)
        good = w_sum > 1e-8
        inv = 1.0 / w_sum.clamp_min(1e-12)
        out["w_sum"][sel] = w_sum
        out["n_hit"][sel] = (w > 0).sum(1).float()
        out["exp"][sel] = (w * z).sum(1) * inv
        j = w.argmax(1)
        out["maxw"][sel] = torch.where(good, z.gather(1, j[:, None])[:, 0],
                                       torch.zeros_like(w_sum))
        cw = torch.cumsum(w, dim=1)
        k = (cw < 0.5 * w_sum[:, None]).sum(1).clamp_max(L - 1)
        z_med = z.gather(1, k[:, None])[:, 0]
        out["median"][sel] = torch.where(good, z_med, torch.zeros_like(w_sum))
        keep = (z - z_med[:, None]).abs() <= band * z_med.clamp_min(1e-6)[:, None]
        wb = torch.where(keep, w, torch.zeros_like(w))
        sb = wb.sum(1)
        out["band"][sel] = torch.where(sb > 1e-8, (wb * z).sum(1) / sb.clamp_min(1e-12), z_med)
        del idx, g, dx, dy, c, sigma, alpha, live, one_m, T_incl, T_excl, killed, w, z, cw, wb
    return out
