"""Uncertainty-weighted 2D--3D matching and the pose-hypothesis mixture
(Sec. IV-C of the draft).

The instance-level score is Eq. 6:

    s_ij = cos(d_i^G, d_j^q) * exp(-lambda_u * (u_i^G + u_j^q))

i.e. semantic similarity modulated by an evidential gate.  Scene-level
retrieval ranks submaps by descriptor similarity; the top-k submaps provide
coarse pose hypotheses with retrieval-score weights, naturally forming a pose
mixture (the SE(3) realization of the non-Gaussian error of KSE-Net).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .primitives import EvidentialMap


@dataclass
class PoseHypothesis:
    T0: torch.Tensor       # (4, 4) coarse pose (camera-from-world)
    weight: float          # retrieval weight pi_tilde_k (unnormalized)
    submap_index: int


def gated_similarity(
    desc_ref: torch.Tensor,
    desc_query: torch.Tensor,
    vac_ref: torch.Tensor,
    vac_query: torch.Tensor,
    lambda_u: float = 2.0,
) -> torch.Tensor:
    """Eq. 6 for all pairs: (M, D), (N, D), (M,), (N,) -> (M, N)."""
    cos = desc_ref @ desc_query.T  # descriptors are l2-normalized
    gate = torch.exp(-lambda_u * (vac_ref[:, None] + vac_query[None, :]))
    return cos * gate


def greedy_instance_matches(
    s: torch.Tensor, min_score: float = 0.2
) -> List[tuple]:
    """Mutually-best correspondences from a gated score matrix (M, N)."""
    matches = []
    s = s.clone()
    while True:
        idx = torch.argmax(s)
        i, j = divmod(int(idx), s.shape[1])
        if s[i, j] < min_score:
            break
        matches.append((i, j, float(s[i, j])))
        s[i, :] = -1.0
        s[:, j] = -1.0
    return matches


@torch.no_grad()
def retrieve_hypotheses(
    emap: EvidentialMap,
    query_scene_desc: torch.Tensor,
    top_k: int = 3,
    min_similarity: float = -1.0,
    candidates: Optional[Sequence[int]] = None,
) -> List[PoseHypothesis]:
    """Rank submaps by scene-descriptor similarity; return top-k hypotheses.

    Submap descriptors must have been cached (``Submap.descriptor``) by the
    pipeline after encoding the map.  Weights are softmax-normalized
    similarities; ambiguous scenes return several comparably-weighted
    hypotheses, which is exactly the multi-modality the GMM posterior keeps.

    ``candidates`` restricts the ranking to a subset of submap indices --
    the hierarchical mode, where a viewpoint-robust global shortlist (e.g.
    CLIP whole-image retrieval) proposes candidates and the evidential gated
    similarity reranks them.
    """
    if candidates is None:
        cand = list(range(len(emap.submaps)))
    else:
        cand = [int(c) for c in candidates]
    descs = torch.stack([emap.submaps[c].descriptor for c in cand])
    sim = descs @ query_scene_desc
    k = min(top_k, sim.shape[0])
    vals, idx = torch.topk(sim, k)
    keep = vals >= min_similarity
    vals, idx = vals[keep], idx[keep]
    weights = torch.softmax(vals / 0.1, dim=0)
    return [
        PoseHypothesis(
            T0=emap.submaps[cand[int(i)]].anchor_pose,
            weight=float(w),
            submap_index=cand[int(i)],
        )
        for i, w in zip(idx, weights)
    ]
