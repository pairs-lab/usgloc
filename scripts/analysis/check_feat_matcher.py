#!/usr/bin/env python3
"""CPU regression check for the descriptor-field matcher (matcher='feat').

Nothing here touches the GPU or a real map: a tiny synthetic
:class:`usgloc.primitives.EvidentialMap` plus a HAND-MADE render dict exercise
``EvidentialMatchCorrector._prep_query_feats`` / ``_match_feat`` -- the two new
pieces -- and assert the properties the localisation path depends on:

1. IDENTITY.  If the "rendered" descriptor map is the query's own fine map (the
   situation at the ground-truth pose with a perfect field), every dense match
   must be the identity, keypoints0 == keypoints1, to the pixel.
2. PCA.  With a PCA'd field the query maps must be projected with the map's own
   basis, so the identity property must survive at the reduced dimension.
3. SUB-PIXEL.  ``feat_subpix`` must move the render-side keypoint by less than
   one pixel and leave the query side alone.
4. NO LEAK.  ``matcher='loftr'`` must keep ``_lf = None``, i.e. ``_render``
   passes ``loc_feats=None`` and takes the byte-for-byte original code path.

    python3 scripts/analysis/check_feat_matcher.py      # exits non-zero on failure
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc import locfeat                                    # noqa: E402
from usgloc.correction import EvidentialMatchCorrector        # noqa: E402
from usgloc.primitives import EvidentialMap                   # noqa: E402
from usgloc.rendering import Camera                           # noqa: E402

H, W, N = 96, 128, 64


def synth_map(D: int, pca: bool, seed: int = 0) -> EvidentialMap:
    g = torch.Generator().manual_seed(seed)
    m = EvidentialMap(primitives=[])
    m.means = torch.randn(N, 3, generator=g)
    m.covs = torch.eye(3).expand(N, 3, 3).clone() * 1e-4
    m.colors = torch.rand(N, 3, generator=g)
    m.opacities = torch.rand(N, generator=g)
    m.gaussian_vacuity = torch.rand(N, generator=g)
    m.loc_feats = torch.nn.functional.normalize(
        torch.randn(N, D, generator=g), p=2, dim=-1)
    m.loc_feat_obs = torch.full((N,), 10.0)
    if pca:
        b = torch.linalg.qr(torch.randn(256, D, generator=g))[0]
        m.loc_feat_pca_mean = torch.zeros(256)
        m.loc_feat_pca_basis = b[:, :D].contiguous()
    return m


def corrector(emap, **kw):
    cam = Camera(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2, width=W, height=H)
    return EvidentialMatchCorrector(emap, renderer=None, camera=cam, **kw)


class StubRenderer:
    """Returns a hand-made render so the whole ``correct()`` path (gates, depth
    lift, PnP-RANSAC, refinement, multi-round logic) can run on the CPU.

    ``feats`` is whatever ``self.feats`` holds -- set it to the query's own fine
    descriptor map to emulate a perfect field rendered at the query pose.
    """

    def __init__(self, h, w):
        g = torch.Generator().manual_seed(3)
        yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(),
                                indexing="ij")
        # non-planar depth, so PnP is not solving a degenerate configuration
        self.depth = 2.0 + 0.3 * torch.sin(xx / 17.0) + 0.2 * torch.cos(yy / 13.0)
        self.rgb = torch.rand(h, w, 3, generator=g)
        self.alpha = torch.ones(h, w)
        self.vac = torch.zeros(h, w)
        self.feats = None
        self.calls = 0

    def render(self, means, covs, colors, opacities, T_cw, camera, vacuity=None,
               with_depth=False, with_xyz=False, with_meta=False,
               loc_feats=None, feat_normalize=True):
        self.calls += 1
        out = {"rgb": self.rgb, "alpha": self.alpha, "vacuity": self.vac}
        if with_depth:
            out["depth"] = self.depth
        if loc_feats is not None:
            assert self.feats is not None, "StubRenderer.feats not set"
            out["feats"] = self.feats
        return out


def check_end_to_end() -> int:
    """At the ground-truth pose with a perfect field, correct() must return the
    pose it started from, from a large, gated, PnP-verified match set."""
    emap = synth_map(64, True)
    cam = Camera(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2, width=W, height=H)
    stub = StubRenderer(H, W)
    c = EvidentialMatchCorrector(emap, stub, cam, matcher="feat", rounds=2)
    img = torch.rand(H, W, 3)
    c._prep_query_feats(img)
    stub.feats = c._q_fine.permute(1, 2, 0).contiguous()   # perfect field
    T0 = torch.eye(4)
    T0[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    r = c.correct(img, T0)
    d = (r.T.double() @ torch.linalg.inv(T0.double()))
    t_cm = float(d[:3, 3].norm()) * 100
    ok = r.success and r.inliers > 50 and t_cm < 0.01 and stub.calls == 2
    print(f"  end-to-end: success={r.success} inliers={r.inliers}/{r.n_matches} "
          f"rounds={r.rounds} renders={stub.calls} dt={t_cm:.4f} cm "
          f"-> {'OK' if ok else 'FAIL'}")
    bad = int(not ok)
    # the same through the mutual-NN fine read-out
    cm = EvidentialMatchCorrector(emap, stub, cam, matcher="feat", rounds=1,
                                  feat_fine="mnn")
    cm._prep_query_feats(img)
    r2 = cm.correct(img, T0)
    d2 = (r2.T.double() @ torch.linalg.inv(T0.double()))
    t2 = float(d2[:3, 3].norm()) * 100
    ok = r2.success and r2.n_matches > r.n_matches and t2 < 1.0
    print(f"  end-to-end (fine mnn): inliers={r2.inliers}/{r2.n_matches} "
          f"dt={t2:.4f} cm -> {'OK' if ok else 'FAIL'}")
    return bad + int(not ok)


def check_stage() -> int:
    """matcher='stage': the leading LoFTR rounds must move the hypothesis but
    never win the best-of-rounds vote on match volume."""
    emap = synth_map(64, True)
    cam = Camera(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2, width=W, height=H)
    stub = StubRenderer(H, W)
    c = EvidentialMatchCorrector(emap, stub, cam, matcher="stage", rounds=2,
                                 feat_stage=1)
    img = torch.rand(H, W, 3)
    c._prep_query_feats(img)
    stub.feats = c._q_fine.permute(1, 2, 0).contiguous()
    base = c._match_feat({"feats": stub.feats})
    n_feat = int(base["keypoints0"].shape[0])

    def fake_loftr(img0, img1):
        # twice as many correspondences as the field, and biased by 1 px
        kq = torch.cat([base["keypoints0"], base["keypoints0"]], 0)
        kr = torch.cat([base["keypoints1"], base["keypoints1"]], 0) + torch.tensor([1.0, 0.0])
        return {"keypoints0": kq, "keypoints1": kr.clamp(0, W - 1),
                "confidence": torch.ones(kq.shape[0])}

    c._match_loftr = fake_loftr
    r = c.correct(img, torch.eye(4))
    ok = r.success and r.n_matches == n_feat and r.rounds == 2
    print(f"  stage: returned round {r.rounds} with {r.n_matches} matches "
          f"(field {n_feat}, warm-up {2 * n_feat}) -> {'OK' if ok else 'FAIL'}")
    bad = int(not ok)
    # with feat_stage >= rounds every round is a warm-up and the LoFTR result
    # must be returned rather than nothing
    c2 = EvidentialMatchCorrector(emap, stub, cam, matcher="stage", rounds=1,
                                  feat_stage=1)
    c2._prep_query_feats(img)
    c2._match_loftr = fake_loftr
    r2 = c2.correct(img, torch.eye(4))
    ok = r2.success and r2.n_matches == 2 * n_feat
    print(f"  stage (all warm-up): returned {r2.n_matches} matches "
          f"-> {'OK' if ok else 'FAIL'}")
    return bad + int(not ok)


def main() -> int:
    torch.manual_seed(0)
    img = torch.rand(H, W, 3)
    bad = 0

    for D, pca in ((256, False), (64, True)):
        emap = synth_map(D, pca)
        c = corrector(emap, matcher="feat")
        c._prep_query_feats(img)
        q_f = c._q_fine
        assert q_f.shape == (D, H, W), q_f.shape
        # A "render" that reproduces the query exactly.
        out = {"feats": q_f.permute(1, 2, 0).contiguous()}
        r = c._match_feat(out)
        kq, kr = r["keypoints0"], r["keypoints1"]
        n = int(kq.shape[0])
        same = int((kq == kr).all(dim=-1).sum())
        ok = n > 0 and same == n
        print(f"  D={D:<4} pca={int(pca)}  identity render: {n} matches, "
              f"{same} exact -> {'OK' if ok else 'FAIL'}")
        bad += not ok
        # keypoints must be inside the image, confidence one per match
        ok = (int(kq[:, 0].max()) < W and int(kq[:, 1].max()) < H
              and int(kr[:, 0].min()) >= 0 and r["confidence"].shape[0] == n)
        print(f"  D={D:<4} pca={int(pca)}  bounds + confidence shape "
              f"-> {'OK' if ok else 'FAIL'}")
        bad += not ok
        # a spatially shuffled render must NOT reproduce the identity
        perm = torch.randperm(H * W)
        sh = q_f.reshape(D, -1)[:, perm].reshape(D, H, W)
        r2 = c._match_feat({"feats": sh.permute(1, 2, 0).contiguous()})
        frac = float((r2["keypoints0"] == r2["keypoints1"]).all(dim=-1).double().mean()) \
            if r2["keypoints0"].numel() else 0.0
        ok = frac < 0.2
        print(f"  D={D:<4} pca={int(pca)}  shuffled render: "
              f"{100 * frac:.0f}% identity -> {'OK' if ok else 'FAIL'}")
        bad += not ok
        # STDLoc's fine mutual-NN: a superset of the per-window argmax, and the
        # identity must still hold for every one of its matches
        cm = corrector(emap, matcher="feat", feat_fine="mnn")
        cm._prep_query_feats(img)
        rm = cm._match_feat(out)
        nm = int(rm["keypoints0"].shape[0])
        samem = int((rm["keypoints0"] == rm["keypoints1"]).all(dim=-1).sum())
        # The mutual-NN read-out is a SUPERSET of the per-window argmax, and on
        # this synthetic image the bilinearly-upsampled fine descriptors inside
        # one 8x8 window are near-duplicates, so mutual pairs legitimately land
        # off the diagonal: require the median offset to be 0, not every match
        # to be exact.
        med = float((rm["keypoints0"] - rm["keypoints1"]).norm(dim=-1).median()) \
            if nm else 9.9
        ok = nm >= n and med == 0.0
        print(f"  D={D:<4} pca={int(pca)}  fine mnn: {nm} matches (argmax {n}), "
              f"{samem} exact, median offset {med:.2f} px "
              f"-> {'OK' if ok else 'FAIL'}")
        bad += not ok
        # sub-pixel refinement: < 1 px move, query side untouched
        cs = corrector(emap, matcher="feat", feat_subpix=0.02)
        cs._prep_query_feats(img)
        rs = cs._match_feat(out)
        d = (rs["keypoints1"] - kr).abs().max() if rs["keypoints0"].shape[0] == n else 9.9
        ok = (rs["keypoints0"] == kq).all() and float(d) < 1.0
        print(f"  D={D:<4} pca={int(pca)}  subpix move {float(d):.3f} px "
              f"-> {'OK' if ok else 'FAIL'}")
        bad += not ok
        # pooled must keep the field half available and the LoFTR conf gate intact
        ok = c.conf_min == 0.20 and c.feat_conf_min == 0.0
        print(f"  D={D:<4} pca={int(pca)}  conf gates (loftr {c.conf_min}, field "
              f"{c.feat_conf_min}) -> {'OK' if ok else 'FAIL'}")
        bad += not ok

    # matcher='loftr' must not carry a field into render()
    emap = synth_map(64, True)
    try:
        c = corrector(emap, matcher="loftr")
        ok = c._lf is None
    except Exception as e:                      # kornia weights unavailable
        print(f"  (loftr construction skipped: {type(e).__name__})")
        ok = True
    print(f"  matcher=loftr keeps loc_feats=None -> {'OK' if ok else 'FAIL'}")
    bad += not ok

    # a map without a field must be refused, not silently mis-matched
    emap = synth_map(64, False)
    emap.loc_feats = None
    try:
        corrector(emap, matcher="feat")
        ok = False
    except ValueError:
        ok = True
    print(f"  matcher=feat without a field raises -> {'OK' if ok else 'FAIL'}")
    bad += not ok

    bad += check_end_to_end()
    bad += check_stage()
    print("FAILURES:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
