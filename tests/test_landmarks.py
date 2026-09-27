"""Pose-free landmark association (usgloc.landmarks)."""

import numpy as np
import pytest
import torch

from usgloc.landmarks import (LandmarkCorrector, build_landmarks,
                              landmark_matches, mo_sample)
from usgloc.primitives import EvidentialMap
from usgloc.rendering import Camera


def _map(n=20, d=8, seed=0, vac=None, obs=None):
    g = torch.Generator().manual_seed(seed)
    feats = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    return EvidentialMap(
        primitives=[],
        means=torch.randn(n, 3, generator=g),
        covs=torch.eye(3).expand(n, 3, 3).clone(),
        colors=torch.rand(n, 3, generator=g),
        opacities=torch.rand(n, generator=g),
        gaussian_vacuity=torch.zeros(n) if vac is None else vac,
        loc_feats=feats,
        loc_feat_obs=torch.ones(n) if obs is None else obs,
    )


# ---------------------------------------------------------------- sampling

def test_mo_sample_returns_unique_indices_inside_the_candidates():
    xyz = torch.rand(200, 3)
    score = torch.rand(200)
    cand = torch.arange(0, 200, 2)
    sel = mo_sample(xyz, score, num=30, k=4, seed=0, candidates=cand)
    assert sel.numel() <= 30
    assert sel.unique().numel() == sel.numel()
    assert set(sel.tolist()).issubset(set(cand.tolist()))


def test_mo_sample_snaps_each_seed_to_the_best_scoring_neighbour():
    # Two tight clusters; inside each, exactly one point has a high score.
    xyz = torch.cat([torch.zeros(5, 3), torch.ones(5, 3) * 10.0])
    xyz += torch.linspace(0, 0.01, 10)[:, None]
    score = torch.zeros(10)
    score[2] = 1.0        # the good one in cluster A
    score[7] = 1.0        # the good one in cluster B
    sel = mo_sample(xyz, score, num=10, k=5, seed=0)
    # every seed lands on a high-score point until those are used up
    assert {2, 7}.issubset(set(sel.tolist()))
    assert sel[:2].numel() == 2


def test_build_landmarks_drops_unobserved_and_vacuous_gaussians():
    n = 20
    obs = torch.ones(n)
    obs[:5] = 0.0                       # never observed
    emap = _map(n=n, obs=obs)
    emap.loc_feats = emap.loc_feats.clone()
    emap.loc_feats[:5] = 0.0            # as the field builder leaves them
    vac = torch.zeros(n)
    vac[5:10] = 0.95
    emap.gaussian_vacuity = vac
    ls = build_landmarks(emap, sample="all")
    assert set(ls.idx.tolist()) == set(range(5, n))
    gated = build_landmarks(emap, sample="all", vac="gate", u_max=0.5)
    assert set(gated.idx.tolist()) == set(range(10, n))
    assert gated.xyz.shape == (10, 3) and gated.feat.shape[0] == 10


def test_build_landmarks_top_sampling_takes_the_best_scores():
    emap = _map(n=20)
    score = torch.arange(20.0) / 20.0
    ls = build_landmarks(emap, score=score, sample="top", num=5)
    assert sorted(ls.idx.tolist()) == [15, 16, 17, 18, 19]


# ---------------------------------------------------------------- matching

class _StubEncoder:
    """Returns a prescribed descriptor map and score map."""

    def __init__(self, desc, scores):
        self._desc, self._scores = desc, scores

    def dense_descriptors(self, img):
        return self._desc, self._scores


def _stub(d=8, H=16, W=24, peaks=((3, 4), (9, 17))):
    desc = torch.nn.functional.normalize(
        torch.randn(1, d, H // 8, W // 8, generator=torch.Generator().manual_seed(1)),
        dim=1)
    scores = torch.zeros(1, H, W)
    for (y, x) in peaks:
        scores[0, y, x] = 1.0
    return desc, scores


def test_landmark_matches_conventions_and_exactness():
    from usgloc.superpoint import sample_descriptors

    d, H, W = 8, 16, 24
    peaks = ((3, 4), (9, 17))
    desc, scores = _stub(d, H, W, peaks)
    kp = torch.tensor([[x, y] for (y, x) in peaks])
    qd = sample_descriptors(desc, kp, H, W)          # (2, d) the true answers
    # A landmark set whose descriptors are those two, plus noise landmarks.
    g = torch.Generator().manual_seed(5)
    extra = torch.nn.functional.normalize(torch.randn(6, d, generator=g), dim=-1)
    feats = torch.cat([extra[:3], qd, extra[3:]])
    xyz = torch.arange(feats.shape[0] * 3, dtype=torch.float32).reshape(-1, 3)
    from usgloc.landmarks import LandmarkSet
    ls = LandmarkSet(idx=torch.arange(feats.shape[0]), xyz=xyz, feat=feats,
                     vac=torch.zeros(feats.shape[0]),
                     score=torch.zeros(feats.shape[0]), info={})
    m = landmark_matches(torch.zeros(H, W), ls, _StubEncoder(desc, scores),
                         detect_num=8, nms=0)
    assert m["n_keypoints"] == 2
    assert m["obj"].dtype == np.float64 and m["img"].dtype == np.float64
    # image points are pixel CENTRES, as correction.py assembles them
    got = sorted(map(tuple, m["img"].tolist()))
    want = sorted([(float(x) + 0.5, float(y) + 0.5) for (y, x) in peaks])
    assert got == want
    # each keypoint matched its own descriptor -> landmark rows 3 and 4
    assert sorted(m["lm"].tolist()) == [3, 4]
    assert np.allclose(np.sort(m["obj"][:, 0]), np.sort(xyz[[3, 4], 0].numpy()))


def test_landmark_matches_topk_returns_k_candidates_per_keypoint():
    d, H, W = 8, 16, 24
    desc, scores = _stub(d, H, W)
    g = torch.Generator().manual_seed(7)
    feats = torch.nn.functional.normalize(torch.randn(9, d, generator=g), dim=-1)
    from usgloc.landmarks import LandmarkSet
    ls = LandmarkSet(idx=torch.arange(9), xyz=torch.rand(9, 3), feat=feats,
                     vac=torch.zeros(9), score=torch.zeros(9), info={})
    enc = _StubEncoder(desc, scores)
    m1 = landmark_matches(torch.zeros(H, W), ls, enc, detect_num=8, nms=0, topk=1)
    m3 = landmark_matches(torch.zeros(H, W), ls, enc, detect_num=8, nms=0, topk=3)
    assert len(m1["obj"]) == 2 and len(m3["obj"]) == 6
    # mutual-NN is a subset of the one-way nearest neighbours
    mm = landmark_matches(torch.zeros(H, W), ls, enc, detect_num=8, nms=0, mnn=True)
    assert len(mm["obj"]) <= len(m1["obj"])


def test_cull_restricts_the_landmarks_to_the_frustum():
    from usgloc.landmarks import LandmarkSet

    cam = Camera(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    c = LandmarkCorrector.__new__(LandmarkCorrector)
    c.camera = cam
    xyz = torch.tensor([[0.0, 0.0, 2.0],      # centre of the image
                        [0.0, 0.0, -2.0],     # behind the camera
                        [2.0, 0.0, 2.0],      # off to the right of the frame
                        [0.1, -0.1, 3.0]])    # inside
    c.landmarks = LandmarkSet(idx=torch.arange(4), xyz=xyz,
                              feat=torch.eye(4), vac=torch.zeros(4),
                              score=torch.zeros(4), info={})
    c.cull = 1.0
    ls, rows = c._visible(torch.eye(4))
    assert rows.tolist() == [0, 3]
    assert ls.info["culled"] == 2
    c.cull = 4.0                      # a wide margin admits the off-frame one
    ls2, rows2 = c._visible(torch.eye(4))
    assert rows2.tolist() == [0, 2, 3]


# ---------------------------------------------------------------- geometry

def _corrector_stub(obj, img, **kw):
    """A LandmarkCorrector with the association short-circuited."""
    c = LandmarkCorrector.__new__(LandmarkCorrector)
    cam = Camera(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    c.camera = cam
    c._K = cam.K(device="cpu", dtype=torch.float64).numpy()
    c._pl_cam = {"model": "PINHOLE", "width": 640, "height": 480,
                 "params": [525.0, 525.0, 320.0, 240.0]}
    c.solver = kw.get("solver", "poselib")
    c.reproj_thresh_px = kw.get("reproj_thresh_px", 12.0)
    c.min_inliers = kw.get("min_inliers", 8)
    c.min_inlier_ratio = 0.0
    c.ransac_iters = 10000
    c.pl_min_iters = 100
    c.pl_loss, c.pl_loss_scale, c.pl_bundle_iters = "CAUCHY", 1.0, 100
    c.refine = kw.get("refine", "none")
    c.refine_sigma, c.refine_iters = 1.0, 10
    c.gamma, c.wvac = 3.0, 0.0
    c._lw = np.ones(len(obj))
    c.cull = 0.0
    c.landmarks = None
    c.match = lambda q, T0=None: dict(obj=obj, img=img, score=np.ones(len(obj)),
                                      lm=np.arange(len(obj)), n_keypoints=len(obj))
    return c


def _synthetic_correspondences(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 3)) + np.array([0.0, 0.0, 3.0])
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    import cv2
    R, _ = cv2.Rodrigues((0.2 * ax).reshape(3, 1))
    t = np.array([0.05, -0.1, 0.2])
    K = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]])
    Xc = X @ R.T + t
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return (np.ascontiguousarray(X), np.ascontiguousarray(uv),
            torch.as_tensor(T, dtype=torch.float32))


def test_correct_recovers_the_pose_from_exact_correspondences():
    obj, img, T_true = _synthetic_correspondences()
    c = _corrector_stub(obj, img)
    r = c.correct(torch.zeros(480, 640), torch.eye(4))
    assert r.success and r.inliers >= 100
    assert torch.allclose(r.T.double(), T_true.double(), atol=1e-4)
    assert r.reproj_px < 1e-3


def test_correct_does_not_depend_on_the_start_pose():
    obj, img, T_true = _synthetic_correspondences(seed=3)
    c = _corrector_stub(obj, img)
    a = c.correct(torch.zeros(480, 640), torch.eye(4))
    far = torch.eye(4)
    far[:3, 3] = torch.tensor([5.0, -3.0, 7.0])
    b = c.correct(torch.zeros(480, 640), far)
    assert torch.equal(a.T, b.T)          # the estimator never sees T0
    assert a.inliers == b.inliers


def test_correct_reports_failure_and_returns_the_input_pose():
    obj, img, _ = _synthetic_correspondences(n=6)
    c = _corrector_stub(obj, img, min_inliers=8)
    T0 = torch.eye(4)
    r = c.correct(torch.zeros(480, 640), T0)
    assert not r.success and torch.equal(r.T, T0)


@pytest.mark.parametrize("refine", ["none", "lm", "cauchy", "lm+huber"])
def test_refine_modes_keep_the_pose(refine):
    obj, img, T_true = _synthetic_correspondences(seed=11)
    c = _corrector_stub(obj, img, refine=refine)
    r = c.correct(torch.zeros(480, 640), torch.eye(4))
    assert r.success
    assert torch.allclose(r.T.double(), T_true.double(), atol=1e-3)


def test_cv2_solver_branch_agrees_with_poselib():
    obj, img, T_true = _synthetic_correspondences(seed=13)
    a = _corrector_stub(obj, img, solver="cv2_iter").correct(
        torch.zeros(480, 640), torch.eye(4))
    b = _corrector_stub(obj, img, solver="poselib").correct(
        torch.zeros(480, 640), torch.eye(4))
    assert a.success and b.success
    assert torch.allclose(a.T.double(), b.T.double(), atol=1e-3)


# ------------------------------------------------- the match-score field

class _StubDS:
    def __init__(self, T, img):
        self._T, self._img = T, img

    def load_image(self, i):
        return self._img

    def __getitem__(self, i):
        class _R:
            pass
        r = _R()
        r.pose_cw = self._T
        return r


class _StubRenderer:
    """Returns depth = the true z of the single Gaussian, alpha = 1."""

    def __init__(self, depth, alpha):
        self._d, self._a = depth, alpha

    def render(self, *a, **kw):
        return {"depth": self._d, "alpha": self._a, "rgb": None}


def test_match_score_field_scores_the_projected_descriptor():
    from usgloc import landmarks as L
    from usgloc.superpoint import sample_descriptors

    H, W, d = 16, 24, 8
    cam = Camera(fx=10.0, fy=10.0, cx=W / 2, cy=H / 2, width=W, height=H)
    # one Gaussian straight ahead, one behind the camera (never observed)
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    desc = torch.nn.functional.normalize(
        torch.randn(1, d, H // 8, W // 8, generator=torch.Generator().manual_seed(2)),
        dim=1)
    scores = torch.full((1, H, W), 0.25)
    # the descriptor the sampler will read at the projection (12.0, 8.0) - 0.5
    uv = torch.tensor([[cam.cx, cam.cy]])
    want = sample_descriptors(desc, uv, H, W)[0]
    feats = torch.stack([want, torch.zeros(d)])
    emap = EvidentialMap(primitives=[], means=xyz,
                         covs=torch.eye(3).expand(2, 3, 3).clone(),
                         colors=torch.zeros(2, 3), opacities=torch.ones(2),
                         gaussian_vacuity=torch.zeros(2), loc_feats=feats,
                         loc_feat_obs=torch.ones(2))
    ds = _StubDS(torch.eye(4), torch.zeros(H, W))
    rend = _StubRenderer(torch.full((H, W), 2.0), torch.ones(H, W))
    out = L.match_score_field(emap, ds, cam, _StubEncoder(desc, scores), rend,
                              [0], log_every=0)
    assert out["obs"].tolist() == [1.0, 0.0]
    assert abs(float(out["score"][0]) - 1.0) < 1e-5     # its own descriptor
    assert float(out["score"][1]) == -1.0               # never observed
    assert abs(float(out["det"][0]) - 0.25) < 1e-6
    assert float(out["det"][1]) == 0.0


def test_match_score_field_rejects_occluded_gaussians():
    from usgloc import landmarks as L

    H, W, d = 16, 24, 8
    cam = Camera(fx=10.0, fy=10.0, cx=W / 2, cy=H / 2, width=W, height=H)
    xyz = torch.tensor([[0.0, 0.0, 2.0]])
    desc = torch.nn.functional.normalize(torch.randn(1, d, H // 8, W // 8), dim=1)
    emap = EvidentialMap(primitives=[], means=xyz,
                         covs=torch.eye(3)[None].clone(),
                         colors=torch.zeros(1, 3), opacities=torch.ones(1),
                         gaussian_vacuity=torch.zeros(1),
                         loc_feats=torch.nn.functional.normalize(torch.randn(1, d), dim=-1),
                         loc_feat_obs=torch.ones(1))
    ds = _StubDS(torch.eye(4), torch.zeros(H, W))
    enc = _StubEncoder(desc, torch.zeros(1, H, W))
    # the map's own rendered surface is at 1 m, the Gaussian sits at 2 m: occluded
    occ = L.match_score_field(emap, ds, cam, enc,
                              _StubRenderer(torch.ones(H, W), torch.ones(H, W)),
                              [0], log_every=0)
    assert occ["obs"].tolist() == [0.0]
    # transparent pixel (alpha below min_alpha) is rejected too
    thin = L.match_score_field(emap, ds, cam, enc,
                               _StubRenderer(torch.full((H, W), 2.0),
                                             torch.full((H, W), 0.05)),
                               [0], log_every=0)
    assert thin["obs"].tolist() == [0.0]
