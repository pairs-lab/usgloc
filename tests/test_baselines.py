import math

import pytest
import torch

from usgloc import se3, synthetic
from usgloc.baselines import (
    BaselineResult,
    as_pose_gaussian,
    icomma_style,
    random_anchor,
    top1_point,
)
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator
from usgloc.metrics import pose_errors
from usgloc.pipeline import PipelineConfig, USGLocPipeline
from usgloc.rendering import TorchSplatRenderer


@pytest.fixture(scope="module")
def setup():
    scene = synthetic.make_scene(num_objects=5, seed=0)
    emap = synthetic.build_map_from_scene(scene, n_anchors=8)
    cam = synthetic.default_camera(64, 48)
    renderer = TorchSplatRenderer()
    return scene, emap, cam, renderer


def test_icomma_style_reduces_pose_error(setup):
    scene, emap, cam, renderer = setup
    T_gt = emap.submaps[2].anchor_pose
    # Query image from the *mapping-time* scene at the GT pose.
    with torch.no_grad():
        img = renderer.render(
            emap.means, emap.covs, emap.colors, emap.opacities, T_gt, cam
        )["rgb"].clamp(0, 1)
    # Perturbed start.
    xi0 = torch.tensor([0.06, -0.04, 0.05, 0.02, -0.02, 0.03])
    T0 = se3.apply_tangent(T_gt, xi0)

    res = icomma_style(emap, renderer, cam, img, T0, iters=80, lr=5e-3)

    assert isinstance(res, BaselineResult)
    assert res.name == "iComMa-style"
    assert res.T_est.shape == (4, 4)
    assert torch.isfinite(res.T_est).all()
    assert math.isfinite(res.psnr)

    err0 = float(torch.linalg.norm(se3.relative_tangent(T_gt, T0)))
    err1 = float(torch.linalg.norm(se3.relative_tangent(T_gt, res.T_est)))
    assert err1 < 0.5 * err0, (err0, err1)


def test_top1_point_end_to_end(setup):
    scene, emap, cam, renderer = setup
    torch.manual_seed(0)
    desc = InstanceDescriptorNet(clip_dim=scene.clip_feats.shape[1], out_dim=64)
    agg = SceneAggregator(dim=64, heads=4)
    cfg = PipelineConfig(refine_iters=80, refine_lr=5e-3, jacobian_pixels=512)
    pipe = USGLocPipeline(emap, renderer, cam, desc, agg, cfg)
    synthetic.train_retrieval(desc, agg, scene, emap, cam, iters=120)
    pipe.encode_map()

    T_gt = emap.submaps[3].anchor_pose
    img = synthetic.render_query_image(scene, renderer, cam, T_gt)
    instances = synthetic.make_query_instances(scene, T_gt, cam)
    assert len(instances) >= 2

    res = top1_point(pipe, img, instances, iters=60)

    assert res.name == "top1-point"
    assert res.T_est.shape == (4, 4)
    assert torch.isfinite(res.T_est).all()
    assert math.isfinite(res.psnr)
    assert "submap_index" in res.extras

    # Retrieval may pick a neighbouring anchor; only require a loose bound.
    t_err, _ = pose_errors(res.T_est, T_gt)
    assert t_err < 0.5, t_err


def test_random_anchor_returns_an_anchor_pose(setup):
    _, emap, _, _ = setup
    g = torch.Generator().manual_seed(7)
    res = random_anchor(emap, generator=g)

    assert res.name == "random-anchor"
    assert math.isnan(res.psnr)
    assert any(
        torch.allclose(res.T_est, sm.anchor_pose) for sm in emap.submaps
    )
    idx = res.extras["submap_index"]
    assert torch.allclose(res.T_est, emap.submaps[idx].anchor_pose)


def test_as_pose_gaussian_diagonal_spd():
    T = torch.eye(4)
    b = BaselineResult(T_est=T, psnr=20.0, name="x")
    pg = as_pose_gaussian(b, sigma_t=0.05, sigma_r=0.02)

    assert pg.Sigma.shape == (6, 6)
    assert torch.allclose(pg.Sigma, pg.Sigma.T)
    assert (torch.linalg.eigvalsh(pg.Sigma) > 0).all()
    expected = torch.tensor([0.05**2] * 3 + [0.02**2] * 3)
    assert torch.allclose(torch.diagonal(pg.Sigma), expected)
    # Off-diagonal must be exactly zero (pure diagonal covariance).
    assert torch.count_nonzero(pg.Sigma - torch.diag(torch.diagonal(pg.Sigma))) == 0
    assert torch.allclose(pg.T_mode, T)
    # Wrapper must be usable by the calibration metrics (finite log-density).
    assert torch.isfinite(pg.log_prob(T))
