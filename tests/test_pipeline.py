import torch

from usgloc import se3, synthetic
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator, info_nce_loss
from usgloc.matching import gated_similarity, greedy_instance_matches
from usgloc.pipeline import PipelineConfig, USGLocPipeline
from usgloc.primitives import build_evidential_map
from usgloc.rendering import TorchSplatRenderer


def test_gated_similarity_suppresses_uncertain_pairs():
    d = torch.eye(4)[:3]  # 3 orthonormal descriptors
    vac_lo = torch.zeros(3)
    vac_hi = torch.tensor([0.0, 0.9, 0.0])
    s_lo = gated_similarity(d, d, vac_lo, vac_lo, lambda_u=2.0)
    s_hi = gated_similarity(d, d, vac_hi, vac_lo, lambda_u=2.0)
    assert torch.allclose(torch.diagonal(s_lo), torch.ones(3))
    assert s_hi[1, 1] < 0.2 < s_hi[0, 0]
    matches = greedy_instance_matches(s_hi, min_score=0.2)
    assert (1, 1) not in [(i, j) for i, j, _ in matches]


def test_map_building_flags_dynamic_objects():
    scene = synthetic.make_scene(num_objects=6, dynamic_fraction=0.34, seed=0)
    emap = synthetic.build_map_from_scene(scene, n_anchors=6)
    vac = {}
    for g in emap.primitives:
        dyn = bool(scene.is_dynamic[g.member_idx[0]])
        vac.setdefault(dyn, []).append(float(g.effective_vacuity))
    assert min(vac[True]) > max(vac[False]), vac


def test_info_nce_trains_retrieval_alignment():
    torch.manual_seed(0)
    f2d = torch.randn(8, 16, requires_grad=True)
    f3d = torch.randn(8, 16, requires_grad=True)
    loss = info_nce_loss(
        torch.nn.functional.normalize(f2d, dim=-1),
        torch.nn.functional.normalize(f3d, dim=-1),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert f2d.grad is not None


def test_end_to_end_localization_and_posterior():
    torch.manual_seed(0)
    scene = synthetic.make_scene(num_objects=6, dynamic_fraction=0.2, seed=3)
    emap = synthetic.build_map_from_scene(scene, n_anchors=10)
    cam = synthetic.default_camera(64, 48)
    renderer = TorchSplatRenderer()

    desc = InstanceDescriptorNet(clip_dim=scene.clip_feats.shape[1], out_dim=64)
    agg = SceneAggregator(dim=64, heads=4)
    cfg = PipelineConfig(top_k=2, refine_iters=80, refine_lr=5e-3, jacobian_pixels=512)
    pipe = USGLocPipeline(emap, renderer, cam, desc, agg, cfg)
    synthetic.train_retrieval(desc, agg, scene, emap, cam, iters=150)
    pipe.encode_map()

    # Query: GT pose perturbed off an anchor; dynamic objects moved.
    T_anchor = emap.submaps[4].anchor_pose
    T_gt = se3.apply_tangent(T_anchor, torch.tensor([0.03, -0.02, 0.02, 0.01, 0.0, 0.02]))
    img = synthetic.render_query_image(scene, renderer, cam, T_gt)
    instances = synthetic.make_query_instances(scene, T_gt, cam)
    assert len(instances) >= 2

    res = pipe.localize(img, instances)
    # Structure checks: a normalized mixture with SPD covariances.
    assert abs(float(res.posterior.weights.sum()) - 1.0) < 1e-5
    for c in res.posterior.components:
        assert (torch.linalg.eigvalsh(c.Sigma) > 0).all()
    # Condensation produces a finite Gaussian summary and finite NLL.
    cond = res.condensed
    assert torch.isfinite(cond.Sigma).all()
    assert torch.isfinite(torch.tensor(res.posterior.nll(T_gt)))

    # The refinement should land near the ground truth (coarse-to-fine works):
    from usgloc.metrics import pose_errors

    t_err, r_err = pose_errors(res.posterior.mode.T_mode, T_gt)
    assert t_err < 0.30 and r_err < 15.0, (t_err, r_err)
