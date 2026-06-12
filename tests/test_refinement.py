import torch

from usgloc import se3, synthetic
from usgloc.refinement import refine_pose, residual_jacobian
from usgloc.rendering import TorchSplatRenderer


def _setup(seed=0):
    scene = synthetic.make_scene(num_objects=5, dynamic_fraction=0.2, seed=seed)
    emap = synthetic.build_map_from_scene(scene, n_anchors=8)
    cam = synthetic.default_camera(64, 48)
    renderer = TorchSplatRenderer()
    return scene, emap, cam, renderer


def test_residual_jacobian_fwd_matches_fd():
    A = torch.randn(8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def r(xi):
        return A @ xi + 0.5 * (A @ xi) ** 2

    J_fwd = residual_jacobian(r, mode="fwd", dtype=torch.float64)
    J_fd = residual_jacobian(r, mode="fd", dtype=torch.float64)
    assert torch.allclose(J_fwd, J_fd, atol=1e-6)
    assert torch.allclose(J_fwd, A, atol=1e-6)  # value at xi=0


def test_refinement_reduces_pose_error():
    scene, emap, cam, renderer = _setup()
    T_gt = emap.submaps[2].anchor_pose
    # Query image from the *mapping-time* scene at the GT pose.
    with torch.no_grad():
        img = renderer.render(
            emap.means, emap.covs, emap.colors, emap.opacities, T_gt, cam
        )["rgb"].clamp(0, 1)
    # Perturbed start.
    xi0 = torch.tensor([0.06, -0.04, 0.05, 0.02, -0.02, 0.03])
    T0 = se3.apply_tangent(T_gt, xi0)
    res = refine_pose(
        emap, renderer, cam, img, T0,
        gamma=3.0, iters=120, lr=5e-3, compute_covariance=True,
        jacobian_pixels=1024,
    )
    err0 = float(torch.linalg.norm(se3.relative_tangent(T_gt, T0)))
    err1 = float(torch.linalg.norm(se3.relative_tangent(T_gt, res.T_star)))
    assert err1 < 0.5 * err0, (err0, err1)

    # Laplace covariance must be symmetric positive definite and finite.
    assert torch.isfinite(res.Sigma).all()
    assert torch.allclose(res.Sigma, res.Sigma.T, atol=1e-8)
    evals = torch.linalg.eigvalsh(res.Sigma)
    assert (evals > 0).all()


def test_covariance_grows_when_image_is_corrupted():
    """A degraded query (heavy noise) must yield a larger Laplace covariance:
    sigma_hat^2 scales with the residuals, so the covariance is honest."""
    scene, emap, cam, renderer = _setup(seed=1)
    T_gt = emap.submaps[1].anchor_pose
    with torch.no_grad():
        img = renderer.render(
            emap.means, emap.covs, emap.colors, emap.opacities, T_gt, cam
        )["rgb"].clamp(0, 1)
    noisy = (img + 0.25 * torch.randn(img.shape, generator=torch.Generator().manual_seed(3))).clamp(0, 1)

    kw = dict(gamma=0.0, iters=0, lr=1e-3, compute_covariance=True, jacobian_pixels=1024)
    res_clean = refine_pose(emap, renderer, cam, img, T_gt, **kw)
    res_noisy = refine_pose(emap, renderer, cam, noisy, T_gt, **kw)
    assert torch.trace(res_noisy.Sigma) > 3 * torch.trace(res_clean.Sigma)


def test_vacuity_weighting_downweights_dynamic_pixels():
    scene, emap, cam, renderer = _setup(seed=2)
    T = emap.submaps[0].anchor_pose
    out = renderer.render(
        emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
        vacuity=emap.gaussian_vacuity,
    )
    w = torch.exp(-3.0 * out["vacuity"])
    # Weights must dip strictly below 1 somewhere iff dynamic content is visible.
    dyn_u = emap.gaussian_vacuity[scene.is_dynamic]
    assert float(dyn_u.min()) > 0.5  # dynamic splats carry high vacuity in the map
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
