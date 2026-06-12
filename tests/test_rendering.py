import torch

from usgloc.rendering import Camera, TorchSplatRenderer
from usgloc import se3


def _single_blob_scene(color=(1.0, 0.0, 0.0), vac=0.8):
    means = torch.tensor([[0.0, 0.0, 2.0]])
    covs = (torch.eye(3) * 0.1**2)[None]
    colors = torch.tensor([list(color)])
    opac = torch.tensor([0.95])
    vacuity = torch.tensor([vac])
    return means, covs, colors, opac, vacuity


def _camera():
    return Camera(fx=60.0, fy=60.0, cx=32.0, cy=24.0, width=64, height=48)


def test_blob_renders_at_projection():
    cam = _camera()
    r = TorchSplatRenderer()
    means, covs, colors, opac, vac = _single_blob_scene()
    out = r.render(means, covs, colors, opac, torch.eye(4), cam, vacuity=vac)
    center = out["rgb"][24, 32]
    corner = out["rgb"][0, 0]
    assert center[0] > 0.5 and center[1] < 0.05  # red at the projected centre
    assert corner.abs().max() < 1e-3              # empty at the corner


def test_vacuity_blending():
    cam = _camera()
    r = TorchSplatRenderer()
    means, covs, colors, opac, vac = _single_blob_scene(vac=0.9)
    out = r.render(means, covs, colors, opac, torch.eye(4), cam, vacuity=vac)
    # Vacuity channel approximately alpha * u at the blob, zero elsewhere.
    assert abs(float(out["vacuity"][24, 32]) - 0.9 * float(out["alpha"][24, 32])) < 1e-4
    assert float(out["vacuity"][0, 0]) < 1e-4


def test_occlusion_order():
    cam = _camera()
    r = TorchSplatRenderer()
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    covs = (torch.eye(3) * 0.15**2).expand(2, 3, 3)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    opac = torch.tensor([0.95, 0.95])
    out = r.render(means, covs, colors, opac, torch.eye(4), cam)
    c = out["rgb"][24, 32]
    assert c[0] > 4 * c[1]  # the near (red) blob dominates


def test_gradient_wrt_pose():
    cam = _camera()
    r = TorchSplatRenderer()
    means, covs, colors, opac, vac = _single_blob_scene()
    target = r.render(means, covs, colors, opac, torch.eye(4), cam)["rgb"].detach()

    xi = torch.zeros(6, requires_grad=True)
    T = se3.apply_tangent(torch.eye(4), xi + torch.tensor([0.05, 0, 0, 0, 0, 0.0]))
    out = r.render(means, covs, colors, opac, T, cam)
    loss = ((out["rgb"] - target) ** 2).mean()
    loss.backward()
    assert xi.grad is not None
    assert torch.isfinite(xi.grad).all()
    assert xi.grad.abs().max() > 0


def test_empty_view():
    cam = _camera()
    r = TorchSplatRenderer()
    means, covs, colors, opac, vac = _single_blob_scene()
    # Camera looking away: blob behind the near plane.
    T = torch.eye(4)
    T[2, 3] = -10.0
    out = r.render(means, covs, colors, opac, T, cam, vacuity=vac)
    assert out["rgb"].abs().max() == 0
    assert out["vacuity"].abs().max() == 0
