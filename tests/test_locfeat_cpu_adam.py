"""finetune_loc_feats(cpu_opt_state=True) must follow torch.optim.Adam (the GPU path)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from usgloc import locfeat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")


class _Renderer:
    """Each pixel's feature is a fixed mix of the (normalised) per-Gaussian features."""

    def __init__(self, n, h, w):
        self.W = torch.rand(h * w, n, generator=torch.Generator().manual_seed(1))
        self.h, self.w = h, w

    def render(self, means, covs, colors, opacities, T, camera, vacuity=None, loc_feats=None, feat_normalize=True):
        f = F.normalize(self.W @ loc_feats, dim=-1).view(self.h, self.w, -1)
        return {"feats": f, "alpha": torch.ones(self.h, self.w)}


class _Frames:
    def __init__(self, imgs):
        self.imgs = imgs

    def load_image(self, i):
        return self.imgs[i]

    def __getitem__(self, i):
        return SimpleNamespace(pose_cw=torch.eye(4))


class _Encoder:
    def dense_descriptors(self, img):
        return F.normalize(img, dim=1), None


@pytest.mark.parametrize("beta1", [0.0, 0.9])
def test_cpu_opt_state_matches_torch_adam(beta1):
    n, d, h, w = 50, 8, 4, 5
    g = torch.Generator().manual_seed(0)
    imgs = [torch.randn(1, d, h, w, generator=g) for _ in range(3)]
    ds = _Frames(imgs)
    emap = SimpleNamespace(means=torch.zeros(n, 3), covs=None, colors=None, opacities=None, gaussian_vacuity=None)
    cam = SimpleNamespace(height=h, width=w)
    init = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    kw = dict(emap=emap, ds=ds, camera=cam, encoder=_Encoder(), renderer=_Renderer(n, h, w), frame_ids=[0, 1, 2],
              init=init, iters=30, lr=1e-2, lr_final=1e-3, beta1=beta1, log_every=0)
    ref = locfeat.finetune_loc_feats(**kw, cpu_opt_state=False)
    cpu = locfeat.finetune_loc_feats(**kw, cpu_opt_state=True)
    assert (ref["loc_feats"] - init).abs().max() > 1e-2          # the features did move
    torch.testing.assert_close(cpu["loc_feats"], ref["loc_feats"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.tensor(cpu["trace"]), torch.tensor(ref["trace"]), atol=1e-5, rtol=1e-5)
