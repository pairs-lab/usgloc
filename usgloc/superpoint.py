"""SuperPoint dense descriptor encoder (the feature field's descriptor source).

Verbatim architecture of the SuperPoint (DeTone et al., CVPRW'18) released
model, as used by STDLoc (CVPR'25) for its per-Gaussian localisation features:
their ``encoders/sp_encoder/export_image_embeddings.py``.  The weights
(``usgloc/weights/superpoint_v1.pth``, 5 MB, MD5
938af9f432d327751dcbc0d6c7a0448b) are the same file that ships with the
official SuperPoint / SuperGluePretrainedNetwork release; they are vendored
here so this package adds NO pip dependency for the descriptor.

The module is a frozen feature extractor: it is never trained here, only the
per-Gaussian descriptors are.

Output convention
-----------------
``dense_descriptors(img)`` returns the ``(B, 256, H/8, W/8)`` descriptor map,
L2-normalised along the channel dimension -- SuperPoint's native descriptor
resolution -- and the ``(B, H, W)`` full-resolution detector heat map.  Nothing
here up- or down-samples for you: the consumer decides (STDLoc's "fine" map is
this map bilinearly upsampled to the image resolution and re-normalised, its
"coarse" map is the 1/8 map itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Vendored SuperPoint weights (see the module docstring).
SUPERPOINT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "superpoint_v1.pth"

#: Native descriptor dimension and spatial stride of the encoder.
SUPERPOINT_DIM = 256
SUPERPOINT_STRIDE = 8


class SuperPoint(nn.Module):
    """Frozen SuperPoint encoder: dense descriptors + detector scores.

    Args:
        weights: checkpoint path (defaults to the vendored one).  ``None``
            leaves the network randomly initialised (tests only).
    """

    def __init__(self, weights: Optional[Path] = SUPERPOINT_WEIGHTS):
        super().__init__()
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, SUPERPOINT_DIM, 1, 1, 0)
        if weights is not None:
            p = Path(weights)
            if not p.is_file():
                raise FileNotFoundError(
                    f"SuperPoint weights not found: {p}. They are vendored in the "
                    "package (usgloc/weights/superpoint_v1.pth); pass weights=None "
                    "only for a randomly-initialised network."
                )
            sd = torch.load(str(p), map_location="cpu", weights_only=True)
            self.load_state_dict(sd, strict=True)
        for q in self.parameters():
            q.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------
    @staticmethod
    def to_gray(img: torch.Tensor) -> torch.Tensor:
        """Accept (H, W), (H, W, 3), (B, 3, H, W) or (B, 1, H, W) in [0, 1] ->
        (B, 1, H, W) luma.

        The grayscale conversion is ``torchvision.transforms.Grayscale``'s
        (and STDLoc's) ITU-R 601-2 luma, which is bit-identical to
        :func:`usgloc.correction._to_gray`.
        """
        if img.dim() == 2:                       # (H, W) luma
            return img[None, None]
        if img.dim() == 3:
            if img.shape[-1] == 3:               # (H, W, 3)
                img = img.permute(2, 0, 1)[None]
            else:                                # (C, H, W)
                img = img[None]
        if img.dim() != 4:
            raise ValueError(f"unsupported image shape {tuple(img.shape)}")
        if img.shape[1] == 1:
            return img
        if img.shape[1] != 3:
            raise ValueError(f"expected 1 or 3 channels, got {img.shape[1]}")
        w = img.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
        return (img * w).sum(dim=1, keepdim=True)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """See :meth:`dense_descriptors`."""
        return self.dense_descriptors(img)

    @torch.no_grad()
    def dense_descriptors(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, 256, H/8, W/8) L2-normalised descriptors + (B, H, W) scores."""
        x = self.to_gray(img).to(next(self.parameters()).dtype)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)
        return desc, scores


def upsample_descriptors(desc: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """STDLoc's "fine" query map: bilinear upsample to (height, width) then
    re-L2-normalise per pixel.  ``desc`` is (B, D, h, w)."""
    out = F.interpolate(desc, size=(height, width), mode="bilinear", align_corners=False)
    return F.normalize(out, p=2, dim=1)


def sample_descriptors(desc: torch.Tensor, uv: torch.Tensor,
                       height: int, width: int) -> torch.Tensor:
    """Bilinearly sample the (1, D, h, w) descriptor map at full-resolution
    pixel coordinates ``uv`` (M, 2) given as (x, y) pixel INDICES, and
    re-normalise.

    The pixel-centre convention matches the rest of the package: index (i, j)
    is the ray through continuous coordinate (j + 0.5, i + 0.5).
    """
    gx = 2.0 * (uv[:, 0] + 0.5) / width - 1.0
    gy = 2.0 * (uv[:, 1] + 0.5) / height - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None, None].to(desc.dtype)
    out = F.grid_sample(desc, grid, mode="bilinear", align_corners=False)[0, :, 0].T
    return F.normalize(out, p=2, dim=-1)
