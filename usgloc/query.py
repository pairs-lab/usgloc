"""Real-data query front end: FastSAM masks + CLIP + the shared evidential head.

Optional dependencies (``pip install usgloc[realdata]``):
* ``ultralytics`` -- FastSAM instance masks.
* ``open_clip_torch`` -- CLIP image embeddings for each instance crop.

Both the map observations and the query crops must pass through the *same*
:class:`usgloc.evidential.EvidentialHead` so that map and query vacuities are
directly comparable (Sec. IV-B).  On machines without these packages, the
synthetic generator in ``scripts/run_demo_synthetic.py`` produces
:class:`usgloc.pipeline.QueryInstance` lists with identical structure.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .evidential import EvidentialHead, evidence_to_subjective_opinion
from .pipeline import QueryInstance


class RealQueryEncoder:
    def __init__(
        self,
        edl_head: EvidentialHead,
        clip_model_name: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        fastsam_weights: str = "FastSAM-s.pt",
        device: str = "cuda",
        min_mask_area: float = 0.002,
    ):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("open_clip_torch is required: pip install usgloc[realdata]") from e
        try:
            from ultralytics import FastSAM
        except ImportError as e:
            raise ImportError("ultralytics is required: pip install usgloc[realdata]") from e

        self.device = device
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=device
        )
        self.clip.eval()
        self.sam = FastSAM(fastsam_weights)
        self.edl_head = edl_head.to(device).eval()
        self.min_mask_area = min_mask_area

    @torch.no_grad()
    def encode(self, image_bgr) -> List[QueryInstance]:
        """numpy HxWx3 (uint8) -> list of evidential query instances."""
        import numpy as np
        from PIL import Image

        H, W = image_bgr.shape[:2]
        results = self.sam(image_bgr, device=self.device, retina_masks=True, verbose=False)
        masks = results[0].masks
        if masks is None:
            return []

        instances: List[QueryInstance] = []
        for m in masks.data:  # (h, w) float mask
            mask = m.bool().cpu().numpy()
            area = float(mask.mean())
            if area < self.min_mask_area:
                continue
            ys, xs = mask.nonzero()
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = image_bgr[y0:y1, x0:x1, ::-1]  # to RGB
            pil = Image.fromarray(np.ascontiguousarray(crop))
            x = self.preprocess(pil)[None].to(self.device)
            f = self.clip.encode_image(x)[0]
            f = f / f.norm().clamp_min(1e-8)

            evidence = self.edl_head(f[None])[0]
            _, belief, vacuity = evidence_to_subjective_opinion(evidence[None])
            centroid = torch.tensor(
                [xs.mean() / W, ys.mean() / H], dtype=f.dtype, device=f.device
            )
            instances.append(
                QueryInstance(
                    clip_feat=f,
                    centroid=centroid,
                    area=torch.tensor([area], dtype=f.dtype, device=f.device),
                    belief=belief[0],
                    vacuity=vacuity[0],
                )
            )
        return instances
