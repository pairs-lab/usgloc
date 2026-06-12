"""Real-data benchmark harness: 7Scenes loading + trained-map ingestion.

Supports the experimental protocol of Sec. V of the RA-L draft
(paper/usgloc_ral_draft.tex): the public indoor benchmarks (7Scenes, and --
through the same directory conventions -- 12Scenes) provide RGB-D frames with
ground-truth camera-to-world poses, while the evidential semantic Gaussian map
(Sec. IV-A) is trained *offline* by an external open-vocabulary 3DGS pipeline
(e.g. Gaussian Grouping / OpenGaussian plus an EDL head, cf. the draft's
"Implementation" paragraph) and handed to this package as a tensor dict; see
:func:`load_gaussian_map` for the exact bridge format.

Conventions
-----------
* The 7Scenes ``frame-XXXXXX.pose.txt`` files store **camera-to-world**
  matrices; :class:`FrameRecord` keeps that raw matrix as ``pose_c2w`` and
  exposes the package-standard **camera-from-world** pose as ``pose_cw``
  (via :func:`usgloc.se3.se3_inverse`).
* Images load as ``(H, W, 3)`` float tensors in ``[0, 1]``.

Expected directory layout (the standard 7Scenes release)::

    root/
      <scene>/                     e.g. chess, fire, office, ...
        TrainSplit.txt             lines like "sequence1"
        TestSplit.txt
        seq-01/
          frame-000000.color.png
          frame-000000.depth.png
          frame-000000.pose.txt    4x4 row-major camera-to-world matrix
          ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

from .primitives import EvidentialMap, build_evidential_map
from .rendering import Camera
from .se3 import se3_inverse

# Standard Kinect intrinsics used by the 7Scenes (and 12Scenes) releases,
# defined for a 640 x 480 sensor.
KINECT_FX = 585.0
KINECT_FY = 585.0
KINECT_CX = 320.0
KINECT_CY = 240.0
KINECT_WIDTH = 640
KINECT_HEIGHT = 480


@dataclass
class FrameRecord:
    """One RGB-D frame of a sequence, with its ground-truth pose.

    Attributes:
        rgb_path: path to ``frame-XXXXXX.color.png``.
        depth_path: path to ``frame-XXXXXX.depth.png``.
        pose_c2w: (4, 4) **camera-to-world** matrix exactly as stored by the
            dataset's ``frame-XXXXXX.pose.txt``.
        seq: sequence directory name, e.g. ``"seq-01"``.
        frame_id: integer frame index within the sequence.
    """

    rgb_path: Path
    depth_path: Path
    pose_c2w: torch.Tensor
    seq: str
    frame_id: int

    @property
    def pose_cw(self) -> torch.Tensor:
        """(4, 4) **camera-from-world** pose -- the package convention."""
        return se3_inverse(self.pose_c2w)


def _read_pose_txt(path: Path) -> torch.Tensor:
    """Parse a 4x4 row-major pose matrix from a whitespace-separated text file."""
    rows = [
        [float(x) for x in line.split()]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    T = torch.tensor(rows, dtype=torch.float32)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 pose matrix, got shape {tuple(T.shape)}")
    return T


class SevenScenes:
    """The Microsoft 7Scenes benchmark (Shotton et al., CVPR'13).

    Parses the standard release layout (see the module docstring).  The same
    ``TrainSplit.txt`` / ``TestSplit.txt`` + ``seq-XX/frame-XXXXXX.*``
    convention covers 12Scenes mirrors organized identically.

    Args:
        root: dataset root containing one directory per scene.
        scene: scene name, e.g. ``"chess"``.
        split: ``"train"`` (mapping sequences) or ``"test"`` (query sequences).

    Raises:
        FileNotFoundError: if the scene directory or split file is missing.
        ValueError: if ``split`` is not ``"train"`` or ``"test"``.
    """

    def __init__(self, root: Union[str, Path], scene: str, split: str = "test"):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.scene = scene
        self.split = split
        scene_dir = self.root / scene
        if not scene_dir.is_dir():
            raise FileNotFoundError(
                f"scene directory not found: {scene_dir} "
                f"(expected the standard 7Scenes layout under --root)"
            )
        split_file = scene_dir / ("TrainSplit.txt" if split == "train" else "TestSplit.txt")
        if not split_file.is_file():
            raise FileNotFoundError(f"split file not found: {split_file}")

        self.sequences: List[str] = []
        for line in split_file.read_text().splitlines():
            m = re.search(r"sequence\s*(\d+)", line, flags=re.IGNORECASE)
            if m:
                self.sequences.append(f"seq-{int(m.group(1)):02d}")
        if not self.sequences:
            raise ValueError(f"{split_file} lists no 'sequenceN' entries")

        self.frames: List[FrameRecord] = []
        for seq in self.sequences:
            seq_dir = scene_dir / seq
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"sequence directory not found: {seq_dir}")
            pose_files = sorted(seq_dir.glob("frame-*.pose.txt"))
            if not pose_files:
                raise FileNotFoundError(f"no frame-*.pose.txt files in {seq_dir}")
            for pf in pose_files:
                frame_id = int(pf.name.split("-")[1].split(".")[0])
                stem = f"frame-{frame_id:06d}"
                self.frames.append(
                    FrameRecord(
                        rgb_path=seq_dir / f"{stem}.color.png",
                        depth_path=seq_dir / f"{stem}.depth.png",
                        pose_c2w=_read_pose_txt(pf),
                        seq=seq,
                        frame_id=frame_id,
                    )
                )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> FrameRecord:
        return self.frames[i]

    def load_image(self, i: int) -> torch.Tensor:
        """Load frame ``i``'s color image as a (H, W, 3) float tensor in [0, 1].

        PIL is imported lazily so the module imports on machines without it.
        """
        from PIL import Image  # local import: optional dependency

        import numpy as np

        with Image.open(self.frames[i].rgb_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    @staticmethod
    def camera(width: int = KINECT_WIDTH, height: int = KINECT_HEIGHT) -> Camera:
        """Kinect intrinsics (fx=fy=585, cx=320, cy=240), rescaled if the
        requested resolution differs from the native 640 x 480."""
        sx = width / KINECT_WIDTH
        sy = height / KINECT_HEIGHT
        return Camera(
            fx=KINECT_FX * sx,
            fy=KINECT_FY * sy,
            cx=KINECT_CX * sx,
            cy=KINECT_CY * sy,
            width=width,
            height=height,
        )


# Keys an exported map checkpoint must provide (per-Gaussian tensors).
MAP_REQUIRED_KEYS = (
    "means",       # (N, 3) splat centres, world frame
    "covs",        # (N, 3, 3) splat covariances, world frame
    "colors",      # (N, 3) RGB in [0, 1]
    "opacities",   # (N,)
    "clip_feats",  # (N, d) open-vocabulary embeddings, l2-normalized
    "beliefs",     # (N, C) DST belief masses from the EDL head
    "vacuities",   # (N,) vacuity masses from the EDL head
)
MAP_OPTIONAL_KEYS = ("instance_ids",)  # (N,) integer cluster/instance ids


def load_gaussian_map(
    path: Union[str, Path],
    fuse_rule: str = "yager",
    device: Optional[Union[str, torch.device]] = None,
) -> EvidentialMap:
    """Load an externally trained evidential 3DGS map (the real-data bridge).

    This is the hand-off point from any open-vocabulary 3DGS training
    pipeline (e.g. Gaussian Grouping or OpenGaussian for the splats and
    per-Gaussian CLIP features, plus an EDL head producing Dirichlet
    evidence; cf. ``usgloc.evidential.EvidentialHead``).  The trainer exports
    a single ``torch.save``'d dict of per-Gaussian tensors:

    Required keys (N Gaussians, C classes, d-dim embeddings):
        * ``means`` (N, 3), ``covs`` (N, 3, 3), ``colors`` (N, 3),
          ``opacities`` (N,) -- raw 3DGS attributes for rendering.
        * ``clip_feats`` (N, d) -- open-vocabulary embeddings.
        * ``beliefs`` (N, C), ``vacuities`` (N,) -- subjective-logic opinions
          from the evidential head (Eq. 3-4 of the draft).

    Optional keys:
        * ``instance_ids`` (N,) -- integer object ids (e.g. Gaussian Grouping
          identity argmax); if absent, spatial clustering is used instead
          (see :func:`usgloc.primitives.cluster_gaussians`).

    Args:
        path: path to the ``torch.save``'d dict.
        fuse_rule: DST fusion rule for cluster-level opinions; ``"yager"``
            (default) routes inter-observation conflict into vacuity, which is
            what the refinement down-weighting needs (see
            ``usgloc.synthetic.build_map_from_scene`` for the rationale).
        device: optional device to move all tensors to before building.

    Returns:
        The :class:`usgloc.primitives.EvidentialMap`, with per-splat vacuity
        already broadcast (``gaussian_vacuity``) for Eq. 8.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the checkpoint is not a dict or lacks required keys
            (the message lists exactly which ones are missing).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian map checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"{path}: expected a torch.save'd dict of tensors, got {type(ckpt).__name__}"
        )
    missing = [k for k in MAP_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"{path}: Gaussian map checkpoint is missing required keys: "
            f"{', '.join(missing)}. Required: {', '.join(MAP_REQUIRED_KEYS)}; "
            f"optional: {', '.join(MAP_OPTIONAL_KEYS)}. "
            "See usgloc.datasets.load_gaussian_map for the expected format."
        )

    def _get(key: str) -> Optional[torch.Tensor]:
        v = ckpt.get(key)
        if v is not None and device is not None:
            v = v.to(device)
        return v

    return build_evidential_map(
        means=_get("means"),
        covs=_get("covs"),
        colors=_get("colors"),
        opacities=_get("opacities"),
        clip_feats=_get("clip_feats"),
        beliefs=_get("beliefs"),
        vacuities=_get("vacuities"),
        instance_ids=_get("instance_ids"),
        fuse_rule=fuse_rule,
    )
