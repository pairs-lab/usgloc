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
from typing import List, Optional, Tuple, Union

import torch

from .primitives import EvidentialMap, build_evidential_map
from .rendering import Camera
from .se3 import quat_to_rotmat, se3_inverse

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


def _read_sfm_pose_file(path: Path) -> dict:
    """Parse a visloc_pseudo_gt_limitations pose file (Brachmann et al., ICCV'21).

    Each line is ``seq-XX/frame-XXXXXX.color.png qw qx qy qz tx ty tz focal``,
    with the pose in the COLMAP **world-to-camera** convention
    (``p_cam = R(q) p_world + t``) -- i.e. exactly the package-standard
    camera-from-world pose.

    Returns:
        Mapping ``"seq-XX/frame-XXXXXX"`` -> ``(T_cw (4, 4) tensor, focal)``.
    """
    entries: dict = {}
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 9:
            raise ValueError(
                f"{path}:{ln}: expected 'image qw qx qy qz tx ty tz focal' "
                f"(9 fields), got {len(parts)}"
            )
        key = parts[0].replace(".color.png", "").replace(".color.jpg", "")
        vals = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float64)
        R = quat_to_rotmat(vals[:4])
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = R
        T[:3, 3] = vals[4:7]
        entries[key] = (T.to(torch.float32), float(vals[7]))
    if not entries:
        raise ValueError(f"{path}: no pose lines parsed")
    return entries


class SevenScenes:
    """The Microsoft 7Scenes benchmark (Shotton et al., CVPR'13).

    Parses the standard release layout (see the module docstring).  The same
    ``TrainSplit.txt`` / ``TestSplit.txt`` + ``seq-XX/frame-XXXXXX.*``
    convention covers 12Scenes mirrors organized identically.

    Args:
        root: dataset root containing one directory per scene.
        scene: scene name, e.g. ``"chess"``.
        split: ``"train"`` (mapping sequences) or ``"test"`` (query sequences).
        sfm_pose_dir: optional directory with the Brachmann et al. (ICCV'21)
            SfM pseudo-ground-truth files ``<scene>_<split>.txt`` (the
            ``pgt/sfm/7scenes`` folder of the ``visloc_pseudo_gt_limitations``
            release).  When given, poses come from that file instead of the
            original DSLAM ``pose.txt`` files -- the protocol Table I of the
            draft commits to -- frames the SfM reconstruction did not register
            are dropped (their count is kept in ``n_unregistered``), and the
            per-scene SfM focal length is exposed as ``focal`` /
            :meth:`scene_camera`.

    Raises:
        FileNotFoundError: if the scene directory or split file is missing.
        ValueError: if ``split`` is not ``"train"`` or ``"test"``.
    """

    def __init__(
        self,
        root: Union[str, Path],
        scene: str,
        split: str = "test",
        sfm_pose_dir: Optional[Union[str, Path]] = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.scene = scene
        self.split = split
        self.focal: Optional[float] = None
        self.n_unregistered = 0

        sfm_poses = None
        if sfm_pose_dir is not None:
            sfm_file = Path(sfm_pose_dir) / f"{scene}_{split}.txt"
            if not sfm_file.is_file():
                raise FileNotFoundError(
                    f"SfM pseudo-GT file not found: {sfm_file} "
                    "(expected the pgt/sfm/7scenes layout of "
                    "visloc_pseudo_gt_limitations)"
                )
            sfm_poses = _read_sfm_pose_file(sfm_file)
            focals = {f for _, f in sfm_poses.values()}
            self.focal = sum(focals) / len(focals)
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
                if sfm_poses is not None:
                    entry = sfm_poses.get(f"{seq}/{stem}")
                    if entry is None:  # frame not registered by the SfM model
                        self.n_unregistered += 1
                        continue
                    pose_c2w = se3_inverse(entry[0])
                else:
                    pose_c2w = _read_pose_txt(pf)
                self.frames.append(
                    FrameRecord(
                        rgb_path=seq_dir / f"{stem}.color.png",
                        depth_path=seq_dir / f"{stem}.depth.png",
                        pose_c2w=pose_c2w,
                        seq=seq,
                        frame_id=frame_id,
                    )
                )
        if sfm_poses is not None and not self.frames:
            raise ValueError(
                f"no frames of {self.scene}/{split} matched the SfM pseudo-GT "
                f"file ({len(sfm_poses)} entries); wrong --sfm-pose-dir?"
            )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> FrameRecord:
        return self.frames[i]

    def load_image(self, i: int) -> torch.Tensor:
        """Load frame ``i``'s color image as a (H, W, 3) float tensor in [0, 1]."""
        return _load_rgb(self.frames[i].rgb_path)

    # Native color resolution (the resolution scene_camera intrinsics refer to).
    color_width = KINECT_WIDTH
    color_height = KINECT_HEIGHT

    def depth_camera(self) -> Camera:
        """Intrinsics to back-project the depth maps with (7Scenes: depth is
        distributed registered to the color frame at the same resolution)."""
        return self.scene_camera(KINECT_WIDTH, KINECT_HEIGHT)

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

    def scene_camera(self, width: int = KINECT_WIDTH, height: int = KINECT_HEIGHT) -> Camera:
        """Like :meth:`camera`, but uses the per-scene SfM focal length when
        the dataset was built with ``sfm_pose_dir`` (the pseudo-GT files store
        the focal each scene's SfM model was calibrated with, e.g. 526.22 for
        chess -- evaluating SfM poses under the default focal mixes
        conventions)."""
        if self.focal is None:
            return self.camera(width, height)
        sx = width / KINECT_WIDTH
        sy = height / KINECT_HEIGHT
        return Camera(
            fx=self.focal * sx,
            fy=self.focal * sy,
            cx=KINECT_CX * sx,
            cy=KINECT_CY * sy,
            width=width,
            height=height,
        )


def _load_rgb(path: Path) -> torch.Tensor:
    """(H, W, 3) float tensor in [0, 1] from any PIL-readable image file."""
    from PIL import Image  # local import: optional dependency

    import numpy as np

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


class TwelveScenes:
    """The 12Scenes benchmark (Valentin et al., 3DV'16).

    Native layout (``scripts/download_12scenes.sh``)::

        root/
          <building>/<scene>/           e.g. apt1/kitchen
            split.txt                   line 1: "sequence0 [frames=NNN]"
            info.txt                    m_calibration{Color,Depth}Intrinsic, ...
            data/
              frame-XXXXXX.color.jpg    1296 x 968
              frame-XXXXXX.depth.png    640 x 480, mm, 65535/0 invalid
              frame-XXXXXX.pose.txt     4x4 camera-to-world

    The FIRST sequence (frames 0..NNN-1) is the TEST split; the rest is
    train, following the dataset's own convention. ``scene`` is
    ``"<building>/<scene>"``. ``sfm_pose_dir`` points at the
    ``pgt/sfm/12scenes`` folder of visloc_pseudo_gt_limitations, whose files
    are ``<building>_<scene>_<split>.txt`` with image keys
    ``data/frame-XXXXXX.color.jpg`` (COLMAP world-to-camera + focal, exactly
    as for 7Scenes).
    """

    def __init__(
        self,
        root: Union[str, Path],
        scene: str,
        split: str = "test",
        sfm_pose_dir: Optional[Union[str, Path]] = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.scene = scene
        self.split = split
        self.focal: Optional[float] = None
        self.n_unregistered = 0

        scene_dir = self.root / scene
        data_dir = scene_dir / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"scene data directory not found: {data_dir} "
                "(expected the 12Scenes layout under --root)"
            )
        split_file = scene_dir / "split.txt"
        m = re.search(r"frames\s*=\s*(\d+)", split_file.read_text())
        if not m:
            raise ValueError(f"{split_file}: no 'frames=N' entry on the first line")
        n_test = int(m.group(1))

        self._info = {}
        for line in (scene_dir / "info.txt").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                self._info[k.strip()] = v.strip()

        sfm_poses = None
        if sfm_pose_dir is not None:
            sfm_file = Path(sfm_pose_dir) / f"{scene.replace('/', '_')}_{split}.txt"
            if not sfm_file.is_file():
                raise FileNotFoundError(f"SfM pseudo-GT file not found: {sfm_file}")
            sfm_poses = _read_sfm_pose_file(sfm_file)
            focals = {f for _, f in sfm_poses.values()}
            self.focal = sum(focals) / len(focals)

        self.frames: List[FrameRecord] = []
        for pf in sorted(data_dir.glob("frame-*.pose.txt")):
            frame_id = int(pf.name.split("-")[1].split(".")[0])
            if (frame_id < n_test) != (split == "test"):
                continue
            stem = f"frame-{frame_id:06d}"
            if sfm_poses is not None:
                entry = sfm_poses.get(f"data/{stem}")
                if entry is None:
                    self.n_unregistered += 1
                    continue
                pose_c2w = se3_inverse(entry[0])
            else:
                pose_c2w = _read_pose_txt(pf)
                if not torch.isfinite(pose_c2w).all():
                    self.n_unregistered += 1  # untracked frame in the raw release
                    continue
            self.frames.append(
                FrameRecord(
                    rgb_path=data_dir / f"{stem}.color.jpg",
                    depth_path=data_dir / f"{stem}.depth.png",
                    pose_c2w=pose_c2w,
                    seq="data",
                    frame_id=frame_id,
                )
            )
        if not self.frames:
            raise ValueError(f"no frames for {scene}/{split}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> FrameRecord:
        return self.frames[i]

    def load_image(self, i: int) -> torch.Tensor:
        return _load_rgb(self.frames[i].rgb_path)

    def _intrinsic(self, key: str) -> Tuple[float, float, float, float]:
        """(fx, fy, cx, cy) from a 4x4 row-major intrinsic line of info.txt."""
        v = [float(x) for x in self._info[key].split()]
        return v[0], v[5], v[2], v[6]

    @property
    def color_width(self) -> int:
        return int(self._info.get("m_colorWidth", 1296))

    @property
    def color_height(self) -> int:
        return int(self._info.get("m_colorHeight", 968))

    def scene_camera(self, width: Optional[int] = None, height: Optional[int] = None) -> Camera:
        """Color-camera intrinsics: the per-scene SfM focal when built with
        ``sfm_pose_dir`` (referred to the native color resolution), else the
        calibrated color intrinsics from info.txt; rescaled to (width, height)."""
        W, H = self.color_width, self.color_height
        width = W if width is None else width
        height = H if height is None else height
        if self.focal is not None:
            fx = fy = self.focal
            cx, cy = W / 2.0, H / 2.0
        else:
            fx, fy, cx, cy = self._intrinsic("m_calibrationColorIntrinsic")
        sx, sy = width / W, height / H
        return Camera(fx=fx * sx, fy=fy * sy, cx=cx * sx, cy=cy * sy,
                      width=width, height=height)

    def depth_camera(self) -> Camera:
        """Native depth intrinsics (640 x 480) for back-projection."""
        fx, fy, cx, cy = self._intrinsic("m_calibrationDepthIntrinsic")
        return Camera(fx=fx, fy=fy, cx=cx, cy=cy,
                      width=int(self._info.get("m_depthWidth", 640)),
                      height=int(self._info.get("m_depthHeight", 480)))


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
MAP_OPTIONAL_KEYS = (
    "instance_ids",          # (N,) integer cluster/instance ids
    "loc_feats",             # (N, D) per-Gaussian LOCAL descriptor field, float16
    "loc_feat_obs",          # (N,) training views each descriptor was averaged over
    "loc_feat_pca_mean",     # (D_src,) PCA mean, if loc_feats were reduced
    "loc_feat_pca_basis",    # (D_src, D) PCA basis, if loc_feats were reduced
    "sh_coeffs",             # (N, K, 3) spherical-harmonic colour, K=(deg+1)^2.
                             # Optional view-dependent appearance: the map keeps
                             # its (N, 3) diffuse `colors` too, and a caller opts in
                             # by passing sh_coeffs to the renderer in place of
                             # colors (GsplatRenderer.render accepts either shape).
)


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
        * ``loc_feats`` (N, D) -- per-Gaussian LOCAL descriptor field (float16
          on disk, L2-normalized rows), built by
          :mod:`usgloc.locfeat`/``scripts/analysis/train_locfeat.py``.  Maps WITHOUT it
          load exactly as before and ``EvidentialMap.loc_feats`` is ``None``.
        * ``loc_feat_obs`` (N,) -- how many training views each descriptor was
          averaged over (0 = never observed); the landmark-selection signal.
        * ``loc_feat_pca_mean`` (D_src,) / ``loc_feat_pca_basis`` (D_src, D) --
          the PCA that produced ``loc_feats`` from the encoder's native
          descriptors; the query side must apply the same projection.

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
        loc_feats=_get("loc_feats"),
        loc_feat_obs=_get("loc_feat_obs"),
        loc_feat_pca_mean=_get("loc_feat_pca_mean"),
        loc_feat_pca_basis=_get("loc_feat_pca_basis"),
        sh_coeffs=_get("sh_coeffs"),
    )
