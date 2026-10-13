# SPDX-License-Identifier: Apache-2.0
"""Camera operations, the MultiViewVideo dataset with lazy image loading, scene manifests, the
dataset converters (Neural3DV, OpenCV-style YAML, SelfCap), image resizing and undistortion, the
YAML camera reader, the frame cache and prefetcher, and the Fastest-10 frames of the ablations.

Spec: §1.4, D-51, D-61, D-69. Band: Foundation (§2.1).
"""

import math
import numbers
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from openftgs.io import read_scene_manifest
from openftgs.types import ImageRecord, SceneManifest, TimeNormalizer, frame_interval

__all__ = ["MultiViewVideo", "fastest_frames"]

_IMAGE_MODES = ("RGB", "RGBA")  # §1.4 "Images": 8-bit RGB, alpha dropped (D-34); Q-65
_MASK_MODES = ("1", "L")  # Q-65: bilevel or 8-bit grayscale masks
_MISSING_SHOWN = 5  # missing files named in the error


# Q-65: the members beyond §2.1's records, time, from_manifest and load_image.
class MultiViewVideo:
    """The image records of a scene manifest with their time normalizer, frame interval Δt_f
    and spatial scale S, from the training images only, and images and test-image masks loaded
    lazily (§2.1, §1.4, E14, D-27, D-46). Built from a parsed manifest (Q-64); `from_manifest`
    reads one."""

    records: list[ImageRecord]
    time: TimeNormalizer

    def __init__(self, manifest: SceneManifest) -> None:
        files = [record.image_path for record in manifest.records]
        files += [mask for mask in manifest.masks if mask is not None]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            shown = ", ".join(missing[:_MISSING_SHOWN])
            more = (
                f" and {len(missing) - _MISSING_SHOWN} more"
                if len(missing) > _MISSING_SHOWN
                else ""
            )
            raise ValueError(f"scene manifest {manifest.path}: missing files: {shown}{more}")
        self.manifest = manifest
        self.records = list(manifest.records)
        train = [record for record in self.records if record.split == "train"]
        # §1.4 "Time": τ₀ and τ₁ are the smallest and largest training timestamps.
        stamps = [record.timestamp for record in train]
        self.time = TimeNormalizer(min(stamps), max(stamps))
        # E14, D-46, Q-8: per training camera, grouped by name (Camera's == compares tensors).
        by_camera: dict[str, list[float]] = {}
        for record in train:
            by_camera.setdefault(record.camera.name, []).append(record.timestamp)
        self.frame_dt = frame_interval(by_camera, self.time)
        # D-27, Q-65: the distinct training poses' centers −R_wᵀt_w (E2), in float64.
        centers: dict[tuple[float, ...], None] = {}
        seen: set[int] = set()
        for record in train:
            if id(record.camera) not in seen:
                seen.add(id(record.camera))
                centers[_center(record)] = None
        self._centers = torch.tensor(list(centers), dtype=torch.float64)

    @classmethod
    def from_manifest(cls, path: Path) -> "MultiViewVideo":
        """The multi-view video of the ftgs-scene file at `path` (§2.2; Q-64)."""
        return cls(read_scene_manifest(Path(path)))

    def spatial_scale(self, factor: float) -> float:
        """S of D-27: `factor` × the largest distance from a distinct training camera center
        to the centroid of those centers (Q-65)."""
        if (
            isinstance(factor, bool)
            or not isinstance(factor, numbers.Real)
            or not (math.isfinite(factor) and factor > 0.0)
        ):
            raise ValueError(f"factor must be a finite number > 0, got {factor!r}")
        if self._centers.shape[0] < 2:
            raise ValueError(
                "the spatial scale needs two distinct training camera centers, got "
                f"{self._centers.shape[0]}"
            )
        offsets = self._centers - self._centers.mean(dim=0)
        return float(factor) * float(torch.linalg.vector_norm(offsets, dim=1).max())

    def _checked_index(self, index: int) -> int:
        if not 0 <= index < len(self.records):
            raise IndexError(f"image index {index} out of range [0, {len(self.records)})")
        return index

    def load_image(self, index: int) -> Tensor:
        """Image `index` as float32 [H, W, 3] in [0, 1], on the CPU: 8-bit values divided by
        255, alpha dropped (§1.4 "Images", D-34; Q-65)."""
        record = self.records[self._checked_index(index)]
        pixels = _read(record.image_path, _size(record), _IMAGE_MODES)
        return torch.from_numpy(pixels[..., :3].astype(np.float32)) / 255.0

    def load_mask(self, index: int) -> Tensor | None:
        """The dynamic-region mask of image `index` as float32 [H, W] in [0, 1], or None when
        it has none: 8-bit values divided by 255, bilevel ones 0 or 1 (§2.2, Q-61, Q-65)."""
        mask_path = self.manifest.masks[self._checked_index(index)]
        if mask_path is None:
            return None
        pixels = _read(mask_path, _size(self.records[index]), _MASK_MODES)
        if pixels.dtype == np.bool_:
            return torch.from_numpy(pixels.astype(np.float32))
        return torch.from_numpy(pixels.astype(np.float32)) / 255.0


def _center(record: ImageRecord) -> tuple[float, ...]:
    """E2's camera center −R_wᵀt_w of the record's pose, in float64."""
    w2c = record.camera.world_to_camera.to(torch.float64)
    return tuple((-(w2c[:3, :3].T @ w2c[:3, 3])).tolist())


def _size(record: ImageRecord) -> tuple[int, int]:
    """The (width, height) of the record's camera, as Pillow gives an image's size."""
    return record.camera.width, record.camera.height


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_BIT_DEPTH = 24  # the IHDR chunk's bit-depth byte: after the signature, length, type, size


def _png_bit_depth(path: Path) -> int | None:
    """The bit depth per channel of a PNG file (its IHDR chunk), or None for other formats.
    Pillow opens 16-bit RGB PNGs in the 8-bit mode "RGB", so the mode alone can't tell."""
    with path.open("rb") as file:
        head = file.read(_PNG_BIT_DEPTH + 1)
    if len(head) <= _PNG_BIT_DEPTH or not head.startswith(_PNG_SIGNATURE):
        return None
    return head[_PNG_BIT_DEPTH]


def _read(path: Path, size: tuple[int, int], modes: tuple[str, ...]) -> np.ndarray:
    """The pixels of an image file in one of `modes`, 8 bits per channel (1 for bilevel), of
    the given (width, height) (Q-65)."""
    with Image.open(path) as image:
        if image.mode not in modes:
            raise ValueError(f"{path}: mode must be one of {modes}, got {image.mode!r}")
        depth = _png_bit_depth(path)
        if depth is not None and depth != (1 if image.mode == "1" else 8):
            raise ValueError(f"{path}: images must have 8 bits per channel, got {depth}")
        if image.size != size:
            raise ValueError(f"{path}: size must be {size} (width, height), got {image.size}")
        return np.asarray(image)


# Q-59: in data, so the converter that stores the range in the manifest (D-69) can call it.
def fastest_frames(frames: Iterable[Tensor], *, count: int = 10) -> int:
    """The 0-based start s of the `count` consecutive frames whose consecutive frames differ
    most (D-69): the window [s, s + count) with the largest sum of the mean absolute
    differences between its consecutive frames, ties going to the lowest s (Q-59).

    `frames` is streamed, two frames at a time; each is [H, W, 3], uint8 or floating, all of
    one shape and dtype. The differences are taken in float64 over pixels and channels.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"count must be an int >= 2, got {count!r}")
    diffs: list[float] = []
    previous: Tensor | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    for frame in frames:
        if frame.dim() != 3 or frame.shape[-1] != 3:
            raise ValueError(f"frames must have shape [H, W, 3], got {list(frame.shape)}")
        if shape is None:
            shape, dtype = tuple(frame.shape), frame.dtype
        elif tuple(frame.shape) != shape:
            raise ValueError(f"frames must share one shape {list(shape)}, got {list(frame.shape)}")
        elif frame.dtype != dtype:
            # Q-59: uint8 is on a 0–255 scale, floating frames on 0–1.
            raise ValueError(f"frames must share one dtype {dtype}, got {frame.dtype}")
        current = frame.to(torch.float64)
        if previous is not None:
            diffs.append(float((current - previous).abs().mean()))
        previous = current
    if len(diffs) + 1 < count:
        raise ValueError(f"needs at least {count} frames, got {len(diffs) + 1 if shape else 0}")
    width = count - 1  # differences inside one window
    scores = torch.tensor(diffs, dtype=torch.float64).unfold(0, width, 1).sum(dim=1)
    # Q-59: the first maximum, so ties go to the lowest start.
    return int(torch.nonzero(scores == scores.max())[0, 0])
