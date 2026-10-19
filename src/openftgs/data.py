# SPDX-License-Identifier: Apache-2.0
"""Camera operations, the MultiViewVideo dataset with lazy image loading, scene manifests, the
dataset converters (Neural3DV, OpenCV-style YAML, SelfCap), image resizing and undistortion, the
YAML camera reader, the frame cache and prefetcher, and the Fastest-10 frames of the ablations.

Spec: §1.4, D-51, D-61, D-69. Band: Foundation (§2.1).
"""

import concurrent.futures
import contextlib
import functools
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import numbers
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from openftgs._stages import recorder, stage
from openftgs.io import read_scene_manifest, write_scene_manifest
from openftgs.types import Camera, ImageRecord, SceneManifest, TimeNormalizer, frame_interval

__all__ = [
    "FrameCache",
    "MultiViewVideo",
    "Prefetcher",
    "camera_from_llff",
    "camera_from_opencv",
    "convert_n3dv",
    "convert_opencv_yaml",
    "convert_selfcap",
    "fastest_frames",
    "read_opencv_yaml",
    "resize_area",
    "resize_camera",
    "resized_size",
    "undistort_image",
    "undistortion_map",
]

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
        if not train:  # read_scene_manifest refuses such a file; a hand-built manifest may not
            raise ValueError(f"scene manifest {manifest.path}: needs at least one training image")
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


# Cameras (§1.4 "World and camera frames", D-28, D-33; Q-84, Q-89).

# Q-105: max |RᵀR − I| accepted from a calibration file. A matrix within D-63's 1e-6 of a
# rotation in every entry has |RᵀR − I| ≤ 2√3·1e-6 + 3e-12 ≈ 3.5e-6, so 1e-5 refuses only
# matrices that are not rotations; R is stored as read.
_ROTATION_TOLERANCE = 1e-5


def _matrix64(value: Any, shape: tuple[int, ...], what: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be a finite {shape} array, got shape {array.shape}")
    return array


def _camera(
    K: np.ndarray, rotation: np.ndarray, translation: np.ndarray, width: int, height: int, name: str
) -> Camera:
    """A Camera from float64 parts in the continuous convention, stored as float32 (§2.1)."""
    for size, what in ((width, "width"), (height, "height")):
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 1:
            raise ValueError(f"{what} must be an int >= 1, got {size!r}")
    if K[0, 1] != 0.0 or K[1, 0] != 0.0 or K[2].tolist() != [0.0, 0.0, 1.0]:
        raise ValueError("K must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew)")
    if not (K[0, 0] > 0.0 and K[1, 1] > 0.0):
        raise ValueError(f"K must have fx > 0 and fy > 0, got {K[0, 0]} and {K[1, 1]}")
    error = np.abs(rotation.T @ rotation - np.eye(3)).max()
    if error > _ROTATION_TOLERANCE or np.linalg.det(rotation) <= 0.0:
        raise ValueError("R must be a rotation matrix (orthonormal, determinant +1)")
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = translation
    return Camera(
        K=torch.from_numpy(K).to(torch.float32),
        world_to_camera=torch.from_numpy(w2c).to(torch.float32),
        width=int(width),
        height=int(height),
        name=name,
    )


def camera_from_opencv(
    K: Any, R: Any, T: Any, *, width: int, height: int, name: str = ""
) -> Camera:
    """A camera from an OpenCV calibration (Q-84): K with pixel centres at integers gains +0.5
    on cx and cy (§1.4), and R, T go straight into world_to_camera, x_cam = R·x + T (D-63)."""
    K64 = _matrix64(K, (3, 3), "K").copy()
    K64[0, 2] += 0.5
    K64[1, 2] += 0.5
    translation = _matrix64(np.asarray(T, dtype=np.float64).reshape(-1), (3,), "T")
    return _camera(K64, _matrix64(R, (3, 3), "R"), translation, width, height, name)


def camera_from_llff(pose: Any, *, name: str = "") -> Camera:
    """A camera from a row of Neural3DV's poses_bounds.npy, or its 3×5 matrix (D-28; Q-84):
    the camera-to-world rotation's columns point down, right and back, so the OpenCV rotation
    is [col₁, col₀, −col₂]; then the camera centre and (H, W, focal); fx = fy = focal and
    c = (W/2, H/2), already the image centre in the continuous convention."""
    array = np.asarray(pose, dtype=np.float64)
    if array.shape == (17,):
        array = array[:15]
    if array.shape == (15,):
        array = array.reshape(3, 5)
    matrix = _matrix64(array, (3, 5), "an LLFF pose (17 values, 15 values or 3x5)")
    down, right, back, centre, hwf = (matrix[:, k] for k in range(5))
    height, width, focal = hwf
    if height != round(height) or width != round(width):
        raise ValueError(f"an LLFF pose's H and W must be integers, got {height} and {width}")
    rotation = np.stack([right, down, -back])  # world-to-camera: rows are the camera's axes
    K = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    return _camera(K, rotation, -rotation @ centre, int(width), int(height), name)


def resized_size(width: int, height: int, ratio: float) -> tuple[int, int]:
    """§1.4's W′ = round(W·r) and H′ = round(H·r), an exact half rounding to even as OpenCV's
    resize by a ratio does, at least 1 (Q-89)."""
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, numbers.Real)
        or not (math.isfinite(ratio) and ratio > 0.0)
    ):
        raise ValueError(f"ratio must be a finite number > 0, got {ratio!r}")
    return max(1, round(width * ratio)), max(1, round(height * ratio))  # half to even


def resize_camera(camera: Camera, width: int, height: int) -> Camera:
    """The camera of the image resized to width × height: fx and cx scale by W′/W, fy and cy by
    H′/H (§1.4, D-33); the pose is unchanged."""
    K = camera.K.to(torch.float64).numpy().copy()
    K[0, [0, 2]] *= width / camera.width
    K[1, [1, 2]] *= height / camera.height
    w2c = camera.world_to_camera.to(torch.float64).numpy()
    return _camera(K, w2c[:3, :3], w2c[:3, 3], width, height, camera.name)


# Area resizing (D-33, D-61; Q-84, Q-89, Q-90).


def _area_weights(size: int, new_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Banded weights of D-33's area averaging along one axis: output pixel j covers
    [j·s, (j + 1)·s) of the input, s = size/new_size, each input pixel a unit square; returns
    the first input index of each output pixel and the weights of its taps, in float64."""
    scale = size / new_size
    edges = np.arange(new_size + 1, dtype=np.float64) * scale
    edges[-1] = size  # exact, whatever the rounding of new_size·scale
    first = np.floor(edges[:-1]).astype(np.int64)
    taps = int(math.ceil(scale)) + 1
    index = first[:, None] + np.arange(taps)[None, :]
    overlap = np.minimum(index + 1, edges[1:, None]) - np.maximum(index, edges[:-1, None])
    weights = np.clip(overlap, 0.0, None) / scale
    weights[index >= size] = 0.0
    return np.minimum(index, size - 1), weights


def resize_area(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """D-33's area resizing of an [H, W] or [H, W, C] array to width × height: every output
    pixel is the mean of the input over its footprint, input pixels being constant squares,
    separably in float64 (Q-84). uint8 input gives uint8 output rounded half to even; floating
    input gives float64."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be [H, W] or [H, W, C], got shape {image.shape}")
    for size, what in ((width, "width"), (height, "height")):
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 1:
            raise ValueError(f"{what} must be an int >= 1, got {size!r}")
    eight_bit = image.dtype == np.uint8
    if not eight_bit and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"image must be uint8 or floating, got {image.dtype}")
    values = image.astype(np.float64)
    rows, row_weights = _area_weights(values.shape[0], height)
    cols, col_weights = _area_weights(values.shape[1], width)
    # Columns, then rows: each a sum over the few taps of the footprint (§2.4 rule 1: no loop
    # over pixels).
    shape = (1, -1) + (1,) * (values.ndim - 2)
    by_col = sum(
        values[:, cols[:, k]] * col_weights[:, k].reshape(shape) for k in range(cols.shape[1])
    )
    shape = (-1,) + (1,) * (values.ndim - 1)
    out: np.ndarray = np.asarray(
        sum(by_col[rows[:, k]] * row_weights[:, k].reshape(shape) for k in range(rows.shape[1]))
    )
    if eight_bit:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


# Undistortion (D-61; Q-84, Q-90): OpenCV's documented camera model (calib3d), written from its
# documentation.

_COEFFICIENT_COUNTS = (0, 4, 5, 8, 12)


def _coefficients(dist: Any) -> np.ndarray:
    """The 12 coefficients k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 of OpenCV's model,
    zero-padded from 0, 4, 5, 8 or 12 given; the tilted model (14) is refused (D-61)."""
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    if values.size == 14:
        raise ValueError("the tilted (14-coefficient) distortion model is not supported (D-61)")
    if values.size not in _COEFFICIENT_COUNTS:
        raise ValueError(
            f"distortion must have 0, 4, 5, 8 or 12 coefficients, got {values.size} coefficients"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("distortion coefficients must be finite")
    return np.concatenate([values, np.zeros(12 - values.size)])


def undistortion_map(K: Any, dist: Any, width: int, height: int) -> np.ndarray:
    """For every pixel of the undistorted width × height image, its source position (x, y) in
    the distorted image, [H, W, 2] float64, by the steps of OpenCV's initUndistortRectifyMap
    with R = I and the new camera matrix equal to K (D-61; Q-84). K uses OpenCV's integer
    pixel centres, as calibration files do."""
    K64 = _matrix64(K, (3, 3), "K")
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = _coefficients(dist)
    fx, fy, cx, cy = K64[0, 0], K64[1, 1], K64[0, 2], K64[1, 2]
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4
    return np.stack([fx * xd + cx, fy * yd + cy], axis=-1)


def _sample_bilinear(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Bilinear samples of an [H, W] or [H, W, C] array at [h, w, 2] positions (x, y), pixel
    centres at integers, pixels outside the image counting as 0 (D-61), in float64."""
    height, width = image.shape[:2]
    values = image.astype(np.float64)
    x, y = positions[..., 0], positions[..., 1]
    x0, y0 = np.floor(x), np.floor(y)
    wx, wy = x - x0, y - y0
    out = np.zeros(positions.shape[:2] + image.shape[2:], dtype=np.float64)
    extra = (...,) + (None,) * (image.ndim - 2)
    for dy, wgt_y in ((0, 1 - wy), (1, wy)):
        for dx, wgt_x in ((0, 1 - wx), (1, wx)):
            xi, yi = x0 + dx, y0 + dy
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            xs = np.clip(xi, 0, width - 1).astype(np.int64)
            ys = np.clip(yi, 0, height - 1).astype(np.int64)
            out += values[ys, xs] * (wgt_x * wgt_y * valid)[extra]
    return out


def undistort_image(image: np.ndarray, K: Any, dist: Any) -> np.ndarray:
    """The undistorted image, same size, K unchanged (D-61; Q-84): bilinear samples at
    `undistortion_map`'s positions, zeros where the source lies outside. uint8 input gives uint8
    output rounded half to even; floating input gives float64 (Q-90: the caller may resize the
    float64 values before rounding). Zero coefficients return the input's values unchanged."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be [H, W] or [H, W, C], got shape {image.shape}")
    eight_bit = image.dtype == np.uint8
    if not eight_bit and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"image must be uint8 or floating, got {image.dtype}")
    coefficients = _coefficients(dist)
    if not coefficients.any():
        _matrix64(K, (3, 3), "K")
        return image.copy() if eight_bit else image.astype(np.float64)
    height, width = image.shape[:2]
    out = _sample_bilinear(image, undistortion_map(K, coefficients, width, height))
    if eight_bit:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


# The OpenCV-style YAML reader (D-61; Q-84, Q-91): PyYAML with a constructor for OpenCV's
# matrices, written from the format OpenCV's FileStorage writes (the probe in the source log).

_OPENCV_DTYPES = {
    "u": np.uint8,
    "c": np.int8,
    "w": np.uint16,
    "s": np.int16,
    "i": np.int32,
    "f": np.float32,
    "d": np.float64,
    "h": np.float16,
}
_DT_PATTERN = re.compile(r"([1-9][0-9]*)?([ucwsifdh])")


class _OpenCVLoader(yaml.SafeLoader):  # type: ignore[misc]  # PyYAML ships no types
    """PyYAML's safe loader plus OpenCV's matrices and its `.Nan` spelling (Q-91)."""


def _opencv_matrix(loader: yaml.SafeLoader, node: yaml.MappingNode) -> np.ndarray:
    fields = loader.construct_mapping(node, deep=True)
    dt = fields.get("dt")
    match = _DT_PATTERN.fullmatch(str(dt))
    if match is None:
        raise yaml.constructor.ConstructorError(
            None, None, f"unknown matrix dt {dt!r}", node.start_mark
        )
    channels = int(match.group(1) or 1)
    try:
        shape: tuple[int, ...] = (int(fields["rows"]), int(fields["cols"]))
    except (KeyError, TypeError, ValueError) as err:
        raise yaml.constructor.ConstructorError(
            None, None, f"a matrix needs integer rows and cols: {err}", node.start_mark
        ) from None
    if channels > 1:
        shape += (channels,)
    data = fields.get("data", [])
    if not isinstance(data, list) or len(data) != math.prod(shape):
        raise yaml.constructor.ConstructorError(
            None, None, f"matrix data must hold {math.prod(shape)} values", node.start_mark
        )
    dtype = _OPENCV_DTYPES[match.group(2)]
    wide = np.float64 if np.issubdtype(dtype, np.floating) else np.int64
    try:
        return np.array(data, dtype=wide).astype(dtype).reshape(shape)
    except (TypeError, ValueError) as err:
        raise yaml.constructor.ConstructorError(
            None, None, f"matrix data must be numbers: {err}", node.start_mark
        ) from None


_OpenCVLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix)
# OpenCV writes NaN as `.Nan`, which YAML 1.1's float pattern (`.nan`, `.NaN`, `.NAN`) misses.
_OpenCVLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(r"^[-+]?\.Nan$"), list("-+.")
)


def read_opencv_yaml(path: Path) -> dict[str, Any]:
    """The top-level mapping of a file written by OpenCV's FileStorage (D-61; Q-84, Q-91):
    `!!opencv-matrix` entries become NumPy arrays of shape (rows, cols) or (rows, cols,
    channels) in the dtype their `dt` names, so values equal what OpenCV reads back; other
    values are as PyYAML reads them. A `%YAML:1.0` first line, which isn't valid YAML, is
    dropped."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as err:
        raise ValueError(f"{path}: not a readable text file: {err}") from None
    if text.startswith("%YAML:"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    try:
        data = yaml.load(text, Loader=_OpenCVLoader)
    except yaml.YAMLError as err:
        raise ValueError(f"{path}: not an OpenCV-style YAML file: {err}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return data


# Dataset conversion (§1.4 "Dataset protocols", D-48, D-80, D-81; Q-86 to Q-91, Q-93 to Q-101):
# ffmpeg runs as a separate program, frames are processed and encoded in a process pool (§2.4
# "Threads and processes").

_HWACCELS = ("none", "cuda")
# D-81: the software format hwdownload hands over, from the stream's pixel format.
_CUDA_FORMATS = {
    "yuv420p": "nv12",
    "yuvj420p": "nv12",
    "nv12": "nv12",
    "yuv420p10le": "p010le",
    "p010le": "p010le",
    "yuv444p": "yuv444p",
    "yuvj444p": "yuv444p",
}
_PNG_LEVEL = 1  # §2.4: a fast PNG level; PNG is lossless, so pixels don't depend on it


@dataclass(frozen=True)
class _Tools:
    ffmpeg: str
    ffprobe: str
    version: str


@dataclass(frozen=True)
class _VideoInfo:
    width: int
    height: int
    pix_fmt: str
    frames: int


def _program(name: str, what: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise ValueError(
            f"{what} not found: {name!r} (D-48: the converters need ffmpeg 5.1 or later)"
        )
    return found


def _version_line(program: str, what: str) -> str:
    try:
        result = subprocess.run([program, "-version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        raise ValueError(f"{what} at {program!r} doesn't run: {err}") from None
    lines = result.stdout.splitlines()
    return lines[0].strip() if lines else ""


def _tools(ffmpeg: str, ffprobe: str | None) -> _Tools:
    """D-80: ffprobe defaults to the one beside a path-given ffmpeg, else the one on PATH."""
    resolved = _program(ffmpeg, "ffmpeg")
    if ffprobe is None:
        separators = [os.sep] + ([os.altsep] if os.altsep else [])
        ffprobe = (
            str(Path(resolved).with_name("ffprobe"))
            if any(sep in ffmpeg for sep in separators)
            else "ffprobe"
        )
    probe = _program(ffprobe, "ffprobe")
    version = _version_line(resolved, "ffmpeg")
    _version_line(probe, "ffprobe")
    return _Tools(resolved, probe, version)


def _probe(tools: _Tools, video: Path) -> _VideoInfo:
    """Width, height, pixel format and frame count of the first video stream (§1.4: read with
    ffprobe from the same installation); `-count_frames` when the container states no count."""

    def stream(extra: list[str], entries: str) -> dict[str, Any]:
        command = [
            tools.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *extra,
            "-show_entries",
            f"stream={entries}",
            "-of",
            "json",
            str(video),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or result.stderr.strip():
            raise ValueError(f"ffprobe failed on {video}: {result.stderr.strip()}")
        streams = json.loads(result.stdout or "{}").get("streams") or []
        if not streams:
            raise ValueError(f"{video}: no video stream")
        found: dict[str, Any] = streams[0]
        return found

    info = stream([], "width,height,pix_fmt,nb_frames")
    try:
        frames = int(info["nb_frames"])
    except (KeyError, TypeError, ValueError):
        frames = int(stream(["-count_frames"], "nb_read_frames").get("nb_read_frames", 0))
    return _VideoInfo(int(info["width"]), int(info["height"]), str(info.get("pix_fmt", "")), frames)


def _decode_command(
    tools: _Tools,
    video: Path,
    info: _VideoInfo,
    *,
    count: int,
    window: tuple[int, int] | None,
    hwaccel: str,
) -> list[str]:
    """The ffmpeg command of §1.4: the first `count` frames, or with `window` = (N0, N1) the
    decode indices N0 ≤ n < N1, as raw RGB24 on stdout (D-48). D-81: with CUDA, hardware frames
    downloaded explicitly, so a decode that can't use CUDA fails instead of falling back."""
    command = [tools.ffmpeg, "-v", "error"]
    filters: list[str] = []
    if hwaccel == "cuda":
        software = _CUDA_FORMATS.get(info.pix_fmt)
        if software is None:
            raise ValueError(
                f"{video}: --hwaccel cuda can't decode pixel format {info.pix_fmt!r}; "
                "use --hwaccel none"
            )
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        # Back to the stream's own format, so the RGB conversion is the CPU path's: the frames
        # then match CPU decoding (U-27).
        native = "" if software == info.pix_fmt else f",format={info.pix_fmt}"
        filters.append(f"hwdownload,format={software}{native}")
    command += ["-i", str(video), "-map", "0:v:0"]
    if window is not None:
        filters.append(f"select='between(n\\,{window[0]}\\,{window[1] - 1})'")
    if filters:
        command += ["-vf", ",".join(filters)]
    if window is not None:
        command += ["-fps_mode", "passthrough", "-frames:v", str(count)]
    else:
        command += ["-frames:v", str(count), "-fps_mode", "passthrough"]
    return command + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _read_exactly(stream: IO[bytes], size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode(
    command: list[str], info: _VideoInfo, count: int, video: Path, *, camera: str
) -> Generator[np.ndarray, None, None]:
    """The frames ffmpeg writes, exactly W·H·3 bytes each (§1.4), each read as the stage
    "convert.decode" keyed by `camera` (§3.4 B-8). Q-93: a short read, a nonzero exit or any
    message at `-v error` is a ValueError."""
    size = info.width * info.height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    errors: list[bytes] = []
    reader = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    read = 0
    try:
        while read < count:
            with stage("convert.decode", key=camera):
                data = _read_exactly(process.stdout, size)
            if len(data) < size:
                break
            read += 1
            yield np.frombuffer(data, dtype=np.uint8).reshape(info.height, info.width, 3)
        process.stdout.read()
    finally:
        if process.poll() is None and read < count:
            process.kill()
        process.wait()
        reader.join()
        process.stdout.close()
        process.stderr.close()
    message = errors[0].decode(errors="replace").strip() if errors else ""
    if read < count or process.returncode != 0 or message:
        raise ValueError(
            f"{video}: ffmpeg gave {read} of {count} frames (exit code {process.returncode})"
            + (f": {message}" if message else "")
            + " (Q-93: errors in frames decoded before a window count too)"
        )


def _pool_init() -> None:
    torch.set_num_threads(1)  # §2.4: one thread per pool worker


def _write_frame(
    source: np.ndarray | str,
    path: str,
    size: tuple[int, int],
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> float:
    """One output frame (a pool task): undistorted with K and dist in OpenCV's convention when
    they distort, resized to `size` by area averaging with one 8-bit rounding (Q-90), and
    written as 8-bit RGB PNG. Returns the task's wall time in seconds, which the converter
    records as the stage "convert.encode" (§3.4 B-8): the pool's process has no clock."""
    began = time.perf_counter()
    with stage("convert.encode"):
        frame = source if isinstance(source, np.ndarray) else _read_rgb(Path(source))
        width, height = size
        resized = (frame.shape[1], frame.shape[0]) != (width, height)
        if dist is not None and _coefficients(dist).any():
            values = undistort_image(frame.astype(np.float64), K, dist)
            if resized:
                values = resize_area(values, width, height)
            out = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        elif resized:
            out = resize_area(frame, width, height)
        else:
            out = frame
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(out), "RGB").save(target, compress_level=_PNG_LEVEL)
    return time.perf_counter() - began


def _read_rgb(path: Path) -> np.ndarray:
    """An 8-bit image file as [H, W, 3] uint8 (alpha dropped; §1.4 "Images"); 16-bit PNGs are
    refused, as `MultiViewVideo.load_image` refuses them (Q-65)."""
    try:
        with Image.open(path) as image:
            if image.mode not in _IMAGE_MODES:
                raise ValueError(f"{path}: mode must be one of {_IMAGE_MODES}, got {image.mode!r}")
            depth = _png_bit_depth(path)
            if depth is not None and depth != 8:
                raise ValueError(f"{path}: images must have 8 bits per channel, got {depth}")
            return np.asarray(image)[..., :3]
    except OSError as err:  # Pillow's UnidentifiedImageError included
        raise ValueError(f"{path}: not a readable image: {err}") from None


@dataclass
class _CameraJob:
    """One camera's frames: decoded from `video`, or read from `images`, then processed into
    `outputs` (frame_index order)."""

    name: str
    outputs: list[Path]
    size: tuple[int, int]
    video: Path | None = None
    info: _VideoInfo | None = None
    window: tuple[int, int] | None = None
    images: list[Path] = field(default_factory=list)
    K: np.ndarray | None = None
    dist: np.ndarray | None = None


def _worker_count(jobs: int) -> int:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
        raise ValueError(f"jobs must be an int >= 0, got {jobs!r}")
    return jobs or max(1, (os.cpu_count() or 1) // 4)  # §2.4: one per four cores


def _record_encoded(
    encoded: Callable[[float], None], done: concurrent.futures.Future[float]
) -> None:
    """Record a pool task's wall time once it has finished without error (§3.4 B-8)."""
    if not done.cancelled() and done.exception() is None:
        encoded(done.result())


def _run_jobs(
    camera_jobs: Sequence[_CameraJob], tools: _Tools | None, *, hwaccel: str, workers: int
) -> dict[str, list[str]]:
    """Decode up to `workers` cameras at once, each frame processed and written by a process
    pool (§2.4); outputs don't depend on `workers`. Returns each camera's ffmpeg command."""
    commands: dict[str, list[str]] = {}
    total = sum(len(job.outputs) for job in camera_jobs)
    pool_size = max(1, min(os.cpu_count() or 1, total))
    in_flight = threading.BoundedSemaphore(2 * pool_size)  # bounds decoded frames held in memory
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        pool_size, mp_context=context, initializer=_pool_init
    ) as pool:

        def submit(
            source: np.ndarray | str,
            job: _CameraJob,
            k: int,
            encoded: Callable[[float], None] | None,
        ) -> concurrent.futures.Future[float]:
            in_flight.acquire()
            try:
                future = pool.submit(
                    _write_frame, source, str(job.outputs[k]), job.size, job.K, job.dist
                )
            except BaseException:
                in_flight.release()
                raise
            future.add_done_callback(lambda _: in_flight.release())
            if encoded is not None:  # §3.4 B-8: the time the pool's process measured
                future.add_done_callback(functools.partial(_record_encoded, encoded))
            return future

        def camera(job: _CameraJob) -> list[concurrent.futures.Future[float]]:
            encoded = recorder("convert.encode", key=job.name)
            futures = []
            if job.video is not None:
                assert tools is not None and job.info is not None
                count = len(job.outputs)
                command = _decode_command(
                    tools, job.video, job.info, count=count, window=job.window, hwaccel=hwaccel
                )
                commands[job.name] = command
                decoded = _decode(command, job.info, count, job.video, camera=job.name)
                with contextlib.closing(decoded) as frames:
                    for k, frame in enumerate(frames):
                        futures.append(submit(frame, job, k, encoded))
            else:
                for k, image in enumerate(job.images):
                    futures.append(submit(str(image), job, k, encoded))
            return futures

        with concurrent.futures.ThreadPoolExecutor(min(workers, len(camera_jobs)) or 1) as threads:
            per_camera = list(threads.map(camera, camera_jobs))
        for futures in per_camera:
            for future in futures:
                future.result()
    return {job.name: commands[job.name] for job in camera_jobs if job.name in commands}


def _prepare_out(dst: Path) -> tuple[Path, bool]:
    """Q-88: OUT is absent or an empty directory; returns it and whether it was created."""
    dst = Path(dst)
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise ValueError(f"{dst}: the output must be absent or an empty directory")
    created = not dst.exists()
    dst.mkdir(parents=True, exist_ok=True)
    return dst, created


def _discard(out: Path, created: bool) -> None:
    """Q-88 (owner answer): after a failure, remove everything the conversion wrote. OUT was
    absent or empty when it started, so all of it is the conversion's own."""
    if created:
        shutil.rmtree(out, ignore_errors=True)
        return
    for child in out.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


def _check_frames(frames: int) -> None:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError(f"frames must be an int >= 1, got {frames!r}")


def _check_hwaccel(hwaccel: str) -> None:
    if hwaccel not in _HWACCELS:
        raise ValueError(f"hwaccel must be one of {_HWACCELS}, got {hwaccel!r}")


def _ffmpeg_metadata(tools: _Tools, hwaccel: str, commands: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "path": tools.ffmpeg,
        "ffprobe": tools.ffprobe,
        "version": tools.version,
        "hwaccel": hwaccel,
        "decoder": "cuda" if hwaccel == "cuda" else "cpu",  # D-81: CUDA runs or fails
        "commands": commands,
    }


_N3DV_FPS = 30  # §1.4: Neural3DV frame k at k/30 s
_N3DV_TEST = "cam00"  # D-29
_N3DV_VIDEO = re.compile(r"cam\d+\.mp4")


def convert_n3dv(
    src: Path,
    dst: Path,
    *,
    frames: int = 300,
    resize: float = 0.5,
    ffmpeg: str = "ffmpeg",
    hwaccel: Literal["none", "cuda"] = "none",
    jobs: int = 0,
    ffprobe: str | None = None,
) -> Path:
    """Convert a Neural3DV scene (D-28, D-48, D-80; Q-88, Q-94, Q-95): `poses_bounds.npy`'s rows
    belong to the `camXX.mp4` files in name order; the first `frames` frames of each video,
    decoded by the user's ffmpeg, area-resized by `resize` and written as PNG; frame k at k/30 s;
    cam00 the test view. Returns the path of OUT/scene.json."""
    src = Path(src)
    _check_frames(frames)
    _check_hwaccel(hwaccel)
    workers = _worker_count(jobs)
    resized_size(1, 1, resize)
    poses_path = src / "poses_bounds.npy"
    if not poses_path.is_file():
        raise ValueError(f"{src}: no poses_bounds.npy")
    try:
        poses = np.load(poses_path, allow_pickle=False)
    except (OSError, ValueError) as err:
        raise ValueError(f"{poses_path}: not a readable array: {err}") from None
    videos = sorted(p for p in src.iterdir() if _N3DV_VIDEO.fullmatch(p.name))
    if poses.ndim != 2 or poses.shape[1] != 17 or poses.shape[0] != len(videos):
        raise ValueError(
            f"{src}: poses_bounds.npy must hold one 17-value row per camXX.mp4 (Q-95), got "
            f"{list(poses.shape)} for {len(videos)} videos"
        )
    out, created = _prepare_out(dst)
    work = out / ".work"
    try:
        tools = _tools(ffmpeg, ffprobe)
        cameras: list[Camera] = []
        camera_jobs: list[_CameraJob] = []
        bounds: dict[str, list[float]] = {}
        for row, video in zip(poses, videos, strict=True):
            name = video.stem
            full = camera_from_llff(row, name=name)
            info = _probe(tools, video)
            if (info.width, info.height) != (full.width, full.height):
                raise ValueError(
                    f"{video}: size {info.width}x{info.height} differs from its poses_bounds row's "
                    f"{full.width}x{full.height} (Q-95)"
                )
            if info.frames < frames:
                raise ValueError(
                    f"{video}: {info.frames} frames, fewer than the {frames} asked for"
                )
            size = resized_size(full.width, full.height, resize)
            cameras.append(resize_camera(full, *size))
            outputs = [out / "images" / name / f"{k:06d}.png" for k in range(frames)]
            camera_jobs.append(_CameraJob(name, outputs, size, video=video, info=info))
            bounds[name] = [float(row[15]), float(row[16])]
        commands = _run_jobs(camera_jobs, tools, hwaccel=hwaccel, workers=workers)
        records = [
            ImageRecord(
                camera, k / _N3DV_FPS, path, "test" if camera.name == _N3DV_TEST else "train", k
            )
            for camera, job in zip(cameras, camera_jobs, strict=True)
            for k, path in enumerate(job.outputs)
        ]
        metadata = {
            "source": f"n3dv/{src.name}",  # Q-62
            "converter": "n3dv",
            "producer": _producer(),
            "resize": resize,
            "fps": _N3DV_FPS,
            "frames": frames,
            "ffmpeg": _ffmpeg_metadata(tools, hwaccel, commands),
            "bounds": bounds,
        }
        write_scene_manifest(out / "scene.json", records, time_unit="s", metadata=metadata)
    except BaseException:
        _discard(out, created)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out / "scene.json"


# OpenCV-style YAML calibrations (D-61, D-63; Q-96, Q-97).

_RODRIGUES_TOLERANCE = 1e-6  # D-63: Rot against Rodrigues(R), max-abs
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    """The rotation matrix of a Rodrigues vector, in float64."""
    theta = float(np.linalg.norm(vector))
    if theta == 0.0:
        return np.eye(3)
    k = vector / theta
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    matrix: np.ndarray = (
        np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * cross
    )
    return matrix


@dataclass(frozen=True)
class _Calibrated:
    """One camera of an OpenCV-style calibration, K in OpenCV's integer-centre convention."""

    name: str
    K: np.ndarray
    dist: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    size: tuple[int, int] | None  # (W, H) from W_ and H_, when present


def _calibration(intri_path: Path, extri_path: Path) -> list[_Calibrated]:
    """The cameras of intri.yml and extri.yml (D-61, D-63; Q-96): `names`, K_, D_ or dist_,
    optional W_ and H_, an optional ccm_ that must be the identity, Rot_ and/or R_ (agreeing
    within 1e-6 when both), T_."""
    for path in (intri_path, extri_path):
        if not path.is_file():
            raise ValueError(f"{path}: calibration file not found")
    intri, extri = read_opencv_yaml(intri_path), read_opencv_yaml(extri_path)
    names = intri.get("names", extri.get("names"))
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{intri_path}: `names` must be a non-empty list of strings")

    def get(source: dict[str, Any], key: str, shape: tuple[int, ...], where: Path) -> np.ndarray:
        if key not in source:
            raise ValueError(f"{where}: no {key}")
        try:
            values = np.asarray(source[key], dtype=np.float64).reshape(shape)
        except (TypeError, ValueError) as err:
            raise ValueError(f"{where}: {key} must hold {shape} numbers: {err}") from None
        return _matrix64(values, shape, key)

    cameras = []
    for name in names:
        K = get(intri, f"K_{name}", (3, 3), intri_path)
        dist_key = f"D_{name}" if f"D_{name}" in intri else f"dist_{name}"
        try:
            dist = (
                np.asarray(intri[dist_key], dtype=np.float64).reshape(-1)
                if dist_key in intri
                else np.zeros(0)
            )
        except (TypeError, ValueError) as err:
            raise ValueError(f"{intri_path}: {dist_key} must hold numbers: {err}") from None
        _coefficients(dist)
        ccm = intri.get(f"ccm_{name}")
        if ccm is not None and not np.array_equal(np.asarray(ccm, dtype=np.float64), np.eye(3)):
            raise ValueError(f"{intri_path}: ccm_{name} is not the identity (D-63)")
        rotations = []
        if f"Rot_{name}" in extri:
            rotations.append(get(extri, f"Rot_{name}", (3, 3), extri_path))
        if f"R_{name}" in extri:
            rotations.append(_rodrigues(get(extri, f"R_{name}", (3,), extri_path)))
        if not rotations:
            raise ValueError(f"{extri_path}: no Rot_{name} or R_{name}")
        if len(rotations) == 2 and np.abs(rotations[0] - rotations[1]).max() > _RODRIGUES_TOLERANCE:
            raise ValueError(
                f"{extri_path}: Rot_{name} and Rodrigues(R_{name}) differ by more than "
                f"{_RODRIGUES_TOLERANCE} (D-63)"
            )
        translation = get(extri, f"T_{name}", (3,), extri_path)
        size = None
        if f"W_{name}" in intri or f"H_{name}" in intri:
            width, height = intri.get(f"W_{name}"), intri.get(f"H_{name}")
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in (width, height)):
                raise ValueError(f"{intri_path}: W_{name} and H_{name} must both be integers")
            size = (int(width), int(height))  # type: ignore[arg-type]
        cameras.append(_Calibrated(name, K, dist, rotations[0], translation, size))
    return cameras


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except OSError as err:
        raise ValueError(f"{path}: not a readable image: {err}") from None


def convert_opencv_yaml(
    src: Path,
    dst: Path,
    *,
    fps: float,
    frames: int,
    resize: float,
    ffmpeg: str = "ffmpeg",
    hwaccel: Literal["none", "cuda"] = "none",
    jobs: int = 0,
    ffprobe: str | None = None,
    test_camera: str | None = None,
) -> Path:
    """Convert a dataset with OpenCV-style YAML calibrations, such as ENeRF-Outdoor (§1.4,
    D-61, D-63, D-80; Q-96): intri.yml and extri.yml in SRC, frames from `images/<camera>/`
    (the first `frames` files by name) or `videos/<camera>.mp4`; frames undistorted with the
    file's K, K shifted by +0.5, frames and K resized; frame k at k/fps s; `test_camera` the
    test view. Returns the path of OUT/scene.json."""
    src = Path(src)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, numbers.Real)
        or not (math.isfinite(fps) and fps > 0)
    ):
        raise ValueError(f"fps must be a finite number > 0, got {fps!r}")
    _check_frames(frames)
    _check_hwaccel(hwaccel)
    workers = _worker_count(jobs)
    resized_size(1, 1, resize)
    calibrated = _calibration(src / "intri.yml", src / "extri.yml")
    if test_camera is not None and test_camera not in [c.name for c in calibrated]:
        raise ValueError(f"test_camera {test_camera!r} is not among the calibrated cameras")
    out, created = _prepare_out(dst)
    try:
        tools: _Tools | None = None
        cameras: list[Camera] = []
        camera_jobs: list[_CameraJob] = []
        for cal in calibrated:
            folder, video = src / "images" / cal.name, src / "videos" / f"{cal.name}.mp4"
            outputs = [out / "images" / cal.name / f"{k:06d}.png" for k in range(frames)]
            if folder.is_dir():
                images = _image_files(folder)[:frames]
                if len(images) < frames:
                    raise ValueError(f"{folder}: {len(images)} images, fewer than {frames}")
                sizes = {_image_size(p) for p in images}
                if len(sizes) != 1:
                    raise ValueError(f"{folder}: the images differ in size: {sorted(sizes)}")
                (width, height), info = sizes.pop(), None
            elif video.is_file():
                tools = tools or _tools(ffmpeg, ffprobe)
                info = _probe(tools, video)
                if info.frames < frames:
                    raise ValueError(f"{video}: {info.frames} frames, fewer than {frames}")
                width, height, images = info.width, info.height, []
            else:
                raise ValueError(f"{src}: no images/{cal.name}/ or videos/{cal.name}.mp4")
            if cal.size is not None and cal.size != (width, height):
                raise ValueError(
                    f"{src}: camera {cal.name}'s W_ and H_ {cal.size} differ from its frames' "
                    f"{(width, height)}"
                )
            size = resized_size(width, height, resize)
            full = camera_from_opencv(
                cal.K, cal.rotation, cal.translation, width=width, height=height, name=cal.name
            )
            cameras.append(resize_camera(full, *size))
            camera_jobs.append(
                _CameraJob(
                    cal.name,
                    outputs,
                    size,
                    video=video if info is not None else None,
                    info=info,
                    images=images,
                    K=cal.K,
                    dist=cal.dist,
                )
            )
        commands = _run_jobs(camera_jobs, tools, hwaccel=hwaccel, workers=workers)
        records = [
            ImageRecord(
                camera,
                k / fps,
                path,
                "test" if camera.name == test_camera else "train",
                k,
            )
            for camera, job in zip(cameras, camera_jobs, strict=True)
            for k, path in enumerate(job.outputs)
        ]
        metadata: dict[str, Any] = {
            "source": f"opencv-yaml/{src.name}",
            "converter": "opencv-yaml",
            "producer": _producer(),
            "resize": resize,
            "fps": fps,
            "frames": frames,
            "test_camera": test_camera,
        }
        if tools is not None:
            metadata["ffmpeg"] = _ffmpeg_metadata(tools, hwaccel, commands)
        write_scene_manifest(out / "scene.json", records, time_unit="s", metadata=metadata)
    except BaseException:
        _discard(out, created)
        raise
    finally:
        shutil.rmtree(out / ".work", ignore_errors=True)
    return out / "scene.json"


# SelfCap (§1.4, D-59, D-60, D-63, D-64, D-69; Q-86, Q-87, Q-93, Q-94, Q-97, Q-101).

# FreeTimeGS scene: (sequence, test camera, window [N0, N1), resize ratio), from §1.4's table.
_SELFCAP_SCENES: dict[str, tuple[str, str, tuple[int, int], float]] = {
    "dance1": ("hair", "0015", (4120, 4180), 0.5),
    "dance2": ("hair", "0015", (5530, 5590), 0.5),
    "corgi1": ("corgi", "0007", (200, 260), 0.5),
    "corgi2": ("corgi", "0007", (2950, 3010), 0.5),
    "bike1": ("bike", "0009", (8900, 8960), 1.0),
    "bike2": ("bike", "0009", (30020, 30080), 1.0),
}
_SELFCAP_DEFAULT_RESIZE = 0.5  # D-80: other scene names, the protocol's ratio for 4K video
_SELFCAP_FPS = 60  # D-59: τ = n/60 s minus the sync offset
_SELFCAP_CALIBRATION = ("intri.yml", "extri.yml", "sync.json")
_FULL_OPENCV_COEFFICIENTS = 8  # k1, k2, p1, p2, k3, k4, k5, k6 (COLMAP's FULL_OPENCV)
_FASTEST = 10  # D-69


class _HashingReader:
    """A binary reader that hashes every byte it passes on (Q-86: the archive's SHA-256 is of
    the bytes as downloaded, read once)."""

    def __init__(self, file: IO[bytes]) -> None:
        self._file = file
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._file.read(size)
        self.digest.update(data)
        return data

    def drain(self) -> None:
        while self.read(1 << 20):
            pass


def _safe_member(member: tarfile.TarInfo) -> tuple[str, ...] | None:
    """A member's path parts, or None for anything but a regular file at a relative path
    without `..` (Q-86: archives are untrusted data)."""
    parts = tuple(Path(member.name).parts)
    if not member.isreg() or not parts or Path(member.name).is_absolute() or ".." in parts:
        return None
    return parts


def _extract_archive(
    archive: Path, work: Path, *, videos: bool, calibration: bool
) -> tuple[str, list[str]]:
    """Stream `archive` once: extract `*/videos/*.mp4` into work/videos and
    `*/optimized/{intri.yml, extri.yml, sync.json}` into work/optimized, hash every byte, and
    return the SHA-256 and the warnings (Q-86). A gzip CRC failure after every member came out
    whole is a warning; one that cuts a member short is an error."""
    if not archive.is_file():
        raise ValueError(
            f"{archive}: archive not found (Q-86: SRC holds the archives as downloaded)"
        )
    notes: list[str] = []
    extracted: set[Path] = set()
    current: Path | None = None  # the member being copied, if any
    with archive.open("rb") as raw:
        reader = _HashingReader(raw)
        try:
            with (
                gzip.GzipFile(fileobj=reader, mode="rb") as stream,  # type: ignore[call-overload]
                tarfile.open(fileobj=stream, mode="r|") as tar,
            ):
                for member in tar:
                    parts = _safe_member(member)
                    if parts is None or len(parts) < 2:
                        continue
                    if videos and parts[-2] == "videos" and parts[-1].endswith(".mp4"):
                        target = work / "videos" / parts[-1]
                    elif (
                        calibration
                        and parts[-2] == "optimized"
                        and parts[-1] in _SELFCAP_CALIBRATION
                    ):
                        target = work / "optimized" / parts[-1]
                    else:
                        continue
                    if target in extracted:
                        raise ValueError(f"{archive}: more than one member matches {target.name}")
                    source = tar.extractfile(member)
                    assert source is not None
                    target.parent.mkdir(parents=True, exist_ok=True)
                    current = target
                    with target.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    current = None
                    if target.stat().st_size != member.size:
                        raise ValueError(f"{archive}: member {member.name} is incomplete")
                    extracted.add(target)
                stream.read()  # to the gzip trailer, where the CRC is checked
        except (gzip.BadGzipFile, zlib.error, EOFError, tarfile.ReadError) as err:
            if current is not None:  # the error cut a needed member short: never use it
                current.unlink(missing_ok=True)
                raise ValueError(f"{archive}: {err} while extracting {current.name}") from None
            if not extracted and (videos or calibration):
                raise ValueError(f"{archive}: not a readable tar.gz archive: {err}") from None
            notes.append(f"{archive.name}: {err} after its members were extracted (§1.4)")
        reader.drain()
    return reader.digest.hexdigest(), notes


def _sync_offsets(path: Path, names: Sequence[str]) -> dict[str, float]:
    """sync.json: an object of camera name → offset in seconds (D-59, Q-101)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{path}: not a readable JSON file: {err}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: must be an object of camera name to offset in seconds")
    offsets = {}
    for name in names:
        value = raw.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{path}: camera {name!r} needs a finite offset, got {value!r}")
        offsets[name] = float(value)
    return offsets


def _colmap_version(colmap: str) -> tuple[str, str]:
    resolved = shutil.which(colmap)
    if resolved is None:
        raise ValueError(
            f"COLMAP not found: {colmap!r} (D-60: SelfCap's protocol undistorts with COLMAP; "
            "undistort='internal' uses the package's own undistortion)"
        )
    try:
        result = subprocess.run([resolved, "help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as err:
        raise ValueError(f"COLMAP at {resolved!r} doesn't run: {err}") from None
    lines = (result.stdout or result.stderr).splitlines()
    return resolved, lines[0].strip() if lines else ""


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix, w ≥ 0 (COLMAP's images.txt order, QW first)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    candidates = [
        (1 + trace, [1 + trace, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]),
        (
            1 + m[0, 0] - m[1, 1] - m[2, 2],
            [
                m[2, 1] - m[1, 2],
                1 + m[0, 0] - m[1, 1] - m[2, 2],
                m[0, 1] + m[1, 0],
                m[0, 2] + m[2, 0],
            ],
        ),
        (
            1 - m[0, 0] + m[1, 1] - m[2, 2],
            [
                m[0, 2] - m[2, 0],
                m[0, 1] + m[1, 0],
                1 - m[0, 0] + m[1, 1] - m[2, 2],
                m[1, 2] + m[2, 1],
            ],
        ),
        (
            1 - m[0, 0] - m[1, 1] + m[2, 2],
            [
                m[1, 0] - m[0, 1],
                m[0, 2] + m[2, 0],
                m[1, 2] + m[2, 1],
                1 - m[0, 0] - m[1, 1] + m[2, 2],
            ],
        ),
    ]
    _, q = max(candidates, key=lambda c: c[0])
    quat: np.ndarray = np.array(q) / np.linalg.norm(q)
    return quat if quat[0] >= 0 else -quat


def _read_colmap_text(
    folder: Path,
) -> tuple[dict[int, tuple[int, int, np.ndarray]], dict[str, int]]:
    """PINHOLE cameras (id → (W, H, [fx, fy, cx, cy])) and image name → camera id of a text
    model (COLMAP's documented format)."""
    cameras: dict[int, tuple[int, int, np.ndarray]] = {}
    try:
        camera_text = (folder / "cameras.txt").read_text()
        image_text = (folder / "images.txt").read_text()
    except OSError as err:
        raise ValueError(f"{folder}: COLMAP wrote no readable text model: {err}") from None
    for line in camera_text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] != "PINHOLE":
            raise ValueError(f"{folder}: COLMAP's undistorted camera is {parts[1]}, not PINHOLE")
        cameras[int(parts[0])] = (
            int(parts[2]),
            int(parts[3]),
            np.array([float(v) for v in parts[4:8]]),
        )
    images: dict[str, int] = {}
    lines = [line for line in image_text.splitlines() if not line.startswith("#")]
    for line in lines[::2]:
        parts = line.split()
        if len(parts) >= 10:
            images[" ".join(parts[9:])] = int(parts[8])
    return cameras, images


def _colmap_undistort(
    colmap: str,
    work: Path,
    calibrated: Sequence[_Calibrated],
    frames: dict[str, list[Path]],
    sizes: dict[str, tuple[int, int]],
) -> tuple[dict[str, tuple[int, int, np.ndarray]], list[list[str]], Path]:
    """D-60, Q-97: COLMAP's image_undistorter with blank_pixels 0 on a text model of FULL_OPENCV
    cameras (cx, cy + 0.5 for COLMAP's pixel centres), its binary output converted to text; the
    read-back PINHOLE camera of each camera (W, H, [fx, fy, cx, cy]) in the continuous
    convention, unshifted; and the folder of undistorted images."""
    model, output, text = (
        work / "colmap" / "sparse_in",
        work / "colmap" / "undistorted",
        work / "colmap" / "text",
    )
    model.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    image_id = 0
    for camera_id, cal in enumerate(calibrated, start=1):
        coefficients = _coefficients(cal.dist)
        width, height = sizes[cal.name]
        params = [cal.K[0, 0], cal.K[1, 1], cal.K[0, 2] + 0.5, cal.K[1, 2] + 0.5, *coefficients[:8]]
        camera_lines.append(
            f"{camera_id} FULL_OPENCV {width} {height} " + " ".join(repr(float(v)) for v in params)
        )
        pose = [*_quaternion(cal.rotation), *cal.translation]
        pose_text = " ".join(repr(float(v)) for v in pose)  # plain decimals for COLMAP's parser
        for path in frames[cal.name]:
            image_id += 1
            name = f"{cal.name}/{path.name}"
            image_lines.append(f"{image_id} {pose_text} {camera_id} {name}\n")
    (model / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (model / "images.txt").write_text("\n".join(image_lines) + "\n")
    (model / "points3D.txt").write_text("")
    image_root = frames[calibrated[0].name][0].parent.parent
    commands = [
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(image_root),
            "--input_path",
            str(model),
            "--output_path",
            str(output),
            "--output_type",
            "COLMAP",
            "--blank_pixels",
            "0",
        ],
        [
            colmap,
            "model_converter",
            "--input_path",
            str(output / "sparse"),
            "--output_path",
            str(text),
            "--output_type",
            "TXT",
        ],
    ]
    text.mkdir(parents=True, exist_ok=True)
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(
                f"COLMAP failed: {' '.join(command[:2])}: {result.stderr.strip()[-2000:]}"
            )
    cameras, images = _read_colmap_text(text)
    read_back: dict[str, tuple[int, int, np.ndarray]] = {}
    for cal in calibrated:
        ids = {images.get(f"{cal.name}/{path.name}") for path in frames[cal.name]}
        if len(ids) != 1 or None in ids:
            raise ValueError(f"COLMAP's model doesn't give camera {cal.name}'s images one camera")
        read_back[cal.name] = cameras[ids.pop()]  # type: ignore[index]
    return read_back, commands, output / "images"


def _masks(folder: Path, indices: Sequence[int]) -> tuple[dict[int, Path], dict[str, Any]]:
    """D-64, Q-87: DIR/<n:06d>.png per test image and DIR/masks.json; a missing mask is
    refused (§1.4: one mask per test image; Q-104)."""
    sidecar = folder / "masks.json"
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{sidecar}: masks need a readable masks.json (Q-87): {err}") from None
    if (
        not isinstance(info, dict)
        or not isinstance(info.get("mask_source"), str)
        or not isinstance(info.get("masks_approximate"), bool)
    ):
        raise ValueError(f"{sidecar}: needs mask_source (a string) and masks_approximate (a bool)")
    paths = {n: folder / f"{n:06d}.png" for n in indices}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError(f"{folder}: no mask for these test images: {', '.join(missing)}")
    return paths, {
        "mask_source": info["mask_source"],
        "masks_approximate": info["masks_approximate"],
    }


def _copy_mask(source: Path, target: Path, size: tuple[int, int]) -> None:
    try:
        image = Image.open(source)
    except OSError as err:
        raise ValueError(f"{source}: not a readable mask: {err}") from None
    with image:
        if image.mode not in _MASK_MODES:
            raise ValueError(
                f"{source}: mask mode must be one of {_MASK_MODES}, got {image.mode!r}"
            )
        if image.size != size:
            raise ValueError(
                f"{source}: mask size must be {size} (the converted test image), got {image.size}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def convert_selfcap(
    src: Path,
    dst: Path,
    *,
    scene: str,
    window: tuple[int, int] | None = None,
    test_camera: str | None = None,
    undistort: Literal["colmap", "internal"] = "colmap",
    colmap: str = "colmap",
    use_sync: bool = True,
    masks: Path | None = None,
    calib: Path | None = None,
    ffmpeg: str = "ffmpeg",
    hwaccel: Literal["none", "cuda"] = "none",
    jobs: int = 0,
    ffprobe: str | None = None,
    resize: float | None = None,
) -> Path:
    """Convert a SelfCap scene (§1.4, D-59, D-60, D-63, D-64, D-80; Q-86, Q-87, Q-97, Q-101): SRC
    holds the archives as downloaded; dance1 … bike2 take window and test camera from the
    card's table, other names need both. `resize` sets the ratio; None takes the table's, or
    0.5 for other names (D-80). Frames N0 ≤ n < N1 of every camera, undistorted by
    COLMAP (or the package with undistort='internal'), area-resized, with τ = n/60 minus the
    camera's sync offset. Returns the path of OUT/scene.json."""
    src = Path(src)
    if undistort not in ("colmap", "internal"):
        raise ValueError(f"undistort must be 'colmap' or 'internal', got {undistort!r}")
    _check_hwaccel(hwaccel)
    workers = _worker_count(jobs)
    if scene in _SELFCAP_SCENES:
        sequence, table_test, table_window, ratio = _SELFCAP_SCENES[scene]
        window = window or table_window
        test_camera = test_camera or table_test
    else:
        if window is None or test_camera is None:
            raise ValueError(
                f"scene {scene!r} isn't in the card's table: pass window and test_camera"
            )
        sequence, ratio = scene, _SELFCAP_DEFAULT_RESIZE
    if resize is not None:
        resized_size(1, 1, resize)  # a finite ratio > 0
        ratio = float(resize)
    n0, n1 = (int(window[0]), int(window[1]))
    if not 0 <= n0 < n1:
        raise ValueError(f"window must be (N0, N1) with 0 <= N0 < N1, got {window!r}")
    indices = list(range(n0, n1))
    colmap_path, colmap_version = _colmap_version(colmap) if undistort == "colmap" else ("", "")
    mask_paths, mask_metadata = _masks(Path(masks), indices) if masks is not None else ({}, {})
    out, created = _prepare_out(dst)
    work = out / ".work"
    try:
        tools = _tools(ffmpeg, ffprobe)
        archives: dict[str, str] = {}
        archive_warnings: list[str] = []
        release = src / f"{sequence}-release.tar.gz"
        calib_archive = src / "hair-calib.tar.gz" if sequence == "hair" and calib is None else None
        from_release = calib is None and calib_archive is None
        digest, found = _extract_archive(
            release, work / "release", videos=True, calibration=from_release
        )
        archives[release.name], archive_warnings = digest, archive_warnings + found
        if calib is not None:
            folder = Path(calib)
            optimized = (
                folder / "optimized" if (folder / "optimized" / "intri.yml").is_file() else folder
            )
            calibration_source = str(folder)
        elif calib_archive is not None:
            digest, found = _extract_archive(
                calib_archive, work / "calib", videos=False, calibration=True
            )
            archives[calib_archive.name], archive_warnings = digest, archive_warnings + found
            optimized, calibration_source = work / "calib" / "optimized", calib_archive.name
        else:
            optimized, calibration_source = work / "release" / "optimized", release.name
        calibrated = _calibration(optimized / "intri.yml", optimized / "extri.yml")
        names = [cal.name for cal in calibrated]
        if test_camera not in names:
            raise ValueError(f"test camera {test_camera!r} is not among the calibrated cameras")
        sync_path = optimized / "sync.json"
        synced = use_sync and sync_path.is_file()
        offsets = _sync_offsets(sync_path, names) if synced else dict.fromkeys(names, 0.0)
        infos: dict[str, _VideoInfo] = {}
        for cal in calibrated:
            video = work / "release" / "videos" / f"{cal.name}.mp4"
            if not video.is_file():
                raise ValueError(f"{release}: no video for camera {cal.name}")
            info = infos[cal.name] = _probe(tools, video)
            if cal.size is not None and cal.size != (info.width, info.height):
                raise ValueError(
                    f"camera {cal.name}: W_ and H_ {cal.size} differ from its video's "
                    f"{(info.width, info.height)}"
                )
            if info.frames < n1:
                raise ValueError(f"{video}: {info.frames} frames, the window needs {n1}")
            if undistort == "colmap" and np.asarray(cal.dist).size > _FULL_OPENCV_COEFFICIENTS:
                raise ValueError(
                    f"camera {cal.name}: COLMAP's FULL_OPENCV holds at most 8 coefficients (Q-97)"
                )
        cameras: dict[str, Camera] = {}
        camera_jobs: list[_CameraJob] = []
        commands: dict[str, list[str]] = {}
        colmap_commands: list[list[str]] = []
        if undistort == "internal":
            for cal in calibrated:
                info = infos[cal.name]
                size = resized_size(info.width, info.height, ratio)
                full = camera_from_opencv(
                    cal.K,
                    cal.rotation,
                    cal.translation,
                    width=info.width,
                    height=info.height,
                    name=cal.name,
                )
                cameras[cal.name] = resize_camera(full, *size)
                outputs = [out / "images" / cal.name / f"{n:06d}.png" for n in indices]
                camera_jobs.append(
                    _CameraJob(
                        cal.name,
                        outputs,
                        size,
                        video=work / "release" / "videos" / f"{cal.name}.mp4",
                        info=info,
                        window=(n0, n1),
                        K=cal.K,
                        dist=cal.dist,
                    )
                )
            commands = _run_jobs(camera_jobs, tools, hwaccel=hwaccel, workers=workers)
        else:
            raw_jobs, raw_frames = [], {}
            for cal in calibrated:
                info = infos[cal.name]
                raw = [work / "colmap" / "images" / cal.name / f"{n:06d}.png" for n in indices]
                raw_frames[cal.name] = raw
                raw_jobs.append(
                    _CameraJob(
                        cal.name,
                        raw,
                        (info.width, info.height),
                        video=work / "release" / "videos" / f"{cal.name}.mp4",
                        info=info,
                        window=(n0, n1),
                    )
                )
            commands = _run_jobs(raw_jobs, tools, hwaccel=hwaccel, workers=workers)
            sizes = {name: (infos[name].width, infos[name].height) for name in names}
            read_back, colmap_commands, undistorted = _colmap_undistort(
                colmap_path, work, calibrated, raw_frames, sizes
            )
            for cal in calibrated:
                width, height, (fx, fy, cx, cy) = read_back[cal.name]
                K = np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
                )  # continuous: no shift
                size = resized_size(width, height, ratio)
                full = _camera(K, cal.rotation, cal.translation, width, height, cal.name)
                cameras[cal.name] = resize_camera(full, *size)
                images = [undistorted / cal.name / f"{n:06d}.png" for n in indices]
                outputs = [out / "images" / cal.name / f"{n:06d}.png" for n in indices]
                camera_jobs.append(_CameraJob(cal.name, outputs, size, images=images))
            _run_jobs(camera_jobs, None, hwaccel=hwaccel, workers=workers)
        records, record_masks = [], []
        for job in camera_jobs:
            camera = cameras[job.name]
            for n, path in zip(indices, job.outputs, strict=True):
                split: Literal["train", "test"] = "test" if job.name == test_camera else "train"
                mask = None
                if split == "test" and n in mask_paths:
                    mask = out / "masks" / job.name / f"{n:06d}.png"
                    _copy_mask(mask_paths[n], mask, (camera.width, camera.height))
                records.append(
                    ImageRecord(camera, n / _SELFCAP_FPS - offsets[job.name], path, split, n)
                )
                record_masks.append(mask)
        metadata: dict[str, Any] = {
            "source": f"selfcap/{scene}",
            "converter": "selfcap",
            "producer": _producer(),
            "resize": ratio,
            "fps": _SELFCAP_FPS,
            "window": [n0, n1],
            "test_camera": test_camera,
            "archives": archives,
            "calibration": calibration_source,
            "sync": synced,
            "undistort": undistort,
            "ffmpeg": _ffmpeg_metadata(tools, hwaccel, commands),
            "warnings": archive_warnings,
            **mask_metadata,
        }
        if undistort == "colmap":
            metadata["colmap"] = {
                "path": colmap_path,
                "version": colmap_version,
                "commands": colmap_commands,
            }
        if scene == "dance1" and len(indices) >= _FASTEST:
            test_job = next(job for job in camera_jobs if job.name == test_camera)
            frames = (torch.from_numpy(_read_rgb(p)) for p in test_job.outputs)
            start = n0 + fastest_frames(frames, count=_FASTEST)
            metadata["fastest_10"] = [start, start + _FASTEST]  # D-69, Q-59: frame_index values
        write_scene_manifest(
            out / "scene.json", records, time_unit="s", masks=record_masks, metadata=metadata
        )
    except BaseException:
        _discard(out, created)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out / "scene.json"


# The frame cache and the prefetcher (D-51; Q-98, Q-99).

_CACHE_FORMAT = "ftgs-frame-cache"
_CACHE_VERSION = "1"
_CACHE_PLACEMENTS = ("gpu", "mmap")


def _cache_groups(dataset: "MultiViewVideo") -> list[tuple[str, list[int]]]:
    """The training records grouped by camera name, in order of first appearance (Q-98)."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(dataset.records):
        if record.split == "train":
            groups.setdefault(record.camera.name, []).append(index)
    return list(groups.items())


def _cache_metadata(dataset: "MultiViewVideo", camera: str, records: list[int]) -> dict[str, str]:
    return {
        "format": _CACHE_FORMAT,
        "version": _CACHE_VERSION,
        "manifest_sha256": dataset.manifest.sha256,
        "camera": camera,
        "records": json.dumps(records, separators=(",", ":")),
    }


def _cache_file_valid(path: Path, metadata: dict[str, str], shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt") as file:
            return bool(
                file.metadata() == metadata and tuple(file.get_slice("frames").get_shape()) == shape
            )
    except Exception:  # an unreadable file is rebuilt like a stale one
        return False


class FrameCache:
    """The uint8 copy of every training image (D-51; Q-84, Q-98): one safetensors file per
    camera beside the scene manifest, keyed by its SHA-256, held in GPU memory ("gpu") or
    memory-mapped and read one frame at a time ("mmap"). `open` builds it when missing or
    stale. Instances come from `FrameCache.open` only: the constructor takes internal
    structures and may change without notice (Q-84)."""

    def __init__(
        self,
        directory: Path,
        placement: str,
        where: dict[int, tuple[int, int]],
        frames: list[Any],
        files: list[Any],
    ) -> None:
        self.directory = directory
        self.placement = placement
        self._where = where
        self._frames = frames  # GPU tensors, or safetensors slices of the memory-mapped files
        self._files = files  # the open memory-mapped files the slices read from

    @staticmethod
    def default_directory(manifest: SceneManifest) -> Path:
        """`<manifest dir>/.ftgs-cache/<manifest sha256>` (Q-98)."""
        return manifest.path.parent / ".ftgs-cache" / manifest.sha256

    @staticmethod
    def size_of(dataset: "MultiViewVideo") -> int:
        """The bytes of the training images as uint8, the cache's payload."""
        return sum(
            r.camera.width * r.camera.height * 3 for r in dataset.records if r.split == "train"
        )

    @classmethod
    def build(
        cls, dataset: "MultiViewVideo", *, directory: Path | None = None, jobs: int = 0
    ) -> Path:
        """Write the cache's files, keeping the valid ones, and return its directory. PNGs are
        decoded by `jobs` threads (0: one per core)."""
        directory = (
            Path(directory) if directory is not None else cls.default_directory(dataset.manifest)
        )
        directory.mkdir(parents=True, exist_ok=True)
        workers = jobs or os.cpu_count() or 1
        for k, (camera, records) in enumerate(_cache_groups(dataset)):
            first = dataset.records[records[0]].camera
            shape = (len(records), first.height, first.width, 3)
            metadata = _cache_metadata(dataset, camera, records)
            path = directory / f"{k:04d}.safetensors"
            if _cache_file_valid(path, metadata, shape):
                continue
            with concurrent.futures.ThreadPoolExecutor(workers) as threads:
                frames = list(
                    threads.map(
                        lambda i: _read(
                            dataset.records[i].image_path, _size(dataset.records[i]), _IMAGE_MODES
                        )[..., :3],
                        records,
                    )
                )
            if any(frame.shape != shape[1:] for frame in frames):
                raise ValueError(f"camera {camera!r}: its training images differ in size")
            temporary = directory / f".{path.name}.tmp-{os.getpid()}"
            try:
                save_file(
                    {"frames": torch.from_numpy(np.stack(frames))},
                    str(temporary),
                    metadata=metadata,
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return directory

    @classmethod
    def open(
        cls,
        dataset: "MultiViewVideo",
        *,
        placement: Literal["gpu", "mmap"],
        device: str | torch.device = "cpu",
        directory: Path | None = None,
    ) -> "FrameCache":
        """The cache of `dataset`, built first when missing or stale; "gpu" loads every camera's
        frames onto `device`, "mmap" keeps the files memory-mapped (D-51, Q-98)."""
        if placement not in _CACHE_PLACEMENTS:
            raise ValueError(f"placement must be one of {_CACHE_PLACEMENTS}, got {placement!r}")
        directory = cls.build(dataset, directory=directory)
        where: dict[int, tuple[int, int]] = {}
        frames: list[Any] = []
        files: list[Any] = []
        for k, (_, records) in enumerate(_cache_groups(dataset)):
            path = directory / f"{k:04d}.safetensors"
            for row, index in enumerate(records):
                where[index] = (k, row)
            if placement == "gpu":
                frames.append(load_file(str(path), device=str(torch.device(device)))["frames"])
            else:
                file = safe_open(str(path), framework="pt")
                files.append(file)
                frames.append(file.get_slice("frames"))  # type: ignore[no-untyped-call]
        return cls(directory, placement, where, frames, files)

    @property
    def nbytes(self) -> int:
        """The size of the cache's files."""
        return sum(p.stat().st_size for p in self.directory.glob("*.safetensors"))

    def frame(self, index: int) -> Tensor:
        """Training image `index` as uint8 [H, W, 3]: a view in GPU memory, or read from the
        memory-mapped file onto the CPU."""
        if index not in self._where:
            raise IndexError(f"record {index} is not a training image of this cache")
        k, row = self._where[index]
        if self.placement == "gpu":
            frame: Tensor = self._frames[k][row]
            return frame
        frame = self._frames[k][row : row + 1][0]
        return frame


class Prefetcher:
    """Images in a known order, read ahead (D-51; Q-84, Q-99): `schedule` gives the upcoming
    training record indices, `take` returns one as float32 [H, W, 3] in [0, 1] on `device`,
    bitwise equal to `MultiViewVideo.load_image`. On CUDA a thread copies frames through a ring
    of pinned buffers on a side stream, and the current stream waits on each copy's event; on the
    CPU the thread reads ahead; `depth` 0 reads each image when it is taken."""

    def __init__(
        self, cache: FrameCache, *, device: str | torch.device = "cpu", depth: int = 8
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError(f"depth must be an int >= 0, got {depth!r}")
        self._cache = cache
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            # "cuda" != "cuda:0": name the index, so a cache already on the device is used in
            # place rather than copied through the host (D-51, §2.4 rule 3).
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._depth = depth
        self._cuda = self._device.type == "cuda"
        # On CUDA, x / <python float> multiplies by the reciprocal, which isn't bitwise the CPU's
        # x / 255.0; dividing by a 0-dim device tensor is (measured; Q-99).
        self._divisor: Tensor | float = (
            torch.full((), 255.0, device=self._device) if self._cuda else 255.0
        )
        self._plan: deque[int] = deque()
        self._pending: deque[tuple[int, concurrent.futures.Future[Any]]] = deque()
        self._executor = concurrent.futures.ThreadPoolExecutor(1) if depth > 0 else None
        self._stream = torch.cuda.Stream(self._device) if self._cuda else None  # type: ignore[no-untyped-call]
        self._free: dict[tuple[int, ...], list[tuple[Tensor, Any]]] = {}
        self._lock = threading.Lock()
        self._misses = 0

    def schedule(self, indices: Sequence[int]) -> None:
        """The upcoming record indices, in the order they will be taken."""
        upcoming = list(indices)
        pending = [index for index, _ in self._pending]
        if upcoming[: len(pending)] == pending:
            self._plan = deque(upcoming[len(pending) :])
        else:
            self._drop()
            self._plan = deque(upcoming)
        self._fill()

    def take(self, index: int) -> Tensor:
        """Record `index`'s image as float32 [H, W, 3] on the device, ready on the current
        stream; read directly when it isn't next in the schedule."""
        if self._pending and self._pending[0][0] == index:
            _, future = self._pending.popleft()
            item = future.result()
        else:
            self._misses += 1  # read directly: not next in the schedule
            self._drop()
            self._plan.clear()
            item = self._load(index)
        image = self._finish(item)
        self._fill()
        return image

    def close(self) -> None:
        self._drop()
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _fill(self) -> None:
        if self._executor is None:
            return
        while self._plan and len(self._pending) < self._depth:
            index = self._plan.popleft()
            self._pending.append((index, self._executor.submit(self._load, index)))

    def _drop(self) -> None:
        while self._pending:
            _, future = self._pending.popleft()
            self._release(future.result())

    def _buffer(self, shape: tuple[int, ...]) -> Tensor:
        with self._lock:
            free = self._free.get(shape)
            entry = free.pop() if free else None
        if entry is None:
            return torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        buffer, event = entry
        if event is not None:
            event.synchronize()  # the buffer's last copy is done (on this worker, not the trainer)
        return buffer

    def _release(self, item: tuple[Tensor, Any, Tensor | None]) -> None:
        _, event, buffer = item
        if buffer is not None:
            with self._lock:
                self._free.setdefault(tuple(buffer.shape), []).append((buffer, event))

    def _load(self, index: int) -> tuple[Tensor, Any, Tensor | None]:
        frame = self._cache.frame(index)
        if frame.device == self._device or not self._cuda:
            return frame, None, None
        assert self._stream is not None
        buffer = self._buffer(tuple(frame.shape))
        buffer.copy_(frame)
        with torch.cuda.stream(self._stream):
            on_device = buffer.to(self._device, non_blocking=True)
            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record(self._stream)
        return on_device, event, buffer

    def _finish(self, item: tuple[Tensor, Any, Tensor | None]) -> Tensor:
        frame, event, _ = item
        if event is not None:
            current = torch.cuda.current_stream(self._device)
            current.wait_event(event)
            frame.record_stream(current)
        image = frame.to(torch.float32) / self._divisor
        self._release(item)
        return image
