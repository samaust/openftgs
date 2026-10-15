# SPDX-License-Identifier: Apache-2.0
"""Camera operations, the MultiViewVideo dataset with lazy image loading, scene manifests, the
dataset converters (Neural3DV, OpenCV-style YAML, SelfCap), image resizing and undistortion, the
YAML camera reader, the frame cache and prefetcher, and the Fastest-10 frames of the ablations.

Spec: §1.4, D-51, D-61, D-69. Band: Foundation (§2.1).
"""

import concurrent.futures
import importlib.metadata
import json
import math
import multiprocessing
import numbers
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

from openftgs.io import read_scene_manifest, write_scene_manifest
from openftgs.types import Camera, ImageRecord, SceneManifest, TimeNormalizer, frame_interval

__all__ = [
    "MultiViewVideo",
    "camera_from_llff",
    "camera_from_opencv",
    "convert_n3dv",
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

_ROTATION_TOLERANCE = 1e-6  # max |RᵀR − I| accepted from a calibration file


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
    """§1.4's W′ = round(W·r) and H′ = round(H·r), an exact half rounding up, at least 1
    (Q-89)."""
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, numbers.Real)
        or not (math.isfinite(ratio) and ratio > 0.0)
    ):
        raise ValueError(f"ratio must be a finite number > 0, got {ratio!r}")
    return max(1, math.floor(width * ratio + 0.5)), max(1, math.floor(height * ratio + 0.5))


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
    if "sizes" in fields:  # opencv-nd-matrix
        shape = tuple(int(n) for n in fields["sizes"])
    else:
        shape = (int(fields["rows"]), int(fields["cols"]))
    if channels > 1:
        shape += (channels,)
    data = fields.get("data", [])
    if not isinstance(data, list) or len(data) != math.prod(shape):
        raise yaml.constructor.ConstructorError(
            None, None, f"matrix data must hold {math.prod(shape)} values", node.start_mark
        )
    dtype = _OPENCV_DTYPES[match.group(2)]
    wide = np.float64 if np.issubdtype(dtype, np.floating) else np.int64
    return np.array(data, dtype=wide).astype(dtype).reshape(shape)


_OpenCVLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix)
_OpenCVLoader.add_constructor("tag:yaml.org,2002:opencv-nd-matrix", _opencv_matrix)
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


# Dataset conversion (§1.4 "Dataset protocols", D-48; Q-85 to Q-101): ffmpeg runs as a separate
# program, frames are processed and encoded in a process pool (§2.4 "Threads and processes").

_HWACCELS = ("none", "cuda")
# Q-92: the software format hwdownload hands over, from the stream's pixel format.
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
    """Q-85: ffprobe defaults to the one beside a path-given ffmpeg, else the one on PATH."""
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
    decode indices N0 ≤ n < N1, as raw RGB24 on stdout (D-48). Q-92: with CUDA, hardware frames
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


def _decode(command: list[str], info: _VideoInfo, count: int, video: Path) -> Iterator[np.ndarray]:
    """The frames ffmpeg writes, exactly W·H·3 bytes each (§1.4). Q-93: a short read, a
    nonzero exit or any message at `-v error` is a ValueError."""
    size = info.width * info.height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    errors: list[bytes] = []
    reader = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    read = 0
    try:
        while read < count:
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
) -> None:
    """One output frame (a pool task): undistorted with K and dist in OpenCV's convention when
    they distort, resized to `size` by area averaging with one 8-bit rounding (Q-90), and
    written as 8-bit RGB PNG."""
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


def _read_rgb(path: Path) -> np.ndarray:
    """An 8-bit image file as [H, W, 3] uint8 (alpha dropped; §1.4 "Images")."""
    with Image.open(path) as image:
        if image.mode not in _IMAGE_MODES:
            raise ValueError(f"{path}: mode must be one of {_IMAGE_MODES}, got {image.mode!r}")
        return np.asarray(image)[..., :3]


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


def _run_jobs(
    jobs: Sequence[_CameraJob], tools: _Tools | None, *, hwaccel: str, workers: int
) -> dict[str, list[str]]:
    """Decode up to `workers` cameras at once, each frame processed and written by a process
    pool (§2.4); outputs don't depend on `workers`. Returns each camera's ffmpeg command."""
    commands: dict[str, list[str]] = {}
    total = sum(len(job.outputs) for job in jobs)
    pool_size = max(1, min(os.cpu_count() or 1, total))
    in_flight = threading.BoundedSemaphore(2 * pool_size)  # bounds decoded frames held in memory
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        pool_size, mp_context=context, initializer=_pool_init
    ) as pool:

        def submit(
            source: np.ndarray | str, job: _CameraJob, k: int
        ) -> concurrent.futures.Future[None]:
            in_flight.acquire()
            future = pool.submit(
                _write_frame, source, str(job.outputs[k]), job.size, job.K, job.dist
            )
            future.add_done_callback(lambda _: in_flight.release())
            return future

        def camera(job: _CameraJob) -> list[concurrent.futures.Future[None]]:
            futures = []
            if job.video is not None:
                assert tools is not None and job.info is not None
                count = len(job.outputs)
                command = _decode_command(
                    tools, job.video, job.info, count=count, window=job.window, hwaccel=hwaccel
                )
                commands[job.name] = command
                for k, frame in enumerate(_decode(command, job.info, count, job.video)):
                    futures.append(submit(frame, job, k))
            else:
                for k, image in enumerate(job.images):
                    futures.append(submit(str(image), job, k))
            return futures

        with concurrent.futures.ThreadPoolExecutor(min(workers, len(jobs)) or 1) as threads:
            per_camera = list(threads.map(camera, jobs))
        for futures in per_camera:
            for future in futures:
                future.result()
    return {job.name: commands[job.name] for job in jobs if job.name in commands}


def _prepare_out(dst: Path) -> Path:
    """Q-88: OUT is absent or an empty directory."""
    dst = Path(dst)
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise ValueError(f"{dst}: the output must be absent or an empty directory")
    dst.mkdir(parents=True, exist_ok=True)
    return dst


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
        "decoder": "cuda" if hwaccel == "cuda" else "cpu",  # Q-92: CUDA runs or fails
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
    """Convert a Neural3DV scene (D-28, D-48; Q-85, Q-88, Q-94, Q-95): `poses_bounds.npy`'s rows
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
    out = _prepare_out(dst)
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
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out / "scene.json"
