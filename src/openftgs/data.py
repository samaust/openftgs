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
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from openftgs.io import read_scene_manifest
from openftgs.types import Camera, ImageRecord, SceneManifest, TimeNormalizer, frame_interval

__all__ = [
    "MultiViewVideo",
    "camera_from_llff",
    "camera_from_opencv",
    "fastest_frames",
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
