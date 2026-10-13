# SPDX-License-Identifier: Apache-2.0
"""The multi-view video (§2.1, E14, D-27, D-46; Q-65) and the Fastest-10 frames (D-69; Q-59)
in openftgs.data.

Scenes are written by hand: manifests whose cameras have exact poses (identity rotations, so a
camera's center is minus its translation), PNG files whose pixel values are known, and timestamp
sets whose medians are worked by hand. For the Fastest-10 frames, sequences are built so that
the window of largest motion is known by construction: frames that stay constant except at
chosen transitions, whose mean absolute differences are worked by hand.
"""

import json
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from openftgs.data import MultiViewVideo, fastest_frames
from openftgs.io import read_scene_manifest
from openftgs.types import TimeNormalizer

F32 = torch.float32


def _frames(levels: list[float], *, shape: tuple[int, int] = (4, 6)) -> list[torch.Tensor]:
    """Constant frames: frame i is `levels[i]` everywhere, so consecutive frames differ by
    |levels[i + 1] − levels[i]| on average."""
    return [torch.full((*shape, 3), level, dtype=F32) for level in levels]


def test_fastest_frames_finds_the_window_of_largest_motion() -> None:
    """D-69: the 10 consecutive frames whose consecutive frames differ most. Twenty frames,
    constant except for jumps between frames 7 and 8, 9 and 10 and 12 and 13: the windows
    containing all three jumps start at 4 … 7, so the lowest, 4, wins (Q-59)."""
    levels = [0.0] * 8 + [0.5] * 2 + [0.1] * 3 + [0.9] * 7
    assert fastest_frames(_frames(levels)) == 4


def test_fastest_frames_counts_only_differences_inside_the_window() -> None:
    """Q-59: a window scores its count − 1 internal differences. With count = 3, the jump
    from frame 5 to 6 lies inside the windows starting at 4 and 5, not at 6, where frame 6 is
    the window's first frame; the larger jump from 9 to 10 is caught by windows 8 and 9."""
    levels = [0.0] * 6 + [0.3] * 4 + [1.0] * 4
    assert fastest_frames(_frames(levels), count=3) == 8


def test_fastest_frames_breaks_ties_to_the_lowest_start() -> None:
    """Q-59: equal scores go to the lowest start (as D-44 breaks ties). Two equal jumps, at
    2→3 and 10→11, each alone in their windows of 4 frames."""
    levels = [0.0] * 3 + [0.5] * 8 + [1.0] * 4
    assert fastest_frames(_frames(levels), count=4) == 0


def test_fastest_frames_streams_frames_and_takes_uint8() -> None:
    """Q-59: frames may come from a generator, and uint8 frames give the start their floating
    counterparts give."""
    levels = [0.0] * 8 + [0.5] * 2 + [0.1] * 3 + [0.9] * 7

    def stream() -> Iterator[torch.Tensor]:
        for frame in _frames(levels):
            yield (frame * 255).round().to(torch.uint8)

    assert fastest_frames(stream()) == fastest_frames(_frames(levels)) == 4


def test_fastest_frames_of_exactly_count_frames_is_zero() -> None:
    assert fastest_frames(_frames([0.0, 1.0] * 5)) == 0


@pytest.mark.parametrize("count", [1, 0, -3, True, 2.0, "10"])
def test_fastest_frames_needs_a_count_of_at_least_two(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        fastest_frames(_frames([0.0] * 12), count=count)  # type: ignore[arg-type]


def test_fastest_frames_needs_count_frames() -> None:
    with pytest.raises(ValueError, match="frames"):
        fastest_frames(_frames([0.0] * 9))


@pytest.mark.parametrize(
    "bad", [torch.zeros(4, 7, 3), torch.zeros(4, 6, 4), torch.zeros(4, 6), torch.zeros(1, 4, 6, 3)]
)
def test_fastest_frames_needs_frames_of_one_hw3_shape(bad: torch.Tensor) -> None:
    frames = _frames([0.0] * 12)
    frames[5] = bad
    with pytest.raises(ValueError, match="shape"):
        fastest_frames(frames)


def test_fastest_frames_needs_frames_of_one_dtype() -> None:
    """Q-59: uint8 frames are on a 0–255 scale and floating frames on 0–1, so a stream mixing
    them would weigh their differences unevenly; it is refused, as mixed shapes are."""
    frames = _frames([0.0] * 12)
    frames[4] = torch.zeros(4, 6, 3, dtype=torch.uint8)
    with pytest.raises(ValueError, match="dtype"):
        fastest_frames(frames)


# MultiViewVideo (§2.1; Q-65).


def _camera(center: tuple[float, float, float], width: int = 4, height: int = 3) -> dict[str, Any]:
    """A camera with the identity rotation at `center`: world_to_camera = [I | −center], so
    E2's center −R_wᵀt_w is `center` exactly."""
    x, y, z = center
    return {
        "width": width,
        "height": height,
        "K": [[5.0, 0.0, 2.0], [0.0, 5.0, 1.5], [0.0, 0.0, 1.0]],
        "world_to_camera": [
            [1.0, 0.0, 0.0, -x],
            [0.0, 1.0, 0.0, -y],
            [0.0, 0.0, 1.0, -z],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _image(camera: str, timestamp: float, split: str = "train", **extra: object) -> dict[str, Any]:
    return {
        "camera": camera,
        "timestamp": timestamp,
        "path": f"images/{camera}/{timestamp!r}.png",
        "split": split,
        **extra,
    }


def _pixels(height: int, width: int, channels: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, channels), dtype=np.uint8)


def _write_scene(
    root: Path,
    cameras: dict[str, dict[str, Any]],
    images: list[dict[str, Any]],
    *,
    mode: str = "RGB",
) -> Path:
    """The manifest and one PNG of the camera's size per image (and per mask), with seeded
    pixel values."""
    manifest = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": cameras,
        "images": images,
        "metadata": {"source": "test"},
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest))
    for k, image in enumerate(images):
        cam = cameras[image["camera"]]
        w, h = cam["width"], cam["height"]
        channels = {"RGB": 3, "RGBA": 4}[mode]
        target = root / image["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_pixels(h, w, channels, k), mode).save(target)
        if "mask" in image:
            mask = root / image["mask"]
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_pixels(h, w, 1, 100 + k)[..., 0], "L").save(mask)
    return path


_RING = {  # four training cameras at distance 2 from their centroid, the origin
    "a": _camera((2.0, 0.0, 0.0)),
    "b": _camera((-2.0, 0.0, 0.0)),
    "c": _camera((0.0, 2.0, 0.0)),
    "d": _camera((0.0, -2.0, 0.0)),
    "test": _camera((100.0, 0.0, 0.0)),
}


def _ring_images(frames: int = 3) -> list[dict[str, Any]]:
    images = [_image(name, k / 30.0) for k in range(frames) for name in "abcd"]
    images += [_image("test", k / 30.0, "test") for k in range(frames)]
    return images


def test_multi_view_video_holds_the_manifest_records(tmp_path: Path) -> None:
    """§2.1: the records of the scene manifest, in file order, and the manifest itself (Q-65)."""
    path = _write_scene(tmp_path, _RING, _ring_images())
    video = MultiViewVideo.from_manifest(path)
    manifest = read_scene_manifest(path)
    assert video.manifest.sha256 == manifest.sha256
    assert [(r.camera.name, r.timestamp, r.split) for r in video.records] == [
        (r.camera.name, r.timestamp, r.split) for r in manifest.records
    ]
    assert isinstance(video.records, list)


def test_time_normalizer_spans_the_training_timestamps(tmp_path: Path) -> None:
    """§1.4 "Time": τ₀ and τ₁ are the smallest and largest training timestamps; a test image
    outside them doesn't move them."""
    images = [_image("a", 1.0), _image("b", 3.0), _image("test", 7.0, "test")]
    images.append(_image("test", -2.0, "test"))
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    assert video.time == TimeNormalizer(1.0, 3.0)


def test_frame_interval_of_the_training_cameras(tmp_path: Path) -> None:
    """E14, D-46: Δt_f is the median over training cameras of each camera's median gap, in
    normalized time. Four synchronized cameras at frames k/30, k = 0 … 4, span T_τ = 4/30, so
    Δt_f = 1/4. A test view whose images are closer in time doesn't count."""
    images = [_image(name, k / 30.0) for k in range(5) for name in "abcd"]
    images += [_image("test", k / 120.0, "test") for k in range(17)]
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    assert video.frame_dt == pytest.approx(0.25, rel=1e-12)


def test_frame_interval_with_unsynchronized_cameras(tmp_path: Path) -> None:
    """D-46, U-18: cameras offset by a fraction of a frame give the same Δt_f as synchronized
    ones. Camera b runs 0.4 frame behind a: T_τ = 4.4/30 and each camera's gap is 1/30, so
    Δt_f = 1/4.4."""
    images = [_image("a", k / 30.0) for k in range(5)]
    images += [_image("b", (k + 0.4) / 30.0) for k in range(5)]
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    assert video.frame_dt == pytest.approx(1.0 / 4.4, rel=1e-12)


def test_frame_interval_of_a_single_timestamp(tmp_path: Path) -> None:
    """§1.4 "Time": with T_τ = 0, Δt_f := 1."""
    images = [_image(name, 0.5) for name in "abcd"]
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    assert video.frame_dt == 1.0
    assert video.time == TimeNormalizer(0.5, 0.5)


def test_spatial_scale_of_a_ring(tmp_path: Path) -> None:
    """D-27: S = 1.1 × the largest distance from a training camera center to the centroid of
    the training camera centers. Four cameras at distance 2 from the origin give 2.2; the test
    view at distance 100 is excluded, and a pose repeated at every frame counts once."""
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, _ring_images(3)))
    assert video.spatial_scale(1.1) == pytest.approx(2.2, rel=1e-12)
    assert video.spatial_scale(2.0) == pytest.approx(4.0, rel=1e-12)


def test_spatial_scale_counts_a_pose_override_as_its_own_center(tmp_path: Path) -> None:
    """Q-65: an image whose pose overrides its camera's adds its own center. With centers
    (±2, 0, 0), (0, ±2, 0) and (0, 0, 4), the centroid is (0, 0, 0.8); the largest distance is
    3.2, from (0, 0, 4), so S = 1.1 × 3.2."""
    images = _ring_images(2)
    images.append(_image("a", 0.5, world_to_camera=_camera((0.0, 0.0, 4.0))["world_to_camera"]))
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    assert video.spatial_scale(1.1) == pytest.approx(1.1 * 3.2, rel=1e-12)


def test_spatial_scale_needs_two_distinct_centers(tmp_path: Path) -> None:
    """Q-65: one training center gives S = 0, which the position rate can't use."""
    images = [_image("a", k / 30.0) for k in range(3)] + [_image("test", 0.0, "test")]
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    with pytest.raises(ValueError, match="two distinct"):
        video.spatial_scale(1.1)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan"), True])
def test_spatial_scale_rejects_bad_factors(tmp_path: Path, factor: float) -> None:
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, _ring_images(1)))
    with pytest.raises(ValueError, match="factor"):
        video.spatial_scale(factor)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_load_image_divides_8_bit_values_by_255(tmp_path: Path, mode: str) -> None:
    """§1.4 "Images": float32 RGB [H, W, 3] in [0, 1], 8-bit values divided by 255, alpha
    dropped (D-34)."""
    images = _ring_images(1)
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images, mode=mode))
    for k in (0, 4):
        image = video.load_image(k)
        expected = torch.from_numpy(_pixels(3, 4, len(mode), k)[..., :3].astype(np.float64) / 255)
        assert image.dtype == torch.float32 and tuple(image.shape) == (3, 4, 3)
        torch.testing.assert_close(image.double(), expected, rtol=0.0, atol=6e-8)


def test_load_image_checks_size_and_mode(tmp_path: Path) -> None:
    """Q-65: the image must match its camera's size, and only 8-bit RGB or RGBA is read."""
    path = _write_scene(tmp_path, _RING, _ring_images(1))
    video = MultiViewVideo.from_manifest(path)
    Image.fromarray(_pixels(4, 4, 3, 0), "RGB").save(video.records[0].image_path)
    with pytest.raises(ValueError, match="size"):
        video.load_image(0)
    Image.fromarray(_pixels(3, 4, 1, 0)[..., 0], "L").save(video.records[1].image_path)
    with pytest.raises(ValueError, match="mode"):
        video.load_image(1)


def _write_png_16(path: Path, pixels: np.ndarray) -> None:
    """A 16-bit-per-channel RGB PNG written from the PNG format's chunks, which Pillow can read
    but not write."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    height, width, _ = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)  # depth 16, truecolor
    rows = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_load_image_refuses_16_bit_files(tmp_path: Path) -> None:
    """Q-65, §1.4 "Images": only 8-bit inputs are defined (divided by 255). Pillow reads a
    16-bit RGB PNG in its 8-bit mode "RGB", keeping the high byte, so the file's own bit depth
    is checked (from the two-axis review of #84)."""
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, _ring_images(1)))
    path = video.records[0].image_path
    _write_png_16(path, np.full((3, 4, 3), 0x1234, dtype=np.uint16))
    with Image.open(path) as image:
        assert image.mode == "RGB"  # the reason for the check
    with pytest.raises(ValueError, match="8 bits"):
        video.load_image(0)


@pytest.mark.parametrize("index", [-1, 7])
def test_load_image_index_out_of_range(tmp_path: Path, index: int) -> None:
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, _ring_images(1)))
    with pytest.raises(IndexError):
        video.load_image(index)


def test_load_mask(tmp_path: Path) -> None:
    """Q-65, Q-61: an 8-bit mask divided by 255 and a bilevel one as 0 or 1, [H, W] float32;
    None for an image without a mask."""
    images = _ring_images(1)
    images[4]["mask"] = "masks/test.png"
    path = _write_scene(tmp_path, _RING, images)
    video = MultiViewVideo.from_manifest(path)
    assert video.load_mask(0) is None
    mask = video.load_mask(4)
    assert mask is not None and mask.dtype == torch.float32 and tuple(mask.shape) == (3, 4)
    expected = torch.from_numpy(_pixels(3, 4, 1, 104)[..., 0].astype(np.float64) / 255)
    torch.testing.assert_close(mask.double(), expected, rtol=0.0, atol=6e-8)
    bits = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    Image.fromarray(bits).save(tmp_path / "masks/test.png")
    mask = video.load_mask(4)
    assert mask is not None
    assert torch.equal(mask, torch.from_numpy(bits.astype(np.float32)))


def test_load_mask_checks_size_and_mode(tmp_path: Path) -> None:
    images = _ring_images(1)
    images[4]["mask"] = "masks/test.png"
    video = MultiViewVideo.from_manifest(_write_scene(tmp_path, _RING, images))
    Image.fromarray(_pixels(3, 5, 1, 0)[..., 0], "L").save(tmp_path / "masks/test.png")
    with pytest.raises(ValueError, match="size"):
        video.load_mask(4)
    Image.fromarray(_pixels(3, 4, 3, 0), "RGB").save(tmp_path / "masks/test.png")
    with pytest.raises(ValueError, match="mode"):
        video.load_mask(4)


def test_missing_files_are_refused_at_construction(tmp_path: Path) -> None:
    """Q-65: every image and mask file must exist before training starts."""
    images = _ring_images(1)
    path = _write_scene(tmp_path, _RING, images)
    (tmp_path / images[2]["path"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        MultiViewVideo.from_manifest(path)
    images[4]["mask"] = "masks/none.png"
    path = _write_scene(tmp_path, _RING, images)
    (tmp_path / "masks/none.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        MultiViewVideo.from_manifest(path)
