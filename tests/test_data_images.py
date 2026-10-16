# SPDX-License-Identifier: Apache-2.0
"""U-34, U-35 and U-36 (D-33, D-61; Q-84, Q-89 to Q-91): area resizing, undistortion and the
OpenCV-style YAML reader of openftgs.data, which replace OpenCV in the core.

The oracles are independent of the code under test: a float64 integration over pixel squares
written here as plain loops (U-34), and OpenCV itself, a test-only dependency (§2.3), for
INTER_AREA, the undistortion maps and images, and FileStorage's readback.
"""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from openftgs.data import (
    read_opencv_yaml,
    resize_area,
    resized_size,
    undistort_image,
    undistortion_map,
)


def _integrate(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """U-34's oracle: each output pixel is the mean of the input over its footprint, the
    input's pixels being constant unit squares, computed by summing exact overlaps in float64."""
    in_h, in_w = image.shape[:2]
    sx, sy = in_w / width, in_h / height
    out = np.zeros((height, width, *image.shape[2:]), dtype=np.float64)
    for v in range(height):
        top, bottom = v * sy, (v + 1) * sy
        for u in range(width):
            left, right = u * sx, (u + 1) * sx
            total = np.zeros(image.shape[2:], dtype=np.float64)
            for y in range(math.floor(top), min(in_h, math.ceil(bottom))):
                dy = min(y + 1, bottom) - max(y, top)
                for x in range(math.floor(left), min(in_w, math.ceil(right))):
                    dx = min(x + 1, right) - max(x, left)
                    total += dx * dy * image[y, x]
            out[v, u] = total / (sx * sy)
    return out


@pytest.mark.parametrize("ratio", [0.5, 0.25, 0.37, 1.0])
def test_u34_area_resizing_matches_integration_over_pixel_squares(ratio: float) -> None:
    """U-34, D-33: random images at ratios 0.5, 0.25, 0.37 and 1.0 against an independent
    float64 integration over pixel squares, to 1e-12."""
    rng = np.random.default_rng(34)
    image = rng.uniform(0.0, 255.0, size=(23, 50, 3))
    width, height = resized_size(50, 23, ratio)
    got = resize_area(image, width, height)
    assert got.dtype == np.float64 and got.shape == (height, width, 3)
    np.testing.assert_allclose(got, _integrate(image, width, height), rtol=0.0, atol=1e-12)


def test_u34_area_resizing_of_a_single_channel_image() -> None:
    """U-34: [H, W] images resize like [H, W, C] ones."""
    rng = np.random.default_rng(35)
    image = rng.uniform(0.0, 1.0, size=(9, 14))
    got = resize_area(image, 5, 3)
    assert got.shape == (3, 5)
    np.testing.assert_allclose(got, _integrate(image, 5, 3), rtol=0.0, atol=1e-12)


def test_u34_ratio_one_half_matches_opencv_inter_area() -> None:
    """U-34: at ratio 0.5, 8-bit images within one 8-bit level of OpenCV's INTER_AREA."""
    rng = np.random.default_rng(36)
    image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    got = resize_area(image, 48, 32)
    want = cv2.resize(image, (48, 32), interpolation=cv2.INTER_AREA)
    assert got.dtype == np.uint8 and got.shape == want.shape
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


def test_u34_eight_bit_output_rounds_half_to_even() -> None:
    """D-33: 8-bit output is rounded half to even: the means 1.5, 2.5 and 3.5 round to 2, 2
    and 4."""
    image = np.array([[1, 2, 2, 3, 3, 4]], dtype=np.uint8)
    assert resize_area(image, 3, 1).tolist() == [[2, 2, 4]]


# U-35 · undistortion (D-61).

_K = np.array([[180.0, 0.0, 97.3], [0.0, 175.0, 61.8], [0.0, 0.0, 1.0]])  # OpenCV convention
_COEFFICIENTS = {
    4: [-0.21, 0.05, 0.0012, -0.0008],
    5: [-0.21, 0.05, 0.0012, -0.0008, -0.01],
    8: [-0.21, 0.05, 0.0012, -0.0008, -0.01, 0.02, -0.003, 0.001],
    12: [-0.21, 0.05, 0.0012, -0.0008, -0.01, 0.02, -0.003, 0.001, 0.002, -0.001, 0.0015, 0.0005],
}


@pytest.mark.parametrize("count", [4, 5, 8, 12])
def test_u35_maps_match_opencv(count: int) -> None:
    """U-35, D-61: the source-position maps for the 4-, 5-, 8- and 12-coefficient models agree
    with OpenCV's initUndistortRectifyMap (R = I, new camera matrix = K) to 1e-3 px."""
    dist = np.array(_COEFFICIENTS[count])
    got = undistortion_map(_K, dist, 200, 120)
    map_x, map_y = cv2.initUndistortRectifyMap(_K, dist, None, _K, (200, 120), cv2.CV_32FC1)
    assert got.shape == (120, 200, 2) and got.dtype == np.float64
    np.testing.assert_allclose(got[..., 0], map_x, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(got[..., 1], map_y, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("count", [4, 5, 8, 12])
def test_u35_images_match_opencv_away_from_the_border(count: int) -> None:
    """U-35, D-61: images of a smooth pattern (gradients well below 32 levels per pixel) agree
    with OpenCV's undistort within one 8-bit level wherever the source position lies at least
    one pixel inside the image."""
    y, x = np.mgrid[0:120, 0:200].astype(np.float64)
    pattern = np.stack(
        [128 + 90 * np.sin(x / 17) * np.cos(y / 23), 40 + 1.5 * x, 200 - 1.2 * y], axis=-1
    )
    image = np.clip(np.rint(pattern), 0, 255).astype(np.uint8)
    dist = np.array(_COEFFICIENTS[count])
    got = undistort_image(image, _K, dist)
    want = cv2.undistort(image, _K, dist)
    map_x, map_y = cv2.initUndistortRectifyMap(_K, dist, None, _K, (200, 120), cv2.CV_32FC1)
    inside = (map_x >= 1) & (map_x <= 200 - 2) & (map_y >= 1) & (map_y <= 120 - 2)
    assert got.dtype == np.uint8 and got.shape == image.shape
    assert inside.mean() > 0.8
    assert np.abs(got.astype(int) - want.astype(int))[inside].max() <= 1


def test_u35_zero_coefficients_are_the_identity() -> None:
    """U-35: zero coefficients (none, or all zero) leave the image unchanged."""
    image = np.random.default_rng(1).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    assert np.array_equal(undistort_image(image, _K, []), image)
    assert np.array_equal(undistort_image(image, _K, np.zeros(5)), image)


def test_u35_pixels_without_a_source_are_zero() -> None:
    """D-61: output pixels whose source position lies outside the image are 0."""
    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    K = np.array([[60.0, 0.0, 39.5], [0.0, 60.0, 29.5], [0.0, 0.0, 1.0]])
    got = undistort_image(image, K, [0.8, 0.0, 0.0, 0.0])  # strong pincushion: corners leave
    assert got[0, 0].tolist() == [0, 0, 0]
    assert got[30, 40].tolist() == [200, 200, 200]


@pytest.mark.parametrize("count", [14, 3, 6])
def test_u35_tilted_and_unknown_models_are_rejected(count: int) -> None:
    """U-35, D-61: a 14-coefficient (tilted) input is rejected with a clear error, as is any
    count OpenCV doesn't define."""
    with pytest.raises(ValueError, match="tilted" if count == 14 else "coefficients"):
        undistortion_map(_K, np.zeros(count), 20, 10)


# U-36 · the OpenCV-style YAML reader (D-61).


def _opencv_values(path: Path) -> dict[str, Any]:
    """U-36's oracle: every top-level value as OpenCV's FileStorage reads it back."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    values: dict[str, Any] = {}
    for key in storage.root().keys():
        node = storage.getNode(key)
        if node.isSeq():
            values[key] = [node.at(i).string() for i in range(node.size())]
        elif node.isString():
            values[key] = node.string()
        elif node.isInt():
            values[key] = int(node.real())
        elif node.isReal():
            values[key] = node.real()
        else:
            values[key] = node.mat()
    storage.release()
    return values


def _assert_same(got: Any, want: Any) -> None:
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray)
        assert got.dtype == want.dtype and got.shape == want.shape
        if np.issubdtype(want.dtype, np.floating):
            # bitwise wherever the value is a number, NaN where OpenCV reads NaN
            nan = np.isnan(want)
            assert np.array_equal(np.isnan(got), nan)
            assert got[~nan].tobytes() == want[~nan].tobytes()
        else:
            assert got.tobytes() == want.tobytes()
    elif isinstance(want, float) and math.isnan(want):
        assert isinstance(got, float) and math.isnan(got)
    else:
        assert got == want and type(got) is type(want)


def _write_with_opencv(path: Path) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("names", ["0000", "0015", "0023"])
    rng = np.random.default_rng(36)
    storage.write("K_0015", np.array([[1460.25, 0.0, 959.5], [0.0, 1461.0, 539.5], [0, 0, 1]]))
    storage.write("D_0015", np.array([[-0.21], [0.05], [1e-5], [-3e-4], [0.0]]))
    storage.write("F_f32", rng.normal(size=(3, 4)).astype(np.float32))
    storage.write("F_special", np.array([[np.nan, np.inf, -np.inf, -0.0, 1 / 3]], np.float32))
    storage.write("I_i32", rng.integers(-1000, 1000, size=(2, 3), dtype=np.int32))
    storage.write("U_u8", rng.integers(0, 256, size=(1, 5), dtype=np.uint8))
    storage.write("C_i8", np.array([[-3, 4]], np.int8))
    storage.write("W_u16", np.array([[65535, 1]], np.uint16))
    storage.write("S_i16", np.array([[-32768, 7]], np.int16))
    storage.write("H_f16", np.array([[1.5, -2.25]], np.float16))
    storage.write("M_3d", rng.normal(size=(2, 2, 3)))
    storage.write("H_0015", 2160)
    storage.write("x", 0.1)
    storage.write("big", 1e300)
    storage.write("label", "hair")
    storage.release()


@pytest.mark.parametrize("header", ["as written", "%YAML:1.0"])
def test_u36_reads_what_opencv_writes(tmp_path: Path, header: str) -> None:
    """U-36, D-61: files written by OpenCV's FileStorage (opencv-matrix entries of several
    shapes and types, a sequence of names, scalars) read back with every value equal to what
    OpenCV reads, bitwise; once as OpenCV 5 writes them (`%YAML 1.2`) and once with the
    `%YAML:1.0` header older OpenCV writes, which isn't valid YAML (Q-91)."""
    path = tmp_path / "calib.yml"
    _write_with_opencv(path)
    if header != "as written":
        lines = path.read_text().splitlines()
        assert lines[0].startswith("%YAML")
        path.write_text("\n".join([header, *lines[1:]]) + "\n")
    want = _opencv_values(path)
    got = read_opencv_yaml(path)
    assert set(got) == set(want)
    for key in want:
        _assert_same(got[key], want[key])


def test_u36_reads_a_hand_written_calibration_in_the_selfcap_shape(tmp_path: Path) -> None:
    """U-36, D-63: a file in the shape of SelfCap's optimized/intri.yml and extri.yml (names,
    K_, D_ 5×1, H_, W_, ccm_, R_ as a Rodrigues vector, Rot_, T_), written by hand with the
    `%YAML:1.0` header and invented values, reads back as OpenCV reads it."""
    text = """%YAML:1.0
---
names:
   - "0000"
   - "0001"
K_0000: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 2000., 0., 1919.5, 0., 2001., 1079.5, 0., 0., 1. ]
D_0000: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ -2.5e-01, 1.25e-01, 0., 0., 0. ]
H_0000: 2160
W_0000: 3840
ccm_0000: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
R_0000: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 1.2e-01, -2.5e+00, 3.0e-02 ]
Rot_0000: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ -0.8, 0., -0.6, 0., 1., 0., 0.6, 0., -0.8 ]
T_0000: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0.5, -1.25, 4. ]
"""
    path = tmp_path / "intri.yml"
    path.write_text(text)
    want = _opencv_values(path)
    got = read_opencv_yaml(path)
    assert (
        set(got)
        == set(want)
        == {
            "names",
            "K_0000",
            "D_0000",
            "H_0000",
            "W_0000",
            "ccm_0000",
            "R_0000",
            "Rot_0000",
            "T_0000",
        }
    )
    for key in want:
        _assert_same(got[key], want[key])
    assert got["names"] == ["0000", "0001"]


def test_yaml_reader_rejects_files_that_are_not_calibration_maps(tmp_path: Path) -> None:
    """Q-91: a file whose top level isn't a mapping, or that isn't YAML, is a ValueError."""
    path = tmp_path / "bad.yml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        read_opencv_yaml(path)
    path.write_text("a: [1, 2\n")
    with pytest.raises(ValueError):
        read_opencv_yaml(path)


def test_yaml_reader_rejects_a_malformed_matrix(tmp_path: Path) -> None:
    """Q-91, Q-26: a matrix without rows and cols, or with data that aren't numbers, is a
    ValueError (exit code 2 at M14), not a KeyError."""
    path = tmp_path / "bad.yml"
    path.write_text("%YAML:1.0\n---\nK: !!opencv-matrix\n   cols: 1\n   dt: d\n   data: [ 1. ]\n")
    with pytest.raises(ValueError):
        read_opencv_yaml(path)
    path.write_text("K: !!opencv-matrix\n   rows: 1\n   cols: 1\n   dt: d\n   data: [ x ]\n")
    with pytest.raises(ValueError):
        read_opencv_yaml(path)
