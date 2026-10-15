# SPDX-License-Identifier: Apache-2.0
"""U-34, U-35 and U-36 (D-33, D-61; Q-84, Q-89 to Q-91): area resizing, undistortion and the
OpenCV-style YAML reader of openftgs.data, which replace OpenCV in the core.

The oracles are independent of the code under test: a float64 integration over pixel squares
written here as plain loops (U-34), and OpenCV itself, a test-only dependency (§2.3), for
INTER_AREA, the undistortion maps and images, and FileStorage's readback.
"""

import math

import cv2
import numpy as np
import pytest

from openftgs.data import resize_area, resized_size, undistort_image, undistortion_map


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
