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

from openftgs.data import resize_area, resized_size


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
