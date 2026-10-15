# SPDX-License-Identifier: Apache-2.0
"""U-20, converter part (§1.4 "World and camera frames", D-28, D-33; Q-84, Q-89): the camera
helpers of openftgs.data that the converters build cameras with.

Expected values are worked by hand: dyadic intrinsics and points, so every projection is exact in
float32, and rotations whose entries are 0 and ±1, so the LLFF conversion is exact too.
"""

import numpy as np
import pytest
import torch

from openftgs.data import camera_from_llff, camera_from_opencv, resize_camera, resized_size
from openftgs.types import Camera


def _project(camera: Camera, point: tuple[float, float, float]) -> tuple[float, float]:
    """A world point's pixel in the continuous convention of §1.4, in float64."""
    w2c = camera.world_to_camera.to(torch.float64)
    x = w2c[:3, :3] @ torch.tensor(point, dtype=torch.float64) + w2c[:3, 3]
    uvw = camera.K.to(torch.float64) @ x
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def test_u20_opencv_calibration_gains_half_a_pixel() -> None:
    """U-20, §1.4: an OpenCV calibration (pixel centres at integers) converts with cx + 0.5 and
    cy + 0.5. With fx = 100, fy = 80, c = (50, 25) and the identity pose, the camera point
    (0.25, −0.5, 1) is at OpenCV pixel (75, −15), so at (75.5, −14.5) in the continuous
    convention."""
    K = np.array([[100.0, 0.0, 50.0], [0.0, 80.0, 25.0], [0.0, 0.0, 1.0]])
    camera = camera_from_opencv(K, np.eye(3), np.zeros(3), width=101, height=51, name="0003")
    assert camera.name == "0003" and (camera.width, camera.height) == (101, 51)
    assert camera.K.dtype == torch.float32 and camera.world_to_camera.dtype == torch.float32
    assert camera.K[0, 2].item() == 50.5 and camera.K[1, 2].item() == 25.5
    assert _project(camera, (0.25, -0.5, 1.0)) == (75.5, -14.5)


def test_u20_opencv_extrinsics_are_world_to_camera() -> None:
    """U-20, §1.4 and D-63: R and T go straight into world_to_camera (x_cam = R·x + T). A
    camera rotated 90° about z (x_cam = −y_w, y_cam = x_w) and translated by T = (0, 0, 2)
    sees the world point (1, 0, 0) at camera (0, 1, 2), so at pixel (c + 0, c + f/2)."""
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    K = np.array([[64.0, 0.0, 31.5], [0.0, 64.0, 31.5], [0.0, 0.0, 1.0]])
    camera = camera_from_opencv(K, R, np.array([[0.0], [0.0], [2.0]]), width=64, height=64)
    expected = torch.eye(4)
    expected[:3, :3] = torch.tensor(R, dtype=torch.float32)
    expected[2, 3] = 2.0
    assert torch.equal(camera.world_to_camera, expected)
    assert _project(camera, (1.0, 0.0, 0.0)) == (32.0, 64.0)


@pytest.mark.parametrize(
    ("width", "height", "ratio", "expected"),
    [((2704, 2028, 0.5, (1352, 1014))), (50, 30, 0.37, (19, 11)), (3, 3, 0.1, (1, 1))],
)
def test_resized_size_rounds_half_up(
    width: int, height: int, ratio: float, expected: tuple[int, int]
) -> None:
    """§1.4, D-33, Q-89: W′ = round(W·r) with an exact half rounding up (50 × 0.37 = 18.5 →
    19; 30 × 0.37 = 11.1 → 11), and at least 1."""
    assert resized_size(width, height, ratio) == expected


def test_u20_resizing_scales_projections_by_the_realized_ratios() -> None:
    """U-20, §1.4: resizing scales fx and cx by W′/W and fy and cy by H′/H, so every projection
    scales by those ratios. 1352×1014 → 676×507 is r = ½ on both axes, and the pixel (75.5,
    −14.5) becomes (37.75, −7.25)."""
    K = np.array([[100.0, 0.0, 50.0], [0.0, 80.0, 25.0], [0.0, 0.0, 1.0]])
    camera = camera_from_opencv(K, np.eye(3), np.zeros(3), width=1352, height=1014)
    resized = resize_camera(camera, 676, 507)
    assert (resized.width, resized.height, resized.name) == (676, 507, camera.name)
    assert torch.equal(resized.world_to_camera, camera.world_to_camera)
    assert _project(resized, (0.25, -0.5, 1.0)) == (37.75, -7.25)


def test_resizing_uses_each_axis_ratio() -> None:
    """§1.4: with W·r or H·r not integral, fx and cx scale by W′/W and fy and cy by H′/H: 3×5
    at r = ½ is 2×3 (1.5 and 2.5 rounding up), ratios 2/3 and 3/5."""
    K = np.array([[3.0, 0.0, 1.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])
    camera = camera_from_opencv(K, np.eye(3), np.zeros(3), width=3, height=5)
    resized = resize_camera(camera, *resized_size(3, 5, 0.5))
    expected = torch.tensor(
        [[3.0 * 2 / 3, 0.0, 1.5 * 2 / 3], [0.0, 5.0 * 3 / 5, 2.5 * 3 / 5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).to(torch.float32)
    assert (resized.width, resized.height) == (2, 3)
    assert torch.equal(resized.K, expected)


def test_u20_llff_row_reproduces_a_known_opencv_pose() -> None:
    """U-20, D-28: a poses_bounds.npy row holds a camera-to-world rotation whose columns point
    down, right and back, the camera centre, and (H, W, focal), then near and far. Built from a
    known OpenCV pose (columns right, down, forward of the camera-to-world rotation), the row
    converts back to exactly that pose, with fx = fy = focal and c = (W/2, H/2)."""
    right, down, forward = np.array([0.0, 1, 0]), np.array([0.0, 0, -1]), np.array([-1.0, 0, 0])
    centre = np.array([3.0, -2.0, 1.5])
    height, width, focal = 2028.0, 2704.0, 1460.25
    matrix = np.stack([down, right, -forward, centre, [height, width, focal]], axis=1)
    row = np.concatenate([matrix.reshape(-1), [0.5, 100.0]])
    camera = camera_from_llff(row, name="cam07")
    rotation = np.stack([right, down, forward])  # world-to-camera: rows are the camera axes
    expected = torch.eye(4, dtype=torch.float64)
    expected[:3, :3] = torch.tensor(rotation)
    expected[:3, 3] = -torch.tensor(rotation) @ torch.tensor(centre)
    assert torch.equal(camera.world_to_camera, expected.to(torch.float32))
    assert torch.equal(
        camera.K,
        torch.tensor([[focal, 0, 1352.0], [0, focal, 1014.0], [0, 0, 1]], dtype=torch.float32),
    )
    assert (camera.width, camera.height, camera.name) == (2704, 2028, "cam07")
    # The same row as its 3×5 matrix.
    assert torch.equal(camera_from_llff(matrix).world_to_camera, camera.world_to_camera)
    # The centre lands on the optical axis: a point one unit ahead projects to c.
    assert _project(camera, tuple(centre + forward)) == (1352.0, 1014.0)


def test_camera_helpers_reject_malformed_input() -> None:
    """Q-84: a skewed K, a non-rotation R and an LLFF row of the wrong size are refused."""
    good = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        camera_from_opencv(
            good + np.diag([0, 0, 0]) + [[0, 0.1, 0], [0, 0, 0], [0, 0, 0]],
            np.eye(3),
            np.zeros(3),
            width=1,
            height=1,
        )
    with pytest.raises(ValueError):
        camera_from_opencv(good, 2 * np.eye(3), np.zeros(3), width=1, height=1)
    with pytest.raises(ValueError):
        camera_from_llff(np.zeros(16))
    with pytest.raises(ValueError):
        resized_size(10, 10, 0.0)
