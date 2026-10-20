# SPDX-License-Identifier: Apache-2.0
"""Cameras for the harness's cases, in §1.4's conventions: OpenCV axes (x right, y down, z
forward), world-to-camera matrices, K in the continuous pixel convention."""

from dataclasses import dataclass

import numpy as np

__all__ = ["Cam", "intrinsics", "look_at"]


@dataclass(frozen=True)
class Cam:
    name: str
    K: np.ndarray  # [3, 3] float64
    world_to_camera: np.ndarray  # [4, 4] float64
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        """E2's camera centre −R_wᵀ t_w."""
        r, t = self.world_to_camera[:3, :3], self.world_to_camera[:3, 3]
        return np.asarray(-(r.T @ t))


def intrinsics(width: int, height: int, focal: float) -> np.ndarray:
    """A pinhole K with the principal point at the image centre (continuous convention)."""
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


_UP = np.array([0.0, 0.0, 1.0])


def look_at(eye: np.ndarray, target: np.ndarray | None = None, up: np.ndarray = _UP) -> np.ndarray:
    """The world-to-camera matrix of a camera at `eye` looking at `target` (the origin by
    default), OpenCV axes, with +z of the world up."""
    forward = (np.zeros(3) if target is None else target) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ eye
    return out
