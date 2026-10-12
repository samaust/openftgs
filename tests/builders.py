# SPDX-License-Identifier: Apache-2.0
"""Shared builders of random models and cameras for the test suite (forward note on PRD #20).

Each test file still writes out its own `PARAM_NAMES` from §1.1: that tuple is the tests'
independent expectation, not a duplicate to share.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from openftgs.model import FreeTimeGaussians
from openftgs.types import Camera, TimeNormalizer

UNIT_TIME = TimeNormalizer(0.0, 1.0)


def random_params(
    n: int,
    *,
    sh_degree: int = 3,
    generator: torch.Generator,
    spread: float = 1.0,
    log_scale: tuple[float, float] = (-3.0, -1.5),
    logit: tuple[float, float] = (-2.0, 2.0),
    log_duration: tuple[float, float] = (-1.5, 0.5),
    speed: float = 0.5,
    sh_scale: float = 0.5,
) -> dict[str, Tensor]:
    """The nine stored tensors of §1.1 (float32) for n random primitives: means in a cube of
    half-width `spread` around the origin, time centers in [0, 1], and the other quantities drawn
    uniformly from the given ranges (logs for scales and durations)."""

    def uniform(*shape: int, low: float, high: float) -> Tensor:
        return low + (high - low) * torch.rand(*shape, generator=generator)

    return {
        "means": uniform(n, 3, low=-spread, high=spread),
        "times": torch.rand(n, generator=generator),
        "log_durations": uniform(n, low=log_duration[0], high=log_duration[1]),
        "velocities": speed * torch.randn(n, 3, generator=generator),
        "log_scales": uniform(n, 3, low=log_scale[0], high=log_scale[1]),
        "quats": torch.randn(n, 4, generator=generator),
        "opacity_logits": uniform(n, low=logit[0], high=logit[1]),
        "sh0": sh_scale * torch.randn(n, 1, 3, generator=generator),
        "shN": sh_scale * torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, generator=generator),
    }


def random_model(
    n: int,
    *,
    seed: int = 0,
    sh_degree: int = 3,
    time: TimeNormalizer = UNIT_TIME,
    **ranges: float | tuple[float, float],
) -> FreeTimeGaussians:
    """A model of n random primitives (see `random_params` for the ranges)."""
    generator = torch.Generator().manual_seed(seed)
    params = random_params(n, sh_degree=sh_degree, generator=generator, **ranges)  # type: ignore[arg-type]
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


def pinhole(
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    world_to_camera: Sequence[Sequence[float]] | Tensor | None = None,
    name: str = "",
) -> Camera:
    """A camera in the continuous pixel convention of §1.4 (identity pose by default)."""
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    if world_to_camera is None:
        w2c = torch.eye(4, dtype=torch.float32)
    else:
        w2c = torch.as_tensor(world_to_camera, dtype=torch.float32)
        if w2c.shape == (3, 4):
            w2c = torch.cat([w2c, torch.tensor([[0.0, 0.0, 0.0, 1.0]])])
    return Camera(K=K, world_to_camera=w2c, width=width, height=height, name=name)


def look_at(
    eye: Sequence[float],
    target: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    width: int,
    height: int,
    focal: float,
    name: str = "",
) -> Camera:
    """A camera at `eye` looking at `target`, with OpenCV axes (x right, y down, z forward;
    §1.4) and the principal point at the image center."""
    eye_t = torch.tensor(eye, dtype=torch.float64)
    z = torch.tensor(target, dtype=torch.float64) - eye_t
    z = z / torch.linalg.vector_norm(z)
    x = torch.linalg.cross(z, torch.tensor(up, dtype=torch.float64))
    x = x / torch.linalg.vector_norm(x)
    y = torch.linalg.cross(z, x)
    rotation = torch.stack([x, y, z])
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ eye_t
    return pinhole(
        width,
        height,
        fx=focal,
        fy=focal,
        cx=width / 2,
        cy=height / 2,
        world_to_camera=w2c.to(torch.float32),
        name=name,
    )


def ring(
    count: int,
    *,
    radius: float = 4.0,
    height: float = 1.0,
    width_px: int = 24,
    height_px: int = 16,
    focal: float = 20.0,
) -> list[Camera]:
    """`count` cameras evenly spaced on a horizontal ring around the origin, looking at it."""
    return [
        look_at(
            (
                radius * math.cos(2 * math.pi * k / count),
                radius * math.sin(2 * math.pi * k / count),
                height,
            ),
            width=width_px,
            height=height_px,
            focal=focal,
            name=f"ring{k:02d}",
        )
        for k in range(count)
    ]
