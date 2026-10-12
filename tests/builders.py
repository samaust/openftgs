# SPDX-License-Identifier: Apache-2.0
"""Shared builders of random models and cameras for the test suite (forward note on PRD #20).

Each test file still writes out its own `PARAM_NAMES` from §1.1: that tuple is the tests'
independent expectation, not a duplicate to share.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openftgs.model import FreeTimeGaussians
from openftgs.types import Camera, ExchangeModel, TimeNormalizer

UNIT_TIME = TimeNormalizer(0.0, 1.0)
C0 = 0.28209479177387814  # §1.1


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
    spread: float = 1.0,
    log_scale: tuple[float, float] = (-3.0, -1.5),
    logit: tuple[float, float] = (-2.0, 2.0),
    log_duration: tuple[float, float] = (-1.5, 0.5),
    speed: float = 0.5,
    sh_scale: float = 0.5,
) -> FreeTimeGaussians:
    """A model of n random primitives from a generator seeded with `seed` (see `random_params`
    for the ranges)."""
    params = random_params(
        n,
        sh_degree=sh_degree,
        generator=torch.Generator().manual_seed(seed),
        spread=spread,
        log_scale=log_scale,
        logit=logit,
        log_duration=log_duration,
        speed=speed,
        sh_scale=sh_scale,
    )
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
    elevation: float = 1.0,
    width_px: int = 24,
    height_px: int = 16,
    focal: float = 20.0,
) -> list[Camera]:
    """`count` cameras evenly spaced on a horizontal ring of the given radius, `elevation` above
    the origin, looking at it."""
    return [
        look_at(
            (
                radius * math.cos(2 * math.pi * k / count),
                radius * math.sin(2 * math.pi * k / count),
                elevation,
            ),
            width=width_px,
            height=height_px,
            focal=focal,
            name=f"ring{k:02d}",
        )
        for k in range(count)
    ]


# Hand-built render scenes, shared by the render tests of both backends (E2, E6).


def model_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sh_degree: int = 0,
    time: TimeNormalizer = UNIT_TIME,
) -> FreeTimeGaussians:
    """A model with one primitive per row. Defaults: time center 0.5, ŝ = 20 (σt = 1 within
    |t − μt| ≤ 1), no velocity, isotropic scale 0.25, identity quaternion, logit 0 (σ = 0.5),
    sh0 = 0 (color 0.5) and no higher bands."""
    k = (sh_degree + 1) ** 2 - 1
    shapes: dict[str, tuple[int, ...]] = {
        "means": (3,),
        "times": (),
        "log_durations": (),
        "velocities": (3,),
        "log_scales": (3,),
        "quats": (4,),
        "opacity_logits": (),
        "sh0": (1, 3),
        "shN": (k, 3),
    }
    cols: dict[str, list[Any]] = {name: [] for name in shapes}
    for row in rows:
        scale = row.get("scale", 0.25)
        scales = [scale] * 3 if isinstance(scale, int | float) else list(scale)
        cols["means"].append(list(row["mean"]))
        cols["times"].append(row.get("time", 0.5))
        cols["log_durations"].append(row.get("log_duration", 20.0))
        cols["velocities"].append(list(row.get("velocity", (0.0, 0.0, 0.0))))
        cols["log_scales"].append([math.log(a) for a in scales])
        cols["quats"].append(list(row.get("quat", (1.0, 0.0, 0.0, 0.0))))
        cols["opacity_logits"].append(row.get("logit", 0.0))
        cols["sh0"].append([list(row.get("sh0", (0.0, 0.0, 0.0)))])
        cols["shN"].append(row.get("shN", [[0.0, 0.0, 0.0]] * k))
    params = {
        name: torch.tensor(cols[name], dtype=torch.float32).reshape(len(rows), *shape)
        for name, shape in shapes.items()
    }
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


# Camera A: the point (X, Y, 4) lands at (32.5 + 4X, 24.5 + 4Y).
CAM_A = pinhole(64, 48, fx=16.0, fy=16.0, cx=32.5, cy=24.5)


def sh0_for(rgb: Sequence[float]) -> tuple[float, float, float]:
    """sh0 giving the color rgb at degree 0 (E2: C₀·sh0 + 0.5)."""
    r, g, b = ((c - 0.5) / C0 for c in rgb)
    return (r, g, b)


def logit_of(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid64(logit: float) -> float:
    """σ of a logit as stored (float32), evaluated in float64."""
    return float(torch.sigmoid(torch.tensor(logit, dtype=torch.float32).to(torch.float64)))


# U-8's scenes (D-32, D-57), shared by both backends' U-8 tests.


def u8_scene(
    n: int, dtype: torch.dtype, *, kept: int | None = None, tie: bool = False
) -> FreeTimeGaussians:
    """n random rows around `ring(3)`, some behind a camera or off the image, with every third row
    below the opacity cull (logit −9). With `kept`, exactly that many rows pass the cull. With
    `tie`, rows 4 and 5 sit at the same place, opaque enough to be seen by every camera, with
    different colors, so they tie in depth."""
    params = random_params(
        n,
        generator=torch.Generator().manual_seed(n),
        spread=4.5,
        log_scale=(-2.5, -0.5),
        logit=(-3.0, 3.0),
    )
    params["opacity_logits"][::3] = -9.0
    if kept is not None:
        params["opacity_logits"] = torch.tensor([3.0] * kept + [-9.0] * (n - kept))
    if tie:
        for row in (4, 5):
            params["means"][row] = torch.tensor([0.1, 0.2, 0.3])
            params["velocities"][row] = 0.0
            params["times"][row] = 0.37
            params["opacity_logits"][row] = 2.0
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME)
    return model.double() if dtype == torch.float64 else model


U8_CASES: dict[str, tuple[int, int | None, bool]] = {  # n, rows passing the cull, depth tie
    "no row kept": (1, 0, False),
    "one row kept": (2, 1, False),
    "a third culled": (7, None, False),
    "depth tie": (33, None, True),
    "depth tie, more rows": (64, None, True),
}


# U-4's change of time unit (E14, §1.4 "Time"), shared by both backends' U-4 tests.


def rescaled_time_unit(exchange: ExchangeModel, a: float, b: float) -> ExchangeModel:
    """The exchange model in the time unit τ′ = aτ + b: time centers aτ + b, durations a·d,
    velocities v_τ/a (float32, as ftgs-if stores them), τ₀ and τ₁ mapped too."""
    arrays = dict(exchange.arrays)
    arrays["time_center"] = a * arrays["time_center"] + b
    arrays["duration"] = a * arrays["duration"]
    arrays["velocity"] = (arrays["velocity"].to(torch.float64) / a).to(torch.float32)
    return ExchangeModel(
        arrays=arrays,
        sh_degree=exchange.sh_degree,
        tau0=a * exchange.tau0 + b,
        tau1=a * exchange.tau1 + b,
        time_unit=exchange.time_unit,
        producer=exchange.producer,
    )
