# SPDX-License-Identifier: Apache-2.0
"""Dataclasses for the package's public data: the frozen Camera, ImageRecord, InitPoints,
TimeNormalizer and ExchangeModel, RenderOutput, and TrainerState, the plain data a checkpoint
holds; and the frame interval Δt_f of E14.

Spec: §1.4, §2.2, E14. Band: Foundation (§2.1).
"""

import math
import statistics
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import torch
from torch import Tensor

__all__ = [
    "Camera",
    "ExchangeModel",
    "ImageRecord",
    "InitPoints",
    "RenderOutput",
    "TimeNormalizer",
    "TrainerState",
    "frame_interval",
]


@dataclass(frozen=True)
class Camera:
    K: Tensor  # [3, 3] float32, continuous pixel convention (§1.4)
    world_to_camera: Tensor  # [4, 4] float32, OpenCV axes
    width: int
    height: int
    name: str = ""


@dataclass(frozen=True)
class ImageRecord:
    camera: Camera
    timestamp: float  # raw τ (float64)
    image_path: Path
    split: Literal["train", "test"]
    frame_index: int | None = None


@dataclass(frozen=True)
class InitPoints:
    positions: Tensor  # [N, 3] float32, world units
    colors: Tensor  # [N, 3] float32 in [0, 1]
    timestamps: Tensor  # [N] float64, raw τ
    velocities: Tensor | None = None  # [N, 3] world units per unit of τ
    durations: Tensor | None = None  # [N] float64, raw τ units (std. dev. of E4)
    frame_ids: Tensor | None = None  # [N] int64; groups points for the 3-NN scale init


@dataclass(frozen=True)
class TimeNormalizer:
    """E14, D-1: t = (τ − τ₀)/(τ₁ − τ₀) over the training timestamps."""

    tau0: float
    tau1: float

    def __post_init__(self) -> None:
        # Q-9: finite bounds with τ₁ ≥ τ₀; τ₁ = τ₀ is the single-frame case of §1.4.
        if not (math.isfinite(self.tau0) and math.isfinite(self.tau1)):
            raise ValueError(f"tau0 and tau1 must be finite, got {self.tau0!r}, {self.tau1!r}")
        if self.tau1 < self.tau0:
            raise ValueError(f"tau1 must be >= tau0, got tau0={self.tau0!r}, tau1={self.tau1!r}")
        object.__setattr__(self, "tau0", float(self.tau0))
        object.__setattr__(self, "tau1", float(self.tau1))

    @overload
    def to_t(self, tau: float) -> float: ...
    @overload
    def to_t(self, tau: Tensor) -> Tensor: ...
    def to_t(self, tau: float | Tensor) -> float | Tensor:
        """Raw τ → normalized t; warns outside [τ₀, τ₁] (§1.4; Q-9)."""
        if isinstance(tau, Tensor):
            # Q-9: the tensor check syncs with the host; training passes floats.
            outside = bool(((tau < self.tau0) | (tau > self.tau1)).any())
        else:
            tau = float(tau)
            outside = tau < self.tau0 or tau > self.tau1
        if outside:
            warnings.warn(
                f"timestamp outside the training span [{self.tau0!r}, {self.tau1!r}]: "
                "motion extrapolates linearly",
                UserWarning,
                stacklevel=2,
            )
        span = self.tau1 - self.tau0
        if span == 0.0:
            # §1.4: one frame, t ≡ 0.
            return torch.zeros_like(tau) if isinstance(tau, Tensor) else 0.0
        return (tau - self.tau0) / span

    @overload
    def to_tau(self, t: float) -> float: ...
    @overload
    def to_tau(self, t: Tensor) -> Tensor: ...
    def to_tau(self, t: float | Tensor) -> float | Tensor:
        """Normalized t → raw τ; never warns (Q-9)."""
        if not isinstance(t, Tensor):
            t = float(t)
        return self.tau0 + t * (self.tau1 - self.tau0)


@dataclass
class RenderOutput:
    image: Tensor  # [C, H, W, 3]
    alpha: Tensor  # [C, H, W, 1]
    # [C, N, 2] unpacked or [nnz, 2] packed; kept for gradient statistics
    means2d: Tensor | None = None
    visible: Tensor | None = None  # [C, N] bool, unpacked mode
    # [nnz] int, packed mode: the primitive behind each row of means2d
    gaussian_ids: Tensor | None = None


@dataclass(frozen=True)
class ExchangeModel:  # FTGS-IF contents (§2.2): physical values, raw time units
    # position, time_center, duration, velocity, scale, rotation, opacity, sh
    arrays: dict[str, Tensor]
    sh_degree: int
    tau0: float
    tau1: float
    time_unit: str
    producer: str


@dataclass
class TrainerState:  # plain tensors and dicts; a §2.2 checkpoint holds exactly this
    iteration: int
    config: dict[str, Any]  # TrainConfig as JSON-ready values
    time: TimeNormalizer
    spatial_scale: float  # S (D-27)
    frame_dt: float  # Δt_f (E14)
    params: dict[str, Tensor]  # the nine stored tensors of §1.1
    optim: dict[str, Any]  # per tensor: exp_avg, exp_avg_sq, step, schedule parameters
    stats: dict[str, Tensor]  # g_sum, count
    rng: dict[str, Any]  # generator states
    sampler: dict[str, Any]  # epoch, cursor, permutation


def frame_interval(camera_timestamps: Mapping[str, Sequence[float]], time: TimeNormalizer) -> float:
    """Δt_f of E14 in normalized time, from each training camera's raw timestamps (D-46).

    The median over cameras with at least two images of each camera's median gap; without
    such a camera, the median gap of the pooled distinct timestamps; 1 for a single timestamp.
    """
    # Q-8: statistical medians (even counts average) over raw gaps, divided by T_τ once;
    # duplicates within a camera are kept, only the pooled fallback takes distinct values.
    if not camera_timestamps:
        raise ValueError("frame_interval needs at least one camera")
    span = time.tau1 - time.tau0
    if span == 0.0:
        return 1.0
    medians = [
        statistics.median(_gaps(sorted(float(x) for x in stamps)))
        for stamps in camera_timestamps.values()
        if len(stamps) >= 2
    ]
    if medians:
        gap = statistics.median(medians)
    else:
        pooled = sorted({float(x) for stamps in camera_timestamps.values() for x in stamps})
        if len(pooled) < 2:
            return 1.0
        gap = statistics.median(_gaps(pooled))
    result = gap / span
    if not (math.isfinite(result) and result > 0.0):
        raise ValueError(f"frame interval must be finite and positive, got {result!r}")
    return result


def _gaps(ordered: list[float]) -> list[float]:
    return [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
