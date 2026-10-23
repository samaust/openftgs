# SPDX-License-Identifier: Apache-2.0
"""Dataclasses for the package's public data, and the frame interval of E14.

The frozen :class:`Camera`, :class:`ImageRecord`, :class:`InitPoints`, :class:`TimeNormalizer` and
:class:`ExchangeModel`, :class:`RenderOutput`, and :class:`TrainerState`, the plain data a
checkpoint holds; the render request and its renders (Q-56); the parsed scene manifest (Q-64); and
the frame interval Δt_f of E14 (:func:`frame_interval`). Raw times τ are float64 in a scene's own
unit; normalized times t span [0, 1] over the training timestamps (§1.4 "Time").

Spec: §1.4, §2.2, E14. Band: Foundation (§2.1).
"""

import itertools
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
    "RenderItem",
    "RenderOutput",
    "RenderRequest",
    "Renders",
    "SceneManifest",
    "TimeNormalizer",
    "TrainerState",
    "frame_interval",
]


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: intrinsics, world-to-camera extrinsics and image size (§1.4).

    The conventions are gsplat's (D-28): OpenCV axes (x right, y down, z forward into the scene)
    and the continuous pixel convention, in which the image spans [0, W] × [0, H] and pixel
    (u, v) is sampled at (u + 0.5, v + 0.5). Lens distortion is not modeled: images are
    undistorted.

    Parameters
    ----------
    K
        Intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], [3, 3] float32, in pixels, in the
        continuous pixel convention (§1.4).
    world_to_camera
        Extrinsics, [4, 4] float32, OpenCV axes: ``x_cam = R·x + t`` in world units (gsplat's
        ``viewmats``).
    width
        Image width W, in pixels.
    height
        Image height H, in pixels.
    name
        The camera's name, as the key of the scene manifest's ``cameras`` object; empty by
        default.
    """

    K: Tensor  # [3, 3] float32, continuous pixel convention (§1.4)
    world_to_camera: Tensor  # [4, 4] float32, OpenCV axes
    width: int
    height: int
    name: str = ""


@dataclass(frozen=True)
class ImageRecord:
    """One image of a scene: its camera, raw timestamp, file and split (§1.4, §2.2).

    Every image has its own timestamp, so unsynchronized cameras need no special case; the frame
    index is optional metadata (§1.4 "Time").

    Parameters
    ----------
    camera
        The camera that took the image, with the image's own ``K`` and ``world_to_camera`` when
        the scene manifest overrides them (§2.2).
    timestamp
        The raw time τ, float64, in the scene's time unit.
    image_path
        The image file; once loaded, float32 RGB in [0, 1] (§1.4 "Images").
    split
        Whether the image is for training or for testing.
    frame_index
        The frame's index, or None when the scene gives none.
    """

    camera: Camera
    timestamp: float  # raw τ (float64)
    image_path: Path
    split: Literal["train", "test"]
    frame_index: int | None = None


@dataclass(frozen=True)
class InitPoints:
    """Initial points: positions, colors, raw times and optional motion (§2.2 ``ftgs-init``).

    The contents of a ``ftgs-init`` file, which :func:`openftgs.io.validate_init_points` checks.
    N ≥ 1 points, every value finite.

    Parameters
    ----------
    positions
        World coordinates, [N, 3] float32, in world units.
    colors
        RGB colors, [N, 3] float32, in [0, 1].
    timestamps
        The raw time τ of each point, [N] float64.
    velocities
        Velocities, [N, 3] float32, in world units per unit of τ; None means 0 (§1.3).
    durations
        Temporal standard deviations d of E4, [N] float64, in units of τ, > 0; None means the
        default duration s₀ = Δt_f (D-21).
    frame_ids
        Frame slots, [N] int64, which group the points for the 3-NN scale initialization; None
        means the slot nearest each point's time (§1.3).
    """

    positions: Tensor  # [N, 3] float32, world units
    colors: Tensor  # [N, 3] float32 in [0, 1]
    timestamps: Tensor  # [N] float64, raw τ
    velocities: Tensor | None = None  # [N, 3] world units per unit of τ
    durations: Tensor | None = None  # [N] float64, raw τ units (std. dev. of E4)
    frame_ids: Tensor | None = None  # [N] int64; groups points for the 3-NN scale init


@dataclass(frozen=True)
class TimeNormalizer:
    """Map raw time τ to normalized time t over the training timestamps (E14, D-1).

    t = (τ − τ₀)/(τ₁ − τ₀), so the training timestamps span [0, 1]. With a single timestamp
    (τ₀ = τ₁), t ≡ 0 (§1.4 "Time"). The bounds are stored as floats.

    Parameters
    ----------
    tau0
        τ₀, the smallest training timestamp, in raw units; finite.
    tau1
        τ₁, the largest training timestamp, in raw units; finite and ≥ ``tau0``.

    Raises
    ------
    ValueError
        If a bound is not finite, or ``tau1`` is below ``tau0``.

    Notes
    -----
    Q-9 fixes the validation, the single-frame case, which conversion warns and the dtypes of
    the results; raw times are float64 (D-30).
    """

    tau0: float
    tau1: float

    def __post_init__(self) -> None:
        """Check the bounds and store them as floats (Q-9)."""
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
        """Convert raw time τ to normalized time t (E14).

        Outside [τ₀, τ₁] the motion extrapolates linearly, so such a time is converted but warns
        (§1.4 "Time"). With τ₀ = τ₁, t is 0.

        Parameters
        ----------
        tau
            Raw time, in the scene's unit: a number, or a tensor of any shape. An integer tensor
            is converted to float64 first.

        Returns
        -------
        float or torch.Tensor
            t = (τ − τ₀)/(τ₁ − τ₀): a float for a number; for a tensor, a tensor of the same
            shape and floating dtype (float64 for an integer tensor).

        Warns
        -----
        UserWarning
            If a value of ``tau`` is not inside [τ₀, τ₁], NaN included.

        Notes
        -----
        The warning, the single-frame case and the dtypes follow Q-9, the float64 of an integer
        tensor D-30. The range check of a tensor synchronizes with the host; the training step
        passes floats.
        """
        # Q-9: NaN counts as outside; the tensor check syncs with the host (training passes
        # floats); integer tensors become float64 (D-30).
        if isinstance(tau, Tensor):
            if not tau.is_floating_point():
                tau = tau.to(torch.float64)
            outside = bool((~((tau >= self.tau0) & (tau <= self.tau1))).any())
        else:
            tau = float(tau)
            outside = not self.tau0 <= tau <= self.tau1
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
        """Convert normalized time t back to raw time τ (E14).

        τ = τ₀ + t·(τ₁ − τ₀); with τ₀ = τ₁, τ is τ₀. It never warns, since the time centers of
        primitives may leave [0, 1] (Q-9).

        Parameters
        ----------
        t
            Normalized time: a number, or a tensor of any shape. An integer tensor is converted
            to float64 first.

        Returns
        -------
        float or torch.Tensor
            Raw time, in the scene's unit: a float for a number; for a tensor, a tensor of the
            same shape and floating dtype (float64 for an integer tensor).

        Notes
        -----
        The dtypes follow Q-9, the float64 of an integer tensor D-30.
        """
        if not isinstance(t, Tensor):
            t = float(t)
        elif not t.is_floating_point():
            t = t.to(torch.float64)  # Q-9
        return self.tau0 + t * (self.tau1 - self.tau0)


@dataclass
class RenderOutput:
    """The images of one call of :func:`openftgs.render.render`, with optional statistics.

    C is the number of cameras rendered at once, H × W their image size, N the number of
    primitives and nnz the number of rows in packed mode. The last three fields are set only when
    the render is asked for them (``return_aux``) and are None otherwise.

    Parameters
    ----------
    image
        The rendered colors, [C, H, W, 3].
    alpha
        The accumulated opacity, [C, H, W, 1].
    means2d
        The projected 2D means, [C, N, 2] unpacked or [nnz, 2] packed, kept for the gradient
        statistics; None unless asked for.
    visible
        Which primitives each camera sees, [C, N] bool, in both layouts, packed and unpacked
        (D-75); None unless asked for.
    gaussian_ids
        In packed mode, the primitive behind each row of ``means2d``, [nnz] int; None otherwise.
    """

    image: Tensor  # [C, H, W, 3]
    alpha: Tensor  # [C, H, W, 1]
    # [C, N, 2] unpacked or [nnz, 2] packed; kept for gradient statistics
    means2d: Tensor | None = None
    visible: Tensor | None = None  # [C, N] bool, in both layouts (packed and unpacked) (D-75)
    # [nnz] int, packed mode: the primitive behind each row of means2d
    gaussian_ids: Tensor | None = None


@dataclass(frozen=True)
class ExchangeModel:  # FTGS-IF contents (§2.2): physical values, raw time units
    """A model in exchange form: physical values in raw time units (§2.2 ``ftgs-if``).

    The contents of a ``ftgs-if`` file, activated values that depend on no implementation's
    internal parameterization; :func:`openftgs.io.validate_exchange` checks them. T_τ = τ₁ − τ₀
    is the training time span.

    Parameters
    ----------
    arrays
        The primitives' values by key, each with N ≥ 1 rows, all finite:

        - ``position``: μx, [N, 3] float32, world units;
        - ``time_center``: τ₀ + μt·T_τ, [N] float64, raw τ;
        - ``duration``: s·T_τ, [N] float64, units of τ, > 0;
        - ``velocity``: v/T_τ, [N, 3] float32, world units per unit of τ;
        - ``scale``: exp(â), [N, 3] float32, > 0;
        - ``rotation``: unit quaternions (w, x, y, z) with w ≥ 0, [N, 4] float32;
        - ``opacity``: sigmoid(ô), [N] float32, in (0, 1);
        - ``sh``: SH coefficients in gsplat's basis order, [N, (L + 1)², 3] float32.
    sh_degree
        The SH degree L, 0 to 3 (D-3).
    tau0
        τ₀, the smallest training timestamp, in raw units.
    tau1
        τ₁, the largest training timestamp, in raw units; ≥ ``tau0``.
    time_unit
        The unit of the raw times, such as ``"s"``.
    producer
        The software that made the model.
    background
        The RGB background the colors were trained against, three numbers in [0, 1]; black
        (D-34) by default and when a file records none (D-84).

    Notes
    -----
    With a single timestamp (τ₀ = τ₁), durations and velocities use a time unit of 1 instead of
    T_τ (D-70), every ``time_center`` is τ₀ and every ``velocity`` is 0 (D-71).
    """

    # position, time_center, duration, velocity, scale, rotation, opacity, sh
    arrays: dict[str, Tensor]
    sh_degree: int
    tau0: float
    tau1: float
    time_unit: str
    producer: str
    # D-84: the background the colors were trained against, recorded in render_conventions and
    # returned by read_exchange; black (D-34) when a file records none.
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TrainerState:  # plain tensors and dicts; a §2.2 checkpoint holds exactly this
    """The trainer's full state in plain tensors and dicts, as a checkpoint holds it (§2.2).

    :func:`openftgs.io.save_checkpoint` writes exactly this and
    :func:`openftgs.io.load_checkpoint` reads it back; a run resumed from it continues as if
    uninterrupted (§2.2 ``ftgs-checkpoint``).

    Parameters
    ----------
    iteration
        The number of iterations completed.
    config
        The run's TrainConfig as JSON-ready values, as
        :func:`openftgs.config.config_to_dict` writes them.
    time
        The time normalizer of the training timestamps (E14).
    spatial_scale
        The spatial scale S, in world units (D-27).
    frame_dt
        The frame interval Δt_f, in normalized time (E14).
    params
        The nine stored tensors of §1.1 by name, raw form, float32.
    optim
        The optimizer state, per tensor: ``exp_avg``, ``exp_avg_sq``, ``step`` and the
        schedule parameters.
    stats
        The growth and relocation accumulators ``g_sum`` and ``count``, [N] each.
    rng
        The generator states.
    sampler
        The sampler's epoch, cursor and permutation.
    scene_sha256
        The SHA-256 of the scene manifest, for resume, or None (Q-17).
    time_unit
        The scene manifest's time unit (Q-17).
    """

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
    scene_sha256: str | None = None  # Q-17: SHA-256 of the scene manifest, for resume
    time_unit: str = "s"  # Q-17: the scene manifest's time_unit


# Q-56: the contents of a ftgs-render-request file and of its ftgs-renders answer (§2.2).
@dataclass(frozen=True)
class RenderItem:
    """One item of a render request: an id, a raw time and a camera (§2.2; Q-56).

    Parameters
    ----------
    id
        The item's id, non-empty and unique within its request; its render is stored under it.
    timestamp
        The raw time τ to render at, float64, in the request's time unit.
    camera
        The camera to render from, named after the item id.
    """

    id: str
    timestamp: float  # raw τ in the request's time unit (float64)
    camera: Camera  # named after the item id


@dataclass(frozen=True)
class RenderRequest:
    """The contents of a ``ftgs-render-request`` file: what to render, where and when (§2.2).

    Parameters
    ----------
    items
        The items to render, in the file's order; at least one, with unique ids.
    time_unit
        The unit of the items' raw times.
    background
        The RGB background, three numbers in [0, 1]; black by default (D-34).
    sh_degree
        The SH degree to render with, 0 to 3, or None for the model's full degree.

    Notes
    -----
    The request and its renders are Q-56's reading of §2.2.
    """

    items: tuple[RenderItem, ...]
    time_unit: str
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)  # D-34
    sh_degree: int | None = None  # None: the model's full degree


@dataclass(frozen=True)
class Renders:
    """The renders that answer a render request, by item id (§2.2 ``ftgs-renders``; Q-56).

    Both maps hold the same ids: in request order from :func:`openftgs.render.render_request`,
    sorted from :func:`openftgs.io.read_renders`, since the file keeps no order.

    Parameters
    ----------
    rgb
        Each item's colors, [H, W, 3] float32, not clamped to 1.
    alpha
        Each item's accumulated opacity, [H, W, 1] float32.
    """

    # Keyed by item id: in request order from render_request, sorted from read_renders (Q-56).
    rgb: dict[str, Tensor]  # id → [H, W, 3] float32, not clamped to 1
    alpha: dict[str, Tensor]  # id → [H, W, 1] float32


# Q-64: the contents of a ftgs-scene file (§2.2), as io's reader returns them.
@dataclass(frozen=True)
class SceneManifest:
    """A parsed ``ftgs-scene`` file: the scene's images, cameras and metadata (§2.2; Q-64).

    :func:`openftgs.io.read_scene_manifest` returns it, and
    :func:`openftgs.io.write_scene_manifest` writes one.

    Parameters
    ----------
    path
        The manifest file.
    sha256
        The SHA-256 of the file's bytes as read, as 64 lowercase hex digits.
    time_unit
        The unit of the images' raw times.
    records
        The image records, in file order, with image paths resolved against the manifest's
        directory.
    masks
        The mask of each record, aligned with ``records``: a path for a test image that carries
        one, else None.
    metadata
        The manifest's ``metadata`` object.
    camera_names
        The keys of the ``cameras`` object, in file order: the manifest's camera order (D-22;
        Q-108).
    """

    path: Path  # the manifest file
    sha256: str  # SHA-256 of the file's bytes as read, as 64 lowercase hex digits
    time_unit: str
    records: tuple[ImageRecord, ...]  # in file order; image paths resolved against path.parent
    masks: tuple[Path | None, ...]  # aligned with records; only test images carry one
    metadata: dict[str, Any]
    # Q-108: the keys of the `cameras` object in file order, D-22's "manifest's camera order".
    camera_names: tuple[str, ...]


def frame_interval(camera_timestamps: Mapping[str, Sequence[float]], time: TimeNormalizer) -> float:
    """Compute the frame interval Δt_f of E14 in normalized time (D-46).

    From each training camera's raw timestamps: the median, over the cameras with at least two
    distinct timestamps, of each camera's median gap; without such a camera, the median gap of
    the pooled distinct timestamps; 1 for a single timestamp. The gap is divided by
    T_τ = τ₁ − τ₀ once. Measured within each camera, it doesn't shrink when unsynchronized
    cameras interleave their timestamps.

    Parameters
    ----------
    camera_timestamps
        Each training camera's raw timestamps τ, by camera name, in any order.
    time
        The time normalizer of the training timestamps.

    Returns
    -------
    float
        Δt_f in normalized time, finite and positive; 1 when τ₀ = τ₁.

    Raises
    ------
    ValueError
        If ``camera_timestamps`` is empty, or the result isn't finite and positive.

    Notes
    -----
    The medians are statistical medians (an even count averages the middle two) over the gaps
    between distinct raw timestamps, so duplicates neither shrink Δt_f nor block the fallback
    (Q-8). Δt_f is computed once per scene and passed on as a value.
    """
    # Q-8: statistical medians (even counts average) over the gaps between distinct raw
    # timestamps, divided by T_τ once; duplicates neither shrink Δt_f nor block the fallback.
    if not camera_timestamps:
        raise ValueError("frame_interval needs at least one camera")
    span = time.tau1 - time.tau0
    if span == 0.0:
        return 1.0
    distinct = [sorted({float(x) for x in stamps}) for stamps in camera_timestamps.values()]
    medians = [statistics.median(_gaps(stamps)) for stamps in distinct if len(stamps) >= 2]
    if medians:
        gap = statistics.median(medians)
    else:
        pooled = sorted({x for stamps in distinct for x in stamps})
        if len(pooled) < 2:
            return 1.0
        gap = statistics.median(_gaps(pooled))
    result = gap / span
    if not (math.isfinite(result) and result > 0.0):
        raise ValueError(f"frame interval must be finite and positive, got {result!r}")
    return result


def _gaps(ordered: list[float]) -> list[float]:
    return [b - a for a, b in itertools.pairwise(ordered)]
