# SPDX-License-Identifier: Apache-2.0
"""render() with two backends, gsplat 1.5.3 (CUDA, float32, production) and reference (pure
PyTorch, the readable oracle for tests); render_request(), which answers a render request with
cameras batched per time; and, from M15, the temporal index used for rendering (D-55).

Spec: E2, E5, E6, D-3, D-4, D-6, D-12, D-28, D-31, D-32, D-34, D-55, D-57. Band: Core model
(§2.1).
"""

import itertools
import math
import numbers
import statistics
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal, get_args

import torch
from torch import Tensor

from openftgs import _shards
from openftgs._stages import point, stage
from openftgs.model import FreeTimeGaussians, GaussiansAtTime, covariance
from openftgs.motion import temporal_slice
from openftgs.sh import sh_basis
from openftgs.types import Camera, RenderItem, RenderOutput, RenderRequest, Renders

__all__ = ["render", "render_request"]

# E6's normative constants (gsplat 1.5.3 in classic mode) and E2's color offset.
_PIXEL_CENTER = 0.5  # E6: pixel (u, v) is sampled at (u + ½, v + ½); §1.4; D-28
_CLAMP_MARGIN = 0.15  # E6: J's x_c/z_c is clamped 0.15·W/f_x beyond the image edges (y likewise)
_EPS2D = 0.3  # E6: ε added to the diagonal of Σ′; D-6
_ALPHA_MAX = 0.999  # E6: α_max; D-31
_ALPHA_SKIP = 1.0 / 255.0  # E6: a term with α < 1/255 is skipped
_T_STOP = 1e-4  # E6: a pixel stops before the primitive that brings T to ≤ 1e-4
_NEAR = 0.01  # E6: z_c < 0.01 is culled; D-6
_FAR = 1e10  # E6: z_c > 1e10 is culled; D-6
_EXTENT_MAX = 3.33  # E6: the screen-space extent never exceeds 3.33 standard deviations
_SH_OFFSET = 0.5  # E2: the SH sum + 0.5, clamped at 0 per channel; D-4
_BACKGROUND = (0.0, 0.0, 0.0)  # E6: c_bg is black; D-34
_CHUNK_ELEMENTS = 1 << 22  # pixel–primitive pairs composited at once (bounds memory only)

# gsplat's settings (D-6, D-57); E6's constants above are passed to it as well.
_RASTERIZE_MODE: Literal["classic", "antialiased"] = (
    "classic"  # D-6; ftgs-if's rasterize_mode (Q-35)
)
_TILE_SIZE = 16  # D-57
_RADIUS_CLIP = 0.0  # D-57: no primitive is skipped for its size

# D-55's temporal index (Q-174).
# Q-174, Q-195: T, the distinct times from which temporal_index None builds the index; B-7's
# recorded run on kitchen (Calibration data, "Q-174's T": the 1 % models break even at 6).
_INDEX_MIN_TIMES = 6
_INDEX_OPACITY_MARGIN = 1e-4  # D-55's margin: rows are listed where 255σ(1 + 10⁻⁴)·σt can reach 1
_INDEX_TIME_MARGIN = 1e-6  # D-55's margin: added to each row's reach, in normalized time
_INDEX_LONG = 8  # Q-174: rows whose reach covers more buckets go to the long list
_INDEX_MIN_WIDTH = 2.0**-30  # keeps bucket numbers small integers; images don't depend on the width
# Q-174's fallback, should U-32 on gsplat differ at the cull bound: True makes the gsplat backend
# slice every row and keep the listed ones after, as the reference backend does.
_GSPLAT_FULL_SLICE = False
_SLICE_INPUTS = (  # the stored tensors the temporal slice reads, in its argument order (Q-12)
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
)

_Backend = Literal["gsplat", "reference"]
_TimeUnits = Literal["raw", "normalized"]
_Precull = Literal["internal", "compact"]  # Q-31: PerfConfig.precull's values (D-32, D-57)
_SliceImpl = Literal["auto", "eager", "compiled", "cuda"]  # Q-66: the model's slice_impl (D-50)


def render(
    model: FreeTimeGaussians,
    cameras: Camera | Sequence[Camera],
    timestamp: float,
    *,
    time_units: _TimeUnits = "raw",
    sh_degree: int | None = None,
    background: Tensor | None = None,
    backend: _Backend = "gsplat",
    return_aux: bool = False,
    precull: _Precull = "internal",  # Q-31
    packed: bool = True,  # Q-36
) -> RenderOutput:
    """Render `model` at one time from one or more cameras of the same size (E2, E5, E6).

    `timestamp` is a raw τ (`time_units="raw"`) or a normalized t. `sh_degree` limits the SH
    bands (default: the model's degree). `background` is [3] or [C, 3], black by default (D-34).
    `precull="compact"` applies the opacity cull of D-32 by compaction before the backend instead
    of inside its projection (Q-31); images are unchanged. `packed` selects gsplat's packed mode
    (D-57, Q-36); the reference backend is always unpacked. With `return_aux`, the output also
    holds the 2D means, whose gradient is retained, the visibility mask and, in packed mode, the
    primitive behind each row of the 2D means (Q-32, D-75, D-76).
    """
    _check_choice("backend", backend, get_args(_Backend))
    _check_choice("time_units", time_units, get_args(_TimeUnits))
    _check_choice("precull", precull, get_args(_Precull))
    if not isinstance(packed, bool):  # Q-36
        raise ValueError(f"packed must be a bool, got {packed!r}")
    cams = _check_cameras(cameras)
    if backend == "gsplat":
        _check_gsplat_model(model)
    t = _normalized_time(model, timestamp, time_units)
    g = model.at_time(t, sh_degree=sh_degree)
    # §3.4: with a stage clock active, the moved means' gradient marks where the rasterization's
    # backward ends and the temporal slice's begins, compaction's gather included.
    point(g.means_t, "backward.raster_done")
    with stage("raster.forward"):
        backgrounds = _backgrounds(background, len(cams), g.means_t)
        rows: Tensor | None = None
        if precull == "compact":
            # Q-31, D-57: compaction after the temporal slice, in ascending row order.
            rows = _kept_rows(g)  # Q-171: the one synchronization compaction adds
            g = _take_rows(g, rows)
        gaussian_ids: Tensor | None = None
        if backend == "gsplat":
            image, alpha, means2d, visible, gaussian_ids = _render_gsplat(
                g, cams, backgrounds, rows=rows, num_rows=model.num_gaussians, packed=packed
            )
        else:
            image, alpha, means2d, visible = _render_reference(
                g, cams, backgrounds, rows=rows, num_rows=model.num_gaussians
            )
    if not return_aux:
        return RenderOutput(image=image, alpha=alpha)
    if means2d.requires_grad:
        means2d.retain_grad()  # Q-32: the gradient statistics read ∂L/∂μ′ (E10)
    return RenderOutput(
        image=image, alpha=alpha, means2d=means2d, visible=visible, gaussian_ids=gaussian_ids
    )


def render_request(
    model: FreeTimeGaussians,
    request: RenderRequest,
    *,
    backend: _Backend = "gsplat",
    temporal_index: bool | None = None,  # Q-174
    frame_dt: float | None = None,  # Q-174
    device: str | torch.device = "cpu",  # Q-174
    devices: Sequence[str] | None = None,  # Q-175
) -> Renders:
    """Answer a render request (§2.2, Q-57): one image and alpha per item, at the item's raw
    timestamp, with the request's background and SH degree, keyed by item id in request order.

    Items that share a timestamp and an image size render in one call (D-55). Item ids must be
    unique; outputs are keyed by them, whatever the cameras' names. Runs in inference mode;
    outputs are inference tensors on `device`, the CPU by default.

    The temporal index of D-55 (Q-174) renders each timestamp inside the training span, a
    normalized time in [0, 1], from the rows that can pass the opacity cull then, with images
    unchanged (U-32); other timestamps render as without it. `temporal_index` None turns it on for
    the gsplat backend when the request has at least `_INDEX_MIN_TIMES` distinct times in
    [0, 1]; True or False forces it. `frame_dt` is the buckets' width in normalized time, Δt_f
    (E14) for a model from a checkpoint; None takes the median gap between the request's
    distinct times in [0, 1].

    `devices` spreads the request's calls over shards, one per entry, each with its own copy of
    the model on its device and its own thread, without changing the images (D-54; Q-175,
    Q-186). None renders on the model's device, in the caller's thread.
    """
    _check_choice("backend", backend, get_args(_Backend))
    if temporal_index is not None and not isinstance(temporal_index, bool):  # Q-174
        raise ValueError(f"temporal_index must be None or a bool, got {temporal_index!r}")
    if frame_dt is not None:  # Q-174
        _check_frame_dt(frame_dt)
    out_device = _output_device(device)
    shard_devices = None if devices is None else _shards.devices(devices)  # Q-186
    # Q-57, Q-17: the timestamps are raw τ in the request's unit, which must be the model's.
    if request.time_unit != model.time_unit:
        raise ValueError(
            f"the request's time_unit {request.time_unit!r} differs from the model's "
            f"{model.time_unit!r}"
        )
    ids = [item.id for item in request.items]
    if len(set(ids)) != len(ids):  # Q-56: unique ids, also in a request built in Python
        raise ValueError("render request item ids must be unique")
    groups: dict[tuple[float, int, int], list[RenderItem]] = {}
    for item in request.items:
        key = (item.timestamp, item.camera.width, item.camera.height)
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        _check_cameras([item.camera for item in group])
    times = {key: _time_in_span(model, key[0]) for key in groups}
    inside = sorted({t for t in times.values() if t is not None})
    # Q-174: decided once for the whole request, so every shard renders alike (Q-186).
    indexed = bool(inside) and _index_on(temporal_index, backend, len(inside))
    plan = _Plan(
        request=request,
        backend=backend,
        index_times=inside if indexed else [],
        width=_bucket_width(frame_dt, inside) if indexed else 1.0,
        out_device=out_device,
    )
    jobs = [_Group(key, group, times[key]) for key, group in groups.items()]
    if shard_devices is None:
        renderer = _GroupRenderer(model, plan)
        outputs = [renderer.render(job) for job in jobs]
    else:
        # D-54, Q-175: each shard is built and warmed up here, one after another, before any
        # worker thread starts (Q-186).
        shards = [_shard_renderer(model, plan, jobs, d) for d in shard_devices]
        outputs = _shards.run(shards, jobs, _GroupRenderer.render, name="render_request shard")
    rgb: dict[str, Tensor] = {}
    alpha: dict[str, Tensor] = {}
    for job, output in zip(jobs, outputs, strict=True):
        for item, (image, alphas) in zip(job.items, output, strict=True):
            rgb[item.id], alpha[item.id] = image, alphas
    return Renders(rgb={i: rgb[i] for i in ids}, alpha={i: alpha[i] for i in ids})


# The front half, shared by every backend.


def _check_choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _check_cameras(cameras: Camera | Sequence[Camera]) -> list[Camera]:
    # Q-32: one Camera or a non-empty sequence of them, all of one positive width and height;
    # the tensors' shapes are checked, their values are not.
    cams = [cameras] if isinstance(cameras, Camera) else list(cameras)
    if not cams:
        raise ValueError("render needs at least one camera")
    for cam in cams:
        if not isinstance(cam, Camera):
            raise ValueError(f"cameras must be Camera objects, got {type(cam).__name__}")
        if tuple(cam.K.shape) != (3, 3) or tuple(cam.world_to_camera.shape) != (4, 4):
            raise ValueError(
                f"camera {cam.name!r}: K must be [3, 3] and world_to_camera [4, 4], got "
                f"{list(cam.K.shape)} and {list(cam.world_to_camera.shape)}"
            )
        if cam.width < 1 or cam.height < 1:
            raise ValueError(f"camera {cam.name!r}: width and height must be positive")
    if len({(cam.width, cam.height) for cam in cams}) > 1:
        raise ValueError("all cameras of one render must have the same width and height")
    return cams


def _check_gsplat_model(model: FreeTimeGaussians) -> None:
    # D-76: gsplat's kernels compute in float32 on a CUDA device; the float64 path of Q-32 is the
    # reference backend's.
    param = model.means
    if param.device.type != "cuda" or param.dtype != torch.float32:
        raise ValueError(
            "the gsplat backend renders float32 models on a CUDA device, got "
            f"{param.dtype} on {param.device}; use backend='reference' for this model"
        )


def _normalized_time(model: FreeTimeGaussians, timestamp: float, time_units: _TimeUnits) -> float:
    # Q-32: a finite real number (not a bool); raw time warns through to_t (Q-9), normalized time
    # here, outside [0, 1] or, with a single timestamp, anywhere but 0.
    if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real):
        raise ValueError(f"timestamp must be a real number, got {timestamp!r}")
    value = float(timestamp)
    if not math.isfinite(value):
        raise ValueError(f"timestamp must be finite, got {value!r}")
    if time_units == "raw":
        return model.time.to_t(value)
    single = model.time.tau1 == model.time.tau0
    inside = value == 0.0 if single else 0.0 <= value <= 1.0
    if not inside:
        warnings.warn(
            f"normalized time {value!r} outside the training range "
            f"{'{0}' if single else '[0, 1]'}: motion extrapolates linearly",
            UserWarning,
            stacklevel=3,
        )
    return value


def _backgrounds(background: Tensor | None, num_cameras: int, like: Tensor) -> Tensor:
    """One background color per camera, [C, 3], in the parameters' dtype and device (D-34)."""
    # Q-32: None (black), [3] for every camera, or [C, 3].
    if background is None:
        # Filled on the device, not copied from the host, so a render doesn't synchronize (D-49).
        color = torch.stack([like.new_full((), value) for value in _BACKGROUND])
    else:
        color = torch.as_tensor(background).to(dtype=like.dtype, device=like.device)
    if tuple(color.shape) == (3,):
        return color.expand(num_cameras, 3)
    if tuple(color.shape) == (num_cameras, 3):
        return color
    raise ValueError(f"background must be [3] or [{num_cameras}, 3], got {list(color.shape)}")


def _kept_rows(g: GaussiansAtTime) -> Tensor:
    """The rows that pass the opacity cull, ascending (Q-31). Selecting them reads their count
    back, the one host synchronization compaction adds (§2.4 rule 3, D-57), which U-29 allows
    here only (Q-171)."""
    return torch.nonzero(g.passes_cull).squeeze(1)


def _take_rows(g: GaussiansAtTime, rows: Tensor) -> GaussiansAtTime:
    """The per-time arrays of the given rows (compaction, Q-31)."""
    return GaussiansAtTime(**{f.name: getattr(g, f.name)[rows] for f in fields(g)})


def _scatter_rows(values: Tensor, rows: Tensor | None, num_rows: int) -> Tensor:
    """Per-row values of a row subset placed back among all rows, zero (or False) elsewhere."""
    if rows is None:
        return values
    full = values.new_zeros((num_rows, *values.shape[1:]))
    return full.index_copy(0, rows, values)


@dataclass(frozen=True)
class _CameraTensors:
    """One camera's parameters as tensors in the model parameters' dtype and device."""

    rotation: Tensor  # [3, 3] R_w
    translation: Tensor  # [3] t_w
    fx: Tensor
    fy: Tensor
    cx: Tensor
    cy: Tensor
    center: Tensor  # [3] o = −R_wᵀ t_w (E2, §1.4)
    width: int
    height: int


def _camera_tensors(camera: Camera, like: Tensor) -> _CameraTensors:
    # Q-32: cast to the parameters' dtype and device; only f_x, f_y, c_x, c_y, R_w and t_w are read.
    K = camera.K.to(dtype=like.dtype, device=like.device)
    w2c = camera.world_to_camera.to(dtype=like.dtype, device=like.device)
    rotation, translation = w2c[:3, :3], w2c[:3, 3]
    return _CameraTensors(
        rotation=rotation,
        translation=translation,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        center=-(rotation * translation[:, None]).sum(dim=0),  # −R_wᵀ t_w
        width=camera.width,
        height=camera.height,
    )


# Rendering a request: its once-per-request parts and the temporal index (D-55; Q-57, Q-174).


def _check_frame_dt(frame_dt: object) -> None:
    # Q-174: a finite positive real number (not a bool), in normalized time.
    if (
        isinstance(frame_dt, bool)
        or not isinstance(frame_dt, numbers.Real)
        or not (math.isfinite(float(frame_dt)) and float(frame_dt) > 0.0)
    ):
        raise ValueError(f"frame_dt must be a finite positive number, got {frame_dt!r}")


def _time_in_span(model: FreeTimeGaussians, timestamp: object) -> float | None:
    """Q-174: the normalized time of a raw timestamp inside the training span [τ₀, τ₁], which
    lies in [0, 1] and draws no warning; None for any other timestamp, which goes to `render()`
    to be checked and warned about as before (Q-9, Q-32)."""
    if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real):
        return None
    value = float(timestamp)
    if not model.time.tau0 <= value <= model.time.tau1:  # NaN included
        return None
    return model.time.to_t(value)


def _output_device(device: str | torch.device) -> torch.device:
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"device must name a torch device, got {device!r}") from error


def _request_background(values: Sequence[float], like: Tensor) -> Tensor:
    """The request's background [3], in float32 as render requests have always passed it (Q-57),
    on the model's device: filled there rather than copied from the host, so building it doesn't
    synchronize (Q-174)."""
    return torch.stack([like.new_full((), float(v), dtype=torch.float32) for v in values])


def _staged_cameras(items: Sequence[RenderItem], device: torch.device) -> dict[str, Camera]:
    """Each item's camera by item id, with its tensors on the model's device (Q-174). Cameras
    held on the host go to a CUDA device in one copy for the whole request, staged in pinned
    memory and sent without blocking, so that no render synchronizes to copy them (§2.4 rule 3);
    their values are unchanged."""
    cameras = {item.id: item.camera for item in items}
    if device.type != "cuda":
        return cameras
    host = [
        item
        for item in items
        if item.camera.K.device.type == "cpu" and item.camera.world_to_camera.device.type == "cpu"
    ]
    if not host:
        return cameras
    flat = torch.cat(
        [
            torch.cat([item.camera.K.reshape(-1), item.camera.world_to_camera.reshape(-1)])
            for item in host
        ]
    )
    sent = flat.reshape(len(host), 25).pin_memory().to(device, non_blocking=True)
    for row, item in zip(sent, host, strict=True):
        cam = item.camera
        cameras[item.id] = Camera(
            row[:9].reshape(3, 3), row[9:].reshape(4, 4), cam.width, cam.height, cam.name
        )
    return cameras


def _index_on(temporal_index: bool | None, backend: str, distinct_times: int) -> bool:
    """Q-174: whether a request with this many distinct times in [0, 1] renders through the
    temporal index. None turns it on for gsplat from `_INDEX_MIN_TIMES` times, so never for one
    time; that threshold is B-7's (Q-195)."""
    if temporal_index is not None:
        return temporal_index
    return backend == "gsplat" and distinct_times >= _INDEX_MIN_TIMES


def _bucket_width(frame_dt: float | None, times: Sequence[float]) -> float:
    """Q-174: the buckets' width in normalized time: `frame_dt` when given, otherwise the median
    gap between the request's distinct times in [0, 1] (`times`, ascending), as Q-8 takes
    medians, and 1 for a single time, as E14 gives Δt_f = 1 for a single timestamp (Q-8). Never
    below `_INDEX_MIN_WIDTH`; the images are the same at any width."""
    if frame_dt is not None:
        width = float(frame_dt)
    elif len(times) >= 2:
        width = statistics.median([b - a for a, b in itertools.pairwise(times)])
    else:
        width = 1.0
    return max(width, _INDEX_MIN_WIDTH)


def _sh_concat(model: FreeTimeGaussians, sh_degree: int | None) -> Tensor:
    """The SH coefficients of every row cut to the request's degree, as `at_time` concatenates
    them (Q-20), built once per request (Q-174); `at_time`'s check of the degree."""
    degree = model.sh_degree if sh_degree is None else sh_degree
    allowed = tuple(range(model.sh_degree + 1))
    if isinstance(degree, bool) or not isinstance(degree, int) or degree not in allowed:
        raise ValueError(f"sh_degree must be one of {allowed}, got {degree!r}")
    return torch.cat([model.sh0, model.shN[:, : (degree + 1) ** 2 - 1]], dim=1)


class _TemporalIndex:
    """D-55's temporal index of one model for the times of one request (Q-174).

    Normalized time is cut into buckets `width` wide; time t falls in bucket ⌊t·(1/width)⌋. A row
    is listed in every bucket that its reach touches inside [0, 1], the reach being the times
    at which its effective opacity can reach 1/255, widened by D-55's margins and computed in
    float64: |t − μt| ≤ s·√(2 ln(255σ(1 + 10⁻⁴))) + 10⁻⁶. A row with 255σ(1 + 10⁻⁴) ≤ 1 can never
    reach it and is never listed. A row whose reach covers more than `_INDEX_LONG` buckets is
    kept once, on a long list that every render takes; the others are stored per bucket, in
    ascending row order. Only the buckets of the request's times are kept.

    Building the index reads counts back from the device, so it synchronizes; a lookup doesn't.
    """

    def __init__(self, model: FreeTimeGaussians, width: float, times: Sequence[float]) -> None:
        self._scale = 1.0 / width
        f64 = torch.float64
        device = model.means.device
        mu = model.times.detach().to(f64)
        peak = 255.0 * torch.sigmoid(model.opacity_logits.detach().to(f64))
        peak = peak * (1.0 + _INDEX_OPACITY_MARGIN)
        reachable = peak > 1.0
        spread = torch.sqrt(2.0 * torch.log(torch.where(reachable, peak, 1.0)))
        reach = torch.exp(model.log_durations.detach().to(f64)) * spread + _INDEX_TIME_MARGIN
        low = torch.clamp(mu - reach, min=0.0)
        high = torch.clamp(mu + reach, max=1.0)
        # The same rounding as `_bucket`: a time in [low, high] falls in a bucket in [first, last].
        first = torch.floor(low * self._scale)
        last = torch.floor(high * self._scale)
        listed = reachable & (low <= high)
        long = listed & (last - first >= _INDEX_LONG)
        self._long = torch.nonzero(long).squeeze(1)
        short = torch.nonzero(listed & ~long).squeeze(1)
        steps = torch.arange(_INDEX_LONG, device=device)
        buckets = first[short].to(torch.int64)[:, None] + steps
        covered = steps <= (last[short] - first[short]).to(torch.int64)[:, None]
        # Each row's buckets in row order, so a stable sort by bucket keeps every bucket's rows
        # ascending (ties in depth then composite in row order, as without the index).
        by_bucket = torch.sort(buckets[covered], stable=True)
        self._rows = short[:, None].expand_as(buckets)[covered][by_bucket.indices]
        keys = sorted({self._bucket(t) for t in times})
        wanted = torch.tensor(keys, dtype=torch.int64, device=device)
        starts = torch.searchsorted(by_bucket.values, wanted)
        stops = torch.searchsorted(by_bucket.values, wanted, right=True)
        bounds = torch.stack([starts, stops], dim=1).tolist()
        self._ranges = {key: (a, b) for key, (a, b) in zip(keys, bounds, strict=True)}

    def _bucket(self, t: float) -> int:
        return math.floor(t * self._scale)

    def rows(self, t: float) -> Tensor:
        """The ascending model rows to render at normalized time t, one of the times the index
        was built for: its bucket's rows and the long rows, a superset of the rows that pass the
        opacity cull at t. Their count is known on the host."""
        start, stop = self._ranges[self._bucket(t)]
        listed = self._rows[start:stop]
        if self._long.shape[0] == 0:
            return listed
        if start == stop:
            return self._long
        return torch.sort(torch.cat([listed, self._long])).values


def _slice(
    model: FreeTimeGaussians, t: float, sh: Tensor, rows: Tensor | None = None
) -> GaussiansAtTime:
    """The per-time arrays of `at_time`, from the request's SH concat `sh`, for every row or
    only for `rows`, which are gathered first (Q-174)."""
    with stage("slice"):
        arrays = [getattr(model, name) for name in _SLICE_INPUTS]
        if rows is not None:
            arrays = [array[rows] for array in arrays]
            sh = sh[rows]
        means, times, log_durations, velocities, log_scales, quats, opacity_logits = arrays
        with stage("slice.kernel"):
            s = temporal_slice(
                means,
                times,
                log_durations,
                velocities,
                log_scales,
                quats,
                opacity_logits,
                t,
                impl=model.slice_impl,
            )
    return GaussiansAtTime(
        means_t=s.means_t,
        opacities_t=s.opacities_t,
        quats_n=s.quats_n,
        scales=s.scales,
        sh=sh,
        opacities=s.opacities,
        reg_terms=s.reg_terms,
        passes_cull=s.passes_cull,
    )


def _render_indexed(
    model: FreeTimeGaussians,
    cameras: list[Camera],
    t: float,
    rows: Tensor,
    *,
    sh: Tensor,
    background: Tensor,
    backend: _Backend,
) -> tuple[Tensor, Tensor]:
    """Image [C, H, W, 3] and alpha [C, H, W, 1] of `cameras` at normalized time t, rendered from
    the ascending model rows `rows` that the temporal index lists then, with `render()`'s
    defaults (D-55, Q-174). The backend's own opacity cull then decides, so the images are those
    of every row (U-32)."""
    backgrounds = _backgrounds(background, len(cameras), model.means)
    if rows.shape[0] == 0:
        # Q-174: no row can pass the cull, so the slice isn't run (dynamo would specialize a
        # graph for zero rows, Q-13), and the render is Q-38's empty render (D-76).
        return _empty_images(backgrounds, cameras[0].width, cameras[0].height)
    # CPU kernels can give other bits on a subset of rows (U-8), so the reference backend slices
    # every row and keeps the listed ones; gsplat slices the listed rows only.
    full = backend == "reference" or _GSPLAT_FULL_SLICE
    g = _slice(model, t, sh, None if full else rows)
    with stage("raster.forward"):
        if full:
            g = _take_rows(g, rows)
        if backend == "gsplat":
            image, alpha, *_ = _render_gsplat(
                g, cameras, backgrounds, rows=rows, num_rows=model.num_gaussians, packed=True
            )
        else:
            image, alpha, *_ = _render_reference(
                g, cameras, backgrounds, rows=rows, num_rows=model.num_gaussians
            )
    return image, alpha


def _empty_images(background: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """Q-38's empty render's image and alpha (D-76): the background [C, 3] composited as both
    backends composite it, image + (1 − alpha)·c_bg with nothing drawn, so its bits are theirs."""
    count = background.shape[0]
    alpha = background.new_zeros((count, height, width, 1))
    image = background.new_zeros((count, height, width, 3))
    return image + (1.0 - alpha) * background[:, None, None, :], alpha


@dataclass(frozen=True)
class _Plan:
    """What `render_request` decides once for the whole request (Q-174, Q-186)."""

    request: RenderRequest
    backend: _Backend
    index_times: list[float]  # the times in [0, 1] the index is built for; empty: no index
    width: float  # the buckets' width
    out_device: torch.device


@dataclass(frozen=True)
class _Group:
    """One render call of a request: items that share a timestamp and an image size (D-55)."""

    key: tuple[float, int, int]  # the raw timestamp, width and height
    items: list[RenderItem]
    t: float | None  # the normalized time inside the training span, else None (Q-174)


class _GroupRenderer:
    """One model's renders of one request's groups. What doesn't change with time is built once
    (Q-174): the background and the cameras on the model's device and, with the index on, the
    SH concat and the temporal index."""

    def __init__(self, model: FreeTimeGaussians, plan: _Plan) -> None:
        self._model = model
        self._plan = plan
        self._index: _TemporalIndex | None = None
        self._sh: Tensor | None = None
        with torch.inference_mode():
            like = model.means
            self._background = _request_background(plan.request.background, like)
            self._cameras = _staged_cameras(plan.request.items, like.device)
            if plan.index_times:
                if plan.backend == "gsplat":
                    _check_gsplat_model(model)
                with stage("slice.sh_concat"):  # Q-174: once per request
                    self._sh = _sh_concat(model, plan.request.sh_degree)
                self._index = _TemporalIndex(model, plan.width, plan.index_times)

    def render(self, group: _Group) -> list[tuple[Tensor, Tensor]]:
        """The image [H, W, 3] and alpha [H, W, 1] of each of the group's items, in its order, on
        the request's output device. Inference mode is entered here, in the thread that renders,
        since it is per thread."""
        plan, model = self._plan, self._model
        with torch.inference_mode():
            cams = [self._cameras[item.id] for item in group.items]
            if self._index is not None and self._sh is not None and group.t is not None:
                image, alphas = _render_indexed(
                    model,
                    cams,
                    group.t,
                    self._index.rows(group.t),
                    sh=self._sh,
                    background=self._background,
                    backend=plan.backend,
                )
            else:
                # D-55: cameras that share a time (and, for render(), a size) render in one call.
                out = render(
                    model,
                    cams,
                    group.key[0],
                    sh_degree=plan.request.sh_degree,
                    background=self._background,
                    backend=plan.backend,
                )
                image, alphas = out.image, out.alpha
            return [
                (image[k].to(plan.out_device), alphas[k].to(plan.out_device))
                for k in range(len(group.items))
            ]

    def warm_up(self, groups: Sequence[_Group]) -> None:
        """Q-175, Q-186: runs the temporal slice once, under inference mode as the renders
        do, on each path that rendering `groups` takes: every row, as `render()` and the
        reference backend's index slice them, and the gathered rows that gsplat slices with the
        index on. So the compiled slice is compiled for this device here, in the caller's
        thread, and not by several worker threads at once."""
        model, plan = self._model, self._plan
        indexed = self._index is not None
        gathered = indexed and plan.backend == "gsplat" and not _GSPLAT_FULL_SLICE
        every_row = not gathered or any(group.t is None for group in groups)
        with torch.inference_mode():
            arrays = [getattr(model, name) for name in _SLICE_INPUTS]
            if every_row:
                _slice_once(arrays, model.slice_impl)
            if gathered and model.num_gaussians > 0:
                rows = torch.arange(model.num_gaussians, device=model.means.device)
                _slice_once([array[rows] for array in arrays], model.slice_impl)


def _slice_once(arrays: Sequence[Tensor], impl: _SliceImpl) -> None:
    """The temporal slice of `_SLICE_INPUTS`' arrays at t = 0, for its side effect: compiling."""
    means, times, log_durations, velocities, log_scales, quats, opacity_logits = arrays
    temporal_slice(
        means, times, log_durations, velocities, log_scales, quats, opacity_logits, 0.0, impl=impl
    )


def _model_copy(model: FreeTimeGaussians, device: torch.device) -> FreeTimeGaussians:
    """Q-175, Q-186: the model on `device`, made with the public constructor from its stored
    tensors, which it copies, keeping its `slice_impl` (Q-66)."""
    copy = FreeTimeGaussians(
        {name: getattr(model, name).detach().to(device) for name in model.PARAM_NAMES},
        sh_degree=model.sh_degree,
        time=model.time,
        time_unit=model.time_unit,
    )
    copy.slice_impl = model.slice_impl
    return copy


def _shard_renderer(
    model: FreeTimeGaussians, plan: _Plan, groups: Sequence[_Group], device: torch.device
) -> _GroupRenderer:
    """One shard of a request (D-54; Q-175, Q-186): a copy of the model on `device`, what the
    request builds once for it, and the warm-up of its temporal slice."""
    copy = _model_copy(model, device)
    if plan.backend == "gsplat":
        _check_gsplat_model(copy)
    renderer = _GroupRenderer(copy, plan)
    renderer.warm_up(groups)
    return renderer


# The reference backend.


@dataclass(frozen=True)
class _Projection:
    """Per-row results of projecting the passed rows into one camera (E2, E6)."""

    means2d: Tensor  # [M, 2] μ′ where visible, 0 elsewhere
    conics: Tensor  # [M, 3] Σ′⁻¹ as (a, b, c): q = ½(a dx² + c dy²) + b dx dy
    depths: Tensor  # [M] z_c, detached, for sorting
    colors: Tensor  # [M, 3] E2
    visible: Tensor  # [M] bool: both screen radii positive


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """a @ b for small matrices, as a broadcast product and a sum, so that every row gives the
    same bits whatever the number of rows (source log #17); batched matmul with a camera matrix
    through BLAS does not."""
    return (a[..., :, :, None] * b[..., None, :, :]).sum(dim=-2)


def _render_reference(
    g: GaussiansAtTime,
    cameras: list[Camera],
    background: Tensor,
    *,
    rows: Tensor | None,
    num_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Image [C, H, W, 3], alpha [C, H, W, 1], 2D means [C, N, 2] and visibility [C, N] of the
    per-time arrays `g`, which hold all N rows or the ascending subset `rows`."""
    cams = [_camera_tensors(camera, g.means_t) for camera in cameras]
    projections = [_project(g, cam) for cam in cams]
    # Every camera is projected before any is composited: compositing reads μ′ from this stacked
    # tensor, so its retained gradient is ∂L/∂μ′ (Q-32).
    means2d = torch.stack([_scatter_rows(p.means2d, rows, num_rows) for p in projections])
    visible = torch.stack([_scatter_rows(p.visible, rows, num_rows) for p in projections])
    images, alphas = [], []
    for c, (cam, p) in enumerate(zip(cams, projections, strict=True)):
        local = torch.nonzero(p.visible).squeeze(1)
        # E6: front to back by z_c; Q-33: a stable sort, so ties go to the lower row.
        local = local[torch.sort(p.depths[local], stable=True).indices]
        ids = local if rows is None else rows[local]
        image, alpha = _rasterize(
            means2d[c, ids],
            p.conics[local],
            g.opacities_t[local],
            p.colors[local],
            background[c],
            cam.width,
            cam.height,
        )
        images.append(image)
        alphas.append(alpha)
    return torch.stack(images), torch.stack(alphas), means2d, visible


def _project(g: GaussiansAtTime, cam: _CameraTensors) -> _Projection:
    """E6's projection, EWA covariance and culls, and E2's colors, for every passed row.

    Every input that could make a local derivative infinite is replaced before use (the depth of
    a row the depth cull removes, the det Σ′ of a row the det cull removes), and the integer
    screen radii are computed without gradient, so a culled row's gradient is exactly zero."""
    x_c = _matmul(cam.rotation, g.means_t[..., None])[..., 0] + cam.translation
    x, y, z = x_c.unbind(-1)
    in_depth = (z >= _NEAR) & (z <= _FAR)
    z = torch.where(in_depth, z, torch.ones_like(z))
    u, v = x / z, y / z
    mean2d = torch.stack([cam.fx * u + cam.cx, cam.fy * v + cam.cy], dim=-1)

    # E6: the EWA Jacobian at x_c, with x_c/z_c and y_c/z_c clamped for J only.
    margin_x = _CLAMP_MARGIN * cam.width / cam.fx
    margin_y = _CLAMP_MARGIN * cam.height / cam.fy
    tx = torch.clamp(u, -(cam.cx / cam.fx + margin_x), (cam.width - cam.cx) / cam.fx + margin_x)
    ty = torch.clamp(v, -(cam.cy / cam.fy + margin_y), (cam.height - cam.cy) / cam.fy + margin_y)
    zero = torch.zeros_like(z)
    jacobian = torch.stack(
        [
            torch.stack([cam.fx / z, zero, -cam.fx * tx / z], dim=-1),
            torch.stack([zero, cam.fy / z, -cam.fy * ty / z], dim=-1),
        ],
        dim=-2,
    )
    sigma_c = _matmul(_matmul(cam.rotation, covariance(g.quats_n, g.scales)), cam.rotation.T)
    cov2d = _matmul(_matmul(jacobian, sigma_c), jacobian.transpose(-1, -2))
    s00 = cov2d[..., 0, 0] + _EPS2D
    s01 = cov2d[..., 0, 1]
    s11 = cov2d[..., 1, 1] + _EPS2D
    det = s00 * s11 - s01 * s01
    det_ok = det > 0.0  # Q-33: a NaN det culls too
    det = torch.where(det_ok, det, torch.ones_like(det))
    conics = torch.stack([s11 / det, -s01 / det, s00 / det], dim=-1)

    with torch.no_grad():
        # E6, D-32: the opacity cull, as the temporal slice evaluated it (passes_cull, Q-12), and
        # the opacity-aware extent √(2 ln(255 σ_eff)), never above 3.33.
        passes = g.passes_cull
        extent = torch.sqrt(2.0 * torch.log(torch.where(passes, 255.0 * g.opacities_t, 1.0)))
        extent = torch.clamp(extent, max=_EXTENT_MAX)
        radius_x = torch.ceil(extent * torch.sqrt(s00))
        radius_y = torch.ceil(extent * torch.sqrt(s11))
        mx, my = mean2d[..., 0], mean2d[..., 1]
        # Q-33: a box that only touches the image counts as off it.
        off_image = (
            (mx + radius_x <= 0.0)
            | (mx - radius_x >= cam.width)
            | (my + radius_y <= 0.0)
            | (my - radius_y >= cam.height)
        )
        kept = in_depth & passes & det_ok & ~off_image
        visible = kept & (radius_x > 0.0) & (radius_y > 0.0)

    return _Projection(
        means2d=torch.where(visible[:, None], mean2d, torch.zeros_like(mean2d)),
        conics=conics,
        depths=z.detach(),
        colors=_colors(g.sh, g.means_t, cam.center),
        visible=visible,
    )


def _colors(sh: Tensor, means_t: Tensor, center: Tensor) -> Tensor:
    """E2: max(0, Σ c_lm Y_lm(d) + 0.5) per channel, with d from the camera center to the moved
    position (D-4); `sh_basis` normalizes d."""
    degree = math.isqrt(sh.shape[1]) - 1
    basis = sh_basis(means_t - center, degree)
    return torch.clamp((sh * basis[..., None]).sum(dim=-2) + _SH_OFFSET, min=0.0)


def _rasterize(
    means2d: Tensor,
    conics: Tensor,
    opacities: Tensor,
    colors: Tensor,
    background: Tensor,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor]:
    """E6's compositing of the visible primitives, sorted front to back, at every pixel center:
    image [H, W, 3] and alpha [H, W, 1] = 1 − T_final (Q-32).

    Every pixel is evaluated against every primitive. That gives gsplat's image: a primitive's α
    reaches the skip threshold only inside its screen box, whose extent cap is never reached
    since σ_eff ≤ 1. Rows of pixels are processed a chunk at a time to bound memory."""
    like = means2d
    px = torch.arange(width, dtype=like.dtype, device=like.device) + _PIXEL_CENTER
    py = torch.arange(height, dtype=like.dtype, device=like.device) + _PIXEL_CENTER
    count = int(means2d.shape[0])
    chunk = max(1, _CHUNK_ELEMENTS // max(1, width * count))
    a, b, c = conics.unbind(-1)
    images, alphas = [], []
    for v0 in range(0, height, chunk):
        dx = px[None, :, None] - means2d[:, 0]  # [h, W, M]
        dy = py[v0 : v0 + chunk, None, None] - means2d[:, 1]
        q = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        alpha = torch.clamp(opacities * torch.exp(-q), max=_ALPHA_MAX)
        # E6: terms with α < 1/255, or a negative quadratic form, are skipped.
        alpha = torch.where((alpha < _ALPHA_SKIP) | (q < 0.0), torch.zeros_like(alpha), alpha)
        # T after each primitive, with 1 before the first. T never increases, so the primitives
        # before E6's stop are those leaving T above the threshold: a prefix, which also excludes
        # the primitive that would cross it.
        ones = alpha.new_ones((*alpha.shape[:-1], 1))
        transmittance = torch.cat([ones, torch.cumprod(1.0 - alpha, dim=-1)], dim=-1)
        included = transmittance[..., 1:] > _T_STOP
        weights = torch.where(included, alpha * transmittance[..., :-1], torch.zeros_like(alpha))
        t_final = torch.gather(transmittance, -1, included.sum(dim=-1, keepdim=True))
        images.append((weights[..., None] * colors).sum(dim=-2) + t_final * background)
        alphas.append(1.0 - t_final)
    return torch.cat(images), torch.cat(alphas)


# The gsplat backend.


def _render_gsplat(
    g: GaussiansAtTime,
    cameras: list[Camera],
    background: Tensor,
    *,
    rows: Tensor | None,
    num_rows: int,
    packed: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Image [C, H, W, 3], alpha [C, H, W, 1], 2D means ([C, N, 2] unpacked, [nnz, 2] packed),
    visibility [C, N] and, packed, the model row behind each row of the 2D means [nnz], of the
    per-time arrays `g`, which hold all N rows or the ascending subset `rows` (D-75, D-76)."""
    num_cams, width, height = len(cameras), cameras[0].width, cameras[0].height
    like = g.means_t
    if like.shape[0] == 0:
        # D-76: gsplat's kernels cannot run on zero rows; this is Q-32's empty render.
        return _empty_gsplat_render(g, background, width, height, num_rows=num_rows, packed=packed)

    import gsplat

    # Adapted from gsplat 1.5.3, gsplat/rendering.py, rasterization() (lines 407-525, 631-648 and
    # 746-759), Apache-2.0: its stage functions in its order, for this package's configuration
    # (pinhole camera, the mode _RASTERIZE_MODE names, classic by default (D-6), no distortion, one
    # GPU, RGB). Calling the stages directly
    # keeps Q-32's 2D means in every layout (D-75); the background and the camera center are this
    # package's own (D-76).
    with stage("raster.forward.projection"):
        cams = [_camera_tensors(camera, like) for camera in cameras]
        viewmats = torch.stack([_view_matrix(cam) for cam in cams])
        intrinsics = torch.stack([_intrinsics(cam) for cam in cams])
        centers = torch.stack([cam.center for cam in cams])
        projection = gsplat.fully_fused_projection(
            g.means_t,
            None,
            g.quats_n,
            g.scales,
            viewmats,
            intrinsics,
            width,
            height,
            eps2d=_EPS2D,
            near_plane=_NEAR,
            far_plane=_FAR,
            radius_clip=_RADIUS_CLIP,
            packed=packed,
            sparse_grad=False,
            # Derived from _RASTERIZE_MODE (D-6): antialiasing scales the effective opacities
            # (below).
            calc_compensations=_RASTERIZE_MODE == "antialiased",
            camera_model="pinhole",
            opacities=g.opacities_t,  # E6, D-32: the opacity cull and the opacity-aware extent
        )
        camera_ids: Tensor | None = None
        local_ids: Tensor | None = None
        gaussian_ids: Tensor | None = None
        if packed:
            _, camera_ids, local_ids, radii, means2d, depths, conics, compensations = projection
            # Every packed row is written and visible: gsplat's radius test drops a row only when
            # both radii are at most radius_clip, here 0 (D-57), and ε keeps both diagonals of Σ′
            # positive while the extent is shared, so a row's radii are both positive or both 0
            # (E6, D-6). gsplat's own 2D means are therefore the anchor (D-75). Rows come
            # camera-major, ascending within a camera (D-76).
            anchor = rasterized = means2d
            passed_visible = (radii > 0).all(dim=-1)
            dirs = g.means_t[local_ids] - centers[camera_ids]
            coeffs = g.sh[local_ids]
            opacities = g.opacities_t[local_ids]
            gaussian_ids = local_ids if rows is None else rows[local_ids]
            visible = torch.zeros((num_cams, num_rows), dtype=torch.bool, device=like.device)
            # A value already on the device, so the scatter doesn't synchronize (D-49).
            visible.index_put_((camera_ids, gaussian_ids), visible.new_ones(()))
        else:
            radii, means2d, depths, conics, compensations = projection
            passed_visible = (radii > 0).all(dim=-1)  # E6: both screen radii positive
            # gsplat leaves the 2D means of culled rows uninitialized; Q-32 wants 0 there, in the
            # tensor the rasterizer reads (D-75).
            means2d = torch.where(passed_visible[..., None], means2d, torch.zeros_like(means2d))
            if rows is None:
                anchor = rasterized = means2d
                visible = passed_visible
            else:
                # Q-31, D-75: the 2D means go back among all N rows, and the rasterizer reads the
                # kept rows from there, so that [C, N, 2] tensor is the anchor.
                anchor = means2d.new_zeros((num_cams, num_rows, 2)).index_copy(1, rows, means2d)
                rasterized = anchor.index_select(1, rows)
                visible = passed_visible.new_zeros((num_cams, num_rows)).index_copy(
                    1, rows, passed_visible
                )
            dirs = g.means_t[None] - centers[:, None]
            coeffs = g.sh.expand(num_cams, *g.sh.shape)
            opacities = g.opacities_t.expand(num_cams, -1)
        if compensations is not None:
            opacities = opacities * compensations  # as gsplat's rasterization() does
    with stage("raster.forward.sh"):
        # E2, D-4: SH from the camera center to the moved position, + 0.5, clamped at 0; at_time has
        # already cut the bands above the active degree (D-3).
        degree = math.isqrt(g.sh.shape[1]) - 1
        colors = gsplat.spherical_harmonics(degree, dirs, coeffs, masks=passed_visible)
        colors = torch.clamp_min(colors + _SH_OFFSET, 0.0)
    with stage("raster.forward.isect"):
        tile_width = math.ceil(width / _TILE_SIZE)
        tile_height = math.ceil(height / _TILE_SIZE)
        _, isect_ids, flatten_ids = gsplat.isect_tiles(
            rasterized,
            radii,
            depths,
            _TILE_SIZE,
            tile_width,
            tile_height,
            segmented=False,  # D-57
            packed=packed,
            n_images=num_cams,
            image_ids=camera_ids,
            gaussian_ids=local_ids,  # gsplat's own ids, local to the rows passed
        )
        offsets = gsplat.isect_offset_encode(isect_ids, num_cams, tile_width, tile_height)
        offsets = offsets.reshape(num_cams, tile_height, tile_width)
    with stage("raster.forward.rasterize"):
        image, alpha = gsplat.rasterize_to_pixels(
            rasterized,
            conics,
            colors,
            opacities,
            width,
            height,
            _TILE_SIZE,
            offsets,
            flatten_ids,
            backgrounds=None,
            packed=packed,
            absgrad=False,
        )
        # D-76: E6's background, T_final·c_bg = (1 − alpha)·c_bg, composited here in both layouts,
        # since gsplat 1.5.3 cannot take a per-camera background in packed mode.
        image = image + (1.0 - alpha) * background[:, None, None, :]
    return image, alpha, anchor, visible, gaussian_ids


def _view_matrix(cam: _CameraTensors) -> Tensor:
    """gsplat's view matrix [4, 4] (world to camera) from R_w and t_w only (§1.4, Q-32)."""
    # Filled on the device, not written from the host, so a render doesn't synchronize (D-49).
    bottom = torch.cat([cam.translation.new_zeros(3), cam.translation.new_ones(1)])[None]
    return torch.cat([torch.cat([cam.rotation, cam.translation[:, None]], dim=1), bottom])


def _intrinsics(cam: _CameraTensors) -> Tensor:
    """gsplat's K [3, 3] from f_x, f_y, c_x and c_y only (§1.4, Q-32)."""
    zero, one = torch.zeros_like(cam.fx), torch.ones_like(cam.fx)
    return torch.stack(
        [
            torch.stack([cam.fx, zero, cam.cx]),
            torch.stack([zero, cam.fy, cam.cy]),
            torch.stack([zero, zero, one]),
        ]
    )


def _empty_gsplat_render(
    g: GaussiansAtTime,
    background: Tensor,
    width: int,
    height: int,
    *,
    num_rows: int,
    packed: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Q-32's empty render, in the gsplat backend's layouts (D-76), for per-time arrays `g` with no
    rows. It stays on the autograd graph, as the reference's empty render does: the outputs depend
    with weight zero on the 2D means, which depend with weight zero on every per-time array, so
    every stored tensor and the 2D means get an exactly zero gradient (U-8, D-57)."""
    num_cams = background.shape[0]
    arrays = (g.means_t, g.opacities_t, g.quats_n, g.scales, g.sh)
    zero = sum((array.sum() for array in arrays), start=g.means_t.new_zeros(())) * 0.0
    means2d = zero.expand(0, 2) if packed else zero.expand(num_cams, num_rows, 2)
    through = means2d.sum() * 0.0  # the outputs read the 2D means, as compositing does
    image = background[:, None, None, :].expand(num_cams, height, width, 3) + through
    alpha = background.new_zeros((num_cams, height, width, 1)) + through
    visible = torch.zeros((num_cams, num_rows), dtype=torch.bool, device=background.device)
    if not packed:
        return image, alpha, means2d, visible, None
    ids = torch.zeros(0, dtype=torch.int64, device=background.device)
    return image, alpha, means2d, visible, ids
