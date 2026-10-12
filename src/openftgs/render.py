# SPDX-License-Identifier: Apache-2.0
"""render() with two backends, gsplat 1.5.3 (CUDA, float32, production) and reference (pure
PyTorch, the readable oracle for tests), and, from M15, the temporal index used for rendering
(D-55).

Spec: E2, E5, E6, D-6, D-31, D-55, D-57. Band: Core model (§2.1).
"""

import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal, get_args

import torch
from torch import Tensor

from openftgs.model import FreeTimeGaussians, GaussiansAtTime, covariance
from openftgs.sh import sh_basis
from openftgs.types import Camera, RenderOutput

__all__ = ["render"]

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
_RASTERIZE_MODE = "classic"  # D-6: no antialiasing compensation; ftgs-if's rasterize_mode (Q-35)
_TILE_SIZE = 16  # D-57
_RADIUS_CLIP = 0.0  # D-57: no primitive is skipped for its size

_Backend = Literal["gsplat", "reference"]
_TimeUnits = Literal["raw", "normalized"]
_Precull = Literal["internal", "compact"]  # Q-31: PerfConfig.precull's values (D-32, D-57)


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
    primitive behind each row of the 2D means (Q-32, Q-37, Q-38).
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
    backgrounds = _backgrounds(background, len(cams), g.means_t)
    rows: Tensor | None = None
    if precull == "compact":
        # Q-31, D-57: compaction after the temporal slice, in ascending row order.
        rows = torch.nonzero(g.passes_cull).squeeze(1)
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
    # Q-38: gsplat's kernels compute in float32 on a CUDA device; the float64 path of Q-32 is the
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
    per-time arrays `g`, which hold all N rows or the ascending subset `rows` (Q-37, Q-38)."""
    num_cams, width, height = len(cameras), cameras[0].width, cameras[0].height
    like = g.means_t
    if like.shape[0] == 0:
        # Q-38: gsplat's kernels cannot run on zero rows; this is Q-32's empty render.
        return _empty_gsplat_render(background, width, height, num_rows=num_rows, packed=packed)

    import gsplat

    # Adapted from gsplat 1.5.3, gsplat/rendering.py, rasterization() (lines 407-525, 631-648 and
    # 746-759), Apache-2.0: its stage functions in its order, for this package's one configuration
    # (pinhole camera, classic mode, no distortion, one GPU, RGB). Calling the stages directly
    # keeps Q-32's 2D means in every layout (Q-37); the background and the camera center are this
    # package's own (Q-38).
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
        calc_compensations=False,  # classic mode (D-6, _RASTERIZE_MODE): no compensation
        camera_model="pinhole",
        opacities=g.opacities_t,  # E6, D-32: the opacity cull and the opacity-aware extent
    )
    camera_ids: Tensor | None = None
    local_ids: Tensor | None = None
    gaussian_ids: Tensor | None = None
    if packed:
        _, camera_ids, local_ids, radii, means2d, depths, conics, _ = projection
        # Every packed row is written and visible (both radii positive), so gsplat's own 2D means
        # are the anchor (Q-37). Rows come camera-major, ascending within a camera (Q-38).
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
        radii, means2d, depths, conics, _ = projection
        passed_visible = (radii > 0).all(dim=-1)  # E6: both screen radii positive
        # gsplat leaves the 2D means of culled rows uninitialized; Q-32 wants 0 there, in the
        # tensor the rasterizer reads (Q-37).
        means2d = torch.where(passed_visible[..., None], means2d, torch.zeros_like(means2d))
        if rows is None:
            anchor = rasterized = means2d
            visible = passed_visible
        else:
            # Q-31, Q-37: the 2D means go back among all N rows, and the rasterizer reads the kept
            # rows from there, so that [C, N, 2] tensor is the anchor.
            anchor = means2d.new_zeros((num_cams, num_rows, 2)).index_copy(1, rows, means2d)
            rasterized = anchor.index_select(1, rows)
            visible = passed_visible.new_zeros((num_cams, num_rows)).index_copy(
                1, rows, passed_visible
            )
        dirs = g.means_t[None] - centers[:, None]
        coeffs = g.sh.expand(num_cams, *g.sh.shape)
        opacities = g.opacities_t.expand(num_cams, -1)
    # E2, D-4: SH from the camera center to the moved position, + 0.5, clamped at 0; at_time has
    # already cut the bands above the active degree (D-3).
    degree = math.isqrt(g.sh.shape[1]) - 1
    colors = gsplat.spherical_harmonics(degree, dirs, coeffs, masks=passed_visible)
    colors = torch.clamp_min(colors + _SH_OFFSET, 0.0)
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
    # Q-38: E6's background, T_final·c_bg = (1 − alpha)·c_bg, composited here in both layouts,
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
    background: Tensor, width: int, height: int, *, num_rows: int, packed: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Q-32's empty render, in the gsplat backend's layouts (Q-38)."""
    num_cams = background.shape[0]
    image = background[:, None, None, :].expand(num_cams, height, width, 3).contiguous()
    alpha = background.new_zeros((num_cams, height, width, 1))
    visible = torch.zeros((num_cams, num_rows), dtype=torch.bool, device=background.device)
    if not packed:
        return image, alpha, background.new_zeros((num_cams, num_rows, 2)), visible, None
    ids = torch.zeros(0, dtype=torch.int64, device=background.device)
    return image, alpha, background.new_zeros((0, 2)), visible, ids
