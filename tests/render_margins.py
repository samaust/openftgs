# SPDX-License-Identifier: Apache-2.0
"""Q-39's boundary margins for comparing the two render backends (E6; U-7, U-12).

Where the gsplat and reference backends' float32 arithmetic can land on opposite sides of an E6
discontinuity, a pixel or a row can change by about 1/255 or more with no bug. This module finds
those places with its own float64 NumPy evaluation of E1 and E3 to E6, sharing no code with either
backend, so the comparisons can exclude them (Q-39): exclusions, not a looser tolerance.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from openftgs.model import FreeTimeGaussians
from openftgs.types import Camera

ALPHA_SKIP = 1.0 / 255.0  # E6
T_STOP = 1e-4  # E6
NEAR, FAR = 0.01, 1e10  # E6, D-6

# Q-39's margins.
ALPHA_REL = 1e-3  # α within this relative distance of 1/255
STOP_REL = 1e-3  # T within this relative distance of 1e-4
DEPTH_REL = 1e-5  # two contributing depths within this relative distance
RADIUS_ABS = 1e-4  # e·√Σ′ᵢᵢ within this distance of an integer
EDGE_ABS = 1e-4  # a box edge within this distance (px) of the image edge
OPACITY_REL = 1e-3  # σ_eff within this relative distance of 1/255
NEAR_ABS = 1e-3  # z within this distance of the near plane

_PARAMS = (
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
)


@dataclass(frozen=True)
class Margins:
    """Per camera, the pixels [C, H, W] and rows [C, N] to leave out of a comparison."""

    pixels: np.ndarray
    rows: np.ndarray


def _per_time(model: FreeTimeGaussians, t: float) -> dict[str, np.ndarray]:
    p = {name: getattr(model, name).detach().cpu().to(torch.float64).numpy() for name in _PARAMS}
    means_t = p["means"] + p["velocities"] * (t - p["times"])[:, None]  # E1
    sigma_t = np.exp(-0.5 * ((t - p["times"]) / np.exp(p["log_durations"])) ** 2)  # E4
    opacity = sigma_t / (1.0 + np.exp(-p["opacity_logits"]))  # E3
    w, x, y, z = (p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)).T
    rot = np.stack(  # E5
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)
    rs = rot * np.exp(p["log_scales"])[:, None, :]
    return {"means_t": means_t, "opacity": opacity, "cov": rs @ rs.transpose(0, 2, 1)}


def opacity_near_cull(model: FreeTimeGaussians, t: float) -> np.ndarray:
    """[N] bool: σ_eff within OPACITY_REL of the opacity cull's 1/255 (Q-39)."""
    opacity = _per_time(model, t)["opacity"]
    return np.abs(opacity - ALPHA_SKIP) <= OPACITY_REL * ALPHA_SKIP


def depth_near_plane(model: FreeTimeGaussians, cameras: Sequence[Camera], t: float) -> np.ndarray:
    """[N] bool: z_c within NEAR_ABS of the near plane in some camera (Q-39)."""
    means_t = _per_time(model, t)["means_t"]
    near = np.zeros(means_t.shape[0], dtype=bool)
    for camera in cameras:
        w2c = camera.world_to_camera.to(torch.float64).numpy()
        z = means_t @ w2c[2, :3] + w2c[2, 3]
        near |= np.abs(z - NEAR) <= NEAR_ABS
    return near


def boundary_margins(model: FreeTimeGaussians, cameras: Sequence[Camera], t: float) -> Margins:
    """The pixels and rows of each camera within Q-39's margins of an E6 discontinuity."""
    q = _per_time(model, t)
    pixels, rows = [], []
    for camera in cameras:
        pix, row = _camera_margins(q, camera)
        pixels.append(pix)
        rows.append(row)
    return Margins(pixels=np.stack(pixels), rows=np.stack(rows))


def _camera_margins(q: dict[str, np.ndarray], camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    means_t, opacity, cov = q["means_t"], q["opacity"], q["cov"]
    n = means_t.shape[0]
    k_mat = camera.K.to(torch.float64).numpy()
    w2c = camera.world_to_camera.to(torch.float64).numpy()
    r_w, t_w = w2c[:3, :3], w2c[:3, 3]
    fx, fy, cx, cy = k_mat[0, 0], k_mat[1, 1], k_mat[0, 2], k_mat[1, 2]
    width, height = camera.width, camera.height

    xc = means_t @ r_w.T + t_w
    z = xc[:, 2]
    in_depth = (z >= NEAR) & (z <= FAR)
    zs = np.where(in_depth, z, 1.0)
    u, v = xc[:, 0] / zs, xc[:, 1] / zs
    mean2d = np.stack([fx * u + cx, fy * v + cy], axis=1)
    # E6: the Jacobian at x_c, with x_c/z_c and y_c/z_c clamped for J only.
    tx = np.clip(u, -(cx / fx + 0.15 * width / fx), (width - cx) / fx + 0.15 * width / fx)
    ty = np.clip(v, -(cy / fy + 0.15 * height / fy), (height - cy) / fy + 0.15 * height / fy)
    jac = np.zeros((n, 2, 3))
    jac[:, 0, 0], jac[:, 0, 2] = fx / zs, -fx * tx / zs
    jac[:, 1, 1], jac[:, 1, 2] = fy / zs, -fy * ty / zs
    cov2d = jac @ (r_w @ cov @ r_w.T) @ jac.transpose(0, 2, 1) + 0.3 * np.eye(2)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    passes = in_depth & (opacity >= ALPHA_SKIP) & (det > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        extent = np.minimum(3.33, np.sqrt(2.0 * np.log(np.where(passes, 255.0 * opacity, 1.0))))
        exact_x = extent * np.sqrt(cov2d[:, 0, 0])
        exact_y = extent * np.sqrt(cov2d[:, 1, 1])
    rx, ry = np.ceil(exact_x), np.ceil(exact_y)
    mx, my = mean2d[:, 0], mean2d[:, 1]
    row = passes & (
        (np.abs(exact_x - np.round(exact_x)) <= RADIUS_ABS)
        | (np.abs(exact_y - np.round(exact_y)) <= RADIUS_ABS)
        | (np.abs(mx + rx) <= EDGE_ABS)
        | (np.abs(mx - rx - width) <= EDGE_ABS)
        | (np.abs(my + ry) <= EDGE_ABS)
        | (np.abs(my - ry - height) <= EDGE_ABS)
    )

    # Per pixel: α of every row that passes the culls, depth order, and T after each term.
    ids = np.nonzero(passes)[0]
    pixel = np.zeros((height, width), dtype=bool)
    if ids.size == 0:
        return pixel, row
    ids = ids[np.lexsort((ids, z[ids]))]
    conic = np.linalg.inv(cov2d[ids])
    px = np.arange(width) + 0.5
    py = np.arange(height) + 0.5
    dx = px[None, :, None] - mx[ids]  # [1, W, M]
    dy = py[:, None, None] - my[ids]  # [H, 1, M]
    quad = 0.5 * (conic[:, 0, 0] * dx * dx + conic[:, 1, 1] * dy * dy) + conic[:, 0, 1] * dx * dy
    alpha = np.minimum(0.999, opacity[ids] * np.exp(-quad))  # [H, W, M]
    pixel |= (np.abs(alpha - ALPHA_SKIP) <= ALPHA_REL * ALPHA_SKIP).any(axis=-1)
    drawn = alpha >= ALPHA_SKIP
    trans = np.cumprod(np.where(drawn, 1.0 - alpha, 1.0), axis=-1)
    pixel |= (np.abs(trans - T_STOP) <= STOP_REL * T_STOP).any(axis=-1)
    depth = z[ids]
    close = np.abs(depth[:, None] - depth[None, :]) <= DEPTH_REL * np.maximum(
        depth[:, None], depth[None, :]
    )
    np.fill_diagonal(close, False)
    near_alpha = alpha >= ALPHA_SKIP * (1.0 - ALPHA_REL)
    pixel |= (near_alpha & ((near_alpha.astype(float) @ close.astype(float)) > 0)).any(axis=-1)
    return pixel, row


def excluded_share(margins: Margins) -> float:
    """The largest share of excluded pixels over the cameras."""
    return float(max(np.mean(p) for p in margins.pixels))
