# SPDX-License-Identifier: Apache-2.0
"""InitPoints to tensors, k-NN velocities, DLT triangulation, and the optional RoMa pipeline on the
vendored copy in openftgs._vendor.romatch.

Spec: §1.3, E15, D-58, D-65. Band: Foundation (§2.1).
"""

import math
from typing import NoReturn

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

__all__ = ["knn_velocities"]  # Q-106

_KNN_TILE = 1 << 24  # distances held at once by the tiled k-NN search (query rows × points)


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


def _check_point_set(name: str, value: Tensor) -> None:
    # Q-T135.1: a floating-point [n, 3] tensor of finite coordinates.
    if value.dim() != 2 or value.shape[1] != 3:
        _fail(f"{name} must be [n, 3], got {list(value.shape)}")
    if not value.is_floating_point():
        _fail(f"{name} must be floating point, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        _fail(f"{name} has a coordinate that isn't finite")


def _nearest_kdtree(query: Tensor, data: Tensor) -> Tensor:
    """The CPU path of `_nearest_indices`: SciPy's cKDTree, whose tie order is undocumented, so
    each row's candidates are re-sorted by (distance, index) and a row is queried again with
    twice the neighbours while its last candidate still ties the nearest distance (D-56)."""
    q = query.detach().cpu().numpy()
    d = data.detach().cpu().numpy()
    m = d.shape[0]
    tree = cKDTree(d)
    out = np.empty(q.shape[0], dtype=np.int64)
    rows = np.arange(q.shape[0])
    k = min(2, m)
    while rows.size:
        dist, idx = tree.query(q[rows], k=list(range(1, k + 1)), workers=-1)
        # Distances come sorted; a tie at the nearest distance can reach past the k candidates
        # only while the last candidate is still at that distance.
        more = (dist[:, -1] == dist[:, 0]) & (k < m)
        done = ~more
        at_nearest = dist[done] == dist[done, :1]
        out[rows[done]] = np.where(at_nearest, idx[done], m).min(axis=1)
        rows = rows[more]
        k = min(2 * k, m)
    return torch.from_numpy(out).to(query.device)


def _nearest_tiled(query: Tensor, data: Tensor) -> Tensor:
    """The CUDA path of `_nearest_indices`: brute force over tiles of query rows, on explicit
    coordinate differences rather than ‖a‖² + ‖b‖² − 2a·b (D-56)."""
    n, m = int(query.shape[0]), int(data.shape[0])
    out = torch.empty(n, dtype=torch.long, device=query.device)
    columns = torch.arange(m, device=query.device)
    rows = max(1, _KNN_TILE // m)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        d = torch.cdist(query[start:stop], data, compute_mode="donot_use_mm_for_euclid_dist")
        # The lowest index among the columns at the nearest distance, whatever order a
        # reduction kernel would pick among ties.
        at_nearest = d == d.amin(dim=1, keepdim=True)
        out[start:stop] = torch.where(at_nearest, columns, m).amin(dim=1)
    return out


def _nearest_indices(query: Tensor, data: Tensor) -> Tensor:
    """For each row of `query` [N, 3], the index in `data` [M, 3] (M ≥ 1) of its exact nearest
    neighbour, ties to the lower index (D-56). Both float64 and finite, on one device; returns
    long [N] there."""
    query, data = query.detach(), data.detach()
    if query.device.type == "cuda":
        return _nearest_tiled(query, data)
    return _nearest_kdtree(query, data)


def knn_velocities(points: Tensor, next_points: Tensor, dt: float) -> Tensor:
    """E15: v_i = (x'_nn(i) − x_i)/dt, with nn(i) the exact nearest neighbour of x_i among
    `next_points`, ties to the lower index, computed in float64 (D-23, D-56).

    `points` [N, 3] and `next_points` [M, 3] sit on one device; `dt` is t_f′ − t_f, negative for
    the backward difference on the last init frame. Returns [N, 3] in the dtype of `points`.
    Raises `ValueError` for M = 0, a `dt` that is zero or not finite (Q-106), malformed shapes,
    non-floating dtypes, two devices or a coordinate that isn't finite (Q-T135.1)."""
    _check_point_set("points", points)
    _check_point_set("next_points", next_points)
    if next_points.device != points.device:
        _fail(f"points and next_points sit on {points.device} and {next_points.device}")
    if next_points.shape[0] == 0:
        _fail("next_points is empty")
    dt = float(dt)
    if dt == 0.0 or not math.isfinite(dt):
        _fail(f"dt must be finite and nonzero, got {dt}")
    p = points.to(torch.float64)
    nxt = next_points.to(torch.float64)
    idx = _nearest_indices(p, nxt)
    return ((nxt[idx] - p) / dt).to(points.dtype)
