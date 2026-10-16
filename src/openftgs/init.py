# SPDX-License-Identifier: Apache-2.0
"""InitPoints to tensors, k-NN velocities, DLT triangulation, and the optional RoMa pipeline on the
vendored copy in openftgs._vendor.romatch.

Spec: §1.3, E15, D-58, D-65. Band: Foundation (§2.1).
"""

import dataclasses
import importlib
import importlib.resources
import json
import math
from dataclasses import dataclass
from typing import Literal, NoReturn

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

__all__ = ["RomaInitConfig", "knn_velocities", "roma_provenance", "triangulate_dlt"]  # Q-106

_KNN_TILE = 1 << 24  # distances held at once by the tiled k-NN search (query rows × points)
_MODELS = ("roma_outdoor", "tiny_roma_v1_outdoor")  # §2.1 RomaInitConfig.model, D-58
_BACKBONE_PATCH = 14  # RoMa asserts coarse resolutions divisible by DINOv2's patch (Q-106)
# Q-116: the XFeat commit whose LICENSE (Apache-2.0) was read (source log row 107).
_XFEAT_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


@dataclass
class RomaInitConfig:  # D-21, D-22, D-24, D-56, D-58
    """`roma_init`'s settings (§2.1), checked on construction (Q-106)."""

    model: Literal["roma_outdoor", "tiny_roma_v1_outdoor"] = "roma_outdoor"
    coarse_res: int = 560
    upsample_res: int = 864
    upsample_preds: bool = True  # False: match at coarse_res only
    matches_per_pair: int = 5000
    min_certainty: float = 0.5  # on match()'s raw certainty
    stride: int = 1
    # B, D-24; the trainer applies InitConfig.points_per_frame again, as the total cap F·B (D-74)
    points_per_frame: int = 1_000
    seed: int = 0

    def __post_init__(self) -> None:
        _check_config(self)


def _check_config(cfg: RomaInitConfig) -> None:
    """Q-106's rules, with the type checks of Q-T136.2: integers in the integer fields, a bool
    in `upsample_preds`, a finite number in `min_certainty` (an int becomes a float, as in
    openftgs.config, Q-10), and no bool in a numeric field."""
    # Q-106, Q-T136.2
    if not isinstance(cfg.model, str) or cfg.model not in _MODELS:
        _fail(f"RomaInitConfig.model must be one of {_MODELS}, got {cfg.model!r}")
    for name in ("coarse_res", "upsample_res", "matches_per_pair", "stride", "points_per_frame"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _fail(f"RomaInitConfig.{name} must be an integer ≥ 1, got {value!r}")
    if cfg.coarse_res % _BACKBONE_PATCH:
        _fail(f"RomaInitConfig.coarse_res must be a multiple of 14, got {cfg.coarse_res!r}")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        _fail(f"RomaInitConfig.seed must be an integer, got {cfg.seed!r}")
    if not isinstance(cfg.upsample_preds, bool):
        _fail(f"RomaInitConfig.upsample_preds must be true or false, got {cfg.upsample_preds!r}")
    certainty = cfg.min_certainty
    if (
        isinstance(certainty, bool)
        or not isinstance(certainty, (int, float))
        or not 0.0 <= certainty <= 1.0
    ):
        _fail(f"RomaInitConfig.min_certainty must be a number in [0, 1], got {certainty!r}")
    cfg.min_certainty = float(certainty)


def _use_custom_corr() -> bool:
    """D-58, Q-112: RoMa's fused local correlation is used only when `local_corr` imports; a
    wheel built for another torch fails with OSError and counts as absent."""
    try:
        importlib.import_module("local_corr")
    except (ImportError, OSError):
        return False
    return True


def _roma_commit() -> str:
    """The upstream commit of the vendored RoMa, from its VENDOR.json (D-65), read without
    importing the vendored package, so it works without the [init] extra."""
    vendor = importlib.resources.files("openftgs._vendor").joinpath("romatch", "VENDOR.json")
    return str(json.loads(vendor.read_text(encoding="utf-8"))["commit"])


def roma_provenance(cfg: RomaInitConfig) -> dict[str, str]:
    """The ftgs-init metadata that records how `roma_init` ran (D-58; Q-107): one key,
    `roma_init`, holding compact JSON of every RomaInitConfig field, `use_custom_corr` as
    `roma_init` resolves it (Q-112), the vendored RoMa's commit and, for Tiny RoMa, XFeat's
    commit (Q-116). For `write_init_points(..., metadata=roma_provenance(cfg))`."""
    _check_config(cfg)
    # Q-107; Q-T136.2: the keys sorted, so the string depends only on the values.
    record: dict[str, object] = {
        **dataclasses.asdict(cfg),
        "use_custom_corr": _use_custom_corr(),
        "roma_commit": _roma_commit(),
    }
    if cfg.model == "tiny_roma_v1_outdoor":
        record["xfeat_commit"] = _XFEAT_COMMIT
    return {"roma_init": json.dumps(record, separators=(",", ":"), sort_keys=True)}


def _check_point_set(name: str, value: Tensor) -> None:
    # Q-120: a floating-point [n, 3] tensor of finite coordinates.
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
    non-floating dtypes, two devices or a coordinate that isn't finite (Q-120)."""
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


def triangulate_dlt(x1: Tensor, x2: Tensor, P1: Tensor, P2: Tensor) -> Tensor:
    """Two-view linear DLT (§1.3 step 3): the point X whose projections by `P1` and `P2` are the
    pixels `x1` and `x2`, solved by SVD in float64 (D-56).

    `x1`, `x2` [..., 2] are pixels in §1.4's continuous convention, the one P's cx and cy use, so
    no half-pixel shift is applied. `P1`, `P2` are [3, 4] or broadcastable [..., 3, 4]. Each
    view gives the rows x·P₂ − P₀ and y·P₂ − P₁ of A [..., 4, 4]; X is the right singular vector
    of A's smallest singular value, dehomogenized. Returns float64 [..., 3] (Q-120)."""
    # Q-120: the shapes, dtypes and device the contract states.
    named = {"x1": x1, "x2": x2, "P1": P1, "P2": P2}
    for name, value in named.items():
        trailing = (2,) if name.startswith("x") else (3, 4)
        if tuple(value.shape[-len(trailing) :]) != trailing:
            shape = ", ".join(["...", *map(str, trailing)])
            _fail(f"{name} must be [{shape}], got {list(value.shape)}")
        if not value.is_floating_point():
            _fail(f"{name} must be floating point, got {value.dtype}")
    if len({value.device for value in named.values()}) > 1:
        devices = ", ".join(f"{name} on {value.device}" for name, value in named.items())
        _fail(f"x1, x2, P1 and P2 must sit on one device, got {devices}")
    try:
        batch = torch.broadcast_shapes(  # type: ignore[no-untyped-call]
            x1.shape[:-1], x2.shape[:-1], P1.shape[:-2], P2.shape[:-2]
        )
    except RuntimeError as error:
        _fail(f"the batch shapes of x1, x2, P1 and P2 don't broadcast: {error}")
    rows = []
    for x, P in ((x1, P1), (x2, P2)):
        x = x.to(torch.float64).expand(*batch, 2)
        P = P.to(torch.float64).expand(*batch, 3, 4)
        rows += [
            x[..., 0:1] * P[..., 2, :] - P[..., 0, :],
            x[..., 1:2] * P[..., 2, :] - P[..., 1, :],
        ]
    A = torch.stack(rows, dim=-2)
    # Q-120: one non-finite system makes a whole CPU batch raise and gives finite vectors on
    # CUDA (source log row 129), so such rows are solved as the identity and then set to NaN.
    finite = torch.isfinite(A).all(dim=-1).all(dim=-1)
    A = torch.where(finite[..., None, None], A, torch.eye(4, dtype=A.dtype, device=A.device))
    h = torch.linalg.svd(A).Vh[..., -1, :]
    X = h[..., :3] / h[..., 3:]
    return torch.where(finite[..., None], X, torch.nan)
