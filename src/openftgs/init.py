# SPDX-License-Identifier: Apache-2.0
"""InitPoints to tensors, k-NN velocities, DLT triangulation, and the optional RoMa pipeline on the
vendored copy in openftgs._vendor.romatch, which is imported only when RoMa is used.

Spec: §1.3, E15, D-21 to D-25, D-47, D-54, D-56, D-58, D-65, D-66, D-70, D-74; Q-106 to Q-112,
Q-116, Q-120 to Q-123; docs/modules/init/roma-pipeline.md. Band: Foundation (§2.1).
"""

import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.resources
import json
import math
import pathlib
import urllib.parse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, Protocol

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch import Tensor

from openftgs import _shards
from openftgs._stages import stage
from openftgs.data import MultiViewVideo
from openftgs.io import validate_init_points
from openftgs.types import Camera, InitPoints

__all__ = [  # Q-106
    "DenseMatcher",
    "RomaInitConfig",
    "knn_velocities",
    "roma_init",
    "roma_provenance",
    "roma_weight_files",  # Q-T220.1
    "triangulate_dlt",
]

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
    """Q-106's rules, with the type checks of Q-122: integers in the integer fields, a bool
    in `upsample_preds`, a finite number in `min_certainty` (an int becomes a float, as in
    openftgs.config, Q-10), and no bool in a numeric field."""
    # Q-106, Q-122
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
    vendor = importlib.resources.files("openftgs._vendor") / "romatch" / "VENDOR.json"
    return str(json.loads(vendor.read_text(encoding="utf-8"))["commit"])


def roma_provenance(cfg: RomaInitConfig) -> dict[str, str]:
    """The ftgs-init metadata that records how `roma_init` ran (D-58; Q-107): one key,
    `roma_init`, holding compact JSON of every RomaInitConfig field, `use_custom_corr` as
    `roma_init` resolves it (Q-112), the vendored RoMa's commit and, for Tiny RoMa, the commit
    of XFeat's code (Q-116; its weights come from Tiny RoMa's checkpoint, Q-123). For
    `write_init_points(..., metadata=roma_provenance(cfg))`."""
    _check_config(cfg)
    # Q-107; Q-122: the keys sorted, so the string depends only on the values.
    record: dict[str, object] = {
        **dataclasses.asdict(cfg),
        "use_custom_corr": _use_custom_corr(),
        "roma_commit": _roma_commit(),
    }
    if cfg.model == "tiny_roma_v1_outdoor":
        record["xfeat_commit"] = _XFEAT_COMMIT
    return {"roma_init": json.dumps(record, separators=(",", ":"), sort_keys=True)}


def _model_zoo() -> Any:
    """The vendored RoMa's model zoo (D-65), imported only when RoMa is used; ImportError
    without the [init] extra."""
    return importlib.import_module("openftgs._vendor.romatch.models.model_zoo")


def _weight_urls(cfg: RomaInitConfig, zoo: Any) -> list[str]:
    """The URLs of the checkpoints the default matcher loads for `cfg.model` (D-58), in load
    order: RoMa's and DINOv2's for roma_outdoor, Tiny RoMa's for Tiny RoMa, whose XFeat is built
    without weights (Q-123)."""
    urls = zoo.weight_urls
    if cfg.model == "roma_outdoor":
        return [urls["romatch"]["outdoor"], urls["dinov2"]]
    return [urls["tiny_roma_v1"]["outdoor"]]


def roma_weight_files(cfg: RomaInitConfig) -> list[pathlib.Path]:  # Q-T220.1
    """The checkpoint files `roma_init` loads for `cfg` with its default matcher (D-58), where
    torch.hub caches them: `torch.hub.get_dir()/checkpoints/`, each under its URL's file name,
    in load order. `roma_init` downloads the ones missing; this only names them, downloading
    and creating nothing. Tiny RoMa also fetches XFeat's code archive through torch.hub, which
    holds no weights (Q-116, Q-123) and isn't listed. Raises ImportError without the [init]
    extra, which the vendored RoMa needs (D-65)."""
    _check_config(cfg)
    folder = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
    names = [
        pathlib.PurePosixPath(urllib.parse.urlparse(url).path).name
        for url in _weight_urls(cfg, _model_zoo())
    ]
    return [folder / name for name in names]


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
    each row's candidates are re-sorted by (distance, index), "with extra neighbours queried
    while the last two distances are equal" (§2.4 "Initialization and conversion", D-56; Q-106).
    A row is queried again with twice the neighbours while its last candidate still ties the
    nearest distance: since the distances come sorted, that tie means the last two are equal,
    and a tie farther out than the nearest distance can't change the lowest index at it, so
    this stricter condition gives the results the spec's does."""
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


# The RoMa pipeline (§1.3 steps 1–7, E15; docs/modules/init/roma-pipeline.md).

_MAX_ERROR_PX = 2.0  # D-22: reprojection error in both views
_MIN_ANGLE = math.radians(1.0)  # D-22: triangulation angle


class DenseMatcher(Protocol):
    """What matches the two images of a camera pair (Q-106): RoMa by default, or one supplied
    to `roma_init`. The images are [H, W, 3] float32 in [0, 1] on the CPU, as
    `MultiViewVideo.load_image` returns them. It returns a warp [M, 4] of rows (x_A, y_A, x_B,
    y_B) in RoMa's normalized coordinates, where ±1 are each image's outer edges, and a
    certainty [M] in [0, 1]."""

    def __call__(self, image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class _Job:
    """One camera pair of one init frame slot (§1.3 step 2): the record indices of the images
    of cameras A and B, A first in the manifest's camera order (Q-108)."""

    slot: int  # k, the slot's frame index
    a: int
    b: int
    camera_a: str
    camera_b: str
    key: tuple[int, int, int]  # (k, A's and B's positions in camera_names): canonical order


@dataclass(frozen=True)
class _PairPoints:
    """The points a pair keeps, in draw order (Q-111)."""

    positions: Tensor  # [n, 3] float32 on the CPU, rounded from the float64 triangulation
    colors: Tensor  # [n, 3] float32
    timestamp: float  # raw τ: the mean of the two images' (§1.3 step 4, D-47)


@dataclass(frozen=True)
class _InitFrame:
    """One init frame: the index k of its slot, its full point set, for E15 against it, and
    its kept subset."""

    k: int
    full: Tensor  # [n, 3] float32, every point of the init frame's pairs in canonical order
    positions: Tensor  # the kept subset (§1.3 step 7), in canonical order
    colors: Tensor
    timestamps: Tensor  # [m] float64, raw τ


def _generator(*parts: object) -> torch.Generator:
    """A CPU generator of its own for one draw (D-56): seeded with the first 8 bytes,
    little-endian, of the SHA-256 of the compact JSON of ("openftgs", "roma_init", *parts), as
    Q-72's pattern (Q-110, Q-111, Q-121)."""
    text = json.dumps(["openftgs", "roma_init", *parts], separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def _frame_count(dataset: MultiViewVideo) -> int:
    """F of §1.3 step 7: ⌊1/Δt_f + ½⌋ + 1, or 1 for a single timestamp (Q-21)."""
    time = dataset.time
    return 1 if time.tau1 == time.tau0 else math.floor(1.0 / dataset.frame_dt + 0.5) + 1


def _admitted(dataset: MultiViewVideo, cfg: RomaInitConfig) -> dict[int, dict[str, int]]:
    """§1.3 step 1, D-21, D-47, D-66; Q-109: for each slot k ∈ {0, s, 2s, …} ≤ F − 1, the record
    index of each training camera's image there. An image joins its nearest slot,
    k = s·⌈t/(s·Δt_f) − ½⌉ (ties to the lower slot, as Q-21's frames), when |t − t_k| ≤ Δt_f/2;
    per camera the closest image wins, then the earlier timestamp, then the lower record index.
    At s = 1 the nearest slot is the nearest frame; at s > 1 an image within Δt_f/2 of a slot
    has that slot as its nearest, whatever its nearest frame."""
    frame_dt = float(dataset.frame_dt)
    names = set(dataset.manifest.camera_names)
    slots: dict[int, dict[str, int]] = {k: {} for k in range(0, _frame_count(dataset), cfg.stride)}
    ranks: dict[tuple[int, str], tuple[float, float, int]] = {}
    for index, record in enumerate(dataset.records):
        if record.split != "train":  # D-66: initialization never reads a test-split image
            continue
        name = record.camera.name
        if name not in names:
            _fail(f"camera {name!r} of image {index} is missing from the manifest's cameras")
        t = float(dataset.time.to_t(float(record.timestamp)))
        k = cfg.stride * math.ceil(t / (cfg.stride * frame_dt) - 0.5)  # Q-21, Q-109
        offset = abs(t - k * frame_dt)
        if k not in slots or offset > frame_dt / 2:
            continue
        rank = (offset, float(record.timestamp), index)
        if (k, name) not in ranks or rank < ranks[k, name]:
            ranks[k, name] = rank
            slots[k][name] = index
    return slots


def _pairs(dataset: MultiViewVideo, chosen: dict[str, int]) -> list[tuple[str, str]]:
    """§1.3 step 2, D-22; Q-108, Q-109: each camera's partner has the largest optical-axis
    cosine (row 3 of R_w of the admitted image's own camera), in float64; ties go to the
    smaller center distance, then to the earlier position in the manifest's camera order. Each
    unordered pair once, as (A, B) in that order, the pairs in that order too."""
    position = {name: i for i, name in enumerate(dataset.manifest.camera_names)}
    names = sorted(chosen, key=position.__getitem__)
    axes: dict[str, Tensor] = {}
    centers: dict[str, Tensor] = {}
    for name in names:
        w2c = dataset.records[chosen[name]].camera.world_to_camera.to(torch.float64)
        axes[name] = w2c[2, :3] / torch.linalg.vector_norm(w2c[2, :3])
        centers[name] = -(w2c[:3, :3].T @ w2c[:3, 3])
    pairs: set[tuple[str, str]] = set()
    for name in names:
        others = [other for other in names if other != name]
        if not others:
            break

        def rank(other: str, name: str = name) -> tuple[float, float, int]:
            cosine = float(axes[name] @ axes[other])
            distance = float(torch.linalg.vector_norm(centers[name] - centers[other]))
            return (-cosine, distance, position[other])

        partner = min(others, key=rank)
        a, b = sorted((name, partner), key=position.__getitem__)
        pairs.add((a, b))
    return sorted(pairs, key=lambda pair: (position[pair[0]], position[pair[1]]))


def _plan(dataset: MultiViewVideo, cfg: RomaInitConfig) -> tuple[list[int], list[_Job]]:
    """The slots and the matching jobs, listed in record order (the PRD's job order); results
    are assembled by `_Job.key`, so the order changes no number."""
    position = {name: i for i, name in enumerate(dataset.manifest.camera_names)}
    admitted = _admitted(dataset, cfg)
    jobs = []
    for k, chosen in admitted.items():
        for a, b in _pairs(dataset, chosen):
            key = (k, position[a], position[b])
            jobs.append(_Job(k, chosen[a], chosen[b], a, b, key))
    jobs.sort(key=lambda job: (job.slot, min(job.a, job.b), max(job.a, job.b)))
    return list(admitted), jobs


def _checked_matches(warp: Tensor, certainty: Tensor) -> tuple[Tensor, Tensor]:
    # Q-106's port, Q-122: a warp [M, 4] and a certainty [M], floating point.
    if not isinstance(warp, Tensor) or warp.dim() != 2 or warp.shape[1] != 4:
        _fail(f"a dense matcher must return a warp [M, 4], got {getattr(warp, 'shape', warp)}")
    if not isinstance(certainty, Tensor) or tuple(certainty.shape) != (warp.shape[0],):
        _fail(
            f"a dense matcher must return a certainty [M] with M = {warp.shape[0]}, got "
            f"{getattr(certainty, 'shape', certainty)}"
        )
    if not (warp.is_floating_point() and certainty.is_floating_point()):
        _fail(f"a dense matcher must return floating point, got {warp.dtype}, {certainty.dtype}")
    return warp.detach().to("cpu"), certainty.detach().to("cpu")


def _draw(warp: Tensor, certainty: Tensor, cfg: RomaInitConfig, job: _Job) -> Tensor:
    """§1.3 step 2, D-22, D-56, D-58; Q-110: of the matches inside both images (every
    coordinate finite and within [−1, 1]) whose raw certainty reaches `min_certainty`, up to
    `matches_per_pair` drawn without replacement in proportion to certainty, on the CPU, with
    the pair's own generator. Returns the drawn warp rows in draw order."""
    inside = (torch.isfinite(warp) & (warp.abs() <= 1.0)).all(dim=1)
    # Q-121: a match of certainty 0 has probability 0, so it is never eligible (this
    # matters only at min_certainty = 0, where torch.multinomial couldn't draw it anyway).
    eligible = inside & torch.isfinite(certainty) & (certainty >= cfg.min_certainty)
    eligible &= certainty > 0
    weights = certainty[eligible]
    count = min(cfg.matches_per_pair, int(weights.shape[0]))
    if count == 0:
        return warp[:0]
    generator = _generator("pair", cfg.seed, job.slot, job.camera_a, job.camera_b)  # Q-110
    drawn = torch.multinomial(weights, count, replacement=False, generator=generator)
    return warp[eligible][drawn]


def _pixels(normalized: Tensor, camera: Camera) -> Tensor:
    """§1.3 step 2: RoMa's [−1, 1] onto [0, W] × [0, H], §1.4's continuous convention, with no
    half-pixel shift, in float64 (Q-110)."""
    size = torch.tensor([camera.width, camera.height], dtype=torch.float64)
    return size / 2 * (normalized.to(torch.float64) + 1.0)


def _projection(camera: Camera) -> Tensor:
    """P = K[R_w | t_w] in float64 (§1.4)."""
    w2c = camera.world_to_camera.to(torch.float64)
    return camera.K.to(torch.float64) @ w2c[:3, :]


def _view(X: Tensor, camera: Camera, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The depth of X in a view, its reprojection error there in pixels, and the ray from the
    view's center to X, in float64 on X's device."""
    w2c = camera.world_to_camera.to(X.device, torch.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    in_camera = X @ R.T + t
    projected = in_camera @ camera.K.to(X.device, torch.float64).T
    error = torch.linalg.vector_norm(projected[:, :2] / projected[:, 2:] - x, dim=1)
    return in_camera[:, 2], error, X - (-(R.T @ t))


def _triangulated(
    xa: Tensor, xb: Tensor, camera_a: Camera, camera_b: Camera, device: torch.device
) -> tuple[Tensor, Tensor]:
    """§1.3 step 3, D-22, D-56: `triangulate_dlt` on the shard's device, then D-22's filters on
    every row, since the solver filters nothing (Q-120): a point is kept when it is finite, its
    depth is positive in both views, its reprojection error is at most 2 px in both, and the
    angle between the rays from the two camera centers is at least 1°. Returns the points
    [n, 3] float64 and the kept mask [n], both on the CPU."""
    xa, xb = xa.to(device), xb.to(device)
    P_a, P_b = _projection(camera_a).to(device), _projection(camera_b).to(device)
    X = triangulate_dlt(xa, xb, P_a, P_b)
    finite = torch.isfinite(X).all(dim=1)  # Q-120: non-finite rows are dropped explicitly
    depth_a, error_a, ray_a = _view(X, camera_a, xa)
    depth_b, error_b, ray_b = _view(X, camera_b, xb)
    sine = torch.linalg.vector_norm(torch.linalg.cross(ray_a, ray_b), dim=1)
    angle = torch.atan2(sine, (ray_a * ray_b).sum(dim=1))
    keep = (
        finite
        & (depth_a > 0)
        & (depth_b > 0)
        & (error_a <= _MAX_ERROR_PX)
        & (error_b <= _MAX_ERROR_PX)
        & (angle >= _MIN_ANGLE)
    )
    return X.cpu(), keep.cpu()


def _source_pixels(image: Tensor, x: Tensor) -> Tensor:
    """Q-110: the pixel that contains each continuous coordinate, floor clamped to the image."""
    height, width = int(image.shape[0]), int(image.shape[1])
    u = torch.floor(x[:, 0]).clamp(0, width - 1).to(torch.long)
    v = torch.floor(x[:, 1]).clamp(0, height - 1).to(torch.long)
    return image[v, u]


def _match_pair(
    job: _Job,
    matcher: DenseMatcher,
    device: torch.device,
    dataset: MultiViewVideo,
    cfg: RomaInitConfig,
) -> _PairPoints:
    """§1.3 steps 2–4 for one camera pair, on one shard: the stage "init.match", keyed by the
    shard's device, with the triangulation's stage inside it (§3.4 B-8)."""
    with stage("init.match", key=str(device)):
        record_a, record_b = dataset.records[job.a], dataset.records[job.b]
        image_a, image_b = dataset.load_image(job.a), dataset.load_image(job.b)
        warp, certainty = _checked_matches(*matcher(image_a, image_b))
        drawn = _draw(warp, certainty, cfg, job)
        xa = _pixels(drawn[:, :2], record_a.camera)
        xb = _pixels(drawn[:, 2:], record_b.camera)
        with stage("init.triangulate", key=str(device)):
            X, keep = _triangulated(xa, xb, record_a.camera, record_b.camera, device)
        # §1.3 step 3, D-22; Q-110: the color is the mean of the two source pixels.
        colors = (_source_pixels(image_a, xa[keep]) + _source_pixels(image_b, xb[keep])) / 2
        return _PairPoints(
            positions=X[keep].to(torch.float32),  # Q-111: the values InitPoints stores
            colors=colors,
            timestamp=(float(record_a.timestamp) + float(record_b.timestamp)) / 2,  # D-47
        )


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    """§2.4 "Initialization and conversion", "Precision", D-56; Q-112: float32 matmul precision
    "highest" (RoMa refuses any other), cuDNN autotuning off, deterministic cuDNN and strict
    deterministic algorithms, for the whole call, restored afterwards."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic)
    algorithms = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    matmul = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        cudnn.benchmark, cudnn.deterministic = False, True
        torch.use_deterministic_algorithms(True)  # Q-112: strict
        yield
    finally:
        cudnn.benchmark, cudnn.deterministic = saved
        torch.use_deterministic_algorithms(algorithms[0], warn_only=algorithms[1])
        torch.set_float32_matmul_precision(matmul)


def _devices(devices: Sequence[str] | None) -> list[torch.device]:
    """Q-106: one shard per entry; None means every visible CUDA device."""
    if devices is None:
        count = torch.cuda.device_count()
        if count == 0:
            _fail("devices is None and no CUDA device is visible; pass devices, e.g. ['cpu']")
        return [torch.device("cuda", i) for i in range(count)]
    return _shards.devices(devices)  # Q-122: a non-empty sequence of device names


def _pil(image: Tensor) -> Image.Image:
    """Q-110: RoMa's input, round(255·x) as uint8, byte-exact for Q-65's 8-bit images."""
    pixels = (image.detach().cpu() * 255.0).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(pixels.contiguous().numpy(), "RGB")


class _RomaMatcher:
    """The roma_outdoor adapter (D-58; Q-106, Q-110): PIL images in, RoMa's whole two-way warp
    and its certainty out, flattened, as float32 on the CPU."""

    def __init__(self, model: Any, device: torch.device) -> None:
        self._model = model
        self._device = device

    def __call__(self, image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]:
        warp, certainty = self._model.match(_pil(image_a), _pil(image_b), device=self._device)
        # match() runs in inference mode; clone() gives ordinary tensors back.
        return (
            warp.reshape(-1, 4).to("cpu", torch.float32).clone(),
            certainty.reshape(-1).to("cpu", torch.float32).clone(),
        )


class _TinyRomaMatcher:
    """The Tiny RoMa adapter (D-58; Q-116): its one-way warp at input resolution, whose matches
    outside image B `_draw` drops, with the call wrapped in inference mode."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]:
        with torch.inference_mode():  # Q-116
            warp, certainty = self._model.match(_pil(image_a), _pil(image_b))
        return (
            warp.reshape(-1, 4).to("cpu", torch.float32).clone(),
            certainty.reshape(-1).to("cpu", torch.float32).clone(),
        )


def _roma_matchers(cfg: RomaInitConfig) -> Callable[[torch.device], DenseMatcher]:
    """The default dense matcher per `cfg.model` (D-58), from the vendored RoMa (D-65), which
    is imported only here. The weights are fetched once, in the caller's thread, and every
    shard's model is built from them (Q-112)."""
    zoo = _model_zoo()
    urls = _weight_urls(cfg, zoo)  # the files roma_weight_files names (Q-T220.1)
    fetch = torch.hub.load_state_dict_from_url
    if cfg.model == "roma_outdoor":
        weights = fetch(urls[0], map_location="cpu")
        dinov2_weights = fetch(urls[1], map_location="cpu")
        use_custom_corr = _use_custom_corr()  # D-58, Q-112

        def build_roma(device: torch.device) -> DenseMatcher:
            model = zoo.roma_outdoor(
                device,
                weights=weights,
                dinov2_weights=dinov2_weights,
                coarse_res=cfg.coarse_res,
                upsample_res=cfg.upsample_res,
                upsample_preds=cfg.upsample_preds,
                use_custom_corr=use_custom_corr,
            )
            return _RomaMatcher(model, device)

        return build_roma
    weights = fetch(urls[0], map_location="cpu")
    # Q-116: XFeat at the commit whose license was read, not the repository's moving default.
    # torch.hub's validation accepts a commit only while it heads a branch or tag (it lists
    # them through GitHub's API; source log row 131), so Q-116's fallback applies: the
    # validation is skipped, and the archive of that commit is downloaded as it is.
    # Q-123: XFeat is built without weights, since its own loader would take them from the
    # repository's moving `main`, and none are loaded into it: Tiny RoMa's checkpoint replaces
    # every XFeat weight a shard keeps (source log row 138).
    xfeat = torch.hub.load(  # type: ignore[no-untyped-call]
        f"verlab/accelerated_features:{_XFEAT_COMMIT}",
        "XFeat",
        pretrained=False,
        top_k=4096,
        trust_repo=True,
        skip_validation=True,
    ).net

    def build_tiny(device: torch.device) -> DenseMatcher:
        # Tiny RoMa deletes XFeat's unused heads when built, so each shard gets its own copy.
        model = zoo.tiny_roma_v1_outdoor(device, weights=weights, xfeat=copy.deepcopy(xfeat))
        return _TinyRomaMatcher(model)

    return build_tiny


def _init_frame(k: int, parts: Sequence[_PairPoints], cfg: RomaInitConfig) -> _InitFrame | None:
    """§1.3 step 7, D-24; Q-111: the init frame of slot k, None when its pairs keep no point.
    Its points come in canonical pair order, of which at most k_stride·B are kept, as a seeded
    uniform subset sorted back to that order; the slot's generator is left untouched when
    everything fits."""
    if not parts or sum(int(part.positions.shape[0]) for part in parts) == 0:
        return None
    positions = torch.cat([part.positions for part in parts])
    colors = torch.cat([part.colors for part in parts])
    timestamps = torch.cat(
        [
            torch.full((int(part.positions.shape[0]),), part.timestamp, dtype=torch.float64)
            for part in parts
        ]
    )
    n, cap = int(positions.shape[0]), cfg.stride * cfg.points_per_frame
    if n <= cap:
        return _InitFrame(k, positions, positions, colors, timestamps)
    drawn = torch.randperm(n, generator=_generator("slot", cfg.seed, k))[:cap]  # Q-111
    keep = torch.sort(drawn).values
    return _InitFrame(k, positions, positions[keep], colors[keep], timestamps[keep])


def roma_init(
    dataset: MultiViewVideo,
    cfg: RomaInitConfig,
    *,
    devices: Sequence[str] | None = None,  # Q-106
    matcher: Callable[[torch.device], DenseMatcher] | None = None,  # Q-106
) -> InitPoints:
    """Initial points from dense matching, two-view triangulation and k-NN velocities (§1.3
    steps 1–7, E15; D-21 to D-24, D-47, D-56, D-58, D-66, D-74), with velocities, durations
    and frame ids filled, on the CPU.

    `devices`: one shard per entry, each with its own dense matcher; None means every visible
    CUDA device. `matcher` builds a shard's dense matcher for its device; None means RoMa per
    `cfg.model`, which needs the [init] extra and downloads weights (Q-106). Raises ValueError
    when `devices` is None and no CUDA device is visible, and when no point survives."""
    _check_config(cfg)
    shard_devices = _devices(devices)
    slots, jobs = _plan(dataset, cfg)
    by_slot: dict[int, list[tuple[int, int, int]]] = {k: [] for k in slots}
    for job in jobs:
        by_slot[job.slot].append(job.key)
    for keys in by_slot.values():
        keys.sort()  # Q-111: canonical pair order

    def work(shard: tuple[DenseMatcher, torch.device], job: _Job) -> _PairPoints:
        shard_matcher, device = shard
        return _match_pair(job, shard_matcher, device, dataset, cfg)

    time = dataset.time
    span = time.tau1 - time.tau0
    unit = span if span > 0 else 1.0  # D-70: a time unit of 1 for a single timestamp
    frame_dt = float(dataset.frame_dt)
    e15_device = shard_devices[0]
    # The finished init frames, each with its kept points' velocities. An init frame's full
    # set is dropped once it is finished and no longer needed by E15 (the PRD's job order).
    finished: list[tuple[_InitFrame, Tensor]] = []

    def finish(frame: _InitFrame, other: _InitFrame | None) -> None:
        if other is None:  # §1.3 step 5: a single init frame gives v = 0
            v = torch.zeros_like(frame.positions)
        else:  # E15, D-23; Q-111: against the other init frame's full set, dt from slot times
            dt = (other.k - frame.k) * frame_dt * unit
            with stage("init.knn"):  # §3.4 B-8, up to the copy back, which waits for it
                v = knn_velocities(frame.positions.to(e15_device), other.full.to(e15_device), dt)
                v = v.cpu()
        finished.append((dataclasses.replace(frame, full=frame.full.new_empty(0, 3)), v.cpu()))

    with _deterministic():
        build = matcher if matcher is not None else _roma_matchers(cfg)
        # D-54, D-56; Q-106, Q-112: one thread and one dense matcher per `devices` entry, the
        # matchers built here, one after another; results come back as pairs finish.
        shards = [(build(device), device) for device in shard_devices]
        with _shards.Shards(shards, jobs, work, name="roma_init shard") as running:
            results: dict[tuple[int, int, int], _PairPoints] = {}
            previous: _InitFrame | None = None  # the last init frame, not yet finished
            before: _InitFrame | None = None  # the init frame before it
            for k in slots:
                while any(key not in results for key in by_slot[k]):
                    job, result = running.next_result()
                    results[job.key] = result
                current = _init_frame(k, [results.pop(key) for key in by_slot[k]], cfg)
                if current is None:  # no point survived at slot k: it has no init frame
                    continue
                if previous is not None:
                    finish(previous, current)  # the next init frame
                before, previous = previous, current
        if previous is None:
            _fail("roma_init: no point survived matching and the filters of D-22")
        finish(previous, before)  # D-23: the last init frame uses the previous one

    duration = cfg.stride * frame_dt * unit  # §1.3 step 6, D-21, D-70
    counts = [int(frame.positions.shape[0]) for frame, _ in finished]
    points = InitPoints(
        positions=torch.cat([frame.positions for frame, _ in finished]),
        colors=torch.cat([frame.colors for frame, _ in finished]),
        timestamps=torch.cat([frame.timestamps for frame, _ in finished]),
        velocities=torch.cat([v for _, v in finished]),
        durations=torch.full((sum(counts),), duration, dtype=torch.float64),
        # Q-111: the slot index, the label from_init_points gives supplied points.
        frame_ids=torch.cat(
            [
                torch.full((n,), frame.k, dtype=torch.int64)
                for (frame, _), n in zip(finished, counts, strict=True)
            ]
        ),
    )
    validate_init_points(points)
    return points
