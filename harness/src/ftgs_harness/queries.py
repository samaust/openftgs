# SPDX-License-Identifier: Apache-2.0
"""L2's seeded forward queries, one generator per query kind (§3.3 L2; Q-207, Q-209): random
queries plus edge queries with s from 10⁻⁶ to 10² (× T_τ), |t − μt|/s up to 10⁴, opacities near
0 and 1, and view directions along the axes."""

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ftgs_harness.cameras import intrinsics, look_at
from ftgs_harness.formats import Exchange, Query, model_tensors
from ftgs_harness.models import random_exchange, unit_quaternions

__all__ = ["GENERATORS", "generate"]

_TAU0, _TAU1 = 0.0, 2.0
_QueryGenerator = Callable[
    [np.random.Generator, bool, Mapping[str, Any]], tuple[dict[str, Any], dict[str, np.ndarray]]
]


def _with_model(ex: Exchange, **params: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tensors, model = model_tensors(ex)
    return {"model": model, **params}, dict(tensors)


def _edge_model(rng: np.random.Generator, n: int, sh_degree: int = 0) -> Exchange:
    """Durations from 10⁻⁶ to 10² × T_τ, opacities near 0 and 1."""
    ex = random_exchange(rng, n, sh_degree=sh_degree, tau0=_TAU0, tau1=_TAU1)
    span = _TAU1 - _TAU0
    ex.arrays["duration"] = np.exp(rng.uniform(math.log(1e-6), math.log(1e2), size=n)) * span
    sigma = ex.arrays["opacity"].copy()
    sigma[: n // 4] = np.float32(1e-6)
    sigma[n // 4 : n // 2] = np.float32(1.0 - 1e-6)
    ex.arrays["opacity"] = sigma
    return ex


def _times(rng: np.random.Generator, ex: Exchange, edge: bool) -> np.ndarray:
    times = [_TAU0, _TAU1, _TAU0 - 0.3, _TAU1 + 0.3, *rng.uniform(_TAU0, _TAU1, size=4)]
    if edge:  # |t − μt|/s of about 10⁴ for the row with the smallest duration
        row = int(np.argmin(ex.arrays["duration"]))
        center, s = float(ex.arrays["time_center"][row]), float(ex.arrays["duration"][row])
        times += [center + 1e4 * s, center - 1e4 * s, center + s]
    return np.array(times, dtype=np.float64)


def _per_time(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ex = (
        _edge_model(rng, 40)
        if edge
        else random_exchange(rng, 40, sh_degree=0, tau0=_TAU0, tau1=_TAU1)
    )
    params, tensors = _with_model(ex)
    tensors["times"] = _times(rng, ex, edge)
    return params, tensors


def _sh_color(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if edge:  # every row at the origin, static: the six centres look along the axes
        ex = random_exchange(rng, 12, sh_degree=3, tau0=_TAU0, tau1=_TAU1, speed=0.0, spread=0.0)
        axes = np.concatenate([np.eye(3), -np.eye(3)]) * 4.0
        params, tensors = _with_model(ex, sh_degree=None)
        tensors["centers"] = axes
        tensors["times"] = rng.uniform(_TAU0, _TAU1, size=6)
        return params, tensors
    degree = int(rng.integers(0, 4))
    ex = random_exchange(rng, 30, sh_degree=3, tau0=_TAU0, tau1=_TAU1)
    params, tensors = _with_model(ex, sh_degree=degree)
    tensors["times"] = rng.uniform(_TAU0 - 0.2, _TAU1 + 0.2, size=5)
    tensors["centers"] = rng.normal(size=(5, 3)) * 4.0
    return params, tensors


def _losses(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ex = (
        _edge_model(rng, 20)
        if edge
        else random_exchange(rng, 20, sh_degree=0, tau0=_TAU0, tau1=_TAU1)
    )
    loss = {k: v for k, v in dict(cfg.get("loss", {})).items() if k != "perc_net"}
    params, tensors = _with_model(
        ex, loss=loss, iteration=int(rng.integers(1, 30_000)), lpips=bool(cfg.get("_lpips", True))
    )
    gt = rng.random((48, 48, 3)).astype(np.float32)
    noise = 0.0 if edge else 0.1
    tensors["gt"] = gt
    tensors["pred"] = np.clip(gt + rng.normal(scale=noise, size=gt.shape), -0.1, 1.2).astype(
        np.float32
    )
    tensors["t"] = np.array(rng.uniform(_TAU0, _TAU1))
    return params, tensors


def _lr_schedule(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    optim = dict(cfg.get("optim", {}))
    total = int(optim.get("iterations", 30_000))
    its = [1, 2, total // 2, total - 1, total, *rng.integers(1, total + 1, size=3)]
    scale, frame = (
        (1e-3, 1e-4) if edge else (float(rng.uniform(0.5, 5.0)), float(rng.uniform(0.01, 0.2)))
    )
    params = {"optim": optim, "spatial_scale": scale, "frame_dt": frame}
    return params, {"iterations": np.array(sorted(set(its)), dtype=np.int64)}


def _relocation_math(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    k = 20
    sigma = rng.uniform(0.01, 0.99, size=k)
    n = rng.integers(1, 52, size=k)
    if edge:
        sigma[:4] = [0.0051, 0.9999, 0.5, 0.01]
        n[:4] = [51, 51, 1, 51]
    params = {
        "reloc": {k2: v for k2, v in dict(cfg.get("reloc", {})).items() if k2 == "dead_opacity"}
    }
    return params, {
        "opacity": sigma,
        "scales": np.exp(rng.uniform(-6, 0, size=(k, 3))),
        "n": n.astype(np.int64),
    }


def _velocity_init(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    points = rng.random((50, 3)).astype(np.float32)
    nxt = (points[rng.permutation(50)[:40]] + rng.normal(scale=0.01, size=(40, 3))).astype(
        np.float32
    )
    dt = -0.05 if edge else float(rng.uniform(0.02, 0.2))
    return {}, {"points": points, "next_points": nxt, "dt": np.array(dt)}


def _relocation_sampling(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """20 live targets and 50 seeds × 200 draws, so every expected cell holds at least 5 (Q-209)."""
    live = rng.uniform(0.3, 1.0, size=20) if not edge else np.geomspace(0.1, 1.0, 20)
    scores = np.concatenate([live, np.zeros(5)]).astype(np.float32)
    first = int(rng.integers(0, 10_000))
    return {"count": 200, "seeds": list(range(first, first + 50))}, {"scores": scores}


def _grad_stats(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    degree = 1
    ex = random_exchange(rng, 40, sh_degree=degree, tau0=_TAU0, tau1=_TAU1, log_duration=(0.0, 1.0))
    angle = float(rng.uniform(0, 2 * math.pi))
    eye = np.array([4.0 * math.cos(angle), 4.0 * math.sin(angle), 1.5])
    width, height = 48, 32
    loss = dict(cfg.get("loss", {}))
    if not cfg.get("_lpips", True):
        loss["lambda_perc"] = 0.0
    optim = {
        k: v for k, v in dict(cfg.get("optim", {})).items() if k in ("iterations", "sh_interval")
    }
    config = {"sh_degree": degree, "loss": loss, "optim": optim}
    if "background" in cfg:
        config["background"] = cfg["background"]
    total = int(optim.get("iterations", 30_000))
    params, tensors = _with_model(
        ex, width=width, height=height, iteration=int(rng.integers(1, total + 1)), config=config
    )
    tensors["K"] = intrinsics(width, height, 40.0)
    tensors["world_to_camera"] = look_at(eye)
    tensors["t"] = np.array(rng.uniform(_TAU0, _TAU1))
    tensors["gt"] = rng.random((height, width, 3)).astype(np.float32)
    return params, tensors


def _relocation_score(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n = 60
    reloc = dict(cfg.get("reloc", {}))
    dead = float(reloc.get("dead_opacity", 0.005))
    sigma = rng.uniform(0.01, 0.99, size=n)
    sigma[:10] = 0.001  # dead rows, well away from the threshold (Q-209)
    if edge:
        sigma[10:15] = 1.0 - 1e-6
    sigma = np.where(np.abs(sigma - dead) < 1e-6, dead + 1e-3, sigma)
    count = rng.integers(0, 20, size=n).astype(np.int32)
    g_sum = (rng.exponential(2e-4, size=n) * count).astype(np.float32)
    params = {
        "reloc": {
            k: v
            for k, v in reloc.items()
            if k in ("lambda_g", "lambda_o", "grad_quantile", "dead_opacity")
        }
    }
    return params, {"opacity": sigma.astype(np.float32), "g_sum": g_sum, "count": count}


def _spatial_scale(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    c = 10
    centers = rng.normal(size=(c, 3)) * (1e3 if edge else 3.0)
    w2c = []
    for center, q in zip(centers, unit_quaternions(rng, c).astype(np.float64), strict=True):
        w, x, y, z = q
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = r, -r @ center
        w2c.append(m)
    train = rng.random(c) < 0.7
    train[:2] = True
    factor = float(cfg.get("scene_scale_factor", 1.1))
    return {"factor": factor}, {"world_to_camera": np.stack(w2c), "train": train}


def _growth_decisions(
    rng: np.random.Generator, edge: bool, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    growth = dict(cfg.get("growth", {}))
    threshold = float(growth.get("grad_threshold", 2e-4))
    split_scale = float(growth.get("split_scale", 0.01))
    n = 80
    s = float(rng.uniform(0.5, 5.0))
    g = (rng.exponential(threshold, size=n)).astype(np.float32)
    if edge:  # seven rows tied at the largest g, every 13th row, with room for three of them:
        # the cap falls inside the tie, so D-44's "ties to the lower row index" decides
        g[::13] = np.float32(np.max(g) * 2)
    scales = (np.exp(rng.uniform(-1.0, 1.0, size=(n, 3))) * split_scale * s).astype(np.float32)
    candidates = int((g > threshold).sum())
    room = 3 if edge else (candidates // 2 if rng.random() < 0.5 else candidates + 10)
    params = {
        "spatial_scale": s,
        "grad_threshold": threshold,
        "split_scale": split_scale,
        "n_max": n + room,
    }
    return params, {"g": g, "scales": scales}


GENERATORS: dict[str, _QueryGenerator] = {
    "motion": _per_time,
    "temporal_opacity": _per_time,
    "effective_opacity": _per_time,
    "sh_color": _sh_color,
    "losses": _losses,
    "lr_schedule": _lr_schedule,
    "relocation_math": _relocation_math,
    "velocity_init": _velocity_init,
    "relocation_sampling": _relocation_sampling,
    "grad_stats": _grad_stats,
    "relocation_score": _relocation_score,
    "spatial_scale": _spatial_scale,
    "growth_decisions": _growth_decisions,
}


def generate(kind: str, count: int, seed: int, config: Mapping[str, Any]) -> list[Query]:
    """`count` random queries of the kind and one edge query, seeded by the run's seed."""
    out = []
    index = list(GENERATORS).index(kind)
    for i in range(count + 1):
        rng = np.random.default_rng([seed, 2, index, i])
        params, tensors = GENERATORS[kind](rng, i == count, config)
        out.append(Query(f"q{i:04d}", params, tensors))
    return out
