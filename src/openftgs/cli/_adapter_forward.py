# SPDX-License-Identifier: Apache-2.0
"""`ftgs-adapter forward QUERIES OUT`: the L2 query kinds of §3.3 answered with the package's
own functions, always on the CPU (Q-202, Q-207).

The files are ftgs-forward-queries and ftgs-forward-answers (Q-207): one kind per file; a
query's tensors under `<id>/<name>`, its FTGS-IF model under `<id>/model/<key>`; each answer
carries a status, `ok`, `error` (the query couldn't be answered) or `not_comparable` (a kind
this adapter doesn't answer, or a query holding a TrainConfig setting it cannot honor, §3.3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs import cli
from openftgs.cli._adapter import _honored
from openftgs.config import GrowthConfig, LossConfig, OptimConfig, RelocConfig, config_from_dict
from openftgs.data import spatial_scale
from openftgs.density import GradStats, draw_targets, growth_masks, mcmc_split, sampling_score
from openftgs.init import knn_velocities
from openftgs.losses import LPIPSLoss, l1_loss, reg4d_loss, ssim, total_loss
from openftgs.model import FreeTimeGaussians
from openftgs.motion import effective_opacity, moved_means, temporal_opacity
from openftgs.optim import build_optimizer
from openftgs.render import render
from openftgs.sh import sh_to_color
from openftgs.train import calendar_entry
from openftgs.types import Camera, ExchangeModel, ImageRecord, TimeNormalizer

__all__ = ["KINDS", "forward"]

_QUERIES = "ftgs-forward-queries"
_ANSWERS = "ftgs-forward-answers"
_VERSION_1 = re.compile(r"1(?:\.(?:0|[1-9][0-9]*))?")

_Handler = Callable[[Mapping[str, Any], Mapping[str, Tensor]], dict[str, Tensor]]

# Q-207: lr_schedule's columns, in the order of §1.3's table.
_RATE_ORDER = (
    "means",
    "velocities",
    "times",
    "log_durations",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
)


def _model(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> FreeTimeGaussians:
    """The query's FTGS-IF model (`model/<key>` and `params.model`, Q-207) on the CPU."""
    meta = params["model"]
    exchange = ExchangeModel(
        arrays={k[len("model/") :]: v for k, v in tensors.items() if k.startswith("model/")},
        sh_degree=int(meta["sh_degree"]),
        tau0=float(meta["tau0"]),
        tau1=float(meta["tau1"]),
        time_unit=str(meta.get("time_unit", "s")),
        producer="ftgs-harness",
    )
    return FreeTimeGaussians.from_exchange(exchange, device="cpu")


def _t(model: FreeTimeGaussians, tau: float) -> float:
    """E14's normalized time of a raw τ; τ outside [τ₀, τ₁] is a query, not a mistake."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(model.time.to_t(tau))


def _per_time(
    params: Mapping[str, Any],
    tensors: Mapping[str, Tensor],
    fn: Callable[[FreeTimeGaussians, float], Tensor],
) -> Tensor:
    model = _model(params, tensors)
    return torch.stack([fn(model, _t(model, float(tau))) for tau in tensors["times"].tolist()])


def _motion(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E1: μx(t) [K, N, 3]."""
    out = _per_time(params, tensors, lambda m, t: moved_means(m.means, m.velocities, m.times, t))
    return {"means_t": out}


def _temporal_opacity(
    params: Mapping[str, Any], tensors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """E4: σt(t) [K, N]."""
    out = _per_time(params, tensors, lambda m, t: temporal_opacity(m.times, m.log_durations, t))
    return {"sigma_t": out}


def _effective_opacity(
    params: Mapping[str, Any], tensors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """E3: σ·σt(t) [K, N]."""
    out = _per_time(
        params,
        tensors,
        lambda m, t: effective_opacity(m.opacity_logits, m.times, m.log_durations, t),
    )
    return {"opacity_eff": out}


def _sh_color(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E2 at each (time, camera centre) pair, with `sh_degree` bands (null: the model's)."""
    model = _model(params, tensors)
    degree = params.get("sh_degree")
    colors = []
    for tau, center in zip(tensors["times"].tolist(), tensors["centers"], strict=True):
        g = model.at_time(_t(model, float(tau)), sh_degree=degree)
        colors.append(sh_to_color(g.sh, g.means_t - center.to(g.means_t.dtype)))
    return {"colors": torch.stack(colors)}


def _losses(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E7 to E9 on one image pair: each unweighted term and the total at `iteration` with the
    query's LossConfig fields; LPIPS-VGG only when `lpips` is true (Q-207)."""
    model = _model(params, tensors)
    cfg = LossConfig(**params.get("loss", {}))
    use_lpips = bool(params.get("lpips", True))
    if not use_lpips:
        cfg = dataclasses.replace(cfg, lambda_perc=0.0)
    pred, gt = tensors["pred"].float(), tensors["gt"].float()
    t = _t(model, float(tensors["t"]))
    it = int(params.get("iteration", 1))
    perceptual = LPIPSLoss(cfg.perc_net) if use_lpips else None
    total, _ = total_loss(pred[None], gt[None], model, t, it, cfg, perceptual=perceptual)
    out = {
        "l1": l1_loss(pred[None], gt[None]),
        "one_minus_ssim": 1.0
        - ssim(
            pred[None],
            gt[None],
            window=cfg.ssim_window,
            sigma=cfg.ssim_sigma,
            padding=cfg.ssim_padding,
        ),
        "reg": reg4d_loss(model, t),
        "total": total,
    }
    if perceptual is not None:
        out["lpips_vgg"] = LPIPSLoss("vgg")(pred[None], gt[None])
    return {k: v.detach().double().reshape(()) for k, v in out.items()}


def _dummy_model(n: int, logits: Tensor | None = None, sh_degree: int = 3) -> FreeTimeGaussians:
    """A valid model of n rows whose only meaningful tensor is the opacity logits."""
    params = {
        "means": torch.zeros(n, 3),
        "times": torch.full((n,), 0.5),
        "log_durations": torch.zeros(n),
        "velocities": torch.zeros(n, 3),
        "log_scales": torch.zeros(n, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacity_logits": torch.zeros(n) if logits is None else logits,
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3),
    }
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=TimeNormalizer(0.0, 1.0))


def _lr_schedule(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E12, E13: each tensor's rate at each iteration, columns in §1.3's order (Q-207)."""
    cfg = OptimConfig(**params.get("optim", {}))
    optimizer = build_optimizer(
        _dummy_model(1),
        cfg,
        spatial_scale=float(params["spatial_scale"]),
        frame_dt=float(params["frame_dt"]),
        adam="foreach",
    )
    rows = []
    for it in tensors["iterations"].tolist():
        rates = optimizer.set_iteration(int(it))
        rows.append([rates[name] for name in _RATE_ORDER])
    return {"rates": torch.tensor(rows, dtype=torch.float64)}


def _relocation_math(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E11 with D-14's whole clamp: `mcmc_split` applies the upper bound, and the lower one is
    applied here as the trainer applies it to the stored logit (Q-50)."""
    cfg = RelocConfig(**params.get("reloc", {}))
    sigma, scales = mcmc_split(
        tensors["opacity"].double(), tensors["scales"].double(), tensors["n"].long()
    )
    # Q-50: ô = max(float32(logit σ_new), ô_dead), ô_dead the float32 logit of dead_opacity;
    # a row the bound binds answers the opacity of ô_dead, the others σ_new.
    p = cfg.dead_opacity
    dead = torch.tensor(math.log(p) - math.log1p(-p), dtype=torch.float32)
    bound = torch.logit(sigma).float() < dead
    opacity = torch.where(bound, torch.sigmoid(dead.double()), sigma)
    return {"opacity_new": opacity, "scales_new": scales}


def _velocity_init(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """E15."""
    dt = float(tensors["dt"])
    return {"velocities": knn_velocities(tensors["points"], tensors["next_points"], dt)}


def _relocation_sampling(
    params: Mapping[str, Any], tensors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """D-15: per seed, `count` targets drawn ∝ the scores with replacement, counted per row."""
    scores = tensors["scores"].float()
    counts = []
    for seed in params["seeds"]:
        draws = draw_targets(
            scores, int(params["count"]), generator=torch.Generator().manual_seed(int(seed))
        )
        counts.append(torch.bincount(draws, minlength=scores.shape[0]))
    return {"counts": torch.stack(counts).long()}


def _grad_stats(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """D-12: one training step's NDC-scaled norm of each primitive's 2D-mean gradient of E9
    (§1.3 steps 3 to 5, as the trainer takes them, on the reference backend), and visibility."""
    model = _model(params, tensors)
    cfg = config_from_dict(params.get("config", {}))
    it = int(params["iteration"])
    width, height = int(params["width"]), int(params["height"])
    camera = Camera(
        K=tensors["K"].float(),
        world_to_camera=tensors["world_to_camera"].float(),
        width=width,
        height=height,
    )
    t = _t(model, float(tensors["t"]))
    out = render(
        model,
        camera,
        t,
        time_units="normalized",
        sh_degree=calendar_entry(cfg, it).sh_degree,
        background=torch.tensor(cfg.background),
        backend="reference",
        return_aux=True,
        packed=False,
    )
    perceptual = LPIPSLoss(cfg.loss.perc_net) if cfg.loss.lambda_perc > 0.0 else None
    loss, _ = total_loss(
        out.image, tensors["gt"].float()[None], model, t, it, cfg.loss, perceptual=perceptual
    )
    torch.autograd.backward(loss)
    assert out.means2d is not None and out.visible is not None
    stats = GradStats(model.num_gaussians)
    if out.means2d.grad is not None:
        stats.update(out.means2d.grad, width=width, height=height, visible=out.visible)
    return {"grad_norm": stats.g_sum, "visible": out.visible[0]}


def _relocation_score(
    params: Mapping[str, Any], tensors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """E10 (D-13) for every row, from the base opacities, gradient sums and visible counts."""
    opacity = tensors["opacity"].double()
    model = _dummy_model(int(opacity.shape[0]), torch.logit(opacity).float(), sh_degree=0)
    stats = GradStats(model.num_gaussians)
    stats.load_state_dict({"g_sum": tensors["g_sum"].float(), "count": tensors["count"].int()})
    return {"score": sampling_score(model, stats, RelocConfig(**params.get("reloc", {})))}


def _spatial_scale(params: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """D-27 over cameras labelled train or test; no image is read."""
    records = [
        ImageRecord(
            Camera(K=torch.eye(3), world_to_camera=w2c.float(), width=1, height=1),
            0.0,
            Path("absent.png"),
            "train" if bool(train) else "test",
        )
        for w2c, train in zip(tensors["world_to_camera"], tensors["train"], strict=True)
    ]
    scale = spatial_scale(records, float(params.get("factor", 1.1)))
    return {"S": torch.tensor(scale, dtype=torch.float64)}


def _growth_decisions(
    params: Mapping[str, Any], tensors: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """D-18, D-44: the clone and split masks after the cap."""
    cfg = GrowthConfig(
        grad_threshold=float(params["grad_threshold"]),
        split_scale=float(params["split_scale"]),
        n_max=int(params["n_max"]),
    )
    clone, split = growth_masks(
        tensors["g"].float(),
        tensors["scales"].float(),
        cfg,
        spatial_scale=float(params["spatial_scale"]),
    )
    return {"clone": clone, "split": split}


# kind → handler(params, tensors) → answer tensors (§3.3 L2, Q-207).
KINDS: dict[str, _Handler] = {
    "motion": _motion,
    "temporal_opacity": _temporal_opacity,
    "effective_opacity": _effective_opacity,
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


def _settings(kind: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """The query's TrainConfig settings, nested as in TrainConfig: grad_stats's `config` subset,
    the other kinds' `loss`, `optim` and `reloc` sections (Q-207)."""
    if kind == "grad_stats":
        return dict(params.get("config", {}))
    return {k: params[k] for k in ("loss", "optim", "reloc") if k in params}


def _read(path: Path) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Tensor]]]]:
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    # §2.2: an unknown major version is rejected; any minor version of 1 is read.
    if meta.get("format") != _QUERIES or not _VERSION_1.fullmatch(str(meta.get("version"))):
        raise ValueError(f"{path}: not a {_QUERIES} v1 file")
    entries = json.loads(meta["queries"])
    queries = []
    for entry in entries:
        prefix = entry["id"] + "/"
        own = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
        queries.append((entry["id"], dict(entry["params"]), own))
    return meta["kind"], queries


def forward(queries_path: Path, out: Path, *, backend: str) -> int:
    """Answer every query of the file; a query that fails is an `error` answer, the others
    are still answered (Q-207). A query with a setting TrainConfig doesn't have, or a value it
    refuses, is `not_comparable`, naming the settings (§3.3 adapter contract)."""
    with cli._as_input_error(f"forward queries {queries_path}", OSError, KeyError):
        sha = hashlib.sha256(queries_path.read_bytes()).hexdigest()
        kind, queries = _read(queries_path)
    handler = KINDS.get(kind)
    entries, tensors = [], {}
    for query_id, params, inputs in queries:
        if handler is None:
            entries.append({"id": query_id, "status": "not_comparable", "message": kind})
            continue
        _, unhonored = _honored(_settings(kind, params))
        if unhonored:
            message = "settings not honored: " + ", ".join(unhonored)
            entries.append({"id": query_id, "status": "not_comparable", "message": message})
            continue
        try:
            with torch.no_grad() if kind != "grad_stats" else torch.enable_grad():
                answer = handler(params, inputs)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            entries.append({"id": query_id, "status": "error", "message": str(error)})
            continue
        entries.append({"id": query_id, "status": "ok", "message": ""})
        tensors.update(
            {f"{query_id}/{k}": v.detach().cpu().contiguous() for k, v in answer.items()}
        )
    meta = {
        "format": _ANSWERS,
        "version": "1",
        "kind": kind,
        "producer": cli._producer(),
        "queries_sha256": sha,
        "answers": json.dumps(entries),
    }
    with cli._output(f"forward answers {out}"):
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out), meta)
    return 0
