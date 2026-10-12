# SPDX-License-Identifier: Apache-2.0
"""FreeTimeGaussians: the nine stored tensors, their activations and at_time().

Spec: §1.1, E5. Band: Core model (§2.1).
"""

import importlib.metadata
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from openftgs.config import InitConfig, config_from_dict
from openftgs.io import load_checkpoint, read_exchange, validate_exchange, validate_init_points
from openftgs.motion import (
    durations_to_normalized,
    durations_to_raw,
    temporal_slice,
    velocities_to_normalized,
    velocities_to_raw,
)
from openftgs.sh import rgb_to_sh0
from openftgs.types import ExchangeModel, InitPoints, TimeNormalizer

__all__ = ["FreeTimeGaussians", "GaussiansAtTime", "covariance", "rotation_matrix"]  # Q-20

_SH_DEGREES = (0, 1, 2, 3)  # D-3
_MIN_QUAT_NORM = 1e-8  # §1.1 validity rules
_OPACITY_EPS = 1e-7  # §2.2 ftgs-if: opacity clamped to [1e-7, 1 − 1e-7] on export and import


# E5 (Q-20): public so the render backends and the split of E17 share one definition.
def rotation_matrix(quats: Tensor) -> Tensor:
    """R(q/‖q‖) of E5 for quaternions (w, x, y, z): ``[..., 4]`` → ``[..., 3, 3]``."""
    q = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def covariance(quats: Tensor, scales: Tensor) -> Tensor:
    """Σ = R(q)·S·Sᵀ·R(q)ᵀ with S = diag(a) (E5): ``[..., 4]``, ``[..., 3]`` → ``[..., 3, 3]``."""
    rs = rotation_matrix(quats) * scales[..., None, :]  # R·diag(a)
    return rs @ rs.transpose(-1, -2)


# Q-20: the return type of at_time, the temporal slice (Q-12) plus the SH coefficients.
@dataclass(frozen=True)
class GaussiansAtTime:
    """The per-time arrays of §1.1 at one time, with the slice's base opacity, E8 terms and
    culling mask (Q-12, Q-20)."""

    means_t: Tensor  # [N, 3] μx(t), E1
    opacities_t: Tensor  # [N] σ_eff = σ·σt, E3
    quats_n: Tensor  # [N, 4] q/‖q‖
    scales: Tensor  # [N, 3] exp(â)
    sh: Tensor  # [N, (L′+1)², 3] concat(sh0, shN), cut to the requested degree
    opacities: Tensor  # [N] σ = sigmoid(ô)
    reg_terms: Tensor  # [N] σ·sg[σt], the per-row terms of E8
    passes_cull: Tensor  # [N] bool, σ_eff ≥ 1/255 (D-32)


def _fail(reason: str) -> None:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(f"FreeTimeGaussians: {reason}")


def _check_sh_degree(sh_degree: object) -> int:
    if (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, int)
        or sh_degree not in _SH_DEGREES
    ):
        _fail(f"sh_degree must be one of {_SH_DEGREES}, got {sh_degree!r}")
    assert isinstance(sh_degree, int)
    return sh_degree


def _knn_mean_distance(positions: Tensor, k: int, floor: float) -> Tensor:
    """Mean distance from each point to its min(k, n − 1) nearest others, floored (D-25, Q-21).

    Brute force in float64 by chunks of rows, with differences rather than the matrix-product
    form, so points far from the origin keep their precision. A lone point gets the floor."""
    n = int(positions.shape[0])
    if n < 2:
        return torch.full((n,), floor, dtype=torch.float64, device=positions.device)
    k = min(k, n - 1)
    x = positions.to(torch.float64)
    out = torch.empty(n, dtype=torch.float64, device=x.device)
    rows = max(1, (1 << 24) // n)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        d = torch.cdist(x[start:stop], x, compute_mode="donot_use_mm_for_euclid_dist")
        local = torch.arange(stop - start, device=x.device)
        d[local, local + start] = math.inf  # a point is not its own neighbour
        out[start:stop] = torch.topk(d, k, dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(floor)


def _canonical_sign(quats: Tensor) -> Tensor:
    """Q-24: the sign that makes the first nonzero component positive: w ≥ 0, and where w = 0
    the next nonzero component decides."""
    first = torch.argmax((quats != 0).to(torch.int8), dim=-1, keepdim=True)
    lead = torch.gather(quats, -1, first)
    return torch.where(lead < 0, -quats, quats)


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


class FreeTimeGaussians(torch.nn.Module):
    """The nine stored tensors of §1.1, registered as parameters under their own names."""

    PARAM_NAMES = (
        "means",
        "times",
        "log_durations",
        "velocities",
        "log_scales",
        "quats",
        "opacity_logits",
        "sh0",
        "shN",
    )

    means: torch.nn.Parameter
    times: torch.nn.Parameter
    log_durations: torch.nn.Parameter
    velocities: torch.nn.Parameter
    log_scales: torch.nn.Parameter
    quats: torch.nn.Parameter
    opacity_logits: torch.nn.Parameter
    sh0: torch.nn.Parameter
    shN: torch.nn.Parameter  # noqa: N815 (§1.1's normative name)

    def __init__(
        self,
        params: Mapping[str, Tensor],
        *,
        sh_degree: int,
        time: TimeNormalizer,
        time_unit: str = "s",  # Q-17
    ) -> None:
        super().__init__()
        self.sh_degree = _check_sh_degree(sh_degree)
        self.time = time
        self.time_unit = time_unit  # Q-17
        self._slice_impl = "auto"  # Q-13; how PerfConfig.slice_impl reaches it is M10's
        self._set_params(params)

    @classmethod
    def from_init_points(
        cls,
        points: InitPoints,
        time: TimeNormalizer,
        cfg: InitConfig,
        *,
        frame_dt: float,
        generator: torch.Generator,
        sh_degree: int = 3,  # Q-16
        n_max: int | None = None,  # Q-16
    ) -> "FreeTimeGaussians":
        """Initialization steps 7–9 of §1.3 for points from roma_init or supplied through the
        public interface: at most F·B points in total (D-24), defaults for missing velocities,
        durations and frame ids (D-23, D-21, D-47), per-frame 3-NN scales (D-25) and the fixed
        initial values of step 9 (D-5). `frame_dt` is Δt_f (E14)."""
        validate_init_points(points)  # Q-22
        sh_degree = _check_sh_degree(sh_degree)
        if not (math.isfinite(frame_dt) and frame_dt > 0.0):
            _fail(f"frame_dt must be finite and positive, got {frame_dt!r}")
        # §1.3 step 7: F frames of the sequence, 1 for a single timestamp; Q-21: round half up.
        frames = 1 if time.tau1 == time.tau0 else math.floor(1.0 / frame_dt + 0.5) + 1
        budget = frames * cfg.points_per_frame
        if n_max is not None and budget > n_max:
            _fail(
                f"the initial budget F·B = {frames}·{cfg.points_per_frame} = {budget} exceeds "
                f"N_max = {n_max} (D-24)"
            )
        n = int(points.positions.shape[0])
        device = points.positions.device
        keep: Tensor | None = None
        if n > budget:
            # Q-21: a seeded uniform subset, back in file order; the generator is used only here.
            draw = torch.randperm(n, generator=generator, device=generator.device)[:budget]
            keep = torch.sort(draw).values.to(device)

        def kept(value: Tensor | None) -> Tensor | None:
            return value if value is None or keep is None else value[keep]

        positions = kept(points.positions)
        colors = kept(points.colors)
        timestamps = kept(points.timestamps)
        velocities_tau = kept(points.velocities)
        durations_tau = kept(points.durations)
        frame_ids = kept(points.frame_ids)
        assert positions is not None and colors is not None and timestamps is not None
        m = int(positions.shape[0])

        t = time.to_t(timestamps.to(torch.float64))
        if velocities_tau is None:
            velocities = torch.zeros(m, 3, dtype=torch.float32, device=device)  # D-23
        else:
            velocities = velocities_to_normalized(velocities_tau, time)  # E14, Q-27
        if durations_tau is None:
            durations = torch.full((m,), frame_dt, dtype=torch.float64, device=device)  # D-21
        else:
            durations = durations_to_normalized(durations_tau, time)  # E14, Q-27
        if frame_ids is None:
            # D-47: the nearest slot t_k = k·Δt_f; Q-21: ties to the lower slot, within [0, F).
            frame_ids = torch.ceil(t / frame_dt - 0.5).clamp(0, frames - 1).to(torch.int64)

        # §1.3 step 8, D-25: isotropic scales from the 3-NN within each init frame.
        mean_distance = torch.empty(m, dtype=torch.float64, device=device)
        for frame in torch.unique(frame_ids):
            rows = torch.nonzero(frame_ids == frame).squeeze(1)
            mean_distance[rows] = _knn_mean_distance(
                positions[rows], cfg.scale_knn, cfg.scale_floor
            )
        log_scales = torch.log(mean_distance)[:, None].expand(m, 3)

        # §1.3 step 9, D-5.
        logit = math.log(cfg.init_opacity / (1.0 - cfg.init_opacity))
        quats = torch.zeros(m, 4, dtype=torch.float32, device=device)
        quats[:, 0] = 1.0
        params = {
            "means": positions,
            "times": t.to(torch.float32),
            "log_durations": torch.log(durations).to(torch.float32),
            "velocities": velocities.to(torch.float32),
            "log_scales": log_scales.to(torch.float32).contiguous(),
            "quats": quats,
            "opacity_logits": torch.full((m,), logit, dtype=torch.float32, device=device),
            "sh0": rgb_to_sh0(colors)[:, None, :],
            "shN": torch.zeros(m, (sh_degree + 1) ** 2 - 1, 3, dtype=torch.float32, device=device),
        }
        return cls(params, sh_degree=sh_degree, time=time)

    @classmethod
    def from_exchange(cls, exchange: ExchangeModel, *, device: str = "cuda") -> "FreeTimeGaussians":
        """A model from exchange-model contents (§2.2 ftgs-if): validated (Q-22), converted to
        the stored forms in float64, the opacity clamped before the logit."""
        validate_exchange(exchange)
        a = exchange.arrays
        time = TimeNormalizer(exchange.tau0, exchange.tau1)
        f64 = torch.float64
        span = time.tau1 - time.tau0
        # E14, μt = (τc − τ₀)/T_τ, computed directly: TimeNormalizer.to_t would warn for time
        # centers outside the training span, which are legitimate (Q-9); t ≡ 0 when T_τ = 0.
        center = a["time_center"].to(f64)
        times = (center - time.tau0) / span if span > 0.0 else torch.zeros_like(center)
        opacity = a["opacity"].to(f64).clamp(_OPACITY_EPS, 1.0 - _OPACITY_EPS)
        sh = a["sh"]
        params = {
            "means": a["position"],
            "times": times,
            "log_durations": torch.log(durations_to_normalized(a["duration"].to(f64), time)),
            "velocities": velocities_to_normalized(a["velocity"].to(f64), time),
            "log_scales": torch.log(a["scale"].to(f64)),
            "quats": a["rotation"],
            "opacity_logits": torch.logit(opacity),
            "sh0": sh[:, :1],
            "shN": sh[:, 1:],
        }
        params = {
            k: v.to(device=device, dtype=torch.float32).contiguous() for k, v in params.items()
        }
        return cls(params, sh_degree=exchange.sh_degree, time=time, time_unit=exchange.time_unit)

    @classmethod
    def load(cls, path: Path, *, device: str = "cuda") -> "FreeTimeGaussians":
        """A model from a checkpoint directory or a ftgs-if file (§2.1). From a checkpoint, a
        model above the stored configuration's N_max is rejected (§1.1; Q-25)."""
        path = Path(path)
        if not path.is_dir():
            return cls.from_exchange(read_exchange(path), device=device)
        state = load_checkpoint(path)
        cfg = config_from_dict(state.config)
        model = cls(
            {k: v.to(device) for k, v in state.params.items()},
            sh_degree=cfg.sh_degree,
            time=state.time,
            time_unit=state.time_unit,
        )
        if model.num_gaussians > cfg.growth.n_max:
            _fail(f"N = {model.num_gaussians} exceeds N_max = {cfg.growth.n_max}")
        return model

    def physical(self) -> dict[str, Tensor]:
        """Activated values in raw time units under the ftgs-if keys (§2.2; Q-20): exactly the
        arrays to_exchange writes, detached."""
        time = self.time
        f64 = torch.float64
        with torch.no_grad():
            quats = self.quats.detach()
            unit = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
            opacity = torch.sigmoid(self.opacity_logits.detach().to(f64))
            return {
                "position": self.means.detach().clone(),
                "time_center": time.to_tau(self.times.detach().to(f64)),
                "duration": durations_to_raw(torch.exp(self.log_durations.detach().to(f64)), time),
                "velocity": velocities_to_raw(self.velocities.detach().to(f64), time).float(),
                "scale": torch.exp(self.log_scales.detach()),
                "rotation": _canonical_sign(unit),
                "opacity": opacity.clamp(_OPACITY_EPS, 1.0 - _OPACITY_EPS).float(),
                "sh": torch.cat([self.sh0.detach(), self.shN.detach()], dim=1),
            }

    def to_exchange(self) -> ExchangeModel:
        """The exchange model of §2.2 ftgs-if: physical values in raw time units."""
        return ExchangeModel(
            arrays=self.physical(),
            sh_degree=self.sh_degree,
            tau0=self.time.tau0,
            tau1=self.time.tau1,
            time_unit=self.time_unit,
            producer=_producer(),
        )

    def _set_params(self, params: Mapping[str, Tensor]) -> None:
        """Check the §1.1 validity rules, then register the nine tensors as parameters."""
        if set(params) != set(self.PARAM_NAMES):
            _fail(f"params must hold exactly {self.PARAM_NAMES}, got {sorted(params)}")
        n = int(params["means"].shape[0]) if params["means"].dim() == 2 else -1
        k = (self.sh_degree + 1) ** 2 - 1
        shapes = {
            "means": (n, 3),
            "times": (n,),
            "log_durations": (n,),
            "velocities": (n, 3),
            "log_scales": (n, 3),
            "quats": (n, 4),
            "opacity_logits": (n,),
            "sh0": (n, 1, 3),
            "shN": (n, k, 3),
        }
        device = params["means"].device
        for name in self.PARAM_NAMES:
            value = params[name]
            if tuple(value.shape) != shapes[name]:
                _fail(f"{name} must have shape {list(shapes[name])}, got {list(value.shape)}")
            if value.dtype != torch.float32:
                _fail(f"{name} must be float32, got {value.dtype}")
            if value.device != device:
                _fail(f"{name} is on {value.device}, means on {device}")
            if not bool(torch.isfinite(value).all()):
                _fail(f"{name} has values that are not finite")
        if not bool((torch.linalg.vector_norm(params["quats"], dim=-1) > _MIN_QUAT_NORM).all()):
            _fail(f"every quaternion norm must exceed {_MIN_QUAT_NORM}")
        for name in self.PARAM_NAMES:
            setattr(self, name, torch.nn.Parameter(params[name].detach()))

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def at_time(self, t: float | Tensor, *, sh_degree: int | None = None) -> GaussiansAtTime:
        """The per-time arrays at normalized time t (§1.1), with SH cut to `sh_degree` bands
        (default: the model's degree; Q-20)."""
        degree = self.sh_degree if sh_degree is None else sh_degree
        if (
            isinstance(degree, bool)
            or not isinstance(degree, int)
            or not 0 <= degree <= self.sh_degree
        ):
            _fail(f"sh_degree must be an int in [0, {self.sh_degree}], got {sh_degree!r}")
        s = temporal_slice(
            self.means,
            self.times,
            self.log_durations,
            self.velocities,
            self.log_scales,
            self.quats,
            self.opacity_logits,
            t,
            impl=self._slice_impl,  # type: ignore[arg-type]
        )
        sh = torch.cat([self.sh0, self.shN[:, : (degree + 1) ** 2 - 1]], dim=1)
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

    def resize_(
        self, *, keep: Tensor | None = None, append: Mapping[str, Tensor] | None = None
    ) -> None:
        """Keep the rows `keep` selects, in their old order, then append the rows of `append`
        last (§1.3 Growth step 4). `keep` is a bool mask [N] or strictly increasing int64
        indices (Q-23). The result is revalidated; on error the model is unchanged."""
        n = self.num_gaussians
        if keep is not None:
            if keep.dtype == torch.bool:
                if tuple(keep.shape) != (n,):
                    _fail(f"a keep mask must have shape [{n}], got {list(keep.shape)}")
            elif keep.dtype == torch.int64:
                if keep.dim() != 1:
                    _fail("keep indices must be one-dimensional")
                if keep.numel() and (int(keep.min()) < 0 or int(keep.max()) >= n):
                    _fail(f"keep indices must lie in [0, {n})")
                if keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()):
                    _fail("keep indices must be strictly increasing")
            else:
                _fail(f"keep must be a bool mask or int64 indices, got {keep.dtype}")
        if append is not None and set(append) != set(self.PARAM_NAMES):
            _fail(f"append must hold exactly {self.PARAM_NAMES}, got {sorted(append)}")
        params: dict[str, Tensor] = {}
        for name in self.PARAM_NAMES:
            value = getattr(self, name).detach()
            if keep is not None:
                value = value[keep.to(value.device)]
            if append is not None:
                extra = append[name]
                if extra.dim() != value.dim() or extra.shape[1:] != value.shape[1:]:
                    _fail(f"append[{name!r}] must have rows of shape {list(value.shape[1:])}")
                value = torch.cat([value, extra.to(value.device)])
            params[name] = value
        self._set_params(params)
