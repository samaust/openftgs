# SPDX-License-Identifier: Apache-2.0
"""FreeTimeGaussians: the nine stored tensors, their activations and at_time().

Spec: §1.1, E5. Band: Core model (§2.1).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from openftgs.motion import temporal_slice
from openftgs.types import TimeNormalizer

__all__ = ["FreeTimeGaussians", "GaussiansAtTime", "covariance", "rotation_matrix"]  # Q-20

_SH_DEGREES = (0, 1, 2, 3)  # D-3
_MIN_QUAT_NORM = 1e-8  # §1.1 validity rules


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
