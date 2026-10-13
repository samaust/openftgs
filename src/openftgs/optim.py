# SPDX-License-Identifier: Apache-2.0
"""Adam construction, learning-rate schedules, row-wise moment reset and optimizer state I/O.

Spec: E12, E13. Band: Mechanics (§2.1).
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

import torch
from torch import Tensor

from openftgs.config import OptimConfig
from openftgs.model import FreeTimeGaussians

__all__ = ["FTGSOptimizer", "build_optimizer"]  # Q-44

_Schedule = Literal["exponential", "annealed", "constant"]
_AdamKind = Literal["fused", "foreach", "selective"]
_ADAM_KINDS = ("fused", "foreach", "selective")


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


@dataclass(frozen=True)
class _Group:
    """One parameter group: its stored tensor's name, base rates η₀, η₁ and schedule."""

    name: str
    eta0: float
    eta1: float
    schedule: _Schedule

    def rate(self, u: float) -> float:
        if self.schedule == "annealed":
            return float(self.eta0 ** (1.0 - u) * self.eta1**u)  # E12 (D-19)
        if self.schedule == "exponential":
            return float(self.eta0 * (self.eta1 / self.eta0) ** u)  # E13
        return self.eta0


def _groups(
    model: FreeTimeGaussians, cfg: OptimConfig, spatial_scale: float, frame_dt: float
) -> list[_Group]:
    """The §1.3 rate table, in §1.1's tensor order, without the tensors that take no updates."""
    groups = [
        _Group(
            "means", cfg.means_lr[0] * spatial_scale, cfg.means_lr[1] * spatial_scale, "exponential"
        ),  # E13, D-27
        _Group("times", cfg.times_lr[0], cfg.times_lr[1], "exponential"),  # D-20
        _Group("log_durations", cfg.log_durations_lr, cfg.log_durations_lr, "constant"),  # D-20
        _Group(
            "velocities",
            cfg.velocities_lr[0] * spatial_scale / frame_dt,
            cfg.velocities_lr[1] * spatial_scale / frame_dt,
            "annealed",
        ),  # E12, D-19
        _Group("log_scales", cfg.log_scales_lr, cfg.log_scales_lr, "constant"),
        _Group("quats", cfg.quats_lr, cfg.quats_lr, "constant"),
        _Group("opacity_logits", cfg.opacity_logits_lr, cfg.opacity_logits_lr, "constant"),
        _Group("sh0", cfg.sh0_lr, cfg.sh0_lr, "constant"),
        _Group("shN", cfg.shN_lr, cfg.shN_lr, "constant"),
    ]
    frozen: set[str] = set()
    if model.sh_degree == 0:
        frozen.add("shN")  # U-26: shN is [N, 0, 3] and has no group
    if model.time.tau0 == model.time.tau1:
        frozen.update(("times", "velocities"))  # D-71 by "no parameter group" (Q-45)
    return [g for g in groups if g.name not in frozen]


class FTGSOptimizer:
    """Adam over the stored tensors of a FreeTimeGaussians, one parameter group per tensor, with
    the learning-rate schedules of E12 and E13. Built by `build_optimizer` (Q-44)."""

    def __init__(
        self,
        model: FreeTimeGaussians,
        cfg: OptimConfig,
        *,
        spatial_scale: float,
        frame_dt: float,
        adam: _AdamKind = "fused",
    ) -> None:
        for label, value in (("spatial_scale", spatial_scale), ("frame_dt", frame_dt)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                _fail(f"{label} must be a number, got {value!r}")
            if not (math.isfinite(value) and value > 0.0):
                _fail(f"{label} must be finite and > 0, got {value!r}")
        if adam not in _ADAM_KINDS:
            _fail(f"adam must be one of {_ADAM_KINDS}, got {adam!r}")
        self._model = model
        self._iterations = cfg.iterations
        self._betas = (float(cfg.betas[0]), float(cfg.betas[1]))
        self._eps = float(cfg.eps)
        self._groups = _groups(model, cfg, float(spatial_scale), float(frame_dt))
        param_groups: list[dict[str, Any]] = [
            {"params": [getattr(model, g.name)], "lr": g.eta0, "name": g.name} for g in self._groups
        ]
        self._adam = adam
        self._torch = self._make_torch_optimizer(param_groups)
        self._init_state()
        self.set_iteration(1)

    # -- construction ---------------------------------------------------------------------------

    def _make_torch_optimizer(self, param_groups: list[dict[str, Any]]) -> torch.optim.Adam:
        if self._adam == "foreach":
            return self._torch_adam(param_groups, fused=False)
        if self._adam == "fused":
            try:
                return self._torch_adam(param_groups, fused=True)
            except RuntimeError as error:
                # Q-45: D-53's "foreach as fallback" where torch refuses the device or dtype.
                warnings.warn(
                    f"fused Adam is unavailable ({error}); using foreach Adam", stacklevel=3
                )
                self._adam = "foreach"
                return self._torch_adam(param_groups, fused=False)
        # D-53: gsplat's SelectiveAdam, a CUDA kernel; imported here so that the module imports
        # on machines without CUDA.
        if any(group["params"][0].device.type != "cuda" for group in param_groups):
            _fail("selective Adam needs a model on a CUDA device (Q-45)")
        from gsplat.optimizers import SelectiveAdam

        optimizer: torch.optim.Adam = SelectiveAdam(param_groups, eps=self._eps, betas=self._betas)
        return optimizer

    def _torch_adam(self, param_groups: list[dict[str, Any]], *, fused: bool) -> torch.optim.Adam:
        # §1.3: β₁, β₂, ε from OptimConfig, no weight decay, dense updates.
        return torch.optim.Adam(
            param_groups,
            betas=self._betas,
            eps=self._eps,
            weight_decay=0.0,
            foreach=None if fused else True,
            fused=True if fused else None,
        )

    def _step_device(self, param: torch.Tensor) -> torch.device:
        # Q-45: where torch's own Adam puts the step counter: on the parameter's device when
        # fused, on the CPU otherwise.
        return param.device if self._adam == "fused" else torch.device("cpu")

    def _init_state(self) -> None:
        """Q-45: zero Adam moments and step counters at build, so that a moment reset, a resize or
        a state dict works before the first step; torch uses them as its own."""
        for group in self._torch.param_groups:
            param = group["params"][0]
            self._torch.state[param] = {
                "step": torch.zeros((), dtype=torch.float32, device=self._step_device(param)),
                "exp_avg": torch.zeros_like(param, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(param, memory_format=torch.preserve_format),
            }

    # -- schedules ------------------------------------------------------------------------------

    def set_iteration(self, it: int) -> dict[str, float]:
        """Set the learning rates of iteration `it` (E12, E13; u from D-41) and return them by
        tensor name, for the tensors that have a parameter group (Q-44, Q-45)."""
        if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= self._iterations:
            _fail(f"it must be an int in [1, {self._iterations}], got {it!r}")
        # Q-45: with a one-iteration run, u = 0.
        u = (it - 1) / (self._iterations - 1) if self._iterations > 1 else 0.0
        rates: dict[str, float] = {}
        for spec, group in zip(self._groups, self._torch.param_groups, strict=True):
            rate = spec.rate(u)
            group["lr"] = rate  # a Python float: no host sync (§2.4 rule 3)
            rates[spec.name] = rate
        return rates

    # -- steps ----------------------------------------------------------------------------------

    def step(self, *, visibility: Tensor | None = None) -> None:
        """One Adam step on every parameter group at the current rates (§1.3 step 6). With
        selective Adam, `visibility` is the bool mask [N] of the rows to update (Q-44)."""
        if self._adam != "selective":
            if visibility is not None:
                _fail(f"visibility is only for selective Adam, not {self._adam!r} (Q-44)")
            self._torch.step()
            return
        n = self._model.num_gaussians
        if visibility is None:
            _fail("selective Adam needs visibility, a bool mask of the rows to update (Q-44)")
        if visibility.dtype != torch.bool or tuple(visibility.shape) != (n,):
            _fail(
                f"visibility must be a bool mask of shape [{n}], "
                f"got {visibility.dtype} {list(visibility.shape)}"
            )
        if visibility.device != self._model.means.device:
            _fail(f"visibility must be on {self._model.means.device}, got {visibility.device}")
        self._torch.step(visibility)  # gsplat's SelectiveAdam
        # Q-45: gsplat's kernel never advances the step counter (D-53); count the steps taken.
        for group in self._torch.param_groups:
            param = group["params"][0]
            if param.grad is not None:
                self._torch.state[param]["step"] += 1

    def zero_grad(self) -> None:
        """Clear the gradients of all nine stored tensors, with or without a group (Q-45)."""
        for name in FreeTimeGaussians.PARAM_NAMES:
            getattr(self._model, name).grad = None

    # -- rows -----------------------------------------------------------------------------------

    def reset_rows(self, rows: Tensor) -> None:
        """Zero the Adam moments of `rows` in every group and keep the step counters (D-16).
        `rows` holds int64 indices on the model's device, in any order, repeats allowed (Q-44)."""
        if rows.dtype != torch.int64 or rows.dim() != 1:
            _fail(f"rows must be int64 indices of shape [k], got {rows.dtype} {list(rows.shape)}")
        if rows.device != self._model.means.device:
            _fail(f"rows must be on {self._model.means.device}, got {rows.device}")
        for group in self._torch.param_groups:
            state = self._torch.state[group["params"][0]]
            state["exp_avg"].index_fill_(0, rows, 0.0)
            state["exp_avg_sq"].index_fill_(0, rows, 0.0)

    def resize_(self, *, keep: Tensor | None = None, n_append: int = 0) -> None:
        """Follow a `FreeTimeGaussians.resize_` that kept the rows of the bool mask `keep` [N] and
        appended `n_append` rows: kept rows keep their moments in order, new rows start with zero
        moments (D-45), each group moves to the model's new tensor, and the step counters stay
        (§1.3 Growth step 4; Q-23, Q-44). Call it right after the model's resize_; on error
        nothing changes."""
        if isinstance(n_append, bool) or not isinstance(n_append, int) or n_append < 0:
            _fail(f"n_append must be an int ≥ 0, got {n_append!r}")
        states = [self._torch.state[group["params"][0]] for group in self._torch.param_groups]
        old_n = int(states[0]["exp_avg"].shape[0]) if states else self._model.num_gaussians
        device = self._model.means.device
        # Q-23: keep is a bool mask [N] on the model's device.
        if keep is not None:
            if keep.dtype != torch.bool or tuple(keep.shape) != (old_n,):
                _fail(
                    f"keep must be a bool mask of shape [{old_n}], "
                    f"got {keep.dtype} {list(keep.shape)}"
                )
            if keep.device != device:
                _fail(f"keep must be on the model's device {device}, got {keep.device}")
        new: list[tuple[Tensor, dict[str, Tensor]]] = []
        for spec, state in zip(self._groups, states, strict=True):
            param = getattr(self._model, spec.name)
            moments = {}
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key] if keep is None else state[key][keep]
                value = torch.cat([value, value.new_zeros((n_append, *value.shape[1:]))])
                if value.shape != param.shape:
                    _fail(
                        f"{spec.name}: the model has shape {list(param.shape)}, but keeping and "
                        f"appending rows gives {list(value.shape)}; call resize_ right after "
                        "the model's, with the same keep (Q-44)"
                    )
                moments[key] = value
            new.append((param, moments))
        for group, state, (param, moments) in zip(
            self._torch.param_groups, states, new, strict=True
        ):
            del self._torch.state[group["params"][0]]
            group["params"] = [param]
            self._torch.state[param] = {"step": state["step"], **moments}

    # -- state ----------------------------------------------------------------------------------

    def _settings(self, spec: _Group) -> dict[str, Any]:
        """Q-46: the JSON half of a group's state (optim.json, §2.2)."""
        return {
            "lr": [spec.eta0, spec.eta1],
            "schedule": spec.schedule,
            "iterations": self._iterations,
            "betas": list(self._betas),
            "eps": self._eps,
        }

    def state_dict(self) -> dict[str, Any]:
        """{tensor name: {exp_avg, exp_avg_sq, step, lr, schedule, iterations, betas, eps}} for
        the tensors with a group (§2.2 optim.safetensors and optim.json; Q-25, Q-46)."""
        out: dict[str, Any] = {}
        for spec, group in zip(self._groups, self._torch.param_groups, strict=True):
            state = self._torch.state[group["params"][0]]
            out[spec.name] = {
                "exp_avg": state["exp_avg"].detach().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
                "step": state["step"].detach().clone(),
                **self._settings(spec),
            }
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the Adam moments and step counters of `state_dict`'s layout, on any device;
        the base rates and settings must equal this optimizer's own (Q-46). On error nothing
        changes."""
        names = [spec.name for spec in self._groups]
        if not isinstance(state, dict) or set(state) != set(names):
            got = sorted(state) if isinstance(state, dict) else type(state).__name__
            _fail(f"optimizer state must hold exactly the groups {names}, got {got}")
        loaded: list[dict[str, Tensor]] = []
        for spec, group in zip(self._groups, self._torch.param_groups, strict=True):
            entry = state[spec.name]
            param = group["params"][0]
            if not isinstance(entry, dict):
                _fail(f"optimizer state for {spec.name} must be a dict")
            expected_keys = {"exp_avg", "exp_avg_sq", "step", *self._settings(spec)}
            if set(entry) != expected_keys:
                _fail(
                    f"optimizer state for {spec.name} must hold {sorted(expected_keys)}, "
                    f"got {sorted(entry)}"
                )
            for key, value in self._settings(spec).items():
                stored = entry[key]
                stored = list(stored) if isinstance(stored, tuple) else stored
                if stored != value:
                    _fail(f"{spec.name}: stored {key} {stored!r} differs from {value!r} (Q-46)")
            moments: dict[str, Tensor] = {}
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry[key]
                if not isinstance(value, Tensor) or value.dtype != param.dtype:
                    _fail(f"{spec.name}.{key} must be a {param.dtype} tensor")
                if value.shape != param.shape:
                    _fail(
                        f"{spec.name}.{key} must have shape {list(param.shape)}, "
                        f"got {list(value.shape)}"
                    )
                moments[key] = value.detach().to(param.device, copy=True)
            step = entry["step"]
            if not isinstance(step, Tensor) or step.dim() != 0 or step.dtype != torch.float32:
                _fail(f"{spec.name}.step must be a 0-d float32 tensor")
            moments["step"] = step.detach().to(self._step_device(param), copy=True)
            loaded.append(moments)
        for group, moments in zip(self._torch.param_groups, loaded, strict=True):
            self._torch.state[group["params"][0]] = moments


def build_optimizer(
    model: FreeTimeGaussians,
    cfg: OptimConfig,
    *,
    spatial_scale: float,
    frame_dt: float,
    adam: _AdamKind = "fused",  # Q-44: PerfConfig.adam (D-53)
) -> FTGSOptimizer:
    """Adam over the model's stored tensors with §1.3's settings and rates, the position rate
    scaled by S and the velocity rate by S/Δt_f (D-19, D-27), at iteration 1."""
    return FTGSOptimizer(model, cfg, spatial_scale=spatial_scale, frame_dt=frame_dt, adam=adam)
