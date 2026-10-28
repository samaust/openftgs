# SPDX-License-Identifier: Apache-2.0
"""Adam construction, learning-rate schedules, row-wise moment reset and optimizer state I/O.

Spec: E12, E13. Band: Mechanics (§2.1).
"""

import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

import torch
from torch import Tensor

from openftgs.config import OptimConfig
from openftgs.model import FreeTimeGaussians

__all__ = ["FTGSOptimizer", "build_optimizer"]  # Q-44

_ScheduleKind = Literal["exponential", "annealed", "constant"]
_AdamKind = Literal["fused", "foreach", "selective"]
_ADAM_KINDS = ("fused", "foreach", "selective")
_MOMENTS = ("exp_avg", "exp_avg_sq")


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


@dataclass(frozen=True)
class _GroupSchedule:
    """One parameter group's stored tensor, rates η₀, η₁ (scaled by S or S/Δt_f) and schedule."""

    name: str
    eta0: float
    eta1: float
    kind: _ScheduleKind

    def rate(self, u: float) -> float:
        if self.kind == "annealed":
            return float(self.eta0 ** (1.0 - u) * self.eta1**u)  # E12 (D-19)
        if self.kind == "exponential":
            # E13 as written; at u = 1 it can differ from η₁ by an ulp, within U-17's tolerance.
            return float(self.eta0 * (self.eta1 / self.eta0) ** u)
        return self.eta0


def _group_schedules(
    model: FreeTimeGaussians, cfg: OptimConfig, spatial_scale: float, frame_dt: float
) -> list[_GroupSchedule]:
    """Build §1.3's rate table in §1.1's tensor order, without the tensors that take no updates."""
    s, v_scale = spatial_scale, spatial_scale / frame_dt
    schedules = [
        _GroupSchedule("means", cfg.means_lr[0] * s, cfg.means_lr[1] * s, "exponential"),  # D-27
        _GroupSchedule("times", cfg.times_lr[0], cfg.times_lr[1], "exponential"),  # D-20
        _GroupSchedule(
            "log_durations", cfg.log_durations_lr, cfg.log_durations_lr, "constant"
        ),  # D-20
        _GroupSchedule(
            "velocities", cfg.velocities_lr[0] * v_scale, cfg.velocities_lr[1] * v_scale, "annealed"
        ),  # D-19
        _GroupSchedule("log_scales", cfg.log_scales_lr, cfg.log_scales_lr, "constant"),
        _GroupSchedule("quats", cfg.quats_lr, cfg.quats_lr, "constant"),
        _GroupSchedule("opacity_logits", cfg.opacity_logits_lr, cfg.opacity_logits_lr, "constant"),
        _GroupSchedule("sh0", cfg.sh0_lr, cfg.sh0_lr, "constant"),
        _GroupSchedule("shN", cfg.shN_lr, cfg.shN_lr, "constant"),
    ]
    frozen: set[str] = set()
    if model.sh_degree == 0:
        frozen.add("shN")  # U-26: shN is [N, 0, 3] and has no group
    if model.time.tau0 == model.time.tau1:
        frozen.update(("times", "velocities"))  # Q-45: D-71 by "no parameter group"
    return [g for g in schedules if g.name not in frozen]


def _fused_supported(params: Sequence[Tensor]) -> bool:
    """Check whether PyTorch's fused Adam takes every (device, dtype) of `params`.

    A one-element probe step per pair, since torch checks support only at a step that creates the
    state (Q-45).
    """
    for device, dtype in {(p.device, p.dtype) for p in params}:
        probe = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
        probe.grad = torch.zeros_like(probe)
        try:
            torch.optim.Adam([probe], fused=True).step()
        except RuntimeError:
            return False
    return True


def _check_row_mask(mask: Tensor, label: str, n: int, device: torch.device) -> None:
    """Check that `mask` is a bool mask [n] on `device` (Q-23, Q-44).

    Only metadata is read, so nothing syncs.
    """
    if mask.dtype != torch.bool or tuple(mask.shape) != (n,):
        _fail(f"{label} must be a bool mask of shape [{n}], got {mask.dtype} {list(mask.shape)}")
    if mask.device != device:
        _fail(f"{label} must be on the model's device {device}, got {mask.device}")


class FTGSOptimizer:
    """Adam over a model's stored tensors, with the learning-rate schedules of E12 and E13.

    One parameter group per stored tensor of a :class:`~openftgs.model.FreeTimeGaussians`, in
    §1.1's tensor order, with Adam's β₁, β₂ and ε from the configuration, no weight decay and
    dense updates (§1.3, "Optimizer and learning rates"), except that gsplat's selective Adam
    (D-53) updates only the visible rows. A tensor that takes no updates has no
    group: ``shN`` at SH degree 0 (U-26), and ``times`` and ``velocities`` when the training set
    has a single timestamp (D-71, Q-45). Build it with :func:`build_optimizer`; the rates of
    iteration 1 are set from the start.

    Parameters
    ----------
    model
        The model whose stored tensors the parameter groups update.
    cfg
        Adam's β₁, β₂ and ε, the run's iteration count and each tensor's base rates η₀ and η₁
        (§1.3).
    spatial_scale
        The scene's spatial scale S (D-27), in world units, finite and > 0. The rates of
        ``means`` are multiplied by S, those of ``velocities`` by S/Δt_f (D-19).
    frame_dt
        The frame interval Δt_f in normalized time (E14), finite and > 0.
    adam
        Which Adam takes the steps (D-53): "fused", PyTorch's fused kernel; "foreach",
        PyTorch's multi-tensor implementation; or "selective", gsplat's SelectiveAdam, which
        updates only the visible rows and needs a model on a CUDA device.

    Raises
    ------
    ValueError
        If ``spatial_scale`` or ``frame_dt`` is not a finite number > 0, ``adam`` is not one of
        the three kinds, or ``adam`` is "selective" and the model is not on a CUDA device.

    Warns
    -----
    UserWarning
        When ``adam`` is "fused" and PyTorch's fused Adam doesn't take the model's device or
        dtype; foreach Adam runs instead (D-53's fallback, Q-45).

    Notes
    -----
    §2.1 lists the class and its methods; Q-44 sets its constructor's arguments, those of
    :func:`build_optimizer`, ``adam`` among them. The Adam state, moments and step counters, is
    created at construction (Q-45), so that a moment reset, a resize or :meth:`state_dict` works
    before the first step.
    """

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
        self._schedules = _group_schedules(model, cfg, float(spatial_scale), float(frame_dt))
        params = [getattr(model, g.name) for g in self._schedules]
        self._adam = self._resolve_kind(adam, params)
        self._adam_impl = self._make_adam(params)
        self._init_state()
        self.set_iteration(1)

    # -- construction ---------------------------------------------------------------------------

    # Q-48 (owner answer): public, so that density can check it was given the model's optimizer.
    @property
    def model(self) -> FreeTimeGaussians:
        """The model this optimizer was built for, whose stored tensors its groups update (Q-48)."""
        return self._model

    @property
    def _device(self) -> torch.device:
        return self._model.means.device

    @staticmethod
    def _resolve_kind(adam: _AdamKind, params: Sequence[Tensor]) -> _AdamKind:
        if adam == "fused" and not _fused_supported(params):
            # Q-45: D-53's "foreach as fallback" where torch refuses the device or dtype.
            warnings.warn("fused Adam is unavailable here; using foreach Adam", stacklevel=4)
            return "foreach"
        if adam == "selective" and any(p.device.type != "cuda" for p in params):
            _fail("selective Adam needs a model on a CUDA device (Q-45)")
        return adam

    def _make_adam(self, params: Sequence[Tensor]) -> torch.optim.Adam:
        param_groups: list[dict[str, Any]] = [
            {"params": [p], "lr": g.eta0, "name": g.name}
            for g, p in zip(self._schedules, params, strict=True)
        ]
        if self._adam == "selective":
            # D-53: gsplat's SelectiveAdam, a CUDA kernel; imported here so that the module
            # imports on machines without CUDA.
            from gsplat.optimizers import SelectiveAdam

            selective: torch.optim.Adam = SelectiveAdam(
                param_groups, eps=self._eps, betas=self._betas
            )
            return selective
        # §1.3: β₁, β₂, ε from OptimConfig, no weight decay, dense updates.
        fused = self._adam == "fused"
        return torch.optim.Adam(
            param_groups,
            betas=self._betas,
            eps=self._eps,
            weight_decay=0.0,
            foreach=None if fused else True,
            fused=True if fused else None,
        )

    def _groups(self) -> Iterator[tuple[_GroupSchedule, Tensor, dict[str, Tensor]]]:
        """Yield each group's schedule, parameter and Adam state, in group order."""
        for schedule, group in zip(self._schedules, self._adam_impl.param_groups, strict=True):
            param = group["params"][0]
            yield schedule, param, self._adam_impl.state[param]

    def _step_device(self, param: Tensor) -> torch.device:
        # Q-45: where torch's own Adam puts the step counter: on the parameter's device when
        # fused, on the CPU otherwise.
        return param.device if self._adam == "fused" else torch.device("cpu")

    def _init_state(self) -> None:
        """Zero the Adam moments and step counters at build (Q-45).

        So a moment reset, a resize or a state dict works before the first step. The keys, the
        0-d float32 step counter and its placement follow the state torch 2.14.1's Adam creates
        lazily (`_init_group` in torch/optim/adam.py, BSD-3-Clause; read, not copied: source-log
        row 50), so torch uses this state as its own.
        """
        for group in self._adam_impl.param_groups:
            param = group["params"][0]
            self._adam_impl.state[param] = {
                "step": torch.zeros((), dtype=torch.float32, device=self._step_device(param)),
                **{
                    key: torch.zeros_like(param, memory_format=torch.preserve_format)
                    for key in _MOMENTS
                },
            }

    # -- learning-rate schedules ----------------------------------------------------------------

    def set_iteration(self, it: int) -> dict[str, float]:
        """Set the learning rates of iteration ``it`` and return them (E12, E13).

        With u = (it − 1)/(T_it − 1) in [0, 1] (D-41), ``means`` and ``times`` follow E13's
        exponential decay η₀·(η₁/η₀)^u, ``velocities`` E12's annealing η₀^(1−u)·η₁^u, and the
        other tensors keep their constant rate (§1.3). A run of one iteration uses u = 0
        (Q-45). The rates are Python floats, so nothing synchronizes with the device.

        Parameters
        ----------
        it
            The iteration, counted from 1, at most the run's iteration count (D-41).

        Returns
        -------
        dict of str to float
            The rate of each tensor that has a parameter group, by tensor name, in group order.

        Raises
        ------
        ValueError
            If ``it`` is not an int from 1 to the run's iteration count.

        Notes
        -----
        The return value, by tensor name and only for the tensors with a group, follows Q-44 and
        Q-45.
        """
        if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= self._iterations:
            _fail(f"it must be an int in [1, {self._iterations}], got {it!r}")
        # Q-45: with a one-iteration run, u = 0.
        u = (it - 1) / (self._iterations - 1) if self._iterations > 1 else 0.0
        rates: dict[str, float] = {}
        for schedule, group in zip(self._schedules, self._adam_impl.param_groups, strict=True):
            rate = schedule.rate(u)
            group["lr"] = rate  # a Python float: no host sync (§2.4 rule 3)
            rates[schedule.name] = rate
        return rates

    # -- steps ----------------------------------------------------------------------------------

    # Q-44: visibility is a keyword-only addition to §2.1's step() (owner answer).
    def step(self, *, visibility: Tensor | None = None) -> None:
        """Take one Adam step on every parameter group at the current rates (§1.3 step 6).

        With selective Adam, only the rows in ``visibility`` are updated, and the step counter of
        every group whose tensor has a gradient advances by one here, since gsplat's kernel
        doesn't count steps (D-53, Q-45).

        Parameters
        ----------
        visibility
            Selective Adam only: the rows to update, a bool mask [N] on the model's device.
            None, the default, for fused and foreach Adam.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``visibility`` is given to fused or foreach Adam, is missing for selective Adam,
            or is not a bool mask [N] on the model's device.

        Notes
        -----
        ``visibility`` is a keyword-only addition to §2.1's ``step()`` (Q-44, owner answer).
        """
        if self._adam != "selective":
            if visibility is not None:
                _fail(f"visibility is only for selective Adam, not {self._adam!r}")
            self._adam_impl.step()
            return
        if visibility is None:
            _fail("selective Adam needs visibility, a bool mask of the rows to update")
        _check_row_mask(visibility, "visibility", self._model.num_gaussians, self._device)
        self._adam_impl.step(visibility)  # gsplat's SelectiveAdam
        # Q-45: gsplat's kernel never advances the step counter (D-53); count the steps taken.
        for _, param, state in self._groups():
            if param.grad is not None:
                state["step"] += 1

    def zero_grad(self) -> None:
        """Clear the gradients of all nine stored tensors (Q-45).

        Each tensor's gradient is set to None, whether or not the tensor has a parameter group.
        """
        for name in FreeTimeGaussians.PARAM_NAMES:
            getattr(self._model, name).grad = None

    # -- rows -----------------------------------------------------------------------------------

    # Q-44: rows are int64 indices in [0, N), in any order, repeats allowed (relocation draws
    # targets with replacement); bool masks are refused, as Q-23 keeps them for row selection.
    # The range is the caller's precondition: checking values would synchronize (§2.4 rule 3),
    # and an index outside it fails inside index_fill_ (on CUDA, as a device-side assert).
    # Q-53 (owner answer): names, a keyword-only addition, limits the reset to those groups.
    def reset_rows(self, rows: Tensor, *, names: Sequence[str] | None = None) -> None:
        """Zero the Adam moments of ``rows`` and keep the step counters (D-16).

        ``exp_avg`` and ``exp_avg_sq`` of the given rows are set to 0 in every parameter group,
        or only in the groups of the tensors named in ``names``. Relocation calls it on its
        targets and moved rows (§1.3 Relocation, step 6). Only metadata is checked, so nothing
        synchronizes (§2.4 rule 3).

        Parameters
        ----------
        rows
            Row indices, [k] int64 in [0, N), on the model's device, in any order, repeats
            allowed. Bool masks are refused (Q-23). The range is not checked: an index outside
            it fails inside PyTorch, on CUDA as a device-side assert.
        names
            The names of the tensors whose groups are reset, each with a parameter group; None,
            the default, resets every group.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``rows`` is not a 1-D int64 tensor on the model's device, or ``names`` is a
            string, holds a non-string or names a tensor without a parameter group.

        Notes
        -----
        Q-44 fixes the format of ``rows``; ``names`` is a keyword-only addition (Q-53, owner
        answer).
        """
        if rows.dtype != torch.int64 or rows.dim() != 1:
            _fail(f"rows must be int64 indices of shape [k], got {rows.dtype} {list(rows.shape)}")
        if rows.device != self._device:
            _fail(f"rows must be on the model's device {self._device}, got {rows.device}")
        grouped = [schedule.name for schedule in self._schedules]
        if names is not None:
            if isinstance(names, str) or not all(isinstance(n, str) for n in names):
                _fail(f"names must be a sequence of tensor names, got {names!r}")
            unknown = sorted(set(names) - set(grouped))
            if unknown:
                _fail(f"names must name tensors with a parameter group {grouped}, got {unknown}")
        for schedule, _, state in self._groups():
            if names is not None and schedule.name not in names:
                continue
            for key in _MOMENTS:
                state[key].index_fill_(0, rows, 0.0)

    # Q-44: called right after FreeTimeGaussians.resize_, with the same keep (Q-23).
    def resize_(self, *, keep: Tensor | None = None, n_append: int = 0) -> None:
        """Follow a model resize: keep the rows of ``keep``, append zero-moment rows (D-45).

        Kept rows keep their moments, in order; the ``n_append`` new rows go last with zero
        moments; each group moves to the model's new tensor; and the step counters stay (§1.3
        Growth, step 4). Call it right after the model's ``resize_``, with the same ``keep``. On
        error nothing changes.

        Parameters
        ----------
        keep
            The rows to keep, a bool mask [N] over the rows before the resize, on the model's
            device (Q-23); None, the default, keeps every row.
        n_append
            The number of rows the model appended after the kept ones, an int ≥ 0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``n_append`` is not an int ≥ 0, ``keep`` is not a bool mask [N] on the model's
            device, or the kept and appended rows don't give the model's new shape.

        Notes
        -----
        With ``keep``, the gather synchronizes with the host; resizes happen only at growth
        events, which synchronize anyway (§2.4 rule 3). The call order is Q-44's.
        """
        if isinstance(n_append, bool) or not isinstance(n_append, int) or n_append < 0:
            _fail(f"n_append must be an int ≥ 0, got {n_append!r}")
        groups = list(self._groups())
        old_n = int(groups[0][2]["exp_avg"].shape[0]) if groups else self._model.num_gaussians
        if keep is not None:
            _check_row_mask(keep, "keep", old_n, self._device)
        resized: list[tuple[Tensor, dict[str, Tensor]]] = []
        for schedule, _, state in groups:
            param = getattr(self._model, schedule.name)
            new_state = {"step": state["step"]}
            for key in _MOMENTS:
                kept = state[key] if keep is None else state[key][keep]
                value = torch.cat([kept, kept.new_zeros((n_append, *kept.shape[1:]))])
                if value.shape != param.shape:
                    _fail(
                        f"{schedule.name}: the model has shape {list(param.shape)}, but keeping "
                        f"and appending rows gives {list(value.shape)}; call resize_ right after "
                        "the model's, with the same keep"
                    )
                new_state[key] = value
            resized.append((param, new_state))
        for group, (param, new_state) in zip(self._adam_impl.param_groups, resized, strict=True):
            del self._adam_impl.state[group["params"][0]]
            group["params"] = [param]
            self._adam_impl.state[param] = new_state

    # -- state ----------------------------------------------------------------------------------

    def _settings(self, schedule: _GroupSchedule) -> dict[str, Any]:
        # Q-46: the JSON half of a group's state (optim.json, §2.2).
        return {
            "lr": [schedule.eta0, schedule.eta1],
            "schedule": schedule.kind,
            "iterations": self._iterations,
            "betas": list(self._betas),
            "eps": self._eps,
        }

    # Q-46: {tensor name: {exp_avg, exp_avg_sq, step, lr, schedule, iterations, betas, eps}}.
    def state_dict(self) -> dict[str, Any]:
        """Return the Adam state of every parameter group, in §2.2's layout (Q-46).

        The layout is that of §2.2's ``optim.safetensors`` and ``optim.json`` (Q-25), keyed by
        tensor name; its tensors are detached clones.

        Returns
        -------
        dict
            ``{tensor name: {"exp_avg", "exp_avg_sq", "step", "lr", "schedule", "iterations",
            "betas", "eps"}}`` for each tensor with a parameter group: the two Adam moments,
            shaped and typed like the tensor; the step counter, a 0-d float32 tensor; the rates
            ``[η₀, η₁]`` after scaling, ``means``' by S and ``velocities``' by S/Δt_f (Q-46);
            the schedule, "exponential", "annealed" or "constant"; the run's
            iteration count; Adam's ``[β₁, β₂]``; and ε.
        """
        return {
            schedule.name: {
                **{key: state[key].detach().clone() for key in ("step", *_MOMENTS)},
                **self._settings(schedule),
            }
            for schedule, _, state in self._groups()
        }

    # Q-46: the scaled rates and settings must equal this optimizer's own; step counters are
    # placed as the current Adam kind needs, so state saved under one kind loads under another.
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the Adam moments and step counters from :meth:`state_dict`'s layout (Q-46).

        The state may come from any device: the moments move to their tensor's device and the
        step counters to where the current Adam kind keeps them, so state saved under one kind
        loads under another. On error nothing changes.

        Parameters
        ----------
        state
            A state in :meth:`state_dict`'s layout, holding exactly this optimizer's groups; its
            rates (scaled by S and Δt_f, Q-46), schedules, iteration count, betas and ε must
            equal this optimizer's own.

        Raises
        ------
        ValueError
            If ``state`` doesn't hold exactly this optimizer's groups and keys, a setting
            differs from this optimizer's, a moment is not a tensor of its tensor's dtype and
            shape, or a step counter is not a 0-d float32 tensor.
        """
        names = [schedule.name for schedule in self._schedules]
        if not isinstance(state, dict) or set(state) != set(names):
            got = sorted(state) if isinstance(state, dict) else type(state).__name__
            _fail(f"optimizer state must hold exactly the groups {names}, got {got}")
        restored: list[dict[str, Tensor]] = []
        for schedule, param, _ in self._groups():
            entry = state[schedule.name]
            if not isinstance(entry, dict):
                _fail(f"optimizer state for {schedule.name} must be a dict")
            settings = self._settings(schedule)
            expected_keys = {"step", *_MOMENTS, *settings}
            if set(entry) != expected_keys:
                _fail(
                    f"optimizer state for {schedule.name} must hold {sorted(expected_keys)}, "
                    f"got {sorted(entry)}"
                )
            for key, value in settings.items():
                stored = entry[key]
                stored = list(stored) if isinstance(stored, tuple) else stored
                if stored != value:
                    _fail(f"{schedule.name}: stored {key} {stored!r} differs from {value!r}")
            group_state: dict[str, Tensor] = {}
            for key in _MOMENTS:
                value = entry[key]
                if not isinstance(value, Tensor) or value.dtype != param.dtype:
                    _fail(f"{schedule.name}.{key} must be a {param.dtype} tensor")
                if value.shape != param.shape:
                    _fail(
                        f"{schedule.name}.{key} must have shape {list(param.shape)}, "
                        f"got {list(value.shape)}"
                    )
                group_state[key] = value.detach().to(param.device, copy=True)
            step = entry["step"]
            if not isinstance(step, Tensor) or step.dim() != 0 or step.dtype != torch.float32:
                _fail(f"{schedule.name}.step must be a 0-d float32 tensor")
            group_state["step"] = step.detach().to(self._step_device(param), copy=True)
            restored.append(group_state)
        for group, group_state in zip(self._adam_impl.param_groups, restored, strict=True):
            self._adam_impl.state[group["params"][0]] = group_state


def build_optimizer(
    model: FreeTimeGaussians,
    cfg: OptimConfig,
    *,
    spatial_scale: float,
    frame_dt: float,
    adam: _AdamKind = "fused",  # Q-44: PerfConfig.adam (D-53), a keyword-only addition
) -> FTGSOptimizer:
    """Build Adam over the model's stored tensors with §1.3's settings and rates, at iteration 1.

    Every stored tensor that takes updates gets a parameter group with its rates from ``cfg``:
    those of ``means`` multiplied by S (D-27), those of ``velocities`` by S/Δt_f (D-19). See
    :class:`FTGSOptimizer` for the groups and the Adam kinds.

    Parameters
    ----------
    model
        The model whose stored tensors the parameter groups update.
    cfg
        Adam's β₁, β₂ and ε, the run's iteration count and each tensor's base rates η₀ and η₁
        (§1.3).
    spatial_scale
        The scene's spatial scale S (D-27), in world units, finite and > 0.
    frame_dt
        The frame interval Δt_f in normalized time (E14), finite and > 0.
    adam
        Which Adam takes the steps (D-53): "fused" (the default), "foreach" or "selective".

    Returns
    -------
    FTGSOptimizer
        The optimizer, with zero moments and the rates of iteration 1 set.

    Raises
    ------
    ValueError
        If ``spatial_scale`` or ``frame_dt`` is not a finite number > 0, ``adam`` is not one of
        the three kinds, or ``adam`` is "selective" and the model is not on a CUDA device.

    Warns
    -----
    UserWarning
        When ``adam`` is "fused" and PyTorch's fused Adam doesn't take the model's device or
        dtype; foreach Adam runs instead (Q-45).

    Notes
    -----
    ``adam`` is ``PerfConfig.adam`` (D-53), a keyword-only addition to §2.1's signature (Q-44).

    Examples
    --------
    One iteration's update, after the loss's backward pass (§1.3, step 6):

    >>> from pathlib import Path
    >>> from openftgs.config import OptimConfig
    >>> from openftgs.model import FreeTimeGaussians
    >>> from openftgs.optim import build_optimizer
    >>> model = FreeTimeGaussians.load(Path("run/checkpoint"))
    >>> optimizer = build_optimizer(model, OptimConfig(), spatial_scale=2.0, frame_dt=1 / 299)
    >>> it = 1
    >>> rates = optimizer.set_iteration(it)
    >>> optimizer.step()
    >>> optimizer.zero_grad()
    """
    return FTGSOptimizer(model, cfg, spatial_scale=spatial_scale, frame_dt=frame_dt, adam=adam)
