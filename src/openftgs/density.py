# SPDX-License-Identifier: Apache-2.0
"""Gradient statistics, growth (clone and split), the relocation score, the dead set, relocation,
and the count-control modes.

Spec: E10, E11, E17, §1.3. Band: Mechanics (§2.1).
"""

import math
from collections.abc import Mapping
from typing import NoReturn

import numpy as np
import torch
from torch import Tensor

from openftgs.config import RelocConfig
from openftgs.model import FreeTimeGaussians

__all__ = ["GradStats", "sampling_score"]  # Q-48


def _fail(reason: str) -> NoReturn:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(reason)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# Q-50: every logit bound in this module (ô_dead of D-14, the ablation's thresholds) is float32
# of the float64 logit; comparing a float32 tensor with this Python float compares in float32.
def _logit_bound(p: float) -> float:
    return float(np.float32(math.log(p) - math.log1p(-p)))


def _check_stats(model: FreeTimeGaussians, stats: "GradStats") -> None:
    """The statistics cover the model's rows, on its device (shapes only: no host sync)."""
    if stats.num_gaussians != model.num_gaussians:
        _fail(f"the statistics hold {stats.num_gaussians} rows, the model {model.num_gaussians}")
    if stats.g_sum.device != model.means.device:
        _fail(f"the statistics are on {stats.g_sum.device}, the model on {model.means.device}")


class GradStats:
    """The gradient statistics g_sum and count of E10, one entry per primitive (Q-48)."""

    def __init__(self, num_gaussians: int, *, device: torch.device | str = "cpu") -> None:
        if not _is_int(num_gaussians) or num_gaussians < 0:
            _fail(f"num_gaussians must be an int ≥ 0, got {num_gaussians!r}")
        self._g_sum = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
        self._count = torch.zeros(num_gaussians, dtype=torch.int32, device=device)

    @property
    def num_gaussians(self) -> int:
        return int(self._g_sum.shape[0])

    @property
    def g_sum(self) -> Tensor:
        """[N] float32: the sum of NDC-scaled 2D-mean gradient norms (E10, D-12). Read-only."""
        return self._g_sum

    @property
    def count(self) -> Tensor:
        """[N] int32: the number of visible camera–primitive pairs summed into g_sum (Q-49)."""
        return self._count

    # Q-49: visible in unpacked mode, gaussian_ids in packed mode; no host synchronization.
    def update(
        self,
        means2d_grad: Tensor,
        *,
        width: int,
        height: int,
        visible: Tensor | None = None,
        gaussian_ids: Tensor | None = None,
    ) -> None:
        """Add the NDC-scaled norm of each visible primitive's 2D-mean gradient to g_sum and one
        to count (§1.3 step 5, D-12). Unpacked mode: `means2d_grad` [C, N, 2] with `visible`
        [C, N]; packed mode: `means2d_grad` [nnz, 2] with `gaussian_ids` [nnz] (Q-49). Checks
        shapes, dtypes and devices only, so nothing synchronizes (D-49)."""
        self._check_update(means2d_grad, width, height, visible, gaussian_ids)
        # D-12: x scaled by W/2 and y by H/2; Python floats, so nothing is built on the host.
        norm = torch.hypot(means2d_grad[..., 0] * (width / 2), means2d_grad[..., 1] * (height / 2))
        norm = norm.to(torch.float32)
        if gaussian_ids is not None:
            # Packed: every row is a visible camera–primitive pair (D-75, D-76); scatter-adds.
            self._g_sum.index_add_(0, gaussian_ids, norm)
            self._count.index_add_(
                0, gaussian_ids, torch.ones_like(gaussian_ids, dtype=torch.int32)
            )
            return
        assert visible is not None  # _check_update
        # Unpacked: masked sums over the cameras (§2.4 stage table).
        self._g_sum += torch.where(visible, norm, 0.0).sum(0)
        self._count += visible.sum(0, dtype=torch.int32)

    def _check_update(
        self,
        grad: Tensor,
        width: object,
        height: object,
        visible: Tensor | None,
        gaussian_ids: Tensor | None,
    ) -> None:
        n = self.num_gaussians
        device = self._g_sum.device
        for label, value in (("width", width), ("height", height)):
            if not _is_int(value) or value < 1:  # type: ignore[operator]
                _fail(f"{label} must be an int ≥ 1, got {value!r}")
        if not isinstance(grad, Tensor) or not grad.is_floating_point():
            _fail("means2d_grad must be a floating-point tensor")
        if (visible is None) == (gaussian_ids is None):
            _fail("give exactly one of visible (unpacked mode) and gaussian_ids (packed mode)")
        if gaussian_ids is not None:
            if grad.dim() != 2 or grad.shape[-1] != 2:
                _fail(f"packed means2d_grad must be [nnz, 2], got {list(grad.shape)}")
            if gaussian_ids.dtype != torch.int64 or tuple(gaussian_ids.shape) != (grad.shape[0],):
                _fail(
                    f"gaussian_ids must be int64 of shape [{grad.shape[0]}], got "
                    f"{gaussian_ids.dtype} {list(gaussian_ids.shape)}"
                )
            selector = gaussian_ids
        else:
            assert visible is not None
            if grad.dim() != 3 or grad.shape[1:] != (n, 2):
                _fail(f"unpacked means2d_grad must be [C, {n}, 2], got {list(grad.shape)}")
            if visible.dtype != torch.bool or visible.shape != grad.shape[:2]:
                _fail(
                    f"visible must be a bool mask of shape {list(grad.shape[:2])}, got "
                    f"{visible.dtype} {list(visible.shape)}"
                )
            selector = visible
        for label, value in (("means2d_grad", grad), ("the selector", selector)):
            if value.device != device:
                _fail(f"{label} must be on the statistics' device {device}, got {value.device}")

    def mean(self) -> Tensor:
        """[N] float32: g of E10, g_sum/count where count > 0 and 0 elsewhere."""
        return torch.where(self._count > 0, self._g_sum / self._count.clamp_min(1), 0.0)

    def reset(self) -> None:
        """Zero g_sum and count (§1.3 step 7)."""
        self._g_sum.zero_()
        self._count.zero_()

    def state_dict(self) -> dict[str, Tensor]:
        """{"g_sum", "count"} as detached clones: TrainerState.stats (§2.1, §2.2; Q-48)."""
        return {"g_sum": self._g_sum.detach().clone(), "count": self._count.detach().clone()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore `state_dict`'s layout from any device; on error nothing changes (Q-48)."""
        n = self.num_gaussians
        if not isinstance(state, Mapping) or set(state) != {"g_sum", "count"}:
            got = sorted(state) if isinstance(state, Mapping) else type(state).__name__
            _fail(f"statistics must hold exactly g_sum and count, got {got}")
        restored = []
        for key, dtype in (("g_sum", torch.float32), ("count", torch.int32)):
            value = state[key]
            if not isinstance(value, Tensor) or value.dtype != dtype or tuple(value.shape) != (n,):
                _fail(f"{key} must be a {dtype} tensor of shape [{n}]")
            restored.append(value.detach().to(self._g_sum.device, copy=True))
        self._g_sum, self._count = restored

    # Q-23: keep is a bool mask; Q-49: append_from indexes the rows before the resize and may
    # repeat a parent (a split's two children); its range is the caller's precondition (Q-44).
    def resize_(self, *, keep: Tensor | None = None, append_from: Tensor | None = None) -> None:
        """Keep the rows of the bool mask `keep` [N] in order, then append rows that copy the
        statistics of the rows `append_from` names (§1.3 Growth step 4, D-45). On error nothing
        changes."""
        n = self.num_gaussians
        device = self._g_sum.device
        if keep is not None:
            if keep.dtype != torch.bool or tuple(keep.shape) != (n,):
                _fail(
                    f"keep must be a bool mask of shape [{n}], got {keep.dtype} {list(keep.shape)}"
                )
            if keep.device != device:
                _fail(f"keep must be on the statistics' device {device}, got {keep.device}")
        if append_from is not None:
            if append_from.dtype != torch.int64 or append_from.dim() != 1:
                _fail(
                    "append_from must be int64 indices of shape [k], got "
                    f"{append_from.dtype} {list(append_from.shape)}"
                )
            if append_from.device != device:
                _fail(
                    f"append_from must be on the statistics' device {device}, "
                    f"got {append_from.device}"
                )
        resized = []
        for value in (self._g_sum, self._count):
            parents = None if append_from is None else value.index_select(0, append_from)
            kept = value if keep is None else value[keep]
            resized.append(kept if parents is None else torch.cat([kept, parents]))
        self._g_sum, self._count = resized


# Q-48: ρ for every row; the procedures mask out the dead set.
def sampling_score(model: FreeTimeGaussians, stats: GradStats, cfg: RelocConfig) -> Tensor:
    """ρ = λ_g·ĝ + λ_o·σ of E10 for every row, [N] float32. ĝ = min(1, g/Q) with Q the
    `cfg.grad_quantile` quantile of g over the live rows visible at least once (D-13), and
    ĝ ≡ 0 when Q is 0 or there is no such row. No host synchronization."""
    _check_stats(model, stats)
    logits = model.opacity_logits.detach()
    sigma = torch.sigmoid(logits)
    if model.num_gaussians == 0:
        return sigma
    g = stats.mean()
    live_seen = (logits >= _logit_bound(cfg.dead_opacity)) & (stats.count > 0)  # D-14
    # The quantile of the selected rows without a gather (no data-dependent shape): the other
    # rows are NaN, which nanquantile ignores; torch's default linear interpolation (D-13).
    q = torch.nanquantile(torch.where(live_seen, g, math.nan), cfg.grad_quantile)
    positive = q > 0  # False for a NaN quantile (no live, visible row): ĝ ≡ 0 (E10)
    g_hat = torch.where(positive, torch.clamp(g / torch.where(positive, q, 1.0), max=1.0), 0.0)
    return cfg.lambda_g * g_hat + cfg.lambda_o * sigma
