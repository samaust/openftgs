# SPDX-License-Identifier: Apache-2.0
"""Gradient statistics, growth, the sampling score, the dead set, relocation and count control.

Growth clones and splits (E17); relocation moves dead primitives onto live targets (E11); the
count-control modes, and the ablation's pruning and opacity reset, follow §1.3.

Spec: E10, E11, E17, §1.3. Band: Mechanics (§2.1).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from openftgs.config import GrowthConfig, RelocConfig
from openftgs.model import FreeTimeGaussians, rotation_matrix
from openftgs.optim import FTGSOptimizer

__all__ = [  # Q-48; Q-67; Q-204
    "OPACITY_RESET_EVERY",
    "GradStats",
    "GrowthReport",
    "RelocationReport",
    "draw_targets",
    "grow",
    "growth_masks",
    "mcmc_split",
    "prune",
    "relocate",
    "reset_opacity",
    "sampling_score",
]

_SPLIT_DIVISOR = 1.6  # E17: children's scales a/1.6


@dataclass(frozen=True)
class GrowthReport:
    """What a growth event did (Q-48).

    Returned by :func:`grow`.

    Parameters
    ----------
    cloned
        The number of candidates cloned ("gradient" mode).
    split
        The number of candidates split, each into two children (E17; "gradient" mode).
    sampled
        The number of rows appended by E11 ("score" mode).
    capped
        The number of candidates left out by the cap N_max (D-44).
    num_gaussians
        The number of primitives N after the event.
    """

    cloned: int
    split: int
    sampled: int
    capped: int
    num_gaussians: int


@dataclass(frozen=True)
class RelocationReport:
    """What a relocation did (Q-48).

    Returned by :func:`relocate`; both counts are 0 when nothing moved.

    Parameters
    ----------
    dead
        The number of dead primitives moved.
    targets
        The number of distinct live targets they were moved onto.
    """

    dead: int
    targets: int


def _fail(reason: str) -> NoReturn:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(reason)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# Q-50: every logit bound in this module (ô_dead of D-14, the ablation's thresholds) is float32
# of the float64 logit; comparing a float32 tensor with this Python float compares in float32.
def _logit_bound(p: float) -> float:
    return float(torch.tensor(math.log(p) - math.log1p(-p), dtype=torch.float32))


def _live(model: FreeTimeGaussians, cfg: RelocConfig) -> Tensor:
    """Return the live set, ô ≥ ô_dead, as a bool mask [N] (D-14, in float32).

    The dead set is the rest.
    """
    return model.opacity_logits.detach() >= _logit_bound(cfg.dead_opacity)


def _check_stats(model: FreeTimeGaussians, stats: "GradStats") -> None:
    """Check that the statistics cover the model's rows, on its device.

    Shapes only: no host sync.
    """
    if stats.num_gaussians != model.num_gaussians:
        _fail(f"the statistics hold {stats.num_gaussians} rows, the model {model.num_gaussians}")
    if stats.g_sum.device != model.means.device:
        _fail(f"the statistics are on {stats.g_sum.device}, the model on {model.means.device}")


# Q-48 (owner answer): an identity check, so another model's optimizer is caught even when it
# has as many rows, before anything changes; it reads no tensor values (D-49).
def _check_optimizer(model: FreeTimeGaussians, optimizer: FTGSOptimizer) -> None:
    if not isinstance(optimizer, FTGSOptimizer) or optimizer.model is not model:
        _fail("the optimizer must be the model's own, built for it by build_optimizer")


class GradStats:
    """The gradient statistics g_sum and count of E10, one entry per primitive (Q-48).

    At every iteration, each primitive visible in the image adds the NDC-scaled norm of its
    2D-mean gradient to ``g_sum`` and one to ``count`` (§1.3 step 5, :meth:`update`);
    :meth:`mean` gives E10's mean gradient g. The statistics are reset after each event (§1.3
    step 7) and resized with the model when it grows (:meth:`resize_`).

    Parameters
    ----------
    num_gaussians
        The number of primitives N, an int ≥ 0.
    device
        The device that holds the statistics, the model's; ``"cpu"`` by default.

    Raises
    ------
    ValueError
        If ``num_gaussians`` is not an int ≥ 0.

    Notes
    -----
    The statistics are part of the checkpoint, as ``TrainerState.stats`` (§2.1, §2.2), through
    :meth:`state_dict` and :meth:`load_state_dict`.
    """

    def __init__(self, num_gaussians: int, *, device: torch.device | str = "cpu") -> None:
        if not _is_int(num_gaussians) or num_gaussians < 0:
            _fail(f"num_gaussians must be an int ≥ 0, got {num_gaussians!r}")
        self._g_sum = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
        self._count = torch.zeros(num_gaussians, dtype=torch.int32, device=device)

    @property
    def num_gaussians(self) -> int:
        """The number of primitives N the statistics cover (Q-48)."""
        return int(self._g_sum.shape[0])

    @property
    def g_sum(self) -> Tensor:
        """The sum of NDC-scaled 2D-mean gradient norms, [N] float32 (E10, D-12). Read-only."""
        return self._g_sum

    @property
    def count(self) -> Tensor:
        """The number of visible camera–primitive pairs summed into g_sum, [N] int32 (Q-49).

        Read-only.
        """
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
        """Add the visible primitives' NDC-scaled 2D-mean gradient norms (§1.3 step 5, D-12).

        Each visible camera–primitive pair adds the norm of its 2D-mean gradient, with x scaled
        by W/2 and y by H/2, to ``g_sum`` and one to ``count``. The two modes follow the
        renderer's (Q-49): unpacked, ``means2d_grad`` [C, N, 2] with the mask ``visible``
        [C, N]; packed, ``means2d_grad`` [nnz, 2] with ``gaussian_ids`` [nnz]. Shapes, dtypes
        and devices are checked, never values, so nothing synchronizes (D-49).

        Parameters
        ----------
        means2d_grad
            The gradient of the loss with respect to the 2D means, in pixel coordinates:
            [C, N, 2] unpacked or [nnz, 2] packed, floating point, on the statistics' device.
        width
            The image width W in pixels, an int ≥ 1.
        height
            The image height H in pixels, an int ≥ 1.
        visible
            Unpacked mode: the visible camera–primitive pairs, a bool mask [C, N].
        gaussian_ids
            Packed mode: the primitive of each row of ``means2d_grad``, [nnz] int64 in [0, N);
            repeats add up. The range is not checked.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``width`` or ``height`` is not an int ≥ 1, ``means2d_grad`` is not a
            floating-point tensor of its mode's shape, not exactly one of ``visible`` and
            ``gaussian_ids`` is given, the one given has the wrong dtype or shape, or a tensor
            is on another device.

        Notes
        -----
        §2.1's signature lists the keyword-only pair ``visible`` and ``gaussian_ids``, whose
        shapes and rules Q-49 defines. Packed rows are scatter-added, since every row is a
        visible pair (D-75, D-76).
        """
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
        """Return E10's mean gradient g, ``g_sum / count`` where ``count`` > 0 and 0 elsewhere.

        Returns
        -------
        torch.Tensor
            g, [N] float32, ≥ 0, in NDC units (D-12).
        """
        return torch.where(self._count > 0, self._g_sum / self._count.clamp_min(1), 0.0)

    def reset(self) -> None:
        """Zero g_sum and count (§1.3 step 7).

        The statistics are reset after every event: by :func:`relocate` itself, or by the
        trainer when relocation is off (Q-51).
        """
        self._g_sum.zero_()
        self._count.zero_()

    def state_dict(self) -> dict[str, Tensor]:
        """Return the statistics as detached clones, for ``TrainerState.stats`` (§2.2, Q-48).

        Returns
        -------
        dict of str to torch.Tensor
            ``{"g_sum": [N] float32, "count": [N] int32}``, detached clones (§2.1, §2.2).
        """
        return {"g_sum": self._g_sum.detach().clone(), "count": self._count.detach().clone()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the statistics from :meth:`state_dict`'s layout, from any device (Q-48).

        The tensors are copied to the statistics' device. On error nothing changes.

        Parameters
        ----------
        state
            ``{"g_sum": [N] float32, "count": [N] int32}``, with N the current number of rows.

        Raises
        ------
        ValueError
            If ``state`` doesn't hold exactly ``g_sum`` and ``count``, or one of them has the
            wrong dtype or shape.
        """
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
        """Keep the rows of ``keep``, then append copies of the rows ``append_from`` names (D-45).

        Kept rows stay in order and new rows go last, each with the statistics of its parent
        (§1.3 Growth, step 4); a parent may appear more than once, as a split's two children
        do. On error nothing changes.

        Parameters
        ----------
        keep
            The rows to keep, a bool mask [N] on the statistics' device (Q-23); None, the
            default, keeps every row.
        append_from
            The parents of the new rows, [k] int64 indices of the rows before the resize, in
            [0, N), repeats allowed; the range is not checked (Q-44, Q-49). None, the default,
            appends nothing.

        Raises
        ------
        ValueError
            If ``keep`` is not a bool mask [N] on the statistics' device, or ``append_from`` is
            not a 1-D int64 tensor on it.
        """
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
    """Compute the relocation sampling score ρ of E10 for every row (E10, D-13).

    ``ρ = λ_g·ĝ + λ_o·σ``, with σ the base opacity and ``ĝ = min(1, g/Q)``: g is the mean
    gradient of :meth:`GradStats.mean` and Q the ``cfg.grad_quantile`` quantile of g over the
    live rows visible at least once, interpolated linearly (D-13). ĝ ≡ 0 when Q is 0 or there
    is no such row. Every row is scored, dead ones included; the procedures mask out the dead
    set (Q-48). No host synchronization.

    Parameters
    ----------
    model
        The model, for its base opacities and its live set (D-14).
    stats
        The gradient statistics, covering the model's rows, on its device.
    cfg
        The weights λ_g and λ_o, the quantile and the dead-opacity threshold (E10, D-13,
        D-14).

    Returns
    -------
    torch.Tensor
        ρ, [N] float32, in [0, λ_g + λ_o], detached from the autograd graph.

    Raises
    ------
    ValueError
        If ``stats`` doesn't hold the model's number of rows or is on another device.
    """
    _check_stats(model, stats)
    sigma = torch.sigmoid(model.opacity_logits.detach())
    if model.num_gaussians == 0:
        return sigma
    g = stats.mean()
    live_seen = _live(model, cfg) & (stats.count > 0)
    # The quantile of the selected rows without a gather (no data-dependent shape): the other
    # rows are NaN, which nanquantile ignores; torch's default linear interpolation (D-13).
    q = torch.nanquantile(torch.where(live_seen, g, math.nan), cfg.grad_quantile)
    positive = q > 0  # False for a NaN quantile (no live, visible row): ĝ ≡ 0 (E10)
    g_hat = torch.where(positive, torch.clamp(g / torch.where(positive, q, 1.0), max=1.0), 0.0)
    return cfg.lambda_g * g_hat + cfg.lambda_o * sigma


_N_COPIES_MAX = 51  # E11: n is clamped to [1, 51]
_OPACITY_MAX = 1.0 - 2.0**-23  # E11: the upper bound of σ_new
_E11_TABLES: dict[torch.device, Tensor] = {}


def _e11_table(device: torch.device) -> Tensor:
    """Return E11's [51, 51] float64 coefficient table on `device`.

    Row n − 1, column k holds Σ_{i=1}^{n} C(i − 1, k)·(−1)^k/√(k + 1), the coefficient of
    σ_new^(k+1) in E11's denominator. The binomial sums are exact Python integers; built once
    per device (its copy to the device is the only host transfer).
    """
    if device not in _E11_TABLES:
        rows = []
        for n in range(1, _N_COPIES_MAX + 1):
            rows.append(
                [
                    (-1) ** k * sum(math.comb(m, k) for m in range(n)) / math.sqrt(k + 1)
                    for k in range(_N_COPIES_MAX)
                ]
            )
        _E11_TABLES[device] = torch.tensor(rows, dtype=torch.float64).to(device)
    return _E11_TABLES[device]


# Q-50: computed in float64 and returned in the input's dtype; n clamped for the formula only;
# the upper bound applied here, the lower bound (D-14) by the callers on the stored logit.
def mcmc_split(opacity: Tensor, scales: Tensor, n: Tensor) -> tuple[Tensor, Tensor]:
    """Apply E11 to k targets that each end with n copies (E11).

    ``σ_new = 1 − (1 − σ)^(1/n)``, clamped to at most 1 − 2⁻²³, and a_new from E11's sum,
    computed from the unclamped σ_new. The formula runs in float64, with n clamped to [1, 51],
    and the results come back in the input's dtype (Q-50). The lower bound of E11's clamp,
    the dead-opacity threshold (D-14), is left to the callers, who apply it to the stored
    logit.

    Parameters
    ----------
    opacity
        The targets' base opacities σ, [k], floating point, in [0, 1].
    scales
        The targets' activated scales a, [k, 3], in ``opacity``'s dtype, in world units.
    n
        Each target's number of copies, itself included, [k], of an integer dtype.

    Returns
    -------
    opacity : torch.Tensor
        σ_new, [k], in ``opacity``'s dtype, at most 1 − 2⁻²³.
    scales : torch.Tensor
        a_new, [k, 3], in ``opacity``'s dtype, in world units.

    Raises
    ------
    ValueError
        If ``opacity`` is not a 1-D floating-point tensor, ``scales`` is not [k, 3] in its
        dtype, ``n`` is not an integer tensor [k], or the three are not on one device.

    Notes
    -----
    Q-50 fixes the precision, the clamping of n and the split of the clamp's two bounds.
    """
    if not isinstance(opacity, Tensor) or not opacity.is_floating_point() or opacity.dim() != 1:
        _fail("opacity must be a floating-point tensor of shape [k]")
    k = int(opacity.shape[0])
    if not isinstance(scales, Tensor) or tuple(scales.shape) != (k, 3):
        _fail(f"scales must have shape [{k}, 3]")
    if scales.dtype != opacity.dtype:
        _fail(f"scales must be {opacity.dtype}, got {scales.dtype}")
    if (
        not isinstance(n, Tensor)
        or n.dtype == torch.bool
        or n.is_floating_point()
        or n.is_complex()
        or tuple(n.shape) != (k,)
    ):
        _fail(f"n must be an integer tensor of shape [{k}]")
    if scales.device != opacity.device or n.device != opacity.device:
        _fail("opacity, scales and n must be on one device")
    sigma = opacity.to(torch.float64)
    new_opacity, new_scales = _e11(sigma, torch.log1p(-sigma), scales.to(torch.float64), n)
    return new_opacity.to(opacity.dtype), new_scales.to(opacity.dtype)


# Q-50: E11 in float64 from σ and log(1 − σ), so that relocation can form log(1 − σ) from the
# stored logit and keep E11 exact where the float64 σ of a saturated logit rounds to 1.
def _e11(sigma: Tensor, log1m_sigma: Tensor, scales: Tensor, n: Tensor) -> tuple[Tensor, Tensor]:
    """Apply E11 in float64, from σ and log(1 − σ).

    σ_new = 1 − (1 − σ)^(1/n), clamped to at most 1 − 2⁻²³, and a_new from the unclamped σ_new
    (E11), all float64; n is clamped to [1, 51] for the formula.
    """
    f64 = torch.float64
    copies = n.clamp(1, _N_COPIES_MAX).to(torch.int64)
    # 1 − (1 − σ)^(1/n), in a form that stays accurate for small σ.
    sigma_new = -torch.expm1(log1m_sigma / copies.to(f64))
    exponents = torch.arange(1, _N_COPIES_MAX + 1, dtype=f64, device=sigma.device)
    powers = sigma_new[:, None] ** exponents  # σ_new^(k+1)
    denominator = (_e11_table(sigma.device)[copies - 1] * powers).sum(dim=1)
    new_scales = scales * (sigma / denominator)[:, None]
    return sigma_new.clamp(max=_OPACITY_MAX), new_scales


def _check_generator(generator: object) -> None:
    if not isinstance(generator, torch.Generator):
        _fail(f"generator must be a torch.Generator, got {type(generator).__name__}")


def draw_targets(scores: Tensor, count: int, *, generator: torch.Generator) -> Tensor:
    """Draw ``count`` rows with replacement, with probability ∝ ``scores`` (D-15).

    The draw runs on the generator's device, as in ``FreeTimeGaussians.from_init_points``, and
    its result comes back on the scores' device. The scores are read on the host, so the call
    synchronizes.

    Parameters
    ----------
    scores
        The sampling weights, [N], finite and non-negative with a positive sum (Q-51).
    count
        The number of rows to draw.
    generator
        The generator of the draw, the trainer's relocation generator (D-15).

    Returns
    -------
    torch.Tensor
        The drawn rows, [count] int64 indices in [0, N), on the scores' device.

    Raises
    ------
    ValueError
        If ``generator`` is not a :class:`torch.Generator`, or the scores are not finite with a
        positive sum (λ_o = 0 with a zero gradient quantile gives all zeros).

    Notes
    -----
    Q-204 makes the draw public, for the differential harness. Q-51: without the check on the
    scores, :func:`torch.multinomial` fails on the CPU and asserts on CUDA.
    """
    _check_generator(generator)
    if not bool(torch.isfinite(scores).all()) or not bool(scores.sum() > 0):
        _fail(
            "the live primitives' sampling scores must be finite with a positive sum "
            "(λ_o = 0 with a zero gradient quantile gives all zeros)"
        )
    draws = torch.multinomial(
        scores.to(generator.device), count, replacement=True, generator=generator
    )
    return draws.to(scores.device)


# Q-50, Q-51: the relocation move once per target, then one row per draw (relocation's moved
# copies, or the score mode's new rows), shared by relocate and grow.
def _relocation_move(
    model: FreeTimeGaussians, draws: Tensor, cfg: RelocConfig, generator: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply E11 in place to every target in `draws`; return the targets and the rows to fill.

    `draws` holds one entry per row to fill, repeats allowed; each target's n is 1 + its number
    of draws. Returns the target rows and, one row per draw, the nine tensors of the row to
    fill: its target's updated tensors, placed per `cfg.copy_position` (D-15).
    """
    n_rows = model.num_gaussians
    device = model.means.device
    counts = torch.zeros(n_rows, dtype=torch.int64, device=device)
    counts.index_add_(0, draws, torch.ones_like(draws))  # bincount would read its max back
    targets = torch.nonzero(counts).squeeze(1)
    with torch.no_grad():
        logits, log_scales = model.opacity_logits, model.log_scales
        # E11's a_j and the target's Gaussian are the target's before the move (Q-51).
        means = model.means.index_select(0, draws)
        rs = rotation_matrix(model.quats.index_select(0, draws))
        rs = rs * torch.exp(log_scales.index_select(0, draws))[:, None, :]  # R·diag(a)
        # Q-50: σ_j and log(1 − σ_j) = −log(1 + e^ô_j) from the stored logit, in float64.
        target_logits = logits[targets].double()
        log1m_sigma = -torch.logaddexp(torch.zeros_like(target_logits), target_logits)
        new_sigma, new_scales = _e11(
            torch.sigmoid(target_logits),
            log1m_sigma,
            torch.exp(log_scales[targets].double()),
            1 + counts[targets],
        )
        # D-14: the lower bound of E11's clamp, on the stored float32 logit (Q-50).
        new_logits = torch.logit(new_sigma).float().clamp_min(_logit_bound(cfg.dead_opacity))
        logits.index_copy_(0, targets, new_logits)
        log_scales.index_copy_(0, targets, torch.log(new_scales).float())
        filled = {
            name: getattr(model, name).detach().index_select(0, draws)
            for name in FreeTimeGaussians.PARAM_NAMES
        }
        if cfg.copy_position == "sample":
            # D-15: μ_j + R_j·diag(a_j)·z, z ~ N(0, I₃), drawn after the targets (Q-51).
            z = torch.randn(
                (int(draws.shape[0]), 3), generator=generator, device=generator.device
            ).to(device)
            filled["means"] = means + (rs @ z[:, :, None]).squeeze(-1)
    return targets, filled


# Q-51: the procedure's details; Q-55: the validity checks before the draws and after the writes.
def relocate(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    cfg: RelocConfig,
    *,
    generator: torch.Generator,
) -> RelocationReport:
    """Move every dead primitive onto a live target (§1.3 Relocation, E11, D-14 to D-16).

    The dead set is the primitives whose base opacity is below ``cfg.dead_opacity``, tested on
    the stored logit (D-14); the live set is the rest. One target per dead primitive is drawn
    from the live set with probability ∝ ρ (E10, :func:`draw_targets`), and E11 is applied
    once per target, with n = 1 + its number of draws. Each dead primitive then takes its
    target's updated tensors, its position drawn from the target's Gaussian or left on the
    target, as ``cfg.copy_position`` says (D-15). The Adam moments of the targets, and of the
    moved rows when ``cfg.reset_moved_moments`` is set, are zeroed (D-16). Last, the gradient
    statistics are reset (§1.3 step 7). If the dead or the live set is empty, only the
    statistics are reset. N doesn't change.

    Parameters
    ----------
    model
        The model, changed in place.
    optimizer
        The model's own optimizer, built for it by :func:`~openftgs.optim.build_optimizer`.
    stats
        The gradient statistics, covering the model's rows, on its device; reset on return.
    cfg
        The relocation settings (E10, E11, D-13 to D-16). ``cfg.enabled`` is the trainer's
        switch and isn't read here.
    generator
        The relocation generator, the trainer's dedicated, checkpointed one (D-15); it draws
        the targets, then the positions.

    Returns
    -------
    RelocationReport
        The number of dead primitives moved and of distinct targets.

    Raises
    ------
    ValueError
        If ``stats`` doesn't cover the model's rows on its device, ``optimizer`` is not the
        model's own (Q-48), ``generator`` is not a :class:`torch.Generator`, the model breaks
        §1.1's validity rules before or after the move (Q-55), or the live primitives' scores
        have no positive sum.

    Notes
    -----
    Synchronizes with the host, as an event does anyway (§2.4 rule 3). Q-51 fixes the
    procedure's details and Q-55 its validity checks.
    """
    _check_stats(model, stats)
    _check_optimizer(model, optimizer)
    _check_generator(generator)
    model.validate()
    dead = ~_live(model, cfg)
    n_dead = int(dead.sum())
    if n_dead == 0 or n_dead == model.num_gaussians:
        stats.reset()
        return RelocationReport(dead=0, targets=0)
    scores = torch.where(dead, 0.0, sampling_score(model, stats, cfg))
    draws = draw_targets(scores, n_dead, generator=generator)
    dead_rows = torch.nonzero(dead).squeeze(1)  # ascending: they take the draws in draw order
    targets, moved = _relocation_move(model, draws, cfg, generator)
    with torch.no_grad():
        for name in FreeTimeGaussians.PARAM_NAMES:
            getattr(model, name).index_copy_(0, dead_rows, moved[name])
    # D-16: zero the moments of the targets and the moved copies, or of the targets only.
    optimizer.reset_rows(torch.cat([targets, dead_rows]) if cfg.reset_moved_moments else targets)
    model.validate()  # §1.1: checked after every relocation
    stats.reset()  # §1.3 step 7 (Q-51, owner answer)
    return RelocationReport(dead=n_dead, targets=int(targets.shape[0]))


def _check_spatial_scale(spatial_scale: object) -> float:
    if isinstance(spatial_scale, bool) or not isinstance(spatial_scale, int | float):
        _fail(f"spatial_scale must be a number, got {spatial_scale!r}")
    if not (math.isfinite(spatial_scale) and spatial_scale > 0.0):
        _fail(f"spatial_scale must be finite and > 0, got {spatial_scale!r}")
    return float(spatial_scale)


def _resize_all(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    *,
    keep: Tensor | None,
    append: Mapping[str, Tensor] | None,
    parents: Tensor | None,
) -> None:
    """Resize the model, optimizer and statistics together with one keep mask (§1.3 Growth step 4).

    Q-23, Q-44: new rows get zero moments and their parents' statistics (D-45).
    """
    model.resize_(keep=keep, append=append)
    optimizer.resize_(keep=keep, n_append=0 if parents is None else int(parents.shape[0]))
    stats.resize_(keep=keep, append_from=parents)


def growth_masks(
    g: Tensor, scales: Tensor, cfg: GrowthConfig, *, spatial_scale: float
) -> tuple[Tensor, Tensor]:
    """Choose the rows a growth event clones and splits (§1.3 Growth, D-18, D-44).

    Candidates have a mean gradient ``g`` above ``cfg.grad_threshold``. When they exceed the
    room below ``cfg.n_max`` (N = len(g); a clone and a split each add one row), the largest g
    that fit are kept, ties to the lower row (D-44). A kept candidate is cloned when its
    largest scale is at most ``cfg.split_scale``·S, and split otherwise. The comparisons run
    in float32. Synchronizes with the host for the cap.

    Parameters
    ----------
    g
        The mean gradients of :meth:`GradStats.mean`, [N] float32.
    scales
        The activated scales, [N, 3], in world units.
    cfg
        The growth settings: the threshold τ_pos, the split scale and N_max.
    spatial_scale
        The scene's spatial scale S (D-27), in world units.

    Returns
    -------
    clone : torch.Tensor
        The rows to clone, a bool mask [N].
    split : torch.Tensor
        The rows to split, a bool mask [N].

    Notes
    -----
    Q-204 makes the masks public, for the differential harness.
    """
    n = int(g.shape[0])
    rows = torch.nonzero(g > cfg.grad_threshold).squeeze(1)  # §1.3 Growth step 1, in float32
    room = max(cfg.n_max - n, 0)
    if int(rows.shape[0]) > room:
        # D-44: the largest g that fit, ties to the lower row (a stable sort of rows in order).
        best = torch.sort(g[rows], descending=True, stable=True).indices[:room]
        rows = torch.sort(rows[best]).values
    # §1.3 Growth step 3: a clone when the largest scale is ≤ split_scale·S (in float32).
    large = scales[rows].amax(dim=-1) > cfg.split_scale * spatial_scale
    clone = torch.zeros(n, dtype=torch.bool, device=g.device)
    split = torch.zeros(n, dtype=torch.bool, device=g.device)
    clone[rows[~large]] = True
    split[rows[large]] = True
    return clone, split


# Q-52: candidates, the cap, the clone/split boundary, the order of new rows and of the draws.
def _grow_by_gradient(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    cfg: GrowthConfig,
    spatial_scale: float,
    generator: torch.Generator,
) -> GrowthReport:
    n = model.num_gaussians
    g = stats.mean()
    with torch.no_grad():
        params = {name: getattr(model, name).detach() for name in FreeTimeGaussians.PARAM_NAMES}
        clone, split = growth_masks(
            g, torch.exp(params["log_scales"]), cfg, spatial_scale=spatial_scale
        )
        n_candidates = int((g > cfg.grad_threshold).sum())
        clone_rows = torch.nonzero(clone).squeeze(1)
        split_rows = torch.nonzero(split).squeeze(1)
        capped = n_candidates - int(clone_rows.shape[0]) - int(split_rows.shape[0])
        if clone_rows.numel() + split_rows.numel() == 0:
            return GrowthReport(cloned=0, split=0, sampled=0, capped=capped, num_gaussians=n)
        n_split = int(split_rows.shape[0])
        new_rows = [{name: p.index_select(0, clone_rows) for name, p in params.items()}]
        keep = None
        if n_split:
            parent = {name: p.index_select(0, split_rows) for name, p in params.items()}
            # E17: μ + R(q)·diag(a)·z_k for k = 1, 2, from one draw, first children first.
            z = torch.randn((2, n_split, 3), generator=generator, device=generator.device)
            rs = rotation_matrix(parent["quats"]) * torch.exp(parent["log_scales"])[:, None, :]
            offsets = (rs[None] @ z.to(rs.device)[..., None]).squeeze(-1)
            child_log_scales = (parent["log_scales"].double() - math.log(_SPLIT_DIVISOR)).float()
            for k in range(2):
                new_rows.append(
                    {
                        **parent,
                        "means": parent["means"] + offsets[k],
                        "log_scales": child_log_scales,
                    }
                )
            keep = torch.ones(n, dtype=torch.bool, device=g.device).index_fill_(
                0, split_rows, False
            )
        append = {name: torch.cat([r[name] for r in new_rows]) for name in params}
    parents = torch.cat([clone_rows, split_rows, split_rows])
    _resize_all(model, optimizer, stats, keep=keep, append=append, parents=parents)
    return GrowthReport(
        cloned=int(clone_rows.shape[0]),
        split=n_split,
        sampled=0,
        capped=capped,
        num_gaussians=model.num_gaussians,
    )


def grow(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    cfg: GrowthConfig,
    *,
    spatial_scale: float,
    reloc: RelocConfig,
    generator: torch.Generator,
) -> GrowthReport:
    """Run one growth event in the configured count-control mode (§1.3 Growth, E17, D-18).

    ``cfg.mode`` picks what the event does (§1.3 Count-control modes, D-42 to D-45):

    - "gradient" clones and splits the candidates of :func:`growth_masks`. A clone copies all
      nine tensors; a split replaces its parent by two children placed by E17, with scales
      a/1.6, from one draw of ``generator``.
    - "score" appends max(0, min(N_max, ⌊(1 + ``cfg.score_rate``)·N⌋) − N) rows drawn ∝ ρ from
      the live set and initialized by E11, as relocation does; the targets' Adam moments are
      zeroed (D-16). Nothing is added when no row is live.
    - "none" does nothing.

    Model, optimizer and statistics are resized together: kept rows stay in order, and new
    rows go last with zero Adam moments and their parents' statistics (D-45).

    Parameters
    ----------
    model
        The model, changed in place.
    optimizer
        The model's own optimizer, built for it by :func:`~openftgs.optim.build_optimizer`.
    stats
        The gradient statistics, covering the model's rows, on its device.
    cfg
        The growth settings: the mode, τ_pos, the split scale, N_max and the score mode's rate.
    spatial_scale
        The scene's spatial scale S (D-27), in world units, finite and > 0.
    reloc
        The relocation settings, for the score mode's ρ, live set and E11 (E10, E11, D-14).
    generator
        The generator of the split offsets or of the score mode's draws.

    Returns
    -------
    GrowthReport
        What the event did, and N after it.

    Raises
    ------
    ValueError
        If ``stats`` doesn't cover the model's rows on its device, ``optimizer`` is not the
        model's own (Q-48), ``generator`` is not a :class:`torch.Generator`, ``spatial_scale``
        is not a finite number > 0, or, in the score mode, the model breaks §1.1's validity
        rules (Q-55) or the live primitives' scores have no positive sum.

    Notes
    -----
    Synchronizes with the host, as an event does anyway (§2.4 rule 3). Q-52 fixes the
    procedure's details: the candidates, the cap, the clone and split boundary, and the order
    of the new rows and of the draws.
    """
    _check_stats(model, stats)
    _check_optimizer(model, optimizer)
    _check_generator(generator)
    scale = _check_spatial_scale(spatial_scale)
    if cfg.mode == "none":
        return GrowthReport(
            cloned=0, split=0, sampled=0, capped=0, num_gaussians=model.num_gaussians
        )
    if cfg.mode == "score":
        return _grow_by_score(model, optimizer, stats, cfg, reloc, generator)
    return _grow_by_gradient(model, optimizer, stats, cfg, scale, generator)


# Q-52: the score mode draws from the live set only, applies E11 like relocation, zeroes the
# targets' moments (D-16) and appends the new rows with their targets' statistics (D-45).
def _grow_by_score(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    cfg: GrowthConfig,
    reloc: RelocConfig,
    generator: torch.Generator,
) -> GrowthReport:
    n = model.num_gaussians
    # §1.3 Count-control modes: max(0, min(N_max, ⌊(1 + rate)·N⌋) − N) new rows.
    n_add = max(0, min(cfg.n_max, math.floor((1.0 + cfg.score_rate) * n)) - n)
    if n_add == 0:
        return GrowthReport(cloned=0, split=0, sampled=0, capped=0, num_gaussians=n)
    model.validate()  # Q-55: a non-finite model raises before the draw
    live = _live(model, reloc)
    if not bool(live.any()):  # Q-52: no live row to draw from, so nothing is added
        return GrowthReport(cloned=0, split=0, sampled=0, capped=0, num_gaussians=n)
    scores = torch.where(live, sampling_score(model, stats, reloc), 0.0)
    draws = draw_targets(scores, n_add, generator=generator)
    targets, new_rows = _relocation_move(model, draws, reloc, generator)
    optimizer.reset_rows(targets)
    _resize_all(model, optimizer, stats, keep=None, append=new_rows, parents=draws)
    return GrowthReport(
        cloned=0, split=0, sampled=n_add, capped=0, num_gaussians=model.num_gaussians
    )


# Q-53: the "w/o periodic relocation" ablation's thresholds and reset value, fixed constants with
# no config field (D-72) [L: 3DGS §5.2; thresholds from gsplat 1.5.3 DefaultStrategy].
# Q-67: the opacity reset's period, public so the trainer's calendar takes it from here; it is
# also the iteration after which large primitives are pruned (iteration > 3000, as gsplat's
# step > reset_every).
OPACITY_RESET_EVERY = 3000
"""The opacity reset's period in iterations, for the "w/o periodic relocation" ablation (Q-67).

The trainer's calendar resets opacities with :func:`reset_opacity` every 3,000 iterations, and
:func:`prune` removes large primitives only after this iteration (§1.3 Count-control modes,
Q-53).
"""
_PRUNE_SCALE = 0.1  # × S: larger primitives are pruned after OPACITY_RESET_EVERY
_RESET_OPACITY = 0.01  # the opacity reset's bound on σ


def prune(
    model: FreeTimeGaussians,
    optimizer: FTGSOptimizer,
    stats: GradStats,
    cfg: RelocConfig,
    *,
    spatial_scale: float,
    iteration: int,
) -> int:
    """Prune primitives for the "w/o periodic relocation" ablation (§1.3 Count-control modes).

    Remove the primitives whose base opacity is below ``cfg.prune_opacity``, tested on the
    stored logit as D-14 tests the dead set, and, after iteration :data:`OPACITY_RESET_EVERY`
    (3,000), those whose largest scale exceeds 0.1·S. Model, optimizer and statistics are
    resized together. With ``cfg.prune_opacity`` None, the default, nothing changes.

    Parameters
    ----------
    model
        The model, changed in place.
    optimizer
        The model's own optimizer, built for it by :func:`~openftgs.optim.build_optimizer`.
    stats
        The gradient statistics, covering the model's rows, on its device.
    cfg
        The relocation settings, for ``cfg.prune_opacity``.
    spatial_scale
        The scene's spatial scale S (D-27), in world units, finite and > 0.
    iteration
        The current iteration, an int ≥ 1 (D-41).

    Returns
    -------
    int
        The number of primitives removed.

    Raises
    ------
    ValueError
        If ``stats`` doesn't cover the model's rows on its device, ``optimizer`` is not the
        model's own (Q-48), ``spatial_scale`` is not a finite number > 0, or ``iteration`` is
        not an int ≥ 1.

    Notes
    -----
    Q-53 fixes the thresholds, as constants with no configuration field (D-72) [L: 3DGS §5.2;
    thresholds from gsplat 1.5.3 DefaultStrategy]; Q-67 the iteration after which large
    primitives go.
    """
    _check_stats(model, stats)
    _check_optimizer(model, optimizer)
    scale = _check_spatial_scale(spatial_scale)
    if not _is_int(iteration) or iteration < 1:
        _fail(f"iteration must be an int ≥ 1, got {iteration!r}")
    if cfg.prune_opacity is None:
        return 0
    with torch.no_grad():
        remove = model.opacity_logits.detach() < _logit_bound(cfg.prune_opacity)
        if iteration > OPACITY_RESET_EVERY:
            largest = torch.exp(model.log_scales.detach()).amax(dim=-1)
            remove |= largest > _PRUNE_SCALE * scale
    n_removed = int(remove.sum())
    if n_removed:
        _resize_all(model, optimizer, stats, keep=~remove, append=None, parents=None)
    return n_removed


def reset_opacity(model: FreeTimeGaussians, optimizer: FTGSOptimizer) -> None:
    """Reset opacities for the "w/o periodic relocation" ablation (§1.3 Count-control modes).

    Set σ ← min(σ, 0.01) on the stored logit, and zero the Adam moments of the opacity logits
    for every row, keeping the step counters (Q-53, owner answer). No host synchronization.

    Parameters
    ----------
    model
        The model, changed in place.
    optimizer
        The model's own optimizer, built for it by :func:`~openftgs.optim.build_optimizer`.

    Raises
    ------
    ValueError
        If ``optimizer`` is not the model's own (Q-48).
    """
    _check_optimizer(model, optimizer)
    with torch.no_grad():
        model.opacity_logits.clamp_(max=_logit_bound(_RESET_OPACITY))
    rows = torch.arange(model.num_gaussians, device=model.means.device)
    optimizer.reset_rows(rows, names=("opacity_logits",))
