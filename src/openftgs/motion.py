# SPDX-License-Identifier: Apache-2.0
"""Pure functions for motion, temporal opacity and effective opacity, time normalization, and the
fused temporal slice with its eager, compiled and optional CUDA implementations.

Spec: E1, E3, E4, E14, D-50. Band: Core model (§2.1).
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import torch
from torch import Tensor

from openftgs.types import TimeNormalizer

__all__ = [
    "TemporalSlice",
    "durations_to_normalized",
    "durations_to_raw",
    "effective_opacity",
    "moved_means",
    "reg4d_terms",
    "temporal_opacity",
    "temporal_slice",
    "velocities_to_normalized",
    "velocities_to_raw",
]

# D-32: gsplat 1.5.3 drops σ_eff < 1/255; on float32 tensors the bound rounds to float32(1/255).
_CULL_OPACITY = 1.0 / 255.0
_Impl = Literal["auto", "eager", "compiled", "cuda"]  # PerfConfig.slice_impl's values (D-50)
_IMPLS: tuple[str, ...] = get_args(_Impl)


# Q-12: the return type of temporal_slice.
@dataclass(frozen=True)
class TemporalSlice:
    """The per-time arrays, activations, E8 terms and culling mask of a model at one time (Q-12)."""

    means_t: Tensor  # [N, 3] μx(t), E1
    opacities_t: Tensor  # [N] σ_eff = σ·σt, E3
    quats_n: Tensor  # [N, 4] q/‖q‖
    scales: Tensor  # [N, 3] exp(â)
    opacities: Tensor  # [N] σ = sigmoid(ô)
    reg_terms: Tensor  # [N] σ·sg[σt], the per-row terms of E8
    passes_cull: Tensor  # [N] bool, σ_eff ≥ 1/255: survives the opacity cull (D-32)


def _as_time(t: float | Tensor, like: Tensor) -> Tensor:
    """t as a 0-d tensor of the parameters' dtype and device, so a compiled graph isn't
    specialized on its value. A Python float becomes a fill, which needs no host-to-device copy
    and doesn't synchronize; a tensor t is moved with ``.to``, a copy if it is on another
    device."""
    if isinstance(t, Tensor):
        return t.to(dtype=like.dtype, device=like.device)
    return torch.full((), float(t), dtype=like.dtype, device=like.device)


def _moved_means(means: Tensor, velocities: Tensor, times: Tensor, t: Tensor) -> Tensor:
    # E1 in this order, so that t = μt gives μx exactly.
    return means + velocities * (t - times)[..., None]


def _sigma_t(times: Tensor, log_durations: Tensor, t: Tensor) -> Tensor:
    # E4 as exp(−½z²) with z = (t − μt)·e^−ŝ: s is never formed and never divided by, so autograd
    # builds no power of 1/s, which underflows in float32 at short durations. Far from μt, σt
    # underflows to exactly 0 and the backward products are 0·finite.
    z = (t - times) * torch.exp(-log_durations)
    return torch.exp(-0.5 * (z * z))


def _opacity_terms(opacity_logits: Tensor, sigma_t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """σ = sigmoid(ô), σ_eff = σ·σt (E3) and the E8 row terms σ·sg[σt], from one σt."""
    sigma = torch.sigmoid(opacity_logits)
    return sigma, sigma * sigma_t, sigma * sigma_t.detach()


# Q-12: the public single-equation functions.
def moved_means(means: Tensor, velocities: Tensor, times: Tensor, t: float | Tensor) -> Tensor:
    """μx(t) = μx + v·(t − μt) (E1), [N, 3]."""
    return _moved_means(means, velocities, times, _as_time(t, times))


def temporal_opacity(times: Tensor, log_durations: Tensor, t: float | Tensor) -> Tensor:
    """σt(t) = exp(−½((t − μt)/s)²) with s = e^ŝ (E4, D-2), [N]."""
    return _sigma_t(times, log_durations, _as_time(t, times))


def effective_opacity(
    opacity_logits: Tensor, times: Tensor, log_durations: Tensor, t: float | Tensor
) -> Tensor:
    """σ_eff(t) = sigmoid(ô)·σt(t) (E3), [N]."""
    return _opacity_terms(opacity_logits, temporal_opacity(times, log_durations, t))[1]


def reg4d_terms(
    opacity_logits: Tensor, times: Tensor, log_durations: Tensor, t: float | Tensor
) -> Tensor:
    """The per-row terms σ·sg[σt(t)] of E8, [N]; their mean is L_reg(t)."""
    return _opacity_terms(opacity_logits, temporal_opacity(times, log_durations, t))[2]


def _span_or_one(time: TimeNormalizer) -> float:
    # D-70: with a single timestamp (T_τ = 0), durations and velocities convert with a unit of 1.
    span = time.tau1 - time.tau0
    return span if span > 0.0 else 1.0


# Q-27: E14's unit conversions of primitive quantities (Q-8); time centers use TimeNormalizer.
def durations_to_normalized(durations: Tensor, time: TimeNormalizer) -> Tensor:
    """s = d/T_τ: durations in raw τ units to normalized time (E14)."""
    return durations / _span_or_one(time)


def durations_to_raw(durations: Tensor, time: TimeNormalizer) -> Tensor:
    """d = s·T_τ: durations in normalized time to raw τ units (E14)."""
    return durations * _span_or_one(time)


def velocities_to_normalized(velocities: Tensor, time: TimeNormalizer) -> Tensor:
    """v = v_τ·T_τ: world units per unit of τ to world units per unit of t (E14)."""
    return velocities * _span_or_one(time)


def velocities_to_raw(velocities: Tensor, time: TimeNormalizer) -> Tensor:
    """v_τ = v/T_τ: world units per unit of t to world units per unit of τ (E14)."""
    return velocities / _span_or_one(time)


def _slice_eager(
    means: Tensor,
    times: Tensor,
    log_durations: Tensor,
    velocities: Tensor,
    log_scales: Tensor,
    quats: Tensor,
    opacity_logits: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # One body for every implementation. No boolean indexing: no shape depends on the data.
    # The tuple is in TemporalSlice's field order.
    sigma, opacities_t, reg_terms = _opacity_terms(
        opacity_logits, _sigma_t(times, log_durations, t)
    )
    return (
        _moved_means(means, velocities, times, t),
        opacities_t,
        quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True),
        torch.exp(log_scales),
        sigma,
        reg_terms,
        opacities_t >= _CULL_OPACITY,
    )


_SliceFn = Callable[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
_compiled: _SliceFn | None = None
_auto_fell_back = False


def _compiled_slice() -> _SliceFn:
    global _compiled
    if _compiled is None:
        # D-50: dynamic shapes, so growth events don't recompile. Q-13: fullgraph, so a graph
        # break raises at compile time and reaching dynamo's recompile limit raises
        # FailOnRecompileLimitHit instead of running the frame eagerly unannounced.
        _compiled = torch.compile(_slice_eager, dynamic=True, fullgraph=True)
    return _compiled


def _rows_dynamic(tensor: Tensor) -> Tensor:
    """A view of `tensor` for the compiled slice: its rows dynamic, its other dims static.

    Q-T226.1: under `dynamic=True` dynamo makes every dim symbolic, the 3 of the means and the 4
    of the quaternions too, and inductor then turns the quaternion norm over the last dim into
    reduction kernels; with the row width static, the norm fuses into the pointwise kernels.
    Only N changes at growth events, so this adds no recompilation, and the eager slice is
    untouched: the compiled slice changes by rounding only (U-28)."""
    import torch._dynamo  # loaded by torch.compile anyway

    view = tensor.view_as(tensor)
    for dim in range(1, view.dim()):
        torch._dynamo.mark_static(view, dim)
    return view


def _compile_failures() -> tuple[type[Exception], ...]:
    """The exceptions Q-13 counts as a failed compilation: dynamo's and its backends' errors, and
    reaching the recompile limit, which is a plain Exception, not a dynamo one."""
    from torch._dynamo import exc as dynamo_exc  # loaded by torch.compile anyway

    # Looked up by name, as #51 asks: on a torch without the class, the TorchDynamoException
    # fallback still works instead of failing with an AttributeError, and Q-13's detecting test
    # is what reports that the limit is no longer caught there.
    limit_hit = getattr(dynamo_exc, "FailOnRecompileLimitHit", None)
    failures: tuple[type[Exception], ...] = (dynamo_exc.TorchDynamoException,)
    return failures if limit_hit is None else (*failures, limit_hit)


def temporal_slice(
    means: Tensor,
    times: Tensor,
    log_durations: Tensor,
    velocities: Tensor,
    log_scales: Tensor,
    quats: Tensor,
    opacity_logits: Tensor,
    t: float | Tensor,
    *,
    impl: _Impl = "auto",
) -> TemporalSlice:
    """The temporal slice of a model at normalized time t (§2.4 "Kernels", D-50; Q-12).

    ``impl`` takes ``PerfConfig.slice_impl`` (Q-13): "eager" is plain PyTorch; "compiled" is the
    same body through torch.compile with dynamic shapes and without graph breaks, raising if
    compilation fails; "auto" compiles on CUDA tensors and stays eager on CPU, falling back to
    eager with one warning if compilation fails, which includes reaching dynamo's recompile
    limit; "cuda" is reserved for D-50's custom operator, which isn't built: B-1 put the slice
    under 10% of an iteration (#226), so it raises NotImplementedError.
    """
    global _auto_fell_back
    if impl not in _IMPLS:
        raise ValueError(f"impl must be one of {_IMPLS}, got {impl!r}")
    if impl == "cuda":
        # Q-13, D-50: the custom CUDA operator is written only if B-1 shows the slice above 10% of
        # an iteration; on kitchen it took 1.2% at most (#226), so it isn't.
        raise NotImplementedError(
            "the CUDA temporal slice is not built (D-50): B-1 measured the temporal slice at "
            "1.2 % of a training iteration at most, under D-50's 10 % (§3.4 B-1 in the "
            "Calibration data of docs/clean-room/questions.md, #226); use impl='compiled' or "
            "'eager'"
        )
    args = (means, times, log_durations, velocities, log_scales, quats, opacity_logits)
    t_tensor = _as_time(t, times)
    # Q-13: "auto" stays eager on CPU and after a failed compilation.
    if impl == "eager" or (impl == "auto" and (times.device.type != "cuda" or _auto_fell_back)):
        out = _slice_eager(*args, t_tensor)
        return TemporalSlice(*out)
    # D-50: dynamo gives nn.Parameter inputs static shapes, so a model's stored tensors would
    # recompile the slice at every growth event; views of them are plain tensors with dynamic
    # shapes, and gradients still reach the parameters (#45, source log #37). Q-T226.1: only
    # their rows are dynamic.
    views = tuple(_rows_dynamic(arg) for arg in args)
    if impl == "compiled":
        out = _compiled_slice()(*views, t_tensor)
    else:
        try:
            out = _compiled_slice()(*views, t_tensor)
        except _compile_failures() as exc:  # Q-13: only "auto" falls back
            _auto_fell_back = True
            warnings.warn(
                f"torch.compile failed for the temporal slice; using the eager slice ({exc})",
                UserWarning,
                stacklevel=2,
            )
            out = _slice_eager(*args, t_tensor)
    return TemporalSlice(*out)
