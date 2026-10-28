# SPDX-License-Identifier: Apache-2.0
"""FreeTimeGaussians: the nine stored tensors, their activations and at_time().

Also initialization from init points (§1.3 steps 7–9), the exchange model in both directions, and
E5's rotation and covariance.

Spec: §1.1, §1.3 Initialization, §2.2 ftgs-if, E5, E14. Band: Core model (§2.1).
"""

import importlib.metadata
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, get_args

import torch
from torch import Tensor

from openftgs._legacy import with_current_loss
from openftgs._stages import stage
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
_SliceImpl = Literal["auto", "eager", "compiled", "cuda"]  # PerfConfig.slice_impl (D-50, Q-13)
_MIN_QUAT_NORM = 1e-8  # §1.1 validity rules
_OPACITY_EPS = 1e-7  # §2.2 ftgs-if: opacity clamped to [1e-7, 1 − 1e-7] on export and import
_KNN_CHUNK = 1 << 24  # distances held at once by the 3-NN search (rows × points)


# E5 (Q-20): public so the render backends and the split of E17 share one definition.
def rotation_matrix(quats: Tensor) -> Tensor:
    """Compute the rotation matrices R(q/‖q‖) of quaternions in (w, x, y, z) order (E5).

    The quaternions are normalized first, so their length doesn't matter.

    Parameters
    ----------
    quats
        Quaternions q, ``[..., 4]`` floating point, in the order (w, x, y, z), of any nonzero
        norm.

    Returns
    -------
    torch.Tensor
        The rotation matrices, ``[..., 3, 3]``, in the dtype of ``quats``.

    Notes
    -----
    Q-20: public, so that the render backends and the split of E17 share one definition.
    """
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
    """Compute the covariances Σ = R(q)·S·Sᵀ·R(q)ᵀ with S = diag(a) (E5).

    R is :func:`rotation_matrix` of the normalized quaternions.

    Parameters
    ----------
    quats
        Quaternions q, ``[..., 4]`` floating point, in the order (w, x, y, z), of any nonzero
        norm.
    scales
        The activated scales a = exp(â), ``[..., 3]``, in world units.

    Returns
    -------
    torch.Tensor
        The covariances, ``[..., 3, 3]``, symmetric, in world units squared.

    Notes
    -----
    Q-20: public, so that the render backends and the split of E17 share one definition.
    """
    rs = rotation_matrix(quats) * scales[..., None, :]  # R·diag(a)
    return rs @ rs.transpose(-1, -2)


# Q-20: the return type of at_time, the temporal slice (Q-12) plus the SH coefficients.
@dataclass(frozen=True)
class GaussiansAtTime:
    """The per-time arrays of §1.1 at one time, with the slice's opacity, E8 terms and cull mask.

    :meth:`FreeTimeGaussians.at_time` returns it: the fields of
    :class:`~openftgs.motion.TemporalSlice` plus the SH coefficients. N is the number of
    primitives, and every tensor is float32 on the model's device.

    Parameters
    ----------
    means_t
        The moved positions μx(t), ``[N, 3]``, in world units (E1).
    opacities_t
        The effective opacities σ_eff = σ·σt, ``[N]``, in [0, 1] (E3).
    quats_n
        The unit quaternions q/‖q‖, ``[N, 4]``, in the order (w, x, y, z).
    scales
        The scales exp(â), ``[N, 3]``, in world units.
    sh
        The SH coefficients concat(``sh0``, ``shN``), ``[N, (L′ + 1)², 3]``, cut to the requested
        degree L′.
    opacities
        The base opacities σ = sigmoid(ô), ``[N]``, in [0, 1].
    reg_terms
        The per-row terms σ·sg[σt] of E8, ``[N]``.
    passes_cull
        ``[N]`` bool, True where σ_eff ≥ 1/255, the rows that survive the opacity cull (D-32).

    Notes
    -----
    Q-12 defines the slice's fields, and Q-20 this type.
    """

    means_t: Tensor  # [N, 3] μx(t), E1
    opacities_t: Tensor  # [N] σ_eff = σ·σt, E3
    quats_n: Tensor  # [N, 4] q/‖q‖
    scales: Tensor  # [N, 3] exp(â)
    sh: Tensor  # [N, (L′+1)², 3] concat(sh0, shN), cut to the requested degree
    opacities: Tensor  # [N] σ = sigmoid(ô)
    reg_terms: Tensor  # [N] σ·sg[σt], the per-row terms of E8
    passes_cull: Tensor  # [N] bool, σ_eff ≥ 1/255 (D-32)


def _fail(reason: str) -> NoReturn:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(f"FreeTimeGaussians: {reason}")


def _check_sh_degree(sh_degree: object, allowed: tuple[int, ...] = _SH_DEGREES) -> int:
    """Return ``sh_degree`` if it is an int (not a bool) among `allowed` (D-3)."""
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree not in allowed:
        _fail(f"sh_degree must be one of {allowed}, got {sh_degree!r}")
    return sh_degree


def _knn_mean_distance(positions: Tensor, k: int, floor: float) -> Tensor:
    """Average each point's distance to its min(k, n − 1) nearest others, floored (D-25, Q-21).

    Brute force in float64 by chunks of rows, with differences rather than the matrix-product
    form, so points far from the origin keep their precision. A lone point gets the floor.
    """
    n = int(positions.shape[0])
    if n < 2:
        return torch.full((n,), floor, dtype=torch.float64, device=positions.device)
    k = min(k, n - 1)
    x = positions.to(torch.float64)
    out = torch.empty(n, dtype=torch.float64, device=x.device)
    rows = max(1, _KNN_CHUNK // n)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        d = torch.cdist(x[start:stop], x, compute_mode="donot_use_mm_for_euclid_dist")
        local = torch.arange(stop - start, device=x.device)
        d[local, local + start] = math.inf  # a point is not its own neighbour
        out[start:stop] = torch.topk(d, k, dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(floor)


def _canonical_sign(quats: Tensor) -> Tensor:
    """Flip each quaternion's sign so that its first nonzero component is positive (Q-24).

    So w ≥ 0, and where w = 0 the next nonzero component decides.
    """
    first = torch.argmax((quats != 0).to(torch.int8), dim=-1, keepdim=True)
    lead = torch.gather(quats, -1, first)
    # + 0.0 turns the −0.0 that a sign flip makes of zero components into +0.0.
    return torch.where(lead < 0, -quats, quats) + 0.0


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


class FreeTimeGaussians(torch.nn.Module):
    """The nine stored tensors of §1.1, registered as parameters under their own names.

    A :class:`torch.nn.Module` whose parameters are the stored forms of §1.1: ``means``,
    ``times``, ``log_durations``, ``velocities``, ``log_scales``, ``quats``, ``opacity_logits``,
    ``sh0`` and ``shN``. :meth:`at_time` gives the per-time arrays a renderer consumes, and
    :meth:`physical` the activated values. The constructor copies the caller's tensors, so the
    parameters get their own storage, and checks §1.1's validity rules. The public attributes
    ``sh_degree``, ``time`` and ``time_unit`` hold the constructor's arguments (Q-20).

    Parameters
    ----------
    params
        The nine stored tensors by name (:attr:`PARAM_NAMES`), each float32, finite and on one
        device, with N rows: ``means`` ``[N, 3]`` in world units; ``times`` ``[N]`` in
        normalized time; ``log_durations`` ``[N]``, log s with s in normalized time;
        ``velocities`` ``[N, 3]`` in world units per unit of normalized time; ``log_scales``
        ``[N, 3]``, log a; ``quats`` ``[N, 4]`` in the order (w, x, y, z), each of norm above
        1e-8; ``opacity_logits`` ``[N]``, logit σ; ``sh0`` ``[N, 1, 3]``; ``shN``
        ``[N, (L + 1)² − 1, 3]``. With a single timestamp (τ₀ = τ₁), ``times`` and
        ``velocities`` must be 0 (D-71).
    sh_degree
        The SH degree L, 0, 1, 2 or 3 (D-3).
    time
        The time normalization of the training timestamps (E14).
    time_unit
        The unit of raw time τ, as the scene manifest names it; "s" by default (Q-17).

    Raises
    ------
    ValueError
        If ``params`` doesn't hold exactly the nine names, a tensor breaks §1.1's validity
        rules (its shape, float32, one device, finite values, quaternion norms, D-71's
        single-timestamp rule), or ``sh_degree`` is not an int among 0 to 3.

    Notes
    -----
    §2.1 fixes the signature; Q-17 adds ``time_unit``. Every rejection is a plain ValueError
    (Q-26). A model starts with :attr:`slice_impl` "auto" (Q-66).

    Examples
    --------
    Build a model of four primitives at SH degree 3 and slice it at normalized time 0.5:

    >>> import torch
    >>> from openftgs.model import FreeTimeGaussians
    >>> from openftgs.types import TimeNormalizer
    >>> n = 4
    >>> params = {
    ...     "means": torch.randn(n, 3),
    ...     "times": torch.linspace(0.0, 1.0, n),
    ...     "log_durations": torch.full((n,), -2.0),
    ...     "velocities": torch.zeros(n, 3),
    ...     "log_scales": torch.full((n, 3), -3.0),
    ...     "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
    ...     "opacity_logits": torch.zeros(n),
    ...     "sh0": torch.zeros(n, 1, 3),
    ...     "shN": torch.zeros(n, 15, 3),
    ... }
    >>> model = FreeTimeGaussians(params, sh_degree=3, time=TimeNormalizer(0.0, 10.0))
    >>> model.num_gaussians
    4
    >>> model.at_time(0.5).means_t.shape
    torch.Size([4, 3])
    """

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
    """The names of the nine stored tensors of §1.1, in order; checkpoints and the optimizer use
    them."""

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
        self._slice_impl: _SliceImpl = "auto"  # Q-66
        # §2.1: the caller's tensors are plain data, so the parameters get their own storage.
        self._set_params({k: v.detach().clone() for k, v in params.items()})

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
        """Initialize a model from init points by steps 7–9 of §1.3 (§1.3 Initialization).

        The points come from ``roma_init`` or are supplied through the public interface. At most
        F·B points are kept in total, a seeded uniform subset in file order, with F frames of
        the sequence (1 for a single timestamp) and B = ``cfg.points_per_frame``. Missing
        velocities become 0, missing durations Δt_f, and missing frame ids the nearest slot
        k·Δt_f. Scales are isotropic: the log of the mean distance to the ``cfg.scale_knn``
        nearest neighbours in the same frame (3 by default), floored at ``cfg.scale_floor``.
        Every primitive starts with opacity ``cfg.init_opacity``, the identity rotation,
        ``sh0`` from its color and ``shN`` = 0.

        Parameters
        ----------
        points
            The initial points: positions ``[N, 3]`` float32 in world units, colors ``[N, 3]``
            float32 in [0, 1], timestamps ``[N]`` float64 in raw τ and, optionally, velocities
            in world units per unit of τ, durations in raw τ units and frame ids.
        time
            The time normalization of the training timestamps (E14).
        cfg
            The initialization settings: B, the opacity, the neighbour count and the scale
            floor.
        frame_dt
            Δt_f, the frame interval in normalized time (E14), finite and positive; it gives F
            and the missing durations.
        generator
            The generator that draws the subset when N exceeds F·B; it isn't used otherwise.
        sh_degree
            The SH degree L, 0, 1, 2 or 3 (D-3); 3 by default.
        n_max
            N_max, the largest number of primitives; None skips the F·B check.

        Returns
        -------
        openftgs.model.FreeTimeGaussians
            The model, with at most F·B primitives, on the device of ``points.positions``.

        Raises
        ------
        ValueError
            If ``points`` breaks ftgs-init's rules, ``sh_degree`` is not an int among 0 to 3,
            ``frame_dt`` is not finite and positive, or F·B exceeds ``n_max``.

        Notes
        -----
        D-24 and D-74: the cap F·B; D-23, D-21 and D-47: the defaults for velocities,
        durations and frame ids; D-25: the scales; D-5: step 9's values; D-71: velocities are 0
        with a single timestamp. Q-16 adds ``sh_degree`` and ``n_max``; Q-21 fixes F, the
        nearest slot and the subset; Q-22 the validation of the points; Q-27 the unit
        conversions (E14).
        """
        validate_init_points(points)  # Q-22
        sh_degree = _check_sh_degree(sh_degree)
        if not (math.isfinite(frame_dt) and frame_dt > 0.0):
            _fail(f"frame_dt must be finite and positive, got {frame_dt!r}")
        # §1.3 step 7: F frames of the sequence, 1 for a single timestamp; Q-21: round half up.
        # D-74: the total cap F·B, for points from every source.
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
        if time.tau1 == time.tau0:
            # §1.4 "Time", D-71: velocities are frozen at 0 when T_τ = 0 (t ≡ 0 already, Q-9).
            velocities = torch.zeros_like(velocities)
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
        """Build a model from exchange-model contents (§2.2 ftgs-if).

        The contents are validated, then converted to the stored forms in float64: time centers
        and durations to normalized time, velocities per unit of normalized time, the logs of
        durations and scales, the rotations normalized, and the opacity clamped to
        [1e-7, 1 − 1e-7] before its logit. The model keeps the exchange model's SH degree, time
        span and time unit.

        Parameters
        ----------
        exchange
            The exchange model, physical values in raw time units.
        device
            The device of the parameters; "cuda" by default.

        Returns
        -------
        openftgs.model.FreeTimeGaussians
            The model, its parameters float32 on ``device``.

        Raises
        ------
        ValueError
            If ``exchange`` breaks ftgs-if's rules, or its values break §1.1's validity rules.

        Notes
        -----
        Q-22 and Q-24: the validation and the rotations; Q-27: E14's conversions; Q-9: time
        centers outside the training span are accepted without a warning; Q-17: the time unit.
        """
        validate_exchange(exchange)
        arrays = exchange.arrays
        time = TimeNormalizer(exchange.tau0, exchange.tau1)
        f64 = torch.float64
        with warnings.catch_warnings():
            # E14 through TimeNormalizer (Q-27): time centers outside the training span are
            # legitimate (Q-9), so to_t's warning doesn't apply to them.
            warnings.simplefilter("ignore", UserWarning)
            times = time.to_t(arrays["time_center"].to(f64))
        opacity = arrays["opacity"].to(f64).clamp(_OPACITY_EPS, 1.0 - _OPACITY_EPS)
        rotation = arrays["rotation"]
        sh = arrays["sh"]
        params = {
            "means": arrays["position"],
            "times": times,
            "log_durations": torch.log(durations_to_normalized(arrays["duration"].to(f64), time)),
            "velocities": velocities_to_normalized(arrays["velocity"].to(f64), time),
            "log_scales": torch.log(arrays["scale"].to(f64)),
            # Q-24: the importer normalizes the rotation.
            "quats": rotation / torch.linalg.vector_norm(rotation, dim=-1, keepdim=True),
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
        """Load a model from a checkpoint directory or an ftgs-if file (§2.1).

        A directory is read as an ftgs-checkpoint, with the SH degree of its stored
        configuration; any other path as an ftgs-if file (§2.2). A checkpoint saved before
        revision 549o, whose configuration names the old perceptual backbone "vgg" or "alex",
        is read as :func:`openftgs.io.load_checkpoint` reads it, with 549o's loss defaults in
        memory only, since a model doesn't depend on the loss it trained with (Q-247, Q-271).

        Parameters
        ----------
        path
            A checkpoint directory or an ftgs-if file.
        device
            The device of the parameters; "cuda" by default.

        Returns
        -------
        openftgs.model.FreeTimeGaussians
            The model, its parameters float32 on ``device``.

        Raises
        ------
        ValueError
            If the checkpoint or file can't be read or breaks its format's rules, or the
            checkpoint holds more primitives than its stored configuration's N_max (§1.1).

        Notes
        -----
        Q-25: :func:`openftgs.io.load_checkpoint` rejects a checkpoint above its N_max. Q-26:
        the CLI reports this ValueError as invalid input.
        """
        path = Path(path)
        if not path.is_dir():
            return cls.from_exchange(read_exchange(path), device=device)
        state = load_checkpoint(path)
        # Q-271, Q-247: the model doesn't depend on the loss it trained with.
        config = config_from_dict(with_current_loss(state.config))
        return cls(
            {k: v.to(device) for k, v in state.params.items()},
            sh_degree=config.sh_degree,
            time=state.time,
            time_unit=state.time_unit,
        )

    def physical(self) -> dict[str, Tensor]:
        """Return the activated values in raw time units under the ftgs-if keys (§2.2).

        They are exactly the arrays :meth:`to_exchange` writes, detached, on the model's
        device.

        Returns
        -------
        dict of str to torch.Tensor
            ``position`` ``[N, 3]`` float32, μx in world units; ``time_center`` ``[N]``
            float64, τ₀ + μt·T_τ in raw τ; ``duration`` ``[N]`` float64, s·T_τ in raw τ units;
            ``velocity`` ``[N, 3]`` float32, v/T_τ in world units per unit of τ; ``scale``
            ``[N, 3]`` float32, exp(â); ``rotation`` ``[N, 4]`` float32, the unit quaternion
            (w, x, y, z) with its first nonzero component positive; ``opacity`` ``[N]``
            float32, sigmoid(ô) computed in float64 and clamped to [1e-7, 1 − 1e-7]; ``sh``
            ``[N, (L + 1)², 3]`` float32. With a single timestamp, durations and velocities use
            a time unit of 1 instead of T_τ (D-70).

        Notes
        -----
        Q-20 names the keys; Q-24 fixes the rotations' sign; Q-27: E14's conversions.
        """
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
        """Return the exchange model of §2.2 ftgs-if, physical values in raw time units.

        Returns
        -------
        openftgs.types.ExchangeModel
            The arrays of :meth:`physical`, with the model's SH degree, τ₀, τ₁ and time unit,
            and the producer "openftgs" followed by the installed version.

        Notes
        -----
        The exchange model's background is left at its default, black (D-34, D-84).
        """
        return ExchangeModel(
            arrays=self.physical(),
            sh_degree=self.sh_degree,
            tau0=self.time.tau0,
            tau1=self.time.tau1,
            time_unit=self.time_unit,
            producer=_producer(),
        )

    def _set_params(self, params: Mapping[str, Tensor]) -> None:
        """Check the §1.1 validity rules, then register the nine tensors as parameters.

        The parameters share the tensors' storage.
        """
        self._check_params(params)
        for name in self.PARAM_NAMES:
            setattr(self, name, torch.nn.Parameter(params[name].detach()))

    # Q-55: public, so that a writer that works in place (relocation) can check §1.1's rules.
    def validate(self) -> None:
        """Check the §1.1 validity rules on the current parameters (§1.1).

        The rules are the constructor's: the nine tensors' shapes, float32, one device, finite
        values, quaternion norms above 1e-8, and zero times and velocities with a single
        timestamp (D-71). It reads values, so it synchronizes on a GPU (events allow it, §2.4
        rule 3).

        Raises
        ------
        ValueError
            If a rule fails.

        Notes
        -----
        Q-55: public, so that a writer that works in place (relocation) can check §1.1's rules.
        """
        self._check_params({name: getattr(self, name).detach() for name in self.PARAM_NAMES})

    def _check_params(self, params: Mapping[str, Tensor]) -> None:
        """Check the §1.1 validity rules, shared by the constructor, resize_ and validate (Q-55)."""
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
        # D-71: §1.4 "Time": with T_τ = 0, t ≡ 0 and velocities are frozen at 0, so time centers
        # and velocities must be 0 (an exchange file can't carry anything else).
        if self.time.tau1 == self.time.tau0 and (
            bool(params["times"].any()) or bool(params["velocities"].any())
        ):
            _fail("with a single timestamp (T_τ = 0), times and velocities must be 0 (§1.4)")

    # Q-66 (owner answer): public, so the trainer can set PerfConfig.slice_impl on any model.
    @property
    def slice_impl(self) -> _SliceImpl:
        """The temporal-slice implementation that at_time runs, "auto" by default (D-50).

        One of ``PerfConfig.slice_impl``'s values, "auto", "eager", "compiled" and "cuda", with
        the meanings of :func:`openftgs.motion.temporal_slice`; the trainer sets it from the
        configuration.

        Raises
        ------
        ValueError
            On setting a value outside those four.

        Notes
        -----
        Q-13: what each value does; Q-66: a public attribute, so the trainer can set it on any
        model.
        """
        return self._slice_impl

    @slice_impl.setter
    def slice_impl(self, value: _SliceImpl) -> None:
        if value not in get_args(_SliceImpl):
            _fail(f"slice_impl must be one of {get_args(_SliceImpl)}, got {value!r}")
        self._slice_impl = value

    @property
    def num_gaussians(self) -> int:
        """The number of primitives N, the rows of every stored tensor (§2.1).

        It changes only through :meth:`resize_`.
        """
        return int(self.means.shape[0])

    def at_time(self, t: float, *, sh_degree: int | None = None) -> GaussiansAtTime:
        """Compute the per-time arrays at normalized time t (§1.1).

        The positions and opacities come from :func:`openftgs.motion.temporal_slice`, run with
        :attr:`slice_impl`; the SH coefficients are cut to ``sh_degree`` bands. Outside [0, 1]
        the motion extrapolates linearly (E1).

        Parameters
        ----------
        t
            The normalized time.
        sh_degree
            The SH degree L′ to keep, an int from 0 to the model's degree L; None, the
            default, keeps L.

        Returns
        -------
        openftgs.model.GaussiansAtTime
            The per-time arrays, float32 on the model's device, differentiable with respect to
            the parameters.

        Raises
        ------
        ValueError
            If ``sh_degree`` is not an int from 0 to L.

        Notes
        -----
        Q-20 adds ``sh_degree`` and the return type; D-50 and Q-66: the slice's
        implementation.
        """
        degree = self.sh_degree if sh_degree is None else sh_degree
        degree = _check_sh_degree(degree, tuple(range(self.sh_degree + 1)))
        # §3.4: the slice's stages wrap the call, never the compiled function's body (D-50).
        with stage("slice"):
            with stage("slice.kernel"):
                s = temporal_slice(
                    self.means,
                    self.times,
                    self.log_durations,
                    self.velocities,
                    self.log_scales,
                    self.quats,
                    self.opacity_logits,
                    t,
                    impl=self._slice_impl,
                )
            with stage("slice.sh_concat"):
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
        """Keep the rows a bool mask selects, in their order, then append new rows last (§1.3).

        This is step 4 of §1.3 Growth. The result is revalidated against §1.1's rules; on error
        the model is unchanged. The nine parameters are replaced by new
        ``torch.nn.Parameter`` objects, so an optimizer over them must be resized too.

        Parameters
        ----------
        keep
            A bool mask ``[N]`` on the model's device, True for the rows to keep; None keeps
            every row.
        append
            The rows to append, under exactly the nine names, each with the trailing shape,
            dtype and device of its stored tensor (``[M, 3]`` float32 for ``means``, and so on);
            None appends nothing.

        Raises
        ------
        ValueError
            If ``keep`` or ``append`` breaks these rules, or the result breaks §1.1's rules.

        Notes
        -----
        Q-23: ``keep`` is a mask only, so it can't reorder rows, and nothing is moved between
        devices silently.
        """
        # Q-23: keep is a bool mask [N], so it can't reorder rows; append matches shape, dtype
        # and device; nothing is moved between devices silently.
        n = self.num_gaussians
        device = self.means.device
        if keep is not None and (keep.dtype != torch.bool or tuple(keep.shape) != (n,)):
            _fail(f"keep must be a bool mask of shape [{n}], got {keep.dtype} {list(keep.shape)}")
        if keep is not None and keep.device != device:
            _fail(f"keep must be on the model's device {device}, got {keep.device}")
        if append is not None and set(append) != set(self.PARAM_NAMES):
            _fail(f"append must hold exactly {self.PARAM_NAMES}, got {sorted(append)}")
        params: dict[str, Tensor] = {}
        for name in self.PARAM_NAMES:
            value = getattr(self, name).detach()
            if keep is not None:
                value = value[keep]
            if append is not None:
                extra = append[name]
                if extra.dim() != value.dim() or extra.shape[1:] != value.shape[1:]:
                    _fail(f"append[{name!r}] must have rows of shape {list(value.shape[1:])}")
                if extra.dtype != value.dtype:
                    _fail(f"append[{name!r}] must be {value.dtype}, got {extra.dtype}")
                if extra.device != device:
                    _fail(
                        f"append[{name!r}] must be on the model's device {device}, "
                        f"got {extra.device}"
                    )
                value = torch.cat([value, extra])
            params[name] = value
        self._set_params(params)
