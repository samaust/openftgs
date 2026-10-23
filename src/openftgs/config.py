# SPDX-License-Identifier: Apache-2.0
"""TrainConfig and its sub-configs, with the defaults of §1.5.

Every field records its provenance tag (§0): [P §x] stated by the paper, [P→3DGS] taken from
3DGS through the paper's "same settings as 3DGS", [L: source] standard Gaussian-splatting
practice, [D-n] a decision of the spec. Each config validates its fields on construction (D-3;
Q-11), and :func:`config_to_dict` and :func:`config_from_dict` give the JSON form the checkpoint
manifest stores (§2.2; Q-10). The field names are the hyperparameter names the harness uses
(§3.3).

Spec: §1.5, §2.1, §2.2, §2.4. Band: Foundation (§2.1).
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

__all__ = [
    "GrowthConfig",
    "InitConfig",
    "LossConfig",
    "OptimConfig",
    "PerfConfig",
    "RelocConfig",
    "TrainConfig",
    "config_from_dict",
    "config_to_dict",
    "resume_conflicts",
]


@dataclass
class InitConfig:  # §1.3 Initialization
    """Settings of the model's initialization from the initial points (§1.3 "Initialization").

    Parameters
    ----------
    points_per_frame
        The initial budget B, in points per frame of the sequence, ≥ 1; F·B must not exceed
        ``GrowthConfig.n_max`` for a sequence of F frames [D-24].
    init_opacity
        The initial opacity of every primitive, in (0, 1) [D-5].
    scale_knn
        How many nearest neighbors' mean distance sets each initial scale, ≥ 1
        [L: 3DGS §5.1].
    scale_floor
        The floor of that mean distance, in world units, > 0 [D-25].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). An int is accepted for a float field and converted; a bool is refused for a
        numeric field.
    """

    points_per_frame: int = 1_000  # B, D-24; F·B must not exceed GrowthConfig.n_max
    init_opacity: float = 0.1  # D-5
    scale_knn: int = 3  # L: 3DGS §5.1
    scale_floor: float = 1e-7  # D-25

    _prefix: ClassVar[str] = "init."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_count(self, "points_per_frame", "scale_knn")
        _check_open_unit(self, "init_opacity")
        _check_positive(self, "scale_floor")


@dataclass
class LossConfig:  # E7–E9
    """Weights and forms of the training loss (E7–E9).

    Parameters
    ----------
    lambda_img
        The weight λ_img of the image loss, the L1 difference (D-7), ≥ 0 [P §3.3].
    lambda_ssim
        The weight λ_ssim of the SSIM loss 1 − SSIM, ≥ 0 [P §3.3].
    lambda_perc
        The weight λ_perc of the perceptual loss, LPIPS, ≥ 0 [P §3.3].
    lambda_reg
        The weight λ_reg of the 4D regularization, ≥ 0; 0 turns it off [P §3.3].
    reg_until
        The last iteration with the 4D regularization, ≥ 0, or None for the whole run [D-10].
    ssim_window
        The side of SSIM's Gaussian window, in pixels, odd and ≥ 1 [D-8].
    ssim_sigma
        The standard deviation σ of that window, in pixels, > 0 [D-8].
    ssim_padding
        ``"valid"`` evaluates SSIM only where the window fits inside the image, ``"same"`` pads
        with zeros [D-8].
    perc_net
        The perceptual loss's LPIPS backbone, VGG or AlexNet [D-9].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). An int is accepted for a float field and converted; a bool is refused for a
        numeric field.
    """

    lambda_img: float = 0.8  # P §3.3
    lambda_ssim: float = 0.2  # P §3.3
    lambda_perc: float = 0.01  # P §3.3
    lambda_reg: float = 1e-2  # P §3.3
    reg_until: int | None = None  # D-10: last iteration with L_reg; None = whole run
    ssim_window: int = 11  # D-8
    ssim_sigma: float = 1.5  # D-8
    ssim_padding: Literal["valid", "same"] = "valid"  # D-8
    perc_net: Literal["vgg", "alex"] = "vgg"  # D-9

    _prefix: ClassVar[str] = "loss."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_non_negative(self, "lambda_img", "lambda_ssim", "lambda_perc", "lambda_reg")
        if self.reg_until is not None:
            _check_non_negative(self, "reg_until")
        _check_count(self, "ssim_window")
        if self.ssim_window % 2 == 0:
            _fail(self, "ssim_window", "must be odd")
        _check_positive(self, "ssim_sigma")


@dataclass
class GrowthConfig:  # §1.3 Growth and count-control modes
    """Settings of the growth of the primitive count (§1.3 "Growth", "Count-control modes").

    Growth events happen at the multiples of ``RelocConfig.every`` from ``start`` to ``stop``,
    both included.

    Parameters
    ----------
    mode
        How the count grows: ``"gradient"`` by 3DGS clone and split, ``"score"`` by primitives
        drawn by their sampling score, ``"none"`` not at all [D-18].
    start
        The first iteration of the growth window, ≤ ``stop`` [D-18].
    stop
        The last iteration of the growth window [D-18].
    grad_threshold
        τ_pos, the mean NDC gradient (E10's g) above which a primitive grows by clone or
        split, > 0 [L: 3DGS §5.2].
    split_scale
        The clone/split boundary, in units of the spatial scale S: a candidate whose largest
        scale exceeds it splits, a smaller one is cloned, > 0 [L: 3DGS §5.2].
    n_max
        N_max, the cap on the number of primitives, ≥ 1; 500,000 for the † rows (D-39)
        [D-44].
    score_rate
        In mode ``"score"``, the fraction by which each event grows the count, in (0, 1]
        [L: 3DGS-MCMC §3.6].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). An int is accepted for a float field and converted; a bool is refused for a
        numeric field.
    """

    mode: Literal["gradient", "score", "none"] = "gradient"  # D-18
    start: int = 500  # D-18
    stop: int = 15_000  # D-18
    grad_threshold: float = 2e-4  # τ_pos; L: 3DGS §5.2
    split_scale: float = 0.01  # × S: larger primitives split, smaller ones clone; L: 3DGS §5.2
    n_max: int = 3_000_000  # D-44; 500,000 for the † rows (D-39)
    score_rate: float = 0.05  # mode "score"; L: 3DGS-MCMC §3.6

    _prefix: ClassVar[str] = "growth."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_window(self)
        _check_positive(self, "grad_threshold", "split_scale")
        _check_count(self, "n_max")
        _check_half_open_unit(self, "score_rate")


@dataclass
class RelocConfig:  # E10, E11, §1.3 Relocation
    """Settings of the periodic relocation of dead primitives (E10, E11, §1.3 "Relocation").

    Relocation events happen at the multiples of ``every`` from ``start`` to ``stop``, both
    included.

    Parameters
    ----------
    enabled
        Whether relocation runs; False only in the "w/o periodic relocation" ablation
        [P §4.2].
    every
        The period of relocation events, in iterations, ≥ 1; growth events share it [P §3.3].
    start
        The first iteration of the relocation window, ≤ ``stop`` [D-17].
    stop
        The last iteration of the relocation window [D-17].
    lambda_g
        The weight λ_g of the normalized gradient in the sampling score ρ (E10), ≥ 0
        [P §3.3].
    lambda_o
        The weight λ_o of the base opacity in the sampling score ρ (E10), ≥ 0 [P §3.3].
    grad_quantile
        The quantile of the live primitives' gradients that normalizes them in the score, in
        (0, 1] [D-13].
    dead_opacity
        The base opacity below which a primitive is dead and relocated, in (0, 1); compared as
        a float32 logit [D-14].
    reset_moved_moments
        Whether the Adam moments of the target and moved rows are zeroed after a move [D-16].
    copy_position
        ``"sample"`` draws each moved copy's position from its target's Gaussian, ``"stack"``
        leaves the copies on the target (U-13) [D-15].
    prune_opacity
        The opacity below which primitives are pruned at growth events, in (0, 1), or None for
        no pruning; set, to 0.005, only in the ablation with relocation off (Q-71)
        [L: 3DGS §5.2].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). An int is accepted for a float field and converted; a bool is refused for a
        numeric field.
    """

    enabled: bool = True  # False only in the "w/o periodic relocation" ablation
    every: int = 100  # P §3.3; growth events share this period
    start: int = 500  # D-17
    stop: int = 25_000  # D-17
    lambda_g: float = 0.5  # P §3.3
    lambda_o: float = 0.5  # P §3.3
    grad_quantile: float = 0.99  # D-13
    dead_opacity: float = 0.005  # D-14; compared as a float32 logit
    reset_moved_moments: bool = True  # D-16
    # D-15; "stack" leaves copies on the target (U-13)
    copy_position: Literal["sample", "stack"] = "sample"
    prune_opacity: float | None = None  # ablation only: 0.005 when relocation is off

    _prefix: ClassVar[str] = "reloc."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_count(self, "every")
        _check_window(self)
        _check_non_negative(self, "lambda_g", "lambda_o")
        _check_half_open_unit(self, "grad_quantile")
        _check_open_unit(self, "dead_opacity")
        if self.prune_opacity is not None:
            _check_open_unit(self, "prune_opacity")


@dataclass
class OptimConfig:  # E12, E13, §1.3 Optimizer
    """Adam's settings and the learning rates of the nine stored tensors (E12, E13).

    A pair is a rate at the first and at the last iteration, annealed between them; a single
    number is a constant rate. Every rate is > 0 (§1.3 "Optimizer and learning rates").

    Parameters
    ----------
    iterations
        T_it, the number of training iterations, ≥ 1; 30,000 for any sequence length
        [P §3.3; D-35].
    betas
        Adam's β₁ and β₂, each in [0, 1) [P→3DGS].
    eps
        Adam's ε, > 0 [P→3DGS].
    means_lr
        The rates of ``means``, in units of the spatial scale S, decaying exponentially (E13)
        [P→3DGS].
    velocities_lr
        The rates of ``velocities``, in units of S/Δt_f, annealed by E12; the form is the
        paper's, the values D-19's [P; D-19].
    times_lr
        The rates of ``times``, in normalized time, decaying exponentially [D-20].
    log_durations_lr
        The constant rate of ``log_durations`` [D-20].
    log_scales_lr
        The constant rate of ``log_scales`` [P→3DGS].
    quats_lr
        The constant rate of ``quats`` [P→3DGS].
    opacity_logits_lr
        The constant rate of ``opacity_logits`` [P→3DGS].
    sh0_lr
        The constant rate of ``sh0`` [P→3DGS].
    shN_lr
        The constant rate of ``shN`` [P→3DGS].
    sh_interval
        Every how many iterations one more SH band becomes active (E13), ≥ 1
        [L: 3DGS §7.1].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). An int is accepted for a float field and converted, a list for a pair; a bool is
        refused for a numeric field.
    """

    iterations: int = 30_000  # P §3.3; D-35
    betas: tuple[float, float] = (0.9, 0.999)  # P→3DGS
    eps: float = 1e-15  # P→3DGS
    means_lr: tuple[float, float] = (1.6e-4, 1.6e-6)  # × S; E13
    velocities_lr: tuple[float, float] = (1.6e-4, 1.6e-6)  # × S/Δt_f; E12, D-19
    times_lr: tuple[float, float] = (1.6e-4, 1.6e-6)  # D-20
    log_durations_lr: float = 5e-3  # D-20
    log_scales_lr: float = 5e-3  # P→3DGS
    quats_lr: float = 1e-3  # P→3DGS
    opacity_logits_lr: float = 5e-2  # P→3DGS
    sh0_lr: float = 2.5e-3  # P→3DGS
    shN_lr: float = 1.25e-4  # P→3DGS
    sh_interval: int = 1000  # L: 3DGS §7.1

    _prefix: ClassVar[str] = "optim."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_count(self, "iterations", "sh_interval")
        if not all(0.0 <= b < 1.0 for b in self.betas):
            _fail(self, "betas", "must lie in [0, 1)")
        _check_positive(
            self,
            "eps",
            "means_lr",
            "velocities_lr",
            "times_lr",
            "log_durations_lr",
            "log_scales_lr",
            "quats_lr",
            "opacity_logits_lr",
            "sh0_lr",
            "shN_lr",
        )


@dataclass
class PerfConfig:  # §2.4 Settings
    """Performance settings (§2.4 "Settings").

    The choices marked approximate change the results; a resume may change only the settings
    that change no number (:func:`resume_conflicts`).

    Parameters
    ----------
    slice_impl
        The implementation of the fused temporal slice: ``"eager"`` plain PyTorch,
        ``"compiled"`` through torch.compile, ``"cuda"`` the custom operator, or ``"auto"``,
        which compiles on CUDA and stays eager on the CPU [D-50].
    packed
        Whether gsplat runs in packed mode; on by default, gsplat's default, which B-2 kept
        [D-57; Q-187].
    precull
        Where the opacity cull of D-32 happens: ``"internal"`` inside gsplat's projection,
        ``"compact"`` by compaction before the backend [D-32, D-57].
    frame_cache
        Where the uint8 frame cache lives: ``"gpu"``, ``"mmap"`` memory-mapped, ``"off"``
        none, or ``"auto"``, on the GPU when it fits and memory-mapped otherwise [D-51].
    prefetch
        How many images ahead of the sampler the prefetcher loads, ≥ 0 [D-51].
    log_every
        Every how many iterations the loss terms are read and their means reported, ≥ 1
        (§2.4 rule 3; Q-69); untagged in §2.4.
    perc_precision
        The precision of the perceptual loss: ``"tf32"``, ``"ieee"`` float32, or ``"bf16"``,
        which is approximate [D-52].
    adam
        Adam's implementation: PyTorch's ``"fused"`` or ``"foreach"``, or gsplat's
        ``"selective"``, which is approximate [D-53].
    images_per_step
        Images per training step, ≥ 1; more than 1 is approximate and uses several GPUs
        [D-54].
    deterministic
        Whether cuDNN autotuning is off and PyTorch's deterministic kernels are used where it
        has them; RoMa initialization always runs this way [D-56].

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (Q-10,
        Q-11). A bool is refused for a numeric field.
    """

    slice_impl: Literal["auto", "eager", "compiled", "cuda"] = "auto"  # D-50
    packed: bool = True  # gsplat's default, which B-2 kept (D-57, Q-187)
    precull: Literal["internal", "compact"] = "internal"  # D-32, D-57
    frame_cache: Literal["auto", "gpu", "mmap", "off"] = "auto"  # D-51
    prefetch: int = 8
    log_every: int = 100
    perc_precision: Literal["tf32", "ieee", "bf16"] = "tf32"  # bf16 is approximate (D-52)
    adam: Literal["fused", "foreach", "selective"] = "fused"  # selective is approximate (D-53)
    images_per_step: int = 1  # > 1 is approximate and uses several GPUs (D-54)
    # cuDNN autotuning off; deterministic kernels where PyTorch has them;
    # RoMa initialization always runs this way (D-56)
    deterministic: bool = False

    _prefix: ClassVar[str] = "perf."

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (Q-10, Q-11)."""
        _check_types(self)
        # Q-11
        _check_non_negative(self, "prefetch")
        _check_count(self, "log_every", "images_per_step")


@dataclass
class TrainConfig:
    """The full training configuration: global settings and one sub-config per part (§2.1).

    Its JSON form, nested by sub-config, is what a checkpoint's manifest stores
    (:func:`config_to_dict`, :func:`config_from_dict`; Q-10). The defaults are those of §1.5.

    Parameters
    ----------
    sh_degree
        The SH degree L, one of 0, 1, 2 and 3 [D-3].
    background
        The RGB background the images are trained against, three numbers in [0, 1]; black by
        default [D-34].
    scene_scale_factor
        The factor of the spatial scale S, which is this factor times the largest distance of
        a training camera center from their centroid, > 0 [D-27].
    seed
        The run's one seed, from which training draws all its randomness; untagged in §2.1.
    init
        The initialization settings (§1.3 "Initialization").
    loss
        The loss settings (E7–E9).
    growth
        The growth settings (§1.3 "Growth").
    reloc
        The relocation settings (E10, E11).
    optim
        The optimizer settings (E12, E13).
    perf
        The performance settings (§2.4).

    Raises
    ------
    ValueError
        If a field has the wrong type or lies outside its range, naming its dotted path (D-3;
        Q-10, Q-11). An int is accepted for a float field and converted, a list for a tuple; a
        bool is refused for a numeric field.
    """

    sh_degree: int = 3  # D-3
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)  # D-34
    scene_scale_factor: float = 1.1  # D-27
    seed: int = 0
    init: InitConfig = field(default_factory=InitConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    reloc: RelocConfig = field(default_factory=RelocConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)  # §2.4

    _prefix: ClassVar[str] = ""

    def __post_init__(self) -> None:
        """Check the fields' types and ranges (D-3; Q-10, Q-11)."""
        _check_types(self)
        if self.sh_degree not in (0, 1, 2, 3):  # D-3
            _fail(self, "sh_degree", "must be one of 0, 1, 2, 3")
        # Q-11
        if not all(0.0 <= c <= 1.0 for c in self.background):
            _fail(self, "background", "components must lie in [0, 1]")
        _check_positive(self, "scene_scale_factor")


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Convert a TrainConfig to JSON-ready values, nested by sub-config (Q-10).

    Tuples become lists and None stays None, null in JSON. This is the form the checkpoint
    manifest stores (§2.2), and :func:`config_from_dict` reads it back.

    Parameters
    ----------
    cfg
        The configuration.

    Returns
    -------
    dict
        One key per field of :class:`TrainConfig`, each sub-config a nested dict, as in
        ``{"sh_degree": 3, …, "growth": {"mode": "gradient", …, "n_max": 3000000, …}, …}``.

    Examples
    --------
    >>> from openftgs.config import GrowthConfig, TrainConfig, config_to_dict
    >>> data = config_to_dict(TrainConfig(growth=GrowthConfig(n_max=500_000)))
    >>> data["growth"]["n_max"]
    500000
    >>> data["background"]
    [0.0, 0.0, 0.0]
    """
    # Q-10: nested objects, tuples as lists, None as null.
    result: dict[str, Any] = _to_json(dataclasses.asdict(cfg))
    return result


def config_from_dict(data: Mapping[str, Any]) -> TrainConfig:
    """Build a TrainConfig from its JSON form, the inverse of :func:`config_to_dict` (Q-10).

    Missing keys take their defaults, so a partial configuration sets only what it names. The
    values are checked as the constructors check them, so ``TrainConfig(...)`` and this
    function accept exactly the same values.

    Parameters
    ----------
    data
        The configuration nested by sub-config, as :func:`config_to_dict` writes it, in full or
        in part.

    Returns
    -------
    openftgs.config.TrainConfig
        The configuration.

    Raises
    ------
    ValueError
        For an unknown key, naming its dotted path; for a sub-config that isn't an object; for
        a value of the wrong type or outside its range (Q-11).

    Examples
    --------
    >>> from openftgs.config import config_from_dict
    >>> cfg = config_from_dict({"seed": 7, "growth": {"n_max": 500_000}})
    >>> cfg.growth.n_max, cfg.growth.stop
    (500000, 15000)
    """
    # Q-10: missing keys take their defaults; unknown keys are errors.
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(TrainConfig)
    for key, value in _reject_unknown_keys(TrainConfig, data).items():
        sub = hints[key]
        if dataclasses.is_dataclass(sub):
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be an object, got {value!r}")
            assert isinstance(sub, type)
            kwargs[key] = sub(**_reject_unknown_keys(sub, value))
        else:
            kwargs[key] = value
    return TrainConfig(**kwargs)


# Q-153: the one rule for Trainer.load_state_dict and ftgs train --resume.
def resume_conflicts(stored: TrainConfig, given: TrainConfig) -> list[str]:
    """List the settings that differ between a checkpoint's configuration and a resume's (Q-153).

    The comparison leaves out the PerfConfig fields that change no number and so may differ on
    resume: ``slice_impl``, ``packed``, ``precull``, ``frame_cache``, ``prefetch``,
    ``log_every`` and ``deterministic`` (Q-134). ``Trainer.load_state_dict`` and
    ``ftgs train --resume`` both decide by it.

    Parameters
    ----------
    stored
        The configuration the checkpoint holds.
    given
        The configuration the resume is given.

    Returns
    -------
    list of str
        The dotted keys whose values differ, such as ``"growth.n_max"``, sorted; an empty list
        means the resume may go on with ``given``.
    """
    a, b = _dotted(config_to_dict(stored)), _dotted(config_to_dict(given))
    differ = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    return sorted(differ - _MAY_DIFFER_ON_RESUME)


# ---------------------------------------------------------------------------------------------
# Private helpers

# Q-134 (as answered): the PerfConfig fields that change no number, which a resume may change.
_MAY_DIFFER_ON_RESUME = frozenset(
    {
        "perf.slice_impl",  # D-50, U-28
        "perf.packed",  # D-57, U-8
        "perf.precull",  # D-57, U-8
        "perf.frame_cache",  # D-51, U-30
        "perf.prefetch",  # D-51, U-30
        "perf.log_every",  # reads only (Q-69)
        "perf.deterministic",  # D-56
    }
)


def _dotted(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            out |= _dotted(value, f"{prefix}{key}.")
        else:
            out[f"{prefix}{key}"] = value
    return out


def _reject_unknown_keys(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    # Q-10: an unknown key is an error naming its dotted path.
    names = {f.name for f in dataclasses.fields(cls)}
    for key in data:
        if key not in names:
            raise ValueError(f"unknown configuration key {_path(cls, key)}")
    return dict(data)


def _path(obj: object, name: str) -> str:
    """Return the dotted path of a field, from its class's prefix ("growth.n_max")."""
    return f"{getattr(obj, '_prefix', '')}{name}"


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json(v) for v in value]
    return value


def _fail(obj: object, name: str, reason: str) -> None:
    raise ValueError(f"{_path(obj, name)} {reason}, got {getattr(obj, name)!r}")


def _check_types(obj: object) -> None:
    """Check each field against its annotation, normalizing ints to floats and lists to tuples."""
    # Q-10: runs on construction, so the constructor and config_from_dict accept the same values.
    hints = get_type_hints(type(obj))
    assert dataclasses.is_dataclass(obj)
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        setattr(obj, f.name, _coerce(obj, f.name, hints[f.name], value))


def _coerce(obj: object, name: str, hint: Any, value: Any) -> Any:
    path = _path(obj, name)
    origin = get_origin(hint)
    if origin in (Union, UnionType):
        args = get_args(hint)
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(obj, name, inner, value)
    if origin is Literal:
        allowed = get_args(hint)
        if not isinstance(value, str) or value not in allowed:
            names = ", ".join(map(repr, allowed))
            raise ValueError(f"{path} must be one of {names}, got {value!r}")
        return value
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(value, (tuple, list)) or len(value) != len(args):
            raise ValueError(f"{path} must be a sequence of {len(args)} numbers, got {value!r}")
        return tuple(_coerce(obj, name, a, v) for a, v in zip(args, value, strict=True))
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, hint):  # type: ignore[arg-type]
            raise ValueError(f"{path} must be a {getattr(hint, '__name__', hint)}, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number, got {value!r}")
        try:
            number = float(value)
        except OverflowError:
            raise ValueError(f"{path} must be a finite number, got {value!r}") from None
        if not math.isfinite(number):  # Q-10
            raise ValueError(f"{path} must be finite, got {value!r}")
        return number
    raise TypeError(f"unsupported annotation {hint!r} for {path}")  # pragma: no cover


def _values(obj: object, name: str) -> tuple[float, ...]:
    value = getattr(obj, name)
    return tuple(value) if isinstance(value, tuple) else (value,)


def _check_positive(obj: object, *names: str) -> None:
    for name in names:
        if not all(v > 0 for v in _values(obj, name)):
            _fail(obj, name, "must be positive")


def _check_non_negative(obj: object, *names: str) -> None:
    for name in names:
        if not all(v >= 0 for v in _values(obj, name)):
            _fail(obj, name, "must not be negative")


def _check_count(obj: object, *names: str) -> None:
    for name in names:
        if getattr(obj, name) < 1:
            _fail(obj, name, "must be at least 1")


def _check_open_unit(obj: object, *names: str) -> None:
    for name in names:
        if not 0.0 < getattr(obj, name) < 1.0:
            _fail(obj, name, "must lie in (0, 1)")


def _check_half_open_unit(obj: object, *names: str) -> None:
    for name in names:
        if not 0.0 < getattr(obj, name) <= 1.0:
            _fail(obj, name, "must lie in (0, 1]")


def _check_window(obj: GrowthConfig | RelocConfig) -> None:
    if obj.start > obj.stop:
        _fail(obj, "start", f"must not exceed stop ({obj.stop!r})")
