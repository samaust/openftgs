# SPDX-License-Identifier: Apache-2.0
"""TrainConfig and its sub-configs, with the defaults of §1.5; every field records its provenance
tag. Validation on construction (D-3; Q-11) and the JSON form the checkpoint manifest stores
(§2.2; Q-10).

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
    points_per_frame: int = 1_000  # B, D-24; F·B must not exceed GrowthConfig.n_max
    init_opacity: float = 0.1  # D-5
    scale_knn: int = 3  # L: 3DGS §5.1
    scale_floor: float = 1e-7  # D-25

    _prefix: ClassVar[str] = "init."

    def __post_init__(self) -> None:
        _check_types(self)
        # Q-11
        _check_count(self, "points_per_frame", "scale_knn")
        _check_open_unit(self, "init_opacity")
        _check_positive(self, "scale_floor")


@dataclass
class LossConfig:  # E7–E9
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
    mode: Literal["gradient", "score", "none"] = "gradient"  # D-18
    start: int = 500  # D-18
    stop: int = 15_000  # D-18
    grad_threshold: float = 2e-4  # τ_pos; L: 3DGS §5.2
    split_scale: float = 0.01  # × S: larger primitives split, smaller ones clone; L: 3DGS §5.2
    n_max: int = 3_000_000  # D-44; 500,000 for the † rows (D-39)
    score_rate: float = 0.05  # mode "score"; L: 3DGS-MCMC §3.6

    _prefix: ClassVar[str] = "growth."

    def __post_init__(self) -> None:
        _check_types(self)
        # Q-11
        _check_window(self)
        _check_positive(self, "grad_threshold", "split_scale")
        _check_count(self, "n_max")
        _check_half_open_unit(self, "score_rate")


@dataclass
class RelocConfig:  # E10, E11, §1.3 Relocation
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
    slice_impl: Literal["auto", "eager", "compiled", "cuda"] = "auto"  # D-50
    packed: bool = True  # gsplat default until B-2 calibrates (D-57)
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
        _check_types(self)
        # Q-11
        _check_non_negative(self, "prefetch")
        _check_count(self, "log_every", "images_per_step")


@dataclass
class TrainConfig:
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
        _check_types(self)
        if self.sh_degree not in (0, 1, 2, 3):  # D-3
            _fail(self, "sh_degree", "must be one of 0, 1, 2, 3")
        # Q-11
        if not all(0.0 <= c <= 1.0 for c in self.background):
            _fail(self, "background", "components must lie in [0, 1]")
        _check_positive(self, "scene_scale_factor")


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """TrainConfig as JSON-ready values, nested by sub-config (§2.2 manifest; Q-10)."""
    # Q-10: nested objects, tuples as lists, None as null.
    result: dict[str, Any] = _to_json(dataclasses.asdict(cfg))
    return result


def config_from_dict(data: Mapping[str, Any]) -> TrainConfig:
    """The inverse of config_to_dict; missing keys take their defaults (Q-10).

    Raises ValueError for an unknown key, naming its dotted path, and for invalid values.
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
    """The dotted keys whose values differ between a checkpoint's configuration and the one a
    resume is given, sorted, leaving out the PerfConfig fields that change no number and so may
    differ (Q-134): an empty list means the resume may go on with `given`."""
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
    """The dotted path of a field, from its class's prefix ("growth.n_max")."""
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
