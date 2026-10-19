# SPDX-License-Identifier: Apache-2.0
"""TrainConfig: §1.5 defaults, validation (U-26's sh_degree part; Q-11) and the JSON form
the checkpoint manifest stores (§2.2; Q-10).

Expected defaults are copied from the config blocks of spec §2.1 and §2.4.
"""

import json
from typing import Any

import pytest

from openftgs.config import (
    GrowthConfig,
    InitConfig,
    LossConfig,
    OptimConfig,
    PerfConfig,
    RelocConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
    resume_conflicts,
)

# Dotted path → default, from spec §2.1 (openftgs.config) and §2.4 (PerfConfig).
DEFAULTS: dict[str, Any] = {
    "sh_degree": 3,
    "background": (0.0, 0.0, 0.0),
    "scene_scale_factor": 1.1,
    "seed": 0,
    "init.points_per_frame": 1_000,
    "init.init_opacity": 0.1,
    "init.scale_knn": 3,
    "init.scale_floor": 1e-7,
    "loss.lambda_img": 0.8,
    "loss.lambda_ssim": 0.2,
    "loss.lambda_perc": 0.01,
    "loss.lambda_reg": 1e-2,
    "loss.reg_until": None,
    "loss.ssim_window": 11,
    "loss.ssim_sigma": 1.5,
    "loss.ssim_padding": "valid",
    "loss.perc_net": "vgg",
    "growth.mode": "gradient",
    "growth.start": 500,
    "growth.stop": 15_000,
    "growth.grad_threshold": 2e-4,
    "growth.split_scale": 0.01,
    "growth.n_max": 3_000_000,
    "growth.score_rate": 0.05,
    "reloc.enabled": True,
    "reloc.every": 100,
    "reloc.start": 500,
    "reloc.stop": 25_000,
    "reloc.lambda_g": 0.5,
    "reloc.lambda_o": 0.5,
    "reloc.grad_quantile": 0.99,
    "reloc.dead_opacity": 0.005,
    "reloc.reset_moved_moments": True,
    "reloc.copy_position": "sample",
    "reloc.prune_opacity": None,
    "optim.iterations": 30_000,
    "optim.betas": (0.9, 0.999),
    "optim.eps": 1e-15,
    "optim.means_lr": (1.6e-4, 1.6e-6),
    "optim.velocities_lr": (1.6e-4, 1.6e-6),
    "optim.times_lr": (1.6e-4, 1.6e-6),
    "optim.log_durations_lr": 5e-3,
    "optim.log_scales_lr": 5e-3,
    "optim.quats_lr": 1e-3,
    "optim.opacity_logits_lr": 5e-2,
    "optim.sh0_lr": 2.5e-3,
    "optim.shN_lr": 1.25e-4,
    "optim.sh_interval": 1000,
    "perf.slice_impl": "auto",
    "perf.packed": True,
    "perf.precull": "internal",
    "perf.frame_cache": "auto",
    "perf.prefetch": 8,
    "perf.log_every": 100,
    "perf.perc_precision": "tf32",
    "perf.adam": "fused",
    "perf.images_per_step": 1,
    "perf.deterministic": False,
}


def _get(cfg: object, path: str) -> Any:
    for part in path.split("."):
        cfg = getattr(cfg, part)
    return cfg


def test_every_default_matches_the_spec() -> None:
    """§1.5, §2.1, §2.4: every field's default, and no field the spec doesn't list."""
    cfg = TrainConfig()
    for path, expected in DEFAULTS.items():
        value = _get(cfg, path)
        assert value == expected, path
        assert type(value) is type(expected), path
    fields: set[str] = set()
    for name, value in vars(cfg).items():
        if hasattr(value, "__dataclass_fields__"):
            fields.update(f"{name}.{sub}" for sub in vars(value))
        else:
            fields.add(name)
    assert fields == set(DEFAULTS)


def _with(path: str, value: Any) -> dict[str, Any]:
    """A partial config dict setting one dotted path."""
    head, _, tail = path.partition(".")
    return {head: {tail: value}} if tail else {head: value}


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_u26_sh_degree_accepted(degree: int) -> None:
    """U-26 (D-3): sh_degree ∈ {0, 1, 2, 3} is accepted."""
    assert TrainConfig(sh_degree=degree).sh_degree == degree
    assert config_from_dict({"sh_degree": degree}).sh_degree == degree


@pytest.mark.parametrize("degree", [-1, 4, True, 2.0, "3", None])
def test_u26_sh_degree_rejected(degree: Any) -> None:
    """U-26 (D-3): other values are rejected, by the constructor and the JSON reader."""
    with pytest.raises(ValueError, match="sh_degree"):
        TrainConfig(sh_degree=degree)
    with pytest.raises(ValueError, match="sh_degree"):
        config_from_dict({"sh_degree": degree})


# Allowed values of every Literal field, from spec §2.1 and §2.4.
LITERALS: dict[str, tuple[str, ...]] = {
    "loss.ssim_padding": ("valid", "same"),
    "loss.perc_net": ("vgg", "alex"),
    "growth.mode": ("gradient", "score", "none"),
    "reloc.copy_position": ("sample", "stack"),
    "perf.slice_impl": ("auto", "eager", "compiled", "cuda"),
    "perf.precull": ("internal", "compact"),
    "perf.frame_cache": ("auto", "gpu", "mmap", "off"),
    "perf.perc_precision": ("tf32", "ieee", "bf16"),
    "perf.adam": ("fused", "foreach", "selective"),
}


@pytest.mark.parametrize(("path", "allowed"), LITERALS.items())
def test_literal_fields_take_exactly_their_spec_values(path: str, allowed: tuple[str, ...]) -> None:
    """§2.1, §2.4: a Literal field takes its listed values and nothing else (Q-10)."""
    for value in allowed:
        assert _get(config_from_dict(_with(path, value)), path) == value
    for bad in ("not-a-choice", allowed[0].upper(), 1, None):
        with pytest.raises(ValueError, match=path.replace(".", r"\.")):
            config_from_dict(_with(path, bad))


# Q-11: (dotted path, an accepted boundary value, a rejected value)
RANGES: list[tuple[str, Any, Any]] = [
    ("init.points_per_frame", 1, 0),
    ("init.scale_knn", 1, 0),
    ("init.init_opacity", 0.999, 1.0),
    ("init.init_opacity", 1e-9, 0.0),
    ("init.scale_floor", 1e-30, 0.0),
    ("loss.lambda_img", 0.0, -1e-9),
    ("loss.lambda_ssim", 0.0, -0.1),
    ("loss.lambda_perc", 0.0, -0.1),
    ("loss.lambda_reg", 0.0, -1e-3),
    ("loss.reg_until", 0, -1),
    ("loss.ssim_window", 1, 0),
    ("loss.ssim_window", 7, 8),
    ("loss.ssim_sigma", 1e-3, 0.0),
    ("growth.grad_threshold", 1e-9, 0.0),
    ("growth.split_scale", 1e-9, -0.01),
    ("growth.n_max", 1, 0),
    ("growth.score_rate", 1.0, 1.01),
    ("growth.score_rate", 1e-9, 0.0),
    ("reloc.every", 1, 0),
    ("reloc.lambda_g", 0.0, -0.5),
    ("reloc.lambda_o", 0.0, -0.5),
    ("reloc.grad_quantile", 1.0, 1.5),
    ("reloc.grad_quantile", 1e-9, 0.0),
    ("reloc.dead_opacity", 1e-9, 0.0),
    ("reloc.dead_opacity", 0.999, 1.0),
    ("reloc.prune_opacity", 0.005, 1.0),
    ("optim.iterations", 1, 0),
    ("optim.sh_interval", 1, 0),
    ("optim.betas", (0.0, 0.0), (0.9, 1.0)),
    ("optim.betas", (0.5, 0.5), (-0.1, 0.999)),
    ("optim.eps", 1e-30, 0.0),
    ("optim.means_lr", (1e-9, 1e-9), (1.6e-4, 0.0)),
    ("optim.velocities_lr", (1e-9, 1e-9), (0.0, 1.6e-6)),
    ("optim.times_lr", (1e-9, 1e-9), (-1.0, 1.6e-6)),
    ("optim.log_durations_lr", 1e-9, 0.0),
    ("optim.log_scales_lr", 1e-9, 0.0),
    ("optim.quats_lr", 1e-9, 0.0),
    ("optim.opacity_logits_lr", 1e-9, 0.0),
    ("optim.sh0_lr", 1e-9, 0.0),
    ("optim.shN_lr", 1e-9, 0.0),
    ("perf.prefetch", 0, -1),
    ("perf.log_every", 1, 0),
    ("perf.images_per_step", 1, 0),
    ("background", (1.0, 1.0, 1.0), (0.0, 0.0, 1.5)),
    ("background", (0.0, 0.5, 0.0), (-0.1, 0.0, 0.0)),
    ("scene_scale_factor", 1e-9, 0.0),
]


@pytest.mark.parametrize(("path", "good", "bad"), RANGES)
def test_q11_ranges(path: str, good: Any, bad: Any) -> None:
    """Q-11: one accepted boundary and one rejected value per range."""
    config_from_dict(_with(path, good))
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        config_from_dict(_with(path, bad))


@pytest.mark.parametrize("section", ["growth", "reloc"])
def test_q11_window_start_must_not_exceed_stop(section: str) -> None:
    """Q-11: a growth or relocation window with start > stop is rejected; start = stop is not."""
    config_from_dict({section: {"start": 700, "stop": 700}})
    with pytest.raises(ValueError, match=rf"{section}\.start"):
        config_from_dict({section: {"start": 701, "stop": 700}})


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        ("seed", 1.5),
        ("seed", True),
        ("optim.eps", "1e-15"),
        ("optim.eps", float("nan")),
        ("optim.betas", (0.9,)),
        ("optim.betas", 0.9),
        ("reloc.enabled", 1),
        ("perf.packed", "true"),
        ("loss.lambda_reg", None),
        ("init.points_per_frame", 1000.0),
        ("optim.eps", 10**400),
    ],
)
def test_q10_field_types_are_checked(path: str, bad: Any) -> None:
    """Q-10: ints for int fields, finite numbers for float fields, booleans for bool fields."""
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        config_from_dict(_with(path, bad))


def _every_field_changed() -> TrainConfig:
    return TrainConfig(
        sh_degree=1,
        background=(1.0, 0.5, 0.25),
        scene_scale_factor=1.3,
        seed=7,
        init=InitConfig(points_per_frame=500, init_opacity=0.2, scale_knn=5, scale_floor=1e-6),
        loss=LossConfig(
            lambda_img=0.7,
            lambda_ssim=0.3,
            lambda_perc=0.0,
            lambda_reg=1e-3,
            reg_until=10_000,
            ssim_window=7,
            ssim_sigma=1.0,
            ssim_padding="same",
            perc_net="alex",
        ),
        growth=GrowthConfig(
            mode="score",
            start=100,
            stop=200,
            grad_threshold=3e-4,
            split_scale=0.02,
            n_max=500_000,
            score_rate=0.1,
        ),
        reloc=RelocConfig(
            enabled=False,
            every=50,
            start=100,
            stop=250,
            lambda_g=0.25,
            lambda_o=0.75,
            grad_quantile=0.95,
            dead_opacity=0.01,
            reset_moved_moments=False,
            copy_position="stack",
            prune_opacity=0.005,
        ),
        optim=OptimConfig(
            iterations=300,
            betas=(0.8, 0.99),
            eps=1e-8,
            means_lr=(1e-3, 1e-5),
            velocities_lr=(2e-4, 2e-6),
            times_lr=(3e-4, 3e-6),
            log_durations_lr=1e-2,
            log_scales_lr=1e-2,
            quats_lr=2e-3,
            opacity_logits_lr=1e-1,
            sh0_lr=5e-3,
            shN_lr=2.5e-4,
            sh_interval=50,
        ),
        perf=PerfConfig(
            slice_impl="eager",
            packed=False,
            precull="compact",
            frame_cache="off",
            prefetch=0,
            log_every=10,
            perc_precision="ieee",
            adam="foreach",
            images_per_step=2,
            deterministic=True,
        ),
    )


def test_every_field_changed_really_changes_every_field() -> None:
    """The round-trip fixture differs from the §2.1 defaults in every field."""
    changed, default = _every_field_changed(), TrainConfig()
    for path in DEFAULTS:
        assert _get(changed, path) != _get(default, path), path


@pytest.mark.parametrize("cfg", [TrainConfig(), _every_field_changed()], ids=["default", "changed"])
def test_train_config_json_round_trip(cfg: TrainConfig) -> None:
    """Added (§2.2 manifest, Q-10): through JSON text the config comes back equal."""
    text = json.dumps(config_to_dict(cfg), allow_nan=False)
    back = config_from_dict(json.loads(text))
    assert back == cfg
    for path in DEFAULTS:
        assert type(_get(back, path)) is type(_get(cfg, path)), path


def test_config_dict_is_nested_by_sub_config() -> None:
    """Q-10: sub-configs are nested objects, tuples are lists, None is null."""
    data = config_to_dict(TrainConfig())
    assert data["growth"]["n_max"] == 3_000_000
    assert data["optim"]["betas"] == [0.9, 0.999]
    assert data["background"] == [0.0, 0.0, 0.0]
    assert data["loss"]["reg_until"] is None
    assert set(data) == {
        "sh_degree", "background", "scene_scale_factor", "seed",
        "init", "loss", "growth", "reloc", "optim", "perf",
    }  # fmt: skip


def test_partial_dict_takes_defaults() -> None:
    """Q-10: missing keys take their defaults."""
    cfg = config_from_dict({"growth": {"n_max": 500_000}, "seed": 3})
    expected = TrainConfig(seed=3, growth=GrowthConfig(n_max=500_000))
    assert cfg == expected
    assert config_from_dict({}) == TrainConfig()


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"sh_degre": 3}, "sh_degre"),
        ({"growth": {"nmax": 10}}, r"growth\.nmax"),
        ({"perf": {"slice": "eager"}}, r"perf\.slice"),
    ],
)
def test_unknown_key_is_rejected_with_its_dotted_path(data: dict[str, Any], path: str) -> None:
    """Q-10: an unknown key raises ValueError naming its dotted path."""
    with pytest.raises(ValueError, match=path):
        config_from_dict(data)


def test_sub_config_must_be_an_object() -> None:
    """Q-10: a sub-config is a nested JSON object."""
    with pytest.raises(ValueError, match="growth"):
        config_from_dict({"growth": 3})


# What a resume may change (Q-134 as answered; Q-153).

EXACT_PERF = {
    "slice_impl": "eager",
    "packed": False,
    "precull": "compact",
    "frame_cache": "mmap",
    "prefetch": 3,
    "log_every": 7,
    "deterministic": True,
}


def _perf(**fields: Any) -> TrainConfig:
    return TrainConfig(perf=PerfConfig(**fields))


def test_resume_conflicts_of_one_configuration_are_none() -> None:
    """Q-153: a configuration has no conflict with itself."""
    assert resume_conflicts(TrainConfig(), TrainConfig()) == []


@pytest.mark.parametrize(("name", "value"), sorted(EXACT_PERF.items()))
def test_resume_conflicts_allow_each_exact_performance_setting(name: str, value: Any) -> None:
    """Q-134 as answered, §2.4 rule 2: the PerfConfig fields that change no number may differ
    on a resume (D-50, D-51, D-56, D-57; log_every only reads)."""
    assert resume_conflicts(TrainConfig(), _perf(**{name: value})) == []


def test_resume_conflicts_allow_all_exact_performance_settings_at_once() -> None:
    """Q-134 as answered: the seven fields may differ together."""
    assert resume_conflicts(TrainConfig(), _perf(**EXACT_PERF)) == []


@pytest.mark.parametrize(
    ("given", "key"),
    [
        (_perf(perc_precision="ieee"), "perf.perc_precision"),  # D-52
        (_perf(adam="foreach"), "perf.adam"),  # D-53
        (_perf(images_per_step=2), "perf.images_per_step"),  # D-54
        (TrainConfig(seed=1), "seed"),
        (TrainConfig(optim=OptimConfig(iterations=20)), "optim.iterations"),
        (TrainConfig(background=(1.0, 1.0, 1.0)), "background"),
    ],
)
def test_resume_conflicts_name_the_fields_that_must_match(given: TrainConfig, key: str) -> None:
    """Q-134 as answered: perc_precision, adam and images_per_step change results, so they must
    match, as must every field outside PerfConfig; each conflict is named by its dotted path."""
    assert resume_conflicts(TrainConfig(), given) == [key]
    assert resume_conflicts(given, TrainConfig()) == [key]


def test_resume_conflicts_are_sorted_and_skip_the_exact_fields() -> None:
    """Q-153: several conflicts come back sorted, exact fields left out."""
    given = TrainConfig(
        seed=5,
        loss=LossConfig(lambda_reg=0.0),
        perf=PerfConfig(adam="selective", frame_cache="off"),
    )
    assert resume_conflicts(TrainConfig(), given) == ["loss.lambda_reg", "perf.adam", "seed"]
