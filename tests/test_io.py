# SPDX-License-Identifier: Apache-2.0
"""The file formats of §2.2 in openftgs.io: ftgs-init, ftgs-if, ftgs-checkpoint, the render
request and renders, the scene manifest, and the PLY snapshot (D-30, D-73; U-22 and U-26 file
parts; Q-17, Q-18, Q-22, Q-24 to Q-26, Q-35, Q-56, Q-64).

Expected values come from the §2.2 tables (keys, dtypes, shapes, validation rules) and from
values worked by hand; files with invalid content are written with safetensors directly, so the
readers' validation is tested without the writers'.
"""

import dataclasses
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from builders import BAD_EXCHANGE_CASES, bad_exchange, exchange_arrays, exchange_metadata
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.config import TrainConfig, config_to_dict
from openftgs.io import (
    check_run_checkpoints,
    load_checkpoint,
    read_exchange,
    read_init_points,
    read_render_request,
    read_renders,
    read_scene_manifest,
    render_conventions,
    save_checkpoint,
    save_run_checkpoint,
    validate_exchange,
    validate_init_points,
    write_exchange,
    write_init_points,
    write_renders,
    write_scene_manifest,
)
from openftgs.types import (
    Camera,
    ExchangeModel,
    ImageRecord,
    InitPoints,
    Renders,
    TimeNormalizer,
    TrainerState,
)

F32 = torch.float32
F64 = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ftgs-init (§2.2; Q-22)


def _init_points(n: int = 5, *, optional: bool = True) -> InitPoints:
    g = _gen(n)
    return InitPoints(
        positions=torch.randn(n, 3, generator=g),
        colors=torch.rand(n, 3, generator=g),
        timestamps=torch.rand(n, generator=g, dtype=F64) * 10.0,
        velocities=torch.randn(n, 3, generator=g) if optional else None,
        durations=torch.rand(n, generator=g, dtype=F64) + 0.1 if optional else None,
        frame_ids=torch.randint(0, 3, (n,), generator=g) if optional else None,
    )


def _init_tensors(n: int = 5) -> dict[str, Tensor]:
    p = _init_points(n)
    return {
        "positions": p.positions,
        "colors": p.colors,
        "timestamps": p.timestamps,
        "velocities": p.velocities,  # type: ignore[dict-item]
        "durations": p.durations,  # type: ignore[dict-item]
        "frame_ids": p.frame_ids,  # type: ignore[dict-item]
    }


def _init_metadata(**override: str) -> dict[str, str]:
    meta = {"format": "ftgs-init", "version": "1", "time_unit": "s", "producer": "test"}
    meta.update(override)
    return meta


@pytest.mark.parametrize("optional", [True, False])
def test_ftgs_init_round_trip(tmp_path: Path, optional: bool) -> None:
    """§2.2 ftgs-init: every key, dtype and value survives write → read; absent optional keys
    stay absent."""
    points = _init_points(optional=optional)
    path = tmp_path / "init.safetensors"
    write_init_points(points, path, time_unit="frame", producer="unit test")
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        keys = set(f.keys())
    assert meta["format"] == "ftgs-init"
    assert meta["version"] == "1"
    assert meta["time_unit"] == "frame"
    assert meta["producer"] == "unit test"
    expected_keys = {"positions", "colors", "timestamps"}
    if optional:
        expected_keys |= {"velocities", "durations", "frame_ids"}
    assert keys == expected_keys
    back = read_init_points(path, time_unit="frame")
    for name in ("positions", "colors", "timestamps", "velocities", "durations", "frame_ids"):
        a, b = getattr(points, name), getattr(back, name)
        if a is None:
            assert b is None
        else:
            assert b.dtype == a.dtype
            assert torch.equal(a, b)


def _bad_init(name: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    t = _init_tensors()
    meta = _init_metadata()
    if name == "nan position":
        t["positions"][1, 2] = float("nan")
    elif name == "inf timestamp":
        t["timestamps"][0] = float("inf")
    elif name == "empty":
        t = {k: v[:0] for k, v in t.items()}
    elif name == "color above 1":
        t["colors"][0, 0] = 1.0001
    elif name == "color below 0":
        t["colors"][2, 1] = -1e-6
    elif name == "zero duration":
        t["durations"][3] = 0.0
    elif name == "negative duration":
        t["durations"][3] = -1.0
    elif name == "float32 timestamps":
        t["timestamps"] = t["timestamps"].float()
    elif name == "float64 positions":
        t["positions"] = t["positions"].double()
    elif name == "int32 frame ids":
        t["frame_ids"] = t["frame_ids"].int()
    elif name == "positions [N, 2]":
        t["positions"] = t["positions"][:, :2].contiguous()
    elif name == "colors of another N":
        t["colors"] = t["colors"][:4].contiguous()
    elif name == "missing colors":
        del t["colors"]
    elif name == "version 2":
        meta["version"] = "2"
    elif name == "other format":
        meta["format"] = "ftgs-if"
    elif name == "no version":
        del meta["version"]
    return t, meta


@pytest.mark.parametrize(
    "case",
    [
        "nan position",
        "inf timestamp",
        "empty",
        "color above 1",
        "color below 0",
        "zero duration",
        "negative duration",
        "float32 timestamps",
        "float64 positions",
        "int32 frame ids",
        "positions [N, 2]",
        "colors of another N",
        "missing colors",
        "version 2",
        "other format",
        "no version",
    ],
)
def test_ftgs_init_reader_rejects_invalid_files(tmp_path: Path, case: str) -> None:
    """§2.2 ftgs-init validation: values finite, N ≥ 1, colors in [0, 1], durations positive,
    the table's dtypes and shapes, required keys, format and major version (Q-22, Q-26)."""
    tensors, meta = _bad_init(case)
    path = tmp_path / "bad.safetensors"
    save_file(tensors, str(path), metadata=meta)
    with pytest.raises(ValueError):
        read_init_points(path)


def test_ftgs_init_reader_ignores_unknown_keys_and_accepts_minor_versions(tmp_path: Path) -> None:
    """§2.2 introduction: readers ignore unknown optional keys; only the major version counts."""
    tensors = _init_tensors()
    tensors["extra_tensor"] = torch.zeros(2)
    path = tmp_path / "init.safetensors"
    save_file(tensors, str(path), metadata=_init_metadata(version="1.3", note="hello"))
    back = read_init_points(path)
    assert torch.equal(back.positions, tensors["positions"])


def test_ftgs_init_reader_rejects_another_time_unit(tmp_path: Path) -> None:
    """Q-22: a file whose time_unit differs from the expected one is rejected."""
    path = tmp_path / "init.safetensors"
    write_init_points(_init_points(), path, time_unit="s", producer="test")
    read_init_points(path, time_unit="s")
    with pytest.raises(ValueError, match="time_unit"):
        read_init_points(path, time_unit="frame")


def test_ftgs_init_writer_adds_extra_metadata(tmp_path: Path) -> None:
    """Q-107: `metadata` keys are written as given beside §2.2's own keys, and the reader, which
    ignores unknown keys (§2.2 introduction), still reads the file."""
    points = _init_points()
    path = tmp_path / "init.safetensors"
    extra = {"roma_init": '{"seed":3}', "note": ""}
    write_init_points(points, path, time_unit="s", producer="test", metadata=extra)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    assert meta == {**_init_metadata(), **extra}
    back = read_init_points(path, time_unit="s")
    assert torch.equal(back.positions, points.positions)


@pytest.mark.parametrize(
    "extra",
    [
        {"format": "other"},
        {"version": "2"},
        {"time_unit": "frame"},
        {"producer": "someone else"},
        {"seed": 3},
        {"seed": None},
        {"flag": True},
        {7: "seven"},
    ],
    ids=["format", "version", "time_unit", "producer", "int", "none", "bool", "int-key"],
)
def test_ftgs_init_writer_refuses_reserved_or_non_string_metadata(
    tmp_path: Path, extra: dict[Any, Any]
) -> None:
    """Q-107: a reserved key (`format`, `version`, `time_unit`, `producer`) or a key or value
    that isn't a string raises ValueError, and no file is written."""
    path = tmp_path / "init.safetensors"
    with pytest.raises(ValueError, match="metadata"):
        write_init_points(_init_points(), path, time_unit="s", producer="test", metadata=extra)
    assert not path.exists()


def test_ftgs_init_writer_validates_before_writing(tmp_path: Path) -> None:
    """Q-22: invalid init points are refused, and no file is written."""
    p = _init_points()
    bad = InitPoints(p.positions, p.colors * 2.0, p.timestamps)
    path = tmp_path / "init.safetensors"
    with pytest.raises(ValueError):
        write_init_points(bad, path, time_unit="s", producer="test")
    assert not path.exists()
    with pytest.raises(ValueError):
        validate_init_points(bad)
    validate_init_points(p)


# ftgs-if (§2.2; U-22 file parts, U-26; Q-22, Q-24)


def _exchange(sh_degree: int = 3, **arrays: Tensor) -> ExchangeModel:
    a = exchange_arrays(sh_degree=sh_degree)
    a.update(arrays)
    return ExchangeModel(a, sh_degree, 2.0, 12.0, "s", "unit test")


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_ftgs_if_round_trip_for_each_sh_degree(tmp_path: Path, sh_degree: int) -> None:
    """U-26 (FTGS-IF round trip per L) and §2.2: every array, dtype and metadata field
    survives write → read bitwise."""
    model = ExchangeModel(exchange_arrays(sh_degree=sh_degree), sh_degree, 0.1, 9.7, "frame", "x")
    path = tmp_path / "model.safetensors"
    write_exchange(model, path)
    back = read_exchange(path)
    assert set(back.arrays) == set(model.arrays)
    for key, value in model.arrays.items():
        assert back.arrays[key].dtype == value.dtype
        assert torch.equal(back.arrays[key], value), key
    assert back.arrays["sh"].shape == (6, (sh_degree + 1) ** 2, 3)
    assert (back.sh_degree, back.tau0, back.tau1) == (sh_degree, 0.1, 9.7)
    assert (back.time_unit, back.producer) == ("frame", "x")


# The render_conventions entries of spec revision 549 (§2.2, Q-24); revision 549a adds D-73's.
_REVISION_549_CONVENTIONS = {
    "alpha_max": 0.999,
    "eps2d": 0.3,
    "near": 0.01,
    "far": 1e10,
    "alpha_skip": 1.0 / 255.0,
    "transmittance_stop": 1e-4,
    "pixel_center_offset": 0.5,
    "background": [0.0, 0.0, 0.0],
}


def test_ftgs_if_metadata_follows_section_2_2(tmp_path: Path) -> None:
    """§2.2 ftgs-if metadata: format, version, sh_degree, time_unit, tau0, tau1 (shortest
    round-trip repr), producer, and render_conventions as compact JSON from E6 and D-34 (Q-24),
    with D-73's Jacobian clamp margin, screen-extent rule and rasterize mode under Q-35's keys."""
    model = ExchangeModel(exchange_arrays(), 3, 0.1, 1.0 / 3.0, "s", "unit test")
    path = tmp_path / "model.safetensors"
    write_exchange(model, path)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    assert meta["format"] == "ftgs-if"
    assert meta["version"] == "1"
    assert meta["sh_degree"] == "3"
    assert meta["time_unit"] == "s"
    assert meta["tau0"] == "0.1"
    assert meta["tau1"] == repr(1.0 / 3.0)
    assert meta["producer"] == "unit test"
    conventions = json.loads(meta["render_conventions"])
    assert " " not in meta["render_conventions"]
    assert conventions == {
        **_REVISION_549_CONVENTIONS,
        "jacobian_clamp_margin": 0.15,
        "screen_extent_rule": "opacity_aware",
        "screen_extent_max": 3.33,
        "rasterize_mode": "classic",
    }


def test_render_conventions_is_the_block_write_exchange_writes(tmp_path: Path) -> None:
    """Q-204, D-73, D-84: render_conventions(background=…) is the block an exchange file with
    that background records, with Q-35's 12 entries; by default the background is black (D-34),
    and every call returns a fresh copy."""
    conventions = render_conventions()
    assert len(conventions) == 12
    assert conventions["background"] == [0.0, 0.0, 0.0]
    conventions["near"] = -1.0
    conventions["background"].append(9.0)
    assert render_conventions()["near"] == 0.01
    assert render_conventions()["background"] == [0.0, 0.0, 0.0]
    model = ExchangeModel(exchange_arrays(), 3, 0.1, 1.0, "s", "t", background=(1.0, 0.5, 0.0))
    path = tmp_path / "model.safetensors"
    write_exchange(model, path)
    with safe_open(str(path), framework="pt") as f:
        written = json.loads(f.metadata()["render_conventions"])
    assert written == render_conventions(background=(1.0, 0.5, 0.0))


@pytest.mark.parametrize(
    "background", [(0.0, 0.0), (0.0, 0.0, 1.5), (0.0, float("nan"), 0.0), (True, 0.0, 0.0)]
)
def test_render_conventions_refuses_a_background_outside_d84(background: tuple[float, ...]) -> None:
    """D-84: three numbers in [0, 1]."""
    with pytest.raises(ValueError, match="background"):
        render_conventions(background=background)


def test_ftgs_if_records_and_returns_the_exchange_models_background(tmp_path: Path) -> None:
    """D-84, U-22: render_conventions records the exchange model's `background`, black by
    default (D-34), and the reader returns it, so the round trip is exact for it too."""
    model = ExchangeModel(exchange_arrays(), 3, 0.1, 1.0, "s", "t", background=(1.0, 0.5, 0.0))
    path = tmp_path / "model.safetensors"
    write_exchange(model, path)
    with safe_open(str(path), framework="pt") as f:
        conventions = json.loads(f.metadata()["render_conventions"])
    assert conventions["background"] == [1.0, 0.5, 0.0]
    assert read_exchange(path).background == (1.0, 0.5, 0.0)
    write_exchange(ExchangeModel(exchange_arrays(), 3, 0.1, 1.0, "s", "t"), path)
    assert read_exchange(path).background == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "conventions",
    [None, "{}", '{"rasterize_mode": "classic"}'],
    ids=["absent", "empty", "other entries"],
)
def test_ftgs_if_reader_reads_black_without_a_recorded_background(
    tmp_path: Path, conventions: str | None
) -> None:
    """D-84, Q-157: a file with no render_conventions, or a block without `background`, reads
    as black, so files of other implementations still read."""
    path = tmp_path / "model.safetensors"
    meta = exchange_metadata()
    if conventions is not None:
        meta["render_conventions"] = conventions
    save_file(exchange_arrays(), str(path), metadata=meta)
    assert read_exchange(path).background == (0.0, 0.0, 0.0)


def test_ftgs_if_reader_reads_an_integer_background_as_floats(tmp_path: Path) -> None:
    """D-84: JSON integers are numbers, returned as floats."""
    path = tmp_path / "model.safetensors"
    meta = exchange_metadata(render_conventions='{"background": [0, 1, 0]}')
    save_file(exchange_arrays(), str(path), metadata=meta)
    background = read_exchange(path).background
    assert background == (0.0, 1.0, 0.0)
    assert all(type(c) is float for c in background)


@pytest.mark.parametrize(
    "background",
    [(1.5, 0.0, 0.0), (-0.1, 0.0, 0.0), (math.nan, 0.0, 0.0), (0.0, 0.0), (True, 0.0, 0.0)],
)
def test_validate_exchange_needs_a_background_in_the_unit_interval(background: object) -> None:
    """D-84 (Q-11's range): three finite numbers in [0, 1], not bools."""
    model = ExchangeModel(exchange_arrays(), 3, 0.1, 1.0, "s", "t", background=background)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="background"):
        validate_exchange(model)


@pytest.mark.parametrize("case", BAD_EXCHANGE_CASES)
def test_u22_ftgs_if_validator_rejects_invalid_files(tmp_path: Path, case: str) -> None:
    """U-22: the validator rejects NaN, durations ≤ 0, opacity outside (0, 1), wrong shapes and
    an unknown major version; and the other §2.2 rules (Q-24): dtypes, keys, ranges, N ≥ 1."""
    arrays, meta = bad_exchange(case)
    path = tmp_path / "bad.safetensors"
    save_file(arrays, str(path), metadata=meta)
    with pytest.raises(ValueError):
        read_exchange(path)


def test_ftgs_if_reader_ignores_unknown_keys_and_other_render_conventions(tmp_path: Path) -> None:
    """§2.2: unknown optional keys are ignored, and so are render_conventions' entries other than
    the background (Q-24, D-84); a unit rotation with w < 0 is accepted on read."""
    arrays = exchange_arrays()
    arrays["rotation"][0] = -arrays["rotation"][0]
    arrays["confidence"] = torch.ones(6)
    path = tmp_path / "model.safetensors"
    meta = exchange_metadata(version="1.2", render_conventions="{}", comment="hi")
    save_file(arrays, str(path), metadata=meta)
    back = read_exchange(path)
    assert set(back.arrays) == set(arrays) - {"confidence"}
    assert torch.equal(back.arrays["rotation"], arrays["rotation"])


def test_ftgs_if_reader_accepts_revision_549_render_conventions(tmp_path: Path) -> None:
    """§2.2 introduction, D-73: a file whose render_conventions holds only the eight entries of
    spec revision 549 (no clamp margin, extent rule or rasterize mode) still reads, its
    background returned (D-84) and its other entries ignored, so D-73 needs no version bump."""
    arrays = exchange_arrays()
    old_block = json.dumps(_REVISION_549_CONVENTIONS, separators=(",", ":"))
    path = tmp_path / "model.safetensors"
    save_file(arrays, str(path), metadata=exchange_metadata(render_conventions=old_block))
    back = read_exchange(path)
    assert set(back.arrays) == set(arrays)
    for key, value in arrays.items():
        assert torch.equal(back.arrays[key], value), key
    assert back.background == (0.0, 0.0, 0.0)  # the block's own entry


def test_ftgs_if_writer_validates_before_writing(tmp_path: Path) -> None:
    """Q-22: write_exchange refuses an invalid model and writes nothing; validate_exchange is the
    same check on in-memory data."""
    bad = _exchange(duration=torch.zeros(6, dtype=F64))
    path = tmp_path / "model.safetensors"
    with pytest.raises(ValueError):
        write_exchange(bad, path)
    assert not path.exists()
    with pytest.raises(ValueError):
        validate_exchange(bad)
    validate_exchange(_exchange())


# ftgs-checkpoint (§2.2; U-21 and U-26 file parts; Q-17, Q-25)

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
CHECKPOINT_FILES = {
    "manifest.json",
    "params.safetensors",
    "optim.safetensors",
    "optim.json",
    "stats.safetensors",
    "rng.safetensors",
    "rng.json",
    "sampler.json",
}


def _params(n: int, sh_degree: int, seed: int = 0) -> dict[str, Tensor]:
    g = _gen(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "times": torch.rand(n, generator=g),
        "log_durations": torch.randn(n, generator=g) - 3.0,
        "velocities": torch.randn(n, 3, generator=g),
        "log_scales": torch.randn(n, 3, generator=g) - 4.0,
        "quats": torch.randn(n, 4, generator=g),
        "opacity_logits": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, generator=g),
    }


def _trainer_state(n: int = 7, sh_degree: int = 3) -> TrainerState:
    params = _params(n, sh_degree)
    g = _gen(99)
    optim: dict[str, object] = {
        name: {
            "exp_avg": torch.randn(p.shape, generator=g),
            "exp_avg_sq": torch.rand(p.shape, generator=g),
            "step": torch.tensor(123.0),
            "lr": [1.6e-4, 1.6e-6],
            "schedule": "exponential",
        }
        for name, p in params.items()
    }
    optim["betas"] = [0.9, 0.999]
    optim["eps"] = 1e-15
    reloc = torch.Generator().manual_seed(5)
    torch.rand(3, generator=reloc)
    py = random.Random(11)
    py.random()
    version, internal, gauss = py.getstate()
    rng = {
        "torch_cpu": torch.Generator().manual_seed(1).get_state(),
        "cuda": {},
        "relocation": reloc.get_state(),
        "sampler": torch.Generator().manual_seed(2).get_state(),
        "python": [version, list(internal), gauss],
        "numpy": np.random.default_rng(3).bit_generator.state,
    }
    return TrainerState(
        iteration=1500,
        config=config_to_dict(TrainConfig(sh_degree=sh_degree, seed=4)),
        time=TimeNormalizer(0.1, 10.0 / 3.0),
        spatial_scale=1.1 * 2.718281828459045,
        frame_dt=1.0 / 299.0,
        params=params,
        optim=optim,
        stats={
            "g_sum": torch.rand(n, generator=g),
            "count": torch.randint(0, 9, (n,), generator=g),
        },
        rng=rng,
        sampler={"epoch": 3, "cursor": 17, "permutation": [4, 0, 2, 1, 3]},
        scene_sha256="ab" * 32,
        time_unit="frame",
    )


def _assert_same(a: object, b: object, path: str = "") -> None:
    """Bitwise equality of nested tensors, dicts, lists and scalars (tuples compare as lists)."""
    if isinstance(a, Tensor):
        assert isinstance(b, Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list | tuple):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}/{i}")
    else:
        assert type(a) is type(b) and a == b, path


def _assert_states_equal(a: TrainerState, b: TrainerState) -> None:
    assert b.iteration == a.iteration
    assert b.time == a.time
    assert (b.spatial_scale, b.frame_dt) == (a.spatial_scale, a.frame_dt)
    assert (b.scene_sha256, b.time_unit) == (a.scene_sha256, a.time_unit)
    for name in ("config", "params", "optim", "stats", "rng", "sampler"):
        _assert_same(getattr(a, name), getattr(b, name), name)


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_checkpoint_round_trip_for_each_sh_degree(tmp_path: Path, sh_degree: int) -> None:
    """U-21 (round trip of every tensor, moment, step counter, generator state, sampler and
    statistic) and U-26 (checkpoint round trip per L, including L = 0's empty shN): bitwise."""
    state = _trainer_state(sh_degree=sh_degree)
    out = save_checkpoint(state, tmp_path / "ckpt")
    assert out == tmp_path / "ckpt"
    back = load_checkpoint(out)
    _assert_states_equal(state, back)
    assert back.params["shN"].shape == (7, (sh_degree + 1) ** 2 - 1, 3)


def test_checkpoint_generator_states_restore_the_same_draws(tmp_path: Path) -> None:
    """U-21: the stored generator states reproduce the draws (torch, Python, NumPy)."""
    state = _trainer_state()
    back = load_checkpoint(save_checkpoint(state, tmp_path / "ckpt"))
    g = torch.Generator()
    g.set_state(back.rng["relocation"])
    h = torch.Generator()
    h.set_state(state.rng["relocation"])
    assert torch.equal(torch.rand(5, generator=g), torch.rand(5, generator=h))
    version, internal, gauss = back.rng["python"]
    py = random.Random()
    py.setstate((version, tuple(internal), gauss))
    ref = random.Random(11)
    ref.random()
    assert py.random() == ref.random()
    bit_generator = np.random.PCG64()
    bit_generator.state = back.rng["numpy"]
    assert np.random.Generator(bit_generator).random() == np.random.default_rng(3).random()


def test_checkpoint_directory_holds_the_section_2_2_files(tmp_path: Path) -> None:
    """§2.2 ftgs-checkpoint: exactly the eight files, and a manifest with format, version,
    iteration, the full TrainConfig, τ₀, τ₁, S, Δt_f, SH degree, seed, scene hash, versions and
    the SHA-256 of every other file (Q-17, Q-25)."""
    state = _trainer_state()
    out = save_checkpoint(state, tmp_path / "ckpt")
    assert {p.name for p in out.iterdir()} == CHECKPOINT_FILES
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["format"] == "ftgs-checkpoint"
    assert manifest["version"] == 1
    assert manifest["iteration"] == 1500
    assert manifest["config"] == state.config
    assert (manifest["tau0"], manifest["tau1"]) == (0.1, 10.0 / 3.0)
    assert manifest["spatial_scale"] == state.spatial_scale
    assert manifest["frame_dt"] == 1.0 / 299.0
    assert manifest["sh_degree"] == 3
    assert manifest["seed"] == 4
    assert manifest["scene_sha256"] == "ab" * 32
    assert manifest["time_unit"] == "frame"
    assert set(manifest["versions"]) == {"openftgs", "torch", "gsplat", "cuda"}
    assert manifest["versions"]["torch"] == torch.__version__
    assert set(manifest["files"]) == CHECKPOINT_FILES - {"manifest.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with safe_open(str(out / "params.safetensors"), framework="pt") as f:
        names = set(f.keys())
        dtypes = {f.get_tensor(k).dtype for k in names}
    assert names == set(PARAM_NAMES)
    assert dtypes == {F32}
    with safe_open(str(out / "optim.safetensors"), framework="pt") as f:
        assert "means/exp_avg" in set(f.keys())
    optim_json = json.loads((out / "optim.json").read_text())
    assert optim_json["means"]["lr"] == [1.6e-4, 1.6e-6]
    assert "exp_avg" not in optim_json["means"]
    sampler = json.loads((out / "sampler.json").read_text())
    assert sampler == {"epoch": 3, "cursor": 17, "permutation": [4, 0, 2, 1, 3]}


@pytest.mark.parametrize(
    "case", ["tampered params", "missing rng.json", "version 2", "eight params"]
)
def test_checkpoint_loader_rejects_damaged_directories(tmp_path: Path, case: str) -> None:
    """§2.2 and §2.1 (integrity check): a changed or missing file, an unknown major version, or
    a params file without the nine tensors is rejected (Q-25, Q-26)."""
    out = save_checkpoint(_trainer_state(), tmp_path / "ckpt")
    if case == "eight params":
        params = _params(7, 3)
        del params["shN"]
        save_file(params, str(out / "params.safetensors"))
        manifest = json.loads((out / "manifest.json").read_text())
        digest = hashlib.sha256((out / "params.safetensors").read_bytes()).hexdigest()
        manifest["files"]["params.safetensors"] = digest
        (out / "manifest.json").write_text(json.dumps(manifest))
    elif case == "tampered params":
        data = bytearray((out / "params.safetensors").read_bytes())
        data[-1] ^= 0x01
        (out / "params.safetensors").write_bytes(bytes(data))
    elif case == "missing rng.json":
        (out / "rng.json").unlink()
    elif case == "version 2":
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["version"] = 2
        (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_checkpoint(out)


def test_checkpoint_save_replaces_an_existing_checkpoint_atomically(tmp_path: Path) -> None:
    """§2.2 "Saves are atomic" (Q-25): saving over a checkpoint replaces it, and no temporary
    directory is left beside it."""
    first = _trainer_state(n=4)
    second = _trainer_state(n=9)
    target = tmp_path / "ckpt"
    save_checkpoint(first, target)
    save_checkpoint(second, target)
    _assert_states_equal(second, load_checkpoint(target))
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt"]


def test_checkpoint_save_refuses_to_replace_a_directory_that_is_not_a_checkpoint(
    tmp_path: Path,
) -> None:
    """Q-25: a save never deletes a directory that isn't a checkpoint."""
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "precious.txt").write_text("keep me")
    with pytest.raises(ValueError):
        save_checkpoint(_trainer_state(), target)
    assert (target / "precious.txt").read_text() == "keep me"


@pytest.mark.parametrize(
    "case", ["slash in key", "tensor in sampler", "nan in json", "eight params", "tensor in list"]
)
def test_checkpoint_save_rejects_state_the_layout_cannot_hold(tmp_path: Path, case: str) -> None:
    """Q-25: keys containing "/" (the tensor key separator), tensors in sampler.json or inside
    lists, non-finite JSON numbers and params without the nine tensors are refused, and nothing
    is written."""
    state = _trainer_state()
    if case == "slash in key":
        state.optim["a/b"] = torch.zeros(1)
    elif case == "tensor in sampler":
        state.sampler["permutation"] = torch.arange(5)
    elif case == "nan in json":
        state.optim["eps"] = float("nan")
    elif case == "eight params":
        del state.params["shN"]
    elif case == "tensor in list":
        state.rng["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    with pytest.raises(ValueError):
        save_checkpoint(state, tmp_path / "ckpt")
    assert list(tmp_path.iterdir()) == []


# PLY snapshot for viewers (§2.1 io row; Q-18)


def _read_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
    """A minimal reader for the test: the header's property names, the vertex count and the
    float32 body as an [N, P] array."""
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    assert header[0] == "ply"
    assert header[1] == "format binary_little_endian 1.0"
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    props = [line.split() for line in header if line.startswith("property")]
    assert all(p[1] == "float" for p in props)
    names = [p[2] for p in props]
    body = np.frombuffer(data[end:], dtype="<f4").reshape(count, len(names))
    return names, count, body


def test_ply_snapshot_uses_the_3dgs_viewer_layout(tmp_path: Path) -> None:
    """Q-18: x y z, nx ny nz (zero), f_dc_0..2, f_rest_* channel-major, opacity as a logit,
    scale_0..2 as logs, rot_0..3; values worked by hand."""
    from openftgs.io import write_ply

    means = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]])
    scales = torch.tensor([[1.0, math.e, 0.5], [2.0, 1.0, 1.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    opacities = torch.tensor([0.5, 0.75])
    sh = torch.arange(2 * 4 * 3, dtype=F32).reshape(2, 4, 3)  # L = 1
    path = tmp_path / "snapshot.ply"
    write_ply(path, means=means, scales=scales, quats=quats, opacities=opacities, sh=sh)
    names, count, body = _read_ply(path)
    rest = [f"f_rest_{i}" for i in range(9)]
    assert names == [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *rest,
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]  # fmt: skip
    assert count == 2
    col = {name: body[:, i] for i, name in enumerate(names)}
    np.testing.assert_array_equal(np.stack([col["x"], col["y"], col["z"]], 1), means.numpy())
    assert not np.any(np.stack([col["nx"], col["ny"], col["nz"]]))
    np.testing.assert_array_equal(col["f_dc_1"], [1.0, 13.0])  # sh[n, 0, 1]
    # channel-major: f_rest_{c·3 + (k − 1)} = sh[n, k, c]
    np.testing.assert_array_equal(col["f_rest_0"], [3.0, 15.0])  # k = 1, c = 0
    np.testing.assert_array_equal(col["f_rest_2"], [9.0, 21.0])  # k = 3, c = 0
    np.testing.assert_array_equal(col["f_rest_3"], [4.0, 16.0])  # k = 1, c = 1
    np.testing.assert_array_equal(col["f_rest_8"], [11.0, 23.0])  # k = 3, c = 2
    np.testing.assert_allclose(col["opacity"], [0.0, math.log(3.0)], rtol=0, atol=1e-6)
    np.testing.assert_allclose(col["scale_1"], [1.0, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(col["scale_2"], [math.log(0.5), 0.0], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(col["rot_0"], [1.0, 0.5])
    np.testing.assert_array_equal(col["rot_3"], [0.0, 0.5])


def test_ply_snapshot_of_degree_0_and_saturated_opacity(tmp_path: Path) -> None:
    """Q-18: L = 0 writes no f_rest; an opacity of 1 or 0 takes the ftgs-if clamp
    [1e-7, 1 − 1e-7] before the logit, so the file stays finite."""
    from openftgs.io import write_ply

    path = tmp_path / "snapshot.ply"
    write_ply(
        path,
        means=torch.zeros(2, 3),
        scales=torch.ones(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        opacities=torch.tensor([1.0, 0.0]),
        sh=torch.zeros(2, 1, 3),
    )
    names, _, body = _read_ply(path)
    assert not any(n.startswith("f_rest") for n in names)
    opacity = body[:, names.index("opacity")]
    logit = math.log((1.0 - 1e-7) / 1e-7)
    np.testing.assert_allclose(opacity, [logit, -logit], rtol=1e-6)


def test_ply_snapshot_rejects_mismatched_arrays(tmp_path: Path) -> None:
    """Q-18, Q-26: arrays of different N or wrong trailing shapes are refused."""
    from openftgs.io import write_ply

    with pytest.raises(ValueError):
        write_ply(
            tmp_path / "x.ply",
            means=torch.zeros(3, 3),
            scales=torch.ones(2, 3),
            quats=torch.ones(2, 4),
            opacities=torch.ones(2) / 2,
            sh=torch.zeros(2, 1, 3),
        )
    assert not (tmp_path / "x.ply").exists()


def test_checkpoint_with_empty_optim_rng_and_stats_round_trips(tmp_path: Path) -> None:
    """§2.2: every file is written even when its part of the state holds no tensor."""
    state = _trainer_state()
    state.optim, state.rng, state.stats, state.sampler = {}, {}, {}, {}
    _assert_states_equal(state, load_checkpoint(save_checkpoint(state, tmp_path / "ckpt")))


def test_unreadable_safetensors_files_are_rejected_with_value_error(tmp_path: Path) -> None:
    """Q-26: a file that isn't safetensors fails schema validation with ValueError."""
    path = tmp_path / "junk.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}garbage")
    for reader in (read_init_points, read_exchange):
        with pytest.raises(ValueError):
            reader(path)


def test_param_names_copy_matches_the_model() -> None:
    """Private: io may not import the model (§2.1), so its copy of the nine names is pinned to
    FreeTimeGaussians.PARAM_NAMES."""
    import openftgs.io
    from openftgs.model import FreeTimeGaussians

    assert openftgs.io._PARAM_NAMES == FreeTimeGaussians.PARAM_NAMES


def test_checkpoint_safetensors_files_carry_format_and_version(tmp_path: Path) -> None:
    """§2.2 introduction: "Every file carries format and version" (Q-25: the JSON parts are
    covered by the manifest's own format, version and hashes)."""
    out = save_checkpoint(_trainer_state(), tmp_path / "ckpt")
    for name in ("params", "optim", "stats", "rng"):
        with safe_open(str(out / f"{name}.safetensors"), framework="pt") as f:
            meta = f.metadata()
        assert meta["format"] == "ftgs-checkpoint", name
        assert meta["version"] == "1", name


@pytest.mark.parametrize("case", ["float64 params", "stats of another N", "2-D stats"])
def test_checkpoint_save_checks_params_and_stats(tmp_path: Path, case: str) -> None:
    """§2.2 ftgs-checkpoint: params are the nine tensors in float32; g_sum and count are [N]
    (Q-25)."""
    state = _trainer_state(n=7)
    if case == "float64 params":
        state.params["times"] = state.params["times"].double()
    elif case == "stats of another N":
        state.stats["count"] = torch.zeros(6, dtype=torch.int64)
    elif case == "2-D stats":
        state.stats["g_sum"] = torch.zeros(7, 1)
    with pytest.raises(ValueError):
        save_checkpoint(state, tmp_path / "ckpt")
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_loader_ignores_unlisted_files(tmp_path: Path) -> None:
    """§2.2 introduction (readers ignore what they don't know; Q-25): a file the manifest doesn't
    list, such as a log or an editor backup, doesn't block loading."""
    state = _trainer_state()
    out = save_checkpoint(state, tmp_path / "ckpt")
    (out / "notes.txt").write_text("hello")
    _assert_states_equal(state, load_checkpoint(out))


def test_checkpoint_save_recovers_from_an_interrupted_swap(tmp_path: Path) -> None:
    """§2.2 "Saves are atomic" (Q-25): a kill between the two renames of a save leaves the old
    checkpoint aside and nothing at the target; the next save of that target puts the new one in
    place and removes every leftover temporary or set-aside directory."""
    target = tmp_path / "ckpt"
    save_checkpoint(_trainer_state(n=4), target)
    target.rename(tmp_path / ".ckpt.old-99999")
    (tmp_path / ".ckpt.tmp-99998").mkdir()
    second = _trainer_state(n=9)
    save_checkpoint(second, target)
    _assert_states_equal(second, load_checkpoint(target))
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt"]


def test_checkpoint_save_over_a_regular_file_is_rejected(tmp_path: Path) -> None:
    """Q-25, Q-26: a target that is a file, not a directory, is refused with ValueError."""
    target = tmp_path / "ckpt"
    target.write_text("not a checkpoint")
    with pytest.raises(ValueError):
        save_checkpoint(_trainer_state(), target)
    assert target.read_text() == "not a checkpoint"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_writers_report_an_unwritable_directory_as_os_error(tmp_path: Path) -> None:
    """Q-158, Q-142: a safetensors writer that can't write its file raises OSError, whatever the
    library's own exception class; nothing is left in the directory."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        model = ExchangeModel(exchange_arrays(), 3, 0.1, 1.0, "s", "t")
        with pytest.raises(OSError):
            write_exchange(model, locked / "model.safetensors")
        with pytest.raises(OSError):
            write_init_points(
                _init_points(), locked / "init.safetensors", time_unit="s", producer="t"
            )
        with pytest.raises(OSError):
            write_renders(
                _renders(), locked / "r.safetensors", producer="t", request_sha256="0" * 64
            )
        with pytest.raises(OSError):
            save_checkpoint(_trainer_state(), locked / "ckpt")
        assert list(locked.iterdir()) == []
    finally:
        locked.chmod(0o755)


# The kept checkpoints of a run directory (D-83; Q-149 to Q-152).


def _at(iteration: int) -> TrainerState:
    return dataclasses.replace(_trainer_state(n=4), iteration=iteration)


def _kept(run_dir: Path) -> dict[str, int]:
    """The run directory's checkpoints by name, with the iteration each one holds."""
    return {
        p.name: load_checkpoint(p).iteration
        for p in sorted(run_dir.iterdir())
        if p.name == "checkpoint" or p.name.startswith("checkpoint-")
    }


def test_save_run_checkpoint_keeps_only_the_latest_by_default(tmp_path: Path) -> None:
    """D-83: with K = 1, each save replaces RUN_DIR/checkpoint and nothing else is kept."""
    for iteration in (5, 10, 15):
        assert save_run_checkpoint(_at(iteration), tmp_path) == tmp_path / "checkpoint"
    assert _kept(tmp_path) == {"checkpoint": 15}
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint"]


def test_save_run_checkpoint_keeps_the_k_latest(tmp_path: Path) -> None:
    """D-83: with K = 2 the latest save is RUN_DIR/checkpoint and the one before it becomes
    RUN_DIR/checkpoint-<its iteration>; the oldest beyond K is removed."""
    for iteration in (5, 10, 15):
        save_run_checkpoint(_at(iteration), tmp_path, keep=2)
    assert _kept(tmp_path) == {"checkpoint": 15, "checkpoint-10": 10}
    _assert_states_equal(_at(15), load_checkpoint(tmp_path / "checkpoint"))
    save_run_checkpoint(_at(20), tmp_path, keep=3)
    assert _kept(tmp_path) == {"checkpoint": 20, "checkpoint-10": 10, "checkpoint-15": 15}


@pytest.mark.parametrize(
    ("iteration", "want"),
    [(7, {"checkpoint": 7, "checkpoint-5": 5}), (10, {"checkpoint": 10, "checkpoint-5": 5})],
)
def test_save_run_checkpoint_removes_an_abandoned_continuation(
    tmp_path: Path, iteration: int, want: dict[str, int]
) -> None:
    """D-83, Q-149: a save at iteration it drops the kept checkpoints of later iterations, the
    continuation a resume from an earlier one abandoned, and one already at it, so no
    iteration is kept twice; RUN_DIR/checkpoint is replaced in place, never renamed."""
    for saved in (5, 10, 15):
        save_run_checkpoint(_at(saved), tmp_path, keep=3)
    save_run_checkpoint(_at(iteration), tmp_path, keep=3)
    assert _kept(tmp_path) == want


def test_save_run_checkpoint_at_the_same_iteration_replaces_in_place(tmp_path: Path) -> None:
    """Q-149: a save at the iteration RUN_DIR/checkpoint already holds replaces it, with no
    rename, so the kept checkpoints are unchanged."""
    save_run_checkpoint(_at(5), tmp_path, keep=2)
    save_run_checkpoint(_at(10), tmp_path, keep=2)
    again = dataclasses.replace(_trainer_state(n=9), iteration=10)
    save_run_checkpoint(again, tmp_path, keep=2)
    assert _kept(tmp_path) == {"checkpoint": 10, "checkpoint-5": 5}
    _assert_states_equal(again, load_checkpoint(tmp_path / "checkpoint"))


def test_save_run_checkpoint_leaves_other_directories_alone(tmp_path: Path) -> None:
    """Q-150, Q-25: only checkpoints named checkpoint-<integer> are kept or removed; a directory
    of another name, or one with that name that isn't a checkpoint, survives the rotation."""
    (tmp_path / "checkpoint-xyz").mkdir()
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-3" / "notes.txt").write_text("mine")
    for iteration in (5, 10):
        save_run_checkpoint(_at(iteration), tmp_path)
    assert (tmp_path / "checkpoint-3" / "notes.txt").read_text() == "mine"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint",
        "checkpoint-3",
        "checkpoint-xyz",
    ]
    assert load_checkpoint(tmp_path / "checkpoint").iteration == 10


def test_save_run_checkpoint_refuses_to_rename_onto_what_is_not_a_checkpoint(
    tmp_path: Path,
) -> None:
    """Q-150, Q-25: with K > 1 the latest is renamed RUN_DIR/checkpoint-<its iteration>; when
    something that isn't a checkpoint has that name, the save is refused with ValueError and
    nothing is removed or replaced."""
    (tmp_path / "checkpoint-5").write_text("mine")
    save_run_checkpoint(_at(5), tmp_path, keep=2)
    with pytest.raises(ValueError, match="checkpoint-5"):
        save_run_checkpoint(_at(10), tmp_path, keep=2)
    assert (tmp_path / "checkpoint-5").read_text() == "mine"
    assert load_checkpoint(tmp_path / "checkpoint").iteration == 5


def test_save_run_checkpoint_keeps_a_killed_saves_checkpoint(tmp_path: Path) -> None:
    """Q-25, D-83: a checkpoint a killed save left set aside is restored first, so with K = 2
    it is kept as RUN_DIR/checkpoint-<its iteration> rather than overwritten."""
    save_run_checkpoint(_at(5), tmp_path, keep=2)
    (tmp_path / "checkpoint").rename(tmp_path / ".checkpoint.old-99999")
    save_run_checkpoint(_at(10), tmp_path, keep=2)
    assert _kept(tmp_path) == {"checkpoint": 10, "checkpoint-5": 5}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


@pytest.mark.parametrize("keep", [0, -1])
def test_save_run_checkpoint_needs_at_least_one_kept(tmp_path: Path, keep: int) -> None:
    """D-83: K ≥ 1; anything else is a ValueError and nothing is written."""
    with pytest.raises(ValueError, match="keep"):
        save_run_checkpoint(_at(5), tmp_path, keep=keep)
    assert list(tmp_path.iterdir()) == []


def test_save_run_checkpoint_that_fails_partway_keeps_the_earlier_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-151: the oldest kept checkpoints are removed only once the new save is in place, and a
    save that fails partway puts the previous latest back at RUN_DIR/checkpoint, so every
    earlier checkpoint stays where it was."""
    for iteration in (5, 10):
        save_run_checkpoint(_at(iteration), tmp_path, keep=2)

    def full(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("openftgs.io.save_file", full)
    with pytest.raises(OSError, match="No space left"):
        save_run_checkpoint(_at(15), tmp_path, keep=2)
    assert _kept(tmp_path) == {"checkpoint": 10, "checkpoint-5": 5}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_check_run_checkpoints_names_an_entry_the_rotation_would_refuse(tmp_path: Path) -> None:
    """Q-163, Q-150: with K > 1 a RUN_DIR/checkpoint-<n> that isn't a checkpoint, which a save
    would refuse to rename onto, is a ValueError naming it, and nothing changes; with K = 1,
    when no save renames, the check passes. Of several, the one of the lowest n is named."""
    (tmp_path / "checkpoint-15").write_text("mine")
    (tmp_path / "checkpoint-100").mkdir()
    with pytest.raises(ValueError, match="checkpoint-15 exists"):
        check_run_checkpoints(tmp_path, keep=3)
    check_run_checkpoints(tmp_path, keep=1)
    check_run_checkpoints(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-100", "checkpoint-15"]
    assert (tmp_path / "checkpoint-15").read_text() == "mine"


def test_check_run_checkpoints_passes_what_the_rotation_counts_or_ignores(tmp_path: Path) -> None:
    """Q-163, Q-150: a checkpoint passes, one whose manifest reads but whose file doesn't match
    its SHA-256 included, since Q-150 counts it as a checkpoint (load_checkpoint would refuse
    it); a name other than checkpoint-<decimal> is ignored."""
    save_checkpoint(_at(5), tmp_path / "checkpoint-5")
    damaged = save_checkpoint(_at(15), tmp_path / "checkpoint-15")
    (damaged / "sampler.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(damaged)
    (tmp_path / "checkpoint-xyz").write_text("mine")
    (tmp_path / "checkpoint-007").mkdir()
    check_run_checkpoints(tmp_path, keep=3)


@pytest.mark.parametrize("keep", [0, -1, True])
def test_check_run_checkpoints_needs_at_least_one_kept(tmp_path: Path, keep: int) -> None:
    """Q-163, D-83: `keep` is an integer of at least 1, as for save_run_checkpoint."""
    with pytest.raises(ValueError, match="keep"):
        check_run_checkpoints(tmp_path, keep=keep)


def test_ftgs_init_reader_requires_time_unit_metadata(tmp_path: Path) -> None:
    """§2.2 ftgs-init metadata includes time_unit (Q-22): a file without it is rejected."""
    meta = _init_metadata()
    del meta["time_unit"]
    path = tmp_path / "init.safetensors"
    save_file(_init_tensors(), str(path), metadata=meta)
    with pytest.raises(ValueError, match="time_unit"):
        read_init_points(path)


@pytest.mark.parametrize("case", ["nan mean", "zero scale", "inf sh"])
def test_ply_snapshot_rejects_values_it_cannot_write(tmp_path: Path, case: str) -> None:
    """Q-18, Q-26: non-finite values and scales ≤ 0 (whose log is not finite) are refused."""
    from openftgs.io import write_ply

    arrays = {
        "means": torch.zeros(2, 3),
        "scales": torch.ones(2, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "opacities": torch.full((2,), 0.5),
        "sh": torch.zeros(2, 1, 3),
    }
    if case == "nan mean":
        arrays["means"][0, 0] = float("nan")
    elif case == "zero scale":
        arrays["scales"][1, 2] = 0.0
    elif case == "inf sh":
        arrays["sh"][0, 0, 1] = float("inf")
    with pytest.raises(ValueError):
        write_ply(tmp_path / "x.ply", **arrays)
    assert not (tmp_path / "x.ply").exists()


def test_checkpoint_loader_enforces_n_max_and_a_valid_config(tmp_path: Path) -> None:
    """§1.1 validity rules, "checked on load": N never exceeds the stored configuration's N_max;
    the stored configuration must itself be valid (Q-10, Q-11, Q-25)."""
    from openftgs.config import GrowthConfig

    state = _trainer_state(n=7)
    state.config = config_to_dict(TrainConfig(growth=GrowthConfig(n_max=6)))
    out = save_checkpoint(state, tmp_path / "over")
    with pytest.raises(ValueError, match="N_max"):
        load_checkpoint(out)

    out = save_checkpoint(_trainer_state(n=7), tmp_path / "bad-config")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["config"]["growth"]["n_max"] = 0
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_checkpoint(out)


def test_save_checkpoint_restores_the_newest_set_aside_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private (Q-25): with several checkpoints left aside by killed saves, recovery restores
    the most recently modified one, whatever the pids in their names."""
    import os

    import openftgs.io

    older, newer = _trainer_state(n=4), _trainer_state(n=5)
    save_checkpoint(older, tmp_path / "a")
    save_checkpoint(newer, tmp_path / "b")
    (tmp_path / "a").rename(tmp_path / ".ckpt.old-99999")
    (tmp_path / "b").rename(tmp_path / ".ckpt.old-1")
    os.utime(tmp_path / ".ckpt.old-99999", (1_000_000, 1_000_000))
    os.utime(tmp_path / ".ckpt.old-1", (2_000_000, 2_000_000))

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(openftgs.io, "_write_safetensors", fail)
    with pytest.raises(OSError):
        save_checkpoint(_trainer_state(n=9), tmp_path / "ckpt")
    monkeypatch.undo()
    _assert_states_equal(newer, load_checkpoint(tmp_path / "ckpt"))


@pytest.mark.parametrize("case", ["velocity", "time_center"])
def test_ftgs_if_single_frame_file_must_have_no_motion(tmp_path: Path, case: str) -> None:
    """§1.4 "Time" and Q-30: with τ₀ = τ₁, every time center must be τ₀ and every velocity 0;
    a file that says otherwise is rejected."""
    arrays = exchange_arrays()
    arrays["time_center"] = torch.full((6,), 2.0, dtype=F64)
    arrays["velocity"] = torch.zeros(6, 3)
    meta = exchange_metadata(tau0="2.0", tau1="2.0")
    good = tmp_path / "good.safetensors"
    save_file(arrays, str(good), metadata=meta)
    read_exchange(good)
    if case == "velocity":
        arrays["velocity"][3, 1] = 1e-3
    else:
        arrays["time_center"][2] = 2.5
    path = tmp_path / "bad.safetensors"
    save_file(arrays, str(path), metadata=meta)
    with pytest.raises(ValueError, match="single timestamp"):
        read_exchange(path)


def test_spec_constant_copies_agree_across_modules() -> None:
    """Private: io may not import the model (§2.1), so the SH degrees (D-3), the quaternion norm
    floor (§1.1) and the opacity clamp (§2.2 ftgs-if) are copied; the copies must agree, and
    TrainConfig must accept exactly the same SH degrees."""
    import openftgs.io
    import openftgs.model

    assert openftgs.io._SH_DEGREES == openftgs.model._SH_DEGREES == (0, 1, 2, 3)
    assert openftgs.io._MIN_ROTATION_NORM == openftgs.model._MIN_QUAT_NORM == 1e-8
    assert openftgs.io._OPACITY_EPS == openftgs.model._OPACITY_EPS == 1e-7
    for degree in range(-1, 5):
        if degree in openftgs.io._SH_DEGREES:
            TrainConfig(sh_degree=degree)
        else:
            with pytest.raises(ValueError):
                TrainConfig(sh_degree=degree)


@pytest.mark.parametrize(
    "rotation",
    [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -0.6, 0.8], [2.0, 0.0, 0.0, 0.0]],
)
def test_ftgs_if_writer_requires_canonical_unit_rotations(
    tmp_path: Path, rotation: list[float]
) -> None:
    """§2.2 ftgs-if table: rotation is a unit quaternion with w ≥ 0. The writer enforces it,
    with Q-24's tie-break at w = 0 (first nonzero component positive); the reader stays lenient
    (Q-24)."""
    arrays = exchange_arrays()
    arrays["rotation"][2] = torch.tensor(rotation)
    model = ExchangeModel(arrays, 3, 2.0, 12.0, "s", "x")
    path = tmp_path / "m.safetensors"
    with pytest.raises(ValueError, match="rotation"):
        write_exchange(model, path)
    assert not path.exists()
    save_file(arrays, str(path), metadata=exchange_metadata())
    assert torch.equal(read_exchange(path).arrays["rotation"][2], torch.tensor(rotation))


@pytest.mark.parametrize("version", ["01", "1.", "+1", " 1", "1.03", "1.x", "1e0"])
def test_readers_reject_non_canonical_version_strings(tmp_path: Path, version: str) -> None:
    """§2.2 introduction: integers are written in decimal; a version is a major number,
    optionally with a minor part, without signs, spaces or leading zeros."""
    path = tmp_path / "init.safetensors"
    save_file(_init_tensors(), str(path), metadata=_init_metadata(version=version))
    with pytest.raises(ValueError, match="version"):
        read_init_points(path)


# ftgs-render-request and ftgs-renders (§2.2; Q-56).


def _request_camera(width: int = 8, height: int = 6) -> dict[str, Any]:
    # A rotation of 90° about z and a translation, written exactly in JSON.
    return {
        "width": width,
        "height": height,
        "K": [[10.0, 0.0, 4.0], [0.0, 12.0, 3.0], [0.0, 0.0, 1.0]],
        "world_to_camera": [
            [0.0, -1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, -0.25],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _request(**overrides: object) -> dict[str, Any]:
    request: dict[str, Any] = {
        "format": "ftgs-render-request",
        "version": 1,
        "time_unit": "s",
        "background": [0.25, 0.5, 1.0],
        "sh_degree": 2,
        "items": [
            {"id": "r0000", "timestamp": 1.2345, "camera": _request_camera()},
            {"id": "r0001", "timestamp": 0.0, "camera": _request_camera(5, 4)},
        ],
    }
    request.update(overrides)
    return request


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_render_request_reads_items_in_order(tmp_path: Path) -> None:
    """§2.2 render request: items in file order, each a camera and a raw timestamp; the
    background and SH degree as given; camera tensors float32 (§2.1 Camera), named by id."""
    request = read_render_request(_write_json(tmp_path / "r.json", _request()))
    assert request.time_unit == "s"
    assert request.background == (0.25, 0.5, 1.0)
    assert request.sh_degree == 2
    assert [item.id for item in request.items] == ["r0000", "r0001"]
    first = request.items[0]
    assert first.timestamp == 1.2345
    assert first.camera.name == "r0000"
    assert (first.camera.width, first.camera.height) == (8, 6)
    assert first.camera.K.dtype == F32 and first.camera.world_to_camera.dtype == F32
    assert torch.equal(first.camera.K, torch.tensor(_request_camera()["K"]))
    assert torch.equal(
        first.camera.world_to_camera, torch.tensor(_request_camera()["world_to_camera"])
    )
    assert (request.items[1].camera.width, request.items[1].camera.height) == (5, 4)


def test_render_request_defaults_and_unknown_keys(tmp_path: Path) -> None:
    """§2.2: background defaults to black (D-34) and sh_degree to null, the model's degree
    (Q-56); readers ignore unknown optional keys, at the top level and in items."""
    raw = _request(extra={"a": 1})
    del raw["background"], raw["sh_degree"]
    raw["items"][0]["note"] = "x"
    raw["items"][0]["camera"]["model"] = "pinhole"
    request = read_render_request(_write_json(tmp_path / "r.json", raw))
    assert request.background == (0.0, 0.0, 0.0)
    assert request.sh_degree is None
    assert len(request.items) == 2
    raw["sh_degree"] = None
    raw["version"] = "1.3"
    assert read_render_request(_write_json(tmp_path / "s.json", raw)).sh_degree is None


def _set(raw: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    target: Any = raw  # dicts and lists along the path
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return raw


_DELETE = object()
_CAM = ("items", 0, "camera")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("format",), "ftgs-renders", "format"),
        (("version",), 2, "version"),
        (("version",), "2.0", "version"),
        (("version",), True, "version"),
        (("version",), 1.0, "version"),
        (("time_unit",), _DELETE, "time_unit"),
        (("time_unit",), 5, "time_unit"),
        (("items",), [], "items"),
        (("items",), _DELETE, "items"),
        (("items", 1, "id"), "r0000", "unique"),
        (("items", 0, "id"), "", "id"),
        (("items", 0, "id"), 7, "id"),
        (("items", 0, "timestamp"), True, "timestamp"),
        (("items", 0, "timestamp"), "1.0", "timestamp"),
        (("items", 0, "timestamp"), _DELETE, "timestamp"),
        ((*_CAM, "width"), True, "width"),
        ((*_CAM, "width"), 0, "width"),
        ((*_CAM, "height"), 2.0, "height"),
        ((*_CAM, "K", 0, 1), 0.5, "skew"),
        ((*_CAM, "K", 0, 0), 0.0, "fx"),
        ((*_CAM, "K", 1, 1), -12.0, "fy"),
        ((*_CAM, "K", 2), [0.0, 0.0, 2.0], "skew"),
        ((*_CAM, "K", 1), [0.0, 12.0], "K"),
        ((*_CAM, "world_to_camera", 3), [0.0, 0.0, 1.0, 1.0], "last row"),
        ((*_CAM, "world_to_camera", 0, 0), 0.01, "orthonormal"),
        ((*_CAM, "world_to_camera", 2, 2), -1.0, "orthonormal"),  # a reflection
        ((*_CAM, "world_to_camera", 0, 3), "x", "finite"),
        (("sh_degree",), 4, "sh_degree"),
        (("sh_degree",), True, "sh_degree"),
        (("background",), [0.0, 0.0, 1.5], "background"),
        (("background",), [0.0, 0.0], "background"),
    ],
)
def test_render_request_rejections(
    tmp_path: Path, path: tuple[str | int, ...], value: object, match: str
) -> None:
    """§2.2 render request, validated as Q-56 states; every failure is a ValueError (Q-26)."""
    raw = _set(_request(), path, value)
    with pytest.raises(ValueError, match=match):
        read_render_request(_write_json(tmp_path / "r.json", raw))


@pytest.mark.parametrize(
    ("value", "shown"), [(_DELETE, "None"), (5, "5"), ("ftgs-if", "'ftgs-if'")]
)
def test_render_request_names_the_format_it_found(
    tmp_path: Path, value: object, shown: str
) -> None:
    """Q-26: the rejection shows the format as the file holds it, a missing one as None, not as
    the string 'None' (from the second M9 milestone review)."""
    raw = _set(_request(), ("format",), value)
    with pytest.raises(ValueError, match=f"format must be 'ftgs-render-request', got {shown}$"):
        read_render_request(_write_json(tmp_path / "r.json", raw))


@pytest.mark.parametrize("text", ["[1, 2]", "{", '{"format": "ftgs-render-request", "x": NaN}'])
def test_render_request_rejects_files_that_are_not_requests(tmp_path: Path, text: str) -> None:
    path = tmp_path / "r.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_render_request(path)


def test_render_request_rejects_non_finite_numbers(tmp_path: Path) -> None:
    """Q-56: JSON's NaN and Infinity extensions are refused wherever a number is read."""
    path = tmp_path / "r.json"
    text = json.dumps(_request()).replace("1.2345", "NaN")
    path.write_text(text)
    with pytest.raises(ValueError, match="finite"):
        read_render_request(path)


def test_render_request_accepts_a_rotation_within_the_tolerance(tmp_path: Path) -> None:
    """Q-56: R orthonormal within 1e-5, so float32-rounded rotations written to JSON pass."""
    c, s = math.cos(0.3), math.sin(0.3)
    w2c = [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
    rounded = torch.tensor(w2c, dtype=F32).tolist()
    raw = _set(_request(), (*_CAM, "world_to_camera"), rounded)
    read_render_request(_write_json(tmp_path / "r.json", raw))


def _renders() -> Renders:
    g = _gen(3)
    rgb = {
        "a": torch.rand(6, 8, 3, generator=g) * 1.5,  # §2.2: not clamped to 1
        "b": torch.rand(4, 5, 3, generator=g),
    }
    alpha = {"a": torch.rand(6, 8, 1, generator=g), "b": torch.rand(4, 5, 1, generator=g)}
    return Renders(rgb=rgb, alpha=alpha)


_SHA = hashlib.sha256(b"request").hexdigest()


def test_renders_round_trip_with_metadata(tmp_path: Path) -> None:
    """§2.2 ftgs-renders: rgb/<id> float32 [H, W, 3], alpha/<id> float32 [H, W, 1]; metadata
    records the format, version, producer and the request's SHA-256 (Q-56)."""
    renders = _renders()
    path = tmp_path / "out.safetensors"
    sha = hashlib.sha256(b"request").hexdigest()
    write_renders(renders, path, producer="openftgs test", request_sha256=sha)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        keys = set(f.keys())
    assert meta == {
        "format": "ftgs-renders",
        "version": "1",
        "producer": "openftgs test",
        "request_sha256": sha,
    }
    assert keys == {"rgb/a", "rgb/b", "alpha/a", "alpha/b"}
    back = read_renders(path, request_sha256=sha)
    assert list(back.rgb) == list(back.alpha) == ["a", "b"]
    for k in ("a", "b"):
        assert torch.equal(back.rgb[k], renders.rgb[k])
        assert torch.equal(back.alpha[k], renders.alpha[k])
    assert float(back.rgb["a"].max()) > 1.0


def test_renders_reader_rejects_another_request(tmp_path: Path) -> None:
    """Q-56: reading with a request hash rejects a file recorded for another request."""
    path = tmp_path / "out.safetensors"
    write_renders(_renders(), path, producer="p", request_sha256=_SHA)
    read_renders(path)
    with pytest.raises(ValueError, match="request"):
        read_renders(path, request_sha256=hashlib.sha256(b"another request").hexdigest())


@pytest.mark.parametrize(
    "sha", ["aa", _SHA.upper(), _SHA[:-1], _SHA + "0", "g" + _SHA[1:], " " + _SHA[1:]]
)
def test_renders_writer_takes_a_sha256_hex_digest(tmp_path: Path, sha: str) -> None:
    """§2.2: the metadata records the SHA-256 of the request, which read_renders compares with
    hashlib's hexdigest, so the writer takes 64 lowercase hex digits and nothing else (Q-56,
    from the second M9 milestone review)."""
    path = tmp_path / "out.safetensors"
    with pytest.raises(ValueError, match="SHA-256"):
        write_renders(_renders(), path, producer="p", request_sha256=sha)
    assert not path.exists()


def test_renders_reader_ignores_other_tensors(tmp_path: Path) -> None:
    """§2.2 introduction: unknown optional content is ignored."""
    r = _renders()
    tensors = {"rgb/a": r.rgb["a"], "alpha/a": r.alpha["a"], "depth/a": torch.zeros(6, 8, 1)}
    path = tmp_path / "out.safetensors"
    save_file(tensors, str(path), metadata={"format": "ftgs-renders", "version": "1"})
    assert set(read_renders(path).rgb) == {"a"}


@pytest.mark.parametrize(
    ("rgb", "alpha", "match"),
    [
        ({"a": torch.zeros(6, 8, 3)}, {}, "both"),
        ({"a": torch.zeros(6, 8, 3, dtype=F64)}, {"a": torch.zeros(6, 8, 1)}, "float32"),
        ({"a": torch.zeros(6, 8, 4)}, {"a": torch.zeros(6, 8, 1)}, "shape"),
        ({"a": torch.zeros(6, 8)}, {"a": torch.zeros(6, 8, 1)}, r"\[H, W, 3\], got \[6, 8\]$"),
        ({"a": torch.zeros(6, 8, 3)}, {"a": torch.zeros(6, 7, 1)}, "shape"),
        ({"a": torch.full((6, 8, 3), math.nan)}, {"a": torch.zeros(6, 8, 1)}, "finite"),
        ({}, {}, "at least one"),
    ],
)
def test_renders_validation(
    tmp_path: Path, rgb: dict[str, Tensor], alpha: dict[str, Tensor], match: str
) -> None:
    """Q-56: the writer and the reader apply the same checks."""
    path = tmp_path / "out.safetensors"
    with pytest.raises(ValueError, match=match):
        write_renders(Renders(rgb, alpha), path, producer="p", request_sha256=_SHA)
    assert not path.exists()
    tensors = {f"rgb/{k}": v for k, v in rgb.items()} | {f"alpha/{k}": v for k, v in alpha.items()}
    save_file(
        tensors or {"x": torch.zeros(1)},
        str(path),
        metadata={"format": "ftgs-renders", "version": "1"},
    )
    with pytest.raises(ValueError, match=match):
        read_renders(path)


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        ({"format": "ftgs-if", "version": "1"}, "format"),
        ({"format": "ftgs-renders", "version": "2"}, "version"),
        ({"format": "ftgs-renders"}, "version"),
    ],
)
def test_renders_reader_checks_format_and_major_version(
    tmp_path: Path, meta: dict[str, str], match: str
) -> None:
    """§2.2 introduction: every file carries format and version; readers reject an unknown
    major version."""
    r = _renders()
    path = tmp_path / "out.safetensors"
    save_file({"rgb/a": r.rgb["a"], "alpha/a": r.alpha["a"]}, str(path), metadata=meta)
    with pytest.raises(ValueError, match=match):
        read_renders(path)


def test_renders_reader_returns_ids_sorted(tmp_path: Path) -> None:
    """Q-56: the file keeps no item order, so ids come back sorted, whatever order they were
    written in."""
    r = _renders()
    renders = Renders(
        rgb={k: r.rgb["b"].clone() for k in ("z", "a", "m")},
        alpha={k: r.alpha["b"].clone() for k in ("z", "a", "m")},
    )
    path = tmp_path / "out.safetensors"
    write_renders(renders, path, producer="p", request_sha256=_SHA)
    back = read_renders(path)
    assert list(back.rgb) == list(back.alpha) == ["a", "m", "z"]


# ftgs-scene (§2.2; Q-64).


def _scene_camera(width: int = 8, height: int = 6) -> dict[str, Any]:
    # A rotation of 90° about z and a translation, written exactly in JSON.
    return {
        "width": width,
        "height": height,
        "model": "pinhole",
        "K": [[10.0, 0.0, 4.0], [0.0, 12.0, 3.0], [0.0, 0.0, 1.0]],
        "world_to_camera": [
            [0.0, -1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, -0.25],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


_MOVED = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 4.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _scene(**overrides: object) -> dict[str, Any]:
    scene: dict[str, Any] = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": {"cam00": _scene_camera(), "cam01": _scene_camera(5, 4)},
        "images": [
            {
                "camera": "cam00",
                "timestamp": 0.0,
                "frame_index": 0,
                "path": "images/cam00/000000.png",
                "split": "train",
            },
            {
                "camera": "cam01",
                "timestamp": 0.1,
                "frame_index": 3,
                "path": "images/cam01/000000.png",
                "split": "test",
                "mask": "masks/cam01/000000.png",
            },
            {
                "camera": "cam00",
                "timestamp": 1.0 / 30.0,
                "path": "images/cam00/000001.png",
                "split": "train",
                "world_to_camera": _MOVED,
            },
        ],
        "metadata": {"source": "n3dv/test", "resize": 0.5},
    }
    scene.update(overrides)
    return scene


def test_scene_manifest_reads_records_in_file_order(tmp_path: Path) -> None:
    """§2.2 scene manifest (Q-64): records in file order with raw float64 timestamps, splits,
    frame indices and paths resolved against the manifest's directory; cameras as in §2.1 with
    float32 tensors and the camera's name; the test image's mask; the metadata, the time unit
    and the SHA-256 of the file's bytes as read."""
    path = _write_json(tmp_path / "scene.json", _scene())
    manifest = read_scene_manifest(path)
    assert manifest.path == path
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.time_unit == "s"
    assert manifest.metadata == {"source": "n3dv/test", "resize": 0.5}
    first, second, third = manifest.records
    assert [r.split for r in manifest.records] == ["train", "test", "train"]
    assert [r.timestamp for r in manifest.records] == [0.0, 0.1, 1.0 / 30.0]
    assert [r.frame_index for r in manifest.records] == [0, 3, None]
    assert first.image_path == tmp_path / "images/cam00/000000.png"
    assert manifest.masks == (None, tmp_path / "masks/cam01/000000.png", None)
    assert first.camera.name == "cam00" and second.camera.name == "cam01"
    assert (second.camera.width, second.camera.height) == (5, 4)
    assert first.camera.K.dtype == F32 and first.camera.world_to_camera.dtype == F32
    assert torch.equal(first.camera.K, torch.tensor(_scene_camera()["K"]))
    assert torch.equal(
        first.camera.world_to_camera, torch.tensor(_scene_camera()["world_to_camera"])
    )
    # §2.2: an image entry may override world_to_camera; the record gets its own camera.
    assert third.camera.name == "cam00"
    assert torch.equal(third.camera.world_to_camera, torch.tensor(_MOVED))
    assert torch.equal(third.camera.K, first.camera.K)


def test_scene_manifest_keeps_the_key_order_of_its_cameras(tmp_path: Path) -> None:
    """Q-108: `camera_names` holds the keys of the manifest's `cameras` object in file order,
    the "manifest's camera order" of D-22, which neither sorted names nor the order of first
    appearance in `images` gives; a camera without images is listed too."""
    raw = _scene()
    raw["cameras"] = {
        "zeta": _scene_camera(),
        "cam01": _scene_camera(5, 4),
        "alpha": _scene_camera(),
        "cam00": _scene_camera(),
    }
    raw["images"][0]["camera"] = "alpha"
    manifest = read_scene_manifest(_write_json(tmp_path / "scene.json", raw))
    assert manifest.camera_names == ("zeta", "cam01", "alpha", "cam00")
    assert [r.camera.name for r in manifest.records] == ["alpha", "cam01", "cam00"]


def test_scene_manifest_defaults_and_unknown_keys(tmp_path: Path) -> None:
    """§2.2: readers ignore unknown optional keys; an absent camera model is pinhole, absent
    metadata is empty, a null frame index is None and a "major.minor" version is read (Q-64)."""
    raw = _scene(extra=[1, 2], version="1.4")
    del raw["metadata"], raw["cameras"]["cam00"]["model"]
    raw["images"][0]["frame_index"] = None
    raw["images"][0]["note"] = "x"
    manifest = read_scene_manifest(_write_json(tmp_path / "scene.json", raw))
    assert manifest.metadata == {}
    assert manifest.records[0].frame_index is None
    assert len(manifest.records) == 3


def test_scene_manifest_images_share_their_camera(tmp_path: Path) -> None:
    """Q-64: images without an override share their camera's object; an override of K alone
    keeps the camera's pose."""
    raw = _scene()
    raw["images"].append(
        {
            "camera": "cam00",
            "timestamp": 0.5,
            "path": "b.png",
            "split": "train",
            "K": [[20.0, 0.0, 4.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]],
        }
    )
    raw["images"].append({"camera": "cam00", "timestamp": 0.6, "path": "c.png", "split": "train"})
    records = read_scene_manifest(_write_json(tmp_path / "scene.json", raw)).records
    assert records[4].camera is records[0].camera
    assert records[3].camera is not records[0].camera
    assert records[3].camera.K[0, 0] == 20.0
    assert torch.equal(records[3].camera.world_to_camera, records[0].camera.world_to_camera)


_IMG = ("images", 0)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("format",), "ftgs-render-request", "format"),
        (("format",), _DELETE, "format"),
        (("version",), 2, "version"),
        (("version",), True, "version"),
        (("time_unit",), "", "time_unit"),
        (("time_unit",), _DELETE, "time_unit"),
        (("cameras",), [], "cameras"),
        (("cameras", "cam00"), 3, "camera"),
        (("cameras", "cam00", "model"), "opencv", "pinhole"),
        (("cameras", "cam00", "width"), 0, "width"),
        (("cameras", "cam00", "K", 0, 1), 0.5, "skew"),
        (("cameras", "cam00", "world_to_camera", 3), [0.0, 0.0, 1.0, 1.0], "last row"),
        (("cameras", "cam00", "world_to_camera", 0, 0), 0.01, "orthonormal"),
        (("images",), [], "images"),
        (("images",), _DELETE, "images"),
        (("images", 0), "x", "object"),
        ((*_IMG, "camera"), "cam09", "camera"),
        ((*_IMG, "camera"), _DELETE, "camera"),
        ((*_IMG, "timestamp"), True, "timestamp"),
        ((*_IMG, "timestamp"), "0", "timestamp"),
        ((*_IMG, "timestamp"), _DELETE, "timestamp"),
        ((*_IMG, "path"), "", "path"),
        ((*_IMG, "path"), "/abs/x.png", "relative"),
        ((*_IMG, "path"), _DELETE, "path"),
        ((*_IMG, "split"), "val", "split"),
        ((*_IMG, "split"), _DELETE, "split"),
        ((*_IMG, "frame_index"), 1.0, "frame_index"),
        ((*_IMG, "frame_index"), True, "frame_index"),
        ((*_IMG, "mask"), "m.png", "mask"),
        (("images", 1, "mask"), "", "mask"),
        ((*_IMG, "world_to_camera"), [[1.0]], "world_to_camera"),
        ((*_IMG, "K"), [[0.0, 0.0, 4.0], [0.0, 12.0, 3.0], [0.0, 0.0, 1.0]], "fx"),
        (("metadata",), [1], "metadata"),
    ],
)
def test_scene_manifest_rejections(
    tmp_path: Path, path: tuple[str | int, ...], value: object, match: str
) -> None:
    """§2.2 scene manifest, validated as Q-64 states (cameras as Q-56's); every failure is a
    ValueError (Q-26)."""
    raw = _set(_scene(), path, value)
    with pytest.raises(ValueError, match=match):
        read_scene_manifest(_write_json(tmp_path / "scene.json", raw))


def test_scene_manifest_needs_a_training_image(tmp_path: Path) -> None:
    """Q-64: training reads the train split, so a manifest without one is refused."""
    raw = _scene()
    raw["images"] = [raw["images"][1]]
    with pytest.raises(ValueError, match="training image"):
        read_scene_manifest(_write_json(tmp_path / "scene.json", raw))


@pytest.mark.parametrize("text", ["[1, 2]", "{", ""])
def test_scene_manifest_rejects_files_that_are_not_manifests(tmp_path: Path, text: str) -> None:
    path = tmp_path / "scene.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_scene_manifest(path)


def test_scene_manifest_rejects_non_finite_timestamps(tmp_path: Path) -> None:
    """Q-56's rule for numbers: JSON's NaN extension is refused."""
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_scene()).replace('"timestamp": 0.1', '"timestamp": NaN'))
    with pytest.raises(ValueError, match="finite"):
        read_scene_manifest(path)


# Scene-manifest writer (§2.2; Q-64, Q-84).


def _camera(name: str, *, fx: float = 676.1, tx: float = 0.0) -> Camera:
    K = torch.tensor([[fx, 0.0, 400.5], [0.0, fx, 300.25], [0.0, 0.0, 1.0]], dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[0, 3] = tx
    return Camera(K=K, world_to_camera=w2c, width=800, height=600, name=name)


def test_scene_manifest_writer_round_trips_through_the_reader(tmp_path: Path) -> None:
    a, b = _camera("cam00"), _camera("cam01", fx=512.3, tx=-0.7)
    records = [
        ImageRecord(a, 0.0, tmp_path / "images/cam00/000000.png", "test", 0),
        ImageRecord(a, 1 / 30, tmp_path / "images/cam00/000001.png", "test", 1),
        ImageRecord(b, 0.1 + 0.2, tmp_path / "images/cam01/000000.png", "train", None),
    ]
    masks = [tmp_path / "masks/cam00/000000.png", None, None]
    metadata = {"source": "n3dv/flame_steak", "resize": 0.5, "nested": {"k": [1, 2]}}
    manifest = write_scene_manifest(
        tmp_path / "scene.json", records, time_unit="s", masks=masks, metadata=metadata
    )
    data = (tmp_path / "scene.json").read_bytes()
    assert manifest.sha256 == hashlib.sha256(data).hexdigest()
    assert read_scene_manifest(tmp_path / "scene.json").sha256 == manifest.sha256
    assert manifest.time_unit == "s" and manifest.metadata == metadata
    assert manifest.masks == tuple(masks)
    for got, want in zip(manifest.records, records, strict=True):
        assert got.image_path == want.image_path and got.split == want.split
        assert got.frame_index == want.frame_index and got.timestamp == want.timestamp
        assert got.camera.name == want.camera.name
        assert (got.camera.width, got.camera.height) == (want.camera.width, want.camera.height)
        assert torch.equal(got.camera.K, want.camera.K)
        assert torch.equal(got.camera.world_to_camera, want.camera.world_to_camera)
    raw = json.loads(data)
    assert raw["format"] == "ftgs-scene" and raw["version"] == 1
    assert set(raw["cameras"]) == {"cam00", "cam01"}
    # Q-84: the shortest decimal that reads back to the same float32.
    assert raw["cameras"]["cam00"]["K"][0][0] == 676.1
    assert raw["images"][0]["path"] == "images/cam00/000000.png"
    assert "frame_index" not in raw["images"][2]


def test_scene_manifest_writer_writes_a_changed_pose_as_an_override(tmp_path: Path) -> None:
    still, moved = _camera("cam00"), _camera("cam00", tx=0.25)
    records = [
        ImageRecord(still, 0.0, tmp_path / "a.png", "train", 0),
        ImageRecord(moved, 1.0, tmp_path / "b.png", "train", 1),
    ]
    manifest = write_scene_manifest(tmp_path / "scene.json", records)
    raw = json.loads((tmp_path / "scene.json").read_text())
    assert "world_to_camera" in raw["images"][1] and "K" not in raw["images"][1]
    assert "world_to_camera" not in raw["images"][0]
    assert torch.equal(manifest.records[1].camera.world_to_camera, moved.world_to_camera)
    assert torch.equal(manifest.records[0].camera.world_to_camera, still.world_to_camera)


@pytest.mark.parametrize("case", ["outside", "mask_on_train", "size", "metadata", "no_train"])
def test_scene_manifest_writer_rejects_what_the_reader_would_refuse(
    tmp_path: Path, case: str
) -> None:
    camera = _camera("cam00")
    records = [ImageRecord(camera, 0.0, tmp_path / "a.png", "train", 0)]
    masks: list[Path | None] | None = None
    metadata: dict[str, Any] = {}
    if case == "outside":
        records = [ImageRecord(camera, 0.0, tmp_path.parent / "a.png", "train", 0)]
    elif case == "mask_on_train":
        masks = [tmp_path / "m.png"]
    elif case == "size":
        other = Camera(camera.K, camera.world_to_camera, 640, 480, "cam00")
        records.append(ImageRecord(other, 1.0, tmp_path / "b.png", "train", 1))
    elif case == "metadata":
        metadata = {"bad": float("nan")}
    elif case == "no_train":
        records = [ImageRecord(camera, 0.0, tmp_path / "a.png", "test", 0)]
    with pytest.raises(ValueError):
        write_scene_manifest(tmp_path / "scene.json", records, masks=masks, metadata=metadata)
    assert not (tmp_path / "scene.json").exists()
    assert list(tmp_path.iterdir()) == []
