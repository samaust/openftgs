# SPDX-License-Identifier: Apache-2.0
"""The file formats of §2.2 in openftgs.io: ftgs-init, ftgs-if, ftgs-checkpoint, and the PLY
snapshot (D-30; U-22 and U-26 file parts; Q-17, Q-18, Q-22, Q-24 to Q-26).

Expected values come from the §2.2 tables (keys, dtypes, shapes, validation rules) and from
values worked by hand; files with invalid content are written with safetensors directly, so the
readers' validation is tested without the writers'.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.io import (
    read_exchange,
    read_init_points,
    validate_exchange,
    validate_init_points,
    write_exchange,
    write_init_points,
)
from openftgs.types import ExchangeModel, InitPoints

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


def _exchange_arrays(n: int = 6, sh_degree: int = 3, seed: int = 0) -> dict[str, Tensor]:
    g = _gen(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[:, :1] < 0, -q, q)
    return {
        "position": torch.randn(n, 3, generator=g),
        "time_center": torch.rand(n, generator=g, dtype=F64) * 10.0 + 2.0,
        "duration": torch.rand(n, generator=g, dtype=F64) + 0.05,
        "velocity": torch.randn(n, 3, generator=g),
        "scale": torch.rand(n, 3, generator=g) + 0.01,
        "rotation": q,
        "opacity": torch.rand(n, generator=g) * 0.98 + 0.01,
        "sh": torch.randn(n, (sh_degree + 1) ** 2, 3, generator=g),
    }


def _exchange(sh_degree: int = 3, **arrays: Tensor) -> ExchangeModel:
    a = _exchange_arrays(sh_degree=sh_degree)
    a.update(arrays)
    return ExchangeModel(a, sh_degree, 2.0, 12.0, "s", "unit test")


def _exchange_metadata(**override: str) -> dict[str, str]:
    meta = {
        "format": "ftgs-if",
        "version": "1",
        "sh_degree": "3",
        "time_unit": "s",
        "tau0": "2.0",
        "tau1": "12.0",
        "producer": "test",
    }
    meta.update(override)
    return meta


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_ftgs_if_round_trip_for_each_sh_degree(tmp_path: Path, sh_degree: int) -> None:
    """U-26 (FTGS-IF round trip per L) and §2.2: every array, dtype and metadata field
    survives write → read bitwise."""
    model = ExchangeModel(_exchange_arrays(sh_degree=sh_degree), sh_degree, 0.1, 9.7, "frame", "x")
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


def test_ftgs_if_metadata_follows_section_2_2(tmp_path: Path) -> None:
    """§2.2 ftgs-if metadata: format, version, sh_degree, time_unit, tau0, tau1 (shortest
    round-trip repr), producer, and render_conventions as compact JSON from E6 and D-34 (Q-24)."""
    model = ExchangeModel(_exchange_arrays(), 3, 0.1, 1.0 / 3.0, "s", "unit test")
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
        "alpha_max": 0.999,
        "eps2d": 0.3,
        "near": 0.01,
        "far": 1e10,
        "alpha_skip": 1.0 / 255.0,
        "transmittance_stop": 1e-4,
        "pixel_center_offset": 0.5,
        "background": [0.0, 0.0, 0.0],
    }


def _bad_exchange(case: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    a = _exchange_arrays()
    meta = _exchange_metadata()
    if case == "nan position":
        a["position"][0, 0] = float("nan")
    elif case == "inf sh":
        a["sh"][1, 4, 2] = float("inf")
    elif case == "zero duration":
        a["duration"][2] = 0.0
    elif case == "negative duration":
        a["duration"][2] = -0.5
    elif case == "opacity 0":
        a["opacity"][3] = 0.0
    elif case == "opacity 1":
        a["opacity"][3] = 1.0
    elif case == "zero scale":
        a["scale"][0, 1] = 0.0
    elif case == "zero rotation":
        a["rotation"][4] = 0.0
    elif case == "rotation [N, 3]":
        a["rotation"] = a["rotation"][:, :3].contiguous()
    elif case == "sh of degree 2 for 3":
        a["sh"] = a["sh"][:, :9].contiguous()
    elif case == "velocity of another N":
        a["velocity"] = a["velocity"][:5].contiguous()
    elif case == "float32 time_center":
        a["time_center"] = a["time_center"].float()
    elif case == "float64 position":
        a["position"] = a["position"].double()
    elif case == "missing opacity":
        del a["opacity"]
    elif case == "empty":
        a = {k: v[:0] for k, v in a.items()}
    elif case == "version 2":
        meta["version"] = "2"
    elif case == "sh_degree 4":
        meta["sh_degree"] = "4"
        a["sh"] = torch.zeros(6, 25, 3)
    elif case == "tau1 < tau0":
        meta["tau1"] = "1.0"
    elif case == "missing tau0":
        del meta["tau0"]
    return a, meta


@pytest.mark.parametrize(
    "case",
    [
        "nan position",
        "inf sh",
        "zero duration",
        "negative duration",
        "opacity 0",
        "opacity 1",
        "zero scale",
        "zero rotation",
        "rotation [N, 3]",
        "sh of degree 2 for 3",
        "velocity of another N",
        "float32 time_center",
        "float64 position",
        "missing opacity",
        "empty",
        "version 2",
        "sh_degree 4",
        "tau1 < tau0",
        "missing tau0",
    ],
)
def test_u22_ftgs_if_validator_rejects_invalid_files(tmp_path: Path, case: str) -> None:
    """U-22: the validator rejects NaN, durations ≤ 0, opacity outside (0, 1), wrong shapes and
    an unknown major version; and the other §2.2 rules (Q-24): dtypes, keys, ranges, N ≥ 1."""
    arrays, meta = _bad_exchange(case)
    path = tmp_path / "bad.safetensors"
    save_file(arrays, str(path), metadata=meta)
    with pytest.raises(ValueError):
        read_exchange(path)


def test_ftgs_if_reader_ignores_unknown_keys_and_render_conventions(tmp_path: Path) -> None:
    """§2.2: unknown optional keys are ignored; render_conventions is accepted and ignored (Q-24);
    a unit rotation with w < 0 is accepted on read."""
    arrays = _exchange_arrays()
    arrays["rotation"][0] = -arrays["rotation"][0]
    arrays["confidence"] = torch.ones(6)
    path = tmp_path / "model.safetensors"
    meta = _exchange_metadata(version="1.2", render_conventions="{}", comment="hi")
    save_file(arrays, str(path), metadata=meta)
    back = read_exchange(path)
    assert set(back.arrays) == set(arrays) - {"confidence"}
    assert torch.equal(back.arrays["rotation"], arrays["rotation"])


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
