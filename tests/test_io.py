# SPDX-License-Identifier: Apache-2.0
"""The file formats of §2.2 in openftgs.io: ftgs-init, ftgs-if, ftgs-checkpoint, and the PLY
snapshot (D-30; U-22 and U-26 file parts; Q-17, Q-18, Q-22, Q-24 to Q-26).

Expected values come from the §2.2 tables (keys, dtypes, shapes, validation rules) and from
values worked by hand; files with invalid content are written with safetensors directly, so the
readers' validation is tested without the writers'.
"""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.config import TrainConfig, config_to_dict
from openftgs.io import (
    load_checkpoint,
    read_exchange,
    read_init_points,
    save_checkpoint,
    validate_exchange,
    validate_init_points,
    write_exchange,
    write_init_points,
)
from openftgs.types import ExchangeModel, InitPoints, TimeNormalizer, TrainerState

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
        stats={"g_sum": torch.rand(n, generator=g), "count": torch.randint(0, 9, (n,))},
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
    "case", ["tampered params", "missing rng.json", "extra file", "version 2", "eight params"]
)
def test_checkpoint_loader_rejects_damaged_directories(tmp_path: Path, case: str) -> None:
    """§2.2 and §2.1 (integrity check): a changed or missing file, an unknown file, an unknown
    major version, or a params file without the nine tensors is rejected (Q-25, Q-26)."""
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
    elif case == "extra file":
        (out / "notes.txt").write_text("hello")
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
