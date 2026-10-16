# SPDX-License-Identifier: Apache-2.0
"""The ftgs command line (§2.1 CLI block and exit codes, §2.2; Q-124 to Q-142): a smoke test of
every command on a small synthetic scene, the CLI round trips of U-26, U-22's rejections with
exit code 2, U-27's exit code, and the train, resume, render, eval and export behaviour.

The tests call `main(argv)` in process, on the CPU unless marked gpu: `torch.cuda.is_available`
is patched for the CLI to read False (Q-125). A few subprocess tests run the real entry points.
"""

import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import (
    random_model,
    ring,
    shortened_config,
    write_scene,
)

from openftgs.cli import main
from openftgs.config import (
    GrowthConfig,
    InitConfig,
    LossConfig,
    PerfConfig,
    TrainConfig,
    config_to_dict,
)
from openftgs.eval import METRIC_NAMES
from openftgs.io import (
    load_checkpoint,
    write_init_points,
)
from openftgs.train import Trainer
from openftgs.types import InitPoints, TimeNormalizer

TAU1 = 2.0 / 30.0  # three frames at 30 FPS
TIME = TimeNormalizer(0.0, TAU1)
ITERATIONS = 20

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)
has_init_extra = all(importlib.util.find_spec(m) for m in ("einops", "loguru", "torchvision"))


@pytest.fixture(autouse=True)
def _cpu_unless_gpu(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-125: the CLI picks CUDA when PyTorch sees it; CPU tests make it see none."""
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _ftgs(*args: object) -> int:
    return main([str(a) for a in args])


def _scene(
    root: Path, *, test: bool = False, masks: bool = False, frames: int = 3, unit: str = "s"
) -> Path:
    """Three 48×32 ring cameras (LPIPS-Alex needs 31 px) training at frames k/30 around a
    random model; with `test`, a fourth camera is the test view."""
    cams = ring(4, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    gt = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images: list[Any] = [(cam, k / 30.0, "train") for k in range(frames) for cam in cams[:3]]
    if test:
        images += [(cams[3], k / 30.0, "test") for k in range(frames)]
    path = write_scene(root, gt, images, mask_tests=masks)
    if unit != "s":
        data = json.loads(path.read_text())
        data["time_unit"] = unit
        path.write_text(json.dumps(data))
    return path


def _init(path: Path, *, n: int = 30, unit: str = "s") -> Path:
    g = torch.Generator().manual_seed(1)
    points = InitPoints(
        positions=(torch.rand(n, 3, generator=g) - 0.5) * 1.2,
        colors=torch.full((n, 3), 0.5),
        timestamps=torch.rand(n, generator=g, dtype=torch.float64) * TAU1,
    )
    write_init_points(points, path, time_unit=unit, producer="openftgs tests")
    return path


def _cfg(iterations: int = ITERATIONS, **fields: Any) -> TrainConfig:
    """No perceptual loss (no downloads), a shortened calendar (§3.2, Q-75), logs every 5."""
    fields.setdefault("perf", PerfConfig(log_every=5))
    base = TrainConfig(loss=LossConfig(lambda_perc=0.0), **fields)
    cfg = shortened_config(iterations, base)
    return cfg


def _config_file(path: Path, cfg: TrainConfig | None = None, **override: Any) -> Path:
    data = config_to_dict(cfg if cfg is not None else _cfg())
    data.update(override)
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def run(tmp_path: Path) -> dict[str, Path]:
    """A scene, init points and a configuration file, without a test split."""
    return {
        "scene": _scene(tmp_path / "scene"),
        "init": _init(tmp_path / "init.safetensors"),
        "config": _config_file(tmp_path / "cfg.json"),
    }


def _train(run: dict[str, Path], out: Path, *extra: object) -> int:
    return _ftgs("train", run["scene"], "--init", run["init"], "--out", out, *extra)


def _params(checkpoint: Path) -> dict[str, torch.Tensor]:
    return load_checkpoint(checkpoint).params


# The entry points (Q-124).


def test_main_is_the_public_api() -> None:
    """Q-124: `openftgs.cli.__all__` is exactly `main`."""
    import openftgs.cli

    assert openftgs.cli.__all__ == ["main"]


def test_help_and_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """§2.1: --help is 0 for ftgs and every command; argparse's usage errors are exit 2."""
    commands = ("convert", "cache", "init", "train", "render", "eval", "export", "bench")
    assert _ftgs("--help") == 0
    for command in commands:
        assert _ftgs(command, "--help") == 0
    assert _ftgs() == 2
    assert _ftgs("fly") == 2
    assert _ftgs("export", "x", "--format", "ply", "--out", "y") == 2
    assert _ftgs("train", "scene.json") == 2  # --init and --out are required
    capsys.readouterr()


def test_the_ftgs_console_script_runs_main() -> None:
    """Q-124: the installed metadata names `ftgs = openftgs.cli:main`."""
    scripts = importlib.metadata.entry_points(group="console_scripts")
    assert [e.value for e in scripts if e.name == "ftgs"] == ["openftgs.cli:main"]


def test_python_m_runs_the_cli_and_reports_errors_without_a_traceback(tmp_path: Path) -> None:
    """Q-124, Q-142: `python -m openftgs.cli` gives 0 for --help, and 2 with a one-line error and
    no traceback for an invalid input."""
    ok = subprocess.run([sys.executable, "-m", "openftgs.cli", "--help"], capture_output=True)
    assert ok.returncode == 0 and b"ftgs" in ok.stdout
    missing = tmp_path / "absent.json"
    bad = subprocess.run(
        [sys.executable, "-m", "openftgs.cli", "bench", "raster", str(missing), "--out", "b"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "ftgs bench: error:" in bad.stderr and "Traceback" not in bad.stderr


# ftgs train (§2.1; Q-131 to Q-135).


def test_train_writes_the_run_directory(run: dict[str, Path], tmp_path: Path) -> None:
    """Q-132: config.json (the resolved configuration), train_log.jsonl (one row per log read:
    iteration, mean loss terms, N), checkpoint/ at T_it; no METRICS.json without a test split."""
    out = tmp_path / "run"
    assert _train(run, out, "--config", run["config"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint", "config.json", "train_log.jsonl"]
    assert json.loads((out / "config.json").read_text()) == config_to_dict(_cfg())
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [5, 10, 15, 20]
    assert all(set(r) == {"iteration", "losses", "num_gaussians"} for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r["losses"].values())
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS


def test_train_seed_overrides_the_configuration(run: dict[str, Path], tmp_path: Path) -> None:
    """§2.1: --seed replaces TrainConfig.seed: equal seeds give bitwise-equal runs (U-24's
    property through the CLI), another seed another run, and config.json holds the CLI seed."""
    cfg = _config_file(tmp_path / "seeded.json", _cfg(seed=5))
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        assert _train(run, tmp_path / name, "--config", cfg, "--seed", seed) == 0
    assert json.loads((tmp_path / "a" / "config.json").read_text())["seed"] == 7
    a, b, c = (_params(tmp_path / n / "checkpoint") for n in "abc")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def _sigterm_at(monkeypatch: pytest.MonkeyPatch, iteration: int) -> None:
    step = Trainer.step

    def stepping(self: Trainer) -> Any:
        log = step(self)
        if self.iteration == iteration:
            os.kill(os.getpid(), signal.SIGTERM)
        return log

    monkeypatch.setattr(Trainer, "step", stepping)


def test_a_signal_saves_a_checkpoint_from_which_resume_is_bitwise(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2.2, Q-133: SIGTERM stops training after the current iteration, saves the checkpoint and
    exits 143 (128 + 15); --resume then continues to T_it, bitwise equal to a straight run on
    the reference backend (§2.2 contract), appending to train_log.jsonl (Q-134)."""
    assert _train(run, tmp_path / "straight", "--config", run["config"]) == 0
    with monkeypatch.context() as m:
        _sigterm_at(m, 12)
        assert _train(run, tmp_path / "cut", "--config", run["config"]) == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    checkpoint = tmp_path / "cut" / "checkpoint"
    assert load_checkpoint(checkpoint).iteration == 12
    assert _train(run, tmp_path / "cut", "--resume", checkpoint) == 0
    straight, resumed = _params(tmp_path / "straight" / "checkpoint"), _params(checkpoint)
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)
    rows = (tmp_path / "cut" / "train_log.jsonl").read_text().splitlines()
    assert [json.loads(r)["iteration"] for r in rows] == [5, 10, 15, 20]


def test_resume_refuses_a_changed_manifest_unless_forced(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2.2, Q-70: a manifest whose hash changed since the checkpoint is exit 2 without --force;
    with --force the run continues. --force without --resume is exit 2 (Q-134)."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    data = json.loads(run["scene"].read_text())
    data["metadata"]["note"] = "edited"
    run["scene"].write_text(json.dumps(data))
    checkpoint = tmp_path / "cut" / "checkpoint"
    assert _train(run, tmp_path / "cut", "--resume", checkpoint) == 2
    assert _train(run, tmp_path / "cut", "--resume", checkpoint, "--force") == 0
    assert _train(run, tmp_path / "fresh", "--force") == 2


def test_resume_refuses_another_configuration(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-134: beside --resume, --config and --seed must give the checkpoint's configuration,
    PerfConfig included; giving it unchanged is accepted."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    ckpt = tmp_path / "cut" / "checkpoint"
    other_loss = _config_file(
        tmp_path / "loss.json", _cfg(), loss={"lambda_perc": 0.0, "lambda_reg": 0.5}
    )
    other_perf = _config_file(tmp_path / "perf.json", _cfg(perf=PerfConfig(log_every=4)))
    for extra in (("--config", other_loss), ("--config", other_perf), ("--seed", 3)):
        assert _train(run, tmp_path / "cut", "--resume", ckpt, *extra) == 2
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--config", run["config"]) == 0


def test_profile_writes_a_trace_of_its_window(run: dict[str, Path], tmp_path: Path) -> None:
    """§3.4, Q-132: --profile START:COUNT writes profile/trace.json, a Chrome trace; a window
    outside (iteration, T_it] or with START or COUNT below 1 is exit 2."""
    out = tmp_path / "run"
    assert _train(run, out, "--config", run["config"], "--profile", "3:2") == 0
    trace = json.loads((out / "profile" / "trace.json").read_text())
    assert trace["traceEvents"]
    for window in ("0:2", "3:0", "19:5", "5", "a:b"):
        assert (
            _train(run, tmp_path / f"bad{window}", "--config", run["config"], "--profile", window)
            == 2
        )


def test_profile_window_starts_after_a_resumed_iteration(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    ckpt = tmp_path / "cut" / "checkpoint"
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--profile", "8:2") == 2
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--profile", "11:2") == 0
    assert (tmp_path / "cut" / "profile" / "trace.json").is_file()


@pytest.mark.parametrize(
    "content",
    ["{", "[1, 2]", '{"colour": 1}', '{"sh_degree": 4}', '{"growth": {"n_max": 0}}'],
    ids=["malformed", "a list", "unknown key", "sh degree 4 (U-26)", "invalid value"],
)
def test_train_refuses_invalid_configurations(
    run: dict[str, Path], tmp_path: Path, content: str
) -> None:
    """Q-10, Q-11, U-26: a configuration file that isn't a valid JSON TrainConfig is exit 2."""
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    assert _train(run, tmp_path / "out", "--config", bad) == 2
    assert not (tmp_path / "out").exists()


def test_train_refuses_invalid_inputs(run: dict[str, Path], tmp_path: Path) -> None:
    """§2.1 exit code 2 (Q-142): a broken init file, an init file in another time unit, a missing
    scene, a non-empty run directory (Q-132), F·B > N_max (D-24, Q-128), and a configuration this
    build can't run (bf16, until M15)."""
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    assert _ftgs("train", run["scene"], "--init", broken, "--out", tmp_path / "a") == 2
    frames = _init(tmp_path / "frames.safetensors", unit="frame")
    assert _ftgs("train", run["scene"], "--init", frames, "--out", tmp_path / "b") == 2
    assert (
        _ftgs("train", tmp_path / "none.json", "--init", run["init"], "--out", tmp_path / "c") == 2
    )
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    assert _train(run, busy, "--config", run["config"]) == 2
    budget = _config_file(
        tmp_path / "budget.json",
        _cfg(init=InitConfig(points_per_frame=10), growth=GrowthConfig(n_max=20)),
    )
    assert _train(run, tmp_path / "d", "--config", budget) == 2
    bf16 = _config_file(tmp_path / "bf16.json", _cfg(perf=PerfConfig(perc_precision="bf16")))
    assert _train(run, tmp_path / "e", "--config", bf16) == 2


def test_divergence_is_exit_code_3_and_saves_nothing(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-69, Q-133: a model that turns NaN is a diverged run, exit code 3, with no checkpoint."""
    step = Trainer.step

    def poisoned(self: Trainer) -> Any:
        log = step(self)
        with torch.no_grad():
            self.model.means[0, 0] = float("nan")
        return log

    monkeypatch.setattr(Trainer, "step", poisoned)
    cfg = _config_file(tmp_path / "every.json", _cfg(perf=PerfConfig(log_every=1)))
    out = tmp_path / "run"
    assert _train(run, out, "--config", cfg) == 3
    assert not (out / "checkpoint").exists()


def test_deterministic_training_sets_the_cublas_workspace(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-131: with perf.deterministic, ftgs train sets CUBLAS_WORKSPACE_CONFIG unless the
    environment already sets it."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cfg = _config_file(
        tmp_path / "det.json", _cfg(perf=PerfConfig(log_every=5, deterministic=True))
    )
    assert _train(run, tmp_path / "a", "--config", cfg) == 0
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert _train(run, tmp_path / "b", "--config", cfg) == 0
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_train_refuses_a_partial_mask_set(tmp_path: Path) -> None:
    """Q-135 (#83): a test split with masks on some images but not all is exit 2 before any
    training."""
    scene = _scene(tmp_path / "scene", test=True, masks=True)
    data = json.loads(scene.read_text())
    tests = [e for e in data["images"] if e["split"] == "test"]
    del tests[0]["mask"]
    scene.write_text(json.dumps(data))
    init = _init(tmp_path / "init.safetensors")
    assert _ftgs("train", scene, "--init", init, "--out", tmp_path / "run") == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.weights
def test_train_evaluates_the_test_split_at_the_end(tmp_path: Path) -> None:
    """§1.3 calendar ("Final checkpoint and evaluation"), Q-132, Q-136: with a test split, the
    run directory gets METRICS.json naming the iteration, with a null model path."""
    scene = _scene(tmp_path / "scene", test=True)
    init = _init(tmp_path / "init.safetensors")
    cfg = _config_file(tmp_path / "cfg.json")
    out = tmp_path / "run"
    assert _ftgs("train", scene, "--init", init, "--out", out, "--config", cfg) == 0
    report = json.loads((out / "METRICS.json").read_text())
    assert report["model"] == {"path": None, "iteration": ITERATIONS}
    assert report["split"] == "test" and set(report["entire"]["scene"]) == set(METRIC_NAMES)
