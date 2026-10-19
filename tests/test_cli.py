# SPDX-License-Identifier: Apache-2.0
"""The ftgs command line (§2.1 CLI block and exit codes, §2.2; D-83; Q-124 to Q-152): a
smoke test of every command on a small synthetic scene, the CLI round trips of U-26, U-22's
rejections with exit code 2, U-27's exit code, and the train, resume, render, eval and export
behaviour.

The tests call `main(argv)` in process, on the CPU unless marked gpu: `torch.cuda.is_available`
is patched for the CLI to read False (Q-125). A few subprocess tests run the real entry points.
"""

import hashlib
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
from typing import Any, Literal

import pytest
import torch
from builders import (
    BAD_EXCHANGE_CASES,
    TexturedPlane,
    bad_exchange,
    look_at,
    n3dv_source,
    random_model,
    ring,
    selfcap_archives,
    shortened_config,
    stub_colmap,
    write_plane_scene,
    write_render_request,
    write_scene,
)
from safetensors import safe_open
from safetensors.torch import save_file

from openftgs.cli import main
from openftgs.config import (
    GrowthConfig,
    InitConfig,
    LossConfig,
    PerfConfig,
    TrainConfig,
    config_to_dict,
)
from openftgs.data import FrameCache
from openftgs.eval import METRIC_NAMES
from openftgs.io import (
    load_checkpoint,
    read_exchange,
    read_init_points,
    read_renders,
    save_checkpoint,
    write_init_points,
    write_renders,
)
from openftgs.model import FreeTimeGaussians
from openftgs.train import Trainer
from openftgs.types import InitPoints, Renders, TimeNormalizer

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


def _sigterm_at(
    monkeypatch: pytest.MonkeyPatch, iteration: int, number: int = signal.SIGTERM
) -> None:
    step = Trainer.step

    def stepping(self: Trainer) -> Any:
        log = step(self)
        if self.iteration == iteration:
            os.kill(os.getpid(), number)
        return log

    monkeypatch.setattr(Trainer, "step", stepping)


def _tensors(tree: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Every tensor of a nested dict, by its path."""
    if isinstance(tree, torch.Tensor):
        return {prefix: tree}
    if isinstance(tree, dict):
        return {
            k: v
            for key, value in tree.items()
            for k, v in _tensors(value, f"{prefix}/{key}").items()
        }
    return {}


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_signal_saves_a_checkpoint_from_which_resume_is_bitwise(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    """§2.2, Q-133: SIGINT or SIGTERM stops training after the current iteration, saves the
    checkpoint and exits 128 + its number (130, 143); --resume then continues to T_it, bitwise
    equal to a straight run on the reference backend in parameters, Adam moments and step
    counters, gradient statistics and generator states (§2.2 contract), appending to
    train_log.jsonl (Q-134). Iteration 12 is not a log read, so the stop doesn't wait for one."""
    before = signal.getsignal(number)
    assert _train(run, tmp_path / "straight", "--config", run["config"]) == 0
    with monkeypatch.context() as m:
        _sigterm_at(m, 12, number)
        assert _train(run, tmp_path / "cut", "--config", run["config"]) == 128 + number
    assert signal.getsignal(number) == before
    checkpoint = tmp_path / "cut" / "checkpoint"
    assert load_checkpoint(checkpoint).iteration == 12
    assert _train(run, tmp_path / "cut", "--resume", checkpoint) == 0
    assert load_checkpoint(checkpoint).iteration == ITERATIONS
    _assert_same_run(tmp_path / "straight" / "checkpoint", checkpoint)
    rows = (tmp_path / "cut" / "train_log.jsonl").read_text().splitlines()
    assert [json.loads(r)["iteration"] for r in rows] == [5, 10, 15, 20]


def test_a_model_that_turns_non_finite_after_the_last_check_is_not_saved(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-133: the trainer checks for divergence at log reads and events (Q-69); a NaN after the
    last one is still caught before the save: exit code 3, no checkpoint."""
    step = Trainer.step

    def late(self: Trainer) -> Any:
        log = step(self)
        if self.iteration == ITERATIONS - 1:
            with torch.no_grad():
                self.model.means[0, 0] = float("nan")
        return log

    monkeypatch.setattr(Trainer, "step", late)
    cfg = _config_file(tmp_path / "seven.json", _cfg(perf=PerfConfig(log_every=7)))
    out = tmp_path / "run"
    assert _train(run, out, "--config", cfg) == 3
    assert not (out / "checkpoint").exists()


def test_a_signal_in_the_last_iteration_finishes_the_run(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-133: a run that reached T_it when the signal came is complete: exit 0."""
    _sigterm_at(monkeypatch, ITERATIONS)
    out = tmp_path / "run"
    assert _train(run, out, "--config", run["config"]) == 0
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS


def test_resume_writes_only_into_its_own_run_directory(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-134: --resume writes into the run directory of its configuration, or an absent or
    empty one; another non-empty directory is exit 2 and left as it was."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    checkpoint = tmp_path / "cut" / "checkpoint"
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("x")
    assert _train(run, other, "--resume", checkpoint) == 2
    assert sorted(p.name for p in other.iterdir()) == ["notes.txt"]
    assert _train(run, tmp_path / "fresh", "--resume", checkpoint) == 0


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


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"loss": {**config_to_dict(_cfg())["loss"], "lambda_reg": 0.5}}, "loss.lambda_reg"),
        ({"perf": {"log_every": 5, "adam": "foreach"}}, "perf.adam"),  # D-53
        ({"perf": {"log_every": 5, "perc_precision": "ieee"}}, "perf.perc_precision"),  # D-52
        ({"perf": {"log_every": 5, "images_per_step": 2}}, "perf.images_per_step"),  # D-54
    ],
    ids=["loss", "adam", "perc_precision", "images_per_step"],
)
def test_resume_refuses_another_configuration(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    override: dict[str, Any],
    key: str,
) -> None:
    """Q-134 (as answered), Q-153: beside --resume, --config and --seed must give the
    checkpoint's configuration outside the PerfConfig fields that change no number:
    perc_precision, adam and images_per_step must match, as must every other field; a
    difference is exit 2 naming its dotted key, and giving the configuration unchanged is
    accepted."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    ckpt = tmp_path / "cut" / "checkpoint"
    other = _config_file(tmp_path / "other.json", _cfg(), **override)
    capsys.readouterr()
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--config", other) == 2
    assert key in capsys.readouterr().err
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--seed", 3) == 2
    assert "seed" in capsys.readouterr().err
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--config", run["config"]) == 0


def test_a_resume_may_change_the_exact_performance_settings(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-134 (as answered), §2.2: a run cut by a signal and resumed into its own run directory
    with a --config that changes only exact PerfConfig fields (the frame cache placement, the
    prefetch depth and the log period) exits 0, gives bitwise the straight run, and its
    config.json and checkpoint record the new settings."""
    assert _train(run, tmp_path / "straight", "--config", run["config"]) == 0
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        assert _train(run, tmp_path / "cut", "--config", run["config"]) == 128 + signal.SIGTERM
    perf = PerfConfig(log_every=4, frame_cache="off", prefetch=2)  # "auto" is mmap on the CPU
    new = _config_file(tmp_path / "mmap.json", _cfg(perf=perf))
    checkpoint = tmp_path / "cut" / "checkpoint"
    assert _train(run, tmp_path / "cut", "--resume", checkpoint, "--config", new) == 0
    _assert_same_run(tmp_path / "straight" / "checkpoint", checkpoint)
    recorded = json.loads((tmp_path / "cut" / "config.json").read_text())
    assert recorded == config_to_dict(_cfg(perf=perf))
    assert load_checkpoint(checkpoint).config == recorded


def test_a_resume_into_its_own_directory_accepts_other_exact_settings(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-134 (as answered): the run-directory check compares only the fields that must match,
    so a run directory whose config.json differs from the checkpoint's only in exact
    PerfConfig fields is the run's own; one differing elsewhere, or holding no configuration
    object, is not, and is exit 2."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    checkpoint = tmp_path / "cut" / "checkpoint"
    recorded = tmp_path / "cut" / "config.json"
    data = json.loads(recorded.read_text())
    data["perf"]["prefetch"] = 3
    own = json.dumps(data)
    for bad in ("null", "[[1]]"):  # not a configuration: exit 2, not a traceback
        recorded.write_text(bad)
        assert _train(run, tmp_path / "cut", "--resume", checkpoint) == 2
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "cut", elsewhere)
    (elsewhere / "config.json").write_text(json.dumps({**data, "seed": 7}))
    assert _train(run, elsewhere, "--resume", checkpoint) == 2
    recorded.write_text(own)
    assert _train(run, tmp_path / "cut", "--resume", checkpoint) == 0


@pytest.mark.parametrize(("preset", "want"), [(None, ":4096:8"), (":16:8", ":16:8")])
def test_a_resume_that_turns_deterministic_on_sets_the_cublas_workspace(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: str | None,
    want: str,
) -> None:
    """Q-131, Q-134 (as answered): the process-wide setting follows the configuration the
    resumed run uses, so a resume whose --config turns deterministic on sets
    CUBLAS_WORKSPACE_CONFIG, unless the environment already sets it."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", preset or "restored after the test")
    if preset is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    on = _config_file(tmp_path / "on.json", _cfg(perf=PerfConfig(log_every=5, deterministic=True)))
    checkpoint = tmp_path / "cut" / "checkpoint"
    assert _train(run, tmp_path / "cut", "--resume", checkpoint, "--config", on) == 0
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == want


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
    """Q-144: after a resume at iteration 10 the window must start after it; a window that a
    signal cuts short still writes the trace of the iterations it ran."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        _train(run, tmp_path / "cut", "--config", run["config"])
    ckpt = tmp_path / "cut" / "checkpoint"
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--profile", "8:2") == 2
    assert _train(run, tmp_path / "cut", "--resume", ckpt, "--profile", "11:2") == 0
    assert (tmp_path / "cut" / "profile" / "trace.json").is_file()
    with monkeypatch.context() as m:
        _sigterm_at(m, 4)
        assert _train(run, tmp_path / "short", "--config", run["config"], "--profile", "3:5") == 143
    assert json.loads((tmp_path / "short" / "profile" / "trace.json").read_text())["traceEvents"]


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


def test_train_refuses_invalid_inputs(
    run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    capsys.readouterr()
    assert _train(run, tmp_path / "d", "--config", budget) == 2
    assert "F·B" in capsys.readouterr().err  # D-24's own error
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


def test_train_refuses_a_partial_mask_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-135 (#83): a test split with masks on some images but not all is exit 2 before any
    training."""
    scene = _scene(tmp_path / "scene", test=True, masks=True)
    data = json.loads(scene.read_text())
    tests = [e for e in data["images"] if e["split"] == "test"]
    del tests[0]["mask"]
    scene.write_text(json.dumps(data))
    init = _init(tmp_path / "init.safetensors")
    assert _ftgs("train", scene, "--init", init, "--out", tmp_path / "run") == 2
    assert tests[0]["path"] in capsys.readouterr().err
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
    assert report["model"] == {"path": None, "iteration": ITERATIONS, "sha256": None}
    assert report["split"] == "test" and set(report["entire"]["scene"]) == set(METRIC_NAMES)


# Periodic checkpoints (D-83; Q-149 to Q-152).


def _kept(run_dir: Path) -> dict[str, int]:
    """The run directory's checkpoints by name, with the iteration each one holds."""
    return {
        p.name: load_checkpoint(p).iteration
        for p in sorted(run_dir.iterdir())
        if p.name == "checkpoint" or p.name.startswith("checkpoint-")
    }


def _assert_same_run(a: Path, b: Path) -> None:
    """§2.2's resume contract, checkpoint against checkpoint: parameters, Adam moments and
    steps, statistics, generator states and sampler, bitwise."""
    x, y = load_checkpoint(a), load_checkpoint(b)
    assert x.iteration == y.iteration
    for part in ("params", "optim", "stats", "rng"):
        p, q = _tensors(getattr(x, part)), _tensors(getattr(y, part))
        assert p.keys() == q.keys() and p, part
        assert all(torch.equal(p[k], q[k]) for k in p), part
    assert x.sampler == y.sampler


def test_periodic_saves_change_nothing_in_the_run(run: dict[str, Path], tmp_path: Path) -> None:
    """D-83, §2.2: a run saving every 5 iterations is bitwise equal to one that doesn't, on the
    reference backend on the CPU; by default only RUN_DIR/checkpoint is kept, and neither
    option is part of the configuration."""
    assert _train(run, tmp_path / "plain", "--config", run["config"]) == 0
    periodic = tmp_path / "periodic"
    assert _train(run, periodic, "--config", run["config"], "--checkpoint-every", 5) == 0
    _assert_same_run(tmp_path / "plain" / "checkpoint", periodic / "checkpoint")
    assert _kept(periodic) == {"checkpoint": ITERATIONS}
    assert (periodic / "config.json").read_text() == (
        tmp_path / "plain" / "config.json"
    ).read_text()


def test_keep_checkpoints_keeps_the_k_latest_saves(run: dict[str, Path], tmp_path: Path) -> None:
    """D-83: with K = 2, exactly two checkpoints remain, the latest at RUN_DIR/checkpoint and the
    one before as RUN_DIR/checkpoint-<its iteration>."""
    out = tmp_path / "run"
    args = ("--config", run["config"], "--checkpoint-every", 5, "--keep-checkpoints", 2)
    assert _train(run, out, *args) == 0
    assert _kept(out) == {"checkpoint": ITERATIONS, "checkpoint-15": 15}


def test_a_resume_from_a_periodic_checkpoint_is_bitwise(
    run: dict[str, Path], tmp_path: Path
) -> None:
    """D-83, §2.2: resuming from a kept periodic checkpoint to T_it gives bitwise the run that
    never stopped."""
    assert _train(run, tmp_path / "straight", "--config", run["config"]) == 0
    out = tmp_path / "run"
    args = ("--config", run["config"], "--checkpoint-every", 5, "--keep-checkpoints", 2)
    assert _train(run, out, *args) == 0
    resumed = tmp_path / "resumed"
    assert _train(run, resumed, "--resume", out / "checkpoint-15") == 0
    _assert_same_run(tmp_path / "straight" / "checkpoint", resumed / "checkpoint")


@pytest.mark.parametrize("stop", [12, 15])
def test_a_resume_from_an_older_checkpoint_removes_the_later_ones(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: int
) -> None:
    """D-83, Q-149: the resumed run's first save drops the kept checkpoints of its iteration or
    later, the continuation the resume from an earlier one abandoned: at 12 a signal's save,
    at 15 the periodic save the abandoned run also made, then the signal's in place."""
    out = tmp_path / "run"
    keep = ("--checkpoint-every", 5, "--keep-checkpoints", 3)
    assert _train(run, out, "--config", run["config"], *keep) == 0
    assert _kept(out) == {"checkpoint": ITERATIONS, "checkpoint-10": 10, "checkpoint-15": 15}
    _sigterm_at(monkeypatch, stop)
    assert _train(run, out, "--resume", out / "checkpoint-10", *keep) == 128 + signal.SIGTERM
    assert _kept(out) == {"checkpoint": stop, "checkpoint-10": 10}


def test_a_resume_of_a_finished_run_keeps_its_checkpoints(
    run: dict[str, Path], tmp_path: Path
) -> None:
    """Q-149: resuming a checkpoint at T_it saves T_it again, in place, so the run directory's
    checkpoints stay as they were."""
    out = tmp_path / "run"
    keep = ("--checkpoint-every", 5, "--keep-checkpoints", 2)
    assert _train(run, out, "--config", run["config"], *keep) == 0
    assert _train(run, out, "--resume", out / "checkpoint", *keep) == 0
    assert _kept(out) == {"checkpoint": ITERATIONS, "checkpoint-15": 15}


@pytest.mark.parametrize(
    "option", [("--checkpoint-every", -1), ("--keep-checkpoints", 0)], ids=["every", "keep"]
)
def test_checkpoint_options_out_of_range_are_exit_code_2(
    run: dict[str, Path], tmp_path: Path, option: tuple[str, int]
) -> None:
    """D-83: N ≥ 0 and K ≥ 1; anything else is exit code 2 and no run directory is made."""
    out = tmp_path / "run"
    assert _train(run, out, "--config", run["config"], *option) == 2
    assert not out.exists()


def test_a_diverged_state_is_never_saved_periodically(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-83, Q-133: every save checks the model and its statistics first, so a model that turns
    NaN before a periodic save is exit code 3, nothing is saved or renamed, and the earlier
    checkpoint is untouched. The NaN comes after the shortened calendar's last event (17) and
    the save at 18 is no log read (7, 14), so the trainer's own checks (Q-69) don't see it
    first."""
    step = Trainer.step

    def late(self: Trainer) -> Any:
        log = step(self)
        if self.iteration == 17:
            with torch.no_grad():
                self.model.means[0, 0] = float("nan")
        return log

    monkeypatch.setattr(Trainer, "step", late)
    cfg = _config_file(tmp_path / "seven.json", _cfg(perf=PerfConfig(log_every=7)))
    out = tmp_path / "run"
    keep = ("--checkpoint-every", 9, "--keep-checkpoints", 2)
    assert _train(run, out, "--config", cfg, *keep) == 3
    assert _kept(out) == {"checkpoint": 9}


@pytest.mark.parametrize(
    ("stop", "want"),
    [(12, {"checkpoint": 12, "checkpoint-10": 10}), (10, {"checkpoint": 10, "checkpoint-5": 5})],
)
def test_a_signal_save_follows_the_rotation(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: int,
    want: dict[str, int],
) -> None:
    """D-83: a signal's save is rotated like any other; Q-149: a signal during an iteration that
    is a multiple of N saves that iteration once, replacing the periodic save in place."""
    _sigterm_at(monkeypatch, stop)
    out = tmp_path / "run"
    keep = ("--checkpoint-every", 5, "--keep-checkpoints", 2)
    assert _train(run, out, "--config", run["config"], *keep) == 128 + signal.SIGTERM
    assert _kept(out) == want


# Models for render, eval and export: a checkpoint trained through the Python API on the CPU.


def _trained(
    root: Path,
    *,
    sh_degree: int = 1,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    masks: bool = False,
) -> tuple[Path, Path]:
    """A scene with a test split and a checkpoint of a 10-iteration run on it."""
    from openftgs.data import MultiViewVideo

    scene = _scene(root / "scene", test=True, masks=masks)
    video = MultiViewVideo.from_manifest(scene)
    cfg = _cfg(10, sh_degree=sh_degree, background=background)
    trainer = Trainer(cfg, video, read_init_points(_init(root / "init.safetensors")), device="cpu")
    trainer.train()
    checkpoint = root / "checkpoint"
    save_checkpoint(trainer.state_dict(), checkpoint)
    return scene, checkpoint


def _test_items(scene: Path) -> list[tuple[str, float, Any]]:
    """One request item per test image of the scene, named after its frame index."""
    from openftgs.data import MultiViewVideo

    video = MultiViewVideo.from_manifest(scene)
    return [
        (f"t{r.frame_index}", r.timestamp, r.camera) for r in video.records if r.split == "test"
    ]


# ftgs render (§2.1, §2.2 render request and renders; Q-125, Q-129).


def test_render_answers_a_request(tmp_path: Path) -> None:
    """§2.2: rgb/<id> and alpha/<id> for every item, and the SHA-256 of the request's bytes."""
    scene, checkpoint = _trained(tmp_path)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "out" / "renders.safetensors"
    assert (
        _ftgs("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
        == 0
    )
    sha = hashlib.sha256(request.read_bytes()).hexdigest()
    renders = read_renders(out, request_sha256=sha)
    assert sorted(renders.rgb) == ["t10", "t11", "t9"] == sorted(renders.alpha)
    assert renders.rgb["t9"].shape == (32, 48, 3)
    with safe_open(str(out), framework="pt") as f:
        assert f.metadata()["request_sha256"] == sha


def test_render_refuses_invalid_requests_and_the_default_backend_without_cuda(
    tmp_path: Path,
) -> None:
    """Q-125: gsplat (the default) without CUDA is exit 2; so are a request in another time
    unit, one that fails §2.2's schema, a missing one and one above the model's SH degree."""
    scene, checkpoint = _trained(tmp_path)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "r.safetensors"
    assert _ftgs("render", checkpoint, "--request", request, "--out", out) == 2
    frames = write_render_request(tmp_path / "frames.json", _test_items(scene), unit="frame")
    ref = ("--backend", "reference")
    assert _ftgs("render", checkpoint, "--request", frames, "--out", out, *ref) == 2
    broken = tmp_path / "broken.json"
    broken.write_text('{"format": "ftgs-render-request", "version": 1}')
    assert _ftgs("render", checkpoint, "--request", broken, "--out", out, *ref) == 2
    assert _ftgs("render", checkpoint, "--request", tmp_path / "none.json", "--out", out, *ref) == 2
    above = json.loads(request.read_text())
    above["sh_degree"] = 3  # the model is of degree 1
    (tmp_path / "above.json").write_text(json.dumps(above))
    assert (
        _ftgs("render", checkpoint, "--request", tmp_path / "above.json", "--out", out, *ref) == 2
    )
    assert not out.exists()


def test_renders_that_are_not_finite_are_exit_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-129: a non-finite render is a numerical failure, and nothing is written."""
    import openftgs.render

    scene, checkpoint = _trained(tmp_path)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    original = openftgs.render.render_request

    def nan(*args: Any, **kwargs: Any) -> Any:
        renders = original(*args, **kwargs)
        renders.rgb["t9"] = torch.full_like(renders.rgb["t9"], float("nan"))
        return renders

    monkeypatch.setattr(openftgs.render, "render_request", nan)
    out = tmp_path / "r.safetensors"
    assert (
        _ftgs("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
        == 3
    )
    assert not out.exists()


@pytest.mark.gpu
def test_render_with_gsplat(tmp_path: Path) -> None:
    """§2.1: the default backend on a CUDA device, close to the reference backend's renders."""
    scene, checkpoint = _trained(tmp_path)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    gsplat, reference = tmp_path / "g.safetensors", tmp_path / "r.safetensors"
    assert _ftgs("render", checkpoint, "--request", request, "--out", gsplat) == 0
    assert (
        _ftgs(
            "render", checkpoint, "--request", request, "--out", reference, "--backend", "reference"
        )
        == 0
    )
    a, b = read_renders(gsplat), read_renders(reference)
    for key in a.rgb:
        torch.testing.assert_close(a.rgb[key], b.rgb[key], rtol=0.0, atol=1e-4)  # U-7


# ftgs export (§2.1; D-84) and U-22's rejections with exit code 2.


def test_export_writes_a_valid_exchange_model_with_the_runs_background(tmp_path: Path) -> None:
    """D-84: the file records the checkpoint's TrainConfig.background in
    render_conventions; a default run records black; an existing file is replaced (Q-126)."""
    _, white = _trained(tmp_path / "white", background=(1.0, 1.0, 1.0))
    _, black = _trained(tmp_path / "black")
    for checkpoint, want in ((white, [1.0, 1.0, 1.0]), (black, [0.0, 0.0, 0.0])):
        out = tmp_path / "exports" / f"{checkpoint.parent.name}.safetensors"
        for _ in range(2):
            assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", out) == 0
        read_exchange(out)
        with safe_open(str(out), framework="pt") as f:
            assert json.loads(f.metadata()["render_conventions"])["background"] == want


def test_export_needs_a_checkpoint_directory(tmp_path: Path) -> None:
    """§2.1: ftgs export reads a checkpoint directory; an exchange model or a missing path is
    exit 2."""
    _, checkpoint = _trained(tmp_path)
    exchange = tmp_path / "m.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", exchange) == 0
    again = tmp_path / "again.safetensors"
    assert _ftgs("export", exchange, "--format", "ftgs-if", "--out", again) == 2
    assert _ftgs("export", tmp_path / "none", "--format", "ftgs-if", "--out", again) == 2


def test_u22_an_opacity_logit_of_20_exports_and_validates(tmp_path: Path) -> None:
    """U-22: a saturated logit exports as 1 − 10⁻⁷ (float64 path, clamped) and the file
    validates."""
    _, checkpoint = _trained(tmp_path)
    state = load_checkpoint(checkpoint)
    state.params["opacity_logits"][0] = 20.0
    saturated = tmp_path / "saturated"
    save_checkpoint(state, saturated)
    out = tmp_path / "m.safetensors"
    assert _ftgs("export", saturated, "--format", "ftgs-if", "--out", out) == 0
    opacity = read_exchange(out).arrays["opacity"]
    assert torch.equal(opacity[0], torch.tensor(1.0 - 1e-7, dtype=torch.float32))


@pytest.mark.parametrize("case", BAD_EXCHANGE_CASES)
def test_u22_invalid_exchange_models_are_exit_code_2(tmp_path: Path, case: str) -> None:
    """U-22: every file the validator rejects is exit code 2 as ftgs render's and ftgs eval's
    MODEL; eval refuses before building its evaluator, so no weights are needed."""
    scene = _scene(tmp_path / "scene", test=True)
    arrays, meta = bad_exchange(case)
    bad = tmp_path / "bad.safetensors"
    save_file(arrays, str(bad), metadata=meta)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "out.safetensors"
    assert _ftgs("render", bad, "--request", request, "--out", out, "--backend", "reference") == 2
    assert _ftgs("eval", bad, scene, "--out", tmp_path / "m.json") == 2
    assert not out.exists() and not (tmp_path / "m.json").exists()


def test_a_corrupted_checkpoint_is_exit_code_2(tmp_path: Path) -> None:
    """§2.2, Q-25: a checkpoint whose file no longer matches its SHA-256 is exit code 2."""
    scene, checkpoint = _trained(tmp_path)
    params = checkpoint / "params.safetensors"
    data = bytearray(params.read_bytes())
    data[-1] ^= 0xFF
    params.write_bytes(bytes(data))
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "out.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", out) == 2
    assert (
        _ftgs("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
        == 2
    )


# U-26 through the command line (D-3).


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_cli_round_trips_for_each_sh_degree(tmp_path: Path, sh_degree: int) -> None:
    """U-26: a 50-iteration ftgs train finishes without NaN; ftgs export writes an FTGS-IF file
    of degree L, exactly the checkpoint's exchange model; ftgs render of the checkpoint and of the
    file agree within U-22's 1e-6."""
    scene = _scene(tmp_path / "scene", test=True)
    init = _init(tmp_path / "init.safetensors")
    cfg = _config_file(tmp_path / "cfg.json", _cfg(50, sh_degree=sh_degree))
    out = tmp_path / "run"
    # Training on a scene with a test split ends with an evaluation, which needs LPIPS weights:
    # train on a copy of the manifest without the test split, and render the test views.
    data = json.loads(scene.read_text())
    data["images"] = [e for e in data["images"] if e["split"] == "train"]
    train_only = scene.parent / "train.json"
    train_only.write_text(json.dumps(data))
    assert _ftgs("train", train_only, "--init", init, "--out", out, "--config", cfg) == 0
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert all(math.isfinite(v) for r in rows for v in r["losses"].values())
    checkpoint = out / "checkpoint"
    exchange = tmp_path / "model.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", exchange) == 0
    written = read_exchange(exchange)
    expected = FreeTimeGaussians.load(checkpoint, device="cpu").to_exchange()
    assert written.sh_degree == sh_degree
    assert written.arrays["sh"].shape[1] == (sh_degree + 1) ** 2
    assert all(torch.equal(written.arrays[k], expected.arrays[k]) for k in expected.arrays)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    ref = ("--backend", "reference")
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    assert _ftgs("render", checkpoint, "--request", request, "--out", a, *ref) == 0
    assert _ftgs("render", exchange, "--request", request, "--out", b, *ref) == 0
    ra, rb = read_renders(a), read_renders(b)
    for key in ra.rgb:
        torch.testing.assert_close(ra.rgb[key], rb.rgb[key], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(ra.alpha[key], rb.alpha[key], rtol=1e-6, atol=1e-6)


# ftgs eval (§2.1 CLI paragraph on METRICS.json; Q-135 to Q-137, D-84).


def _body(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k not in ("producer", "model", "renders")}


def _assert_close(a: Any, b: Any, path: str = "body") -> None:
    """Equal JSON trees, floats within 1e-6, relative or absolute (U-22's render tolerance, for
    metrics of renders in [0, 1]; Q-162)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _assert_close(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{k}]")
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-6, abs=1e-6), path
    else:
        assert a == b, path


def _renders_dir(tmp_path: Path, checkpoint: Path, items: list[tuple[str, float, Any]]) -> Path:
    """DIR of Q-137: request.json and its renders by the reference backend."""
    directory = tmp_path / "supplied"
    request = write_render_request(directory / "request.json", items)
    out = directory / "renders.safetensors"
    assert (
        _ftgs("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
        == 0
    )
    return directory


@pytest.mark.weights
def test_eval_writes_metrics_for_a_checkpoint_and_an_exchange_model(tmp_path: Path) -> None:
    """§2.1, Q-63, Q-136: METRICS.json with every metric under `entire`, `dynamic` for a masked
    split (with no image missing a mask, Q-135), and the header: `model` = {path, iteration,
    sha256}, the iteration null for an exchange model, the hash that of the exchange file or of
    the checkpoint's manifest.json (Q-136 as answered), and the scene manifest."""
    scene, checkpoint = _trained(tmp_path, masks=True)
    out = tmp_path / "m.json"
    assert _ftgs("eval", checkpoint, scene, "--split", "test", "--out", out) == 0
    report = json.loads(out.read_text())
    assert report["format"] == "ftgs-metrics" and report["split"] == "test"
    manifest_sha = hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
    assert report["model"] == {"path": str(checkpoint), "iteration": 10, "sha256": manifest_sha}
    assert report["manifest"]["sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert set(report["entire"]["scene"]) == set(METRIC_NAMES)
    assert len(report["entire"]["images"]) == 3
    # The masks' 24-px halves are below Q-61's smallest region, so each image is skipped.
    assert report["dynamic"]["missing"] == 0 and len(report["dynamic"]["images"]) == 3
    exchange = tmp_path / "m.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", exchange) == 0
    assert _ftgs("eval", exchange, scene, "--out", out) == 0
    assert json.loads(out.read_text())["model"] == {
        "path": str(exchange),
        "iteration": None,
        "sha256": hashlib.sha256(exchange.read_bytes()).hexdigest(),
    }


def test_eval_of_a_model_that_changes_while_it_is_read_is_exit_code_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-154, Q-136: ftgs eval hashes MODEL before and after reading it; a checkpoint whose
    manifest.json changes in between, as a periodic save changes RUN_DIR/checkpoint (D-83),
    is exit 2 and nothing is written, before any LPIPS weights are needed."""
    import openftgs.io

    scene, checkpoint = _trained(tmp_path)
    original = openftgs.io.load_checkpoint

    def saved_meanwhile(directory: Path) -> Any:
        state = original(directory)
        manifest = Path(directory) / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return state

    monkeypatch.setattr("openftgs.io.load_checkpoint", saved_meanwhile)
    out = tmp_path / "m.json"
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 2
    assert not out.exists()


@pytest.mark.weights
def test_eval_of_a_checkpoint_matches_the_trainers_evaluation(tmp_path: Path) -> None:
    """D-84, Q-74: ftgs eval of a checkpoint renders against its configuration's background,
    so a white-background run's report body equals Trainer.evaluate's; its export records that
    background and ftgs eval of the exchange model gives the same body."""
    from openftgs.data import MultiViewVideo

    scene, checkpoint = _trained(tmp_path, background=(1.0, 1.0, 1.0))
    out = tmp_path / "cli.json"
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 0
    trainer = Trainer.from_checkpoint(checkpoint, MultiViewVideo.from_manifest(scene), device="cpu")
    trainer.evaluate(out_dir=tmp_path / "trainer")
    want = json.loads((tmp_path / "trainer" / "METRICS.json").read_text())
    got = json.loads(out.read_text())
    assert _body(got) == _body(want)
    assert set(got) == set(want) and set(got["model"]) == set(want["model"])  # Q-136
    # D-84: the exported model records the background, and ftgs eval renders it on that.
    exchange = tmp_path / "white.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", exchange) == 0
    exported = tmp_path / "exported.json"
    assert _ftgs("eval", exchange, scene, "--out", exported) == 0
    # Q-162: the exchange model stores opacities and durations, not their logits, so its
    # renders match the checkpoint's within U-22's 1e-6, and the bodies are compared there.
    _assert_close(_body(json.loads(exported.read_text())), _body(got))


@pytest.mark.weights
def test_eval_scores_supplied_renders_as_its_own(tmp_path: Path) -> None:
    """Q-137: ftgs render --backend reference then ftgs eval --renders-dir gives the body of
    ftgs eval without it; the header names what was scored."""
    scene, checkpoint = _trained(tmp_path)
    own, supplied = tmp_path / "own.json", tmp_path / "supplied.json"
    assert _ftgs("eval", checkpoint, scene, "--out", own) == 0
    directory = _renders_dir(tmp_path, checkpoint, _test_items(scene))
    assert _ftgs("eval", checkpoint, scene, "--out", supplied, "--renders-dir", directory) == 0
    report = json.loads(supplied.read_text())
    assert _body(report) == _body(json.loads(own.read_text()))
    sha = hashlib.sha256((directory / "request.json").read_bytes()).hexdigest()
    assert report["renders"]["request_sha256"] == sha
    assert report["renders"]["producer"].startswith("openftgs")
    manifest_sha = hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
    assert report["model"]["sha256"] == manifest_sha  # Q-136: still MODEL's


def _broken_first_render(
    monkeypatch: pytest.MonkeyPatch,
    scene: Path,
    value: float = math.nan,
    part: Literal["rgb", "alpha"] = "rgb",
) -> list[str]:
    """Patch the evaluation's renderer so that its first render's `part` is all `value`; the
    list receives the manifest path of that render's image. Training renders through another
    function, so only evaluations see the patch."""
    import openftgs.render
    from openftgs.data import MultiViewVideo

    video = MultiViewVideo.from_manifest(scene)
    original = openftgs.render.render_request  # the object openftgs.eval imports
    paths: list[str] = []

    def broken(*args: Any, **kwargs: Any) -> Renders:
        renders = original(*args, **kwargs)
        if not paths:
            tensors = getattr(renders, part)
            key = next(iter(tensors))
            tensors[key] = torch.full_like(tensors[key], value)
            image = video.records[int(key)].image_path
            paths.append(image.relative_to(scene.parent).as_posix())
        return renders

    monkeypatch.setattr("openftgs.eval.render_request", broken)
    return paths


@pytest.mark.weights
@pytest.mark.parametrize(
    ("part", "value"),
    [("rgb", math.nan), ("rgb", math.inf), ("rgb", -math.inf), ("alpha", math.nan)],
    ids=["nan", "inf", "-inf", "alpha-nan"],
)
@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_eval_of_a_render_that_is_not_finite_is_exit_code_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
    part: Literal["rgb", "alpha"],
    value: float,
) -> None:
    """Q-129 (as answered), Q-147: a render with NaN or ±Inf colors, or a NaN alpha, makes
    ftgs eval exit 3, naming the image, and write no METRICS.json, leaving an existing one as
    it was (Q-126). An Inf render would score as a plausible image after E16's clamp, and the
    metrics never see alpha."""
    scene, checkpoint = _trained(tmp_path)
    out = tmp_path / "metrics" / "METRICS.json"
    before = None
    if existing:
        assert _ftgs("eval", checkpoint, scene, "--out", out) == 0
        before = out.read_bytes()
    else:
        out.parent.mkdir()
    paths = _broken_first_render(monkeypatch, scene, value, part)
    capsys.readouterr()
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 3
    assert paths and paths[0] in capsys.readouterr().err
    _assert_untouched(out, before)


@pytest.mark.weights
def test_a_render_that_is_not_finite_in_the_final_evaluation_is_exit_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-129 (as answered), Q-147: ftgs train's final evaluation of a render that isn't finite
    exits 3, naming the image, with no METRICS.json; the checkpoint, saved before it (Q-132),
    is kept."""
    scene = _scene(tmp_path / "scene", test=True)
    init = _init(tmp_path / "init.safetensors")
    cfg = _config_file(tmp_path / "cfg.json")
    out = tmp_path / "run"
    paths = _broken_first_render(monkeypatch, scene)
    capsys.readouterr()
    assert _ftgs("train", scene, "--init", init, "--out", out, "--config", cfg) == 3
    assert paths and paths[0] in capsys.readouterr().err
    assert not (out / "METRICS.json").exists()
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS


@pytest.mark.weights
def test_eval_of_a_perfect_render_is_exit_code_0_with_a_null_psnr(tmp_path: Path) -> None:
    """Q-129, Q-63: supplied renders equal to the ground-truth images have an infinite PSNR,
    which is no failure: exit 0, PSNR null in the rows and the scene means, SSIM 1, DSSIM 0 and
    every LPIPS 0 (a worked example: identical images)."""
    from openftgs.data import MultiViewVideo

    scene, checkpoint = _trained(tmp_path)
    video = MultiViewVideo.from_manifest(scene)
    directory = tmp_path / "perfect"
    request = write_render_request(directory / "request.json", _test_items(scene))
    rgb = {
        f"t{r.frame_index}": video.load_image(i)
        for i, r in enumerate(video.records)
        if r.split == "test"
    }
    alpha = {k: torch.ones(*v.shape[:2], 1) for k, v in rgb.items()}
    sha = hashlib.sha256(request.read_bytes()).hexdigest()
    write_renders(
        Renders(rgb, alpha),
        directory / "renders.safetensors",
        producer="openftgs tests",
        request_sha256=sha,
    )
    out = tmp_path / "m.json"
    assert _ftgs("eval", checkpoint, scene, "--out", out, "--renders-dir", directory) == 0
    entire = json.loads(out.read_text())["entire"]
    assert entire["scene"]["psnr"] is None
    assert all(row["psnr"] is None for row in entire["images"])
    scene_means = entire["scene"]
    assert scene_means["ssim1"] == pytest.approx(1.0) and scene_means["ssim2"] == pytest.approx(1.0)
    for name in ("dssim1", "dssim2", "lpips_alex", "lpips_vgg", "lpips_vgg_raw"):
        assert scene_means[name] == pytest.approx(0.0, abs=1e-6), name


def test_eval_refuses_a_supplied_renders_file_that_is_not_finite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-129, Q-142: a supplied renders file holding a NaN is invalid input, exit 2 as
    read_renders refuses it, not exit 3; nothing is written."""
    scene, checkpoint = _trained(tmp_path)
    directory = _renders_dir(tmp_path, checkpoint, _test_items(scene))
    path = directory / "renders.safetensors"
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    tensors["rgb/t9"][0, 0, 0] = math.nan
    save_file(tensors, path, metadata=metadata)
    out = tmp_path / "m.json"
    capsys.readouterr()
    assert _ftgs("eval", checkpoint, scene, "--out", out, "--renders-dir", directory) == 2
    assert "t9 has values that are not finite" in capsys.readouterr().err
    assert not out.exists()


def test_eval_refuses_supplied_renders_that_do_not_match_the_split(tmp_path: Path) -> None:
    """Q-137: a renders file made for another request, a request missing a test image, two items
    matching one image, a renders file missing a matched item or holding it at another size, and
    a missing DIR are exit 2, before any weights are needed."""
    scene, checkpoint = _trained(tmp_path)
    items = _test_items(scene)
    out = tmp_path / "m.json"
    stale = _renders_dir(tmp_path / "stale", checkpoint, items)
    write_render_request(
        stale / "request.json", items[::-1]
    )  # the renders were made for the first one
    missing = _renders_dir(tmp_path / "missing", checkpoint, items[:2])
    twice = _renders_dir(tmp_path / "twice", checkpoint, [*items, ("again", *items[0][1:])])
    # Files another implementation could write: the right request hash, a render missing or of
    # another size.
    lacking = _renders_dir(tmp_path / "lacking", checkpoint, items)
    small = _renders_dir(tmp_path / "small", checkpoint, items)
    for directory, edit in ((lacking, "drop"), (small, "shrink")):
        path = directory / "renders.safetensors"
        sha = hashlib.sha256((directory / "request.json").read_bytes()).hexdigest()
        renders = read_renders(path, request_sha256=sha)
        if edit == "drop":
            del renders.rgb["t9"], renders.alpha["t9"]
        else:
            renders.rgb["t9"] = renders.rgb["t9"][:-1].contiguous()
            renders.alpha["t9"] = renders.alpha["t9"][:-1].contiguous()
        write_renders(renders, path, producer="another", request_sha256=sha)
    for directory in (stale, missing, twice, lacking, small, tmp_path / "absent"):
        assert _ftgs("eval", checkpoint, scene, "--out", out, "--renders-dir", directory) == 2
    assert not out.exists()


def test_unavailable_lpips_weights_are_exit_code_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-142: a download of the LPIPS weights that fails (an OSError) is an environment error,
    in ftgs eval and in ftgs train's final evaluation, which runs after the checkpoint is saved
    (Q-132)."""
    import openftgs.eval

    def offline(*args: Any, **kwargs: Any) -> None:
        raise OSError("no network")

    monkeypatch.setattr(openftgs.eval.Evaluator, "__init__", offline)
    scene, checkpoint = _trained(tmp_path)
    assert _ftgs("eval", checkpoint, scene, "--out", tmp_path / "m.json") == 2
    init = _init(tmp_path / "again.safetensors")
    cfg = _config_file(tmp_path / "cfg.json")
    out = tmp_path / "run"
    assert _ftgs("train", scene, "--init", init, "--out", out, "--config", cfg) == 2
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS


def test_eval_refuses_a_partial_mask_set_and_an_empty_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-135 (#83): masks on some test images but not all is exit 2, naming the missing one; a
    split with no image is exit 2; both before the evaluator is built."""
    scene, checkpoint = _trained(tmp_path, masks=True)
    data = json.loads(scene.read_text())
    tests = [e for e in data["images"] if e["split"] == "test"]
    del tests[1]["mask"]
    scene.write_text(json.dumps(data))
    out = tmp_path / "m.json"
    capsys.readouterr()
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 2
    assert tests[1]["path"] in capsys.readouterr().err
    data["images"] = [e for e in data["images"] if e["split"] == "train"]
    scene.write_text(json.dumps(data))
    assert _ftgs("eval", checkpoint, scene, "--split", "test", "--out", out) == 2
    assert not out.exists()


# ftgs cache (D-51; Q-127).


def test_cache_builds_the_frame_cache(run: dict[str, Path], tmp_path: Path) -> None:
    """D-51, Q-98, Q-127: the cache beside the manifest; mmap opens it; gpu without CUDA is
    exit 2; a second manifest in the same folder keeps the first one's cache."""
    from openftgs.data import MultiViewVideo

    scene = run["scene"]
    assert _ftgs("cache", scene) == 0
    first = FrameCache.default_directory(MultiViewVideo.from_manifest(scene).manifest)
    assert sorted(first.glob("*.safetensors"))
    assert _ftgs("cache", scene, "--placement", "mmap") == 0
    assert _ftgs("cache", scene, "--placement", "gpu") == 2
    data = json.loads(scene.read_text())
    data["metadata"]["note"] = "a second manifest"
    other = scene.parent / "other.json"
    other.write_text(json.dumps(data))
    assert _ftgs("cache", other) == 0
    second = FrameCache.default_directory(MultiViewVideo.from_manifest(other).manifest)
    assert first != second and first.is_dir() and second.is_dir()
    assert _ftgs("cache", tmp_path / "none.json") == 2


def test_a_cache_that_does_not_fit_the_gpu_is_exit_code_2(
    run: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-142: out of GPU memory while opening --placement gpu is an environment error."""

    def full(*args: Any, **kwargs: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(FrameCache, "open", full)
    assert _ftgs("cache", run["scene"], "--placement", "gpu") == 2


@pytest.mark.gpu
def test_cache_opens_on_the_gpu(run: dict[str, Path]) -> None:
    """Q-127: --placement gpu opens the cache in GPU memory when it fits."""
    assert _ftgs("cache", run["scene"], "--placement", "gpu") == 0


# ftgs convert (§1.4 "Dataset protocols"; U-27's exit code; D-80, Q-88).


@needs_ffmpeg
def test_convert_n3dv(tmp_path: Path) -> None:
    """§2.1: ftgs convert n3dv writes OUT/scene.json; a non-empty OUT is exit 2 (Q-88)."""
    src = n3dv_source(tmp_path / "src", cameras=2, frames=4)
    out = tmp_path / "out"
    args = ("convert", "n3dv", src, out, "--frames", 4, "--resize", 1.0, "--jobs", 1)
    assert _ftgs(*args) == 0
    from openftgs.io import read_scene_manifest

    manifest = read_scene_manifest(out / "scene.json")
    assert len(manifest.records) == 8
    assert _ftgs(*args) == 2


@needs_ffmpeg
def test_convert_selfcap_with_a_stub_colmap(tmp_path: Path) -> None:
    """§2.1, D-80: ftgs convert selfcap on a generated mini-archive (U-37's), with --window,
    --test-camera and a stub COLMAP: frames 4 ≤ n < 8 of three cameras, the test camera's
    images in the test split, and COLMAP called."""
    src = selfcap_archives(tmp_path, "mini")
    colmap, log = stub_colmap(tmp_path)
    out = tmp_path / "out"
    args = (
        "convert",
        "selfcap",
        src,
        out,
        "--scene",
        "mini",
        "--window",
        "4:8",
        "--test-camera",
        "0002",
        "--colmap",
        colmap,
        "--jobs",
        1,
    )
    assert _ftgs(*args) == 0
    from openftgs.io import read_scene_manifest

    manifest = read_scene_manifest(out / "scene.json")
    assert {r.frame_index for r in manifest.records} == {4, 5, 6, 7}
    assert {r.camera.name for r in manifest.records if r.split == "test"} == {"0002"}
    assert (log / "calls.jsonl").is_file()


@needs_ffmpeg
def test_u27_a_wrong_ffmpeg_or_ffprobe_path_is_exit_code_2(tmp_path: Path) -> None:
    """U-27, D-48: a wrong --ffmpeg or --ffprobe path stops the converter with exit code 2."""
    src = n3dv_source(tmp_path / "src", cameras=1, frames=2)
    for flag in ("--ffmpeg", "--ffprobe"):
        out = tmp_path / flag.strip("-")
        args = ("convert", "n3dv", src, out, "--frames", 2, flag, tmp_path / "no-such-program")
        assert _ftgs(*args) == 2
        assert not out.exists() or not any(out.iterdir())


def test_convert_maps_options_to_the_converter_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2.1 CLI line, D-80: each option reaches its keyword, --window N0:N1 as (N0, N1) and
    --no-sync as use_sync=False; an option of another kind, or a missing required one, is
    exit 2 before the converter runs."""
    import openftgs.data

    calls: dict[str, dict[str, Any]] = {}

    def recorder(name: str) -> Any:
        def convert(src: Path, dst: Path, **kwargs: Any) -> Path:
            calls[name] = kwargs
            return dst / "scene.json"

        return convert

    for name in ("convert_n3dv", "convert_opencv_yaml", "convert_selfcap"):
        monkeypatch.setattr(openftgs.data, name, recorder(name))
    src, out = tmp_path / "src", tmp_path / "out"
    selfcap = (
        "convert",
        "selfcap",
        src,
        out,
        "--scene",
        "yoga",
        "--window",
        "4:8",
        "--test-camera",
        "0002",
        "--undistort",
        "internal",
        "--no-sync",
        "--masks",
        tmp_path / "m",
        "--calib",
        tmp_path / "c",
        "--resize",
        0.25,
        "--jobs",
        2,
    )
    assert _ftgs(*selfcap) == 0
    assert calls["convert_selfcap"] == {
        "scene": "yoga",
        "window": (4, 8),
        "test_camera": "0002",
        "undistort": "internal",
        "use_sync": False,
        "masks": tmp_path / "m",
        "calib": tmp_path / "c",
        "resize": 0.25,
        "jobs": 2,
    }
    yaml = (
        "convert",
        "opencv-yaml",
        src,
        out,
        "--fps",
        60,
        "--frames",
        5,
        "--resize",
        0.5,
        "--test-camera",
        "c03",
        "--hwaccel",
        "cuda",
        "--ffprobe",
        "fp",
    )
    assert _ftgs(*yaml) == 0
    assert calls["convert_opencv_yaml"] == {
        "fps": 60.0,
        "frames": 5,
        "resize": 0.5,
        "test_camera": "c03",
        "hwaccel": "cuda",
        "ffprobe": "fp",
    }
    calls.clear()
    for wrong in (
        ("convert", "n3dv", src, out, "--scene", "x"),
        ("convert", "n3dv", src, out, "--no-sync"),
        ("convert", "selfcap", src, out, "--scene", "x", "--fps", 30),
        ("convert", "opencv-yaml", src, out, "--fps", 60, "--frames", 5),
        ("convert", "selfcap", src, out),
        ("convert", "selfcap", src, out, "--scene", "x", "--window", "4-8"),
    ):
        assert _ftgs(*wrong) == 2
    assert not calls


# ftgs init (Q-107, Q-128, Q-142).


def test_init_needs_cuda_and_the_init_extra(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-125, Q-142: roma_init without a CUDA device is exit 2, as is a missing [init] extra and
    an invalid setting."""
    import openftgs.init

    out = tmp_path / "roma.safetensors"
    assert _ftgs("init", "roma", run["scene"], out) == 2
    assert _ftgs("init", "roma", run["scene"], out, "--stride", 0) == 2

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("No module named 'einops'")

    monkeypatch.setattr(openftgs.init, "roma_init", missing)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # past the device check
    assert _ftgs("init", "roma", run["scene"], out) == 2
    assert not out.exists()


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not has_init_extra, reason="needs the [init] extra")
def test_init_roma_writes_init_points_with_provenance(tmp_path: Path) -> None:
    """Q-107: ftgs init roma on a textured plane writes valid init points with roma_init's
    provenance metadata."""
    plane = TexturedPlane(seed=3, velocity=(0.6, 0.3, 0.0))
    images = []
    for tau in (0.0, 1 / 30):
        for k, yaw in enumerate((-15.0, -5.0, 5.0, 15.0)):
            theta = math.radians(yaw)
            eye = (3.0 * math.sin(theta), -3.0 * math.cos(theta), 0.3)
            cam = look_at(eye, width=320, height=256, focal=280.0, name=f"C{k}")
            images.append((cam, tau, "train"))
    scene = write_plane_scene(tmp_path / "plane", plane, images)
    out = tmp_path / "init.safetensors"
    assert _ftgs("init", "roma", scene, out, "--points-per-frame", 200, "--seed", 3) == 0
    points = read_init_points(out, time_unit="s")
    assert points.positions.shape[0] == 2 * 200
    with safe_open(str(out), framework="pt") as f:
        provenance = json.loads(f.metadata()["roma_init"])
    assert provenance["points_per_frame"] == 200 and provenance["seed"] == 3


# ftgs bench (Q-130).


def test_bench_is_a_stub_until_m15(run: dict[str, Path], tmp_path: Path) -> None:
    """Q-130: it checks its arguments and the manifest, writes nothing and returns 1."""
    out = tmp_path / "bench.json"
    assert _ftgs("bench", "train-step", run["scene"], "--out", out) == 1
    assert not out.exists()
    assert _ftgs("bench", "warp", run["scene"], "--out", out) == 2
    assert _ftgs("bench", "raster", tmp_path / "none.json", "--out", out) == 2


# Single-file outputs are replaced atomically (Q-126 as answered; Q-25's temporary naming; #179).


def _fail_partway(monkeypatch: pytest.MonkeyPatch, out: Path) -> None:
    """Make every safetensors write into `out`'s directory stop partway: it writes a few bytes
    of its file, then raises OSError, as a full disk would."""

    def partial(tensors: Any, filename: Any, metadata: Any = None) -> None:
        if Path(filename).parent != out.parent:
            save_file(tensors, filename, metadata=metadata)
            return
        Path(filename).write_bytes(b"\x08\x00")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("openftgs.io.save_file", partial)


def _assert_untouched(out: Path, before: bytes | None) -> None:
    """Q-126: the target is as it was (bitwise, or still absent) and no temporary is left."""
    if before is None:
        assert not out.exists()
    else:
        assert out.read_bytes() == before
    assert sorted(p.name for p in out.parent.iterdir() if p.name != out.name) == []


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_export_that_fails_partway_leaves_the_target_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    """Q-126: an exchange model written partway leaves the old file, or no file, and no
    temporary; the failure is exit code 2 naming the output (Q-142 as answered)."""
    _, checkpoint = _trained(tmp_path)
    out = tmp_path / "exports" / "model.safetensors"
    before = None
    if existing:
        assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", out) == 0
        before = out.read_bytes()
    else:
        out.parent.mkdir()
    _fail_partway(monkeypatch, out)
    capsys.readouterr()
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", out) == 2
    assert str(out) in capsys.readouterr().err
    _assert_untouched(out, before)


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_render_that_fails_partway_leaves_the_target_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    """Q-126: renders written partway leave the old file, or no file, and no temporary; the
    failure is exit code 2 naming the output (Q-142 as answered)."""
    scene, checkpoint = _trained(tmp_path)
    request = write_render_request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "renders" / "out.safetensors"
    args = ("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
    before = None
    if existing:
        assert _ftgs(*args) == 0
        before = out.read_bytes()
    else:
        out.parent.mkdir()
    _fail_partway(monkeypatch, out)
    capsys.readouterr()
    assert _ftgs(*args) == 2
    assert str(out) in capsys.readouterr().err
    _assert_untouched(out, before)


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_init_roma_that_fails_partway_leaves_the_target_as_it_was(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    """Q-126: init points written partway leave the old file, or no file, and no temporary;
    the failure is exit code 2 naming the output (Q-142 as answered). roma_init is stood in
    for, so no CUDA device or [init] extra is needed."""
    import openftgs.init

    points = read_init_points(run["init"])
    monkeypatch.setattr(openftgs.init, "roma_init", lambda video, cfg: points)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # past the device check
    out = tmp_path / "points" / "roma.safetensors"
    before = None
    if existing:
        assert _ftgs("init", "roma", run["scene"], out) == 0
        before = out.read_bytes()
    else:
        out.parent.mkdir()
    _fail_partway(monkeypatch, out)
    capsys.readouterr()
    assert _ftgs("init", "roma", run["scene"], out) == 2
    assert str(out) in capsys.readouterr().err
    _assert_untouched(out, before)


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_eval_that_fails_partway_leaves_the_target_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: bool,
) -> None:
    """Q-126: METRICS.json written partway leaves the old file, or no file, and no temporary;
    the failure is exit code 2 naming the output (Q-142 as answered). Text writes into its
    directory write a few characters, then raise OSError, as a full disk would."""
    scene, checkpoint = _trained(tmp_path)
    out = tmp_path / "metrics" / "METRICS.json"
    before = None
    if existing:
        assert _ftgs("eval", checkpoint, scene, "--out", out) == 0
        before = out.read_bytes()
    else:
        out.parent.mkdir()
    original = Path.write_text

    def partial(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.parent != out.parent:
            return original(self, data, *args, **kwargs)
        original(self, data[:5], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", partial)
    capsys.readouterr()
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 2
    assert str(out) in capsys.readouterr().err
    _assert_untouched(out, before)


def test_export_replaces_a_symlinked_target_by_a_regular_file(tmp_path: Path) -> None:
    """Q-146: the atomic replace (Q-126) swaps the directory entry the path names, so a symlink
    there becomes a regular file and the file it pointed to keeps its content."""
    _, checkpoint = _trained(tmp_path)
    elsewhere = tmp_path / "elsewhere.safetensors"
    elsewhere.write_bytes(b"not overwritten")
    out = tmp_path / "exports" / "model.safetensors"
    out.parent.mkdir()
    out.symlink_to(elsewhere)
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", out) == 0
    assert not out.is_symlink()
    read_exchange(out)
    assert elsewhere.read_bytes() == b"not overwritten"


# Out of GPU memory in training and unwritable outputs (Q-142 as answered; Q-158 to Q-161).


def _cut_at_ten(run: dict[str, Path], out: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A run directory whose run SIGTERM stopped at iteration 10, its checkpoint saved."""
    with monkeypatch.context() as m:
        _sigterm_at(m, 10)
        assert _train(run, out, "--config", run["config"]) == 128 + signal.SIGTERM
    return out


def _blocked_profile(cut: Path, out: Path) -> Path:
    """A run directory of `cut`'s run (its config.json) where a file stands in the way of the
    profile trace's folder."""
    out.mkdir()
    shutil.copy(cut / "config.json", out / "config.json")
    (out / "profile").write_text("a file where the trace's folder goes")
    return out


def _out_of_memory_at(monkeypatch: pytest.MonkeyPatch, iteration: int) -> None:
    """Trainer.step raises torch.OutOfMemoryError in iteration `iteration`, as CUDA would."""
    step = Trainer.step

    def stepping(self: Trainer) -> Any:
        if self.iteration + 1 == iteration:
            raise torch.OutOfMemoryError("CUDA out of memory (stood in for)")
        return step(self)

    monkeypatch.setattr(Trainer, "step", stepping)


@pytest.mark.parametrize("every", [5, 0], ids=["periodic", "none"])
def test_out_of_gpu_memory_in_training_is_exit_code_2(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    every: int,
) -> None:
    """Q-142 (as answered): out of GPU memory inside Trainer.train is exit code 2; nothing is
    saved for the iteration cut off, and the message names it, the latest checkpoint when there
    is one, and the frame-cache remedy (D-51, Q-99, Q-134)."""
    _out_of_memory_at(monkeypatch, 7)
    out = tmp_path / "run"
    capsys.readouterr()
    assert _train(run, out, "--config", run["config"], "--checkpoint-every", every) == 2
    err = capsys.readouterr().err
    assert "after iteration 6" in err and "frame_cache" in err
    if every:
        assert str(out / "checkpoint") in err
        assert _kept(out) == {"checkpoint": 5}
    else:
        assert "no checkpoint was saved" in err
        assert not (out / "checkpoint").exists()


def test_out_of_gpu_memory_in_a_resumed_run_names_its_checkpoint(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-142 (as answered): a resumed run that runs out of GPU memory before its first save
    names the checkpoint it was resumed from."""
    cut = _cut_at_ten(run, tmp_path / "cut", monkeypatch)
    _out_of_memory_at(monkeypatch, 13)
    capsys.readouterr()
    assert _train(run, tmp_path / "resumed", "--resume", cut / "checkpoint") == 2
    err = capsys.readouterr().err
    assert str(cut / "checkpoint") in err and "after iteration 12" in err
    assert not (tmp_path / "resumed" / "checkpoint").exists()


def test_out_of_memory_or_a_read_error_elsewhere_keeps_its_traceback(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-26, Q-142 (as answered): only Trainer.train's out-of-memory error and the outputs'
    write errors are exit code 2; out of memory while the trainer is built, or an OSError of a
    read during training, stays a traceback."""

    def no_memory(*args: Any, **kwargs: Any) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory (stood in for)")

    with monkeypatch.context() as m:
        m.setattr(Trainer, "__init__", no_memory)
        with pytest.raises(torch.OutOfMemoryError):
            _train(run, tmp_path / "a", "--config", run["config"])
    cut = _cut_at_ten(run, tmp_path / "cut", monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(Trainer, "from_checkpoint", no_memory)
        with pytest.raises(torch.OutOfMemoryError):
            _train(run, tmp_path / "resumed", "--resume", cut / "checkpoint")
    step = Trainer.step

    def unreadable(self: Trainer) -> Any:
        if self.iteration == 3:
            raise OSError(5, "Input/output error reading a frame")
        return step(self)

    monkeypatch.setattr(Trainer, "step", unreadable)
    with pytest.raises(OSError, match="reading a frame"):
        _train(run, tmp_path / "b", "--config", run["config"])


def test_a_checkpoint_that_cannot_be_written_is_exit_code_2(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-142 (as answered), Q-159: a checkpoint save that fails partway, a periodic one here, is
    exit code 2 naming the run directory; no further save is attempted, the checkpoint already
    kept is as it was, and no temporary is left (Q-25, Q-151)."""
    import safetensors.torch

    import openftgs.io

    save_file = safetensors.torch.save_file  # the function openftgs.io writes with
    save_checkpoint = openftgs.io.save_checkpoint
    full = {"now": False}

    def writing(*args: Any, **kwargs: Any) -> None:
        if full["now"]:
            raise OSError(28, "No space left on device")
        save_file(*args, **kwargs)

    def saving(state: Any, directory: Path) -> Path:
        full["now"] = state.iteration == 10
        return save_checkpoint(state, directory)

    monkeypatch.setattr("openftgs.io.save_file", writing)
    monkeypatch.setattr("openftgs.io.save_checkpoint", saving)
    out = tmp_path / "run"
    keep = ("--checkpoint-every", 5, "--keep-checkpoints", 2)
    capsys.readouterr()
    assert _train(run, out, "--config", run["config"], *keep) == 2
    assert str(out) in capsys.readouterr().err
    full["now"] = False
    assert _kept(out) == {"checkpoint": 5}
    assert not [p.name for p in out.iterdir() if p.name.startswith(".")]


def test_a_train_log_that_cannot_be_written_is_exit_code_2(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-142 (as answered), Q-159: an OSError while appending to train_log.jsonl is exit code 2
    naming it, at once: no checkpoint is saved after it."""
    original = Path.open

    def full(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "train_log.jsonl":
            raise OSError(28, "No space left on device")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", full)
    out = tmp_path / "run"
    capsys.readouterr()
    assert _train(run, out, "--config", run["config"]) == 2
    assert "train_log.jsonl" in capsys.readouterr().err
    assert not (out / "checkpoint").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_outputs_into_a_read_only_directory_are_exit_code_2(
    run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-142 (as answered), Q-158: the real failure of a write into a directory without write
    permission, the safetensors writer's included, is exit code 2 naming the output, for
    ftgs export, ftgs render and ftgs train's run directory."""
    scene, checkpoint = _trained(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        request = write_render_request(tmp_path / "req.json", _test_items(scene))
        commands = [
            ("export", checkpoint, "--format", "ftgs-if", "--out", locked / "m.safetensors"),
            (
                "render",
                checkpoint,
                "--request",
                request,
                "--out",
                locked / "r.safetensors",
                "--backend",
                "reference",
            ),
        ]
        for command in commands:
            capsys.readouterr()
            assert _ftgs(*command) == 2
            assert str(locked) in capsys.readouterr().err
        capsys.readouterr()
        assert _train(run, locked / "run", "--config", run["config"]) == 2
        assert str(locked / "run") in capsys.readouterr().err
        assert list(locked.iterdir()) == []
    finally:
        locked.chmod(0o755)


def test_a_trace_that_cannot_be_written_is_exit_code_2_after_the_save(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-142 (as answered): the profile trace is an output, so a trace that can't be written is
    exit code 2 naming it; the run goes on to T_it and saves its checkpoint first, so no training
    is lost. A file where the trace's folder goes makes the write fail at the window's end."""
    cut = _cut_at_ten(run, tmp_path / "cut", monkeypatch)
    out = _blocked_profile(cut, tmp_path / "resumed")
    capsys.readouterr()
    assert _train(run, out, "--resume", cut / "checkpoint", "--profile", "12:2") == 2
    assert "profile" in capsys.readouterr().err
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS


def test_a_signal_with_a_trace_that_cannot_be_written_keeps_its_exit_code(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-133, Q-142 (as answered): a signal still saves and exits 128 + its number when the
    trace can't be written; the trace's failure is a warning."""
    cut = _cut_at_ten(run, tmp_path / "cut", monkeypatch)
    out = _blocked_profile(cut, tmp_path / "resumed")
    _sigterm_at(monkeypatch, 16)
    capsys.readouterr()
    args = ("--resume", cut / "checkpoint", "--profile", "12:2")
    assert _train(run, out, *args) == 128 + signal.SIGTERM
    assert "warning" in capsys.readouterr().err
    assert load_checkpoint(out / "checkpoint").iteration == 16


def test_out_of_memory_with_an_unwritable_trace_reports_the_out_of_memory(
    run: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-160: when training runs out of memory inside a profile window and the trace can't be
    written either, the out-of-memory error is what the command reports (exit 2), the trace's
    failure a warning."""
    cut = _cut_at_ten(run, tmp_path / "cut", monkeypatch)
    out = _blocked_profile(cut, tmp_path / "resumed")
    _out_of_memory_at(monkeypatch, 13)
    capsys.readouterr()
    assert _train(run, out, "--resume", cut / "checkpoint", "--profile", "12:5") == 2
    err = capsys.readouterr().err
    assert "out of GPU memory" in err and "warning" in err


def test_out_of_memory_in_a_profile_window_writes_the_trace(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-160, Q-144: an out-of-memory error inside a profile window still writes the trace of the
    iterations the window ran, and the command reports the out-of-memory error (exit 2)."""
    _out_of_memory_at(monkeypatch, 6)
    out = tmp_path / "run"
    assert _train(run, out, "--config", run["config"], "--profile", "3:10") == 2
    assert json.loads((out / "profile" / "trace.json").read_text())["traceEvents"]


@pytest.mark.weights
def test_metrics_that_cannot_be_written_in_training_are_exit_code_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-142 (as answered), Q-161: a METRICS.json write that fails in ftgs train's final
    evaluation is exit code 2 naming METRICS.json, after the checkpoint is saved (Q-132)."""

    def full(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("openftgs.train.write_metrics", full)
    scene = _scene(tmp_path / "scene", test=True)
    init = _init(tmp_path / "init.safetensors")
    cfg = _config_file(tmp_path / "cfg.json")
    out = tmp_path / "run"
    capsys.readouterr()
    assert _ftgs("train", scene, "--init", init, "--out", out, "--config", cfg) == 2
    err = capsys.readouterr().err
    assert "METRICS.json" in err and "No space left" in err
    assert load_checkpoint(out / "checkpoint").iteration == ITERATIONS
