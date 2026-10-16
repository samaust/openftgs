# SPDX-License-Identifier: Apache-2.0
"""The ftgs command line (§2.1 CLI block and exit codes, §2.2; Q-124 to Q-142): a smoke test of
every command on a small synthetic scene, the CLI round trips of U-26, U-22's rejections with
exit code 2, U-27's exit code, and the train, resume, render, eval and export behaviour.

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
from typing import Any

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
    shortened_config,
    write_plane_scene,
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
)
from openftgs.model import FreeTimeGaussians
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


def _camera(camera: Any) -> dict[str, Any]:
    return {
        "width": camera.width,
        "height": camera.height,
        "K": camera.K.tolist(),
        "world_to_camera": camera.world_to_camera.tolist(),
    }


def _request(path: Path, items: list[tuple[str, float, Any]], *, unit: str = "s") -> Path:
    """An ftgs-render-request v1 (§2.2), written by hand."""
    body = {
        "format": "ftgs-render-request",
        "version": 1,
        "time_unit": unit,
        "background": [0, 0, 0],
        "sh_degree": None,
        "items": [{"id": i, "timestamp": tau, "camera": _camera(cam)} for i, tau, cam in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


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
    request = _request(tmp_path / "req.json", _test_items(scene))
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
    """Q-125: gsplat (the default) without CUDA is exit 2; so are a request in another time unit
    and a request that fails §2.2's schema."""
    scene, checkpoint = _trained(tmp_path)
    request = _request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "r.safetensors"
    assert _ftgs("render", checkpoint, "--request", request, "--out", out) == 2
    frames = _request(tmp_path / "frames.json", _test_items(scene), unit="frame")
    ref = ("--backend", "reference")
    assert _ftgs("render", checkpoint, "--request", frames, "--out", out, *ref) == 2
    broken = tmp_path / "broken.json"
    broken.write_text('{"format": "ftgs-render-request", "version": 1}')
    assert _ftgs("render", checkpoint, "--request", broken, "--out", out, *ref) == 2
    assert _ftgs("render", checkpoint, "--request", tmp_path / "none.json", "--out", out, *ref) == 2
    assert not out.exists()


def test_renders_that_are_not_finite_are_exit_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-129: a non-finite render is a numerical failure, and nothing is written."""
    import openftgs.render

    scene, checkpoint = _trained(tmp_path)
    request = _request(tmp_path / "req.json", _test_items(scene))
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
    request = _request(tmp_path / "req.json", _test_items(scene))
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
        torch.testing.assert_close(a.rgb[key], b.rgb[key], rtol=0.0, atol=2e-3)


# ftgs export (§2.1; Q-141) and U-22's rejections with exit code 2.


def test_export_writes_a_valid_exchange_model_with_the_runs_background(tmp_path: Path) -> None:
    """Q-141 (Q-24): the file records the checkpoint's TrainConfig.background in
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
    assert opacity[0] == torch.tensor(1.0 - 1e-7, dtype=torch.float32)


@pytest.mark.parametrize("case", BAD_EXCHANGE_CASES)
def test_u22_invalid_exchange_models_are_exit_code_2(tmp_path: Path, case: str) -> None:
    """U-22: every file the validator rejects is exit code 2 as ftgs render's and ftgs eval's
    MODEL; eval refuses before building its evaluator, so no weights are needed."""
    scene = _scene(tmp_path / "scene", test=True)
    arrays, meta = bad_exchange(case)
    bad = tmp_path / "bad.safetensors"
    save_file(arrays, str(bad), metadata=meta)
    request = _request(tmp_path / "req.json", _test_items(scene))
    out = tmp_path / "out.safetensors"
    assert _ftgs("render", bad, "--request", request, "--out", out, "--backend", "reference") == 2
    assert _ftgs("eval", bad, scene, "--out", tmp_path / "m.json") == 2
    assert not out.exists() and not (tmp_path / "m.json").exists()


def test_u22_a_corrupted_checkpoint_is_exit_code_2(tmp_path: Path) -> None:
    """§2.2, Q-25: a checkpoint whose file no longer matches its SHA-256 is exit code 2."""
    scene, checkpoint = _trained(tmp_path)
    params = checkpoint / "params.safetensors"
    data = bytearray(params.read_bytes())
    data[-1] ^= 0xFF
    params.write_bytes(bytes(data))
    request = _request(tmp_path / "req.json", _test_items(scene))
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
    request = _request(tmp_path / "req.json", _test_items(scene))
    ref = ("--backend", "reference")
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    assert _ftgs("render", checkpoint, "--request", request, "--out", a, *ref) == 0
    assert _ftgs("render", exchange, "--request", request, "--out", b, *ref) == 0
    ra, rb = read_renders(a), read_renders(b)
    for key in ra.rgb:
        torch.testing.assert_close(ra.rgb[key], rb.rgb[key], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(ra.alpha[key], rb.alpha[key], rtol=1e-6, atol=1e-6)


# ftgs eval (§2.1 CLI paragraph on METRICS.json; Q-135 to Q-137, Q-141).


def _body(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k not in ("producer", "model", "renders")}


def _renders_dir(tmp_path: Path, checkpoint: Path, items: list[tuple[str, float, Any]]) -> Path:
    """DIR of Q-137: request.json and its renders by the reference backend."""
    directory = tmp_path / "supplied"
    request = _request(directory / "request.json", items)
    out = directory / "renders.safetensors"
    assert (
        _ftgs("render", checkpoint, "--request", request, "--out", out, "--backend", "reference")
        == 0
    )
    return directory


@pytest.mark.weights
def test_eval_writes_metrics_for_a_checkpoint_and_an_exchange_model(tmp_path: Path) -> None:
    """§2.1, Q-63, Q-136: METRICS.json with every metric under `entire`, `dynamic` for a masked
    split (with no image missing a mask, Q-135), and the header: `model` = {path, iteration},
    the iteration null for an exchange model, and the scene manifest."""
    scene, checkpoint = _trained(tmp_path, masks=True)
    out = tmp_path / "m.json"
    assert _ftgs("eval", checkpoint, scene, "--split", "test", "--out", out) == 0
    report = json.loads(out.read_text())
    assert report["format"] == "ftgs-metrics" and report["split"] == "test"
    assert report["model"] == {"path": str(checkpoint), "iteration": 10}
    assert report["manifest"]["sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert set(report["entire"]["scene"]) == set(METRIC_NAMES)
    assert len(report["entire"]["images"]) == 3
    # The masks' 24-px halves are below Q-61's smallest region, so each image is skipped.
    assert report["dynamic"]["missing"] == 0 and len(report["dynamic"]["images"]) == 3
    exchange = tmp_path / "m.safetensors"
    assert _ftgs("export", checkpoint, "--format", "ftgs-if", "--out", exchange) == 0
    assert _ftgs("eval", exchange, scene, "--out", out) == 0
    assert json.loads(out.read_text())["model"] == {"path": str(exchange), "iteration": None}


@pytest.mark.weights
def test_eval_of_a_checkpoint_matches_the_trainers_evaluation(tmp_path: Path) -> None:
    """Q-141, Q-74: ftgs eval of a checkpoint renders against its configuration's background,
    so a white-background run's report body equals Trainer.evaluate's."""
    from openftgs.data import MultiViewVideo

    scene, checkpoint = _trained(tmp_path, background=(1.0, 1.0, 1.0))
    out = tmp_path / "cli.json"
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 0
    trainer = Trainer.from_checkpoint(checkpoint, MultiViewVideo.from_manifest(scene), device="cpu")
    trainer.evaluate(out_dir=tmp_path / "trainer")
    want = json.loads((tmp_path / "trainer" / "METRICS.json").read_text())
    assert _body(json.loads(out.read_text())) == _body(want)


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


def test_eval_refuses_supplied_renders_that_do_not_match_the_split(tmp_path: Path) -> None:
    """Q-137: a renders file made for another request, a request missing a test image, and two
    items matching one image are exit 2, before any weights are needed."""
    scene, checkpoint = _trained(tmp_path)
    items = _test_items(scene)
    out = tmp_path / "m.json"
    stale = _renders_dir(tmp_path / "stale", checkpoint, items)
    _request(stale / "request.json", items[::-1])  # the renders were made for the first one
    missing = _renders_dir(tmp_path / "missing", checkpoint, items[:2])
    twice = _renders_dir(tmp_path / "twice", checkpoint, [*items, ("again", *items[0][1:])])
    for directory in (stale, missing, twice, tmp_path / "absent"):
        assert _ftgs("eval", checkpoint, scene, "--out", out, "--renders-dir", directory) == 2
    assert not out.exists()


def test_eval_refuses_a_partial_mask_set_and_an_empty_split(tmp_path: Path) -> None:
    """Q-135 (#83): masks on some test images but not all is exit 2, naming the missing one; a
    split with no image is exit 2; both before the evaluator is built."""
    scene, checkpoint = _trained(tmp_path, masks=True)
    data = json.loads(scene.read_text())
    tests = [e for e in data["images"] if e["split"] == "test"]
    del tests[1]["mask"]
    scene.write_text(json.dumps(data))
    out = tmp_path / "m.json"
    assert _ftgs("eval", checkpoint, scene, "--out", out) == 2
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


@pytest.mark.gpu
def test_cache_opens_on_the_gpu(run: dict[str, Path]) -> None:
    assert _ftgs("cache", run["scene"], "--placement", "gpu") == 0


# ftgs convert (§1.4 "Dataset protocols"; U-27's exit code; Q-85, Q-88).


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
    """§2.1 CLI line, Q-85: each option reaches its keyword, --window N0:N1 as (N0, N1) and
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
