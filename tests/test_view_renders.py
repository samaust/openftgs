# SPDX-License-Identifier: Apache-2.0
"""`scripts/view_renders.py`, the test-view renders of a trained model through the public CLI
(§2.2 "Rendering at any camera and time", E16, D-84; Q-56, Q-137; #327).

Added tests, not §3 rows: the request it writes is matched record for record by `ftgs eval
--renders-dir` (Q-137's rule); its background is the model's training background (D-84); the
whole run leaves one 8-bit PNG per test record equal to the clamped render, a METRICS.json equal
to the run's own, and no renders file. The script runs as #327 runs it, in a process of its own,
here on the CPU (the reference backend).
"""

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from builders import random_model, ring, shortened_config, write_scene
from PIL import Image

from openftgs.cli import main
from openftgs.config import (
    LossConfig,
    PerfConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
)
from openftgs.io import (
    load_checkpoint,
    read_render_request,
    read_scene_manifest,
    write_exchange,
    write_init_points,
)
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import InitPoints, TimeNormalizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view_renders.py"
TAU1 = 3.0 / 30.0  # four frames at 30 FPS
TIME = TimeNormalizer(0.0, TAU1)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-125: the CLI picks CUDA when PyTorch sees it; these runs use the reference backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _script(*args: Path | str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # the script and its ftgs runs on the CPU
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=env,
    )


def _scene(root: Path) -> Path:
    """Three cameras over four frames: cameras 0 and 1 train, camera 2 is the test view."""
    cams = ring(3, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    gt = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images = [
        (c, k / 30.0, "test" if i == 2 else "train") for k in range(4) for i, c in enumerate(cams)
    ]
    return write_scene(root, gt, images)


def test_request_is_matched_record_for_record_by_eval(tmp_path: Path) -> None:
    """Q-56, Q-137, D-84: one item per test record, with its exact timestamp, size, K and pose;
    the model's training background and full SH degree; renders answering it pass `ftgs eval
    --renders-dir`, which matches every record of the split to exactly one item."""
    scene = _scene(tmp_path / "scene")
    model = random_model(12, seed=3, sh_degree=1, time=TIME, spread=0.6)
    exchange = dataclasses.replace(model.to_exchange(), background=(0.25, 0.5, 0.75))
    model_path = tmp_path / "model.safetensors"
    write_exchange(exchange, model_path)
    out = tmp_path / "renders"
    done = _script("request", model_path, scene, "--out", out / "request.json")
    assert done.returncode == 0, done.stderr
    request = read_render_request(out / "request.json")
    tests = [r for r in read_scene_manifest(scene).records if r.split == "test"]
    assert len(request.items) == len(tests) == 4
    for item, record in zip(request.items, tests, strict=True):
        assert item.timestamp == record.timestamp
        assert item.camera.width == record.camera.width
        assert item.camera.height == record.camera.height
        assert torch.equal(item.camera.K, record.camera.K)
        assert torch.equal(item.camera.world_to_camera, record.camera.world_to_camera)
    assert request.background == (0.25, 0.5, 0.75)
    assert request.sh_degree is None and request.time_unit == "s"
    render_argv = ["render", model_path, "--request", out / "request.json",
                   "--out", out / "renders.safetensors", "--backend", "reference"]  # fmt: skip
    assert main([str(a) for a in render_argv]) == 0
    eval_argv = ["eval", model_path, scene, "--renders-dir", out, "--out", out / "METRICS.json"]
    assert main([str(a) for a in eval_argv]) == 0
    assert json.loads((out / "METRICS.json").read_text())["renders"]["dir"] == str(out)


def test_request_for_an_empty_split_is_exit_2(tmp_path: Path) -> None:
    """A split without records is exit 2, with nothing written."""
    cams = ring(2, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    gt = random_model(8, seed=0, sh_degree=1, time=TIME)
    scene = write_scene(tmp_path / "scene", gt, [(c, 0.0, "train") for c in cams])
    model_path = tmp_path / "model.safetensors"
    write_exchange(gt.to_exchange(), model_path)
    out = tmp_path / "renders" / "request.json"
    done = _script("request", model_path, scene, "--out", out)
    assert done.returncode == 2 and "test split" in done.stderr
    assert not out.exists()


def _trained_run(tmp_path: Path, scene: Path) -> Path:
    """A short CPU `ftgs train` run against a white background, whose METRICS.json scores the
    test view."""
    g = torch.Generator().manual_seed(1)
    init = tmp_path / "init.safetensors"
    points = InitPoints(
        positions=(torch.rand(30, 3, generator=g) - 0.5) * 1.2,
        colors=torch.full((30, 3), 0.5),
        timestamps=torch.rand(30, generator=g, dtype=torch.float64) * TAU1,
    )
    write_init_points(points, init, time_unit="s", producer="openftgs tests")
    base = TrainConfig(
        background=(1.0, 1.0, 1.0), loss=LossConfig(lambda_perc=0.0), perf=PerfConfig(log_every=5)
    )
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(config_to_dict(shortened_config(20, base))))
    run = tmp_path / "run"
    argv = ["train", scene, "--init", init, "--out", run, "--config", config]
    assert main([str(a) for a in argv]) == 0
    return run


def _scene_means(body: dict[str, Any]) -> dict[str, float]:
    return dict(body["entire"]["scene"])


def test_run_leaves_pngs_metrics_and_no_renders(tmp_path: Path) -> None:
    """#327: request, `ftgs render`, `ftgs eval --renders-dir`, PNGs. Each test record gets an
    8-bit PNG at its image path relative to the scene, equal to the render clamped to [0, 1]
    and rounded (E16's clamp) against the run's white background (D-84); METRICS.json equals the
    run's own; the comparison with it is written; renders.safetensors is deleted and
    request.json kept."""
    scene = _scene(tmp_path / "scene")
    run = _trained_run(tmp_path, scene)
    out = run / "renders"
    done = _script(
        "run", run / "checkpoint", scene, out,
        "--compare", run / "METRICS.json", "--backend", "reference",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert not (out / "renders.safetensors").exists()
    assert (out / "request.json").is_file()
    state = load_checkpoint(run / "checkpoint")
    degree = config_from_dict(state.config).sh_degree
    model = FreeTimeGaussians(state.params, sh_degree=degree, time=state.time)
    manifest = read_scene_manifest(scene)
    tests = [r for r in manifest.records if r.split == "test"]
    for record in tests:
        png = out / record.image_path.relative_to(scene.parent)
        with Image.open(png) as image:
            assert image.mode == "RGB"
            pixels = np.asarray(image)
        with torch.no_grad():
            rgb = render(
                model,
                record.camera,
                record.timestamp,
                background=torch.ones(3),
                backend="reference",
            ).image[0]
        expected = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        assert np.array_equal(pixels, expected)
    ours = json.loads((out / "METRICS.json").read_text())
    theirs = json.loads((run / "METRICS.json").read_text())
    for name, value in _scene_means(theirs).items():
        assert _scene_means(ours)[name] == pytest.approx(value, rel=1e-6, abs=1e-9)
    comparison = json.loads((out / "COMPARISON.json").read_text())
    assert comparison["findings"] == []
    assert set(comparison["blocks"]["entire"]) == set(_scene_means(theirs))
    assert all(abs(d) < 1e-6 for d in comparison["blocks"]["entire"].values())


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("view_renders", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(psnr: float | None, ssim1: float, dynamic_psnr: float | None) -> dict[str, Any]:
    image = {"path": "images/cam02/000000.png", "psnr": psnr, "ssim1": ssim1}
    return {
        "entire": {"scene": {"psnr": psnr, "ssim1": ssim1}, "images": [image]},
        "dynamic": {"scene": {"psnr": dynamic_psnr}, "images": []},
    }


def test_comparison_flags_psnr_ssim_and_skips_nulls() -> None:
    """#327: a PSNR difference above 0.01 dB and an SSIM difference above 0.001 are findings,
    smaller ones aren't; a null metric (METRICS.json's non-finite value, Q-63) gives a null
    difference rather than an error; the dynamic block is compared beside the entire one."""
    compare = _module().compare
    small = compare(_metrics(30.005, 0.9005, None), _metrics(30.0, 0.9, 25.0))
    assert small["findings"] == []
    assert small["blocks"]["entire"]["psnr"] == pytest.approx(0.005)
    assert small["blocks"]["dynamic"] == {"psnr": None}
    large = compare(_metrics(30.02, 0.902, 25.0), _metrics(30.0, 0.9, 25.0))
    assert [f.split(" differs")[0] for f in large["findings"]] == [
        "entire: scene psnr",
        "entire: scene ssim1",
        "entire: an image's PSNR",
    ]
    assert large["largest_image_psnr_difference"]["entire"] == pytest.approx(0.02)
