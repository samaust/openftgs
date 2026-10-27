# SPDX-License-Identifier: Apache-2.0
"""`scripts/view_renders.py`, the test-view renders of a trained model through the public CLI
(§2.2 "Rendering at any camera and time", E16, D-84; Q-56, Q-137, Q-224; #327, #333).

Added tests, not §3 rows: the request it writes is matched record for record by `ftgs eval
--renders-dir` (Q-137's rule); its background is the model's training background (D-84); the
whole run leaves one 8-bit PNG per test record equal to the clamped render, a METRICS.json equal
to the run's own, and no renders file; its comparison's findings follow Q-224's bounds, and it
reads METRICS.json versions 1 and 2 and says which it read (Q-256), leaving uncompared the
metrics one version holds alone (Q-266). The script runs as #327 runs it, in a process of its
own, here on the CPU (the reference backend).
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
from builders import pre_549o_checkpoint, random_model, ring, shortened_config, write_scene
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
    assert comparison["versions"] == {"this_run": 2, "given": 2}  # Q-256
    assert comparison["not_compared"] == {"this_run": [], "given": []}  # Q-266
    assert comparison["findings"] == []
    assert set(comparison["blocks"]["entire"]) == set(_scene_means(theirs))
    assert all(abs(d) < 1e-6 for d in comparison["blocks"]["entire"].values())


def test_run_reads_a_checkpoint_saved_before_549o(tmp_path: Path) -> None:
    """Q-271 (Q-247): the viewer runs on a checkpoint whose configuration names LPIPS-VGG,
    as M17's runs saved theirs before revision 549o: it renders the model against the white
    background the run trained with (D-84) and scores it as the run's own METRICS.json does."""
    scene = _scene(tmp_path / "scene")
    run = _trained_run(tmp_path, scene)
    legacy = pre_549o_checkpoint(run / "checkpoint", tmp_path / "legacy")
    out = tmp_path / "renders"
    done = _script("run", legacy, scene, out, "--backend", "reference")
    assert done.returncode == 0, done.stderr
    ours = json.loads((out / "METRICS.json").read_text())
    theirs = json.loads((run / "METRICS.json").read_text())
    for name, value in _scene_means(theirs).items():
        assert _scene_means(ours)[name] == pytest.approx(value, rel=1e-6, abs=1e-9)


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("view_renders", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(**changes: float | None) -> dict[str, Any]:
    """A version-2 METRICS.json body whose scene means and one image, in both blocks, hold the
    same values; `changes` sets `<block>_<scene|image>_<metric>` to another value."""
    base = {"psnr": 30.0, "ssim1": 0.9, "flip": 0.2}
    body: dict[str, Any] = {"format": "ftgs-metrics", "version": 2}
    for block in ("entire", "dynamic"):
        scene = {k: changes.get(f"{block}_scene_{k}", v) for k, v in base.items()}
        values = {k: changes.get(f"{block}_image_{k}", v) for k, v in base.items()}
        image = {"camera": "02", "path": "images/cam02/000000.png", **values}
        body[block] = {"scene": scene, "images": [image]}
    return body


def _flagged(findings: list[str]) -> list[str]:
    return [f.split(" differs")[0] for f in findings]


@pytest.mark.parametrize("block", ["entire", "dynamic"])
@pytest.mark.parametrize("level", ["scene", "image"])
def test_comparison_bounds_are_q224s(block: str, level: str) -> None:
    """Q-224: a difference above 0.01 dB in PSNR or 10⁻⁴ in SSIM and ꟻLIP is a finding, per
    scene mean and per image, in both blocks; 0.005 dB and 5·10⁻⁵ aren't. Every difference is
    written."""
    compare = _module().compare
    theirs = _metrics()
    small = {"psnr": 30.005, "ssim1": 0.9 + 5e-5, "flip": 0.2 - 5e-5}
    large = {"psnr": 30.02, "ssim1": 0.9 + 2e-4, "flip": 0.2 - 2e-4}
    quiet = compare(_metrics(**{f"{block}_{level}_{k}": v for k, v in small.items()}), theirs)
    assert quiet["findings"] == []
    loud = compare(_metrics(**{f"{block}_{level}_{k}": v for k, v in large.items()}), theirs)
    where = f"{block}: " + ("scene" if level == "scene" else "images/cam02/000000.png")
    assert _flagged(loud["findings"]) == [f"{where} {k}" for k in large]
    for result, values in ((quiet, small), (loud, large)):
        written = (
            result["blocks"][block]
            if level == "scene"
            else result["largest_image_difference"][block]
        )
        expected = {"psnr": 30.0, "ssim1": 0.9, "flip": 0.2}
        for k, v in values.items():
            assert written[k] == pytest.approx(v - expected[k], abs=1e-12)
    psnr = loud["largest_image_psnr_difference"][block]
    assert psnr == pytest.approx(0.02 if level == "image" else 0.0, abs=1e-9)


def test_comparison_flags_nulls_on_one_side_only() -> None:
    """Q-224: a value null on one side and a number on the other is a finding, per scene and
    per image, with a null difference; null on both sides is a null difference and no
    finding."""
    compare = _module().compare
    both = compare(_metrics(dynamic_scene_psnr=None), _metrics(dynamic_scene_psnr=None))
    assert both["findings"] == []
    assert both["blocks"]["dynamic"]["psnr"] is None
    one = compare(_metrics(dynamic_scene_psnr=None), _metrics(entire_image_ssim1=None))
    assert one["findings"] == [
        "entire: images/cam02/000000.png ssim1 is null in the given METRICS.json",
        "dynamic: scene psnr is null in this run's METRICS.json",
    ]
    assert one["blocks"]["dynamic"]["psnr"] is None
    assert "ssim1" not in one["largest_image_difference"]["entire"]


def test_comparison_flags_what_one_file_holds_alone() -> None:
    """Q-224: a block, a metric and an image path present in one file only are each a finding,
    whichever file holds it; what both hold is still compared."""
    compare = _module().compare
    ours, theirs = _metrics(), _metrics()
    del ours["dynamic"]
    assert compare(ours, theirs)["findings"] == ["block dynamic is only in the given METRICS.json"]
    assert compare(theirs, ours)["findings"] == ["block dynamic is only in this run's METRICS.json"]
    ours = _metrics()
    del ours["entire"]["scene"]["flip"]
    del theirs["dynamic"]["images"][0]["ssim1"]
    theirs["dynamic"]["images"].append({**theirs["dynamic"]["images"][0], "path": "b.png"})
    result = compare(ours, theirs)
    assert result["findings"] == [
        "entire: scene metric flip is only in the given METRICS.json",
        "dynamic: image b.png is only in the given METRICS.json",
        "dynamic: images/cam02/000000.png metric ssim1 is only in this run's METRICS.json",
    ]
    assert result["blocks"]["entire"] == {"psnr": 0.0, "ssim1": 0.0}
    assert result["largest_image_difference"]["entire"]["flip"] == 0.0  # both rows hold it
    assert result["largest_image_difference"]["dynamic"] == {"psnr": 0.0, "flip": 0.0}


def _version_1(body: dict[str, Any]) -> dict[str, Any]:
    """A version-1 copy of a `_metrics` body, as M17's runs wrote it: LPIPS-Alex, LPIPS-VGG and
    lpips_vgg_raw in place of ꟻLIP, and the backbone label (Q-62, Q-63)."""
    old: dict[str, Any] = json.loads(json.dumps(body))
    old["version"], old["lpips_backbone"] = 1, "vgg"
    for block in ("entire", "dynamic"):
        for values in (old[block]["scene"], *old[block]["images"]):
            value = values.pop("flip")
            values |= {"lpips_alex": value, "lpips_vgg": value, "lpips_vgg_raw": value}
    return old


def test_comparison_says_which_versions_it_read() -> None:
    """Q-256, Q-266: a version-2 run compared with a version-1 file says which version each
    holds. The metrics both versions hold (PSNR and SSIM₁ in these bodies) are compared as
    Q-224 compares them, the PSNR difference above 0.01 dB being the only finding; the metrics
    one version holds alone (ꟻLIP; LPIPS-Alex, LPIPS-VGG, lpips_vgg_raw) are compared nowhere,
    listed once under `not_compared`, and are no finding, in scene means or image rows."""
    compare = _module().compare
    ours = _metrics(entire_scene_psnr=30.02)
    result = compare(ours, _version_1(_metrics()))
    assert result["versions"] == {"this_run": 2, "given": 1}
    assert result["not_compared"] == {
        "this_run": ["flip"],
        "given": ["lpips_alex", "lpips_vgg", "lpips_vgg_raw"],
    }
    assert result["findings"] == ["entire: scene psnr differs by +0.02"]
    assert result["blocks"]["entire"]["psnr"] == pytest.approx(0.02, abs=1e-9)
    for block in ("entire", "dynamic"):
        assert set(result["blocks"][block]) == {"psnr", "ssim1"}
        assert set(result["largest_image_difference"][block]) == {"psnr", "ssim1"}
    same = compare(ours, ours)
    assert same["versions"] == {"this_run": 2, "given": 2}
    assert same["not_compared"] == {"this_run": [], "given": []}


def test_comparison_across_versions_still_flags_a_shared_metric_in_one_file_only() -> None:
    """Q-266, Q-224: across versions, a metric both versions hold that one file lacks, in
    its scene means or in an image row, is still a finding."""
    compare = _module().compare
    theirs = _version_1(_metrics())
    del theirs["entire"]["scene"]["ssim1"]
    del theirs["dynamic"]["images"][0]["psnr"]
    assert compare(_metrics(), theirs)["findings"] == [
        "entire: scene metric ssim1 is only in this run's METRICS.json",
        "dynamic: images/cam02/000000.png metric psnr is only in this run's METRICS.json",
    ]


@pytest.mark.parametrize(
    "header",
    [{"format": "ftgs-metrics", "version": 3}, {"format": "another", "version": 2}, {}],
    ids=["version-3", "another-format", "untagged"],
)
def test_run_refuses_to_compare_with_another_version(
    tmp_path: Path, header: dict[str, Any]
) -> None:
    """Q-256: `--compare` reads METRICS.json versions 1 and 2; a file of another version or
    format, or without its tag, is exit 2, checked before anything is rendered or written."""
    given = tmp_path / "given.json"
    body = {key: value for key, value in _metrics().items() if key not in ("format", "version")}
    given.write_text(json.dumps({**header, **body}))
    out = tmp_path / "renders"
    done = _script("run", tmp_path / "model.safetensors", tmp_path / "scene.json", out,
                   "--compare", given, "--backend", "reference")  # fmt: skip
    assert done.returncode == 2 and "--compare" in done.stderr
    assert not out.exists()
