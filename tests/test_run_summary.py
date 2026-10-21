# SPDX-License-Identifier: Apache-2.0
"""`scripts/run_summary.py`, the summary of a `ftgs train` run directory that M17 records in the
Calibration data table (D-10, D-18, D-44; Q-220; #317).

Added tests, not §3 rows: the N trace and the cap from `train_log.jsonl` against a log written
by hand; the opacity statistics against the checkpoints read directly; the metrics against the
run's METRICS.json. The script runs as M17 runs it, in a process of its own.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import random_model, ring, shortened_config, write_scene

from openftgs.cli import main
from openftgs.config import GrowthConfig, LossConfig, PerfConfig, TrainConfig, config_to_dict
from openftgs.io import load_checkpoint, write_init_points
from openftgs.types import InitPoints, TimeNormalizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_summary.py"
TAU1 = 2.0 / 30.0  # three frames at 30 FPS
TIME = TimeNormalizer(0.0, TAU1)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-125: the CLI picks CUDA when PyTorch sees it; these runs use the reference backend."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _summary(run: Path, *args: str) -> tuple[int, dict[str, Any], str]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(run), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    out = run / "SUMMARY.json"
    body = json.loads(out.read_text()) if done.returncode == 0 and out.is_file() else {}
    return done.returncode, body, done.stderr


def _log_row(iteration: int, n: int, reg: float, total: float) -> dict[str, Any]:
    losses = {"l1": 0.1, "one_minus_ssim": 0.2, "perceptual": 0.0, "reg": reg, "total": total}
    return {"iteration": iteration, "losses": losses, "num_gaussians": n}


def test_summary_reads_the_n_trace_the_cap_and_the_loss_blocks(tmp_path: Path) -> None:
    """D-18, D-44, Q-220: from a log written by hand (rows every 500 iterations to 3,000,
    growth from 500 to 1,500, n_max 400): N at the first row, at the end of growth and at the
    end; the first iteration at n_max; the mean reg and total losses over each block of 1,000
    iterations (rows 500 and 1,000 in the first block, 1,500 and 2,000 in the second, 2,500 and
    3,000 in the third)."""
    run = tmp_path / "run"
    run.mkdir()
    cfg = TrainConfig(growth=GrowthConfig(start=500, stop=1500, n_max=400))
    (run / "config.json").write_text(json.dumps(config_to_dict(cfg)))
    rows = [
        _log_row(500, 100, 0.4, 1.0),
        _log_row(1000, 250, 0.2, 0.8),
        _log_row(1500, 400, 0.1, 0.6),
        _log_row(2000, 400, 0.3, 0.4),
        _log_row(2500, 400, 0.0, 0.3),
        _log_row(3000, 400, 0.0, 0.1),
    ]
    (run / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, body, err = _summary(run)
    assert code == 0, err
    trace = body["n_trace"]
    assert trace["first"] == {"iteration": 500, "num_gaussians": 100}
    assert trace["growth_end"] == {"iteration": 1500, "num_gaussians": 400}
    assert trace["last"] == {"iteration": 3000, "num_gaussians": 400}
    assert trace["n_max"] == 400 and trace["first_at_n_max"] == 1500
    blocks = body["loss_blocks"]
    assert [b["iterations"] for b in blocks] == [[1, 1000], [1001, 2000], [2001, 3000]]
    assert [b["rows"] for b in blocks] == [2, 2, 2]
    assert [b["reg"] for b in blocks] == pytest.approx([0.3, 0.2, 0.0])
    assert [b["total"] for b in blocks] == pytest.approx([0.9, 0.5, 0.2])
    assert body["checkpoints"] == [] and body["metrics"] is None


def test_summary_of_a_folder_that_isnt_a_run_exits_2(tmp_path: Path) -> None:
    """A folder without config.json and train_log.jsonl is exit 2, with nothing written."""
    code, body, err = _summary(tmp_path)
    assert code == 2 and "config.json" in err
    assert not (tmp_path / "SUMMARY.json").exists()


def test_summary_opacity_statistics_are_the_checkpoints(tmp_path: Path) -> None:
    """D-10, Q-220: a short CPU `ftgs train` run keeping checkpoints every 10 of 30 iterations
    (D-83) gives one entry per checkpoint, in order of iteration, whose N, mean σ, 10-bin
    histogram of σ over [0, 1] and share σ > 0.9 are those of the checkpoint read directly."""
    cams = ring(3, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    gt = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    scene = write_scene(
        tmp_path / "scene", gt, [(c, k / 30.0, "train") for k in range(3) for c in cams]
    )
    g = torch.Generator().manual_seed(1)
    init = tmp_path / "init.safetensors"
    points = InitPoints(
        positions=(torch.rand(30, 3, generator=g) - 0.5) * 1.2,
        colors=torch.full((30, 3), 0.5),
        timestamps=torch.rand(30, generator=g, dtype=torch.float64) * TAU1,
    )
    write_init_points(points, init, time_unit="s", producer="openftgs tests")
    base = TrainConfig(loss=LossConfig(lambda_perc=0.0), perf=PerfConfig(log_every=5))
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(config_to_dict(shortened_config(30, base))))
    run = tmp_path / "run"
    argv = ["train", scene, "--init", init, "--out", run, "--config", config,
            "--checkpoint-every", "10", "--keep-checkpoints", "3"]  # fmt: skip
    assert main([str(a) for a in argv]) == 0
    code, body, err = _summary(run, "--block", "10")
    assert code == 0, err
    assert [c["iteration"] for c in body["checkpoints"]] == [10, 20, 30]
    for entry in body["checkpoints"]:
        state = load_checkpoint(run / entry["path"])
        assert state.iteration == entry["iteration"]
        sigma = torch.sigmoid(state.params["opacity_logits"].double())
        assert entry["num_gaussians"] == sigma.numel()
        assert entry["opacity_mean"] == pytest.approx(float(sigma.mean()), rel=1e-12)
        expected = [int(((sigma >= k / 10) & (sigma < (k + 1) / 10)).sum()) for k in range(10)]
        expected[-1] += int((sigma == 1.0).sum())
        assert entry["opacity_histogram"] == expected
        share = float((sigma > 0.9).double().mean())
        assert entry["opacity_share_above_0_9"] == pytest.approx(share, rel=1e-12)
    assert body["n_trace"]["last"]["iteration"] == 30
    assert [b["iterations"] for b in body["loss_blocks"]] == [[1, 10], [11, 20], [21, 30]]


def test_summary_reads_metrics_and_the_wall_time(tmp_path: Path) -> None:
    """The METRICS.json part (scene means over the entire images, the label, the dynamic and
    fastest-10 blocks without their per-image rows) and the wall-time part, from a METRICS.json,
    a `/usr/bin/time -v` report and `nvidia-smi` samples written by hand: 1:02:03.5 is 3,723.5 s;
    the GPU peak is the largest sample."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config_to_dict(TrainConfig())))
    (run / "train_log.jsonl").write_text(json.dumps(_log_row(100, 10, 0.1, 0.5)) + "\n")
    scene = {"psnr": 30.0, "lpips_vgg": 0.1, "lpips_vgg_raw": 0.2}
    row = {"camera": "0007", "psnr": 30.0}
    metrics = {
        "format": "ftgs-metrics",
        "version": 1,
        "split": "test",
        "lpips_backbone": "vgg",
        "entire": {"scene": scene, "images": [row, row]},
        "dynamic": {"mask_source": "m", "masks_approximate": False, "count": 2, "skipped": 0,
                    "missing": 0, "scene": {"psnr": 25.0}, "images": [row, row]},
        "fastest_10": {"range": [3, 13], "count": 2, "entire": {"psnr": 29.0}},
    }  # fmt: skip
    (run / "METRICS.json").write_text(json.dumps(metrics))
    time_log = tmp_path / "time.txt"
    time_log.write_text(
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.5\n"
        "\tMaximum resident set size (kbytes): 123456\n"
    )
    gpu_log = tmp_path / "gpu.csv"
    gpu_log.write_text("500 MiB\n9000 MiB\n8000 MiB\n")
    code, body, err = _summary(run, "--time", str(time_log), "--gpu-log", str(gpu_log))
    assert code == 0, err
    assert body["metrics"] == {
        "split": "test",
        "lpips_backbone": "vgg",
        "count": 2,
        "entire": scene,
        "dynamic": {"mask_source": "m", "masks_approximate": False, "count": 2, "skipped": 0,
                    "missing": 0, "scene": {"psnr": 25.0}},
        "fastest_10": {"range": [3, 13], "count": 2, "entire": {"psnr": 29.0}},
    }  # fmt: skip
    assert body["wall"] == {
        "elapsed_s": pytest.approx(3723.5),
        "max_rss_kib": 123456,
        "gpu_memory_used_peak_mib": 9000,
    }
