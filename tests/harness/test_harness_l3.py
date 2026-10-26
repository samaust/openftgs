# SPDX-License-Identifier: Apache-2.0
"""L3 · multi-seed statistics, its plumbing (§3.3 L3; D-54; Q-203, Q-210, Q-211), with a replay
adapter whose `train` copies pre-made tiny CPU runs. The verdicts and statistics themselves are
pinned in test_harness_stats.py; the S-9 run is test_harness_e2e.py. L3 computes ꟻLIP with the
harness's own copy on the CPU and downloads nothing, so these tests run in CPU CI (Q-254)."""

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import random_model, ring, write_scene
from harness_runs import ADAPTER, BIN, HERE, checks, run_harness

from ftgs_harness.level3 import QUANTITIES, trace_targets
from ftgs_harness.stats import holm, tost_p, welch_p
from openftgs.config import LossConfig, PerfConfig, TrainConfig, config_to_dict
from openftgs.io import write_init_points
from openftgs.types import InitPoints, TimeNormalizer

ITERATIONS = 4
# §3.3 L3 step 4: the TOST margins of the four tested metrics (Q-210).
MARGINS = {"psnr": 0.25, "ssim1": 0.005, "ssim2": 0.005, "flip": 0.005}


def test_trace_targets_scale_with_the_runs_length() -> None:
    """§3.3 L3 step 6 at 1k, 7k, 15k and 30k, scaled by T_it/30,000 as §3.2's calendar is."""
    assert trace_targets(30_000) == [1_000, 7_000, 15_000, 30_000]
    assert trace_targets(7_000) == [233, 1_633, 3_500, 7_000]
    assert trace_targets(4) == [1, 1, 2, 4]


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A 48×32 scene with three training cameras and a held-out one, its init points, a
    harness config, and three tiny CPU runs to replay."""
    root = tmp_path_factory.mktemp("l3")
    time = TimeNormalizer(0.0, 2.0 / 30.0)
    cams = ring(4, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    gt = random_model(16, seed=0, sh_degree=0, time=time, spread=0.6, log_scale=(-2.5, -1.5))
    images: list[Any] = [(c, k / 30.0, "train") for k in range(3) for c in cams[:3]]
    images += [(cams[3], k / 30.0, "test") for k in range(3)]
    manifest = write_scene(root / "scene", gt, images)
    g = torch.Generator().manual_seed(1)
    points = InitPoints(
        positions=torch.rand(30, 3, generator=g) - 0.5,
        colors=torch.full((30, 3), 0.5),
        timestamps=torch.rand(30, generator=g, dtype=torch.float64) * (2.0 / 30.0),
    )
    write_init_points(points, root / "init.safetensors", time_unit="s", producer="test")
    cfg = TrainConfig(sh_degree=0, loss=LossConfig(lambda_perc=0.0), perf=PerfConfig(log_every=1))
    data = config_to_dict(cfg)
    data["optim"]["iterations"] = ITERATIONS
    del data["seed"]
    config = root / "harness_config.json"
    config.write_text(json.dumps(data))
    for seed in range(3):
        argv = [ADAPTER, "--backend", "reference", "train", manifest, root / "init.safetensors"]
        argv += [config, seed, root / "source" / f"run{seed}"]
        subprocess.run([str(a) for a in argv], check=True, capture_output=True)
    return {
        "manifest": manifest,
        "init": root / "init.safetensors",
        "config": config,
        "source": root / "source",
    }


def _replay(source: Path, *extra: str) -> str:
    return shlex.join(
        [sys.executable, str(HERE / "replay_adapter.py"), "--source", str(source), *extra]
    )


def _assert_holm_across(metrics: dict[str, dict[str, Any]]) -> None:
    """§3.3 L3 step 4, Q-210: each metric check's TOST and Welch p are Holm-corrected across
    every metric check of the run, the four metrics times the scenes, recomputed from the
    per-seed values the checks report."""
    rows = [(c["inputs"]["metric"], c["localization"]) for c in metrics.values()]
    raw_tost = [tost_p(loc["per_seed_A"], loc["per_seed_B"], MARGINS[m]) for m, loc in rows]
    raw_welch = [welch_p(loc["per_seed_A"], loc["per_seed_B"]) for _, loc in rows]
    for (_, loc), p_tost, p_welch in zip(rows, holm(raw_tost), holm(raw_welch), strict=True):
        assert loc["p_tost_holm"] == pytest.approx(p_tost, rel=1e-12)
        assert loc["p_welch_holm"] == pytest.approx(p_welch, rel=1e-12)


def test_l3_runs_compares_and_reports(tmp_path: Path, scene: dict[str, Path]) -> None:
    """K = 5: five runs of A and of B and five of A′ on seeds 5 to 9, round robin over two GPU
    slots (D-54); the four tested image metrics, PSNR, SSIM₁, SSIM₂ and ꟻLIP, get verdicts,
    Holm-corrected across the four (§3.3 L3 steps 3 and 4, Q-210), the distribution checks are
    inconclusive because no Holm-corrected flag is reachable at K = 5 (Q-210), and the traces sit
    at the scaled iterations. A setting B can't honor is not comparable."""
    settings = {
        "k": 5,
        "gpus": ["0", "1"],
        "l3": {
            "scenes": [
                {"name": "tiny", "manifest": str(scene["manifest"]), "init": str(scene["init"])}
            ]
        },
    }
    config = json.loads(scene["config"].read_text())
    code, report = run_harness(
        tmp_path,
        "L3",
        a=_replay(scene["source"]),
        b=_replay(scene["source"], "--not-comparable", "growth.unknown"),
        settings=settings,
        config=config,
        jobs=1,
    )
    assert code == 0, [c["id"] for c in report.get("checks", []) if c["status"] == "fail"]
    run_root = tmp_path / "run" / "L3" / "tiny"
    seeds = {
        impl: sorted(p.name for p in (run_root / impl).iterdir() if p.is_dir())
        for impl in ("A", "B", "A2")
    }
    assert seeds["A"] == seeds["B"] == [f"seed{s}" for s in range(5)]
    assert seeds["A2"] == [f"seed{s}" for s in range(5, 10)]
    devices = {
        json.loads((p / "env.json").read_text())["CUDA_VISIBLE_DEVICES"]
        for p in run_root.glob("*/seed*")
        if p.is_dir()
    }
    assert devices == {"0", "1"}
    metrics = checks(report, "L3.metric.tiny.")
    assert sorted(metrics) == sorted(f"L3.metric.tiny.{m}" for m in MARGINS)
    assert all(
        c["localization"]["verdict"] in ("equivalent", "inconclusive") for c in metrics.values()
    )
    assert "dssim_A" in metrics["L3.metric.tiny.ssim1"]["localization"]
    flip = metrics["L3.metric.tiny.flip"]
    assert flip["metric"] == "flip_mean_B_minus_A" and "0.005" in flip["expected"]
    assert all(0.0 < v < 1.0 for v in flip["localization"]["per_seed_A"])
    _assert_holm_across(metrics)
    dists = checks(report, "L3.dist.tiny.")
    assert len(dists) == 2 * len(QUANTITIES)
    assert {c["status"] for c in dists.values()} == {"inconclusive"}
    assert "7 seeds a side" in next(iter(dists.values()))["symptom"]
    traces = checks(report, "L3.trace.tiny.")
    assert [c["inputs"]["iteration"] for c in traces.values()] == [1, 1, 2, 4]
    assert all(c["status"] == "pass" for c in traces.values())
    nc = checks(report, "L3.config.tiny.")
    assert list(nc) == ["L3.config.tiny.B.growth.unknown"]
    assert report["summary"]["L3"] == "inconclusive"


def test_a_failed_run_is_reported(tmp_path: Path, scene: dict[str, Path]) -> None:
    settings = {
        "k": 2,
        "l3": {
            "scenes": [
                {"name": "tiny", "manifest": str(scene["manifest"]), "init": str(scene["init"])}
            ]
        },
    }
    code, report = run_harness(
        tmp_path,
        "L3",
        a=_replay(scene["source"]),
        b=_replay(tmp_path / "nowhere"),
        settings=settings,
        jobs=1,
    )
    assert code == 1
    failed = [c for c in checks(report, "L3.run.tiny.B.").values() if c["status"] == "fail"]
    assert len(failed) == 2 and failed[0]["pattern"] == "crash"


def test_a_rerun_of_an_l3_check_reruns_its_own_scene_only(
    tmp_path: Path, scene: dict[str, Path]
) -> None:
    """Q-203: `rerun` of an L3 check replays that scene's 3K runs, not the other scenes'."""
    entry = {"manifest": str(scene["manifest"]), "init": str(scene["init"])}
    settings = {"k": 2, "l3": {"scenes": [{"name": "one", **entry}, {"name": "two", **entry}]}}
    code, _ = run_harness(
        tmp_path,
        "L3",
        a=_replay(scene["source"]),
        b=_replay(scene["source"]),
        settings=settings,
        jobs=1,
    )
    assert code == 0
    done = subprocess.run(
        [str(BIN / "ftgs-harness"), "rerun", "L3.metric.two.psnr", "--out", str(tmp_path / "run")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["id"] == "L3.metric.two.psnr"
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    metrics = checks(report, "L3.metric.")
    assert len(metrics) == 2 * len(MARGINS)
    _assert_holm_across(metrics)
    rerun = tmp_path / "run" / "reruns" / "L3.metric.two.psnr" / "L3"
    assert sorted(p.name for p in rerun.iterdir()) == ["two"]
