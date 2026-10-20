# SPDX-License-Identifier: Apache-2.0
"""Milestone M16's test (PRD #279): the differential harness run with openftgs as both
implementations, every level at §3.3's full case sizes, on S-9 for L3 (K = 5).

"Passes every level" is read as no failed check and exit 0 (Q-211): with K = 5 no
Holm-corrected distribution flag is reachable, so those checks, and L3's summary, are
`inconclusive` (Q-210). PSNR's verdict depends on S-9's seed-to-seed spread (Calibration data):
it may be "equivalent" or "inconclusive", never "different"; SSIM and LPIPS, far inside their
margins, are "equivalent". The verdicts and the wall time are recorded for the Calibration data
table."""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from harness_runs import ADAPTER, BIN, checks

ROOT = Path(__file__).resolve().parents[2]

Record = Callable[[str, object], None]


@pytest.mark.gpu
@pytest.mark.weights
def test_the_harness_passes_every_level_with_openftgs_as_both(
    tmp_path: Path, record_property: Record
) -> None:
    s9 = tmp_path / "s9"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "harness_s9.py"), str(s9)], check=True)
    adapter = f"{ADAPTER} --backend gsplat"
    started = time.perf_counter()
    done = subprocess.run(
        [
            str(BIN / "ftgs-harness"),
            "run",
            "--a",
            adapter,
            "--b",
            adapter,
            "--config",
            str(s9 / "harness_config.s9.json"),
            "--settings",
            str(s9 / "settings.s9.json"),
            "--levels",
            "L0,L1,L2,L3",
            "--out",
            str(tmp_path / "run"),
            "--jobs",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    failed = [c["id"] for c in report["checks"] if c["status"] == "fail"]
    record_property("seconds", seconds)
    assert done.returncode == 0, (failed, done.stderr[-2000:])
    summary = report["summary"]
    assert summary["failed_checks"] == 0
    assert summary["L0"] == summary["L1"] == summary["L2"] == "pass"
    assert summary["L3"] == "inconclusive"
    metrics = checks(report, "L3.metric.s9.")
    verdicts = {k.rsplit(".", 1)[1]: c["localization"]["verdict"] for k, c in metrics.items()}
    for metric in ("ssim1", "ssim2", "lpips_alex", "lpips_vgg"):
        assert verdicts[metric] == "equivalent", verdicts
    assert verdicts["psnr"] in ("equivalent", "inconclusive"), verdicts
    psnr = metrics["L3.metric.s9.psnr"]["localization"]
    record_property("psnr_verdict", verdicts["psnr"])
    record_property("psnr_per_seed_A", psnr["per_seed_A"])
    record_property("psnr_per_seed_A2", psnr["per_seed_A2"])
    dists = checks(report, "L3.dist.s9.")
    assert dists and {c["status"] for c in dists.values()} == {"inconclusive"}
