# SPDX-License-Identifier: Apache-2.0
"""L1 · identical model, near-exact renders (§3.3 L1; Q-206, Q-209, Q-211): openftgs as both
implementations, the reference backend against gsplat (the milestone's test), and faults."""

from pathlib import Path

import pytest
from harness_runs import checks, faulty, openftgs, run_harness

SMALL = {"l1": {"items": 30}}


def test_l1_main_case_passes_with_openftgs_as_both_implementations(tmp_path: Path) -> None:
    """Identical renders: the same tier, and PSNR is null with an "inf" note (Q-211)."""
    code, report = run_harness(tmp_path, "L1", a=openftgs(), b=openftgs(), settings=SMALL)
    assert code == 0 and report["run"]["tier"] == "same"
    main = checks(report, "L1.render.main.")
    assert len(main) == 30
    assert all(c["observed"] is None and c["note"] == "inf" for c in main.values())


@pytest.mark.gpu
def test_l1_reference_against_gsplat_passes_at_its_tier(tmp_path: Path) -> None:
    """Milestone M16: the reference backend as A and gsplat as B, at §3.3's full 200 items,
    judged in the "different rasterizers" tier (Q-206): PSNR(A, B) ≥ 45 dB and at most 0.5 % of
    pixels with |Δ| > 1/255, on every item."""
    code, report = run_harness(tmp_path, "L1", a=openftgs("reference"), b=openftgs("gsplat"))
    assert report["run"]["tier"] == "different"
    main = checks(report, "L1.render.main.")
    assert len(main) >= 200
    failed = {
        k: (c["metric"], c["observed"]) for k, c in report_checks(report) if c["status"] == "fail"
    }
    assert code == 0, failed


def report_checks(report: dict) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    return [(c["id"], c) for c in report["checks"]]


def test_l1_reports_a_shifted_principal_point_as_an_offset(tmp_path: Path) -> None:
    """B renders with cx + 0.5 px: items fail with the "offset" pattern, about 0.5 px to the
    right, with artifacts and a pixel box."""
    code, report = run_harness(
        tmp_path, "L1", a=openftgs(), b=faulty("cx=0.5", on=("render",)), settings=SMALL
    )
    assert code == 1
    failed = [c for c in checks(report, "L1.render.main.").values() if c["status"] == "fail"]
    assert failed
    offsets = [c for c in failed if c["pattern"] == "offset"]
    assert len(offsets) >= 0.8 * len(failed)
    assert "right of A's" in offsets[0]["symptom"] and " px " in offsets[0]["symptom"]
    assert offsets[0]["artifacts"] and "pixel_bbox" in offsets[0]["localization"]
    assert (tmp_path / "run" / offsets[0]["artifacts"][0]).is_file()


def test_l1_reports_non_finite_renders_and_crashes(tmp_path: Path) -> None:
    code, report = run_harness(
        tmp_path, "L1", a=openftgs(), b=faulty("nan", on=("render",)), settings=SMALL
    )
    assert code == 1
    assert {c["pattern"] for c in checks(report, "L1.render.main.").values()} == {"NaN/Inf"}
    code, report = run_harness(
        tmp_path, "L1", a=openftgs(), b=faulty("exit=1", on=("render",)), settings=SMALL, out="run2"
    )
    assert code == 1
    crash = checks(report)["L1.render.main.B"]
    assert crash["pattern"] == "crash" and crash["status"] == "fail"
