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


PROBE_SIZES = {"l1": {"items": 4, "probe_items": 11}}


@pytest.mark.parametrize(
    ("fault", "probe", "pattern", "trend"),
    [
        ("velocity=1.1", "P2", "offset", "linear in |τ − time_center|"),
        ("duration=1.2", "P3", "temporal widening", None),
        ("cx=0.5", "P1", "offset", "constant over sub-pixel position"),
        ("dc=0.05", "P1", "color shift", None),
        ("sh_sign", "P5", "view-dependent color shift", None),
        ("scale=1.1", "P1", "scale change", None),
        ("drop_primitive", "P1", "missing primitive", None),
    ],
)
def test_a_probe_points_at_the_changed_quantity(
    tmp_path: Path, fault: str, probe: str, pattern: str, trend: str | None
) -> None:
    """§3.3 L1 probes: B's renders with one quantity changed fail the probe that sweeps it, with
    the pattern that names it and, for the sweeps whose trend is known, that trend (Q-211)."""
    code, report = run_harness(
        tmp_path, "L1", a=openftgs(), b=faulty(fault, on=("render",)), settings=PROBE_SIZES
    )
    assert code == 1
    failed = [c for c in checks(report, f"L1.render.{probe}.").values() if c["status"] == "fail"]
    assert failed, f"{probe} did not fail"
    assert {c["pattern"] for c in failed} == {pattern}
    assert all(c["localization"].get("trend") for c in failed)
    if trend is not None:
        assert trend in failed[0]["symptom"], failed[0]["symptom"]
    assert any(a.endswith("_trend.png") for a in failed[0]["artifacts"])


def test_probes_pass_with_openftgs_as_both_implementations(tmp_path: Path) -> None:
    code, report = run_harness(tmp_path, "L1", a=openftgs(), b=openftgs(), settings=PROBE_SIZES)
    assert code == 0
    assert {c["id"].split(".")[2] for c in checks(report, "L1.render.P").values()} == {
        f"P{k}" for k in range(1, 11)
    }


# A private test of the probe analysis: it holds which item names a failing probe's pattern
# when its items failed on different measures.
def test_probes_name_a_failing_probe_by_its_lowest_psnr_item(tmp_path: Path) -> None:
    """§3.3 L1 probes: the probe takes its worst item's pattern, the item of lowest PSNR,
    whichever measure failed each item (PSNR, where low is bad, or max |Δ|, where high is)."""
    from types import SimpleNamespace

    import numpy as np

    from ftgs_harness import probes
    from ftgs_harness.checks import Check

    probe = probes.PROBES[0]  # P1, neither time-swept nor view-dependent
    _, items = probe.build(3)
    ids = [e.item.id for e in items]
    base = np.random.default_rng(0).random((16, 24, 3)).astype(np.float32)
    alpha = np.ones((*base.shape[:2], 1), np.float32)
    a = {i: (base, alpha) for i in ids}
    b = {**a, ids[0]: (base + 0.002, alpha), ids[1]: (base + 0.02, alpha)}  # ≈ 54 dB, ≈ 34 dB

    def failed(i: int, metric: str, observed: float, pattern: str) -> Check:
        return Check(
            id=f"L1.render.P1.{ids[i]}",
            level="L1",
            status="fail",
            spec_refs=probe.spec_refs,
            inputs={"items": [ids[i]]},
            metric=metric,
            expected="",
            observed=observed,
            symptom="s",
            pattern=pattern,
        )

    found = [
        failed(0, "max_abs_A_vs_B", 0.002, "offset"),
        failed(1, "psnr_A_vs_B_db", 34.0, "color shift"),
    ]
    ctx = SimpleNamespace(out=tmp_path, run_dir=tmp_path)
    probes._analyse(ctx, probe, items, a, b, found)  # type: ignore[arg-type]
    assert {c.pattern for c in found} == {"color shift"}
