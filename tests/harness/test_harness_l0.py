# SPDX-License-Identifier: Apache-2.0
"""L0 · boundary converters (§3.3 L0; Q-206, Q-208, Q-209), with openftgs as both
implementations and with the fault-injecting wrapper as B. Reduced case sizes keep the CPU run
short (Q-203)."""

from pathlib import Path

import pytest
from harness_runs import checks, faulty, openftgs, run_harness


def test_l0_passes_with_openftgs_as_both_implementations(tmp_path: Path) -> None:
    """§3.3 L0: every model's import → export round trip matches within 1e-6 relative, passes
    the schema validator, and renders as before, in each implementation."""
    code, report = run_harness(
        tmp_path, "L0", a=openftgs(), b=openftgs(), settings={"l0": {"models": 5}}
    )
    assert code == 0, report.get("summary")
    assert report["summary"] == {"L0": "pass", "failed_checks": 0}
    assert len(checks(report, "L0.roundtrip.")) == 10 and len(checks(report, "L0.render.")) == 10
    assert report["run"]["settings"]["reduced"] is True


def test_l0_reports_a_changed_field_and_a_schema_violation(tmp_path: Path) -> None:
    """A velocity ×1.1 at B's import is an "offset" in B's round trip only; a key missing from
    B's export is a "schema violation"; both exit 1."""
    settings = {"l0": {"models": 1}}
    code, report = run_harness(
        tmp_path, "L0", a=openftgs(), b=faulty("velocity=1.1", on=("import",)), settings=settings
    )
    assert code == 1
    found = checks(report, "L0.roundtrip.m00")
    assert found["L0.roundtrip.m00.A"]["status"] == "pass"
    b = found["L0.roundtrip.m00.B"]
    assert b["status"] == "fail" and b["pattern"] == "offset" and "velocity" in b["symptom"]
    assert b["observed"] > 0.05
    code, report = run_harness(
        tmp_path,
        "L0",
        a=openftgs(),
        b=faulty("drop_key=velocity", on=("export",)),
        settings=settings,
        out="run2",
    )
    assert code == 1
    assert checks(report)["L0.roundtrip.m00.B"]["pattern"] == "schema violation"


def test_l0_passes_positions_stored_in_a_normalised_frame(tmp_path: Path) -> None:
    """Q-208: a B that stores positions as (x − c)/r in float32 holds them only to their rounding
    at the field's largest magnitude, which is position's floor, so its round trips pass."""
    code, report = run_harness(
        tmp_path,
        "L0",
        a=openftgs(),
        b=faulty("position_frame", on=("import",)),
        settings={"l0": {"models": 5}},
    )
    assert code == 0, [c["symptom"] for c in checks(report, "L0.roundtrip.").values()]


@pytest.mark.parametrize(
    ("fault", "field", "model"),
    [("position_zero=0.01", "position", "m00"), ("duration_small=1e-4", "duration", "m03")],
)
def test_l0_fails_errors_in_a_fields_small_entries(
    tmp_path: Path, fault: str, field: str, model: str
) -> None:
    """Q-208: positions zeroed below 10⁻²·M fail against position's largest magnitude, and a
    10⁻⁴ relative error in only the durations below 1 % of the largest (m03 has durations from
    e⁻⁶ to e⁴ × T_τ, Q-209) fails duration's floor of 1e-3 of it."""
    code, report = run_harness(
        tmp_path,
        "L0",
        a=openftgs(),
        b=faulty(fault, on=("import",)),
        settings={"l0": {"models": int(model[1:]) + 1}},
    )
    assert code == 1
    found = checks(report, f"L0.roundtrip.{model}.")
    assert found[f"L0.roundtrip.{model}.A"]["status"] == "pass"
    b = found[f"L0.roundtrip.{model}.B"]
    assert b["status"] == "fail" and b["localization"]["fields"] == [field]


def test_l0_reports_a_crash(tmp_path: Path) -> None:
    code, report = run_harness(
        tmp_path,
        "L0",
        a=openftgs(),
        b=faulty("exit=3", on=("import",)),
        settings={"l0": {"models": 1}},
    )
    assert code == 1
    b = checks(report)["L0.roundtrip.m00.B"]
    assert b["pattern"] == "crash" and "exited with code 3" in b["symptom"]


def test_l0_renders_use_the_strict_tolerance_whatever_the_tier(tmp_path: Path) -> None:
    """Q-206: B names another rasterizer, so A and B are in the "different rasterizers" tier,
    yet B's renders before and after its own round trip are judged at max |Δ| ≤ 1e-3. A DC
    change of 0.006 at B's import moves colours by about C0·0.006 ≈ 1.7e-3, below 1/255."""
    code, report = run_harness(
        tmp_path,
        "L0",
        a=openftgs(),
        b=faulty("rasterizer=other", "dc=0.006", on=("import", "info")),
        settings={"l0": {"models": 1}},
    )
    assert report["run"]["tier"] == "different"
    render = checks(report)["L0.render.m00.B"]
    assert render["status"] == "fail" and render["metric"] == "max_abs_A_vs_B"
    assert checks(report)["L0.render.m00.A"]["status"] == "pass"
    assert code == 1
