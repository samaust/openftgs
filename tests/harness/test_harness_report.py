# SPDX-License-Identifier: Apache-2.0
"""The harness's run mechanics and report (§3.3 "Report format", D-73; Q-203, Q-205, Q-206,
Q-211): the check record, the level summary, report.json and report.md, the `info` checks and
the L1 tier, the exit codes and `rerun`. A stub adapter answers `info`, and a stand-in level
module produces checks, so these tests don't depend on any real level."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from stub_adapter import CONVENTIONS, INFO

from ftgs_harness import runner
from ftgs_harness.checks import Check, close_absolute, close_relative, level_summary
from ftgs_harness.cli import main
from ftgs_harness.info import choose_tier, info_problems
from ftgs_harness.report import build_report, render_markdown

HERE = Path(__file__).parent
STUB = f"{sys.executable} {HERE / 'stub_adapter.py'}"

# §3.3's example: the top-level keys of report.json and of a check.
REPORT_KEYS = {"format", "version", "run", "implementations", "summary", "checks"}
CHECK_KEYS = {
    "id",
    "level",
    "status",
    "spec_refs",
    "inputs",
    "metric",
    "expected",
    "observed",
    "symptom",
    "localization",
    "repro",
    "artifacts",
}


def _check(check_id: str, status: str, pattern: str | None = None) -> Check:
    return Check(
        id=check_id,
        level=check_id.split(".")[0],
        status=status,  # type: ignore[arg-type]
        spec_refs=["E1"],
        inputs={},
        metric="m",
        expected="e",
        observed=1.0,
        symptom="s" if status == "fail" else "",
        pattern=pattern,
    )


def test_level_summary_fails_on_any_failure_then_inconclusive_then_passes() -> None:
    """Q-211: fail if any check failed, otherwise inconclusive if any was, otherwise pass; a
    not-comparable check fails nothing."""
    assert level_summary(["pass", "inconclusive", "fail"]) == "fail"
    assert level_summary(["pass", "inconclusive", "not_comparable"]) == "inconclusive"
    assert level_summary(["pass", "not_comparable"]) == "pass"
    assert level_summary([]) == "pass"


def test_check_writes_non_finite_values_as_null_with_a_note() -> None:
    """Q-63, Q-211: no NaN or Infinity in report.json; an infinite PSNR is null with "inf"."""
    check = Check("L1.render.main.r0000", "L1", "pass", [], {}, "psnr", ">= 60", float("inf"))
    body = check.to_json()
    assert body["observed"] is None and body["note"] == "inf"
    json.dumps(body, allow_nan=False)
    nested = Check("L1.x", "L1", "pass", [], {"v": [float("nan"), 1.0]}, "m", "e", 0.0).to_json()
    assert nested["inputs"] == {"v": [None, 1.0]}


def test_check_refuses_a_pattern_outside_the_vocabulary() -> None:
    """§3.3 symptom rule 4: the fixed pattern vocabulary."""
    with pytest.raises(ValueError, match="vocabulary"):
        _check("L1.x", "fail", "wobble")


def test_report_has_section_3_3s_keys_and_counts_failed_checks() -> None:
    checks = [_check("L0.a", "pass"), _check("L0.b", "fail", "offset"), _check("L2.c", "pass")]
    report = build_report(
        run={"started": "t", "harness_version": "v", "config_sha256": "h"},
        implementations={"A": {"name": "a", "version": "1", "info": {}}},
        levels=["L0", "L2"],
        checks=checks,
    )
    assert set(report) == REPORT_KEYS
    assert report["format"] == "ftgs-harness-report" and report["version"] == 1
    assert report["summary"] == {"L0": "fail", "L2": "pass", "failed_checks": 1}
    assert all(CHECK_KEYS <= set(c) for c in report["checks"])
    json.dumps(report, allow_nan=False)


def test_report_md_lists_failed_checks_first_grouped_by_pattern() -> None:
    """§3.3: report.md lists failed checks first, grouped by pattern."""
    checks = [
        _check("L0.a", "pass"),
        _check("L1.b", "fail", "offset"),
        _check("L1.c", "inconclusive"),
        _check("L1.d", "fail", "color shift"),
        _check("L1.e", "fail", "offset"),
    ]
    report = build_report(
        run={"started": "t", "harness_version": "v", "config_sha256": "h"},
        implementations={},
        levels=["L0", "L1"],
        checks=checks,
    )
    md = render_markdown(report)
    order = [md.index(f"`{i}`") for i in ("L1.b", "L1.e", "L1.d", "L1.c", "L0.a")]
    assert order == sorted(order)
    assert md.index("### offset (2)") < md.index("### color shift (1)")


def test_info_problems_names_each_missing_render_convention() -> None:
    """D-73: an info reply reports at least the 12 entries of render_conventions (Q-35)."""
    assert info_problems(INFO) == []
    partial = {**INFO, "render_conventions": {k: v for k, v in CONVENTIONS.items() if k != "near"}}
    assert info_problems(partial) == ["`render_conventions` lacks near"]
    assert info_problems([]) == ["the reply is not a JSON object"]


@pytest.mark.parametrize(
    ("change", "tier"),
    [
        ({}, "same"),
        ({"rasterizer": {"name": "other", "version": "0"}}, "different"),
        ({"rasterizer": {"name": "stub", "version": "1"}}, "different"),
        ({"render_conventions": {**CONVENTIONS, "eps2d": 0.1}}, "different"),
        ({"render_conventions": {**CONVENTIONS, "extra": 1}}, "different"),
        ({"name": "other implementation"}, "same"),
        ({"render_conventions": {**CONVENTIONS, "background": [1.0, 1.0, 1.0]}}, "same"),
    ],
)
def test_tier_is_same_only_for_the_same_rasterizer_and_conventions(
    change: dict[str, Any], tier: str
) -> None:
    """Q-206: the same-constants tier needs the same rasterizer identity and every
    render_conventions entry equal, the background aside, since it is each model's (D-84); the
    implementations' own names don't matter."""
    assert choose_tier(INFO, {**INFO, **change}) == tier
    assert choose_tier(INFO, None) == "different"


def test_relative_tolerance_judges_near_zero_entries_against_the_arrays_scale() -> None:
    """Q-208: |a − b| ≤ tol·max(|a|, |b|, 1e-3·‖b‖∞); NaN never passes."""
    b = [1000.0, 0.0, 5.0]
    # The floor is 1e-3 · 1000 = 1, so the zero entry may move by 1e-6 · 1 = 1e-6.
    assert close_relative([1000.0005, 5e-7, 5.000004], b, 1e-6).tolist() == [True, True, True]
    assert close_relative([1000.0, 2e-6, float("nan")], b, 1e-6).tolist() == [True, False, False]
    assert close_absolute([1.0, 1.1], [1.0, 1.0], 0.05).tolist() == [True, False]


def _config(tmp_path: Path, body: object = None) -> Path:
    path = tmp_path / "harness_config.json"
    path.write_text(json.dumps({"optim": {"iterations": 100}} if body is None else body))
    return path


def _run(tmp_path: Path, *extra: str, a: str = f"{STUB} A", b: str = f"{STUB} B") -> int:
    return main(
        [
            "run",
            "--a",
            a,
            "--b",
            b,
            "--config",
            str(_config(tmp_path)),
            "--out",
            str(tmp_path / "run"),
            *extra,
        ]
    )


# Private tests of the runner (`test_runner_…`): its level registry is pointed at a stand-in
# level, so they pin the run mechanics apart from any real level.
@pytest.fixture
def fake_l0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(runner._LEVEL_MODULES, "L0", "fake_level")


def test_runner_with_passing_checks_exits_0_and_writes_the_run(
    tmp_path: Path, fake_l0: None
) -> None:
    """§3.3: exit 0 when all checks passed; run.json, report.json and report.md are written,
    the info checks under the lowest level, with the tier in the run record."""
    assert _run(tmp_path, "--levels", "L0") == 0
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    ids = [c["id"] for c in report["checks"]]
    assert ids == ["L0.info.A", "L0.info.B", "L0.fake.A", "L0.fake.B", "L0.fake.inf"]
    assert report["summary"] == {"L0": "pass", "failed_checks": 0}
    assert report["run"]["tier"] == "same"
    assert report["implementations"]["A"]["name"] == "stub"
    assert (tmp_path / "run" / "report.md").read_text().startswith("# Differential")
    assert (tmp_path / "run" / "run.json").is_file()


def test_runner_with_a_failed_check_exits_1(
    tmp_path: Path, fake_l0: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.3: exit 1 when at least one check failed; a crashing call is the "crash" pattern."""
    monkeypatch.setenv("STUB_EXIT", "3")
    assert _run(tmp_path, "--levels", "L0") == 1
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    assert [c["id"] for c in failed] == ["L0.fake.A", "L0.fake.B"]
    assert all(c["pattern"] == "crash" and "exited with code 3" in c["symptom"] for c in failed)


def test_runner_fails_the_check_of_an_incomplete_info_reply(
    tmp_path: Path, fake_l0: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-73: an info reply without one of render_conventions' entries is a failed info check,
    and the tier falls back to different rasterizers."""
    info = {**INFO, "render_conventions": {k: v for k, v in CONVENTIONS.items() if k != "far"}}
    monkeypatch.setenv("STUB_INFO_B", json.dumps(info))
    assert _run(tmp_path, "--levels", "L0") == 1
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    check = next(c for c in report["checks"] if c["id"] == "L0.info.B")
    assert check["status"] == "fail" and check["pattern"] == "schema violation"
    assert "far" in check["observed"]
    assert report["run"]["tier"] == "different"


def test_runner_rerun_reproduces_one_check(
    tmp_path: Path,
    fake_l0: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-203: `rerun CHECK_ID --out RUN_DIR` replays that check's adapter calls into
    reruns/<id>/, prints it, and exits by its status."""
    assert _run(tmp_path, "--levels", "L0") == 0
    capsys.readouterr()
    run_dir = tmp_path / "run"
    assert main(["rerun", "L0.fake.B", "--out", str(run_dir)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["id"] == "L0.fake.B" and printed["status"] == "pass"
    assert (run_dir / "reruns" / "L0.fake.B" / "check.json").is_file()
    assert list((run_dir / "reruns" / "L0.fake.B" / "logs" / "B").glob("*-echo.log"))
    assert not list((run_dir / "reruns" / "L0.fake.B" / "logs" / "A").glob("*-echo.log"))
    monkeypatch.setenv("STUB_EXIT", "1")
    assert main(["rerun", "L0.fake.A", "--out", str(run_dir)]) == 1
    assert main(["rerun", "L0.info.A", "--out", str(run_dir)]) == 0
    assert main(["rerun", "L0.nothing", "--out", str(run_dir)]) == 2


@pytest.mark.parametrize(
    "case",
    ["config_not_json", "seed_in_config", "missing_adapter", "l3_without_scenes", "bad_level"],
)
def test_configuration_errors_exit_2(tmp_path: Path, case: str) -> None:
    """§3.3: exit 2 for a harness or configuration error (Q-203)."""
    levels = "L3" if case == "l3_without_scenes" else "L0"
    if case == "bad_level":
        levels = "L4"
    if case == "config_not_json":
        _config(tmp_path).write_text("{")
        args = ["--config", str(tmp_path / "harness_config.json")]
    elif case == "seed_in_config":
        args = ["--config", str(_config(tmp_path, {"seed": 3}))]
    else:
        args = ["--config", str(_config(tmp_path))]
    a = "no-such-adapter-executable" if case == "missing_adapter" else f"{STUB} A"
    code = main(
        ["run", "--a", a, "--b", f"{STUB} B", *args, "--levels", levels]
        + ["--out", str(tmp_path / "run")]
    )
    assert code == 2


def test_a_non_empty_run_directory_or_bad_settings_exit_2(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "x").write_text("")
    assert _run(tmp_path, "--levels", "L0") == 2
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"l1": {"items": 0}}))
    assert _run(tmp_path, "--levels", "L0", "--settings", str(settings)) == 2
    settings.write_text(json.dumps({"unknown": 1}))
    assert _run(tmp_path, "--levels", "L0", "--settings", str(settings)) == 2


def test_l3_without_the_lpips_extra_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-201: L3 computes LPIPS itself, so without torch and lpips it is a configuration
    error, before any adapter runs."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "lpips" else real(name, *a)
    )
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"l3": {"scenes": [{"name": "s", "manifest": str(scene), "init": str(scene)}]}})
    )
    assert _run(tmp_path, "--levels", "L3", "--settings", str(settings)) == 2
    assert not (tmp_path / "run" / "run.json").exists()
