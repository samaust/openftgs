# SPDX-License-Identifier: Apache-2.0
"""`report.json` and `report.md` (§3.3 "Report format"; Q-211)."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ftgs_harness.checks import Check, json_value, level_summary
from ftgs_harness.settings import sizes_reduced

__all__ = ["REPORT_FORMAT", "build_report", "render_markdown", "write_report"]

REPORT_FORMAT = "ftgs-harness-report"


def build_report(
    *,
    run: Mapping[str, Any],
    implementations: Mapping[str, Mapping[str, Any]],
    levels: Sequence[str],
    checks: Sequence[Check],
) -> dict[str, Any]:
    """The body of report.json: §3.3's example, with a summary for each level that ran."""
    summary: dict[str, Any] = {
        level: level_summary([c.status for c in checks if c.level == level]) for level in levels
    }
    summary["failed_checks"] = sum(c.status == "fail" for c in checks)
    return {
        "format": REPORT_FORMAT,
        "version": 1,
        "run": dict(json_value(run)),
        "implementations": dict(json_value(implementations)),
        "summary": summary,
        "checks": [c.to_json() for c in checks],
    }


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return json.dumps(value) if not isinstance(value, str) else value


def _check_block(check: Mapping[str, Any]) -> list[str]:
    lines = [f"#### `{check['id']}`", ""]
    if check.get("symptom"):
        lines += [check["symptom"], ""]
    observed = check["observed"]
    note = f" ({check['note']})" if check.get("note") else ""
    lines += [
        f"- {check['metric']}: observed {_number(observed)}{note}, expected {check['expected']}",
    ]
    if check.get("spec_refs"):
        lines.append("- spec: " + ", ".join(check["spec_refs"]))
    if check.get("localization"):
        lines.append("- localization: " + json.dumps(check["localization"]))
    if check.get("repro"):
        lines.append(f"- repro: `{check['repro']}`")
    images = [a for a in check.get("artifacts", []) if a.endswith(".png")]
    if images:
        lines += ["", " ".join(f'<img src="{a}" width="160" alt="{a}">' for a in images)]
    lines.append("")
    return lines


def _reduced_note(settings: Mapping[str, Any]) -> str:
    """What makes the run reduced, for the header: case sizes below §3.3's (Q-203), LPIPS left
    out of L2 (Q-212), or both."""
    if not settings.get("reduced"):
        return ""
    parts = []
    if sizes_reduced(settings.get("sizes", {})):
        parts.append("case sizes below §3.3's")
    if settings.get("l2_lpips") is False:
        parts.append("LPIPS left out of L2")
    return f" (reduced: {'; '.join(parts)})"


def render_markdown(report: Mapping[str, Any]) -> str:
    """report.md from report.json: the summary, failed checks first grouped by pattern, then
    the inconclusive and not-comparable checks, then the passed ones as a list (Q-211)."""
    run = report["run"]
    out = [
        "# Differential harness report",
        "",
        f"Started {run.get('started', '?')}, harness {run.get('harness_version', '?')}, "
        f"config SHA-256 `{str(run.get('config_sha256', '?'))[:12]}…`"
        + _reduced_note(run.get("settings", {})),
        "",
        "| Implementation | Name | Version |",
        "| --- | --- | --- |",
    ]
    for label, impl in report["implementations"].items():
        out.append(f"| {label} | {impl.get('name')} | {impl.get('version')} |")
    summary = report["summary"]
    out += ["", "| Level | Result |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in summary.items() if k != "failed_checks"]
    out += ["", f"Failed checks: {summary['failed_checks']}", ""]
    checks = report["checks"]
    failed = [c for c in checks if c["status"] == "fail"]
    if failed:
        out += ["## Failed checks", ""]
        patterns: dict[str, list[Mapping[str, Any]]] = {}
        for c in failed:
            patterns.setdefault(c.get("pattern") or "other", []).append(c)
        for pattern, group in patterns.items():
            out += [f"### {pattern} ({len(group)})", ""]
            for c in group:
                out += _check_block(c)
    for status, title in (("inconclusive", "Inconclusive"), ("not_comparable", "Not comparable")):
        group = [c for c in checks if c["status"] == status]
        if group:
            out += [f"## {title} checks", ""]
            for c in group:
                out += _check_block(c)
    passed = [c for c in checks if c["status"] == "pass"]
    if passed:
        out += ["## Passed checks", "", f"<details><summary>{len(passed)} checks</summary>", ""]
        out += [f"- `{c['id']}`: {c['metric']} {_number(c['observed'])}" for c in passed]
        out += ["", "</details>", ""]
    return "\n".join(out)


def write_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    """Write report.json (no NaN or Infinity, Q-211) and report.md."""
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (run_dir / "report.md").write_text(render_markdown(report) + "\n")
