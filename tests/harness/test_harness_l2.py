# SPDX-License-Identifier: Apache-2.0
"""L2 · forward behaviour on random inputs (§3.3 L2; Q-207, Q-208, Q-209), with openftgs as
both implementations and with the fault-injecting wrapper as B. LPIPS is off (l2.lpips), so no
weights are downloaded."""

from pathlib import Path

import numpy as np
import pytest
from harness_runs import checks, faulty, openftgs, run_harness

from ftgs_harness.info import QUERY_KINDS

SETTINGS = {"l2": {"queries": 2, "lpips": False}}


def test_l2_passes_with_openftgs_as_both_implementations(tmp_path: Path) -> None:
    """Every kind, random and edge queries, at §3.3's tolerances."""
    code, report = run_harness(tmp_path, "L2", a=openftgs(), b=openftgs(), settings=SETTINGS)
    assert code == 0, [c["id"] for c in report["checks"] if c["status"] != "pass"]
    kinds = {c["id"].split(".")[2] for c in checks(report, "L2.forward.").values()}
    assert kinds == set(QUERY_KINDS)
    assert len(checks(report, "L2.forward.motion.")) == 3  # two random, one edge
    # Q-212: with LPIPS off the run is reduced, and report.md's header says why.
    assert report["run"]["settings"]["reduced"] is True
    md = (tmp_path / "run" / "report.md").read_text().splitlines()
    assert "LPIPS left out of L2" in next(line for line in md if line.startswith("Started "))


@pytest.mark.parametrize(("scale", "fails"), [(2e-6, True), (5e-7, False)])
def test_answers_off_by_twice_the_tolerance_fail_and_by_half_pass(
    tmp_path: Path, scale: float, fails: bool
) -> None:
    """Q-208: motion's answers × (1 + 2·10⁻⁶) exceed its 1e-6 relative tolerance; × (1 + 5·10⁻⁷)
    stay within it."""
    _, report = run_harness(
        tmp_path,
        "L2",
        a=openftgs(),
        b=faulty(f"answers={scale}", on=("forward",)),
        settings=SETTINGS,
    )
    motion = checks(report, "L2.forward.motion.")
    assert {c["status"] for c in motion.values()} == ({"fail"} if fails else {"pass"})
    if fails:
        symptom = next(iter(motion.values()))["symptom"]
        assert "means_t" in symptom and "1e-06 relative" in symptom


def test_a_biased_sampler_fails_the_chi2_test(tmp_path: Path) -> None:
    """§3.3: relocation_sampling's counts against the scores, p > 0.001."""
    code, report = run_harness(
        tmp_path, "L2", a=openftgs(), b=faulty("biased_sampler", on=("forward",)), settings=SETTINGS
    )
    assert code == 1
    sampling = checks(report, "L2.forward.relocation_sampling.")
    assert sampling and all(c["status"] == "fail" for c in sampling.values())
    assert all(
        c["localization"]["p_A"] > 0.001 > c["localization"]["p_B"] for c in sampling.values()
    )


def test_a_kind_missing_from_info_is_not_comparable(tmp_path: Path) -> None:
    """§3.3: "A query kind missing from an adapter's info reply is reported as not comparable",
    and fails nothing."""
    code, report = run_harness(
        tmp_path, "L2", a=openftgs(), b=faulty("no_kind=motion", on=("info",)), settings=SETTINGS
    )
    assert code == 0
    motion = checks(report)["L2.forward.motion"]
    assert motion["status"] == "not_comparable" and "B" in motion["observed"]
    assert report["summary"]["L2"] == "pass"


def test_the_growth_edge_query_puts_the_cap_inside_a_tie() -> None:
    """D-44: "ties to the lower row index" decides only when the cap falls inside a group of
    equal g, so the edge query has seven tied rows at the top and room for three."""
    from ftgs_harness.queries import generate

    edge = generate("growth_decisions", 0, seed=0, config={})[-1]
    g = edge.tensors["g"]
    top = np.flatnonzero(g == g.max())
    room = edge.params["n_max"] - g.size
    assert top.size == 7 and room == 3
