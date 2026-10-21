# SPDX-License-Identifier: Apache-2.0
"""L2 · forward behaviour on random inputs (§3.3 L2; Q-207, Q-208, Q-209), with openftgs as
both implementations and with the fault-injecting wrapper as B. LPIPS is off (l2.lpips), or
left out of the losses queries by the wrapper's lpips_off, so no weights are downloaded."""

from pathlib import Path

import numpy as np
import pytest
from harness_runs import checks, faulty, openftgs, run_harness

from ftgs_harness.info import QUERY_KINDS
from ftgs_harness.level2 import TOLERANCES

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


def test_an_exact_float64_motion_passes_against_openftgs(tmp_path: Path) -> None:
    """Q-208: means_t is signed, so it is judged against the answer's largest magnitude. An
    exact float64 evaluation of E1 differs from openftgs's float32 answers by their rounding at
    that scale, and passes."""
    _, report = run_harness(
        tmp_path,
        "L2",
        a=openftgs(),
        b=faulty("exact_motion", on=("forward",)),
        settings=SETTINGS,
    )
    motion = checks(report, "L2.forward.motion.")
    assert len(motion) == 3 and {c["status"] for c in motion.values()} == {"pass"}


def test_signed_answers_zeroed_below_a_hundredth_of_their_scale_fail(tmp_path: Path) -> None:
    """Q-208: the largest-magnitude floor still sees means_t's entries below 10⁻²·M set to 0."""
    _, report = run_harness(
        tmp_path,
        "L2",
        a=openftgs(),
        b=faulty("zero_answers=0.01", on=("forward",)),
        settings=SETTINGS,
    )
    motion = checks(report, "L2.forward.motion.")
    assert len(motion) == 3 and {c["status"] for c in motion.values()} == {"fail"}


@pytest.mark.parametrize("lacking", [("A", "B"), ("B",)])
def test_ok_answers_without_a_tensor_of_their_kind_are_schema_violations(
    tmp_path: Path, lacking: tuple[str, ...]
) -> None:
    """Q-207: an ok answer carries every tensor of its kind's row. Each kind compared by
    tolerance loses one tensor, from both answers or from B's only: each of its queries fails
    as a schema violation naming the adapters and the tensor."""
    first = {kind: next(iter(names)) for kind, names in TOLERANCES.items()}
    dropping = faulty("drop_answer=" + ",".join(first.values()), on=("forward",))
    a = dropping if "A" in lacking else openftgs()
    code, report = run_harness(tmp_path, "L2", a=a, b=dropping, settings=SETTINGS)
    assert code == 1
    for kind, name in first.items():
        found = checks(report, f"L2.forward.{kind}.")
        assert len(found) == 3, kind  # two random, one edge
        for check in found.values():
            assert check["status"] == "fail" and check["pattern"] == "schema violation"
            assert check["observed"] == "; ".join(f"{label} lacks {name}" for label in lacking)
            assert check["symptom"].endswith(f"lacks 1 ({name}).")
    sampling = checks(report, "L2.forward.relocation_sampling.")
    assert sampling and all(c["status"] == "pass" for c in sampling.values())


def test_losses_with_lpips_on_fail_without_lpips_vgg(tmp_path: Path) -> None:
    """Q-212: with `lpips` true, losses answers that both lack lpips_vgg fail; with it false
    the other four tensors suffice (the lpips-off runs above). The fault answers each losses
    query as if `lpips` were false, and λ_perc = 0 keeps grad_stats off LPIPS, so no weights
    are downloaded."""
    adapter = faulty("lpips_off", on=("forward",))
    code, report = run_harness(
        tmp_path,
        "L2",
        a=adapter,
        b=adapter,
        settings={"l2": {"queries": 1, "lpips": True}},
        config={"loss": {"lambda_perc": 0.0}},
    )
    assert code == 1
    losses = checks(report, "L2.forward.losses.")
    assert len(losses) == 2  # one random, one edge
    for check in losses.values():
        assert check["status"] == "fail" and check["pattern"] == "schema violation"
        assert check["observed"] == "A lacks lpips_vgg; B lacks lpips_vgg"
    others = {k: c for k, c in checks(report, "L2.forward.").items() if k not in losses}
    assert others and all(c["status"] == "pass" for c in others.values()), others.keys()


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


def test_queries_with_a_setting_the_adapters_cannot_honor_are_not_comparable(
    tmp_path: Path,
) -> None:
    """§3.3 adapter contract: a harness_config setting the adapter cannot honor is reported as
    not comparable, never silently dropped; it fails nothing, and the other kinds are judged."""
    code, report = run_harness(
        tmp_path,
        "L2",
        a=openftgs(),
        b=openftgs(),
        settings=SETTINGS,
        config={"loss": {"not_a_field": 1.0}},
    )
    assert code == 0
    for kind in ("losses", "grad_stats"):
        found = checks(report, f"L2.forward.{kind}.")
        assert found and {c["status"] for c in found.values()} == {"not_comparable"}
        messages = next(iter(found.values()))["localization"]["adapter_messages"]
        assert all("loss.not_a_field" in m for m in messages.values())
    assert {c["status"] for c in checks(report, "L2.forward.motion.").values()} == {"pass"}
