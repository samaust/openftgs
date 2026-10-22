# SPDX-License-Identifier: Apache-2.0
"""`scripts/reserve_trace.py`'s summary of the allocator over a first epoch (#350; Q-238, Q-232,
Q-225).

Added tests, not §3 rows, on synthetic readings: k is the iteration at which the last training
size first appears and the decision is read after k + 1; the steady reserve is the median over
the later iterations, and the smallest of them is reported; the first-at-a-size iterations that
returned blocks to the driver are counted against the count before iteration 1; the verdict
turns at 90 % of the steady reserve; a trace that misses a size or stops at k + 1 is refused.
The trace itself needs CUDA and a scene, and is a recorded run (Calibration data).
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
GB = 10**9


def _module() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("reserve_trace")


def _rows(
    reserves: list[float], firsts: set[int], frees: list[int], allocated: float = 1.0
) -> list[dict[str, Any]]:
    """Iterations 1, 2, … with the given reserves (GB), first-at-a-size iterations and
    cumulative device frees."""
    return [
        {
            "iteration": it,
            "first_at_size": it in firsts,
            "free_bytes": int(20 * GB),
            "reserved_bytes": int((allocated + reserve) * GB),
            "allocated_bytes": int(allocated * GB),
            "num_device_free": count,
        }
        for it, (reserve, count) in enumerate(zip(reserves, frees, strict=True), start=1)
    ]


def test_summary_reads_the_decision_after_the_last_new_size() -> None:
    """Sizes first come at iterations 1, 2 and 4, so k = 4 and the decision is read after 5.
    The iterations after it hold 5.0, 6.0, 4.0 and 5.5 GB: steady 5.25, smallest 4.0. Iterations
    1 and 4 raised the device frees (1 from the count before it), 3 did without a new size."""
    rows = _rows(
        [0.5, 1.0, 3.0, 0.8, 5.0, 5.0, 6.0, 4.0, 5.5],
        firsts={1, 2, 4},
        frees=[10, 10, 12, 15, 15, 15, 15, 15, 15],
    )
    out = _module().summary(rows, training_sizes=3, frees_before=7)
    assert out["k"] == 4
    assert out["decision_iteration"] == 5
    assert out["decision_reserve_bytes"] == 5 * GB
    assert out["decision_allocated_bytes"] == 1 * GB
    assert out["decision_reserved_bytes"] == 6 * GB
    assert out["steady_reserve_bytes"] == pytest.approx(5.25 * GB)
    assert out["smallest_later_reserve_bytes"] == 4 * GB
    assert out["later_iterations"] == 4
    assert out["firsts_returning_blocks"] == 2
    assert out["others_returning_blocks"] == 1
    assert out["ratio"] == pytest.approx(5.0 / 5.25)
    assert out["passes"] is True


def test_summary_fails_below_ninety_percent_of_the_steady_reserve() -> None:
    """A decision reserve of 0.89 of the steady reserve fails, 0.90 passes."""
    module = _module()
    for decision, passes in ((8.9, False), (9.0, True)):
        rows = _rows([0.5, decision, 10.0, 10.0, 10.0], firsts={1}, frees=[1] * 5)
        out = module.summary(rows, training_sizes=1, frees_before=0)
        assert out["k"] == 1 and out["decision_iteration"] == 2
        assert out["ratio"] == pytest.approx(decision / 10.0)
        assert out["passes"] is passes
        assert out["firsts_returning_blocks"] == 1


def test_summary_refuses_a_trace_that_misses_a_size_or_stops_at_the_decision() -> None:
    module = _module()
    rows = _rows([1.0, 2.0, 3.0], firsts={1, 2}, frees=[0, 0, 0])
    with pytest.raises(ValueError, match="reaches 2 of the 3"):
        module.summary(rows, training_sizes=3, frees_before=0)
    with pytest.raises(ValueError, match="before an iteration after k"):
        module.summary(rows[:3], training_sizes=2, frees_before=0)
