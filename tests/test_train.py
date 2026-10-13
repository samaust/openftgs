# SPDX-License-Identifier: Apache-2.0
"""The trainer in openftgs.train: the calendar part of U-17, U-21, U-24, the smoke runs of U-26,
S-10, and the added tests of PRD #84 (§1.3, §2.2, D-17, D-18, D-26, D-27, D-45, D-46, D-71;
Q-69 to Q-75).

Expected values come from the §1.3 calendar table and its rules, from rates and scales worked
by hand for scenes whose cameras and timestamps are chosen so the answer is known, and, for the
resume and determinism tests, from a second run of the same configuration: the property U-21
and U-24 state.
"""

import pytest

from openftgs.config import GrowthConfig, RelocConfig, TrainConfig
from openftgs.train import calendar_entry

# The calendar (§1.3 Iteration calendar; U-17; Q-71).


def test_u17_growth_and_relocation_events_fall_on_the_calendar() -> None:
    """U-17: growth exactly at 500, 600, …, 15,000 (146 events) and relocation exactly at 500,
    600, …, 25,000 (246 events), with the default configuration (D-17, D-18)."""
    cfg = TrainConfig()
    entries = [calendar_entry(cfg, it) for it in range(1, 30_001)]
    grows = [it for it, e in enumerate(entries, start=1) if e.grow]
    relocates = [it for it, e in enumerate(entries, start=1) if e.relocate]
    assert grows == list(range(500, 15_001, 100)) and len(grows) == 146
    assert relocates == list(range(500, 25_001, 100)) and len(relocates) == 246
    assert [it for it, e in enumerate(entries, start=1) if e.event] == relocates
    assert not any(e.prune or e.reset_opacity for e in entries)


def test_u17_sh_bands_activate_one_per_thousand_iterations() -> None:
    """U-17, E13: L_active steps at 1,000/1,001, 2,000/2,001 and 3,000/3,001, and stops at L
    (D-3)."""
    cfg = TrainConfig()
    steps = {1: 0, 1000: 0, 1001: 1, 2000: 1, 2001: 2, 3000: 2, 3001: 3, 30_000: 3}
    assert {it: calendar_entry(cfg, it).sh_degree for it in steps} == steps
    low = TrainConfig(sh_degree=1)
    assert [calendar_entry(low, it).sh_degree for it in (1000, 1001, 3001, 30_000)] == [0, 1, 1, 1]


def test_ablation_prunes_at_growth_events_and_resets_opacity_in_the_growth_window() -> None:
    """§1.3 Count-control modes, the "w/o periodic relocation" ablation (Q-71, owner answer):
    with relocation off and a pruning opacity, pruning runs at every growth event and the
    opacity reset at every multiple of 3,000 inside the growth window: 3,000 … 15,000."""
    cfg = TrainConfig(reloc=RelocConfig(enabled=False, prune_opacity=0.005))
    entries = {it: calendar_entry(cfg, it) for it in range(1, 30_001)}
    assert [it for it, e in entries.items() if e.prune] == list(range(500, 15_001, 100))
    assert [it for it, e in entries.items() if e.reset_opacity] == [
        3000,
        6000,
        9000,
        12_000,
        15_000,
    ]
    assert not any(e.relocate for e in entries.values())
    assert [it for it, e in entries.items() if e.event] == list(range(500, 15_001, 100))


def test_calendar_windows_and_period_come_from_the_configuration() -> None:
    """Q-71: growth and relocation share the event period and keep their own windows; growth
    mode "none" grows nothing. U-21's test calendar: events every 50 iterations from 100, growth
    up to 200, relocation up to 250."""
    cfg = TrainConfig(
        growth=GrowthConfig(start=100, stop=200),
        reloc=RelocConfig(every=50, start=100, stop=250),
    )
    entries = {it: calendar_entry(cfg, it) for it in range(1, 30_001)}
    assert [it for it, e in entries.items() if e.grow] == [100, 150, 200]
    assert [it for it, e in entries.items() if e.relocate] == [100, 150, 200, 250]
    none = TrainConfig(growth=GrowthConfig(mode="none"))
    assert not any(calendar_entry(none, it).grow for it in range(1, 30_001))


@pytest.mark.parametrize("it", [0, 30_001, True, 5.0])
def test_calendar_rejects_iterations_outside_the_run(it: object) -> None:
    """D-41: it counts completed steps from 1 to T_it (Q-71)."""
    with pytest.raises(ValueError, match="it"):
        calendar_entry(TrainConfig(), it)  # type: ignore[arg-type]
