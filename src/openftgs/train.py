# SPDX-License-Identifier: Apache-2.0
"""Trainer: the iteration loop, the sampler, the schedules, and the evaluation and checkpoint
hooks.

Spec: §1.3. Band: Orchestration (§2.1).
"""

from dataclasses import dataclass

from openftgs.config import TrainConfig
from openftgs.density import OPACITY_RESET_EVERY

__all__ = ["CalendarEntry", "calendar_entry"]


# Q-71: what the calendar (§1.3 Iteration calendar) holds for one iteration.
@dataclass(frozen=True)
class CalendarEntry:
    sh_degree: int  # L_active (E13)
    grow: bool  # a growth event (D-18)
    relocate: bool  # a relocation event (D-17)
    prune: bool  # the ablation's pruning (§1.3 Count-control modes)
    reset_opacity: bool  # the ablation's opacity reset (§1.3 Count-control modes)

    @property
    def event(self) -> bool:
        """An event whose gradient statistics are reset afterwards (§1.3 step 7, D-45)."""
        return self.grow or self.relocate or self.prune


def calendar_entry(cfg: TrainConfig, it: int) -> CalendarEntry:
    """The calendar of `cfg` at iteration `it` (§1.3 step 7 and Iteration calendar, E13): the
    active SH degree and the events, every window and period read from the configuration."""
    iterations = cfg.optim.iterations
    if isinstance(it, bool) or not isinstance(it, int) or not 1 <= it <= iterations:
        raise ValueError(f"it must be an int in [1, {iterations}], got {it!r}")
    growth, reloc = cfg.growth, cfg.reloc
    on_period = it % reloc.every == 0  # growth events share the relocation period (§2.1)
    in_growth = growth.start <= it <= growth.stop
    ablation = reloc.prune_opacity is not None  # Q-71: the ablation's densification
    return CalendarEntry(
        sh_degree=min((it - 1) // cfg.optim.sh_interval, cfg.sh_degree),  # E13
        grow=on_period and in_growth and growth.mode != "none",
        relocate=on_period and reloc.enabled and reloc.start <= it <= reloc.stop,
        prune=on_period and in_growth and ablation,
        # Q-71 (owner answer): inside the growth window only; Q-67: the period from density.
        reset_opacity=ablation and in_growth and it % OPACITY_RESET_EVERY == 0,
    )
