# SPDX-License-Identifier: Apache-2.0
"""The trainer in openftgs.train: the calendar part of U-17, U-21, U-24, the smoke runs of U-26,
S-10, and the added tests of PRD #84 (§1.3, §2.2, D-17, D-18, D-26, D-27, D-45, D-46, D-71;
Q-69 to Q-75).

Expected values come from the §1.3 calendar table and its rules, from rates and scales worked
by hand for scenes whose cameras and timestamps are chosen so the answer is known, and, for the
resume and determinism tests, from a second run of the same configuration: the property U-21
and U-24 state.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from builders import UNIT_TIME, pinhole, random_model, ring, write_scene

from openftgs.config import (
    GrowthConfig,
    LossConfig,
    OptimConfig,
    PerfConfig,
    RelocConfig,
    TrainConfig,
)
from openftgs.data import MultiViewVideo
from openftgs.train import StepLog, Trainer, calendar_entry
from openftgs.types import Camera, InitPoints, TimeNormalizer

F64 = torch.float64
TAU1 = 4.0 / 30.0  # five frames at 30 FPS
TIME = TimeNormalizer(0.0, TAU1)


def _scene(root: Path, *, frames: int = 5) -> MultiViewVideo:
    """A ring of four 32×24 cameras around a random ground-truth model: three train at frames
    k/30, the fourth is the test view."""
    cams = ring(4, radius=4.0, elevation=1.0, width_px=32, height_px=24, focal=30.0)
    gt = random_model(24, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    views = [(cam, k / 30.0, "train") for k in range(frames) for cam in cams[:3]]
    views += [(cams[3], k / 30.0, "test") for k in range(frames)]
    return MultiViewVideo.from_manifest(write_scene(root, gt, views))


def _init_points(n: int = 30, seed: int = 1) -> InitPoints:
    g = torch.Generator().manual_seed(seed)
    return InitPoints(
        positions=(torch.rand(n, 3, generator=g) - 0.5) * 1.2,
        colors=torch.full((n, 3), 0.5),
        timestamps=torch.rand(n, generator=g, dtype=F64) * TAU1,
    )


def _no_perceptual(**fields: object) -> TrainConfig:
    """A configuration without the perceptual loss, so CPU tests download nothing."""
    return TrainConfig(loss=LossConfig(lambda_perc=0.0), **fields)  # type: ignore[arg-type]


def _run(trainer: Trainer, until: int | None = None) -> list[StepLog]:
    logs: list[StepLog] = []
    trainer.train(until, callbacks=[lambda _, log: logs.append(log)])
    return logs


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


# Smoke runs (U-26).


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_smoke_run_for_each_sh_degree(tmp_path: Path, sh_degree: int) -> None:
    """U-26: a 50-iteration run for each SH degree finishes without NaN, with shN of shape
    [N, (L + 1)² − 1, 3]; at L = 0, shN has no parameter group, so no rate (D-3). With one band
    per 10 iterations, every band activates (E13)."""
    video = _scene(tmp_path)
    cfg = _no_perceptual(sh_degree=sh_degree, optim=OptimConfig(iterations=50, sh_interval=10))
    trainer = Trainer(cfg, video, _init_points(), device="cpu")
    logs = _run(trainer)
    assert trainer.iteration == 50 and [log.iteration for log in logs] == list(range(1, 51))
    assert all(math.isfinite(float(log.losses["total"])) for log in logs)
    assert [log.sh_degree for log in logs] == [
        min((it - 1) // 10, sh_degree) for it in range(1, 51)
    ]
    params = trainer.state_dict().params
    n = params["means"].shape[0]
    assert params["shN"].shape == (n, (sh_degree + 1) ** 2 - 1, 3)
    assert ("shN" in logs[0].rates) == (sh_degree > 0)


# What the run is built from (D-26, D-27, E14, D-71; Q-65, Q-69, Q-73).


def _axis_camera(center: tuple[float, float, float], name: str) -> Camera:
    """A camera with the identity rotation at `center`, looking along +z: its center
    −R_wᵀt_w is `center` exactly."""
    x, y, z = center
    w2c = [[1.0, 0.0, 0.0, -x], [0.0, 1.0, 0.0, -y], [0.0, 0.0, 1.0, -z]]
    return pinhole(32, 24, fx=30.0, fy=30.0, cx=16.0, cy=12.0, world_to_camera=w2c, name=name)


def _axis_scene(root: Path, timestamps: Sequence[float]) -> MultiViewVideo:
    """Four training cameras at (±2, 0, −5) and (0, ±2, −5), 2 from their centroid, and a test
    camera far away, at the given raw timestamps."""
    centers = [(2.0, 0.0, -5.0), (-2.0, 0.0, -5.0), (0.0, 2.0, -5.0), (0.0, -2.0, -5.0)]
    cams = [_axis_camera(c, f"cam{k}") for k, c in enumerate(centers)]
    test = _axis_camera((40.0, 0.0, -5.0), "far")
    # The ground truth keeps its own time normalizer: a single timestamp holds no motion (D-71),
    # but the ground truth is only rendered at the scene's raw timestamps.
    gt = random_model(16, seed=0, sh_degree=1, time=UNIT_TIME, spread=0.6, log_scale=(-2.5, -1.5))
    views = [(cam, tau, "train") for tau in timestamps for cam in cams]
    views += [(test, tau, "test") for tau in timestamps]
    return MultiViewVideo.from_manifest(write_scene(root, gt, views))


def test_spatial_scale_and_frame_interval_reach_the_rates(tmp_path: Path) -> None:
    """§1.3 rate table, D-19, D-27, E14: S = 1.1 × 2 from the training cameras alone and, at
    frames k/30 for k = 0 … 4, Δt_f = 1/4, so at it = 1 the means rate is 1.6e-4·S and the
    velocity rate 1.6e-4·S/Δt_f, worked by hand."""
    video = _axis_scene(tmp_path, [k / 30.0 for k in range(5)])
    log = Trainer(_no_perceptual(), video, _init_points(), device="cpu").step()
    s = 1.1 * 2.0
    assert log.rates["means"] == pytest.approx(1.6e-4 * s, rel=1e-12)
    assert log.rates["velocities"] == pytest.approx(1.6e-4 * s * 4.0, rel=1e-12)
    assert log.rates["times"] == pytest.approx(1.6e-4, rel=1e-12)


def test_single_timestamp_takes_no_time_or_velocity_updates(tmp_path: Path) -> None:
    """D-71, §1.4 "Time": with one timestamp, Δt_f = 1 and time centers and velocities have no
    parameter group, so no rate."""
    video = _axis_scene(tmp_path, [0.25])
    init = InitPoints(
        positions=(torch.rand(20, 3, generator=torch.Generator().manual_seed(1)) - 0.5),
        colors=torch.full((20, 3), 0.5),
        timestamps=torch.full((20,), 0.25, dtype=F64),
    )
    log = Trainer(_no_perceptual(), video, init, device="cpu").step()
    assert "times" not in log.rates and "velocities" not in log.rates
    assert log.t == 0.0 and math.isfinite(float(log.losses["total"]))


def test_each_epoch_takes_every_training_image_once(tmp_path: Path) -> None:
    """D-26: a seeded permutation of all training images, redrawn each epoch; test images are
    never drawn. Another seed gives another order (Q-72)."""
    video = _scene(tmp_path)
    training = sorted(i for i, r in enumerate(video.records) if r.split == "train")
    order = [
        log.image_index
        for log in _run(
            Trainer(_no_perceptual(), video, _init_points(), device="cpu"), 3 * len(training)
        )
    ]
    epochs = [order[k * len(training) : (k + 1) * len(training)] for k in range(3)]
    assert all(sorted(epoch) == training for epoch in epochs)
    assert epochs[0] != epochs[1]
    other = Trainer(_no_perceptual(seed=1), video, _init_points(), device="cpu")
    assert [log.image_index for log in _run(other, len(training))] != epochs[0]


def test_a_model_must_belong_to_the_run(tmp_path: Path) -> None:
    """Q-69: a model passed in must have the configuration's SH degree, the dataset's time
    normalizer and time unit, and at most N_max primitives."""
    video = _scene(tmp_path)
    cases = {
        "SH degree": random_model(8, sh_degree=2, time=video.time),
        "time": random_model(8, sh_degree=3, time=TimeNormalizer(0.0, 1.0)),
        "N_max": random_model(8, sh_degree=3, time=video.time),
    }
    unit = random_model(8, sh_degree=3, time=video.time)
    unit.time_unit = "frame"
    for match, model in cases.items():
        cfg = _no_perceptual(growth=GrowthConfig(n_max=4 if match == "N_max" else 3_000_000))
        with pytest.raises(ValueError, match=match):
            Trainer(cfg, video, model, device="cpu")
    with pytest.raises(ValueError, match="time unit"):
        Trainer(_no_perceptual(), video, unit, device="cpu")


@pytest.mark.parametrize(
    "perf",
    [
        PerfConfig(perc_precision="bf16"),
        PerfConfig(images_per_step=2),
        PerfConfig(frame_cache="gpu"),
        PerfConfig(frame_cache="mmap"),
    ],
)
def test_options_of_later_milestones_are_refused(tmp_path: Path, perf: PerfConfig) -> None:
    """Q-73: bf16 LPIPS and several images per step (M15), and the GPU or memory-mapped frame
    cache (M12), fail at construction rather than being ignored."""
    video = _scene(tmp_path)
    with pytest.raises(NotImplementedError):
        Trainer(_no_perceptual(perf=perf), video, _init_points(), device="cpu")
