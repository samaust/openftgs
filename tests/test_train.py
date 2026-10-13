# SPDX-License-Identifier: Apache-2.0
"""The trainer in openftgs.train: the calendar part of U-17, U-21, U-24, the smoke runs of U-26,
S-10, and the added tests of PRD #84 (§1.3, §2.2, D-17, D-18, D-26, D-27, D-45, D-46, D-71;
Q-69 to Q-75).

Expected values come from the §1.3 calendar table and its rules, from rates and scales worked
by hand for scenes whose cameras and timestamps are chosen so the answer is known, and, for the
resume and determinism tests, from a second run of the same configuration: the property U-21
and U-24 state.
"""

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from builders import (
    UNIT_TIME,
    logit_of,
    model_from_rows,
    pinhole,
    random_model,
    ring,
    shortened_config,
    write_scene,
)

from openftgs.config import (
    GrowthConfig,
    InitConfig,
    LossConfig,
    OptimConfig,
    PerfConfig,
    RelocConfig,
    TrainConfig,
)
from openftgs.data import MultiViewVideo
from openftgs.eval import METRIC_NAMES
from openftgs.io import load_checkpoint, save_checkpoint
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.train import StepLog, Trainer, calendar_entry
from openftgs.types import Camera, InitPoints, TimeNormalizer, TrainerState

F64 = torch.float64
TAU1 = 4.0 / 30.0  # five frames at 30 FPS
TIME = TimeNormalizer(0.0, TAU1)


def _scene(root: Path, *, frames: int = 5) -> MultiViewVideo:
    """A ring of four 32×24 cameras around a random ground-truth model: three train at frames
    k/30, the fourth is the test view."""
    cams = ring(4, radius=4.0, elevation=1.0, width_px=32, height_px=24, focal=30.0)
    gt = random_model(24, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images = [(cam, k / 30.0, "train") for k in range(frames) for cam in cams[:3]]
    images += [(cams[3], k / 30.0, "test") for k in range(frames)]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images))


def _init_points(n: int = 30, seed: int = 1, *, extent: float = 1.2) -> InitPoints:
    g = torch.Generator().manual_seed(seed)
    return InitPoints(
        positions=(torch.rand(n, 3, generator=g) - 0.5) * extent,
        colors=torch.full((n, 3), 0.5),
        timestamps=torch.rand(n, generator=g, dtype=F64) * TAU1,
    )


def _no_perceptual(**fields: object) -> TrainConfig:
    """A configuration without the perceptual loss, so CPU tests download nothing."""
    return TrainConfig(loss=LossConfig(lambda_perc=0.0), **fields)  # type: ignore[arg-type]


def _all_finite(logs: Sequence[StepLog]) -> bool:
    """No loss term of any iteration is NaN or infinite."""
    return all(math.isfinite(float(value)) for log in logs for value in log.losses.values())


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
    degrees = {1: 0, 1000: 0, 1001: 1, 2000: 1, 2001: 2, 3000: 2, 3001: 3, 30_000: 3}
    assert {it: calendar_entry(cfg, it).sh_degree for it in degrees} == degrees
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
    assert _all_finite(logs)
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
    view far away, at the given raw timestamps."""
    centers = [(2.0, 0.0, -5.0), (-2.0, 0.0, -5.0), (0.0, 2.0, -5.0), (0.0, -2.0, -5.0)]
    cams = [_axis_camera(c, f"cam{k}") for k, c in enumerate(centers)]
    test = _axis_camera((40.0, 0.0, -5.0), "far")
    # The ground truth keeps its own time normalizer: a single timestamp holds no motion (D-71),
    # but the ground truth is only rendered at the scene's raw timestamps.
    gt = random_model(16, seed=0, sh_degree=1, time=UNIT_TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images = [(cam, tau, "train") for tau in timestamps for cam in cams]
    images += [(test, tau, "test") for tau in timestamps]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images))


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


def test_the_ablation_turns_relocation_off(tmp_path: Path) -> None:
    """§1.3 Count-control modes, Q-71: a pruning opacity selects the "w/o periodic relocation"
    ablation, so with relocation on the configuration is refused rather than run in a mode the
    spec doesn't define (from the M10 milestone review)."""
    cfg = _no_perceptual(reloc=RelocConfig(prune_opacity=0.005))
    with pytest.raises(ValueError, match="prune_opacity"):
        Trainer(cfg, _scene(tmp_path), _init_points(), device="cpu")


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


# Checkpoints and exact resume (U-21), determinism (U-24), stability (S-10).

# U-21's test calendar: events every 50 iterations from 100, growth up to 200, relocation up to
# 250, so events fall on both sides of a resume at 150. A low gradient threshold makes every
# event grow, and a dead threshold of 0.2 leaves dead rows at every event (Q-11 ranges hold).
U21 = TrainConfig(
    sh_degree=1,
    loss=LossConfig(lambda_perc=0.0),
    growth=GrowthConfig(start=100, stop=200, grad_threshold=1e-7, n_max=400),
    reloc=RelocConfig(every=50, start=100, stop=250, dead_opacity=0.2),
    optim=OptimConfig(iterations=300, sh_interval=50),
    perf=PerfConfig(log_every=50),
)


def _u21_trainer(video: MultiViewVideo, cfg: TrainConfig = U21) -> Trainer:
    model = random_model(
        40,
        seed=2,
        sh_degree=1,
        time=video.time,
        spread=0.6,
        log_scale=(-2.5, -1.5),
        logit=(-3.0, 2.0),
    )
    return Trainer(cfg, video, model, device="cpu")


def _assert_same(a: object, b: object, path: str = "state") -> None:
    """Bitwise equality of nested dicts, lists and tensors."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype, path
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list | tuple):
        assert isinstance(b, list | tuple) and len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}[{k}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_states_equal(a: TrainerState, b: TrainerState, *, global_rng: bool = True) -> None:
    for name in ("iteration", "config", "time", "spatial_scale", "frame_dt", "params", "optim"):
        _assert_same(getattr(a, name), getattr(b, name), name)
    for name in ("stats", "sampler", "scene_sha256", "time_unit"):
        _assert_same(getattr(a, name), getattr(b, name), name)
    roles = set(a.rng) if global_rng else {"sampler", "relocation"}
    _assert_same({k: a.rng[k] for k in roles}, {k: b.rng[k] for k in roles}, "rng")


def _assert_logs_equal(a: Sequence[StepLog], b: Sequence[StepLog]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert (x.iteration, x.image_index, x.timestamp, x.t, x.sh_degree) == (
            y.iteration,
            y.image_index,
            y.timestamp,
            y.t,
            y.sh_degree,
        )
        assert x.rates == y.rates and x.mean_losses == y.mean_losses, x.iteration
        assert (x.num_gaussians, x.growth, x.relocation, x.pruned, x.opacity_reset) == (
            y.num_gaussians,
            y.growth,
            y.relocation,
            y.pruned,
            y.opacity_reset,
        ), x.iteration
        _assert_same(x.losses, y.losses, f"losses at {x.iteration}")


def _events_did_work(logs: Sequence[StepLog]) -> bool:
    grew = any(log.growth is not None and log.growth.cloned + log.growth.split > 0 for log in logs)
    moved = any(log.relocation is not None and log.relocation.dead > 0 for log in logs)
    return grew and moved


def test_u21_checkpoint_round_trip_keeps_every_part_of_the_state(tmp_path: Path) -> None:
    """U-21: every tensor, Adam moment, step counter, generator state, sampler state and
    statistic survives a save and a load bitwise, and a trainer resumed from the checkpoint
    holds the same state (§2.2; Q-25, Q-72)."""
    video = _scene(tmp_path / "scene")
    trainer = _u21_trainer(video)
    _run(trainer, 120)  # after an event, in the middle of an epoch
    state = trainer.state_dict()
    save_checkpoint(state, tmp_path / "ckpt")
    _assert_states_equal(state, load_checkpoint(tmp_path / "ckpt"))
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    assert resumed.iteration == 120
    _assert_states_equal(state, resumed.state_dict())


def test_u21_resumed_run_is_bitwise_equal_to_the_straight_run(tmp_path: Path) -> None:
    """U-21 (reference backend, CPU): 300 iterations straight against 150, a resume through a
    checkpoint directory, and 150 more give the same images, rates, loss terms, event reports
    and final state, bitwise. Events grow and relocate rows on both sides of the resume."""
    video = _scene(tmp_path / "scene")
    straight_trainer = _u21_trainer(video)
    straight = _run(straight_trainer)
    first = _u21_trainer(video)
    before = _run(first, 150)
    save_checkpoint(first.state_dict(), tmp_path / "ckpt")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    after = _run(resumed)
    assert _events_did_work(before) and _events_did_work(after)
    _assert_logs_equal(straight, before + after)
    _assert_states_equal(straight_trainer.state_dict(), resumed.state_dict(), global_rng=False)


def test_u24_two_seeded_runs_are_bitwise_identical(tmp_path: Path) -> None:
    """U-24: two 100-iteration runs from one configuration and seed, with events, give the same
    step logs and final state, bitwise; neither seeds the global generators (Q-72)."""
    video = _scene(tmp_path)
    cfg = dataclasses.replace(
        U21,
        growth=dataclasses.replace(U21.growth, start=20, stop=60),
        reloc=dataclasses.replace(U21.reloc, every=20, start=20, stop=80),
        optim=dataclasses.replace(U21.optim, iterations=100, sh_interval=20),
        perf=dataclasses.replace(U21.perf, log_every=20),
    )
    a, b = _u21_trainer(video, cfg), _u21_trainer(video, cfg)
    logs_a, logs_b = _run(a), _run(b)
    assert _events_did_work(logs_a)
    _assert_logs_equal(logs_a, logs_b)
    _assert_states_equal(a.state_dict(), b.state_dict(), global_rng=False)


def _shared_setup_scene(root: Path) -> MultiViewVideo:
    """§3.2 shared setup: eight 64×64 ring cameras looking at the origin, a black background
    and S ≈ 1 (the ring's radius 0.9 is each center's distance from their centroid, so
    S = 1.1 × 0.9), over five frames."""
    cams = ring(8, radius=0.9, elevation=0.3, width_px=64, height_px=64, focal=64.0)
    gt = random_model(24, seed=0, sh_degree=1, time=TIME, spread=0.15, log_scale=(-4.0, -3.0))
    images = [(cam, k / 30.0, "train") for k in range(5) for cam in cams]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images))


def test_s10_tiny_scene_trains_stably_and_resumes(tmp_path: Path) -> None:
    """S-10: a tiny scene of §3.2's shared setup, 200 iterations on the CPU on its shortened
    calendar (Q-75): the mean loss of the last 20 iterations is below that of the first 20, no
    loss term is NaN, and a checkpoint at 100 resumes to 200. N never decreases and stops
    growing after the growth window (§1.1, D-18, U-25)."""
    video = _shared_setup_scene(tmp_path / "scene")
    assert video.spatial_scale(1.1) == pytest.approx(0.99, rel=1e-5)
    base = _no_perceptual(init=InitConfig(points_per_frame=10), growth=GrowthConfig(n_max=150))
    cfg = shortened_config(200, base)
    trainer = Trainer(cfg, video, _init_points(extent=0.3), device="cpu")
    logs = _run(trainer, 100)
    save_checkpoint(trainer.state_dict(), tmp_path / "ckpt")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    logs += _run(resumed)
    totals = [float(log.losses["total"]) for log in logs]
    assert [log.iteration for log in logs] == list(range(1, 201))
    assert _all_finite(logs)
    assert sum(totals[-20:]) < sum(totals[:20])
    counts = [log.num_gaussians for log in logs]
    assert counts == sorted(counts)
    after = [log for log in logs if log.iteration > cfg.growth.stop]
    assert all(log.growth is None and log.num_gaussians == counts[-1] for log in after)


def test_the_initial_subset_is_seeded(tmp_path: Path) -> None:
    """§1.3 step 7, D-24, D-74, Q-72: init points beyond F·B are cut to a seeded uniform subset
    drawn from the run's initialization generator: the same seed keeps the same points,
    another seed others. Five frames at B = 4 keep 20 of 100 points."""
    video = _scene(tmp_path)

    def means(seed: int) -> torch.Tensor:
        cfg = _no_perceptual(seed=seed, init=InitConfig(points_per_frame=4))
        trainer = Trainer(cfg, video, _init_points(100), device="cpu")
        return trainer.state_dict().params["means"]

    first, again, other = means(0), means(0), means(1)
    assert first.shape[0] == 20
    assert torch.equal(first, again) and not torch.equal(first, other)


# Events (§1.3 step 7, D-45; Q-51, Q-71).

_EVERY_ITERATION = dict(
    growth=GrowthConfig(start=1, stop=1, grad_threshold=1e-12, split_scale=10.0),
    reloc=RelocConfig(every=1, start=1, stop=1),
    optim=OptimConfig(iterations=2),
)


def _one_live_row(video: MultiViewVideo) -> FreeTimeGaussians:
    """One live primitive at the origin, in every camera's view, and ten primitives at
    σ = 0.001 far above the scene, in no camera's view, so they are never growth candidates."""
    rows = [{"mean": (0.0, 0.0, 0.0), "scale": 0.2, "logit": 1.0, "sh0": (0.5, 0.5, 0.5)}]
    rows += [{"mean": (0.1 * k, 0.0, 40.0), "logit": logit_of(0.001)} for k in range(10)]
    return model_from_rows(rows, sh_degree=3, time=video.time)


def test_growth_runs_before_relocation_in_an_event(tmp_path: Path) -> None:
    """§1.3 step 7, D-45, U-25: growth runs before relocation in an event, so the relocation of
    the same event can draw the rows growth added. The live row is cloned (its scale is below
    the split boundary), and relocation's ten draws fall on both copies: two targets, which only
    growth-then-relocation allows."""
    video = _scene(tmp_path)
    trainer = Trainer(_no_perceptual(**_EVERY_ITERATION), video, _one_live_row(video), device="cpu")
    log = trainer.step()
    assert log.growth is not None and (log.growth.cloned, log.growth.split) == (1, 0)
    assert log.relocation is not None
    assert (log.relocation.dead, log.relocation.targets) == (10, 2)
    assert log.num_gaussians == 12


def test_trainer_resets_the_statistics_when_relocation_is_off(tmp_path: Path) -> None:
    """D-45, Q-51: an event ends with the statistics reset; relocation resets them itself, and
    with relocation off the trainer does."""
    video = _scene(tmp_path)
    cfg = _no_perceptual(**{**_EVERY_ITERATION, "reloc": RelocConfig(enabled=False, every=1)})
    trainer = Trainer(cfg, video, _one_live_row(video), device="cpu")
    log = trainer.step()
    assert log.growth is not None and log.growth.cloned == 1 and log.relocation is None
    stats = trainer.state_dict().stats
    assert not stats["g_sum"].any() and not stats["count"].any()


def test_divergence_raises_floating_point_error(tmp_path: Path) -> None:
    """Q-69: a model that fails §1.1's validity rules at a log read is a diverged run,
    reported as FloatingPointError (exit code 3 in M14), not as an input error. The model passed
    in is the one the trainer trains, so a NaN written into it is seen."""
    video = _scene(tmp_path)
    model = random_model(20, sh_degree=3, time=video.time, spread=0.6)
    trainer = Trainer(_no_perceptual(perf=PerfConfig(log_every=1)), video, model, device="cpu")
    assert trainer.step().mean_losses is not None
    with torch.no_grad():
        model.means[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="iteration 2"):
        trainer.step()


def test_non_finite_statistics_at_an_event_raise_floating_point_error(tmp_path: Path) -> None:
    """Q-69: gradient statistics that aren't finite at an event mean the run diverged, reported
    as FloatingPointError rather than as relocation's ValueError (from the two-axis review of
    #84)."""
    video = _scene(tmp_path)
    trainer = Trainer(_no_perceptual(**_EVERY_ITERATION), video, _one_live_row(video), device="cpu")
    state = trainer.state_dict()
    state.stats["g_sum"][0] = float("nan")
    trainer.load_state_dict(state)
    with pytest.raises(FloatingPointError, match="statistics"):
        trainer.step()


# train(), callbacks and global flags (Q-69, Q-73).


def test_train_calls_callbacks_in_order_and_stops_when_one_asks(tmp_path: Path) -> None:
    """Q-69: every callback runs after each step, in order; one returning True stops training
    after that step. `until` lies in [iteration, T_it]; past T_it the run is complete."""
    video = _scene(tmp_path)
    trainer = Trainer(
        _no_perceptual(optim=OptimConfig(iterations=20)), video, _init_points(), device="cpu"
    )
    calls: list[tuple[str, int]] = []

    def first(_: Trainer, log: StepLog) -> None:
        calls.append(("first", log.iteration))

    def stop_at_5(_: Trainer, log: StepLog) -> bool:
        calls.append(("second", log.iteration))
        return log.iteration == 5

    trainer.train(callbacks=[first, stop_at_5])
    assert trainer.iteration == 5
    assert calls == [(name, it) for it in range(1, 6) for name in ("first", "second")]
    trainer.train(until=8)
    assert trainer.iteration == 8
    for until in (7, 21, True, 9.0):
        with pytest.raises(ValueError, match="until"):
            trainer.train(until=until)  # type: ignore[arg-type]
    trainer.train()
    assert trainer.iteration == 20
    with pytest.raises(ValueError, match="complete"):
        trainer.step()


def test_a_training_iteration_leaves_the_global_torch_flags_as_they_were(tmp_path: Path) -> None:
    """Q-73: deterministic mode, the TF32 setting and the matmul precision (D-52) apply during a
    training iteration only."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    matmul = torch.get_float32_matmul_precision()
    algorithms = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = True, False, True
        torch.set_float32_matmul_precision("high")
        perf = PerfConfig(deterministic=True, perc_precision="ieee")
        trainer = Trainer(_no_perceptual(perf=perf), _scene(tmp_path), _init_points(), device="cpu")
        trainer.step()
        assert (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32) == (True, False, True)
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ) == algorithms
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = saved
        torch.set_float32_matmul_precision(matmul)


# Resuming another run's state (§2.2; Q-46, Q-69, Q-70, Q-72).


def test_a_resume_refuses_a_changed_scene_manifest_unless_forced(tmp_path: Path) -> None:
    """§2.2, Q-70: a checkpoint of another scene manifest is refused; forced, the resume keeps
    the run's S and Δt_f, so its learning rates continue those of the straight run (Q-46's
    note), although the moved camera changes the dataset's S."""
    video = _scene(tmp_path / "scene")
    cfg = _no_perceptual(optim=OptimConfig(iterations=30))
    straight = _run(Trainer(cfg, video, _init_points(), device="cpu"))
    first = Trainer(cfg, video, _init_points(), device="cpu")
    _run(first, 20)
    save_checkpoint(first.state_dict(), tmp_path / "ckpt")
    manifest = json.loads(video.manifest.path.read_text())
    manifest["cameras"]["ring00"]["world_to_camera"][2][3] += 10.0  # back along its axis
    changed_path = tmp_path / "scene" / "changed.json"
    changed_path.write_text(json.dumps(manifest))
    changed = MultiViewVideo.from_manifest(changed_path)
    assert changed.spatial_scale(1.1) != video.spatial_scale(1.1)
    with pytest.raises(ValueError, match="force"):
        Trainer.from_checkpoint(tmp_path / "ckpt", changed, device="cpu")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", changed, device="cpu", force=True)
    after = _run(resumed)
    assert [log.rates for log in after] == [log.rates for log in straight[20:]]
    assert [log.image_index for log in after] == [log.image_index for log in straight[20:]]


def test_load_state_dict_refuses_another_run_and_changes_nothing(tmp_path: Path) -> None:
    """Q-69, Q-72: a state of another configuration, whose sampler doesn't fit the dataset,
    whose iteration isn't an int, or whose global generator states are malformed, is a
    ValueError, and the trainer keeps its own state, global generators included."""
    video = _scene(tmp_path)
    trainer = _u21_trainer(video)
    _run(trainer, 10)
    before = trainer.state_dict()
    other = _u21_trainer(video, dataclasses.replace(U21, seed=5)).state_dict()
    with pytest.raises(ValueError, match="configuration"):
        trainer.load_state_dict(other)
    broken = [
        dataclasses.replace(before, sampler={**before.sampler, "permutation": [0, 1, 2]}),
        dataclasses.replace(
            before,
            sampler={
                **before.sampler,
                "permutation": [float(i) for i in before.sampler["permutation"]],
            },
        ),
        dataclasses.replace(before, iteration=10.0),  # type: ignore[arg-type]
        dataclasses.replace(before, rng={**before.rng, "python": "x"}),
        dataclasses.replace(before, rng={**before.rng, "numpy": {}}),
        dataclasses.replace(
            before, rng={**before.rng, "torch_cpu": torch.zeros(3, dtype=torch.uint8)}
        ),
    ]
    for state in broken:
        with pytest.raises(ValueError):
            trainer.load_state_dict(state)
    _assert_states_equal(trainer.state_dict(), before)


# On the GPU: gsplat (U-21's GPU variant; D-57, Q-36, Q-73).


def _flat(prefix: str, tree: object) -> dict[str, torch.Tensor]:
    """The tensors of a tree of dicts, by their dotted paths."""
    if isinstance(tree, torch.Tensor):
        return {prefix: tree}
    if isinstance(tree, dict):
        return {k: v for key, sub in tree.items() for k, v in _flat(f"{prefix}.{key}", sub).items()}
    return {}


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / ((a - b) ** 2).mean()))


@pytest.mark.gpu
def test_u21_on_gsplat_without_events(tmp_path: Path) -> None:
    """U-21 on the GPU, with no event in the window: 300 iterations straight against 150, a
    resume and 150 more give the same image order and rates, every stored tensor (parameters,
    Adam moments and step counters, statistics) within 1e-4 relative L2 error, and renders at
    least 50 dB apart (gsplat's backward pass reorders its sums, §2.2)."""
    video = _scene(tmp_path / "scene")
    cfg = dataclasses.replace(
        U21,
        growth=dataclasses.replace(U21.growth, mode="none"),
        reloc=dataclasses.replace(U21.reloc, enabled=False),
    )

    def trainer() -> Trainer:
        model = random_model(40, seed=2, sh_degree=1, time=video.time, spread=0.6)
        return Trainer(cfg, video, model, device="cuda")

    straight_trainer = trainer()
    straight = _run(straight_trainer)
    first = trainer()
    before = _run(first, 150)
    save_checkpoint(first.state_dict(), tmp_path / "ckpt")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cuda")
    after = _run(resumed)
    assert all(log.growth is None and log.relocation is None for log in straight)
    assert [(log.image_index, log.rates) for log in straight] == [
        (log.image_index, log.rates) for log in before + after
    ]
    a, b = straight_trainer.state_dict(), resumed.state_dict()
    tensors = {**_flat("params", a.params), **_flat("optim", a.optim), **_flat("stats", a.stats)}
    others = {**_flat("params", b.params), **_flat("optim", b.optim), **_flat("stats", b.stats)}
    assert set(tensors) == set(others)
    for name, value in tensors.items():
        x, y = value.double(), others[name].double()
        scale = float(torch.linalg.vector_norm(x))
        error = float(torch.linalg.vector_norm(x - y))
        assert error <= 1e-4 * scale or error == 0.0, name
    test = next(r for r in video.records if r.split == "test")
    renders = []
    for state in (a.params, b.params):
        model = FreeTimeGaussians(state, sh_degree=1, time=video.time).cuda()
        with torch.no_grad():
            renders.append(render(model, test.camera, test.timestamp).image)
    assert _psnr(renders[0], renders[1]) >= 50.0


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [True, False])
def test_gpu_runs_in_both_layouts(tmp_path: Path, packed: bool) -> None:
    """D-57, Q-36: training on gsplat in packed and unpacked mode, with events, stays finite."""
    video = _scene(tmp_path)
    base = _no_perceptual(
        init=InitConfig(points_per_frame=10),
        growth=GrowthConfig(n_max=200),
        perf=PerfConfig(packed=packed, log_every=10),
    )
    trainer = Trainer(shortened_config(50, base), video, _init_points(), device="cuda")
    logs = _run(trainer)
    assert len(logs) == 50 and _all_finite(logs)
    assert any(log.relocation is not None for log in logs)


@pytest.mark.gpu
def test_gpu_deterministic_run(tmp_path: Path) -> None:
    """Q-73: deterministic mode trains on gsplat and restores the flags afterwards."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
    perf = PerfConfig(deterministic=True)
    trainer = Trainer(_no_perceptual(perf=perf), _scene(tmp_path), _init_points(), device="cuda")
    logs = _run(trainer, 10)
    assert _all_finite(logs)
    assert (
        cudnn.benchmark,
        cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    ) == saved


# With the LPIPS networks (weights; D-77, Q-74).


@pytest.mark.weights
def test_training_with_the_perceptual_loss(tmp_path: Path) -> None:
    """D-77: with λ_perc > 0 the trainer builds the perceptual network once and passes it to
    total_loss, whose "perceptual" term is reported."""
    trainer = Trainer(TrainConfig(), _scene(tmp_path), _init_points(), device="cpu")
    logs = _run(trainer, 3)
    assert all(math.isfinite(float(log.losses["perceptual"])) for log in logs)


def _evaluation_scene(root: Path) -> MultiViewVideo:
    """64×48 cameras, so the masks' left halves (32 px) are usable dynamic regions (Q-61)."""
    cams = ring(3, radius=4.0, elevation=1.0, width_px=64, height_px=48, focal=60.0)
    gt = random_model(24, seed=0, sh_degree=1, time=TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images = [(cam, k / 30.0, "train") for k in range(5) for cam in cams[:2]]
    images += [(cams[2], k / 30.0, "test") for k in (0, 2, 4)]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images, mask_tests=True))


@pytest.mark.weights
def test_evaluate_returns_scene_means_and_writes_the_metrics_report(tmp_path: Path) -> None:
    """Q-74: the scene means of METRIC_NAMES for the entire images and, with masks, under
    "dynamic/<name>"; with out_dir, METRICS.json in Q-63's layout with the trainer's header."""
    video = _evaluation_scene(tmp_path / "scene")
    trainer = Trainer(_no_perceptual(), video, _init_points(), device="cpu")
    _run(trainer, 5)
    result = trainer.evaluate(out_dir=tmp_path / "out")
    assert set(result) == set(METRIC_NAMES) | {f"dynamic/{name}" for name in METRIC_NAMES}
    report = json.loads((tmp_path / "out" / "METRICS.json").read_text())
    assert report["format"] == "ftgs-metrics" and report["split"] == "test"
    assert report["model"] == {"iteration": 5}
    assert report["manifest"] == {
        "path": str(video.manifest.path),
        "sha256": video.manifest.sha256,
    }
    assert report["producer"].startswith("openftgs")
    for name in METRIC_NAMES:
        assert report["entire"]["scene"][name] == pytest.approx(result[name], rel=1e-12)
        assert report["dynamic"]["scene"][name] == pytest.approx(
            result[f"dynamic/{name}"], rel=1e-12
        )
    assert len(report["entire"]["images"]) == 3


@pytest.mark.weights
def test_an_evaluation_between_steps_leaves_the_run_unchanged(tmp_path: Path) -> None:
    """Q-74: evaluate draws from no generator and trains nothing, so a run with an evaluation
    after its fifth iteration equals one without, bitwise."""
    video = _evaluation_scene(tmp_path / "scene")
    cfg = _no_perceptual(optim=OptimConfig(iterations=10))
    plain = Trainer(cfg, video, _init_points(), device="cpu")
    plain_logs = _run(plain)
    evaluated = Trainer(cfg, video, _init_points(), device="cpu")

    def evaluate_at_5(trainer: Trainer, log: StepLog) -> None:
        if log.iteration == 5:
            trainer.evaluate()

    logs: list[StepLog] = []
    evaluated.train(callbacks=[lambda _, log: logs.append(log), evaluate_at_5])
    _assert_logs_equal(plain_logs, logs)
    _assert_states_equal(plain.state_dict(), evaluated.state_dict(), global_rng=False)
