# SPDX-License-Identifier: Apache-2.0
"""The trainer in openftgs.train: the calendar part of U-17, U-21, U-24, the smoke runs of U-26,
S-10, the added tests of PRD #84 (§1.3, §2.2, D-17, D-18, D-26, D-27, D-45, D-46, D-71;
Q-69 to Q-75), and #97's tests of the model the trainer trains (Q-69 as answered, Q-76).

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
from typing import Any, Literal

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

import openftgs.render
from openftgs.config import (
    GrowthConfig,
    InitConfig,
    LossConfig,
    OptimConfig,
    PerfConfig,
    RelocConfig,
    TrainConfig,
    config_to_dict,
)
from openftgs.data import FrameCache, MultiViewVideo
from openftgs.eval import METRIC_NAMES
from openftgs.io import load_checkpoint, save_checkpoint
from openftgs.losses import LPIPSLoss
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.train import StepLog, Trainer, _Sampler, calendar_entry
from openftgs.types import Camera, InitPoints, Renders, TimeNormalizer, TrainerState

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


def test_several_images_per_step_are_refused_naming_two_gpus_and_l3(tmp_path: Path) -> None:
    """Q-177 (owner's answer, 2026-10-19): multi-GPU training is deferred, so images_per_step
    > 1 (D-54) still fails at construction rather than being ignored, and says why: the mode
    needs two GPUs and must pass M16's L3 first. The frame cache's placements arrived with M12
    (Q-99) and bf16 LPIPS with M15 (D-52, #228)."""
    video = _scene(tmp_path)
    with pytest.raises(NotImplementedError, match=r"two GPUs.*M16's L3"):
        Trainer(
            _no_perceptual(perf=PerfConfig(images_per_step=2)), video, _init_points(), device="cpu"
        )


def test_the_cuda_slice_is_refused_at_construction(tmp_path: Path) -> None:
    """Q-73, D-50: slice_impl "cuda" names the custom operator B-1 decided against (#226), so the
    trainer refuses it at construction rather than at its first slice (from the M15 milestone
    review)."""
    video = _scene(tmp_path)
    cfg = _no_perceptual(perf=PerfConfig(slice_impl="cuda"))
    with pytest.raises(NotImplementedError, match=r"D-50"):
        Trainer(cfg, video, _init_points(), device="cpu")


def test_bf16_perceptual_precision_is_accepted(tmp_path: Path) -> None:
    """D-52, Q-73's note (4): bf16 LPIPS is built in M15 (#228), so the trainer no longer refuses
    it; without the perceptual term it trains as any other run."""
    video = _scene(tmp_path)
    cfg = _no_perceptual(perf=PerfConfig(perc_precision="bf16"))
    trainer = Trainer(cfg, video, _init_points(), device="cpu")
    assert _all_finite([trainer.step()])


# Checkpoints and exact resume (U-21), determinism (U-24), stability (S-10).

# U-21's test calendar: events every 50 iterations from 100, growth up to 200, relocation up to
# 250, so events fall on both sides of a resume at 150. A low gradient threshold makes every
# event grow, and a dead threshold of 0.2 leaves dead rows at every event (Q-11 ranges hold).
# U-21's scene has 12 training images (four frames), so 150 and 125 fall in mid-epoch.
U21_FRAMES = 4
U21 = TrainConfig(
    sh_degree=1,
    loss=LossConfig(lambda_perc=0.0),
    growth=GrowthConfig(start=100, stop=200, grad_threshold=1e-7, n_max=400),
    reloc=RelocConfig(every=50, start=100, stop=250, dead_opacity=0.2),
    optim=OptimConfig(iterations=300, sh_interval=50),
    perf=PerfConfig(log_every=50),
)

# U-24's 100-iteration runs, with events every 20 iterations from 20.
U24 = dataclasses.replace(
    U21,
    growth=dataclasses.replace(U21.growth, start=20, stop=60),
    reloc=dataclasses.replace(U21.reloc, every=20, start=20, stop=80),
    optim=dataclasses.replace(U21.optim, iterations=100, sh_interval=20),
    perf=dataclasses.replace(U21.perf, log_every=20),
)

# U-21's calendar cut to 120 iterations: one event, at 100, which grows and relocates.
ONE_EVENT = dataclasses.replace(U21, optim=dataclasses.replace(U21.optim, iterations=120))


def _u21_model(video: MultiViewVideo) -> FreeTimeGaussians:
    return random_model(
        40,
        seed=2,
        sh_degree=1,
        time=video.time,
        spread=0.6,
        log_scale=(-2.5, -1.5),
        logit=(-3.0, 2.0),
    )


def _u21_trainer(video: MultiViewVideo, cfg: TrainConfig = U21) -> Trainer:
    return Trainer(cfg, video, _u21_model(video), device="cpu")


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


def _assert_logs_equal(
    a: Sequence[StepLog], b: Sequence[StepLog], *, mean_losses: bool = True
) -> None:
    """Bitwise-equal step logs; `mean_losses=False` leaves out the log window's means, which a
    resume restarts (§2.2's contract covers images, rates, draws and parameters, not logs)."""
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert (x.iteration, x.image_index, x.timestamp, x.t, x.sh_degree) == (
            y.iteration,
            y.image_index,
            y.timestamp,
            y.t,
            y.sh_degree,
        )
        assert x.rates == y.rates, x.iteration
        assert not mean_losses or x.mean_losses == y.mean_losses, x.iteration
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
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    trainer = _u21_trainer(video)
    _run(trainer, 125)  # after an event, in the middle of an epoch
    state = trainer.state_dict()
    assert 0 < state.sampler["cursor"] < len(state.sampler["permutation"])
    save_checkpoint(state, tmp_path / "ckpt")
    _assert_states_equal(state, load_checkpoint(tmp_path / "ckpt"))
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    assert resumed.iteration == 125
    _assert_states_equal(state, resumed.state_dict())


def test_u21_resumed_run_is_bitwise_equal_to_the_straight_run(tmp_path: Path) -> None:
    """U-21 (reference backend, CPU): 300 iterations straight against 150, a resume through a
    checkpoint directory, and 150 more give the same images, rates, loss terms, event reports
    and final state, bitwise. Events grow and relocate rows on both sides of the resume, which
    falls in the middle of an epoch, so a partly used permutation is restored. The log window
    restarts at a resume, so its means are compared before the resume only."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    straight_trainer = _u21_trainer(video)
    straight = _run(straight_trainer)
    first = _u21_trainer(video)
    before = _run(first, 150)
    state = first.state_dict()
    assert 0 < state.sampler["cursor"] < len(state.sampler["permutation"])
    save_checkpoint(state, tmp_path / "ckpt")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    after = _run(resumed)
    assert _events_did_work(before) and _events_did_work(after)
    _assert_logs_equal(straight[:150], before)
    _assert_logs_equal(straight[150:], after, mean_losses=False)
    _assert_states_equal(straight_trainer.state_dict(), resumed.state_dict(), global_rng=False)


def test_u24_two_seeded_runs_are_bitwise_identical(tmp_path: Path) -> None:
    """U-24: two 100-iteration runs from one configuration and seed, with events, give the same
    step logs and final state, bitwise; neither seeds the global generators (Q-72)."""
    video = _scene(tmp_path)
    a, b = _u21_trainer(video, U24), _u21_trainer(video, U24)
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
    reported as FloatingPointError (exit code 3 in M14), not as an input error. The trainer
    trains a copy of the model passed in (#97), so a NaN written into the caller's object isn't
    seen, while one written into `trainer.model`, the model being trained, is."""
    video = _scene(tmp_path)
    model = random_model(20, sh_degree=3, time=video.time, spread=0.6)
    trainer = Trainer(_no_perceptual(perf=PerfConfig(log_every=1)), video, model, device="cpu")
    assert trainer.step().mean_losses is not None
    with torch.no_grad():
        model.means[0, 0] = float("nan")
    assert trainer.step().mean_losses is not None
    with torch.no_grad():
        trainer.model.means[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="iteration 3"):
        trainer.step()


# The model the trainer trains (Q-69 as answered, #97; Q-76).


def _params_of(model: FreeTimeGaussians) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach().clone() for name in model.PARAM_NAMES}


def test_a_passed_model_is_never_modified(tmp_path: Path) -> None:
    """Q-69 (owner answer, #97): the trainer trains a copy of a passed model, so the caller's
    object keeps its tensors, its device and its slice_impl after construction, after training
    iterations with events, and after a construction refused once the copy is in place
    (selective Adam on the CPU, Q-45); the copy takes PerfConfig.slice_impl (Q-66)."""
    video = _scene(tmp_path)
    model = _u21_model(video)
    model.slice_impl = "compiled"  # not the default, so a reset would show
    before = _params_of(model)

    def unchanged() -> None:
        _assert_same(_params_of(model), before)
        assert model.means.device.type == "cpu" and model.slice_impl == "compiled"

    cfg = dataclasses.replace(ONE_EVENT, perf=dataclasses.replace(U21.perf, slice_impl="eager"))
    trainer = Trainer(cfg, video, model, device="cpu")
    unchanged()
    assert trainer.model.slice_impl == "eager"
    assert _events_did_work(_run(trainer))
    unchanged()
    selective = dataclasses.replace(cfg, perf=dataclasses.replace(cfg.perf, adam="selective"))
    with pytest.raises(ValueError, match="selective"):
        Trainer(selective, video, model, device="cpu")
    unchanged()


def test_the_copy_keeps_the_time_unit(tmp_path: Path) -> None:
    """Q-69 (owner answer, #97), Q-17: the copy keeps the passed model's time unit, which must
    be the scene's; the trainer state records it."""
    scene = _scene(tmp_path)
    manifest = json.loads(scene.manifest.path.read_text())
    manifest["time_unit"] = "frame"
    path = scene.manifest.path.with_name("frames.json")
    path.write_text(json.dumps(manifest))
    video = MultiViewVideo.from_manifest(path)
    model = _u21_model(video)
    model.time_unit = "frame"
    trainer = Trainer(_no_perceptual(sh_degree=1), video, model, device="cpu")
    assert trainer.model.time_unit == "frame" and trainer.state_dict().time_unit == "frame"


def test_two_trainers_from_one_model_give_identical_runs(tmp_path: Path) -> None:
    """Q-69 (owner answer, #97), U-24's property: a trainer built from a model object after
    another trainer ran from it gives the same step logs and final state, bitwise, with events,
    since the first one didn't train the object itself."""
    video = _scene(tmp_path)
    model = _u21_model(video)
    a = Trainer(U24, video, model, device="cpu")
    logs_a = _run(a)
    b = Trainer(U24, video, model, device="cpu")
    logs_b = _run(b)
    assert _events_did_work(logs_a)
    _assert_logs_equal(logs_a, logs_b)
    _assert_states_equal(a.state_dict(), b.state_dict(), global_rng=False)


def test_trainer_model_is_the_model_being_trained(tmp_path: Path) -> None:
    """Q-69 (owner answer, #97): `Trainer.model` is the model the trainer trains: its stored
    tensors are the state's; events resize it in place, so it stays the same object;
    `load_state_dict` and `from_checkpoint` replace it with the loaded state's model; it is
    read-only."""
    video = _scene(tmp_path / "scene")
    trainer = Trainer(ONE_EVENT, video, _u21_model(video), device="cpu")
    model = trainer.model
    _assert_same(_params_of(model), trainer.state_dict().params)
    assert _events_did_work(_run(trainer))
    assert trainer.model is model
    state = trainer.state_dict()
    _assert_same(_params_of(trainer.model), state.params)
    trainer.load_state_dict(state)
    assert trainer.model is not model
    _assert_same(_params_of(trainer.model), state.params)
    save_checkpoint(state, tmp_path / "ckpt")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu")
    _assert_same(_params_of(resumed.model), state.params)
    with pytest.raises(AttributeError):
        trainer.model = model  # type: ignore[misc]


def test_a_passed_model_that_breaks_the_validity_rules_is_refused(tmp_path: Path) -> None:
    """Q-76 (owner agreed while triaging #97): the trainer's copy goes through the model's
    constructor, so a model that fails §1.1's validity rules is a ValueError at construction,
    an input error (Q-26), and the caller's object is untouched."""
    video = _scene(tmp_path)
    model = random_model(20, sh_degree=3, time=video.time, spread=0.6)
    with torch.no_grad():
        model.means[0, 0] = float("nan")
    before = _params_of(model)
    with pytest.raises(ValueError, match="not finite"):
        Trainer(_no_perceptual(), video, model, device="cpu")
    for name, value in before.items():
        torch.testing.assert_close(
            getattr(model, name).detach(), value, rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.gpu
def test_a_passed_cpu_model_stays_on_the_cpu(tmp_path: Path) -> None:
    """Q-69 (owner answer, #97): a CPU model passed to a CUDA trainer stays on the CPU,
    bitwise unchanged, while the model being trained is on the CUDA device."""
    video = _scene(tmp_path)
    model = random_model(20, sh_degree=3, time=video.time, spread=0.6)
    before = _params_of(model)
    trainer = Trainer(_no_perceptual(), video, model, device="cuda")
    _run(trainer, 5)
    assert model.means.device.type == "cpu" and trainer.model.means.device.type == "cuda"
    _assert_same(_params_of(model), before)


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


# Q-173 (changed, #326), Q-223: cuDNN autotuning is on in training steps that aren't deterministic.
AUTOTUNING = True


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("caller", [False, True])
def test_cudnn_autotuning_during_a_step_follows_b4_and_deterministic(
    tmp_path: Path, deterministic: bool, caller: bool
) -> None:
    """D-52, Q-73, Q-173: during a training iteration's backward pass, cuDNN autotuning is off
    under `deterministic`; otherwise it is on when Q-173 adopted it (#326), and else left as the
    caller set it. Every flag is the caller's again after the step."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    try:
        cudnn.benchmark = caller
        perf = PerfConfig(deterministic=deterministic)
        trainer = Trainer(_no_perceptual(perf=perf), _scene(tmp_path), _init_points(), device="cpu")
        seen: list[bool] = []
        trainer.model.means.register_hook(lambda _: seen.append(cudnn.benchmark))  # type: ignore[no-untyped-call]
        trainer.step()
        expected = False if deterministic else (True if AUTOTUNING else caller)
        assert seen == [expected]
        assert (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32) == (caller, *saved[1:])
    finally:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = saved


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


# A resume with other exact performance settings (Q-134 as answered; Q-153).


def test_a_resume_with_other_exact_performance_settings_is_bitwise(tmp_path: Path) -> None:
    """Q-134 as answered, §2.2, §2.4 rule 2: a run checkpointed in mid-epoch with the frame
    cache on the device and resumed through from_checkpoint(perf=…) with the memory-mapped
    cache, another prefetch depth, another log period and deterministic kernels gives bitwise
    the state of a straight run, on the reference backend on the CPU, where the "gpu"
    placement holds whole tensors; the resumed trainer runs with, and records, the new
    settings."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    straight = _u21_trainer(video, _with_cache(ONE_EVENT, "gpu"))
    _run(straight)
    first = _u21_trainer(video, _with_cache(ONE_EVENT, "gpu"))
    _run(first, 65)
    save_checkpoint(first.state_dict(), tmp_path / "ckpt")
    perf = dataclasses.replace(
        ONE_EVENT.perf, frame_cache="mmap", prefetch=3, log_every=7, deterministic=True
    )
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu", perf=perf)
    logs = _run(resumed)
    assert [log.iteration for log in logs if log.mean_losses is not None] == list(
        range(70, ONE_EVENT.optim.iterations + 1, 7)
    )
    state = resumed.state_dict()
    assert state.config["perf"] == config_to_dict(dataclasses.replace(ONE_EVENT, perf=perf))["perf"]
    want = straight.state_dict()
    _assert_states_equal(want, dataclasses.replace(state, config=want.config), global_rng=False)


@pytest.mark.parametrize(
    "change", [{"adam": "foreach"}, {"perc_precision": "ieee"}, {"images_per_step": 2}]
)
def test_from_checkpoint_refuses_performance_settings_that_change_results(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """Q-134 as answered: perc_precision, adam and images_per_step change results (D-52 to
    D-54), so from_checkpoint(perf=…) refuses a change to one of them with ValueError naming
    its dotted key, before anything is built."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    trainer = _u21_trainer(video, ONE_EVENT)
    _run(trainer, 5)
    save_checkpoint(trainer.state_dict(), tmp_path / "ckpt")
    perf = dataclasses.replace(ONE_EVENT.perf, **change)
    (key,) = change
    with pytest.raises(ValueError, match=f"perf.{key}"):
        Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cpu", perf=perf)


@pytest.mark.parametrize("change", [{"adam": "foreach"}, {"perc_precision": "ieee"}])
def test_load_state_dict_refuses_a_state_whose_results_would_differ(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    """Q-134 as answered, Q-69: a state whose configuration differs from the trainer's in a
    PerfConfig field that changes results is a ValueError naming it, and the trainer keeps its
    own state."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    source = _u21_trainer(video, ONE_EVENT)
    _run(source, 5)
    mine = dataclasses.replace(ONE_EVENT, perf=dataclasses.replace(ONE_EVENT.perf, **change))
    trainer = _u21_trainer(video, mine)
    before = trainer.state_dict()
    (key,) = change
    with pytest.raises(ValueError, match=f"perf.{key}"):
        trainer.load_state_dict(source.state_dict())
    _assert_states_equal(trainer.state_dict(), before)


def test_load_state_dict_accepts_other_exact_settings_and_keeps_its_own(tmp_path: Path) -> None:
    """Q-134 as answered, Q-69: a state whose configuration differs from the trainer's only in
    exact PerfConfig fields is loaded, and the trainer keeps its own values for them."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    source = _u21_trainer(video, ONE_EVENT)
    _run(source, 5)
    mine = dataclasses.replace(ONE_EVENT, perf=dataclasses.replace(ONE_EVENT.perf, log_every=3))
    trainer = _u21_trainer(video, mine)
    trainer.load_state_dict(source.state_dict())
    assert trainer.iteration == 5
    assert trainer.state_dict().config["perf"]["log_every"] == 3


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
def test_a_gsplat_resume_from_the_device_cache_to_the_mapped_one(tmp_path: Path) -> None:
    """Q-134 as answered, §2.2's gsplat clause: a run checkpointed with frame_cache = "gpu"
    and resumed with "mmap" matches a straight run within U-21's GPU tolerance (1e-4 relative
    L2 error of every stored tensor, the same image order and rates)."""
    video = _scene(tmp_path / "scene")
    cfg = _with_cache(
        dataclasses.replace(
            U21,
            growth=dataclasses.replace(U21.growth, mode="none"),
            reloc=dataclasses.replace(U21.reloc, enabled=False),
        ),
        "gpu",
    )

    def trainer() -> Trainer:
        model = random_model(40, seed=2, sh_degree=1, time=video.time, spread=0.6)
        return Trainer(cfg, video, model, device="cuda")

    straight_trainer = trainer()
    straight = _run(straight_trainer)
    first = trainer()
    before = _run(first, 150)
    save_checkpoint(first.state_dict(), tmp_path / "ckpt")
    perf = dataclasses.replace(cfg.perf, frame_cache="mmap")
    resumed = Trainer.from_checkpoint(tmp_path / "ckpt", video, device="cuda", perf=perf)
    after = _run(resumed)
    assert [(log.image_index, log.rates) for log in straight] == [
        (log.image_index, log.rates) for log in before + after
    ]
    a, b = straight_trainer.state_dict(), resumed.state_dict()
    assert b.config["perf"]["frame_cache"] == "mmap"
    tensors = {**_flat("params", a.params), **_flat("optim", a.optim), **_flat("stats", a.stats)}
    others = {**_flat("params", b.params), **_flat("optim", b.optim), **_flat("stats", b.stats)}
    assert set(tensors) == set(others)
    for name, value in tensors.items():
        x, y = value.double(), others[name].double()
        error = float(torch.linalg.vector_norm(x - y))
        assert error <= 1e-4 * float(torch.linalg.vector_norm(x)) or error == 0.0, name


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


def _step_with_gradients(trainer: Trainer) -> tuple[StepLog, dict[str, torch.Tensor]]:
    """One training iteration, with the gradients its backward pass gives the model's stored
    tensors, caught by hooks on them (the optimizer zeroes them before the step returns)."""
    grads: dict[str, torch.Tensor] = {}
    model = trainer.model
    for name in model.PARAM_NAMES:
        tensor = getattr(model, name)
        tensor.register_hook(lambda g, name=name: grads.__setitem__(name, g.detach().clone()))
    return trainer.step(), grads


@pytest.mark.weights
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_bf16_step_gives_a_finite_perceptual_term_near_float32(
    tmp_path: Path, device: str
) -> None:
    """D-52, Q-172 (#228): with perc_precision = "bf16" one training step runs LPIPS-VGG under
    bfloat16 autocast; its perceptual term is finite and within 5e-2 relative of the float32
    ("ieee") step from the same seed, which renders the same image, and the gradients that reach
    the model are finite. bf16 is approximate (D-52), so the two terms differ."""
    video = _scene(tmp_path)
    terms, gradients = {}, {}
    for precision in ("ieee", "bf16"):
        cfg = TrainConfig(perf=PerfConfig(perc_precision=precision, frame_cache="off"))
        trainer = Trainer(cfg, video, _init_points(), device=device)
        log, grads = _step_with_gradients(trainer)
        terms[precision] = float(log.losses["perceptual"])
        gradients[precision] = grads
    assert math.isfinite(terms["bf16"]) and terms["ieee"] > 0.0
    assert terms["bf16"] != terms["ieee"]
    assert abs(terms["bf16"] - terms["ieee"]) <= 5e-2 * terms["ieee"]
    assert "means" in gradients["bf16"]
    assert all(bool(torch.isfinite(g).all()) for g in gradients["bf16"].values())


@pytest.mark.weights
@pytest.mark.parametrize("precision", ["ieee", "tf32"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_the_float32_perceptual_precisions_run_a_plain_lpips_loss_on_nchw_inputs_bitwise(
    tmp_path: Path, device: str, precision: Literal["ieee", "tf32"]
) -> None:
    """D-52, Q-172 (#228, PRD #220: "the float32 path is bitwise unchanged"), Q-223 (#326): with
    perc_precision "ieee" or "tf32" a training step's perceptual term equals, bitwise, a direct
    call of a separately built plain `LPIPSLoss`, not bf16's, on the same two images laid out
    so that their batch-first views are contiguous NCHW, as the trainer's network copies them;
    made at the same moment, so under the global flags the step sets (`cudnn.allow_tf32` per
    precision, autotuning). The images are caught by a global forward hook, PyTorch's public
    `register_module_forward_hook`."""
    from torch.nn.modules.module import register_module_forward_hook

    def nchw(image: torch.Tensor) -> torch.Tensor:
        # [..., H, W, 3] whose batch-first view [B, 3, H, W] is contiguous
        return image.detach().movedim(-1, -3).contiguous().movedim(-3, -1)

    cfg = TrainConfig(perf=PerfConfig(perc_precision=precision, frame_cache="off"))
    trainer = Trainer(cfg, _scene(tmp_path), _init_points(), device=device)
    direct = LPIPSLoss(cfg.loss.perc_net).to(device)
    seen: list[tuple[type, torch.Tensor, torch.Tensor]] = []
    busy = False

    def hook(module: torch.nn.Module, args: Any, output: Any) -> None:
        nonlocal busy
        if busy or not isinstance(module, LPIPSLoss):
            return
        busy = True  # the direct call below runs the same hook
        try:
            expected = direct(*(nchw(arg) for arg in args))
        finally:
            busy = False
        seen.append((type(module), output.detach().clone(), expected.detach()))

    handle = register_module_forward_hook(hook)
    try:
        log = trainer.step()
    finally:
        handle.remove()
    assert len(seen) == 1
    kind, term, expected = seen[0]
    assert issubclass(kind, LPIPSLoss)
    assert torch.equal(term, expected)
    assert float(log.losses["perceptual"]) == float(expected)


@pytest.mark.weights
@pytest.mark.parametrize("precision", ["ieee", "tf32", "bf16"])
def test_the_trainers_perceptual_network_gets_nchw_inputs_except_under_bf16(
    tmp_path: Path, precision: Literal["ieee", "tf32", "bf16"]
) -> None:
    """Q-173 (changed, #326), Q-223: during a training step, lpips's network gets both images as
    contiguous NCHW tensors under perc_precision "ieee" and "tf32", and as the channels-last view
    of the render under "bf16", for which B-4 measured NCHW inputs slower. A plain `LPIPSLoss`,
    as the evaluator builds (E16), still passes the channels-last view. The inputs are caught
    by PyTorch's public `register_module_forward_pre_hook`."""
    import lpips
    from torch.nn.modules.module import register_module_forward_pre_hook

    layouts: list[list[tuple[bool, bool]]] = []

    def hook(module: torch.nn.Module, args: Any) -> None:
        if isinstance(module, lpips.LPIPS):
            last = torch.channels_last
            layouts.append([(a.is_contiguous(), a.is_contiguous(memory_format=last)) for a in args])

    cfg = TrainConfig(perf=PerfConfig(perc_precision=precision, frame_cache="off"))
    trainer = Trainer(cfg, _scene(tmp_path), _init_points(), device="cpu")
    plain = LPIPSLoss(cfg.loss.perc_net)
    handle = register_module_forward_pre_hook(hook)
    try:
        trainer.step()
        plain(torch.rand(32, 40, 3), torch.rand(32, 40, 3))
    finally:
        handle.remove()
    nchw, channels_last = (True, False), (False, True)
    expected = channels_last if precision == "bf16" else nchw
    assert layouts == [[expected, expected], [channels_last, channels_last]]


@pytest.mark.parametrize(("precision", "tf32"), [("ieee", False), ("tf32", True), ("bf16", True)])
def test_cudnn_tf32_during_a_step_follows_the_perceptual_precision(
    tmp_path: Path, precision: Literal["tf32", "ieee", "bf16"], tf32: bool
) -> None:
    """D-52, Q-172 (#245): during a training iteration's backward pass, cuDNN's TF32 flag is off
    under perc_precision "ieee" and on under "tf32" and "bf16", whose autocast is local to the
    perceptual network, so every other convolution (the SSIM fallback's) keeps TF32. The flag
    is the caller's again after the step."""
    cudnn = torch.backends.cudnn
    saved = cudnn.allow_tf32
    try:
        cudnn.allow_tf32 = not tf32
        perf = PerfConfig(perc_precision=precision)
        trainer = Trainer(_no_perceptual(perf=perf), _scene(tmp_path), _init_points(), device="cpu")
        seen: list[bool] = []
        trainer.model.means.register_hook(lambda _: seen.append(cudnn.allow_tf32))  # type: ignore[no-untyped-call]
        trainer.step()
        assert seen == [tf32]
        assert cudnn.allow_tf32 is not tf32
    finally:
        cudnn.allow_tf32 = saved


@pytest.mark.weights
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_bf16_step_changes_only_the_perceptual_term(tmp_path: Path, device: str) -> None:
    """D-52, Q-172 (#245): bf16's autocast is local to the perceptual network, so a "bf16" step
    and a "tf32" step from the same seed, both with TF32 on, give bitwise the same L1, 1 − SSIM
    and L_reg terms, and only the perceptual term differs. On the CPU, SSIM takes its PyTorch
    fallback, whose convolution an autocast around the whole loss would reach."""
    video = _scene(tmp_path)
    terms: dict[str, dict[str, torch.Tensor]] = {}
    for precision in ("tf32", "bf16"):
        cfg = TrainConfig(perf=PerfConfig(perc_precision=precision, frame_cache="off"))
        trainer = Trainer(cfg, video, _init_points(), device=device)
        terms[precision] = {name: value.cpu() for name, value in trainer.step().losses.items()}
    assert terms["bf16"].keys() == terms["tf32"].keys()
    for name in ("l1", "one_minus_ssim", "reg"):
        assert torch.equal(terms["bf16"][name], terms["tf32"][name]), name
    assert not torch.equal(terms["bf16"]["perceptual"], terms["tf32"]["perceptual"])


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
    assert report["model"] == {"path": None, "iteration": 5, "sha256": None}  # Q-136
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


def _nan_first_render(monkeypatch: pytest.MonkeyPatch, video: MultiViewVideo) -> list[str]:
    """Patch the evaluation's renderer so that its first render is NaN; the list receives the
    manifest path of that render's image."""
    original = openftgs.render.render_request  # the object openftgs.eval imports
    paths: list[str] = []

    def nan(*args: Any, **kwargs: Any) -> Renders:
        renders = original(*args, **kwargs)
        if not paths:
            key = next(iter(renders.rgb))
            renders.rgb[key] = torch.full_like(renders.rgb[key], math.nan)
            image = video.records[int(key)].image_path
            paths.append(image.relative_to(video.manifest.path.parent).as_posix())
        return renders

    monkeypatch.setattr("openftgs.eval.render_request", nan)
    return paths


@pytest.mark.weights
@pytest.mark.parametrize("written", [False, True])
def test_evaluate_refuses_a_render_that_is_not_finite_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, written: bool
) -> None:
    """Q-147, Q-129: a NaN render makes evaluate raise FloatingPointError naming its image,
    before writing anything, with or without out_dir."""
    video = _evaluation_scene(tmp_path / "scene")
    trainer = Trainer(_no_perceptual(), video, _init_points(), device="cpu")
    _run(trainer, 5)
    paths = _nan_first_render(monkeypatch, video)
    out = tmp_path / "out"
    with pytest.raises(FloatingPointError) as raised:
        trainer.evaluate(out_dir=out if written else None)
    assert paths and paths[0] in str(raised.value)
    assert not (out / "METRICS.json").exists()


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


# The frame cache and the prefetcher in training (U-30, D-51; Q-99).


def _with_cache(cfg: TrainConfig, placement: Literal["auto", "gpu", "mmap", "off"]) -> TrainConfig:
    return dataclasses.replace(cfg, perf=dataclasses.replace(cfg.perf, frame_cache=placement))


def test_u30_cached_training_is_bitwise_the_png_training(tmp_path: Path) -> None:
    """U-30, D-51: a run reading its images from the memory-mapped cache through the prefetcher
    gives bitwise the logs and state of a run decoding every PNG; resumed in mid-epoch (an
    epoch is 12 images; 65 is 5 into the sixth), it gives the same images, in the same order,
    and the same final state."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    png = _u21_trainer(video, _with_cache(ONE_EVENT, "off"))
    cached = _u21_trainer(video, _with_cache(ONE_EVENT, "mmap"))
    png_logs, cached_logs = _run(png), _run(cached)
    _assert_logs_equal(png_logs, cached_logs)
    # The states differ only in the configuration's frame_cache, which they record.
    cached_state = dataclasses.replace(cached.state_dict(), config=png.state_dict().config)
    _assert_states_equal(png.state_dict(), cached_state, global_rng=False)
    assert FrameCache.default_directory(video.manifest).is_dir()
    first = _u21_trainer(video, _with_cache(ONE_EVENT, "mmap"))
    before = _run(first, 65)
    state = first.state_dict()
    assert 0 < state.sampler["cursor"] < len(state.sampler["permutation"])
    resumed = _u21_trainer(video, _with_cache(ONE_EVENT, "mmap"))
    resumed.load_state_dict(state)
    after = _run(resumed)
    _assert_logs_equal(png_logs[:65], before)
    _assert_logs_equal(png_logs[65:], after, mean_losses=False)
    resumed_state = dataclasses.replace(resumed.state_dict(), config=png.state_dict().config)
    _assert_states_equal(png.state_dict(), resumed_state, global_rng=False)


def test_sampler_peek_matches_the_next_draws() -> None:
    """Q-72 note 3, Q-99: `peek` returns the rest of the permutation and the next epoch's,
    drawn on a copy of the generator, so the stream doesn't move and `next` then returns the
    same indices."""
    sampler = _Sampler([3, 5, 8, 13], torch.Generator().manual_seed(7))
    for _ in range(3):
        sampler.next()
    state = sampler.generator.get_state()
    upcoming = sampler.peek(6)
    assert torch.equal(sampler.generator.get_state(), state)
    assert upcoming == [sampler.next() for _ in range(6)][: len(upcoming)]
    assert len(upcoming) == 5  # one left of this epoch, four of the next


def test_auto_placement_falls_back_when_no_cache_can_be_made(tmp_path: Path) -> None:
    """Q-99: when the cache directory can't be created, "auto" decodes PNGs with a warning,
    and an explicit placement is a ValueError (Q-26)."""
    video = _scene(tmp_path / "scene")
    (tmp_path / "scene" / ".ftgs-cache").write_text("not a directory")
    with pytest.warns(UserWarning, match="frame cache"):
        trainer = Trainer(_no_perceptual(), video, _init_points(), device="cpu")
    trainer.step()
    with pytest.raises(ValueError, match="frame cache"):
        Trainer(
            _no_perceptual(perf=PerfConfig(frame_cache="mmap")), video, _init_points(), device="cpu"
        )


def test_prefetcher_follows_the_samplers_order(tmp_path: Path) -> None:
    """Q-72 note 3, Q-99: across epochs and a resume, every image comes from the prefetch
    queue, scheduled at construction and after each take, so the prefetcher reads in the
    sampler's order."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    trainer = _u21_trainer(video, _with_cache(ONE_EVENT, "mmap"))
    _run(trainer, 40)  # 12 images an epoch: three epoch boundaries
    assert trainer._frames is not None and trainer._frames._misses == 0
    resumed = _u21_trainer(video, _with_cache(ONE_EVENT, "mmap"))
    resumed.load_state_dict(trainer.state_dict())
    _run(resumed, 60)
    assert resumed._frames is not None and resumed._frames._misses == 0


@pytest.mark.gpu
def test_cuda_auto_placement_falls_back_after_the_first_iteration(tmp_path: Path) -> None:
    """Q-99: on CUDA "auto" decides after the first iteration; a cache directory that can't be
    created then gives a warning and PNG decoding, not an error."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    (tmp_path / "scene" / ".ftgs-cache").write_text("not a directory")
    trainer = Trainer(_with_cache(ONE_EVENT, "auto"), video, _u21_model(video), device="cuda")
    with pytest.warns(UserWarning, match="frame cache"):
        trainer.step()
    assert trainer._frames is None
    trainer.step()


@pytest.mark.gpu
@pytest.mark.parametrize("placement", ["gpu", "mmap", "auto"])
def test_u30_cuda_training_reads_from_each_placement(
    tmp_path: Path, placement: Literal["auto", "gpu", "mmap"]
) -> None:
    """U-30 (GPU), Q-99: training on CUDA reads its images from the cache in GPU memory, from
    the memory-mapped cache through the CUDA prefetcher, or, with "auto", from whichever the
    free memory allows after the first iteration; the losses stay finite."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    cfg = _with_cache(
        dataclasses.replace(ONE_EVENT, optim=dataclasses.replace(ONE_EVENT.optim, iterations=30)),
        placement,
    )
    trainer = Trainer(cfg, video, _u21_model(video), device="cuda")
    logs = _run(trainer)
    assert len(logs) == 30 and _all_finite(logs)
    assert FrameCache.default_directory(video.manifest).is_dir()


@pytest.mark.gpu
@pytest.mark.parametrize("placement", ["off", "auto"])
def test_u30_cuda_png_path_gives_the_first_losses_of_the_gpu_cache_bitwise(
    tmp_path: Path, placement: Literal["off", "auto"]
) -> None:
    """U-30 (GPU), D-51, §2.4 rule 3: on CUDA the PNG path ("off", and "auto" until its first
    iteration has chosen a placement) stages each decoded image in pinned memory and copies it
    without blocking. Its first iteration gives bitwise the loss terms of a first iteration
    reading the cache in GPU memory, before any update could make the runs differ (gsplat's
    backward pass accumulates with atomic additions)."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    first = [
        Trainer(_with_cache(ONE_EVENT, cache), video, _u21_model(video), device="cuda").step()
        for cache in (placement, "gpu")
    ]
    assert first[0].image_index == first[1].image_index
    assert first[0].losses.keys() == first[1].losses.keys()
    for name, value in first[0].losses.items():
        assert torch.equal(value, first[1].losses[name]), name


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
@pytest.mark.parametrize("placement", ["off", "auto"])
def test__image_on_cuda_is_the_decoded_png_bitwise_and_never_synchronizes(
    tmp_path: Path, placement: Literal["off", "auto"]
) -> None:
    """D-51, §2.4 rule 3: on the PNG path the trainer's CUDA image of every training record is
    bitwise what `MultiViewVideo.load_image` decodes, and taking it makes no host–device
    synchronization (torch's sync debug mode)."""
    video = _scene(tmp_path / "scene", frames=U21_FRAMES)
    trainer = Trainer(_with_cache(ONE_EVENT, placement), video, _u21_model(video), device="cuda")
    training = [i for i, record in enumerate(video.records) if record.split == "train"]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        images = [trainer._image(index) for index in training]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    for index, image in zip(training, images, strict=True):
        assert image.device.type == "cuda"
        assert torch.equal(image, video.load_image(index).cuda()), index
