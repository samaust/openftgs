# SPDX-License-Identifier: Apache-2.0
"""§3.2 synthetic-scene recovery tests S-1, S-2, S-3, S-5, S-6, S-7 and S-8 (M11, PRD #99).

Each scene trains from an initial model derived from committed ground truth and checks what was
recovered against §3.2's pass criteria, on the CPU with the reference backend and on the GPU with
gsplat (S-5 on the GPU only, Q-78). The setup choices §3.2 leaves open are Q-77 to Q-79, the
measures Q-80. Every test records its measured values with `record_property`, for the
Calibration data table of docs/clean-room/questions.md; no criterion is loosened there.
"""

import dataclasses
import functools
import math
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from synthetic_scenes import (
    C0,
    DATA,
    FPS,
    GROUND_TRUTH_FILES,
    assert_rows_matched,
    at_trajectory_points,
    concatenated,
    config,
    dead_rows,
    draw_taus,
    frame_taus,
    generate_ground_truth,
    grid,
    ground_truth,
    hemisphere_rig,
    load_seeds,
    mean_psnr,
    model_under,
    physical,
    replaced,
    ring_rig,
    rows,
    scene_video,
    split_indices,
    tau,
    train,
    trajectory_error,
    velocity_errors,
    with_degree,
)

from openftgs.config import GrowthConfig, LossConfig, RelocConfig, TrainConfig
from openftgs.data import MultiViewVideo
from openftgs.io import read_exchange
from openftgs.train import StepLog, Trainer
from openftgs.types import ExchangeModel

DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]
Record = Callable[[str, object], None]


def test_committed_ground_truth_reproduces_from_its_seeds() -> None:
    """§3.2 ("ground-truth files and seeds are committed"): the generator, run from the committed
    seeds, gives the committed ftgs-if files bitwise."""
    for scene, model in generate_ground_truth(load_seeds()).items():
        committed = read_exchange(DATA / GROUND_TRUTH_FILES[scene])
        assert (committed.sh_degree, committed.tau0, committed.tau1) == (
            model.sh_degree,
            model.tau0,
            model.tau1,
        )
        assert committed.arrays.keys() == model.arrays.keys()
        for key, value in model.arrays.items():
            assert torch.equal(committed.arrays[key], value), (scene, key)


@pytest.mark.parametrize("device", DEVICES)
def test_s1_one_static_gaussian_recovers_position_and_appearance(
    device: str, tmp_path: Path, record_property: Record
) -> None:
    """S-1: one static Gaussian, 8 ring cameras, 5 frames, s = 10; the initialization offset by
    0.05·S with a gray color (and D-5's opacity 0.1, Q-79); 2,000 iterations with relocation and
    regularization off. Position error ≤ 1e-3·S, held-out view PSNR ≥ 40 dB, and color and
    opacity compared as their product σ·c ≤ 0.01 (Q-81: one Gaussian over a uniform background
    renders as σ·G·c, so σ and c are not identifiable apart; both are recorded)."""
    gt = ground_truth("s1")
    rig = ring_rig()
    taus = [tau(k) for k in range(5)]
    video = scene_video(
        tmp_path, gt, grid(rig.train, taus, "train") + grid(rig.test, taus, "test"), device
    )
    S = video.spatial_scale(1.1)
    assert S == pytest.approx(0.99, rel=1e-6)

    g = torch.Generator().manual_seed(load_seeds()["s1"]["offset"])
    direction = torch.randn(3, generator=g, dtype=torch.float64)
    offset = 0.05 * S * direction / torch.linalg.vector_norm(direction)
    init = replaced(
        gt,
        position=gt.arrays["position"].to(torch.float64) + offset,
        sh=torch.zeros_like(gt.arrays["sh"]),  # gray: C₀·0 + ½
        opacity=torch.full_like(gt.arrays["opacity"], 0.1),  # D-5 (Q-79)
    )
    cfg = config(
        2_000,
        TrainConfig(
            sh_degree=0,
            loss=LossConfig(lambda_reg=0.0),
            growth=GrowthConfig(mode="none"),
            reloc=RelocConfig(enabled=False),
        ),
    )
    trainer, logs = train(cfg, video, model_under(init, video.time, sh_degree=0), device)
    assert_rows_matched(logs, 1)

    model = trainer.model
    position = trajectory_error(model, gt, torch.tensor(taus, dtype=torch.float64))
    p = physical(model)
    color = C0 * p["sh"][:, 0] + 0.5
    color_gt = C0 * gt.arrays["sh"][:, 0] + 0.5
    opacity, opacity_gt = p["opacity"][:, None], gt.arrays["opacity"][:, None]
    premultiplied_error = float((opacity * color - opacity_gt * color_gt).abs().max())  # Q-81
    held_out = mean_psnr(model, video, split_indices(video, "test"))
    _record(
        record_property,
        f"s1_{device}",
        {
            "position_error_over_S": position / S,
            "premultiplied_color_error": premultiplied_error,
            "color_error": float((color - color_gt).abs().max()),  # recorded only (Q-81)
            "opacity_error": float((opacity - opacity_gt).abs().max()),  # recorded only (Q-81)
            "held_out_psnr": held_out,
        },
    )
    assert position <= 1e-3 * S
    assert premultiplied_error <= 0.01  # Q-81
    assert held_out >= 40.0


# The S-2 family: 8 Gaussians moving linearly on the hemisphere rig (Q-77, Q-79).

S2_ITERATIONS = 5_000


def _s2_config(iterations: int = S2_ITERATIONS, **fields: object) -> TrainConfig:
    """S-2's configuration: growth off (§3.2), relocation and λ_reg at their defaults, SH degree
    3 (Q-79), on the shortened calendar of `iterations` (5,000 by default)."""
    base = TrainConfig(growth=GrowthConfig(mode="none"))
    return config(iterations, dataclasses.replace(base, **fields))  # type: ignore[arg-type]


def _s2_init(gt: ExchangeModel, seed: int, choices: list[float]) -> ExchangeModel:
    """Q-79: each primitive at the ground-truth point of a seeded random training time, μt at
    that time and v = 0, at SH degree 3."""
    return with_degree(at_trajectory_points(gt, draw_taus(seed, choices, 8)), 3)


@functools.cache
def _s2_run(device: str) -> dict[str, float]:
    """S-2's run on `device` and its measures; cached so S-8 compares against the same
    synchronized run (Q-80)."""
    gt = ground_truth("s2")
    rig = hemisphere_rig()
    taus = [tau(k) for k in range(30)]
    with tempfile.TemporaryDirectory() as root:
        video = scene_video(
            Path(root), gt, grid(rig.train, taus, "train") + grid(rig.test, taus, "test"), device
        )
        S = video.spatial_scale(1.1)
        init = _s2_init(gt, load_seeds()["s2"]["init_frames"], taus)
        trainer, logs = train(
            _s2_config(), video, model_under(init, video.time, sh_degree=3), device
        )
        assert_rows_matched(logs, 8)
        model = trainer.model
        velocity = velocity_errors(model, gt)
        return {
            "S": S,
            "trajectory_error_over_S": trajectory_error(model, gt, frame_taus()) / S,
            "velocity_magnitude": velocity["magnitude"],
            "velocity_angle_deg": velocity["angle_deg"],
            "velocity_vector": velocity["vector"],
            "held_out_psnr": mean_psnr(model, video, split_indices(video, "test")),
        }


def _record(record_property: Record, prefix: str, values: dict[str, float]) -> None:
    for name, value in values.items():
        record_property(f"{prefix}_{name}", value)


@pytest.mark.parametrize("device", DEVICES)
def test_s2_linear_motion_recovers_trajectories_and_velocities(
    device: str, record_property: Record
) -> None:
    """S-2: 8 Gaussians moving linearly (|v| = 0.125–0.5 per unit t, Q-77), s = 2, 12 hemisphere
    cameras, 30 frames; the initialization at true trajectory points with v = 0; 5,000
    iterations. Max over frames of ‖μx(t) − μx_gt(t)‖ ≤ 0.01·S; velocity error ≤ 5 % and ≤ 3°;
    held-out cameras PSNR ≥ 35 dB."""
    measures = _s2_run(device)
    _record(record_property, f"s2_{device}", measures)
    assert measures["S"] == pytest.approx(1.0, abs=0.02)
    assert measures["trajectory_error_over_S"] <= 0.01
    assert measures["velocity_magnitude"] <= 0.05
    assert measures["velocity_angle_deg"] <= 3.0
    assert measures["held_out_psnr"] >= 35.0


@pytest.mark.parametrize("device", DEVICES)
def test_s7_interpolates_at_unseen_odd_frames(
    device: str, tmp_path: Path, record_property: Record
) -> None:
    """S-7: S-2 trained on even frames and tested at the odd frames inside the training span
    (1–27, Q-79): odd-frame PSNR within 1 dB of even-frame PSNR, from the same 12 cameras."""
    gt = ground_truth("s2")
    rig = hemisphere_rig()
    even = [tau(k) for k in range(0, 30, 2)]
    odd = [tau(k) for k in range(1, 28, 2)]
    video = scene_video(
        tmp_path, gt, grid(rig.train, even, "train") + grid(rig.train, odd, "test"), device
    )
    init = _s2_init(gt, load_seeds()["s7"]["init_frames"], even)  # D-66: training frames only
    trainer, logs = train(_s2_config(), video, model_under(init, video.time, sh_degree=3), device)
    assert_rows_matched(logs, 8)
    psnr_even = mean_psnr(trainer.model, video, split_indices(video, "train"))
    psnr_odd = mean_psnr(trainer.model, video, split_indices(video, "test"))
    _record(record_property, f"s7_{device}", {"psnr_even": psnr_even, "psnr_odd": psnr_odd})
    assert abs(psnr_odd - psnr_even) <= 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_s8_per_image_timestamps_recover_unsynchronized_cameras(
    device: str, tmp_path: Path, record_property: Record
) -> None:
    """S-8: S-2 with each camera offset by a seeded random fraction of a frame (Q-79), images at
    τ = (k + δ_c)/30: the velocity error is at most 1.2 × that of the synchronized S-2 run on the
    same device (the relative vector error, Q-80)."""
    gt = ground_truth("s2")
    rig = hemisphere_rig()
    seeds = load_seeds()["s8"]
    g = torch.Generator().manual_seed(seeds["offsets"])
    offsets = torch.rand(len(rig.train), generator=g, dtype=torch.float64).tolist()
    images = [
        (camera, tau(k + delta), "train")
        for k in range(30)
        for camera, delta in zip(rig.train, offsets, strict=True)
    ]
    video = scene_video(tmp_path, gt, images, device)  # type: ignore[arg-type]
    choices = sorted({t for _, t, _ in images})
    init = with_degree(at_trajectory_points(gt, draw_taus(seeds["init_images"], choices, 8)), 3)
    trainer, logs = train(_s2_config(), video, model_under(init, video.time, sh_degree=3), device)
    assert_rows_matched(logs, 8)
    unsynchronized = velocity_errors(trainer.model, gt)["vector"]
    synchronized = _s2_run(device)["velocity_vector"]
    _record(
        record_property,
        f"s8_{device}",
        {"velocity_vector": unsynchronized, "synchronized_velocity_vector": synchronized},
    )
    assert unsynchronized <= 1.2 * synchronized


S3_ITERATIONS = 10_000  # Q-79, changed from 5,000 after the first measurement


@pytest.mark.parametrize("device", DEVICES)
def test_s3_recovers_time_centers_and_durations(
    device: str, tmp_path: Path, record_property: Record
) -> None:
    """S-3: 12 Gaussians with s_gt = 3 frames and centers spread over 30 frames, on the
    hemisphere rig (Q-79); the initialization with μt shifted by +2 frames and s = 1 frame;
    10,000 iterations (Q-79: the shortened calendar shortens the decay of the time centers'
    rate, D-20, so 5,000 left about 1.7 frames of effective budget for a 2-frame shift).
    |μt − μt_gt| ≤ 0.5 frame; s within 25 %; PSNR at held-out times (the training cameras at
    mid-frame times, Q-79) ≥ 33 dB."""
    gt = ground_truth("s3")
    rig = hemisphere_rig()
    frames = [tau(k) for k in range(30)]
    between = [tau(k + 0.5) for k in range(29)]
    video = scene_video(
        tmp_path, gt, grid(rig.train, frames, "train") + grid(rig.train, between, "test"), device
    )
    a = gt.arrays
    init = replaced(
        gt,
        time_center=a["time_center"] + tau(2),
        duration=torch.full_like(a["duration"], tau(1)),
    )
    cfg = _s2_config(S3_ITERATIONS)
    trainer, logs = train(cfg, video, model_under(init, video.time, sh_degree=3), device)
    assert_rows_matched(logs, 12)
    p = physical(trainer.model)
    center_error = float(((p["time_center"] - a["time_center"]).abs() * FPS).max())
    duration_error = float((p["duration"] / a["duration"] - 1.0).abs().max())
    held_out = mean_psnr(trainer.model, video, split_indices(video, "test"))
    _record(
        record_property,
        f"s3_{device}",
        {
            "time_center_error_frames": center_error,
            "duration_relative_error": duration_error,
            "held_out_time_psnr": held_out,
        },
    )
    assert center_error <= 0.5
    assert duration_error <= 0.25
    assert held_out >= 33.0


def _s2_scene(
    root: Path, device: str, scene: str = "s2"
) -> tuple[ExchangeModel, MultiViewVideo, list[float]]:
    """A ground truth of the S-2 family and its scene (12 training and 4 held-out hemisphere
    cameras, 30 frames)."""
    gt = ground_truth(scene)
    rig = hemisphere_rig()
    taus = [tau(k) for k in range(30)]
    images = grid(rig.train, taus, "train") + grid(rig.test, taus, "test")
    return gt, scene_video(root, gt, images, device), taus


@pytest.mark.gpu
def test_s5_regularization_keeps_fewer_primitives_opaque(
    tmp_path: Path, record_property: Record
) -> None:
    """S-5 (GPU only, Q-78): S-2 plus 2,000 small random Gaussians in the ground truth, every row
    initialized at its true point with opacity 0.1 (D-5, Q-78, Q-79; S-2's 8 objects as in S-2,
    with v = 0), trained with λ_reg = 0 and 10⁻², growth "gradient" (N_max = 10,000): at the end
    the share of primitives with σ > 0.9 is lower with regularization, and the final held-out
    PSNR is not lower by more than 0.5 dB."""
    device = "cuda"
    gt, video, taus = _s2_scene(tmp_path, device, "s5")
    objects = _s2_init(rows(gt, slice(0, 8)), load_seeds()["s2"]["init_frames"], taus)
    init = concatenated(objects, with_degree(rows(gt, slice(8, None)), 3))
    init = replaced(init, opacity=torch.full_like(init.arrays["opacity"], 0.1))
    start = model_under(init, video.time, sh_degree=3)
    growth = GrowthConfig(n_max=10_000)  # mode "gradient" (Q-78)
    results: dict[str, float] = {}
    for name, lambda_reg in (("no_reg", 0.0), ("reg", 1e-2)):
        cfg = config(
            S2_ITERATIONS, TrainConfig(loss=LossConfig(lambda_reg=lambda_reg), growth=growth)
        )
        trainer, _ = train(cfg, video, start, device)
        opacity = physical(trainer.model)["opacity"]
        results[f"{name}_opaque_share"] = float((opacity > 0.9).to(torch.float64).mean())
        results[f"{name}_count"] = float(opacity.shape[0])
        results[f"{name}_held_out_psnr"] = mean_psnr(
            trainer.model, video, split_indices(video, "test")
        )
    _record(record_property, f"s5_{device}", results)
    assert results["reg_opaque_share"] < results["no_reg_opaque_share"]
    assert results["reg_held_out_psnr"] >= results["no_reg_held_out_psnr"] - 0.5


# D-14: a row is dead when its float32 opacity logit lies below the float32 logit of 0.005.
DEAD_LOGIT = float(torch.tensor(math.log(0.005) - math.log1p(-0.005), dtype=torch.float32))


@dataclasses.dataclass(frozen=True)
class _S6:
    counts: frozenset[int]  # N over every iteration of both runs
    first_event_dead: int  # dead rows the first relocation event moved
    dead_after_event: tuple[int, ...]  # event iterations that left a dead row
    events: int
    psnr_on: float
    psnr_off: float


@functools.cache
def _s6_runs(device: str) -> _S6:
    """S-6's two runs on `device` (relocation on and off, Q-78), shared by its two tests."""
    with tempfile.TemporaryDirectory() as root:
        gt, video, taus = _s2_scene(Path(root), device)
        seeds = load_seeds()
        dead = dead_rows(seeds["s6"]["dead"], gt, 8, frame_taus())
        init = concatenated(_s2_init(gt, seeds["s2"]["init_frames"], taus), with_degree(dead, 3))
        start = model_under(init, video.time, sh_degree=3)
        events: list[StepLog] = []
        dead_after: list[int] = []

        def check_event(trainer: Trainer, log: StepLog) -> None:
            if log.relocation is not None:
                events.append(log)
                if not bool((trainer.model.opacity_logits.detach() >= DEAD_LOGIT).all()):
                    dead_after.append(log.iteration)

        counts: set[int] = set()
        psnrs: dict[str, float] = {}
        for name, enabled in (("on", True), ("off", False)):
            cfg = _s2_config(reloc=RelocConfig(enabled=enabled))
            trainer, logs = train(cfg, video, start, device, [check_event] if enabled else [])
            counts.update(log.num_gaussians for log in logs)
            psnrs[name] = mean_psnr(trainer.model, video, split_indices(video, "test"))
        first = events[0].relocation
        assert first is not None
        return _S6(
            counts=frozenset(counts),
            first_event_dead=first.dead,
            dead_after_event=tuple(dead_after),
            events=len(events),
            psnr_on=psnrs["on"],
            psnr_off=psnrs["off"],
        )


@pytest.mark.parametrize("device", DEVICES)
def test_s6_relocation_keeps_n_and_revives_every_dead_primitive(device: str) -> None:
    """S-6: S-2's initialization plus as many dead primitives (σ = 10⁻³) away from the objects,
    N = 16 (Q-78), relocation on, growth off. N stays constant; the first relocation event moves
    all 8 dead rows (Q-82); right after each event no row is dead (Q-78)."""
    runs = _s6_runs(device)
    assert runs.counts == {16}
    assert runs.events > 0
    assert runs.first_event_dead == 8
    assert runs.dead_after_event == ()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.xfail(
    strict=True,
    reason="Q-82: unattainable under E10 and E11 for S-2's one-Gaussian objects; the measured "
    "gain is recorded in the Calibration data table, pending a spec decision",
)
def test_s6_relocation_gains_one_db(device: str, record_property: Record) -> None:
    """S-6: PSNR with relocation ≥ PSNR without + 1 dB (calibrated). Kept as §3.2 states it
    under a strict xfail (Q-82), so the measurement is recorded and a pass is loud."""
    runs = _s6_runs(device)
    gain = runs.psnr_on - runs.psnr_off
    _record(
        record_property,
        f"s6_{device}",
        {"psnr_on": runs.psnr_on, "psnr_off": runs.psnr_off, "gain_db": gain},
    )
    assert gain >= 1.0
