# SPDX-License-Identifier: Apache-2.0
"""§3.2 synthetic-scene recovery tests S-1, S-2, S-3, S-6, S-7 and S-8 (M11, PRD #99), and
S-4 (M13, PRD #133). S-5 runs on the ray-cast scene, in `test_end_to_end.py` (D-86).

Each scene trains from an initial model derived from committed ground truth and checks what was
recovered against §3.2's pass criteria, on the CPU with the reference backend and on the GPU with
gsplat. S-4 instead starts from init points sampled on its moving spheres, with E15 velocities or
none (Q-119). The setup choices §3.2 leaves open are Q-77, Q-79 and Q-119, the measures Q-80;
S-1 departs from §3.2's row where the owner decided (Q-81), and S-6 runs on D-87's clusters.
Every test records its measured values with `record_property`, for the Calibration data table of
docs/clean-room/questions.md; no criterion is loosened there.
"""

import dataclasses
import functools
import math
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch
from builders import C0
from synthetic_scenes import (
    DATA,
    FPS,
    GROUND_TRUTH_FILES,
    S4_FRAMES,
    S4_GT_POINTS,
    SceneImage,
    assert_rows_matched,
    at_trajectory_points,
    concatenated,
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
    s4_init,
    same_exchange,
    scene_video,
    split_indices,
    synthetic_config,
    tau,
    train,
    trajectory_error,
    velocity_error_rows,
    velocity_errors,
    with_degree,
)

from openftgs.config import GrowthConfig, LossConfig, RelocConfig, TrainConfig
from openftgs.data import MultiViewVideo
from openftgs.io import read_exchange
from openftgs.train import StepLog, Trainer
from openftgs.types import ExchangeModel, TimeNormalizer

DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]
Record = Callable[[str, object], None]


def _record(record_property: Record, prefix: str, values: dict[str, float]) -> None:
    for name, value in values.items():
        record_property(f"{prefix}_{name}", value)


def test_committed_ground_truth_reproduces_from_its_seeds() -> None:
    """§3.2 ("ground-truth files and seeds are committed"): the generator, run from the committed
    seeds, gives the committed ftgs-if files bitwise."""
    for scene, model in generate_ground_truth(load_seeds()).items():
        assert same_exchange(read_exchange(DATA / GROUND_TRUTH_FILES[scene]), model), scene


def test_recovery_measures_on_a_ground_truth_with_known_errors() -> None:
    """Q-80's measures against S-2's ground truth perturbed by known amounts: a position shift d
    gives a trajectory error of ‖d‖; shifting μt by δ and μx by v·δ leaves the trajectory
    unchanged (§3.2's note on S-2); velocities × 1.1 give a magnitude and vector error of 0.1
    and no angle; velocities turned by 2° in a plane containing them give an angle of 2°, no
    magnitude error and a vector error of 2·sin 1°."""
    gt = ground_truth("s2")
    a = gt.arrays
    frame = TimeNormalizer(gt.tau0, gt.tau1)
    position, velocity = a["position"].to(torch.float64), a["velocity"].to(torch.float64)
    taus = frame_taus()

    d = torch.tensor([0.003, 0.0, 0.004], dtype=torch.float64)
    shifted = model_under(replaced(gt, position=position + d), frame, sh_degree=0)
    assert trajectory_error(shifted, gt, taus) == pytest.approx(0.005, abs=1e-6)
    delta = tau(1.5)
    moved = replaced(gt, position=position + velocity * delta, time_center=a["time_center"] + delta)
    assert trajectory_error(model_under(moved, frame, sh_degree=0), gt, taus) <= 1e-6

    faster = model_under(replaced(gt, velocity=1.1 * velocity), frame, sh_degree=0)
    errors = velocity_errors(faster, gt)
    assert errors["magnitude"] == pytest.approx(0.1, abs=1e-5)
    assert errors["vector"] == pytest.approx(0.1, abs=1e-5)
    assert errors["angle_deg"] <= 0.05

    theta = math.radians(2.0)
    side = torch.linalg.cross(velocity, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    side = side / torch.linalg.vector_norm(side, dim=1, keepdim=True)
    speed = torch.linalg.vector_norm(velocity, dim=1, keepdim=True)
    turned_velocity = math.cos(theta) * velocity + math.sin(theta) * speed * side
    turned = model_under(replaced(gt, velocity=turned_velocity), frame, sh_degree=0)
    errors = velocity_errors(turned, gt)
    assert errors["angle_deg"] == pytest.approx(2.0, abs=0.01)
    assert errors["magnitude"] <= 1e-5
    assert errors["vector"] == pytest.approx(2 * math.sin(theta / 2), abs=1e-5)


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
    cfg = synthetic_config(
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


def _s2_config(iterations: int = S2_ITERATIONS, *, reloc: RelocConfig | None = None) -> TrainConfig:
    """S-2's configuration: growth off (§3.2), relocation (unless `reloc` is given) and λ_reg at
    their defaults, SH degree 3 (Q-79), on the shortened calendar of `iterations` (5,000 by
    default)."""
    base = TrainConfig(growth=GrowthConfig(mode="none"), reloc=reloc or RelocConfig())
    return synthetic_config(iterations, base)


def _train(
    cfg: TrainConfig,
    video: MultiViewVideo,
    init: ExchangeModel,
    device: str,
    callbacks: Sequence[Callable[[Trainer, StepLog], None]] = (),
) -> tuple[Trainer, list[StepLog]]:
    """A run from `init` under the video's time normalizer, at SH degree 3."""
    return train(cfg, video, model_under(init, video.time, sh_degree=3), device, callbacks)


def _held_out_psnr(trainer: Trainer, video: MultiViewVideo) -> float:
    return mean_psnr(trainer.model, video, split_indices(video, "test"))  # Q-80


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


def _s2_init(gt: ExchangeModel, seed: int, choices: list[float]) -> ExchangeModel:
    """Q-79: each primitive at the ground-truth point of a seeded random training time, μt at
    that time and v = 0, at SH degree 3."""
    return with_degree(at_trajectory_points(gt, draw_taus(seed, choices, 8)), 3)


@functools.cache
def _s2_run(device: str) -> dict[str, float]:
    """S-2's run on `device` and its measures; cached so S-8 compares against the same
    synchronized run (Q-80)."""
    with tempfile.TemporaryDirectory() as root:
        gt, video, taus = _s2_scene(Path(root), device)
        S = video.spatial_scale(1.1)
        init = _s2_init(gt, load_seeds()["s2"]["init_frames"], taus)
        trainer, logs = _train(_s2_config(), video, init, device)
        assert_rows_matched(logs, 8)
        velocity = velocity_errors(trainer.model, gt)  # Q-80
        return {
            "S": S,
            "trajectory_error_over_S": trajectory_error(trainer.model, gt, frame_taus()) / S,
            "velocity_magnitude": velocity["magnitude"],
            "velocity_angle_deg": velocity["angle_deg"],
            "velocity_vector": velocity["vector"],
            "held_out_psnr": _held_out_psnr(trainer, video),
        }


# Q-214: on one xdist worker, which runs the cached run once.
@pytest.mark.xdist_group("s2-run")
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
    trainer, logs = _train(_s2_config(), video, init, device)
    assert_rows_matched(logs, 8)
    # Q-79, Q-80: even frames are the training images themselves, odd frames the test split.
    psnr_even = mean_psnr(trainer.model, video, split_indices(video, "train"))
    psnr_odd = mean_psnr(trainer.model, video, split_indices(video, "test"))
    _record(record_property, f"s7_{device}", {"psnr_even": psnr_even, "psnr_odd": psnr_odd})
    assert abs(psnr_odd - psnr_even) <= 1.0


# Q-214: on one xdist worker, which runs the cached run once.
@pytest.mark.xdist_group("s2-run")
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
    images: list[SceneImage] = [
        (camera, tau(k + delta), "train")
        for k in range(30)
        for camera, delta in zip(rig.train, offsets, strict=True)
    ]
    video = scene_video(tmp_path, gt, images, device)
    choices = sorted({t for _, t, _ in images})
    init = _s2_init(gt, seeds["init_images"], choices)
    trainer, logs = _train(_s2_config(), video, init, device)
    assert_rows_matched(logs, 8)
    unsynchronized = velocity_errors(trainer.model, gt)["vector"]  # Q-80
    synchronized = _s2_run(device)["velocity_vector"]
    _record(
        record_property,
        f"s8_{device}",
        {"velocity_vector": unsynchronized, "synchronized_velocity_vector": synchronized},
    )
    assert unsynchronized <= 1.2 * synchronized


@pytest.mark.parametrize("device", DEVICES)
def test_s4_e15_init_recovers_the_spheres_velocities(device: str, record_property: Record) -> None:
    """S-4: init points sampled on S-4's three moving spheres, one fixed jittered lattice per
    sphere translated to each of the 10 frames (Q-119), with E15 velocities (`knn_velocities` on
    `device`, a backward difference on the last frame, D-23): the median velocity error over the
    init points is ≤ 5 %. The error is Q-80's relative vector error ‖v − v_gt‖/‖v_gt‖ (S-8's
    single number, which bounds the relative magnitude error) against the velocity of the
    point's sphere, in world units per τ. §3.2's precondition, a displacement per frame below
    half the sampling spacing, is checked first."""
    gt = ground_truth("s4")
    init = s4_init(gt, load_seeds()["s4"]["init"], device)
    v_gt = gt.arrays["velocity"].to(torch.float64)[init.sphere * S4_GT_POINTS]
    largest_step = float(torch.linalg.vector_norm(v_gt, dim=1).max()) / FPS
    assert largest_step < 0.5 * init.spacing  # §3.2 S-4
    velocities = init.points.velocities
    assert velocities is not None
    errors = velocity_error_rows(velocities, v_gt)  # Q-80
    _record(
        record_property,
        f"s4_{device}",
        {
            "median_velocity_vector": float(errors["vector"].median()),
            "median_velocity_magnitude": float(errors["magnitude"].median()),
            "median_velocity_angle_deg": float(errors["angle_deg"].median()),
            "max_velocity_vector": float(errors["vector"].max()),
            "largest_step_over_half_spacing": largest_step / (0.5 * init.spacing),
        },
    )
    assert float(errors["vector"].median()) <= 0.05


# Q-119: S-4's iteration count, the same as S-1's. The gain of E15 shrinks as both runs
# converge (scratch probes on other draws of this scene gave 0.9–1.6 dB at 1,000–2,000
# iterations and about 0.5 dB at 3,000–5,000, the CPU and the GPU within 0.01 dB), and 2,000
# keeps the CPU variant under its 3-minute target.
S4_ITERATIONS = 2_000


@pytest.mark.parametrize("device", DEVICES)
def test_s4_zero_velocity_init_ends_below_e15_init(
    device: str, tmp_path: Path, record_property: Record
) -> None:
    """S-4 (the direction of Table 4): with the same budget, the zero-velocity init ends with a
    lower held-out PSNR than the E15 init. Both runs train from the same init points (Q-119),
    once as made and once with `velocities=None` (v = 0, D-23's fallback), each for
    S4_ITERATIONS on the shortened calendar (Q-75) with growth "none", relocation and λ_reg at
    their defaults, SH degree 3 and λ_perc = 0 (Q-79), so N stays at the init's 420 rows in
    both. PSNR is the mean over the 4 held-out hemisphere cameras at the 10 training frames
    (Q-79, Q-80)."""
    gt = ground_truth("s4")
    rig = hemisphere_rig()
    taus = [tau(k) for k in range(S4_FRAMES)]
    video = scene_video(
        tmp_path, gt, grid(rig.train, taus, "train") + grid(rig.test, taus, "test"), device
    )
    points = s4_init(gt, load_seeds()["s4"]["init"], device).points
    n = int(points.positions.shape[0])
    cfg = synthetic_config(S4_ITERATIONS, TrainConfig(growth=GrowthConfig(mode="none")))
    held_out: dict[str, float] = {}
    zero_velocity = dataclasses.replace(points, velocities=None)
    for name, init in (("e15", points), ("zero_velocity", zero_velocity)):
        trainer, logs = train(cfg, video, init, device)
        assert all(log.num_gaussians == n for log in logs)  # the same budget
        held_out[name] = _held_out_psnr(trainer, video)
    _record(
        record_property,
        f"s4_{device}",
        {
            "e15_held_out_psnr": held_out["e15"],
            "zero_velocity_held_out_psnr": held_out["zero_velocity"],
            "e15_gain_db": held_out["e15"] - held_out["zero_velocity"],
        },
    )
    assert held_out["zero_velocity"] < held_out["e15"]


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
    trainer, logs = _train(_s2_config(S3_ITERATIONS), video, init, device)
    assert_rows_matched(logs, 12)
    p = physical(trainer.model)
    center_error = float(((p["time_center"] - a["time_center"]).abs() * FPS).max())
    duration_error = float((p["duration"] / a["duration"] - 1.0).abs().max())
    held_out = _held_out_psnr(trainer, video)
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


# D-14: a row is dead when its float32 opacity logit lies below the float32 logit of 0.005.
DEAD_LOGIT = float(torch.tensor(math.log(0.005) - math.log1p(-0.005), dtype=torch.float32))


@dataclasses.dataclass(frozen=True)
class _S6Runs:
    counts: frozenset[int]  # N over every iteration of both runs
    first_event_dead: int  # dead rows the first relocation event moved
    dead_after_event: tuple[int, ...]  # event iterations that left a dead row
    events: int
    psnr_on: float
    psnr_off: float


@functools.cache
def _s6_runs(device: str) -> _S6Runs:
    """S-6's two runs on `device` (relocation on and off), shared by its two tests: the
    clustered ground truth (D-87), initialized as S-2 (one live row per object, at its center)
    plus 8 dead rows (N = 16, D-87)."""
    with tempfile.TemporaryDirectory() as root:
        _, video, taus = _s2_scene(Path(root), device, "s6")
        objects = ground_truth("s2")
        seeds = load_seeds()
        dead = dead_rows(seeds["s6"]["dead"], objects, 8, frame_taus())
        live = _s2_init(objects, seeds["s2"]["init_frames"], taus)
        init = concatenated(live, with_degree(dead, 3))
        events: list[StepLog] = []
        dead_after: list[int] = []

        def check_event(trainer: Trainer, log: StepLog) -> None:
            if log.relocation is not None:
                events.append(log)
                # D-87: right after an event, where E11's lower clamp leaves no dead row.
                if not bool((trainer.model.opacity_logits.detach() >= DEAD_LOGIT).all()):
                    dead_after.append(log.iteration)

        counts: set[int] = set()
        psnrs: dict[str, float] = {}
        for name, enabled in (("on", True), ("off", False)):
            cfg = _s2_config(reloc=RelocConfig(enabled=enabled))
            trainer, logs = _train(cfg, video, init, device, [check_event] if enabled else [])
            counts.update(log.num_gaussians for log in logs)
            psnrs[name] = _held_out_psnr(trainer, video)
        first = events[0].relocation
        assert first is not None
        return _S6Runs(
            counts=frozenset(counts),
            first_event_dead=first.dead,
            dead_after_event=tuple(dead_after),
            events=len(events),
            psnr_on=psnrs["on"],
            psnr_off=psnrs["off"],
        )


# Q-214: on one xdist worker, which runs the cached run once.
@pytest.mark.xdist_group("s6-runs")
@pytest.mark.parametrize("device", DEVICES)
def test_s6_relocation_keeps_n_and_revives_every_dead_primitive(
    device: str, record_property: Record
) -> None:
    """S-6: S-2's objects as clusters of three Gaussians (D-87), S-2's initialization plus as
    many dead primitives (σ = 10⁻³) away from the objects, N = 16 (D-87), relocation on, growth
    off. N stays constant; the first relocation event moves all 8 dead rows; right after each
    event no row is dead (D-87)."""
    runs = _s6_runs(device)
    _record(
        record_property,
        f"s6_{device}",
        {
            "events": float(runs.events),
            "first_event_dead": float(runs.first_event_dead),
            "events_leaving_a_dead_row": float(len(runs.dead_after_event)),
        },
    )
    assert runs.counts == {16}
    assert runs.events > 0
    assert runs.first_event_dead == 8
    assert runs.dead_after_event == ()


# Q-214: on one xdist worker, which runs the cached run once.
@pytest.mark.xdist_group("s6-runs")
@pytest.mark.parametrize("device", DEVICES)
def test_s6_relocation_gains_one_db(device: str, record_property: Record) -> None:
    """S-6: PSNR with relocation ≥ PSNR without + 1 dB (calibrated; the measured gain goes to
    the Calibration data table), on the clustered ground truth whose clusters one live row
    can't fit (D-87)."""
    runs = _s6_runs(device)
    gain = runs.psnr_on - runs.psnr_off
    _record(
        record_property,
        f"s6_{device}",
        {"psnr_on": runs.psnr_on, "psnr_off": runs.psnr_off, "gain_db": gain},
    )
    assert gain >= 1.0
