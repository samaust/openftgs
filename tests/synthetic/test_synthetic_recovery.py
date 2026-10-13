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
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from synthetic_scenes import (
    C0,
    DATA,
    GROUND_TRUTH_FILES,
    assert_rows_matched,
    at_trajectory_points,
    config,
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
    scene_video,
    split_indices,
    tau,
    train,
    trajectory_error,
    velocity_errors,
    with_degree,
)

from openftgs.config import GrowthConfig, LossConfig, RelocConfig, TrainConfig
from openftgs.io import read_exchange
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


def _s2_config(**fields: object) -> TrainConfig:
    """S-2's configuration: growth off (§3.2), relocation and λ_reg at their defaults, SH degree
    3 (Q-79), on the shortened 5,000-iteration calendar."""
    base = TrainConfig(growth=GrowthConfig(mode="none"))
    return config(S2_ITERATIONS, dataclasses.replace(base, **fields))  # type: ignore[arg-type]


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
