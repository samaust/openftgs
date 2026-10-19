# SPDX-License-Identifier: Apache-2.0
"""§3.2's end-to-end scenes on non-Gaussian data (M14, PRD #157): S-9 through the command line
(ftgs train, export and render; ftgs init roma when the [init] extra is installed) and S-11's
coverage of surfaces with no initial points, both on the textured spheres of the analytic ray
caster in `raycast.py` (Q-138 to Q-140), and S-5's effect of E8 on the same scene (#100, D-86).
All train on the GPU: on the reference backend their 320 images at 128×96 would far exceed
Q-119's 3-minute CPU target, as M11's S-5 did (Q-138).

Each test records its measured values with `record_property`, for the Calibration data table of
docs/clean-room/questions.md; the thresholds are §3.2's and never loosened there.
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import raycast
import torch
from builders import look_at, write_render_request
from synthetic_scenes import load_seeds, mean_psnr, smallest_spacing, synthetic_config, train

from openftgs.cli import main
from openftgs.config import GrowthConfig, LossConfig, TrainConfig, config_to_dict
from openftgs.data import MultiViewVideo
from openftgs.eval import psnr, ssim_k
from openftgs.io import read_renders, write_init_points
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.sh import rgb_to_sh0
from openftgs.train import Trainer
from openftgs.types import ImageRecord, InitPoints

ITERATIONS = 7_000  # Q-140: set during the M14 build and recorded
PER_SPHERE = 300  # Q-140: lattice points per sphere and frame, 900 per frame (B = 1,000)
S9_PSNR, S9_SSIM = 28.0, 0.9  # §3.2 S-9 (calibrated)
S11_WITHIN_DB = 1.0  # §3.2 S-11 (calibrated)
FILLER_OPACITY = 0.01  # Q-140: S-11's random low-opacity primitives
S5_MAX_COST_DB = 0.5  # §3.2 S-5: "final PSNR not lower by more than 0.5 dB"

has_init_extra = all(importlib.util.find_spec(m) for m in ("einops", "loguru", "torchvision"))
Record = Callable[[str, object], None]


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> raycast.Scene:
    return raycast.write_scene(tmp_path_factory.mktemp("spheres"), load_seeds()["s9"]["spheres"])


def _full_init(scene: raycast.Scene) -> InitPoints:
    return raycast.surface_samples(scene.spheres, load_seeds()["s9"]["lattice"], PER_SPHERE)


def _cfg(**fields: Any) -> TrainConfig:
    return synthetic_config(ITERATIONS, TrainConfig(**fields))


def _test_records(scene: raycast.Scene) -> tuple[MultiViewVideo, list[int]]:
    video = MultiViewVideo.from_manifest(scene.manifest)
    return video, [i for i, r in enumerate(video.records) if r.split == "test"]


# The ray caster itself, against the cameras' own projection (§1.4).


def test_cast_sees_each_sphere_where_the_camera_projects_its_center() -> None:
    """For one sphere at a time, the pixel nearest its projected center (K and world_to_camera
    of §1.4, continuous convention) shows the texture where the line toward the camera meets the
    surface, within the texture's slope over that pixel: the colors are 0.5 + 0.4 × a mean of
    sinusoids, so they change by at most 0.4 × the top texture frequency per radian of the local
    direction, and one pixel spans depth / focal / radius radians there. Every hit pixel has a
    positive coverage, the pixel at a sphere's center a full one."""
    rig = raycast.rig()
    scene = raycast.spheres(load_seeds()["s9"]["spheres"], rig.train + rig.test)
    for camera in (rig.train[0], rig.train[9], rig.test[2]):
        w2c, K = camera.world_to_camera.double().numpy(), camera.K.double().numpy()
        eye = -w2c[:3, :3].T @ w2c[:3, 3]
        for sphere in scene:
            cast = raycast.cast([sphere], camera, 0.0, supersample=1)
            center = sphere.center_at(0.0)
            pix = K @ (w2c[:3, :3] @ center + w2c[:3, 3])
            u, v = math.floor(pix[0] / pix[2]), math.floor(pix[1] / pix[2])
            toward = (eye - center) / np.linalg.norm(eye - center)
            depth = float(np.linalg.norm(eye - center)) - raycast.RADIUS
            footprint = depth / raycast.FOCAL / raycast.RADIUS  # rad of the local direction
            slope = 0.4 * raycast.TEXTURE_FREQUENCY[1]
            assert np.abs(cast.image[v, u] - sphere.color(toward)).max() < slope * footprint
            assert np.array_equal(cast.coverage > 0, cast.image.sum(axis=-1) > 0)
            assert raycast.cast([sphere], camera, 0.0).coverage[v, u] == 1.0
    # Q-140: seen from the -x side of a sphere every visible point has local x < 0, from +x
    # every one has x > 0.
    for sphere in scene:
        center = sphere.center_at(0.0)
        for side, negative in ((-1.0, True), (1.0, False)):
            eye = tuple(float(c) for c in center + np.array([side, 0.0, 0.0]))
            camera = look_at(
                eye,
                target=tuple(center),
                up=(0, 0, 1),
                width=raycast.WIDTH,
                height=raycast.HEIGHT,
                focal=raycast.FOCAL,
            )
            cast = raycast.cast([sphere], camera, 0.0)
            hit = cast.coverage > 0
            assert hit.sum() > 100
            # `half` is the share of all of a pixel's rays, so a silhouette pixel shows below 1.
            assert np.array_equal(cast.half > 0, hit) if negative else not cast.half.any()
            if negative:
                assert float(np.median(cast.half[hit])) == 1.0


def test_motion_stays_below_half_the_init_lattice_spacing(record_property: Record) -> None:
    """§3.2 S-4's precondition as Q-119 applies it to S-9 (D-23, Q-155, Q-156): every sphere's
    displacement per frame, and the top of the drawn range, is below half the smallest spacing
    of the init's own lattice on that sphere (the surface samples of one frame), so E15's
    nearest neighbour of a point in the next frame is its own moved copy; the ratio reached is
    recorded."""
    rig = raycast.rig()
    scene = raycast.spheres(load_seeds()["s9"]["spheres"], rig.train + rig.test)
    points = raycast.surface_samples(scene, load_seeds()["s9"]["lattice"], PER_SPHERE)
    first = points.positions[: PER_SPHERE * len(scene)].double()  # frame 0, sphere by sphere
    lattices = [first[k * PER_SPHERE : (k + 1) * PER_SPHERE] for k in range(len(scene))]
    half = 0.5 * min(smallest_spacing(lattice) for lattice in lattices)
    step = max(float(np.linalg.norm(s.velocity)) / raycast.FPS for s in scene)
    record_property("s9_step_over_half_spacing", step / half)
    assert step < half
    assert raycast.STEP[1] < half  # Q-155: the whole range, not only this seed's draws


def test_surface_samples_of_the_positive_x_half_leave_the_cast_half_bare() -> None:
    """Q-140: S-11's half init keeps exactly the lattice points with local x ≥ 0, the
    complement of the local x < 0 half the caster counts as uninitialized."""
    rig = raycast.rig()
    scene = raycast.spheres(load_seeds()["s9"]["spheres"], rig.train + rig.test)
    seed = load_seeds()["s9"]["lattice"]
    full = raycast.surface_samples(scene, seed, PER_SPHERE)
    half = raycast.surface_samples(scene, seed, PER_SPHERE, positive_x_only=True)
    centers = [torch.from_numpy(s.center_at(raycast.tau(0))) for s in scene]

    def local_x(points: InitPoints) -> torch.Tensor:
        first = points.positions[points.frame_ids == 0].double()
        near = torch.stack([torch.linalg.vector_norm(first - c, dim=1) for c in centers])
        return (first - torch.stack(centers)[near.argmin(dim=0)])[:, 0]

    assert bool((local_x(half) >= -1e-6).all())
    negatives = int((local_x(full) < -1e-6).sum())
    assert 0 < negatives and int((local_x(full) >= -1e-6).sum()) == local_x(half).numel()


# S-9 (§3.2): end-to-end quality on non-Gaussian data, through the command line.


def _view(record: ImageRecord) -> tuple[str, int]:
    """The caster's key for a held-out image: its camera and frame."""
    assert record.frame_index is not None
    return record.camera.name, record.frame_index


def _psnr_over(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """The PSNR of one image over the pixels of `mask`, after E16's clamp."""
    mse = (pred.clamp(0.0, 1.0).double() - gt.double())[mask].square().mean()
    return float(-10.0 * torch.log10(mse))


def _s9(
    scene: raycast.Scene,
    tmp_path: Path,
    record_property: Record,
    init: Path,
    prefix: str,
    started: float,
) -> None:
    """`started`: when the run began, init included, for its wall time."""
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(config_to_dict(_cfg())))
    run = tmp_path / "run"
    args = ["train", scene.train_manifest, "--init", init, "--out", run, "--config", config]
    assert main([str(a) for a in args]) == 0
    exchange = tmp_path / "model.safetensors"
    assert (
        main(["export", str(run / "checkpoint"), "--format", "ftgs-if", "--out", str(exchange)])
        == 0
    )
    video, tests = _test_records(scene)
    request = write_render_request(
        tmp_path / "request.json",
        [(str(i), video.records[i].timestamp, video.records[i].camera) for i in tests],
    )
    out = tmp_path / "renders.safetensors"
    assert main(["render", str(exchange), "--request", str(request), "--out", str(out)]) == 0
    renders = read_renders(out, request_sha256=hashlib.sha256(request.read_bytes()).hexdigest())
    pairs = [(renders.rgb[str(i)], video.load_image(i)) for i in tests]
    covered = [scene.coverage[_view(video.records[i])] for i in tests]
    psnrs = [float(psnr(pred, gt)) for pred, gt in pairs]
    ssims = [float(ssim_k(pred, gt, 1)) for pred, gt in pairs]  # Q-139
    ssims2 = [float(ssim_k(pred, gt, 2)) for pred, gt in pairs]  # Q-139: recorded only
    # Q-138: the PSNR over the pixels any ray of the caster hits a sphere in, the black
    # background left out (a sphere-coverage PSNR; the glossary keeps "foreground" off it).
    sphere_psnrs = [
        _psnr_over(pred, gt, torch.from_numpy(c > 0))
        for (pred, gt), c in zip(pairs, covered, strict=True)
    ]
    mean_psnr, mean_ssim = sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
    mean_spheres = sum(sphere_psnrs) / len(sphere_psnrs)
    rows = (run / "train_log.jsonl").read_text().splitlines()
    record_property(f"{prefix}_psnr", mean_psnr)
    record_property(f"{prefix}_ssim1", mean_ssim)
    record_property(f"{prefix}_ssim2", sum(ssims2) / len(ssims2))
    record_property(f"{prefix}_psnr_spheres", mean_spheres)  # Q-138
    record_property(f"{prefix}_coverage", float(np.mean([c.mean() for c in covered])))
    record_property(f"{prefix}_final_n", json.loads(rows[-1])["num_gaussians"])
    record_property(f"{prefix}_seconds", time.perf_counter() - started)
    print(
        f"{prefix}: held-out PSNR {mean_psnr:.2f} dB (spheres {mean_spheres:.2f}),"
        f" SSIM {mean_ssim:.4f}"
    )
    assert mean_psnr >= S9_PSNR
    assert mean_ssim >= S9_SSIM


@pytest.mark.gpu
def test_s9_textured_spheres_end_to_end_from_surface_samples(
    scene: raycast.Scene, tmp_path: Path, record_property: Record
) -> None:
    """S-9: 16 cameras, 20 frames, 128×96, initialized from surface samples (Q-140), trained,
    exported and rendered through ftgs: held-out PSNR ≥ 28 dB and SSIM ≥ 0.9 (calibrated)."""
    started = time.perf_counter()
    init = tmp_path / "init.safetensors"
    write_init_points(_full_init(scene), init, time_unit="s", producer="openftgs tests (S-9)")
    _s9(scene, tmp_path, record_property, init, "s9", started)


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not has_init_extra, reason="needs the [init] extra")
def test_s9_textured_spheres_end_to_end_from_roma(
    scene: raycast.Scene, tmp_path: Path, record_property: Record
) -> None:
    """S-9 with the RoMa initialization (§3.2: "or RoMa when [init] is installed"), made by
    ftgs init roma on the training images; the same criteria."""
    started = time.perf_counter()  # the matching included, as bcb5e70's row timed it
    init = tmp_path / "init.safetensors"
    assert main(["init", "roma", str(scene.train_manifest), str(init)]) == 0
    _s9(scene, tmp_path, record_property, init, "s9_roma", started)


# S-11 (§3.2): coverage of surfaces with no initial points.


def _masked_psnr(model: FreeTimeGaussians, scene: raycast.Scene) -> float:
    """Q-140: held-out PSNR over the pixels where at least half the rays hit a local x < 0 half,
    per image (E16's clamp), averaged over the images that show any."""
    video, tests = _test_records(scene)
    values: list[float] = []
    with torch.no_grad():
        for i in tests:
            record = video.records[i]
            mask = (torch.from_numpy(scene.halves[_view(record)]) >= 0.5).to("cuda")
            if not bool(mask.any()):
                continue
            out = render(model, record.camera, record.timestamp, backend="gsplat")
            values.append(_psnr_over(out.image[0], video.load_image(i).to("cuda"), mask))
    return sum(values) / len(values)


def _filled(
    trainer_points: InitPoints,
    video: MultiViewVideo,
    cfg: TrainConfig,
    n: int,
    scene: raycast.Scene,
) -> FreeTimeGaussians:
    """Q-140, run 3: the half init's model plus random low-opacity primitives up to N = n:
    uniform in the box of the spheres' trajectories, at uniform random times with the init's
    duration Δt_f, still, with the init's median scale, opacity 0.01 and random colors."""
    g = torch.Generator().manual_seed(load_seeds()["s11"]["filler"])
    model = FreeTimeGaussians.from_init_points(
        trainer_points,
        video.time,
        cfg.init,
        frame_dt=video.frame_dt,
        generator=g,
        sh_degree=cfg.sh_degree,
        n_max=cfg.growth.n_max,
    )
    model = model.to("cuda")
    count = n - model.num_gaussians
    assert count > 0
    tracks = np.array(
        [[s.center_at(raycast.tau(k)) for k in range(raycast.FRAMES)] for s in scene.spheres]
    ).reshape(-1, 3)
    low = torch.tensor(tracks.min(axis=0) - raycast.RADIUS, dtype=torch.float32)
    high = torch.tensor(tracks.max(axis=0) + raycast.RADIUS, dtype=torch.float32)
    sh_rest = model.shN.shape[1:]
    append = {
        "means": (low + (high - low) * torch.rand(count, 3, generator=g)),
        "times": torch.rand(count, generator=g),
        "log_durations": torch.full((count,), math.log(video.frame_dt)),
        "velocities": torch.zeros(count, 3),
        "log_scales": model.log_scales.detach().median(dim=0).values.cpu().expand(count, 3),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4),
        "opacity_logits": torch.full((count,), math.log(FILLER_OPACITY / (1 - FILLER_OPACITY))),
        "sh0": rgb_to_sh0(torch.rand(count, 3, generator=g))[:, None, :],
        "shN": torch.zeros(count, *sh_rest),
    }
    model.resize_(append={k: v.contiguous().to("cuda") for k, v in append.items()})
    return model


@pytest.mark.gpu
def test_s11_growth_covers_surfaces_with_no_initial_points(
    scene: raycast.Scene, record_property: Record
) -> None:
    """S-11: initialized on each sphere's local x ≥ 0 half only, growth "gradient" ends within
    1 dB of a fully initialized run in held-out PSNR on the uninitialized half (calibrated); the
    same metric is reported for growth "none" started at the same final N, the missing points
    replaced by random low-opacity primitives (Q-140)."""
    started = time.perf_counter()
    video = MultiViewVideo.from_manifest(scene.train_manifest)
    cfg = _cfg()
    full = Trainer(cfg, video, _full_init(scene), device="cuda")
    full.train()
    record_property("s11_seconds_full", time.perf_counter() - started)
    half_points = raycast.surface_samples(
        scene.spheres, load_seeds()["s9"]["lattice"], PER_SPHERE, positive_x_only=True
    )
    half = Trainer(cfg, video, half_points, device="cuda")
    half.train()
    p1, p2 = _masked_psnr(full.model, scene), _masked_psnr(half.model, scene)
    none_cfg = dataclasses.replace(cfg, growth=dataclasses.replace(cfg.growth, mode="none"))
    filled = _filled(half_points, video, none_cfg, half.model.num_gaussians, scene)
    still = Trainer(none_cfg, video, filled, device="cuda")
    still.train()
    p3 = _masked_psnr(still.model, scene)
    for name, value in (("full", p1), ("half_gradient", p2), ("half_none", p3)):
        record_property(f"s11_psnr_{name}", value)
    record_property("s11_seconds", time.perf_counter() - started)
    record_property(
        "s11_n",
        json.dumps([full.model.num_gaussians, half.model.num_gaussians, still.model.num_gaussians]),
    )
    print(
        f"S-11 uninitialized-half PSNR: full {p1:.2f}, half+growth {p2:.2f}, "
        f"half+filler {p3:.2f} dB; N {full.model.num_gaussians}, {half.model.num_gaussians}"
    )
    assert abs(p2 - p1) <= S11_WITHIN_DB


# S-5 (§3.2) on the ray-cast scene: the effect of E8 on dense textured surfaces (#100, D-86).


@dataclasses.dataclass(frozen=True)
class _S5Runs:
    opaque: dict[str, int]  # rows with σ > 0.9 at the end, per run
    rows: dict[str, int]  # N at the end, per run
    psnr: dict[str, float]  # held-out PSNR, E16 (criterion 2)
    psnr_spheres: dict[str, float]  # over the pixels any ray hits a sphere in (Q-138), recorded
    reg_term_reported: bool  # every iteration of the regularized run reported a nonzero L_reg
    reg_term_last: float  # the regularized run's L_reg at its last iteration
    seconds: float  # both runs, their held-out evaluations included


def _sphere_psnr(
    model: FreeTimeGaussians, video: MultiViewVideo, tests: list[int], scene: raycast.Scene
) -> float:
    """Q-138: the mean held-out PSNR over the pixels any ray of the caster hits a sphere in."""
    values: list[float] = []
    with torch.no_grad():
        for i in tests:
            record = video.records[i]
            out = render(model, record.camera, record.timestamp, backend="gsplat")
            mask = torch.from_numpy(scene.coverage[_view(record)] > 0).to("cuda")
            values.append(_psnr_over(out.image[0], video.load_image(i).to("cuda"), mask))
    return sum(values) / len(values)


@pytest.fixture(scope="module")
def s5_runs(scene: raycast.Scene) -> _S5Runs:
    """D-86: S-5's two runs on S-9's scene and initialization as they stand (N = 18,000), growth
    "none" so N stays the same in both, relocation at its default, S-9's calendar, and
    λ_reg = 0 against 10⁻² (E8, D-10) the only difference."""
    started = time.perf_counter()
    video = MultiViewVideo.from_manifest(scene.train_manifest)
    scene_video, tests = _test_records(scene)  # both splits, for the held-out images
    opaque: dict[str, int] = {}
    rows: dict[str, int] = {}
    psnrs: dict[str, float] = {}
    spheres: dict[str, float] = {}
    reported, last = False, 0.0
    for name, lambda_reg in (("no_reg", 0.0), ("reg", 1e-2)):
        # D-86: growth "none" keeps N the same in both runs; λ_reg is the only difference.
        cfg = _cfg(loss=LossConfig(lambda_reg=lambda_reg), growth=GrowthConfig(mode="none"))
        trainer, logs = train(cfg, video, _full_init(scene), "cuda")
        model = trainer.model
        opaque[name] = int((torch.sigmoid(model.opacity_logits.detach()) > 0.9).sum())  # §1.1, D-5
        rows[name] = model.num_gaussians
        psnrs[name] = mean_psnr(model, scene_video, tests)
        spheres[name] = _sphere_psnr(model, scene_video, tests, scene)
        if lambda_reg > 0.0:
            reported = all("reg" in log.losses and float(log.losses["reg"]) > 0.0 for log in logs)
            last = float(logs[-1].losses["reg"]) if reported else 0.0
    return _S5Runs(
        opaque=opaque,
        rows=rows,
        psnr=psnrs,
        psnr_spheres=spheres,
        reg_term_reported=reported,
        reg_term_last=last,
        seconds=time.perf_counter() - started,
    )


@pytest.mark.gpu
def test_s5_raycast_regularization_keeps_fewer_primitives_opaque(
    s5_runs: _S5Runs, record_property: Record
) -> None:
    """S-5 on the ray-cast scene (D-86): at the end of training the share of primitives with
    σ > 0.9 is lower with λ_reg = 10⁻² than without, both runs keeping the same N (growth
    "none"); the regularized run reports a nonzero L_reg at every iteration, so E8 entered its
    loss."""
    runs = s5_runs
    for name in ("no_reg", "reg"):
        record_property(f"s5_raycast_{name}_opaque", runs.opaque[name])
        record_property(f"s5_raycast_{name}_rows", runs.rows[name])
    record_property("s5_raycast_reg_term_last", runs.reg_term_last)
    record_property("s5_raycast_seconds", runs.seconds)
    print(
        f"s5 raycast: σ > 0.9 in {runs.opaque['no_reg']} and {runs.opaque['reg']} of"
        f" {runs.rows['no_reg']} and {runs.rows['reg']} rows"
    )
    assert runs.rows["no_reg"] == runs.rows["reg"]
    assert runs.reg_term_reported and runs.reg_term_last > 0.0
    assert runs.opaque["reg"] / runs.rows["reg"] < runs.opaque["no_reg"] / runs.rows["no_reg"]


@pytest.mark.gpu
def test_s5_raycast_regularization_costs_at_most_half_a_db(
    s5_runs: _S5Runs, record_property: Record
) -> None:
    """S-5 on the ray-cast scene (D-86): the final held-out PSNR with λ_reg = 10⁻² is not lower
    than without by more than 0.5 dB; the sphere-coverage PSNR is recorded beside it (Q-138)."""
    runs = s5_runs
    for name in ("no_reg", "reg"):
        record_property(f"s5_raycast_{name}_psnr", runs.psnr[name])
        record_property(f"s5_raycast_{name}_psnr_spheres", runs.psnr_spheres[name])
    print(
        f"s5 raycast: held-out PSNR {runs.psnr['no_reg']:.2f} and {runs.psnr['reg']:.2f} dB"
        f" (spheres {runs.psnr_spheres['no_reg']:.2f} and {runs.psnr_spheres['reg']:.2f})"
    )
    assert runs.psnr["reg"] >= runs.psnr["no_reg"] - S5_MAX_COST_DB
