# SPDX-License-Identifier: Apache-2.0
"""§3.2's end-to-end scenes on non-Gaussian data (M14, PRD #157): S-9 through the command line
(ftgs train, export and render; ftgs init roma when the [init] extra is installed) and S-11's
coverage of surfaces with no initial points, both on the textured spheres of the analytic ray
caster in `raycast.py` (Q-138 to Q-140). Both train on the GPU: on the reference backend their
320 images at 128×96 would far exceed Q-119's 3-minute CPU target, as S-5 did (Q-138).

Each test records its measured values with `record_property`, for the Calibration data table of
docs/clean-room/questions.md; the thresholds are §3.2's and never loosened there.
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import raycast
import torch
from builders import look_at
from synthetic_scenes import load_seeds, synthetic_config

from openftgs.cli import main
from openftgs.config import TrainConfig, config_to_dict
from openftgs.data import MultiViewVideo
from openftgs.eval import psnr, ssim_k
from openftgs.io import read_renders, write_init_points
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.sh import rgb_to_sh0
from openftgs.train import Trainer
from openftgs.types import InitPoints

ITERATIONS = 7_000  # Q-140: set during the M14 build and recorded
PER_SPHERE = 300  # Q-140: lattice points per sphere and frame, 900 per frame (B = 1,000)
S9_PSNR, S9_SSIM = 28.0, 0.9  # §3.2 S-9 (calibrated)
S11_WITHIN_DB = 1.0  # §3.2 S-11 (calibrated)
FILLER_OPACITY = 0.01  # Q-140: S-11's random low-opacity primitives

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
    surface, within the texture's slope over that pixel; seen from below, the hit pixels show
    the local z < 0 half, from above the z ≥ 0 half."""
    rig = raycast.rig()
    scene = raycast.spheres(load_seeds()["s9"]["spheres"], rig.train + rig.test)
    for camera in (rig.train[0], rig.train[9], rig.test[2]):
        w2c, K = camera.world_to_camera.double().numpy(), camera.K.double().numpy()
        eye = -w2c[:3, :3].T @ w2c[:3, 3]
        for sphere in scene:
            image, _ = raycast.cast([sphere], camera, 0.0, supersample=1)
            center = sphere.center_at(0.0)
            pix = K @ (w2c[:3, :3] @ center + w2c[:3, 3])
            u, v = math.floor(pix[0] / pix[2]), math.floor(pix[1] / pix[2])
            toward = (eye - center) / np.linalg.norm(eye - center)
            # A pixel is about 0.014 world units at this depth, 0.12 rad of the local direction,
            # over which the texture (slope ≤ 0.4 · 5 per rad) changes by at most 0.24.
            assert np.abs(image[v, u] - sphere.color(toward)).max() < 0.24
    # From straight below a sphere every visible point has local z < 0, from above z > 0.
    center = scene[0].center_at(0.0)
    for side, lower in ((-1.0, True), (1.0, False)):
        eye = tuple(float(c) for c in center + np.array([0.0, 0.0, side]))
        camera = look_at(eye, target=tuple(center), up=(0, 1, 0), width=128, height=96, focal=64.0)
        image, half = raycast.cast(scene[:1], camera, 0.0)
        hit = image.sum(axis=-1) > 0
        assert hit.sum() > 100
        # `half` is the share of all of a pixel's rays, so a silhouette pixel shows below 1.
        assert np.array_equal(half > 0, hit) if lower else not half.any()
        if lower:
            assert float(np.median(half[hit])) == 1.0


# S-9 (§3.2): end-to-end quality on non-Gaussian data, through the command line.


def _s9(
    scene: raycast.Scene, tmp_path: Path, record_property: Record, init: Path, prefix: str
) -> None:
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
    items = [
        {
            "id": str(i),
            "timestamp": video.records[i].timestamp,
            "camera": {
                "width": video.records[i].camera.width,
                "height": video.records[i].camera.height,
                "K": video.records[i].camera.K.tolist(),
                "world_to_camera": video.records[i].camera.world_to_camera.tolist(),
            },
        }
        for i in tests
    ]
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"format": "ftgs-render-request", "version": 1, "time_unit": "s", "items": items}
        )
    )
    out = tmp_path / "renders.safetensors"
    assert main(["render", str(exchange), "--request", str(request), "--out", str(out)]) == 0
    renders = read_renders(out, request_sha256=hashlib.sha256(request.read_bytes()).hexdigest())
    psnrs = [float(psnr(renders.rgb[str(i)], video.load_image(i))) for i in tests]
    ssims = [float(ssim_k(renders.rgb[str(i)], video.load_image(i), 1)) for i in tests]  # Q-139
    mean_psnr, mean_ssim = sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
    rows = (run / "train_log.jsonl").read_text().splitlines()
    record_property(f"{prefix}_psnr", mean_psnr)
    record_property(f"{prefix}_ssim1", mean_ssim)
    record_property(f"{prefix}_final_n", json.loads(rows[-1])["num_gaussians"])
    print(f"{prefix}: held-out PSNR {mean_psnr:.2f} dB, SSIM {mean_ssim:.4f}")
    assert mean_psnr >= S9_PSNR
    assert mean_ssim >= S9_SSIM


@pytest.mark.gpu
def test_s9_textured_spheres_end_to_end_from_surface_samples(
    scene: raycast.Scene, tmp_path: Path, record_property: Record
) -> None:
    """S-9: 16 cameras, 20 frames, 128×96, initialized from surface samples (Q-140), trained,
    exported and rendered through ftgs: held-out PSNR ≥ 28 dB and SSIM ≥ 0.9 (calibrated)."""
    init = tmp_path / "init.safetensors"
    write_init_points(_full_init(scene), init, time_unit="s", producer="openftgs tests (S-9)")
    _s9(scene, tmp_path, record_property, init, "s9")


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not has_init_extra, reason="needs the [init] extra")
def test_s9_textured_spheres_end_to_end_from_roma(
    scene: raycast.Scene, tmp_path: Path, record_property: Record
) -> None:
    """S-9 with the RoMa initialization (§3.2: "or RoMa when [init] is installed"), made by
    ftgs init roma on the training images; the same criteria."""
    init = tmp_path / "init.safetensors"
    assert main(["init", "roma", str(scene.train_manifest), str(init)]) == 0
    _s9(scene, tmp_path, record_property, init, "s9_roma")


# S-11 (§3.2): coverage of surfaces with no initial points.


def _masked_psnr(model: FreeTimeGaussians, scene: raycast.Scene) -> float:
    """Q-140: held-out PSNR over the pixels where at least half the rays hit a local z < 0 half,
    per image (E16's clamp), averaged over the images that show any."""
    video, tests = _test_records(scene)
    values: list[float] = []
    with torch.no_grad():
        for i in tests:
            record = video.records[i]
            mask = torch.from_numpy(scene.halves[(record.camera.name, record.frame_index or 0)])
            mask = (mask >= 0.5).to("cuda")
            if not bool(mask.any()):
                continue
            out = render(model, record.camera, record.timestamp, backend="gsplat")
            pred = out.image[0].clamp(0.0, 1.0).double()
            gt = video.load_image(i).to("cuda").double()
            mse = (pred - gt)[mask].square().mean()
            values.append(float(-10.0 * torch.log10(mse)))
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
    """S-11: initialized on each sphere's local z ≥ 0 half only, growth "gradient" ends within
    1 dB of a fully initialized run in held-out PSNR on the uninitialized half (calibrated); the
    same metric is reported for growth "none" started at the same final N, the missing points
    replaced by random low-opacity primitives (Q-140)."""
    video = MultiViewVideo.from_manifest(scene.train_manifest)
    cfg = _cfg()
    full = Trainer(cfg, video, _full_init(scene), device="cuda")
    full.train()
    half_points = raycast.surface_samples(
        scene.spheres, load_seeds()["s9"]["lattice"], PER_SPHERE, upper_only=True
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
    record_property(
        "s11_n",
        json.dumps([full.model.num_gaussians, half.model.num_gaussians, still.model.num_gaussians]),
    )
    print(
        f"S-11 uninitialized-half PSNR: full {p1:.2f}, half+growth {p2:.2f}, "
        f"half+filler {p3:.2f} dB; N {full.model.num_gaussians}, {half.model.num_gaussians}"
    )
    assert abs(p2 - p1) <= S11_WITHIN_DB
