# SPDX-License-Identifier: Apache-2.0
"""Shared setup of the §3.2 synthetic-scene recovery tests (M11, PRD #99): the camera rigs, the
ground truth drawn from committed seeds (stored as ftgs-if files in `data/`), the initial models
derived from it, the scenes written to disk, the training runs and the recovery measures.

The ground truth lives in raw time τ, with frames at τ = k/30 s. A model is rebuilt under each
dataset's own time normalizer (`model_under`), since the trainer refuses a model whose normalizer
differs from the dataset's (Q-69). Rows are matched to the ground truth by index (Q-80).
"""

import dataclasses
import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

import torch
from builders import C0, look_at, ring, shortened_config, write_scene
from torch import Tensor

from openftgs.config import TrainConfig
from openftgs.data import MultiViewVideo
from openftgs.eval import psnr
from openftgs.init import knn_velocities
from openftgs.io import read_exchange
from openftgs.model import FreeTimeGaussians, rotation_matrix
from openftgs.render import render
from openftgs.train import StepLog, Trainer
from openftgs.types import Camera, ExchangeModel, InitPoints, TimeNormalizer

DATA = Path(__file__).parent / "data"
SEEDS_FILE = DATA / "seeds.json"
FPS = 30.0
F64 = torch.float64
F32 = torch.float32

SIZE = 64  # Q-79: 64×64 images, the low end of §3.2's sizes
EDGE_MARGIN = 4.0  # pixels between a trajectory point and the image edge (Q-77)
NEAREST_DEPTH = 0.2  # every trajectory point at least this far in front of every camera
SCALE_RANGE = (0.02, 0.035)  # S-2 family and S-3 scales (Q-79)
SEPARATION = 4.0 * SCALE_RANGE[1]  # Q-79: trajectories at least 4 × the largest scale apart
DEAD_CLEARANCE = 5.0 * SCALE_RANGE[1]  # Q-78: dead rows of S-6 away from every trajectory
SPEED_RANGE = (0.125, 0.5)  # Q-77: §3.2's 0.5–2 per unit t, × ¼
CENTER_RADIUS = 0.2  # Q-79: S-2 centers at t = ½ within 0.2 of the origin
VOLUME = 0.35  # half-width of the cube that holds every trajectory: the view volume


def tau(frame: float) -> float:
    """The raw timestamp of a (possibly fractional) frame."""
    return frame / FPS


def backend_for(device: str) -> Literal["reference", "gsplat"]:
    """§3.2: the reference backend on the CPU, gsplat on the GPU (the trainer's rule, Q-69)."""
    return "gsplat" if device == "cuda" else "reference"


# Rigs (§3.2 shared setup; Q-79).


@dataclasses.dataclass(frozen=True)
class Rig:
    train: list[Camera]
    test: list[Camera]


def ring_rig() -> Rig:
    """S-1: S-10's ring of eight 64×64 cameras (radius 0.9, elevation 0.3, so S = 0.99) and one
    held-out ring camera between two of them, at another elevation."""
    train = ring(8, radius=0.9, elevation=0.3, width_px=SIZE, height_px=SIZE, focal=64.0)
    angle = 2 * math.pi / 16
    held_out = look_at(
        (0.9 * math.cos(angle), 0.9 * math.sin(angle), 0.5),  # Q-79
        width=SIZE,
        height=SIZE,
        focal=64.0,
        name="heldout00",
    )
    return Rig(train, [held_out])


# Q-79: the hemisphere rig.
_TRAIN_ELEVATIONS = (20.0, 40.0, 60.0)  # degrees; four azimuths each
_TEST_VIEWS = ((30.0, 45.0), (50.0, 135.0), (25.0, 225.0), (55.0, 315.0))  # (elev, azim)
HEMISPHERE_FOCAL = 56.0  # Q-77, Q-79: every trajectory in view


def _on_sphere(radius: float, elevation: float, azimuth: float) -> tuple[float, float, float]:
    e, a = math.radians(elevation), math.radians(azimuth)
    return (
        radius * math.cos(e) * math.cos(a),
        radius * math.cos(e) * math.sin(a),
        radius * math.sin(e),
    )


def _hemisphere_directions() -> list[tuple[float, float]]:
    return [
        (elevation, 90.0 * k + 30.0 * j)
        for j, elevation in enumerate(_TRAIN_ELEVATIONS)
        for k in range(4)
    ]


def hemisphere_radius() -> float:
    """The radius that gives S = 1 (D-27: 1.1 × the largest distance of a training camera
    center from their centroid) for the twelve training directions."""
    unit = torch.tensor([_on_sphere(1.0, e, a) for e, a in _hemisphere_directions()], dtype=F64)
    spread = float(torch.linalg.vector_norm(unit - unit.mean(dim=0), dim=1).max())
    return 1.0 / (1.1 * spread)


def hemisphere_rig() -> Rig:
    """The S-2 family and S-3: twelve 64×64 training cameras on a hemisphere at elevations
    20°, 40° and 60°, and four held-out ones between them, all looking at the origin, with the
    radius that gives S = 1 (Q-79)."""
    radius = hemisphere_radius()

    def camera(elevation: float, azimuth: float, name: str) -> Camera:
        eye = _on_sphere(radius, elevation, azimuth)
        return look_at(eye, width=SIZE, height=SIZE, focal=HEMISPHERE_FOCAL, name=name)

    train = [camera(e, a, f"hemi{k:02d}") for k, (e, a) in enumerate(_hemisphere_directions())]
    test = [camera(e, a, f"heldout{k:02d}") for k, (e, a) in enumerate(_TEST_VIEWS)]
    return Rig(train, test)


def project(cameras: Sequence[Camera], points: Tensor) -> tuple[Tensor, Tensor]:
    """Pixel coordinates [C, P, 2] and depths [C, P] of world points [P, 3] (§1.4, float64)."""
    w2c = torch.stack([c.world_to_camera for c in cameras]).to(F64)
    K = torch.stack([c.K for c in cameras]).to(F64)
    cam = points.to(F64) @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, None, :3, 3]
    depth = cam[..., 2]
    pix = cam @ K.transpose(1, 2)
    return pix[..., :2] / pix[..., 2:], depth


def in_view(cameras: Sequence[Camera], points: Tensor) -> bool:
    """Every point lies in front of every camera and inside its image by EDGE_MARGIN pixels."""
    uv, depth = project(cameras, points)
    width, height = cameras[0].width, cameras[0].height
    inside = (
        (uv[..., 0] >= EDGE_MARGIN)
        & (uv[..., 0] <= width - EDGE_MARGIN)
        & (uv[..., 1] >= EDGE_MARGIN)
        & (uv[..., 1] <= height - EDGE_MARGIN)
    )
    return bool((inside & (depth >= NEAREST_DEPTH)).all())


# Ground truth (§3.2: "Ground-truth files and seeds are committed").


def load_seeds() -> dict[str, dict[str, int]]:
    seeds: dict[str, dict[str, int]] = json.loads(SEEDS_FILE.read_text())
    return seeds


def _uniform(g: torch.Generator, low: float, high: float, *shape: int) -> Tensor:
    return low + (high - low) * torch.rand(*shape, generator=g, dtype=F64)


def _rotations(g: torch.Generator, n: int) -> Tensor:
    """Uniform random unit quaternions with the canonical sign (w ≥ 0, ftgs-if)."""
    q = torch.randn(n, 4, generator=g, dtype=F64)
    q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


def _log_uniform(g: torch.Generator, low: float, high: float, *shape: int) -> Tensor:
    return torch.exp(_uniform(g, math.log(low), math.log(high), *shape))


def sh_of(color: Tensor) -> Tensor:
    """sh [N, 1, 3] giving the RGB color at degree 0 (E2: C₀·sh0 + ½)."""
    return ((color - 0.5) / C0).reshape(-1, 1, 3)


def exchange(
    arrays: dict[str, Tensor],
    *,
    tau0: float,
    tau1: float,
    degree: int,
    producer: str = "openftgs tests (M11 ground truth)",
) -> ExchangeModel:
    """ftgs-if contents in the dtypes §2.2 fixes, from float64 arrays."""
    dtypes = {"time_center": F64, "duration": F64}
    return ExchangeModel(
        arrays={k: v.to(dtypes.get(k, F32)) for k, v in arrays.items()},
        sh_degree=degree,
        tau0=tau0,
        tau1=tau1,
        time_unit="s",
        producer=producer,
    )


def _gt_s1(seed: int) -> ExchangeModel:
    """S-1: one static anisotropic Gaussian near the origin, σ = 0.7, a non-gray color, scales
    0.08–0.12, s = 10 in normalized time, over five frames (Q-79)."""
    g = torch.Generator().manual_seed(seed)
    span = tau(4)
    arrays = {
        "position": _uniform(g, -0.03, 0.03, 1, 3),
        "time_center": torch.tensor([tau(2)], dtype=F64),
        "duration": torch.tensor([10.0 * span], dtype=F64),
        "velocity": torch.zeros(1, 3, dtype=F64),
        "scale": _uniform(g, 0.08, 0.12, 1, 3),
        "rotation": _rotations(g, 1),
        "opacity": torch.tensor([0.7], dtype=F64),
        "sh": sh_of(torch.tensor([[0.8, 0.45, 0.25]], dtype=F64)),
    }
    return exchange(arrays, tau0=0.0, tau1=span, degree=0)


def frame_taus(count: int = 30) -> Tensor:
    return torch.arange(count, dtype=F64) / FPS


def trajectory_points(
    position: Tensor, velocity: Tensor, time_center: Tensor, taus: Tensor
) -> Tensor:
    """μx(τ) of E1 in raw units, [N, T, 3] in float64."""
    dt = taus.to(F64)[None, :, None] - time_center.to(F64)[:, None, None]
    return position.to(F64)[:, None, :] + velocity.to(F64)[:, None, :] * dt


def _gt_s2(seed: int) -> ExchangeModel:
    """S-2: 8 Gaussians moving linearly, |v| in [0.125, 0.5] per unit t (Q-77) in uniform random
    directions, s = 2 (always present), time centers uniform over the sequence, centers at t = ½
    within CENTER_RADIUS of the origin; each drawn again until its trajectory stays in view of
    every camera of the hemisphere rig and at least SEPARATION from the others' (Q-79)."""
    g = torch.Generator().manual_seed(seed)
    span = tau(29)
    taus = frame_taus()
    rig = hemisphere_rig()
    cameras = rig.train + rig.test
    drawn: list[dict[str, Tensor]] = []
    tracks: list[Tensor] = []
    while len(drawn) < 8:
        center = _uniform(g, -CENTER_RADIUS, CENTER_RADIUS, 3)
        direction = torch.randn(3, generator=g, dtype=F64)
        direction = direction / torch.linalg.vector_norm(direction)
        speed_t = _uniform(g, *SPEED_RANGE, 1)
        velocity = direction * speed_t / span  # world units per τ
        time_center = _uniform(g, 0.0, span, 1)
        row = {
            "position": center + velocity * (time_center - span / 2),
            "time_center": time_center,
            "duration": torch.tensor([2.0 * span], dtype=F64),
            "velocity": velocity,
            "scale": _log_uniform(g, *SCALE_RANGE, 3),
            "rotation": _rotations(g, 1)[0],
            "opacity": _uniform(g, 0.6, 0.9, 1),
            "sh": sh_of(_uniform(g, 0.2, 0.9, 3))[0],
        }
        if float(torch.linalg.vector_norm(center)) > CENTER_RADIUS:
            continue
        track = trajectory_points(row["position"][None], velocity[None], time_center, taus)[0]
        if not in_view(cameras, track):
            continue
        if any(
            float(torch.linalg.vector_norm(track - t, dim=1).min()) < SEPARATION for t in tracks
        ):
            continue
        drawn.append(row)
        tracks.append(track)
    arrays = {k: torch.stack([row[k] for row in drawn]) for k in drawn[0]}
    for k in ("time_center", "duration", "opacity"):
        arrays[k] = arrays[k].reshape(-1)
    return exchange(arrays, tau0=0.0, tau1=span, degree=0)


def _gt_s3(seed: int) -> ExchangeModel:
    """S-3: 12 static Gaussians with s_gt = 3 frames and time centers uniform over frames 3–26,
    in view of every hemisphere camera and at least SEPARATION apart (Q-79)."""
    g = torch.Generator().manual_seed(seed)
    span = tau(29)
    rig = hemisphere_rig()
    cameras = rig.train + rig.test
    positions: list[Tensor] = []
    while len(positions) < 12:
        p = _uniform(g, -0.3, 0.3, 3)
        if float(torch.linalg.vector_norm(p)) > 0.3 or not in_view(cameras, p[None]):
            continue
        if any(float(torch.linalg.vector_norm(p - q)) < SEPARATION for q in positions):
            continue
        positions.append(p)
    n = 12
    arrays = {
        "position": torch.stack(positions),
        "time_center": _uniform(g, tau(3), tau(26), n),
        "duration": torch.full((n,), tau(3), dtype=F64),
        "velocity": torch.zeros(n, 3, dtype=F64),
        "scale": _log_uniform(g, *SCALE_RANGE, n, 3),
        "rotation": _rotations(g, n),
        "opacity": _uniform(g, 0.6, 0.9, n),
        "sh": sh_of(_uniform(g, 0.2, 0.9, n, 3)),
    }
    return exchange(arrays, tau0=0.0, tau1=span, degree=0)


# S-4 (Q-119): three spheres translating linearly over 10 frames, on the hemisphere rig. §3.2's
# precondition, a displacement per frame below half the sampling spacing, binds the init
# lattice: 14 points per sphere leave a spacing of about 0.08, so a step of up to 0.025
# (about 1.5 px at the rig's focal length) stays below 0.63 of half of it.
S4_FRAMES = 10
S4_RADIUS = 0.1  # about 12 px across at the hemisphere rig's focal length and radius
S4_GT_POINTS = 80  # ground-truth Gaussians per sphere, on a Fibonacci lattice
S4_GT_SCALE = 0.45  # their isotropic scale, × the lattice's smallest spacing (about 1 px)
S4_GT_OPACITY = 0.9
S4_GT_DURATION = 10.0  # long durations (S-1's s = 10): present at every frame
S4_INIT_POINTS = 14  # init points per sphere and frame, on a rotated, jittered lattice
S4_INIT_JITTER = 0.05  # the largest jitter of an init point, × the lattice's smallest spacing
S4_STEP = (0.018, 0.025)  # displacement per frame (|v|/30 per τ), in world units
S4_CENTER_RADIUS = 0.2  # sphere centers at the middle time within this of the origin
S4_CLEARANCE = 0.06  # surfaces this far apart at every frame: more than twice the largest step,
# so a sphere's points can't find their k-NN partner on another sphere


def fibonacci_sphere(n: int) -> Tensor:
    """n points [n, 3] of the Fibonacci lattice on the unit sphere, in float64: z_i = 1 −
    (2i + 1)/n and azimuth i·π(3 − √5) (the golden angle)."""
    i = torch.arange(n, dtype=F64)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = torch.sqrt(1.0 - z * z)
    azimuth = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(azimuth), r * torch.sin(azimuth), z], dim=1)


def smallest_spacing(points: Tensor) -> float:
    """The smallest distance between two of `points` [n, 3] (n ≥ 2), in float64."""
    d = torch.cdist(points.to(F64), points.to(F64))
    d.fill_diagonal_(math.inf)
    return float(d.min())


def _gt_s4(seed: int) -> ExchangeModel:
    """S-4 (Q-119): three spheres of radius S4_RADIUS translating linearly, each S4_GT_POINTS
    Gaussians on a Fibonacci lattice that share its velocity, with isotropic scales, opacity
    0.9, long durations, time centers at the middle frame and uniform random colors in
    [0.2, 0.9] (a texture that shows the motion). A sphere's center at the middle time lies
    within S4_CENTER_RADIUS of the origin and its step per frame is uniform in S4_STEP, in a
    uniform random direction; each is drawn again until its surface stays in view of every
    hemisphere camera at every frame and S4_CLEARANCE from the others'. Rows are sphere by
    sphere, S4_GT_POINTS each."""
    g = torch.Generator().manual_seed(seed)
    taus = frame_taus(S4_FRAMES)
    span = tau(S4_FRAMES - 1)
    middle = span / 2
    rig = hemisphere_rig()
    cameras = rig.train + rig.test
    unit = fibonacci_sphere(S4_GT_POINTS)
    centers: list[Tensor] = []
    velocities: list[Tensor] = []
    tracks: list[Tensor] = []
    while len(centers) < 3:
        center = _uniform(g, -S4_CENTER_RADIUS, S4_CENTER_RADIUS, 3)
        direction = torch.randn(3, generator=g, dtype=F64)
        direction = direction / torch.linalg.vector_norm(direction)
        velocity = direction * _uniform(g, *S4_STEP, 1) * FPS  # world units per τ
        if float(torch.linalg.vector_norm(center)) > S4_CENTER_RADIUS:
            continue
        track = center + velocity * (taus[:, None] - middle)  # [frames, 3]
        surface = (track[:, None, :] + S4_RADIUS * unit).reshape(-1, 3)
        if not in_view(cameras, surface):
            continue
        apart = 2.0 * S4_RADIUS + S4_CLEARANCE
        if any(float(torch.linalg.vector_norm(track - t, dim=1).min()) < apart for t in tracks):
            continue
        centers.append(center)
        velocities.append(velocity)
        tracks.append(track)
    n = 3 * S4_GT_POINTS
    scale = S4_GT_SCALE * S4_RADIUS * smallest_spacing(unit)
    arrays = {
        "position": torch.cat([c + S4_RADIUS * unit for c in centers]),
        "time_center": torch.full((n,), middle, dtype=F64),
        "duration": torch.full((n,), S4_GT_DURATION * span, dtype=F64),
        "velocity": torch.cat([v.expand(S4_GT_POINTS, 3) for v in velocities]),
        "scale": torch.full((n, 3), scale, dtype=F64),
        "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F64).repeat(n, 1),
        "opacity": torch.full((n,), S4_GT_OPACITY, dtype=F64),
        "sh": sh_of(_uniform(g, 0.2, 0.9, n, 3)),
    }
    return exchange(
        arrays, tau0=0.0, tau1=span, degree=0, producer="openftgs tests (M13 ground truth)"
    )


CLUTTER_COUNT = 2_000
CLUTTER_SCALES = (0.005, 0.015)  # Q-78: smaller than the objects', the "small" Gaussians
CLUTTER_OPACITY = (0.3, 0.99)  # Q-78: up to about 1, so an unregularized fit can saturate


def _gt_clutter(seed: int, gt: ExchangeModel) -> ExchangeModel:
    """S-5's 2,000 small random Gaussians (Q-78): static (v = 0) and always present (s = 2 in
    normalized time, as S-2's objects), uniform in the view volume, with uniform time centers,
    log-uniform scales in CLUTTER_SCALES, uniform opacities in CLUTTER_OPACITY and colors."""
    g = torch.Generator().manual_seed(seed)
    span = gt.tau1 - gt.tau0
    n = CLUTTER_COUNT
    arrays = {
        "position": _uniform(g, -VOLUME, VOLUME, n, 3),
        "time_center": _uniform(g, 0.0, span, n),
        "duration": torch.full((n,), 2.0 * span, dtype=F64),
        "velocity": torch.zeros(n, 3, dtype=F64),
        "scale": _log_uniform(g, *CLUTTER_SCALES, n, 3),
        "rotation": _rotations(g, n),
        "opacity": _uniform(g, *CLUTTER_OPACITY, n),
        "sh": sh_of(_uniform(g, 0.2, 0.9, n, 3)),
    }
    return exchange(arrays, tau0=gt.tau0, tau1=gt.tau1, degree=0)


CLUSTER_SIZE = 3  # Q-83: each S-2 object and two companions
CLUSTER_OFFSET = 1.5  # Q-83: companions this many of the object's largest scales away
CLUSTER_SCALE = 0.8  # Q-83: a companion's scales, as a fraction of the object's


def _gt_s6(seed: int, gt: ExchangeModel) -> ExchangeModel:
    """S-6's under-fitted ground truth (Q-83): each of S-2's 8 objects followed by two companions
    that share its motion, time, duration, rotation and opacity, offset in a uniform random
    direction by CLUSTER_OFFSET × its largest scale, with CLUSTER_SCALE × its scales and a
    random color. At focal length 56 px and radius 0.93 a scale of 0.035 is about 2 px, so a
    cluster spans a few pixels that one primitive can't fit: a capacity deficit relocation can
    fill."""
    g = torch.Generator().manual_seed(seed)
    a = gt.arrays
    drawn: list[dict[str, Tensor]] = []
    for i in range(a["position"].shape[0]):
        base = {k: v[i].to(F64) for k, v in a.items()}
        drawn.append(base)
        for _ in range(CLUSTER_SIZE - 1):
            d = torch.randn(3, generator=g, dtype=F64)
            d = d / torch.linalg.vector_norm(d)
            companion = dict(base)
            companion["position"] = base["position"] + d * CLUSTER_OFFSET * float(
                base["scale"].max()
            )
            companion["scale"] = base["scale"] * CLUSTER_SCALE
            companion["sh"] = sh_of(_uniform(g, 0.2, 0.9, 3))[0]
            drawn.append(companion)
    arrays = {k: torch.stack([row[k] for row in drawn]) for k in drawn[0]}
    return exchange(arrays, tau0=gt.tau0, tau1=gt.tau1, degree=0)


GROUND_TRUTH_FILES = {
    "s1": "s1_gt.safetensors",
    "s2": "s2_gt.safetensors",
    "s3": "s3_gt.safetensors",
    "s4": "s4_gt.safetensors",
    "s5": "s5_gt.safetensors",
    "s6": "s6_gt.safetensors",
}


def generate_ground_truth(seeds: dict[str, dict[str, int]]) -> dict[str, ExchangeModel]:
    """The ground truth of S-1 to S-6 from the committed seeds: S-4's is three spheres of
    Gaussians (Q-119); S-5's is S-2's 8 Gaussians (its first rows) followed by 2,000 small ones
    (Q-78); S-6's is S-2's objects as clusters of three (Q-83); S-7 and S-8 reuse S-2's."""
    s2 = _gt_s2(seeds["s2"]["ground_truth"])
    return {
        "s1": _gt_s1(seeds["s1"]["ground_truth"]),
        "s2": s2,
        "s3": _gt_s3(seeds["s3"]["ground_truth"]),
        "s4": _gt_s4(seeds["s4"]["ground_truth"]),
        "s5": concatenated(s2, _gt_clutter(seeds["s5"]["clutter"], s2)),
        "s6": _gt_s6(seeds["s6"]["ground_truth"], s2),
    }


def same_exchange(a: ExchangeModel, b: ExchangeModel) -> bool:
    """Whether two exchange models hold the same header fields and bitwise-equal arrays."""
    fields = ("sh_degree", "tau0", "tau1", "time_unit", "producer")
    return (
        all(getattr(a, f) == getattr(b, f) for f in fields)
        and a.arrays.keys() == b.arrays.keys()
        and all(torch.equal(a.arrays[k], b.arrays[k]) for k in a.arrays)
    )


def ground_truth(scene: str) -> ExchangeModel:
    """A committed ground-truth file (§3.2)."""
    return read_exchange(DATA / GROUND_TRUTH_FILES[scene])


# Models under a dataset's time normalizer, and the initial models (Q-79).


def with_degree(model: ExchangeModel, degree: int) -> ExchangeModel:
    """The same model with `sh` padded with zeros (or cut) to `degree` bands."""
    sh = model.arrays["sh"]
    count = (degree + 1) ** 2
    if sh.shape[1] < count:
        sh = torch.cat([sh, sh.new_zeros(sh.shape[0], count - sh.shape[1], 3)], dim=1)
    arrays = dict(model.arrays, sh=sh[:, :count].contiguous())
    return dataclasses.replace(model, arrays=arrays, sh_degree=degree)


def model_under(
    model: ExchangeModel, time: TimeNormalizer, *, sh_degree: int, device: str = "cpu"
) -> FreeTimeGaussians:
    """The model's raw quantities as a FreeTimeGaussians under `time` (a dataset's normalizer)."""
    framed = dataclasses.replace(with_degree(model, sh_degree), tau0=time.tau0, tau1=time.tau1)
    return FreeTimeGaussians.from_exchange(framed, device=device)


def replaced(model: ExchangeModel, **arrays: Tensor) -> ExchangeModel:
    """The model with some arrays replaced, cast to the dtypes of the originals."""
    new = dict(model.arrays)
    for key, value in arrays.items():
        new[key] = value.to(model.arrays[key].dtype)
    return dataclasses.replace(model, arrays=new)


def rows(model: ExchangeModel, index: slice) -> ExchangeModel:
    """The model's rows selected by `index`."""
    return dataclasses.replace(model, arrays={k: v[index] for k, v in model.arrays.items()})


def concatenated(first: ExchangeModel, second: ExchangeModel) -> ExchangeModel:
    """The rows of `first`, then those of `second` (same SH degree and time frame)."""
    arrays = {
        k: torch.cat([first.arrays[k], second.arrays[k].to(first.arrays[k].dtype)])
        for k in first.arrays
    }
    return dataclasses.replace(first, arrays=arrays)


def at_trajectory_points(gt: ExchangeModel, taus: Tensor) -> ExchangeModel:
    """Q-79, S-2: each primitive at its ground-truth position at its own raw time taus[i], with
    μt = taus[i] and v = 0; everything else as the ground truth."""
    a = gt.arrays
    # E1 at each primitive's own time: the diagonal of the all-times trajectories.
    points = torch.diagonal(
        trajectory_points(a["position"], a["velocity"], a["time_center"], taus), dim1=0, dim2=1
    ).T
    return replaced(
        gt, position=points, time_center=taus.to(F64), velocity=torch.zeros_like(points)
    )


def draw_taus(seed: int, choices: Sequence[float], n: int) -> Tensor:
    """n raw times drawn uniformly, with replacement, from `choices` (seeded)."""
    g = torch.Generator().manual_seed(seed)
    index = torch.randint(len(choices), (n,), generator=g)
    return torch.tensor(choices, dtype=F64)[index]


@dataclasses.dataclass(frozen=True)
class S4Init:
    points: InitPoints  # on the CPU, frame by frame, then sphere by sphere
    sphere: Tensor  # [N] int64: each point's sphere, whose rows in the ground truth start there
    spacing: float  # the smallest spacing of the jittered lattices: §3.2's sampling spacing


def _s4_lattice(g: torch.Generator) -> Tensor:
    """Q-119: one init lattice on the unit sphere, other than the ground truth's: the Fibonacci
    lattice of S4_INIT_POINTS points under a uniform random rotation, each point moved by up to
    S4_INIT_JITTER of the lattice's smallest spacing in a uniform random direction and put back
    on the sphere."""
    base = fibonacci_sphere(S4_INIT_POINTS)
    turned = base @ rotation_matrix(_rotations(g, 1))[0].T
    direction = torch.randn(S4_INIT_POINTS, 3, generator=g, dtype=F64)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    amount = S4_INIT_JITTER * smallest_spacing(base) * torch.rand(S4_INIT_POINTS, 1, generator=g)
    moved = turned + amount.to(F64) * direction
    lattice: Tensor = moved / torch.linalg.vector_norm(moved, dim=1, keepdim=True)
    return lattice


def s4_init(gt: ExchangeModel, seed: int, device: str) -> S4Init:
    """Q-119: S-4's init points from its ground truth. Each sphere gets one fixed lattice
    (`_s4_lattice`, seeded), scaled to S4_RADIUS and translated with the sphere to each of the
    S4_FRAMES frames, so the same material points persist; the positions are rounded to float32,
    the values `InitPoints` holds, before E15 runs on them. A point's color is that of the
    ground-truth Gaussian nearest it on its sphere, as colors taken from the images would be.
    Timestamps are the frames' τ and frame ids the frame index; durations are left to
    `from_init_points` (s₀ = Δt_f, D-21). Velocities are E15's: `knn_velocities` on `device`
    against the next frame's points, the last frame against the previous one's (D-23)."""
    g = torch.Generator().manual_seed(seed)
    a = gt.arrays
    taus = frame_taus(S4_FRAMES)
    unit_gt = fibonacci_sphere(S4_GT_POINTS)
    shells: list[Tensor] = []  # per sphere, [P, 3] at frame 0
    motions: list[Tensor] = []  # per sphere, its velocity [3] in world units per τ
    colors: list[Tensor] = []
    lattices: list[Tensor] = []
    for s in range(3):
        rows = slice(s * S4_GT_POINTS, (s + 1) * S4_GT_POINTS)
        # The ground truth's rows are center + S4_RADIUS·unit_gt at their time center.
        center = a["position"][rows].to(F64).mean(dim=0) - S4_RADIUS * unit_gt.mean(dim=0)
        velocity = a["velocity"][rows][0].to(F64)
        start = center + velocity * (taus[0] - a["time_center"][rows][0])
        lattice = _s4_lattice(g)
        lattices.append(lattice)
        shells.append(start + S4_RADIUS * lattice)
        motions.append(velocity)
        nearest = (lattice @ unit_gt.T).argmax(dim=1)
        colors.append((C0 * a["sh"][rows][nearest, 0] + 0.5).clamp(0.0, 1.0))
    frames = [
        torch.cat([shell + v * (t - taus[0]) for shell, v in zip(shells, motions, strict=True)])
        .to(F32)
        .to(device)
        for t in taus
    ]
    velocities: list[Tensor] = []
    for k in range(S4_FRAMES):
        other = k + 1 if k + 1 < S4_FRAMES else k - 1  # D-23: backward on the last frame
        dt = float(taus[other] - taus[k])
        velocities.append(knn_velocities(frames[k], frames[other], dt).cpu())  # E15
    per_frame = 3 * S4_INIT_POINTS
    points = InitPoints(
        positions=torch.cat(frames).cpu(),
        colors=torch.cat(colors).to(F32).repeat(S4_FRAMES, 1),
        timestamps=taus.repeat_interleave(per_frame),
        velocities=torch.cat(velocities),
        frame_ids=torch.arange(S4_FRAMES).repeat_interleave(per_frame),
    )
    sphere = torch.arange(3).repeat_interleave(S4_INIT_POINTS).repeat(S4_FRAMES)
    spacing = min(S4_RADIUS * smallest_spacing(lattice) for lattice in lattices)
    return S4Init(points=points, sphere=sphere, spacing=spacing)


# Scenes on disk.


SceneImage = tuple[Camera, float, Literal["train", "test"]]


def scene_video(
    root: Path, gt: ExchangeModel, images: Sequence[SceneImage], device: str
) -> MultiViewVideo:
    """The ground truth rendered on `device` with that device's backend into a scene manifest
    and 8-bit PNGs (§3.2), and its multi-view video. The ground truth renders under the
    normalizer of the training timestamps, so no image's τ is outside it."""
    stamps = [t for _, t, split in images if split == "train"]
    time = TimeNormalizer(min(stamps), max(stamps))
    model = model_under(gt, time, sh_degree=gt.sh_degree, device=device)
    path = write_scene(root, model, images, backend=backend_for(device))
    return MultiViewVideo.from_manifest(path)


def grid(
    cameras: Iterable[Camera], taus: Iterable[float], split: Literal["train", "test"]
) -> list[SceneImage]:
    return [(camera, t, split) for t in taus for camera in cameras]


# Training (§3.2 shared setup; Q-75, Q-79).


def synthetic_config(iterations: int, base: TrainConfig) -> TrainConfig:
    """`base` without the perceptual loss (Q-79), on the shortened calendar of §3.2 (Q-75)."""
    loss = dataclasses.replace(base.loss, lambda_perc=0.0)
    return shortened_config(iterations, dataclasses.replace(base, loss=loss))


def train(
    cfg: TrainConfig,
    video: MultiViewVideo,
    init: FreeTimeGaussians,
    device: str,
    callbacks: Sequence[Callable[[Trainer, StepLog], bool | None]] = (),
) -> tuple[Trainer, list[StepLog]]:
    """A full run of `cfg` from `init`, with its step logs."""
    trainer = Trainer(cfg, video, init, device=device)
    logs: list[StepLog] = []

    def keep(_: Trainer, log: StepLog) -> None:
        logs.append(log)

    trainer.train(callbacks=[keep, *callbacks])
    return trainer, logs


def assert_rows_matched(logs: Sequence[StepLog], n: int) -> None:
    """Q-80: one-to-one matching by row holds, since N never changed and no relocation event
    moved a row. A report of 0 dead rows also covers relocation's no-op when every row is dead
    (§1.3), which the scenes' recovery criteria would fail anyway."""
    assert all(log.num_gaussians == n for log in logs)
    assert all(log.relocation is None or log.relocation.dead == 0 for log in logs)


# Recovery measures (Q-80), in raw units.


def physical(model: FreeTimeGaussians) -> dict[str, Tensor]:
    return {k: v.detach().cpu() for k, v in model.physical().items()}


def trajectory_error(model: FreeTimeGaussians, gt: ExchangeModel, taus: Tensor) -> float:
    """The largest ‖μx(τ) − μx_gt(τ)‖ over primitives and the given raw times (E1)."""
    p, a = physical(model), gt.arrays
    ours = trajectory_points(p["position"], p["velocity"], p["time_center"], taus)
    theirs = trajectory_points(a["position"], a["velocity"], a["time_center"], taus)
    return float(torch.linalg.vector_norm(ours - theirs, dim=-1).max())


def velocity_error_rows(velocity: Tensor, velocity_gt: Tensor) -> dict[str, Tensor]:
    """Q-80's velocity errors of each row of [N, 3] velocities against [N, 3] true ones, in
    float64: the relative magnitude error, the angle in degrees and the relative vector error."""
    v, v_gt = velocity.to(F64), velocity_gt.to(F64)
    norm, norm_gt = torch.linalg.vector_norm(v, dim=1), torch.linalg.vector_norm(v_gt, dim=1)
    cosine = (v * v_gt).sum(dim=1) / (norm * norm_gt)
    return {
        "magnitude": (norm - norm_gt).abs() / norm_gt,
        "angle_deg": torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0))),
        "vector": torch.linalg.vector_norm(v - v_gt, dim=1) / norm_gt,
    }


def velocity_errors(model: FreeTimeGaussians, gt: ExchangeModel) -> dict[str, float]:
    """Per primitive, in world units per τ, each a max over primitives: the relative magnitude
    error, the angle in degrees and the relative vector error."""
    errors = velocity_error_rows(physical(model)["velocity"], gt.arrays["velocity"])
    return {name: float(value.max()) for name, value in errors.items()}


def mean_psnr(model: FreeTimeGaussians, video: MultiViewVideo, indices: Iterable[int]) -> float:
    """The mean over images of E16 PSNR, each record rendered at its raw τ with the backend of
    the model's device against its 8-bit image."""
    backend = backend_for(model.means.device.type)
    values: list[float] = []
    with torch.no_grad():
        for index in indices:
            record = video.records[index]
            out = render(model, record.camera, record.timestamp, backend=backend)
            gt = video.load_image(index).to(out.image.device)
            values.append(float(psnr(out.image[0], gt)))
    return sum(values) / len(values)


def split_indices(video: MultiViewVideo, split: str) -> list[int]:
    return [i for i, record in enumerate(video.records) if record.split == split]


def dead_rows(seed: int, gt: ExchangeModel, count: int, taus: Tensor) -> ExchangeModel:
    """Q-78, S-6: `count` dead primitives (σ = 10⁻³) at seeded places in the view volume more than
    DEAD_CLEARANCE from every ground-truth trajectory point, with scales like the ground truth's,
    random colors and rotations, S-2's duration, v = 0 and a random frame each."""
    g = torch.Generator().manual_seed(seed)
    a = gt.arrays
    tracks = trajectory_points(a["position"], a["velocity"], a["time_center"], taus).reshape(-1, 3)
    positions: list[Tensor] = []
    while len(positions) < count:
        p = _uniform(g, -VOLUME, VOLUME, 3)
        if float(torch.linalg.vector_norm(tracks - p, dim=1).min()) > DEAD_CLEARANCE:
            positions.append(p)
    arrays = {
        "position": torch.stack(positions),
        "time_center": taus[torch.randint(len(taus), (count,), generator=g)],
        "duration": torch.full((count,), float(a["duration"][0]), dtype=F64),
        "velocity": torch.zeros(count, 3, dtype=F64),
        "scale": _log_uniform(g, *SCALE_RANGE, count, 3),
        "rotation": _rotations(g, count),
        "opacity": torch.full((count,), 1e-3, dtype=F64),
        "sh": sh_of(_uniform(g, 0.2, 0.9, count, 3)),
    }
    return exchange(arrays, tau0=gt.tau0, tau1=gt.tau1, degree=0)
