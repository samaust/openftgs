# SPDX-License-Identifier: Apache-2.0
"""The analytic ray caster of §3.2's S-9 and S-11 (M14, PRD #157; Q-138, Q-140): textured
spheres translating linearly, rendered in NumPy float64 from first principles, so the model is
graded on data that isn't of its own form (§3.2) and the renderer under test shares no code
with the images it is graded on.

Each pixel averages `supersample`² rays through the continuous-convention pixel (§1.4: pixel
(i, j) covers [i, i + 1) × [j, j + 1), its center at i + ½), each hitting the nearest sphere or
the black background (D-34). A sphere's surface is emissive and view-independent, colored by a
smooth procedural texture of the sphere-local direction, colors in [0.1, 0.9] (Q-138). The
caster also returns, per pixel, the share of its rays that hit a sphere's local z < 0 half, the
half S-11 leaves without initial points.
"""

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from builders import look_at
from PIL import Image
from synthetic_scenes import fibonacci_sphere, smallest_spacing

from openftgs.init import knn_velocities
from openftgs.io import write_scene_manifest
from openftgs.types import Camera, ImageRecord, InitPoints

FPS = 30.0
FRAMES = 20  # §3.2 S-9
WIDTH, HEIGHT = 128, 96  # §3.2 S-9
FOCAL = 64.0  # Q-138: every sphere in view at the rig's radius of about 0.89
RADIUS = 0.12  # sphere radius, world units (S ≈ 1): about 17 px across
CENTER_RADIUS = 0.25  # centers at the middle time within this of the origin
STEP = (0.004, 0.007)  # displacement per frame, below half the init lattice's spacing (E15)
CLEARANCE = 0.04  # surfaces this far apart at every frame
EDGE_MARGIN = 4.0  # silhouettes this many pixels inside every image
TEXTURE_WAVES = 6  # sinusoids per color channel
TEXTURE_FREQUENCY = (2.0, 5.0)  # rad per unit of the local direction: smooth at RADIUS

_TRAIN_ELEVATIONS = (15.0, 30.0, 45.0, 60.0)  # Q-138: 16 training cameras, four azimuths each
_TEST_VIEWS = ((25.0, 56.25), (40.0, 146.25), (52.0, 236.25), (33.0, 326.25))  # (elev, azim)


def tau(frame: float) -> float:
    return frame / FPS


@dataclasses.dataclass(frozen=True)
class Sphere:
    center: np.ndarray  # [3], at the middle time
    velocity: np.ndarray  # [3], world units per τ
    waves: np.ndarray  # [3 channels, TEXTURE_WAVES, 3] frequency vectors
    phases: np.ndarray  # [3, TEXTURE_WAVES]

    def center_at(self, t: float) -> np.ndarray:
        return self.center + self.velocity * (t - tau((FRAMES - 1) / 2))

    def color(self, direction: np.ndarray) -> np.ndarray:
        """The texture at unit local directions [..., 3]: per channel the mean of sinusoids of
        the direction, mapped to [0.1, 0.9]."""
        arg = np.einsum("...d,cwd->...cw", direction, self.waves) + self.phases
        texture: np.ndarray = 0.5 + 0.4 * np.sin(arg).mean(axis=-1)
        return texture


@dataclasses.dataclass(frozen=True)
class Rig:
    train: list[Camera]
    test: list[Camera]


def _on_sphere(radius: float, elevation: float, azimuth: float) -> tuple[float, float, float]:
    e, a = math.radians(elevation), math.radians(azimuth)
    return (
        radius * math.cos(e) * math.cos(a),
        radius * math.cos(e) * math.sin(a),
        radius * math.sin(e),
    )


def _train_views() -> list[tuple[float, float]]:
    return [(e, 90.0 * k + 22.5 * j) for j, e in enumerate(_TRAIN_ELEVATIONS) for k in range(4)]


def rig() -> Rig:
    """Q-138: 16 hemisphere training cameras and 4 held-out ones between them, 128×96, looking
    at the origin, at the radius that gives S = 1 (D-27: 1.1 × the largest distance of a training
    camera center from their centroid)."""
    unit = np.array([_on_sphere(1.0, e, a) for e, a in _train_views()])
    spread = np.linalg.norm(unit - unit.mean(axis=0), axis=1)
    radius = 1.0 / (1.1 * float(np.max(spread)))

    def camera(elevation: float, azimuth: float, name: str) -> Camera:
        eye = _on_sphere(radius, elevation, azimuth)
        return look_at(eye, width=WIDTH, height=HEIGHT, focal=FOCAL, name=name)

    train = [camera(e, a, f"cam{k:02d}") for k, (e, a) in enumerate(_train_views())]
    test = [camera(e, a, f"heldout{k:02d}") for k, (e, a) in enumerate(_TEST_VIEWS)]
    return Rig(train, test)


def _project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = camera.world_to_camera.double().numpy()
    K = camera.K.double().numpy()
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    pix = cam @ K.T
    return pix[:, :2] / pix[:, 2:], cam[:, 2]


def _in_view(cameras: Sequence[Camera], points: np.ndarray) -> bool:
    for camera in cameras:
        uv, depth = _project(camera, points)
        inside = (
            (uv[:, 0] >= EDGE_MARGIN)
            & (uv[:, 0] <= camera.width - EDGE_MARGIN)
            & (uv[:, 1] >= EDGE_MARGIN)
            & (uv[:, 1] <= camera.height - EDGE_MARGIN)
            & (depth > 0.2)
        )
        if not inside.all():
            return False
    return True


def spheres(seed: int, cameras: Sequence[Camera]) -> list[Sphere]:
    """Three spheres, drawn again until each one's surface stays in view of every camera at every
    frame and CLEARANCE from the others' (as S-4's are, Q-119), each with its own texture."""
    g = np.random.default_rng(seed)
    taus = [tau(k) for k in range(FRAMES)]
    shell = fibonacci_sphere(200).numpy() * RADIUS
    found: list[Sphere] = []
    for _ in range(100_000):
        if len(found) == 3:
            return found
        center = g.uniform(-CENTER_RADIUS, CENTER_RADIUS, 3)
        direction = g.normal(size=3)
        velocity = direction / np.linalg.norm(direction) * g.uniform(*STEP) * FPS
        waves = g.normal(size=(3, TEXTURE_WAVES, 3))
        waves /= np.linalg.norm(waves, axis=-1, keepdims=True)  # random wave directions
        waves *= g.uniform(*TEXTURE_FREQUENCY, (3, TEXTURE_WAVES, 1))
        sphere = Sphere(center, velocity, waves, g.uniform(0.0, 2 * math.pi, (3, TEXTURE_WAVES)))
        if np.linalg.norm(center) > CENTER_RADIUS:
            continue
        track = np.array([sphere.center_at(t) for t in taus])
        if not _in_view(cameras, (track[:, None, :] + shell).reshape(-1, 3)):
            continue
        apart = 2.0 * RADIUS + CLEARANCE
        others = [np.array([s.center_at(t) for t in taus]) for s in found]
        if any(np.linalg.norm(track - o, axis=1).min() < apart for o in others):
            continue
        found.append(sphere)
    raise AssertionError("no three spheres fit the rig")


def cast(
    scene: Sequence[Sphere], camera: Camera, t: float, *, supersample: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """The image [H, W, 3] in [0, 1] (float64) at raw time `t`, and per pixel the share of rays
    that hit a sphere's local z < 0 half [H, W]."""
    w2c = camera.world_to_camera.double().numpy()
    K = camera.K.double().numpy()
    R, translation = w2c[:3, :3], w2c[:3, 3]
    origin = -R.T @ translation
    offsets = (np.arange(supersample) + 0.5) / supersample
    u = (np.arange(camera.width)[:, None] + offsets[None, :]).reshape(-1)
    v = (np.arange(camera.height)[:, None] + offsets[None, :]).reshape(-1)
    uu, vv = np.meshgrid(u, v)  # [H·s, W·s]
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    directions = pixels @ np.linalg.inv(K).T @ R  # camera rays to world (Rᵀ·d as a row)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    best = np.full(uu.shape, np.inf)
    color = np.zeros((*uu.shape, 3))
    lower = np.zeros(uu.shape)
    for sphere in scene:
        center = sphere.center_at(t)
        oc = origin - center
        b = directions @ oc
        disc = b * b - (oc @ oc - RADIUS * RADIUS)
        hit = disc > 0.0
        depth = np.where(hit, -b - np.sqrt(np.where(hit, disc, 0.0)), np.inf)
        nearer = hit & (depth > 0.0) & (depth < best)
        if not nearer.any():
            continue
        local = (origin + depth[nearer, None] * directions[nearer] - center) / RADIUS
        best[nearer] = depth[nearer]
        color[nearer] = sphere.color(local)
        lower[nearer] = (local[:, 2] < 0.0).astype(np.float64)
    s = supersample
    image = color.reshape(camera.height, s, camera.width, s, 3).mean(axis=(1, 3))
    half = lower.reshape(camera.height, s, camera.width, s).mean(axis=(1, 3))
    return image, half


@dataclasses.dataclass(frozen=True)
class Scene:
    manifest: Path  # train and test images
    train_manifest: Path  # the training images only
    spheres: list[Sphere]
    rig: Rig
    halves: dict[tuple[str, int], np.ndarray]  # (camera, frame) → local z < 0 share, test views


def write_scene(root: Path, seed: int) -> Scene:
    """The ray-cast images of every camera at every frame as 8-bit PNGs, with two manifests:
    all images (the held-out cameras as the test split) and the training images only."""
    cameras = rig()
    scene = spheres(seed, cameras.train + cameras.test)
    records: list[ImageRecord] = []
    halves: dict[tuple[str, int], np.ndarray] = {}
    for k in range(FRAMES):
        for split, group in (("train", cameras.train), ("test", cameras.test)):
            for camera in group:
                image, half = cast(scene, camera, tau(k))
                path = root / "images" / camera.name / f"{k:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(pixels, "RGB").save(path)
                kind: Literal["train", "test"] = "train" if split == "train" else "test"
                records.append(ImageRecord(camera, tau(k), path, kind, k))
                if kind == "test":
                    halves[(camera.name, k)] = half
    metadata = {"source": "synthetic/raycast"}
    full = write_scene_manifest(root / "scene.json", records, metadata=metadata)
    train = write_scene_manifest(
        root / "train.json", [r for r in records if r.split == "train"], metadata=metadata
    )
    return Scene(full.path, train.path, scene, cameras, halves)


def _lattice(g: torch.Generator, n: int) -> torch.Tensor:
    """A Fibonacci lattice of n points under a uniform random rotation (float64)."""
    q = torch.randn(4, generator=g, dtype=torch.float64)
    w, x, y, z = (q / torch.linalg.vector_norm(q)).tolist()
    rotation = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )
    return fibonacci_sphere(n) @ rotation.T


def surface_samples(
    scene: Sequence[Sphere], seed: int, per_sphere: int, *, upper_only: bool = False
) -> InitPoints:
    """Init points from the surfaces (Q-140, after Q-119's S-4): one fixed lattice per sphere,
    translated with it to each frame, colored by the texture, with frame ids and E15 velocities
    (`knn_velocities` against the next frame, backward on the last, D-23). With `upper_only`,
    only the lattice points of each sphere's local z ≥ 0 half (S-11)."""
    g = torch.Generator().manual_seed(seed)
    lattices = [_lattice(g, per_sphere) for _ in scene]
    if upper_only:
        lattices = [lattice[lattice[:, 2] >= 0.0] for lattice in lattices]
    colors = torch.cat(
        [torch.from_numpy(s.color(lat.numpy())) for s, lat in zip(scene, lattices, strict=True)]
    ).to(torch.float32)
    frames = [
        torch.cat(
            [
                torch.from_numpy(s.center_at(tau(k))) + RADIUS * lat
                for s, lat in zip(scene, lattices, strict=True)
            ]
        ).to(torch.float32)
        for k in range(FRAMES)
    ]
    velocities = []
    for k in range(FRAMES):
        other = k + 1 if k + 1 < FRAMES else k - 1  # D-23
        velocities.append(knn_velocities(frames[k], frames[other], tau(other) - tau(k)))  # E15
    per_frame = frames[0].shape[0]
    return InitPoints(
        positions=torch.cat(frames),
        colors=colors.repeat(FRAMES, 1),
        timestamps=torch.tensor(
            [tau(k) for k in range(FRAMES)], dtype=torch.float64
        ).repeat_interleave(per_frame),
        velocities=torch.cat(velocities),
        frame_ids=torch.arange(FRAMES).repeat_interleave(per_frame),
    )


def lattice_spacing(per_sphere: int) -> float:
    return RADIUS * smallest_spacing(fibonacci_sphere(per_sphere))
