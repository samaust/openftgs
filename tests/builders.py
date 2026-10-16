# SPDX-License-Identifier: Apache-2.0
"""Shared test scenes for the test suite (forward note on PRD #20): random models and cameras,
the hand-built render scenes and U-8's scenes that both render backends' tests use, U-4's
change of time unit, and scenes written to disk as a scene manifest with its images, with the
shortened calendars of §3.2 (PRD #84, shared with M11).

Each test file still writes out its own `PARAM_NAMES` from §1.1: that tuple is the tests'
independent expectation, not a duplicate to share.
"""

import dataclasses
import json
import math
import stat
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from openftgs.config import TrainConfig
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera, ExchangeModel, TimeNormalizer

UNIT_TIME = TimeNormalizer(0.0, 1.0)
C0 = 0.28209479177387814  # §1.1


def random_params(
    n: int,
    *,
    sh_degree: int = 3,
    generator: torch.Generator,
    spread: float = 1.0,
    log_scale: tuple[float, float] = (-3.0, -1.5),
    logit: tuple[float, float] = (-2.0, 2.0),
    log_duration: tuple[float, float] = (-1.5, 0.5),
    speed: float = 0.5,
    sh_scale: float = 0.5,
) -> dict[str, Tensor]:
    """The nine stored tensors of §1.1 (float32) for n random primitives: means in a cube of
    half-width `spread` around the origin, time centers in [0, 1], and the other quantities drawn
    uniformly from the given ranges (logs for scales and durations)."""

    def uniform(*shape: int, low: float, high: float) -> Tensor:
        return low + (high - low) * torch.rand(*shape, generator=generator)

    return {
        "means": uniform(n, 3, low=-spread, high=spread),
        "times": torch.rand(n, generator=generator),
        "log_durations": uniform(n, low=log_duration[0], high=log_duration[1]),
        "velocities": speed * torch.randn(n, 3, generator=generator),
        "log_scales": uniform(n, 3, low=log_scale[0], high=log_scale[1]),
        "quats": torch.randn(n, 4, generator=generator),
        "opacity_logits": uniform(n, low=logit[0], high=logit[1]),
        "sh0": sh_scale * torch.randn(n, 1, 3, generator=generator),
        "shN": sh_scale * torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, generator=generator),
    }


def random_model(
    n: int,
    *,
    seed: int = 0,
    sh_degree: int = 3,
    time: TimeNormalizer = UNIT_TIME,
    spread: float = 1.0,
    log_scale: tuple[float, float] = (-3.0, -1.5),
    logit: tuple[float, float] = (-2.0, 2.0),
    log_duration: tuple[float, float] = (-1.5, 0.5),
    speed: float = 0.5,
    sh_scale: float = 0.5,
) -> FreeTimeGaussians:
    """A model of n random primitives from a generator seeded with `seed` (see `random_params`
    for the ranges)."""
    params = random_params(
        n,
        sh_degree=sh_degree,
        generator=torch.Generator().manual_seed(seed),
        spread=spread,
        log_scale=log_scale,
        logit=logit,
        log_duration=log_duration,
        speed=speed,
        sh_scale=sh_scale,
    )
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


def pinhole(
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    world_to_camera: Sequence[Sequence[float]] | Tensor | None = None,
    name: str = "",
) -> Camera:
    """A camera in the continuous pixel convention of §1.4 (identity pose by default)."""
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    if world_to_camera is None:
        w2c = torch.eye(4, dtype=torch.float32)
    else:
        w2c = torch.as_tensor(world_to_camera, dtype=torch.float32)
        if w2c.shape == (3, 4):
            w2c = torch.cat([w2c, torch.tensor([[0.0, 0.0, 0.0, 1.0]])])
    return Camera(K=K, world_to_camera=w2c, width=width, height=height, name=name)


def look_at(
    eye: Sequence[float],
    target: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    width: int,
    height: int,
    focal: float,
    name: str = "",
) -> Camera:
    """A camera at `eye` looking at `target`, with OpenCV axes (x right, y down, z forward;
    §1.4) and the principal point at the image center."""
    eye_t = torch.tensor(eye, dtype=torch.float64)
    z = torch.tensor(target, dtype=torch.float64) - eye_t
    z = z / torch.linalg.vector_norm(z)
    x = torch.linalg.cross(z, torch.tensor(up, dtype=torch.float64))
    x = x / torch.linalg.vector_norm(x)
    y = torch.linalg.cross(z, x)
    rotation = torch.stack([x, y, z])
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ eye_t
    return pinhole(
        width,
        height,
        fx=focal,
        fy=focal,
        cx=width / 2,
        cy=height / 2,
        world_to_camera=w2c.to(torch.float32),
        name=name,
    )


def ring(
    count: int,
    *,
    radius: float = 4.0,
    elevation: float = 1.0,
    width_px: int = 24,
    height_px: int = 16,
    focal: float = 20.0,
) -> list[Camera]:
    """`count` cameras evenly spaced on a horizontal ring of the given radius, `elevation` above
    the origin, looking at it."""
    return [
        look_at(
            (
                radius * math.cos(2 * math.pi * k / count),
                radius * math.sin(2 * math.pi * k / count),
                elevation,
            ),
            width=width_px,
            height=height_px,
            focal=focal,
            name=f"ring{k:02d}",
        )
        for k in range(count)
    ]


# Hand-built render scenes, shared by the render tests of both backends (E2, E6).


_ROW_KEYS = {"mean", "time", "log_duration", "velocity", "scale", "quat", "logit", "sh0", "shN"}


def model_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sh_degree: int = 0,
    time: TimeNormalizer = UNIT_TIME,
) -> FreeTimeGaussians:
    """A model with one primitive per row. Defaults: time center 0.5, ŝ = 20 (σt = 1 within
    |t − μt| ≤ 1), no velocity, isotropic scale 0.25, identity quaternion, logit 0 (σ = 0.5),
    sh0 = 0 (color 0.5) and no higher bands. An unknown key raises ValueError."""
    k = (sh_degree + 1) ** 2 - 1
    shapes: dict[str, tuple[int, ...]] = {
        "means": (3,),
        "times": (),
        "log_durations": (),
        "velocities": (3,),
        "log_scales": (3,),
        "quats": (4,),
        "opacity_logits": (),
        "sh0": (1, 3),
        "shN": (k, 3),
    }
    cols: dict[str, list[Any]] = {name: [] for name in shapes}
    for row in rows:
        unknown = set(row) - _ROW_KEYS
        if unknown:
            raise ValueError(f"unknown row keys {sorted(unknown)}")
        scale = row.get("scale", 0.25)
        scales = [scale] * 3 if isinstance(scale, int | float) else list(scale)
        cols["means"].append(list(row["mean"]))
        cols["times"].append(row.get("time", 0.5))
        cols["log_durations"].append(row.get("log_duration", 20.0))
        cols["velocities"].append(list(row.get("velocity", (0.0, 0.0, 0.0))))
        cols["log_scales"].append([math.log(a) for a in scales])
        cols["quats"].append(list(row.get("quat", (1.0, 0.0, 0.0, 0.0))))
        cols["opacity_logits"].append(row.get("logit", 0.0))
        cols["sh0"].append([list(row.get("sh0", (0.0, 0.0, 0.0)))])
        cols["shN"].append(row.get("shN", [[0.0, 0.0, 0.0]] * k))
    params = {
        name: torch.tensor(cols[name], dtype=torch.float32).reshape(len(rows), *shape)
        for name, shape in shapes.items()
    }
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


# Camera A: the point (X, Y, 4) lands at (32.5 + 4X, 24.5 + 4Y).
CAM_A = pinhole(64, 48, fx=16.0, fy=16.0, cx=32.5, cy=24.5)


def sh0_for(rgb: Sequence[float]) -> tuple[float, float, float]:
    """sh0 giving the color rgb at degree 0 (E2: C₀·sh0 + 0.5)."""
    r, g, b = ((c - 0.5) / C0 for c in rgb)
    return (r, g, b)


def logit_of(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid64(logit: float) -> float:
    """σ of a logit as stored (float32), evaluated in float64."""
    return float(torch.sigmoid(torch.tensor(logit, dtype=torch.float32).to(torch.float64)))


# U-8's scenes (D-32, D-57), shared by both backends' U-8 tests.


def u8_scene(
    n: int, dtype: torch.dtype, *, kept: int | None = None, tie: bool = False
) -> FreeTimeGaussians:
    """n random rows around `ring(3)`, some behind a camera or off the image, with every third row
    below the opacity cull (logit −9). With `kept`, exactly that many rows pass the cull. With
    `tie`, rows 4 and 5 sit at the same place, opaque enough to be seen by every camera, with
    different colors, so they tie in depth."""
    params = random_params(
        n,
        generator=torch.Generator().manual_seed(n),
        spread=4.5,
        log_scale=(-2.5, -0.5),
        logit=(-3.0, 3.0),
    )
    params["opacity_logits"][::3] = -9.0
    if kept is not None:
        params["opacity_logits"] = torch.tensor([3.0] * kept + [-9.0] * (n - kept))
    if tie:
        for row in (4, 5):
            params["means"][row] = torch.tensor([0.1, 0.2, 0.3])
            params["velocities"][row] = 0.0
            params["times"][row] = 0.37
            params["opacity_logits"][row] = 2.0
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME)
    return model.double() if dtype == torch.float64 else model


U8_CASES: dict[str, tuple[int, int | None, bool]] = {  # n, rows passing the cull, depth tie
    "no row kept": (1, 0, False),
    "one row kept": (2, 1, False),
    "a third culled": (7, None, False),
    "depth tie": (33, None, True),
    "depth tie, more rows": (64, None, True),
}


# U-4's change of time unit (E14, §1.4 "Time"), shared by both backends' U-4 tests.


def rescaled_time_unit(exchange: ExchangeModel, a: float, b: float) -> ExchangeModel:
    """The exchange model in the time unit τ′ = aτ + b: time centers aτ + b, durations a·d,
    velocities v_τ/a (float32, as ftgs-if stores them), τ₀ and τ₁ mapped too."""
    arrays = dict(exchange.arrays)
    arrays["time_center"] = a * arrays["time_center"] + b
    arrays["duration"] = a * arrays["duration"]
    arrays["velocity"] = (arrays["velocity"].to(torch.float64) / a).to(torch.float32)
    return ExchangeModel(
        arrays=arrays,
        sh_degree=exchange.sh_degree,
        tau0=a * exchange.tau0 + b,
        tau1=a * exchange.tau1 + b,
        time_unit=exchange.time_unit,
        producer=exchange.producer,
    )


# Scenes on disk (§2.2 scene manifest) and the shortened calendars of §3.2.


def _camera_json(camera: Camera) -> dict[str, Any]:
    return {
        "width": camera.width,
        "height": camera.height,
        "model": "pinhole",
        "K": camera.K.tolist(),
        "world_to_camera": camera.world_to_camera.tolist(),
    }


def write_scene(
    root: Path,
    model: FreeTimeGaussians,
    images: Sequence[tuple[Camera, float, str]],
    *,
    mask_tests: bool = False,
    backend: Literal["reference", "gsplat"] = "reference",
) -> Path:
    """A scene manifest and its images in `root`, returning the manifest's path. Each image, a
    (camera, raw τ, data split) triple, is rendered from `model` with `backend` on the model's
    device at that τ, clamped to [0, 1] and rounded to an 8-bit RGB PNG. Cameras are keyed by
    name, so the images of one camera share it. With `mask_tests`, each test image gets an 8-bit
    mask whose left half is dynamic."""
    root.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for k, (camera, tau, split) in enumerate(images):
        cameras.setdefault(camera.name, _camera_json(camera))
        with torch.no_grad():
            rgb = render(model, camera, tau, backend=backend).image[0]
        pixels = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        entry: dict[str, Any] = {
            "camera": camera.name,
            "timestamp": tau,
            "frame_index": k,
            "path": f"images/{camera.name}/{k:06d}.png",
            "split": split,
        }
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, "RGB").save(target)
        if mask_tests and split == "test":
            mask = torch.zeros(camera.height, camera.width, dtype=torch.uint8)
            mask[:, : camera.width // 2] = 255
            entry["mask"] = f"masks/{camera.name}/{k:06d}.png"
            (root / entry["mask"]).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.numpy(), "L").save(root / entry["mask"])
        entries.append(entry)
    manifest = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": cameras,
        "images": entries,
        "metadata": {"source": "synthetic"},
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


def shortened_config(iterations: int, base: TrainConfig) -> TrainConfig:
    """§3.2 shared setup: `base` with T_it = `iterations` and every
    iteration count of its calendar (the growth and relocation windows, the event period, the
    SH interval and, when set, the last iteration of the 4D regularization, given for
    T_it = 30,000) multiplied by T_it/30,000 and rounded, half up (Q-75), with a minimum of 1."""
    cfg = base

    def scaled(count: int) -> int:
        return max(1, math.floor(count * iterations / 30_000 + 0.5))  # Q-75

    reg_until = cfg.loss.reg_until
    return dataclasses.replace(
        cfg,
        loss=dataclasses.replace(
            cfg.loss, reg_until=None if reg_until is None else scaled(reg_until)
        ),
        growth=dataclasses.replace(
            cfg.growth, start=scaled(cfg.growth.start), stop=scaled(cfg.growth.stop)
        ),
        reloc=dataclasses.replace(
            cfg.reloc,
            every=scaled(cfg.reloc.every),
            start=scaled(cfg.reloc.start),
            stop=scaled(cfg.reloc.stop),
        ),
        optim=dataclasses.replace(
            cfg.optim, iterations=iterations, sh_interval=scaled(cfg.optim.sh_interval)
        ),
    )


# A textured plane that a dense matcher can match (#137): RoMa's tests need real images, which
# the coordinate-encoded scenes of the init tests are not. The plane y = y₀ + v_y·τ carries a
# seeded multi-octave value-noise texture that moves with it by (v_x, v_z)·τ; each pixel is the
# mean of a grid of exact ray–plane hits inside it, so no image is resampled.


@dataclasses.dataclass(frozen=True)
class TexturedPlane:
    """The plane y = y0 + velocity[1]·τ, its texture at (x − velocity[0]·τ, z − velocity[2]·τ)."""

    seed: int
    y0: float = 0.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cells: tuple[float, ...] = (0.5, 0.25, 0.12, 0.06, 0.03)  # octave cell sizes, world units

    def texture(self, a: Tensor, b: Tensor) -> Tensor:
        """RGB in [0, 1] at plane coordinates (a, b), float64 [..., 3]: smoothstep value noise,
        one 64 × 64 periodic grid per octave and channel, octaves weighted by √(cell size),
        then stretched about ½ for contrast."""
        generator = torch.Generator().manual_seed(self.seed)
        total = torch.zeros(*a.shape, 3, dtype=torch.float64)
        for cell in self.cells:
            grid = torch.rand(64, 64, 3, generator=generator, dtype=torch.float64)
            p, q = a / cell, b / cell
            i, j = torch.floor(p), torch.floor(q)
            fp, fq = p - i, q - j
            fp, fq = fp * fp * (3 - 2 * fp), fq * fq * (3 - 2 * fq)  # smoothstep
            i0, j0 = i.long() % 64, j.long() % 64
            i1, j1 = (i0 + 1) % 64, (j0 + 1) % 64
            top = grid[i0, j0] * (1 - fp)[..., None] + grid[i1, j0] * fp[..., None]
            bottom = grid[i0, j1] * (1 - fp)[..., None] + grid[i1, j1] * fp[..., None]
            total += math.sqrt(cell) * (top * (1 - fq)[..., None] + bottom * fq[..., None])
        mean = total / sum(math.sqrt(cell) for cell in self.cells)
        return (0.5 + 2.5 * (mean - 0.5)).clamp(0.0, 1.0)

    def render(self, camera: Camera, tau: float, *, supersample: int = 3) -> Tensor:
        """The camera's 8-bit RGB image [H, W, 3] at raw τ; rays that miss the plane are grey."""
        w2c = camera.world_to_camera.to(torch.float64)
        rotation, t = w2c[:3, :3], w2c[:3, 3]
        center = -rotation.T @ t
        offsets = (torch.arange(supersample, dtype=torch.float64) + 0.5) / supersample
        u = (torch.arange(camera.width, dtype=torch.float64)[:, None] + offsets).reshape(-1)
        v = (torch.arange(camera.height, dtype=torch.float64)[:, None] + offsets).reshape(-1)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
        rays = pixels @ torch.linalg.inv(camera.K.to(torch.float64)).T @ rotation
        y = self.y0 + self.velocity[1] * tau
        s = (y - center[1]) / rays[..., 1]
        hits = center + s[..., None] * rays
        color = self.texture(
            hits[..., 0] - self.velocity[0] * tau, hits[..., 2] - self.velocity[2] * tau
        )
        color = torch.where((s > 0)[..., None], color, torch.full_like(color, 0.5))
        h, w, n = camera.height, camera.width, supersample
        color = color.reshape(h, n, w, n, 3).mean(dim=(1, 3))
        return (color * 255.0).round().to(torch.uint8)


def write_plane_scene(
    root: Path, plane: TexturedPlane, images: Sequence[tuple[Camera, float, str]]
) -> Path:
    """A scene manifest and its images in `root`, laid out as `write_scene` writes them, each
    (camera, raw τ, data split) image rendered from `plane`. Cameras are keyed by name, in the
    order they first appear."""
    root.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for k, (camera, tau, split) in enumerate(images):
        cameras.setdefault(camera.name, _camera_json(camera))
        entry: dict[str, Any] = {
            "camera": camera.name,
            "timestamp": tau,
            "frame_index": k,
            "path": f"images/{camera.name}/{k:06d}.png",
            "split": split,
        }
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plane.render(camera, tau).numpy(), "RGB").save(target)
        entries.append(entry)
    manifest = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": cameras,
        "images": entries,
        "metadata": {"source": "synthetic"},
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


# Render requests (§2.2 ftgs-render-request v1), written by hand for the CLI tests.


def write_render_request(
    path: Path, items: Sequence[tuple[str, float, Camera]], *, unit: str = "s"
) -> Path:
    """An ftgs-render-request v1 file of (id, raw τ, camera) items, black background, the
    model's full SH degree."""
    body = {
        "format": "ftgs-render-request",
        "version": 1,
        "time_unit": unit,
        "background": [0, 0, 0],
        "sh_degree": None,
        "items": [
            {
                "id": item_id,
                "timestamp": tau,
                "camera": {
                    "width": camera.width,
                    "height": camera.height,
                    "K": camera.K.tolist(),
                    "world_to_camera": camera.world_to_camera.tolist(),
                },
            }
            for item_id, tau, camera in items
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


# ffmpeg-made clips and Neural3DV sources (from the converter tests; shared with the CLI tests).


def ffmpeg_video(
    path: Path,
    *,
    width: int = 64,
    height: int = 48,
    frames: int = 10,
    lossless: bool = True,
    smooth: bool = False,
) -> Path:
    """A lavfi testsrc clip; the lossless clips decode to exactly what ffmpeg's PNG decode
    gives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lossless:
        codec = ["-c:v", "libx264rgb", "-qp", "0"]
    else:
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    pattern = "gradients" if smooth else "testsrc"  # gradients: smooth content (U-35's oracle)
    source = f"{pattern}=size={width}x{height}:rate=30"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-frames:v",
            str(frames),
            *codec,
            str(path),
        ],
        check=True,
    )
    return path


def llff_row(centre_x: float, *, width: int, height: int, focal: float) -> np.ndarray:
    """A poses_bounds.npy row (D-28) for a camera at (centre_x, 0, 0) looking down +z, with
    OpenCV axes x right, y down: the LLFF columns are down, right, back."""
    down, right, back = [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]
    matrix = np.array([down, right, back, [centre_x, 0.0, 0.0], [height, width, focal]]).T
    return np.concatenate([matrix.reshape(-1), [0.5, 50.0]])


def n3dv_source(root: Path, cameras: int = 2, **video: Any) -> Path:
    width, height = video.get("width", 64), video.get("height", 48)
    rows = [llff_row(0.25 * k, width=width, height=height, focal=50.0) for k in range(cameras)]
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "poses_bounds.npy", np.stack(rows))
    for k in range(cameras):
        ffmpeg_video(root / f"cam{k:02d}.mp4", **video)
    return root


# OpenCV-style calibrations and SelfCap mini-archives (from the converter tests; shared with the
# CLI tests).


YAML_K = np.array([[60.0, 0.0, 31.5], [0.0, 60.0, 23.5], [0.0, 0.0, 1.0]])  # OpenCV convention
YAML_D = np.array([[-0.12], [0.03], [0.0005], [-0.0004], [0.0]])


def rodrigues(vector: np.ndarray) -> np.ndarray:
    import cv2  # OpenCV, the converter tests' oracle (D-61), imported where needed

    matrix, _ = cv2.Rodrigues(vector)
    return np.asarray(matrix)


def write_opencv_calibration(
    root: Path,
    names: list[str],
    *,
    rot_error: float = 0.0,
    ccm: np.ndarray | None = None,
    distortion: np.ndarray = YAML_D,
    sizes: tuple[int, int] | None = (64, 48),
) -> None:
    """intri.yml and extri.yml as OpenCV's FileStorage writes them, in the SelfCap key family
    (D-63): names, K_, D_, H_, W_, (ccm_), R_ (Rodrigues), Rot_, T_."""
    import cv2

    root.mkdir(parents=True, exist_ok=True)
    intri = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    intri.write("names", names)
    extri.write("names", names)
    for k, name in enumerate(names):
        intri.write(f"K_{name}", YAML_K)
        intri.write(f"D_{name}", distortion)
        if sizes is not None:
            intri.write(f"W_{name}", sizes[0])
            intri.write(f"H_{name}", sizes[1])
        if ccm is not None:
            intri.write(f"ccm_{name}", ccm)
        vector = np.array([[0.0], [0.1 * k], [0.0]])
        rot = rodrigues(vector)
        rot[0, 0] += rot_error
        extri.write(f"R_{name}", vector)
        extri.write(f"Rot_{name}", rot)
        extri.write(f"T_{name}", np.array([[0.2 * k], [0.0], [3.0]]))
    intri.release()
    extri.release()


SELFCAP_NAMES = ["0000", "0001", "0002"]
SELFCAP_OFFSETS = {"0000": 0.004, "0001": -0.0125, "0002": 0.02}  # seconds, as sync.json holds them


def selfcap_archives(
    root: Path,
    seq: str,
    *,
    calib_archive: bool = False,
    sync: bool = True,
    frames: int = 12,
    distortion: np.ndarray | None = None,
    **calibration: Any,
) -> Path:
    """A generated mini-archive in SelfCap's layout (D-63, Q-101): `<seq>-release.tar.gz` with
    `<seq>-release/videos/<name>.mp4` and `<seq>-release/optimized/{intri.yml, extri.yml,
    sync.json}`, sync.json an object of camera name → seconds; with `calib_archive`, the
    calibration goes to `<seq>-calib.tar.gz` instead (as hair-calib does)."""
    staging = root / f"staging-{seq}"
    release, optimized = staging / f"{seq}-release", staging / "calib"
    zero = np.zeros((5, 1)) if distortion is None else distortion
    write_opencv_calibration(optimized, SELFCAP_NAMES, distortion=zero, **calibration)
    if sync:
        (optimized / "sync.json").write_text(json.dumps(SELFCAP_OFFSETS))
    for name in SELFCAP_NAMES:
        ffmpeg_video(release / "videos" / f"{name}.mp4", frames=frames)
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src / f"{seq}-release.tar.gz", "w:gz") as tar:
        tar.add(release / "videos", arcname=f"{seq}-release/videos")
        if not calib_archive:
            tar.add(optimized, arcname=f"{seq}-release/optimized")
    if calib_archive:
        with tarfile.open(src / f"{seq}-calib.tar.gz", "w:gz") as tar:
            tar.add(optimized, arcname=f"{seq}-calib/optimized")
    return src


_COLMAP_STUB = """#!{python}
# A stub COLMAP (U-37): records its arguments and input model, undistorts by copying, and
# writes a PINHOLE model with the FULL_OPENCV camera's intrinsics.
import json, pathlib, shutil, sys
args = sys.argv[1:]
log = pathlib.Path({log!r})
log.mkdir(parents=True, exist_ok=True)
if not args or args[0] == "help":
    print("COLMAP stub -- for tests")
    sys.exit(0)
opts = {{args[i]: args[i + 1] for i in range(1, len(args) - 1, 2)}}
with open(log / "calls.jsonl", "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "image_undistorter":
    model, out = pathlib.Path(opts["--input_path"]), pathlib.Path(opts["--output_path"])
    shutil.copytree(model, log / "input_model", dirs_exist_ok=True)
    text = [l for l in (model / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for line in text[::2]:
        [float(v) for v in line.split()[1:8]]  # COLMAP parses QW … TZ as numbers
    shutil.copytree(opts["--image_path"], out / "images", dirs_exist_ok=True)
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    lines = []
    for line in (model / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            lines.append(" ".join([p[0], "PINHOLE", p[2], p[3], *p[4:8]]))
    (out / "sparse" / "cameras.txt").write_text("\\n".join(lines) + "\\n")
    shutil.copy(model / "images.txt", out / "sparse" / "images.txt")
    (out / "sparse" / "points3D.txt").write_text("")
elif args[0] == "model_converter":
    out = pathlib.Path(opts["--output_path"])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(pathlib.Path(opts["--input_path"]) / name, out / name)
else:
    sys.exit(f"unknown command {{args[0]}}")
"""


def stub_colmap(root: Path) -> tuple[Path, Path]:
    path, log = root / "bin" / "colmap", root / "colmap-log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_COLMAP_STUB.format(python=sys.executable, log=str(log)))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path, log


# Invalid ftgs-if files (U-22; from the io tests, shared with the CLI tests).


def exchange_arrays(n: int = 6, sh_degree: int = 3, seed: int = 0) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[:, :1] < 0, -q, q)
    return {
        "position": torch.randn(n, 3, generator=g),
        "time_center": torch.rand(n, generator=g, dtype=torch.float64) * 10.0 + 2.0,
        "duration": torch.rand(n, generator=g, dtype=torch.float64) + 0.05,
        "velocity": torch.randn(n, 3, generator=g),
        "scale": torch.rand(n, 3, generator=g) + 0.01,
        "rotation": q,
        "opacity": torch.rand(n, generator=g) * 0.98 + 0.01,
        "sh": torch.randn(n, (sh_degree + 1) ** 2, 3, generator=g),
    }


def exchange_metadata(**override: str) -> dict[str, str]:
    meta = {
        "format": "ftgs-if",
        "version": "1",
        "sh_degree": "3",
        "time_unit": "s",
        "tau0": "2.0",
        "tau1": "12.0",
        "producer": "test",
    }
    meta.update(override)
    return meta


BAD_EXCHANGE_CASES = (
    "nan position",
    "inf sh",
    "zero duration",
    "negative duration",
    "opacity 0",
    "opacity 1",
    "zero scale",
    "zero rotation",
    "rotation [N, 3]",
    "sh of degree 2 for 3",
    "velocity of another N",
    "float32 time_center",
    "float64 position",
    "missing opacity",
    "empty",
    "version 2",
    "sh_degree 4",
    "tau1 < tau0",
    "missing tau0",
)


def bad_exchange(case: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    a = exchange_arrays()
    meta = exchange_metadata()
    if case == "nan position":
        a["position"][0, 0] = float("nan")
    elif case == "inf sh":
        a["sh"][1, 4, 2] = float("inf")
    elif case == "zero duration":
        a["duration"][2] = 0.0
    elif case == "negative duration":
        a["duration"][2] = -0.5
    elif case == "opacity 0":
        a["opacity"][3] = 0.0
    elif case == "opacity 1":
        a["opacity"][3] = 1.0
    elif case == "zero scale":
        a["scale"][0, 1] = 0.0
    elif case == "zero rotation":
        a["rotation"][4] = 0.0
    elif case == "rotation [N, 3]":
        a["rotation"] = a["rotation"][:, :3].contiguous()
    elif case == "sh of degree 2 for 3":
        a["sh"] = a["sh"][:, :9].contiguous()
    elif case == "velocity of another N":
        a["velocity"] = a["velocity"][:5].contiguous()
    elif case == "float32 time_center":
        a["time_center"] = a["time_center"].float()
    elif case == "float64 position":
        a["position"] = a["position"].double()
    elif case == "missing opacity":
        del a["opacity"]
    elif case == "empty":
        a = {k: v[:0] for k, v in a.items()}
    elif case == "version 2":
        meta["version"] = "2"
    elif case == "sh_degree 4":
        meta["sh_degree"] = "4"
        a["sh"] = torch.zeros(6, 25, 3)
    elif case == "tau1 < tau0":
        meta["tau1"] = "1.0"
    elif case == "missing tau0":
        del meta["tau0"]
    return a, meta
