# SPDX-License-Identifier: Apache-2.0
"""The synthetic scene of the GPU CI (§3.4: "a synthetic scene the size of the Neural3DV setup
(20 cameras, 300 frames, 1352×1014, 1M primitives)"; Q-179, Q-T235.2; #235). Datasets never
enter CI (§2.3), so the CI generates its scene.

    python scripts/ci_scene.py OUT [--jobs J] [--seed S] [--cameras C] [--frames F]
                               [--width W] [--height H] [--short-frames G]

Writes, under the folder OUT:

- `images/camXX/NNNNNN.png`: C cameras × F frames of 8-bit RGB PNGs, encoded by J processes
  (default: one per CPU) at the converters' fast PNG level;
- `scene.json`: the ftgs-scene manifest of every frame (§2.2), written with
  `openftgs.io.write_scene_manifest`; frame k at k/30 s, with frame index k; cam00 is the
  held-out test camera, as Neural3DV's is (§1.4);
- `scene-G.json` (G = 60 by default): the first G frames of the same PNGs, on which B-6
  measures the `gpu` placement of the frame cache, since the full scene's cache doesn't fit on a
  24 GiB GPU (Q-178).

The cameras stand on a horizontal arc around a common point and look at it (Q-T235.2), with a
60° horizontal field of view and the principal point at the image centre, as the Neural3DV
converter sets it (D-28). The images are procedural, not renders of a 3D scene: the benchmarks
time computations whose cost doesn't depend on the pixels, and the models they time are
synthesized around the cameras' common point (Q-181), at the sizes each benchmark sets (B-2 and
B-5 at 1M and 3M rows, B-3 from 10⁵ to 3·10⁶, B-6 at 1M). What the pixels do set is the size of a
PNG and the time to decode one, which B-6's `off` placement pays: they hold smooth colour, fine
texture, moving discs and sensor-like noise, so a PNG compresses about as a converted
openftgs-synth frame does (Q-T235.2).

Every frame comes from its own seed (the scene's seed, the camera, the frame), so the output
doesn't depend on the number of processes, and the manifests hold only the parameters, so their
bytes repeat. A folder generated before with the same parameters is completed rather than
refused: PNGs already there are kept, which lets a runner keep its scene and its frame cache
between runs. Exit 0 once the scene is written, 2 for invalid arguments or a folder that holds
something else.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_VERSION = 1  # the generator's; a change to the images or the rig changes it, and the baselines
_FPS = 30  # §1.4: Neural3DV's frame k at k/30 s
_TEST_CAMERA = 0  # §1.4: cam00 is held out, as Neural3DV's
_RADIUS = 4.0  # Q-T235.2: the arc's radius around the common point
_HEIGHT = 1.0  # Q-T235.2: the cameras' height above it
_ARC = math.radians(90.0)  # Q-T235.2: the arc's span
_HFOV = math.radians(60.0)  # Q-T235.2: the horizontal field of view
_PNG_LEVEL = 1  # as the converters' (openftgs.data, §2.4): PNG is lossless, so pixels don't move
_CHUNK = 25  # frames per pool task
_STAMP = "ci-scene.json"  # the parameters of the generation the folder holds
_DISCS = 12  # Q-T235.2: moving discs over the static content
_NOISE = 1.0  # Q-T235.2: the noise's standard deviation, in 8-bit levels


class InputError(Exception):
    """Invalid arguments or output folder: exit 2."""


@dataclass(frozen=True)
class Params:
    """What a generation depends on; recorded in the manifests' metadata and the folder's
    stamp."""

    seed: int = 0
    cameras: int = 20
    frames: int = 300
    width: int = 1352
    height: int = 1014
    short_frames: int = 60

    def check(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InputError(f"--{name.replace('_', '-')} must be an int >= 0, got {value}")
        if self.cameras < 2:
            raise InputError("--cameras must be at least 2: one test camera and a training one")
        if min(self.frames, self.width, self.height) < 1:
            raise InputError("--frames, --width and --height must be at least 1")
        if not 1 <= self.short_frames <= self.frames:
            raise InputError(f"--short-frames must lie in [1, {self.frames}]")

    def stamp(self) -> dict[str, Any]:
        return {"generator": "scripts/ci_scene.py", "version": _VERSION, **asdict(self)}


def camera_name(index: int) -> str:
    return f"cam{index:02d}"


def image_path(out: Path, camera: int, frame: int) -> Path:
    return out / "images" / camera_name(camera) / f"{frame:06d}.png"


# -- the rig (Q-T235.2) ----------------------------------------------------------------------------


def rig(params: Params) -> list[tuple[list[list[float]], list[list[float]]]]:
    """Per camera, its K (continuous pixel convention, §1.4) and its world-to-camera matrix
    (OpenCV axes): evenly spaced on a horizontal arc of radius `_RADIUS` at `_HEIGHT` above the
    origin, z up, each looking at the origin. Python's math module, not NumPy's vector kernels,
    so the values repeat across machines."""
    focal = 0.5 * params.width / math.tan(0.5 * _HFOV)
    K = [[focal, 0.0, 0.5 * params.width], [0.0, focal, 0.5 * params.height], [0.0, 0.0, 1.0]]
    cameras = []
    for k in range(params.cameras):
        angle = _ARC * (k / (params.cameras - 1) - 0.5)
        centre = (_RADIUS * math.sin(angle), -_RADIUS * math.cos(angle), _HEIGHT)
        forward = _unit(tuple(-c for c in centre))
        right = _unit(_cross(forward, (0.0, 0.0, 1.0)))
        down = _cross(forward, right)
        rows = (right, down, forward)  # OpenCV: x right, y down, z forward
        t = [-sum(r[i] * centre[i] for i in range(3)) for r in rows]
        w2c = [[*rows[i], t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        cameras.append((K, w2c))
    return cameras


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v: tuple[float, ...]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(x * x for x in v))
    return (v[0] / norm, v[1] / norm, v[2] / norm)


# -- the images (Q-T235.2) -------------------------------------------------------------------------


def _interpolation(size: int, points: int) -> np.ndarray:
    """[size, points] weights of linear interpolation from `points` evenly spaced samples, the
    first and last on the first and last pixel centres."""
    x = np.linspace(0.0, points - 1.0, size)
    low = np.minimum(np.floor(x).astype(np.int64), points - 2)
    frac = x - low
    weights = np.zeros((size, points))
    weights[np.arange(size), low] = 1.0 - frac
    weights[np.arange(size), low + 1] = frac
    return weights


@functools.lru_cache(maxsize=4)
def _background(seed: int, camera: int, width: int, height: int) -> np.ndarray:
    """A camera's static content, [H, W, 3] float32 in 8-bit levels: smooth colour from a coarse
    random grid, plus three oriented sinusoids of 6 to 48 px periods."""
    rng = np.random.default_rng([seed, 1, camera])
    grid = rng.uniform(40.0, 215.0, (9, 12, 3))
    rows, cols = _interpolation(height, 9), _interpolation(width, 12)
    image: np.ndarray = np.einsum("hjc,wj->hwc", np.einsum("hi,ijc->hjc", rows, grid), cols)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    for _ in range(3):
        angle = rng.uniform(0.0, math.pi)
        period = rng.uniform(6.0, 48.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        wave = np.sin(
            (x * math.cos(angle) + y * math.sin(angle)) * (2.0 * math.pi / period) + phase
        )
        image += wave[..., None] * rng.uniform(-14.0, 14.0, 3)
    return image.astype(np.float32)


@functools.lru_cache(maxsize=4)
def _discs(seed: int, width: int, height: int) -> np.ndarray:
    """Per disc: x, y (px at t = 0), vx, vy (px/s), radius (px) and an RGB colour."""
    rng = np.random.default_rng([seed, 0])
    size = min(width, height)
    return np.column_stack(
        [
            rng.uniform(0.0, width, _DISCS),
            rng.uniform(0.0, height, _DISCS),
            rng.uniform(-0.1, 0.1, (_DISCS, 2)) * size,
            rng.uniform(0.03, 0.12, _DISCS) * size,
            rng.uniform(20.0, 235.0, (_DISCS, 3)),
        ]
    )


def frame(params: Params, camera: int, index: int) -> np.ndarray:
    """One frame as [H, W, 3] uint8: the camera's background, the discs where they are at
    index/30 s (shifted per camera, as parallax would), and the frame's own noise."""
    width, height = params.width, params.height
    image = _background(params.seed, camera, width, height).copy()
    t = index / _FPS
    shift = (camera - 0.5 * (params.cameras - 1)) * 0.01 * width
    for x0, y0, vx, vy, radius, *colour in _discs(params.seed, width, height):
        cx = (x0 + vx * t + shift) % width
        cy = (y0 + vy * t) % height
        top, bottom = max(0, int(cy - radius - 2)), min(height, int(cy + radius + 3))
        left, right = max(0, int(cx - radius - 2)), min(width, int(cx + radius + 3))
        if top >= bottom or left >= right:
            continue
        y, x = np.mgrid[top:bottom, left:right]
        alpha = np.clip(radius - np.hypot(x + 0.5 - cx, y + 0.5 - cy), 0.0, 1.0)[..., None]
        patch = image[top:bottom, left:right]
        patch += alpha.astype(np.float32) * (np.asarray(colour, np.float32) - patch)
    rng = np.random.default_rng([params.seed, 2, camera, index])
    image += rng.standard_normal(image.shape, dtype=np.float32) * np.float32(_NOISE)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class _Task:
    params: Params
    out: Path
    camera: int
    frames: tuple[int, ...]


def _write_frames(task: _Task) -> int:
    """A pool task: the frames of one camera not yet written, each PNG written to a temporary
    sibling and renamed, so a PNG that exists is whole. Returns how many it wrote."""
    written = 0
    for index in task.frames:
        path = image_path(task.out, task.camera, index)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            image = Image.fromarray(frame(task.params, task.camera, index))  # [H, W, 3] uint8: RGB
            image.save(temporary, format="PNG", compress_level=_PNG_LEVEL)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        written += 1
    return written


# -- the folder and the manifests ------------------------------------------------------------------


def _prepare(out: Path, params: Params) -> None:
    """The folder must be absent, empty, or hold a generation with the same parameters."""
    stamp = out / _STAMP
    if out.exists() and not out.is_dir():
        raise InputError(f"{out}: not a folder")
    if out.is_dir() and any(out.iterdir()):
        try:
            same = json.loads(stamp.read_text(encoding="utf-8")) == params.stamp()
        except (OSError, ValueError):
            same = False
        if not same:
            raise InputError(
                f"{out} holds something other than a CI scene with these parameters; "
                "remove it or choose another folder"
            )
    out.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(params.stamp(), indent=1) + "\n", encoding="utf-8")


def _write_images(out: Path, params: Params, jobs: int) -> int:
    tasks = [
        _Task(params, out, camera, tuple(range(start, min(start + _CHUNK, params.frames))))
        for camera in range(params.cameras)
        for start in range(0, params.frames, _CHUNK)
    ]
    written = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for done, count in enumerate(pool.map(_write_frames, tasks), start=1):
            written += count
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tasks", file=sys.stderr, flush=True)
    return written


def _write_manifests(out: Path, params: Params) -> list[Path]:
    """`scene.json` and `scene-G.json` through the package's manifest writer (§2.2)."""
    import torch

    from openftgs.io import write_scene_manifest
    from openftgs.types import Camera, ImageRecord

    cameras = [
        Camera(
            K=torch.tensor(K, dtype=torch.float32),
            world_to_camera=torch.tensor(w2c, dtype=torch.float32),
            width=params.width,
            height=params.height,
            name=camera_name(k),
        )
        for k, (K, w2c) in enumerate(rig(params))
    ]
    written = []
    for name, frames in (
        ("scene.json", params.frames),
        (f"scene-{params.short_frames}.json", params.short_frames),
    ):
        records = [
            ImageRecord(
                camera,
                index / _FPS,
                image_path(out, k, index),
                "test" if k == _TEST_CAMERA else "train",
                index,
            )
            for k, camera in enumerate(cameras)
            for index in range(frames)
        ]
        metadata = {
            **params.stamp(),
            "manifest_frames": frames,
            "fps": _FPS,
            "test_camera": camera_name(_TEST_CAMERA),
            # No Q-n here: renumbering a row must not change the manifest's bytes, by which
            # B-6's frame cache is keyed.
            "purpose": "the GPU CI scene of §3.4, with procedural images",
        }
        write_scene_manifest(out / name, records, time_unit="s", metadata=metadata)
        written.append(out / name)
    return written


def generate(out: Path, params: Params, *, jobs: int = 0) -> list[Path]:
    """Write the scene under `out` (module docstring) and return its two manifests."""
    params.check()
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
        raise InputError(f"--jobs must be an int >= 0, got {jobs}")
    _prepare(out, params)
    began = time.perf_counter()
    written = _write_images(out, params, jobs or os.cpu_count() or 1)
    manifests = _write_manifests(out, params)
    total = params.cameras * params.frames
    print(
        f"{written} PNGs written, {total - written} kept, in {time.perf_counter() - began:.0f} s",
        file=sys.stderr,
    )
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the synthetic scene of the GPU CI (§3.4)")
    parser.add_argument("out", type=Path, metavar="OUT")
    parser.add_argument("--jobs", type=int, default=0, help="processes; 0: one per CPU")
    defaults = Params()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=value)
    args = parser.parse_args(argv)
    params = Params(**{name: getattr(args, name) for name in asdict(defaults)})
    try:
        for manifest in generate(args.out, params, jobs=args.jobs):
            print(manifest)
    except InputError as err:
        print(f"ci_scene: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
