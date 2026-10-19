# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench`: the performance benchmarks B-1 to B-8 of §3.4, each a subcommand that writes
BENCH.json.

Spec: §2.1 (the `ftgs bench` line and the exit codes), §3.4, D-50. Q-166 (BENCH.json and the
exit codes), Q-167 (`--model`), Q-168 (the setups). docs/modules/cli/benchmarks.md maps each
subcommand to its B-n.

Each subcommand has an entry in `_BENCHMARKS`: its B-n, the function that runs it, and whether
it needs CUDA and takes `--model`. The framework checks the scene and the environment, runs
the function and writes BENCH.json; a benchmark whose ticket hasn't landed has no function yet
and keeps Q-130's behaviour: it says so, writes nothing and returns 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openftgs.cli import _INPUT, _fail, _load_model, _scene

if TYPE_CHECKING:
    from torch import Tensor

    from openftgs.data import MultiViewVideo
    from openftgs.model import FreeTimeGaussians
    from openftgs.types import TimeNormalizer

_UNBUILT = 1  # Q-130: a benchmark whose ticket hasn't landed


@dataclass(frozen=True)
class _Benchmark:
    """One subcommand: its §3.4 ID, the ticket that builds it, the function that runs it (None
    until that ticket lands), whether it needs CUDA (all but `convert`, Q-166) and whether it
    takes `--model` (Q-167)."""

    id: str
    ticket: int
    run: Callable[..., Any] | None = None
    cuda: bool = True
    model: bool = False


_BENCHMARKS: dict[str, _Benchmark] = {
    "train-step": _Benchmark("B-1", 226, model=True),
    "raster": _Benchmark("B-2", 227),
    "slice": _Benchmark("B-3", 225),
    "losses": _Benchmark("B-4", 228),
    "adam": _Benchmark("B-5", 229),
    "data": _Benchmark("B-6", 232),
    "render": _Benchmark("B-7", 231, model=True),
    "init": _Benchmark("B-8", 234),
    "convert": _Benchmark("B-8", 234, cuda=False),
}


def add_parser(commands: Any) -> None:
    """The `bench` command and its options, on the `ftgs` subparsers `commands`."""
    p = commands.add_parser("bench", help="performance benchmarks (§3.4)")
    p.add_argument("benchmark", choices=tuple(_BENCHMARKS))
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--out", type=Path, required=True, metavar="BENCH.json")
    # Q-167: train-step and render only; the spec's CLI line takes no model (a change request).
    p.add_argument("--model", type=Path, metavar="MODEL")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    """`ftgs bench BENCHMARK SCENE.json --out BENCH.json`."""
    import torch

    bench = _BENCHMARKS[args.benchmark]
    if args.model is not None and not bench.model:  # Q-167
        takers = " and ".join(name for name, b in _BENCHMARKS.items() if b.model)
        _fail(f"--model applies only to {takers}", _INPUT)
    _scene(args.scene)
    if bench.cuda and not torch.cuda.is_available():  # Q-166: an invalid environment
        _fail(f"{bench.id} needs a CUDA device", _INPUT)
    if args.model is not None:  # Q-167: MODEL as `ftgs render` reads it
        _load_model(args.model, "cuda")
    if bench.run is None:  # Q-130 until its ticket lands
        print(
            f"ftgs bench {args.benchmark}: {bench.id} comes with #{bench.ticket}; "
            "nothing was run or written",
            file=sys.stderr,
        )
        return _UNBUILT
    result: int = bench.run(args)
    return result


# -- the model synthesizer (Q-168, Q-T225.2) -------------------------------------------------------

_SPHERES = 32  # Q-168
_BACKGROUND_SHARE = 0.25  # Q-T225.2: the share of rows on the background sphere
_OPACITY = (0.3, 0.95)  # Q-168: σ ~ U[0.3, 0.95]
_NORMAL_AXIS = 0.2  # Q-168: the normal axis at 1/5 of the rows' spacing
_ALWAYS = 10.0  # Q-T225.2: s for a fully active model; its reach exceeds [0, 1] for every σ ≥ 0.3


@dataclass(frozen=True)
class _Volume:
    """Where a synthesized model goes (Q-T225.2): the point nearest every training camera's
    optical axis, the radius around it that every training camera sees, the largest distance
    from it to a training camera, the scene's S (D-27) and its time normalizer."""

    centre: tuple[float, float, float]
    view_radius: float
    camera_reach: float
    spatial_scale: float
    time: TimeNormalizer
    time_unit: str


def _volume(video: MultiViewVideo) -> _Volume:
    """The volume of `video`'s training cameras (Q-T225.2). ValueError for a scene with a single
    timestamp, where no fraction of rows can be inactive (D-71), or one training pose (D-27)."""
    import torch

    if video.time.tau1 == video.time.tau0:
        raise ValueError("a benchmark scene needs more than one timestamp")
    scale = video.spatial_scale(1.1)  # D-27's factor, TrainConfig.scene_scale_factor
    cameras = {r.camera.name: r.camera for r in video.records if r.split == "train"}
    centres, axes, half_fov = [], [], []
    for camera in cameras.values():
        w2c = camera.world_to_camera.to(torch.float64)
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        centres.append(-rotation.T @ translation)  # E2
        axes.append(rotation[2])  # the camera's +z axis in world coordinates
        K = camera.K.to(torch.float64)
        half_x = min(float(K[0, 2]), camera.width - float(K[0, 2])) / float(K[0, 0])
        half_y = min(float(K[1, 2]), camera.height - float(K[1, 2])) / float(K[1, 1])
        half_fov.append(min(half_x, half_y))  # tan of the narrower half field of view
    c, f = torch.stack(centres), torch.stack(axes)
    # Least squares for the point nearest every optical axis, Σ(I − f fᵀ)(p − c) = 0, with a
    # ridge toward a point S ahead of the cameras' centroid along their mean axis, which settles
    # the directions parallel axes leave free and nothing else.
    projectors = torch.eye(3, dtype=torch.float64) - f[:, :, None] * f[:, None, :]
    a, b = projectors.sum(0), (projectors @ c[:, :, None]).sum(0)[:, 0]
    mean_axis = f.mean(0)
    norm = float(torch.linalg.vector_norm(mean_axis))
    ahead = c.mean(0) + (scale * mean_axis / norm if norm > 1e-6 else 0.0)
    ridge = 1e-6 * float(a.trace())
    centre = torch.linalg.solve(a + ridge * torch.eye(3, dtype=torch.float64), b + ridge * ahead)
    distances = torch.linalg.vector_norm(c - centre, dim=1)
    view = min(float(d) * h for d, h in zip(distances, half_fov, strict=True))
    return _Volume(
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        view_radius=view,
        camera_reach=float(distances.max()),
        spatial_scale=scale,
        time=video.time,
        time_unit=video.manifest.time_unit,
    )


def _synthesize(
    volume: _Volume, n: int, *, active: float, seed: int, device: str, sh_degree: int = 3
) -> FreeTimeGaussians:
    """A seeded model of `n` rows in `volume` (Q-168, Q-T225.2): a quarter on a background
    sphere twice the cameras' reach around the centre, the rest on 32 random spheres in view of
    every camera, each moving at its own velocity; each row's normal axis along its sphere's
    normal, at 1/5 of its tangent scales, which are the rows' spacing on that sphere. A fraction
    `active` of the rows, in (0, 1], passes the opacity cull (D-32) at any t in [0, 1], in
    expectation; rows come in a random order, so any prefix is a smaller model of the same kind."""
    import math

    import torch

    from openftgs.model import FreeTimeGaussians
    from openftgs.sh import rgb_to_sh0

    if not 0.0 < active <= 1.0:
        raise ValueError(f"the active fraction must be in (0, 1], got {active}")
    g = torch.Generator(device=device).manual_seed(seed)

    def uniform(lo: float, hi: float, *shape: int) -> Tensor:
        return torch.empty(*shape, dtype=torch.float64, device=device).uniform_(lo, hi, generator=g)

    def directions(count: int) -> Tensor:
        d = torch.randn(count, 3, dtype=torch.float64, device=device, generator=g)
        unit: Tensor = d / torch.linalg.vector_norm(d, dim=1, keepdim=True).clamp_min(1e-12)
        return unit

    centre = torch.tensor(volume.centre, dtype=torch.float64, device=device)
    view = volume.view_radius
    sphere_centres = centre + directions(_SPHERES) * view * 0.5 * uniform(0, 1, _SPHERES, 1) ** (
        1 / 3
    )
    sphere_radii = uniform(0.05, 0.15, _SPHERES) * view
    sphere_speeds = directions(_SPHERES) * uniform(0.0, 0.25, _SPHERES, 1) * view  # per unit t
    sphere_colors = uniform(0.2, 0.8, _SPHERES, 3)

    n_back = int(n * _BACKGROUND_SHARE)
    n_fore = n - n_back
    weights = sphere_radii**2  # rows by area, so every sphere has about the same spacing
    which = torch.multinomial(weights, n_fore, replacement=True, generator=g)
    counts = torch.bincount(which, minlength=_SPHERES).to(torch.float64)
    back_radius = 2.0 * volume.camera_reach
    spacing_fore = torch.sqrt(4 * math.pi * sphere_radii**2 / counts.clamp_min(1))[which]
    spacing_back = math.sqrt(4 * math.pi * back_radius**2 / max(n_back, 1))

    normals = directions(n)  # the background rows first, then the spheres'
    radius = torch.cat([sphere_radii.new_full((n_back,), back_radius), sphere_radii[which]])
    spacing = torch.cat([spacing_fore.new_full((n_back,), spacing_back), spacing_fore])
    origin = torch.cat([centre.expand(n_back, 3), sphere_centres[which]])
    velocity = torch.cat([sphere_speeds.new_zeros(n_back, 3), sphere_speeds[which]])

    # Durations for the active fraction: a row passes the cull while |t − μt| ≤ reach, with
    # reach = s·√(2 ln(255σ)) (E3, E4, D-32). With μt ~ U[−r, 1 + r] and reach r, a time t in
    # [0, 1] finds 2r/(1 + 2r) of the rows active; r = f/(2(1 − f)) makes that f.
    sigma = uniform(*_OPACITY, n)
    if active < 1.0:
        reach = active / (2.0 * (1.0 - active))
        times = uniform(-reach, 1.0 + reach, n)
        durations = reach / torch.sqrt(2.0 * torch.log(255.0 * sigma))
    else:
        times = uniform(0.0, 1.0, n)
        durations = torch.full((n,), _ALWAYS, dtype=torch.float64, device=device)

    # Each row on its sphere as it is at the row's own time μt (the sphere's centre moves as
    # origin + v·(t − ½)), so E1 keeps it on the moving sphere at every t.
    means = origin + velocity * (times[:, None] - 0.5) + radius[:, None] * normals
    # The rotation taking +z to the normal: q = (1 + z·n, z × n), normalized; −z needs 180°.
    w = 1.0 + normals[:, 2]
    quats = torch.stack([w, -normals[:, 1], normals[:, 0], torch.zeros_like(w)], dim=1)
    flipped = w < 1e-9
    quats[flipped] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    quats = quats / torch.linalg.vector_norm(quats, dim=1, keepdim=True)
    log_scales = torch.log(spacing)[:, None] + torch.log(
        torch.tensor([1.0, 1.0, _NORMAL_AXIS], dtype=torch.float64, device=device)
    )
    colors = torch.cat([uniform(0.3, 0.7, n_back, 3), sphere_colors[which]])
    colors = colors + 0.05 * torch.randn(n, 3, dtype=torch.float64, device=device, generator=g)
    bands = (sh_degree + 1) ** 2 - 1
    params = {
        "means": means,
        "times": times,
        "log_durations": torch.log(durations),
        "velocities": velocity,
        "log_scales": log_scales,
        "quats": quats,
        "opacity_logits": torch.log(sigma / (1.0 - sigma)),
        "sh0": rgb_to_sh0(colors.clamp(0.0, 1.0))[:, None, :],
        "shN": 0.02 * torch.randn(n, bands, 3, dtype=torch.float64, device=device, generator=g),
    }
    order = torch.randperm(n, device=device, generator=g)
    return FreeTimeGaussians(
        {name: value[order].to(torch.float32) for name, value in params.items()},
        sh_degree=sh_degree,
        time=volume.time,
        time_unit=volume.time_unit,
    )
