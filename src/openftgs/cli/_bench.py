# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench`: the performance benchmarks B-1 to B-8 of §3.4, each a subcommand that writes
BENCH.json.

Spec: §2.1 (the `ftgs bench` line and the exit codes), §3.4, D-50, D-51, D-55, D-56, D-81,
D-88 (`--model`). Q-166 (BENCH.json and the exit codes), Q-168 (the setups), Q-178 and Q-189
(B-6), Q-174 (the index threshold T that B-7 sets), Q-194 and Q-195 (B-7), Q-169, Q-192
and Q-193 (B-8). docs/modules/cli/benchmarks.md maps each subcommand to its B-n.

Each subcommand has an entry in `_BENCHMARKS`, at the end of this module: its B-n, the function
that runs it, and whether it needs CUDA and takes `--model`. The framework checks the scene and
the environment, runs the function, which returns its setup, cases and budgets, and writes
BENCH.json. Every subcommand runs its benchmark: #225 to #235 retired Q-130's stub, which
returned 1. Adding one is a function and its entry.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Literal

from openftgs.cli import (
    _INPUT,
    _NUMERIC,
    _as_input_error,
    _fail,
    _json_text,
    _load_model,
    _Loaded,
    _output,
    _scene,
)
from openftgs.cli import _environment as _environment_error

if TYPE_CHECKING:
    from torch import Tensor

    from openftgs._stages import Timing
    from openftgs.config import TrainConfig
    from openftgs.data import MultiViewVideo
    from openftgs.eval import Evaluator
    from openftgs.model import FreeTimeGaussians
    from openftgs.optim import FTGSOptimizer
    from openftgs.train import StepLog, Trainer
    from openftgs.types import Camera, RenderRequest, TimeNormalizer

__all__ = ["add_parser", "run"]  # what `openftgs.cli` takes from this private module

_FORMAT, _VERSION = "ftgs-bench", 1  # Q-166


@dataclass(frozen=True)
class _Context:
    """What a benchmark gets: the parsed arguments, the scene, and `--model` read on CUDA
    (D-88), None when not given."""

    args: argparse.Namespace
    video: MultiViewVideo
    model: _Loaded | None


@dataclass
class _Result:
    """What a benchmark returns for BENCH.json (Q-166): its setup, its cases, each
    `{"case": {parameter: value}, "figures": {name: value}}`, and its budgets, each
    `{"case", "figure", "limit", "value", "pass", "kind"}`."""

    setup: dict[str, Any]
    cases: list[dict[str, Any]]
    budgets: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Benchmark:
    """One subcommand: its §3.4 ID, the function that runs it, whether it needs CUDA (all but
    `convert`, Q-166) and whether it takes `--model` (D-88)."""

    id: str
    run: Callable[[_Context], _Result]
    cuda: bool = True
    model: bool = False


def add_parser(commands: Any) -> None:
    """The `bench` command and its options, on the `ftgs` subparsers `commands`."""
    p = commands.add_parser("bench", help="performance benchmarks (§3.4)")
    p.add_argument("benchmark", choices=tuple(_BENCHMARKS))
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--out", type=Path, required=True, metavar="BENCH.json")
    # D-88: train-step and render only.
    p.add_argument("--model", type=Path, metavar="MODEL")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    """`ftgs bench BENCHMARK SCENE.json --out BENCH.json [--model MODEL]`: exit 0 once the
    benchmark ran, whatever its budgets; 2 for invalid input or environment, out of GPU memory
    included; 3 for a numerical failure (Q-166 as answered, Q-199)."""
    import torch

    bench = _BENCHMARKS[args.benchmark]
    if args.model is not None and not bench.model:  # D-88
        takers = " and ".join(name for name, b in _BENCHMARKS.items() if b.model)
        _fail(f"--model applies only to {takers}", _INPUT)
    video = _scene(args.scene)
    if bench.cuda and not torch.cuda.is_available():  # Q-166: an invalid environment
        _fail(f"{bench.id} needs a CUDA device", _INPUT)
    # Q-166 (as answered): a run cut short measured nothing, so it writes nothing; out of GPU
    # memory is exit 2 and a numerical failure exit 3, as in `ftgs train` (Q-142; Q-199).
    try:
        model = None if args.model is None else _load_model(args.model, "cuda")  # D-88
        with _output(f"BENCH.json {args.out}"):  # Q-142: found out before the run, not after
            args.out.parent.mkdir(parents=True, exist_ok=True)
        date = datetime.now(UTC).isoformat(timespec="seconds")
        result = bench.run(_Context(args, video, model))
    except torch.OutOfMemoryError as error:
        _fail(f"{bench.id} ran out of GPU memory: {error}", _INPUT)
    except FloatingPointError as error:  # the trainer's divergence (Q-69), a render's (Q-129)
        _fail(f"{bench.id}: {error}", _NUMERIC)
    command = ["ftgs", "bench", args.benchmark, str(args.scene), "--out", str(args.out)]
    if args.model is not None:
        command += ["--model", str(args.model)]
    body = {
        "format": _FORMAT,
        "version": _VERSION,
        "benchmark": bench.id,
        "command": command,
        "environment": _environment(),
        "scene": _scene_entry(args.scene, video),
        "date": date,
        "setup": result.setup,
        "cases": result.cases,
        "budgets": result.budgets,
    }
    _write(args.out, body)
    print(args.out)
    return 0


# -- BENCH.json (Q-166, Q-180) ---------------------------------------------------------------------


def _write(path: Path, body: dict[str, Any]) -> None:
    """Q-126: replaced atomically through a temporary sibling, as the other outputs are."""
    text = _json_text(body)  # Q-25: no NaN or Infinity
    with _output(f"BENCH.json {path}"):  # Q-142
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(text + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(*command: str, cwd: Path | None = None) -> str | None:
    """A program's standard output, stripped, or None when it can't run or fails."""
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _driver() -> str | None:
    """The NVIDIA driver's version from nvidia-smi (PyTorch has no public accessor; source log
    162): `driver_version`, deprecated for `kmd_version`, which older drivers lack."""
    for query in ("driver_version", "kmd_version"):
        out = _run("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader")
        if out:
            return out.splitlines()[0].strip()
    return None


def _commit() -> tuple[str | None, bool | None]:
    """Q-180: the commit of the git checkout this openftgs is imported from, and whether its
    tracked files differ from it; (None, None) when the package isn't a tracked file of one, as
    when installed from a wheel."""
    import openftgs

    package = Path(openftgs.__file__).resolve().parent
    if _run("git", "ls-files", "--error-unmatch", "__init__.py", cwd=package) is None:
        return None, None
    commit = _run("git", "rev-parse", "HEAD", cwd=package)
    status = _run("git", "status", "--porcelain", "--untracked-files=no", cwd=package)
    return commit, None if status is None else bool(status)


def _cpu() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _environment() -> dict[str, Any]:
    """Q-166, §3.4: the GPU's model and count, the driver, CUDA, cuDNN, PyTorch, gsplat,
    fused-ssim and Python versions, and the commit with its dirty flag; also openftgs's version
    and the CPU's model (Q-180)."""
    import torch

    cuda = torch.cuda.is_available()
    commit, dirty = _commit()
    return {
        "gpu": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()) if cuda else None,
            "count": torch.cuda.device_count() if cuda else 0,
        },
        "driver": _driver(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "torch": torch.__version__,
        "gsplat": _version("gsplat"),
        "fused_ssim": _version("fused_ssim"),
        "python": platform.python_version(),
        "openftgs": _version("openftgs"),
        "cpu": _cpu(),
        "commit": commit,
        "dirty": dirty,
    }


def _scene_entry(path: Path, video: MultiViewVideo) -> dict[str, Any]:
    """Q-166, Q-180: the manifest's path and SHA-256, its cameras, the most images any camera
    has (its frames), and the size of the first training image."""
    per_camera: dict[str, int] = {}
    for record in video.records:
        per_camera[record.camera.name] = per_camera.get(record.camera.name, 0) + 1
    first = next(r.camera for r in video.records if r.split == "train")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cameras": len(per_camera),
        "frames": max(per_camera.values()),
        "width": first.width,
        "height": first.height,
    }


# -- measurement -----------------------------------------------------------------------------------

_WARMUP = 10  # Q-168, Q-182: untimed calls before the timed ones


def _first_call(body: Callable[[], None]) -> float:
    """The wall time of one call of `body` from an idle device until its work is done, so it
    includes compilation and the first launches; it synchronizes, so it never runs inside a
    timed repetition."""
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    body()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _repeat(body: Callable[[], None], repetitions: int) -> list[Timing]:
    """`_WARMUP` calls of `body`, then `repetitions` calls under the stage clock's CUDA events
    (§3.4, `openftgs._stages`); `body` names its parts with `stage`. Nothing synchronizes until
    the clock is read after the last call."""
    from openftgs._stages import Clock

    for _ in range(_WARMUP):
        body()
    with Clock("cuda") as clock:
        for _ in range(repetitions):
            body()
    return clock.read()


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


# -- the model synthesizer (Q-168, Q-181) ----------------------------------------------------------

_SPHERES = 32  # Q-168
_BACKGROUND_SHARE = 0.25  # Q-181: the share of rows on the background sphere
_OPACITY = (0.3, 0.95)  # Q-168: σ ~ U[0.3, 0.95]
_NORMAL_AXIS = 0.2  # Q-168: the normal axis at 1/5 of the rows' spacing
_ALWAYS = 10.0  # Q-181: s for a fully active model; its reach exceeds [0, 1] for every σ ≥ 0.3


@dataclass(frozen=True)
class _Volume:
    """Where a synthesized model goes (Q-181): the point nearest every training camera's
    optical axis, the radius around it that every training camera sees, the largest distance
    from it to a training camera, the scene's S (D-27) and its time normalizer."""

    centre: tuple[float, float, float]
    view_radius: float
    camera_reach: float
    spatial_scale: float
    time: TimeNormalizer
    time_unit: str


def _volume(video: MultiViewVideo) -> _Volume:
    """The volume of `video`'s training cameras (Q-181). ValueError for a scene with a single
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
    """A seeded model of `n` rows in `volume` (Q-168, Q-181): a quarter on a background
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
    offsets = directions(_SPHERES) * uniform(0.0, 1.0, _SPHERES, 1) ** (1 / 3)  # in a unit ball
    sphere_centres = centre + 0.5 * view * offsets
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


# -- B-3: the temporal slice's implementations (§3.4, D-50; Q-13, Q-182) --------------------------

# Q-182: 11 sizes, so 10 resizes, each keeping the rows before it and appending new ones, as
# growth events do; B-1's 0.5M, 1M and 3M among them.
_SLICE_SIZES = (
    100_000,
    150_000,
    200_000,
    300_000,
    400_000,
    500_000,
    700_000,
    1_000_000,
    1_500_000,
    2_000_000,
    3_000_000,
)
_SLICE_REPETITIONS = 50  # #225: the median of 50
# Explicit, never "auto", which may fall back to eager unannounced in the numbers (Q-13).
_SLICE_IMPLS: tuple[Literal["eager", "compiled"], ...] = ("eager", "compiled")
_SLICE_INPUTS = (
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
)
# The outputs a training iteration differentiates: the render's four and E8's terms (`at_time`).
_SLICE_GRADIENTS = ("means_t", "opacities_t", "quats_n", "scales", "reg_terms")
_RECOMPILE_LIMIT = 2  # U-28: at most 2 recompilations over 10 resizes
_SEED = 0  # Q-168: the synthesized models are seeded
_SLICE_ACTIVE = 0.2  # Q-182: the share active at any time, as B-1's main cases


def _slice_call(
    leaves: list[Tensor], grads: list[Tensor], impl: Literal["eager", "compiled"], times: Random
) -> Callable[[], None]:
    """One B-3 call: the slice at the next time, as the stage `slice.kernel`, then its backward
    from `grads` into `leaves`, as the stage `backward`; `torch.autograd.grad` accumulates
    nothing between calls."""
    import torch

    from openftgs._stages import stage
    from openftgs.motion import temporal_slice

    means, times_, log_durations, velocities, log_scales, quats, opacity_logits = leaves

    def call() -> None:
        t = times.random()
        with stage("slice.kernel"):
            out = temporal_slice(
                means,
                times_,
                log_durations,
                velocities,
                log_scales,
                quats,
                opacity_logits,
                t,
                impl=impl,
            )
        outputs = [getattr(out, name) for name in _SLICE_GRADIENTS]
        with stage("backward"):
            torch.autograd.grad(outputs, leaves, grads)

    return call


def _slice(ctx: _Context) -> _Result:
    """B-3: the forward and backward of `motion.temporal_slice`, eager and compiled, at N from
    10⁵ to 3·10⁶, grown as at growth events: the time of the first call at each size, which
    includes compilation, then the medians of 50 calls after a warm-up; the recompilations over
    the 10 resizes, counted as Q-13 counts them, against U-28's limit."""
    import random

    import torch
    from torch._dynamo.utils import counters  # Q-13: dynamo's count of unique graphs

    with _as_input_error(f"scene manifest {ctx.args.scene}"):
        volume = _volume(ctx.video)
    model = _synthesize(volume, _SLICE_SIZES[-1], active=_SLICE_ACTIVE, seed=_SEED, device="cuda")
    stored = [getattr(model, name).detach() for name in _SLICE_INPUTS]
    del model
    times = random.Random(_SEED)  # the times of the calls, uniform in [0, 1]

    torch.compiler.reset()  # the first compiled call compiles, whatever ran before
    counters.clear()
    cases: list[dict[str, Any]] = []
    for n in _SLICE_SIZES:
        leaves = [p[:n].clone().requires_grad_() for p in stored]
        # Fixed upstream gradients, the same for both implementations, shaped as the outputs.
        g = torch.Generator(device="cuda").manual_seed(n)
        like = dict(zip(_SLICE_INPUTS, leaves, strict=True))
        shapes = {"means_t": "means", "opacities_t": "times", "quats_n": "quats"}
        shapes |= {"scales": "log_scales", "reg_terms": "times"}
        grads = [
            torch.randn(like[shapes[name]].shape, generator=g, device="cuda")
            for name in _SLICE_GRADIENTS
        ]
        for impl in _SLICE_IMPLS:
            call = _slice_call(leaves, grads, impl, times)
            torch.cuda.reset_peak_memory_stats()
            first = _first_call(call)
            timings = _repeat(call, _SLICE_REPETITIONS)
            forward = [t for t in timings if t.name == "slice.kernel"]
            backward = [t for t in timings if t.name == "backward"]
            figures: dict[str, Any] = {
                "forward_backward_s": _median(
                    [b.end - f.start for f, b in zip(forward, backward, strict=True)]
                ),
                "forward_s": _median([t.seconds for t in forward]),
                "backward_s": _median([t.seconds for t in backward]),
                "first_call_s": first,
                "repetitions": len(forward),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            if impl == "compiled":
                figures["unique_graphs"] = int(counters["stats"]["unique_graphs"])
            cases.append({"case": {"impl": impl, "n": n}, "figures": figures})
        del leaves
    graphs = int(counters["stats"]["unique_graphs"])
    resizes = len(_SLICE_SIZES) - 1
    recompilations = graphs - 1
    cases.append(
        {
            "case": {"impl": "compiled", "resizes": resizes},
            "figures": {"unique_graphs": graphs, "recompilations": recompilations},
        }
    )
    budgets = [
        {
            "case": {"impl": "compiled", "resizes": resizes},
            "figure": "recompilations",
            "limit": _RECOMPILE_LIMIT,
            "value": recompilations,
            "pass": recompilations <= _RECOMPILE_LIMIT,
            "kind": "limit",
        }
    ]
    setup = {
        "model": "synthesized (Q-168, Q-181)",
        "seed": _SEED,
        "active_fraction": _SLICE_ACTIVE,
        "sizes": list(_SLICE_SIZES),
        "resizes": resizes,
        "implementations": list(_SLICE_IMPLS),
        "warmup": _WARMUP,
        "repetitions": _SLICE_REPETITIONS,
        "gradients": list(_SLICE_GRADIENTS),
        "times": "uniform in [0, 1], seeded",
        "clock": "CUDA events: slice.kernel (forward), backward",
        "recompile_limit": _RECOMPILE_LIMIT,
    }
    return _Result(setup, cases, budgets)


# -- B-1: a training iteration by stage (§3.4, D-49, D-50; Q-168, Q-185) -----------------------

# #226: N = 0.5M, 1M and 3M with 20 % of the rows active, and 3M at 5 % as the conservative case
# for D-50, where rasterization does the least per row.
_TRAIN_STEP_CASES: tuple[tuple[int, float], ...] = (
    (500_000, 0.2),
    (1_000_000, 0.2),
    (3_000_000, 0.2),
    (3_000_000, 0.05),
)
_EVENTS_FROM = 100  # #226: growth and relocation start at iteration 100
_HEADROOM_PERCENT = 2  # #226: n_max = ⌈1.02·N⌉
_TIMED = (121, 420)  # #226: the iterations timed by the stage clock, events included
_UNCLOCKED = (421, 720)  # Q-185: whole iterations timed without the clock
_BOUND_LIMIT_S = 3600.0  # §3.4 B-1: a full training run ≤ 60 min on an RTX 4090 (Q-168's bound)
_SLICE_SHARE_LIMIT = 0.10  # D-50: the CUDA operator only above 10 % of an iteration
_COPY_REPETITIONS = 50  # Q-72 note 5: copies of the relocation scores, after _WARMUP


def _train_step_config(n: int, *, sh_degree: int = 3) -> TrainConfig:
    """B-1's configuration for a model of `n` rows (#226): the default, LPIPS-VGG included,
    except that growth and relocation start at iteration 100, n_max = ⌈1.02·N⌉ and the SH bands
    all open by iteration 4 (sh_interval = 1); `sh_degree` is the model's (Q-185)."""
    import dataclasses

    from openftgs.config import TrainConfig

    base = TrainConfig()
    n_max = -(-(100 + _HEADROOM_PERCENT) * n // 100)  # ⌈1.02·N⌉ in integers, so never one over
    return dataclasses.replace(
        base,
        sh_degree=sh_degree,
        growth=dataclasses.replace(base.growth, start=_EVENTS_FROM, n_max=n_max),
        reloc=dataclasses.replace(base.reloc, start=_EVENTS_FROM),
        optim=dataclasses.replace(base.optim, sh_interval=1),
    )


# The backward stage in three parts, split by its two points (`openftgs._stages.POINTS`): every
# loss's backward, then the rasterization's (projection and SH included), then the temporal
# slice's, until `backward()` returns.
_BACKWARD_PARTS = ("backward.losses", "backward.raster", "backward.slice")


def _iteration_parts(timings: Sequence[Timing]) -> list[dict[str, float]]:
    """One dict per `iteration` record, in order, from the records of a stage clock: each
    stage's seconds in that iteration, and the three parts of its backward stage when both
    points fired. Records belong to the last iteration that began before them (by `index`), so
    the points recorded on the autograd thread fall in their own iteration."""
    from openftgs._stages import POINTS

    groups: list[list[Timing]] = []
    for timing in sorted(timings, key=lambda t: t.index):
        if timing.name == "iteration":
            groups.append([])
        if groups:
            groups[-1].append(timing)
    parts: list[dict[str, float]] = []
    for group in groups:
        seconds: dict[str, float] = {}
        points: dict[str, float] = {}
        backward: Timing | None = None
        for timing in group:
            if timing.name in POINTS:
                points[timing.name] = timing.start
                continue
            seconds[timing.name] = seconds.get(timing.name, 0.0) + timing.seconds
            if timing.name == "backward":
                backward = timing
        if backward is not None and set(POINTS) <= set(points):
            losses, raster = points["backward.losses_done"], points["backward.raster_done"]
            seconds["backward.losses"] = losses - backward.start
            seconds["backward.raster"] = raster - losses
            seconds["backward.slice"] = backward.end - raster
        parts.append(seconds)
    return parts


def _training_bound(iteration_s: float, event_s: float) -> tuple[float, int, int]:
    """Q-168's upper bound on a default training run, with T_it and the number of events of the
    default calendar (§1.3): T_it × `iteration_s` + events × `event_s`, where `event_s` is what
    an event iteration takes beyond an iteration without one."""
    from openftgs.config import TrainConfig
    from openftgs.train import calendar_entry

    cfg = TrainConfig()
    iterations = cfg.optim.iterations
    events = 0
    for it in range(1, iterations + 1):
        entry = calendar_entry(cfg, it)
        events += entry.event or entry.reset_opacity
    return iterations * iteration_s + events * event_s, iterations, events


def _stage_figures(
    parts: Sequence[dict[str, float]], *, events: Sequence[bool]
) -> dict[str, dict[str, float]]:
    """§3.4 B-1 (#226): per stage, over the iterations without an event in which it ran, the
    mean and median seconds, the count, and its share of an iteration, the ratio of its mean to
    the mean iteration (Q-185)."""
    quiet = [p for p, event in zip(parts, events, strict=True) if not event]
    iteration = statistics.fmean(p["iteration"] for p in quiet)
    names = dict.fromkeys(name for p in quiet for name in p)  # first-seen order
    figures: dict[str, dict[str, float]] = {}
    for name in names:
        values = [p[name] for p in quiet if name in p]
        mean = statistics.fmean(values)
        figures[name] = {
            "mean_s": mean,
            "median_s": _median(values),
            "share": mean / iteration,
            "count": len(values),
        }
    return figures


def _is_event(cfg: TrainConfig, it: int) -> bool:
    """Whether iteration `it` of `cfg`'s calendar runs an event (`Trainer.step`'s `_event`)."""
    from openftgs.train import calendar_entry

    entry = calendar_entry(cfg, it)
    return entry.event or entry.reset_opacity


def _fractions(trainer: Trainer, video: MultiViewVideo, logs: Sequence[StepLog]) -> list[float]:
    """The shares of the rows that pass the opacity cull (active, D-32) and that are visible in
    the image (E6), averaged over the images and times of `logs`, on the trainer's model as it is
    now, without gradients (Q-185)."""
    import torch

    from openftgs.render import render

    model = trainer.model
    totals = torch.zeros(2, dtype=torch.float64, device=model.means.device)
    with torch.no_grad():
        for log in logs:
            camera = video.records[log.image_index].camera
            out = render(
                model,
                camera,
                log.t,
                time_units="normalized",
                sh_degree=log.sh_degree,
                backend="gsplat",
                return_aux=True,
            )
            assert out.visible is not None  # return_aux
            totals[0] += model.at_time(log.t).passes_cull.double().mean()
            totals[1] += out.visible.double().mean()
    return [float(v) / len(logs) for v in totals.tolist()]


def _score_copy(n: int) -> float:
    """Q-72 note 5: the median wall time of copying N float32 sampling scores from the GPU to
    the CPU, as relocation does before its CPU generator draws the targets; the copy
    synchronizes, as an event does anyway."""
    import torch

    scores = torch.rand(n, device="cuda")
    times: list[float] = []
    for k in range(_WARMUP + _COPY_REPETITIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores.to("cpu")
        if k >= _WARMUP:
            times.append(time.perf_counter() - start)
    return _median(times)


def _train_step_case(
    ctx: _Context, case: dict[str, Any], model: FreeTimeGaussians
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One B-1 case: a trainer on `model` with B-1's configuration, iterations 1 to 120 as a
    warm-up, 121 to 420 under the stage clock and 421 to 720 without it (Q-185). Returns the
    case's per-stage and per-event entries, and its summary figures."""
    import gc

    import torch

    from openftgs._stages import Clock
    from openftgs.train import Trainer

    cfg = _train_step_config(model.num_gaussians, sh_degree=model.sh_degree)
    given = f"model {ctx.args.model}" if ctx.model is not None else f"scene {ctx.args.scene}"
    # A model of another scene's time can't train on this one; the LPIPS weights may download.
    with _environment_error("LPIPS networks"), _as_input_error(given):
        trainer = Trainer(cfg, ctx.video, model, device="cuda")
    del model
    n_start = trainer.model.num_gaussians
    first, last = _TIMED
    while trainer.iteration < first - 1:
        trainer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    n_timed_start = trainer.model.num_gaussians
    logs: list[StepLog] = []
    with Clock("cuda") as clock:
        while trainer.iteration < last:
            logs.append(trainer.step())
    timings = clock.read()  # synchronizes, after the last timed iteration
    peaks = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    n_end = trainer.model.num_gaussians
    active, visible = _fractions(trainer, ctx.video, logs)

    # Q-185: the stage clock costs host time per stage, which shows where an iteration is
    # host-bound; whole iterations timed by one pair of events each, without it, measure that.
    marks: list[tuple[int, Any, Any]] = []
    while trainer.iteration < _UNCLOCKED[1]:
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        iteration = trainer.step().iteration
        end.record()
        marks.append((iteration, start, end))
    torch.cuda.synchronize()
    unclocked = [(it, start.elapsed_time(end) / 1000.0) for it, start, end in marks]
    copy_s = _score_copy(trainer.model.num_gaussians)
    del trainer, marks
    gc.collect()
    torch.cuda.empty_cache()

    parts = _iteration_parts(timings)
    if len(parts) != len(logs):
        raise RuntimeError(f"the stage clock holds {len(parts)} iterations, not {len(logs)}")
    events = [_is_event(cfg, log.iteration) for log in logs]
    stages = _stage_figures(parts, events=events)
    entries = [
        {"case": {**case, "stage": name}, "figures": figures} for name, figures in stages.items()
    ]
    event_parts = [(log, part) for log, part, ev in zip(logs, parts, events, strict=True) if ev]
    for log, part in event_parts:
        figures = {
            "events_s": part["events"],
            "iteration_s": part["iteration"],
            "n_after": log.num_gaussians,
        }
        entries.append({"case": {**case, "iteration": log.iteration}, "figures": figures})

    iteration_s = stages["iteration"]["mean_s"]
    quiet = [s for it, s in unclocked if not _is_event(cfg, it)]
    loud = [s for it, s in unclocked if _is_event(cfg, it)]
    unclocked_s = statistics.fmean(quiet)
    if loud:  # what an event iteration takes beyond one without an event
        event_extra_s = statistics.fmean(loud) - unclocked_s
    else:
        event_extra_s = statistics.fmean(p["iteration"] for _, p in event_parts) - iteration_s
    bound, _, _ = _training_bound(unclocked_s, event_extra_s)
    slice_s = stages["slice.kernel"]["mean_s"] + stages["backward.slice"]["mean_s"]
    summary = {
        "iterations": stages["iteration"]["count"],
        "slice_forward_s": stages["slice.kernel"]["mean_s"],
        "slice_backward_s": stages["backward.slice"]["mean_s"],
        "slice_share": slice_s / iteration_s,  # D-50
        "iteration_unclocked_mean_s": unclocked_s,
        "iteration_unclocked_median_s": _median(quiet),
        "unclocked_iterations": len(quiet),
        "clock_overhead_s": iteration_s - unclocked_s,
        "events_s": statistics.fmean(p["events"] for _, p in event_parts),
        "event_iteration_s": statistics.fmean(p["iteration"] for _, p in event_parts),
        "event_extra_s": event_extra_s,
        "training_bound_s": bound,
        "peak_allocated_bytes": peaks[0],
        "peak_reserved_bytes": peaks[1],
        "active_fraction": active,
        "visible_fraction": visible,
        "n_start": n_start,
        "n_timed_start": n_timed_start,
        "n_end": n_end,
        "score_copy_s": copy_s,
    }
    return entries, summary


def _train_step(ctx: _Context) -> _Result:
    """B-1: a training iteration by stage, from the stage clock's CUDA events, on synthesized
    models of 0.5M, 1M and 3M rows with 20 % active and 3M with 5 % (Q-168, Q-181), or on the
    model given with `--model` (D-88); with peak memory, the events, and a bound on a default
    training run (Q-168); and D-50's share of the temporal slice (#226, Q-185)."""
    import functools

    from openftgs._stages import STAGES
    from openftgs.data import FrameCache

    with _as_input_error(f"frame cache of {ctx.args.scene}", OSError):
        FrameCache.build(ctx.video)  # #226: built before timing, so "auto" only opens it
    makers: list[tuple[dict[str, Any], Callable[[], FreeTimeGaussians]]] = []
    if ctx.model is not None:
        given = ctx.model.model
        listed: list[list[Any]] = [[given.num_gaussians, None]]
        makers.append(({"model": "given", "n": given.num_gaussians}, lambda: given))
    else:
        with _as_input_error(f"scene manifest {ctx.args.scene}"):
            volume = _volume(ctx.video)
        listed = [[n, active] for n, active in _TRAIN_STEP_CASES]
        for n, active in _TRAIN_STEP_CASES:
            synthesize = functools.partial(
                _synthesize, volume, n, active=active, seed=_SEED, device="cuda"
            )
            makers.append(({"n": n, "active": active}, synthesize))

    cases: list[dict[str, Any]] = []
    summaries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for case, make in makers:
        entries, summary = _train_step_case(ctx, case, make())
        cases += entries
        cases.append({"case": case, "figures": summary})
        summaries.append((case, summary))

    # The bound (Q-168) from the largest N, and among equal N the slowest; D-50's share from the
    # case where the slice takes the largest share (#226).
    case, summary = max(summaries, key=lambda s: (s[0]["n"], s[1]["training_bound_s"]))
    budgets = [_budget(case, summary, "training_bound_s", _BOUND_LIMIT_S, "bound")]
    case, summary = max(summaries, key=lambda s: s[1]["slice_share"])
    budgets.append(_budget(case, summary, "slice_share", _SLICE_SHARE_LIMIT, "limit"))

    any_cfg = _train_step_config(1)  # the events don't depend on N
    _, iterations, events = _training_bound(0.0, 0.0)
    first, last = _TIMED
    setup = {
        "model": "given (--model)" if ctx.model is not None else "synthesized (Q-168, Q-181)",
        "seed": None if ctx.model is not None else _SEED,
        "cases": listed,
        "configuration": (
            "the default, LPIPS-VGG included, except growth.start = reloc.start = "
            f"{_EVENTS_FROM}, growth.n_max = ⌈1.02·N⌉, optim.sh_interval = 1, and the "
            "model's SH degree"
        ),
        "frame_cache": 'built before timing; placement "auto"',
        "warmup": first - 1,
        "timed": [first, last],
        "unclocked": list(_UNCLOCKED),
        "events": [it for it in range(first, last + 1) if _is_event(any_cfg, it)],
        "clock": "CUDA events (openftgs._stages), read after the last timed iteration",
        "stages": list(STAGES[: STAGES.index("log") + 1]),
        "backward_parts": list(_BACKWARD_PARTS),
        "figures": "per stage over the iterations without an event; shares are ratios of means",
        "training_iterations": iterations,
        "default_events": events,
        "bound_limit_s": _BOUND_LIMIT_S,
        "slice_share_limit": _SLICE_SHARE_LIMIT,
        "copy_repetitions": _COPY_REPETITIONS,
    }
    return _Result(setup, cases, budgets)


# -- B-2: the rasterization variants (§3.4, D-32, D-57; Q-31, Q-36, Q-187) ----------------------

_RASTER_TIMES = 10  # #227: one training camera at 10 fixed times


def _raster_frames(video: MultiViewVideo) -> list[int]:
    """The records B-2 renders (Q-187): the first training camera in the manifest's order, at
    `_RASTER_TIMES` of its frames evenly spaced in time, round(k·(F − 1)/9) of its F frames for
    k = 0…9, repeating frames when F < 10."""
    first = next(r.camera.name for r in video.records if r.split == "train")
    frames = sorted(
        (r.timestamp, i)
        for i, r in enumerate(video.records)
        if r.split == "train" and r.camera.name == first
    )
    last = len(frames) - 1
    return [frames[round(k * last / (_RASTER_TIMES - 1))][1] for k in range(_RASTER_TIMES)]


def _raster_figures(timings: Sequence[Timing]) -> dict[str, Any]:
    """B-2's times from the stage clock's records of its calls (Q-187), each the median over
    the calls: the whole call, from the temporal slice's start to the backward's end (the figure
    the decision rule reads); the rasterization, from `raster.forward`'s start to the point where
    its backward is done; that forward and that backward (from the losses' backward done to the
    rasterization's); and the backward."""

    def named(name: str) -> list[Timing]:
        return [t for t in timings if t.name == name]

    calls = list(
        zip(
            named("slice"),
            named("raster.forward"),
            named("backward"),
            named("backward.losses_done"),
            named("backward.raster_done"),
            strict=True,
        )
    )
    return {
        "forward_backward_s": _median([b.end - s.start for s, _, b, _, _ in calls]),
        "raster_s": _median([done.start - r.start for _, r, _, _, done in calls]),
        "raster_forward_s": _median([r.seconds for _, r, _, _, _ in calls]),
        "raster_backward_s": _median([done.start - lo.start for _, _, _, lo, done in calls]),
        "backward_s": _median([b.seconds for _, _, b, _, _ in calls]),
        "repetitions": len(calls),
    }


def _intersections(model: FreeTimeGaussians, camera: Camera, timestamp: float) -> list[int]:
    """At raw time `timestamp`: the rows that pass the opacity cull (D-32), the rows visible in
    `camera` and gsplat's tile intersections (Q-187), from gsplat's own projection and tile
    intersection with the arguments `render` gives them (E6, D-6, D-57). Every variant sorts and
    rasterizes the same intersections (U-8), so they are counted once."""
    import math

    import gsplat
    import torch

    with torch.no_grad():
        g = model.at_time(model.time.to_t(timestamp))
        like = g.means_t
        radii, means2d, depths, _, _ = gsplat.fully_fused_projection(
            g.means_t,
            None,
            g.quats_n,
            g.scales,
            camera.world_to_camera.to(like)[None],
            camera.K.to(like)[None],
            camera.width,
            camera.height,
            eps2d=0.3,  # E6, D-6
            near_plane=0.01,  # E6, D-6
            far_plane=1e10,  # E6, D-6
            radius_clip=0.0,  # D-57
            packed=False,
            opacities=g.opacities_t,  # E6, D-32: the opacity cull and the extent
        )
        visible_rows = (radii > 0).all(dim=-1)
        # As `render` does: gsplat leaves the 2D means of culled rows unwritten (D-75).
        means2d = torch.where(visible_rows[..., None], means2d, torch.zeros_like(means2d))
        tile = 16  # D-57
        _, isect_ids, _ = gsplat.isect_tiles(
            means2d,
            radii,
            depths,
            tile,
            math.ceil(camera.width / tile),
            math.ceil(camera.height / tile),
            packed=False,
            n_images=1,
        )
        active = int(g.passes_cull.sum())
        visible = int(visible_rows.sum())
    return [active, visible, int(isect_ids.shape[0])]


_RASTER_SIZES = (1_000_000, 3_000_000)  # #227
_RASTER_ACTIVE = (0.05, 0.2, 1.0)  # #227
_RASTER_REPETITIONS = 50  # #227: the median of 50
_RasterVariant = tuple[bool, Literal["internal", "compact"]]
# packed × precull (Q-36, Q-31), D-57's default first: the spot check's reference.
_RASTER_VARIANTS: tuple[_RasterVariant, ...] = (
    (True, "internal"),
    (True, "compact"),
    (False, "internal"),
    (False, "compact"),
)
_SPOT_LIMIT = 1e-6  # U-8 on the GPU: images within 1e-6


def _spot_check(
    model: FreeTimeGaussians, camera: Camera, timestamps: list[float]
) -> dict[_RasterVariant, dict[str, Any]]:
    """B-2's spot check (#227, Q-187), under `no_grad` at every time: each variant's image
    against packed/internal's, its largest absolute difference and the pixels where some channel
    differs by more than 1e-6, summed over the times; and each compacted image against the same
    layout's internal one, U-8 itself."""
    import torch

    from openftgs.render import render

    default = _RASTER_VARIANTS[0]
    checks: dict[_RasterVariant, dict[str, Any]] = {
        variant: {"image_max_abs_diff": 0.0, "image_pixels_over_limit": 0}
        for variant in _RASTER_VARIANTS[1:]
    }
    with torch.no_grad():
        for timestamp in timestamps:
            images = {
                variant: render(
                    model, camera, timestamp, packed=variant[0], precull=variant[1]
                ).image
                for variant in _RASTER_VARIANTS
            }
            for variant, check in checks.items():
                diff = (images[variant] - images[default]).abs().amax(dim=-1)
                check["image_max_abs_diff"] = max(check["image_max_abs_diff"], float(diff.max()))
                check["image_pixels_over_limit"] += int((diff > _SPOT_LIMIT).sum())
                packed, precull = variant
                if precull == "compact":
                    same = float((images[variant] - images[(packed, "internal")]).abs().max())
                    check["precull_max_abs_diff"] = max(
                        check.get("precull_max_abs_diff", 0.0), same
                    )
    return checks


def _raster_call(
    model: FreeTimeGaussians,
    camera: Camera,
    frames: list[tuple[float, Tensor]],
    variant: _RasterVariant,
) -> Callable[[], None]:
    """One B-2 call (Q-187): `render` at the next of `frames` (raw τ and its training image
    [1, H, W, 3]) from `camera`, moved to the model's device first, the L1 loss against that
    image, then its backward into the model's nine stored tensors, as the stage `backward`;
    `torch.autograd.grad` accumulates nothing between calls."""
    import torch

    from openftgs._stages import point, stage
    from openftgs.losses import l1_loss
    from openftgs.render import render

    packed, precull = variant
    params = [getattr(model, name) for name in model.PARAM_NAMES]
    order = itertools.cycle(frames)
    # On the model's device once, as `Trainer` keeps its cameras: `render` copies a camera held on
    # the host in every call, which synchronizes (§2.4 rule 3).
    device = model.means.device
    camera = replace(
        camera, K=camera.K.to(device), world_to_camera=camera.world_to_camera.to(device)
    )

    def call() -> None:
        timestamp, target = next(order)
        out = render(model, camera, timestamp, packed=packed, precull=precull)
        point(out.image, "backward.losses_done")  # as `Trainer.step` marks it
        with stage("loss.l1"):
            loss = l1_loss(out.image, target)
        with stage("backward"):
            torch.autograd.grad(loss, params)

    return call


def _raster(ctx: _Context) -> _Result:
    """B-2: `render` forward plus an L1 backward for every `packed` × `precull` combination
    (D-57), on synthesized models at N = 1M and 3M with 5 %, 20 % and 100 % of rows active, from
    one training camera at 10 fixed times (Q-187). Per model, the rows active, the rows
    visible and the tile intersections at each time, and the spot check (`_spot_check`), each
    image held to 1e-6 of packed/internal's (#227) and each compacted one to 1e-6 of its layout's
    internal one (U-8); per variant, the first call and the medians of 50 calls, and the peak
    allocated memory."""
    import torch

    from openftgs.config import PerfConfig

    with _as_input_error(f"scene manifest {ctx.args.scene}"):
        volume = _volume(ctx.video)
    indices = _raster_frames(ctx.video)
    records = [ctx.video.records[i] for i in indices]
    camera = records[0].camera
    frames = [
        (r.timestamp, ctx.video.load_image(i).to("cuda")[None])
        for i, r in zip(indices, records, strict=True)
    ]
    cases: list[dict[str, Any]] = []
    budgets: list[dict[str, Any]] = []
    for n in _RASTER_SIZES:
        for active in _RASTER_ACTIVE:
            model = _synthesize(volume, n, active=active, seed=_SEED, device="cuda")
            # Q-187: the eager slice, the same in every variant, compiles nothing into the first
            # call and can't fall back unannounced ("auto", Q-13).
            model.slice_impl = "eager"
            counts = [_intersections(model, camera, timestamp) for timestamp, _ in frames]
            resident = torch.cuda.memory_allocated()
            cases.append(
                {
                    "case": {"n": n, "active": active},
                    "figures": {
                        "active_rows": [c[0] for c in counts],
                        "visible_rows": [c[1] for c in counts],
                        "tile_intersections": [c[2] for c in counts],
                        "resident_memory_bytes": resident,
                    },
                }
            )
            spot = _spot_check(model, camera, [timestamp for timestamp, _ in frames])
            for variant in _RASTER_VARIANTS:
                packed, precull = variant
                case = {"n": n, "active": active, "packed": packed, "precull": precull}
                call = _raster_call(model, camera, frames, variant)
                torch.cuda.reset_peak_memory_stats()
                first = _first_call(call)
                figures = _raster_figures(_repeat(call, _RASTER_REPETITIONS))
                figures["first_call_s"] = first
                figures["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                checked = spot.get(variant, {})
                figures |= checked
                for name in ("image_max_abs_diff", "precull_max_abs_diff"):
                    if name in checked:
                        budgets.append(_budget(case, figures, name, _SPOT_LIMIT, "limit"))
                cases.append({"case": case, "figures": figures})
            del model, call  # the last call holds the model; the next resident figure must not
    setup = {
        "model": "synthesized (Q-168, Q-181)",
        "seed": _SEED,
        "sizes": list(_RASTER_SIZES),
        "active_fractions": list(_RASTER_ACTIVE),
        "variants": [{"packed": p, "precull": c} for p, c in _RASTER_VARIANTS],
        "default": {"packed": PerfConfig.packed, "precull": PerfConfig.precull},  # D-57
        "camera": camera.name,
        "width": camera.width,
        "height": camera.height,
        "timestamps": [timestamp for timestamp, _ in frames],
        "slice_impl": "eager",
        "loss": "L1 against the camera's training image at that time (losses.l1_loss)",
        "backward": "torch.autograd.grad into the nine stored tensors",
        "warmup": _WARMUP,
        "repetitions": _RASTER_REPETITIONS,
        "clock": (
            "CUDA events: slice to backward (forward_backward_s), raster.forward to "
            "backward.raster_done (raster_s), backward.losses_done to backward.raster_done "
            "(raster_backward_s)"
        ),
        "spot_check_limit": _SPOT_LIMIT,
    }
    return _Result(setup, cases, budgets)


# -- B-5: the Adam kinds (§3.4, D-53; Q-168, Q-188) --------------------------------------------


def _visibility(n: int, fraction: float, *, seed: int, device: str) -> Tensor:
    """Q-188: selective Adam's mask, a bool [n] with exactly round(`fraction`·n) rows true,
    chosen by a seeded permutation, so the visible rows are spread over the tensors as a render
    leaves them."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[torch.randperm(n, generator=g, device=device)[: round(fraction * n)]] = True
    return mask


def _seed_gradients(model: FreeTimeGaussians, *, seed: int) -> None:
    """#229, Q-188: a seeded N(0, 1) gradient on each of the model's nine stored tensors."""
    import torch

    g = torch.Generator(device=model.means.device).manual_seed(seed)
    for name in model.PARAM_NAMES:
        param = getattr(model, name)
        param.grad = torch.randn(param.shape, generator=g, dtype=param.dtype, device=param.device)


_ADAM_LIMIT = 1.0  # §3.4 B-5: fused no slower than foreach, with no tolerance (Q-188)


def _adam_budgets(
    medians: Mapping[tuple[str, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§3.4 B-5's budget (Q-188): from the median of each `(kind, n)`, per N in the order
    given a case `{"n"}` with fused's median over foreach's, and its budget entry against 1.0."""
    cases: list[dict[str, Any]] = []
    budgets: list[dict[str, Any]] = []
    for n in dict.fromkeys(n for _, n in medians):
        case = {"n": n}
        figures = {"fused_over_foreach": medians["fused", n] / medians["foreach", n]}
        cases.append({"case": case, "figures": figures})
        budgets.append(_budget(case, figures, "fused_over_foreach", _ADAM_LIMIT, "budget"))
    return cases, budgets


_AdamKind = Literal["fused", "foreach", "selective"]
_ADAM_KINDS: tuple[_AdamKind, ...] = ("fused", "foreach", "selective")  # D-53, PerfConfig.adam
# #229: N = 1M and 3M, each with selective Adam's mask at the visible fraction B-1 measured at
# that N with 20 % of the rows active, to three decimals (Calibration data, Q-185; Q-188).
_ADAM_CASES: tuple[tuple[int, float], ...] = ((1_000_000, 0.173), (3_000_000, 0.170))
_ADAM_ACTIVE = 0.2  # Q-188: B-1's models at these N
_ADAM_REPETITIONS = 50  # #229: the median of 50


def _adam_call(optimizer: FTGSOptimizer, visibility: Tensor | None) -> Callable[[], None]:
    """One B-5 call (#229): the next iteration's rates, then one Adam step, as the stage
    `adam`; the gradients stay, since no `zero_grad` clears them (Q-188)."""
    import itertools

    from openftgs._stages import stage

    iterations = itertools.count(1)

    def call() -> None:
        it = next(iterations)
        with stage("adam"):
            optimizer.set_iteration(it)
            if visibility is None:
                optimizer.step()
            else:
                optimizer.step(visibility=visibility)

    return call


def _adam(ctx: _Context) -> _Result:
    """B-5: `FTGSOptimizer.set_iteration` plus `step` for fused, foreach and selective Adam
    (D-53) at N = 1M and 3M, on B-1's synthesized models with seeded gradients on all nine
    stored tensors: the time of the first call, then the median of 50 calls after a warm-up;
    selective Adam's mask at the visible fraction B-1 measured; and per N the budget, fused no
    slower than foreach (#229, Q-188)."""
    import gc

    import torch

    from openftgs.config import OptimConfig
    from openftgs.optim import build_optimizer

    with _as_input_error(f"scene manifest {ctx.args.scene}"):
        volume = _volume(ctx.video)
    cases: list[dict[str, Any]] = []
    medians: dict[tuple[str, int], float] = {}
    for n, fraction in _ADAM_CASES:
        model = _synthesize(volume, n, active=_ADAM_ACTIVE, seed=_SEED, device="cuda")
        _seed_gradients(model, seed=_SEED)
        mask = _visibility(n, fraction, seed=_SEED, device="cuda")
        for kind in _ADAM_KINDS:
            # The same model for every kind: Adam's cost doesn't depend on the values.
            optimizer = build_optimizer(
                model,
                OptimConfig(),
                spatial_scale=volume.spatial_scale,
                frame_dt=ctx.video.frame_dt,
                adam=kind,
            )
            call = _adam_call(optimizer, mask if kind == "selective" else None)
            torch.cuda.reset_peak_memory_stats()
            first = _first_call(call)
            timings = [t for t in _repeat(call, _ADAM_REPETITIONS) if t.name == "adam"]
            figures: dict[str, Any] = {
                "step_s": _median([t.seconds for t in timings]),
                "first_call_s": first,
                "repetitions": len(timings),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            if kind == "selective":
                figures["visible_fraction"] = round(fraction * n) / n
            medians[kind, n] = figures["step_s"]
            cases.append({"case": {"adam": kind, "n": n}, "figures": figures})
            del optimizer, call
        del model, mask
        gc.collect()
        torch.cuda.empty_cache()
    compared, budgets = _adam_budgets(medians)
    setup = {
        "model": "synthesized (Q-168, Q-181)",
        "seed": _SEED,
        "active_fraction": _ADAM_ACTIVE,
        "cases": [[n, fraction] for n, fraction in _ADAM_CASES],
        "kinds": list(_ADAM_KINDS),
        "configuration": "OptimConfig() with the scene's S (D-27) and Δt_f",
        "gradients": "N(0, 1), seeded, on all nine stored tensors, set once and kept",
        "call": "set_iteration(it) then step(), it = 1, 2, …, without zero_grad",
        "visibility": (
            "selective Adam's mask: round(f·N) rows chosen by a seeded permutation, f being "
            "B-1's visible fraction at N with 20 % active (Calibration data, Q-185)"
        ),
        "warmup": _WARMUP,
        "repetitions": _ADAM_REPETITIONS,
        "clock": "CUDA events: adam",
        "budget": f"fused_over_foreach ≤ {_ADAM_LIMIT} at each N",
    }
    return _Result(setup, cases + compared, budgets)


# -- B-6: the data wait per cache placement (§3.4, D-51; Q-99, Q-178, Q-189) -------------------

_DATA_N = 1_000_000  # #232: B-1's configuration at N = 1M
_DATA_ACTIVE = 0.2  # Q-189: as B-1's case at 1M rows (#226)
_MEMORY_N = 3_000_000  # #232, Q-99's note: peak memory at N = 3M with the cache on the GPU
_DATA_TIMED = (121, 320)  # #232: 200 iterations under the CUDA clock, after B-1's warm-up
_DATA_HOST = (321, 420)  # Q-189: 100 iterations under the wall clock, a diagnostic
_DATA_SHARE_LIMIT = 0.01  # §3.4 B-6, D-51: the data wait under 1 % of an iteration
# #232, Q-178: each placement, with the prefetch depth where it has one ("gpu" takes its frames in
# place and "off" decodes PNGs, so neither reads ahead).
_DATA_PLACEMENTS: tuple[tuple[Literal["gpu", "mmap", "off"], int | None], ...] = (
    ("gpu", None),
    ("mmap", 8),
    ("mmap", 0),
    ("off", None),
)


def _data_config(
    n: int, placement: Literal["gpu", "mmap", "off"], prefetch: int | None
) -> TrainConfig:
    """B-6's configuration (#232): B-1's at `n` rows, with the case's frame-cache placement and
    prefetch depth; None keeps the default depth, which "gpu" and "off" don't use."""
    import dataclasses

    cfg = _train_step_config(n)
    perf = dataclasses.replace(cfg.perf, frame_cache=placement)
    if prefetch is not None:
        perf = dataclasses.replace(perf, prefetch=prefetch)
    return dataclasses.replace(cfg, perf=perf)


@dataclass(frozen=True)
class _AutoEstimate:
    """What "auto" measures after a warm-up iteration on CUDA (Q-99), in bytes: the free GPU
    memory, `mem_get_info`'s plus what the allocator holds unused; what growth to N_max adds;
    the cache's size; the memory allocated and reserved then; and the device's total."""

    free: int
    growth: int
    cache: int
    allocated: int
    reserved: int
    total: int

    @property
    def limit(self) -> float:
        """D-51, Q-99: the largest cache "auto" puts on the GPU, ½·free − growth."""
        return 0.5 * self.free - self.growth

    @property
    def fits(self) -> bool:
        return self.cache <= self.limit

    def figures(self) -> dict[str, float]:
        return {
            "auto_free_bytes": self.free,
            "auto_growth_bytes": self.growth,
            "auto_allocated_bytes": self.allocated,
            "auto_reserved_bytes": self.reserved,
            "auto_total_bytes": self.total,
            # Q-189: the device's memory outside this process's allocator, other processes'
            # and the CUDA context's, which the free memory leaves out.
            "auto_outside_bytes": self.total - self.free - self.allocated,
            "cache_bytes": self.cache,
            "auto_limit_bytes": self.limit,
        }

    def margin(self, *, peak_reserved: int) -> dict[str, float]:
        """Q-99's note (Q-189): what a run with the cache on the GPU took after the warm-up
        iteration beyond the cache and the growth, its peak reserved memory less what was
        allocated then, and that as a share of the ½·free the rule leaves for it; above 1, a cache
        at the rule's limit wouldn't have fit."""
        beyond = peak_reserved - self.allocated - self.cache - self.growth
        return {"beyond_estimate_bytes": beyond, "margin_used": beyond / (0.5 * self.free)}


def _data_figures(
    timed: Sequence[dict[str, float]],
    timed_events: Sequence[bool],
    host: Sequence[dict[str, float]],
    host_events: Sequence[bool],
) -> dict[str, Any]:
    """§3.4 B-6 (Q-189): from the per-iteration stage times of the CUDA window (`timed`) and
    of the wall clock's (`host`), over the iterations without an event: the data stage's share
    of an iteration, a ratio of means as B-1's (Q-185), with its mean and median, the mean
    iteration and the count; then the host times of the data stage and of an iteration."""
    stages = _stage_figures(timed, events=timed_events)
    on_host = _stage_figures(host, events=host_events)
    return {
        "data_share": stages["data"]["share"],
        "data_mean_s": stages["data"]["mean_s"],
        "data_median_s": stages["data"]["median_s"],
        "iteration_mean_s": stages["iteration"]["mean_s"],
        "iterations": stages["iteration"]["count"],
        "data_host_mean_s": on_host["data"]["mean_s"],
        "iteration_host_mean_s": on_host["iteration"]["mean_s"],
        "host_iterations": on_host["iteration"]["count"],
    }


def _data_budgets(cases: Sequence[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    """§3.4 B-6, D-51 (#232): a budget entry for the data wait's share of every case that reads
    the cache ("gpu", "mmap" at either depth) at B-6's `n` rows, strictly under 1 % (§3.4:
    "< 1%"), as scripts/bench_compare.py judges it (Q-196)."""
    return [
        _budget(c["case"], c["figures"], "data_share", _DATA_SHARE_LIMIT, "budget", strict=True)
        for c in cases
        if c["case"]["placement"] in ("gpu", "mmap") and c["case"]["n"] == n
    ]


def _release_cuda() -> None:
    """Frees what the last trainer left in the allocator, so the next case starts alike."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _data_trainer(ctx: _Context, volume: _Volume, cfg: TrainConfig, n: int) -> Trainer:
    """A trainer of `cfg` on a synthesized model of `n` rows (Q-168, Q-181), B-1's case at 1M."""
    from openftgs.train import Trainer

    model = _synthesize(volume, n, active=_DATA_ACTIVE, seed=_SEED, device="cuda")
    with _environment_error("LPIPS networks"):  # the LPIPS weights may download
        return Trainer(cfg, ctx.video, model, device="cuda")


def _probe(ctx: _Context, volume: _Volume, n: int) -> _AutoEstimate:
    """Q-99, Q-189: what "auto" measures on CUDA, taken as it takes it: a trainer of B-6's
    configuration at `n` rows runs its first iteration on decoded PNGs, as "auto" does, and the
    free memory and growth are read when that iteration returns."""
    import torch

    from openftgs.data import FrameCache

    cfg = _data_config(n, "off", None)
    trainer = _data_trainer(ctx, volume, cfg, n)
    trainer.step()
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    model = trainer.model
    # Q-99: the bytes of a row in training, its stored floats (§1.1) for the parameter, its
    # gradient and Adam's two moments, plus the gradient statistics' two floats, all float32.
    floats = sum(getattr(model, name)[0].numel() for name in model.PARAM_NAMES)
    growth = max(0, cfg.growth.n_max - model.num_gaussians) * 4 * (4 * floats + 2)
    estimate = _AutoEstimate(
        free=free + reserved - allocated,
        growth=growth,
        cache=FrameCache.size_of(ctx.video),
        allocated=allocated,
        reserved=reserved,
        total=total,
    )
    del trainer, model
    _release_cuda()
    return estimate


def _data_case(ctx: _Context, volume: _Volume, cfg: TrainConfig, n: int) -> dict[str, Any]:
    """One B-6 case (#232, Q-189): a fresh trainer of `cfg` on `n` rows, iterations 1 to 120
    as B-1's warm-up, 121 to 320 under the stage clock's CUDA events, then 321 to 420 under the
    wall clock; with peak memory from the first iteration on."""
    import torch

    from openftgs._stages import Clock

    trainer = _data_trainer(ctx, volume, cfg, n)
    torch.cuda.reset_peak_memory_stats()  # after the cache's placement, which it includes
    windows: list[tuple[list[dict[str, float]], list[bool]]] = []
    for clock, (first, last) in ((Clock("cuda"), _DATA_TIMED), (Clock("wall"), _DATA_HOST)):
        while trainer.iteration < first - 1:
            trainer.step()
        with clock:
            iterations = [trainer.step().iteration for _ in range(first, last + 1)]
        parts = _iteration_parts(clock.read())  # the CUDA clock synchronizes, after `last`
        if len(parts) != len(iterations):
            raise RuntimeError(
                f"the stage clock holds {len(parts)} iterations, not {len(iterations)}"
            )
        windows.append((parts, [_is_event(cfg, it) for it in iterations]))
    torch.cuda.synchronize()
    peaks = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    del trainer
    _release_cuda()
    (timed, timed_events), (host, host_events) = windows
    figures = _data_figures(timed, timed_events, host, host_events)
    figures |= {"peak_allocated_bytes": peaks[0], "peak_reserved_bytes": peaks[1]}
    return figures


def _data(ctx: _Context) -> _Result:
    """B-6: the data wait's share of an iteration per frame-cache placement (§3.4, D-51), with
    B-1's configuration at 1M rows (#232): the GPU cache, the memory-mapped cache at prefetch
    depths 8 and 0, and PNG decoding; then the GPU cache at 3M rows with peak memory against
    what "auto" estimated (Q-99's note). The GPU cache is measured only where "auto" would place
    it on the GPU (Q-178, Q-189), and skipped with that estimate otherwise."""
    from openftgs.data import FrameCache

    with _as_input_error(f"frame cache of {ctx.args.scene}", OSError):
        FrameCache.build(ctx.video)  # as B-1: built before timing, so the cases only open it
    with _as_input_error(f"scene manifest {ctx.args.scene}"):
        volume = _volume(ctx.video)
    runs = [(_DATA_N, placement, prefetch) for placement, prefetch in _DATA_PLACEMENTS]
    runs.append((_MEMORY_N, "gpu", None))
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for n, placement, prefetch in runs:
        case: dict[str, Any] = {"n": n, "placement": placement}
        if prefetch is not None:
            case["prefetch"] = prefetch
        estimate = None
        if placement == "gpu":  # Q-178, Q-189
            estimate = _probe(ctx, volume, n)
            if not estimate.fits:
                reason = (
                    f'"auto" would memory-map the cache (Q-99): its {estimate.cache / 1e9:.2f} GB '
                    f"exceed half the {estimate.free / 1e9:.2f} GB free after a warm-up "
                    f"iteration less growth's {estimate.growth / 1e9:.3f} GB"
                )
                skipped.append({"case": case, "reason": reason, "estimate": estimate.figures()})
                continue
        figures = _data_case(ctx, volume, _data_config(n, placement, prefetch), n)
        if estimate is not None:
            figures |= estimate.figures()
            figures |= estimate.margin(peak_reserved=figures["peak_reserved_bytes"])
        cases.append({"case": case, "figures": figures})

    any_cfg = _data_config(1, "off", None)  # the events don't depend on N
    first, last = _DATA_TIMED[0], _DATA_HOST[1]
    setup = {
        "model": "synthesized (Q-168, Q-181)",
        "seed": _SEED,
        "active_fraction": _DATA_ACTIVE,
        "cases": [[n, placement, prefetch] for n, placement, prefetch in runs],
        "configuration": (
            "B-1's (#226): the default, LPIPS-VGG included, except growth.start = reloc.start = "
            f"{_EVENTS_FROM}, growth.n_max = ⌈1.02·N⌉ and optim.sh_interval = 1; with the "
            "case's perf.frame_cache and perf.prefetch"
        ),
        "frame_cache": "built before timing",
        "gpu": (
            'measured where "auto" would place the cache on the GPU (Q-99): a trainer decoding '
            "PNGs runs one iteration first, and the free memory and growth are read after it"
        ),
        "warmup": _DATA_TIMED[0] - 1,
        "timed": list(_DATA_TIMED),
        "host": list(_DATA_HOST),
        "events": [it for it in range(first, last + 1) if _is_event(any_cfg, it)],
        "clock": "CUDA events (openftgs._stages) over timed; the wall clock over host",
        "figure": (
            "data_share: the data stage's mean over the iteration's, over the timed iterations "
            "without an event"
        ),
        "share_limit": _DATA_SHARE_LIMIT,
        "memory_n": _MEMORY_N,
        "skipped": skipped,
    }
    return _Result(setup, cases, _data_budgets(cases, n=_DATA_N))


def _budget(
    case: dict[str, Any],
    figures: dict[str, Any],
    figure: str,
    limit: float,
    kind: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """A budget entry of BENCH.json (Q-166, Q-180) for one figure of one case: it passes at
    most at `limit`, or, `strict`, only under it (B-6's "< 1%")."""
    value = figures[figure]
    return {
        "case": case,
        "figure": figure,
        "limit": limit,
        "value": value,
        "pass": value < limit if strict else value <= limit,
        "kind": kind,
    }


# -- B-4: the losses' precisions and layouts (§3.4, D-52; Q-42, Q-172, Q-173, Q-190) -----------

_LOSS_SIZES: tuple[tuple[int, int], ...] = ((1352, 1014), (1920, 1080))  # §3.4 B-4: width, height
_LOSS_PRECISIONS = ("ieee", "tf32", "bf16")  # D-52
# Q-190: contiguous NCHW inputs; channels-last inputs, as `LPIPSLoss` passes a render (its
# batch-first view of [..., H, W, 3] is channels-last), so training's path until #228; and both
# the inputs and the network's weights channels-last.
_LOSS_LAYOUTS = ("nchw", "channels_last_inputs", "channels_last_inputs_and_weights")
_LOSS_AUTOTUNING = (False, True)  # cuDNN autotuning, one child process each (_CHILD)
_LOSS_REPETITIONS = 30  # Q-190
_LOSS_NOISE = 0.05  # Q-190: the prediction's seeded Gaussian noise
_GAIN_THRESHOLD = 0.03  # Q-173: a gain is at least 3 % at both sizes
# Q-190: the comparisons the turn-on rule reads (Q-173), and the others reported beside
# them, each as (baseline, variant) LPIPS cases `(precision, layout, autotuning)`.
_LPIPS_COMPARISONS: dict[str, tuple[tuple[str, str, bool], tuple[str, str, bool]]] = {
    "autotuning": (("tf32", "channels_last_inputs", False), ("tf32", "channels_last_inputs", True)),
    "channels_last_weights": (
        ("tf32", "channels_last_inputs", False),
        ("tf32", "channels_last_inputs_and_weights", False),
    ),
    "autotuning_and_channels_last_weights": (
        ("tf32", "channels_last_inputs", False),
        ("tf32", "channels_last_inputs_and_weights", True),
    ),
    "tf32": (("ieee", "channels_last_inputs", False), ("tf32", "channels_last_inputs", False)),
    "bf16": (("tf32", "channels_last_inputs", False), ("bf16", "channels_last_inputs", False)),
}
_TURN_ON = ("autotuning", "channels_last_weights")  # the exact options Q-173's rule decides
# The child process of one autotuning setting (Q-190). cuDNN caches the algorithm it picked
# for a configuration and looks there first whatever `cudnn.benchmark` says, so autotuning off
# and on can't be compared in one process. `fused_ssim` can't be imported there, so
# `openftgs.losses.ssim` takes its PyTorch path without a change to its signature (PRD #220,
# owner's decision): a None in sys.modules makes `import fused_ssim` raise ImportError.
_CHILD = (
    "import sys; sys.modules['fused_ssim'] = None; "
    "from openftgs.cli._bench import _losses_child; _losses_child(sys.argv[1])"
)
# Q-166 (as answered): the child's exit code when it runs out of GPU memory, its error last on
# stderr; Python's own for an uncaught exception is 1.
_CHILD_OUT_OF_MEMORY = 2


def _loss_images(
    video: MultiViewVideo, width: int, height: int, *, seed: int, device: str
) -> tuple[Tensor, Tensor]:
    """B-4's target and prediction, [height, width, 3] float32 on `device` (Q-190): the
    scene's first training image, resized bilinearly with antialiasing when its size differs,
    and that image plus Gaussian noise of σ 0.05 from a CPU generator seeded with `seed`."""
    import torch
    import torch.nn.functional as F

    first = next(i for i, record in enumerate(video.records) if record.split == "train")
    image = video.load_image(first)
    if tuple(image.shape[:2]) != (height, width):
        resized = F.interpolate(
            image.permute(2, 0, 1)[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = resized[0].permute(1, 2, 0).contiguous()
    g = torch.Generator().manual_seed(seed)
    noise = _LOSS_NOISE * torch.randn(image.shape, generator=g, dtype=torch.float32)
    return image.to(device), (image + noise).to(device)


def _deviation(
    value: Tensor, grad: Tensor, value_ieee: Tensor, grad_ieee: Tensor
) -> dict[str, Any]:
    """Q-190: a precision's deviation from IEEE float32, in float64: the values and their
    relative difference; the gradients' relative L2 difference, and their largest difference
    over the largest IEEE gradient element."""
    v, v_ieee = float(value), float(value_ieee)
    g, g_ieee = grad.double(), grad_ieee.double()
    return {
        "value": v,
        "value_ieee": v_ieee,
        "value_relative": abs(v - v_ieee) / abs(v_ieee),
        "gradient_relative_l2": float((g - g_ieee).norm() / g_ieee.norm()),
        "gradient_max_relative": float((g - g_ieee).abs().max() / g_ieee.abs().max()),
    }


def _comparison(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Q-173, Q-190: a variant's median forward plus backward against its baseline's, and
    its gain, 1 − variant / baseline: positive when the variant is faster."""
    base, other = baseline["forward_backward_s"], variant["forward_backward_s"]
    return {"baseline_s": base, "variant_s": other, "gain": 1.0 - other / base}


@contextlib.contextmanager
def _cudnn_flags(*, tf32: bool, autotuning: bool) -> Iterator[None]:
    """A B-4 case's process-global flags, restored afterwards: cuDNN's TF32 and autotuning as
    given, its deterministic algorithms off, and float32 matrix products at "highest" (D-52),
    as a training iteration sets them (Q-73)."""
    import torch

    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    matmul = torch.get_float32_matmul_precision()
    try:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = autotuning, False, tf32
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = saved
        torch.set_float32_matmul_precision(matmul)


def _lpips_networks() -> dict[bool, Any]:
    """lpips 0.1.4's LPIPS-VGG on CUDA, frozen and in evaluation mode as `LPIPSLoss` keeps it
    (Q-41), called directly (PRD #220); keyed by whether its weights are channels-last."""
    import copy
    import warnings

    import lpips
    import torch

    with warnings.catch_warnings():
        # As in `LPIPSLoss`: lpips passes torchvision's deprecated `pretrained=True`.
        warnings.filterwarnings("ignore", "The parameter 'pretrained'", UserWarning)
        warnings.filterwarnings("ignore", "Arguments other than a weight enum", UserWarning)
        network = lpips.LPIPS(net="vgg", verbose=False)
    network.requires_grad_(False)
    network.eval()
    network = network.to("cuda")
    channels_last = copy.deepcopy(network).to(memory_format=torch.channels_last)
    return {False: network, True: channels_last}


def _lpips_forward(network: Any, x: Tensor, y: Tensor, precision: str) -> Tensor:
    """LPIPS of [1, 3, H, W] images in [0, 1], mapped to [−1, 1] as `LPIPSLoss` maps them (D-68),
    under bfloat16 autocast for "bf16" (Q-172), as float32."""
    import torch

    if precision == "bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = network(x, y, normalize=True).mean()
    else:
        value = network(x, y, normalize=True).mean()
    out: Tensor = value.float()
    return out


def _lpips_inputs(gt: Tensor, pred: Tensor, layout: str) -> tuple[Tensor, Tensor]:
    """The prediction, a leaf that requires its gradient, and the target, [1, 3, H, W] in the
    layout's memory format."""
    import torch

    memory = torch.contiguous_format if layout == "nchw" else torch.channels_last
    x = pred[None].permute(0, 3, 1, 2).clone(memory_format=memory).requires_grad_()
    y = gt[None].permute(0, 3, 1, 2).clone(memory_format=memory)
    return x, y


def _lpips_call(
    network: Any, gt: Tensor, pred: Tensor, *, precision: str, layout: str
) -> Callable[[], None]:
    """One B-4 LPIPS call: the forward on both images as the stage `loss.perceptual`, then the
    prediction's gradient as the stage `backward`."""
    import torch

    from openftgs._stages import stage

    x, y = _lpips_inputs(gt, pred, layout)

    def call() -> None:
        with stage("loss.perceptual"):
            value = _lpips_forward(network, x, y, precision)
        with stage("backward"):
            torch.autograd.grad(value, x)

    return call


def _ssim_call(gt: Tensor, pred: Tensor) -> Callable[[], None]:
    """One B-4 SSIM call: `openftgs.losses.ssim` with training's window, σ and padding
    (`LossConfig`; E7, D-8), as the stage `loss.ssim`, then the prediction's gradient as the
    stage `backward`. It takes fused-ssim when that imports (Q-42)."""
    import torch

    from openftgs._stages import stage
    from openftgs.config import LossConfig
    from openftgs.losses import ssim

    cfg = LossConfig()
    x, y = pred[None].clone().requires_grad_(), gt[None]

    def call() -> None:
        with stage("loss.ssim"):
            value = ssim(
                x, y, window=cfg.ssim_window, sigma=cfg.ssim_sigma, padding=cfg.ssim_padding
            )
        with stage("backward"):
            torch.autograd.grad(value, x)

    return call


def _call_figures(call: Callable[[], None], forward: str, repetitions: int) -> dict[str, Any]:
    """The figures of one B-4 case (Q-168, Q-190): the first call, then the medians of the
    forward stage, the backward stage and the span from one's start to the other's end over
    `repetitions` calls after the warm-up, and the peak allocated memory."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    first = _first_call(call)
    timings = _repeat(call, repetitions)
    forwards = [t for t in timings if t.name == forward]
    backwards = [t for t in timings if t.name == "backward"]
    return {
        "forward_backward_s": _median(
            [b.end - f.start for f, b in zip(forwards, backwards, strict=True)]
        ),
        "forward_s": _median([t.seconds for t in forwards]),
        "backward_s": _median([t.seconds for t in backwards]),
        "first_call_s": first,
        "repetitions": len(forwards),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _fused_ssim_importable() -> bool:
    try:
        importlib.import_module("fused_ssim")
    except ImportError:
        return False
    return True


def _losses_child(spec: str) -> None:
    """B-4's cases of one cuDNN autotuning setting, run as `python -c _CHILD SPEC` (Q-190),
    where `fused_ssim` can't be imported (PRD #220): `SPEC` is JSON with the scene's path, the
    sizes, the seed, the repetitions and the setting. Every LPIPS case, then the PyTorch SSIM
    path with TF32 on as in training (D-52), at every size. Prints its cases as one line of
    JSON; out of GPU memory, prints the error on stderr and exits with `_CHILD_OUT_OF_MEMORY`
    (Q-166 as answered)."""
    import torch

    if _fused_ssim_importable():
        raise RuntimeError("fused_ssim can be imported in B-4's child process")
    try:
        cases = _losses_child_cases(json.loads(spec))
    except torch.OutOfMemoryError as error:
        print(error, file=sys.stderr)
        raise SystemExit(_CHILD_OUT_OF_MEMORY) from None
    print(json.dumps(cases))


def _losses_child_cases(given: dict[str, Any]) -> list[dict[str, Any]]:
    from openftgs.data import MultiViewVideo

    autotuning, repetitions = bool(given["autotuning"]), int(given["repetitions"])
    video = MultiViewVideo.from_manifest(Path(given["scene"]))
    images: _Images = {
        (int(w), int(h)): _loss_images(video, int(w), int(h), seed=given["seed"], device="cuda")
        for w, h in given["sizes"]
    }
    timed = _lpips_timed(_lpips_networks(), images, autotuning=autotuning, repetitions=repetitions)
    cases = [{"case": _lpips_case(*key), "figures": figures} for key, figures in timed.items()]
    for (width, height), (gt, pred) in images.items():
        with _cudnn_flags(tf32=True, autotuning=autotuning):
            figures = _call_figures(_ssim_call(gt, pred), "loss.ssim", repetitions)
        case = {"loss": "ssim", "width": width, "height": height, "path": "torch"}
        case |= {"precision": "tf32", "autotuning": autotuning}
        cases.append({"case": case, "figures": figures})
    return cases


def _child_cases(
    scene: Path, sizes: Sequence[tuple[int, int]], *, autotuning: bool
) -> list[dict[str, Any]]:
    """The cases of one autotuning setting, from a child process (`_losses_child`) running this
    openftgs: the folder it is imported from goes first on the child's PYTHONPATH. A child out
    of GPU memory is torch.OutOfMemoryError here, as in this process (Q-166 as answered)."""
    import torch

    import openftgs

    root = str(Path(openftgs.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    spec = {
        "scene": str(scene.resolve()),
        "sizes": [list(size) for size in sizes],
        "seed": _SEED,
        "repetitions": _LOSS_REPETITIONS,
        "autotuning": autotuning,
    }
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if done.returncode == _CHILD_OUT_OF_MEMORY:
        error = (done.stderr.strip().splitlines() or [""])[-1]
        setting = "on" if autotuning else "off"
        raise torch.OutOfMemoryError(f"in the child process with autotuning {setting}: {error}")
    if done.returncode != 0:
        raise RuntimeError(f"B-4's child process failed:\n{done.stderr[-4000:]}")
    cases: list[dict[str, Any]] = json.loads(done.stdout.strip().splitlines()[-1])
    return cases


# B-4's images per (width, height), as (target, prediction); an LPIPS case's key.
_Images = dict[tuple[int, int], tuple["Tensor", "Tensor"]]
_LpipsKey = tuple[int, int, str, str, bool]


def _lpips_case(
    width: int, height: int, precision: str, layout: str, autotuning: bool
) -> dict[str, Any]:
    """An LPIPS case's parameters (Q-190, Q-180)."""
    case = {"loss": "lpips_vgg", "width": width, "height": height, "precision": precision}
    return case | {"layout": layout, "autotuning": autotuning}


def _lpips_deviations(network: Any, images: _Images) -> list[dict[str, Any]]:
    """Per size, TF32's and bf16's deviation from IEEE float32 in value and in the prediction's
    gradient (Q-190), with the network as training runs it: NCHW weights, channels-last
    inputs, autotuning off."""
    import torch

    cases: list[dict[str, Any]] = []
    for (width, height), (gt, pred) in images.items():
        results: dict[str, tuple[Tensor, Tensor]] = {}
        for precision in _LOSS_PRECISIONS:
            x, y = _lpips_inputs(gt, pred, "channels_last_inputs")
            with _cudnn_flags(tf32=precision != "ieee", autotuning=False):
                value = _lpips_forward(network, x, y, precision)
                (grad,) = torch.autograd.grad(value, x)
            results[precision] = (value.detach(), grad)
        for precision in ("tf32", "bf16"):
            figures = _deviation(*results[precision], *results["ieee"])
            case = {"deviation": precision, "width": width, "height": height}
            cases.append({"case": case, "figures": figures})
    return cases


def _lpips_timed(
    networks: dict[bool, Any], images: _Images, *, autotuning: bool, repetitions: int
) -> dict[_LpipsKey, dict[str, Any]]:
    """Every LPIPS case's figures under one autotuning setting (Q-190)."""
    timed: dict[_LpipsKey, dict[str, Any]] = {}
    for (width, height), (gt, pred) in images.items():
        for precision in _LOSS_PRECISIONS:
            for layout in _LOSS_LAYOUTS:
                network = networks[layout == "channels_last_inputs_and_weights"]
                call = _lpips_call(network, gt, pred, precision=precision, layout=layout)
                with _cudnn_flags(tf32=precision != "ieee", autotuning=autotuning):
                    figures = _call_figures(call, "loss.perceptual", repetitions)
                timed[(width, height, precision, layout, autotuning)] = figures
    return timed


def _ssim_fused_timed(images: _Images) -> dict[tuple[int, int], dict[str, Any]]:
    """fused-ssim's figures per size, in this process (Q-42, Q-190)."""
    timed: dict[tuple[int, int], dict[str, Any]] = {}
    for size, (gt, pred) in images.items():
        with _cudnn_flags(tf32=True, autotuning=False):
            timed[size] = _call_figures(_ssim_call(gt, pred), "loss.ssim", _LOSS_REPETITIONS)
    return timed


def _losses(ctx: _Context) -> _Result:
    """B-4 (§3.4, D-52): LPIPS-VGG's forward and the prediction's backward in IEEE float32,
    TF32 and bfloat16, with NCHW inputs, channels-last inputs and channels-last inputs and
    weights, with cuDNN autotuning off and on; SSIM through fused-ssim and through the PyTorch
    path; bf16's (and TF32's) deviation from IEEE float32 in value and gradient; at 1352×1014
    and 1920×1080 (Q-190). The deviations and fused-ssim run in this process, everything
    else in one child process per autotuning setting. The gains Q-173's rule reads are cases
    too; the rule's B-1 half is a rerun of B-1."""
    import gc

    import torch

    sizes = [(int(w), int(h)) for w, h in _LOSS_SIZES]
    with _environment_error("LPIPS networks"):  # exit 2 here, before any child starts
        networks = _lpips_networks()
    images = {size: _loss_images(ctx.video, *size, seed=_SEED, device="cuda") for size in sizes}
    cases = _lpips_deviations(networks[False], images)
    fused = _fused_ssim_importable()
    ssim_fused = _ssim_fused_timed(images) if fused else {}
    for (width, height), figures in ssim_fused.items():
        case = {"loss": "ssim", "width": width, "height": height, "path": "fused"}
        cases.append({"case": case, "figures": figures})
    del networks, images
    gc.collect()
    torch.cuda.empty_cache()  # for the child processes
    children: list[dict[str, Any]] = []
    for autotuning in _LOSS_AUTOTUNING:
        children += _child_cases(ctx.args.scene, sizes, autotuning=autotuning)
    cases += children
    timed = {
        (c["width"], c["height"], c["precision"], c["layout"], c["autotuning"]): entry["figures"]
        for entry in children
        if (c := entry["case"])["loss"] == "lpips_vgg"
    }
    ssim_torch = [entry for entry in children if entry["case"]["loss"] == "ssim"]

    def compare(name: str, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        gains = []
        for (width, height), (base, variant) in zip(sizes, pairs, strict=True):
            figures = _comparison(base, variant)
            gains.append(figures["gain"])
            case = {"comparison": name, "width": width, "height": height}
            cases.append({"case": case, "figures": figures})
        smallest = min(gains)
        figures = {"smallest_gain": smallest, "meets_gain_threshold": smallest >= _GAIN_THRESHOLD}
        cases.append({"case": {"comparison": name}, "figures": figures})

    for name, (base, variant) in _LPIPS_COMPARISONS.items():
        compare(name, [(timed[(*size, *base)], timed[(*size, *variant)]) for size in sizes])
    if fused:
        torch_off = {
            (c["case"]["width"], c["case"]["height"]): c["figures"]
            for c in ssim_torch
            if not c["case"]["autotuning"]
        }
        compare("fused_ssim", [(torch_off[size], ssim_fused[size]) for size in sizes])

    from openftgs.config import LossConfig

    loss_cfg = LossConfig()
    cfg_ssim = {
        "window": loss_cfg.ssim_window,
        "sigma": loss_cfg.ssim_sigma,
        "padding": loss_cfg.ssim_padding,
    }
    setup = {
        "sizes": [list(size) for size in sizes],
        "images": (
            "the scene's first training image, resized bilinearly with antialiasing, as the "
            f"target; the prediction adds Gaussian noise of σ {_LOSS_NOISE} (Q-190)"
        ),
        "seed": _SEED,
        "lpips": "lpips 0.1.4, VGG, called directly; inputs mapped to [−1, 1] (D-68)",
        "precisions": list(_LOSS_PRECISIONS),
        "layouts": list(_LOSS_LAYOUTS),
        "training_layout": "channels_last_inputs",
        "autotuning": list(_LOSS_AUTOTUNING),
        "processes": (
            "the deviations and fused-ssim in the benchmark's process; LPIPS and the PyTorch "
            "SSIM path in one child process per autotuning setting, in which fused_ssim can't "
            "be imported"
        ),
        "ssim": cfg_ssim,
        "fused_ssim": fused,
        "warmup": _WARMUP,
        "repetitions": _LOSS_REPETITIONS,
        "clock": "CUDA events: loss.perceptual or loss.ssim (forward), backward (the prediction's)",
        "comparisons": {name: [list(b), list(v)] for name, (b, v) in _LPIPS_COMPARISONS.items()},
        "turn_on": list(_TURN_ON),
        "gain_threshold": _GAIN_THRESHOLD,
    }
    return _Result(setup, cases)


# -- B-7: rendering at a time, with and without the temporal index (§3.4, D-55; Q-174, Q-194,
# Q-195) ---------------------------------------------------------------------------------------

_B7_WIDTH, _B7_HEIGHT = 1920, 1080  # §3.4 B-7
_B7_SIZES = (1_000_000, 3_000_000)  # #231: N = 1M and 3M
# Q-194: on kitchen's one-frame buckets a 1 % row spans about 4 buckets, the 5 % and 20 %
# rows 17 and 76, which puts them on the long list.
_B7_ACTIVE = (0.01, 0.05, 0.2)
_B7_REPETITIONS = 10  # Q-194: the median of 10 requests
# Q-194: the request sizes below F that the index's break-even is read from; then F.
_B7_REQUEST_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128)
_B7_OUTPUTS = ("cuda", "cpu")  # #231: `render_request`'s outputs on the GPU and on the CPU
_B7_BUDGET_FPS = 450.0  # §3.4 B-7: ≥ 450 FPS on an RTX 4090 (calibrated)
_B7_THRESHOLD_FLOOR = 2  # Q-195: a request at a single time never builds an index


def _b7_camera(video: MultiViewVideo) -> Camera:
    """B-7's camera (Q-168, Q-194): the pose of the first camera in the manifest's camera
    order that has test images, else of the first camera; its fx and fy scaled by 1080/H, H
    its height, and the principal point at the centre of the 1920×1080 image."""
    import torch

    from openftgs.types import Camera

    by_name: dict[str, list[bool]] = {}
    sources: dict[str, Camera] = {}
    for record in video.records:
        by_name.setdefault(record.camera.name, []).append(record.split == "test")
        sources.setdefault(record.camera.name, record.camera)
    names = [name for name in video.manifest.camera_names if name in by_name]
    tested = [name for name in names if any(by_name[name])]
    source = sources[(tested or names)[0]]
    scale = _B7_HEIGHT / source.height
    K = source.K.to(torch.float64)
    intrinsics = [
        [float(K[0, 0]) * scale, 0.0, _B7_WIDTH / 2.0],
        [0.0, float(K[1, 1]) * scale, _B7_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ]
    return Camera(
        torch.tensor(intrinsics, dtype=torch.float32),
        source.world_to_camera.clone(),
        _B7_WIDTH,
        _B7_HEIGHT,
        source.name,
    )


def _spread_times(count: int, k: int) -> list[int]:
    """Q-194: k of `count` timestamps evenly spread, in order: round(j(count − 1)/(k − 1))
    rounded half up, j = 0 … k − 1, in integer arithmetic; the one at ⌊(count − 1)/2⌋ for
    k = 1."""
    if k == 1:
        return [(count - 1) // 2]
    return [(2 * j * (count - 1) + (k - 1)) // (2 * (k - 1)) for j in range(k)]


def _request_sizes(count: int) -> list[int]:
    """Q-194: the sizes of B-7's requests on a scene of `count` distinct training times."""
    return [k for k in _B7_REQUEST_SIZES if k < count] + [count]


def _index_figures(times: dict[int, tuple[float, float]]) -> dict[str, Any]:
    """One model's index figures (Q-194, Q-195), from its median request times with the
    index on and off by request size, outputs on the GPU: the build time, the one-time request
    with the index forced on less the same request forced off (owner answer: `render_request`
    alone, no private name of `render`), which the render's own saving can make negative; and
    the break-even."""
    on, off = times[1]
    return {"build_s": on - off, "break_even_times": _break_even(times)}


def _break_even(times: dict[int, tuple[float, float]]) -> int | None:
    """Q-195: the smallest request size at which the request with the index takes no longer
    than without it (`times` maps a size to the two times, with and without), there and at every
    larger size; None when the index is slower at the largest."""
    found = None
    for k in sorted(times, reverse=True):
        on, off = times[k]
        if on > off:
            break
        found = k
    return found


def _threshold(break_evens: Sequence[int | None]) -> int | None:
    """Q-195: T, the largest break-even over the models that have one, and at least
    `_B7_THRESHOLD_FLOOR`; None when no model has one."""
    known = [k for k in break_evens if k is not None]
    return max(*known, _B7_THRESHOLD_FLOOR) if known else None


def _wall_calls(body: Callable[[], None], repetitions: int) -> list[float]:
    """`_WARMUP` calls of `body`, then `repetitions` calls, each timed on the wall clock from a
    synchronized device to a synchronize after it (Q-194): a render request synchronizes
    inside every render anyway, and its outputs on the GPU are complete only then."""
    import torch

    for _ in range(_WARMUP):
        body()
    seconds = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        body()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return seconds


def _b7_request(camera: Camera, timestamps: Sequence[float], time_unit: str) -> RenderRequest:
    """One item per raw timestamp, ids "0", "1", …, each with `camera`, whose tensors stay on
    the host as a request file gives them (Q-194); black background, the model's SH degree."""
    from openftgs.types import RenderItem, RenderRequest

    items = tuple(RenderItem(str(j), tau, camera) for j, tau in enumerate(timestamps))
    return RenderRequest(items, time_unit=time_unit)


def _request_figures(
    model: FreeTimeGaussians,
    request: RenderRequest,
    *,
    index: bool,
    frame_dt: float,
    device: str,
) -> dict[str, Any]:
    """B-7's figures of one request (Q-194): the first call, then the medians of
    `_B7_REPETITIONS` timed calls of `render_request`: the request's time, FPS and the time
    per render; the peak allocated memory."""
    import torch

    from openftgs.render import render_request

    def call() -> None:
        render_request(
            model,
            request,
            backend="gsplat",
            temporal_index=index,
            frame_dt=frame_dt,
            device=device,
        )

    torch.cuda.reset_peak_memory_stats()
    first = _first_call(call)
    seconds = _median(_wall_calls(call, _B7_REPETITIONS))
    renders = len(request.items)
    return {
        "fps": renders / seconds,
        "request_s": seconds,
        "render_s": seconds / renders,
        "renders": renders,
        "first_call_s": first,
        "repetitions": _B7_REPETITIONS,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _evaluation_figures(
    model: FreeTimeGaussians,
    video: MultiViewVideo,
    evaluator: Evaluator,
    split: str,
    frame_dt: float,
) -> dict[str, Any]:
    """The share of `evaluate_images` spent rendering (PRD #220's 10 % rule, Q-194): the
    evaluation of `split` on the wall clock, once, against the render requests it sends, one
    per timestamp with the split's own cameras and the outputs on the CPU, timed once the same
    way; both after a warm-up of the first request and one Evaluator call."""
    import torch

    from openftgs.eval import evaluate_images
    from openftgs.render import render_request
    from openftgs.types import RenderItem, RenderRequest

    groups: dict[float, list[int]] = {}
    for i, record in enumerate(video.records):
        if record.split == split:
            groups.setdefault(record.timestamp, []).append(i)
    requests = [
        RenderRequest(
            tuple(RenderItem(str(i), tau, video.records[i].camera) for i in group),
            time_unit=video.manifest.time_unit,
        )
        for tau, group in groups.items()
    ]

    def render_all() -> None:
        for request in requests:
            render_request(model, request, backend="gsplat", frame_dt=frame_dt)

    first = requests[0].items[0]
    warm = render_request(model, requests[0], backend="gsplat", frame_dt=frame_dt)
    evaluator(warm.rgb[first.id], video.load_image(int(first.id)))
    torch.cuda.synchronize()
    start = time.perf_counter()
    render_all()
    torch.cuda.synchronize()
    rendering = time.perf_counter() - start
    start = time.perf_counter()
    images = evaluate_images(
        model, video, evaluator, split=split, backend="gsplat", frame_dt=frame_dt
    )
    torch.cuda.synchronize()
    evaluation = time.perf_counter() - start
    return {
        "evaluation_s": evaluation,
        "render_s": rendering,
        "render_share": rendering / evaluation,
        "images": len(images),
        "requests": len(requests),
    }


def _render(ctx: _Context) -> _Result:
    """B-7: the speed of `render.render_request` from one 1920×1080 camera at every distinct
    training time, with the temporal index on and off and the outputs on the GPU and on the CPU,
    and with the index on and off at fewer times, on synthesized models at N = 1M and 3M with
    1 %, 5 % and 20 % of rows active, or on the model given with `--model` (D-88); per model the
    index's build time, its break-even and the share of evaluation spent rendering; then T, the
    number of distinct times from which "auto" builds the index (Q-174, Q-194, Q-195).
    Everything goes through `render_request` and `evaluate_images`, public names only (Q-153;
    the owner's answer in Q-194)."""
    from openftgs.eval import Evaluator

    video = ctx.video
    unit = video.manifest.time_unit
    if ctx.model is not None and ctx.model.model.time_unit != unit:
        _fail(f"the model's time unit {ctx.model.model.time_unit!r} isn't the scene's {unit!r}")
    with _environment_error("LPIPS networks"):  # as `ftgs eval` builds them; found before the run
        evaluator = Evaluator(device="cuda")
    camera = _b7_camera(video)
    taus = sorted({r.timestamp for r in video.records if r.split == "train"})
    count = len(taus)
    sizes = _request_sizes(count)
    # D-55: buckets one training frame interval wide, a checkpoint's Δt_f or the scene's.
    frame_dt = video.frame_dt
    if ctx.model is not None and ctx.model.frame_dt is not None:
        frame_dt = ctx.model.frame_dt
    split = "test" if any(r.split == "test" for r in video.records) else "train"

    def models() -> Iterator[tuple[dict[str, Any], FreeTimeGaussians]]:
        if ctx.model is not None:
            yield {"n": ctx.model.model.num_gaussians, "active": None}, ctx.model.model
            return
        with _as_input_error(f"scene manifest {ctx.args.scene}"):
            volume = _volume(video)
        for n in _B7_SIZES:
            for active in _B7_ACTIVE:
                yield (
                    {"n": n, "active": active},
                    _synthesize(volume, n, active=active, seed=_SEED, device="cuda"),
                )

    cases: list[dict[str, Any]] = []
    budgets: list[dict[str, Any]] = []
    break_evens: list[int | None] = []
    slice_impls: list[str] = []
    for base, model in models():
        slice_impls.append(model.slice_impl)
        on_off: dict[int, tuple[float, float]] = {}
        for k in sizes:
            request = _b7_request(camera, [taus[i] for i in _spread_times(count, k)], unit)
            for device in _B7_OUTPUTS if k == count else _B7_OUTPUTS[:1]:
                timed: dict[bool, float] = {}
                for index in (True, False):
                    case = {**base, "times": k, "index": index, "outputs": device}
                    figures = _request_figures(
                        model, request, index=index, frame_dt=frame_dt, device=device
                    )
                    timed[index] = figures["request_s"]
                    cases.append({"case": case, "figures": figures})
                    if k == count and device == "cuda":
                        budgets.append(
                            {
                                "case": case,
                                "figure": "fps",
                                "limit": _B7_BUDGET_FPS,
                                "value": figures["fps"],
                                "pass": figures["fps"] >= _B7_BUDGET_FPS,
                                "kind": "budget",
                            }
                        )
                if device == "cuda":
                    on_off[k] = (timed[True], timed[False])
        index_figures = _index_figures(on_off)
        break_evens.append(index_figures["break_even_times"])
        cases.append({"case": base, "figures": index_figures})
        cases.append(
            {
                "case": {**base, "evaluation": split},
                "figures": _evaluation_figures(model, video, evaluator, split, frame_dt),
            }
        )
        del model
    cases.append(
        {"case": {"threshold": "T"}, "figures": {"distinct_times": _threshold(break_evens)}}
    )
    setup = {
        "model": (
            "synthesized (Q-168, Q-181)" if ctx.model is None else f"given: {ctx.args.model}"
        ),
        "seed": _SEED if ctx.model is None else None,
        "sizes": list(_B7_SIZES) if ctx.model is None else [ctx.model.model.num_gaussians],
        "active_fractions": list(_B7_ACTIVE) if ctx.model is None else None,
        "slice_impl": sorted(set(slice_impls)),
        "camera": {
            "pose": camera.name,
            "width": camera.width,
            "height": camera.height,
            "K": camera.K.tolist(),
        },
        "times": count,
        "request_sizes": sizes,
        "frame_dt": frame_dt,
        "outputs": list(_B7_OUTPUTS),
        "warmup": _WARMUP,
        "repetitions": _B7_REPETITIONS,
        "clock": "wall clock, from a synchronized device to a synchronize after each request",
        "budget_fps": _B7_BUDGET_FPS,
        "build_time": (
            "Q-194 as answered: the one-time request with the index forced on less the same "
            "request forced off, outputs on the GPU"
        ),
        "threshold_rule": (
            "Q-195: the largest break-even over the models that have one, at least "
            f"{_B7_THRESHOLD_FLOOR}; outputs on the GPU"
        ),
        "evaluation_split": split,
    }
    return _Result(setup, cases, budgets)


# -- B-8: initialization on the wall clock (§3.4, D-56, D-58; Q-106, Q-192) ---------------------


def _spread(values: Sequence[float]) -> dict[str, float]:
    return {
        "total_s": float(sum(values)),
        "median_s": _median(values),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }


def _init(ctx: _Context) -> _Result:
    """B-8's initialization (§3.4, D-56): `roma_init` with the default RomaInitConfig on every
    visible GPU (Q-106), once, under the wall clock: matching per pair and in total per GPU
    (`init.match`, which holds the pair's two image reads and its triangulation), triangulation
    per GPU (`init.triangulate`), the k-NN searches of E15 (`init.knn`), and the whole call. It
    needs the [init] extra and RoMa's weights in torch.hub's cache, and never downloads them, so
    no download is timed (Q-192)."""
    import torch

    from openftgs._stages import Clock
    from openftgs.init import RomaInitConfig, roma_init, roma_provenance, roma_weight_files

    cfg = RomaInitConfig()  # D-58's defaults (Q-192)
    try:
        weights = roma_weight_files(cfg)  # Q-198: where torch.hub keeps them
    except ImportError as error:  # Q-142, Q-166
        _fail(f"bench init needs the [init] extra (pip install 'openftgs[init]'): {error}")
    missing = [str(path) for path in weights if not path.is_file()]
    if missing:  # Q-166, Q-192
        _fail(
            "bench init needs RoMa's weights in torch.hub's cache and doesn't download them; "
            f"missing {', '.join(missing)} (one `ftgs init roma` run fetches them)"
        )
    devices = [str(torch.device("cuda", i)) for i in range(torch.cuda.device_count())]  # Q-106
    try:
        with _as_input_error("bench init: RoMa initialization"), Clock("wall") as clock:
            start = time.perf_counter()
            points = roma_init(ctx.video, cfg)  # devices None: every visible GPU (Q-106)
            wall = time.perf_counter() - start
    except ImportError as error:  # Q-142: the [init] extra, found missing only now
        _fail(f"bench init needs the [init] extra (pip install 'openftgs[init]'): {error}")
    timings = clock.read()

    def of(name: str) -> dict[str | None, list[Timing]]:
        by_key: dict[str | None, list[Timing]] = {}
        for t in timings:
            if t.name == name:
                by_key.setdefault(t.key, []).append(t)
        return by_key

    cases: list[dict[str, Any]] = []
    matches, triangulations = of("init.match"), of("init.triangulate")
    for device in sorted(k for k in matches if k is not None):
        seconds = [t.seconds for t in matches[device]]
        figures = {"pairs": len(seconds), **_spread(seconds)}
        cases.append({"case": {"part": "init.match", "device": device}, "figures": figures})
    for device in sorted(k for k in triangulations if k is not None):
        seconds = [t.seconds for t in triangulations[device]]
        figures = {"pairs": len(seconds), **_spread(seconds)}
        cases.append({"case": {"part": "init.triangulate", "device": device}, "figures": figures})
    searches = [t.seconds for found in of("init.knn").values() for t in found]
    if searches:
        figures = {"frames": len(searches), **_spread(searches)}
        cases.append({"case": {"part": "init.knn"}, "figures": figures})
    pairs = [t for found in matches.values() for t in found]
    frames = 0 if points.frame_ids is None else int(torch.unique(points.frame_ids).numel())
    whole = {
        "wall_s": wall,
        # Loading the weights and building a model per GPU, before the first pair starts.
        "startup_s": min((t.start for t in pairs), default=wall),
        "pairs": len(pairs),
        "init_frames": frames,
        "points": int(points.positions.shape[0]),
    }
    cases.append({"case": {"part": "roma_init"}, "figures": whole})
    setup = {
        "roma_init": json.loads(roma_provenance(cfg)["roma_init"]),  # Q-107's record
        "devices": devices,
        "weights": [str(path) for path in weights],
        "repetitions": 1,
        "clock": "wall: init.match per camera pair, keyed by its shard's device, with "
        "init.triangulate inside it; init.knn per init frame",
    }
    return _Result(setup, cases)


# -- B-8: conversion on the wall clock (§3.4, D-48, D-81; Q-169, Q-193) -------------------------

# Q-193: B-8's scene is a Neural3DV one; selfcap's recorded `-i` paths are inside the
# conversion's own work folder, which it removes, so they can never be found again.
_RERUNS = ("n3dv",)


@dataclass(frozen=True)
class _RecordedConversion:
    """What a manifest records of the conversion that made it (Q-169): the converter, the source
    folder its ffmpeg commands read, and the options to run it with again."""

    converter: str
    source: Path
    frames: Any
    resize: Any
    ffmpeg: str
    ffprobe: str


def _recorded_conversion(ctx: _Context) -> _RecordedConversion:
    """Q-169: the converter and its options from the manifest's metadata, and the source from
    the `-i` paths of its ffmpeg commands; exit 2, naming the path, when one isn't there."""
    where = f"scene manifest {ctx.args.scene}"
    metadata = ctx.video.manifest.metadata
    converter = metadata.get("converter")
    ffmpeg = metadata.get("ffmpeg")
    commands = ffmpeg.get("commands") if isinstance(ffmpeg, dict) else None
    if not isinstance(converter, str) or not isinstance(commands, dict) or not commands:
        _fail(f"{where}: its metadata records no converter run with ffmpeg commands (Q-169)")
    if converter not in _RERUNS:
        _fail(
            f"{where}: recorded by the {converter} converter; bench convert reruns "
            f"{', '.join(_RERUNS)} conversions (B-8, Q-193)"
        )
    assert isinstance(ffmpeg, dict)
    sources: list[Path] = []
    for camera, command in commands.items():
        if not isinstance(command, list) or "-i" not in command[:-1]:
            _fail(f"{where}: the recorded ffmpeg command of {camera} has no -i path (Q-169)")
        path = Path(str(command[command.index("-i") + 1]))
        if not path.is_file():
            _fail(f"source video {path} not found (the ffmpeg command recorded for {camera})")
        sources.append(path)
    folders = sorted({str(path.parent) for path in sources})
    if len(folders) != 1:
        _fail(f"{where}: the recorded ffmpeg commands read from {len(folders)} folders: {folders}")
    for key in ("path", "ffprobe"):
        if not isinstance(ffmpeg.get(key), str):
            _fail(f"{where}: its metadata records no ffmpeg {key} (Q-169)")
    return _RecordedConversion(
        converter=converter,
        source=Path(folders[0]).resolve(),
        frames=metadata.get("frames"),
        resize=metadata.get("resize"),
        ffmpeg=ffmpeg["path"],
        ffprobe=ffmpeg["ffprobe"],
    )


def _cuda_decoding(ffmpeg: str) -> str | None:
    """Q-193: None when the recorded ffmpeg can be asked to decode with CUDA here, otherwise
    why not: a CUDA device must be visible, and the build must list `cuda` among its hardware
    acceleration methods (source log 179)."""
    import torch

    if not torch.cuda.is_available():
        return "not run: no CUDA device is visible"
    listed = _run(ffmpeg, "-hide_banner", "-hwaccels")
    if listed is None:
        return f"not run: {ffmpeg} -hwaccels failed"
    if "cuda" not in [line.strip() for line in listed.splitlines()[1:]]:
        return f"not run: {ffmpeg} lists no cuda hardware acceleration"
    return None


@dataclass(frozen=True)
class _Conversion:
    """One run of a converter: its call's wall time and the stages it recorded."""

    wall: float
    timings: list[Timing]


def _conversion(recorded: _RecordedConversion, hwaccel: Literal["none", "cuda"]) -> _Conversion:
    """The recorded converter, run once with `hwaccel` into a temporary directory that is
    removed afterwards, under the wall clock (§3.4 B-8). A ValueError of the converter (a
    missing ffmpeg, an unreadable source, a pixel format CUDA can't decode) propagates."""
    from openftgs._stages import Clock
    from openftgs.data import convert_n3dv

    with tempfile.TemporaryDirectory(prefix="ftgs-bench-convert-") as folder:
        with _output(f"the temporary conversion in {folder}"), Clock("wall") as clock:
            start = time.perf_counter()
            convert_n3dv(
                recorded.source,
                Path(folder) / "scene",
                frames=recorded.frames,
                resize=recorded.resize,
                ffmpeg=recorded.ffmpeg,
                ffprobe=recorded.ffprobe,
                hwaccel=hwaccel,
            )  # jobs: the converter's default; the manifest doesn't record it (Q-193)
            wall = time.perf_counter() - start
        return _Conversion(wall, clock.read())


def _conversion_cases(hwaccel: str, run: _Conversion) -> list[dict[str, Any]]:
    """Per camera: its frames, the sum of its frames' reads from ffmpeg (`convert.decode`) and
    of their processing and writing in the pool (`convert.encode`, measured in the pool's
    processes, so they overlap), and its wall time, from its first read to its last frame
    written. In total: the converter's call on the wall clock, and the sums."""
    decode: dict[str, list[Timing]] = {}
    encode: dict[str, list[Timing]] = {}
    for t in run.timings:
        parts = {"convert.decode": decode, "convert.encode": encode}.get(t.name)
        if parts is not None and t.key is not None:
            parts.setdefault(t.key, []).append(t)
    cases: list[dict[str, Any]] = []
    for camera in sorted(set(decode) | set(encode)):
        mine = decode.get(camera, []) + encode.get(camera, [])
        figures = {
            "frames": len(encode.get(camera, [])),
            "wall_s": max(t.end for t in mine) - min(t.start for t in mine),
            "decode_s": sum(t.seconds for t in decode.get(camera, [])),
            "encode_s": sum(t.seconds for t in encode.get(camera, [])),
        }
        cases.append({"case": {"hwaccel": hwaccel, "camera": camera}, "figures": figures})
    total = {
        "cameras": len(cases),
        "frames": sum(c["figures"]["frames"] for c in cases),
        "wall_s": run.wall,
        "decode_s": sum(c["figures"]["decode_s"] for c in cases),
        "encode_s": sum(c["figures"]["encode_s"] for c in cases),
    }
    return cases + [{"case": {"hwaccel": hwaccel}, "figures": total}]


def _convert(ctx: _Context) -> _Result:
    """B-8's conversion (§3.4, D-48, D-81): the scene's recorded converter rerun from its
    recorded source, once with `--hwaccel none` and once with `cuda` when the recorded ffmpeg
    supports it here, each into a temporary directory; the wall time per camera and in total.
    It runs once: there is nothing to warm up in a conversion (Q-193)."""
    recorded = _recorded_conversion(ctx)
    where = f"bench convert: the {recorded.converter} converter on {recorded.source}"
    with _as_input_error(where):  # Q-142: the converters' input and environment errors
        runs = {"none": _conversion(recorded, "none")}
    skipped: dict[str, str] = {}
    reason = _cuda_decoding(recorded.ffmpeg)
    if reason is None:
        try:
            runs["cuda"] = _conversion(recorded, "cuda")
        except ValueError as error:  # Q-193: the CPU's numbers stand; the refusal is noted
            reason = f"failed: {error}"
    if reason is not None:
        skipped["cuda"] = reason
    cases = [case for hwaccel, run in runs.items() for case in _conversion_cases(hwaccel, run)]
    version = _run(recorded.ffmpeg, "-version")
    setup = {
        "converter": recorded.converter,
        "source": str(recorded.source),
        "frames": recorded.frames,
        "resize": recorded.resize,
        "ffmpeg": recorded.ffmpeg,
        "ffprobe": recorded.ffprobe,
        "ffmpeg_version": version.splitlines()[0] if version else None,
        "jobs": 0,  # the converter's default, one decoding process per four cores (§2.4)
        "cpus": os.cpu_count(),
        "temporary": tempfile.gettempdir(),
        "hwaccel": list(runs),
        "skipped": skipped,
        "repetitions": 1,
        "clock": "wall: convert.decode per frame read from ffmpeg, convert.encode per frame "
        "processed and written in the pool, keyed by camera",
    }
    return _Result(setup, cases)


# -- the subcommands (§2.1, §3.4) ------------------------------------------------------------------

_BENCHMARKS: dict[str, _Benchmark] = {
    "train-step": _Benchmark("B-1", _train_step, model=True),
    "raster": _Benchmark("B-2", _raster),
    "slice": _Benchmark("B-3", _slice),
    "losses": _Benchmark("B-4", _losses),
    "adam": _Benchmark("B-5", _adam),
    "data": _Benchmark("B-6", _data),
    "render": _Benchmark("B-7", _render, model=True),
    "init": _Benchmark("B-8", _init),
    "convert": _Benchmark("B-8", _convert, cuda=False),
}
