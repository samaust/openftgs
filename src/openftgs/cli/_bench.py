# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench`: the performance benchmarks B-1 to B-8 of §3.4, each a subcommand that writes
BENCH.json.

Spec: §2.1 (the `ftgs bench` line and the exit codes), §3.4, D-50. Q-166 (BENCH.json and the
exit codes), Q-167 (`--model`), Q-168 (the setups). docs/modules/cli/benchmarks.md maps each
subcommand to its B-n.

Each subcommand has an entry in `_BENCHMARKS`, at the end of this module: its B-n, the function
that runs it, and whether it needs CUDA and takes `--model`. The framework checks the scene and
the environment, runs the function, which returns its setup, cases and budgets, and writes
BENCH.json. A benchmark whose ticket hasn't landed has no function yet and keeps Q-130's
behaviour: it says so, writes nothing and returns 1. Adding one is a function and its entry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Literal

from openftgs.cli import (
    _INPUT,
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
    from openftgs.model import FreeTimeGaussians
    from openftgs.optim import FTGSOptimizer
    from openftgs.train import StepLog, Trainer
    from openftgs.types import TimeNormalizer

__all__ = ["add_parser", "run"]  # what `openftgs.cli` takes from this private module

_UNBUILT = 1  # Q-130: a benchmark whose ticket hasn't landed
_FORMAT, _VERSION = "ftgs-bench", 1  # Q-166


@dataclass(frozen=True)
class _Context:
    """What a benchmark gets: the parsed arguments, the scene, and `--model` read on CUDA
    (Q-167), None when not given."""

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
    """One subcommand: its §3.4 ID, the ticket that builds it, the function that runs it (None
    until that ticket lands), whether it needs CUDA (all but `convert`, Q-166) and whether it
    takes `--model` (Q-167)."""

    id: str
    ticket: int
    run: Callable[[_Context], _Result] | None = None
    cuda: bool = True
    model: bool = False


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
    """`ftgs bench BENCHMARK SCENE.json --out BENCH.json [--model MODEL]`: exit 0 once the
    benchmark ran, whatever its budgets; 2 for invalid input or environment (Q-166)."""
    import torch

    bench = _BENCHMARKS[args.benchmark]
    if args.model is not None and not bench.model:  # Q-167
        takers = " and ".join(name for name, b in _BENCHMARKS.items() if b.model)
        _fail(f"--model applies only to {takers}", _INPUT)
    video = _scene(args.scene)
    if bench.cuda and not torch.cuda.is_available():  # Q-166: an invalid environment
        _fail(f"{bench.id} needs a CUDA device", _INPUT)
    model = None if args.model is None else _load_model(args.model, "cuda")  # Q-167
    if bench.run is None:  # Q-130 until its ticket lands
        print(
            f"ftgs bench {args.benchmark}: {bench.id} comes with #{bench.ticket}; "
            "nothing was run or written",
            file=sys.stderr,
        )
        return _UNBUILT
    with _output(f"BENCH.json {args.out}"):  # Q-142: found out before the run, not after it
        args.out.parent.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).isoformat(timespec="seconds")
    result = bench.run(_Context(args, video, model))
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
    model given with `--model` (Q-167); with peak memory, the events, and a bound on a default
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


# -- B-5: the Adam kinds (§3.4, D-53; Q-168, Q-T229.1) --------------------------------------------


def _visibility(n: int, fraction: float, *, seed: int, device: str) -> Tensor:
    """Q-T229.1: selective Adam's mask, a bool [n] with exactly round(`fraction`·n) rows true,
    chosen by a seeded permutation, so the visible rows are spread over the tensors as a render
    leaves them."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[torch.randperm(n, generator=g, device=device)[: round(fraction * n)]] = True
    return mask


def _seed_gradients(model: FreeTimeGaussians, *, seed: int) -> None:
    """#229, Q-T229.1: a seeded N(0, 1) gradient on each of the model's nine stored tensors."""
    import torch

    g = torch.Generator(device=model.means.device).manual_seed(seed)
    for name in model.PARAM_NAMES:
        param = getattr(model, name)
        param.grad = torch.randn(param.shape, generator=g, dtype=param.dtype, device=param.device)


_ADAM_LIMIT = 1.0  # §3.4 B-5: fused no slower than foreach, with no tolerance (Q-T229.1)


def _adam_budgets(
    medians: Mapping[tuple[str, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§3.4 B-5's budget (Q-T229.1): from the median of each `(kind, n)`, per N in the order
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
# that N with 20 % of the rows active, to three decimals (Calibration data, Q-185; Q-T229.1).
_ADAM_CASES: tuple[tuple[int, float], ...] = ((1_000_000, 0.173), (3_000_000, 0.170))
_ADAM_ACTIVE = 0.2  # Q-T229.1: B-1's models at these N
_ADAM_REPETITIONS = 50  # #229: the median of 50


def _adam_call(optimizer: FTGSOptimizer, visibility: Tensor | None) -> Callable[[], None]:
    """One B-5 call (#229): the next iteration's rates, then one Adam step, as the stage
    `adam`; the gradients stay, since no `zero_grad` clears them (Q-T229.1)."""
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
    slower than foreach (#229, Q-T229.1)."""
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


def _budget(
    case: dict[str, Any], figures: dict[str, Any], figure: str, limit: float, kind: str
) -> dict[str, Any]:
    """A budget entry of BENCH.json (Q-166, Q-180) for one figure of one case."""
    value = figures[figure]
    return {
        "case": case,
        "figure": figure,
        "limit": limit,
        "value": value,
        "pass": value <= limit,
        "kind": kind,
    }


# -- the subcommands (§2.1, §3.4) ------------------------------------------------------------------

_BENCHMARKS: dict[str, _Benchmark] = {
    "train-step": _Benchmark("B-1", 226, _train_step, model=True),
    "raster": _Benchmark("B-2", 227),
    "slice": _Benchmark("B-3", 225, _slice),
    "losses": _Benchmark("B-4", 228),
    "adam": _Benchmark("B-5", 229, _adam),
    "data": _Benchmark("B-6", 232),
    "render": _Benchmark("B-7", 231, model=True),
    "init": _Benchmark("B-8", 234),
    "convert": _Benchmark("B-8", 234, cuda=False),
}
