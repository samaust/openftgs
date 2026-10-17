# SPDX-License-Identifier: Apache-2.0
"""The ftgs command line: convert, cache, init, train, render, eval, export and bench.

Spec: §2.1 (CLI block and exit codes), §2.2, §3.4, D-62. Band: Interface (§2.1).

Each command parses its options and calls the library; heavy imports happen inside the
commands, so `ftgs --help` stays fast and `ftgs train` can set environment variables before
CUDA starts. The exit codes are §2.1's: 0 success, 2 invalid input or environment, 3 a
numerical failure (Q-142); `ftgs train` stopped by a signal exits with 128 plus its number
(Q-133). Any other exception is a bug and keeps its traceback (Q-26).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal, NoReturn

if TYPE_CHECKING:
    from openftgs.config import TrainConfig
    from openftgs.data import MultiViewVideo
    from openftgs.model import FreeTimeGaussians
    from openftgs.train import StepLog, Trainer
    from openftgs.types import RenderRequest

__all__ = ["main"]

_INPUT = 2  # §2.1: invalid input or environment
_NUMERIC = 3  # §2.1: a numerical failure (NaN or Inf)
_STUB = 1  # Q-130: ftgs bench until M15

_BENCHMARKS = (
    "train-step",
    "raster",
    "slice",
    "losses",
    "adam",
    "data",
    "render",
    "init",
    "convert",
)
# Q-137: K and world_to_camera of a supplied render's item, relative (absolute near zero), so a
# producer's own float32 rounding of pixel intrinsics still matches.
_CAMERA_TOLERANCE = 1e-6


class _Failure(Exception):
    """A failure the CLI reports in one line, with an exit code of §2.1."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, code: int = _INPUT) -> NoReturn:
    raise _Failure(code, message)


@contextlib.contextmanager
def _as_input_error(what: str, *also: type[Exception]) -> Iterator[None]:
    """Q-26, Q-142: a ValueError (and the listed types) of a call that reads or checks user
    input is exit code 2; anywhere else it stays a bug with its traceback."""
    caught: tuple[type[Exception], ...] = (ValueError, *also)
    try:
        yield
    except caught as error:
        raise _Failure(_INPUT, f"{what}: {error}") from error


@contextlib.contextmanager
def _environment(what: str) -> Iterator[None]:
    """Q-142: an OSError of a run-time download (the LPIPS weights, §2.3 notes) is an
    environment error, exit code 2."""
    try:
        yield
    except OSError as error:
        raise _Failure(_INPUT, f"{what}: {error}") from error


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


def _device() -> str:
    """Q-125: CUDA when PyTorch sees a device, else the CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _backend(device: str) -> Literal["gsplat", "reference"]:
    return "gsplat" if device == "cuda" else "reference"  # Q-69, Q-125


def _empty_or_absent(path: Path, what: str) -> None:
    """Q-126: a directory output must be absent or an empty directory."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        _fail(f"{what} {path} must be absent or an empty directory")


def _json_text(value: Any) -> str:
    return json.dumps(value, allow_nan=False, indent=1)  # Q-25: no NaN or Infinity


def _scene(path: Path) -> MultiViewVideo:
    from openftgs.data import MultiViewVideo

    with _as_input_error(f"scene manifest {path}"):
        return MultiViewVideo.from_manifest(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- argument types ------------------------------------------------------------------------------


def _pair(text: str) -> tuple[int, int]:
    """`A:B` with two integers (`--window N0:N1`, `--profile START:COUNT`)."""
    parts = text.split(":")
    try:
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected A:B with two integers, got {text!r}") from None


# -- convert -------------------------------------------------------------------------------------

# The options each converter takes (§2.1 CLI line, with --test-camera and --window since revision
# 549f; D-80), by
# the keyword they map to; anything else given for a kind is exit code 2.
_CONVERT_OPTIONS: dict[str, tuple[str, ...]] = {
    "n3dv": ("frames", "resize", "ffmpeg", "ffprobe", "hwaccel", "jobs"),
    "opencv-yaml": (
        "fps",
        "frames",
        "resize",
        "ffmpeg",
        "ffprobe",
        "hwaccel",
        "jobs",
        "test_camera",
    ),
    "selfcap": (
        "scene",
        "window",
        "test_camera",
        "undistort",
        "colmap",
        "use_sync",
        "masks",
        "calib",
        "ffmpeg",
        "ffprobe",
        "hwaccel",
        "jobs",
        "resize",
    ),
}
_CONVERT_REQUIRED = {"opencv-yaml": ("fps", "frames", "resize"), "selfcap": ("scene",)}
_FLAGS = {
    "fps": "--fps",
    "frames": "--frames",
    "resize": "--resize",
    "ffmpeg": "--ffmpeg",
    "ffprobe": "--ffprobe",
    "hwaccel": "--hwaccel",
    "jobs": "--jobs",
    "test_camera": "--test-camera",
    "scene": "--scene",
    "window": "--window",
    "undistort": "--undistort",
    "colmap": "--colmap",
    "use_sync": "--no-sync",
    "masks": "--masks",
    "calib": "--calib",
}


def _convert(args: argparse.Namespace) -> int:
    from openftgs import data

    given = {name: getattr(args, name) for name in _FLAGS if getattr(args, name) is not None}
    allowed = _CONVERT_OPTIONS[args.kind]
    wrong = [_FLAGS[name] for name in given if name not in allowed]
    if wrong:
        _fail(f"{', '.join(wrong)} not valid for the {args.kind} converter")
    missing = [_FLAGS[name] for name in _CONVERT_REQUIRED.get(args.kind, ()) if name not in given]
    if missing:
        _fail(f"the {args.kind} converter needs {', '.join(missing)}")
    if "use_sync" in given:
        given["use_sync"] = False  # --no-sync
    converters: dict[str, Callable[..., Path]] = {
        "n3dv": data.convert_n3dv,
        "opencv-yaml": data.convert_opencv_yaml,
        "selfcap": data.convert_selfcap,
    }
    with _as_input_error(f"ftgs convert {args.kind}"):  # Q-88: input and environment errors
        manifest = converters[args.kind](args.src, args.out, **given)
    print(manifest)
    return 0


# -- cache ---------------------------------------------------------------------------------------


def _cache(args: argparse.Namespace) -> int:
    from openftgs.data import FrameCache

    video = _scene(args.scene)
    with _as_input_error("frame cache"):
        directory = FrameCache.build(video)
    print(f"{directory} ({FrameCache.size_of(video)} bytes)")
    if args.placement != "auto":  # Q-127: show that it fits or maps
        device = "cpu"
        if args.placement == "gpu":
            device = _device()
            if device != "cuda":
                _fail("--placement gpu needs a CUDA device")
        import torch

        # Q-142: a cache that doesn't fit in GPU memory is an environment error.
        with _as_input_error(f"frame cache, placement {args.placement}", torch.OutOfMemoryError):
            cache = FrameCache.open(video, placement=args.placement, device=device)
        print(f"opened in placement {args.placement}: {cache.nbytes} bytes")
    return 0


# -- init ----------------------------------------------------------------------------------------


def _init(args: argparse.Namespace) -> int:
    from openftgs.init import RomaInitConfig, roma_init, roma_provenance
    from openftgs.io import write_init_points

    video = _scene(args.scene)
    fields = {
        "stride": args.stride,
        "points_per_frame": args.points_per_frame,
        "seed": args.seed,
        "model": args.model,
    }
    with _as_input_error("RoMa settings"):
        cfg = RomaInitConfig(
            **{k: v for k, v in fields.items() if v is not None},
            upsample_preds=not args.no_upsample,
        )
    if _device() != "cuda":  # Q-125: roma_init matches on every visible CUDA device
        _fail("RoMa initialization needs a CUDA device")
    try:
        with _as_input_error("RoMa initialization"):
            points = roma_init(video, cfg)
    except ImportError as error:  # Q-142: the [init] extra is missing
        _fail(f"RoMa initialization needs the [init] extra (pip install 'openftgs[init]'): {error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with _as_input_error(f"init points {args.out}"):
        write_init_points(
            points,
            args.out,
            time_unit=video.manifest.time_unit,
            producer=_producer(),
            metadata=roma_provenance(cfg),  # Q-107
        )
    print(f"{args.out}: {points.positions.shape[0]} points")
    return 0


# -- train ---------------------------------------------------------------------------------------


def _read_config(path: Path | None, seed: int | None) -> TrainConfig:
    """§2.1: --config is a JSON TrainConfig (Q-10); --seed replaces its seed."""
    from openftgs.config import config_from_dict

    with _as_input_error(f"configuration {path}", OSError):
        data = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("a TrainConfig file holds a JSON object")
        cfg = config_from_dict(data)
    if seed is not None:
        with _as_input_error("--seed"):
            cfg = dataclasses.replace(cfg, seed=seed)
    return cfg


def _flat(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            out |= _flat(value, f"{prefix}{key}.")
        else:
            out[f"{prefix}{key}"] = value
    return out


def _check_resume_dir(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Q-134: a resumed run writes into an absent or empty directory, or into the run directory
    of its own configuration (its config.json)."""
    if not run_dir.exists() or (run_dir.is_dir() and not any(run_dir.iterdir())):
        return
    try:
        recorded = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = None
    if recorded != json.loads(json.dumps(config)):
        _fail(f"run directory {run_dir} isn't this run's: its config.json differs or is missing")


def _check_masks(video: MultiViewVideo, split: str) -> None:
    """Q-135 (#83): a split's masks are all there or none are."""
    indices = [i for i, r in enumerate(video.records) if r.split == split]
    absent = [i for i in indices if video.manifest.masks[i] is None]
    if absent and len(absent) < len(indices):
        root = video.manifest.path.parent
        names = [video.records[i].image_path.relative_to(root).as_posix() for i in absent]
        _fail(
            f"the {split} split has masks on some images but not on {len(absent)}: {names[:5]}"
            " (§1.4: one mask per test image, or none)"
        )


class _Signals:
    """§2.2, Q-133: SIGINT and SIGTERM during training are recorded, and the trainer stops
    after the current iteration; the old handlers come back afterwards."""

    def __init__(self) -> None:
        self.received: int | None = None
        self._old: dict[int, Any] = {}

    def __enter__(self) -> _Signals:
        for number in (signal.SIGINT, signal.SIGTERM):
            self._old[number] = signal.signal(number, self._record)
        return self

    def __exit__(self, *exc: object) -> None:
        for number, handler in self._old.items():
            signal.signal(number, handler)

    def _record(self, number: int, frame: FrameType | None) -> None:
        self.received = number


class _Profile:
    """§3.4: a torch.profiler trace of iterations START … START + COUNT − 1 (NVTX ranges per
    stage come with M15)."""

    def __init__(self, start: int, count: int, path: Path, device: str) -> None:
        import torch.profiler

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(activities=activities)
        self._first, self._last, self._path = start, start + count - 1, path
        self._running = False

    def begin_if_due(self, iteration: int) -> None:
        """Start before iteration `iteration` + 1 when that is START."""
        if iteration + 1 == self._first and not self._running:
            self._profiler.start()
            self._running = True

    def step_done(self, iteration: int) -> None:
        """After iteration `iteration`: start before the window's first, finish after its
        last."""
        self.begin_if_due(iteration)
        if iteration == self._last:
            self.finish()

    def finish(self) -> None:
        """Stop and write the trace, if the window has started (Q-144)."""
        if self._running:
            self._profiler.stop()
            self._running = False
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(str(self._path))


def _train(args: argparse.Namespace) -> int:
    from openftgs.config import config_from_dict, config_to_dict
    from openftgs.io import load_checkpoint, read_init_points, save_checkpoint
    from openftgs.train import Trainer

    if args.force and args.resume is None:
        _fail("--force only applies with --resume")
    run_dir: Path = args.out
    if args.resume is None:
        _empty_or_absent(run_dir, "run directory")  # Q-132
    video = _scene(args.scene)
    with _as_input_error(f"init points {args.init}"):
        points = read_init_points(args.init, time_unit=video.manifest.time_unit)
    has_test = any(r.split == "test" for r in video.records)
    if has_test:
        _check_masks(video, "test")
    if args.resume is None:
        cfg = _read_config(args.config, args.seed)
    else:
        with _as_input_error(f"checkpoint {args.resume}"):
            stored = load_checkpoint(args.resume).config
            cfg = config_from_dict(stored)
        if args.config is not None or args.seed is not None:  # Q-134
            if args.config is not None:
                given = config_to_dict(_read_config(args.config, args.seed))
            else:
                with _as_input_error("--seed"):
                    given = config_to_dict(dataclasses.replace(cfg, seed=args.seed))
            want, have = _flat(json.loads(json.dumps(stored))), _flat(given)
            differ = sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))
            if differ:
                _fail(f"--resume continues the checkpoint's configuration; these differ: {differ}")
    if args.resume is not None:  # Q-134: the run being resumed, or a new run directory
        _check_resume_dir(run_dir, config_to_dict(cfg))
    if cfg.perf.deterministic:  # Q-131 (Q-73 note 2), before any CUDA call
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _device()
    if args.resume is None:
        with _as_input_error("training setup", NotImplementedError):
            trainer = Trainer(cfg, video, points, device=device)
    else:
        with _as_input_error(f"resume from {args.resume}", NotImplementedError):
            trainer = Trainer.from_checkpoint(args.resume, video, device=device, force=args.force)
    last = cfg.optim.iterations
    profile = None
    if args.profile is not None:
        start, count = args.profile
        # Q-144: the window lies inside the iterations this command runs.
        if start < 1 or count < 1 or start <= trainer.iteration or start + count - 1 > last:
            _fail(
                f"--profile {start}:{count} must cover iterations in "
                f"({trainer.iteration}, {last}], with START and COUNT at least 1"
            )
        profile = _Profile(start, count, run_dir / "profile" / "trace.json", device)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(_json_text(config_to_dict(cfg)), encoding="utf-8")
    log_path = run_dir / "train_log.jsonl"

    def record(trainer: Trainer, log: StepLog) -> bool:
        if profile is not None:
            profile.step_done(log.iteration)
        if log.mean_losses is not None:  # Q-132: read every log_every iterations, no extra sync
            row = {
                "iteration": log.iteration,
                "losses": log.mean_losses,
                "num_gaussians": log.num_gaussians,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            total = sum(log.mean_losses.values())
            print(
                f"iteration {log.iteration}/{last}  loss {total:.6g}  N {log.num_gaussians}",
                file=sys.stderr,
            )
        return signals.received is not None

    if profile is not None:
        profile.begin_if_due(trainer.iteration)
    # Q-133: the guard stays on through the save, so a signal can't interrupt it.
    with _Signals() as signals:
        try:
            trainer.train(callbacks=[record])
        except FloatingPointError as error:  # Q-69: divergence; nothing is saved
            _fail(str(error), _NUMERIC)
        finally:
            if profile is not None:
                profile.finish()  # Q-144: a window cut short by a signal is still written
        state = trainer.state_dict()
        _check_finite(trainer, state.stats)
        save_checkpoint(state, run_dir / "checkpoint")
    if signals.received is not None and trainer.iteration < last:  # Q-133; a finished run goes on
        print(
            f"stopped by signal {signals.received} at iteration {trainer.iteration}; "
            f"checkpoint in {run_dir / 'checkpoint'}",
            file=sys.stderr,
        )
        return 128 + signals.received
    if has_test:  # §1.3 calendar: the final evaluation, after the checkpoint (Q-132)
        with _environment("LPIPS networks"):
            trainer.evaluate("test", out_dir=run_dir)
    return 0


def _check_finite(trainer: Trainer, stats: Mapping[str, Any]) -> None:
    """Q-133: the trainer checks for divergence at log reads and events (Q-69), so a model that
    turned non-finite since then is caught here, and a diverged state is never saved."""
    import torch

    try:
        trainer.model.validate()  # Q-55: §1.1's rules, finite values among them
    except ValueError as error:
        _fail(f"training diverged by iteration {trainer.iteration}: {error}", _NUMERIC)
    if not all(bool(torch.isfinite(v).all()) for v in stats.values()):
        _fail(f"training diverged by iteration {trainer.iteration}: statistics", _NUMERIC)


# -- models --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Loaded:
    model: FreeTimeGaussians
    iteration: int | None  # a checkpoint's
    # D-84: a checkpoint's configuration's background, else black until #187
    background: tuple[float, float, float]


def _load_model(path: Path, device: str) -> _Loaded:
    """MODEL of §2.1: a checkpoint directory or an FTGS-IF file."""
    from openftgs.config import config_from_dict
    from openftgs.io import load_checkpoint, read_exchange
    from openftgs.model import FreeTimeGaussians

    with _as_input_error(f"model {path}"):
        if not path.is_dir():
            return _Loaded(
                FreeTimeGaussians.from_exchange(read_exchange(path), device=device),
                None,
                (0.0, 0.0, 0.0),
            )
        state = load_checkpoint(path)
        cfg = config_from_dict(state.config)
        model = FreeTimeGaussians(
            {k: v.to(device) for k, v in state.params.items()},
            sh_degree=cfg.sh_degree,
            time=state.time,
            time_unit=state.time_unit,
        )
        return _Loaded(model, state.iteration, cfg.background)


def _render_device(backend: str) -> str:
    if backend == "reference":
        return "cpu"  # Q-125
    if _device() != "cuda":
        _fail("the gsplat backend needs a CUDA device; pass --backend reference")
    return "cuda"


# -- render --------------------------------------------------------------------------------------


def _render(args: argparse.Namespace) -> int:
    import torch

    from openftgs.io import read_render_request, write_renders
    from openftgs.render import render_request

    device = _render_device(args.backend)
    loaded = _load_model(args.model, device)
    with _as_input_error(f"render request {args.request}", OSError):
        sha = _sha256(args.request)
        request = read_render_request(args.request)
    if request.time_unit != loaded.model.time_unit:
        _fail(
            f"the request's time unit {request.time_unit!r} isn't the model's "
            f"{loaded.model.time_unit!r}"
        )
    if request.sh_degree is not None and request.sh_degree > loaded.model.sh_degree:
        _fail(
            f"the request's sh_degree {request.sh_degree} is above the model's "
            f"{loaded.model.sh_degree}"
        )
    renders = render_request(loaded.model, request, backend=args.backend)
    bad = [i for i, rgb in renders.rgb.items() if not bool(torch.isfinite(rgb).all())]
    bad += [i for i, a in renders.alpha.items() if not bool(torch.isfinite(a).all())]
    if bad:  # Q-129
        _fail(f"renders that aren't finite: {sorted(set(bad))[:5]}", _NUMERIC)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_renders(renders, args.out, producer=_producer(), request_sha256=sha)
    return 0


# -- eval ----------------------------------------------------------------------------------------


def _match_renders(
    video: MultiViewVideo, split: str, directory: Path
) -> tuple[dict[int, Any], dict[str, str]]:
    """Q-137: each record of the split matched to exactly one item of DIR's request, by its
    timestamp, image size and camera, and that item's render."""
    import torch
    from safetensors import safe_open

    from openftgs.io import read_render_request, read_renders

    request_path, renders_path = directory / "request.json", directory / "renders.safetensors"
    with _as_input_error(f"--renders-dir {directory}", OSError):
        sha = _sha256(request_path)
        request: RenderRequest = read_render_request(request_path)
        renders = read_renders(renders_path, request_sha256=sha)
        with safe_open(str(renders_path), framework="pt") as f:
            producer = (f.metadata() or {}).get("producer", "")
    if request.time_unit != video.manifest.time_unit:
        _fail(
            f"--renders-dir: the request's time unit {request.time_unit!r} isn't the scene's "
            f"{video.manifest.time_unit!r}"
        )
    root = video.manifest.path.parent
    found: dict[int, Any] = {}
    for i, record in enumerate(video.records):
        if record.split != split:
            continue
        cam = record.camera
        hits = [
            item
            for item in request.items
            if item.timestamp == record.timestamp
            and (item.camera.width, item.camera.height) == (cam.width, cam.height)
            and torch.allclose(
                item.camera.K.double(),
                cam.K.double(),
                rtol=_CAMERA_TOLERANCE,
                atol=_CAMERA_TOLERANCE,
            )
            and torch.allclose(
                item.camera.world_to_camera.double(),
                cam.world_to_camera.double(),
                rtol=_CAMERA_TOLERANCE,
                atol=_CAMERA_TOLERANCE,
            )
        ]
        name = record.image_path.relative_to(root).as_posix()
        if len(hits) != 1:
            _fail(f"--renders-dir: {len(hits)} request items match {name}; it needs exactly one")
        rgb = renders.rgb.get(hits[0].id)
        if rgb is None or tuple(rgb.shape) != (cam.height, cam.width, 3):
            _fail(f"--renders-dir: no render of {cam.height}×{cam.width} for {name}")
        found[i] = rgb
    entry = {"dir": str(directory), "request_sha256": sha, "producer": producer}
    return found, entry


def _eval(args: argparse.Namespace) -> int:
    from openftgs.eval import Evaluator, evaluate_images, metrics_report, write_metrics

    device = _device()
    loaded = _load_model(args.model, device)
    video = _scene(args.scene)
    if not any(r.split == args.split for r in video.records):
        _fail(f"the {args.split} split of {args.scene} holds no images")
    if loaded.model.time_unit != video.manifest.time_unit:
        _fail(
            f"the model's time unit {loaded.model.time_unit!r} isn't the scene's "
            f"{video.manifest.time_unit!r}"
        )
    _check_masks(video, args.split)
    supplied, entry = (None, None)
    if args.renders_dir is not None:
        supplied, entry = _match_renders(video, args.split, args.renders_dir)
    with _environment("LPIPS networks"):
        evaluator = Evaluator(device=device)
    images = evaluate_images(
        loaded.model,
        video,
        evaluator,
        split=args.split,
        backend=_backend(device),
        background=loaded.background,
        renders=supplied,
    )
    body = metrics_report(images, split=args.split, metadata=video.manifest.metadata)
    write_metrics(
        args.out,
        body,
        manifest=video.manifest,
        model_path=args.model,
        iteration=loaded.iteration,
        renders=entry,
    )
    return 0


# -- export --------------------------------------------------------------------------------------


def _export(args: argparse.Namespace) -> int:
    from openftgs.io import write_exchange

    if not args.checkpoint.is_dir():
        _fail(f"{args.checkpoint} is not a checkpoint directory")
    loaded = _load_model(args.checkpoint, "cpu")
    # D-84: the file records the background the run was trained against.
    exchange = dataclasses.replace(loaded.model.to_exchange(), background=loaded.background)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with _as_input_error(f"exchange model {args.out}"):
        write_exchange(exchange, args.out)
    return 0


# -- bench ---------------------------------------------------------------------------------------


def _bench(args: argparse.Namespace) -> int:
    _scene(args.scene)
    print(
        f"ftgs bench {args.benchmark}: the benchmarks of §3.4 arrive with milestone M15; "
        "nothing was run or written",
        file=sys.stderr,
    )
    return _STUB  # Q-130


# -- parser --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftgs",
        description="openftgs, an independent clean-room implementation of FreeTimeGS "
        "(Wang et al., CVPR 2025), not affiliated with its authors.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = commands.add_parser("convert", help="convert a dataset into a scene manifest")
    p.add_argument("kind", choices=tuple(_CONVERT_OPTIONS))
    p.add_argument("src", type=Path, metavar="SRC")
    p.add_argument("out", type=Path, metavar="OUT")
    p.add_argument("--scene", metavar="NAME")
    p.add_argument("--fps", type=float, metavar="F")
    p.add_argument("--frames", type=int, metavar="N")
    p.add_argument("--resize", type=float, metavar="R")
    p.add_argument("--ffmpeg", metavar="PATH")
    p.add_argument("--ffprobe", metavar="PATH")
    p.add_argument("--hwaccel", choices=("none", "cuda"))
    p.add_argument("--jobs", type=int, metavar="J")
    p.add_argument("--test-camera", dest="test_camera", metavar="NAME")
    p.add_argument("--window", type=_pair, metavar="N0:N1")
    p.add_argument("--undistort", choices=("colmap", "internal"))
    p.add_argument("--colmap", metavar="PATH")
    p.add_argument("--no-sync", dest="use_sync", action="store_const", const=False)
    p.add_argument("--masks", type=Path, metavar="DIR")
    p.add_argument("--calib", type=Path, metavar="DIR")
    p.set_defaults(run=_convert)

    p = commands.add_parser("cache", help="build the frame cache of a scene")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--placement", choices=("auto", "gpu", "mmap"), default="auto")
    p.set_defaults(run=_cache)

    p = commands.add_parser(
        "init",
        help="initial points from a scene",
        description="Initial points for training. D-24's check of F·B against N_max happens "
        "when `ftgs train` builds the model, where it is exit code 2 (Q-128).",
    )
    p.add_argument("method", choices=("roma",))
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("out", type=Path, metavar="OUT.safetensors")
    p.add_argument("--stride", type=int, metavar="K")
    p.add_argument("--points-per-frame", dest="points_per_frame", type=int, metavar="B")
    p.add_argument("--seed", type=int, metavar="S")
    p.add_argument("--model", choices=("roma_outdoor", "tiny_roma_v1_outdoor"))
    p.add_argument("--no-upsample", dest="no_upsample", action="store_true")
    p.set_defaults(run=_init)

    p = commands.add_parser("train", help="train a model")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--init", type=Path, required=True, metavar="INIT.safetensors")
    p.add_argument("--out", type=Path, required=True, metavar="RUN_DIR")
    p.add_argument("--config", type=Path, metavar="CFG.json")
    p.add_argument("--seed", type=int, metavar="S")
    p.add_argument("--resume", type=Path, metavar="CKPT_DIR")
    p.add_argument("--force", action="store_true")
    p.add_argument("--profile", type=_pair, metavar="START:COUNT")
    p.set_defaults(run=_train)

    p = commands.add_parser("render", help="answer a render request")
    p.add_argument("model", type=Path, metavar="MODEL")
    p.add_argument("--request", type=Path, required=True, metavar="REQUEST.json")
    p.add_argument("--out", type=Path, required=True, metavar="RENDERS.safetensors")
    p.add_argument("--backend", choices=("gsplat", "reference"), default="gsplat")
    p.set_defaults(run=_render)

    p = commands.add_parser("eval", help="evaluate a model on a scene")
    p.add_argument("model", type=Path, metavar="MODEL")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--split", choices=("test", "train"), default="test")  # Q-143
    p.add_argument("--out", type=Path, required=True, metavar="METRICS.json")
    p.add_argument("--renders-dir", dest="renders_dir", type=Path, metavar="DIR")
    p.set_defaults(run=_eval)

    p = commands.add_parser("export", help="export a checkpoint as an exchange model")
    p.add_argument("checkpoint", type=Path, metavar="CKPT_DIR")
    p.add_argument("--format", choices=("ftgs-if",), required=True)
    p.add_argument("--out", type=Path, required=True, metavar="MODEL.safetensors")
    p.set_defaults(run=_export)

    p = commands.add_parser("bench", help="performance benchmarks (M15)")
    p.add_argument("benchmark", choices=_BENCHMARKS)
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--out", type=Path, required=True, metavar="BENCH.json")
    p.set_defaults(run=_bench)
    return parser


# Q-124: the module's one public name; the ftgs console script and `python -m openftgs.cli`.
def main(argv: Sequence[str] | None = None) -> int:
    """Run `ftgs` with `argv` (the process's arguments by default) and return its exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:  # argparse: --help is 0, a usage error 2 (§2.1)
        return stop.code if isinstance(stop.code, int) else _INPUT
    try:
        code: int = args.run(args)
    except _Failure as failure:
        print(f"ftgs {args.command}: error: {failure}", file=sys.stderr)
        return failure.code
    return code


if __name__ == "__main__":
    sys.exit(main())
