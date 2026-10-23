# SPDX-License-Identifier: Apache-2.0
"""The ``ftgs`` command line of §2.1: convert, cache, init, train, render, eval, export, bench.

Each command parses its options and calls the library; heavy imports happen inside the
commands, so ``ftgs --help`` stays fast and ``ftgs train`` can set environment variables
before CUDA starts. The module's one public name is ``main``, which the ``ftgs`` console
script and ``python -m openftgs.cli`` run.

Notes
-----
Spec: §2.1 (CLI block and exit codes), §2.2, §3.4, D-62. Band: Interface (§2.1).

The exit codes are §2.1's: 0 success, 2 invalid input or environment, 3 a numerical failure
(Q-142); ``ftgs train`` stopped by a signal exits with 128 plus its number (Q-133). Any other
exception is a bug and keeps its traceback (Q-26). ``ftgs bench`` and its benchmarks are in
the private module ``_bench`` (Q-166 to Q-168).
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
    """Turn a ValueError, or one of the listed types, into exit code 2.

    Q-26, Q-142: for a call that reads or checks user input; anywhere else a ValueError stays
    a bug with its traceback.
    """
    caught: tuple[type[Exception], ...] = (ValueError, *also)
    try:
        yield
    except caught as error:
        raise _Failure(_INPUT, f"{what}: {error}") from error


@contextlib.contextmanager
def _environment(what: str) -> Iterator[None]:
    """Turn an OSError of a run-time download into exit code 2.

    Q-142: an environment error, as for the LPIPS weights (§2.3 notes).
    """
    try:
        yield
    except OSError as error:
        raise _Failure(_INPUT, f"{what}: {error}") from error


@contextlib.contextmanager
def _output(what: str) -> Iterator[None]:
    """Turn an OSError while writing a user-chosen output into exit code 2.

    Q-142 (as answered): making the output's directory included; the atomic writes leave the
    old file or checkpoint (Q-25, Q-126).
    """
    try:
        yield
    except OSError as error:
        raise _Failure(_INPUT, f"cannot write {what}: {error}") from error


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


def _device() -> str:
    """Return CUDA when PyTorch sees a device, else the CPU (Q-125)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _backend(device: str) -> Literal["gsplat", "reference"]:
    return "gsplat" if device == "cuda" else "reference"  # Q-69, Q-125


def _shard_devices(device: str) -> list[str] | None:
    """Return the devices to shard over, or None to run unsharded.

    D-54, Q-175, Q-186: on CUDA with two or more visible CUDA devices, which
    `CUDA_VISIBLE_DEVICES` restricts, every one of them, one shard each; None with one, or on
    the CPU, which renders unsharded with no copy of the model.
    """
    import torch

    if device != "cuda":
        return None
    count = torch.cuda.device_count()
    if count < 2:  # Q-186: one GPU shards nothing, so nothing is copied
        return None
    return [f"cuda:{k}" for k in range(count)]


def _empty_or_absent(path: Path, what: str) -> None:
    """Refuse a directory output that is neither absent nor an empty directory (Q-126)."""
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
    """Parse `A:B` with two integers (`--window N0:N1`, `--profile START:COUNT`)."""
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
    with _output(f"init points {args.out}"), _as_input_error(f"init points {args.out}"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
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
    """Read --config, a JSON TrainConfig (§2.1, Q-10), with --seed replacing its seed."""
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


def _check_resume_dir(run_dir: Path, cfg: TrainConfig) -> None:
    """Refuse a resumed run's directory unless it is absent, empty or the run's own (Q-134).

    A resumed run writes into an absent or empty directory, or into the run directory of its
    own configuration: its config.json differs at most in the performance settings a resume
    may change (Q-134 as answered, Q-153).
    """
    from openftgs.config import config_from_dict, resume_conflicts

    if not run_dir.exists() or (run_dir.is_dir() and not any(run_dir.iterdir())):
        return
    try:
        data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        recorded = config_from_dict(data) if isinstance(data, dict) else None
    except (OSError, ValueError):
        recorded = None
    if recorded is None or resume_conflicts(recorded, cfg):
        _fail(f"run directory {run_dir} isn't this run's: its config.json differs or is missing")


def _check_masks(video: MultiViewVideo, split: str) -> None:
    """Refuse a split whose masks aren't all there or all absent (Q-135, #83)."""
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
    """The record of SIGINT and SIGTERM during training (§2.2, Q-133).

    The trainer stops after the current iteration; the old handlers come back afterwards.
    """

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
    """A torch.profiler trace of iterations START … START + COUNT − 1 (§3.4).

    Every stage of an iteration is a named range in it; the stages' NVTX ranges run with or
    without it.
    """

    def __init__(self, start: int, count: int, path: Path, device: str) -> None:
        import torch.profiler

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(activities=activities)
        self._first, self._last, self.path = start, start + count - 1, path
        self._running = False

    def begin_if_due(self, iteration: int) -> None:
        """Start before iteration `iteration` + 1 when that is START."""
        if iteration + 1 == self._first and not self._running:
            from openftgs._stages import set_profiling

            self._profiler.start()
            self._running = True
            set_profiling(True)  # §3.4: the trace carries the stage names

    def step_done(self, iteration: int) -> None:
        """Start before the window's first iteration, and finish after its last.

        Called after iteration `iteration`.
        """
        self.begin_if_due(iteration)
        if iteration == self._last:
            self.finish()

    def finish(self) -> None:
        """Stop and write the trace, if the window has started (Q-144)."""
        if self._running:
            from openftgs._stages import set_profiling

            set_profiling(False)
            self._profiler.stop()
            self._running = False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(str(self.path))


def _train(args: argparse.Namespace) -> int:
    import torch

    from openftgs.config import config_from_dict, config_to_dict, resume_conflicts
    from openftgs.io import (
        check_run_checkpoints,
        load_checkpoint,
        read_init_points,
        save_run_checkpoint,
    )
    from openftgs.train import Trainer

    if args.force and args.resume is None:
        _fail("--force only applies with --resume")
    if args.checkpoint_every < 0:  # D-83
        _fail(f"--checkpoint-every must be at least 0, got {args.checkpoint_every}")
    if args.keep_checkpoints < 1:  # D-83
        _fail(f"--keep-checkpoints must be at least 1, got {args.keep_checkpoints}")
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
            cfg = config_from_dict(load_checkpoint(args.resume).config)
        if args.config is not None or args.seed is not None:  # Q-134
            if args.config is not None:
                given = _read_config(args.config, args.seed)
            else:
                with _as_input_error("--seed"):
                    given = dataclasses.replace(cfg, seed=args.seed)
            # Q-134 (as answered), Q-153: only the performance settings that change no number
            # may differ, and the resumed run uses the given ones.
            differ = resume_conflicts(cfg, given)
            if differ:
                _fail(f"--resume continues the checkpoint's configuration; these differ: {differ}")
            cfg = given
    if args.resume is not None:  # Q-134: the run being resumed, or a new run directory
        _check_resume_dir(run_dir, cfg)
    with _as_input_error("--keep-checkpoints", OSError):  # Q-150 (as answered), Q-163
        check_run_checkpoints(run_dir, keep=args.keep_checkpoints)
    if cfg.perf.deterministic:  # Q-131 (Q-73 note 2), before any CUDA call
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _device()
    if args.resume is None:
        with _as_input_error("training setup", NotImplementedError):
            trainer = Trainer(cfg, video, points, device=device)
    else:
        with _as_input_error(f"resume from {args.resume}", NotImplementedError):
            trainer = Trainer.from_checkpoint(
                args.resume, video, device=device, force=args.force, perf=cfg.perf
            )
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

    with _output(f"run directory {run_dir}"):
        run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    with _output(str(config_path)):  # Q-142: replaced atomically, as the other outputs (Q-126)
        temporary = config_path.with_name(f".{config_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(_json_text(config_to_dict(cfg)), encoding="utf-8")
            os.replace(temporary, config_path)
        finally:
            temporary.unlink(missing_ok=True)
    log_path = run_dir / "train_log.jsonl"
    trace_failure: _Failure | None = None  # Q-142: reported once the checkpoint is safe
    log_failure: _Failure | None = None  # Q-159 (as answered): reported once the checkpoint is safe

    def save() -> None:
        """Save a checkpoint, the state checked first, and rotate the kept ones (D-83).

        Every save, periodic, final, on a signal or after a train_log.jsonl that can't be
        written (Q-159 as answered), checks the state first (Q-133) and rotates the run
        directory's kept checkpoints.
        """
        state = trainer.state_dict()
        _check_finite(trainer, state.stats)
        # Q-159: no further save is attempted; Q-150 (as answered): a rotation refused onto an
        # entry that appeared since the check before training is exit 2 too, not a traceback.
        with (
            _output(f"a checkpoint in {run_dir}"),
            _as_input_error(f"cannot write a checkpoint in {run_dir}"),
        ):
            save_run_checkpoint(state, run_dir, keep=args.keep_checkpoints)

    def end_profile(stage: Callable[[], None], *, failed: bool) -> None:
        """Write the trace, holding back a failure to write it (Q-144).

        Q-142: a trace that can't be written is exit 2, reported after the checkpoint is
        saved; Q-160: after a training failure, only a warning.
        """
        nonlocal trace_failure
        try:
            with _output(f"the profile trace {profile.path if profile else ''}"):
                stage()
        except _Failure as failure:
            if failed:
                print(f"warning: {failure}", file=sys.stderr)
            elif trace_failure is None:
                trace_failure = failure

    def record(_: Trainer, log: StepLog) -> bool:
        nonlocal log_failure
        if profile is not None:
            end_profile(lambda: profile.step_done(log.iteration), failed=False)
        if log.mean_losses is not None:  # Q-132: read every log_every iterations, no extra sync
            row = {
                "iteration": log.iteration,
                "losses": log.mean_losses,
                "num_gaussians": log.num_gaussians,
            }
            try:
                with _output(str(log_path)), log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            except _Failure as failure:  # Q-159 (as answered): stop here, as a signal does
                log_failure = failure
                return True
            total = log.mean_losses["total"]  # E9; D-77 reports the terms beside it
            print(
                f"iteration {log.iteration}/{last}  loss {total:.6g}  N {log.num_gaussians}",
                file=sys.stderr,
            )
        every = args.checkpoint_every
        if every and log.iteration % every == 0 and log.iteration < last:  # D-83
            save()
        return signals.received is not None

    if profile is not None:
        profile.begin_if_due(trainer.iteration)
    # Q-133: the guard stays on through the save, so a signal can't interrupt it.
    with _Signals() as signals:
        failed = True
        try:
            trainer.train(callbacks=[record])
            failed = False
        except FloatingPointError as error:  # Q-69: divergence; nothing is saved
            _fail(str(error), _NUMERIC)
        except torch.OutOfMemoryError:  # Q-142: the step in progress is lost; nothing is saved
            _fail(_out_of_memory(trainer.iteration, run_dir, args.resume))
        finally:
            if profile is not None:  # Q-144: a window cut short is still written; Q-160
                end_profile(profile.finish, failed=failed or log_failure is not None)
        try:
            save()
        except _Failure as failure:
            if trace_failure is not None:
                print(f"warning: {trace_failure}", file=sys.stderr)
            if log_failure is None:
                raise
            # Q-159 (as answered): the log's failure is the error, the save's a warning after
            # it; a diverged state (Q-133) is exit 3 still, the log's failure its warning.
            if failure.code == _NUMERIC:
                failure.add_note(str(log_failure))
                raise
            log_failure.add_note(str(failure))
            raise log_failure from failure
    stopped = signals.received is not None and trainer.iteration < last  # Q-133
    # Q-159 (as answered): not a finished run, so no evaluation; Q-164: a signal's exit code
    # stands over it, as over a trace failure (Q-160).
    if log_failure is not None and not stopped:
        if trace_failure is not None:  # Q-160, Q-164: after a training failure, a warning
            print(f"warning: {trace_failure}", file=sys.stderr)
        _fail(
            f"{log_failure}; training stopped at iteration {trainer.iteration}, "
            f"checkpoint in {run_dir / 'checkpoint'}"
        )
    if stopped and signals.received is not None:  # Q-133; a finished run goes on
        for warned in (trace_failure, log_failure):  # the signal's exit code stands (Q-133)
            if warned is not None:
                print(f"warning: {warned}", file=sys.stderr)
        print(
            f"stopped by signal {signals.received} at iteration {trainer.iteration}; "
            f"checkpoint in {run_dir / 'checkpoint'}",
            file=sys.stderr,
        )
        return 128 + signals.received
    if trace_failure is not None:
        raise trace_failure
    if has_test:  # §1.3 calendar: the final evaluation, after the checkpoint (Q-132)
        # Q-161: one call downloads the networks and writes the file, so the message names both.
        with _environment(f"LPIPS networks or {run_dir / 'METRICS.json'}"):
            try:
                trainer.evaluate("test", out_dir=run_dir)
            except FloatingPointError as error:  # Q-129, Q-147: a render isn't finite; no file
                _fail(str(error), _NUMERIC)
    return 0


def _check_finite(trainer: Trainer, stats: Mapping[str, Any]) -> None:
    """Refuse a model or statistics that aren't finite, as exit code 3 (Q-133).

    The trainer checks for divergence at log reads and events (Q-69), so a model that turned
    non-finite since then is caught here, and a diverged state is never saved.
    """
    import torch

    try:
        trainer.model.validate()  # Q-55: §1.1's rules, finite values among them
    except ValueError as error:
        _fail(f"training diverged by iteration {trainer.iteration}: {error}", _NUMERIC)
    if not all(bool(torch.isfinite(v).all()) for v in stats.values()):
        _fail(f"training diverged by iteration {trainer.iteration}: statistics", _NUMERIC)


def _out_of_memory(completed: int, run_dir: Path, resumed: Path | None) -> str:
    """Describe what an out-of-memory failure in training lost, and the way on.

    Q-142 (as answered): `completed` is the trainer's iteration, the step in progress being
    lost.
    """
    latest = run_dir / "checkpoint"
    if latest.is_dir():
        where = f"resume from {latest}"
    elif resumed is not None:
        where = f"resume from {resumed}"
    else:
        where = "no checkpoint was saved; start again"
    return (
        f"out of GPU memory after iteration {completed}, the step in progress unsaved; {where}"
        ' with perf.frame_cache = "mmap" or lower memory settings (D-51, Q-99)'
    )


# -- models --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Loaded:
    model: FreeTimeGaussians
    iteration: int | None  # a checkpoint's
    # D-84: a checkpoint's configuration's background, or the one an exchange file records
    background: tuple[float, float, float]
    # Q-136 (as answered): an exchange model's file, or a checkpoint's manifest.json, which
    # holds every other file's SHA-256 (§2.2, Q-25); only when asked for
    sha256: str | None
    # Q-174: a checkpoint's Δt_f in normalized time (E14), the temporal index's bucket width;
    # an exchange model carries none
    frame_dt: float | None = None


def _model_sha256(path: Path) -> str:
    return _sha256(path if not path.is_dir() else path / "manifest.json")


def _load_model(path: Path, device: str, *, hashed: bool = False) -> _Loaded:
    """Read MODEL of §2.1, a checkpoint directory or an FTGS-IF file.

    With `hashed`, also the SHA-256 that names it (Q-136), taken before and after reading, so
    that it names the bytes read: a model that changed meanwhile, as a periodic save (D-83)
    changes RUN_DIR/checkpoint, is exit 2 (Q-154). So is a model whose second hash can't be
    taken, and a read that fails while the hash changes, as when a save tears the read; a read
    that fails with the hash unchanged keeps the reader's error (Q-154 as answered).
    """
    changed_message = f"model {path} changed while it was read; evaluate it again"
    with _as_input_error(f"model {path}", OSError):  # Q-142, Q-154
        try:  # a mistyped checkpoint directory isn't an unreadable FTGS-IF file
            path.stat()
        except FileNotFoundError:
            _fail(
                f"model {path}: No such file or directory; MODEL is a checkpoint directory or "
                "an FTGS-IF file"
            )
        before = _model_sha256(path) if hashed else None
        try:
            loaded = _read_model(path, device)
        except (ValueError, OSError):
            if hashed and _model_sha256_or_none(path) != before:
                _fail(changed_message)
            raise
        if hashed and _model_sha256_or_none(path) != before:
            _fail(changed_message)
    return dataclasses.replace(loaded, sha256=before)


def _model_sha256_or_none(path: Path) -> str | None:
    """Return the model's SHA-256, or None when it is gone or unreadable (Q-154).

    Q-154 (as answered): gone or unreadable as within a save.
    """
    try:
        return _model_sha256(path)
    except OSError:
        return None


def _read_model(path: Path, device: str) -> _Loaded:
    from openftgs.config import config_from_dict
    from openftgs.io import load_checkpoint, read_exchange
    from openftgs.model import FreeTimeGaussians

    if not path.is_dir():
        exchange = read_exchange(path)
        return _Loaded(
            FreeTimeGaussians.from_exchange(exchange, device=device),
            None,
            exchange.background,  # D-84
            None,
        )
    state = load_checkpoint(path)
    cfg = config_from_dict(state.config)
    model = FreeTimeGaussians(
        {k: v.to(device) for k, v in state.params.items()},
        sh_degree=cfg.sh_degree,
        time=state.time,
        time_unit=state.time_unit,
    )
    return _Loaded(model, state.iteration, cfg.background, None, frame_dt=state.frame_dt)


def _render_device(backend: str) -> str:
    if backend == "reference":
        return "cpu"  # Q-125
    if _device() != "cuda":
        _fail("the gsplat backend needs a CUDA device; pass --backend reference")
    return "cuda"


# -- render --------------------------------------------------------------------------------------


def _render(args: argparse.Namespace) -> int:
    device = _render_device(args.backend)
    return _render_loaded(
        _load_model(args.model, device), args.request, args.out, args.backend, device
    )


def _render_loaded(
    loaded: _Loaded, request_path: Path, out: Path, backend: str, device: str
) -> int:
    """Run `ftgs render`'s body once MODEL is read; also `ftgs-adapter render`'s (Q-202)."""
    import torch

    from openftgs.io import read_render_request, write_renders
    from openftgs.render import render_request

    with _as_input_error(f"render request {request_path}", OSError):
        sha = _sha256(request_path)
        request = read_render_request(request_path)
    if request.time_unit != loaded.model.time_unit:
        _fail(
            f"the request's time unit {request.time_unit!r} isn't the model's "
            f"{loaded.model.time_unit!r}"
        )
    # Q-20: at_time refuses L′ > L, which is exit 2 (Q-26, Q-142); checked before rendering
    if request.sh_degree is not None and request.sh_degree > loaded.model.sh_degree:
        _fail(
            f"the request's sh_degree {request.sh_degree} is above the model's "
            f"{loaded.model.sh_degree}"
        )
    renders = render_request(
        loaded.model,
        request,
        backend=backend,  # type: ignore[arg-type]
        frame_dt=loaded.frame_dt,  # Q-174: a checkpoint's Δt_f is the index's bucket width
        devices=_shard_devices(device),  # Q-175, Q-186
    )
    bad = [i for i, rgb in renders.rgb.items() if not bool(torch.isfinite(rgb).all())]
    bad += [i for i, a in renders.alpha.items() if not bool(torch.isfinite(a).all())]
    if bad:  # Q-129
        _fail(f"renders that aren't finite: {sorted(set(bad))[:5]}", _NUMERIC)
    with _output(f"renders {out}"):  # Q-142
        out.parent.mkdir(parents=True, exist_ok=True)
        write_renders(renders, out, producer=_producer(), request_sha256=sha)
    return 0


# -- eval ----------------------------------------------------------------------------------------


def _match_renders(
    video: MultiViewVideo, split: str, directory: Path
) -> tuple[dict[int, Any], dict[str, str]]:
    """Match each record of the split to exactly one item of DIR's request (Q-137).

    The match is by timestamp, image size and camera; each record gets that item's render.
    """
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
    loaded = _load_model(args.model, device, hashed=True)  # Q-136
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
    try:
        images = evaluate_images(
            loaded.model,
            video,
            evaluator,
            split=args.split,
            backend=_backend(device),
            background=loaded.background,
            renders=supplied,
            frame_dt=loaded.frame_dt,  # Q-174, Q-183
            devices=_shard_devices(device),  # Q-175, Q-186
        )
    except FloatingPointError as error:  # Q-129, Q-147: a render isn't finite; nothing is written
        _fail(str(error), _NUMERIC)
    body = metrics_report(images, split=args.split, metadata=video.manifest.metadata)
    with _output(f"metrics {args.out}"):  # Q-142
        write_metrics(
            args.out,
            body,
            manifest=video.manifest,
            model_path=args.model,
            iteration=loaded.iteration,
            renders=entry,
            model_sha256=loaded.sha256,
        )
    return 0


# -- export --------------------------------------------------------------------------------------


def _export(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_dir():
        _fail(f"{args.checkpoint} is not a checkpoint directory")
    return _export_loaded(_load_model(args.checkpoint, "cpu"), args.out)


def _export_loaded(loaded: _Loaded, out: Path) -> int:
    """Run `ftgs export`'s body once the model is read; also `ftgs-adapter export`'s (Q-202)."""
    from openftgs.io import write_exchange

    # D-84: the file records the background the run was trained against.
    exchange = dataclasses.replace(loaded.model.to_exchange(), background=loaded.background)
    with _output(f"exchange model {out}"), _as_input_error(f"exchange model {out}"):
        out.parent.mkdir(parents=True, exist_ok=True)  # Q-142
        write_exchange(exchange, out)
    return 0


# -- parser --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftgs",
        description="openftgs, an independent clean-room implementation of FreeTimeGS "
        "(Wang et al., CVPR 2025), not affiliated with its authors.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # Q-242: the help text, written from §2.1's CLI block, the paragraph after it and the module
    # docs; it changes no default, type, choice or requirement. A literal percent sign is "%%".
    p = commands.add_parser("convert", help="convert a dataset into a scene manifest")
    p.add_argument(
        "kind",
        choices=tuple(_CONVERT_OPTIONS),
        help="the converter: n3dv (Neural3DV), opencv-yaml (OpenCV-style YAML calibrations, "
        "such as ENeRF-Outdoor) or selfcap (SelfCap's archives)",
    )
    p.add_argument("src", type=Path, metavar="SRC", help="the dataset's scene folder")
    p.add_argument(
        "out",
        type=Path,
        metavar="OUT",
        help="the output folder, absent or empty; it receives the frames and OUT/scene.json, "
        "whose path is printed",
    )
    p.add_argument(
        "--scene",
        metavar="NAME",
        help="selfcap, required: the scene; a scene outside the dataset card's table needs "
        "--window and --test-camera",
    )
    p.add_argument("--fps", type=float, metavar="F", help="opencv-yaml, required: frame rate")
    p.add_argument(
        "--frames",
        type=int,
        metavar="N",
        help="n3dv and opencv-yaml: the first N frames of each camera (n3dv: 300 by default; "
        "required for opencv-yaml)",
    )
    p.add_argument(
        "--resize",
        type=float,
        metavar="R",
        help="the resize ratio of frames and intrinsics (n3dv: 0.5 by default; required for "
        "opencv-yaml; selfcap: the card's ratio, or 0.5 for other scenes)",
    )
    p.add_argument("--ffmpeg", metavar="PATH", help="the ffmpeg program (default: on PATH)")
    p.add_argument(
        "--ffprobe",
        metavar="PATH",
        help="the ffprobe program (default: beside a given --ffmpeg, else on PATH)",
    )
    p.add_argument(
        "--hwaccel",
        choices=("none", "cuda"),
        help="ffmpeg's video decoding: none (the default) or cuda, which forces hardware frames",
    )
    p.add_argument(
        "--jobs",
        type=int,
        metavar="J",
        help="cameras decoded at once (0, the default: one per four cores); outputs don't "
        "depend on it",
    )
    p.add_argument(
        "--test-camera",
        dest="test_camera",
        metavar="NAME",
        help="opencv-yaml and selfcap: the test view (opencv-yaml: none by default, so every "
        "image trains; selfcap: the card's)",
    )
    p.add_argument(
        "--window",
        type=_pair,
        metavar="N0:N1",
        help="selfcap: frames N0 <= n < N1 of every camera (default: the card's)",
    )
    p.add_argument(
        "--undistort",
        choices=("colmap", "internal"),
        help="selfcap: undistort with COLMAP (the default) or with the package's own code",
    )
    p.add_argument(
        "--colmap", metavar="PATH", help="selfcap: the COLMAP program (default: on PATH)"
    )
    p.add_argument(
        "--no-sync",
        dest="use_sync",
        action="store_const",
        const=False,
        help="selfcap: ignore the cameras' sync offsets, so frame n is at n/60 s",
    )
    p.add_argument(
        "--masks",
        type=Path,
        metavar="DIR",
        help="selfcap: one mask per test image, DIR/<n:06d>.png, with DIR/masks.json giving "
        "their source",
    )
    p.add_argument(
        "--calib",
        type=Path,
        metavar="DIR",
        help="selfcap: a calibration folder that overrides the archives' calibration",
    )
    p.set_defaults(run=_convert)

    p = commands.add_parser("cache", help="build the frame cache of a scene")
    p.add_argument("scene", type=Path, metavar="SCENE.json", help="the scene manifest")
    p.add_argument(
        "--placement",
        choices=("auto", "gpu", "mmap"),
        default="auto",
        help="auto (the default) only builds the cache; gpu or mmap also opens it in that "
        "placement, which shows that it fits",
    )
    p.set_defaults(run=_cache)

    p = commands.add_parser(
        "init",
        help="initial points from a scene",
        description="Initial points for training. D-24's check of F·B against N_max happens "
        "when `ftgs train` builds the model, where it is exit code 2 (Q-128).",
    )
    p.add_argument("method", choices=("roma",), help="the method: roma (needs a CUDA device)")
    p.add_argument("scene", type=Path, metavar="SCENE.json", help="the scene manifest")
    p.add_argument(
        "out", type=Path, metavar="OUT.safetensors", help="the initial points file to write"
    )
    p.add_argument(
        "--stride",
        type=int,
        metavar="K",
        help="init frames at every K-th frame of the sequence (default: 1, every frame)",
    )
    p.add_argument(
        "--points-per-frame",
        dest="points_per_frame",
        type=int,
        metavar="B",
        help="the point budget B per frame of the sequence; an init frame keeps at most K·B "
        "(default: 1000)",
    )
    p.add_argument("--seed", type=int, metavar="S", help="the random seed (default: 0)")
    p.add_argument(
        "--model",
        choices=("roma_outdoor", "tiny_roma_v1_outdoor"),
        help="RoMa's model (default: roma_outdoor)",
    )
    p.add_argument(
        "--no-upsample",
        dest="no_upsample",
        action="store_true",
        help="match at RoMa's coarse resolution only",
    )
    p.set_defaults(run=_init)

    p = commands.add_parser("train", help="train a model")
    p.add_argument("scene", type=Path, metavar="SCENE.json", help="the scene manifest")
    p.add_argument(
        "--init",
        type=Path,
        required=True,
        metavar="INIT.safetensors",
        help="the initial points, from ftgs init",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="RUN_DIR",
        help="the run directory, absent or empty for a new run; it receives config.json, "
        "train_log.jsonl, the checkpoints and, when the scene has a test split, METRICS.json",
    )
    p.add_argument(
        "--config",
        type=Path,
        metavar="CFG.json",
        help="a TrainConfig as JSON (default: TrainConfig's defaults)",
    )
    p.add_argument("--seed", type=int, metavar="S", help="the seed, replacing the configuration's")
    p.add_argument(
        "--resume",
        type=Path,
        metavar="CKPT_DIR",
        help="continue from a checkpoint, on its own configuration",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="with --resume: resume despite a changed scene manifest",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help="also save a checkpoint after every N-th iteration (0, the default, saves only at "
        "the end and on SIGINT or SIGTERM)",
    )  # D-83
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=1,
        metavar="K",
        help="keep the K latest saves: RUN_DIR/checkpoint and RUN_DIR/checkpoint-<iteration> "
        "(default: 1)",
    )  # D-83
    p.add_argument(
        "--profile",
        type=_pair,
        metavar="START:COUNT",
        help="write a torch.profiler trace of COUNT iterations from START to "
        "RUN_DIR/profile/trace.json",
    )
    p.set_defaults(run=_train)

    p = commands.add_parser("render", help="answer a render request")
    p.add_argument(
        "model", type=Path, metavar="MODEL", help="a checkpoint directory or an FTGS-IF file"
    )
    p.add_argument(
        "--request", type=Path, required=True, metavar="REQUEST.json", help="the render request"
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="RENDERS.safetensors",
        help="the renders file to write",
    )
    p.add_argument(
        "--backend",
        choices=("gsplat", "reference"),
        default="gsplat",
        help="gsplat (the default, needs a CUDA device) or reference, which renders on the CPU",
    )
    p.set_defaults(run=_render)

    p = commands.add_parser("eval", help="evaluate a model on a scene")
    p.add_argument(
        "model", type=Path, metavar="MODEL", help="a checkpoint directory or an FTGS-IF file"
    )
    p.add_argument("scene", type=Path, metavar="SCENE.json", help="the scene manifest")
    p.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="the split to evaluate (default: test)",
    )  # Q-143
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="METRICS.json",
        help="the metrics file to write, per image and per scene",
    )
    p.add_argument(
        "--renders-dir",
        dest="renders_dir",
        type=Path,
        metavar="DIR",
        help="score the renders in DIR (request.json and renders.safetensors) instead of "
        "rendering the model",
    )
    p.set_defaults(run=_eval)

    p = commands.add_parser("export", help="export a checkpoint as an exchange model")
    p.add_argument("checkpoint", type=Path, metavar="CKPT_DIR", help="the checkpoint directory")
    p.add_argument(
        "--format", choices=("ftgs-if",), required=True, help="the exchange format: ftgs-if"
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="MODEL.safetensors",
        help="the exchange model file to write",
    )
    p.set_defaults(run=_export)

    from openftgs.cli import _bench  # the package is loaded by now; the module imports from it

    _bench.add_parser(commands)
    return parser


# Q-176 (as answered, #274): the cores of the affinity set, which PyTorch's own default follows.
def _physical_cores() -> int:
    """Count the physical cores of the CPUs this process may run on (Q-176).

    The distinct (package, core) pairs, in Linux's CPU topology, of the CPUs this process may
    run on; the number of those CPUs where the topology can't be read, and the logical CPUs
    where Python has no `os.sched_getaffinity` (macOS, Windows).
    """
    if not hasattr(os, "sched_getaffinity"):
        return os.cpu_count() or 1
    cpus = os.sched_getaffinity(0)
    cores: set[tuple[str, str]] = set()
    try:
        for cpu in cpus:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = (topology / "physical_package_id").read_text().strip()
            cores.add((package, (topology / "core_id").read_text().strip()))
    except OSError:
        return len(cpus)
    return len(cores)


# Q-176: §2.4 "Threads and processes", set at startup by the entry point (as Q-131 sets
# CUBLAS_WORKSPACE_CONFIG); the library leaves the setting alone. After parsing, so --help stays
# fast; PyTorch has already taken either variable, when set, at its own start.
def _set_intra_op_threads() -> None:
    if "OMP_NUM_THREADS" in os.environ or "MKL_NUM_THREADS" in os.environ:
        return
    import torch

    torch.set_num_threads(_physical_cores())


# Q-124: the module's one public name; the ftgs console script and `python -m openftgs.cli`.
def main(argv: Sequence[str] | None = None) -> int:
    """Run ``ftgs`` with ``argv`` and return its exit code (§2.1).

    Parse the arguments with the parser of §2.1's CLI block, set PyTorch's intra-op threads to
    the physical cores, unless ``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS`` is set, then run the
    command and map its failures to §2.1's exit codes. ``--help`` and a usage error run no
    command and change nothing.

    Parameters
    ----------
    argv
        The arguments after the program's name, as ``["train", "scene.json", "--init",
        "init.safetensors", "--out", "run"]``; None, the default, reads the process's
        arguments, ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code: 0 success (``--help`` included), 2 invalid input or environment (a
        usage error included), 3 a numerical failure, and 128 plus the signal's number when
        SIGINT or SIGTERM stops ``ftgs train`` after it saved its checkpoint.

    Notes
    -----
    Q-124: the module's one public name, run by the ``ftgs`` console script and by
    ``python -m openftgs.cli``. A failure the CLI reports is one line on stderr,
    ``ftgs <command>: error: <message>``, followed by a warning line for each failure that
    came after it (Q-159); any other exception is a bug and keeps its traceback (Q-26,
    Q-142). Signals: Q-133. Threads: §2.4, Q-176.

    Examples
    --------
    Train on a converted scene from Python, as ``ftgs train`` would:

    >>> from openftgs.cli import main
    >>> main(["train", "scene/scene.json", "--init", "init.safetensors", "--out", "run"])
    0
    """
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:  # argparse: --help is 0, a usage error 2 (§2.1)
        return stop.code if isinstance(stop.code, int) else _INPUT
    _set_intra_op_threads()
    try:
        code: int = args.run(args)
    except _Failure as failure:
        print(f"ftgs {args.command}: error: {failure}", file=sys.stderr)
        for note in getattr(failure, "__notes__", ()):  # Q-159: failures that followed it
            print(f"warning: {note}", file=sys.stderr)
        return failure.code
    return code
