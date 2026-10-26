# SPDX-License-Identifier: Apache-2.0
"""Trainer: the iteration loop, the sampler, the schedules, and the evaluation and checkpoint
hooks.

Spec: §1.3, §2.2 (Full checkpoint), §2.4 rule 3, D-26, D-45. Band: Orchestration (§2.1).
"""

import contextlib
import dataclasses
import hashlib
import math
import random
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from openftgs._stages import point, stage
from openftgs.config import (
    LossConfig,
    PerfConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
    resume_conflicts,
)
from openftgs.data import FrameCache, MultiViewVideo, Prefetcher
from openftgs.density import (
    OPACITY_RESET_EVERY,
    GradStats,
    GrowthReport,
    RelocationReport,
    grow,
    prune,
    relocate,
    reset_opacity,
)
from openftgs.eval import (
    Evaluator,
    average_metrics,
    evaluate_images,
    metrics_report,
    write_metrics,
)
from openftgs.io import load_checkpoint
from openftgs.losses import LPIPSLoss, PerceptualLoss, total_loss
from openftgs.model import FreeTimeGaussians
from openftgs.optim import FTGSOptimizer, build_optimizer
from openftgs.render import render
from openftgs.types import Camera, InitPoints, TimeNormalizer, TrainerState

__all__ = ["Callback", "CalendarEntry", "StepLog", "Trainer", "calendar_entry"]  # Q-69


def _check_int(value: object, low: int, high: int | None, what: str) -> int:
    """An int (not a bool) in [low, high], or at least low when high is None."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < low
        or (high is not None and value > high)
    ):
        bounds = f"in [{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{what} must be an int {bounds}, got {value!r}")
    return value


# The calendar (§1.3 Iteration calendar).


# Q-71: what the calendar holds for one iteration.
@dataclass(frozen=True)
class CalendarEntry:
    sh_degree: int  # L_active (E13)
    grow: bool  # a growth event (D-18)
    relocate: bool  # a relocation event (D-17)
    prune: bool  # the ablation's pruning (§1.3 Count-control modes)
    reset_opacity: bool  # the ablation's opacity reset (§1.3 Count-control modes)

    @property
    def event(self) -> bool:
        """An event whose gradient statistics are reset afterwards (§1.3 step 7, D-45)."""
        return self.grow or self.relocate or self.prune


def calendar_entry(cfg: TrainConfig, it: int) -> CalendarEntry:
    """The calendar of `cfg` at iteration `it` (§1.3 step 7 and Iteration calendar, E13): the
    active SH degree and the events, every window and period read from the configuration."""
    _check_int(it, 1, cfg.optim.iterations, "it")  # D-41
    growth, reloc = cfg.growth, cfg.reloc
    on_period = it % reloc.every == 0  # growth events share the relocation period (§2.1)
    in_growth = growth.start <= it <= growth.stop
    ablation = reloc.prune_opacity is not None  # Q-71: the ablation's densification
    return CalendarEntry(
        sh_degree=min((it - 1) // cfg.optim.sh_interval, cfg.sh_degree),  # E13
        grow=on_period and in_growth and growth.mode != "none",
        relocate=on_period and reloc.enabled and reloc.start <= it <= reloc.stop,
        prune=on_period and in_growth and ablation,
        # Q-71 (owner answer): inside the growth window only; Q-67: the period from density.
        reset_opacity=ablation and in_growth and it % OPACITY_RESET_EVERY == 0,
    )


# What a training iteration reports (Q-69).


@dataclass(frozen=True)
class StepLog:
    iteration: int  # the iteration just completed
    image_index: int  # into the dataset's records
    timestamp: float  # the image's raw τ
    t: float  # its normalized time (E14)
    sh_degree: int  # L_active (E13)
    rates: dict[str, float]  # the learning rates of this iteration (E12, E13)
    losses: dict[str, Tensor]  # total_loss's detached 0-d terms, on the device, unread (D-77)
    num_gaussians: int  # N after the iteration
    growth: GrowthReport | None = None
    relocation: RelocationReport | None = None
    pruned: int | None = None
    opacity_reset: bool = False
    # Each term's mean over the iterations since the last read, read when the iteration is a
    # multiple of PerfConfig.log_every (§2.4 rule 3); None otherwise.
    mean_losses: dict[str, float] | None = None


Callback = Callable[["Trainer", StepLog], bool | None]  # Q-69: True stops train()


# Generators and the sampler (D-26; Q-72).


def _seeded(seed: int, role: str) -> torch.Generator:
    """A CPU generator for one role, seeded from TrainConfig.seed (Q-72)."""
    digest = hashlib.sha256(f"openftgs/{role}/{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def _restored_generator(state: object, role: str) -> torch.Generator:
    if not isinstance(state, Tensor) or state.dtype != torch.uint8:
        raise ValueError(f"rng[{role!r}] must be a uint8 generator state")
    generator = torch.Generator()
    try:
        generator.set_state(state)
    except RuntimeError as err:
        raise ValueError(f"rng[{role!r}] is not a CPU generator state: {err}") from None
    return generator


class _Sampler:
    """D-26: the seeded permutation of the training images, redrawn each epoch, as dataset
    record indices; the next permutation is drawn when the current one runs out (Q-72)."""

    def __init__(
        self,
        indices: Sequence[int],
        generator: torch.Generator,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self._indices = list(indices)
        self.generator = generator
        if state is None:
            self.epoch, self.cursor, self.permutation = 0, 0, self._draw()
            return
        permutation = state.get("permutation")
        if not isinstance(permutation, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in permutation
        ):
            raise ValueError("the sampler's permutation must be a list of record indices")
        if sorted(permutation) != sorted(self._indices):
            raise ValueError(
                "the sampler's permutation must permute this dataset's training images"
            )
        self.epoch = _check_int(state.get("epoch"), 0, None, "the sampler's epoch")
        self.cursor = _check_int(state.get("cursor"), 0, len(permutation), "the sampler's cursor")
        self.permutation = list(permutation)

    def _draw(self) -> list[int]:
        order = torch.randperm(len(self._indices), generator=self.generator).tolist()
        return [self._indices[k] for k in order]

    def next(self) -> int:
        if self.cursor == len(self.permutation):
            self.epoch, self.cursor, self.permutation = self.epoch + 1, 0, self._draw()
        index = self.permutation[self.cursor]
        self.cursor += 1
        return index

    def peek(self, count: int) -> list[int]:
        """Up to `count` upcoming indices: the rest of this permutation and the next epoch's,
        drawn on a copy of the generator's state, so the stream doesn't move (Q-72 note 3,
        Q-99)."""
        upcoming = self.permutation[self.cursor : self.cursor + count]
        if len(upcoming) < count:
            clone = torch.Generator()
            clone.set_state(self.generator.get_state())
            order = torch.randperm(len(self._indices), generator=clone).tolist()
            upcoming += [self._indices[k] for k in order][: count - len(upcoming)]
        return upcoming

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "cursor": self.cursor, "permutation": list(self.permutation)}


def _global_rng() -> dict[str, Any]:
    """The global generators, recorded for code outside the trainer (Q-72); CUDA states only
    when CUDA is already initialized, so a CPU run never initializes it."""
    cuda: dict[str, Tensor] = {}
    if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        cuda = {str(i): state for i, state in enumerate(torch.cuda.get_rng_state_all())}
    version, internal, gauss = random.getstate()
    numpy_state = np.random.get_state(legacy=False)
    assert isinstance(numpy_state, dict)
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda,
        "python": [version, list(internal), gauss],
        "numpy": {
            "bit_generator": numpy_state["bit_generator"],
            "state": {
                "key": numpy_state["state"]["key"].tolist(),
                "pos": int(numpy_state["state"]["pos"]),
            },
            "has_gauss": int(numpy_state["has_gauss"]),
            "gauss": float(numpy_state["gauss"]),
        },
    }


def _global_rng_restorer(rng: Mapping[str, Any]) -> Callable[[], None]:
    """Q-72: the recorded global generator states, checked on throwaway generators so that a
    malformed state is a ValueError before anything changes, and the function that restores
    them: CUDA states only when CUDA is initialized, for recorded devices that exist."""
    try:
        cpu = rng["torch_cpu"]
        torch.Generator().set_state(cpu)
        version, internal, gauss = rng["python"]
        python_state = (version, tuple(internal), gauss)
        random.Random().setstate(python_state)
        numpy = rng["numpy"]
        numpy_state = {
            "bit_generator": numpy["bit_generator"],
            "state": {
                "key": np.array(numpy["state"]["key"], dtype=np.uint32),
                "pos": numpy["state"]["pos"],
            },
            "has_gauss": numpy["has_gauss"],
            "gauss": numpy["gauss"],
        }
        np.random.RandomState().set_state(numpy_state)
        cuda = {int(index): state for index, state in rng.get("torch_cuda", {}).items()}
    except (AttributeError, KeyError, OverflowError, RuntimeError, TypeError, ValueError) as err:
        raise ValueError(f"malformed global generator states in rng: {err!r}") from None
    initialized = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    for index, state in cuda.items():
        if not isinstance(state, Tensor) or state.dtype != torch.uint8:
            raise ValueError(f"rng['torch_cuda'][{index}] must be a uint8 generator state")
        if initialized and index < torch.cuda.device_count():
            if state.numel() != torch.cuda.get_rng_state(index).numel():
                raise ValueError(f"rng['torch_cuda'][{index}] has the wrong size")

    def restore() -> None:
        torch.set_rng_state(cpu)
        if initialized:
            for index, state in cuda.items():
                if index < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, device=index)
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    return restore


# Global PyTorch flags around a training iteration (Q-73).

# Q-173 (changed, #326), Q-223: cuDNN autotuning during a training iteration that isn't
# deterministic (§2.4 rule 2, D-52). B-4 measured it alone at +4.3 % and −1.0 % (#228); with the
# perceptual network's NCHW inputs (_NCHWLPIPSLoss) at +7.2 % and +3.9 % (Calibration data).
_AUTOTUNE = True


class _PlacementWait:
    """Q-225 (changed, #341), Q-232 (changed, #348): when "auto" decides on CUDA. An iteration
    at an image size not yet seen in the process may autotune cuDNN, whose trials return the
    allocator's blocks to the driver, so the reserve read after it is close to nothing. "auto"
    therefore decides after the iteration that follows the first iteration at the last of the
    training images' sizes to appear: after the second when they all have one size."""

    def __init__(self, dataset: MultiViewVideo) -> None:
        self._sizes = [(r.camera.width, r.camera.height) for r in dataset.records]
        self._unseen = {
            size for size, r in zip(self._sizes, dataset.records, strict=True) if r.split == "train"
        }

    def decides_after(self, index: int) -> bool:
        """Records that an iteration trained on record `index`, and says whether "auto" decides
        after it; called once an iteration, in order."""
        if not self._unseen:
            return True
        self._unseen.discard(self._sizes[index])
        return False


@contextlib.contextmanager
def _perf_flags(perf: PerfConfig) -> Iterator[None]:
    """Q-73: PerfConfig's process-global flags for one training iteration, restored
    afterwards."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)
    algorithms = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    matmul = torch.get_float32_matmul_precision()
    try:
        # D-52; Q-172: under "bf16" the other convolutions (the SSIM fallback's) keep TF32.
        cudnn.allow_tf32 = perf.perc_precision != "ieee"
        torch.set_float32_matmul_precision("highest")  # D-52: no TF32 in matrix products
        if perf.deterministic:
            # §2.4 Settings: autotuning off, deterministic kernels where PyTorch has them.
            # fused-ssim stays: its backward writes each gradient element once (source log #73).
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif _AUTOTUNE:
            cudnn.benchmark = True  # Q-173; otherwise the flag stays the caller's (Q-73)
        yield
    finally:
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = saved
        torch.use_deterministic_algorithms(algorithms[0], warn_only=algorithms[1])
        torch.set_float32_matmul_precision(matmul)


# Q-173: the training perceptual network's weights in channels-last memory, turned on only when
# B-4 shows a gain and B-1 is no worse (§2.4 rule 2, D-52). Off: B-4 measured +1.0 % and +2.9 %
# at its two sizes (Calibration data, #228), with the inputs channels-last as `LPIPSLoss` views
# a render's [..., H, W, 3] batch first; since #326 mostly NCHW (_lpips_network, Q-223). D-89's
# networks are measured by B-4's rerun (Q-251), which #394 reads.
_CHANNELS_LAST_WEIGHTS = False


# Q-173 (changed, #326), Q-223: B-4 measured NCHW inputs the fastest layout for LPIPS-VGG in IEEE
# float32, and in TF32 with autotuning; slower in TF32 without it and in bfloat16, which keep the
# plain view. LPIPS's backbones only, until #392 removes them (#380's expand step).
class _NCHWLPIPSLoss(LPIPSLoss):
    """LPIPS with its two images copied so that the network gets them as contiguous NCHW
    tensors: the trainer's network under `perc_precision` "ieee", and "tf32" with autotuning.
    The copy is exact (§2.4 rule 2); the evaluator's networks stay plain."""

    def forward(self, pred: Tensor, gt: Tensor, *, map_inputs: bool = True) -> Tensor:
        # [..., H, W, 3] in memory whose batch-first view, `LPIPSLoss`'s, is contiguous NCHW
        pred, gt = (image.movedim(-1, -3).contiguous().movedim(-3, -1) for image in (pred, gt))
        return super().forward(pred, gt, map_inputs=map_inputs)


# Q-172: autocast is local to the perceptual network, which only the trainer builds this way.
class _Bf16LPIPSLoss(LPIPSLoss):
    """D-52's approximate option, `perc_precision = "bf16"`: LPIPS with its network run under
    bfloat16 autocast on the images' device, returning float32. The evaluator's networks stay
    float32 (#228)."""

    def forward(self, pred: Tensor, gt: Tensor, *, map_inputs: bool = True) -> Tensor:
        # Q-191: on the CPU too; the network's own reductions as autocast leaves them.
        with torch.autocast(pred.device.type, dtype=torch.bfloat16):
            out = super().forward(pred, gt, map_inputs=map_inputs)
        return out.float()


# Q-172, Q-250, Q-251: the bfloat16 path generalized to D-89's trunk networks.
class _Bf16PerceptualLoss(PerceptualLoss):
    """D-52's approximate option, `perc_precision = "bf16"`, for a trunk network: the network
    runs under bfloat16 autocast on the images' device, the CPU included (Q-191), and returns
    float32. The images enter as float32, as the networks require (Q-263, Q-265), and the
    module keeps its float32 weights and constants; only the operations autocast covers run
    in bfloat16. The evaluator's networks stay float32."""

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        with torch.autocast(pred.device.type, dtype=torch.bfloat16):
            out = super().forward(pred, gt)
        return out.float()


def _lpips_network(perf: PerfConfig) -> type[LPIPSLoss]:
    """The trainer's LPIPS network for `perf`: bf16's (Q-172), NCHW inputs where B-4 measured
    them faster (Q-223), else plain `LPIPSLoss` with its channels-last view."""
    if perf.perc_precision == "bf16":
        return _Bf16LPIPSLoss
    autotuned = _AUTOTUNE and not perf.deterministic
    if perf.perc_precision == "ieee" or autotuned:
        return _NCHWLPIPSLoss
    return LPIPSLoss


def _builds_perceptual(loss: LossConfig) -> bool:
    """D-77, Q-69, Q-249: the trainer builds its perceptual network when λ_perc is None, which
    `total_loss` resolves to the network's calibrated weight (D-89), or positive."""
    return loss.lambda_perc is None or loss.lambda_perc > 0.0


# Q-250: D-89's networks without trained weights, which have no bfloat16 path (B-4 measures
# bfloat16 for the trunk networks only).
_WEIGHT_FREE = ("flip", "haarpsi")


def _check_perceptual(cfg: TrainConfig) -> None:
    """Q-250: bf16 with a weight-free network is refused before anything is built; the
    `TrainConfig` stays valid on its own, so it can move to a trunk network. Without a network
    (λ_perc = 0) nothing would run in bfloat16, and nothing is refused (Q-T388.1)."""
    net = cfg.loss.perc_net
    # Q-T388.1: refused only when the network is built.
    if cfg.perf.perc_precision == "bf16" and net in _WEIGHT_FREE and _builds_perceptual(cfg.loss):
        raise ValueError(
            f"perf.perc_precision 'bf16' runs a trunk network in bfloat16 (D-52), and "
            f"loss.perc_net {net!r} has no bfloat16 path: use 'tf32' or 'ieee', or a trunk network"
        )


def _perceptual_network(cfg: TrainConfig) -> PerceptualLoss | LPIPSLoss:
    """The trainer's perceptual network for `cfg`, on the CPU. LPIPS's two backbones, still
    accepted as `perc_net` until #392 (#380's expand step), are built as before (Q-172, Q-223);
    every network of D-89 takes the render's view as it comes, its layout and autotuning to be
    set per network after B-4 (Q-251), and a trunk network runs under bfloat16 autocast when
    asked (Q-250; `_check_perceptual` refuses bf16 for the weight-free ones)."""
    net = cfg.loss.perc_net
    if net == "vgg" or net == "alex":
        return _lpips_network(cfg.perf)(net)
    if cfg.perf.perc_precision == "bf16":
        return _Bf16PerceptualLoss(net)
    return PerceptualLoss(net)


def _check_ablation(cfg: TrainConfig) -> None:
    """Q-71: the "w/o periodic relocation" ablation turns relocation off (§1.3 Count-control
    modes), so a pruning opacity with relocation on, a mode the spec doesn't define, is
    refused rather than run."""
    if cfg.reloc.enabled and cfg.reloc.prune_opacity is not None:
        raise ValueError(
            "reloc.prune_opacity selects the ablation, which turns relocation off: set "
            "reloc.enabled to False"
        )


def _check_perf(perf: PerfConfig) -> None:
    """Q-73: an option that isn't built fails at construction rather than being ignored."""
    if perf.images_per_step != 1:
        # Q-177 (owner's answer): multi-GPU training is deferred.
        raise NotImplementedError(
            "images_per_step > 1 (D-54) is deferred: training on several GPUs needs at least "
            "two GPUs, and must pass M16's L3 against single-GPU training before it is offered"
        )
    if perf.slice_impl == "cuda":
        # D-50: B-1 put the slice under 10% of an iteration (#226), so the operator isn't built.
        raise NotImplementedError(
            "perf.slice_impl 'cuda' (D-50) is not built: B-1 measured the temporal slice under "
            "D-50's 10 % of a training iteration (#226); use 'auto', 'compiled' or 'eager'"
        )


def _bytes_per_row(model: FreeTimeGaussians) -> int:
    """Q-99: the bytes a primitive row takes during training: its floats in the nine stored
    tensors (§1.1) for the parameter, its gradient and Adam's two moments, plus the gradient
    statistics' two floats, all float32."""
    floats = sum(tensor[0].numel() for tensor in _stored_tensors(model).values())
    return 4 * (4 * floats + 2)


def _fits_on_gpu(cache: int, *, free: int, reserve: int, growth: int) -> bool:
    """D-51, Q-99, Q-225: whether "auto" puts a cache of `cache` bytes on the GPU: it fits in
    half the free memory less the working memory `reserve` held for the iterations, minus what
    growth to N_max adds."""
    return cache <= 0.5 * (free - reserve) - growth


def _stored_tensors(model: FreeTimeGaussians) -> dict[str, Tensor]:
    """The model's nine stored tensors (§1.1), by name."""
    return {name: getattr(model, name) for name in FreeTimeGaussians.PARAM_NAMES}


def _copy_of(model: FreeTimeGaussians) -> FreeTimeGaussians:
    """Q-69 (owner answer, #97): a copy of a passed model, made as the model copies the tensors
    it is given, so the caller's object is never modified or moved; it keeps the model's SH
    degree, time normalizer and time unit (Q-17). Q-76: the constructor checks §1.1, so a model
    that breaks the validity rules is refused here."""
    return FreeTimeGaussians(
        _stored_tensors(model),
        sh_degree=model.sh_degree,
        time=model.time,
        time_unit=model.time_unit,
    )


def _cpu(tree: Any) -> Any:
    """A copy of a tree of dicts with every tensor detached and cloned to the CPU."""
    if isinstance(tree, Tensor):
        return tree.detach().to("cpu", copy=True)
    if isinstance(tree, Mapping):
        return {key: _cpu(value) for key, value in tree.items()}
    return tree


@dataclass
class _Run:
    """Everything a training run changes, built in full before it replaces the old one."""

    model: FreeTimeGaussians
    optimizer: FTGSOptimizer
    stats: GradStats
    sampler: _Sampler
    relocation: torch.Generator  # every event draw (D-15; Q-72)
    time: TimeNormalizer  # the run's τ₀, τ₁
    spatial_scale: float  # S (D-27)
    frame_dt: float  # Δt_f (E14)
    iteration: int


class Trainer:
    """The training loop of §1.3 on a multi-view video: one image per training iteration, the
    events of the calendar, and the trainer state a checkpoint holds (§2.2). The seed is
    `cfg.seed`; the device selects the backend: gsplat on CUDA, the reference elsewhere (Q-69)."""

    def __init__(
        self,
        cfg: TrainConfig,
        dataset: MultiViewVideo,
        init: InitPoints | FreeTimeGaussians,
        *,
        device: str = "cuda",
    ) -> None:
        self._configure(cfg, dataset, device)
        time_unit = dataset.manifest.time_unit
        if isinstance(init, FreeTimeGaussians):
            # Q-69: a model passed in must belong to this run; the trainer trains a copy.
            if init.sh_degree != cfg.sh_degree:
                raise ValueError(f"the model's SH degree {init.sh_degree} isn't {cfg.sh_degree}")
            if init.time != dataset.time:
                raise ValueError(f"the model's time {init.time} isn't the dataset's {dataset.time}")
            if init.time_unit != time_unit:
                raise ValueError(f"the model's time unit {init.time_unit!r} isn't {time_unit!r}")
            if init.num_gaussians > cfg.growth.n_max:
                raise ValueError(
                    f"the model's N = {init.num_gaussians} exceeds N_max = {cfg.growth.n_max}"
                )
            model = _copy_of(init)
        elif isinstance(init, InitPoints):
            # §1.3 steps 7–9 (D-24, D-74; Q-16), with the run's own initialization generator.
            model = FreeTimeGaussians.from_init_points(
                init,
                dataset.time,
                cfg.init,
                frame_dt=dataset.frame_dt,
                generator=_seeded(cfg.seed, "init"),
                sh_degree=cfg.sh_degree,
                n_max=cfg.growth.n_max,
            )
            model.time_unit = time_unit
        else:
            raise ValueError(f"init must be InitPoints or a model, got {type(init).__name__}")
        spatial_scale = dataset.spatial_scale(cfg.scene_scale_factor)  # D-27
        model = self._place(model)
        self._install(
            _Run(
                model=model,
                optimizer=self._optimizer_for(model, spatial_scale, dataset.frame_dt),
                stats=GradStats(model.num_gaussians, device=self._device),
                sampler=_Sampler(self._training, _seeded(cfg.seed, "sampler")),
                relocation=_seeded(cfg.seed, "relocation"),
                time=dataset.time,
                spatial_scale=spatial_scale,
                frame_dt=dataset.frame_dt,
                iteration=0,
            )
        )

    # Q-70 (owner answer): force resumes despite a changed scene manifest.
    @classmethod
    def from_checkpoint(
        cls,
        directory: Path,
        dataset: MultiViewVideo,
        *,
        device: str = "cuda",
        force: bool = False,
        perf: PerfConfig | None = None,
    ) -> "Trainer":
        """Resume the run saved in a ftgs-checkpoint directory (io.load_checkpoint, then
        load_state_dict). A checkpoint of another scene manifest is refused unless forced; a
        forced resume keeps the run's τ₀, τ₁, S and Δt_f (§2.2; Q-70). Given `perf`, the run
        continues with those performance settings, which may differ from the checkpoint's only
        in the fields that change no number (Q-134, Q-153); else with the checkpoint's."""
        state = load_checkpoint(Path(directory))
        if state.scene_sha256 != dataset.manifest.sha256 and not force:
            raise ValueError(
                f"checkpoint {directory} was made from the scene manifest {state.scene_sha256}, "
                f"not {dataset.manifest.sha256}; pass force=True to resume anyway"
            )
        cfg = config_from_dict(state.config)
        # Q-134 (as answered): checked before anything is configured, so a refused setting opens
        # no frame cache and reaches no NotImplementedError; load_state_dict checks again.
        if perf is not None:
            given = dataclasses.replace(cfg, perf=perf)
            conflicts = resume_conflicts(cfg, given)
            if conflicts:
                raise ValueError(
                    f"perf differs from the checkpoint's where it must match: {conflicts}"
                )
            cfg = given
        trainer = cls.__new__(cls)
        trainer._configure(cfg, dataset, device)
        trainer.load_state_dict(state)
        return trainer

    def _configure(self, cfg: TrainConfig, dataset: MultiViewVideo, device: str) -> None:
        _check_ablation(cfg)
        _check_perf(cfg.perf)
        _check_perceptual(cfg)  # Q-250
        self._cfg = cfg
        self._dataset = dataset
        self._device = torch.device(device)
        # Q-69, D-76: gsplat renders on CUDA; the reference backend everywhere else.
        self._backend: Literal["gsplat", "reference"] = (
            "gsplat" if self._device.type == "cuda" else "reference"
        )
        self._training = [i for i, r in enumerate(dataset.records) if r.split == "train"]
        self._background = torch.tensor(cfg.background, dtype=torch.float32, device=self._device)
        # D-77: one perceptual network, built when its weight is None or positive (Q-69, Q-249).
        self._perceptual: PerceptualLoss | LPIPSLoss | None = None
        if _builds_perceptual(cfg.loss):
            # Q-72: lpips's trunk and the vendored DINOv2 builders initialize their weights from
            # torch's global CPU generator before the publisher's are loaded; its state is
            # restored, so the trainer draws nothing from it.
            with torch.random.fork_rng(devices=[]):
                perceptual = _perceptual_network(cfg)
            perceptual = perceptual.to(self._device)
            if _CHANNELS_LAST_WEIGHTS:  # Q-173
                perceptual = perceptual.to(memory_format=torch.channels_last)
            self._perceptual = perceptual
        self._evaluator: Evaluator | None = None
        self._frames: Prefetcher | None = None  # D-51: training images through the frame cache
        self._placement_wait: _PlacementWait | None = None  # "auto" pending on CUDA (Q-232)
        self._open_frame_cache(cfg.perf.frame_cache)
        # §2.4 rule 3: every training record's camera is copied to the device here, once, so a
        # training iteration's render copies no camera from the host; records share camera
        # objects, so they are keyed by identity (Camera's == compares tensors).
        self._cameras: dict[int, Camera] = {}
        for index in self._training:
            camera = dataset.records[index].camera
            if id(camera) not in self._cameras:
                self._cameras[id(camera)] = dataclasses.replace(
                    camera,
                    K=camera.K.to(self._device),
                    world_to_camera=camera.world_to_camera.to(self._device),
                )

    def _place(self, model: FreeTimeGaussians) -> FreeTimeGaussians:
        model.slice_impl = self._cfg.perf.slice_impl  # Q-66
        return model.to(self._device)

    def _optimizer_for(
        self, model: FreeTimeGaussians, spatial_scale: float, frame_dt: float
    ) -> FTGSOptimizer:
        return build_optimizer(
            model,
            self._cfg.optim,
            spatial_scale=spatial_scale,
            frame_dt=frame_dt,
            adam=self._cfg.perf.adam,  # Q-44
        )

    def _install(self, run: _Run) -> None:
        self._run = run
        self._window: dict[str, Tensor] = {}  # loss sums since the last read (§2.4 rule 3)
        self._window_count = 0
        self._schedule()  # Q-99: a restored sampler gets its own queue

    def _open_frame_cache(self, placement: str, *, automatic: bool = False) -> None:
        """D-51, Q-99: "off" decodes PNGs; "gpu" and "mmap" open the cache now; "auto" is mmap on
        a CPU device and, on CUDA, decided after the iteration that follows the first at the last
        training image size to appear (Q-232). A cache directory that can't be made turns "auto"
        into "off" with a warning, and is a ValueError for an explicit placement (Q-26)."""
        if placement == "off":
            return
        if placement == "auto" and self._device.type == "cuda":
            self._await_placement()
            return
        automatic = automatic or placement == "auto"
        chosen: Literal["gpu", "mmap"] = "gpu" if placement == "gpu" else "mmap"
        try:
            cache = FrameCache.open(self._dataset, placement=chosen, device=self._device)
        except OSError as err:
            if not automatic:
                raise ValueError(f"the frame cache can't be made: {err}") from None
            warnings.warn(
                f"no frame cache, decoding PNGs instead: {err}", UserWarning, stacklevel=3
            )
            return
        depth = 0 if chosen == "gpu" else self._cfg.perf.prefetch
        self._frames = Prefetcher(cache, device=self._device, depth=depth)
        weakref.finalize(self, self._frames.close)  # the reader thread ends with the trainer

    def _await_placement(self) -> None:
        """Q-232 (changed, #348): "auto" on CUDA decodes PNGs until `_PlacementWait` says it
        decides, counting from this process's first iteration."""
        self._placement_wait = _PlacementWait(self._dataset)

    def _choose_placement(self) -> None:
        """Q-99: once `_PlacementWait` says "auto" decides (Q-232), the cache goes to GPU memory
        when it fits in half the free memory, less the working memory the iteration left
        reserved, minus what growth to N_max will add (D-51, Q-225), and is memory-mapped
        otherwise; a cache that can't be made falls back to PNGs with a warning."""
        unheld, _ = torch.cuda.mem_get_info(self._device)
        # Q-225: the allocator's unused reserve is the iteration's working memory (the render's
        # and the losses' buffers at the image's size, the gradients), which Q-99's free memory
        # counts and the rule sets aside again.
        reserve = torch.cuda.memory_reserved(self._device) - torch.cuda.memory_allocated(
            self._device
        )
        model = self._run.model
        growth = max(0, self._cfg.growth.n_max - model.num_gaussians) * _bytes_per_row(model)
        fits = _fits_on_gpu(
            FrameCache.size_of(self._dataset),
            free=unheld + reserve,
            reserve=reserve,
            growth=growth,
        )
        self._open_frame_cache("gpu" if fits else "mmap", automatic=True)
        self._schedule()

    def _schedule(self) -> None:
        if self._frames is not None:
            self._frames.schedule(self._run.sampler.peek(self._cfg.perf.prefetch))

    def _image(self, index: int) -> Tensor:
        """Training image `index` on the device: from the frame cache when there is one, bitwise
        what `MultiViewVideo.load_image` decodes (D-51)."""
        if self._frames is None:
            image = self._dataset.load_image(index)
            if self._device.type != "cuda":
                return image.to(self._device)
            # §2.4 rule 3, D-51: a copy from pageable memory blocks the host, so the decoded image
            # is staged in pinned memory and sent without blocking; PyTorch's pinned allocator
            # keeps the staging block until the copy is done. The bytes are copied as they are.
            return image.pin_memory().to(self._device, non_blocking=True)
        image = self._frames.take(index)
        self._schedule()
        return image

    # -- training -------------------------------------------------------------------------------

    # Q-69 (owner answer, #97): read-only; load_state_dict replaces the model it returns.
    @property
    def model(self) -> FreeTimeGaussians:
        """The model the trainer is training: the same object across training iterations and
        events, which resize it in place; `load_state_dict` and `from_checkpoint` replace it with
        the model of the loaded state."""
        return self._run.model

    @property
    def iteration(self) -> int:
        """Completed optimizer steps."""
        return self._run.iteration

    def step(self) -> StepLog:
        """One training iteration (§1.3 steps 1–7) at it = iteration + 1."""
        run, cfg = self._run, self._cfg
        it = run.iteration + 1
        if it > cfg.optim.iterations:
            raise ValueError(f"the run is complete at iteration {run.iteration}")
        entry = calendar_entry(cfg, it)
        # §3.4: the iteration's stages (B-1); the slice, rasterization and loss stages are
        # render's, at_time's and total_loss's.
        with stage("iteration"):
            with _perf_flags(cfg.perf), torch.enable_grad():
                # Steps 1–2: the next image of the sampler, its normalized time and L_active.
                index = run.sampler.next()
                record = self._dataset.records[index]
                with stage("data"):
                    gt = self._image(index)
                t = float(run.time.to_t(record.timestamp))
                # Step 3: the per-time arrays and the render, retaining the 2D means' gradient.
                out = render(
                    run.model,
                    self._cameras[id(record.camera)],
                    t,
                    time_units="normalized",
                    sh_degree=entry.sh_degree,
                    background=self._background,
                    backend=self._backend,
                    return_aux=True,
                    precull=cfg.perf.precull,  # Q-31
                    packed=cfg.perf.packed,  # Q-36
                )
                # §3.4: with a stage clock active, the image's gradient marks where every loss's
                # backward ends.
                point(out.image, "backward.losses_done")
                # Step 4: E9.
                loss, terms = total_loss(
                    out.image, gt[None], run.model, t, it, cfg.loss, perceptual=self._perceptual
                )
                # Step 5: backpropagate, then the gradient statistics (E10, D-12).
                with stage("backward"):
                    torch.autograd.backward(loss)
                assert out.means2d is not None and out.visible is not None  # return_aux
                with stage("statistics"):
                    grad = out.means2d.grad
                    if grad is not None:  # a check for None, not a synchronization
                        width, height = record.camera.width, record.camera.height
                        if out.gaussian_ids is not None:  # packed mode (Q-36)
                            run.stats.update(
                                grad, width=width, height=height, gaussian_ids=out.gaussian_ids
                            )
                        else:
                            run.stats.update(grad, width=width, height=height, visible=out.visible)
                # Step 6: this iteration's rates and one Adam step.
                with stage("adam"):
                    rates = run.optimizer.set_iteration(it)
                    if cfg.perf.adam == "selective":
                        run.optimizer.step(visibility=out.visible.any(dim=0))
                    else:
                        run.optimizer.step()
                    run.optimizer.zero_grad()
                # Step 7: the event.
                with stage("events"):
                    growth, relocation, pruned = self._event(entry, it)
            run.iteration = it
            if self._placement_wait is not None and self._placement_wait.decides_after(index):
                self._placement_wait = None
                self._choose_placement()
            with stage("log"):
                mean_losses = self._log(terms, it)
        return StepLog(
            iteration=it,
            image_index=index,
            timestamp=record.timestamp,
            t=t,
            sh_degree=entry.sh_degree,
            rates=rates,
            losses=terms,
            num_gaussians=run.model.num_gaussians,
            growth=growth,
            relocation=relocation,
            pruned=pruned,
            opacity_reset=entry.reset_opacity,
            mean_losses=mean_losses,
        )

    def _check_valid(self, it: int) -> None:
        """Q-69: a model that fails §1.1's validity rules, or non-finite gradient statistics,
        mean the run diverged. Reads values, so it runs only where the run synchronizes."""
        try:
            self._run.model.validate()
        except ValueError as err:
            raise FloatingPointError(f"training diverged by iteration {it}: {err}") from err
        if not bool(torch.isfinite(self._run.stats.g_sum).all()):
            raise FloatingPointError(
                f"training diverged by iteration {it}: the gradient statistics are not finite"
            )

    def _event(
        self, entry: CalendarEntry, it: int
    ) -> tuple[GrowthReport | None, RelocationReport | None, int | None]:
        """§1.3 step 7 in its order (D-45; Q-71): growth, relocation, the ablation's pruning
        and opacity reset, then the statistics reset, which relocation does itself (Q-51)."""
        if not (entry.event or entry.reset_opacity):
            return None, None, None
        run, cfg = self._run, self._cfg
        self._check_valid(it)
        growth = relocation = pruned = None
        if entry.grow:
            growth = grow(
                run.model,
                run.optimizer,
                run.stats,
                cfg.growth,
                spatial_scale=run.spatial_scale,
                reloc=cfg.reloc,
                generator=run.relocation,
            )
        if entry.relocate:
            relocation = relocate(
                run.model, run.optimizer, run.stats, cfg.reloc, generator=run.relocation
            )
        if entry.prune:
            pruned = prune(
                run.model,
                run.optimizer,
                run.stats,
                cfg.reloc,
                spatial_scale=run.spatial_scale,
                iteration=it,
            )
        if entry.reset_opacity:
            reset_opacity(run.model, run.optimizer)
        if entry.event and not entry.relocate:
            run.stats.reset()  # Q-51: with relocation off, the trainer resets them
        return growth, relocation, pruned

    def _log(self, terms: Mapping[str, Tensor], it: int) -> dict[str, float] | None:
        """§2.4 rule 3: loss terms add up on the device and are read every log_every
        iterations, with the validity check; a non-finite mean means divergence (Q-69)."""
        for name, value in terms.items():
            self._window[name] = value if name not in self._window else self._window[name] + value
        self._window_count += 1
        if it % self._cfg.perf.log_every != 0:
            return None
        means = {name: float(total) / self._window_count for name, total in self._window.items()}
        self._window, self._window_count = {}, 0
        if not all(math.isfinite(value) for value in means.values()):
            raise FloatingPointError(f"training diverged by iteration {it}: mean losses {means}")
        self._check_valid(it)
        return means

    def train(self, until: int | None = None, callbacks: Sequence[Callback] = ()) -> None:
        """Step until `until` (default T_it), calling every callback after each step; a
        callback that returns True stops training after that step (Q-69)."""
        iterations = self._cfg.optim.iterations
        target = iterations if until is None else until
        _check_int(target, self._run.iteration, iterations, "until")
        while self._run.iteration < target:
            log = self.step()
            stops = [bool(callback(self, log)) for callback in callbacks]
            if any(stops):
                return

    # -- evaluation -----------------------------------------------------------------------------

    # Q-74: the returned keys and what out_dir receives.
    def evaluate(self, split: str = "test", *, out_dir: Path | None = None) -> dict[str, float]:
        """The E16 metrics of a data split's images: the scene means of the entire images, and
        under "dynamic/<name>" those of the dynamic regions when there are any. With `out_dir`,
        METRICS.json is written there (Q-63, Q-74). A render that isn't finite raises
        FloatingPointError from evaluate_images, before anything is written (Q-129, Q-147)."""
        if self._evaluator is None:
            self._evaluator = Evaluator(device=self._device)
        images = evaluate_images(
            self._run.model,
            self._dataset,
            self._evaluator,
            split=split,
            backend=self._backend,
            background=self._cfg.background,
        )
        result = average_metrics([image.entire for image in images])
        dynamic = [image.dynamic for image in images if image.dynamic is not None]
        if dynamic:
            result |= {f"dynamic/{k}": v for k, v in average_metrics(dynamic).items()}
        if out_dir is not None:
            manifest = self._dataset.manifest
            body = metrics_report(images, split=split, metadata=manifest.metadata)
            write_metrics(  # Q-136
                Path(out_dir) / "METRICS.json",
                body,
                manifest=manifest,
                iteration=self._run.iteration,
            )
        return result

    # -- state ----------------------------------------------------------------------------------

    def state_dict(self) -> TrainerState:
        """The trainer state of §2.2's checkpoint, with every tensor a CPU copy (Q-69, Q-72)."""
        run = self._run
        return TrainerState(
            iteration=run.iteration,
            config=config_to_dict(self._cfg),
            time=run.time,
            spatial_scale=run.spatial_scale,
            frame_dt=run.frame_dt,
            params=_cpu(_stored_tensors(run.model)),
            optim=_cpu(run.optimizer.state_dict()),
            stats=_cpu(run.stats.state_dict()),
            rng=_cpu(
                {
                    "sampler": run.sampler.generator.get_state(),
                    "relocation": run.relocation.get_state(),
                    **_global_rng(),
                }
            ),
            sampler=run.sampler.state_dict(),
            scene_sha256=self._dataset.manifest.sha256,
            time_unit=run.model.time_unit,
        )

    def load_state_dict(self, state: TrainerState) -> None:
        """Continue the run `state` holds: its configuration must be this trainer's, apart from
        the performance settings that change no number, for which the trainer keeps its own
        (Q-134, Q-153); its τ₀, τ₁, S and Δt_f are adopted. Everything is built before anything
        is replaced, so a rejected state leaves the trainer as it was (Q-69, Q-72)."""
        cfg = self._cfg
        conflicts = resume_conflicts(config_from_dict(state.config), cfg)
        if conflicts:
            raise ValueError(f"the state's configuration differs from the trainer's: {conflicts}")
        time_unit = self._dataset.manifest.time_unit
        if state.time_unit != time_unit:
            raise ValueError(f"the state's time unit {state.time_unit!r} isn't {time_unit!r}")
        iteration = _check_int(state.iteration, 0, cfg.optim.iterations, "the state's iteration")
        if int(state.params["means"].shape[0]) > cfg.growth.n_max:
            raise ValueError(f"the state's N exceeds N_max = {cfg.growth.n_max}")
        model = FreeTimeGaussians(
            state.params, sh_degree=cfg.sh_degree, time=state.time, time_unit=state.time_unit
        )
        model = self._place(model)
        optimizer = self._optimizer_for(model, state.spatial_scale, state.frame_dt)  # Q-46
        optimizer.load_state_dict(state.optim)
        stats = GradStats(model.num_gaussians, device=self._device)
        stats.load_state_dict(state.stats)
        rng = state.rng
        restore_global_rng = _global_rng_restorer(rng)
        sampler = _Sampler(
            self._training, _restored_generator(rng.get("sampler"), "sampler"), state.sampler
        )
        run = _Run(
            model=model,
            optimizer=optimizer,
            stats=stats,
            sampler=sampler,
            relocation=_restored_generator(rng.get("relocation"), "relocation"),
            time=state.time,
            spatial_scale=state.spatial_scale,
            frame_dt=state.frame_dt,
            iteration=iteration,
        )
        self._install(run)
        restore_global_rng()
