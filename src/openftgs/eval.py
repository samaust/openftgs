# SPDX-License-Identifier: Apache-2.0
"""PSNR, SSIM_k, DSSIM_k, LPIPS-Alex and LPIPS-VGG, the dynamic-region crop and mask, the
evaluation of a model on a scene's images, and the metrics report that `ftgs eval` writes.

Spec: E16, App. B.1, D-29, D-52, D-55, D-64, D-67, D-68, D-69. Band: Orchestration (§2.1).
"""

import contextlib
import importlib.metadata
import json
import math
import os
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from openftgs.data import MultiViewVideo
from openftgs.losses import LPIPSLoss, ssim
from openftgs.model import FreeTimeGaussians
from openftgs.render import render_request
from openftgs.types import RenderItem, RenderRequest, SceneManifest

__all__ = [
    "METRIC_NAMES",
    "Evaluator",
    "ImageMetrics",
    "average_metrics",
    "dssim_k",
    "dynamic_region",
    "evaluate_images",
    "lpips_backbone",
    "metrics_report",
    "psnr",
    "ssim_k",
    "write_metrics",
]

# §2.1 CLI paragraph: the metrics METRICS.json reports, in this order (Q-60).
METRIC_NAMES = (
    "psnr",
    "ssim1",
    "ssim2",
    "dssim1",
    "dssim2",
    "lpips_alex",
    "lpips_vgg",
    "lpips_vgg_raw",
)

# Q-61: the smallest side of a dynamic region every metric accepts: SSIM's valid window needs
# 11 px, LPIPS-VGG 16 and LPIPS-Alex 31 (lpips 0.1.4; source log #67).
_MIN_REGION_SIDE = 31
_MASK_THRESHOLD = 0.5  # D-67: soft masks are binarized at 0.5; Q-61: 0.5 counts as dynamic


@contextlib.contextmanager
def _metric_mode() -> Iterator[None]:
    """D-52, §2.4 Precision: evaluation is a forward pass with plain float32 convolutions, so
    cuDNN's TF32 flag is cleared, and restored on exit; inference mode, as for rendering."""
    before = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = before


# Q-60: images are [..., H, W, 3] of one shape, floating dtype and device (Q-41); one value
# per image.
def _check_pair(pred: Tensor, gt: Tensor) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have one shape, got {list(pred.shape)}, {list(gt.shape)}"
        )
    if pred.dim() < 3 or pred.shape[-1] != 3:
        raise ValueError(f"images must be [..., H, W, 3], got {list(pred.shape)}")
    if not pred.is_floating_point() or pred.dtype != gt.dtype:
        raise ValueError(f"images must share one floating dtype, got {pred.dtype} and {gt.dtype}")
    if pred.device != gt.device:
        raise ValueError(f"images must be on one device, got {pred.device} and {gt.device}")


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k not in (1, 2):
        raise ValueError(f"k must be 1 or 2 (E16), got {k!r}")


def psnr(pred: Tensor, gt: Tensor) -> Tensor:
    """PSNR = 10·log10(1/MSE) per image, the render clamped to [0, 1] first (E16, D-67); the MSE
    is taken in float64 over pixels and channels, and an MSE of 0 gives +inf (Q-60)."""
    _check_pair(pred, gt)
    with _metric_mode():
        diff = pred.to(torch.float64).clamp(0.0, 1.0) - gt.to(torch.float64)
        mse = diff.square().mean(dim=(-3, -2, -1))
        return -10.0 * torch.log10(mse)


def ssim_k(pred: Tensor, gt: Tensor, k: int) -> Tensor:
    """SSIM_k per image: E7's SSIM with data range k ∈ {1, 2}, so C₁ = (0.01k)² and
    C₂ = (0.03k)², the render clamped to [0, 1] first (E16, D-67)."""
    _check_pair(pred, gt)
    _check_k(k)
    with _metric_mode():
        lead = pred.shape[:-3]
        x = pred.clamp(0.0, 1.0).reshape(-1, *pred.shape[-3:])
        y = gt.reshape(-1, *gt.shape[-3:])
        if x.shape[0] == 0:  # Q-60: no images, no values
            return pred.new_empty(lead)
        # One call per image through the loss's public SSIM, which averages over its batch.
        values = [ssim(x[i], y[i], data_range=float(k)) for i in range(x.shape[0])]
        return torch.stack(values).reshape(lead)


def dssim_k(pred: Tensor, gt: Tensor, k: int) -> Tensor:
    """DSSIM_k = (1 − SSIM_k)/2 per image, a dissimilarity (E16, D-29, D-40)."""
    value = ssim_k(pred, gt, k)
    with _metric_mode():
        return (1.0 - value) / 2.0


def dynamic_region(pred: Tensor, gt: Tensor, mask: Tensor) -> tuple[Tensor, Tensor] | None:
    """The pair a dynamic-region metric compares (App. B.1, D-67): the render clamped to
    [0, 1], then both images set to black outside the mask and cropped to the mask's bounding
    box. `mask` is [H, W] or [H, W, 1], bool or floating, binarized at 0.5 (≥ 0.5 is dynamic).
    None, with a warning, for an empty mask or a box narrower than 31 px on a side, which some
    metric can't evaluate (Q-61)."""
    if pred.dim() != 3:
        raise ValueError(f"dynamic_region takes one image pair [H, W, 3], got {list(pred.shape)}")
    _check_pair(pred, gt)
    height, width = pred.shape[0], pred.shape[1]
    if mask.dim() == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if tuple(mask.shape) != (height, width):
        raise ValueError(
            f"mask must be [{height}, {width}] or [{height}, {width}, 1], got {list(mask.shape)}"
        )
    if mask.device != pred.device:
        raise ValueError(f"mask must be on the images' device {pred.device}, got {mask.device}")
    # Q-61: an 8-bit mask is divided by 255 first, as every 8-bit input is (§1.4 "Images").
    if mask.dtype == torch.bool:
        dynamic = mask
    elif mask.is_floating_point():
        dynamic = mask >= _MASK_THRESHOLD
    else:
        raise ValueError(f"mask must be bool or floating in [0, 1], got {mask.dtype}")
    rows = torch.nonzero(dynamic.any(dim=1)).flatten()
    cols = torch.nonzero(dynamic.any(dim=0)).flatten()
    if rows.numel() == 0:
        warnings.warn(
            "dynamic-region mask is empty: no dynamic metrics for this image",
            UserWarning,
            stacklevel=2,
        )
        return None
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    if min(r1 - r0, c1 - c0) < _MIN_REGION_SIDE:
        warnings.warn(
            f"dynamic-region box {r1 - r0}x{c1 - c0} px is smaller than {_MIN_REGION_SIDE} px on "
            "a side: no dynamic metrics for this image",
            UserWarning,
            stacklevel=2,
        )
        return None
    keep = dynamic[r0:r1, c0:c1, None]
    zero = pred.new_zeros(())
    pred_crop = torch.where(keep, pred[r0:r1, c0:c1].clamp(0.0, 1.0), zero)
    gt_crop = torch.where(keep, gt[r0:r1, c0:c1], gt.new_zeros(()))
    return pred_crop, gt_crop


# Q-62: the dataset is read from metadata.source.
def lpips_backbone(source: str | None) -> Literal["alex", "vgg"]:
    """The LPIPS backbone the paper reports for a dataset (D-29): AlexNet for Neural3DV, VGG
    otherwise, read from the scene manifest's `metadata.source`."""
    if source is not None and not isinstance(source, str):
        raise ValueError(f"metadata.source must be a string, got {source!r}")
    if source is not None and (source == "n3dv" or source.startswith("n3dv/")):
        return "alex"
    return "vgg"


def average_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The arithmetic mean of each metric over the rows (D-67): images → scene, then scenes →
    overall. Every row must hold the same keys."""
    if not rows:
        raise ValueError("average_metrics needs at least one row")
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("every row must hold the same keys")
    return {key: math.fsum(row[key] for row in rows) / len(rows) for key in keys}


class Evaluator:
    """All eight metrics of METRIC_NAMES for one image pair, with the LPIPS networks built once
    on `device` (E16, D-68; Q-60). Calling it with a dynamic-region mask also gives the same
    metrics on `dynamic_region`'s pair, or None when the mask has no usable region (Q-61)."""

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self._device = torch.device(device)
        self._alex = LPIPSLoss("alex").to(self._device)
        self._vgg = LPIPSLoss("vgg").to(self._device)

    def __call__(
        self, pred: Tensor, gt: Tensor, *, mask: Tensor | None = None
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        if pred.dim() != 3:
            raise ValueError(f"Evaluator takes one image pair [H, W, 3], got {list(pred.shape)}")
        _check_pair(pred, gt)
        # D-52: reported metrics are plain float32, the networks' dtype (Q-60).
        pred = pred.to(self._device, torch.float32)
        gt = gt.to(self._device, torch.float32)
        entire = self._metrics(pred, gt)
        if mask is None:
            return entire, None
        region = dynamic_region(pred, gt, mask.to(self._device))
        return entire, None if region is None else self._metrics(*region)

    def _metrics(self, pred: Tensor, gt: Tensor) -> dict[str, float]:
        ssim1, ssim2 = ssim_k(pred, gt, 1), ssim_k(pred, gt, 2)
        values = {"psnr": psnr(pred, gt), "ssim1": ssim1, "ssim2": ssim2}
        with _metric_mode():
            values["dssim1"] = (1.0 - ssim1) / 2.0  # as dssim_k, from the same SSIM
            values["dssim2"] = (1.0 - ssim2) / 2.0
            clamped = pred.clamp(0.0, 1.0)  # D-67
            values["lpips_alex"] = self._alex(clamped, gt)
            values["lpips_vgg"] = self._vgg(clamped, gt)
            # D-68, Q-58: the same VGG network on the unmapped [0, 1] images.
            values["lpips_vgg_raw"] = self._vgg(clamped, gt, map_inputs=False)
        return {name: float(values[name]) for name in METRIC_NAMES}


# Q-60: one test image's results, as metrics_report reads them.
@dataclass(frozen=True)
class ImageMetrics:
    camera: str
    timestamp: float  # raw τ
    frame_index: int | None
    path: str  # the image path of the scene manifest
    entire: Mapping[str, float]  # the eight METRIC_NAMES values
    dynamic: Mapping[str, float] | None  # None without a usable dynamic region (Q-61)
    masked: bool  # the image carries a dynamic-region mask


# Q-68: the images of one data split, rendered through render_request (Q-57's shared path).
def evaluate_images(
    model: FreeTimeGaussians,
    dataset: MultiViewVideo,
    evaluator: Evaluator,
    *,
    split: str = "test",
    backend: Literal["gsplat", "reference"] = "gsplat",
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    renders: Mapping[int, Tensor] | None = None,
) -> list[ImageMetrics]:
    """The metrics of every image of one data split (E16, D-67): each record rendered at its
    raw timestamp with the model's full SH degree and `background`, through `render_request`
    one timestamp at a time, so cameras that share a time render together (D-55) and one
    time's renders are held at once, then evaluated by `evaluator` with the image's mask.
    Results come in the manifest's order. Given `renders`, a prediction [H, W, 3] for every
    record of the split by record index, nothing is rendered and those are scored (Q-137); one
    that isn't finite is a ValueError.

    A render of the model that isn't finite, in its colors or its alpha, is not scored, nor is
    anything after it: once the whole data split is rendered, FloatingPointError names up to
    five such images (Q-129, Q-147)."""
    indices = [i for i, record in enumerate(dataset.records) if record.split == split]
    if not indices:
        raise ValueError(f"the data split {split!r} holds no images")
    root = dataset.manifest.path.parent
    if renders is not None:
        absent = [i for i in indices if i not in renders]
        if absent:
            names = _image_names(dataset, absent)
            raise ValueError(f"no supplied render for {len(absent)} image(s): {names[:5]}")
        broken = [i for i in indices if not _finite(renders[i])]
        if broken:  # Q-147: input, as read_renders would refuse it (Q-142)
            names = _image_names(dataset, broken)
            raise ValueError(
                f"supplied renders that aren't finite for {len(names)} image(s): {names[:5]}"
            )
    groups: dict[float, list[int]] = {}
    for i in indices:
        groups.setdefault(dataset.records[i].timestamp, []).append(i)
    results: dict[int, ImageMetrics] = {}
    broken = []
    for timestamp, group in groups.items():
        if renders is None:
            items = tuple(RenderItem(str(i), timestamp, dataset.records[i].camera) for i in group)
            request = RenderRequest(
                items, time_unit=dataset.manifest.time_unit, background=background
            )
            rendered = render_request(model, request, backend=backend)
            predictions = {i: rendered.rgb[str(i)] for i in group}
            # Q-147: before E16's clamp, which would turn ±Inf into a plausible color.
            broken += [
                i for i in group if not _finite(rendered.rgb[str(i)], rendered.alpha[str(i)])
            ]
        else:
            predictions = {i: renders[i] for i in group}
        for i in group:
            if broken:  # Q-147: the data split fails, so nothing more is scored
                continue
            record = dataset.records[i]
            mask = dataset.load_mask(i)
            entire, dynamic = evaluator(predictions[i], dataset.load_image(i), mask=mask)
            results[i] = ImageMetrics(
                camera=record.camera.name,
                timestamp=record.timestamp,
                frame_index=record.frame_index,
                path=record.image_path.relative_to(root).as_posix(),
                entire=entire,
                dynamic=dynamic,
                masked=mask is not None,
            )
    if broken:
        names = _image_names(dataset, sorted(broken))
        raise FloatingPointError(
            f"renders that aren't finite for {len(names)} image(s): {names[:5]}"
        )
    return [results[i] for i in indices]


def _image_names(dataset: MultiViewVideo, indices: Sequence[int]) -> list[str]:
    root = dataset.manifest.path.parent
    return [dataset.records[i].image_path.relative_to(root).as_posix() for i in indices]


def _finite(*tensors: Tensor) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


_REPORT = "ftgs-metrics"  # Q-63
_SHA256 = re.compile(r"[0-9a-f]{64}")  # Q-136, as Q-56 spells a digest
_FASTEST_COUNT = 10  # §2.2: metadata.fastest_10 holds [n0, n0 + 10) (D-69)


def _json_value(value: float) -> float | None:
    # Q-63: JSON without NaN or Infinity (Q-25); a perfect render's PSNR becomes null.
    return value if math.isfinite(value) else None


def _check_values(values: Mapping[str, float], path: str) -> None:
    if set(values) != set(METRIC_NAMES):
        raise ValueError(f"{path}: metrics must hold exactly {list(METRIC_NAMES)}")


def _means(rows: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
    if not rows:
        return dict.fromkeys(METRIC_NAMES)
    means = average_metrics(rows)
    return {name: _json_value(means[name]) for name in METRIC_NAMES}


def _row(image: ImageMetrics, values: Mapping[str, float] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "camera": image.camera,
        "timestamp": image.timestamp,
        "frame_index": image.frame_index,
        "path": image.path,
    }
    for name in METRIC_NAMES:
        row[name] = None if values is None else _json_value(values[name])
    return row


def metrics_report(
    images: Sequence[ImageMetrics], *, split: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """The body of METRICS.json (§2.1 CLI paragraph; Q-63): per-image rows and per-scene means
    under `entire`; under `dynamic` when an image is masked, with the manifest's mask source and
    approximate flag (D-64); under `fastest_10` the means over the Fastest-10 frames when the
    manifest names them (D-69). `metadata` is the scene manifest's `metadata`. Non-finite values
    and means over no image are null. The caller adds the header (model, scene, producer)."""
    if not images:
        raise ValueError("metrics_report needs at least one image")
    for image in images:
        _check_values(image.entire, image.path)
        if image.dynamic is not None:
            if not image.masked:
                raise ValueError(f"{image.path}: dynamic metrics need a masked image")
            _check_values(image.dynamic, image.path)
    approximate = metadata.get("masks_approximate", False)
    if not isinstance(approximate, bool):  # D-64
        raise ValueError(f"metadata.masks_approximate must be a bool, got {approximate!r}")
    masked = [image for image in images if image.masked]
    report: dict[str, Any] = {
        "format": _REPORT,
        "version": 1,
        "split": split,
        "metrics": list(METRIC_NAMES),
        "lpips_backbone": lpips_backbone(metadata.get("source")),
        "entire": {
            "scene": _means([image.entire for image in images]),
            "images": [_row(image, image.entire) for image in images],
        },
    }
    if masked:
        usable = [image.dynamic for image in masked if image.dynamic is not None]
        report["dynamic"] = {
            "mask_source": metadata.get("mask_source"),
            "masks_approximate": approximate,
            "count": len(usable),
            "skipped": len(masked) - len(usable),
            "missing": len(images) - len(masked),  # Q-135: images of the split without a mask
            "scene": _means(usable),
            "images": [_row(image, image.dynamic) for image in masked],
        }
    fastest = metadata.get("fastest_10")
    if fastest is not None:
        if (
            not isinstance(fastest, list | tuple)
            or len(fastest) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in fastest)
            or fastest[1] - fastest[0] != _FASTEST_COUNT
        ):
            raise ValueError(f"metadata.fastest_10 must be [n0, n0 + 10), got {fastest!r}")
        n0, n1 = fastest
        inside = [
            image
            for image in images
            if image.frame_index is not None and n0 <= image.frame_index < n1
        ]
        block: dict[str, Any] = {
            "range": [n0, n1],
            "count": len(inside),
            "entire": _means([image.entire for image in inside]),
        }
        if masked:
            block["dynamic"] = _means(
                [image.dynamic for image in inside if image.masked and image.dynamic is not None]
            )
        report["fastest_10"] = block
    return report


_HEADER = ("producer", "model", "manifest", "renders")


def _producer() -> str:
    try:
        return f"openftgs {importlib.metadata.version('openftgs')}"
    except importlib.metadata.PackageNotFoundError:
        return "openftgs"


# Q-136: the one writer of METRICS.json, for Trainer.evaluate and ftgs eval (Q-63, Q-74).
def write_metrics(
    path: Path,
    body: Mapping[str, Any],
    *,
    manifest: SceneManifest,
    model_path: Path | None = None,
    iteration: int | None = None,
    renders: Mapping[str, Any] | None = None,
    model_sha256: str | None = None,
) -> None:
    """Writes METRICS.json: the header (`producer`; `model` as {path, iteration, sha256}, any of
    which may be null, the hash being of an exchange model's file or of a checkpoint's
    manifest.json; the scene manifest as `manifest`, {path, sha256}; `renders` when given,
    Q-137), then `body`, the `metrics_report` dict, as JSON without NaN or Infinity (Q-25).
    A body holding header keys, or a `model_sha256` that isn't 64 lowercase hex digits, is a
    ValueError, raised before anything is written."""
    clash = [key for key in _HEADER if key in body]
    if clash:
        raise ValueError(f"the report body holds header keys {clash}")
    # Q-136 (as answered), Q-56: 64 lowercase hex digits, as the scene manifest's.
    if model_sha256 is not None and not (
        isinstance(model_sha256, str) and _SHA256.fullmatch(model_sha256)
    ):
        raise ValueError(f"model_sha256 must be 64 lowercase hex digits, got {model_sha256!r}")
    header: dict[str, Any] = {
        "producer": _producer(),
        "model": {
            "path": None if model_path is None else str(model_path),
            "iteration": iteration,
            "sha256": model_sha256,
        },
        "manifest": {"path": str(manifest.path), "sha256": manifest.sha256},
    }
    if renders is not None:
        header["renders"] = dict(renders)
    text = json.dumps({**header, **body}, allow_nan=False, indent=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Q-126 (as answered, #179): written to a temporary sibling named as Q-25's
    # `.<name>.tmp-<pid>` and renamed over the target, so a write that fails partway leaves the
    # old file, or no file, and no temporary. A symlink at `path` becomes a regular file (Q-146).
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
