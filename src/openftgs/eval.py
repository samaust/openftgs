# SPDX-License-Identifier: Apache-2.0
"""PSNR, SSIM_k, DSSIM_k, LPIPS-Alex and LPIPS-VGG, and the dynamic-region crop and mask.

Spec: E16, App. B.1, D-29, D-52, D-67, D-68. Band: Orchestration (§2.1).
"""

import contextlib
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor

from openftgs.losses import ssim

__all__ = [
    "METRIC_NAMES",
    "average_metrics",
    "dssim_k",
    "dynamic_region",
    "lpips_backbone",
    "psnr",
    "ssim_k",
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


# Q-60: images are [..., H, W, 3] of one shape and floating dtype (Q-41); one value per image.
def _check_pair(pred: Tensor, gt: Tensor) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have one shape, got {list(pred.shape)}, {list(gt.shape)}"
        )
    if pred.dim() < 3 or pred.shape[-1] != 3:
        raise ValueError(f"images must be [..., H, W, 3], got {list(pred.shape)}")
    if not (pred.is_floating_point() and gt.is_floating_point()):
        raise ValueError(f"images must be floating, got {pred.dtype} and {gt.dtype}")


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
        # One call per image through the loss's public SSIM, which averages over its batch.
        values = [ssim(x[i], y[i], data_range=float(k)) for i in range(x.shape[0])]
        return torch.stack(values).reshape(lead)


def dssim_k(pred: Tensor, gt: Tensor, k: int) -> Tensor:
    """DSSIM_k = (1 − SSIM_k)/2 per image, a dissimilarity (E16, D-29, D-40)."""
    value = ssim_k(pred, gt, k)
    with _metric_mode():
        return (1.0 - value) / 2.0


def dynamic_region(pred: Tensor, gt: Tensor, mask: Tensor) -> tuple[Tensor, Tensor] | None:
    """The pair a dynamic-region metric compares (App. B.1, D-67): both images clamped, set to
    black outside the mask and cropped to the mask's bounding box. `mask` is [H, W] or [H, W, 1],
    bool or floating, binarized at 0.5 (≥ 0.5 is dynamic). None, with a warning, for an empty
    mask or a box narrower than 31 px on a side, which some metric can't evaluate (Q-61)."""
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
    if mask.dtype == torch.bool:
        dynamic = mask
    elif mask.is_floating_point():
        dynamic = mask >= _MASK_THRESHOLD
    else:
        raise ValueError(f"mask must be bool or floating, got {mask.dtype}")
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


def lpips_backbone(source: str | None) -> Literal["alex", "vgg"]:
    """The LPIPS backbone the paper reports for a dataset (D-29): AlexNet for Neural3DV, VGG
    otherwise, read from the scene manifest's `metadata.source` (Q-62)."""
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
