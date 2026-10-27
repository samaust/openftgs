# SPDX-License-Identifier: Apache-2.0
"""E16's image metrics, computed by the harness itself so that metric code cannot differ
between A and B (§3.3 L3 step 3), written from E16 and E7's wording: renders clamped to [0, 1];
PSNR = 10 log₁₀(1/MSE); SSIM_k with an 11×11 Gaussian window (σ = 1.5) over the valid region,
C₁ = (0.01k)², C₂ = (0.03k)², per channel and averaged over the map and channels;
DSSIM_k = (1 − SSIM_k)/2; ꟻLIP (LDR) at 67.0 pixels per degree, the mean of its error map,
through the harness's own vendored copy of NVIDIA's loss (Q-254), which needs torch, the
harness's optional extra; the rest needs no torch (Q-201)."""

import math
from typing import Any

import numpy as np
import scipy.ndimage

__all__ = ["FLIP_PIXELS_PER_DEGREE", "dssim", "flip", "psnr", "ssim"]

_WINDOW, _SIGMA = 11, 1.5
# Q-259: E16's and D-89's 67.0 pixels per degree, passed to ꟻLIP as written.
FLIP_PIXELS_PER_DEGREE = 67.0


def _clamped(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float64), 0.0, 1.0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """E16: 10 log₁₀(1/MSE) of the clamped render against the ground truth, [H, W, 3]."""
    mse = float(np.mean((_clamped(pred) - gt.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def _window() -> np.ndarray:
    x = np.arange(_WINDOW, dtype=np.float64) - (_WINDOW - 1) / 2.0
    w = np.exp(-(x**2) / (2.0 * _SIGMA**2))
    return w / w.sum()


def _filtered(image: np.ndarray) -> np.ndarray:
    """The Gaussian-weighted local mean over the valid region, per channel."""
    w = _window()
    out = scipy.ndimage.correlate1d(image, w, axis=0, mode="constant")
    out = scipy.ndimage.correlate1d(out, w, axis=1, mode="constant")
    half = _WINDOW // 2
    return np.asarray(out[half:-half, half:-half])


def ssim(pred: np.ndarray, gt: np.ndarray, k: int = 1) -> float:
    """E16's SSIM_k (E7's SSIM with data range k)."""
    x, y = _clamped(pred), gt.astype(np.float64)
    c1, c2 = (0.01 * k) ** 2, (0.03 * k) ** 2
    mx, my = _filtered(x), _filtered(y)
    sxx = _filtered(x * x) - mx * mx
    syy = _filtered(y * y) - my * my
    sxy = _filtered(x * y) - mx * my
    value = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return float(value.mean())


def dssim(pred: np.ndarray, gt: np.ndarray, k: int = 1) -> float:
    """E16: DSSIM_k = (1 − SSIM_k)/2."""
    return (1.0 - ssim(pred, gt, k)) / 2.0


def flip(pred: np.ndarray, gt: np.ndarray) -> float:
    """E16's ꟻLIP (LDR) at 67.0 pixels per degree (Q-259): the mean of its error map for the
    clamped render against the ground truth, [H, W, 3], in float32 on the CPU, through the
    harness's own vendored copy of NVIDIA's loss (Q-254), which needs torch."""
    import torch

    from ftgs_harness._vendor.flip import LDRFLIPLoss

    def nchw(image: np.ndarray) -> Any:
        planar = np.ascontiguousarray(image.astype(np.float32).transpose(2, 0, 1))
        return torch.from_numpy(planar)[None]

    with torch.no_grad():
        value = LDRFLIPLoss()(nchw(_clamped(pred)), nchw(gt), FLIP_PIXELS_PER_DEGREE)
    return float(value)
