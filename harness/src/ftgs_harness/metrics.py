# SPDX-License-Identifier: Apache-2.0
"""E16's image metrics, computed by the harness itself so that metric code cannot differ
between A and B (§3.3 L3 step 3), written from E16 and E7's wording: renders clamped to [0, 1];
PSNR = 10 log₁₀(1/MSE); SSIM_k with an 11×11 Gaussian window (σ = 1.5) over the valid region,
C₁ = (0.01k)², C₂ = (0.03k)², per channel and averaged over the map and channels;
DSSIM_k = (1 − SSIM_k)/2; LPIPS-Alex and LPIPS-VGG on inputs mapped from [0, 1] to [−1, 1]
(D-68), through the [lpips] extra (Q-201)."""

import math
import warnings
from typing import Any

import numpy as np
import scipy.ndimage

__all__ = ["Lpips", "dssim", "psnr", "ssim"]

_WINDOW, _SIGMA = 11, 1.5


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


class Lpips:
    """LPIPS with the AlexNet or VGG backbone (lpips 0.1.4), inputs mapped to [−1, 1] (D-68).
    Needs the [lpips] extra; the network weights are downloaded on first use."""

    def __init__(self, net: str) -> None:
        import lpips
        import torch

        self._torch: Any = torch
        with warnings.catch_warnings():  # torchvision's notes on lpips's backbone arguments
            warnings.simplefilter("ignore", UserWarning)
            self._model: Any = lpips.LPIPS(net=net, verbose=False).eval()

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        torch = self._torch
        x = torch.from_numpy(_clamped(pred) * 2.0 - 1.0).float().permute(2, 0, 1)[None]
        y = torch.from_numpy(gt.astype(np.float64) * 2.0 - 1.0).float().permute(2, 0, 1)[None]
        with torch.no_grad():
            return float(self._model(x, y).item())
