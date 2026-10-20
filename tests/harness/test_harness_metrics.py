# SPDX-License-Identifier: Apache-2.0
"""The harness's E16 metrics against openftgs.eval's: two implementations written separately
from E16 and E7, each checking the other (§3.3 L3 step 3)."""

import numpy as np
import pytest
import torch

from ftgs_harness.metrics import Lpips, dssim, psnr, ssim
from openftgs import eval as ev


def _pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    gt = g.random((40, 52, 3)).astype(np.float32)
    pred = np.clip(gt + g.normal(scale=0.08, size=gt.shape), -0.2, 1.3).astype(np.float32)
    return pred, gt


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_psnr_and_ssim_agree_with_openftgs_eval(seed: int) -> None:
    pred, gt = _pair(seed)
    p, g = torch.from_numpy(pred), torch.from_numpy(gt)
    assert psnr(pred, gt) == pytest.approx(float(ev.psnr(p, g)), rel=1e-10)
    for k in (1, 2):
        assert ssim(pred, gt, k) == pytest.approx(float(ev.ssim_k(p, g, k)), abs=1e-6)
        assert dssim(pred, gt, k) == pytest.approx((1 - float(ev.ssim_k(p, g, k))) / 2, abs=1e-6)


def test_identical_images() -> None:
    _, gt = _pair(3)
    assert psnr(gt, gt) == float("inf") and ssim(gt, gt) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.weights
@pytest.mark.parametrize("net", ["alex", "vgg"])
def test_lpips_agrees_with_openftgs_eval(net: str) -> None:
    """E16, D-68: inputs mapped to [−1, 1] for both backbones."""
    from openftgs.losses import LPIPSLoss

    pred, gt = _pair(4)
    p, g = torch.from_numpy(np.clip(pred, 0, 1))[None], torch.from_numpy(gt)[None]
    expected = float(LPIPSLoss(net)(p, g).detach())  # type: ignore[arg-type]
    assert Lpips(net)(pred, gt) == pytest.approx(expected, abs=1e-5)
