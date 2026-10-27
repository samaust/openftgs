# SPDX-License-Identifier: Apache-2.0
"""The harness's E16 metrics against openftgs.eval's: two implementations written separately
from E16 and E7, each checking the other (§3.3 L3 step 3). The harness's ꟻLIP runs its own
vendored copy on the CPU (Q-254), checked against openftgs's copy, which U-23 checks against
NVIDIA's file as published."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from ftgs_harness.metrics import dssim, flip, psnr, ssim
from openftgs import eval as ev
from openftgs._vendor.flip import LDRFLIPLoss


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


def _nchw(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]


def test_flip_of_identical_images_is_zero() -> None:
    """E16, U-23's criterion: ꟻLIP of identical images is 0 within 1e-6 (its stabilizer leaves
    a few 1e-7)."""
    _, gt = _pair(5)
    assert flip(gt, gt) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", [6, 7])
def test_flip_agrees_with_openftgs_vendored_copy_at_67_pixels_per_degree(seed: int) -> None:
    """E16, Q-254, Q-259: the harness's ꟻLIP is LDR-ꟻLIP at 67.0 pixels per degree on the
    clamped render against the ground truth, the mean of its error map; on the CPU it agrees
    with openftgs's vendored copy, which U-23 checks against NVIDIA's file as published, within
    U-23's 1e-6."""
    pred, gt = _pair(seed)
    with torch.no_grad():
        expected = LDRFLIPLoss()(_nchw(np.clip(pred, 0.0, 1.0)), _nchw(gt), 67.0)
    value = flip(pred, gt)
    assert 0.0 < value < 1.0
    assert value == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize("seed", [6, 7])
def test_flip_agrees_with_openftgs_eval(seed: int) -> None:
    """E16, Q-259 (#387): the harness's ꟻLIP and the one METRICS.json reports, `openftgs.eval`'s
    Evaluator on the CPU, agree within U-23's 1e-6, both at 67.0 pixels per degree on the
    clamped render; the published default, 67.02, would put them about 3·10⁻⁵ apart."""
    pred, gt = _pair(seed)
    entire, _ = ev.Evaluator(device="cpu")(torch.from_numpy(pred), torch.from_numpy(gt))
    assert flip(pred, gt) == pytest.approx(entire["flip"], abs=1e-6)


def test_flip_clamps_the_render_first() -> None:
    """E16: a render above 1 or below 0 is clamped to [0, 1] before ꟻLIP."""
    pred, gt = _pair(8)
    assert pred.max() > 1.0 and pred.min() < 0.0
    assert flip(pred, gt) == flip(np.clip(pred, 0.0, 1.0), gt)


_IMPORT_WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None  # as if torch weren't installed
import ftgs_harness.metrics as metrics
import numpy as np
image = np.full((16, 16, 3), 0.5, dtype=np.float32)
print(metrics.psnr(image * 0.9, image) > 0)
"""


def test_metrics_import_without_torch() -> None:
    """Q-201, Q-254: L0 to L2 run without the harness's extra, so its metrics module imports,
    and its PSNR and SSIM run, without torch; only ꟻLIP needs it."""
    done = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_TORCH],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True"]
