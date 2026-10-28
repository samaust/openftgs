# SPDX-License-Identifier: Apache-2.0
"""The harness's E16 metrics against openftgs.eval's: two implementations written separately
from E16 and E7, each checking the other (§3.3 L3 step 3). The harness's ꟻLIP runs its own
vendored copy on the CPU (Q-254), checked against openftgs's copy, which U-23 checks against
NVIDIA's file as published, and, on a GPU, against that file directly."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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


# Q-252: NVIDIA's flip_loss.py at d3ab02a as published, kept beside the tests.
_PUBLISHED_FLIP = Path(__file__).resolve().parents[1] / "reference" / "flip" / "flip_loss.py"


def _published_flip() -> Any:
    """NVIDIA's LDR-ꟻLIP loss as published, loaded from the reference copy under a name of its
    own (Q-252); the file runs on CUDA only."""
    spec = importlib.util.spec_from_file_location("flip_published_for_harness", _PUBLISHED_FLIP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LDRFLIPLoss()


@pytest.mark.gpu
def test_flip_on_the_cpu_agrees_with_the_published_file_on_cuda() -> None:
    """E16, §3.3 L3 step 3, U-23's tolerance (Q-254, Q-252, Q-259; #424): on a fixed pair whose
    render leaves [0, 1] in places, the harness's ꟻLIP on the CPU is NVIDIA's LDR-ꟻLIP as
    published, run on CUDA in IEEE float32 (D-52) on the clamped render against the ground
    truth at 67.0 pixels per degree, within 1e-6. It pins the CPU path of the harness's device
    change to the published CUDA path directly, with no openftgs between them. 67.0 is written
    here rather than read from the harness, so that a change of the harness's constant, to the
    published default 67.02 say, about 3·10⁻⁵ away on this pair, fails the test."""
    g = np.random.default_rng(424)
    gt: np.ndarray = g.random((96, 128, 3)).astype(np.float32)
    pred: np.ndarray = (gt + g.normal(scale=0.12, size=gt.shape)).astype(np.float32)
    pred[:6] += 0.6
    pred[-5:, :9] -= 0.4
    assert pred.max() > 1.0 and pred.min() < 0.0
    value = flip(pred, gt)
    saved = (torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            render, truth = _nchw(np.clip(pred, 0.0, 1.0)).cuda(), _nchw(gt).cuda()
            expected = float(_published_flip()(render, truth, 67.0))
    finally:
        torch.backends.cudnn.allow_tf32 = saved[0]
        torch.set_float32_matmul_precision(saved[1])
    assert 0.0 < value < 1.0 and 0.0 < expected < 1.0
    assert value == pytest.approx(expected, abs=1e-6)


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
