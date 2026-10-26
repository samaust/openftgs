# SPDX-License-Identifier: Apache-2.0
"""The vendored ꟻLIP loss in ``openftgs._vendor.flip`` (D-65, D-89, U-23; Q-252; #383).

NVIDIA's ``src/pytorch/flip_loss.py`` at commit d3ab02a, made by ``scripts/vendor_flip.py``
from the reference copy beside these tests (``tests/reference/flip``, Q-252), with marked
changes that are device and caching only (D-89): device-agnostic tensors, the filters built
once per device and size (§2.4's perceptual-loss row), and the colour transforms' constants
and the colour maximum kept per device, so that no call after the first on a device
synchronizes (§2.4 rule 3).
"""

import contextlib
import importlib.util
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch

from openftgs._vendor.flip import HDRFLIPLoss, LDRFLIPLoss

REFERENCE = Path(__file__).resolve().parent / "reference" / "flip"  # Q-252


def _published() -> ModuleType:
    """NVIDIA's file as published, loaded from the reference copy under a name of its own
    (Q-252)."""
    spec = importlib.util.spec_from_file_location("flip_published", REFERENCE / "flip_loss.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _ieee_float32() -> Iterator[None]:
    """IEEE float32 convolutions and matrix products, as for reported metrics (D-52),
    restored afterwards."""
    saved = (torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = saved[0]
        torch.set_float32_matmul_precision(saved[1])


def _pair(seed: int, shape: tuple[int, ...] = (2, 3, 48, 64)) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed image pair in NCHW: a seeded reference in [0, 1] and a noisy test image that
    leaves [0, 1] in places, which ꟻLIP clamps (D-89)."""
    generator = torch.Generator().manual_seed(seed)
    reference = torch.rand(shape, generator=generator)
    test = reference + 0.1 * torch.randn(shape, generator=generator)
    return test, reference


def test_vendored_flip_runs_on_the_cpu() -> None:
    """The device-agnostic change (#383): on CPU tensors the loss returns a finite value in
    [0, 1] on the CPU, and its gradient with respect to the test image is finite."""
    test, reference = _pair(0)
    test.requires_grad_(True)
    value = LDRFLIPLoss()(test, reference)
    assert value.device.type == "cpu" and value.dim() == 0
    assert 0.0 < float(value.detach()) < 1.0
    value.backward()
    assert test.grad is not None and test.grad.device.type == "cpu"
    assert bool(torch.isfinite(test.grad).all())


@pytest.mark.gpu
def test_u23_vendored_flip_agrees_with_the_published_file_on_cuda(
    record_property: Callable[[str, object], None],
) -> None:
    """U-23 (D-89; Q-252): on a fixed image pair on CUDA, the vendored LDR-ꟻLIP and NVIDIA's
    file as published, loaded from the reference copy, agree within 1e-6, and so do their
    gradients with respect to the test image, to 1e-5 relative: the copy's changes are device
    and caching only. Both run in IEEE float32, as reported metrics do (D-52). The values were
    bitwise equal on the RTX 4090; the gradients differed by under 5e-10, as much as two runs of
    the published file differ from each other there (its CUDA backward isn't bitwise
    reproducible)."""
    published = _published()
    device = torch.device("cuda:0")
    test, reference = (image.to(device) for image in _pair(1))
    values: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    with _ieee_float32():
        for loss in (published.LDRFLIPLoss(), LDRFLIPLoss()):
            image = test.clone().requires_grad_(True)
            value = loss(image, reference)
            value.backward()
            assert image.grad is not None
            values.append(value.detach())
            gradients.append(image.grad)
    differences = {
        "value": float((values[1] - values[0]).abs()),
        "gradient": float((gradients[1] - gradients[0]).abs().max()),
    }
    record_property("max_abs_differences", json.dumps(differences))
    print(f"U-23 ꟻLIP oracle: largest differences {differences}")
    torch.testing.assert_close(values[1], values[0], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-5, atol=1e-9)
    assert 0.0 < float(values[0]) < 1.0  # the pair differs, and ꟻLIP is in [0, 1]


@pytest.mark.gpu
@pytest.mark.parametrize("tone_mapper", ["aces", "reinhard", "hable"])
def test_vendored_hdr_flip_agrees_with_the_published_file_on_cuda(tone_mapper: str) -> None:
    """The device-agnostic change reaches HDR-ꟻLIP's code too (#383): on a fixed HDR pair on
    CUDA, the vendored and published HDR-ꟻLIP agree within U-23's 1e-6 for each tone mapper."""
    published = _published()
    device = torch.device("cuda:0")
    test, reference = (4.0 * image.clamp(min=0.0).to(device) for image in _pair(2))
    with _ieee_float32():
        expected = published.HDRFLIPLoss()(test, reference, tone_mapper=tone_mapper)
        actual = HDRFLIPLoss()(test, reference, tone_mapper=tone_mapper)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
