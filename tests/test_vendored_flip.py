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
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

import openftgs
from openftgs._vendor.flip import HDRFLIPLoss, LDRFLIPLoss

REFERENCE = Path(__file__).resolve().parent / "reference" / "flip"  # Q-252
VENDORED = Path(openftgs.__file__).resolve().parent / "_vendor" / "flip"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vendor_flip.py"
COMMIT = "d3ab02a4425eb167d8a4021b382a2bafc6e6c838"
# Source log rows 425 and T383.1: the published flip_loss.py and LICENSE at COMMIT.
PUBLISHED_SHA256 = "bf2b83aeb2427a960abe1143d4436d48609f1cc4f256e685daa08d86cc07c4fc"
LICENSE_SHA256 = "13b955078ffb4a3215757038ab2e5fcf0cc66349d0990faec9adba8ad034e578"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vendor_flip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vendor_json() -> dict[str, Any]:
    record = json.loads((VENDORED / "VENDOR.json").read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


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


def test_reference_copy_is_the_published_file() -> None:
    """Q-252: U-23's reference copy is NVIDIA's file and license at d3ab02a, unchanged (the
    SHA-256 of source log rows 425 and T383.1)."""
    assert hashlib.sha256((REFERENCE / "flip_loss.py").read_bytes()).hexdigest() == (
        PUBLISHED_SHA256
    )
    assert hashlib.sha256((REFERENCE / "LICENSE").read_bytes()).hexdigest() == LICENSE_SHA256


def test_vendored_files_have_the_sha256_of_vendor_json() -> None:
    """D-65: VENDOR.json records the commit, the published file's SHA-256 and the copy's and
    its license's, which the files on disk have."""
    record = _vendor_json()
    assert record["commit"] == COMMIT
    (entry,) = record["files"]
    assert entry["path"] == "flip_loss.py"
    assert entry["upstream_sha256"] == PUBLISHED_SHA256
    (license_entry,) = record["licenses"]
    assert (license_entry["path"], license_entry["license"]) == ("LICENSE", "BSD-3-Clause")
    assert license_entry["sha256"] == LICENSE_SHA256
    for listed in (entry, license_entry):
        data = (VENDORED / listed["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == listed["sha256"], listed["path"]


def test_vendoring_script_check_mode_reproduces_the_copy(tmp_path: Path) -> None:
    """D-65, Q-252: the script's check mode rebuilds the copy from the reference copy, offline,
    and finds it byte for byte the same; a copy changed by one byte fails the check, and a
    changed reference copy stops the script with exit code 2."""
    script = _script()
    assert script.main(["--check", "--reference", str(REFERENCE), "--dest", str(VENDORED)]) == 0

    changed = tmp_path / "copy"
    shutil.copytree(VENDORED, changed, ignore=shutil.ignore_patterns("__pycache__"))
    source = changed / "flip_loss.py"
    source.write_bytes(source.read_bytes() + b"\n")
    assert script.main(["--check", "--reference", str(REFERENCE), "--dest", str(changed)]) == 1

    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    (reference / "flip_loss.py").write_bytes((REFERENCE / "flip_loss.py").read_bytes() + b"#")
    assert script.main(["--check", "--reference", str(reference), "--dest", str(VENDORED)]) == 2


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


def test_vendored_flip_builds_its_filters_once_per_device_and_size() -> None:
    """§2.4's perceptual-loss row (#383): ꟻLIP's spatial and feature filters are built once per
    device and size, the size being set by the pixels per degree alone: a second request
    returns the same tensors, another pixels per degree others, of another size."""
    from openftgs._vendor.flip import flip_loss

    cpu = torch.device("cpu")
    default = (0.7 * 3840 / 0.7) * math.pi / 180  # the published default, 67.02
    for channel in ("A", "RG", "BY"):
        first, radius = flip_loss.generate_spatial_filter(default, channel, cpu)
        again, _ = flip_loss.generate_spatial_filter(default, channel, cpu)
        assert again is first
        other, other_radius = flip_loss.generate_spatial_filter(30.0, channel, cpu)
        assert other is not first and other_radius < radius
    for feature in ("edge", "point"):
        first, radius = flip_loss.generate_feature_filter(default, feature, cpu)
        again, _ = flip_loss.generate_feature_filter(default, feature, cpu)
        assert again is first
        other, other_radius = flip_loss.generate_feature_filter(30.0, feature, cpu)
        assert other is not first and other_radius < radius


def test_vendored_flip_filters_first_built_in_inference_mode_serve_a_backward_pass() -> None:
    """The caching changes (#383): tensors the copy keeps are made outside inference mode, so
    a loss first evaluated under `torch.inference_mode()`, as a metric is, still trains: the
    next call with a gradient saves the same filters for its backward pass."""
    test, reference = _pair(3, (1, 3, 24, 24))
    pixels_per_degree = 41.0  # no other test uses it, so the filters are first built here
    with torch.inference_mode():
        LDRFLIPLoss()(test, reference, pixels_per_degree=pixels_per_degree)
    image = test.clone().requires_grad_(True)
    LDRFLIPLoss()(image, reference, pixels_per_degree=pixels_per_degree).backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())


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
