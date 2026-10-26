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
import os
import shutil
import subprocess
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
# Source log rows 425 and 444: the published flip_loss.py and LICENSE at COMMIT.
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
    SHA-256 of source log rows 425 and 444)."""
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


def test_vendored_flip_is_deterministic_on_the_cpu() -> None:
    """#380: CPU tests that keep the default perceptual weight run ꟻLIP on the CPU, so two runs
    on the same pair give bitwise the same value and gradient."""
    test, reference = _pair(5)
    runs = []
    for _ in range(2):
        image = test.clone().requires_grad_(True)
        value = LDRFLIPLoss()(image, reference)
        value.backward()
        assert image.grad is not None
        runs.append((value.detach(), image.grad))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


@pytest.mark.gpu
def test_vendored_flip_on_the_cpu_agrees_with_cuda() -> None:
    """The device-agnostic change (#383): the same pair on the CPU and on CUDA, in IEEE
    float32, gives values within U-23's 1e-6."""
    test, reference = _pair(6)
    with _ieee_float32():
        on_cpu = LDRFLIPLoss()(test, reference)
        on_cuda = LDRFLIPLoss()(test.cuda(), reference.cuda())
    assert on_cuda.device.type == "cuda"
    torch.testing.assert_close(on_cuda.cpu(), on_cpu, rtol=0.0, atol=1e-6)


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


# Runs in a fresh process, so that every tensor the copy keeps is first made in inference mode.
_INFERENCE_THEN_TRAINING = """
import torch
from openftgs._vendor.flip import LDRFLIPLoss

generator = torch.Generator().manual_seed(3)
reference = torch.rand((1, 3, 24, 24), generator=generator)
test = reference + 0.1 * torch.randn((1, 3, 24, 24), generator=generator)
with torch.inference_mode():
    LDRFLIPLoss()(test, reference)
image = test.clone().requires_grad_(True)
LDRFLIPLoss()(image, reference).backward()
assert image.grad is not None and bool(torch.isfinite(image.grad).all())
print("trained")
"""


def test_vendored_flip_first_used_in_inference_mode_still_trains() -> None:
    """The caching changes (#383): the tensors the copy keeps are made outside inference
    mode, so a loss first evaluated under `torch.inference_mode()`, as a metric is, still
    trains: the next call with a gradient saves the same filters and constants for its
    backward pass."""
    source_root = Path(openftgs.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(source_root)}
    result = subprocess.run(
        [sys.executable, "-c", _INFERENCE_THEN_TRAINING],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "trained"


def _footprint(pixels_per_degree: float, colour: tuple[float, float, float]) -> tuple[int, int]:
    """The rows and columns, as (first, last) offsets from a one-pixel difference on a flat
    grey pair, of the error-map pixels whose ꟻLIP changes, on the CPU; asserts the change is
    centred."""
    from openftgs._vendor.flip import flip_loss

    loss = LDRFLIPLoss()
    size, centre = 96, 48
    reference = torch.full((1, 3, size, size), 0.5)
    test = reference.clone()
    test[0, :, centre, centre] = torch.tensor(colour)

    def error_map(image: torch.Tensor) -> torch.Tensor:
        test_opponent = flip_loss.color_space_transform(image, "srgb2ycxcz")
        reference_opponent = flip_loss.color_space_transform(reference, "srgb2ycxcz")
        errors: torch.Tensor = flip_loss.compute_ldrflip(
            test_opponent,
            reference_opponent,
            pixels_per_degree,
            loss.qc,
            loss.qf,
            loss.pc,
            loss.pt,
            loss.eps,
        )
        return errors[0, 0]

    changed = error_map(test) != error_map(reference)
    rows = changed.any(dim=1).nonzero().flatten() - centre
    columns = changed.any(dim=0).nonzero().flatten() - centre
    assert (int(rows.min()), int(rows.max())) == (int(columns.min()), int(columns.max()))
    return int(rows.min()), int(rows.max())


@pytest.mark.parametrize("pixels_per_degree", [(0.7 * 3840 / 0.7) * math.pi / 180, 67.0])
def test_flip_filter_support_at_67_pixels_per_degree(pixels_per_degree: float) -> None:
    """Q-257 (#383): ꟻLIP's filter support at 67.0 pixels per degree, measured with the copy,
    is 21 px. Worked by hand: the spatial filters' radius is ceil(3·√(0.04/(2π²))·ppd) =
    ceil(9.05) = 10, so they are 21 × 21, and the feature filters' is ceil(3·0.5·0.082·ppd) =
    ceil(8.24) = 9, so 19 × 19, at the published default (67.02) as at 67.0. Measured: a
    one-pixel chromatic difference on a flat grey pair changes ꟻLIP within 10 px of it and no
    farther (the blue-yellow filter reaches that far), an achromatic one within 9 px (the
    feature filters; the achromatic filter's tail at 10 px is below float32's resolution).
    VENDOR.json records the 21 px, which #387 carries into Q-61."""
    from openftgs._vendor.flip import flip_loss

    cpu = torch.device("cpu")
    for channel in ("A", "RG", "BY"):
        weights, radius = flip_loss.generate_spatial_filter(pixels_per_degree, channel, cpu)
        assert radius == 10 and tuple(weights.shape) == (1, 1, 21, 21)
    for feature in ("edge", "point"):
        weights, radius = flip_loss.generate_feature_filter(pixels_per_degree, feature, cpu)
        assert radius == 9 and tuple(weights.shape) == (1, 1, 19, 19)
    assert _footprint(pixels_per_degree, (0.9, 0.2, 0.7)) == (-10, 10)
    assert _footprint(pixels_per_degree, (0.6, 0.6, 0.6)) == (-9, 9)
    support = _vendor_json()["filter_support"]
    assert (support["side"], support["radius"]) == (21, 10)
    assert support["test"].endswith("::test_flip_filter_support_at_67_pixels_per_degree")


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
@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the recorder needs sys.monitoring (Python 3.12)"
)
def test_vendored_flip_forward_and_backward_on_cuda_make_no_synchronization() -> None:
    """§2.4 rule 3 (#383): after one call on a device, which builds and copies the filters and
    constants there and reads the colour maximum back once, a forward and backward pass of
    the vendored loss on CUDA make no host synchronization under
    `torch.cuda.set_sync_debug_mode("warn")`, by any route the recorder sees."""
    from sync_recorder import SyncRecorder, described

    device = torch.device("cuda:0")
    test, reference = (image.to(device) for image in _pair(4))
    loss = LDRFLIPLoss()
    warm_up = test.clone().requires_grad_(True)
    loss(warm_up, reference).backward()
    image = test.clone().requires_grad_(True)
    torch.cuda.synchronize()
    with SyncRecorder() as recorder:
        loss(image, reference).backward()
    torch.cuda.synchronize()
    assert recorder.syncs == [], described(recorder.syncs, __file__)
    assert recorder.unattributed == 0
    assert image.grad is not None


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
