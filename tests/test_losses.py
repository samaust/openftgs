# SPDX-License-Identifier: Apache-2.0
"""openftgs.losses (E7, E8, E9; D-7 to D-11, D-37, D-38, D-52, D-68, D-77, D-78; Q-41, Q-42):
U-9, U-10, U-11 and the added tests of PRD #53.

Expected values are worked by hand from the spec, come from scikit-image's
`structural_similarity` or from a brute-force float64 SSIM written in this file from Wang et al.
2004 and E7, or from E8's closed-form gradient. Images are in §1.4's layout, [..., H, W, 3].
"""

import copy
import importlib
import sys
import warnings
from collections.abc import Iterator
from typing import get_args

import numpy as np
import pytest
import torch
from builders import CAM_A, UNIT_TIME, model_from_rows, pinhole, random_model
from skimage.metrics import structural_similarity
from torch import Tensor

import openftgs.losses
from openftgs.config import LossConfig, OptimConfig, PercNet
from openftgs.losses import LPIPSLoss, PerceptualLoss, l1_loss, reg4d_loss, ssim, total_loss
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera

F32 = torch.float32
F64 = torch.float64

PARAM_NAMES = (  # §1.1, written out: the tests' own expectation
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
)


# U-9: L1 (E7, D-7)


@pytest.mark.parametrize("shape", [(5, 7, 3), (2, 5, 7, 3)])
def test_u9_l1_of_constant_images_is_their_difference(shape: tuple[int, ...]) -> None:
    """U-9: L1 between constant images a and b is |a − b| (mean over pixels and channels)."""
    pred = torch.full(shape, 0.75)
    gt = torch.full(shape, 0.25)
    assert l1_loss(pred, gt).item() == 0.5
    assert l1_loss(gt, pred).item() == 0.5


# U-9: SSIM (E7, D-8)


def _image_pair(
    shape: tuple[int, ...], *, seed: int, correlated: bool, dtype: torch.dtype = F64
) -> tuple[Tensor, Tensor]:
    """Two images in [0, 1]: independent noise, or a target and the target plus noise."""
    gen = torch.Generator().manual_seed(seed)
    gt = torch.rand(shape, generator=gen, dtype=F64)
    if correlated:
        pred = (gt + 0.1 * torch.randn(shape, generator=gen, dtype=F64)).clamp(0.0, 1.0)
    else:
        pred = torch.rand(shape, generator=gen, dtype=F64)
    return pred.to(dtype), gt.to(dtype)


def _skimage_ssim(pred: Tensor, gt: Tensor) -> float:
    """scikit-image's SSIM with Wang et al.'s settings (Gaussian weights, σ = 1.5, population
    covariance, data range 1); its mean runs over the valid region. A batch is the mean of its
    images."""
    a = pred.to(F64).reshape(-1, *pred.shape[-3:]).numpy()
    b = gt.to(F64).reshape(-1, *gt.shape[-3:]).numpy()
    values = [
        structural_similarity(  # type: ignore[no-untyped-call]
            x,
            y,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        for x, y in zip(a, b, strict=True)
    ]
    return float(np.mean(values))


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("correlated", [True, False])
@pytest.mark.parametrize(
    ("shape", "seed"), [((32, 40, 3), 0), ((27, 19, 3), 1), ((2, 24, 30, 3), 2)]
)
def test_u9_ssim_matches_scikit_image(
    shape: tuple[int, ...], seed: int, correlated: bool, dtype: torch.dtype
) -> None:
    """U-9: SSIM (valid padding, data range 1) agrees with scikit-image's structural_similarity
    to 1e-4."""
    pred, gt = _image_pair(shape, seed=seed, correlated=correlated, dtype=dtype)
    value = ssim(pred, gt)
    assert value.shape == () and value.dtype == dtype
    torch.testing.assert_close(value.item(), _skimage_ssim(pred, gt), rtol=0.0, atol=1e-4)


def _brute_force_ssim(pred: np.ndarray, gt: np.ndarray, *, padding: str) -> float:
    """SSIM of two [H, W, 3] float64 images written from Wang et al. 2004 and E7, one window at a
    time: an 11×11 Gaussian window (σ = 1.5) normalized to sum 1, population moments under it,
    C₁ = 0.01², C₂ = 0.03². "valid" visits only windows wholly inside the image; "same" visits
    every pixel, with zeros outside the image (Q-42)."""
    r = 5
    offsets = np.arange(-r, r + 1, dtype=np.float64)
    g1 = np.exp(-(offsets**2) / (2 * 1.5**2))
    w = np.outer(g1, g1)
    w /= w.sum()
    h, wd, _ = pred.shape
    if padding == "same":
        pad = ((r, r), (r, r), (0, 0))
        x_img, y_img = np.pad(pred, pad), np.pad(gt, pad)
        rows, cols = range(h), range(wd)
    else:
        x_img, y_img = pred, gt
        rows, cols = range(h - 2 * r), range(wd - 2 * r)
    values = []
    for i in rows:
        for j in cols:
            for c in range(3):
                x = x_img[i : i + 2 * r + 1, j : j + 2 * r + 1, c]
                y = y_img[i : i + 2 * r + 1, j : j + 2 * r + 1, c]
                mx, my = (w * x).sum(), (w * y).sum()
                vx = (w * (x - mx) ** 2).sum()
                vy = (w * (y - my) ** 2).sum()
                cxy = (w * (x - mx) * (y - my)).sum()
                c1, c2 = 0.01**2, 0.03**2
                values.append(
                    (2 * mx * my + c1) * (2 * cxy + c2) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
                )
    return float(np.mean(values))


@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("correlated", [True, False])
def test_ssim_matches_a_brute_force_evaluation(padding: str, correlated: bool) -> None:
    """Added (E7, D-8, Q-42): SSIM in float64 equals a window-by-window evaluation of its
    definition to 1e-10, for valid and zero ("same") padding."""
    pred, gt = _image_pair((16, 18, 3), seed=3, correlated=correlated)
    value = ssim(pred, gt, padding=padding)  # type: ignore[arg-type]
    expected = _brute_force_ssim(pred.numpy(), gt.numpy(), padding=padding)
    torch.testing.assert_close(value.item(), expected, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_u9_ssim_of_an_image_with_itself_gives_a_zero_ssim_loss(
    padding: str, dtype: torch.dtype
) -> None:
    """U-9: L_ssim = 1 − SSIM(I, I) = 0."""
    image, _ = _image_pair((2, 20, 23, 3), seed=4, correlated=False, dtype=dtype)
    assert 1.0 - ssim(image, image.clone(), padding=padding).item() == 0.0  # type: ignore[arg-type]


# Q-41: input validation


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (torch.zeros(4, 5, 3), torch.zeros(5, 4, 3)),
        (torch.zeros(4, 5, 4), torch.zeros(4, 5, 4)),
        (torch.zeros(5, 3), torch.zeros(5, 3)),
        (torch.zeros(4, 5, 3, dtype=torch.int64), torch.zeros(4, 5, 3, dtype=torch.int64)),
        (torch.zeros(4, 5, 3, dtype=F64), torch.zeros(4, 5, 3)),
    ],
    ids=["shapes", "channels", "rank", "integer", "dtypes"],
)
def test_image_pairs_must_match_and_be_float_rgb(pred: Tensor, gt: Tensor) -> None:
    """Q-41: pred and gt share one [..., H, W, 3] shape and one floating dtype."""
    with pytest.raises(ValueError):
        l1_loss(pred, gt)
    with pytest.raises(ValueError):
        ssim(pred, gt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 10},
        {"window": 0},
        {"window": -3},
        {"window": True},
        {"window": 11.0},
        {"sigma": 0.0},
        {"sigma": -1.5},
        {"data_range": 0.0},
        {"padding": "reflect"},
    ],
)
def test_ssim_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    """Q-41: the window is an odd int ≥ 1, σ and the data range are positive, and the padding is
    valid or same."""
    image = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        ssim(image, image, **kwargs)  # type: ignore[arg-type]


def test_ssim_with_valid_padding_needs_an_image_as_large_as_the_window() -> None:
    """Q-41: "valid" padding on an image smaller than the window would leave an empty map."""
    image = torch.rand(10, 16, 3, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        ssim(image, image)
    with pytest.raises(ValueError):
        ssim(image.transpose(0, 1), image.transpose(0, 1))
    assert ssim(image, image, padding="same").item() == 1.0
    corner = image[:, :11]
    assert ssim(corner, corner.clone(), window=9).item() == 1.0


# Q-42: the SSIM paths on the GPU (fused-ssim when installed, the PyTorch fallback otherwise)


def _relative(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def _cuda_ssim_against_float64(padding: str) -> None:
    """SSIM and its gradient for CUDA float32 images against the float64 evaluation on the CPU:
    value to 1e-5 relative, gradient to 1e-4 relative L2. TF32 is off, so the fallback's
    convolution runs in plain float32 (D-52)."""
    pred64, gt64 = _image_pair((2, 40, 52, 3), seed=5, correlated=True)
    pred64.requires_grad_()
    expected = ssim(pred64, gt64, padding=padding)  # type: ignore[arg-type]
    (expected_grad,) = torch.autograd.grad(expected, pred64)
    pred = pred64.detach().to("cuda", F32).requires_grad_()
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        value = ssim(pred, gt64.to("cuda", F32), padding=padding)  # type: ignore[arg-type]
        (grad,) = torch.autograd.grad(value, pred)
    assert value.device.type == "cuda" and value.dtype == F32 and value.shape == ()
    torch.testing.assert_close(value.item(), expected.item(), rtol=1e-5, atol=0.0)
    assert _relative(grad.cpu(), expected_grad) <= 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_ssim_on_the_gpu_agrees_with_float64(padding: str) -> None:
    """Added (§2.4 SSIM row, D-52, Q-42): on CUDA float32 images, the path ssim takes
    (fused-ssim when installed) gives the float64 value and gradient."""
    _cuda_ssim_against_float64(padding)


@pytest.mark.gpu
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_fused_ssim_agrees_with_float64(padding: str) -> None:
    """Added (§2.4 SSIM row, Q-2, Q-42): with fused-ssim installed, CUDA float32 SSIM matches the
    float64 fallback in value and gradient."""
    pytest.importorskip("fused_ssim")
    _cuda_ssim_against_float64(padding)


@pytest.fixture
def without_fused_ssim(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """openftgs.losses reloaded with fused-ssim made unimportable; reloaded again afterwards."""
    monkeypatch.setitem(sys.modules, "fused_ssim", None)
    importlib.reload(openftgs.losses)
    yield
    monkeypatch.undo()
    importlib.reload(openftgs.losses)


@pytest.mark.gpu
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_ssim_without_fused_ssim_falls_back_on_the_gpu(
    without_fused_ssim: None, padding: str
) -> None:
    """Added (Q-2's optional-import test, Q-42): without fused-ssim, SSIM on CUDA still works and
    gives the float64 value and gradient."""
    assert openftgs.losses.ssim is not ssim
    ssim_reloaded = openftgs.losses.ssim
    pred64, gt64 = _image_pair((24, 30, 3), seed=6, correlated=True)
    pred64.requires_grad_()
    expected = ssim(pred64, gt64, padding=padding)  # type: ignore[arg-type]
    (expected_grad,) = torch.autograd.grad(expected, pred64)
    pred = pred64.detach().to("cuda", F32).requires_grad_()
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        value = ssim_reloaded(
            pred,
            gt64.to("cuda", F32),
            padding=padding,  # type: ignore[arg-type]
        )
        (grad,) = torch.autograd.grad(value, pred)
    torch.testing.assert_close(value.item(), expected.item(), rtol=1e-5, atol=0.0)
    assert _relative(grad.cpu(), expected_grad) <= 1e-4


@pytest.mark.gpu
def test_ssim_gives_the_target_a_gradient_when_it_requires_one() -> None:
    """Added (Q-42): fused-ssim differentiates only its first image, so a target that requires a
    gradient takes the fallback and receives the float64 gradient."""
    pred64, gt64 = _image_pair((24, 30, 3), seed=7, correlated=True)
    gt64.requires_grad_()
    (expected,) = torch.autograd.grad(ssim(pred64, gt64), gt64)
    gt = gt64.detach().to("cuda", F32).requires_grad_()
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        (grad,) = torch.autograd.grad(ssim(pred64.to("cuda", F32), gt), gt)
    assert _relative(grad.cpu(), expected) <= 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("padding", ["valid", "same"])
def test_u9_ssim_of_an_image_with_itself_on_the_gpu(padding: str) -> None:
    """U-9: L_ssim = 1 − SSIM(I, I) = 0 on CUDA float32 too."""
    image, _ = _image_pair((2, 20, 23, 3), seed=4, correlated=False, dtype=F32)
    image = image.cuda()
    value = ssim(image, image.clone(), padding=padding)  # type: ignore[arg-type]
    assert 1.0 - value.item() == 0.0


@pytest.mark.gpu
def test__fused_ssim_finds_the_installed_package() -> None:
    """The private loader returns fused-ssim's function when the package imports."""
    fused = pytest.importorskip("fused_ssim")
    assert openftgs.losses._fused_ssim() is fused.fused_ssim


@pytest.mark.gpu
def test__fused_ssim_is_called_only_under_q42_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q-42: fused-ssim runs only for CUDA float32 images, window 11, σ 1.5, data range 1, and a
    target that doesn't require a gradient; `train` follows grad mode and pred.requires_grad."""
    calls: list[tuple[tuple[int, ...], str, bool]] = []

    def stub(img1: Tensor, img2: Tensor, padding: str = "same", train: bool = True) -> Tensor:
        assert img1.is_contiguous() and img2.is_contiguous()
        calls.append((tuple(img1.shape), padding, train))
        return img1.sum() * 0.0 + 0.5

    monkeypatch.setattr(openftgs.losses, "_fused_ssim", lambda: stub)
    pred, gt = (x.to("cuda", F32) for x in _image_pair((2, 16, 20, 3), seed=8, correlated=True))
    assert ssim(pred, gt, padding="same").item() == 0.5
    assert calls == [((2, 3, 16, 20), "same", False)]
    assert ssim(pred.requires_grad_(), gt).item() == 0.5
    assert calls[-1] == ((2, 3, 16, 20), "valid", True)
    with torch.no_grad():
        ssim(pred, gt)
    assert calls[-1][2] is False
    calls.clear()
    ssim(pred, gt, window=9)
    ssim(pred, gt, sigma=1.0)
    ssim(pred, gt, data_range=2.0)
    ssim(pred, gt.clone().requires_grad_())
    ssim(pred.double(), gt.double())
    ssim(pred.cpu(), gt.cpu())
    assert calls == []


# U-9: LPIPS (E7, D-9, D-68; Q-41)


@pytest.fixture(scope="module")
def lpips_nets() -> dict[str, LPIPSLoss]:
    """One LPIPSLoss per backbone, built once (the trunk weights are downloaded)."""
    return {net: LPIPSLoss(net) for net in ("vgg", "alex")}


def test_lpips_rejects_an_unknown_backbone() -> None:
    """Q-41: the backbone is vgg or alex, checked before anything is downloaded."""
    with pytest.raises(ValueError):
        LPIPSLoss("squeeze")  # type: ignore[arg-type]


@pytest.mark.weights
@pytest.mark.parametrize("net", ["vgg", "alex"])
def test_u9_lpips_of_an_image_with_itself_is_zero(
    lpips_nets: dict[str, LPIPSLoss], net: str
) -> None:
    """U-9: LPIPS(I, I) = 0, also after .train(), which must not switch on lpips's dropout."""
    loss = lpips_nets[net]
    image, _ = _image_pair((2, 32, 40, 3), seed=9, correlated=False, dtype=F32)
    assert loss(image, image.clone()).item() == 0.0
    loss.train()
    try:
        assert loss(image, image.clone()).item() == 0.0
        pred, gt = _image_pair((32, 40, 3), seed=10, correlated=True, dtype=F32)
        assert loss(pred, gt).item() == loss(pred, gt).item()
    finally:
        loss.eval()


@pytest.mark.weights
@pytest.mark.parametrize("net", ["vgg", "alex"])
def test_u9_lpips_maps_inputs_from_unit_range_to_plus_minus_one(
    lpips_nets: dict[str, LPIPSLoss], net: str
) -> None:
    """U-9 (D-9, D-68): LPIPSLoss on [0, 1] images equals lpips 0.1.4's network on the images
    mapped to [−1, 1] by 2x − 1, exactly, as a mean over the batch. The reference gets the
    [B, 3, H, W] view of the same channels-last memory, since the CPU convolutions pick their
    kernels by memory format and would otherwise differ by rounding (about 1e-8)."""
    import lpips

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torchvision's `pretrained` deprecation
        reference = lpips.LPIPS(net=net, verbose=False)
    pred, gt = _image_pair((2, 32, 40, 3), seed=11, correlated=True, dtype=F32)
    nchw = [x.permute(0, 3, 1, 2) for x in (pred, gt)]
    with torch.no_grad():
        expected = reference(2 * nchw[0] - 1, 2 * nchw[1] - 1).mean()
        unmapped = reference(nchw[0], nchw[1]).mean()
        value = lpips_nets[net](pred, gt)
    assert value.shape == ()
    assert value.item() == expected.item()
    assert value.item() != unmapped.item()


@pytest.mark.weights
@pytest.mark.parametrize("net", ["vgg", "alex"])
def test_lpips_without_mapping_feeds_the_unit_range_images_to_the_network(
    lpips_nets: dict[str, LPIPSLoss], net: str
) -> None:
    """D-68, Q-58: with map_inputs=False, LPIPSLoss equals lpips 0.1.4's network on the [0, 1]
    images themselves (lpips_vgg_raw), exactly, as a mean over the batch; the default still
    maps."""
    import lpips

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torchvision's `pretrained` deprecation
        reference = lpips.LPIPS(net=net, verbose=False)
    pred, gt = _image_pair((2, 32, 40, 3), seed=12, correlated=True, dtype=F32)
    nchw = [x.permute(0, 3, 1, 2) for x in (pred, gt)]
    with torch.no_grad():
        expected = reference(nchw[0], nchw[1]).mean()
        value = lpips_nets[net](pred, gt, map_inputs=False)
        mapped = lpips_nets[net](pred, gt)
    assert value.shape == ()
    assert value.item() == expected.item()
    assert mapped.item() != value.item()


@pytest.mark.weights
@pytest.mark.parametrize("flag", [1, 0, None, "yes"])
def test_lpips_map_inputs_must_be_a_bool(lpips_nets: dict[str, LPIPSLoss], flag: object) -> None:
    """Q-58: map_inputs is a bool."""
    image = torch.zeros(32, 32, 3)
    with pytest.raises(ValueError, match="map_inputs"):
        lpips_nets["vgg"](image, image, map_inputs=flag)


@pytest.mark.weights
def test_lpips_weights_are_frozen_and_only_the_prediction_gets_a_gradient(
    lpips_nets: dict[str, LPIPSLoss],
) -> None:
    """Q-41, D-52: every LPIPS parameter is frozen; the loss differentiates the prediction, and a
    target that doesn't require a gradient gets none."""
    loss = lpips_nets["vgg"]
    assert all(not p.requires_grad for p in loss.parameters())
    pred, gt = _image_pair((32, 40, 3), seed=12, correlated=True, dtype=F32)
    pred.requires_grad_()
    loss(pred, gt).backward()
    assert pred.grad is not None and bool(pred.grad.abs().sum() > 0)
    assert gt.grad is None
    assert all(p.grad is None for p in loss.parameters())


@pytest.mark.gpu
@pytest.mark.weights
def test_u9_lpips_of_an_image_with_itself_is_zero_on_the_gpu() -> None:
    """U-9: LPIPS(I, I) = 0 on CUDA too."""
    loss = LPIPSLoss("vgg").cuda()
    image, _ = _image_pair((2, 32, 40, 3), seed=13, correlated=False, dtype=F32)
    image = image.cuda()
    assert loss(image, image.clone()).item() == 0.0


# U-9: the perceptual networks (E7, D-9, D-89; Q-41)

TRUNK_NETWORKS = (  # D-89's table, written out: the networks built on pretrained weights
    "vgg16_oxford",
    "alexnet_bvlc",
    "dinov2_s14",
    "dinov2_b14",
    "dino_s16",
    "dino_b16",
)
WEIGHT_FREE_NETWORKS: tuple[PercNet, ...] = ("flip", "haarpsi")  # D-89: the two without weights
# D-89's two Caffe trunks (#385), whose weights are downloaded at first use, hence `weights`
# (Q-7); they compute in float32, their weights' dtype (Q-263).
CAFFE_NETWORKS: tuple[PercNet, ...] = ("vgg16_oxford", "alexnet_bvlc")
CAFFE_PARAMS = [pytest.param(net, marks=pytest.mark.weights) for net in CAFFE_NETWORKS]


@pytest.mark.parametrize("net", ["vgg", "alex", "lpips", "FLIP", ""])
def test_perceptual_loss_rejects_a_name_outside_d89(net: str) -> None:
    """§2.1, D-89: `PerceptualLoss(net)` takes one of D-89's eight names; LPIPS's backbones, the
    names before revision 549o, are not among them."""
    with pytest.raises(ValueError, match="D-89"):
        PerceptualLoss(net)  # type: ignore[arg-type]


@pytest.mark.parametrize("net", [n for n in TRUNK_NETWORKS if n not in CAFFE_NETWORKS])
def test_perceptual_loss_trunk_networks_are_not_built_yet(net: str) -> None:
    """#381: the four DINO networks of D-89 are built by #386; until then building one raises
    an error naming its ticket. #386 replaces this test; #385 built the Caffe trunks."""
    with pytest.raises(NotImplementedError, match="#386"):
        PerceptualLoss(net)  # type: ignore[arg-type]


def _with_noise(image: Tensor, *, seed: int, sigma: float = 16 / 255) -> Tensor:
    """The image plus seeded Gaussian noise of standard deviation σ, clipped to [0, 1]."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(image.shape, generator=gen, dtype=F64)
    return (image.double() + sigma * noise).clamp(0.0, 1.0).to(image.dtype)


def _pattern(shape: tuple[int, ...], phase: float) -> Tensor:
    """A closed-form textured image in [0.05, 0.95], float64: three sinusoids of different
    frequencies and orientations per channel, shifted by `phase` and by the batch index. The
    script that computed the piq values below builds the same images."""
    h, w = shape[-3], shape[-2]
    batch = 1
    for n in shape[:-3]:
        batch *= n
    k = torch.arange(batch, dtype=F64).view(batch, 1, 1, 1)
    y = torch.arange(h, dtype=F64).view(1, h, 1, 1)
    x = torch.arange(w, dtype=F64).view(1, w, 1)[None]
    c = torch.arange(3, dtype=F64).view(1, 1, 1, 3)
    image = (
        0.5
        + 0.2 * torch.sin(0.9 * x + 0.4 * y + 2.0 * c + phase + k)
        + 0.15 * torch.sin(0.13 * x - 0.29 * y + c * phase)
        + 0.1 * torch.sin(2.3 * x + 1.7 * y + phase + 0.5 * k)
    )
    return image.reshape(shape)


# Q-262: ꟻLIP runs in float32 only (Q-261); HaarPSI's identity case runs in float64, where
# its float32 rounding (the logit of float32's sigmoid(4.2)) doesn't reach U-9's bound.
@pytest.mark.parametrize(
    ("net", "dtype"),
    [
        ("flip", F32),
        ("haarpsi", F64),
        *(pytest.param(net, F32, marks=pytest.mark.weights) for net in CAFFE_NETWORKS),
    ],
)
def test_u9_perceptual_network_of_an_image_with_itself(net: PercNet, dtype: torch.dtype) -> None:
    """U-9 (D-89): on identical images a network gives 0, or at most 10⁻⁵ of its value under
    Gaussian noise of σ = 16/255 (float32 rounding, ꟻLIP's stabilizer), on each of five images:
    uniform noise and closed-form textures, one and two to an image batch. ꟻLIP is checked in
    float32, its only dtype (Q-261), and HaarPSI in float64 (Q-262)."""
    loss = PerceptualLoss(net)
    images = [
        _image_pair((2, 32, 40, 3), seed=40, correlated=False)[0],
        _image_pair((64, 96, 3), seed=40, correlated=False)[0],
        _pattern((2, 32, 40, 3), 0.0),
        _pattern((64, 96, 3), 0.0),
        _pattern((128, 160, 3), 0.3),
    ]
    ratios = []
    for image in (image.to(dtype) for image in images):
        same = loss(image, image.clone())
        noise = loss(_with_noise(image, seed=41), image)
        assert same.shape == () and same.dtype == dtype
        assert noise.item() > 0.0
        ratios.append(abs(same.item()) / noise.item())
    assert max(ratios) <= 1e-5, ratios


@pytest.mark.parametrize("net", [*WEIGHT_FREE_NETWORKS, *CAFFE_PARAMS])
def test_perceptual_loss_is_frozen_in_evaluation_mode_and_differentiates_only_the_render(
    net: PercNet,
) -> None:
    """D-89 (common to all, Q-41): the network is frozen and in evaluation mode, also after
    `.train()`; the render gets a gradient and the reference none, even when it asks for one."""
    loss = PerceptualLoss(net)
    loss.train()
    assert not any(module.training for module in loss.modules())
    assert all(not p.requires_grad for p in loss.parameters())
    pred, gt = _image_pair((32, 40, 3), seed=42, correlated=True, dtype=F32)
    pred.requires_grad_()
    gt.requires_grad_()
    loss(pred, gt).backward()
    assert pred.grad is not None and bool(pred.grad.abs().sum() > 0)
    assert gt.grad is None


def test_d89_flip_is_the_vendored_ldr_flip_at_67_pixels_per_degree() -> None:
    """D-89's `flip` row: NVIDIA's LDR-ꟻLIP loss (the vendored copy, #383) at 67.0 pixels per
    degree, the mean of its error map over the batch, on the [B, 3, H, W] views of the images,
    bitwise; a render above 1 gives the loss of the render clamped to [0, 1], since the loss
    clamps both images. The published default, 67.02 pixels per degree, gives another value."""
    from openftgs._vendor.flip import LDRFLIPLoss

    pred, gt = _image_pair((2, 32, 40, 3), seed=44, correlated=True, dtype=F32)
    pred[:, 4:12, 6:18] += 0.4
    nchw = [x.permute(0, 3, 1, 2) for x in (pred, gt)]
    reference = LDRFLIPLoss()
    value = PerceptualLoss("flip")(pred, gt)
    assert value.item() == reference(*nchw, pixels_per_degree=67.0).item()
    assert value.item() != reference(*nchw).item()
    assert value.item() == PerceptualLoss("flip")(pred.clamp(0.0, 1.0), gt).item()


# 1 − HaarPSI from piq 0.8.0's `haarpsi` at its defaults (3 scales, subsampled, c = 30, α = 4.2,
# data range 1), in float64, on the render clamped to [0, 1] and the reference (source log
# 448): (shape, phase of the render, a patch raised by 0.3, the value).
PIQ_HAARPSI = [
    ((2, 33, 47, 3), 0.35, True, 0.35540822979323505),
    ((40, 64, 3), 1.1, False, 0.5703410520768328),
    ((16, 17, 3), 0.7, False, 0.4269709218029545),
]


@pytest.mark.parametrize(("shape", "phase", "raised", "expected"), PIQ_HAARPSI)
def test_d89_haarpsi_is_one_minus_piq_haarpsi_with_the_render_clamped(
    shape: tuple[int, ...], phase: float, raised: bool, expected: float
) -> None:
    """D-89's `haarpsi` row and §2.3's HaarPSI row: 1 − HaarPSI (3 scales, subsampled, c = 30,
    α = 4.2) as piq 0.8.0 computes it, the render clamped to [0, 1], as the mean over the batch;
    against piq's values in float64, to 1e-12. Odd sides take piq's padding before the
    subsampling, and 16 pixels is the smallest side 3 scales allow."""
    gt = _pattern(shape, 0.0)
    pred = _pattern(shape, phase)
    if raised:
        pred[..., 5:20, 10:30, :] += 0.3
        assert bool((pred > 1.0).any())
    value = PerceptualLoss("haarpsi")(pred, gt)
    assert value.shape == () and value.dtype == F64
    torch.testing.assert_close(value.item(), expected, rtol=0.0, atol=1e-12)


def test__haarpsi_constants_built_in_inference_mode_still_serve_a_backward() -> None:
    """The YIQ matrix and Haar filters are kept per device and dtype: when the first call is a
    metric's, under `torch.inference_mode()`, a later training call still differentiates the
    render, since the kept tensors aren't inference tensors."""
    openftgs.losses._haarpsi_constants.cache_clear()
    gt = _pattern((32, 40, 3), 0.0)
    pred = _pattern((32, 40, 3), 0.5)
    loss = PerceptualLoss("haarpsi")
    with torch.inference_mode():
        metric = loss(pred, gt)
    pred.requires_grad_()
    value = loss(pred, gt)
    value.backward()
    assert value.item() == metric.item()
    assert pred.grad is not None and bool(torch.isfinite(pred.grad).all())


@pytest.mark.parametrize("shape", [(15, 16, 3), (16, 15, 3), (2, 15, 40, 3)])
def test_haarpsi_needs_sides_of_16_pixels(shape: tuple[int, ...]) -> None:
    """Q-261: as in piq, three Haar scales need a height and width of at least 2^(3+1) = 16
    pixels, checked before the subsampling; a smaller image is a ValueError."""
    image = _pattern(shape, 0.0)
    with pytest.raises(ValueError, match="16"):
        PerceptualLoss("haarpsi")(image, image.clone())


def test_haarpsi_checks_no_values() -> None:
    """Q-261: piq's check that both images lie in [0, 1] is gone, since it reads them back
    to the host: a render outside [0, 1] is clamped (D-89), and a reference outside it is used
    as given, without an error."""
    gt = _pattern((32, 40, 3), 0.0)
    pred = _pattern((32, 40, 3), 0.5)
    outside = gt.clone()
    outside[4:10, 4:10] = torch.tensor([1.5, -0.25, 0.5], dtype=F64)
    value = PerceptualLoss("haarpsi")(pred, outside)
    assert bool(torch.isfinite(value))
    assert value.item() != PerceptualLoss("haarpsi")(pred, gt).item()
    low = pred - 0.6
    assert bool((low < 0.0).any())
    assert PerceptualLoss("haarpsi")(low, gt).item() == (
        PerceptualLoss("haarpsi")(low.clamp(0.0, 1.0), gt).item()
    )


@pytest.mark.parametrize("dtype", [F64, torch.float16, torch.bfloat16])
def test_flip_takes_float32_images_only(dtype: torch.dtype) -> None:
    """Q-261: NVIDIA's ꟻLIP loss computes with float32 constants, so `flip` refuses images
    of another dtype with a ValueError, before any work, rather than casting them."""
    image = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(45)).to(dtype)
    with pytest.raises(ValueError, match="float32"):
        PerceptualLoss("flip")(image, image.clone())


@pytest.mark.parametrize("net", [*WEIGHT_FREE_NETWORKS, *CAFFE_PARAMS])
def test_u9_perceptual_network_gradient_is_finite_on_a_render_above_one(net: PercNet) -> None:
    """U-9 (E7: Î isn't clamped, so it can exceed 1): the gradient with respect to a render
    with a raised patch, above 1 where the image is bright, is finite everywhere and nonzero."""
    pred, gt = _image_pair((2, 32, 40, 3), seed=43, correlated=True, dtype=F32)
    pred = pred.clone()
    pred[:, 8:24, 10:30] += 0.3
    assert bool((pred > 1.0).any())
    pred.requires_grad_()
    PerceptualLoss(net)(pred, gt).backward()
    assert pred.grad is not None
    assert bool(torch.isfinite(pred.grad).all()) and bool(pred.grad.abs().sum() > 0)


@pytest.mark.gpu
@pytest.mark.parametrize(("shape", "phase", "raised", "expected"), PIQ_HAARPSI)
def test_d89_haarpsi_on_cuda_is_one_minus_piq_haarpsi(
    shape: tuple[int, ...], phase: float, raised: bool, expected: float
) -> None:
    """D-89's `haarpsi` row on CUDA: piq's values in float64, to 1e-12, as on the CPU."""
    gt = _pattern(shape, 0.0)
    pred = _pattern(shape, phase)
    if raised:
        pred[..., 5:20, 10:30, :] += 0.3
    value = PerceptualLoss("haarpsi")(pred.cuda(), gt.cuda())
    assert value.device.type == "cuda"
    torch.testing.assert_close(value.item(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("net", [*WEIGHT_FREE_NETWORKS, *CAFFE_PARAMS])
def test_u9_perceptual_network_of_an_image_with_itself_on_the_gpu(net: PercNet) -> None:
    """U-9 on CUDA, in float64 for HaarPSI (Q-262) and float32 for the others (Q-261,
    Q-263): identical images give 0, or at most 10⁻⁵ of the value under σ = 16/255 noise."""
    dtype = F64 if net == "haarpsi" else F32  # Q-262
    loss = PerceptualLoss(net).to("cuda")
    image = _pattern((2, 64, 96, 3), 0.0).to("cuda", dtype)
    same = loss(image, image.clone())
    noise = loss(_with_noise(image.cpu(), seed=46).cuda(), image)
    assert same.device.type == "cuda"
    assert noise.item() > 0.0
    assert abs(same.item()) <= 1e-5 * noise.item()


@pytest.mark.gpu
@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the recorder needs sys.monitoring (Python 3.12)"
)
@pytest.mark.parametrize("net", [*WEIGHT_FREE_NETWORKS, *CAFFE_PARAMS])
def test_perceptual_loss_forward_and_backward_on_cuda_make_no_synchronization(net: PercNet) -> None:
    """§2.4 rule 3 (#381, #385): after one call on a device, which builds the networks'
    constants and filters there (and, for ꟻLIP, reads its colour maximum back once), a
    forward and backward pass of the network on a render-sized pair make no host
    synchronization under `torch.cuda.set_sync_debug_mode("warn")`, by any route the recorder
    sees. The Caffe trunks' weights are moved to the device with the module."""
    from sync_recorder import SyncRecorder, described

    loss = PerceptualLoss(net).to("cuda")
    pred, gt = (
        x.to("cuda", F32) for x in (_pattern((1, 96, 128, 3), 0.4), _pattern((1, 96, 128, 3), 0.0))
    )
    warm_up = pred.clone().requires_grad_(True)
    loss(warm_up, gt).backward()
    image = pred.clone().requires_grad_(True)
    torch.cuda.synchronize()
    with SyncRecorder() as recorder:
        loss(image, gt).backward()
    torch.cuda.synchronize()
    assert recorder.syncs == [], described(recorder.syncs, __file__)
    assert recorder.unattributed == 0
    assert image.grad is not None


# U-10: the 4D regularization (E8, D-11)


def _reg_oracle(model: FreeTimeGaussians, t: float) -> tuple[float, np.ndarray]:
    """E8 in float64 from the stored tensors: L_reg = mean(σ·σt(t)) with σ = 1/(1 + e^−ô) and
    σt = exp(−½((t − μt)/e^ŝ)²), and its closed-form gradient (1/N)·σt·σ(1 − σ) w.r.t. ô."""
    logits = model.opacity_logits.detach().double().numpy()
    times = model.times.detach().double().numpy()
    log_durations = model.log_durations.detach().double().numpy()
    sigma = 1.0 / (1.0 + np.exp(-logits))
    sigma_t = np.exp(-0.5 * ((t - times) / np.exp(log_durations)) ** 2)
    n = len(logits)
    return float(np.mean(sigma * sigma_t)), sigma_t * sigma * (1.0 - sigma) / n


def _all_grads(loss: Tensor, model: FreeTimeGaussians) -> dict[str, Tensor]:
    params = [getattr(model, name) for name in PARAM_NAMES]
    grads = torch.autograd.grad(loss, params, allow_unused=True, materialize_grads=True)
    return dict(zip(PARAM_NAMES, grads, strict=True))


@pytest.mark.parametrize("t", [0.0, 0.37, 1.0])
def test_u10_regularization_is_the_mean_of_opacity_times_temporal_opacity(t: float) -> None:
    """U-10: L_reg(t) = mean(σ·σt(t)) over all N primitives, to 1e-7 (float64 model; a float32
    model too)."""
    model = random_model(40, seed=20, logit=(-4.0, 4.0)).double()
    expected, _ = _reg_oracle(model, t)
    value = reg4d_loss(model, t)
    assert value.shape == () and value.dtype == F64
    torch.testing.assert_close(value.item(), expected, rtol=0.0, atol=1e-7)
    model32 = random_model(40, seed=20, logit=(-4.0, 4.0))
    torch.testing.assert_close(reg4d_loss(model32, t).item(), expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("t", [0.0, 0.37, 1.0])
def test_u10_regularization_gradient_reaches_only_the_opacity_logits(t: float) -> None:
    """U-10: ∂L_reg/∂ô = (1/N)·σt·σ(1 − σ) to 1e-7 relative; every other gradient, μt and ŝ
    included (the stop-gradient on σt), is exactly 0."""
    model = random_model(40, seed=21, logit=(-4.0, 4.0)).double()
    _, expected = _reg_oracle(model, t)
    grads = _all_grads(reg4d_loss(model, t), model)
    torch.testing.assert_close(
        grads["opacity_logits"], torch.from_numpy(expected), rtol=1e-7, atol=0.0
    )
    for name in PARAM_NAMES:
        if name != "opacity_logits":
            assert bool((grads[name] == 0).all()), name


def test_regularization_of_an_empty_model_is_zero() -> None:
    """Q-41: with N = 0 the mean of E8 is taken as 0, not NaN."""
    model = random_model(0, seed=0)
    assert reg4d_loss(model, 0.5).item() == 0.0


# U-10 and U-9: the total loss (E9, D-10, D-37, D-38, D-77; Q-41)

NO_PERC = LossConfig(lambda_perc=0.0)  # E9 at its defaults, without the downloaded network
T_IT = OptimConfig().iterations  # T_it, the default run length


def _detached_pair(seed: int, dtype: torch.dtype = F64) -> tuple[Tensor, Tensor]:
    """A render-sized image pair that doesn't depend on any model."""
    return _image_pair((48, 64, 3), seed=seed, correlated=True, dtype=dtype)


@pytest.mark.parametrize("it", [1, T_IT])
def test_u10_regularization_weight_is_one_hundredth_at_every_iteration_by_default(it: int) -> None:
    """U-10 (E9, D-10, D-38): with reg_until unset, λ_reg = 10⁻² applies at it = 1 and at
    it = T_it: the total is the weighted sum of the terms to 1e-6, and with images independent of
    the model ∂L/∂ô is 10⁻² times E8's gradient, to 1e-7 relative."""
    model = random_model(30, seed=22, logit=(-4.0, 4.0)).double()
    pred, gt = _detached_pair(23)
    loss, terms = total_loss(pred, gt, model, 0.4, it, NO_PERC)
    reg, expected_grad = _reg_oracle(model, 0.4)
    assert set(terms) == {"l1", "one_minus_ssim", "reg", "total"}
    weighted = 0.8 * l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt)) + 1e-2 * reg
    torch.testing.assert_close(loss.item(), weighted.item(), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(terms["reg"].item(), reg, rtol=0.0, atol=1e-7)
    grads = _all_grads(loss, model)
    torch.testing.assert_close(
        grads["opacity_logits"], 1e-2 * torch.from_numpy(expected_grad), rtol=1e-7, atol=0.0
    )


def test_u10_regularization_stops_after_reg_until() -> None:
    """U-10 (D-10, Q-41): reg_until is the last iteration with L_reg: at it = reg_until it
    applies; after it, and at every iteration with reg_until = 0, the total leaves it out and
    ∂L/∂ô is exactly 0, while "reg" still reports L_reg's value, detached (owner's answer to
    Q-41)."""
    model = random_model(30, seed=24).double()
    pred, gt = _detached_pair(25)
    reg, _ = _reg_oracle(model, 0.4)
    cfg = LossConfig(lambda_perc=0.0, reg_until=100)
    loss, terms = total_loss(pred, gt, model, 0.4, 100, cfg)
    assert "reg" in terms
    assert bool((_all_grads(loss, model)["opacity_logits"] != 0).all())
    for it, until in [(101, 100), (T_IT, 100), (1, 0)]:
        cfg = LossConfig(lambda_perc=0.0, reg_until=until)
        # pred requires a gradient so that the loss has a graph whose ô gradient can be read.
        loss, terms = total_loss(pred.clone().requires_grad_(), gt, model, 0.4, it, cfg)
        assert set(terms) == {"l1", "one_minus_ssim", "reg", "total"}
        assert not terms["reg"].requires_grad
        torch.testing.assert_close(terms["reg"].item(), reg, rtol=0.0, atol=1e-7)
        expected = 0.8 * l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
        assert loss.item() == expected.item()
        assert bool((_all_grads(loss, model)["opacity_logits"] == 0).all())


def _off_screen_scene() -> FreeTimeGaussians:
    """Two primitives in front of camera A and one 100 units to its side, off its image."""
    return model_from_rows(
        [
            {"mean": (0.0, 0.0, 4.0), "logit": 1.0},
            {"mean": (1.0, 0.5, 5.0), "logit": -0.5, "log_duration": -1.0},
            {"mean": (100.0, 0.0, 4.0), "logit": 0.3, "log_duration": -0.5, "time": 0.2},
        ]
    ).double()


def test_u10_regularization_is_independent_of_the_camera() -> None:
    """U-10 (D-11): L_reg is identical for renders from two cameras, and a primitive off one
    camera's image still gets E8's gradient λ_reg·(1/N)·σt·σ(1 − σ) through the total loss."""
    model = _off_screen_scene()
    cam_b = pinhole(
        64,
        48,
        fx=16.0,
        fy=16.0,
        cx=32.5,
        cy=24.5,
        world_to_camera=[[1.0, 0.0, 0.0, -100.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    gt = _detached_pair(26)[1]
    regs, losses = [], []
    for camera, visible in ((CAM_A, [True, True, False]), (cam_b, [False, False, True])):
        out = render(
            model, camera, 0.4, time_units="normalized", backend="reference", return_aux=True
        )
        assert out.visible is not None and out.visible[0].tolist() == visible
        loss, terms = total_loss(out.image[0], gt, model, 0.4, 1, NO_PERC)
        regs.append(terms["reg"])
        losses.append(loss)
    assert torch.equal(regs[0], regs[1])
    loss = losses[0]
    _, expected = _reg_oracle(model, 0.4)
    grad = _all_grads(loss, model)["opacity_logits"]
    torch.testing.assert_close(
        grad[2], torch.tensor(1e-2 * expected[2], dtype=F64), rtol=1e-7, atol=0.0
    )


def test_total_loss_returns_detached_scalars_on_the_image_device() -> None:
    """D-77: the terms are detached 0-d tensors on the images' device, and "total" is the
    returned loss's value."""
    model = random_model(10, seed=27)
    pred, gt = _detached_pair(28, dtype=F32)
    loss, terms = total_loss(pred.requires_grad_(), gt, model, 0.5, 1, NO_PERC)
    assert loss.shape == () and loss.requires_grad
    for value in terms.values():
        assert isinstance(value, Tensor) and value.shape == () and not value.requires_grad
        assert value.device == pred.device
    assert terms["total"].item() == loss.item()


@pytest.mark.parametrize("it", [0, -1, True, 1.0], ids=["zero", "negative", "bool", "float"])
def test_total_loss_rejects_an_iteration_below_one(it: object) -> None:
    """Q-41: iterations are counted from 1 (D-41)."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(29, dtype=F32)
    with pytest.raises(ValueError):
        total_loss(pred, gt, model, 0.5, it, NO_PERC)  # type: ignore[arg-type]


def test_total_loss_needs_the_perceptual_network_when_its_weight_is_positive() -> None:
    """D-77: λ_perc > 0 with no network passed is a ValueError."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(30, dtype=F32)
    with pytest.raises(ValueError):
        total_loss(pred, gt, model, 0.5, 1, LossConfig())


def test_total_loss_skips_terms_whose_weight_is_zero() -> None:
    """Q-41: a term whose λ is 0 is not computed and has no key, L_reg included, also after
    reg_until."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(31, dtype=F32)
    cfg = LossConfig(lambda_img=0.0, lambda_ssim=0.0, lambda_perc=0.0, lambda_reg=1e-2)
    loss, terms = total_loss(pred, gt, model, 0.5, 1, cfg)
    assert set(terms) == {"reg", "total"}
    torch.testing.assert_close(loss, 1e-2 * reg4d_loss(model, 0.5), rtol=0.0, atol=0.0)
    for it in (1, 200):
        cfg = LossConfig(lambda_perc=0.0, lambda_reg=0.0, reg_until=100)
        _, terms = total_loss(pred, gt, model, 0.5, it, cfg)
        assert set(terms) == {"l1", "one_minus_ssim", "total"}


@pytest.mark.weights
def test_u9_total_loss_is_the_weighted_sum_of_its_terms(lpips_nets: dict[str, LPIPSLoss]) -> None:
    """U-9 (E7, E9, D-37): L = 0.8·L1 + 0.2·(1 − SSIM) + 0.01·LPIPS-VGG + 0.01·L_reg, to 1e-6,
    each term recomputed with the public functions."""
    model = random_model(30, seed=32)
    pred, gt = _detached_pair(33, dtype=F32)
    perc = lpips_nets["vgg"]
    loss, terms = total_loss(pred, gt, model, 0.6, 1, LossConfig(), perceptual=perc)
    assert set(terms) == {"l1", "one_minus_ssim", "perceptual", "reg", "total"}
    expected = (
        0.8 * l1_loss(pred, gt).item()
        + 0.2 * (1.0 - ssim(pred, gt).item())
        + 0.01 * perc(pred, gt).item()
        + 0.01 * reg4d_loss(model, 0.6).item()
    )
    torch.testing.assert_close(loss.item(), expected, rtol=0.0, atol=1e-6)
    assert terms["perceptual"].item() == perc(pred, gt).item()


@pytest.mark.weights
def test_total_loss_needs_the_backbone_the_config_names(lpips_nets: dict[str, LPIPSLoss]) -> None:
    """Q-41 (D-9): a perceptual network with another backbone than cfg.perc_net is a
    ValueError, whatever λ_perc."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(34, dtype=F32)
    for cfg in (LossConfig(), NO_PERC):
        with pytest.raises(ValueError):
            total_loss(pred, gt, model, 0.5, 1, cfg, perceptual=lpips_nets["alex"])
    cfg = LossConfig(perc_net="alex")
    _, terms = total_loss(pred, gt, model, 0.5, 1, cfg, perceptual=lpips_nets["alex"])
    assert "perceptual" in terms


# D-89's calibrated λ_perc of the two weight-free networks and the two Caffe trunks, written
# out from its table.
CALIBRATED = {"flip": 0.415, "haarpsi": 0.0617, "vgg16_oxford": 0.00164, "alexnet_bvlc": 0.00408}


@pytest.mark.parametrize("net", [*WEIGHT_FREE_NETWORKS, *CAFFE_PARAMS])
def test_u9_lambda_perc_none_takes_the_network_calibrated_weight(net: PercNet) -> None:
    """U-9 (D-9, D-89; Q-249): with λ_perc = None, total_loss weights the perceptual term by the
    network's calibrated weight, 0.415 for ꟻLIP, 0.0617 for HaarPSI, 0.00164 for VGG16 and
    0.00408 for AlexNet: the loss and every term are exactly those of the same configuration
    with that weight written in."""
    model = random_model(10, seed=35)
    pred, gt = _detached_pair(36, dtype=F32)
    perc = PerceptualLoss(net)
    by_none = total_loss(
        pred, gt, model, 0.5, 1, LossConfig(perc_net=net, lambda_perc=None), perceptual=perc
    )
    written = LossConfig(perc_net=net, lambda_perc=CALIBRATED[net])
    by_value = total_loss(pred, gt, model, 0.5, 1, written, perceptual=perc)
    assert by_none[0].item() == by_value[0].item()
    assert {k: v.item() for k, v in by_none[1].items()} == {
        k: v.item() for k, v in by_value[1].items()
    }
    assert "perceptual" in by_none[1]
    for weight in (0.0, 1.0):
        other = LossConfig(perc_net=net, lambda_perc=weight)
        assert total_loss(pred, gt, model, 0.5, 1, other, perceptual=perc)[0].item() != (
            by_none[0].item()
        )


@pytest.mark.parametrize("net", ["flip", "haarpsi", *TRUNK_NETWORKS])
def test_total_loss_needs_the_perceptual_network_when_lambda_perc_is_none(net: PercNet) -> None:
    """D-77 with Q-249: λ_perc = None resolves to the network's calibrated weight, which is
    positive for each of D-89's networks, so perceptual=None is a ValueError; λ_perc = 0 needs
    no network."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(37, dtype=F32)
    with pytest.raises(ValueError, match="perceptual"):
        total_loss(pred, gt, model, 0.5, 1, LossConfig(perc_net=net, lambda_perc=None))
    _, terms = total_loss(pred, gt, model, 0.5, 1, LossConfig(perc_net=net, lambda_perc=0.0))
    assert "perceptual" not in terms


def test_u9_total_loss_with_flip_is_the_weighted_sum_of_its_terms() -> None:
    """U-9 (E7, E9, D-37, D-89): at λ_perc = None, L = 0.8·L1 + 0.2·(1 − SSIM) + 0.415·ꟻLIP +
    0.01·L_reg, to 1e-6, each term recomputed with the public functions; the terms are
    detached 0-d tensors on the images' device and "total" is the loss's value (D-77)."""
    model = random_model(30, seed=39)
    pred, gt = _detached_pair(40, dtype=F32)
    perc = PerceptualLoss("flip")
    cfg = LossConfig(perc_net="flip", lambda_perc=None)
    loss, terms = total_loss(pred.requires_grad_(), gt, model, 0.6, 1, cfg, perceptual=perc)
    assert set(terms) == {"l1", "one_minus_ssim", "perceptual", "reg", "total"}
    for value in terms.values():
        assert isinstance(value, Tensor) and value.shape == () and not value.requires_grad
        assert value.device == pred.device
    with torch.no_grad():
        expected = (
            0.8 * l1_loss(pred, gt).item()
            + 0.2 * (1.0 - ssim(pred, gt).item())
            + 0.415 * perc(pred, gt).item()
            + 0.01 * reg4d_loss(model, 0.6).item()
        )
        assert terms["perceptual"].item() == perc(pred, gt).item()
    torch.testing.assert_close(loss.item(), expected, rtol=0.0, atol=1e-6)
    assert terms["total"].item() == loss.item()


def test_total_loss_needs_the_perceptual_network_the_config_names() -> None:
    """Q-41 (D-9): a PerceptualLoss built for another network than cfg.perc_net is a
    ValueError, whatever λ_perc."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(41, dtype=F32)
    for weight in (None, 0.0):
        cfg = LossConfig(perc_net="haarpsi", lambda_perc=weight)
        with pytest.raises(ValueError, match="perc_net"):
            total_loss(pred, gt, model, 0.5, 1, cfg, perceptual=PerceptualLoss("flip"))
    cfg = LossConfig(perc_net="haarpsi", lambda_perc=None)
    _, terms = total_loss(pred, gt, model, 0.5, 1, cfg, perceptual=PerceptualLoss("haarpsi"))
    assert "perceptual" in terms


def test__calibrated_weights_are_d89s_table() -> None:
    """D-89's table: the calibrated λ_perc of each of the eight networks, the trunk networks'
    included, so that #385 and #386 find theirs; one per `PercNet` value."""
    assert openftgs.losses._CALIBRATED_WEIGHTS == {
        "flip": 0.415,
        "haarpsi": 0.0617,
        "vgg16_oxford": 0.00164,
        "alexnet_bvlc": 0.00408,
        "dinov2_s14": 0.00185,
        "dinov2_b14": 0.00204,
        "dino_s16": 0.00358,
        "dino_b16": 0.00484,
    }
    assert set(openftgs.losses._CALIBRATED_WEIGHTS) == set(get_args(PercNet))


@pytest.mark.parametrize("net", ["vgg", "alex"])
def test_total_loss_refuses_lambda_perc_none_for_an_lpips_backbone(net: str) -> None:
    """#380's expand step (Q-249): LPIPS's backbones stay valid `perc_net` values until the
    contract ticket (#392), but D-89's table gives them no calibrated weight, so λ_perc = None
    with one of them is a ValueError naming D-89, raised before any network is used."""
    model = random_model(4, seed=0)
    pred, gt = _detached_pair(38, dtype=F32)
    cfg = LossConfig(perc_net=net, lambda_perc=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="D-89"):
        total_loss(pred, gt, model, 0.5, 1, cfg)


# U-11: the full forward through the reference backend


class _LossModule(torch.nn.Module):
    """Holds a model, and a copy of it for L_reg, so that torch.func.functional_call can
    substitute their tensors; forward renders `model` with the reference backend and returns
    L1 + L_reg, with L_reg evaluated on `reg_model`."""

    def __init__(self, model: FreeTimeGaussians, camera: Camera, t: float, gt: Tensor) -> None:
        super().__init__()
        self.model = model
        self.reg_model = copy.deepcopy(model)
        self.camera = camera
        self.t = t
        self.gt = gt

    def forward(self) -> Tensor:
        out = render(self.model, self.camera, self.t, time_units="normalized", backend="reference")
        return l1_loss(out.image[0], self.gt) + reg4d_loss(self.reg_model, self.t)


def test_u11_l1_plus_regularization_passes_gradcheck_through_the_reference_renderer() -> None:
    """U-11: float64 gradcheck of L1 + L_reg through the reference backend, 3 primitives on an
    8×8 image, all nine stored tensors, at eps 1e-6 and atol 1e-5. The scene stays away from the
    clamps and thresholds of E6 (σ′ ≈ 3 px, σ ≤ 0.6, distinct depths, colors away from 0), and
    the target sits 0.1 from the initial render in every pixel and channel, away from L1's kink.

    D-78: E8's sg[σt] gives L_reg no gradient through μt and ŝ while finite differences see its
    value move, so L_reg is evaluated with the substituted opacity logits and μt and ŝ held at
    their base values, which is what the stop-gradient means to first order; L1 sees all nine
    substituted tensors. U-10 pins the exact zeros."""
    camera = pinhole(8, 8, fx=8.0, fy=8.0, cx=4.0, cy=4.0)
    gen = torch.Generator().manual_seed(11)
    params = {
        "means": torch.tensor([[0.2, -0.1, 3.6], [-0.3, 0.2, 4.0], [0.1, 0.3, 4.4]]),
        "times": torch.tensor([0.5, 0.6, 0.4]),
        "log_durations": torch.zeros(3),
        "velocities": 0.3 * torch.randn(3, 3, generator=gen),
        "log_scales": torch.log(1.2 + 0.6 * torch.rand(3, 3, generator=gen)),
        "quats": torch.randn(3, 4, generator=gen),
        "opacity_logits": torch.tensor([-0.8, 0.0, 0.4]),
        "sh0": 0.3 + 0.2 * torch.rand(3, 1, 3, generator=gen),
        "shN": 0.05 * torch.randn(3, 15, 3, generator=gen),
    }
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME).double()
    t = 0.3
    with torch.no_grad():
        image = render(model, camera, t, time_units="normalized", backend="reference").image[0]
    signs = torch.randint(0, 2, image.shape, generator=gen).to(F64) * 2 - 1
    module = _LossModule(model, camera, t, image + 0.1 * signs)
    inputs = tuple(getattr(model, name).detach().clone().requires_grad_() for name in PARAM_NAMES)

    def fn(*tensors: Tensor) -> Tensor:
        substituted = {f"model.{name}": x for name, x in zip(PARAM_NAMES, tensors, strict=True)}
        # D-78: L_reg's μt and ŝ stay at their base values.
        substituted["reg_model.opacity_logits"] = substituted["model.opacity_logits"]
        loss: Tensor = torch.func.functional_call(module, substituted, ())
        return loss

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)
    # Every tensor gets a gradient: L1 reaches all nine through the render, L_reg the logits.
    grads = torch.autograd.grad(fn(*inputs), inputs)
    assert all(bool(g.abs().sum() > 0) for g in grads)
