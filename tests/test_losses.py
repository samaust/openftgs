# SPDX-License-Identifier: Apache-2.0
"""openftgs.losses (E7, E8, E9; D-7 to D-11, D-37, D-38, D-52, D-68; Q-40 to Q-42): U-9, U-10,
U-11 and the added tests of PRD #53.

Expected values are worked by hand from the spec, come from scikit-image's
`structural_similarity` or from a brute-force float64 SSIM written in this file from Wang et al.
2004 and E7, or from E8's closed-form gradient. Images are in §1.4's layout, [..., H, W, 3].
"""

import importlib
import sys
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity
from torch import Tensor

import openftgs.losses
from openftgs.losses import l1_loss, ssim

F32 = torch.float32
F64 = torch.float64


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
    assert abs(value.item() - _skimage_ssim(pred, gt)) <= 1e-4


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
    assert abs(value.item() - expected) <= 1e-10


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
    assert abs(value.item() - expected.item()) <= 1e-5 * abs(expected.item())
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
    expected = ssim(pred64, gt64, padding=padding)  # type: ignore[arg-type]
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        value = ssim_reloaded(
            pred64.to("cuda", F32),
            gt64.to("cuda", F32),
            padding=padding,  # type: ignore[arg-type]
        )
    assert abs(value.item() - expected.item()) <= 1e-5 * abs(expected.item())


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
    """U-9: L_ssim = 1 − SSIM(I, I) is 0 on CUDA float32 too, to 1e-6."""
    image, _ = _image_pair((2, 20, 23, 3), seed=4, correlated=False, dtype=F32)
    image = image.cuda()
    value = ssim(image, image.clone(), padding=padding)  # type: ignore[arg-type]
    assert abs(1.0 - value.item()) <= 1e-6


@pytest.mark.gpu
def test_fused_ssim_loader_finds_the_installed_package() -> None:
    """The private loader returns fused-ssim's function when the package imports."""
    fused = pytest.importorskip("fused_ssim")
    assert openftgs.losses._fused_ssim() is fused.fused_ssim


@pytest.mark.gpu
def test_ssim_dispatch_takes_fused_ssim_only_under_q42_conditions(
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
