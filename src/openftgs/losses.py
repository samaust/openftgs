# SPDX-License-Identifier: Apache-2.0
"""L1, SSIM, the LPIPS wrapper, 4D regularization and the total loss.

Spec: E7, E8, E9. Band: Mechanics (§2.1).
"""

import functools
import warnings
from collections.abc import Callable
from typing import Literal, NoReturn

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["LPIPSLoss", "l1_loss", "ssim"]


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


# Q-41: images are [..., H, W, 3] (§1.4), with any leading dimensions forming a batch.
def _check_pair(pred: Tensor, gt: Tensor) -> None:
    if pred.shape != gt.shape:
        _fail(f"pred and gt must have one shape, got {tuple(pred.shape)} and {tuple(gt.shape)}")
    if pred.dim() < 3 or pred.shape[-1] != 3:
        _fail(f"images must be [..., H, W, 3], got {tuple(pred.shape)}")
    if not pred.is_floating_point() or pred.dtype != gt.dtype:
        _fail(f"images must share one floating dtype, got {pred.dtype} and {gt.dtype}")
    if pred.device != gt.device:
        _fail(f"images must be on one device, got {pred.device} and {gt.device}")


def _batch_first(image: Tensor) -> Tensor:
    """[..., H, W, 3] as [B, 3, H, W]; a contiguous input gives a channels-last view, no copy."""
    return image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2)


def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """L_img = mean |Î − I| over the batch, pixels and channels (E7, D-7)."""
    _check_pair(pred, gt)
    return (pred - gt).abs().mean()


# Q-2: fused-ssim is optional and not on PyPI; tested at commit
# a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 of https://github.com/rahul-goel/fused-ssim (MIT).
@functools.cache
def _fused_ssim() -> Callable[..., Tensor] | None:
    """fused-ssim's `fused_ssim(img1, img2, padding, train)`, or None when it doesn't import."""
    try:
        from fused_ssim import fused_ssim
    except ImportError:
        return None
    return fused_ssim  # type: ignore[no-any-return]


def _gaussian_window(window: int, sigma: float, like: Tensor) -> Tensor:
    """The normalized 1-D Gaussian of `window` taps and standard deviation `sigma`, centred on the
    middle tap, in `like`'s dtype and on its device (D-8)."""
    x = torch.arange(window, dtype=like.dtype, device=like.device) - (window - 1) / 2
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def _ssim_torch(
    x: Tensor, y: Tensor, *, window: int, sigma: float, data_range: float, padding: str
) -> Tensor:
    """SSIM of [B, 3, H, W] images (Wang et al. 2004; E7, D-8): local moments under the Gaussian
    window, population (weighted) variances, the map averaged over batch, channels and pixels.
    "same" zero-pads the images (Q-42)."""
    channels = x.shape[1]
    g = _gaussian_window(window, sigma, x)
    kernel = (g[:, None] * g[None, :]).expand(5 * channels, 1, window, window)
    maps = torch.cat([x, y, x * x, y * y, x * y], dim=1)
    pad = window // 2 if padding == "same" else 0
    filtered = F.conv2d(maps, kernel, padding=pad, groups=5 * channels)
    mu_x, mu_y, e_xx, e_yy, e_xy = filtered.chunk(5, dim=1)
    # Built symmetrically in x and y, so SSIM(I, I) is exactly 1.
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    var_x, var_y, cov = e_xx - mu_xx, e_yy - mu_yy, e_xy - mu_xy
    # Q-41: the constants scale with the data range (SSIM_k).
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * cov + c2)
    den = (mu_xx + mu_yy + c1) * (var_x + var_y + c2)
    return (num / den).mean()


def ssim(
    pred: Tensor,
    gt: Tensor,
    *,
    window: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    padding: Literal["valid", "same"] = "valid",
) -> Tensor:
    """SSIM(Î, I) with a `window`×`window` Gaussian window of standard deviation `sigma`, per
    channel, averaged over the map, channels and batch (E7, D-8). Returns SSIM, not 1 − SSIM."""
    _check_pair(pred, gt)
    # Q-41
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        _fail(f"window must be an odd int >= 1, got {window!r}")
    if not sigma > 0.0:
        _fail(f"sigma must be positive, got {sigma!r}")
    if not data_range > 0.0:
        _fail(f"data_range must be positive, got {data_range!r}")
    if padding not in ("valid", "same"):
        _fail(f'padding must be "valid" or "same", got {padding!r}')
    if padding == "valid" and min(pred.shape[-3], pred.shape[-2]) < window:
        _fail(f"valid padding needs H and W >= window {window}, got {tuple(pred.shape[-3:-1])}")
    x, y = _batch_first(pred), _batch_first(gt)
    # Q-42: fused-ssim hardcodes E7's constants, runs in CUDA float32 and differentiates only its
    # first image; anything else takes the PyTorch fallback.
    fused = (
        _fused_ssim()
        if pred.is_cuda
        and pred.dtype == torch.float32
        and (window, sigma, data_range) == (11, 1.5, 1.0)
        and not gt.requires_grad
        else None
    )
    if fused is not None:
        train = torch.is_grad_enabled() and pred.requires_grad
        return fused(x.contiguous(), y.contiguous(), padding=padding, train=train)
    return _ssim_torch(
        x,
        y,
        window=window,
        sigma=sigma,
        data_range=data_range,
        padding=padding,
    )


class LPIPSLoss(torch.nn.Module):
    """L_perc: LPIPS with the given backbone on images in [0, 1], mapped to [−1, 1] (E7, D-9,
    D-68), as the mean over the batch. The network is lpips 0.1.4's, frozen and kept in
    evaluation mode (Q-41)."""

    def __init__(self, net: Literal["vgg", "alex"] = "vgg") -> None:
        super().__init__()
        if net not in ("vgg", "alex"):
            _fail(f'net must be "vgg" or "alex", got {net!r}')
        import lpips  # torchvision comes with it; imported only when a network is built

        with warnings.catch_warnings():
            # lpips 0.1.4 (pinned, §2.3) passes torchvision's deprecated `pretrained=True`, which
            # torchvision maps to the ImageNet weights with two UserWarnings.
            warnings.filterwarnings("ignore", "The parameter 'pretrained'", UserWarning)
            warnings.filterwarnings("ignore", "Arguments other than a weight enum", UserWarning)
            self.net: torch.nn.Module = lpips.LPIPS(net=net, verbose=False)
        # lpips freezes only its trunk; the linear layers would stay trainable.
        self.net.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        # The linear layers hold dropout, which training mode would switch on (Q-41).
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt)
        x, y = _batch_first(pred), _batch_first(gt)
        # D-68: the network's documented input range. With frozen weights and a target that
        # doesn't require a gradient, autograd records nothing on the target's branch (§2.4).
        out: Tensor = self.net(2 * x - 1, 2 * y - 1)
        return out.mean()
