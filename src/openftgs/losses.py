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

from openftgs._stages import stage
from openftgs.config import LossConfig
from openftgs.model import FreeTimeGaussians
from openftgs.motion import reg4d_terms

__all__ = ["LPIPSLoss", "l1_loss", "reg4d_loss", "ssim", "total_loss"]


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
    """Reshape [..., H, W, 3] images to [B, 3, H, W].

    A contiguous input gives a channels-last view, no copy.
    """
    return image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2)


def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """Compute the image loss L_img, the mean of ``|Î − I|`` (E7, D-7).

    The L1 term of E7: the absolute difference averaged over the batch, the pixels and the
    channels.

    Parameters
    ----------
    pred
        The render Î, [..., H, W, 3], floating point; any leading dimensions form a batch
        (Q-41). It isn't clamped, so it may exceed 1 (E7).
    gt
        The ground-truth image I, RGB in [0, 1], with ``pred``'s shape, dtype and device.

    Returns
    -------
    torch.Tensor
        L_img, a 0-d tensor in ``pred``'s dtype, on its device.

    Raises
    ------
    ValueError
        If the images differ in shape, dtype or device, are not [..., H, W, 3], or are not
        floating point.
    """
    _check_pair(pred, gt)
    return (pred - gt).abs().mean()


# Q-42: the window, σ and data range fused-ssim hardcodes (E7's).
_FUSED_SSIM_CONSTANTS = (11, 1.5, 1.0)


# Q-2: fused-ssim is optional and not on PyPI; tested at commit
# a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 of https://github.com/rahul-goel/fused-ssim (MIT).
@functools.cache
def _fused_ssim() -> Callable[..., Tensor] | None:
    """Return fused-ssim's `fused_ssim(img1, img2, padding, train)`, or None if it won't import."""
    try:
        from fused_ssim import fused_ssim
    except ImportError:
        return None
    return fused_ssim  # type: ignore[no-any-return]


def _gaussian_window(window: int, sigma: float, like: Tensor) -> Tensor:
    """Build the normalized 1-D Gaussian of `window` taps and standard deviation `sigma` (D-8).

    Centred on the middle tap, in `like`'s dtype and on its device.
    """
    x = torch.arange(window, dtype=like.dtype, device=like.device) - (window - 1) / 2
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def _ssim_torch(
    x: Tensor, y: Tensor, *, window: int, sigma: float, data_range: float, padding: str
) -> Tensor:
    """Compute the SSIM of [B, 3, H, W] images (Wang et al. 2004; E7, D-8).

    Local moments under the Gaussian window, population (weighted) variances, the map averaged
    over batch, channels and pixels. "same" zero-pads the images (Q-42).
    """
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
    """Compute SSIM(Î, I) under a Gaussian window, per channel, averaged (E7, D-8).

    The local means, variances and covariance are taken under a ``window`` × ``window``
    Gaussian window of standard deviation ``sigma``, with population (weighted) variances and the
    constants C₁ = (0.01·``data_range``)² and C₂ = (0.03·``data_range``)² (Wang et al. 2004).
    The SSIM map is averaged over its pixels, the channels and the batch. This is SSIM, not
    1 − SSIM.

    On CUDA, for float32 images with E7's window, σ and data range and a ``gt`` that doesn't
    require a gradient, fused-ssim computes it when it is installed (Q-2, Q-42); a PyTorch
    implementation does otherwise.

    Parameters
    ----------
    pred
        The first image, Î, [..., H, W, 3], floating point; any leading dimensions form a batch
        (Q-41).
    gt
        The second image, I, with ``pred``'s shape, dtype and device.
    window
        The window's width and height in pixels, an odd int ≥ 1 (E7: 11).
    sigma
        The Gaussian's standard deviation in pixels, > 0 (E7: 1.5).
    data_range
        The images' value range, > 0: 1 for images in [0, 1] (E7), k for E16's SSIM_k.
    padding
        "valid" keeps only the window positions inside the image (E7); "same" zero-pads the
        images by ``window // 2`` pixels, so the map has the image's size (Q-42).

    Returns
    -------
    torch.Tensor
        SSIM, a 0-d tensor; 1 for identical images.

    Raises
    ------
    ValueError
        If the images differ in shape, dtype or device or are not [..., H, W, 3] floating
        point, ``window`` is not an odd int ≥ 1, ``sigma`` or ``data_range`` is not > 0,
        ``padding`` is not "valid" or "same", or, with "valid" padding, H or W is smaller
        than ``window``.

    Notes
    -----
    Q-41 fixes the image layout and the checks, and scales the constants with the data range;
    Q-42 the fused path and "same" padding.
    """
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
        and (window, sigma, data_range) == _FUSED_SSIM_CONSTANTS
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
    """The perceptual loss L_perc, LPIPS on images in [0, 1] (E7, D-9).

    The images are mapped from [0, 1] to [−1, 1], the network's documented input range (D-68),
    and the LPIPS distance is averaged over the batch. ``forward(…, map_inputs=False)`` skips
    the mapping, for the ``lpips_vgg_raw`` metric (Q-58). The network is lpips 0.1.4's, frozen
    and kept in evaluation mode, whatever :meth:`train` is given (Q-41). Calling the module,
    ``loss(pred, gt)``, runs :meth:`forward`.

    Parameters
    ----------
    net
        The backbone: "vgg", the training loss's (D-9), or "alex", for the AlexNet LPIPS
        metric (E16).

    Raises
    ------
    ValueError
        If ``net`` is not "vgg" or "alex".

    Notes
    -----
    The trainer builds one network and passes it to :func:`total_loss` (D-77).
    """

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
            self._net: torch.nn.Module = lpips.LPIPS(net=net, verbose=False)
        self._backbone = net
        # lpips freezes only its trunk; the linear layers would stay trainable.
        self._net.requires_grad_(False)
        self._net.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        """Set this module's training mode and keep the LPIPS network in evaluation mode (Q-41).

        The network's linear layers hold dropout, which training mode would switch on, so only
        this module's own flag follows ``mode``.

        Parameters
        ----------
        mode
            True for training mode, False for evaluation mode, as for
            :meth:`torch.nn.Module.train`.

        Returns
        -------
        LPIPSLoss
            This module.
        """
        # The linear layers hold dropout, which training mode would switch on (Q-41).
        super().train(mode)
        self._net.eval()
        return self

    # Q-58: map_inputs=False feeds the [0, 1] images unmapped, D-68's lpips_vgg_raw metric.
    def forward(self, pred: Tensor, gt: Tensor, *, map_inputs: bool = True) -> Tensor:
        """Compute the LPIPS distance between ``pred`` and ``gt``, averaged over the batch (E7).

        Parameters
        ----------
        pred
            The render Î, [..., H, W, 3], floating point, nominally in [0, 1]; any leading
            dimensions form a batch (Q-41).
        gt
            The ground-truth image I, RGB in [0, 1], with ``pred``'s shape, dtype and device.
        map_inputs
            True, the default, maps both images from [0, 1] to [−1, 1] first (D-68); False
            feeds them unmapped, for D-68's ``lpips_vgg_raw`` metric (Q-58).

        Returns
        -------
        torch.Tensor
            The mean LPIPS distance, a 0-d tensor; 0 for identical images.

        Raises
        ------
        ValueError
            If ``map_inputs`` is not a bool, or the images differ in shape, dtype or device or
            are not [..., H, W, 3] floating point.
        """
        if not isinstance(map_inputs, bool):
            _fail(f"map_inputs must be a bool, got {map_inputs!r}")
        _check_pair(pred, gt)
        x, y = _batch_first(pred), _batch_first(gt)
        if map_inputs:
            # D-68: the network's documented input range.
            x, y = 2 * x - 1, 2 * y - 1
        # With frozen weights and a target that doesn't require a gradient, autograd records
        # nothing on the target's branch (§2.4).
        out: Tensor = self._net(x, y)
        return out.mean()


def reg4d_loss(model: FreeTimeGaussians, t: float) -> Tensor:
    """Compute the 4D regularization L_reg(t) at normalized time ``t`` (E8, D-11).

    ``L_reg(t) = (1/N)·Σᵢ σᵢ·sg[σt,ᵢ(t)]`` over all N primitives, visible or not, with σᵢ the
    base opacity and σt,ᵢ(t) the temporal opacity. The stop-gradient is motion's (Q-12), so the
    gradient reaches only the opacity logits (E8). The parameters are read at call time. An
    empty model gives 0 (Q-41).

    Parameters
    ----------
    model
        The model whose opacities, time centers and durations enter the sum.
    t
        The training image's normalized time (E14), in [0, 1] for a training timestamp.

    Returns
    -------
    torch.Tensor
        L_reg(t), a 0-d float32 tensor in [0, 1], on the model's device.
    """
    terms = reg4d_terms(model.opacity_logits, model.times, model.log_durations, t)
    # Q-41: an empty model gives 0, not the NaN of an empty mean.
    return terms.mean() if terms.numel() > 0 else terms.sum()


# D-77: the dict holds detached 0-d tensors, so the loss never syncs with the host (§2.4 rule 3),
# and the perceptual network is passed in.
def total_loss(
    pred: Tensor,
    gt: Tensor,
    model: FreeTimeGaussians,
    t: float,
    it: int,
    cfg: LossConfig,
    *,
    perceptual: LPIPSLoss | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the total loss L of E9 at iteration ``it``, and its unweighted terms (E9, D-77).

    ``L = λ_img·L1 + λ_ssim·(1 − SSIM) + λ_perc·LPIPS + λ_reg·L_reg(t)`` (E7, E9, D-37), from
    :func:`l1_loss`, :func:`ssim`, :class:`LPIPSLoss` and :func:`reg4d_loss`. A term whose λ is
    0 is neither computed nor reported. After ``cfg.reg_until``, L_reg leaves the total but is
    still reported, computed without a gradient, so the keys stay fixed for a run (D-10,
    Q-41). Each term runs as a stage of its own, for the benchmarks (§3.4, B-1).

    Parameters
    ----------
    pred
        The render Î, [..., H, W, 3], floating point, not clamped (E7); any leading dimensions
        form a batch (Q-41).
    gt
        The training image I, RGB in [0, 1], with ``pred``'s shape, dtype and device.
    model
        The model, for L_reg.
    t
        The training image's normalized time (§1.4), as for :func:`reg4d_loss`.
    it
        The iteration, counted from 1 (D-41); it decides whether L_reg enters the total.
    cfg
        The weights λ and the SSIM settings (E7 to E9, D-8), the end of L_reg (D-10) and the
        LPIPS backbone (D-9).
    perceptual
        The LPIPS network, built once by the caller as ``LPIPSLoss(cfg.perc_net)``; required
        when ``cfg.lambda_perc`` > 0 and unused otherwise. Its backbone must be
        ``cfg.perc_net`` in either case (Q-41).

    Returns
    -------
    loss : torch.Tensor
        L, a 0-d tensor that carries the gradient; 0 when every λ is 0.
    terms : dict of str to torch.Tensor
        The unweighted terms, "l1", "one_minus_ssim", "perceptual" and "reg", each present
        when its λ > 0, and the total under "total": detached 0-d tensors on the images'
        device (D-77).

    Raises
    ------
    ValueError
        If ``it`` is not an int ≥ 1, ``cfg.lambda_perc`` > 0 and ``perceptual`` is None,
        ``perceptual``'s backbone differs from ``cfg.perc_net``, or the images fail
        :func:`l1_loss`'s checks.

    Notes
    -----
    D-77 changes §2.1's signature: the terms stay on the device, so the loss never synchronizes
    with the host (§2.4 rule 3), and the perceptual network is passed in rather than built
    here. The trainer accumulates the terms and reads them every ``log_every`` iterations.

    Examples
    --------
    >>> from openftgs.config import LossConfig
    >>> from openftgs.losses import LPIPSLoss, total_loss
    >>> cfg = LossConfig()
    >>> perceptual = LPIPSLoss(cfg.perc_net).to(pred.device)
    >>> loss, terms = total_loss(pred, gt, model, t, it, cfg, perceptual=perceptual)
    >>> loss.backward()
    >>> sorted(terms)
    ['l1', 'one_minus_ssim', 'perceptual', 'reg', 'total']
    """
    # Q-41: iterations count from 1 (D-41).
    if isinstance(it, bool) or not isinstance(it, int) or it < 1:
        _fail(f"it must be an int >= 1, got {it!r}")
    if cfg.lambda_perc > 0.0 and perceptual is None:
        _fail("lambda_perc > 0 needs the perceptual network (perceptual=LPIPSLoss(...))")
    # Q-41: a network passed in must be the backbone LossConfig names (D-9), whatever λ_perc.
    if perceptual is not None and perceptual._backbone != cfg.perc_net:
        _fail(f"perceptual uses {perceptual._backbone!r}, but cfg.perc_net is {cfg.perc_net!r}")
    _check_pair(pred, gt)
    terms: dict[str, Tensor] = {}
    weighted: list[Tensor] = []
    # §3.4: each term is a stage of its own (B-1).
    if cfg.lambda_img > 0.0:
        with stage("loss.l1"):
            terms["l1"] = l1_loss(pred, gt)
            weighted.append(cfg.lambda_img * terms["l1"])
    if cfg.lambda_ssim > 0.0:
        with stage("loss.ssim"):
            value = ssim(
                pred, gt, window=cfg.ssim_window, sigma=cfg.ssim_sigma, padding=cfg.ssim_padding
            )
            terms["one_minus_ssim"] = 1.0 - value
            weighted.append(cfg.lambda_ssim * terms["one_minus_ssim"])
    if perceptual is not None and cfg.lambda_perc > 0.0:
        with stage("loss.perceptual"):
            terms["perceptual"] = perceptual(pred, gt)
            weighted.append(cfg.lambda_perc * terms["perceptual"])
    if cfg.lambda_reg > 0.0:
        with stage("loss.reg"):
            if cfg.reg_until is None or it <= cfg.reg_until:
                terms["reg"] = reg4d_loss(model, t)
                weighted.append(cfg.lambda_reg * terms["reg"])
            else:
                # Q-41 (owner's answer): reported after the cutoff, for D-10's loss trace.
                with torch.no_grad():
                    terms["reg"] = reg4d_loss(model, t)
    loss = functools.reduce(torch.add, weighted) if weighted else pred.new_zeros(())
    report = {name: value.detach() for name, value in terms.items()}
    report["total"] = loss.detach()
    return loss, report
