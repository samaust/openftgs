# SPDX-License-Identifier: Apache-2.0
"""L1, SSIM, the perceptual networks (D-89), 4D regularization and the total loss; the LPIPS
wrapper stays until the contract ticket of #380 (#392) removes it.

Spec: E7, E8, E9, D-89. Band: Mechanics (§2.1).
"""

import functools
import warnings
from collections.abc import Callable
from typing import Literal, NoReturn, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from openftgs._stages import stage
from openftgs._vendor.flip import LDRFLIPLoss
from openftgs.config import LossConfig, PercNet
from openftgs.model import FreeTimeGaussians
from openftgs.motion import reg4d_terms

__all__ = ["LPIPSLoss", "PerceptualLoss", "l1_loss", "reg4d_loss", "ssim", "total_loss"]


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


# Q-42: the window, σ and data range fused-ssim hardcodes (E7's).
_FUSED_SSIM_CONSTANTS = (11, 1.5, 1.0)


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


# D-89: the eight networks `perc_net` selects, and the trunk networks among them.
_PERC_NETS: tuple[str, ...] = get_args(PercNet)
_TRUNK_TICKETS = {
    "vgg16_oxford": "#385",
    "alexnet_bvlc": "#385",
    "dinov2_s14": "#386",
    "dinov2_b14": "#386",
    "dino_s16": "#386",
    "dino_b16": "#386",
}
# D-89, E16: ꟻLIP's pixels per degree, a Python float, as the vendored copy's filter cache keys it.
_FLIP_PIXELS_PER_DEGREE = 67.0
# D-89's `haarpsi` row: piq's defaults.
_HAARPSI_SCALES = 3
_HAARPSI_C = 30.0
_HAARPSI_ALPHA = 4.2


# The HaarPSI below is adapted from piq 0.8.0 (https://github.com/photosynthesis-team/piq):
# `haarpsi` in piq/haarpsi.py, with the helpers it calls, `rgb2yiq`
# (piq/functional/colour_conversion.py), `haar_filter` (piq/functional/filters.py) and
# `similarity_map` (piq/functional/base.py). piq is by Sergey Kastryulin and its contributors,
# licensed under the Apache License, Version 2.0 (NOTICE; source log T381.1, T381.2).
# Changed for openftgs (#381): the defaults D-89 sets (3 scales, subsampled, c = 30, α = 4.2) and
# a data range of 1 are fixed; `_validate_input`'s asserts give way to Q-41's checks, and its
# check of the values, which reads them back to the host, to D-89's clamp of the render
# (Q-T381.1); the YIQ matrix and the Haar filters are built once per device and dtype rather
# than at every call, and the pairs of Haar channels are taken by strided slices rather than by
# tuple indices, since both copies to the GPU synchronize (§2.4 rule 3); the helpers are inlined;
# "mean" is the only reduction.


@functools.cache
def _haarpsi_constants(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, list[Tensor]]:
    """The RGB-to-YIQ matrix, transposed for a matmul over the channels-last image, and the
    stacked horizontal and vertical Haar filters of each scale, on `device` in `dtype`. Made
    outside inference mode, so that a loss first called by a metric under
    `torch.inference_mode()` still serves a training step's backward."""
    with torch.inference_mode(False):
        yiq = torch.tensor(
            [[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]],
            dtype=dtype,
            device=device,
        ).t()
        kernels = []
        for scale in range(_HAARPSI_SCALES):
            size = 2 ** (scale + 1)
            haar = torch.ones((size, size), device=device, dtype=dtype) / size
            haar[size // 2 :, :] = -haar[size // 2 :, :]
            haar = haar.unsqueeze(0)
            kernels.append(torch.stack([haar, haar.transpose(-1, -2)]))
    return yiq, kernels


def _similarity(x: Tensor, y: Tensor, c: float) -> Tensor:
    """piq's `similarity_map` with its masking coefficient at its default 0, whose terms are
    dropped."""
    return (2.0 * x * y + c) / (x**2 + y**2 + c)


def _haarpsi(x: Tensor, y: Tensor) -> Tensor:
    """HaarPSI of [B, 3, H, W] images in [0, 1], the mean over the batch, as piq 0.8.0's
    `haarpsi` at D-89's settings computes it."""
    # Q-T381.1: piq's minimal size, checked before the subsampling.
    kernel_size = 2 ** (_HAARPSI_SCALES + 1)
    if x.size(-1) < kernel_size or x.size(-2) < kernel_size:
        _fail(f"HaarPSI needs H and W >= {kernel_size}, got {tuple(x.shape[-2:])}")
    yiq_weights, kernels = _haarpsi_constants(x.device, x.dtype)

    # Rescale images, from a data range of 1
    x = x * 255
    y = y * 255

    # Convert RGB to YIQ color space
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    y_yiq = torch.matmul(y.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)

    # Downscale input to simulate the typical distance between an image and its viewer.
    up_pad = 0
    down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
    pad_to_use = [up_pad, down_pad, up_pad, down_pad]
    x_yiq = F.pad(x_yiq, pad=pad_to_use)
    y_yiq = F.pad(y_yiq, pad=pad_to_use)
    x_yiq = F.avg_pool2d(x_yiq, kernel_size=2, stride=2, padding=0)
    y_yiq = F.avg_pool2d(y_yiq, kernel_size=2, stride=2, padding=0)

    # Haar wavelet decomposition
    coefficients_x, coefficients_y = [], []
    for scale in range(_HAARPSI_SCALES):
        kernel_size = 2 ** (scale + 1)
        # Asymmetrical padding due to even kernel size. Matches MATLAB conv2(A, B, 'same')
        upper_pad = kernel_size // 2 - 1
        bottom_pad = kernel_size // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        coefficients_x.append(F.conv2d(F.pad(x_yiq[:, :1], pad=pad_to_use), kernels[scale]))
        coefficients_y.append(F.conv2d(F.pad(y_yiq[:, :1], pad=pad_to_use), kernels[scale]))

    # Shape (N, {scales * 2}, H, W)
    coeffs_x = torch.cat(coefficients_x, dim=1)
    coeffs_y = torch.cat(coefficients_y, dim=1)

    # Low-frequency coefficients used as weights
    # Shape (N, 2, H, W)
    weights = torch.max(torch.abs(coeffs_x[:, 4:]), torch.abs(coeffs_y[:, 4:]))

    # High-frequency coefficients used for similarity computation in 2 orientations
    sim_map = []
    for orientation in range(2):
        # The channels (orientation, orientation + 2), as a strided slice: piq's tuple index
        # copies an index tensor to the device, which synchronizes (§2.4 rule 3).
        pair = slice(orientation, orientation + 3, 2)
        magnitude_x = torch.abs(coeffs_x[:, pair])
        magnitude_y = torch.abs(coeffs_y[:, pair])
        similarity = _similarity(magnitude_x, magnitude_y, _HAARPSI_C)
        sim_map.append(similarity.sum(dim=1, keepdim=True) / 2)

    pad_to_use = [0, 1, 0, 1]
    x_yiq = F.pad(x_yiq, pad=pad_to_use)
    y_yiq = F.pad(y_yiq, pad=pad_to_use)
    coefficients_x_iq = torch.abs(F.avg_pool2d(x_yiq[:, 1:], kernel_size=2, stride=1, padding=0))
    coefficients_y_iq = torch.abs(F.avg_pool2d(y_yiq[:, 1:], kernel_size=2, stride=1, padding=0))

    # Compute weights and similarity
    weights = torch.cat([weights, weights.mean(dim=1, keepdim=True)], dim=1)
    similarity = _similarity(coefficients_x_iq, coefficients_y_iq, _HAARPSI_C)
    sim_map.append(similarity.sum(dim=1, keepdim=True) / 2)

    sim = torch.cat(sim_map, dim=1)

    # Calculate the final score
    eps = torch.finfo(sim.dtype).eps
    score = (((sim * _HAARPSI_ALPHA).sigmoid() * weights).sum(dim=[1, 2, 3]) + eps) / (
        torch.sum(weights, dim=[1, 2, 3]) + eps
    )
    # Logit of score
    score = (torch.log(score / (1 - score)) / _HAARPSI_ALPHA) ** 2
    return score.mean(dim=0)


# End of the code adapted from piq.


class PerceptualLoss(torch.nn.Module):
    """L_perc: the perceptual network `net` of D-89 on the render and the training image, as the
    mean over the batch (E7, D-9). `forward(pred, gt)` takes Q-41's [..., H, W, 3] images; the
    reference gets no gradient. The network is frozen and stays in evaluation mode under
    `.train()` (D-89). "flip" is the vendored LDR-ꟻLIP loss at 67.0 pixels per degree, on
    float32 images; "haarpsi" is 1 − HaarPSI, the render clamped to [0, 1], on sides of at least
    16 pixels (Q-T381.1). The six trunk networks are built by #385 and #386."""

    def __init__(self, net: PercNet = "flip") -> None:
        super().__init__()
        if net not in _PERC_NETS:
            _fail(f"net must be one of D-89's perceptual networks {_PERC_NETS}, got {net!r}")
        if net in _TRUNK_TICKETS:
            # #381: the trunk networks come with #385 and #386.
            raise NotImplementedError(
                f"PerceptualLoss({net!r}) is not built yet: the trunk networks of D-89 come with "
                f"{_TRUNK_TICKETS[net]}"
            )
        self._name = net
        if net == "flip":
            self._flip = LDRFLIPLoss()
        # D-89: frozen and in evaluation mode, whatever a parent module asks.
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "PerceptualLoss":
        # D-89: in evaluation mode, also when a parent module calls `.train()`.
        return super().train(False)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt)
        # D-89: the reference without a gradient.
        x, y = _batch_first(pred), _batch_first(gt.detach())
        if self._name == "haarpsi":
            # D-89: 1 − HaarPSI, Î clamped to [0, 1].
            return 1.0 - _haarpsi(x.clamp(0.0, 1.0), y)
        # Q-T381.1: the published loss's constants are float32, and other dtypes fail inside it.
        if pred.dtype != torch.float32:
            _fail(f"ꟻLIP takes float32 images, got {pred.dtype}")
        out: Tensor = self._flip(x, y, pixels_per_degree=_FLIP_PIXELS_PER_DEGREE)
        return out


class LPIPSLoss(torch.nn.Module):
    """L_perc: LPIPS with the given backbone on images in [0, 1], mapped to [−1, 1] (E7, D-9,
    D-68), as the mean over the batch; `forward(…, map_inputs=False)` skips the mapping, for
    the lpips_vgg_raw metric (Q-58). The network is lpips 0.1.4's, frozen and kept in
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
            self._net: torch.nn.Module = lpips.LPIPS(net=net, verbose=False)
        self._name = net  # the perc_net value it was built for, as PerceptualLoss's
        # lpips freezes only its trunk; the linear layers would stay trainable.
        self._net.requires_grad_(False)
        self._net.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        # The linear layers hold dropout, which training mode would switch on (Q-41).
        super().train(mode)
        self._net.eval()
        return self

    # Q-58: map_inputs=False feeds the [0, 1] images unmapped, D-68's lpips_vgg_raw metric.
    def forward(self, pred: Tensor, gt: Tensor, *, map_inputs: bool = True) -> Tensor:
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
    """L_reg(t) = (1/N)·Σ σᵢ·sg[σt,ᵢ(t)] over all N primitives at normalized time t (E8, D-11).
    The stop-gradient is motion's (Q-12); the parameters are read at call time."""
    terms = reg4d_terms(model.opacity_logits, model.times, model.log_durations, t)
    # Q-41: an empty model gives 0, not the NaN of an empty mean.
    return terms.mean() if terms.numel() > 0 else terms.sum()


# D-89's table: each network's calibrated λ_perc, the weight `lambda_perc = None` stands for.
_CALIBRATED_WEIGHTS = {
    "flip": 0.415,
    "haarpsi": 0.0617,
    "vgg16_oxford": 0.00164,
    "alexnet_bvlc": 0.00408,
    "dinov2_s14": 0.00185,
    "dinov2_b14": 0.00204,
    "dino_s16": 0.00358,
    "dino_b16": 0.00484,
}


# Q-249: a private resolver, so that no §2.1 addition is needed.
def _resolve_lambda_perc(cfg: LossConfig) -> float:
    """λ_perc as `total_loss` applies it: `cfg.lambda_perc`, or the calibrated weight of
    `cfg.perc_net` when it is None (D-9, D-89)."""
    if cfg.lambda_perc is not None:
        return cfg.lambda_perc
    if cfg.perc_net not in _CALIBRATED_WEIGHTS:
        # #380's expand step: LPIPS's backbones, still accepted as perc_net until #392, have no
        # calibrated weight.
        _fail(
            f"lambda_perc = None takes perc_net's calibrated weight in D-89's table, which has "
            f"none for {cfg.perc_net!r}; set lambda_perc or choose one of {_PERC_NETS}"
        )
    return _CALIBRATED_WEIGHTS[cfg.perc_net]


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
    perceptual: PerceptualLoss | LPIPSLoss | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """L = λ_img·L1 + λ_ssim·(1 − SSIM) + λ_perc·L_perc + λ_reg·L_reg(t) at iteration `it` (E7,
    E9, D-37), and its unweighted terms with the total under "total". t is the training image's
    normalized time (§1.4), as for `reg4d_loss`. λ_perc = None is the calibrated weight of
    `cfg.perc_net` (D-89; Q-249). A term whose λ is 0 is neither computed nor reported. After
    `cfg.reg_until`, L_reg leaves the total but is still reported, computed without a gradient,
    so the keys stay fixed for a run (D-10, Q-41)."""
    # Q-41: iterations count from 1 (D-41).
    if isinstance(it, bool) or not isinstance(it, int) or it < 1:
        _fail(f"it must be an int >= 1, got {it!r}")
    lambda_perc = _resolve_lambda_perc(cfg)
    # D-77, Q-249: the resolved weight, not the configured one, asks for the network.
    if lambda_perc > 0.0 and perceptual is None:
        _fail(
            f"lambda_perc {lambda_perc} (configured {cfg.lambda_perc!r}) needs the perceptual "
            f"network (perceptual=PerceptualLoss({cfg.perc_net!r}))"
        )
    # Q-41: a network passed in must be the one LossConfig names (D-9), whatever λ_perc.
    if perceptual is not None and perceptual._name != cfg.perc_net:
        _fail(f"perceptual is {perceptual._name!r}, but cfg.perc_net is {cfg.perc_net!r}")
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
    if perceptual is not None and lambda_perc > 0.0:
        with stage("loss.perceptual"):
            terms["perceptual"] = perceptual(pred, gt)
            weighted.append(lambda_perc * terms["perceptual"])
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
