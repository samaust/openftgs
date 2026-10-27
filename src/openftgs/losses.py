# SPDX-License-Identifier: Apache-2.0
"""L1, SSIM, the perceptual networks (D-89), 4D regularization and the total loss.

Spec: E7, E8, E9, D-89. Band: Mechanics (§2.1).
"""

import functools
import importlib
from collections.abc import Callable
from typing import Any, Literal, NoReturn, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from openftgs import _weights
from openftgs._caffe_trunks import CAFFE_TRUNKS, CaffeTrunk
from openftgs._stages import stage
from openftgs._vendor.flip import LDRFLIPLoss
from openftgs._weights import trunk_tensors
from openftgs.config import LossConfig, PercNet
from openftgs.model import FreeTimeGaussians
from openftgs.motion import reg4d_terms

__all__ = ["PerceptualLoss", "l1_loss", "reg4d_loss", "ssim", "total_loss"]


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


# D-89: the eight networks `perc_net` selects.
_PERC_NETS: tuple[str, ...] = get_args(PercNet)
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
# licensed under the Apache License, Version 2.0 (NOTICE; source log 446, 447).
# Changed for openftgs (#381): the defaults D-89 sets (3 scales, subsampled, c = 30, α = 4.2) and
# a data range of 1 are fixed; `_validate_input`'s asserts give way to Q-41's checks, and its
# check of the values, which reads them back to the host, to D-89's clamp of the render
# (Q-261); the YIQ matrix and the Haar filters are built once per device and dtype rather
# than at every call, and the pairs of Haar channels are taken by strided slices rather than by
# tuple indices, since both copies to the GPU synchronize (§2.4 rule 3); the helpers are inlined,
# `similarity_map` without its masking terms, whose coefficient is 0 here; "mean" is the only
# reduction. On random images it agrees with piq bitwise (source log 448).


@functools.cache
def _haarpsi_constants(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, list[Tensor]]:
    """Build HaarPSI's RGB-to-YIQ matrix and Haar filters on `device` in `dtype`.

    The matrix is transposed for a matmul over the channels-last image, and the horizontal and
    vertical Haar filters of each scale are stacked. Made outside inference mode, so that a
    loss first called by a metric under `torch.inference_mode()` still serves a training
    step's backward.
    """
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
    """Compute piq's `similarity_map` with its masking coefficient at its default 0.

    The masking terms, whose coefficient is 0, are dropped.
    """
    return (2.0 * x * y + c) / (x**2 + y**2 + c)


def _haarpsi(x: Tensor, y: Tensor) -> Tensor:
    """Compute HaarPSI of [B, 3, H, W] images in [0, 1], the mean over the batch.

    As piq 0.8.0's `haarpsi` at D-89's settings computes it.
    """
    # Q-261: piq's minimal size, checked before the subsampling.
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


# The DINO networks (D-89's `dinov2_*` and `dino_*` rows; #386), on the vendored DINOv2 layers
# (D-65) and their publishers' weights (`openftgs._weights`). The construction is the one #374
# tested (D-89), adapted from its prototype `candidates.py` (`ViTDistance` and the helpers it
# calls; the owner's Apache-2.0 research code, SHA-256
# 4386e7ce8bc97a2f94ca6b9e0eb11cc0d5559b20ccc08b72cb59c85c8e3486eb; source log 452).

# D-89: each network's vendored builder, its patch size, the side of the square image its
# publisher's position embeddings were made for, and its layer scale's initial value (DINOv2's;
# DINO v1 has none), so that the publisher's state dict loads into the builder as saved.
_DINO_TRUNKS: dict[str, tuple[str, int, int, float | None]] = {
    "dinov2_s14": ("vit_small", 14, 518, 1.0),
    "dinov2_b14": ("vit_base", 14, 518, 1.0),
    "dino_s16": ("vit_small", 16, 224, None),
    "dino_b16": ("vit_base", 16, 224, None),
}
# D-89: the patch tokens after blocks 3, 6, 9 and 12 of the twelve, counted from 0.
_DINO_BLOCKS = [2, 5, 8, 11]
# D-89: ImageNet's standardization.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# lpips's ε in the unit normalization, which #374's distances share (source log 435).
_UNIT_EPS = 1e-10


def _vendored_dinov2(net: str) -> Any:
    """Import the vendored DINOv2 builders (D-65), only when a DINO network is built.

    Their package's `__init__` modules import torchvision, einops and loguru, which the [init]
    extra installs (Q-248).
    """
    try:
        return importlib.import_module("openftgs._vendor.romatch.models.transformer.dinov2")
    except ImportError as error:  # Q-248; Q-142: the CLI's exit 2
        raise ImportError(
            f"PerceptualLoss({net!r}) needs the [init] extra (pip install 'openftgs[init]'), "
            f"whose packages the vendored DINOv2 layers import: {error}"
        ) from error


def _halved(image: Tensor) -> tuple[int, int]:
    """Return D-89's half resolution of [..., H, W] images (Q-265).

    Each side is halved and rounded half to even, as #374's prototype rounds it.
    """
    return round(image.shape[-2] / 2), round(image.shape[-1] / 2)


def _unit_tokens(tokens: Tensor) -> Tensor:
    """Scale each token's vector [..., C] to unit length, as lpips normalizes its features."""
    return tokens / (tokens.pow(2).sum(dim=-1, keepdim=True).sqrt() + _UNIT_EPS)


class _DinoDistance(torch.nn.Module):
    """Compute D-89's distance over four blocks of the DINO network `net`.

    The mean over a batch of [B, 3, H, W] images: each image at half resolution (bilinear,
    antialiased), centre-cropped to a multiple of the patch and standardized; the patch tokens
    after blocks 3, 6, 9 and 12, unit-normalized, their squared differences summed over
    channels, averaged over tokens and summed over the blocks. The position embeddings are
    interpolated by the vendored model's own code. It computes in its weights' dtype, float32
    as built, and holds them as buffers that no `state_dict` carries and `.to()` converts, as
    the Caffe trunks (Q-263, Q-265).
    """

    _mean: Tensor
    _std: Tensor

    def __init__(self, net: str) -> None:
        super().__init__()
        builder, patch, img_size, layer_scale = _DINO_TRUNKS[net]
        vendored = _vendored_dinov2(net)  # Q-248: before anything is downloaded
        # block_chunks=0 keeps the publisher's key names (`blocks.<i>.…`).
        vit = getattr(vendored, builder)(
            patch_size=patch, img_size=img_size, init_values=layer_scale, block_chunks=0
        )
        loaded = vit.load_state_dict(_weights.trunk_tensors(net), strict=False)
        # DINO v1 has no mask token, which the vendored model uses only with masks (#374).
        missing = [] if net.startswith("dinov2") else ["mask_token"]
        if loaded.unexpected_keys or loaded.missing_keys != missing:
            # The file passed its hash, so the builder's arguments are wrong.
            raise RuntimeError(
                f"the publisher's state dict of {net} doesn't fit the vendored {builder}: "
                f"missing {loaded.missing_keys}, unexpected {loaded.unexpected_keys}"
            )
        # Q-265: the weights as buffers no `state_dict` carries, as the Caffe trunks (Q-263;
        # §2.3: never redistributed), which `.to()` still moves and converts. The vendored
        # layers read them as attributes, so their files stay unchanged (D-65).
        for module in vit.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                delattr(module, name)
                module.register_buffer(name, parameter.detach(), persistent=False)
        self._vit: Any = vit
        self._net = net
        self._patch = patch
        dtype = vit.patch_embed.proj.weight.dtype  # Q-265: the weights' dtype, float32 as built
        mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1)
        # Buffers, so that `.to(device)` moves them and no call copies them to the device.
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _tokens(self, image: Tensor) -> list[Tensor]:
        """Return the patch tokens [B, T, C] after the four blocks, for [B, 3, H, W] images."""
        # D-89: half resolution, bilinear and antialiased (Q-265: the rounding).
        half = _halved(image)
        image = F.interpolate(
            image, size=half, mode="bilinear", align_corners=False, antialias=True
        )
        # D-89: centre-cropped to a multiple of the patch.
        crop_h, crop_w = (side // self._patch * self._patch for side in half)
        top, left = (half[0] - crop_h) // 2, (half[1] - crop_w) // 2
        image = image[..., top : top + crop_h, left : left + crop_w]
        image = (image - self._mean) / self._std  # D-89: ImageNet's standardization
        # D-89: after blocks 3, 6, 9 and 12, before the final norm, without the class token.
        return list(self._vit.get_intermediate_layers(image, n=_DINO_BLOCKS, norm=False))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Q-265: the weights' dtype, float32 as built, never a cast, as the Caffe trunks
        # (Q-263); at least one patch at half resolution, 2p − 1 pixels for an even patch p.
        # Shapes and dtypes only.
        dtype = self._mean.dtype
        if x.dtype != dtype:
            _fail(
                f"PerceptualLoss({self._net!r}) computes in its weights' dtype {dtype}, so it "
                f"takes {dtype} images, got {x.dtype}"
            )
        if min(_halved(x)) < self._patch:
            _fail(
                f"PerceptualLoss({self._net!r}) needs H and W >= {2 * self._patch - 1}, one "
                f"{self._patch}-pixel patch at half resolution, got {tuple(x.shape[-2:])}"
            )
        with torch.no_grad():  # D-89: the reference without a gradient
            reference = self._tokens(y)
        total = x.new_zeros(x.shape[0])
        for a, b in zip(self._tokens(x), reference, strict=True):
            total = total + (_unit_tokens(a) - _unit_tokens(b)).pow(2).sum(dim=-1).mean(dim=-1)
        return total.mean()


class PerceptualLoss(torch.nn.Module):
    """The perceptual loss L_perc, one of D-89's eight networks on images in [0, 1] (E7, D-9).

    ``net`` selects the network (:data:`openftgs.config.PercNet`); the module computes the
    network's distance between the render and the training image, as the mean over the batch,
    the reference without a gradient. The network is frozen and stays in evaluation mode,
    whatever :meth:`train` is given (D-89). Calling the module, ``loss(pred, gt)``, runs
    :meth:`forward`. The trainer builds one network and passes it to :func:`total_loss`
    (D-77); the evaluator builds ``"flip"`` for its ꟻLIP metric (E16).

    Parameters
    ----------
    net
        The network, ``"flip"`` by default (D-89):

        - ``"flip"``: NVIDIA's LDR-ꟻLIP loss, vendored (D-65), at 67.0 pixels per degree, the
          mean of its error map; it clamps both images to [0, 1] (Q-259);
        - ``"haarpsi"``: 1 − HaarPSI at 3 scales, subsampled, c = 30, α = 4.2, the render
          clamped to [0, 1];
        - ``"vgg16_oxford"``, ``"alexnet_bvlc"``: the distance between unit-normalized
          features of five layers of the Caffe releases of Oxford's VGG16 and BVLC's AlexNet,
          every channel weighted equally, at full resolution;
        - ``"dinov2_s14"``, ``"dinov2_b14"``, ``"dino_s16"``, ``"dino_b16"``: the distance
          between the patch tokens of DINOv2's or DINO's blocks 3, 6, 9 and 12, at half
          resolution.

    Raises
    ------
    ValueError
        If ``net`` is not one of D-89's eight networks.
    ImportError
        If ``net`` is a DINO network and the ``[init]`` extra isn't installed, naming the
        extra (Q-248).
    OSError
        If a trunk network's weights can't be downloaded, fail their check, or can't be
        converted into the cache (Q-258).

    Notes
    -----
    ``"flip"`` and ``"haarpsi"`` have no weights. The six trunk networks download their
    publisher's weights at first use into torch.hub's cache, ``torch.hub.get_dir()``, and check
    them against their byte count and SHA-256 before every use; a Caffe file is converted once
    into a safetensors file beside it (§2.3; Q-258). Their weights are buffers that no
    ``state_dict`` carries, which ``.to()`` moves and converts (Q-263, Q-265).

    The inputs each network refuses, with a ValueError raised on shapes and dtypes alone,
    before any work and without a cast (Q-261, Q-263, Q-265): ``"flip"`` takes float32 images
    only; ``"haarpsi"`` float32 or float64 images whose height and width are at least 16
    pixels; the trunk networks images of their weights' dtype, float32 as built, and
    ``"alexnet_bvlc"`` sides of at least 23 pixels, the DINOv2 networks 27 and the DINO
    networks 31, one patch at half resolution.

    Under a caller's :class:`torch.autocast`, ``"flip"`` and ``"haarpsi"`` turn autocast off
    for their own computation, which stays in their images' dtype; the trunk networks run
    under it, as the trainer's bfloat16 path asks (Q-275, Q-250).

    Examples
    --------
    >>> import torch
    >>> from openftgs.losses import PerceptualLoss
    >>> loss = PerceptualLoss("flip")
    >>> pred = torch.rand(64, 96, 3, requires_grad=True)
    >>> gt = torch.rand(64, 96, 3)
    >>> value = loss(pred, gt)
    >>> value.shape
    torch.Size([])
    >>> value.backward()
    """

    def __init__(self, net: PercNet = "flip") -> None:
        super().__init__()
        if net not in _PERC_NETS:
            _fail(f"net must be one of D-89's perceptual networks {_PERC_NETS}, got {net!r}")
        self._name = net
        if net == "flip":
            self._flip = LDRFLIPLoss()
        elif net in CAFFE_TRUNKS:
            # D-89 (#385): the Caffe trunks on their publishers' weights, fetched and checked
            # at first use (#384; Q-142: OSError, exit 2).
            self._caffe = CaffeTrunk(net, trunk_tensors(net))
        elif net in _DINO_TRUNKS:
            self._dino = _DinoDistance(net)
        # D-89: frozen and in evaluation mode, whatever a parent module asks.
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "PerceptualLoss":
        """Keep the network in evaluation mode, whatever ``mode`` asks (D-89).

        A parent module's ``.train()`` reaches this module too; it stays frozen and in
        evaluation mode.

        Parameters
        ----------
        mode
            Ignored; accepted for :meth:`torch.nn.Module.train`'s signature.

        Returns
        -------
        openftgs.losses.PerceptualLoss
            This module.
        """
        # D-89: in evaluation mode, also when a parent module calls `.train()`.
        return super().train(False)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute the network's distance between ``pred`` and ``gt`` (E7, D-89).

        Parameters
        ----------
        pred
            The render Î, [..., H, W, 3], floating point, not clamped (E7); any leading
            dimensions form a batch (Q-41).
        gt
            The training image I, RGB in [0, 1], with ``pred``'s shape, dtype and device; it
            gets no gradient.

        Returns
        -------
        torch.Tensor
            The distance, a 0-d tensor, the mean over the batch.

        Raises
        ------
        ValueError
            If the images differ in shape, dtype or device, aren't [..., H, W, 3] floating
            point, or are of a dtype or size the network refuses.
        """
        _check_pair(pred, gt)
        # D-89: the reference without a gradient.
        x, y = _batch_first(pred), _batch_first(gt.detach())
        if self._name in CAFFE_TRUNKS:
            # D-89: the images as the caller lays them out; the trainer's subclass gives a
            # Caffe trunk NCHW inputs where B-4 measured them faster (Q-273).
            caffe: Tensor = self._caffe(x, y).mean()
            return caffe
        if self._name in _DINO_TRUNKS:
            distance: Tensor = self._dino(x, y)
            return distance
        if self._name == "haarpsi":
            # Q-261: computed in the images' dtype, never a cast; float16 overflows to a NaN and
            # bfloat16 is an approximation B-4 never measured. The dtype only, so nothing
            # synchronizes.
            if pred.dtype not in (torch.float32, torch.float64):
                _fail(f"HaarPSI takes float32 or float64 images, got {pred.dtype}")
            # Q-275: in the images' dtype under a caller's autocast too.
            with torch.autocast(x.device.type, enabled=False):
                # D-89: 1 − HaarPSI, Î clamped to [0, 1].
                return 1.0 - _haarpsi(x.clamp(0.0, 1.0), y)
        # Q-261: the published loss's constants are float32, and other dtypes fail inside it.
        if pred.dtype != torch.float32:
            _fail(f"ꟻLIP takes float32 images, got {pred.dtype}")
        # Q-275: in float32 under a caller's autocast too.
        with torch.autocast(x.device.type, enabled=False):
            out: Tensor = self._flip(x, y, pixels_per_degree=_FLIP_PIXELS_PER_DEGREE)
        return out


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
    """Return λ_perc as `total_loss` applies it (D-9, D-89).

    `cfg.lambda_perc`, or the calibrated weight of `cfg.perc_net` when it is None.
    """
    if cfg.lambda_perc is not None:
        return cfg.lambda_perc
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
    perceptual: PerceptualLoss | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the total loss L of E9 at iteration ``it``, and its unweighted terms (E9, D-77).

    ``L = λ_img·L1 + λ_ssim·(1 − SSIM) + λ_perc·L_perc + λ_reg·L_reg(t)`` (E7, E9, D-37), from
    :func:`l1_loss`, :func:`ssim`, :class:`PerceptualLoss` and :func:`reg4d_loss`.
    ``cfg.lambda_perc = None`` stands for the calibrated weight of ``cfg.perc_net``, 0.415 for
    ``"flip"`` (D-9, D-89; Q-249). A term whose λ is 0 is neither computed nor reported. After
    ``cfg.reg_until``, L_reg leaves the total but is still reported, computed without a
    gradient, so the keys stay fixed for a run (D-10, Q-41). Each term runs as a stage of its
    own, for the benchmarks (§3.4, B-1).

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
        perceptual network (D-9).
    perceptual
        The perceptual network, built once by the caller as ``PerceptualLoss(cfg.perc_net)``;
        required when the resolved λ_perc is positive, and unused otherwise. Its network must
        be ``cfg.perc_net`` in either case (Q-41).

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
        If ``it`` is not an int ≥ 1, the resolved λ_perc is positive and ``perceptual`` is
        None (D-77; Q-249), ``perceptual``'s network differs from ``cfg.perc_net``, or the
        images fail :func:`l1_loss`'s checks.

    Notes
    -----
    D-77 changes §2.1's signature: the terms stay on the device, so the loss never synchronizes
    with the host (§2.4 rule 3), and the perceptual network is passed in rather than built
    here. The trainer accumulates the terms and reads them every ``log_every`` iterations.
    Q-249: D-77's raise reads the resolved weight, so the default configuration, whose
    ``lambda_perc`` is None, needs the network.

    Examples
    --------
    >>> from openftgs.config import LossConfig
    >>> from openftgs.losses import PerceptualLoss, total_loss
    >>> cfg = LossConfig()  # perc_net "flip", lambda_perc None: 0.415
    >>> perceptual = PerceptualLoss(cfg.perc_net).to(pred.device)
    >>> loss, terms = total_loss(pred, gt, model, t, it, cfg, perceptual=perceptual)
    >>> loss.backward()
    >>> sorted(terms)
    ['l1', 'one_minus_ssim', 'perceptual', 'reg', 'total']
    """
    # Q-41: iterations count from 1 (D-41).
    if isinstance(it, bool) or not isinstance(it, int) or it < 1:
        _fail(f"it must be an int >= 1, got {it!r}")
    lambda_perc = _resolve_lambda_perc(cfg)
    # D-77, Q-249: the resolved weight, not the configured one, asks for the network.
    if lambda_perc > 0.0 and perceptual is None:
        _fail(
            f"lambda_perc {lambda_perc} (configured {cfg.lambda_perc!r}) needs the perceptual "
            f"network of perc_net {cfg.perc_net!r}, passed as perceptual="
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
