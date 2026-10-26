# SPDX-License-Identifier: Apache-2.0
"""D-89's two Caffe trunk networks, `vgg16_oxford` and `alexnet_bvlc` (`openftgs._caffe_trunks`,
built by `PerceptualLoss`; #385): lpips's uncalibrated distance on Oxford's VGG16 and BVLC's
AlexNet, with Caffe's input map and ceiling pooling.

The module is private, so these are private tests. The CPU tests run the trunks on seeded
stand-in weights of the publishers' layer shapes, written out here from the layer
configurations (VGG16's configuration D, source log 420; BVLC's AlexNet prototxt, source log
421). The expected tap sizes are worked by hand from those configurations."""

import pytest
import torch
from torch import Tensor

from openftgs import _caffe_trunks

F64 = torch.float64

# The publishers' convolutions, [out, in / groups, kh, kw], in network order (source log 420,
# 421): written out, the tests' own expectation.
VGG16_SHAPES = {
    "conv1_1": (64, 3, 3, 3),
    "conv1_2": (64, 64, 3, 3),
    "conv2_1": (128, 64, 3, 3),
    "conv2_2": (128, 128, 3, 3),
    "conv3_1": (256, 128, 3, 3),
    "conv3_2": (256, 256, 3, 3),
    "conv3_3": (256, 256, 3, 3),
    "conv4_1": (512, 256, 3, 3),
    "conv4_2": (512, 512, 3, 3),
    "conv4_3": (512, 512, 3, 3),
    "conv5_1": (512, 512, 3, 3),
    "conv5_2": (512, 512, 3, 3),
    "conv5_3": (512, 512, 3, 3),
}
ALEXNET_SHAPES = {
    "conv1": (96, 3, 11, 11),
    "conv2": (256, 48, 5, 5),
    "conv3": (384, 256, 3, 3),
    "conv4": (384, 192, 3, 3),
    "conv5": (256, 192, 3, 3),
}
SHAPES = {"vgg16_oxford": VGG16_SHAPES, "alexnet_bvlc": ALEXNET_SHAPES}


def _stand_in(name: str, *, seed: int, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    """Seeded weights of the publisher's shapes under `openftgs._weights`'s keys: He-normal
    filters and small biases, so that the features neither vanish nor saturate."""
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, Tensor] = {}
    for layer, shape in SHAPES[name].items():
        fan_in = shape[1] * shape[2] * shape[3]
        weight = torch.randn(shape, generator=gen, dtype=F64) * (2.0 / fan_in) ** 0.5
        bias = 0.01 * torch.randn(shape[0], generator=gen, dtype=F64)
        tensors[f"{layer}.weight"] = weight.to(dtype)
        tensors[f"{layer}.bias"] = bias.to(dtype)
    return tensors


def _images(shape: tuple[int, ...], *, seed: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Uniform images in [0, 1] of shape [B, 3, H, W]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=gen, dtype=F64).to(dtype)


# (network, image [B, 3, H, W], each tap's channels, heights and widths). VGG16: 3×3
# convolutions keep the size, and 2×2 pooling of stride 2 in Caffe's ceiling mode gives
# ⌈n/2⌉ before blocks 2 to 5, so 37 → 19 → 10 → 5 → 3 (floor: 18, 9, 4, 2) and
# 41 → 21 → 11 → 6 → 3 (floor: 20, 10, 5, 2). AlexNet: conv1's 11×11 of stride 4 gives
# (n − 11)/4 + 1, then 3×3 pooling of stride 2 gives ⌈(n − 3)/2⌉ + 1 after conv1 and conv2,
# whose padding 2 keeps the size, as conv3 to conv5's padding 1 does: 23 → 4 → 2 → 1 (floor:
# 4 → 1, and then no 3×3 window fits) and 39 → 8 → 4 → 2 (floor: 8 → 3 → 1).
TAPS = [
    (
        "vgg16_oxford",
        (2, 3, 37, 41),
        (64, 128, 256, 512, 512),
        (37, 19, 10, 5, 3),
        (41, 21, 11, 6, 3),
    ),
    (
        "alexnet_bvlc",
        (1, 3, 23, 39),
        (96, 256, 384, 384, 256),
        (4, 2, 1, 1, 1),
        (8, 4, 2, 2, 2),
    ),
]


@pytest.mark.parametrize(("name", "shape", "channels", "heights", "widths"), TAPS)
def test__caffe_trunk_taps_d89s_layers_with_caffes_ceiling_pooling(
    name: str,
    shape: tuple[int, ...],
    channels: tuple[int, ...],
    heights: tuple[int, ...],
    widths: tuple[int, ...],
) -> None:
    """D-89's rows: VGG16's features after relu1_2, relu2_2, relu3_3, relu4_3 and relu5_3, and
    AlexNet's after relu1 to relu5, with Caffe's ceiling pooling (D-89's `vgg16_oxford` row;
    BVLC's own pooling, the construction #374 tested): five taps of the layers' channels, at
    the sizes the strides, paddings and ceiling pooling give, each after a ReLU."""
    trunk = _caffe_trunks.CaffeTrunk(name, _stand_in(name, seed=1))
    taps = trunk.features(_images(shape, seed=2))
    assert [tuple(tap.shape) for tap in taps] == [
        (shape[0], c, h, w) for c, h, w in zip(channels, heights, widths, strict=True)
    ]
    assert all(bool((tap >= 0.0).all()) and bool((tap > 0.0).any()) for tap in taps)


# D-89's mean pixels, BGR on 0 to 255: Oxford's as its row gives it, BVLC's as #374 tested it,
# the per-channel means of Caffe's ilsvrc_2012_mean.npy to eight decimals, which D-89's row
# rounds to (104.007, 116.669, 122.679) (Q-264).
MEAN_BGR = {
    "vgg16_oxford": (103.939, 116.779, 123.68),
    "alexnet_bvlc": (104.00698793, 116.66876762, 122.67891434),
}


def _caffe_lrn(x: Tensor) -> Tensor:
    """Caffe's cross-channel LRN of local size n = 5, α = 10⁻⁴, β = 0.75 and k = 1:
    b_c = a_c / (k + (α/n)·Σ a_c'²)^β over the n channels centred on c, written here from
    Caffe's definition, zero beyond the first and last channels."""
    n, alpha, beta, k = 5, 1e-4, 0.75, 1.0
    squares = torch.nn.functional.pad(x * x, (0, 0, 0, 0, n // 2, n // 2))
    window = sum(squares[:, i : i + x.shape[1]] for i in range(n))
    return x / (k + alpha / n * window) ** beta


def _caffe_pool(x: Tensor, kernel: int, stride: int, *, ceil: bool = True) -> Tensor:
    """Caffe's max pooling without padding, its output side ⌈(n − kernel)/stride⌉ + 1: the
    input padded at the bottom and right with −∞ until the last window fits, then pooled
    with whole windows only. ceil=False is the floor convention, for the test's control."""

    def extra(n: int) -> int:
        out = -(-(n - kernel) // stride) + 1 if ceil else (n - kernel) // stride + 1
        return max(0, (out - 1) * stride + kernel - n)

    h, w = x.shape[-2:]
    x = torch.nn.functional.pad(x, (0, extra(w), 0, extra(h)), value=float("-inf"))
    return torch.nn.functional.max_pool2d(x, kernel, stride)


def _oracle(
    name: str,
    tensors: dict[str, Tensor],
    x: Tensor,
    y: Tensor,
    *,
    bgr: bool = True,
    ceil: bool = True,
    lrn: bool = True,
) -> Tensor:
    """D-89's distance for [B, 3, H, W] RGB images in [0, 1], in float64, written from its rows:
    the input as BGR × 255 minus the mean pixel; VGG16's 3×3 convolutions (padding 1) with a
    ReLU each, the five blocks separated by 2×2 pooling of stride 2, tapped after each block;
    AlexNet's conv1 (stride 4), conv2 (padding 2, 2 groups), conv3 (padding 1), conv4 and
    conv5 (padding 1, 2 groups), each with a ReLU and tapped after it, the taps of conv1 and
    conv2 followed by LRN and 3×3 pooling of stride 2; each tap's feature vectors divided by
    their norm plus 10⁻¹⁰, the squared differences summed over channels, averaged over space
    and summed over the taps. bgr, ceil and lrn switch one part off, for the test's controls."""
    w = {key: value.to(F64) for key, value in tensors.items()}
    mean = torch.tensor(MEAN_BGR[name], dtype=F64).view(1, 3, 1, 1)

    def conv(f: Tensor, layer: str, stride: int = 1, padding: int = 1, groups: int = 1) -> Tensor:
        out = torch.nn.functional.conv2d(
            f, w[f"{layer}.weight"], w[f"{layer}.bias"], stride, padding, 1, groups
        )
        return out.clamp_min(0.0)

    def taps(image: Tensor) -> list[Tensor]:
        f = (image[:, [2, 1, 0]] if bgr else image).to(F64) * 255.0 - mean
        out = []
        if name == "vgg16_oxford":
            blocks = [["conv1_1", "conv1_2"], ["conv2_1", "conv2_2"]]
            blocks += [[f"conv{b}_{i}" for i in (1, 2, 3)] for b in (3, 4, 5)]
            for index, block in enumerate(blocks):
                if index:
                    f = _caffe_pool(f, 2, 2, ceil=ceil)
                for layer in block:
                    f = conv(f, layer)
                out.append(f)
            return out
        for layer, stride, padding, groups, after in [
            ("conv1", 4, 0, 1, True),
            ("conv2", 1, 2, 2, True),
            ("conv3", 1, 1, 1, False),
            ("conv4", 1, 1, 2, False),
            ("conv5", 1, 1, 2, False),
        ]:
            f = conv(f, layer, stride, padding, groups)
            out.append(f)
            if after:
                f = _caffe_pool(_caffe_lrn(f) if lrn else f, 3, 2, ceil=ceil)
        return out

    total = torch.zeros(x.shape[0], dtype=F64)
    for a, b in zip(taps(x), taps(y), strict=True):
        ua = a / (a.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
        ub = b / (b.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
        total = total + (ua - ub).square().sum(dim=1).mean(dim=(1, 2))
    return total


# (network, the images' [B, 3, H, W], the controls that must move the value): sizes at which
# Caffe's ceiling pooling and the floor convention give different taps (VGG16: 37 → 19, floor
# 18; AlexNet: 71 → 16 → 8, floor 7, then 4, floor 3).
DISTANCE = [
    ("vgg16_oxford", (2, 3, 37, 41), ("bgr", "ceil")),
    ("alexnet_bvlc", (2, 3, 71, 51), ("bgr", "ceil", "lrn")),
]


@pytest.mark.parametrize(("name", "shape", "controls"), DISTANCE)
def test__caffe_trunk_is_lpips_uncalibrated_distance_on_caffes_input_and_layers(
    name: str, shape: tuple[int, ...], controls: tuple[str, ...]
) -> None:
    """D-89's `vgg16_oxford` and `alexnet_bvlc` rows: the trunk's value per image is the float64
    distance the rows describe, written out in `_oracle` (its layers, Caffe's input map with
    the release's mean pixel, its LRN and its ceiling pooling), to 1e-10 relative, on
    float64 stand-in weights. Each control, the oracle with one part changed (RGB for BGR, the
    floor convention, no LRN), moves the value by more than 10⁻⁴ relative, so each part is
    tested."""
    tensors = _stand_in(name, seed=3, dtype=F64)
    trunk = _caffe_trunks.CaffeTrunk(name, tensors)
    x = _images(shape, seed=4, dtype=F64)
    y = (x + 0.1 * _images(shape, seed=5, dtype=F64)).clamp(0.0, 1.0)
    value = trunk(x, y)
    expected = _oracle(name, tensors, x, y)
    assert value.shape == (shape[0],) and value.dtype == F64
    torch.testing.assert_close(value, expected, rtol=1e-10, atol=0.0)
    for control in controls:
        changed = _oracle(name, tensors, x, y, **{control: False})
        assert bool(((changed - expected).abs() > 1e-4 * expected).all()), control


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("dtype", [F64, torch.float16, torch.bfloat16])
def test__caffe_trunk_takes_images_of_its_weights_dtype_only(name: str, dtype: torch.dtype) -> None:
    """Q-263: a trunk computes in its weights' dtype, float32 as converted, and refuses
    images of another dtype with a ValueError naming it, rather than casting them."""
    trunk = _caffe_trunks.CaffeTrunk(name, _stand_in(name, seed=6))
    image = _images((1, 3, 24, 24), seed=7, dtype=dtype)
    with pytest.raises(ValueError, match="float32"):
        trunk(image, image.clone())


@pytest.mark.parametrize("name", list(SHAPES))
def test__caffe_trunk_keeps_its_weights_out_of_the_state_dict(name: str) -> None:
    """Q-263 (§2.3: never redistributed): the weights and the mean pixel are buffers that no
    `state_dict` carries, which `.to()` still converts: converted to float64, the trunk
    takes float64 images, and gives on them the value of a trunk built in float64, to the
    float32 rounding of the mean pixel it was built with (1e-7 relative)."""
    tensors = _stand_in(name, seed=8)
    trunk = _caffe_trunks.CaffeTrunk(name, tensors)
    assert trunk.state_dict() == {}
    assert list(trunk.parameters()) == []
    assert len(list(trunk.buffers())) == 2 * len(SHAPES[name]) + 1
    built = _caffe_trunks.CaffeTrunk(name, {key: t.to(F64) for key, t in tensors.items()})
    x = _images((1, 3, 24, 28), seed=9, dtype=F64)
    y = _images((1, 3, 24, 28), seed=10, dtype=F64)
    trunk.to(F64)
    assert all(buffer.dtype == F64 for buffer in trunk.buffers())
    torch.testing.assert_close(trunk(x, y), built(x, y), rtol=1e-7, atol=0.0)


@pytest.mark.parametrize("shape", [(1, 3, 22, 40), (1, 3, 40, 22), (2, 3, 22, 22)])
def test__caffe_trunk_alexnet_needs_sides_of_23_pixels(shape: tuple[int, ...]) -> None:
    """Q-263: AlexNet's layers need a height and width of at least 23 pixels (conv1's 11×11
    of stride 4 gives 4, which its two 3×3 poolings of stride 2 take to 2 and 1); a smaller
    side is a ValueError naming 23, before any work, and 23 runs. VGG16, whose ceiling pooling
    takes a side of 1 to 1, runs at any size."""
    alexnet = _caffe_trunks.CaffeTrunk("alexnet_bvlc", _stand_in("alexnet_bvlc", seed=11))
    image = _images(shape, seed=12)
    with pytest.raises(ValueError, match="23"):
        alexnet(image, image.clone())
    smallest = _images((1, 3, 23, 23), seed=13)
    assert alexnet(smallest, smallest.clone()).shape == (1,)
    vgg = _caffe_trunks.CaffeTrunk("vgg16_oxford", _stand_in("vgg16_oxford", seed=14))
    pixel = _images((1, 3, 1, 1), seed=15)
    assert vgg(pixel, pixel.clone()).shape == (1,)


@pytest.mark.weights
@pytest.mark.parametrize("name", list(SHAPES))
def test_d89_caffe_trunk_is_the_rows_distance_on_the_publishers_weights(name: str) -> None:
    """D-89's `vgg16_oxford` and `alexnet_bvlc` rows at the public seam: `PerceptualLoss(name)`
    on [..., H, W, 3] float32 images is the mean over the batch of the float64 distance the
    rows describe (`_oracle`), on the publisher's converted weights (Q-7, Q-258), to 1e-5
    relative (float32's rounding). The render isn't clamped: a patch raised above 1 changes
    the value from that of the render clamped to [0, 1]."""
    from openftgs._weights import trunk_tensors
    from openftgs.losses import PerceptualLoss

    loss = PerceptualLoss(name)  # type: ignore[arg-type]
    gt = _images((2, 48, 56, 3), seed=16)
    pred = (gt + 0.2 * _images((2, 48, 56, 3), seed=17) - 0.1).clamp(0.0, 1.0)
    pred[:, 10:30, 12:40] += 0.3
    assert bool((pred > 1.0).any())
    value = loss(pred, gt)
    tensors = trunk_tensors(name)
    nchw = [image.permute(0, 3, 1, 2).to(F64) for image in (pred, gt)]
    expected = _oracle(name, tensors, *nchw).mean()
    assert value.shape == () and value.dtype == torch.float32
    torch.testing.assert_close(value.double(), expected, rtol=1e-5, atol=0.0)
    clamped = loss(pred.clamp(0.0, 1.0), gt)
    assert abs(clamped.item() - value.item()) > 1e-3 * value.item()
