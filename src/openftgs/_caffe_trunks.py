# SPDX-License-Identifier: Apache-2.0
"""D-89's two Caffe trunk networks, `vgg16_oxford` and `alexnet_bvlc`: lpips's uncalibrated
distance on Oxford's VGG16 and BVLC's AlexNet, built from the convolutions `openftgs._weights`
converts from the publishers' Caffe files (#384).

Spec: D-89, §2.3's pretrained-trunk row. A private top-level module that imports nothing from
openftgs: `openftgs.losses` passes it the tensors and wraps it in `PerceptualLoss` (#385).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CAFFE_TRUNKS", "CaffeTrunk"]


# The construction below is adapted from #374's prototype `candidates.py` (the owner's
# Apache-2.0 research code, SHA-256
# 4386e7ce8bc97a2f94ca6b9e0eb11cc0d5559b20ccc08b72cb59c85c8e3486eb; source log T385.1), its
# `_CaffeTrunk`, `VGG16CC` and `AlexNetCC`: the construction #374 tested, which D-89 names.
# The layers are the publishers' (VGG16's configuration D, source log 420; BVLC's AlexNet
# prototxt, source log 421).


@dataclass(frozen=True)
class _Conv:
    """A convolution under its Caffe layer name, followed by its ReLU (Caffe's in-place one)."""

    name: str
    stride: int = 1
    padding: int = 1
    groups: int = 1


@dataclass(frozen=True)
class _Pool:
    """Max pooling in Caffe's ceiling mode (D-89)."""

    kernel: int
    stride: int


# A step of a trunk, in network order: a convolution and its ReLU, a pooling, Caffe's local
# response normalization ("lrn"), or a feature D-89's distance reads ("tap").
_Step = _Conv | _Pool | Literal["tap", "lrn"]

_VGG16: tuple[_Step, ...] = (
    *(_Conv("conv1_1"), _Conv("conv1_2"), "tap"),
    *(_Pool(2, 2), _Conv("conv2_1"), _Conv("conv2_2"), "tap"),
    *(_Pool(2, 2), _Conv("conv3_1"), _Conv("conv3_2"), _Conv("conv3_3"), "tap"),
    *(_Pool(2, 2), _Conv("conv4_1"), _Conv("conv4_2"), _Conv("conv4_3"), "tap"),
    *(_Pool(2, 2), _Conv("conv5_1"), _Conv("conv5_2"), _Conv("conv5_3"), "tap"),
)
# D-89: the taps after relu1 to relu5, before the normalization and pooling that follow relu1
# and relu2; conv2, conv4 and conv5 in two groups.
_ALEXNET: tuple[_Step, ...] = (
    *(_Conv("conv1", stride=4, padding=0), "tap", "lrn", _Pool(3, 2)),
    *(_Conv("conv2", padding=2, groups=2), "tap", "lrn", _Pool(3, 2)),
    *(_Conv("conv3"), "tap"),
    *(_Conv("conv4", groups=2), "tap"),
    *(_Conv("conv5", groups=2), "tap"),
)
_STEPS: dict[str, tuple[_Step, ...]] = {"vgg16_oxford": _VGG16, "alexnet_bvlc": _ALEXNET}

# D-89: each release's mean pixel, BGR on Caffe's 0 to 255 scale: Oxford's as published
# (source log 420); BVLC's the per-channel means of Caffe's ilsvrc_2012_mean.npy (source log
# 426) to eight decimals, as #374 tested them, which D-89's table rounds to three (Q-T385.2).
_MEAN_BGR = {
    "vgg16_oxford": (103.939, 116.779, 123.68),
    "alexnet_bvlc": (104.00698793, 116.66876762, 122.67891434),  # Q-T385.2
}

# The two trunks, in D-89's table's order.
CAFFE_TRUNKS: tuple[str, ...] = tuple(_STEPS)


class CaffeTrunk(torch.nn.Module):
    """The Caffe trunk `name` (one of `CAFFE_TRUNKS`) on `tensors`, its convolutions'
    `<layer>.weight` and `<layer>.bias` under Caffe's layer names, as `openftgs._weights`'s
    `trunk_tensors` returns them (D-89)."""

    def __init__(self, name: str, tensors: Mapping[str, Tensor]) -> None:
        super().__init__()
        self._steps = _STEPS[name]
        self._convs = [step for step in self._steps if isinstance(step, _Conv)]
        for conv in self._convs:
            self.register_buffer(f"{conv.name}_weight", tensors[f"{conv.name}.weight"])
            self.register_buffer(f"{conv.name}_bias", tensors[f"{conv.name}.bias"])
        dtype = tensors[f"{self._convs[0].name}.weight"].dtype
        mean = torch.tensor(_MEAN_BGR[name], dtype=dtype).view(1, 3, 1, 1)
        self.register_buffer("mean_bgr", mean)

    def features(self, image: Tensor) -> list[Tensor]:
        """The taps of D-89's distance for a [B, 3, H, W] RGB image in [0, 1]: Caffe's input,
        BGR × 255 minus the mean pixel, through the trunk's layers in network order."""
        x = image.flip(1) * 255.0 - self.get_buffer("mean_bgr")
        taps: list[Tensor] = []
        for step in self._steps:
            if step == "tap":
                taps.append(x)
            elif step == "lrn":
                x = F.local_response_norm(x, 5, alpha=1e-4, beta=0.75, k=1.0)
            elif isinstance(step, _Pool):
                x = F.max_pool2d(x, step.kernel, step.stride, ceil_mode=True)
            else:
                weight = self.get_buffer(f"{step.name}_weight")
                bias = self.get_buffer(f"{step.name}_bias")
                x = F.conv2d(
                    x, weight, bias, stride=step.stride, padding=step.padding, groups=step.groups
                )
                x = F.relu(x)
        return taps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """D-89's distance between the [B, 3, H, W] RGB images x (the render) and y (the
        reference), per image: lpips's uncalibrated distance, its `lpips=False` (source log
        350), over the five taps."""
        # D-89: the reference without a gradient.
        with torch.no_grad():
            reference = [_unit(tap) for tap in self.features(y)]
        total: Tensor | None = None
        for tap, unit_y in zip(self.features(x), reference, strict=True):
            # Squared differences summed over channels, averaged over space.
            layer = (_unit(tap) - unit_y).pow(2).sum(dim=1).mean(dim=(1, 2))
            total = layer if total is None else total + layer
        assert total is not None  # five taps
        return total


# D-89: lpips's `normalize_tensor` ε (source log 435).
_EPS = 1e-10


def _unit(features: Tensor) -> Tensor:
    """Each feature vector (dim 1) divided by its norm plus ε, as lpips's `normalize_tensor`."""
    return features / (features.pow(2).sum(dim=1, keepdim=True).sqrt() + _EPS)
