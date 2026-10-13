# SPDX-License-Identifier: Apache-2.0
"""E16's metrics in openftgs.eval: U-23 and the added tests of PRD #74 (D-29, D-40, D-52, D-64,
D-67 to D-69; Q-60 to Q-63).

Expected values are worked by hand (PSNR at a known MSE, SSIM_k of constant images in closed
form, crops built pixel by pixel) or come from torchmetrics' LPIPS, an oracle that shares no
code with lpips's wrapper here beyond the published weights.
"""

import math
import warnings

import pytest
import torch
from torch import Tensor

import openftgs.eval
from openftgs.eval import (
    METRIC_NAMES,
    average_metrics,
    dssim_k,
    dynamic_region,
    lpips_backbone,
    psnr,
    ssim_k,
)

F32 = torch.float32
F64 = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _noisy_pair(shape: tuple[int, ...], seed: int, *, noise: float = 0.1) -> tuple[Tensor, Tensor]:
    g = _gen(seed)
    gt = torch.rand(*shape, generator=g, dtype=F64)
    pred = (gt + noise * torch.randn(*shape, generator=g, dtype=F64)).clamp(0.0, 1.0)
    return pred, gt


def test_public_api() -> None:
    """Q-60: `__all__` is exactly the public API, and the metric names are those §2.1's CLI
    paragraph gives for METRICS.json."""
    assert sorted(openftgs.eval.__all__) == [
        "METRIC_NAMES",
        "average_metrics",
        "dssim_k",
        "dynamic_region",
        "lpips_backbone",
        "psnr",
        "ssim_k",
    ]
    assert METRIC_NAMES == (
        "psnr",
        "ssim1",
        "ssim2",
        "dssim1",
        "dssim2",
        "lpips_alex",
        "lpips_vgg",
        "lpips_vgg_raw",
    )


def test_u23_psnr_at_an_mse_of_1e_minus_4_is_40_db() -> None:
    """U-23: PSNR = 10·log10(1/MSE) (E16); |Δ| = 0.01 on every value gives MSE 10⁻⁴ and
    40 dB, to 1e-6."""
    gt = torch.full((12, 16, 3), 0.5, dtype=F64)
    pred = gt + 0.01
    pred[::2] = gt[::2] - 0.01
    value = psnr(pred, gt)
    assert value.shape == ()
    assert abs(value.item() - 40.0) <= 1e-6


def test_psnr_of_identical_images_is_infinite() -> None:
    """Q-60: MSE 0 gives +inf, not an error."""
    image = torch.rand(8, 8, 3, generator=_gen(0))
    assert psnr(image, image.clone()).item() == math.inf


@pytest.mark.parametrize("k", [1, 2])
def test_u23_dssim_is_one_minus_ssim_over_two(k: int) -> None:
    """U-23: DSSIM_k = (1 − SSIM_k)/2 (E16, D-29, D-40), to 1e-6, per image."""
    pred, gt = _noisy_pair((3, 24, 20, 3), seed=k)
    s = ssim_k(pred, gt, k)
    d = dssim_k(pred, gt, k)
    assert s.shape == d.shape == (3,)
    torch.testing.assert_close(d, (1.0 - s) / 2.0, rtol=0.0, atol=1e-6)


def test_u23_dssim2_is_below_dssim1_on_a_noisy_pair() -> None:
    """U-23: a larger data range raises C₁ and C₂, so SSIM₂ > SSIM₁ and DSSIM₂ < DSSIM₁."""
    pred, gt = _noisy_pair((24, 20, 3), seed=5)
    assert dssim_k(pred, gt, 2).item() < dssim_k(pred, gt, 1).item()


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(("a", "b"), [(0.25, 0.75), (0.9, 0.1), (0.5, 0.5)])
def test_u23_ssim_of_constant_images_has_its_closed_form(k: int, a: float, b: float) -> None:
    """U-23: for constant images a and b, every local variance and covariance is 0, so SSIM_k
    = (2ab + C₁)/(a² + b² + C₁) with C₁ = (0.01k)² (E16, D-67), to 1e-6."""
    c1 = (0.01 * k) ** 2
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    pred = torch.full((16, 13, 3), a, dtype=F64)
    gt = torch.full((16, 13, 3), b, dtype=F64)
    assert abs(ssim_k(pred, gt, k).item() - expected) <= 1e-6


def test_u23_a_render_above_one_is_clamped_before_metrics() -> None:
    """U-23, D-67: renders are clamped to [0, 1] before every metric, so a render with values
    above 1 (and below 0) scores exactly as its clamped copy."""
    pred, gt = _noisy_pair((2, 20, 24, 3), seed=7)
    raw = pred.clone()
    raw[:, :5] = 1.0 + torch.rand(2, 5, 24, 3, generator=_gen(8), dtype=F64)
    raw[:, -3:, :4] = -0.2
    clamped = raw.clamp(0.0, 1.0)
    assert not torch.equal(raw, clamped)
    assert torch.equal(psnr(raw, gt), psnr(clamped, gt))
    for k in (1, 2):
        assert torch.equal(ssim_k(raw, gt, k), ssim_k(clamped, gt, k))
        assert torch.equal(dssim_k(raw, gt, k), dssim_k(clamped, gt, k))


def test_u23_metrics_are_averaged_per_image() -> None:
    """U-23, D-67: each metric is computed per test image, then averaged. Images with MSE 10⁻²
    and 10⁻⁴ (20 and 40 dB) give a scene PSNR of 30 dB, not the 22.97 dB of the pooled MSE."""
    gt = torch.full((2, 8, 8, 3), 0.5, dtype=F64)
    pred = gt.clone()
    pred[0] += 0.1
    pred[1] += 0.01
    per_image = psnr(pred, gt)
    assert per_image.shape == (2,)
    rows = [{"psnr": float(v)} for v in per_image]
    assert abs(average_metrics(rows)["psnr"] - 30.0) <= 1e-6
    pooled = 10 * math.log10(1 / ((1e-2 + 1e-4) / 2))
    assert abs(pooled - 22.97) < 0.01


def test_average_metrics_is_the_mean_per_key() -> None:
    """D-67, Q-60: the arithmetic mean of each key, at both levels (images → scene, scenes →
    overall)."""
    rows = [
        {"psnr": 30.0, "ssim1": 0.5},
        {"psnr": 20.0, "ssim1": 0.75},
        {"psnr": 25.0, "ssim1": 1.0},
    ]
    assert average_metrics(rows) == {"psnr": 25.0, "ssim1": 0.75}
    with pytest.raises(ValueError):
        average_metrics([])
    with pytest.raises(ValueError, match="keys"):
        average_metrics([{"psnr": 1.0}, {"ssim1": 1.0}])


@pytest.mark.parametrize(
    ("source", "backbone"),
    [
        ("n3dv/flame_steak", "alex"),
        ("n3dv", "alex"),
        ("selfcap/dance1", "vgg"),
        ("enerf_outdoor/actor1", "vgg"),
        ("synthetic", "vgg"),
        ("n3dvx/scene", "vgg"),
        (None, "vgg"),
    ],
)
def test_u23_lpips_backbone_per_dataset(source: str | None, backbone: str) -> None:
    """U-23, D-29: LPIPS-Alex for Neural3DV, VGG otherwise, read from the scene manifest's
    metadata.source (Q-62)."""
    assert lpips_backbone(source) == backbone


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (torch.zeros(16, 16, 3), torch.zeros(16, 15, 3)),
        (torch.zeros(16, 16, 4), torch.zeros(16, 16, 4)),
        (torch.zeros(16, 3), torch.zeros(16, 3)),
        (torch.zeros(16, 16, 3, dtype=torch.int64), torch.zeros(16, 16, 3, dtype=torch.int64)),
    ],
)
def test_metrics_need_matching_rgb_images(pred: Tensor, gt: Tensor) -> None:
    """Q-60: images are [..., H, W, 3] of one shape and a floating dtype (Q-41)."""
    with pytest.raises(ValueError):
        psnr(pred, gt)
    with pytest.raises(ValueError):
        ssim_k(pred, gt, 1)


@pytest.mark.parametrize("k", [0, 3, 1.0, True, 1.5])
def test_ssim_k_takes_a_data_range_of_one_or_two(k: object) -> None:
    """E16, Q-60: k ∈ {1, 2}."""
    image = torch.rand(16, 16, 3, generator=_gen(1))
    with pytest.raises(ValueError, match="k"):
        ssim_k(image, image, k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k"):
        dssim_k(image, image, k)  # type: ignore[arg-type]


def test_metrics_run_without_tf32_and_restore_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-52, §2.4 Precision: evaluation clears cuDNN's TF32 flag for its forward pass and
    restores it afterwards (Q-60)."""
    seen: list[bool] = []
    original = openftgs.eval.ssim

    def spy(*args: object, **kwargs: object) -> Tensor:
        seen.append(torch.backends.cudnn.allow_tf32)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(openftgs.eval, "ssim", spy)
    pred, gt = _noisy_pair((16, 16, 3), seed=2)
    before = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        ssim_k(pred, gt, 1)
        dssim_k(pred, gt, 2)
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = before
    assert seen == [False, False]


# The dynamic region (App. B.1, D-67; Q-61).


def _l_mask() -> Tensor:
    """A 50×60 soft mask whose dynamic pixels (≥ 0.5) form an L: column 10 from row 5 to row
    44, and row 44 from column 10 to column 49. (5, 10) holds exactly 0.5, the rest of the L
    0.51; a pixel at (2, 30) holds 0.49 and lies outside, as does everything else (0.2)."""
    mask = torch.full((50, 60), 0.2, dtype=F32)
    mask[5:45, 10] = 0.51
    mask[44, 10:50] = 0.51
    mask[5, 10] = 0.5
    mask[2, 30] = 0.49
    return mask


def test_u23_dynamic_region_crops_to_the_mask_box_and_blacks_out_both_images() -> None:
    """U-23, App. B.1, D-67: masks binarize at 0.5 (Q-61: 0.5 counts as dynamic); both images
    are clamped, set to black outside the mask, and cropped to the mask's bounding box, here
    rows 5 … 44 and columns 10 … 49 (40×40). Exact."""
    g = _gen(3)
    pred = torch.rand(50, 60, 3, generator=g) * 1.4
    gt = torch.rand(50, 60, 3, generator=g)
    region = dynamic_region(pred, gt, _l_mask())
    assert region is not None
    pred_crop, gt_crop = region
    expected_pred = torch.zeros(40, 40, 3)
    expected_gt = torch.zeros(40, 40, 3)
    for row in range(5, 45):
        expected_pred[row - 5, 0] = pred[row, 10].clamp(0.0, 1.0)
        expected_gt[row - 5, 0] = gt[row, 10]
    for col in range(10, 50):
        expected_pred[39, col - 10] = pred[44, col].clamp(0.0, 1.0)
        expected_gt[39, col - 10] = gt[44, col]
    assert torch.equal(pred_crop, expected_pred)
    assert torch.equal(gt_crop, expected_gt)


def test_u23_pixels_outside_the_mask_do_not_count() -> None:
    """D-67: masking both images keeps the background outside the mask out of the metric; a
    render that differs from the ground truth only there scores as a perfect render."""
    g = _gen(4)
    gt = torch.rand(50, 60, 3, generator=g)
    pred = torch.rand(50, 60, 3, generator=g)
    mask = _l_mask() >= 0.5
    pred[mask] = gt[mask]
    region = dynamic_region(pred, gt, mask)
    assert region is not None
    assert torch.equal(region[0], region[1])
    assert psnr(*region).item() == math.inf


def test_dynamic_region_takes_masks_with_a_channel_and_bool_masks() -> None:
    """Q-61: masks are [H, W] or [H, W, 1], bool or floating."""
    g = _gen(5)
    pred, gt = torch.rand(50, 60, 3, generator=g), torch.rand(50, 60, 3, generator=g)
    soft = _l_mask()
    expected = dynamic_region(pred, gt, soft)
    assert expected is not None
    for mask in (soft[..., None], soft >= 0.5, (soft >= 0.5)[..., None]):
        region = dynamic_region(pred, gt, mask)
        assert region is not None
        assert torch.equal(region[0], expected[0]) and torch.equal(region[1], expected[1])


@pytest.mark.parametrize(
    ("rows", "cols", "usable"),
    [((0, 0), (0, 0), False), ((3, 33), (2, 40), False), ((3, 34), (2, 32), False),
     ((3, 34), (2, 33), True)],
)  # fmt: skip
def test_dynamic_region_needs_a_box_every_metric_accepts(
    rows: tuple[int, int], cols: tuple[int, int], usable: bool
) -> None:
    """Q-61: an empty mask, or a box narrower than 31 px on a side (LPIPS-Alex's smallest
    input), gives no region and a warning; a 31×31 box is usable."""
    g = _gen(6)
    pred, gt = torch.rand(50, 60, 3, generator=g), torch.rand(50, 60, 3, generator=g)
    mask = torch.zeros(50, 60, dtype=torch.bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    if usable:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            region = dynamic_region(pred, gt, mask)
        assert region is not None and region[0].shape == (31, 31, 3)
    else:
        with pytest.warns(UserWarning, match="dynamic"):
            assert dynamic_region(pred, gt, mask) is None


@pytest.mark.parametrize(
    "mask",
    [torch.zeros(50, 59), torch.zeros(50, 60, 3), torch.zeros(50, 60, dtype=torch.int64)],
)
def test_dynamic_region_rejects_a_mask_of_another_shape_or_type(mask: Tensor) -> None:
    image = torch.zeros(50, 60, 3)
    with pytest.raises(ValueError, match="mask"):
        dynamic_region(image, image, mask)
