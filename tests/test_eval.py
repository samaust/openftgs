# SPDX-License-Identifier: Apache-2.0
"""E16's metrics in openftgs.eval: U-23 and the added tests of PRD #74 (D-29, D-40, D-52, D-64,
D-67 to D-69; Q-60 to Q-63).

Expected values are worked by hand (PSNR at a known MSE, SSIM_k of constant images in closed
form, crops built pixel by pixel) or come from torchmetrics' LPIPS, an oracle reached through
another library's code path to the same published weights (§2.3).
"""

import json
import math
import warnings
from typing import Any, Literal

import pytest
import torch
from torch import Tensor

import openftgs.eval
import openftgs.losses
from openftgs.eval import (
    METRIC_NAMES,
    Evaluator,
    ImageMetrics,
    average_metrics,
    dssim_k,
    dynamic_region,
    lpips_backbone,
    metrics_report,
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
        "Evaluator",
        "ImageMetrics",
        "METRIC_NAMES",
        "average_metrics",
        "dssim_k",
        "dynamic_region",
        "lpips_backbone",
        "metrics_report",
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


@pytest.mark.parametrize("dtype", [F32, F64])
def test_metrics_of_an_empty_batch_are_empty(dtype: torch.dtype) -> None:
    """Q-60: one value per image, so a batch of no images gives no values from every metric,
    not an error (from the second M9 milestone review)."""
    pred = torch.zeros(0, 16, 16, 3, dtype=dtype)
    gt = torch.zeros(0, 16, 16, 3, dtype=dtype)
    assert psnr(pred, gt).shape == (0,)
    for k in (1, 2):
        s, d = ssim_k(pred, gt, k), dssim_k(pred, gt, k)
        assert s.shape == d.shape == (0,)
        assert s.dtype == d.dtype == dtype
    assert ssim_k(torch.zeros(2, 0, 16, 16, 3), torch.zeros(2, 0, 16, 16, 3), 1).shape == (2, 0)


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
    original = openftgs.losses.ssim  # the function eval imports

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
    """U-23, App. B.1, D-67: masks binarize at 0.5 (Q-61: 0.5 counts as dynamic); the render is
    clamped (the ground truth is used as given), then both images are set to black outside the
    mask and cropped to the mask's bounding box, here rows 5 … 44 and columns 10 … 49 (40×40).
    Exact."""
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
    [
        torch.zeros(50, 59),
        torch.zeros(50, 60, 3),
        torch.zeros(50, 60, dtype=torch.int64),
        torch.zeros(50, 60, dtype=torch.uint8),  # 8-bit masks are divided by 255 first (§1.4)
        torch.zeros(50, 60, device="meta"),
    ],
)
def test_dynamic_region_rejects_a_mask_of_another_shape_or_type(mask: Tensor) -> None:
    image = torch.zeros(50, 60, 3)
    with pytest.raises(ValueError, match="mask"):
        dynamic_region(image, image, mask)


# The Evaluator: all eight metrics of an image (E16, D-68; Q-60).


@pytest.fixture(scope="module")
def evaluator() -> Evaluator:
    """Built once on the CPU (the LPIPS trunk weights are downloaded)."""
    return Evaluator(device="cpu")


def _torchmetrics_lpips(
    net: Literal["alex", "vgg"], pred: Tensor, gt: Tensor, *, normalize: bool
) -> float:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torchvision's `pretrained` deprecation
        metric = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=normalize)
    with torch.no_grad():
        value = metric(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    return float(value)


@pytest.mark.weights
def test_u23_lpips_against_torchmetrics(evaluator: Evaluator) -> None:
    """U-23, D-68: LPIPS-Alex and LPIPS-VGG with inputs mapped to [−1, 1] match torchmetrics'
    LPIPS with normalize=True, and lpips_vgg_raw (unmapped [0, 1] inputs) matches it with
    normalize=False, within 1e-4. The render is clamped first (D-67)."""
    pred, gt = _noisy_pair((48, 64, 3), seed=11, noise=0.15)
    pred, gt = pred.to(F32), gt.to(F32)
    raw = pred.clone()
    raw[:4] += 0.5  # above 1 in places: the metrics see the clamped render
    pred = raw.clamp(0.0, 1.0)
    entire, dynamic = evaluator(raw, gt)
    assert dynamic is None
    assert abs(entire["lpips_alex"] - _torchmetrics_lpips("alex", pred, gt, normalize=True)) <= 1e-4
    assert abs(entire["lpips_vgg"] - _torchmetrics_lpips("vgg", pred, gt, normalize=True)) <= 1e-4
    expected_raw = _torchmetrics_lpips("vgg", pred, gt, normalize=False)
    assert abs(entire["lpips_vgg_raw"] - expected_raw) <= 1e-4
    assert abs(entire["lpips_vgg_raw"] - entire["lpips_vgg"]) > 1e-3


@pytest.mark.weights
def test_evaluator_gives_every_metric_of_the_image_and_of_its_region(evaluator: Evaluator) -> None:
    """Q-60: the eight metrics, in METRIC_NAMES order, agree with the metric functions; with a
    mask, the dynamic values are the same metrics on dynamic_region's pair (Q-61)."""
    g = _gen(12)
    gt = torch.rand(50, 60, 3, generator=g)
    pred = (gt + 0.1 * torch.randn(50, 60, 3, generator=g)).clamp(0.0, 1.2)
    entire, dynamic = evaluator(pred, gt, mask=_l_mask())
    assert list(entire) == list(METRIC_NAMES)
    assert entire["psnr"] == psnr(pred, gt).item()
    assert entire["ssim2"] == ssim_k(pred, gt, 2).item()
    assert entire["dssim1"] == dssim_k(pred, gt, 1).item()
    region = dynamic_region(pred, gt, _l_mask())
    assert region is not None and dynamic is not None
    assert dynamic == evaluator(*region)[0]
    with pytest.warns(UserWarning, match="dynamic"):
        _, none = evaluator(pred, gt, mask=torch.zeros(50, 60))
    assert none is None


@pytest.mark.weights
def test_evaluator_takes_one_image_pair(evaluator: Evaluator) -> None:
    with pytest.raises(ValueError):
        evaluator(torch.zeros(2, 40, 40, 3), torch.zeros(2, 40, 40, 3))


# The metrics report (§2.1 CLI paragraph, D-64, D-69; Q-63).


def _values(base: float) -> dict[str, float]:
    return {name: base + k for k, name in enumerate(METRIC_NAMES)}


def _image(
    k: int, *, frame: int | None, dynamic: dict[str, float] | None = None, masked: bool = False
) -> ImageMetrics:
    return ImageMetrics(
        camera="cam00",
        timestamp=k / 30,
        frame_index=frame,
        path=f"images/cam00/{k:06d}.png",
        entire=_values(10.0 * k),
        dynamic=dynamic,
        masked=masked,
    )


def test_metrics_report_layout_without_masks() -> None:
    """Q-63: format and version, the split, the metric names, the backbone from the source
    (Q-62), and under `entire` the per-scene means and one row per image; no `dynamic` or
    `fastest_10` block without masks or a range."""
    images = [_image(0, frame=0), _image(1, frame=1), _image(2, frame=None)]
    report = metrics_report(images, split="test", metadata={"source": "n3dv/flame_steak"})
    assert set(report) == {"format", "version", "split", "metrics", "lpips_backbone", "entire"}
    assert report["format"] == "ftgs-metrics" and report["version"] == 1
    assert report["split"] == "test"
    assert report["metrics"] == list(METRIC_NAMES)
    assert report["lpips_backbone"] == "alex"
    assert report["entire"]["scene"] == _values(10.0)  # the mean of 0, 10 and 20, plus k
    rows = report["entire"]["images"]
    assert [r["path"] for r in rows] == [im.path for im in images]
    assert rows[1] == {
        "camera": "cam00",
        "timestamp": 1 / 30,
        "frame_index": 1,
        "path": "images/cam00/000001.png",
        **_values(10.0),
    }
    assert rows[2]["frame_index"] is None
    json.dumps(report, allow_nan=False)


def test_metrics_report_writes_non_finite_values_as_null() -> None:
    """Q-63: a perfect render's PSNR is +inf (Q-60); JSON gets null, in rows and means."""
    perfect = ImageMetrics("c", 0.0, 0, "p.png", {**_values(0.0), "psnr": math.inf}, None, False)
    report = metrics_report([perfect, _image(1, frame=1)], split="test", metadata={})
    assert report["entire"]["images"][0]["psnr"] is None
    assert report["entire"]["scene"]["psnr"] is None
    assert report["entire"]["scene"]["ssim1"] == (1.0 + 11.0) / 2
    assert report["lpips_backbone"] == "vgg"
    json.dumps(report, allow_nan=False)


def test_metrics_report_dynamic_block() -> None:
    """§2.1, D-64, Q-63: with masked images, `dynamic` holds the mask source and approximate
    flag from the manifest metadata, the count of usable regions and of skipped ones, their
    means, and one row per masked image (null values where no region was usable, Q-61)."""
    images = [
        _image(0, frame=0, dynamic=_values(1.0), masked=True),
        _image(1, frame=1, dynamic=None, masked=True),
        _image(2, frame=2, dynamic=_values(3.0), masked=True),
        _image(3, frame=3),  # no mask
    ]
    metadata: dict[str, Any] = {"mask_source": "bgmv2", "masks_approximate": True}
    dynamic = metrics_report(images, split="test", metadata=metadata)["dynamic"]
    assert dynamic["mask_source"] == "bgmv2"
    assert dynamic["masks_approximate"] is True
    assert (dynamic["count"], dynamic["skipped"]) == (2, 1)
    assert dynamic["scene"] == _values(2.0)
    assert [r["path"] for r in dynamic["images"]] == [im.path for im in images[:3]]
    assert all(dynamic["images"][1][name] is None for name in METRIC_NAMES)
    assert dynamic["images"][2]["psnr"] == 3.0
    plain = metrics_report(images[1:2], split="test", metadata={})["dynamic"]
    assert plain["mask_source"] is None and plain["masks_approximate"] is False
    assert all(plain["scene"][name] is None for name in METRIC_NAMES)


def test_metrics_report_fastest_10_block() -> None:
    """D-69, Q-63: with metadata.fastest_10 = [n0, n0 + 10), the means over the test images
    whose frame_index lies in the range, for `entire` and, with masks, `dynamic`."""
    images = [_image(k, frame=k, dynamic=_values(float(k)), masked=True) for k in range(12)] + [
        _image(99, frame=None)
    ]
    report = metrics_report(images, split="test", metadata={"fastest_10": [2, 12]})
    block = report["fastest_10"]
    assert block["range"] == [2, 12]
    assert block["count"] == 10
    assert block["entire"] == _values(10.0 * 6.5)  # frames 2 … 11
    assert block["dynamic"] == _values(6.5)
    plain = metrics_report(images[:3], split="test", metadata={"fastest_10": [5, 15]})
    assert plain["fastest_10"]["count"] == 0
    assert all(plain["fastest_10"]["entire"][name] is None for name in METRIC_NAMES)


@pytest.mark.parametrize(
    "bad",
    [
        {"metadata": {"fastest_10": [3]}},
        {"metadata": {"fastest_10": [5, 3]}},
        {"metadata": {"fastest_10": [0, 50]}},  # §2.2: [n0, n0 + 10)
        {"metadata": {"fastest_10": "2-12"}},
        {"metadata": {"masks_approximate": "false"}},
        {"images": [ImageMetrics("c", 0.0, 0, "p", _values(0.0), _values(0.0), False)]},
        {"images": []},
        {"images": [ImageMetrics("c", 0.0, 0, "p", {"psnr": 1.0}, None, False)]},
    ],
)
def test_metrics_report_rejects_malformed_input(bad: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = {"images": [_image(0, frame=0)], "metadata": {}} | bad
    with pytest.raises(ValueError):
        metrics_report(kwargs["images"], split="test", metadata=kwargs["metadata"])


def test_u23_report_scene_means_are_means_of_per_image_values() -> None:
    """U-23, D-67: the report's scene PSNR is the mean of the per-image values (30 dB for
    images at 20 and 40 dB), not the PSNR of a pooled MSE (22.97 dB)."""
    gt = torch.full((2, 8, 8, 3), 0.5, dtype=F64)
    pred = gt.clone()
    pred[0] += 0.1
    pred[1] += 0.01
    values = psnr(pred, gt)
    images = [
        ImageMetrics(
            "c", float(k), k, f"{k}.png", {**_values(0.0), "psnr": float(values[k])}, None, False
        )
        for k in range(2)
    ]
    scene = metrics_report(images, split="test", metadata={})["entire"]["scene"]
    assert abs(scene["psnr"] - 30.0) <= 1e-6


def test_dynamic_region_clamps_the_render_only() -> None:
    """D-67, Q-60, Q-61: renders are clamped to [0, 1]; the ground truth is used as given."""
    pred = torch.full((40, 40, 3), 1.5)
    gt = torch.full((40, 40, 3), 1.25)
    region = dynamic_region(pred, gt, torch.ones(40, 40, dtype=torch.bool))
    assert region is not None
    assert float(region[0].max()) == 1.0
    assert float(region[1].max()) == 1.25


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (torch.zeros(16, 16, 3), torch.zeros(16, 16, 3, dtype=F64)),
        (torch.zeros(16, 16, 3, device="meta"), torch.zeros(16, 16, 3)),
    ],
)
def test_metrics_need_one_dtype_and_device(pred: Tensor, gt: Tensor) -> None:
    """Q-60: images share one floating dtype and one device (Q-41)."""
    with pytest.raises(ValueError):
        psnr(pred, gt)


@pytest.mark.parametrize("source", [5, ["n3dv"], b"n3dv"])
def test_lpips_backbone_needs_a_string_source(source: object) -> None:
    """Q-62: metadata.source is a string or absent."""
    with pytest.raises(ValueError, match="source"):
        lpips_backbone(source)  # type: ignore[arg-type]


@pytest.mark.weights
def test_evaluator_reports_float32_metrics_for_float64_images(evaluator: Evaluator) -> None:
    """D-52, Q-60: reported metrics are plain float32: the Evaluator takes any floating pair
    and computes every metric on its float32 copy."""
    pred, gt = _noisy_pair((40, 48, 3), seed=13)
    from64, _ = evaluator(pred, gt)
    from32, _ = evaluator(pred.to(F32), gt.to(F32))
    assert from64 == from32


@pytest.mark.weights
def test_evaluator_runs_lpips_without_tf32_and_restores_the_flag(
    evaluator: Evaluator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-52, §2.4 Precision: cuDNN's TF32 flag is cleared for every LPIPS pass of the Evaluator
    (LPIPS-Alex, LPIPS-VGG and lpips_vgg_raw), the convolutions the flag would reach, and
    restored afterwards (Q-60)."""
    from openftgs.losses import LPIPSLoss

    seen: list[tuple[bool, bool]] = []
    original = LPIPSLoss.forward

    def spy(self: LPIPSLoss, pred: Tensor, gt: Tensor, *, map_inputs: bool = True) -> Tensor:
        seen.append((torch.backends.cudnn.allow_tf32, map_inputs))
        return original(self, pred, gt, map_inputs=map_inputs)

    monkeypatch.setattr(LPIPSLoss, "forward", spy)
    pred, gt = _noisy_pair((40, 48, 3), seed=14)
    before = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        evaluator(pred, gt)
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = before
    assert sorted(seen) == [(False, False), (False, True), (False, True)]
