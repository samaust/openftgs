# SPDX-License-Identifier: Apache-2.0
"""E16's metrics in openftgs.eval: U-23 and the added tests of PRD #74 (D-29, D-40, D-52, D-64,
D-67, D-69; Q-60, Q-61, Q-63) and of #387 (D-89; Q-256, Q-257, Q-259), and the evaluation of a
model on a scene's images (Q-68).

Expected values are worked by hand (PSNR at a known MSE, SSIM_k of constant images in closed
form, crops built pixel by pixel) or come from NVIDIA's ꟻLIP file as published, kept beside the
tests (Q-252), the oracle U-23 names (D-89).
"""

import importlib.util
import json
import math
import warnings
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from builders import random_model, ring, write_scene
from torch import Tensor

import openftgs.eval
import openftgs.losses
from openftgs.data import MultiViewVideo
from openftgs.eval import (
    METRIC_NAMES,
    Evaluator,
    ImageMetrics,
    average_metrics,
    dssim_k,
    dynamic_region,
    evaluate_images,
    metrics_report,
    psnr,
    ssim_k,
    write_metrics,
)
from openftgs.render import render_request
from openftgs.types import RenderItem, RenderRequest, TimeNormalizer

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
    """Q-60: `__all__` is exactly the public API, without Q-62's retired `lpips_backbone`, and
    the metric names are those §2.1's CLI paragraph gives for METRICS.json since 549o: psnr,
    ssim1, ssim2, dssim1, dssim2 and flip (D-89)."""
    assert sorted(openftgs.eval.__all__) == [
        "Evaluator",
        "ImageMetrics",
        "METRIC_NAMES",
        "average_metrics",
        "dssim_k",
        "dynamic_region",
        "evaluate_images",
        "metrics_report",
        "psnr",
        "ssim_k",
        "write_metrics",
    ]
    assert METRIC_NAMES == ("psnr", "ssim1", "ssim2", "dssim1", "dssim2", "flip")
    assert not hasattr(openftgs.eval, "lpips_backbone")


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
    [((0, 0), (0, 0), False), ((3, 23), (2, 40), False), ((3, 24), (2, 22), False),
     ((3, 24), (2, 23), True)],
)  # fmt: skip
def test_dynamic_region_needs_a_box_every_metric_accepts(
    rows: tuple[int, int], cols: tuple[int, int], usable: bool
) -> None:
    """Q-61, Q-257: an empty mask, or a box narrower than 21 px on a side, the larger of SSIM's
    11-px window and the 21 px of ꟻLIP's filters at 67.0 pixels per degree, gives no region and
    a warning; a 21×21 box is usable."""
    g = _gen(6)
    pred, gt = torch.rand(50, 60, 3, generator=g), torch.rand(50, 60, 3, generator=g)
    mask = torch.zeros(50, 60, dtype=torch.bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    if usable:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            region = dynamic_region(pred, gt, mask)
        assert region is not None and region[0].shape == (21, 21, 3)
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


# The Evaluator: the six metrics of an image, ꟻLIP among them (E16, D-89; Q-60).


@pytest.fixture(scope="module")
def evaluator() -> Evaluator:
    """Built once on the CPU; ꟻLIP needs no weights (D-89)."""
    return Evaluator(device="cpu")


def test_u23_flip_of_identical_images_is_zero(evaluator: Evaluator) -> None:
    """U-23: ꟻLIP of identical images is 0, within 1e-6 (E16, D-89), for the entire image and
    for its dynamic region."""
    image = torch.rand(50, 60, 3, generator=_gen(10))
    entire, dynamic = evaluator(image, image.clone(), mask=_l_mask())
    assert dynamic is not None
    assert abs(entire["flip"]) <= 1e-6
    assert abs(dynamic["flip"]) <= 1e-6


# Q-252: NVIDIA's flip_loss.py at d3ab02a as published, kept beside the tests.
_PUBLISHED_FLIP = Path(__file__).resolve().parent / "reference" / "flip" / "flip_loss.py"


def _published_flip() -> Any:
    """NVIDIA's LDR-ꟻLIP loss as published, loaded from the reference copy under a name of its
    own (Q-252); the file runs on CUDA only."""
    spec = importlib.util.spec_from_file_location("flip_published_for_eval", _PUBLISHED_FLIP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LDRFLIPLoss()


@pytest.mark.gpu
def test_u23_flip_against_the_published_file_on_cuda() -> None:
    """U-23 (E16, D-89; Q-252, Q-259): on a fixed image pair on CUDA, the Evaluator's ꟻLIP is
    NVIDIA's LDR-ꟻLIP as published, run on the clamped render against the reference at 67.0
    pixels per degree in IEEE float32 (D-52), within 1e-6. The render leaves [0, 1] in places,
    which E16 clamps and the loss clamps again (D-89). At the file's default 67.02 pixels per
    degree the values differ by about 3·10⁻⁵, so the test tells the two apart (Q-259)."""
    pred, gt = _noisy_pair((96, 128, 3), seed=15, noise=0.12)
    raw = pred.to(F32)
    raw[:6] += 0.6
    raw[-5:, :9] -= 0.4
    gt = gt.to(F32)
    entire, _ = Evaluator(device="cuda")(raw.cuda(), gt.cuda())

    def nchw(image: Tensor) -> Tensor:
        return image.permute(2, 0, 1)[None].contiguous().cuda()

    saved = (torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            expected = float(_published_flip()(nchw(raw.clamp(0.0, 1.0)), nchw(gt), 67.0))
    finally:
        torch.backends.cudnn.allow_tf32 = saved[0]
        torch.set_float32_matmul_precision(saved[1])
    assert 0.0 < expected < 1.0
    assert abs(entire["flip"] - expected) <= 1e-6


def test_evaluator_gives_every_metric_of_the_image_and_of_its_region(evaluator: Evaluator) -> None:
    """Q-60: the six metrics, in METRIC_NAMES order, agree with the metric functions; with a
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
    """§2.1, Q-63, Q-256: format and version 2, the split, the metric names psnr, ssim1, ssim2,
    dssim1, dssim2 and flip, and under `entire` the per-scene means and one row per image with
    exactly those values; no `lpips_backbone`, whatever the manifest's source (Q-62, retired),
    and no `dynamic` or `fastest_10` block without masks or a range."""
    images = [_image(0, frame=0), _image(1, frame=1), _image(2, frame=None)]
    report = metrics_report(images, split="test", metadata={"source": "n3dv/flame_steak"})
    assert set(report) == {"format", "version", "split", "metrics", "entire"}
    assert report["format"] == "ftgs-metrics" and report["version"] == 2
    assert report["split"] == "test"
    assert report["metrics"] == ["psnr", "ssim1", "ssim2", "dssim1", "dssim2", "flip"]
    assert set(report["entire"]["scene"]) == set(report["metrics"])
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


def test_metrics_report_counts_the_images_without_a_mask() -> None:
    """Q-135 (#83): with masks on some images of the split, `dynamic.missing` is the number of
    images without one, so a partial set shows in the report; a full set gives 0."""
    masked = [_image(k, frame=k, dynamic=_values(1.0), masked=True) for k in range(3)]
    partial = metrics_report(
        [*masked, _image(3, frame=3), _image(4, frame=4)], split="test", metadata={}
    )
    assert partial["dynamic"]["missing"] == 2
    assert metrics_report(masked, split="test", metadata={})["dynamic"]["missing"] == 0


def _manifest(tmp_path: Path) -> MultiViewVideo:
    cams = ring(2, width_px=8, height_px=6, focal=8.0)
    gt = random_model(4, seed=0, sh_degree=0, time=TimeNormalizer(0.0, 0.1), spread=0.5)
    images = [(cams[0], 0.0, "train"), (cams[1], 0.1, "train"), (cams[0], 0.1, "test")]
    return MultiViewVideo.from_manifest(write_scene(tmp_path, gt, images))


def test_write_metrics_writes_the_header_and_the_body(tmp_path: Path) -> None:
    """Q-136 (Q-63, Q-74 note 1): one writer of METRICS.json: `producer`, `model` as {path,
    iteration, sha256}, any of which may be null, `manifest` as {path, sha256}, `renders` only
    when given, then the report body; parent folders are created and the JSON has no NaN."""
    video = _manifest(tmp_path / "scene")
    body = metrics_report([_image(0, frame=0)], split="test", metadata={})
    out = tmp_path / "out" / "deep" / "METRICS.json"
    write_metrics(out, body, manifest=video.manifest, iteration=7)
    written = json.loads(out.read_text())
    assert written["producer"].startswith("openftgs")
    assert written["model"] == {"path": None, "iteration": 7, "sha256": None}
    assert written["manifest"] == {
        "path": str(video.manifest.path),
        "sha256": video.manifest.sha256,
    }
    assert "renders" not in written
    assert {k: written[k] for k in body} == json.loads(json.dumps(body))
    renders = {"dir": "r", "request_sha256": "0" * 64, "producer": "other 1.0"}
    write_metrics(
        out,
        body,
        manifest=video.manifest,
        model_path=Path("m.safetensors"),
        renders=renders,
        model_sha256="ab" * 32,
    )
    again = json.loads(out.read_text())
    assert again["model"] == {"path": "m.safetensors", "iteration": None, "sha256": "ab" * 32}
    assert again["renders"] == renders
    with pytest.raises(ValueError, match="header"):
        write_metrics(out, {**body, "model": 1}, manifest=video.manifest)


@pytest.mark.parametrize(
    "digest",
    ["AB" * 32, "a" * 63, "a" * 65, "g" * 64, b"a" * 64, 7],
    ids=["uppercase", "short", "long", "not-hex", "bytes", "int"],
)
def test_write_metrics_refuses_a_malformed_model_sha256(tmp_path: Path, digest: Any) -> None:
    """Q-136 (as answered), Q-56: the model's SHA-256 is 64 lowercase hex digits, as the scene
    manifest's is; anything else is a ValueError and nothing is written."""
    video = _manifest(tmp_path / "scene")
    body = metrics_report([_image(0, frame=0)], split="test", metadata={})
    out = tmp_path / "METRICS.json"
    with pytest.raises(ValueError, match="sha256"):
        write_metrics(out, body, manifest=video.manifest, model_sha256=digest)
    assert not out.exists()


class _Unused:
    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("nothing should be evaluated")


def test_evaluate_images_passes_frame_dt_to_every_render_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-183, Q-174: `frame_dt` reaches each of the split's render requests, one per
    timestamp, as the temporal index's bucket width; None by default."""
    video = _manifest(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    seen: list[Any] = []

    class _Stop(Exception):
        pass

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("frame_dt", "absent"))
        raise _Stop

    monkeypatch.setattr("openftgs.eval.render_request", spy)
    evaluator: Any = _Unused()
    for frame_dt in (None, 0.04):
        with pytest.raises(_Stop):
            evaluate_images(model, video, evaluator, backend="reference", frame_dt=frame_dt)
    with pytest.raises(_Stop):
        evaluate_images(model, video, evaluator, backend="reference")
    assert seen == [None, 0.04, None]


def test_evaluate_images_needs_a_supplied_render_for_every_record(tmp_path: Path) -> None:
    """Q-137: supplied renders must cover every record of the split, which is checked before
    anything is rendered or evaluated."""
    video = _manifest(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    evaluator: Any = _Unused()
    with pytest.raises(ValueError, match="000002"):
        evaluate_images(model, video, evaluator, backend="reference", renders={})


# A render that isn't finite is a numerical failure (§2.1 exit code 3; Q-129, Q-147).


def _broken_renders(
    monkeypatch: pytest.MonkeyPatch, part: Literal["rgb", "alpha"], value: float
) -> None:
    """Patch the evaluation's renderer so that one pixel of every render's `part` is `value`."""
    original = render_request  # the object openftgs.eval imports

    def broken(*args: Any, **kwargs: Any) -> Any:
        renders = original(*args, **kwargs)
        tensors = getattr(renders, part)
        for key, tensor in tensors.items():
            tensors[key] = tensor.clone()  # renders are inference tensors
            tensors[key][0, 0, 0] = value
        return renders

    monkeypatch.setattr("openftgs.eval.render_request", broken)


@pytest.mark.parametrize(
    ("part", "value"),
    [("rgb", math.nan), ("rgb", math.inf), ("rgb", -math.inf), ("alpha", math.nan)],
)
def test_evaluate_images_refuses_a_render_that_is_not_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: Literal["rgb", "alpha"], value: float
) -> None:
    """Q-129, Q-147: a render of the model that isn't finite, in its colors or its alpha, raises
    FloatingPointError naming its image, and isn't scored. E16's clamp would map ±Inf to a
    plausible color, and the metrics never see alpha."""
    video = _manifest(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    _broken_renders(monkeypatch, part, value)
    evaluator: Any = _Unused()
    with pytest.raises(FloatingPointError, match="000002"):
        evaluate_images(model, video, evaluator, backend="reference")


def test_evaluate_images_names_up_to_five_renders_that_are_not_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-129, Q-147: the whole data split is rendered first, so the message gives the number of
    images whose render isn't finite, here 7 at 7 timestamps, and names the first five in the
    manifest's order."""
    cams = ring(2, width_px=8, height_px=6, focal=8.0)
    gt = random_model(4, seed=0, sh_degree=0, time=TimeNormalizer(0.0, 0.6), spread=0.5)
    images = [(cams[0], 0.0, "train"), (cams[0], 0.6, "train")]
    images += [(cams[1], k / 10, "test") for k in range(6, -1, -1)]
    video = MultiViewVideo.from_manifest(write_scene(tmp_path, gt, images))
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    _broken_renders(monkeypatch, "rgb", math.nan)
    evaluator: Any = _Unused()
    with pytest.raises(FloatingPointError) as raised:
        evaluate_images(model, video, evaluator, backend="reference")
    message = str(raised.value)
    root = video.manifest.path.parent
    paths = [r.image_path.relative_to(root).as_posix() for r in video.records[2:]]
    assert "7 image(s)" in message
    assert message.index(paths[0]) < message.index(paths[4])
    assert all(path in message for path in paths[:5])
    assert paths[5] not in message and paths[6] not in message


def test_evaluate_images_refuses_a_supplied_render_that_is_not_finite(tmp_path: Path) -> None:
    """Q-147, Q-137: a supplied render is input, so one that isn't finite is a ValueError, as
    read_renders refuses such a file (Q-142), checked before anything is evaluated."""
    video = _manifest(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    evaluator: Any = _Unused()
    supplied = {2: torch.full((6, 8, 3), math.inf)}
    with pytest.raises(ValueError, match="000002"):
        evaluate_images(model, video, evaluator, backend="reference", renders=supplied)


def test_evaluate_images_scores_supplied_renders_as_its_own(
    tmp_path: Path, evaluator: Evaluator
) -> None:
    """Q-137: given the renders `render_request` makes of each record, the results equal those
    of the rendered path; other renders give other values, so the supplied ones are scored."""
    video = _scene_for_evaluation(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5)
    own = evaluate_images(model, video, evaluator, backend="reference")
    tests = [i for i, record in enumerate(video.records) if record.split == "test"]
    items = tuple(
        RenderItem(str(i), video.records[i].timestamp, video.records[i].camera) for i in tests
    )
    renders = render_request(model, RenderRequest(items, time_unit="s"), backend="reference")
    supplied = {i: renders.rgb[str(i)] for i in tests}
    assert evaluate_images(model, video, evaluator, backend="reference", renders=supplied) == own
    black = {i: torch.zeros_like(supplied[i]) for i in tests}
    other = evaluate_images(model, video, evaluator, backend="reference", renders=black)
    assert [m.entire for m in other] != [m.entire for m in own]


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


def test_evaluator_reports_float32_metrics_for_float64_images(evaluator: Evaluator) -> None:
    """D-52, Q-60: reported metrics are plain float32: the Evaluator takes any floating pair
    and computes every metric on its float32 copy."""
    pred, gt = _noisy_pair((40, 48, 3), seed=13)
    from64, _ = evaluator(pred, gt)
    from32, _ = evaluator(pred.to(F32), gt.to(F32))
    assert from64 == from32


def test_evaluator_runs_flip_in_ieee_float32_and_restores_the_flags(
    evaluator: Evaluator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-52, §2.4 Precision: the Evaluator's ꟻLIP, a convolution and a matrix product, runs in
    IEEE float32, with cuDNN's TF32 flag cleared and float32 matrix products at "highest"
    precision, and both are restored afterwards (Q-60)."""
    from openftgs.losses import PerceptualLoss

    seen: list[tuple[bool, str]] = []
    original = PerceptualLoss.forward

    def spy(self: PerceptualLoss, pred: Tensor, gt: Tensor) -> Tensor:
        seen.append((torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()))
        return original(self, pred, gt)

    monkeypatch.setattr(PerceptualLoss, "forward", spy)
    pred, gt = _noisy_pair((40, 48, 3), seed=14)
    before = (torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        evaluator(pred, gt)
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cudnn.allow_tf32 = before[0]
        torch.set_float32_matmul_precision(before[1])
    assert seen == [(False, "highest")]


# Evaluating a model on a scene's images (Q-68).


def _scene_for_evaluation(root: Path) -> MultiViewVideo:
    """Three 64×48 ring cameras over τ ∈ [0, 0.2]: ring00 trains; ring01 and ring02 are test
    images, two of them sharing τ = 0.1, with masks whose left half is dynamic (32 px wide, so
    every metric is defined on the region, Q-61)."""
    cams = ring(3, width_px=64, height_px=48, focal=60.0)
    gt = random_model(12, seed=0, sh_degree=1, time=TimeNormalizer(0.0, 0.2), spread=0.5)
    images = [(cams[0], tau, "train") for tau in (0.0, 0.1, 0.2)]
    images += [(cams[1], 0.0, "test"), (cams[2], 0.1, "test"), (cams[1], 0.1, "test")]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images, mask_tests=True))


def test_evaluate_images_are_the_evaluator_on_render_request_images(
    tmp_path: Path, evaluator: Evaluator
) -> None:
    """Q-68: each test image's metrics are the Evaluator's on render_request's image of that
    record, with the image's mask; results come in manifest order, with the record's camera,
    raw timestamp, frame index and path relative to the manifest."""
    video = _scene_for_evaluation(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5)
    results = evaluate_images(model, video, evaluator, backend="reference")
    tests = [i for i, record in enumerate(video.records) if record.split == "test"]
    assert [(m.camera, m.timestamp, m.frame_index, m.path) for m in results] == [
        ("ring01", 0.0, 3, "images/ring01/000003.png"),
        ("ring02", 0.1, 4, "images/ring02/000004.png"),
        ("ring01", 0.1, 5, "images/ring01/000005.png"),
    ]
    items = tuple(
        RenderItem(str(i), video.records[i].timestamp, video.records[i].camera) for i in tests
    )
    renders = render_request(model, RenderRequest(items, time_unit="s"), backend="reference")
    for metrics, i in zip(results, tests, strict=True):
        mask = video.load_mask(i)
        entire, dynamic = evaluator(renders.rgb[str(i)], video.load_image(i), mask=mask)
        assert metrics.masked and dynamic is not None
        assert metrics.entire == entire and metrics.dynamic == dynamic


def test_evaluate_images_of_the_training_split(tmp_path: Path, evaluator: Evaluator) -> None:
    """Q-68: any data split; an empty one is a ValueError."""
    video = _scene_for_evaluation(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5)
    results = evaluate_images(model, video, evaluator, split="train", backend="reference")
    assert [m.timestamp for m in results] == [0.0, 0.1, 0.2]
    assert not any(m.masked for m in results)
    with pytest.raises(ValueError, match="split"):
        evaluate_images(model, video, evaluator, split="val", backend="reference")


# D-54's sharding of an evaluation across devices (Q-175, Q-186).


def _sharding_scene(root: Path) -> MultiViewVideo:
    """Three 64×48 ring cameras over τ ∈ [0, 0.3]: ring00 trains; ring01 and ring02 are test
    images at four timestamps listed out of time order, with masks whose left half is dynamic,
    so the split has several timestamps to spread and its records' order isn't theirs."""
    cams = ring(3, width_px=64, height_px=48, focal=60.0)
    gt = random_model(12, seed=0, sh_degree=1, time=TimeNormalizer(0.0, 0.3), spread=0.5)
    taus = (0.0, 0.1, 0.2, 0.3)
    images = [(cams[0], tau, "train") for tau in taus]
    images += [(cams[1 + k % 2], tau, "test") for k, tau in enumerate((0.2, 0.0, 0.3, 0.1))]
    images += [(cams[2 - k % 2], tau, "test") for k, tau in enumerate((0.1, 0.3, 0.0, 0.2))]
    return MultiViewVideo.from_manifest(write_scene(root, gt, images, mask_tests=True))


def test_evaluate_images_sharded_on_the_cpu_equals_unsharded(
    tmp_path: Path, evaluator: Evaluator
) -> None:
    """D-54 ("without changing results"), Q-175: an evaluation spread over three shards on the
    CPU (`devices=["cpu"] * 3`, each with its own model copy and Evaluator) gives exactly the
    metrics of `devices=None`, entire and dynamic, in record order."""
    video = _sharding_scene(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5)
    unsharded = evaluate_images(model, video, evaluator, backend="reference")
    sharded = evaluate_images(model, video, evaluator, backend="reference", devices=["cpu"] * 3)
    assert len(unsharded) == 8 and all(m.dynamic is not None for m in unsharded)
    assert sharded == unsharded


@pytest.mark.gpu
def test_evaluate_images_sharded_on_one_gpu_equals_unsharded(tmp_path: Path) -> None:
    """D-54 ("without changing results"), Q-175 (GPU): three shards on one GPU
    (`devices=["cuda:0"] * 3`, shards simulated as U-33 simulates them), each rendering on
    gsplat with its own model copy and scoring with its own Evaluator, give metrics equal to
    those of `devices=None`, in record order; the caller's TF32 flag is unchanged afterwards."""
    video = _sharding_scene(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5).to("cuda")
    evaluator = Evaluator(device="cuda")
    before = torch.backends.cudnn.allow_tf32
    unsharded = evaluate_images(model, video, evaluator, backend="gsplat")
    sharded = evaluate_images(model, video, evaluator, backend="gsplat", devices=["cuda:0"] * 3)
    assert torch.backends.cudnn.allow_tf32 is before
    assert len(unsharded) == 8 and all(m.dynamic is not None for m in unsharded)
    assert sharded == unsharded


def test_sharded_evaluation_clears_tf32_once_and_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-52, Q-175, Q-186: a sharded evaluation clears cuDNN's process-wide TF32 flag once,
    in the caller's thread, around the whole run: the flag is already False whenever a shard's
    Evaluator is called in its thread, before any metric's own save and restore, and the
    caller's True is back afterwards, also when a render that isn't finite fails the data split
    (Q-147); so is float32 matrix products' "highest" precision (D-52, Q-60). (The Evaluator is
    stood in for, so that only the flags are seen.)"""
    import threading

    video = _sharding_scene(tmp_path)
    model = random_model(12, seed=1, sh_degree=1, time=video.time, spread=0.5)
    seen: list[tuple[str, tuple[bool, str]]] = []

    def build(self: Evaluator, *, device: str | torch.device = "cuda") -> None:
        pass

    def call(self: Evaluator, pred: Tensor, gt: Tensor, *, mask: Tensor | None = None) -> Any:
        flags = torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()
        seen.append((threading.current_thread().name, flags))
        return dict.fromkeys(METRIC_NAMES, 0.0), None

    monkeypatch.setattr(Evaluator, "__init__", build)
    monkeypatch.setattr(Evaluator, "__call__", call)
    kw: dict[str, Any] = {"backend": "reference", "devices": ["cpu"] * 3}
    before = torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        results = evaluate_images(model, video, Evaluator(), **kw)
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
        _broken_renders(monkeypatch, "alpha", math.nan)
        with pytest.raises(FloatingPointError):
            evaluate_images(model, video, Evaluator(), **kw)
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cudnn.allow_tf32 = before[0]
        torch.set_float32_matmul_precision(before[1])
    assert len(results) == len(seen) == 8
    assert {flags for _, flags in seen} == {(False, "highest")}
    assert {name for name, _ in seen} <= {f"evaluate_images shard {k}" for k in range(3)}


def test_sharded_evaluation_names_the_renders_that_are_not_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-147, Q-186: spread over shards, a data split whose renders aren't finite raises the
    FloatingPointError of an unsharded evaluation, naming the same first five images in the
    manifest's order, whichever shard rendered them. (Nothing is scored, so the shards'
    Evaluators are stood in for.)"""
    video = _sharding_scene(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    monkeypatch.setattr(Evaluator, "__init__", lambda self, **kwargs: None)
    _broken_renders(monkeypatch, "rgb", math.nan)
    evaluator: Any = _Unused()
    messages = []
    for devices in (None, ["cpu"] * 3):
        with pytest.raises(FloatingPointError) as raised:
            evaluate_images(model, video, evaluator, backend="reference", devices=devices)
        messages.append(str(raised.value))
    assert "8 image(s)" in messages[0]
    assert messages[1] == messages[0]


@pytest.mark.parametrize(
    "devices", ["cpu", [], ["not a device"]], ids=["string", "empty", "unknown"]
)
def test_evaluate_images_refuses_malformed_devices(tmp_path: Path, devices: Any) -> None:
    """Q-186, as Q-122 for roma_init: `devices` is a non-empty sequence of device names;
    anything else is a ValueError, before anything is rendered or scored."""
    video = _manifest(tmp_path)
    model = random_model(4, seed=1, sh_degree=0, time=video.time, spread=0.5)
    evaluator: Any = _Unused()
    with pytest.raises(ValueError, match="device"):
        evaluate_images(model, video, evaluator, backend="reference", devices=devices)
