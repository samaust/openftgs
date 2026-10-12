# SPDX-License-Identifier: Apache-2.0
"""render() with the reference backend (E2, E5, E6, D-4, D-6, D-28, D-31, D-32, D-34; §1.4; Q-31 to
Q-33): the rest of U-6, U-4, U-8, the projection cases of U-20, the render part of U-22, and U-7's
single-backend cases.

Expected values are worked by hand from the spec in scenes whose values are exact in float32, or
come from an independent float64 evaluation of E1 to E6 written in this file. Pixels are written
as (row v, column u).
"""

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

import numpy as np
import pytest
import torch
from builders import UNIT_TIME, look_at, pinhole, random_model, random_params, ring
from real_sh import real_sh
from torch import Tensor

from openftgs.io import read_exchange, write_exchange
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera, ExchangeModel, RenderOutput, TimeNormalizer

F32 = torch.float32
F64 = torch.float64
C0 = 0.28209479177387814  # §1.1
C1 = 0.4886025119029199  # √(3/(4π)), the l = 1 real SH constant
ALPHA_MAX_F32 = torch.tensor(0.999, dtype=F32)  # E6, D-31

PARAM_NAMES = (  # §1.1, written out: the tests' own expectation
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
)

# Camera A: the point (X, Y, 4) lands at (32.5 + 4X, 24.5 + 4Y).
CAM_A = pinhole(64, 48, fx=16.0, fy=16.0, cx=32.5, cy=24.5)


def _model(
    rows: Sequence[dict[str, Any]],
    *,
    sh_degree: int = 0,
    time: TimeNormalizer = UNIT_TIME,
) -> FreeTimeGaussians:
    """A model with one primitive per row. Defaults: time center 0.5, ŝ = 20 (σt = 1 within
    |t − μt| ≤ 1), no velocity, isotropic scale 0.25, identity quaternion, logit 0 (σ = 0.5),
    sh0 = 0 (color 0.5) and no higher bands."""
    k = (sh_degree + 1) ** 2 - 1
    cols: dict[str, list[Any]] = {name: [] for name in PARAM_NAMES}
    for row in rows:
        scale = row.get("scale", 0.25)
        scales = [scale] * 3 if isinstance(scale, int | float) else list(scale)
        cols["means"].append(list(row["mean"]))
        cols["times"].append(row.get("time", 0.5))
        cols["log_durations"].append(row.get("log_duration", 20.0))
        cols["velocities"].append(list(row.get("velocity", (0.0, 0.0, 0.0))))
        cols["log_scales"].append([math.log(a) for a in scales])
        cols["quats"].append(list(row.get("quat", (1.0, 0.0, 0.0, 0.0))))
        cols["opacity_logits"].append(row.get("logit", 0.0))
        cols["sh0"].append([list(row.get("sh0", (0.0, 0.0, 0.0)))])
        cols["shN"].append(row.get("shN", [[0.0, 0.0, 0.0]] * k))
    shapes = {
        "means": (3,),
        "times": (),
        "log_durations": (),
        "velocities": (3,),
        "log_scales": (3,),
        "quats": (4,),
        "opacity_logits": (),
        "sh0": (1, 3),
        "shN": (k, 3),
    }
    params = {
        name: torch.tensor(cols[name], dtype=F32).reshape(len(rows), *shapes[name])
        for name in PARAM_NAMES
    }
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


def _render(
    model: FreeTimeGaussians, cameras: Camera | Sequence[Camera], t: float, **kw: Any
) -> RenderOutput:
    """The reference backend at normalized time t."""
    kw.setdefault("time_units", "normalized")
    return render(model, cameras, t, backend="reference", **kw)


def _assert_background(out: RenderOutput, background: Tensor) -> None:
    """Every pixel shows the background and alpha is 0 (D-34)."""
    assert torch.equal(out.image, background.expand(out.image.shape))
    assert torch.equal(out.alpha, torch.zeros_like(out.alpha))


# Empty renders and the background (Q-32, D-34)


@pytest.mark.parametrize("precull", ["internal", "compact"])
def test_empty_model_renders_the_background(precull: str) -> None:
    """Q-32: a model with no primitives renders the background with alpha 0, in the §2.1 shapes
    [C, H, W, 3] and [C, H, W, 1]."""
    background = torch.tensor([0.25, 0.5, 0.75])
    out = _render(_model([]), CAM_A, 0.5, background=background, precull=precull)
    assert out.image.shape == (1, 48, 64, 3)
    assert out.alpha.shape == (1, 48, 64, 1)
    _assert_background(out, background)


# U-7's single-backend cases (E6, D-31)


def test_u7_opaque_primitive_on_a_pixel_center_has_alpha_0999() -> None:
    """U-7: a primitive with σ_eff = 1 centered on a pixel center has alpha α_max = 0.999 there
    (logit 20 rounds to σ = 1 in float32; σt = 1)."""
    model = _model([{"mean": (0.0, 0.0, 4.0), "logit": 20.0}])
    out = _render(model, CAM_A, 0.5)
    assert out.alpha[0, 24, 32, 0] == ALPHA_MAX_F32
    assert float(out.alpha.detach().max()) == float(ALPHA_MAX_F32)


def test_u7_tiny_gaussian_peaks_at_its_pixel() -> None:
    """U-7, U-20: a tiny Gaussian at (1.25, −0.75, 4) projects to (37.5, 21.5) = (u + ½, v + ½)
    exactly (§1.4), and its alpha peaks at pixel (21, 37)."""
    model = _model([{"mean": (1.25, -0.75, 4.0), "scale": 1e-3, "logit": 20.0}])
    out = _render(model, CAM_A, 0.5, return_aux=True)
    assert out.means2d is not None and out.visible is not None
    assert torch.equal(out.means2d, torch.tensor([[[37.5, 21.5]]]))
    assert out.visible.tolist() == [[True]]
    peak = int(torch.argmax(out.alpha.detach()[0, :, :, 0]))
    assert divmod(peak, 64) == (21, 37)


def _sh0(rgb: Sequence[float]) -> tuple[float, float, float]:
    """sh0 giving the color rgb at degree 0 (E2: C₀·sh0 + 0.5)."""
    r, g, b = ((c - 0.5) / C0 for c in rgb)
    return (r, g, b)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid64(logit: float) -> float:
    """σ of a logit as stored (float32), evaluated in float64."""
    return float(torch.sigmoid(torch.tensor(logit, dtype=F32).to(F64)))


def test_u7_second_of_two_stacked_opaque_layers_is_excluded() -> None:
    """U-7, E6: two stacked primitives with α = 0.999 give 0.999·c₁ + 0.001·c_bg; the far layer
    would bring T to about 1e-6 ≤ 1e-4, so it is excluded (it would add about 1e-3). The far row
    comes first, so the order is by depth, not by row."""
    c1, c2, background = (1.0, 0.5, 0.25), (0.0, 1.0, 0.0), (0.25, 0.5, 0.75)
    model = _model(
        [
            {"mean": (0.0, 0.0, 6.0), "logit": 20.0, "sh0": _sh0(c2)},
            {"mean": (0.0, 0.0, 4.0), "logit": 20.0, "sh0": _sh0(c1)},
        ]
    )
    out = _render(model, CAM_A, 0.5, background=torch.tensor(background))
    expected = torch.tensor([0.999 * a + 0.001 * b for a, b in zip(c1, background, strict=True)])
    torch.testing.assert_close(out.image[0, 24, 32].detach(), expected, rtol=0.0, atol=1e-6)
    assert out.alpha[0, 24, 32, 0] == ALPHA_MAX_F32


def test_u7_primitives_outside_the_depth_range_are_culled() -> None:
    """U-7, E6: primitives at z_c = 0.005 (the U-7 case), −4 and 2·10¹⁰ are culled, so the image
    is the background and nothing is visible; z_c = 0.01 itself is kept."""
    background = torch.tensor([0.25, 0.5, 0.75])
    rows = [{"mean": (0.0, 0.0, z), "logit": 20.0} for z in (0.005, -4.0, 2e10)]
    out = _render(_model(rows), CAM_A, 0.5, background=background, return_aux=True)
    _assert_background(out, background)
    assert out.visible is not None and out.visible.tolist() == [[False, False, False]]
    near = _render(_model([{"mean": (0.0, 0.0, 0.01)}]), CAM_A, 0.5, return_aux=True)
    assert near.visible is not None and near.visible.tolist() == [[True]]


def test_e6_pixel_stops_before_the_primitive_reaching_the_transmittance_threshold() -> None:
    """E6: four layers with σ = 0.9, 0.9, 0.95, 0.9 in depth order take T to 0.1, 0.01, 5e-4 and
    then 5e-5 ≤ 1e-4, so the fourth is excluded and the background shows through 5e-4 of T."""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    sigmas = [0.9, 0.9, 0.95, 0.9]  # in depth order
    depths = [2.0, 3.0, 4.0, 5.0]
    order = [2, 0, 3, 1]  # row r holds depth layer order[r]
    model = _model(
        [
            {
                "mean": (0.0, 0.0, depths[k]),
                "logit": _logit(sigmas[k]),
                "sh0": _sh0(colors[k]),
            }
            for k in order
        ]
    )
    background = (0.25, 0.5, 0.75)
    out = _render(model, CAM_A, 0.5, background=torch.tensor(background))
    alphas = [_sigmoid64(_logit(s)) for s in sigmas]
    expected = [0.0, 0.0, 0.0]
    t = 1.0
    for k in range(3):
        expected = [e + alphas[k] * t * c for e, c in zip(expected, colors[k], strict=True)]
        t *= 1.0 - alphas[k]
    expected = [e + t * b for e, b in zip(expected, background, strict=True)]
    torch.testing.assert_close(
        out.image[0, 24, 32].detach().to(F64), torch.tensor(expected, dtype=F64), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        out.alpha[0, 24, 32, 0].detach().to(F64),
        torch.tensor(1.0 - t, dtype=F64),
        rtol=0,
        atol=1e-6,
    )


def test_e6_terms_below_the_alpha_threshold_are_skipped() -> None:
    """E6: with Σ′ = 1.3·I (a = 0.25 on the axis) and σ = 0.5, pixel (24, 36), 4 px from the
    mean, has α = 0.5·e^(−8/1.3) ≈ 0.00106 < 1/255 and stays exactly black; pixel (24, 35) has
    α = 0.5·e^(−4.5/1.3) and shows the color 0.5."""
    out = _render(_model([{"mean": (0.0, 0.0, 4.0)}]), CAM_A, 0.5)
    image = out.image.detach()
    assert torch.equal(image[0, 24, 36], torch.zeros(3))
    alpha = 0.5 * math.exp(-4.5 / 1.3)
    torch.testing.assert_close(image[0, 24, 35], torch.full((3,), 0.5 * alpha), rtol=0, atol=1e-6)


def test_e6_jacobian_uses_the_clamped_direction_and_the_mean_does_not() -> None:
    """E6: camera B (c_x = 20) clamps x_c/z_c to [−(20/16 + 0.6), 44/16 + 0.6] = [−1.85, 3.35] for
    J only. A unit-scale row at x_c/z_c = −2.5 projects unclamped to μ′ = (−20, 24.5), but its
    Σ′₀₀ = 4² + (4·1.85)² + 0.3 = 71.06 (116.3 unclamped); a row at +3.5 checks the upper bound
    (Σ′₀₀ = 4² + (4·3.35)² + 0.3 = 195.86, 212.3 unclamped)."""
    cam_b = pinhole(64, 48, fx=16.0, fy=16.0, cx=20.0, cy=24.5)
    for x, mean2d, pixel, cov00 in [
        (-10.0, (-20.0, 24.5), (24, 0), 71.06),
        (14.0, (76.0, 24.5), (24, 63), 195.86),
    ]:
        model = _model([{"mean": (x, 0.0, 4.0), "scale": 1.0}])
        out = _render(model, cam_b, 0.5, return_aux=True)
        assert out.means2d is not None and torch.equal(out.means2d[0, 0], torch.tensor(mean2d))
        dx = pixel[1] + 0.5 - mean2d[0]
        alpha = 0.5 * math.exp(-0.5 * dx * dx / cov00)
        torch.testing.assert_close(
            out.image[0, pixel[0], pixel[1]].detach(),
            torch.full((3,), 0.5 * alpha),
            rtol=0,
            atol=1e-6,
        )


def test_e6_screen_radii_follow_the_opacity_aware_extent() -> None:
    """E6: a row at x_c/z_c = 2.5 (μ′x = 72.5, off the 64-px image) with a = 0.25 has
    Σ′₀₀ = (4² + 10²)·0.0625 + 0.3 = 7.55. With σ = 1 the extent is √(2 ln 255) = 3.329 and
    r_x = ⌈9.15⌉ = 10 reaches the image; with σ = 0.05 it is √(2 ln 12.75) = 2.256 and
    r_x = ⌈6.20⌉ = 7 does not (a fixed 3σ extent would give 9 and reach it)."""
    for logit, visible in [(20.0, True), (_logit(0.05), False)]:
        model = _model([{"mean": (10.0, 0.0, 4.0), "logit": logit}])
        out = _render(model, CAM_A, 0.5, return_aux=True)
        assert out.visible is not None and out.visible.tolist() == [[visible]]


def test_e6_box_touching_the_image_edge_counts_as_off_it() -> None:
    """E6, Q-33: with a = 10⁻³ and σ = 1 the radii are ⌈3.329·√0.3001⌉ = 2. A box μ′ ± 2 that
    only touches the image (μ′x = 66 or −2) is culled; one that overlaps it by 1/16 px
    (65.9375, −1.9375) is visible."""
    rows = [
        {"mean": (x, 0.0, 4.0), "scale": 1e-3, "logit": 20.0}
        for x in (8.375, 8.359375, -8.625, -8.609375)
    ]
    out = _render(_model(rows), CAM_A, 0.5, return_aux=True)
    assert out.means2d is not None and out.visible is not None
    assert out.means2d[0, [1, 3], 0].tolist() == [65.9375, -1.9375]
    assert out.visible.tolist() == [[False, True, False, True]]


def test_e6_depth_ties_composite_the_lower_row_first() -> None:
    """E6, Q-33: two rows at the same place tie in depth and composite in row order: red row 0
    (σ = 0.5) in front of green row 1 gives 0.5·(1, 0, 0) + 0.5·0.5·(0, 1, 0) = (0.5, 0.25, 0)."""
    rows = [
        {"mean": (0.0, 0.0, 4.0), "sh0": _sh0((1.0, 0.0, 0.0))},
        {"mean": (0.0, 0.0, 4.0), "sh0": _sh0((0.0, 1.0, 0.0))},
    ]
    image = _render(_model(rows), CAM_A, 0.5).image.detach()
    torch.testing.assert_close(image[0, 24, 32], torch.tensor([0.5, 0.25, 0.0]), rtol=0, atol=1e-6)


# U-6 · E2 color, the parts that need a renderer (D-4; the basis parts are in test_sh.py)


def _band1(index: int, values: Sequence[float]) -> list[list[float]]:
    """shN of degree 1 with one l = 1 coefficient set: index 0, 1, 2 is m = −1, 0, +1, whose
    basis functions are C₁·(−y), C₁·z and C₁·(−x) (Q-14)."""
    rows = [[0.0, 0.0, 0.0] for _ in range(3)]
    rows[index] = list(values)
    return rows


def test_u6_direction_from_the_moved_mean_flips_the_l1_response() -> None:
    """U-6: a primitive moving past the camera, from (−3, 0, 4) at t = 0 to (3, 0, 4) at t = 1
    (μx = (0, 0, 4) at μt = 0.5), with only the m = +1 coefficient 1/(2.4·C₁) in channel 0. The
    direction from the moved mean, (∓0.6, 0, 0.8), gives the l = 1 response ±0.25, so the centered
    pixels read 0.5·0.75 = 0.375 at t = 0 and 0.5·0.25 = 0.125 at t = 1; the stored mean would give
    the response 0 and 0.25 both times."""
    s = 1.0 / (2.4 * C1)
    row = {"mean": (0.0, 0.0, 4.0), "velocity": (6.0, 0.0, 0.0), "shN": _band1(2, (s, 0.0, 0.0))}
    model = _model([row], sh_degree=1)
    for t, pixel, red in [(0.0, (24, 20), 0.375), (1.0, (24, 44), 0.125)]:
        image = _render(model, CAM_A, t).image.detach()
        torch.testing.assert_close(
            image[0, pixel[0], pixel[1]], torch.tensor([red, 0.25, 0.25]), rtol=0, atol=1e-6
        )


def test_u6_direction_starts_at_the_camera_center() -> None:
    """U-6, E2: world_to_camera = [[0, 0, −1, 0], [0, 1, 0, 0], [1, 0, 0, 4]] puts the camera
    center o = −R_wᵀ t_w at (−4, 0, 0), so a row at the origin is seen along d = (1, 0, 0) and lands
    on pixel (24, 32). The m = +1 coefficient 1/(4·C₁) in channel 0 gives −0.25 (0.5·0.25 = 0.125
    on screen) and the m = 0 coefficient 1/(4·C₁) in channel 1 gives 0 (0.25 on screen); the
    camera-frame direction (0, 0, 1) would give 0.25 and 0.375 instead."""
    w2c = [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 4.0]]
    camera = pinhole(64, 48, fx=16.0, fy=16.0, cx=32.5, cy=24.5, world_to_camera=w2c)
    k = 1.0 / (4.0 * C1)
    sh_n = [[0.0, 0.0, 0.0], [0.0, k, 0.0], [k, 0.0, 0.0]]
    model = _model([{"mean": (0.0, 0.0, 0.0), "shN": sh_n}], sh_degree=1)
    image = _render(model, camera, 0.5).image.detach()
    torch.testing.assert_close(
        image[0, 24, 32], torch.tensor([0.125, 0.25, 0.25]), rtol=0, atol=1e-6
    )


def test_u6_negative_sums_clamp_to_zero_per_channel() -> None:
    """U-6, E2: sh0 = (−5, 0, 1) gives the color (max(0, 0.5 − 5·C₀), 0.5, 0.5 + C₀) = (0, 0.5,
    0.782): the red channel is exactly 0 on screen and its gradient with respect to the clamped
    coefficient is exactly 0, while the other channels keep theirs."""
    model = _model([{"mean": (0.0, 0.0, 4.0), "sh0": (-5.0, 0.0, 1.0)}])
    pixel = _render(model, CAM_A, 0.5).image[0, 24, 32]
    torch.testing.assert_close(
        pixel.detach(), torch.tensor([0.0, 0.25, 0.5 * (0.5 + C0)]), rtol=0, atol=1e-6
    )
    assert pixel[0].item() == 0.0
    (grad,) = torch.autograd.grad(pixel.sum(), model.sh0)
    assert grad[0, 0, 0].item() == 0.0
    torch.testing.assert_close(grad[0, 0, 1:], torch.full((2,), 0.5 * C0), rtol=1e-6, atol=0)


def test_e2_colors_are_not_clamped_at_one() -> None:
    """E2, E7: colors are clamped only at 0, so a color of 1.6 at α = 0.999 renders 0.999·1.6."""
    model = _model([{"mean": (0.0, 0.0, 4.0), "logit": 20.0, "sh0": _sh0((1.6, 1.6, 1.6))}])
    image = _render(model, CAM_A, 0.5).image.detach()
    torch.testing.assert_close(image[0, 24, 32], torch.full((3,), 0.999 * 1.6), rtol=0, atol=1e-6)


def _band_scene(seed: int, *, sh_degree: int = 3) -> FreeTimeGaussians:
    return random_model(20, seed=seed, sh_degree=sh_degree, spread=0.6)


def test_u6_inactive_bands_contribute_nothing() -> None:
    """U-6, E13: rendered with sh_degree = 1, a degree-3 model equals the degree-1 model holding
    only its first three higher bands, whatever its bands 2 and 3 hold, and those bands get exactly
    zero gradient."""
    cameras = ring(2)
    model = _band_scene(0)
    params = {name: getattr(model, name).detach() for name in PARAM_NAMES}
    low = FreeTimeGaussians({**params, "shN": params["shN"][:, :3]}, sh_degree=1, time=model.time)
    expected = _render(low, cameras, 0.4).image.detach()
    out = _render(model, cameras, 0.4, sh_degree=1)
    assert torch.equal(out.image.detach(), expected)
    (grad,) = torch.autograd.grad(out.image.sum(), model.shN)
    assert torch.equal(grad[:, 3:], torch.zeros_like(grad[:, 3:]))
    assert bool(grad[:, :3].abs().sum() > 0)
    other = FreeTimeGaussians(
        {**params, "shN": torch.cat([params["shN"][:, :3], -params["shN"][:, 3:]], dim=1)},
        sh_degree=3,
        time=model.time,
    )
    assert torch.equal(_render(other, cameras, 0.4, sh_degree=1).image.detach(), expected)


def test_u6_degree_zero_renders_the_dc_color() -> None:
    """U-6: with sh_degree = 0 a centered pixel of a row with higher-band energy reads
    σ·(C₀·c₀₀ + 0.5) = 0.5·(C₀·c₀₀ + 0.5)."""
    sh0 = (0.3, -0.2, 0.9)
    sh_n = [[0.4, -0.3, 0.2]] * 15
    model = _model([{"mean": (0.0, 0.0, 4.0), "sh0": sh0, "shN": sh_n}], sh_degree=3)
    image = _render(model, CAM_A, 0.5, sh_degree=0).image.detach()
    expected = torch.tensor([0.5 * (C0 * c + 0.5) for c in sh0])
    torch.testing.assert_close(image[0, 24, 32], expected, rtol=0, atol=1e-6)


# An independent float64 evaluation of E1 to E6, pixel by pixel


def _oracle(
    model: FreeTimeGaussians, camera: Camera, t: float, background: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image [H, W, 3], alpha [H, W] and visibility [N] of one camera by E1–E6, in float64 NumPy
    with an explicit loop over pixels and depth-sorted primitives, sharing no code with the code
    under test: the SH basis is `real_sh` (U-6 pins `sh_basis` to it). Q-33's readings: ties in
    depth to the lower row, a box touching the image counts as off it, det Σ′ > 0 required."""
    p = {name: getattr(model, name).detach().to(F64).numpy() for name in PARAM_NAMES}
    n = p["means"].shape[0]
    means_t = p["means"] + p["velocities"] * (t - p["times"])[:, None]  # E1
    sigma_t = np.exp(-0.5 * ((t - p["times"]) / np.exp(p["log_durations"])) ** 2)  # E4
    opacity = sigma_t / (1.0 + np.exp(-p["opacity_logits"]))  # E3
    w, x, y, z = (p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)).T
    rot = np.stack(  # E5
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)
    rs = rot * np.exp(p["log_scales"])[:, None, :]
    cov = rs @ rs.transpose(0, 2, 1)
    k_mat = camera.K.to(F64).numpy()
    w2c = camera.world_to_camera.to(F64).numpy()
    r_w, t_w = w2c[:3, :3], w2c[:3, 3]
    fx, fy, cx, cy = k_mat[0, 0], k_mat[1, 1], k_mat[0, 2], k_mat[1, 2]
    width, height = camera.width, camera.height
    center = -r_w.T @ t_w
    sh = np.concatenate([p["sh0"], p["shN"]], axis=1)
    basis = real_sh(means_t - center, model.sh_degree)
    colors = np.maximum(0.0, (sh * basis[:, :, None]).sum(axis=1) + 0.5)  # E2

    mean2d = np.zeros((n, 2))
    conic = np.zeros((n, 2, 2))
    depth = np.zeros(n)
    visible = np.zeros(n, dtype=bool)
    for i in range(n):
        xc = r_w @ means_t[i] + t_w
        depth[i] = xc[2]
        if not 0.01 <= xc[2] <= 1e10 or opacity[i] < 1.0 / 255.0:
            continue
        mean2d[i] = (fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy)
        lim_x = (cx / fx + 0.15 * width / fx, (width - cx) / fx + 0.15 * width / fx)
        lim_y = (cy / fy + 0.15 * height / fy, (height - cy) / fy + 0.15 * height / fy)
        tx = min(max(xc[0] / xc[2], -lim_x[0]), lim_x[1])
        ty = min(max(xc[1] / xc[2], -lim_y[0]), lim_y[1])
        jac = np.array([[fx / xc[2], 0.0, -fx * tx / xc[2]], [0.0, fy / xc[2], -fy * ty / xc[2]]])
        cov2d = jac @ r_w @ cov[i] @ r_w.T @ jac.T + 0.3 * np.eye(2)
        det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] ** 2
        if not det > 0:
            continue
        extent = min(3.33, math.sqrt(2.0 * math.log(255.0 * opacity[i])))
        rx = math.ceil(extent * math.sqrt(cov2d[0, 0]))
        ry = math.ceil(extent * math.sqrt(cov2d[1, 1]))
        mx, my = mean2d[i]
        if mx + rx <= 0 or mx - rx >= width or my + ry <= 0 or my - ry >= height:
            continue
        visible[i] = rx > 0 and ry > 0
        conic[i] = np.linalg.inv(cov2d)
    order = [i for i in np.lexsort((np.arange(n), depth)) if visible[i]]

    image = np.zeros((height, width, 3))
    alpha_out = np.zeros((height, width))
    for v in range(height):
        for u in range(width):
            pix = np.array([u + 0.5, v + 0.5])
            color, trans = np.zeros(3), 1.0
            for i in order:
                d = pix - mean2d[i]
                quad = 0.5 * d @ conic[i] @ d
                if quad < 0:
                    continue
                alpha = min(0.999, opacity[i] * math.exp(-quad))
                if alpha < 1.0 / 255.0:
                    continue
                if trans * (1.0 - alpha) <= 1e-4:
                    break
                color += colors[i] * alpha * trans
                trans *= 1.0 - alpha
            image[v, u] = color + trans * np.asarray(background)
            alpha_out[v, u] = 1.0 - trans
    return image, alpha_out, visible


def _oracle_scene(seed: int) -> FreeTimeGaussians:
    """Up to 32 primitives spread around two ring cameras: some behind or close to a camera,
    some below the opacity cull or away from their time center, anisotropic and rotated."""
    n = 5 + 3 * seed  # 5 to 32
    return random_model(
        n, seed=seed, spread=4.5, log_scale=(-2.5, -0.5), logit=(-6.0, 3.0), speed=0.5
    )


@pytest.mark.parametrize("seed", range(10))
def test_e6_reference_matches_an_independent_evaluation(seed: int) -> None:
    """E1–E6: the reference backend in float64 against the pixel-by-pixel float64 evaluation, to
    1e-9, with the same visible primitives; and in float32, to 1e-5."""
    cameras = ring(2)
    background = (0.125, 0.25, 0.375)  # exact in float32
    model = _oracle_scene(seed)
    out32 = _render(model, cameras, 0.37, background=torch.tensor(background))
    model64 = _oracle_scene(seed).double()
    out64 = _render(model64, cameras, 0.37, background=torch.tensor(background), return_aux=True)
    assert out64.visible is not None
    for c, camera in enumerate(cameras):
        image, alpha, visible = _oracle(model64, camera, 0.37, background)
        assert out64.visible[c].tolist() == visible.tolist()
        torch.testing.assert_close(
            out64.image[c].detach(), torch.from_numpy(image), rtol=0, atol=1e-9
        )
        torch.testing.assert_close(
            out64.alpha[c, ..., 0].detach(), torch.from_numpy(alpha), rtol=0, atol=1e-9
        )
        torch.testing.assert_close(
            out32.image[c].detach().to(F64), torch.from_numpy(image), rtol=0, atol=1e-5
        )
        torch.testing.assert_close(
            out32.alpha[c, ..., 0].detach().to(F64), torch.from_numpy(alpha), rtol=0, atol=1e-5
        )


# U-8 · culling σ_eff < 1/255 before the backend changes nothing (D-32, D-57; Q-31)


def _u8_scene(
    n: int, dtype: torch.dtype, *, kept: int | None = None, tie: bool = False
) -> FreeTimeGaussians:
    """n random rows around `ring(3)`, some behind a camera or off the image, with every third row
    below the opacity cull (logit −9). With `kept`, exactly that many rows pass the cull. With
    `tie`, rows 4 and 5 sit at the same place, opaque enough to be seen by every camera, with
    different colors, so they tie in depth."""
    params = random_params(
        n,
        generator=torch.Generator().manual_seed(n),
        spread=4.5,
        log_scale=(-2.5, -0.5),
        logit=(-3.0, 3.0),
    )
    params["opacity_logits"][::3] = -9.0
    if kept is not None:
        params["opacity_logits"] = torch.tensor([3.0] * kept + [-9.0] * (n - kept))
    if tie:
        for row in (4, 5):
            params["means"][row] = torch.tensor([0.1, 0.2, 0.3])
            params["velocities"][row] = 0.0
            params["times"][row] = 0.37
            params["opacity_logits"][row] = 2.0
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME)
    return model.double() if dtype == F64 else model


_U8_CASES: dict[str, tuple[int, int | None, bool]] = {  # n, rows passing the cull, depth tie
    "no row kept": (1, 0, False),
    "one row kept": (2, 1, False),
    "a third culled": (7, None, False),
    "depth tie": (33, None, True),
    "depth tie, more rows": (64, None, True),
}


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("case", list(_U8_CASES))
def test_u8_compaction_leaves_renders_bitwise_unchanged(case: str, dtype: torch.dtype) -> None:
    """U-8: rendering with compaction (precull = "compact") and with the opacity cull inside the
    backend's projection ("internal") gives bitwise-identical images, alpha, 2D means and
    visibility on the reference backend, with 0 or 1 rows kept, a third of the rows culled, and a
    depth tie between visible rows."""
    n, kept, tie = _U8_CASES[case]
    cameras = ring(3)
    model = _u8_scene(n, dtype, kept=kept, tie=tie)
    passing = int(model.at_time(0.37).passes_cull.sum())
    assert passing == kept if kept is not None else 0 < passing < n
    out = {
        precull: _render(model, cameras, 0.37, return_aux=True, precull=precull)
        for precull in ("internal", "compact")
    }
    for field in ("image", "alpha", "means2d", "visible"):
        a, b = getattr(out["internal"], field), getattr(out["compact"], field)
        assert torch.equal(a, b), field
    if tie:
        visible = out["internal"].visible
        assert visible is not None and bool(visible[:, [4, 5]].all())


@pytest.mark.parametrize("precull", ["internal", "compact"])
@pytest.mark.parametrize("dtype", [F32, F64])
def test_u8_culled_primitives_get_exactly_zero_render_gradient(
    precull: str, dtype: torch.dtype
) -> None:
    """U-8: after backpropagating a random weighting of image and alpha, every primitive visible
    in no camera (below the opacity cull, behind a camera or off the image) has an exactly zero
    gradient in all nine stored tensors, in both placements of the opacity cull."""
    cameras = ring(3)
    model = _u8_scene(64, dtype, tie=True)
    out = _render(model, cameras, 0.37, return_aux=True, precull=precull)
    weights = torch.Generator().manual_seed(1)
    loss = (out.image * torch.rand(out.image.shape, generator=weights, dtype=dtype)).sum() + (
        out.alpha * torch.rand(out.alpha.shape, generator=weights, dtype=dtype)
    ).sum()
    params = [getattr(model, name) for name in PARAM_NAMES]
    grads = torch.autograd.grad(loss, params)
    assert out.visible is not None
    culled = ~out.visible.any(dim=0)
    assert int(culled.sum()) >= 20 and int((~culled).sum()) >= 10
    for name, grad in zip(PARAM_NAMES, grads, strict=True):
        assert torch.equal(grad[culled], torch.zeros_like(grad[culled])), name
        assert bool(torch.isfinite(grad).all()), name


# U-20 · camera conventions, the projection cases (§1.4, D-28)

_U20_W2C = [[0.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 2.0], [1.0, 0.0, 0.0, 3.0]]
_U20_POINT = (1.0, 0.5, -1.0)  # x_c = R_w·p + t_w = (0.5, 3, 4)


def _means2d(model: FreeTimeGaussians, camera: Camera) -> Tensor:
    out = _render(model, camera, 0.5, return_aux=True)
    assert out.means2d is not None and out.visible is not None and bool(out.visible.all())
    return out.means2d[0].detach()


def test_u20_known_point_projects_to_the_expected_pixel() -> None:
    """U-20, §1.4: with world_to_camera = [[0, −1, 0, 1], [0, 0, −1, 2], [1, 0, 0, 3]] the point
    (1, 0.5, −1) is at x_c = (0.5, 3, 4); with f = (32, 24) and c = (20.5, 10.5) it projects to
    (32·0.125 + 20.5, 24·0.75 + 10.5) = (24.5, 28.5), the center of pixel (28, 24), where the
    render peaks."""
    camera = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.5, cy=10.5, world_to_camera=_U20_W2C)
    model = _model([{"mean": _U20_POINT, "scale": 1e-3, "logit": 20.0}])
    out = _render(model, camera, 0.5, return_aux=True)
    assert out.means2d is not None
    assert torch.equal(out.means2d[0, 0].detach(), torch.tensor([24.5, 28.5]))
    assert divmod(int(torch.argmax(out.alpha.detach()[0, :, :, 0])), 48) == (28, 24)


def test_u20_opencv_calibration_converts_with_a_half_pixel_shift() -> None:
    """U-20, §1.4: an OpenCV calibration with c = (20, 10) (pixel centers at integers) puts the
    point at (24, 28), a pixel center in its convention; converted with c + 0.5 it projects to
    (24.5, 28.5), the same pixel's center in the continuous convention."""
    model = _model([{"mean": _U20_POINT, "scale": 1e-3, "logit": 20.0}])
    opencv = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.0, cy=10.0, world_to_camera=_U20_W2C)
    shifted = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.5, cy=10.5, world_to_camera=_U20_W2C)
    assert torch.equal(_means2d(model, opencv)[0], torch.tensor([24.0, 28.0]))
    assert torch.equal(_means2d(model, shifted)[0], torch.tensor([24.5, 28.5]))


def test_u20_resizing_by_one_half_halves_projections() -> None:
    """U-20, §1.4: resizing 48×40 by r = 0.5 multiplies f and c by 0.5, so (24.5, 28.5) becomes
    (12.25, 14.25)."""
    model = _model([{"mean": _U20_POINT, "scale": 1e-3, "logit": 20.0}])
    half = pinhole(24, 20, fx=16.0, fy=12.0, cx=10.25, cy=5.25, world_to_camera=_U20_W2C)
    assert torch.equal(_means2d(model, half)[0], torch.tensor([12.25, 14.25]))


def _backproject(camera: Camera, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """World points that the camera (its float32 values, in float64 NumPy) sees at the given
    continuous pixel coordinates and depths, by inverting §1.4's projection."""
    k = camera.K.to(F64).numpy()
    w2c = camera.world_to_camera.to(F64).numpy()
    x_c = np.stack(
        [
            (pixels[:, 0] - k[0, 2]) / k[0, 0] * depths,
            (pixels[:, 1] - k[1, 2]) / k[1, 1] * depths,
            depths,
        ],
        axis=1,
    )
    # R_w⁻¹ (x_c − t_w): the float32 rotation is orthonormal only to float32 precision.
    world: np.ndarray = np.linalg.solve(w2c[:3, :3], (x_c - w2c[:3, 3]).T).T
    return world


@pytest.mark.parametrize("seed", range(3))
def test_u20_points_project_to_their_pixels_at_full_resolution(seed: int) -> None:
    """U-20, §1.4: on random rigid cameras at 1352×1014, one to three points back-projected in
    NumPy from random continuous pixel coordinates project back to them within 1e-5 px (float64);
    resizing by r = 0.37 (500×375) scales the x projections by W′/W = 500/1352 and the y ones by
    H′/H = 375/1014."""
    rng = np.random.default_rng(seed)
    count = 1 + seed
    eye, target = (rng.normal(size=3) * 3.0).tolist(), (rng.normal(size=3) * 0.5).tolist()
    camera = look_at(eye, target, width=1352, height=1014, focal=1100.0)
    pixels = rng.uniform(size=(count, 2)) * np.array([1352.0, 1014.0])
    depths = 2.0 + 3.0 * rng.uniform(size=count)
    points = _backproject(camera, pixels, depths)
    rows = [{"mean": tuple(p), "scale": 1e-2, "logit": 20.0} for p in points.tolist()]
    model = _model(rows).double()
    model.means.data = torch.from_numpy(points)  # the float64 positions, not float32 roundings
    torch.testing.assert_close(_means2d(model, camera), torch.from_numpy(pixels), rtol=0, atol=1e-5)

    # The resize rule of §1.4, with K kept in float64 so the scaled intrinsics are exact.
    w_ratio, h_ratio = 500 / 1352, 375 / 1014
    scale = torch.tensor([[w_ratio], [h_ratio], [1.0]], dtype=F64)
    resized = Camera(
        K=camera.K.to(F64) * scale,
        world_to_camera=camera.world_to_camera,
        width=500,
        height=375,
    )
    expected = torch.from_numpy(pixels * np.array([w_ratio, h_ratio]))
    torch.testing.assert_close(_means2d(model, resized), expected, rtol=0, atol=1e-5)


# U-4 · renders are invariant under an affine change of the time unit (E14, §1.4 "Time")

_SPAN = TimeNormalizer(2.0, 12.0)


def _moving_scene(seed: int, *, time: TimeNormalizer = _SPAN, **ranges: Any) -> FreeTimeGaussians:
    """Moving primitives near the origin, seen by `ring(2)`."""
    return random_model(16, seed=seed, time=time, spread=0.6, log_scale=(-2.0, -0.7), **ranges)


def _rescaled(exchange: ExchangeModel, a: float, b: float) -> ExchangeModel:
    """The exchange model in the time unit τ′ = aτ + b: time centers aτ + b, durations a·d,
    velocities v_τ/a (float32, as ftgs-if stores them), τ₀ and τ₁ mapped too."""
    arrays = dict(exchange.arrays)
    arrays["time_center"] = a * arrays["time_center"] + b
    arrays["duration"] = a * arrays["duration"]
    arrays["velocity"] = (arrays["velocity"].to(F64) / a).to(F32)
    return ExchangeModel(
        arrays=arrays,
        sh_degree=exchange.sh_degree,
        tau0=a * exchange.tau0 + b,
        tau1=a * exchange.tau1 + b,
        time_unit=exchange.time_unit,
        producer=exchange.producer,
    )


def _raw_render(model: FreeTimeGaussians, tau: float) -> RenderOutput:
    with torch.no_grad():
        return render(model, ring(2), tau, backend="reference")


@pytest.mark.parametrize(("a", "b"), [(2.0, 5.0), (1e-3, -3.0), (1e3, 0.0)])
def test_u4_renders_are_invariant_under_an_affine_time_change(a: float, b: float) -> None:
    """U-4: (τ, μτ, d, v_τ) → (aτ + b, aμτ + b, a·d, v_τ/a) leaves renders unchanged, max abs
    difference ≤ 1e-5 on the reference backend, at τ₀, inside the span and at τ₁; no warning is
    raised inside the span (Q-9)."""
    exchange = _moving_scene(0).to_exchange()
    before = FreeTimeGaussians.from_exchange(exchange, device="cpu")
    after = FreeTimeGaussians.from_exchange(_rescaled(exchange, a, b), device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tau in (2.0, 7.3, 12.0):
            x, y = _raw_render(before, tau), _raw_render(after, a * tau + b)
            assert float(x.alpha.max()) > 0.1
            torch.testing.assert_close(y.image, x.image, rtol=0, atol=1e-5)
            torch.testing.assert_close(y.alpha, x.alpha, rtol=0, atol=1e-5)


@pytest.mark.parametrize(("a", "b"), [(2.0, 5.0), (1e-3, -3.0), (1e3, 0.0)])
def test_u4_single_timestamp_renders_are_invariant_too(a: float, b: float) -> None:
    """U-4 with one timestamp (Q-30): t ≡ 0 and every time center is τ₀, so scaling durations by a
    changes nothing on screen (≤ 1e-5), and τ₀ itself raises no warning."""
    single = TimeNormalizer(5.0, 5.0)
    params = random_params(16, generator=torch.Generator().manual_seed(3), spread=0.6)
    params["times"] = torch.zeros(16)
    params["velocities"] = torch.zeros(16, 3)
    exchange = FreeTimeGaussians(params, sh_degree=3, time=single).to_exchange()
    before = FreeTimeGaussians.from_exchange(exchange, device="cpu")
    after = FreeTimeGaussians.from_exchange(_rescaled(exchange, a, b), device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, y = _raw_render(before, 5.0), _raw_render(after, a * 5.0 + b)
    assert float(x.alpha.max()) > 0.1
    torch.testing.assert_close(y.image, x.image, rtol=0, atol=1e-5)
    torch.testing.assert_close(y.alpha, x.alpha, rtol=0, atol=1e-5)


# U-22 · renders before and after an exchange round trip (§2.2 ftgs-if)


def test_u22_renders_agree_before_and_after_export_and_import(tmp_path: Path) -> None:
    """U-22: export → write → read → import leaves renders within 1e-6 at four times on two
    cameras. The scene keeps σ′ ≥ 2 px, |v·(t − μt)| small and transmittance far above 1e-4, and
    holds one opacity logit of 20 (exported as 1 − 10⁻⁷), away from the others."""
    model = random_model(
        6,
        seed=5,
        time=_SPAN,
        spread=0.5,
        log_scale=(-0.9, -0.5),
        speed=0.25,
        logit=(-2.0, 2.0),
    )
    params = {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}
    speed = torch.linalg.vector_norm(params["velocities"], dim=1, keepdim=True)
    params["velocities"] *= torch.clamp(0.5 / speed, max=1.0)  # |v·(t − μt)| ≤ 0.5
    params["means"][0] = torch.tensor([0.0, 0.0, 2.5])
    params["opacity_logits"][0] = 20.0
    model = FreeTimeGaussians(params, sh_degree=3, time=_SPAN)
    write_exchange(model.to_exchange(), tmp_path / "model.safetensors")
    again = FreeTimeGaussians.from_exchange(
        read_exchange(tmp_path / "model.safetensors"), device="cpu"
    )
    for tau in (2.0, 4.5, 9.1, 12.0):
        x, y = _raw_render(model, tau), _raw_render(again, tau)
        assert 0.5 < float(x.alpha.max()) <= 0.9991  # T stays at 1e-3 or above, far from 1e-4
        for field in ("image", "alpha"):
            torch.testing.assert_close(
                getattr(y, field).detach(), getattr(x, field).detach(), rtol=0, atol=1e-6
            )


# Gradients through the reference backend (E1–E6; U-11 runs the losses through it in M6)


class _RenderModule(torch.nn.Module):
    """Holds a model so that torch.func.functional_call can substitute its nine tensors."""

    def __init__(self, model: FreeTimeGaussians, camera: Camera, t: float) -> None:
        super().__init__()
        self.model = model
        self.camera = camera
        self.t = t

    def forward(self) -> tuple[Tensor, Tensor]:
        out = _render(self.model, self.camera, self.t)
        return out.image, out.alpha


def test_reference_render_gradients_pass_gradcheck() -> None:
    """E1–E6 are differentiable end to end: float64 gradcheck of image and alpha with respect to
    all nine stored tensors, 3 primitives on an 8×8 image, kept away from the clamps and thresholds
    of E6 (σ′ ≈ 3 px so every α is above 1/255, σ ≤ 0.6, distinct depths, means near the axis,
    colors away from 0), at eps 1e-6 and atol 1e-5."""
    camera = pinhole(8, 8, fx=8.0, fy=8.0, cx=4.0, cy=4.0)
    gen = torch.Generator().manual_seed(7)
    params = {
        "means": torch.tensor([[0.2, -0.1, 3.6], [-0.3, 0.2, 4.0], [0.1, 0.3, 4.4]]),
        "times": torch.tensor([0.5, 0.6, 0.4]),
        "log_durations": torch.zeros(3),
        "velocities": 0.3 * torch.randn(3, 3, generator=gen),
        "log_scales": torch.log(1.2 + 0.6 * torch.rand(3, 3, generator=gen)),
        "quats": torch.randn(3, 4, generator=gen),
        "opacity_logits": torch.tensor([-0.8, 0.0, 0.4]),
        "sh0": 0.3 + 0.2 * torch.rand(3, 1, 3, generator=gen),
        "shN": 0.05 * torch.randn(3, 15, 3, generator=gen),
    }
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME).double()
    module = _RenderModule(model, camera, 0.3)
    inputs = tuple(getattr(model, name).detach().clone().requires_grad_() for name in PARAM_NAMES)

    def fn(*tensors: Tensor) -> tuple[Tensor, Tensor]:
        substituted = {f"model.{name}": x for name, x in zip(PARAM_NAMES, tensors, strict=True)}
        image, alpha = torch.func.functional_call(module, substituted, ())
        return image, alpha

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("precull", ["internal", "compact"])
def test_means2d_gradient_is_the_gradient_with_respect_to_the_2d_mean(precull: str) -> None:
    """Q-32: with return_aux, `means2d` keeps its gradient, in both placements of the opacity cull
    (Q-31). For the on-axis row (Σ′ = 1.3·I, σ = 0.5, color 0.5) and L = the red value of pixel
    (24, 34), 2 px right of μ′, L = 0.5·0.5·e^(−q) with q = ½·(34.5 − μ′x)²/1.3, so
    ∂L/∂μ′ = (L·2/1.3, 0); the second row is below the opacity cull and gets (0, 0)."""
    model = _model([{"mean": (0.0, 0.0, 4.0)}, {"mean": (1.0, 0.0, 4.0), "logit": -9.0}]).double()
    model.log_scales.data.fill_(math.log(0.25))  # a = 0.25 exactly, not via float32
    out = _render(model, CAM_A, 0.5, return_aux=True, precull=precull)
    assert out.means2d is not None
    torch.autograd.backward(out.image[0, 24, 34, 0])
    value = 0.25 * math.exp(-0.5 * 4.0 / 1.3)
    assert out.means2d.grad is not None
    expected = torch.tensor([[[value * 2.0 / 1.3, 0.0], [0.0, 0.0]]], dtype=F64)
    torch.testing.assert_close(out.means2d.grad, expected, rtol=1e-12, atol=0)


def test_row_at_the_camera_center_keeps_gradients_finite() -> None:
    """E6: a row exactly at the camera center (z_c = 0, a zero view direction) is culled; every
    gradient stays finite and that row's are exactly 0."""
    model = _model([{"mean": (0.0, 0.0, 0.0)}, {"mean": (0.0, 0.0, 4.0)}], sh_degree=1)
    out = _render(model, CAM_A, 0.5)
    params = [getattr(model, name) for name in PARAM_NAMES]
    grads = torch.autograd.grad(out.image.sum() + out.alpha.sum(), params)
    for name, grad in zip(PARAM_NAMES, grads, strict=True):
        assert bool(torch.isfinite(grad).all()), name
        assert torch.equal(grad[0], torch.zeros_like(grad[0])), name
    assert bool(grads[0][1].abs().sum() > 0)


# The render() API (§2.1, Q-31, Q-32)


def _api_model() -> FreeTimeGaussians:
    return _model([{"mean": (0.0, 0.0, 4.0)}, {"mean": (0.5, 0.25, 5.0), "logit": 1.0}])


def test_all_lists_the_public_api() -> None:
    """§2.1: render is the module's only public name."""
    import openftgs.render

    assert openftgs.render.__all__ == ["render"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "cuda"}, "backend"),
        ({"time_units": "seconds"}, "time_units"),
        ({"precull": "always"}, "precull"),
        ({"background": torch.zeros(4)}, "background"),
        ({"background": torch.zeros(2, 3)}, "background"),
        ({"sh_degree": 1}, "sh_degree"),
    ],
)
def test_render_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    """Q-32: unknown choices, background shapes other than [3] and [C, 3], and an SH degree above
    the model's (Q-20) raise ValueError."""
    with pytest.raises(ValueError, match=match):
        render(_api_model(), CAM_A, 0.5, **{"backend": "reference", **kwargs})


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf, True, "0.5"])
def test_render_rejects_timestamps_that_are_not_finite_reals(timestamp: Any) -> None:
    """Q-32: a timestamp must be a finite real number."""
    with pytest.raises(ValueError, match="timestamp"):
        render(_api_model(), CAM_A, timestamp, backend="reference")


def test_render_rejects_cameras_it_cannot_batch() -> None:
    """Q-32: one Camera or a non-empty sequence of them, of one positive width and height, with K
    [3, 3] and world_to_camera [4, 4]."""
    model = _api_model()
    other_size = pinhole(32, 48, fx=16.0, fy=16.0, cx=16.0, cy=24.0)
    bad_k = Camera(K=torch.eye(4), world_to_camera=torch.eye(4), width=64, height=48)
    bad_pose = Camera(K=CAM_A.K, world_to_camera=torch.eye(3), width=64, height=48)
    no_pixels = Camera(K=CAM_A.K, world_to_camera=CAM_A.world_to_camera, width=0, height=48)
    cases: list[tuple[Any, str]] = [
        ([], "camera"),
        ([CAM_A, other_size], "same width and height"),
        (bad_k, "K"),
        (bad_pose, "world_to_camera"),
        (no_pixels, "positive"),
        ([CAM_A, "camera"], "Camera objects"),
    ]
    for cameras, match in cases:
        with pytest.raises(ValueError, match=match):
            render(model, cameras, 0.5, backend="reference")


def test_gsplat_backend_arrives_in_m5() -> None:
    """Q-32: until M5, backend="gsplat" (the §2.1 default) raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="M5"):
        render(_api_model(), CAM_A, 0.5)


def test_render_warns_outside_the_training_time_range() -> None:
    """§1.4 "Time", Q-9, Q-32: exactly one UserWarning for a raw τ outside [τ₀, τ₁] and for a
    normalized t outside [0, 1] (≠ 0 with a single timestamp); none inside, ends included."""
    model = _model([{"mean": (0.0, 0.0, 4.0)}], time=_SPAN)
    single = _model([{"mean": (0.0, 0.0, 4.0), "time": 0.0}], time=TimeNormalizer(3.0, 3.0))
    warned: list[tuple[FreeTimeGaussians, float, Literal["raw", "normalized"]]] = [
        (model, 1.5, "raw"),
        (model, 12.5, "raw"),
        (model, -0.1, "normalized"),
        (model, 1.25, "normalized"),
        (single, 0.5, "normalized"),
        (single, 4.0, "raw"),
    ]
    for m, timestamp, units in warned:
        with pytest.warns(UserWarning) as record:
            render(m, CAM_A, timestamp, time_units=units, backend="reference")
        assert len(record) == 1, (timestamp, units)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet: list[tuple[FreeTimeGaussians, float, Literal["raw", "normalized"]]] = [
            (model, 2.0, "raw"),
            (model, 12.0, "raw"),
            (model, 0.0, "normalized"),
            (model, 1.0, "normalized"),
            (single, 0.0, "normalized"),
            (single, 3.0, "raw"),
        ]
        for m, timestamp, units in quiet:
            render(m, CAM_A, timestamp, time_units=units, backend="reference")


def test_raw_and_normalized_time_give_the_same_render() -> None:
    """E14: a raw τ renders exactly as its normalized t = (τ − τ₀)/(τ₁ − τ₀)."""
    model = _moving_scene(1)
    raw = render(model, ring(2), 7.3, backend="reference")
    t = model.time.to_t(7.3)
    normalized = render(model, ring(2), t, time_units="normalized", backend="reference")
    assert torch.equal(raw.image, normalized.image)
    assert torch.equal(raw.alpha, normalized.alpha)


def test_render_outputs_follow_return_aux() -> None:
    """§2.1, Q-32: image [C, H, W, 3] and alpha [C, H, W, 1] always; with return_aux, means2d
    [C, N, 2] (0 for primitives that are not visible) and visible [C, N]; gaussian_ids stays None
    on the unpacked reference backend."""
    model = _api_model()
    cameras = [
        CAM_A,
        pinhole(
            64,
            48,
            fx=16.0,
            fy=16.0,
            cx=32.5,
            cy=24.5,
            world_to_camera=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -4.5]],
        ),
    ]
    plain = _render(model, cameras, 0.5)
    assert plain.image.shape == (2, 48, 64, 3) and plain.alpha.shape == (2, 48, 64, 1)
    assert plain.means2d is None and plain.visible is None and plain.gaussian_ids is None
    aux = _render(model, cameras, 0.5, return_aux=True)
    assert aux.means2d is not None and aux.visible is not None and aux.gaussian_ids is None
    assert aux.means2d.shape == (2, 2, 2) and aux.visible.dtype == torch.bool
    assert aux.visible.tolist() == [[True, True], [False, True]]
    assert torch.equal(aux.means2d[1, 0], torch.zeros(2))
    assert torch.equal(aux.image, plain.image)


def test_cameras_render_independently() -> None:
    """Q-32: a single Camera renders like a one-element list, and a batch of cameras like each
    camera alone, with a per-camera background [C, 3]."""
    model = _moving_scene(2)
    cameras = ring(3)
    backgrounds = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.25, 0.125], [1.0, 1.0, 1.0]])
    batch = _render(model, cameras, 0.4, background=backgrounds)
    for c, camera in enumerate(cameras):
        alone = _render(model, camera, 0.4, background=backgrounds[c])
        listed = _render(model, [camera], 0.4, background=backgrounds[c])
        assert torch.equal(alone.image, listed.image)
        assert torch.equal(batch.image[c], alone.image[0])
        assert torch.equal(batch.alpha[c], alone.alpha[0])


def test_float64_models_render_in_float64() -> None:
    """Q-32: the render follows the parameters' dtype (float64 through .double(), for gradient
    checks), whatever the float32 camera tensors."""
    out = _render(_api_model().double(), CAM_A, 0.5, return_aux=True)
    assert out.image.dtype == out.alpha.dtype == F64
    assert out.means2d is not None and out.means2d.dtype == F64


@pytest.mark.parametrize("precull", ["internal", "compact"])
def test_scene_with_every_primitive_culled_renders_the_background(precull: str) -> None:
    """Q-32, E6: when every primitive is culled (below the opacity cull, behind the camera), the
    render is the background with alpha 0, in both placements of the opacity cull."""
    model = _model([{"mean": (0.0, 0.0, 4.0), "logit": -8.0}, {"mean": (0.0, 0.0, -4.0)}])
    background = torch.tensor([0.25, 0.5, 0.75])
    out = _render(model, CAM_A, 0.5, background=background, precull=precull, return_aux=True)
    _assert_background(out, background)
    assert out.visible is not None and not bool(out.visible.any())


def test_render_constants_agree_with_their_copies() -> None:
    """Private: E6's constants appear in render and in the ftgs-if render_conventions (Q-24, and
    D-73's entries under Q-35's keys); the opacity cull that render reads is motion's (D-32); the
    default background is TrainConfig's (D-34); the precull choices are PerfConfig's (Q-31)."""
    import openftgs.io
    import openftgs.motion
    import openftgs.render
    from openftgs.config import PerfConfig, TrainConfig

    conventions = openftgs.io._RENDER_CONVENTIONS
    r = openftgs.render
    assert conventions["alpha_max"] == r._ALPHA_MAX == 0.999
    assert conventions["eps2d"] == r._EPS2D == 0.3
    assert conventions["near"] == r._NEAR == 0.01
    assert conventions["far"] == r._FAR == 1e10
    assert conventions["alpha_skip"] == r._ALPHA_SKIP == 1.0 / 255.0
    assert openftgs.motion._CULL_OPACITY == 1.0 / 255.0  # the opacity cull render reads (D-32)
    assert conventions["transmittance_stop"] == r._T_STOP == 1e-4
    assert conventions["pixel_center_offset"] == r._PIXEL_CENTER == 0.5
    assert conventions["background"] == list(r._BACKGROUND) == list(TrainConfig().background)
    assert r._EXTENT_MAX == 3.33 and r._CLAMP_MARGIN == 0.15 and r._SH_OFFSET == 0.5
    # D-73, Q-35: the conventions files record are the ones render applies (D-72).
    assert conventions["jacobian_clamp_margin"] == r._CLAMP_MARGIN
    assert conventions["screen_extent_max"] == r._EXTENT_MAX
    # The extent rule and the rasterize mode have no constant in render to agree with: the
    # reference backend has one rule and one mode. The io metadata test pins their entries; M5's
    # gsplat backend, which takes the mode as an argument, can tie the mode to it.
    # Q-31: precull takes PerfConfig.precull's values.
    assert get_args(r._Precull) == get_args(get_type_hints(PerfConfig)["precull"])


def test_rasterize_composites_chunks_of_pixel_rows_like_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private: compositing one row of pixels at a time (a tiny _CHUNK_ELEMENTS) gives the image
    of a single chunk, bitwise."""
    import openftgs.render

    model = _moving_scene(4)
    whole = _render(model, ring(2), 0.4)
    monkeypatch.setattr(openftgs.render, "_CHUNK_ELEMENTS", 64)
    chunked = _render(model, ring(2), 0.4)
    assert torch.equal(chunked.image, whole.image)
    assert torch.equal(chunked.alpha, whole.alpha)
