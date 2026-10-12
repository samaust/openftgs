# SPDX-License-Identifier: Apache-2.0
"""render() with the reference backend (E2, E5, E6, D-4, D-6, D-28, D-31, D-32, D-34; §1.4; Q-31 to
Q-33): the rest of U-6, U-4, U-8, the projection cases of U-20, the render part of U-22, and U-7's
single-backend cases.

Expected values are worked by hand from the spec in scenes whose values are exact in float32, or
come from an independent float64 evaluation of E1 to E6 written in this file. Pixels are written
as (row v, column u).
"""

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from builders import UNIT_TIME, pinhole

from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera, RenderOutput, TimeNormalizer

F32 = torch.float32
F64 = torch.float64
C0 = 0.28209479177387814  # §1.1
C1 = 0.4886025119029199  # √(3/(4π)), the l = 1 real SH constant
ALPHA_MAX_F32 = torch.tensor(0.999, dtype=F32)  # E6, D-31

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
    cols: dict[str, list[Any]] = {name: [] for name in _PARAM_NAMES}
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
        for name in _PARAM_NAMES
    }
    return FreeTimeGaussians(params, sh_degree=sh_degree, time=time)


_PARAM_NAMES = (
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


def _render(
    model: FreeTimeGaussians, cameras: Camera | Sequence[Camera], t: float, **kw: Any
) -> RenderOutput:
    """The reference backend at normalized time t."""
    kw.setdefault("time_units", "normalized")
    return render(model, cameras, t, backend="reference", **kw)


# Empty renders and the background (Q-32, D-34)


@pytest.mark.parametrize("precull", ["internal", "compact"])
def test_empty_model_renders_the_background(precull: str) -> None:
    """Q-32: a model with no primitives renders the background with alpha 0, in the §2.1 shapes
    [C, H, W, 3] and [C, H, W, 1]."""
    background = torch.tensor([0.25, 0.5, 0.75])
    out = _render(_model([]), CAM_A, 0.5, background=background, precull=precull)
    assert out.image.shape == (1, 48, 64, 3)
    assert out.alpha.shape == (1, 48, 64, 1)
    assert torch.equal(out.image, background.expand(1, 48, 64, 3))
    assert torch.equal(out.alpha, torch.zeros(1, 48, 64, 1))


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
    assert torch.equal(out.image, background.expand(1, 48, 64, 3))
    assert torch.equal(out.alpha, torch.zeros(1, 48, 64, 1))
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
