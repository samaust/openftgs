# SPDX-License-Identifier: Apache-2.0
"""render() with the gsplat backend on a CUDA device (E2, E6, D-6, D-31, D-32, D-57; §1.4; Q-31 to
Q-33, Q-36 to Q-39): U-7, U-12, the GPU variants of U-4, U-8 and U-20, the Q-33 boundary cases and
the exact 1/255 boundary, both layouts (packed and unpacked) and a cross-check against gsplat's own
rasterization().

Expected values are M4's worked values (`test_render.py`), or come from the reference backend
rendering the same CUDA model, so both backends see bit-identical per-time arrays. Pixels are
written as (row v, column u).
"""

import math
from typing import Any

import pytest
import torch
from builders import (
    CAM_A,
    logit_of,
    model_from_rows,
    pinhole,
    random_model,
    ring,
    sh0_for,
    sigmoid64,
)
from torch import Tensor

from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera, RenderOutput

pytestmark = pytest.mark.gpu

F32 = torch.float32
ALPHA_MAX_F32 = torch.tensor(0.999, dtype=F32)  # E6, D-31
LAYOUTS = pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])


def _cuda(model: FreeTimeGaussians) -> FreeTimeGaussians:
    return model.to("cuda")


def _gsplat(
    model: FreeTimeGaussians, cameras: Camera | list[Camera], t: float, **kw: Any
) -> RenderOutput:
    """The gsplat backend at normalized time t."""
    kw.setdefault("time_units", "normalized")
    return render(model, cameras, t, backend="gsplat", **kw)


def _unpacked(out: RenderOutput) -> Tensor:
    """The 2D means as [C, N, 2], zero where not visible, from either layout: packed rows are
    camera-major and ascending within a camera (Q-38), which is the order of the True entries of
    `visible` read row by row."""
    assert out.means2d is not None and out.visible is not None
    if out.gaussian_ids is None:
        return out.means2d.detach()
    full = out.means2d.new_zeros((*out.visible.shape, 2))
    full[out.visible] = out.means2d.detach()
    return full


def _background_only(out: RenderOutput, background: Tensor) -> bool:
    return torch.equal(out.image, background.expand(out.image.shape)) and torch.equal(
        out.alpha, torch.zeros_like(out.alpha)
    )


# U-7's closed cases on gsplat (E6, D-31; M4's worked values)


@LAYOUTS
def test_u7_opaque_primitive_on_a_pixel_center_has_alpha_0999(packed: bool) -> None:
    """U-7: a primitive with σ_eff = 1 centered on a pixel center has alpha α_max = 0.999 there,
    and nowhere more (logit 20 rounds to σ = 1 in float32; σt = 1)."""
    model = _cuda(model_from_rows([{"mean": (0.0, 0.0, 4.0), "logit": 20.0}]))
    alpha = _gsplat(model, CAM_A, 0.5, packed=packed).alpha.detach().cpu()
    assert alpha[0, 24, 32, 0] == ALPHA_MAX_F32
    assert float(alpha.max()) == float(ALPHA_MAX_F32)


@LAYOUTS
def test_u7_tiny_gaussian_peaks_at_its_pixel(packed: bool) -> None:
    """U-7, U-20: a tiny Gaussian at (1.25, −0.75, 4) projects to (37.5, 21.5) = (u + ½, v + ½)
    exactly (§1.4), and its alpha peaks at pixel (21, 37)."""
    model = _cuda(model_from_rows([{"mean": (1.25, -0.75, 4.0), "scale": 1e-3, "logit": 20.0}]))
    out = _gsplat(model, CAM_A, 0.5, return_aux=True, packed=packed)
    assert out.visible is not None and out.visible.tolist() == [[True]]
    assert torch.equal(_unpacked(out).cpu(), torch.tensor([[[37.5, 21.5]]]))
    peak = int(torch.argmax(out.alpha.detach()[0, :, :, 0]))
    assert divmod(peak, 64) == (21, 37)


@LAYOUTS
def test_u7_second_of_two_stacked_opaque_layers_is_excluded(packed: bool) -> None:
    """U-7, E6: two stacked primitives with α = 0.999 give 0.999·c₁ + 0.001·c_bg; the far layer
    would bring T to about 1e-6 ≤ 1e-4, so it is excluded. The far row comes first, so the order
    is by depth, not by row."""
    c1, c2, background = (1.0, 0.5, 0.25), (0.0, 1.0, 0.0), (0.25, 0.5, 0.75)
    model = _cuda(
        model_from_rows(
            [
                {"mean": (0.0, 0.0, 6.0), "logit": 20.0, "sh0": sh0_for(c2)},
                {"mean": (0.0, 0.0, 4.0), "logit": 20.0, "sh0": sh0_for(c1)},
            ]
        )
    )
    out = _gsplat(model, CAM_A, 0.5, background=torch.tensor(background), packed=packed)
    expected = torch.tensor([0.999 * a + 0.001 * b for a, b in zip(c1, background, strict=True)])
    torch.testing.assert_close(out.image[0, 24, 32].cpu(), expected, rtol=0.0, atol=1e-6)
    assert out.alpha[0, 24, 32, 0].cpu() == ALPHA_MAX_F32


@LAYOUTS
def test_u7_primitives_outside_the_depth_range_are_culled(packed: bool) -> None:
    """U-7, E6: primitives at z_c = 0.005 (the U-7 case), −4 and 2·10¹⁰ are culled, so the image
    is the background and nothing is visible; z_c = 0.01 itself is kept."""
    background = torch.tensor([0.25, 0.5, 0.75], device="cuda")
    rows = [{"mean": (0.0, 0.0, z), "logit": 20.0} for z in (0.005, -4.0, 2e10)]
    out = _gsplat(
        _cuda(model_from_rows(rows)),
        CAM_A,
        0.5,
        background=background,
        return_aux=True,
        packed=packed,
    )
    assert _background_only(out, background)
    assert out.visible is not None and out.visible.tolist() == [[False, False, False]]
    near = _gsplat(
        _cuda(model_from_rows([{"mean": (0.0, 0.0, 0.01)}])),
        CAM_A,
        0.5,
        return_aux=True,
        packed=packed,
    )
    assert near.visible is not None and near.visible.tolist() == [[True]]


# E6's other closed forms on gsplat (M4's worked values; they pin gsplat's hardcoded α_max, skip
# and stop, D-72)


@LAYOUTS
def test_e6_pixel_stops_before_the_primitive_reaching_the_transmittance_threshold(
    packed: bool,
) -> None:
    """E6: four layers with σ = 0.9, 0.9, 0.95, 0.9 in depth order take T to 0.1, 0.01, 5e-4 and
    then 5e-5 ≤ 1e-4, so the fourth is excluded and the background shows through 5e-4 of T."""
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    sigmas = [0.9, 0.9, 0.95, 0.9]  # in depth order
    depths = [2.0, 3.0, 4.0, 5.0]
    order = [2, 0, 3, 1]  # row r holds depth layer order[r]
    rows = [
        {"mean": (0.0, 0.0, depths[k]), "logit": logit_of(sigmas[k]), "sh0": sh0_for(colors[k])}
        for k in order
    ]
    background = (0.25, 0.5, 0.75)
    out = _gsplat(
        _cuda(model_from_rows(rows)), CAM_A, 0.5, background=torch.tensor(background), packed=packed
    )
    alphas = [sigmoid64(logit_of(s)) for s in sigmas]
    expected = [0.0, 0.0, 0.0]
    t = 1.0
    for k in range(3):
        expected = [e + alphas[k] * t * c for e, c in zip(expected, colors[k], strict=True)]
        t *= 1.0 - alphas[k]
    expected = [e + t * b for e, b in zip(expected, background, strict=True)]
    torch.testing.assert_close(
        out.image[0, 24, 32].cpu().double(), torch.tensor(expected).double(), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        out.alpha[0, 24, 32, 0].cpu().double(), torch.tensor(1.0 - t).double(), rtol=0, atol=1e-6
    )


@LAYOUTS
def test_e6_terms_below_the_alpha_threshold_are_skipped(packed: bool) -> None:
    """E6: with Σ′ = 1.3·I (a = 0.25 on the axis) and σ = 0.5, pixel (24, 36), 4 px from the
    mean, has α = 0.5·e^(−8/1.3) ≈ 0.00106 < 1/255 and stays exactly black; pixel (24, 35) has
    α = 0.5·e^(−4.5/1.3) and shows the color 0.5."""
    out = _gsplat(_cuda(model_from_rows([{"mean": (0.0, 0.0, 4.0)}])), CAM_A, 0.5, packed=packed)
    image = out.image.detach().cpu()
    assert torch.equal(image[0, 24, 36], torch.zeros(3))
    alpha = 0.5 * math.exp(-4.5 / 1.3)
    torch.testing.assert_close(image[0, 24, 35], torch.full((3,), 0.5 * alpha), rtol=0, atol=1e-6)


@LAYOUTS
def test_e6_jacobian_uses_the_clamped_direction_and_the_mean_does_not(packed: bool) -> None:
    """E6: camera B (c_x = 20) clamps x_c/z_c to [−1.85, 3.35] for J only. A unit-scale row at
    x_c/z_c = −2.5 projects unclamped to μ′ = (−20, 24.5) with Σ′₀₀ = 71.06 (116.3 unclamped); a
    row at +3.5 checks the upper bound (Σ′₀₀ = 195.86, 212.3 unclamped)."""
    cam_b = pinhole(64, 48, fx=16.0, fy=16.0, cx=20.0, cy=24.5)
    for x, mean2d, pixel, cov00 in [
        (-10.0, (-20.0, 24.5), (24, 0), 71.06),
        (14.0, (76.0, 24.5), (24, 63), 195.86),
    ]:
        model = _cuda(model_from_rows([{"mean": (x, 0.0, 4.0), "scale": 1.0}]))
        out = _gsplat(model, cam_b, 0.5, return_aux=True, packed=packed)
        assert torch.equal(_unpacked(out)[0, 0].cpu(), torch.tensor(mean2d))
        dx = pixel[1] + 0.5 - mean2d[0]
        alpha = 0.5 * math.exp(-0.5 * dx * dx / cov00)
        torch.testing.assert_close(
            out.image[0, pixel[0], pixel[1]].detach().cpu(),
            torch.full((3,), 0.5 * alpha),
            rtol=0,
            atol=1e-6,
        )


@LAYOUTS
def test_e6_screen_radii_follow_the_opacity_aware_extent(packed: bool) -> None:
    """E6: a row at μ′x = 72.5, off the 64-px image, with Σ′₀₀ = 7.55 is visible with σ = 1
    (extent 3.329, r_x = 10 reaches the image) and invisible with σ = 0.05 (extent 2.256, r_x = 7;
    a fixed 3σ extent would give 9 and reach it)."""
    for logit, visible in [(20.0, True), (logit_of(0.05), False)]:
        model = _cuda(model_from_rows([{"mean": (10.0, 0.0, 4.0), "logit": logit}]))
        out = _gsplat(model, CAM_A, 0.5, return_aux=True, packed=packed)
        assert out.visible is not None and out.visible.tolist() == [[visible]]


@LAYOUTS
def test_e2_colors_are_not_clamped_at_one(packed: bool) -> None:
    """E2, E7: colors are clamped only at 0, so a color of 1.6 at α = 0.999 renders 0.999·1.6."""
    model = _cuda(
        model_from_rows([{"mean": (0.0, 0.0, 4.0), "logit": 20.0, "sh0": sh0_for((1.6,) * 3)}])
    )
    image = _gsplat(model, CAM_A, 0.5, packed=packed).image.detach().cpu()
    torch.testing.assert_close(image[0, 24, 32], torch.full((3,), 0.999 * 1.6), rtol=0, atol=1e-6)


# The backend against gsplat's own rasterization() (Q-37, Q-38)


@LAYOUTS
def test_backend_matches_gsplat_rasterization(packed: bool) -> None:
    """Q-37: the backend runs the kernels gsplat's rasterization() runs, with D-6's and D-57's
    arguments: on the same per-time arrays and no background, images and alpha agree within 1e-6
    (only the camera center is computed differently, Q-38), and so do the 2D means and the
    visible primitives."""
    import gsplat

    cameras = ring(2, width_px=64, height_px=48, focal=40.0)
    model = _cuda(random_model(40, seed=3, spread=1.5, log_scale=(-2.5, -1.0)))
    out = _gsplat(model, cameras, 0.37, return_aux=True, packed=packed)
    g = model.at_time(0.37)
    image, alpha, meta = gsplat.rasterization(
        g.means_t,
        g.quats_n,
        g.scales,
        g.opacities_t,
        g.sh,
        torch.stack([cam.world_to_camera for cam in cameras]).cuda(),
        torch.stack([cam.K for cam in cameras]).cuda(),
        64,
        48,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        eps2d=0.3,
        sh_degree=3,
        packed=packed,
        tile_size=16,
        rasterize_mode="classic",
    )
    torch.testing.assert_close(out.image, image, rtol=0, atol=1e-6)
    torch.testing.assert_close(out.alpha, alpha, rtol=0, atol=1e-6)
    visible = (meta["radii"] > 0).all(dim=-1)
    assert out.visible is not None and out.means2d is not None
    if packed:
        assert out.gaussian_ids is not None
        assert torch.equal(out.gaussian_ids, meta["gaussian_ids"])
        assert torch.equal(out.means2d, meta["means2d"])
    else:
        assert torch.equal(out.visible, visible)
        assert torch.equal(out.means2d[visible], meta["means2d"][visible])
    assert int(out.visible.sum()) >= 20
