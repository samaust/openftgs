# SPDX-License-Identifier: Apache-2.0
"""render() with the gsplat backend on a CUDA device (E2, E6, D-6, D-31, D-32, D-57, D-75, D-76;
§1.4; Q-31 to Q-33, Q-36, Q-39): U-7, U-12, the GPU variants of U-4, U-8 and U-20, the Q-33
boundary cases and the exact 1/255 boundary, both layouts (packed and unpacked) and a cross-check
against gsplat's own rasterization().

Expected values are M4's worked values (`test_render.py`), or come from the reference backend
rendering the same CUDA model, so both backends see bit-identical per-time arrays. Pixels are
written as (row v, column u).
"""

import math
import warnings
from typing import Any, Literal

import pytest
import torch
from builders import (
    CAM_A,
    U8_CASES,
    UNIT_TIME,
    logit_of,
    model_from_rows,
    pinhole,
    random_model,
    random_params,
    rescaled_time_unit,
    ring,
    sh0_for,
    sigmoid64,
    u8_scene,
)
from render_margins import (
    boundary_margins,
    depth_near_plane,
    excluded_share,
    opacity_near_cull,
)
from torch import Tensor

from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.types import Camera, RenderOutput, TimeNormalizer

pytestmark = pytest.mark.gpu

F32 = torch.float32
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
LAYOUTS = pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
_CAMERAS = ring(2, width_px=64, height_px=48, focal=40.0)  # two ring cameras, 64×48


def _cuda(model: FreeTimeGaussians) -> FreeTimeGaussians:
    return model.to("cuda")


def _gsplat(
    model: FreeTimeGaussians, cameras: Camera | list[Camera], t: float, **kw: Any
) -> RenderOutput:
    """The gsplat backend at normalized time t."""
    kw.setdefault("time_units", "normalized")
    return render(model, cameras, t, backend="gsplat", **kw)


def _reference(
    model: FreeTimeGaussians, cameras: Camera | list[Camera], t: float, **kw: Any
) -> RenderOutput:
    """The reference backend at normalized time t, on the same model and device."""
    return render(model, cameras, t, time_units="normalized", backend="reference", **kw)


@pytest.fixture(autouse=True, scope="module")
def _warm_up() -> None:
    """Compiles the temporal slice once before the tests, for one row and for several (dynamo
    specializes a single row; Q-13)."""
    for rows in ([{"mean": (0.0, 0.0, 4.0)}], [{"mean": (0.0, 0.0, 4.0)}] * 3):
        for packed in (True, False):
            _gsplat(_cuda(model_from_rows(rows)), CAM_A, 0.5, packed=packed)


def _unpacked(out: RenderOutput) -> Tensor:
    """The 2D means as [C, N, 2], zero where not visible, from either layout: packed rows are
    camera-major and ascending within a camera (D-76), which is the order of the True entries of
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


# The backend against gsplat's own rasterization() (D-75, D-76)


@LAYOUTS
def test_backend_matches_gsplat_rasterization(packed: bool) -> None:
    """D-75: the backend runs the kernels gsplat's rasterization() runs, with D-6's and D-57's
    arguments: on the same per-time arrays and no background, images and alpha agree within 1e-6
    (only the camera center is computed differently, D-76), and so do the 2D means and the
    visible primitives; in unpacked mode, a per-camera background composited outside the kernel
    agrees with gsplat's own within 1e-6 too (D-76)."""
    import gsplat

    cameras = _CAMERAS
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
    if not packed:
        # D-76: the background composited outside the kernel agrees with gsplat's own, which
        # gsplat 1.5.3 accepts per camera in unpacked mode only.
        backgrounds = torch.tensor([[0.25, 0.5, 0.75], [1.0, 0.0, 0.5]], device="cuda")
        with_background = _gsplat(model, cameras, 0.37, background=backgrounds, packed=False)
        in_kernel, _, _ = gsplat.rasterization(
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
            packed=False,
            tile_size=16,
            rasterize_mode="classic",
            backgrounds=backgrounds,
        )
        torch.testing.assert_close(with_background.image, in_kernel, rtol=0, atol=1e-6)
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


@LAYOUTS
def test__backend_follows_the_rasterize_mode_constant(
    packed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private, D-6, D-72 (#52): the rasterize mode has one source, `_RASTERIZE_MODE`. Set to
    "antialiased" (which D-6 doesn't use), the backend computes gsplat's compensations and applies
    them to the opacities, so it matches gsplat's own antialiased rasterization() within 1e-6."""
    import gsplat

    import openftgs.render

    monkeypatch.setattr(openftgs.render, "_RASTERIZE_MODE", "antialiased")
    model = _cuda(random_model(40, seed=4, spread=1.5, log_scale=(-2.5, -1.0)))
    out = _gsplat(model, _CAMERAS, 0.37, packed=packed)
    g = model.at_time(0.37)
    image, alpha, _ = gsplat.rasterization(
        g.means_t,
        g.quats_n,
        g.scales,
        g.opacities_t,
        g.sh,
        torch.stack([cam.world_to_camera for cam in _CAMERAS]).cuda(),
        torch.stack([cam.K for cam in _CAMERAS]).cuda(),
        64,
        48,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        eps2d=0.3,
        sh_degree=3,
        packed=packed,
        tile_size=16,
        rasterize_mode="antialiased",
    )
    torch.testing.assert_close(out.image, image, rtol=0, atol=1e-6)
    torch.testing.assert_close(out.alpha, alpha, rtol=0, atol=1e-6)
    monkeypatch.undo()
    classic = _gsplat(model, _CAMERAS, 0.37, packed=packed)
    assert float((out.image - classic.image).detach().abs().max()) > 1e-3, (
        "the mode changed nothing"
    )


# Layouts: packed and unpacked give the same render (D-57, D-75, D-76; Q-36)

PLACEMENTS = pytest.mark.parametrize("precull", ["internal", "compact"])


def _culling_scene(seed: int) -> FreeTimeGaussians:
    """Rows around two ring cameras (64×48), some below the opacity cull (every third row), some
    behind a camera or off the image."""
    model = random_model(48, seed=seed, spread=4.5, log_scale=(-2.5, -0.5), logit=(-3.0, 3.0))
    with torch.no_grad():
        model.opacity_logits[::3] = -9.0
    return _cuda(model)


@PLACEMENTS
def test_packed_and_unpacked_layouts_render_alike(precull: str) -> None:
    """D-57, D-75, D-76: packed mode changes memory and speed, not results. Images and alpha agree
    within 1e-6, `visible` is equal, and the packed 2D means are the unpacked ones at the visible
    entries read camera by camera in ascending row order, labeled by gaussian_ids."""
    model = _culling_scene(0)
    out = {
        packed: _gsplat(model, _CAMERAS, 0.37, return_aux=True, precull=precull, packed=packed)
        for packed in (True, False)
    }
    packed, unpacked = out[True], out[False]
    torch.testing.assert_close(packed.image, unpacked.image, rtol=0, atol=1e-6)
    torch.testing.assert_close(packed.alpha, unpacked.alpha, rtol=0, atol=1e-6)
    assert packed.visible is not None and unpacked.visible is not None
    assert torch.equal(packed.visible, unpacked.visible)
    assert unpacked.gaussian_ids is None and unpacked.means2d is not None
    assert packed.gaussian_ids is not None and packed.means2d is not None
    assert packed.gaussian_ids.dtype == torch.int64
    cams, ids = torch.nonzero(unpacked.visible, as_tuple=True)
    assert torch.equal(packed.gaussian_ids, ids)
    assert torch.equal(packed.means2d, unpacked.means2d[cams, ids])
    culled = int((~unpacked.visible).sum())
    assert culled >= 20 and int(unpacked.visible.sum()) >= 10


@PLACEMENTS
def test_unpacked_means2d_is_zero_off_the_visible_rows(precull: str) -> None:
    """Q-32, D-75: in unpacked mode, means2d is μ′ for visible rows and exactly 0 elsewhere (gsplat
    leaves those rows uninitialized), with and without compaction."""
    model = _culling_scene(1)
    out = _gsplat(model, _CAMERAS, 0.37, return_aux=True, precull=precull, packed=False)
    assert out.means2d is not None and out.visible is not None
    assert out.means2d.shape == (2, 48, 2) and out.visible.shape == (2, 48)
    hidden = out.means2d[~out.visible]
    assert hidden.numel() > 0 and torch.equal(hidden, torch.zeros_like(hidden))


@LAYOUTS
@PLACEMENTS
def test_means2d_gradient_is_the_gradient_with_respect_to_the_2d_mean(
    packed: bool, precull: str
) -> None:
    """Q-32, D-75: with return_aux, means2d keeps its gradient in all four layouts and placements.
    For the on-axis row (Σ′ = 1.3·I, σ = 0.5, color 0.5) and L = the red value of pixel (24, 34),
    2 px right of μ′, L = 0.5·0.5·e^(−q) with q = ½·2²/1.3, so ∂L/∂μ′ = (L·2/1.3, 0); the second
    row is below the opacity cull and has no visible entry, so its gradient is 0 (unpacked) or
    absent (packed)."""
    model = _cuda(
        model_from_rows([{"mean": (0.0, 0.0, 4.0)}, {"mean": (1.0, 0.0, 4.0), "logit": -9.0}])
    )
    out = _gsplat(model, CAM_A, 0.5, return_aux=True, precull=precull, packed=packed)
    assert out.means2d is not None
    torch.autograd.backward(out.image[0, 24, 34, 0])
    value = 0.25 * math.exp(-0.5 * 4.0 / 1.3)
    grad = out.means2d.grad
    assert grad is not None
    if packed:
        assert out.gaussian_ids is not None and out.gaussian_ids.tolist() == [0]
        expected = torch.tensor([[value * 2.0 / 1.3, 0.0]], dtype=torch.float64)
    else:
        expected = torch.tensor([[[value * 2.0 / 1.3, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(grad.cpu().double(), expected, rtol=1e-5, atol=1e-9)
    if not packed:
        assert float(grad[0, 1].abs().sum()) == 0.0


@LAYOUTS
@PLACEMENTS
def test_empty_and_fully_culled_scenes_render_the_background(packed: bool, precull: str) -> None:
    """Q-32, D-76: a model with no primitives, and a scene whose primitives are all culled (below
    the opacity cull, behind the camera), render the background with alpha 0 in every layout and
    placement, with no visible entry: zero means2d [C, N, 2] unpacked, empty [0, 2] means2d and
    gaussian_ids packed."""
    background = torch.tensor([0.25, 0.5, 0.75], device="cuda")
    models = [
        _cuda(model_from_rows([])),
        _cuda(
            model_from_rows([{"mean": (0.0, 0.0, 4.0), "logit": -8.0}, {"mean": (0.0, 0.0, -4.0)}])
        ),
    ]
    for model in models:
        n = model.num_gaussians
        out = _gsplat(
            model,
            [CAM_A, CAM_A],
            0.5,
            background=background,
            return_aux=True,
            precull=precull,
            packed=packed,
        )
        assert out.image.shape == (2, 48, 64, 3) and out.alpha.shape == (2, 48, 64, 1)
        assert _background_only(out, background)
        assert out.visible is not None and out.visible.shape == (2, n)
        assert not bool(out.visible.any())
        assert out.means2d is not None
        if packed:
            assert out.gaussian_ids is not None and out.gaussian_ids.shape == (0,)
            assert out.means2d.shape == (0, 2)
        else:
            assert out.gaussian_ids is None
            assert torch.equal(out.means2d, torch.zeros(2, n, 2, device="cuda"))


@LAYOUTS
@PLACEMENTS
def test_renders_with_no_visible_row_keep_exactly_zero_gradients(
    packed: bool, precull: str
) -> None:
    """U-8, D-57, Q-32, D-76: when no row is visible, whether none reaches gsplat (an empty model,
    or compaction keeping none) or gsplat culls them all, the render stays on the autograd graph:
    backpropagating image and alpha gives every stored tensor an exactly zero gradient and the 2D
    means a zero gradient, in every layout and placement, as on the reference backend."""
    for scene in ("empty model", "no row kept"):
        for backend in ("gsplat", "reference"):
            built = model_from_rows([]) if scene == "empty model" else u8_scene(1, F32, kept=0)
            model = _cuda(built)
            out = (_gsplat if backend == "gsplat" else _reference)(
                model, ring(3), 0.37, return_aux=True, packed=packed, precull=precull
            )
            assert out.visible is not None and not bool(out.visible.any())
            torch.autograd.backward(out.image.sum() + out.alpha.sum())
            for name in PARAM_NAMES:
                grad = getattr(model, name).grad
                assert grad is not None and torch.equal(grad, torch.zeros_like(grad)), (
                    backend,
                    name,
                )
            assert out.means2d is not None and out.means2d.grad is not None, backend
            assert torch.equal(out.means2d.grad, torch.zeros_like(out.means2d)), backend


@LAYOUTS
def test_cameras_render_independently(packed: bool) -> None:
    """Q-32, D-76: a single Camera renders like a one-element list, and a batch of cameras like
    each camera alone, with a per-camera background [C, 3] in both layouts."""
    model = _culling_scene(2)
    cameras = ring(3, width_px=64, height_px=48, focal=40.0)
    backgrounds = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.25, 0.125], [1.0, 1.0, 1.0]])
    batch = _gsplat(model, cameras, 0.4, background=backgrounds, packed=packed)
    for c, camera in enumerate(cameras):
        alone = _gsplat(model, camera, 0.4, background=backgrounds[c], packed=packed)
        listed = _gsplat(model, [camera], 0.4, background=backgrounds[c], packed=packed)
        assert torch.equal(alone.image, listed.image)
        assert torch.equal(batch.image[c], alone.image[0])
        assert torch.equal(batch.alpha[c], alone.alpha[0])
    assert float(batch.alpha.detach().max()) > 0.5


def test_gsplat_backend_rejects_float64_models_on_the_device() -> None:
    """D-76 (GPU): gsplat's kernels compute in float32, so a float64 model on a CUDA device raises
    ValueError naming the reference backend, in both layouts."""
    model = _cuda(model_from_rows([{"mean": (0.0, 0.0, 4.0)}])).double()
    for packed in (True, False):
        with pytest.raises(ValueError, match="reference"):
            _gsplat(model, CAM_A, 0.5, packed=packed)


# U-8 · culling σ_eff < 1/255 before the backend changes nothing (D-32, D-57; Q-31), on gsplat


@LAYOUTS
@pytest.mark.parametrize("case", list(U8_CASES))
def test_u8_compaction_leaves_renders_unchanged(case: str, packed: bool) -> None:
    """U-8 (GPU): rendering with compaction (precull = "compact") and with the opacity cull inside
    gsplat's projection ("internal") gives images and alpha within 1e-6 and equal 2D means,
    visibility and gaussian_ids, with 0 or 1 rows kept, a third of the rows culled, and a depth
    tie between visible rows."""
    n, kept, tie = U8_CASES[case]
    cameras = ring(3)
    model = _cuda(u8_scene(n, F32, kept=kept, tie=tie))
    passing = int(model.at_time(0.37).passes_cull.sum())
    assert passing == kept if kept is not None else 0 < passing < n
    out = {
        precull: _gsplat(model, cameras, 0.37, return_aux=True, precull=precull, packed=packed)
        for precull in ("internal", "compact")
    }
    internal, compact = out["internal"], out["compact"]
    torch.testing.assert_close(compact.image, internal.image, rtol=0, atol=1e-6)
    torch.testing.assert_close(compact.alpha, internal.alpha, rtol=0, atol=1e-6)
    for field in ("means2d", "visible"):
        assert torch.equal(getattr(compact, field), getattr(internal, field)), field
    if packed:
        assert compact.gaussian_ids is not None and internal.gaussian_ids is not None
        assert torch.equal(compact.gaussian_ids, internal.gaussian_ids)
    if tie:
        assert internal.visible is not None and bool(internal.visible[:, [4, 5]].all())


@LAYOUTS
@PLACEMENTS
def test_u8_culled_primitives_get_exactly_zero_render_gradient(packed: bool, precull: str) -> None:
    """U-8 (GPU): after backpropagating a random weighting of image and alpha, every primitive
    visible in no camera (below the opacity cull, behind a camera or off the image) has an exactly
    zero gradient in all nine stored tensors, in both placements and both layouts; every gradient
    is finite."""
    cameras = ring(3)
    model = _cuda(u8_scene(64, F32, tie=True))
    out = _gsplat(model, cameras, 0.37, return_aux=True, precull=precull, packed=packed)
    weights = torch.Generator(device="cuda").manual_seed(1)
    loss = (out.image * torch.rand(out.image.shape, generator=weights, device="cuda")).sum() + (
        out.alpha * torch.rand(out.alpha.shape, generator=weights, device="cuda")
    ).sum()
    params = [getattr(model, name) for name in PARAM_NAMES]
    grads = torch.autograd.grad(loss, params)
    assert out.visible is not None
    culled = ~out.visible.any(dim=0)
    assert int(culled.sum()) >= 20 and int((~culled).sum()) >= 10
    for name, grad in zip(PARAM_NAMES, grads, strict=True):
        assert torch.equal(grad[culled], torch.zeros_like(grad[culled])), name
        assert bool(torch.isfinite(grad).all()), name
        assert float(grad[~culled].abs().sum()) > 0.0, name


# U-7's random scenes and U-12, gsplat against the reference (Q-39's margins)

_T = 0.37


def _random_scene(seed: int, *, min_rows: int = 1) -> FreeTimeGaussians:
    """U-7's and U-12's random scenes: N from `min_rows` to 64 rows around two ring cameras, some
    behind a camera, off the image or below the opacity cull, anisotropic and rotated, SH degree
    3, σ_eff ≤ 0.95. Rows whose σ_eff or z_c lands within Q-39's margins of the culls' bounds are
    moved off them."""
    n = min_rows + (37 * seed) % (65 - min_rows)
    params = random_params(
        n,
        generator=torch.Generator().manual_seed(500 + seed),
        spread=4.5,
        log_scale=(-2.5, -0.5),
        logit=(-6.0, 2.9),
        speed=0.5,
    )
    model = FreeTimeGaussians(params, sh_degree=3, time=UNIT_TIME)
    for _ in range(10):
        near_cull = torch.from_numpy(opacity_near_cull(model, _T))
        near_plane = torch.from_numpy(depth_near_plane(model, _CAMERAS, _T))
        if not (bool(near_cull.any()) or bool(near_plane.any())):
            return _cuda(model)
        with torch.no_grad():
            model.opacity_logits[near_cull] += 0.05
            model.means[near_plane] += 0.01
    raise AssertionError(f"scene {seed} keeps rows near a cull's bound")


def _compare(
    seed: int, gsplat_out: RenderOutput, reference: RenderOutput, model: FreeTimeGaussians
) -> tuple[float, float, int, int]:
    """U-7 on one scene: image and alpha within 1e-4 outside Q-39's excluded pixels, `visible`
    equal outside its excluded rows, 2D means of the rows visible in both within 1e-4 px. Returns
    the largest difference outside the mask, the excluded share, the number of excluded pixels,
    and how many of them differ by more than 1e-4 in image or alpha (Q-39)."""
    margins = boundary_margins(model, _CAMERAS, _T)
    share = excluded_share(margins)
    assert share <= 0.01, (seed, share)
    keep = torch.from_numpy(~margins.pixels).to("cuda")
    worst = 0.0
    differs = torch.zeros_like(keep)
    for field in ("image", "alpha"):
        diff = (getattr(gsplat_out, field) - getattr(reference, field)).detach().abs()
        a, b = getattr(gsplat_out, field).detach()[keep], getattr(reference, field).detach()[keep]
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4, msg=lambda m: f"seed {seed}: {m}")
        worst = max(worst, float((a - b).abs().max()))
        differs |= diff.amax(dim=-1) > 1e-4
    assert gsplat_out.visible is not None and reference.visible is not None
    rows = torch.from_numpy(~margins.rows).to("cuda")
    assert torch.equal(gsplat_out.visible[rows], reference.visible[rows]), seed
    both = gsplat_out.visible & reference.visible
    torch.testing.assert_close(
        _unpacked(gsplat_out)[both], _unpacked(reference)[both], rtol=0, atol=1e-4
    )
    return worst, share, int((~keep).sum()), int((differs & ~keep).sum())


@LAYOUTS
def test_u7_random_scenes_match_the_reference(packed: bool) -> None:
    """U-7: 50 random scenes (N ≤ 64, 64×48, two cameras) rendered by gsplat and by the reference
    from the same CUDA model agree within 1e-4 in RGB and alpha, outside the pixels Q-39 excludes
    (at most 1% of a camera's pixels); `visible` agrees outside the excluded rows, and the 2D
    means of rows visible in both within 1e-4 px. Prints the largest difference outside the mask,
    the largest excluded share, and how many excluded pixels differ by more than 1e-4 (Q-39)."""
    worst, share = 0.0, 0.0
    excluded, excluded_differ = 0, 0
    visible_rows = 0
    for seed in range(50):
        model = _random_scene(seed)
        out = _gsplat(model, _CAMERAS, _T, return_aux=True, packed=packed)
        reference = _reference(model, _CAMERAS, _T, return_aux=True)
        scene_worst, scene_share, scene_excluded, scene_differ = _compare(
            seed, out, reference, model
        )
        worst, share = max(worst, scene_worst), max(share, scene_share)
        excluded, excluded_differ = excluded + scene_excluded, excluded_differ + scene_differ
        assert reference.visible is not None
        visible_rows += int(reference.visible.sum())
    print(
        f"\nU-7 ({'packed' if packed else 'unpacked'}): largest |Δ| outside the mask {worst:.3g}, "
        f"largest excluded share {share:.3g}, excluded pixels differing by more than 1e-4 "
        f"{excluded_differ} of {excluded}, visible primitive–camera pairs {visible_rows}"
    )
    assert visible_rows >= 500


def _weighted_loss(out: RenderOutput, keep: Tensor, seed: int) -> Tensor:
    """A random weighting of image and alpha, zero at the pixels Q-39 excludes."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    image_w = torch.rand(out.image.shape, generator=gen, device="cuda") * keep[..., None]
    alpha_w = torch.rand(out.alpha.shape, generator=gen, device="cuda") * keep[..., None]
    return (out.image * image_w).sum() + (out.alpha * alpha_w).sum()


@LAYOUTS
def test_u12_gradients_match_the_reference(packed: bool) -> None:
    """U-12: for 20 random scenes (U-7's generator, at least 8 rows), the gradients of a random
    weighting of image and alpha with respect to each of the nine stored tensors agree between
    gsplat and the reference: cosine ≥ 0.999 and relative L2 error ≤ 1e-3. The weights are zero
    at the pixels Q-39 excludes, at most 1% of a camera's pixels. Prints the worst cosine and
    relative error and the largest excluded share."""
    worst_cos, worst_rel, worst_share = 1.0, 0.0, 0.0
    for seed in range(20):
        model = _random_scene(100 + seed, min_rows=8)
        margins = boundary_margins(model, _CAMERAS, _T)
        share = excluded_share(margins)
        assert share <= 0.01, (seed, share)
        worst_share = max(worst_share, share)
        keep = torch.from_numpy(~margins.pixels).to("cuda")
        params = [getattr(model, name) for name in PARAM_NAMES]
        grads = {}
        backends: tuple[Literal["gsplat", "reference"], ...] = ("gsplat", "reference")
        for backend in backends:
            out = render(
                model,
                _CAMERAS,
                _T,
                time_units="normalized",
                backend=backend,
                packed=packed,
            )
            grads[backend] = torch.autograd.grad(_weighted_loss(out, keep, seed), params)
        for name, a, b in zip(PARAM_NAMES, grads["gsplat"], grads["reference"], strict=True):
            a64, b64 = a.double().flatten(), b.double().flatten()
            norm = float(torch.linalg.vector_norm(b64))
            assert norm > 0.0, (seed, name)
            cos = float(a64 @ b64) / (float(torch.linalg.vector_norm(a64)) * norm)
            rel = float(torch.linalg.vector_norm(a64 - b64)) / norm
            assert cos >= 0.999 and rel <= 1e-3, (seed, name, cos, rel)
            worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    print(
        f"\nU-12 ({'packed' if packed else 'unpacked'}): worst cosine {worst_cos:.7f}, "
        f"worst relative L2 error {worst_rel:.3g}, largest excluded share {worst_share:.3g}"
    )


# U-4 · renders are invariant under an affine change of the time unit (E14, §1.4 "Time"), on gsplat

_SPAN = TimeNormalizer(2.0, 12.0)


def _raw(model: FreeTimeGaussians, tau: float, packed: bool) -> RenderOutput:
    with torch.no_grad():
        return render(model, ring(2), tau, packed=packed)


@LAYOUTS
@pytest.mark.parametrize(("a", "b"), [(2.0, 5.0), (1e-3, -3.0), (1e3, 0.0)])
def test_u4_renders_are_invariant_under_an_affine_time_change(
    a: float, b: float, packed: bool
) -> None:
    """U-4 (GPU): (τ, μτ, d, v_τ) → (aτ + b, aμτ + b, a·d, v_τ/a) leaves gsplat renders unchanged,
    max abs difference ≤ 1e-4, at τ₀, inside the span and at τ₁; no warning is raised inside the
    span (Q-9). A first render outside the strict block compiles the temporal slice."""
    model = random_model(16, seed=0, time=_SPAN, spread=0.6, log_scale=(-2.0, -0.7))
    exchange = model.to_exchange()
    before = FreeTimeGaussians.from_exchange(exchange, device="cuda")
    after = FreeTimeGaussians.from_exchange(rescaled_time_unit(exchange, a, b), device="cuda")
    _raw(before, 7.3, packed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tau in (2.0, 7.3, 12.0):
            x, y = _raw(before, tau, packed), _raw(after, a * tau + b, packed)
            assert float(x.alpha.max()) > 0.1
            torch.testing.assert_close(y.image, x.image, rtol=0, atol=1e-4)
            torch.testing.assert_close(y.alpha, x.alpha, rtol=0, atol=1e-4)


@LAYOUTS
@pytest.mark.parametrize(("a", "b"), [(2.0, 5.0), (1e-3, -3.0), (1e3, 0.0)])
def test_u4_single_timestamp_renders_are_invariant_too(a: float, b: float, packed: bool) -> None:
    """U-4 (GPU) with one timestamp (Q-30): t ≡ 0 and every time center is τ₀, so scaling
    durations by a changes nothing on screen (≤ 1e-4), and τ₀ itself raises no warning."""
    single = TimeNormalizer(5.0, 5.0)
    params = random_params(16, generator=torch.Generator().manual_seed(3), spread=0.6)
    params["times"] = torch.zeros(16)
    params["velocities"] = torch.zeros(16, 3)
    exchange = FreeTimeGaussians(params, sh_degree=3, time=single).to_exchange()
    before = FreeTimeGaussians.from_exchange(exchange, device="cuda")
    after = FreeTimeGaussians.from_exchange(rescaled_time_unit(exchange, a, b), device="cuda")
    _raw(before, 5.0, packed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, y = _raw(before, 5.0, packed), _raw(after, a * 5.0 + b, packed)
    assert float(x.alpha.max()) > 0.1
    torch.testing.assert_close(y.image, x.image, rtol=0, atol=1e-4)
    torch.testing.assert_close(y.alpha, x.alpha, rtol=0, atol=1e-4)


# U-20 · camera conventions, the projection cases (§1.4, D-28), on gsplat: the dyadic float32 cases,
# which are exact, so they are compared exactly, within U-20's 1e-5 px (float32 cannot reach 1e-5 px
# at 1000 px in general, so random poses stay with the float64 reference)

_U20_W2C = [[0.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 2.0], [1.0, 0.0, 0.0, 3.0]]
_U20_POINT = (1.0, 0.5, -1.0)  # x_c = R_w·p + t_w = (0.5, 3, 4)


def _projected(camera: Camera, packed: bool) -> RenderOutput:
    model = _cuda(model_from_rows([{"mean": _U20_POINT, "scale": 1e-3, "logit": 20.0}]))
    out = _gsplat(model, camera, 0.5, return_aux=True, packed=packed)
    assert out.visible is not None and out.visible.tolist() == [[True]]
    return out


@LAYOUTS
def test_u20_known_point_projects_to_the_expected_pixel(packed: bool) -> None:
    """U-20 (GPU), §1.4: with world_to_camera = [[0, −1, 0, 1], [0, 0, −1, 2], [1, 0, 0, 3]], the
    point (1, 0.5, −1) is at x_c = (0.5, 3, 4); with f = (32, 24) and c = (20.5, 10.5) it projects
    to (24.5, 28.5), the center of pixel (28, 24), where the render peaks."""
    camera = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.5, cy=10.5, world_to_camera=_U20_W2C)
    out = _projected(camera, packed)
    assert torch.equal(_unpacked(out)[0, 0].cpu(), torch.tensor([24.5, 28.5]))
    assert divmod(int(torch.argmax(out.alpha.detach()[0, :, :, 0])), 48) == (28, 24)


@LAYOUTS
def test_u20_opencv_calibration_converts_with_a_half_pixel_shift(packed: bool) -> None:
    """U-20 (GPU), §1.4: an OpenCV calibration with c = (20, 10) puts the point at (24, 28);
    converted with c + 0.5 it projects to (24.5, 28.5), the same pixel's center in the continuous
    convention."""
    opencv = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.0, cy=10.0, world_to_camera=_U20_W2C)
    shifted = pinhole(48, 40, fx=32.0, fy=24.0, cx=20.5, cy=10.5, world_to_camera=_U20_W2C)
    for camera, expected in [(opencv, [24.0, 28.0]), (shifted, [24.5, 28.5])]:
        assert torch.equal(
            _unpacked(_projected(camera, packed))[0, 0].cpu(), torch.tensor(expected)
        )


@LAYOUTS
def test_u20_resizing_by_one_half_halves_projections(packed: bool) -> None:
    """U-20 (GPU), §1.4: resizing 48×40 by r = 0.5 multiplies f and c by 0.5, so (24.5, 28.5)
    becomes (12.25, 14.25)."""
    resized = pinhole(24, 20, fx=16.0, fy=12.0, cx=10.25, cy=5.25, world_to_camera=_U20_W2C)
    assert torch.equal(
        _unpacked(_projected(resized, packed))[0, 0].cpu(), torch.tensor([12.25, 14.25])
    )


# Q-33's boundary cases (forward notes on #37), gsplat against the reference


@LAYOUTS
@PLACEMENTS
def test_box_touching_the_image_edge_counts_as_off_it(packed: bool, precull: str) -> None:
    """E6, Q-33 (GPU): with a = 10⁻³ and σ = 1 the radii are ⌈3.329·√0.3001⌉ = 2. A box μ′ ± 2
    that only touches the image (μ′x = 66 or −2) is culled; one that overlaps it by 1/16 px
    (65.9375, −1.9375) is visible, as on the reference."""
    rows = [
        {"mean": (x, 0.0, 4.0), "scale": 1e-3, "logit": 20.0}
        for x in (8.375, 8.359375, -8.625, -8.609375)
    ]
    model = _cuda(model_from_rows(rows))
    out = _gsplat(model, CAM_A, 0.5, return_aux=True, packed=packed, precull=precull)
    reference = _reference(model, CAM_A, 0.5, return_aux=True)
    assert out.visible is not None and out.visible.tolist() == [[False, True, False, True]]
    assert reference.visible is not None and torch.equal(out.visible, reference.visible)
    assert _unpacked(out)[0, [1, 3], 0].tolist() == [65.9375, -1.9375]
    torch.testing.assert_close(out.image, reference.image, rtol=0, atol=1e-4)


@LAYOUTS
@PLACEMENTS
def test_depth_ties_composite_the_lower_row_first(packed: bool, precull: str) -> None:
    """E6, Q-33 (GPU): two rows at the same place tie in depth and composite in row order, red in
    front of green: 0.5·(1, 0, 0) + 0.5·0.5·(0, 1, 0) = (0.5, 0.25, 0), as on the reference. Rows
    below the opacity cull between and before them change nothing, which under compaction checks
    that the kept rows keep their relative order."""
    red: dict[str, Any] = {"mean": (0.0, 0.0, 4.0), "sh0": sh0_for((1.0, 0.0, 0.0))}
    green: dict[str, Any] = {"mean": (0.0, 0.0, 4.0), "sh0": sh0_for((0.0, 1.0, 0.0))}
    culled: dict[str, Any] = {"mean": (0.0, 0.0, 4.0), "logit": -9.0, "sh0": sh0_for((0, 0, 1))}
    scenes: list[list[dict[str, Any]]] = [[red, green], [culled, red, culled, culled, green]]
    for rows in scenes:
        model = _cuda(model_from_rows(rows))
        image = _gsplat(model, CAM_A, 0.5, packed=packed, precull=precull).image.detach()
        reference = _reference(model, CAM_A, 0.5)
        torch.testing.assert_close(
            image[0, 24, 32].cpu(), torch.tensor([0.5, 0.25, 0.0]), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(image, reference.image, rtol=0, atol=1e-4)


# The exact 1/255 boundary (E6, D-32; #37, M4 milestone review finding 5; Q-39)


def _threshold_rows() -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Rows whose σ_eff straddles float32(1/255) in steps of about one float32 ulp: 81 logits
    around logit(1/255) with σt = 1, then the first logit above the threshold with time centers
    0.5 − δ (s = 1), whose σt < 1 lowers σ_eff in sub-ulp steps so one row can land exactly on
    the threshold. Each row sits at its own pixel center (v, u) of camera A, 3 px from the next,
    with color 1; the pixels are returned beside the rows."""
    threshold = torch.tensor(1.0 / 255.0, dtype=F32)
    logit = torch.tensor(math.log(1.0 / 254.0), dtype=F32)
    up, down = [logit], [logit]
    for _ in range(40):
        up.append(torch.nextafter(up[-1], torch.tensor(math.inf)))
        down.append(torch.nextafter(down[-1], torch.tensor(-math.inf)))
    logits = [float(x) for x in reversed(down[1:])] + [float(x) for x in up]
    probe = _cuda(model_from_rows([{"mean": (0.0, 0.0, 4.0), "logit": x} for x in logits]))
    sigma = probe.at_time(0.5).opacities_t.cpu()
    first_above = logits[int(torch.nonzero(sigma > threshold)[0])]
    params = [(x, 0.5, 20.0) for x in logits] + [
        (first_above, 0.5 - 5e-5 * j, 0.0) for j in range(1, 201)
    ]
    pixels = [(v, u) for v in range(2, 48, 3) for u in range(2, 63, 3)][: len(params)]
    color = sh0_for((1.0, 1.0, 1.0))
    rows = [
        {
            "mean": ((u - 32) / 4, (v - 24) / 4, 4.0),
            "logit": x,
            "time": time,
            "log_duration": log_duration,
            "sh0": color,
        }
        for (x, time, log_duration), (v, u) in zip(params, pixels, strict=True)
    ]
    return rows, pixels


@LAYOUTS
def test_backends_agree_at_the_opacity_cull_threshold(packed: bool) -> None:
    """E6, D-32, Q-39 (GPU): below float32(1/255) a row is culled in both backends, and exactly at
    it both give it a zero extent, so it is invisible and its pixel shows the background. Just
    above, gsplat's extent from σ_eff / (1/255) and the reference's from 255·σ_eff may round to
    zero in one backend only: then `visible` differs and the row's own pixel differs by at most
    σ_eff·|c − c_bg|; nothing else differs. Prints how many rows sat exactly on the threshold and
    how many differ in visibility."""
    rows, pixels = _threshold_rows()
    model = _cuda(model_from_rows(rows))
    g = model.at_time(0.5)
    sigma, passes = g.opacities_t.cpu(), g.passes_cull.cpu()
    threshold = torch.tensor(1.0 / 255.0, dtype=F32)
    out = _gsplat(model, CAM_A, 0.5, return_aux=True, packed=packed)
    reference = _reference(model, CAM_A, 0.5, return_aux=True)
    assert out.visible is not None and reference.visible is not None
    image, ref_image = out.image.detach()[0].cpu(), reference.image.detach()[0].cpu()
    owned = torch.zeros(image.shape[:2], dtype=torch.bool)
    counts = {"below": 0, "at": 0, "above": 0, "differ": 0}
    for i, (v, u) in enumerate(pixels):
        owned[v, u] = True
        vis, ref_vis = bool(out.visible[0, i]), bool(reference.visible[0, i])
        if sigma[i] <= threshold:
            counts["below" if sigma[i] < threshold else "at"] += 1
            assert bool(passes[i]) == (sigma[i] == threshold), i
            assert not vis and not ref_vis, i
            assert torch.equal(image[v, u], torch.zeros(3)), i
            assert torch.equal(ref_image[v, u], torch.zeros(3)), i
            continue
        counts["above"] += 1
        bound = 1e-6 if vis == ref_vis else float(sigma[i]) * 1.0 + 1e-6  # |c − c_bg| = 1
        counts["differ"] += vis != ref_vis
        torch.testing.assert_close(image[v, u], ref_image[v, u], rtol=0, atol=bound)
    assert torch.equal(image[~owned], torch.zeros_like(image[~owned]))
    assert torch.equal(ref_image[~owned], torch.zeros_like(ref_image[~owned]))
    assert counts["below"] >= 10 and counts["above"] >= 10
    print(f"\n1/255 boundary ({'packed' if packed else 'unpacked'}): {counts}")


# Host synchronization (§2.4 rule 3, D-49)


def _on_device(camera: Camera) -> Camera:
    return Camera(
        K=camera.K.cuda(),
        world_to_camera=camera.world_to_camera.cuda(),
        width=camera.width,
        height=camera.height,
        name=camera.name,
    )


@LAYOUTS
def test_render_synchronizes_only_inside_gsplat(packed: bool) -> None:
    """§2.4 rule 3, D-49, D-76: with cameras and background already on the device, a render
    (opacity cull inside the projection) synchronizes with the host only inside gsplat's own
    calls: every synchronization torch's debug mode reports comes from gsplat's files, with the
    default black background and with a given one. The debug mode is a prototype that misses some
    synchronizations, so this pins only the ones it sees."""
    cameras = [_on_device(camera) for camera in _CAMERAS]
    model = _random_scene(7, min_rows=20)
    for background in (None, torch.tensor([0.25, 0.5, 0.75], device="cuda")):
        _gsplat(model, cameras, _T, background=background, packed=packed)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                _gsplat(model, cameras, _T, background=background, packed=packed)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        sites = {
            f"{record.filename}:{record.lineno}"
            for record in records
            if "synchronizing CUDA operation" in str(record.message)
        }
        assert all("/gsplat/" in site for site in sites), sites
