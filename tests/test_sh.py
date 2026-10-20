# SPDX-License-Identifier: Apache-2.0
"""U-6, basis parts: the real SH basis and the RGB ↔ DC conversion (E2, D-3, D-4; Q-14).

The independent evaluation, `real_sh` in tests/real_sh.py, is written from the general definition
of the real spherical harmonics, not from the table `openftgs.sh` uses. The direction, clamp and
inactive-band parts of U-6 are in test_render.py (M4), and the inactive-band parts on the gsplat
backend in test_render_gsplat.py (#52).
"""

import math

import numpy as np
import pytest
import scipy.special
import torch
from real_sh import real_sh

import openftgs.sh
from openftgs.sh import C0, rgb_to_sh0, sh0_to_rgb, sh_basis, sh_to_color


def _directions(n: int, seed: int) -> np.ndarray:
    """n random unit directions, n random non-unit ones, and the axes."""
    rng = np.random.default_rng(seed)
    unit = rng.normal(size=(n, 3))
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    non_unit = rng.normal(size=(n, 3)) * rng.uniform(0.1, 10.0, size=(n, 1))
    axes = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1]],
        dtype=np.float64,
    )
    return np.concatenate([unit, non_unit, axes], axis=0)


def test_all_lists_the_public_api() -> None:
    """Q-14: `__all__` is exactly the public API."""
    assert sorted(openftgs.sh.__all__) == [
        "C0",
        "rgb_to_sh0",
        "sh0_to_rgb",
        "sh_basis",
        "sh_to_color",
    ]


def test_c0_is_the_l0_constant() -> None:
    """§1.1: C₀ = 0.28209479177387814 = 1/(2√π)."""
    assert C0 == 0.28209479177387814
    assert abs(C0 - 1.0 / (2.0 * math.sqrt(math.pi))) <= 1e-16


def test_reference_matches_scipy_complex_sh() -> None:
    """The test oracle itself: √2·Re/Im of SciPy's complex Y_l^|m| (Condon–Shortley phase)."""
    d = _directions(200, 1)
    ref = real_sh(d, 3)
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))  # polar angle
    phi = np.arctan2(u[:, 1], u[:, 0])  # azimuth
    # Pin SciPy's argument order and phase on two known values first.
    np.testing.assert_allclose(
        scipy.special.sph_harm_y(1, 0, theta, phi).real,
        math.sqrt(3 / (4 * math.pi)) * u[:, 2],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scipy.special.sph_harm_y(1, 1, theta, phi),
        -math.sqrt(3 / (8 * math.pi)) * (u[:, 0] + 1j * u[:, 1]),
        atol=1e-12,
    )
    col = 0
    for ell in range(4):
        for m in range(-ell, ell + 1):
            y = scipy.special.sph_harm_y(ell, abs(m), theta, phi)
            expected = y.real if m == 0 else math.sqrt(2) * (y.real if m > 0 else y.imag)
            np.testing.assert_allclose(ref[:, col], expected, atol=1e-12)
            col += 1


def test_u6_degree_0_gives_c0_and_dc_color() -> None:
    """U-6: degree 0 is the constant C₀, so the DC color is C₀·c₀₀ + 0.5: worked by hand for
    c₀₀ = (1, −1, 0) as (C₀ + 0.5, 0.5 − C₀, 0.5), through the basis and through sh0_to_rgb.
    Every channel stays non-negative, so E2's clamp at 0 doesn't apply; the clamp belongs to the
    render backend and is tested with it (M4)."""
    d = torch.from_numpy(_directions(50, 2))
    basis = sh_basis(d, 0)
    assert basis.shape == (d.shape[0], 1)
    torch.testing.assert_close(
        basis, torch.full_like(basis, 0.28209479177387814), rtol=0.0, atol=1e-6
    )
    c00 = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    want = torch.tensor([0.78209479177387814, 0.21790520822612186, 0.5], dtype=torch.float64)
    torch.testing.assert_close(
        basis[:, :1] * c00 + 0.5, want.expand(d.shape[0], 3), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(sh0_to_rgb(c00), want, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_u6_degrees_1_to_3_match_independent_real_sh(degree: int, dtype: torch.dtype) -> None:
    """U-6: degrees 1–3 against the independent evaluation, within 1e-6; directions need not be
    unit vectors (Q-14)."""
    d = _directions(500, 3 + degree)
    expected = torch.from_numpy(real_sh(d, degree)).to(dtype)
    got = sh_basis(torch.from_numpy(d).to(dtype), degree)
    assert got.shape == (d.shape[0], (degree + 1) ** 2)
    assert got.dtype == dtype
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-6)


def test_sh_basis_keeps_batch_dimensions() -> None:
    """Q-14: [..., 3] → [..., (L+1)²]."""
    d = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert sh_basis(d, 3).shape == (4, 5, 16)


@pytest.mark.parametrize("degree", [-1, 4, 2.0, True])
def test_sh_basis_rejects_degrees_outside_0_to_3(degree: object) -> None:
    """D-3: the SH degree is one of 0, 1, 2, 3."""
    with pytest.raises(ValueError, match="degree"):
        sh_basis(torch.ones(1, 3), degree)  # type: ignore[arg-type]


def test_rgb_dc_round_trip() -> None:
    """§1.1, D-4: sh0 = (rgb − 0.5)/C₀, worked by hand (0.5 ↦ 0, 1 ↦ √π, 0 ↦ −√π,
    0.5 + C₀ ↦ 1), and back."""
    sqrt_pi = 1.7724538509055159
    rgb = torch.tensor([0.5, 1.0, 0.0, 0.5 + 0.28209479177387814], dtype=torch.float64)
    want = torch.tensor([0.0, sqrt_pi, -sqrt_pi, 1.0], dtype=torch.float64)
    torch.testing.assert_close(rgb_to_sh0(rgb), want, rtol=0.0, atol=1e-12)
    rand = torch.rand(100, 1, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    torch.testing.assert_close(sh0_to_rgb(rgb_to_sh0(rand)), rand, rtol=0.0, atol=1e-12)
    assert torch.equal(rgb_to_sh0(torch.full((3,), 0.5)), torch.zeros(3))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_basis_matches_gsplat_spherical_harmonics(degree: int) -> None:
    """E2: gsplat's constants and band ordering. `sh_basis` contracted with random coefficients
    equals gsplat 1.5.3's `spherical_harmonics`, which returns the plain sum (source log #6)."""
    from gsplat import spherical_harmonics

    gen = torch.Generator(device="cuda").manual_seed(degree)
    n, k = 1000, (degree + 1) ** 2
    dirs = torch.randn(n, 3, generator=gen, device="cuda")
    coeffs = torch.randn(n, k, 3, generator=gen, device="cuda")
    expected = spherical_harmonics(degree, dirs, coeffs)
    got = (sh_basis(dirs, degree)[..., None] * coeffs).sum(dim=-2)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


# sh_to_color: E2's colour from the basis (Q-204).


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_to_color_is_e2_against_independent_real_sh(degree: int) -> None:
    """Q-204: sh_to_color(sh, d) = max(0, Σ c_lm Y_lm(d/|d|) + 0.5) per channel (E2), with the
    degree read from the number of coefficients; checked against the independent real-SH
    evaluation in float64."""
    dirs = _directions(40, seed=7 + degree)
    rng = np.random.default_rng(degree)
    coeffs = rng.normal(scale=0.6, size=(len(dirs), (degree + 1) ** 2, 3))
    unit = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    expected = np.maximum((coeffs * real_sh(unit, degree)[..., None]).sum(axis=-2) + 0.5, 0.0)
    got = sh_to_color(torch.from_numpy(coeffs), torch.from_numpy(dirs))
    torch.testing.assert_close(got, torch.from_numpy(expected), rtol=1e-12, atol=1e-12)


def test_sh_to_color_clamps_each_channel_at_zero() -> None:
    """E2: the max is per channel. A DC coefficient of −1/C0 on one channel gives
    C0·(−1/C0) + 0.5 = −0.5, clamped to 0, while the other channels keep 0.5 + C0·c."""
    sh = torch.tensor([[[-1.0 / C0, 0.0, 0.4]]], dtype=torch.float64)
    got = sh_to_color(sh, torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        got, torch.tensor([[0.0, 0.5, 0.5 + C0 * 0.4]], dtype=torch.float64), rtol=0, atol=1e-15
    )


def test_sh_to_color_rejects_a_coefficient_count_that_is_not_a_square() -> None:
    """Q-204: the coefficient axis holds (L + 1)² entries for L in 0 to 3."""
    with pytest.raises(ValueError, match="coefficients"):
        sh_to_color(torch.zeros(2, 5, 3), torch.ones(2, 3))
