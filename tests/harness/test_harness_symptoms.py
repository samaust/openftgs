# SPDX-License-Identifier: Apache-2.0
"""Symptoms and trends (§3.3 "Rules for the `symptom` field" and the trend paragraph; Q-211),
on constructed image pairs whose difference is known."""

import re

import numpy as np
import pytest

from ftgs_harness.checks import PATTERNS
from ftgs_harness.symptoms import classify, describe, measure_pair
from ftgs_harness.trends import fit_trend

H, W = 48, 64


def _blob(
    cx: float = 32.0,
    cy: float = 24.0,
    sigma: float = 3.0,
    color: tuple[float, ...] = (0.8, 0.4, 0.2),
    opacity: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:H, 0:W] + 0.5
    alpha = opacity * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    rgb = alpha[..., None] * np.array(color)
    return rgb.astype(np.float32), alpha[..., None].astype(np.float32)


def _pattern(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[str, str]:
    m = measure_pair(a[0], b[0], a[1], b[1])
    pattern = classify(m)
    return pattern, describe(pattern, m, "in item r0000 (camera c00, τ = 0.5 s)")


@pytest.mark.parametrize(
    ("b", "pattern", "number"),
    [
        (_blob(cx=33.5), "offset", r"1\.50 px right of A's"),
        (_blob(cy=22.0), "offset", r"2\.00 px above"),
        (_blob(sigma=3.3), "scale change", r"1\.100× the spread"),
        (_blob(opacity=0.3), "missing primitive", r"-62% of A's"),
        (_blob(color=(0.9, 0.4, 0.2)), "color shift", r"\(\+0\.\d+, \+0\.0000, \+0\.0000\)"),
    ],
)
def test_constructed_pairs_give_their_pattern_and_number(
    b: tuple[np.ndarray, np.ndarray], pattern: str, number: str
) -> None:
    got, sentence = _pattern(_blob(), b)
    assert got == pattern
    assert re.search(number, sentence), sentence


def test_an_extra_primitive_and_non_finite_values() -> None:
    a = _blob()
    rgb2, alpha2 = _blob(cx=12.0)
    both = (a[0] + rgb2, np.clip(a[1] + alpha2, 0, 1))
    assert _pattern(a, both)[0] == "extra primitive"
    nan = (a[0].copy(), a[1])
    nan[0][24, 32] = np.nan
    pattern, sentence = _pattern(a, nan)
    assert pattern == "NaN/Inf" and "3 non-finite values" in sentence


def test_a_border_band_and_a_blur() -> None:
    a = _blob(sigma=8.0)
    band = (a[0].copy(), a[1])
    band[0][:2, :, :] += 0.05
    assert _pattern(a, band)[0] == "border band"
    # A blur keeps the mass, centre and spread almost unchanged but softens the alpha.
    rgb, alpha = _blob()
    blurred_alpha = alpha.copy()
    blurred_alpha[::2, :] = 0.5 * (alpha[::2, :] + np.roll(alpha, 1, axis=0)[::2, :])
    noisy = alpha + 0.2 * alpha * ((np.indices((H, W)).sum(0) % 2)[..., None] - 0.5)
    pattern, _ = _pattern((rgb, noisy.astype(np.float32)), (rgb, alpha))
    assert pattern == "blur"


@pytest.mark.parametrize(
    ("degree", "coefficients"), [(0, (2.0,)), (1, (0.5, 3.0)), (2, (1.0, -2.0, 0.7))]
)
def test_bic_picks_the_true_degree(degree: int, coefficients: tuple[float, ...]) -> None:
    """§3.3: constant, linear and quadratic fits; the lowest BIC (Q-211) picks the degree that
    generated the data, under small noise."""
    x = np.linspace(-2.0, 3.0, 21)
    y = np.polynomial.polynomial.polyval(x, coefficients)
    y += np.random.default_rng(degree).normal(scale=1e-3, size=x.size)
    trend = fit_trend(x, y)
    assert trend.model == ("constant", "linear", "quadratic")[degree]
    np.testing.assert_allclose(trend.coefficients, coefficients, atol=5e-3)


def test_a_perfect_fit_keeps_the_lower_degree() -> None:
    assert fit_trend([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0]).model == "linear"
    assert fit_trend([0.0, 1.0, 2.0], [4.0, 4.0, 4.0]).model == "constant"


def test_a_discrepancy_of_zero_everywhere_is_a_constant_zero() -> None:
    """§3.3: a probe whose discrepancy is 0 on every item fits the constant 0."""
    trend = fit_trend([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert trend.model == "constant" and trend.coefficients == (0.0,)


_IDENTIFIER = re.compile(r"\b[a-z]+_[a-z_]+\(|\.py\b|\bdef\b|\bline \d+")


@pytest.mark.parametrize("pattern", [p for p in PATTERNS if p not in ("crash", "schema violation")])
def test_every_sentence_has_a_number_a_unit_and_no_code_identifier(pattern: str) -> None:
    """§3.3 symptom rules 2 and 3: at least one number with its unit and the controlled variable
    (here τ and the camera), and no function, file or line of either implementation."""
    m = measure_pair(*_blob()[:1], *_blob(cx=33.0)[:1], _blob()[1], _blob(cx=33.0)[1])
    sentence = describe(pattern, m, "in item r0003 (camera c01, τ = 0.75 s)")
    assert re.search(r"\d", sentence) and "τ = 0.75 s" in sentence
    assert not _IDENTIFIER.search(sentence), sentence
