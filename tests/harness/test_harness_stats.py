# SPDX-License-Identifier: Apache-2.0
"""The harness's statistics (§3.3 L2 and L3; Q-209, Q-210) on constructed samples."""

import numpy as np
import pytest

from ftgs_harness.stats import (
    chi2_goodness_of_fit,
    chi2_homogeneity,
    holm,
    k_for_power,
    k_needed_for_flags,
    ks_distance,
    permutation_p,
    smallest_permutation_p,
    tost_p,
    w1_distance,
    welch_p,
)


def test_welch_ks_and_w1_on_worked_samples() -> None:
    """Worked by hand: equal samples give Welch p = 1; disjoint samples give a KS distance of 1;
    W1 between two samples one shifted by 3 is 3; KS between (0, 1, 2, 3) and (2, 3, 4, 5) is
    0.5, where half of the first lies below every value of the second."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert welch_p(a, a) == pytest.approx(1.0)
    assert welch_p(a, a + 100.0) < 1e-6
    assert ks_distance(a, a + 10.0) == 1.0
    assert w1_distance(a, a + 3.0) == pytest.approx(3.0)
    assert ks_distance([0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.5)


def test_tost_declares_equivalence_only_within_the_margin() -> None:
    """TOST with margin 0.25: two samples 0.01 apart with spread 0.05 are equivalent
    (p < 0.05); 0.5 apart, they are not; with too few or too noisy seeds, it can't tell."""
    g = np.random.default_rng(1)
    a = 30.0 + g.normal(scale=0.05, size=5)
    assert tost_p(a, a + 0.01, 0.25) < 0.05
    assert tost_p(a, a + 0.5, 0.25) > 0.5
    noisy = 30.0 + g.normal(scale=1.0, size=5)
    assert tost_p(noisy, noisy + 0.01, 0.25) > 0.05
    # A margin at the observed difference leaves one one-sided test at p = 0.5.
    x, y = np.array([1.0, 1.2, 0.9, 1.1]), np.array([1.05, 1.0, 1.15, 1.0])
    d = abs(float(x.mean() - y.mean()))
    assert tost_p(x, y, d) == pytest.approx(0.5)


def test_holm_steps_down() -> None:
    """Holm: sorted 0.01, 0.03, 0.04 of three become 0.03, 0.06, 0.06, back in input order."""
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0]


def test_k_for_power_reaches_80_percent_power() -> None:
    """Q-210: with the K returned, TOST at α = 0.05 declares two identical normal samples of
    spread 0.1 equivalent at margin 0.25 in at least 80 % of 4,000 seeded simulations, and with
    one seed fewer in fewer."""
    k = k_for_power(0.1, 0.25, alpha=0.05)
    g = np.random.default_rng(2)

    def power(n: int) -> float:
        hits = sum(
            tost_p(g.normal(scale=0.1, size=n), g.normal(scale=0.1, size=n), 0.25) < 0.05
            for _ in range(4000)
        )
        return hits / 4000

    assert power(k) >= 0.8
    assert k >= 2 and k_for_power(0.2, 0.25) > k


def test_permutation_flags_a_shifted_distribution_and_not_an_equal_one() -> None:
    """§3.3 L3 step 5: with K = 5 a side, B shifted far from A gives the smallest p, 1/252;
    B drawn as A′ is gives a large p."""
    g = np.random.default_rng(3)
    a = [g.normal(size=200) for _ in range(5)]
    a2 = [g.normal(size=200) for _ in range(5)]
    far = [g.normal(3.0, size=200) for _ in range(5)]
    same = [g.normal(size=200) for _ in range(5)]
    _, p = permutation_p(a, far, a2, ks_distance)
    assert p == pytest.approx(1 / 252)
    _, p = permutation_p(a, same, a2, w1_distance)
    assert p > 0.05


def test_flag_reachability() -> None:
    """Q-210: at K = 5 the smallest p is 1/252, above Holm's first threshold 0.01/14; K = 7 is
    the first that reaches it."""
    assert smallest_permutation_p(5) == pytest.approx(1 / 252)
    assert smallest_permutation_p(5) > 0.01 / 14
    assert k_needed_for_flags(14, 0.01) == 7


def test_chi2_tests() -> None:
    """χ² against the expected draws and between two count vectors; a count in a zero-weight
    cell fails outright."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.random.default_rng(4)
    counts = g.multinomial(10_000, weights / weights.sum())
    assert chi2_goodness_of_fit(counts, weights) > 0.001
    biased = counts.copy()
    biased[0] += biased[3] // 2
    biased[3] -= biased[3] // 2
    assert chi2_goodness_of_fit(biased, weights) < 1e-6
    assert chi2_goodness_of_fit([1, 5], [0.0, 1.0]) == 0.0
    other = g.multinomial(10_000, weights / weights.sum())
    assert chi2_homogeneity(counts, other) > 0.001
    assert chi2_homogeneity(counts, biased) < 1e-6
