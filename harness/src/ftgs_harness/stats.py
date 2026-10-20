# SPDX-License-Identifier: Apache-2.0
"""The statistics of L2 and L3 (§3.3; Q-209, Q-210): Welch's t-test, TOST as two one-sided
Welch tests, Holm's step-down correction, the seeds needed for 80 % power, KS and
Wasserstein-1 distances, the exact seed-permutation test with its reachability, and χ² tests."""

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.stats

__all__ = [
    "chi2_goodness_of_fit",
    "chi2_homogeneity",
    "holm",
    "k_for_power",
    "k_needed_for_flags",
    "ks_distance",
    "permutation_p",
    "smallest_permutation_p",
    "tost_p",
    "w1_distance",
    "welch_p",
]


def welch_p(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Welch's two-sided t-test p-value; 1 when both samples are constant and equal."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if np.ptp(x) == 0 and np.ptp(y) == 0:
        return 1.0 if x.mean() == y.mean() else 0.0
    return float(scipy.stats.ttest_ind(x, y, equal_var=False).pvalue)


def _welch_se_df(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = math.sqrt(va + vb)
    if se == 0:
        return 0.0, float(x.size + y.size - 2)
    df = (va + vb) ** 2 / (va**2 / (x.size - 1) + vb**2 / (y.size - 1))
    return se, df


def tost_p(a: npt.ArrayLike, b: npt.ArrayLike, margin: float) -> float:
    """The TOST p-value for |mean(a) − mean(b)| < margin: the larger p of the two one-sided Welch
    tests against −margin and +margin."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(x.mean() - y.mean())
    se, df = _welch_se_df(x, y)
    if se == 0:
        return 0.0 if abs(d) < margin else 1.0
    p_low = float(scipy.stats.t.sf((d + margin) / se, df))
    p_high = float(scipy.stats.t.cdf((d - margin) / se, df))
    return max(p_low, p_high)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the given order: the i-th smallest p times
    (m − i + 1), made monotone, capped at 1."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def k_for_power(sd: float, margin: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """The seeds per implementation for TOST to declare equivalence with probability `power`
    when the true difference is 0: K = 2σ²(z₁₋α + z₁₋β/₂)²/δ² by the normal approximation, then
    refined with t quantiles at df = 2K − 2 (Q-210). At least 2."""
    if sd <= 0:
        return 2
    beta = 1.0 - power
    k = 2.0 * sd**2 * (scipy.stats.norm.ppf(1 - alpha) + scipy.stats.norm.ppf(1 - beta / 2)) ** 2
    k /= margin**2
    for _ in range(50):
        df = max(2.0 * math.ceil(max(k, 2.0)) - 2.0, 1.0)
        t = scipy.stats.t.ppf(1 - alpha, df) + scipy.stats.t.ppf(1 - beta / 2, df)
        refined = 2.0 * sd**2 * t**2 / margin**2
        if abs(refined - k) < 1e-9:
            break
        k = refined
    return max(2, int(math.ceil(k)))


def ks_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    return float(scipy.stats.ks_2samp(np.asarray(x), np.asarray(y)).statistic)


def w1_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    return float(scipy.stats.wasserstein_distance(np.asarray(x), np.asarray(y)))


def smallest_permutation_p(k: int) -> float:
    """The smallest p the exact seed permutation can give with K runs a side: 1/C(2K, K)."""
    return 1.0 / math.comb(2 * k, k)


def k_needed_for_flags(tests: int, alpha: float) -> int:
    """The smallest K at which Holm's first threshold α/m is reachable (Q-210)."""
    k = 2
    while smallest_permutation_p(k) > alpha / tests:
        k += 1
    return k


def permutation_p(
    runs_a: Sequence[np.ndarray],
    runs_b: Sequence[np.ndarray],
    runs_a2: Sequence[np.ndarray],
    distance: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    """The exact seed-permutation test of §3.3 L3 step 5 (Q-210): the statistic
    d(A, B) − d(A, A′) on pooled samples, against its value over every split of the runs of B
    and A′ into two groups of K (the observed split included). Returns (statistic, p)."""
    pooled_a = np.concatenate([np.ravel(r) for r in runs_a])
    others = list(runs_b) + list(runs_a2)
    k = len(runs_b)

    def statistic(first: Sequence[int]) -> float:
        group = set(first)
        b = np.concatenate([np.ravel(others[i]) for i in range(len(others)) if i in group])
        a2 = np.concatenate([np.ravel(others[i]) for i in range(len(others)) if i not in group])
        return distance(pooled_a, b) - distance(pooled_a, a2)

    observed = statistic(range(k))
    splits = list(itertools.combinations(range(len(others)), k))
    count = sum(statistic(split) >= observed - 1e-12 for split in splits)
    return observed, count / len(splits)


def chi2_goodness_of_fit(observed: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """χ² p-value of counts against probabilities ∝ `weights` (cells of weight 0 left out; a
    count in such a cell gives p = 0)."""
    obs, w = np.asarray(observed, np.float64), np.asarray(weights, np.float64)
    if np.any(obs[w <= 0] > 0):
        return 0.0
    keep = w > 0
    expected = w[keep] / w[keep].sum() * obs[keep].sum()
    return float(scipy.stats.chisquare(obs[keep], expected).pvalue)


def chi2_homogeneity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """χ² p-value that two count vectors come from one distribution (cells empty in both left
    out)."""
    table = np.stack([np.asarray(a, np.float64), np.asarray(b, np.float64)])
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(scipy.stats.chi2_contingency(table).pvalue)
