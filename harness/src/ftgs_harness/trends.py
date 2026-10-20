# SPDX-License-Identifier: Apache-2.0
"""Trends of a discrepancy against the swept variable (§3.3 "Report format": "the harness fits
constant, linear and quadratic models to the discrepancy and reports the best fit with its
coefficients"; Q-211: the best is the lowest BIC)."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Trend", "fit_trend"]

_NAMES = ("constant", "linear", "quadratic")


@dataclass(frozen=True)
class Trend:
    model: str  # "constant", "linear" or "quadratic"
    coefficients: tuple[float, ...]  # c0, c1, c2: y ≈ c0 + c1·x + c2·x²

    def describe(self, variable: str, unit: str) -> str:
        """For example "linear in |τ − time_center| (7.3 px per s, 0.0 px at 0)"."""
        c = self.coefficients
        if self.model == "constant":
            return f"constant over {variable} ({c[0]:.3g} {unit})"
        if self.model == "linear":
            return f"linear in {variable} ({c[1]:.3g} {unit} per unit, {c[0]:.3g} {unit} at 0)"
        return (
            f"quadratic in {variable} ({c[2]:.3g} {unit} per unit², {c[1]:.3g} {unit} per unit, "
            f"{c[0]:.3g} {unit} at 0)"
        )


def fit_trend(x: npt.ArrayLike, y: npt.ArrayLike) -> Trend:
    """Least-squares fits of degree 0, 1 and 2, chosen by the Bayesian information criterion
    n·ln(RSS/n) + k·ln(n) (k the number of coefficients). Residuals within 0.1 % of the data's
    largest magnitude count as an exact fit, so a negligible slope or curvature doesn't win;
    the lower degree wins a tie (Q-211)."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    n = int(xs.size)
    best: tuple[float, int, np.ndarray] | None = None
    scale = max(float(np.abs(ys).max()) if n else 0.0, 1e-300)
    for degree in range(min(3, max(n, 1))):
        if n <= degree:
            break
        coeffs = np.polynomial.polynomial.polyfit(xs, ys, degree)
        rss = float(((np.polynomial.polynomial.polyval(xs, coeffs) - ys) ** 2).sum())
        floor = (1e-3 * scale) ** 2 * n  # a fit within 0.1 % of the data is exact
        bic = n * math.log(max(rss, floor) / n) + (degree + 1) * math.log(max(n, 2))
        if best is None or bic < best[0] - 1e-9:
            best = (bic, degree, coeffs)
    if best is None:
        return Trend("constant", (float("nan"),))
    _, degree, coeffs = best
    return Trend(_NAMES[degree], tuple(float(c) for c in coeffs))
