# SPDX-License-Identifier: Apache-2.0
"""L1's per-item measures and the tier's pass rule (§3.3 L1; Q-206, Q-211).

Renders are compared as stored, unclamped (§2.2), PSNR with a peak of 1: clamping would hide a
difference above 1."""

import math
from dataclasses import dataclass

import numpy as np

from ftgs_harness.info import TIERS, Tier

__all__ = ["STEP", "ItemMeasures", "measure", "verdict"]

STEP = 1.0 / 255.0  # §3.3 L1: the share of pixels with |Δ| > 1/255


@dataclass(frozen=True)
class ItemMeasures:
    psnr_db: float  # +inf for identical renders
    max_abs: float
    mean_abs: float
    share_above: float  # pixels with |Δ| > 1/255 in any channel
    alpha_max_abs: float
    alpha_mean_abs: float
    finite: bool

    def to_json(self) -> dict[str, float | bool]:
        return {
            "psnr_db": self.psnr_db,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "share_above_1_255": self.share_above,
            "alpha_max_abs": self.alpha_max_abs,
            "alpha_mean_abs": self.alpha_mean_abs,
        }


def measure(
    rgb_a: np.ndarray, rgb_b: np.ndarray, alpha_a: np.ndarray, alpha_b: np.ndarray
) -> ItemMeasures:
    """§3.3 L1: PSNR(A, B), max and mean |Δ|, the share of pixels with |Δ| > 1/255 and the
    alpha difference, in float64."""
    a, b = rgb_a.astype(np.float64), rgb_b.astype(np.float64)
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    finite = finite and bool(np.all(np.isfinite(alpha_a)) and np.all(np.isfinite(alpha_b)))
    if not finite:
        nan = float("nan")
        return ItemMeasures(nan, nan, nan, nan, nan, nan, False)
    diff = np.abs(a - b)
    mse = float(np.mean(diff**2))
    alpha = np.abs(alpha_a.astype(np.float64) - alpha_b.astype(np.float64))
    return ItemMeasures(
        psnr_db=math.inf if mse == 0.0 else -10.0 * math.log10(mse),
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        share_above=float(np.mean(np.any(diff > STEP, axis=-1))),
        alpha_max_abs=float(alpha.max()),
        alpha_mean_abs=float(alpha.mean()),
        finite=True,
    )


def verdict(m: ItemMeasures, tier: Tier) -> tuple[bool, str, str, float]:
    """Whether the item passes the tier, and the check's metric, expected and observed: the
    criterion that failed first, or PSNR when both hold."""
    rule = TIERS[tier]
    if not m.finite:
        return False, "finite_render", "finite values", float("nan")
    if m.psnr_db < rule["psnr_min_db"]:
        return False, "psnr_A_vs_B_db", f">= {rule['psnr_min_db']:g}", m.psnr_db
    if tier == "same" and m.max_abs > rule["max_abs_max"]:
        return False, "max_abs_A_vs_B", f"<= {rule['max_abs_max']:g}", m.max_abs
    if tier == "different" and m.share_above > rule["share_above_max"]:
        return False, "share_above_1_255", f"<= {rule['share_above_max']:g}", m.share_above
    return True, "psnr_A_vs_B_db", f">= {rule['psnr_min_db']:g}", m.psnr_db
