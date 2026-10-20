# SPDX-License-Identifier: Apache-2.0
"""Symptoms of a failed render comparison, in input and output terms only (§3.3 "Rules for the
`symptom` field"; Q-211): measurements of B's render against A's, a pattern from the fixed
vocabulary, and a sentence with numbers, units and the controlled variable.

Measurements (Q-211): the alpha-weighted centroid (offset, px), the alpha-weighted spread
(scale change), the alpha mass (missing or extra primitive), the mean colour under the alpha
mask (colour shift), the share of |Δ| near the image border (border band), the alpha where A is
near opaque (opacity saturation), the alpha's gradient energy (blur), and non-finite values.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PairMeasures", "classify", "describe", "measure_pair"]

_BORDER_PX = 3  # the width of the border band


@dataclass(frozen=True)
class PairMeasures:
    """B's render against A's, for one item."""

    nonfinite: int  # non-finite values in B's render
    offset: tuple[float, float]  # B's alpha centroid minus A's, px (x right, y down)
    mass_ratio: float  # B's alpha mass over A's
    spread_ratio: float  # B's alpha spread over A's
    color_diff: tuple[float, float, float]  # B − A, mean colour under the alpha mask
    border_share: float  # share of Σ|Δ| within the border band
    saturation_diff: float  # max |Δα| where A's α > 0.9
    sharpness_ratio: float  # B's alpha gradient energy over A's
    alpha_diff: float  # max |Δα|

    @property
    def offset_px(self) -> float:
        return math.hypot(*self.offset)


def _moments(alpha: np.ndarray) -> tuple[float, np.ndarray, float]:
    h, w = alpha.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs, ys = xs + 0.5, ys + 0.5  # pixel centres (§1.4)
    mass = float(alpha.sum())
    if mass <= 0.0:
        return 0.0, np.array([w / 2.0, h / 2.0]), 0.0
    cx, cy = float((alpha * xs).sum() / mass), float((alpha * ys).sum() / mass)
    spread = float(np.sqrt((alpha * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / mass))
    return mass, np.array([cx, cy]), spread


def _gradient_energy(alpha: np.ndarray) -> float:
    gx = np.diff(alpha, axis=1)
    gy = np.diff(alpha, axis=0)
    return float((gx**2).sum() + (gy**2).sum())


def measure_pair(
    rgb_a: np.ndarray, rgb_b: np.ndarray, alpha_a: np.ndarray, alpha_b: np.ndarray
) -> PairMeasures:
    """The measurements of B's render against A's (float64; `alpha_*` are [H, W, 1])."""
    nonfinite = int((~np.isfinite(rgb_b)).sum() + (~np.isfinite(alpha_b)).sum())
    a, b = np.nan_to_num(rgb_a.astype(np.float64)), np.nan_to_num(rgb_b.astype(np.float64))
    aa = np.nan_to_num(alpha_a[..., 0].astype(np.float64))
    ab = np.nan_to_num(alpha_b[..., 0].astype(np.float64))
    mass_a, center_a, spread_a = _moments(aa)
    mass_b, center_b, spread_b = _moments(ab)
    weight = np.maximum(aa, ab)
    total_weight = float(weight.sum())
    if total_weight > 0.0:
        color = ((b - a) * weight[..., None]).sum(axis=(0, 1)) / total_weight
    else:
        color = (b - a).mean(axis=(0, 1))
    diff = np.abs(b - a).sum(axis=-1)
    band = np.zeros(diff.shape, bool)
    band[:_BORDER_PX], band[-_BORDER_PX:] = True, True
    band[:, :_BORDER_PX], band[:, -_BORDER_PX:] = True, True
    total = float(diff.sum())
    near_opaque = aa > 0.9
    energy_a, energy_b = _gradient_energy(aa), _gradient_energy(ab)
    return PairMeasures(
        nonfinite=nonfinite,
        offset=(float(center_b[0] - center_a[0]), float(center_b[1] - center_a[1])),
        mass_ratio=mass_b / mass_a if mass_a > 0 else (math.inf if mass_b > 0 else 1.0),
        spread_ratio=spread_b / spread_a if spread_a > 0 else 1.0,
        color_diff=(float(color[0]), float(color[1]), float(color[2])),
        border_share=float(diff[band].sum() / total) if total > 0 else 0.0,
        saturation_diff=float(np.abs(ab - aa)[near_opaque].max()) if near_opaque.any() else 0.0,
        sharpness_ratio=energy_b / energy_a if energy_a > 0 else 1.0,
        alpha_diff=float(np.abs(ab - aa).max()),
    )


def classify(m: PairMeasures, *, view_dependent: bool = False) -> str:
    """The pattern of §3.3's vocabulary that best names the difference (Q-211): checked in this
    order, the first that applies."""
    if m.nonfinite:
        return "NaN/Inf"
    if m.mass_ratio < 0.8:
        return "missing primitive"
    if m.mass_ratio > 1.25:
        return "extra primitive"
    if m.offset_px > 0.2:
        return "offset"
    if abs(m.spread_ratio - 1.0) > 0.03:
        return "scale change"
    if m.border_share > 0.8:
        return "border band"
    if m.saturation_diff > 0.01:
        return "opacity saturation"
    if m.sharpness_ratio < 0.85:
        return "blur"
    return "view-dependent color shift" if view_dependent else "color shift"


def _direction(dx: float, dy: float) -> str:
    horizontal = "right" if dx > 0 else "left"
    vertical = "below" if dy > 0 else "above"
    if abs(dx) >= 2 * abs(dy):
        return f"{horizontal} of"
    if abs(dy) >= 2 * abs(dx):
        return f"{vertical}"
    return f"{vertical} and {horizontal} of"


def describe(pattern: str, m: PairMeasures, where: str) -> str:
    """One sentence for the pattern, with numbers and units; `where` names the items, times and
    cameras ("in item r0007 (camera c03, τ = 0.83 s)")."""
    if pattern == "NaN/Inf":
        return f"B's render holds {m.nonfinite} non-finite values {where}."
    if pattern in ("missing primitive", "extra primitive"):
        change = (m.mass_ratio - 1.0) * 100.0
        what = "content A renders is absent from B" if change < 0 else "B shows content A lacks"
        return f"B's alpha coverage is {change:+.0f}% of A's {where}: {what}."
    if pattern == "offset":
        dx, dy = m.offset
        return (
            f"Both renders show the content, but B's copy sits {m.offset_px:.2f} px "
            f"{_direction(dx, dy)} A's {where} (Δx = {dx:+.2f} px, Δy = {dy:+.2f} px); alpha "
            f"coverage agrees within {abs(m.mass_ratio - 1.0) * 100:.1f}%."
        )
    if pattern == "scale change":
        return (
            f"B's footprint is {m.spread_ratio:.3f}× the spread of A's {where}, its centre "
            f"within {m.offset_px:.2f} px of A's."
        )
    if pattern == "border band":
        return (
            f"{m.border_share * 100:.0f}% of the difference lies within {_BORDER_PX} px of the "
            f"image border {where}."
        )
    if pattern == "opacity saturation":
        return f"Where A's alpha exceeds 0.9, B's differs by up to {m.saturation_diff:.4f} {where}."
    if pattern == "blur":
        return (
            f"B's render is softer than A's {where}: {m.sharpness_ratio:.2f}× A's gradient energy."
        )
    r, g, b = m.color_diff
    kind = "view-dependent colour" if pattern == "view-dependent color shift" else "colour"
    return (
        f"B's {kind} differs from A's by ({r:+.4f}, {g:+.4f}, {b:+.4f}) in RGB over the covered "
        f"pixels {where}; alpha agrees within {m.alpha_diff:.4f}."
    )
