# SPDX-License-Identifier: Apache-2.0
"""The adapters' `info` replies: their schema (D-73; Q-205) and the L1 tier chosen from both
(§3.3 L1; Q-206)."""

from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "QUERY_KINDS",
    "RENDER_CONVENTIONS",
    "TIERS",
    "Tier",
    "choose_tier",
    "info_problems",
]

# §3.3 L2: the thirteen query kinds.
QUERY_KINDS = (
    "motion",
    "temporal_opacity",
    "effective_opacity",
    "sh_color",
    "losses",
    "lr_schedule",
    "relocation_math",
    "velocity_init",
    "relocation_sampling",
    "grad_stats",
    "relocation_score",
    "spatial_scale",
    "growth_decisions",
)

# §2.2 ftgs-if `render_conventions` with D-73's entries, under Q-35's keys: an `info` reply
# reports at least these.
RENDER_CONVENTIONS = (
    "alpha_max",
    "eps2d",
    "near",
    "far",
    "alpha_skip",
    "transmittance_stop",
    "pixel_center_offset",
    "background",
    "jacobian_clamp_margin",
    "screen_extent_rule",
    "screen_extent_max",
    "rasterize_mode",
)

Tier = Literal["same", "different"]


# §3.3 L1's tier table.
TIERS: dict[Tier, dict[str, float]] = {
    "same": {"psnr_min_db": 60.0, "max_abs_max": 1e-3},
    "different": {"psnr_min_db": 45.0, "share_above_max": 0.005},
}


def info_problems(info: Any) -> list[str]:
    """What an `info` reply lacks against Q-205's layout and D-73's entries; empty when none."""
    if not isinstance(info, Mapping):
        return ["the reply is not a JSON object"]
    problems = []
    for key in ("name", "version"):
        if not isinstance(info.get(key), str):
            problems.append(f"`{key}` is not a string")
    kinds = info.get("query_kinds")
    if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
        problems.append("`query_kinds` is not a list of strings")
    rasterizer = info.get("rasterizer")
    if not (
        isinstance(rasterizer, Mapping)
        and isinstance(rasterizer.get("name"), str)
        and isinstance(rasterizer.get("version"), str)
    ):
        problems.append("`rasterizer` is not an object with string `name` and `version`")
    conventions = info.get("render_conventions")
    if not isinstance(conventions, Mapping):
        problems.append("`render_conventions` is not an object")
    else:
        missing = [k for k in RENDER_CONVENTIONS if k not in conventions]
        if missing:
            problems.append("`render_conventions` lacks " + ", ".join(missing))
    return problems


def choose_tier(info_a: Any, info_b: Any) -> Tier:
    """Q-206: "same" only when both replies name the same rasterizer and every
    `render_conventions` entry of either matches the other's; "different" otherwise, also when
    a reply is unusable."""
    if info_problems(info_a) or info_problems(info_b):
        return "different"
    if dict(info_a["rasterizer"]) != dict(info_b["rasterizer"]):
        return "different"
    if dict(info_a["render_conventions"]) != dict(info_b["render_conventions"]):
        return "different"
    return "same"
