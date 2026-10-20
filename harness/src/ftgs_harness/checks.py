# SPDX-License-Identifier: Apache-2.0
"""The check record of the report (§3.3 "Report format") and the tolerance readings (Q-208).

A check has exactly the fields of §3.3's example, plus `pattern` (the symptom's word from the
fixed vocabulary) and `note`, both written only when set (Q-211).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "LEVELS",
    "PATTERNS",
    "STATUSES",
    "Check",
    "Status",
    "close_absolute",
    "close_relative",
    "json_value",
    "level_summary",
]

LEVELS = ("L0", "L1", "L2", "L3")
Status = Literal["pass", "fail", "inconclusive", "not_comparable"]
STATUSES: tuple[Status, ...] = ("pass", "fail", "inconclusive", "not_comparable")  # Q-211

# §3.3 symptom rule 4: the fixed pattern vocabulary.
PATTERNS = (
    "offset",
    "scale change",
    "blur",
    "missing primitive",
    "extra primitive",
    "color shift",
    "view-dependent color shift",
    "temporal shift",
    "temporal widening",
    "temporal narrowing",
    "opacity saturation",
    "border band",
    "NaN/Inf",
    "crash",
    "schema violation",
)


def json_value(value: Any) -> Any:
    """`value` with every non-finite float replaced by None, recursively (Q-63, Q-211), and
    NumPy scalars and arrays turned into Python values."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(v) for v in value]
    return value


@dataclass
class Check:
    """One check of the report (§3.3). `observed` may be any JSON value; a non-finite number is
    written as null, with `note` saying what it was (Q-211)."""

    id: str
    level: str
    status: Status
    spec_refs: list[str]
    inputs: dict[str, Any]
    metric: str
    expected: str
    observed: Any
    symptom: str = ""
    localization: dict[str, Any] = field(default_factory=dict)
    repro: str = ""
    artifacts: list[str] = field(default_factory=list)
    pattern: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.pattern is not None and self.pattern not in PATTERNS:
            raise ValueError(f"pattern {self.pattern!r} is not in §3.3's vocabulary")
        if isinstance(self.observed, float) and not math.isfinite(self.observed):
            self.note = self.note or str(self.observed)  # "inf", "-inf" or "nan"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "level": self.level,
            "status": self.status,
            "spec_refs": list(self.spec_refs),
            "inputs": self.inputs,
            "metric": self.metric,
            "expected": self.expected,
            "observed": self.observed,
            "symptom": self.symptom,
            "localization": self.localization,
            "repro": self.repro,
            "artifacts": list(self.artifacts),
        }
        if self.pattern is not None:
            out["pattern"] = self.pattern
        if self.note is not None:
            out["note"] = self.note
        return dict(json_value(out))


def level_summary(statuses: Sequence[str]) -> str:
    """Q-211: `fail` if any check failed, otherwise `inconclusive` if any was, otherwise `pass`;
    `not_comparable` checks fail nothing."""
    if "fail" in statuses:
        return "fail"
    if "inconclusive" in statuses:
        return "inconclusive"
    return "pass"


def close_relative(a: npt.ArrayLike, b: npt.ArrayLike, tol: float) -> np.ndarray:
    """Q-208: |a − b| ≤ tol · max(|a|, |b|, 1e-3 · ‖b‖∞), elementwise. NaN never passes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    floor = 1e-3 * float(np.max(np.abs(b))) if b.size and np.all(np.isfinite(b)) else 0.0
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.abs(a - b) <= tol * scale)


def close_absolute(a: npt.ArrayLike, b: npt.ArrayLike, tol: float) -> np.ndarray:
    """Q-208: a bare tolerance is absolute. NaN never passes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.abs(a - b) <= tol)
