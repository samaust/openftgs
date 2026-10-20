# SPDX-License-Identifier: Apache-2.0
"""Seeded FTGS-IF models for the harness's cases (§2.2 ftgs-if; Q-209)."""

import numpy as np

from ftgs_harness.formats import Exchange

__all__ = ["random_exchange", "single_primitive", "unit_quaternions"]


def unit_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random unit quaternions (w, x, y, z) with w ≥ 0 (§2.2), float32."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q.astype(np.float32)


def random_exchange(
    rng: np.random.Generator,
    n: int,
    *,
    sh_degree: int = 3,
    tau0: float = 0.0,
    tau1: float = 1.9,
    spread: float = 0.6,
    log_scale: tuple[float, float] = (-4.0, -2.5),
    log_duration: tuple[float, float] = (-2.5, 0.0),
    speed: float = 0.15,
    opacity: tuple[float, float] = (0.05, 0.95),
    saturated: bool = False,
    sh_scale: float = 0.3,
) -> Exchange:
    """A random model with N rows. Durations are exp(U(log_duration)) × T_τ, velocities about
    `speed`·spread per T_τ; with τ₀ = τ₁ every time centre is τ₀ and every velocity 0 (D-71).
    With `saturated`, a third of the opacities sit at the file's clamp, 1e-7 or 1 − 1e-7."""
    span = tau1 - tau0
    unit = span if span > 0 else 1.0  # D-70: a time unit of 1 with a single timestamp
    sigma = rng.uniform(*opacity, size=n)
    if saturated:
        k = n // 3
        sigma[:k] = np.where(rng.random(k) < 0.5, 1e-7, 1.0 - 1e-7)
    arrays = {
        "position": (rng.normal(size=(n, 3)) * spread).astype(np.float32),
        "time_center": rng.uniform(tau0, tau1, size=n) if span > 0 else np.full(n, tau0),
        "duration": np.exp(rng.uniform(*log_duration, size=n)) * unit,
        "velocity": (
            (rng.normal(size=(n, 3)) * speed * spread / unit).astype(np.float32)
            if span > 0
            else np.zeros((n, 3), np.float32)
        ),
        "scale": np.exp(rng.uniform(*log_scale, size=(n, 3))).astype(np.float32),
        "rotation": unit_quaternions(rng, n),
        "opacity": sigma.astype(np.float32),
        "sh": (rng.normal(size=(n, (sh_degree + 1) ** 2, 3)) * sh_scale).astype(np.float32),
    }
    return Exchange(arrays, sh_degree, tau0, tau1)


def single_primitive(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    time_center: float = 0.95,
    duration: float = 10.0,
    scale: tuple[float, float, float] = (0.05, 0.05, 0.05),
    opacity: float = 0.8,
    color_sh: tuple[float, float, float] = (0.5, 0.2, -0.3),
    sh_degree: int = 0,
    tau0: float = 0.0,
    tau1: float = 1.9,
) -> Exchange:
    """A one-row model for a probe (§3.3 L1), axis-aligned."""
    sh = np.zeros((1, (sh_degree + 1) ** 2, 3), np.float32)
    sh[0, 0] = color_sh
    arrays = {
        "position": np.array([position], np.float32),
        "time_center": np.array([time_center]),
        "duration": np.array([duration]),
        "velocity": np.array([velocity], np.float32),
        "scale": np.array([scale], np.float32),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        "opacity": np.array([opacity], np.float32),
        "sh": sh,
    }
    return Exchange(arrays, sh_degree, tau0, tau1)
