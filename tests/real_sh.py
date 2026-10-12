# SPDX-License-Identifier: Apache-2.0
"""An independent float64 evaluation of the real spherical harmonics, shared by the tests that need
one: the basis tests of U-6 (test_sh.py) and the E6 oracle (test_render.py).

It is written from the general definition of the real SH (normalization from factorials,
associated Legendre functions with the Condon–Shortley phase by the standard recurrence, cos/sin
combinations; source log #9), not from the Cartesian table `openftgs.sh` uses.
"""

import math

import numpy as np


def _legendre_cs(l_max: int, x: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """P_l^m(x) for 0 ≤ m ≤ l ≤ l_max, with the Condon–Shortley phase (−1)^m."""
    p: dict[tuple[int, int], np.ndarray] = {(0, 0): np.ones_like(x)}
    s = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for m in range(1, l_max + 1):
        p[(m, m)] = -(2 * m - 1) * s * p[(m - 1, m - 1)]
    for m in range(0, l_max):
        p[(m + 1, m)] = (2 * m + 1) * x * p[(m, m)]
    for m in range(0, l_max + 1):
        for ell in range(m + 2, l_max + 1):
            p[(ell, m)] = (
                (2 * ell - 1) * x * p[(ell - 1, m)] - (ell + m - 1) * p[(ell - 2, m)]
            ) / (ell - m)
    return p


def real_sh(dirs: np.ndarray, l_max: int) -> np.ndarray:
    """Real SH keeping the Condon–Shortley phase, ordered l = 0 … l_max, m = −l … l (float64)."""
    d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    phi = np.arctan2(y, x)
    p = _legendre_cs(l_max, z)
    out = []
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            a = abs(m)
            k = math.sqrt(
                (2 * ell + 1) / (4 * math.pi) * math.factorial(ell - a) / math.factorial(ell + a)
            )
            if m == 0:
                out.append(k * p[(ell, 0)])
            elif m > 0:
                out.append(math.sqrt(2) * k * np.cos(a * phi) * p[(ell, a)])
            else:
                out.append(math.sqrt(2) * k * np.sin(a * phi) * p[(ell, a)])
    return np.stack(out, axis=-1)
