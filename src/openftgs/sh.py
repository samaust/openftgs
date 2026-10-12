# SPDX-License-Identifier: Apache-2.0
"""The real spherical-harmonics basis (reference implementation) and the RGB to DC conversion.

Spec: E2. Band: Core model (§2.1).
"""

import torch
from torch import Tensor

__all__ = ["C0", "rgb_to_sh0", "sh0_to_rgb", "sh_basis"]  # Q-14

C0: float = 0.28209479177387814  # §1.1: the l = 0 real SH constant, 1/(2√π)

# Normalization constants of the real SH, from the standard Cartesian table (source log #9),
# named _C<l>_<|m|>. The m = −2 terms use 2·_C<l>_2: _C2_1 = 2·_C2_2, and 2·_C3_2 for l = 3.
_C1 = 0.4886025119029199  # √(3/(4π))
_C2_0 = 0.31539156525252005  # ¼√(5/π)
_C2_1 = 1.0925484305920792  # ½√(15/π)
_C2_2 = 0.5462742152960396  # ¼√(15/π)
_C3_0 = 0.3731763325901154  # ¼√(7/π)
_C3_1 = 0.4570457994644658  # ¼√(21/(2π))
_C3_2 = 1.445305721320277  # ¼√(105/π)
_C3_3 = 0.5900435899266435  # ¼√(35/(2π))


def sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Real SH basis Y_lm(d) for l ≤ ``degree``: ``[..., 3]`` → ``[..., (degree + 1)²]``.

    Ordered by band l and, within a band, m = −l … l, with gsplat 1.5.3's signs (the
    Condon–Shortley phase is kept, so l = 1 is C₁·(−y, z, −x)). Directions are normalized here,
    as E2's d is a unit vector (Q-14).
    """
    # Q-14: the degree is checked here as well as in TrainConfig (D-3); bools are rejected.
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= 3:
        raise ValueError(f"SH degree must be 0, 1, 2 or 3, got {degree!r}")
    d = torch.nn.functional.normalize(dirs, dim=-1)  # Q-14: directions normalized here
    x, y, z = d.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        out += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            _C2_1 * x * y,
            -_C2_1 * y * z,
            _C2_0 * (3.0 * zz - 1.0),
            -_C2_1 * x * z,
            _C2_2 * (xx - yy),
        ]
    if degree >= 3:
        out += [
            -_C3_3 * y * (3.0 * xx - yy),
            2.0 * _C3_2 * x * y * z,
            -_C3_1 * y * (5.0 * zz - 1.0),
            _C3_0 * z * (5.0 * zz - 3.0),
            -_C3_1 * x * (5.0 * zz - 1.0),
            _C3_2 * z * (xx - yy),
            -_C3_3 * x * (xx - 3.0 * yy),
        ]
    return torch.stack(out, dim=-1)


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    """SH DC band from a color: (rgb − 0.5)/C₀ (§1.1, D-4)."""
    return (rgb - 0.5) / C0


def sh0_to_rgb(sh0: Tensor) -> Tensor:
    """Color of the SH DC band alone: C₀·sh0 + 0.5 (E2 at degree 0, D-4)."""
    return C0 * sh0 + 0.5
