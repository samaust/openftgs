# SPDX-License-Identifier: Apache-2.0
"""The real spherical-harmonics basis (reference implementation) and the RGB to DC conversion.

Spec: E2. Band: Core model (§2.1).
"""

import torch
from torch import Tensor

__all__ = ["C0", "rgb_to_sh0", "sh0_to_rgb", "sh_basis"]

C0: float = 0.28209479177387814  # §1.1: the l = 0 real SH constant, 1/(2√π)

# Normalization constants of the real SH, from the published Cartesian table (√((2l+1)/(4π)·…)).
_C1 = 0.4886025119029199  # √(3/(4π))
_C2 = (1.0925484305920792, 0.31539156525252005, 0.5462742152960396)
_C3 = (
    0.5900435899266435,
    2.890611442640554,
    0.4570457994644658,
    0.3731763325901154,
    1.445305721320277,
)


def sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Real SH basis Y_lm(d) for l ≤ ``degree``: ``[..., 3]`` → ``[..., (degree + 1)²]``.

    Ordered by band l and, within a band, m = −l … l, with gsplat 1.5.3's signs (the
    Condon–Shortley phase is kept, so l = 1 is C₁·(−y, z, −x)). Directions are normalized here,
    as E2's d is a unit vector (Q-14).
    """
    # Q-14: the degree is checked here as well as in TrainConfig (D-3).
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= 3:
        raise ValueError(f"SH degree must be 0, 1, 2 or 3, got {degree!r}")
    d = torch.nn.functional.normalize(dirs, dim=-1)
    x, y, z = d.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        out += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            _C2[0] * x * y,
            -_C2[0] * y * z,
            _C2[1] * (3.0 * zz - 1.0),
            -_C2[0] * x * z,
            _C2[2] * (xx - yy),
        ]
    if degree >= 3:
        out += [
            -_C3[0] * y * (3.0 * xx - yy),
            _C3[1] * x * y * z,
            -_C3[2] * y * (5.0 * zz - 1.0),
            _C3[3] * z * (5.0 * zz - 3.0),
            -_C3[2] * x * (5.0 * zz - 1.0),
            _C3[4] * z * (xx - yy),
            -_C3[0] * x * (xx - 3.0 * yy),
        ]
    return torch.stack(out, dim=-1)


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    """SH DC band from a color: (rgb − 0.5)/C₀ (§1.1, D-4)."""
    return (rgb - 0.5) / C0


def sh0_to_rgb(sh0: Tensor) -> Tensor:
    """Color of the SH DC band alone: C₀·sh0 + 0.5 (E2 at degree 0, D-4)."""
    return C0 * sh0 + 0.5
