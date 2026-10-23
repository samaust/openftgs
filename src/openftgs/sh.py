# SPDX-License-Identifier: Apache-2.0
"""The real spherical-harmonics basis (reference implementation) and the RGB to DC conversion.

Spec: E2. Band: Core model (§2.1).
"""

import math

import torch
from torch import Tensor

__all__ = ["C0", "rgb_to_sh0", "sh0_to_rgb", "sh_basis", "sh_to_color"]  # Q-14, Q-204

C0: float = 0.28209479177387814  # §1.1: the l = 0 real SH constant, 1/(2√π)
"""The l = 0 real spherical-harmonics constant, 1/(2√π) (§1.1, Q-14)."""
_COLOR_OFFSET = 0.5  # E2: the SH sum + 0.5, clamped at 0 per channel; D-4

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
    """Evaluate the real SH basis Y_lm(d) for every band l ≤ ``degree`` (E2, Q-14).

    The basis maps ``[..., 3]`` directions to ``[..., (degree + 1)²]`` values, ordered by band l
    and, within a band, m = −l … l, with gsplat 1.5.3's signs (the Condon–Shortley phase is kept,
    so l = 1 is C₁·(−y, z, −x)). Directions are normalized here, as E2's d is a unit vector. This
    is the reference implementation; the gsplat backend evaluates its own.

    Parameters
    ----------
    dirs
        Directions, ``[..., 3]`` floating point, of any nonzero length.
    degree
        The highest band L, 0, 1, 2 or 3 (D-3); a bool is rejected.

    Returns
    -------
    torch.Tensor
        The basis values, ``[..., (degree + 1)²]``, in the dtype and on the device of ``dirs``;
        entry 0 is :data:`C0`.

    Raises
    ------
    ValueError
        If ``degree`` is not an int among 0, 1, 2 and 3.

    Notes
    -----
    D-3: the degree is also checked by the training configuration. The normalization constants
    come from the standard Cartesian table of the real SH (source log #9).
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


def sh_to_color(sh: Tensor, dirs: Tensor) -> Tensor:
    """Evaluate E2's color, max(0, Σ c_lm Y_lm(d) + 0.5) per channel (E2).

    The degree L comes from the number of coefficients, and :func:`sh_basis` normalizes the
    directions. The sum is offset by 0.5 and clamped at 0 per channel, never above, so a color
    can exceed 1 (D-4).

    Parameters
    ----------
    sh
        SH coefficients, ``[..., (L + 1)², 3]`` floating point, L from 0 to 3: the DC band
        ``sh0`` followed by the higher bands ``shN`` of §1.1.
    dirs
        Viewing directions, ``[..., 3]`` with the same leading dimensions, of any nonzero length;
        E2's d runs from the camera center to the moved mean.

    Returns
    -------
    torch.Tensor
        The colors, ``[..., 3]``, each channel ≥ 0.

    Raises
    ------
    ValueError
        If the number of coefficients ``sh.shape[-2]`` is not a square (L + 1)², or L is above 3.

    Notes
    -----
    Q-204: public so that the differential harness's ``sh_color`` quantity calls the package's
    own code; the reference backend of :func:`openftgs.render.render` calls it too.
    """
    degree = math.isqrt(sh.shape[-2]) - 1
    if (degree + 1) ** 2 != sh.shape[-2]:
        raise ValueError(f"expected (L + 1)² SH coefficients, got {sh.shape[-2]}")
    basis = sh_basis(dirs, degree)
    return torch.clamp((sh * basis[..., None]).sum(dim=-2) + _COLOR_OFFSET, min=0.0)


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    """Convert a color to its SH DC coefficient, (rgb − 0.5)/C₀ (§1.1).

    It is the initial value of ``sh0`` (§1.1), and the inverse of :func:`sh0_to_rgb`.

    Parameters
    ----------
    rgb
        Colors, ``[..., 3]`` floating point, nominally in [0, 1].

    Returns
    -------
    torch.Tensor
        The DC coefficients, ``[..., 3]``, in the dtype of ``rgb``.

    Notes
    -----
    D-4: the 0.5 offset is E2's; :data:`C0` is the l = 0 constant.
    """
    return (rgb - 0.5) / C0


def sh0_to_rgb(sh0: Tensor) -> Tensor:
    """Convert an SH DC coefficient to its color, C₀·sh0 + 0.5 (E2 at degree 0).

    It is the inverse of :func:`rgb_to_sh0`. No clamp is applied, unlike E2's color.

    Parameters
    ----------
    sh0
        DC coefficients, ``[..., 3]`` floating point.

    Returns
    -------
    torch.Tensor
        The colors, ``[..., 3]``, in the dtype of ``sh0``.

    Notes
    -----
    D-4: the 0.5 offset; :data:`C0` is the l = 0 constant.
    """
    return C0 * sh0 + 0.5
