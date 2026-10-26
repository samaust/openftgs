# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's ꟻLIP loss, vendored (D-65, D-89; #383).

``flip_loss.py`` is ``src/pytorch/flip_loss.py`` of https://github.com/NVlabs/flip at commit
d3ab02a, made by ``scripts/vendor_flip.py`` from the reference copy in ``tests/reference/flip``
(Q-252) and never edited by hand. Its BSD-3-Clause license is in its header and in LICENSE
beside it; VENDOR.json records the hashes and every edit. The changes are device and caching
only (D-89); the lines that carry them are marked ``openftgs (<change>)``.
"""

from .flip_loss import HDRFLIPLoss, LDRFLIPLoss

__all__ = ["HDRFLIPLoss", "LDRFLIPLoss"]
