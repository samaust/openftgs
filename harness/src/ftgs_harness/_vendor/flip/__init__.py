# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's ꟻLIP loss, the harness's own copy (§3.3 L3 step 3; Q-254; #391).

``flip_loss.py`` is ``src/pytorch/flip_loss.py`` of https://github.com/NVlabs/flip at commit
d3ab02a, made by ``scripts/vendor_flip.py --target harness`` from the reference copy in
``tests/reference/flip`` (Q-252) and never edited by hand. Its BSD-3-Clause license is in its
header and in LICENSE beside it; VENDOR.json records the hashes and every edit. The only change
is the device-agnostic one, on the lines marked ``ftgs_harness (device)``, so that it runs on
the CPU. Importing it needs torch, which the harness's optional extra holds (Q-254).
"""

from .flip_loss import LDRFLIPLoss

__all__ = ["LDRFLIPLoss"]
