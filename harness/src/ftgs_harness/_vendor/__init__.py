# SPDX-License-Identifier: Apache-2.0
"""Third-party code the harness carries (Q-254, #391).

``flip`` is NVIDIA's ꟻLIP loss at commit d3ab02a, made by ``scripts/vendor_flip.py --target
harness`` from the reference copy in ``tests/reference/flip`` (Q-252) with the device-agnostic
changes only, so that L3 computes ꟻLIP itself on the CPU (§3.3 L3 step 3). The harness never
imports openftgs (§2.1), so it doesn't share openftgs's copy.
"""
