# SPDX-License-Identifier: Apache-2.0
"""Third-party code vendored into openftgs (D-65).

``romatch`` is RoMa's inference code at commit 77f8d68, made by ``scripts/vendor_roma.py``
and never edited by hand; its licenses, notices and patch record sit beside it (LICENSE,
LICENSES, VENDOR.json). The package is private and imported lazily by ``openftgs.init``.

``flip`` is NVIDIA's ꟻLIP loss at commit d3ab02a, made by ``scripts/vendor_flip.py`` from the
reference copy in ``tests/reference/flip`` and never edited by hand, with changes that are
device and caching only (D-89); its license and vendoring record sit beside it (LICENSE,
VENDOR.json).
"""

__all__: list[str] = []
