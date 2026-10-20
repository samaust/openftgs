# SPDX-License-Identifier: Apache-2.0
"""ftgs_harness: black-box differential harness (§3.3).

It compares two implementations through the CLI and the file formats of §2.2 only,
and never imports openftgs (§2.1). The command is `ftgs-harness` (`ftgs_harness.cli`).

The modules are internal to the distribution (Q-213): each `__all__` lists the names other
harness modules import, not a stable API.
"""

__all__: list[str] = []
