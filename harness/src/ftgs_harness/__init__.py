# SPDX-License-Identifier: Apache-2.0
"""ftgs_harness: black-box differential harness (§3.3).

It compares two implementations through the CLI and the file formats of §2.2 only,
and never imports openftgs (§2.1). The command is `ftgs-harness` (`ftgs_harness.cli`).

The modules are internal to the distribution (Q-213): as in openftgs's modules, each
module's `__all__` lists exactly its top-level names that don't start with an underscore
(the names other harness modules import, the `run` functions the runner loads through
importlib, the types in their signatures and names the tests use), and none of it is a
stable API. `tests/test_skeleton.py` checks it.
"""

__all__: list[str] = []
