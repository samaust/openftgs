# SPDX-License-Identifier: Apache-2.0
"""`python -m openftgs.cli`: runs the same `main` as the ftgs console script (Q-124)."""

import sys

from openftgs.cli import main

sys.exit(main())
