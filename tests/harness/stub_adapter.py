# SPDX-License-Identifier: Apache-2.0
"""A stub adapter for the harness's own tests: `info` prints the JSON in $STUB_INFO_<LABEL>
(or a complete reply), and every other command exits with $STUB_EXIT (0 by default) after
printing its arguments. Run as `python stub_adapter.py LABEL <command> …`."""

import json
import os
import sys

CONVENTIONS = {
    "alpha_max": 0.999,
    "eps2d": 0.3,
    "near": 0.01,
    "far": 1e10,
    "alpha_skip": 1.0 / 255.0,
    "transmittance_stop": 1e-4,
    "pixel_center_offset": 0.5,
    "background": [0.0, 0.0, 0.0],
    "jacobian_clamp_margin": 0.15,
    "screen_extent_rule": "opacity_aware",
    "screen_extent_max": 3.33,
    "rasterize_mode": "classic",
}
INFO = {
    "name": "stub",
    "version": "0",
    "query_kinds": [],
    "rasterizer": {"name": "stub", "version": "0"},
    "render_conventions": CONVENTIONS,
}


def main() -> int:
    label, command, *args = sys.argv[1:]
    if command == "info":
        print(os.environ.get(f"STUB_INFO_{label}", json.dumps(INFO)))
        return int(os.environ.get(f"STUB_INFO_EXIT_{label}", "0"))
    print(json.dumps([command, *args]))
    return int(os.environ.get("STUB_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
