# SPDX-License-Identifier: Apache-2.0
"""Small FTGS-IF files and render requests for the adapter's tests and the fault adapter's."""

import json
from pathlib import Path

from builders import exchange_arrays, ring

from openftgs.io import write_exchange
from openftgs.types import ExchangeModel


def exchange_file(path: Path, *, n: int = 12, sh_degree: int = 2, seed: int = 0) -> Path:
    arrays = exchange_arrays(n, sh_degree, seed)
    arrays["position"] = arrays["position"] * 0.3  # inside the request's view
    arrays["scale"] = arrays["scale"] * 0.2
    write_exchange(
        ExchangeModel(arrays, sh_degree, 2.0, 12.0, "s", "test", background=(0.0, 0.2, 1.0)), path
    )
    return path


def request_file(path: Path) -> Path:
    items = []
    for k, cam in enumerate(ring(3, radius=4.0, elevation=1.0, width_px=24, height_px=16)):
        items.append(
            {
                "id": f"r{k}",
                "timestamp": 2.0 + 3.0 * k,
                "camera": {
                    "width": cam.width,
                    "height": cam.height,
                    "K": cam.K.tolist(),
                    "world_to_camera": cam.world_to_camera.tolist(),
                },
            }
        )
    path.write_text(
        json.dumps(
            {"format": "ftgs-render-request", "version": 1, "time_unit": "s", "items": items}
        )
    )
    return path
