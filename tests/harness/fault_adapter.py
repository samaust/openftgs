# SPDX-License-Identifier: Apache-2.0
"""A black-box fault-injecting adapter for the harness's own tests (PRD #279, ticket #284).

    python fault_adapter.py --inner "ftgs-adapter --backend reference" [--fault NAME[=V] …]
        <command> <args…>

It runs the inner adapter's command and changes one thing, using NumPy and safetensors only,
so a level's test can check that the harness sees that change, with the expected pattern:

- model faults, applied to the FTGS-IF file given to `import` and `render`: `velocity=F`
  (× F), `duration=F` (× F), `scale=F` (× F), `dc=D` (+ D on the l = 0 coefficients),
  `sh_sign` (l ≥ 1 coefficients negated), `drop_primitive` (row 0's opacity to the floor
  1e-7, so it all but vanishes; a file can't hold zero rows), `position_frame` (positions
  through a normalized scene frame (x − c)/r in float32 and back, as a faithful implementation
  may store them), `position_zero=F` (position entries below F·‖position‖∞ set to 0),
  `duration_small=E` (durations below 1 % of the largest × (1 + E));
- request faults, applied to `render`'s request, the renders then filed under the original
  request's SHA-256: `cx=D` (+ D px on every item's principal point);
- output faults: `nan` (a NaN at every render's centre pixel), `drop_key=K` (key K removed
  from `export`'s file), `answers=E` (every float answer × (1 + E)), `exact_motion`
  (`motion` answered by E1 in float64 from the queries file), `zero_answers=F` (every float
  answer's entries below F·‖answer‖∞ set to 0), `biased_sampler`
  (`relocation_sampling` counts moved from the last target with draws to the first),
  `no_kind=K` (kind K left out of `info`), `drop_convention=K` (entry K left out of `info`'s
  render_conventions), `rasterizer=NAME` (`info` names another rasterizer);
- process faults: `exit=C` (exit with C without running), `hang` (sleep past any timeout),
  `garbage` (OUT written as bytes that are no format).

`--on CMD` (repeatable) limits the faults to those commands; the others run unchanged.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

MODEL_FAULTS = {
    "velocity",
    "duration",
    "scale",
    "dc",
    "sh_sign",
    "drop_primitive",
    "position_frame",
    "position_zero",
    "duration_small",
}
FORWARD_FAULTS = {"answers", "biased_sampler", "exact_motion", "zero_answers"}


def _read(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    with safe_open(str(path), framework="np") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})  # noqa: SIM118


def _write(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]) -> None:
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path), meta)


def _perturb_model(source: Path, target: Path, faults: dict[str, str]) -> None:
    arrays, meta = _read(source)
    if "velocity" in faults:
        arrays["velocity"] = arrays["velocity"] * np.float32(faults["velocity"])
    if "duration" in faults:
        arrays["duration"] = arrays["duration"] * float(faults["duration"])
    if "scale" in faults:
        arrays["scale"] = arrays["scale"] * np.float32(faults["scale"])
    if "dc" in faults:
        arrays["sh"] = arrays["sh"].copy()
        arrays["sh"][:, 0, :] += np.float32(faults["dc"])
    if "sh_sign" in faults:
        arrays["sh"] = arrays["sh"].copy()
        arrays["sh"][:, 1:, :] *= np.float32(-1.0)
    if "drop_primitive" in faults:
        arrays["opacity"] = arrays["opacity"].copy()
        arrays["opacity"][0] = np.float32(1e-7)
    if "position_frame" in faults:
        x = arrays["position"].astype(np.float32)
        c = x.mean(axis=0, dtype=np.float32)
        r = np.float32(np.linalg.norm(x - c, axis=1).max() or 1.0)
        arrays["position"] = ((x - c) / r).astype(np.float32) * r + c
    if "position_zero" in faults:
        x = arrays["position"].copy()
        x[np.abs(x) < float(faults["position_zero"]) * np.abs(x).max()] = 0.0
        arrays["position"] = x
    if "duration_small" in faults:
        d = arrays["duration"].copy()
        small = d < 0.01 * d.max()
        d[small] *= 1.0 + float(faults["duration_small"])
        arrays["duration"] = d
    _write(target, arrays, meta)


def _exact_motion(queries: Path, tensors: dict[str, np.ndarray]) -> None:
    """E1 in float64 from the FTGS-IF model, whose velocity is per unit of τ: μx(τ) = μx +
    v·(τ − τc), for each answer's `means_t` [K, N, 3]."""
    model, _ = _read(queries)
    for key in tensors:
        query_id, _, name = key.partition("/")
        if name != "means_t":
            continue
        position = model[f"{query_id}/model/position"].astype(np.float64)
        velocity = model[f"{query_id}/model/velocity"].astype(np.float64)
        center = model[f"{query_id}/model/time_center"].astype(np.float64)
        times = model[f"{query_id}/times"].astype(np.float64)
        tensors[key] = position + velocity * (times[:, None, None] - center[None, :, None])


def _inner(argv: list[str], command: str, args: list[str]) -> int:
    return subprocess.run([*argv, command, *args], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", required=True)
    parser.add_argument("--fault", action="append", default=[])
    parser.add_argument("--on", action="append", default=[])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()
    inner = shlex.split(opts.inner)
    faults = dict(f.partition("=")[::2] for f in opts.fault)
    command, args = opts.command, list(opts.args)
    if opts.on and command not in opts.on:
        faults = {}
    if "exit" in faults:
        return int(faults["exit"])
    if "hang" in faults:
        time.sleep(3600)
    if "garbage" in faults and command != "info":
        Path(args[-1]).write_bytes(b"\x00 not a file of any format")
        return 0
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        if command == "info":
            done = subprocess.run([*inner, "info"], capture_output=True, text=True, check=False)
            info = json.loads(done.stdout)
            if "no_kind" in faults:
                info["query_kinds"] = [k for k in info["query_kinds"] if k != faults["no_kind"]]
            if "drop_convention" in faults:
                info["render_conventions"].pop(faults["drop_convention"], None)
            if "rasterizer" in faults:
                info["rasterizer"] = {"name": faults["rasterizer"], "version": "0"}
            print(json.dumps(info))
            return done.returncode
        if command in ("import", "render") and MODEL_FAULTS & faults.keys():
            source = Path(args[0])
            if source.is_file():
                args[0] = str(tmp / "model.safetensors")
                _perturb_model(source, Path(args[0]), faults)
        original_sha = None
        if command == "render" and "cx" in faults:
            request_path = Path(args[1])
            original_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
            request = json.loads(request_path.read_text())
            for item in request["items"]:
                item["camera"]["K"][0][2] += float(faults["cx"])
            args[1] = str(tmp / "request.json")
            Path(args[1]).write_text(json.dumps(request))
        code = _inner(inner, command, args)
        if code != 0:
            return code
        out = Path(args[-1])
        if command == "render" and (original_sha is not None or "nan" in faults):
            tensors, meta = _read(out)
            if original_sha is not None:
                meta["request_sha256"] = original_sha
            if "nan" in faults:
                for key, value in tensors.items():
                    if key.startswith("rgb/"):
                        value = value.copy()
                        value[value.shape[0] // 2, value.shape[1] // 2, :] = np.nan
                        tensors[key] = value
            _write(out, tensors, meta)
        if command == "export" and "drop_key" in faults:
            tensors, meta = _read(out)
            tensors.pop(faults["drop_key"], None)
            _write(out, tensors, meta)
        if command == "forward" and FORWARD_FAULTS & faults.keys():
            tensors, meta = _read(out)
            if "exact_motion" in faults and meta.get("kind") == "motion":
                _exact_motion(Path(args[0]), tensors)
            if "zero_answers" in faults:
                for key, value in tensors.items():
                    if value.dtype.kind == "f" and value.size:
                        value = value.copy()
                        value[
                            np.abs(value) < float(faults["zero_answers"]) * np.abs(value).max()
                        ] = 0
                        tensors[key] = value
            if "answers" in faults:
                factor = 1.0 + float(faults["answers"])
                for key, value in tensors.items():
                    if value.dtype.kind == "f":
                        tensors[key] = (value * factor).astype(value.dtype)
            if "biased_sampler" in faults and meta.get("kind") == "relocation_sampling":
                for key, value in tensors.items():
                    if key.endswith("/counts"):
                        value = value.copy()
                        last = int(np.nonzero(value.sum(0))[0].max())
                        value[:, 0] += value[:, last]
                        value[:, last] = 0
                        tensors[key] = value
            _write(out, tensors, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
