# SPDX-License-Identifier: Apache-2.0
"""`ftgs-adapter`: openftgs's adapter for the differential harness (§3.3 adapter contract;
Q-202). A second console script, so §2.1's `ftgs` command list stays as it is.

    ftgs-adapter [--backend reference|gsplat] info
    ftgs-adapter [--backend …] export MODEL OUT
    ftgs-adapter [--backend …] import IF OUT
    ftgs-adapter [--backend …] render MODEL REQUEST OUT
    ftgs-adapter [--backend …] forward QUERIES OUT
    ftgs-adapter [--backend …] train SCENE INIT CONFIG SEED OUT_DIR

Exit codes follow `ftgs`'s (§2.1): 2 for invalid input, 3 for a render that isn't finite.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openftgs import cli

if TYPE_CHECKING:
    from openftgs.model import FreeTimeGaussians

__all__ = ["main"]

_NATIVE_PARAMS = "params.safetensors"  # Q-202: the native directory's nine §1.1 tensors
_NATIVE_MODEL = "model.json"  # Q-202: sh_degree, tau0, tau1, time_unit, background
_NOT_COMPARABLE = "not_comparable.json"  # Q-202, §3.3: settings the run couldn't honor


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _info(backend: str) -> dict[str, Any]:
    """Q-205: name, version, the query kinds `forward` answers, the rasterizer identity and the
    render conventions (D-73) with the default black background (D-34)."""
    from openftgs.cli import _adapter_forward
    from openftgs.io import render_conventions

    rasterizer = (
        {"name": "gsplat", "version": _version("gsplat")}
        if backend == "gsplat"
        else {"name": "openftgs-reference", "version": _version("openftgs")}
    )
    return {
        "name": "openftgs",
        "version": _version("openftgs"),
        "query_kinds": sorted(_adapter_forward.KINDS),
        "rasterizer": rasterizer,
        "render_conventions": render_conventions(),
    }


def _write_native(model: FreeTimeGaussians, background: Sequence[float], out: Path) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    params = {name: getattr(model, name).detach().cpu().contiguous() for name in model.PARAM_NAMES}
    save_file(params, str(out / _NATIVE_PARAMS))
    meta = {
        "sh_degree": model.sh_degree,
        "tau0": model.time.tau0,
        "tau1": model.time.tau1,
        "time_unit": model.time_unit,
        "background": [float(c) for c in background],
    }
    (out / _NATIVE_MODEL).write_text(json.dumps(meta) + "\n")


def _load(path: Path, device: str) -> cli._Loaded:
    """MODEL (Q-202): a native directory, a run directory (its checkpoint), a checkpoint or an
    FTGS-IF file."""
    from safetensors.torch import load_file

    from openftgs.model import FreeTimeGaussians
    from openftgs.types import TimeNormalizer

    if path.is_dir() and (path / _NATIVE_MODEL).is_file():
        with cli._as_input_error(f"native model {path}", OSError, KeyError):
            meta = json.loads((path / _NATIVE_MODEL).read_text())
            params = load_file(str(path / _NATIVE_PARAMS), device=device)
            model = FreeTimeGaussians(
                params,
                sh_degree=int(meta["sh_degree"]),
                time=TimeNormalizer(float(meta["tau0"]), float(meta["tau1"])),
                time_unit=str(meta["time_unit"]),
            )
            background = tuple(float(c) for c in meta["background"])
        return cli._Loaded(model, None, (background[0], background[1], background[2]), None)
    if path.is_dir() and (path / "checkpoint").is_dir():
        path = path / "checkpoint"
    return cli._load_model(path, device)


def _import(exchange_path: Path, out: Path) -> int:
    from openftgs.io import read_exchange
    from openftgs.model import FreeTimeGaussians

    with cli._as_input_error(f"exchange model {exchange_path}", OSError):
        exchange = read_exchange(exchange_path)
        model = FreeTimeGaussians.from_exchange(exchange, device="cpu")
    with cli._output(f"native model {out}"):
        _write_native(model, exchange.background, out)
    return 0


def _export(model_path: Path, out: Path) -> int:
    return cli._export_loaded(_load(model_path, "cpu"), out)


def _render(backend: str, model_path: Path, request_path: Path, out: Path) -> int:
    """As `ftgs render MODEL --request REQUEST --out OUT --backend BACKEND`, with MODEL read
    as `_load` reads it."""
    device = cli._render_device(backend)
    return cli._render_loaded(_load(model_path, device), request_path, out, backend, device)


def _unknown_keys(data: Mapping[str, Any], known: Mapping[str, Any], prefix: str = "") -> list[str]:
    """The dotted keys of `data` that TrainConfig doesn't have."""
    out = []
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if key not in known:
            out.append(dotted)
        elif isinstance(known[key], dict) and isinstance(value, dict):
            out += _unknown_keys(value, known[key], dotted + ".")
    return out


def _drop(data: dict[str, Any], dotted: str) -> None:
    *path, last = dotted.split(".")
    for key in path:
        data = data[key]
    del data[last]


def _train(scene: Path, init: Path, config: Path, seed: str, out: Path) -> int:
    """`ftgs train` in process (Q-202). Settings TrainConfig doesn't have are left out of the
    run and listed in OUT_DIR/not_comparable.json (§3.3: reported, never dropped silently)."""
    from openftgs.config import TrainConfig, config_to_dict

    with cli._as_input_error(f"config {config}", OSError):
        data = json.loads(config.read_text())
        if not isinstance(data, dict):
            raise ValueError("the config must be a JSON object")
    unknown = _unknown_keys(data, config_to_dict(TrainConfig()))
    for dotted in unknown:
        _drop(data, dotted)
    with tempfile.TemporaryDirectory() as scratch:
        honored = Path(scratch) / "config.json"
        honored.write_text(json.dumps(data))
        args = ["train", str(scene), "--init", str(init), "--config", str(honored)]
        code = cli.main([*args, "--seed", seed, "--out", str(out)])
    if unknown and out.is_dir():
        (out / _NOT_COMPARABLE).write_text(json.dumps(unknown) + "\n")
    return code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftgs-adapter", description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=("reference", "gsplat"), default="gsplat")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    for name, operands in (
        ("export", ("model", "out")),
        ("import", ("exchange", "out")),
        ("render", ("model", "request", "out")),
        ("forward", ("queries", "out")),
        ("train", ("scene", "init", "config", "seed", "out")),
    ):
        p = commands.add_parser(name)
        for operand in operands:
            p.add_argument(operand, type=str if operand == "seed" else Path)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "info":
        print(json.dumps(_info(args.backend)))
        return 0
    if args.command == "import":
        return _import(args.exchange, args.out)
    if args.command == "export":
        return _export(args.model, args.out)
    if args.command == "render":
        return _render(args.backend, args.model, args.request, args.out)
    if args.command == "forward":
        from openftgs.cli import _adapter_forward

        return _adapter_forward.forward(args.queries, args.out, backend=args.backend)
    return _train(args.scene, args.init, args.config, args.seed, args.out)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if args.backend == "reference":
        # Q-202: the reference backend runs on the CPU, training included (Q-69); set before
        # torch is first imported, so CUDA stays hidden.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return _run(args)
    except cli._Failure as failure:
        print(f"ftgs-adapter {args.command}: error: {failure}", file=sys.stderr)
        return failure.code
