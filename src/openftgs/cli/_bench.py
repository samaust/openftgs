# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench`: the performance benchmarks B-1 to B-8 of §3.4, each a subcommand that writes
BENCH.json.

Spec: §2.1 (the `ftgs bench` line and the exit codes), §3.4, D-50. Q-166 (BENCH.json and the
exit codes), Q-167 (`--model`), Q-168 (the setups). docs/modules/cli/benchmarks.md maps each
subcommand to its B-n.

Each subcommand has an entry in `_BENCHMARKS`: its B-n, the function that runs it, and whether
it needs CUDA and takes `--model`. The framework checks the scene and the environment, runs
the function and writes BENCH.json; a benchmark whose ticket hasn't landed has no function yet
and keeps Q-130's behaviour: it says so, writes nothing and returns 1.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openftgs.cli import _INPUT, _fail, _load_model, _scene

_UNBUILT = 1  # Q-130: a benchmark whose ticket hasn't landed


@dataclass(frozen=True)
class _Benchmark:
    """One subcommand: its §3.4 ID, the ticket that builds it, the function that runs it (None
    until that ticket lands), whether it needs CUDA (all but `convert`, Q-166) and whether it
    takes `--model` (Q-167)."""

    id: str
    ticket: int
    run: Callable[..., Any] | None = None
    cuda: bool = True
    model: bool = False


_BENCHMARKS: dict[str, _Benchmark] = {
    "train-step": _Benchmark("B-1", 226, model=True),
    "raster": _Benchmark("B-2", 227),
    "slice": _Benchmark("B-3", 225),
    "losses": _Benchmark("B-4", 228),
    "adam": _Benchmark("B-5", 229),
    "data": _Benchmark("B-6", 232),
    "render": _Benchmark("B-7", 231, model=True),
    "init": _Benchmark("B-8", 234),
    "convert": _Benchmark("B-8", 234, cuda=False),
}


def add_parser(commands: Any) -> None:
    """The `bench` command and its options, on the `ftgs` subparsers `commands`."""
    p = commands.add_parser("bench", help="performance benchmarks (§3.4)")
    p.add_argument("benchmark", choices=tuple(_BENCHMARKS))
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--out", type=Path, required=True, metavar="BENCH.json")
    # Q-167: train-step and render only; the spec's CLI line takes no model (a change request).
    p.add_argument("--model", type=Path, metavar="MODEL")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    """`ftgs bench BENCHMARK SCENE.json --out BENCH.json`."""
    import torch

    bench = _BENCHMARKS[args.benchmark]
    if args.model is not None and not bench.model:  # Q-167
        takers = " and ".join(name for name, b in _BENCHMARKS.items() if b.model)
        _fail(f"--model applies only to {takers}", _INPUT)
    _scene(args.scene)
    if bench.cuda and not torch.cuda.is_available():  # Q-166: an invalid environment
        _fail(f"{bench.id} needs a CUDA device", _INPUT)
    if args.model is not None:  # Q-167: MODEL as `ftgs render` reads it
        _load_model(args.model, "cuda")
    if bench.run is None:  # Q-130 until its ticket lands
        print(
            f"ftgs bench {args.benchmark}: {bench.id} comes with #{bench.ticket}; "
            "nothing was run or written",
            file=sys.stderr,
        )
        return _UNBUILT
    result: int = bench.run(args)
    return result
