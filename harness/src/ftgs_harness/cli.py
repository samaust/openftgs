# SPDX-License-Identifier: Apache-2.0
"""`ftgs-harness run` and `ftgs-harness rerun` (§3.3; Q-203). Exit 0 when no check failed, 1
when one did, 2 for a harness or configuration error."""

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from ftgs_harness.runner import rerun, run
from ftgs_harness.settings import HarnessError

__all__ = ["main"]

_ERROR = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftgs-harness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="compare two implementations")
    p.add_argument("--a", required=True, help="adapter A's command line")
    p.add_argument("--b", required=True, help="adapter B's command line")
    p.add_argument("--config", required=True, type=Path, help="harness_config.json")
    p.add_argument("--levels", required=True, help="for example L0,L1,L2,L3")
    p.add_argument("--out", required=True, type=Path, help="the run directory")
    p.add_argument("--settings", type=Path, help="the run settings (Q-203)")
    p.add_argument("--jobs", type=int, default=1, help="adapter calls at a time")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per adapter call")
    r = commands.add_parser("rerun", help="reproduce one check of a run")
    r.add_argument("check_id")
    r.add_argument("--out", type=Path, default=Path("."), help="the run directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:  # argparse's usage errors are exit 2 already
        return int(stop.code or 0)
    try:
        if args.command == "run":
            return run(
                a=args.a,
                b=args.b,
                config=args.config,
                levels=args.levels,
                out=args.out,
                settings_path=args.settings,
                jobs=args.jobs,
                timeout=args.timeout,
            )
        return rerun(args.check_id, args.out)
    except HarnessError as error:
        print(f"ftgs-harness: error: {error}", file=sys.stderr)
        return _ERROR
    except Exception:  # §3.3: any other failure of the harness itself is exit 2 too
        traceback.print_exc()
        print("ftgs-harness: error: the harness failed (traceback above)", file=sys.stderr)
        return _ERROR
