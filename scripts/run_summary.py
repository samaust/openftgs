# SPDX-License-Identifier: Apache-2.0
"""The summary of a `ftgs train` run directory that M17 records in the Calibration data table
(D-10, D-18, D-44; Q-220; #317).

    python scripts/run_summary.py RUN_DIR [--out SUMMARY.json] [--block B]
                                  [--time TIME.txt] [--gpu-log GPU.csv] [--fastest N0:N1]

Reads the run directory's `config.json`, `train_log.jsonl`, every checkpoint it kept
(`checkpoint/` and `checkpoint-<iteration>/`, with `openftgs.io.load_checkpoint`) and its
METRICS.json when present, and writes, as JSON (default `RUN_DIR/SUMMARY.json`):

- `n_trace`: N at the log's first row, at its last row at or before `growth.stop` (where growth
  ends, D-18), and at its last row; `growth.n_max` and the first iteration whose N reached it,
  or null (D-44). The log is read every `perf.log_every` iterations, 100 by default, which lands
  on every event, so this is the count after each event. That holds only while `log_every`
  divides 100, the events' spacing: with a larger one, the N at the end of growth and the first
  iteration at the cap are read up to `log_every` − 100 iterations late;
- `loss_blocks`: the mean `reg` and `total` of the log's rows over each block of B iterations
  (default 1,000): iterations 1 to B, B + 1 to 2B, …;
- `checkpoints`: per checkpoint, in order of iteration, its iteration, N, the mean base opacity
  σ = sigmoid(opacity logit), its histogram in 10 equal bins of [0, 1] and the share σ > 0.9:
  D-10's opacity trace, where a cutoff of the regularization shows as a rise in mean opacity;
- `metrics`: METRICS.json's version, split and scene means over the entire images, a version-1
  file's LPIPS backbone label, and its `dynamic` and `fastest_10` blocks without their per-image
  rows; null without one. Versions 1 (M17's runs, LPIPS) and 2 (since 549o, ꟻLIP) are read, and
  `version` says which; a file of another version or format is exit 2 (Q-256). With
  `--fastest N0:N1`, a range of 10 frames, also `fastest_10_derived`: the means of the file's own
  metrics over the per-image rows with N0 ≤ frame_index < N1, entire and dynamic, as `ftgs eval`
  writes `fastest_10` for a manifest that carries the range, for a scene whose manifest has none
  (studio's, Q-222);
- `wall`: with `--time`, the elapsed wall time in seconds and the peak resident memory in KiB
  from `/usr/bin/time -v`'s report; with `--gpu-log`, the peak of the MiB values that
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` sampled.

Exit 0 once written, 2 for a folder that isn't a run directory or a file that can't be read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

from openftgs.eval import average_metrics
from openftgs.io import load_checkpoint

_BINS = 10  # Q-220: D-10's opacity trace as a 10-bin histogram of σ over [0, 1]
_HIGH = 0.9  # Q-220: the share of primitives above it, as S-5 counts them
_FASTEST_COUNT = 10  # D-69: the range holds the 10 fastest frames
_FORMAT = "ftgs-metrics"  # Q-63: METRICS.json's format tag
_VERSIONS = (1, 2)  # Q-256: version 1 (LPIPS, with its backbone label) and version 2 (ꟻLIP)


def _n_trace(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    # Exact at the events only while perf.log_every divides 100 (module docstring).
    growth = config["growth"]
    stop, n_max = int(growth["stop"]), int(growth["n_max"])

    def at(row: dict[str, Any]) -> dict[str, int]:
        return {"iteration": int(row["iteration"]), "num_gaussians": int(row["num_gaussians"])}

    grown = [row for row in rows if int(row["iteration"]) <= stop]
    capped = [int(row["iteration"]) for row in rows if int(row["num_gaussians"]) >= n_max]
    return {
        "first": at(rows[0]) if rows else None,
        "growth_end": at(grown[-1]) if grown else None,
        "last": at(rows[-1]) if rows else None,
        "n_max": n_max,
        "first_at_n_max": capped[0] if capped else None,
    }


def _loss_blocks(rows: list[dict[str, Any]], block: int) -> list[dict[str, Any]]:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(math.ceil(int(row["iteration"]) / block), []).append(row)
    out = []
    for b in sorted(by_block):
        group = by_block[b]
        out.append(
            {
                "iterations": [(b - 1) * block + 1, b * block],
                "rows": len(group),
                "reg": sum(float(r["losses"]["reg"]) for r in group) / len(group),
                "total": sum(float(r["losses"]["total"]) for r in group) / len(group),
            }
        )
    return out


def _checkpoint(directory: Path) -> dict[str, Any]:
    state = load_checkpoint(directory)
    sigma = torch.sigmoid(state.params["opacity_logits"].to(torch.float64).reshape(-1))
    counts = torch.histc(sigma, bins=_BINS, min=0.0, max=1.0)
    n = int(sigma.numel())
    return {
        "path": directory.name,
        "iteration": state.iteration,
        "num_gaussians": n,
        "opacity_mean": float(sigma.mean()) if n else None,
        "opacity_histogram": [int(c) for c in counts.tolist()],
        "opacity_share_above_0_9": float((sigma > _HIGH).sum()) / n if n else None,
    }


def _fastest_means(
    rows: list[dict[str, Any]], fastest: tuple[int, int], names: list[str]
) -> dict[str, Any]:
    # Q-222: the means eval.metrics_report gives a fastest_10 range, over the rows inside it
    # that carry values (a skipped dynamic row has none); Q-256: of the file's own metrics.
    n0, n1 = fastest
    inside = [
        {name: float(r[name]) for name in names}
        for r in rows
        if r.get("frame_index") is not None
        and n0 <= int(r["frame_index"]) < n1
        and all(r.get(name) is not None for name in names)
    ]
    return average_metrics(inside) if inside else dict.fromkeys(names)


def _metrics(path: Path, fastest: tuple[int, int] | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        if fastest is not None:
            raise ValueError(f"--fastest needs the run's METRICS.json, and {path} isn't a file")
        return None
    body = json.loads(path.read_text())
    version = body.get("version")
    if body.get("format") != _FORMAT or version not in _VERSIONS:  # Q-256
        raise ValueError(
            f"{path} isn't a METRICS.json of version 1 or 2: format {body.get('format')!r}, "
            f"version {version!r}"
        )
    out: dict[str, Any] = {"version": version, "split": body["split"]}
    if version == 1:  # Q-62: version 1 labels the backbone the paper reports
        out["lpips_backbone"] = body["lpips_backbone"]
    out |= {"count": len(body["entire"]["images"]), "entire": body["entire"]["scene"]}
    if "dynamic" in body:
        out["dynamic"] = {k: v for k, v in body["dynamic"].items() if k != "images"}
    if "fastest_10" in body:
        out["fastest_10"] = body["fastest_10"]
    if fastest is not None:
        n0, n1 = fastest
        names = [str(name) for name in body["metrics"]]  # Q-63: the file's metric names
        entire = body["entire"]["images"]
        derived: dict[str, Any] = {
            "range": [n0, n1],
            "count": sum(
                1
                for r in entire
                if r.get("frame_index") is not None and n0 <= int(r["frame_index"]) < n1
            ),
            "entire": _fastest_means(entire, fastest, names),
        }
        if "dynamic" in body:
            derived["dynamic"] = _fastest_means(body["dynamic"]["images"], fastest, names)
        derived["source"] = "derived from METRICS.json's per-image rows (Q-222)"
        out["fastest_10_derived"] = derived
    return out


def _wall(time_log: Path | None, gpu_log: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if time_log is not None:
        text = time_log.read_text()
        elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([\d:.]+)", text)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        if elapsed is None or rss is None:
            raise ValueError(f"{time_log} isn't a report of /usr/bin/time -v")
        seconds = 0.0
        for part in elapsed.group(1).split(":"):
            seconds = seconds * 60.0 + float(part)
        out["elapsed_s"] = seconds
        out["max_rss_kib"] = int(rss.group(1))
    if gpu_log is not None:
        values = [int(m) for m in re.findall(r"(\d+)\s*MiB", gpu_log.read_text())]
        out["gpu_memory_used_peak_mib"] = max(values) if values else None
    return out


def summarize(
    run: Path,
    *,
    block: int = 1000,
    time_log: Path | None = None,
    gpu_log: Path | None = None,
    fastest: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """The summary of a run directory, as the script writes it."""
    for name in ("config.json", "train_log.jsonl"):
        if not (run / name).is_file():
            raise ValueError(f"{run} has no {name}: not a ftgs train run directory")
    config = json.loads((run / "config.json").read_text())
    lines = (run / "train_log.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows.sort(key=lambda r: int(r["iteration"]))
    folders = [
        p for p in run.iterdir() if p.is_dir() and re.fullmatch(r"checkpoint(-\d+)?", p.name)
    ]
    checkpoints = sorted((_checkpoint(p) for p in folders), key=lambda c: int(c["iteration"]))
    return {
        "run": str(run),
        "iterations": int(config["optim"]["iterations"]),
        "n_trace": _n_trace(rows, config),
        "loss_blocks": _loss_blocks(rows, block),
        "checkpoints": checkpoints,
        "metrics": _metrics(run / "METRICS.json", fastest),
        "wall": _wall(time_log, gpu_log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the summary of a ftgs train run directory")
    parser.add_argument("run", type=Path, help="the run directory (ftgs train --out)")
    parser.add_argument("--out", type=Path, help="where to write it (RUN_DIR/SUMMARY.json)")
    parser.add_argument(
        "--block", type=int, default=1000, help="iterations per loss block"
    )  # Q-220
    parser.add_argument("--time", type=Path, help="the report of /usr/bin/time -v")
    parser.add_argument("--gpu-log", type=Path, help="nvidia-smi's memory.used samples")
    parser.add_argument(
        "--fastest", metavar="N0:N1", help="the 10 fastest frames' range to average (Q-222)"
    )
    args = parser.parse_args(argv)
    if args.block < 1:
        parser.error("--block must be at least 1")
    fastest = None
    if args.fastest is not None:
        match = re.fullmatch(r"(\d+):(\d+)", args.fastest)
        if match is None or int(match.group(2)) - int(match.group(1)) != _FASTEST_COUNT:
            parser.error(f"--fastest must be N0:N1 with N1 = N0 + 10, got {args.fastest!r}")
        fastest = (int(match.group(1)), int(match.group(2)))
    try:
        body = summarize(
            args.run, block=args.block, time_log=args.time, gpu_log=args.gpu_log, fastest=fastest
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"run_summary: {error}", file=sys.stderr)
        return 2
    out = args.out if args.out is not None else args.run / "SUMMARY.json"
    out.write_text(json.dumps(body, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
