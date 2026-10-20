# SPDX-License-Identifier: Apache-2.0
"""`ftgs-adapter forward QUERIES OUT`: the L2 query kinds of §3.3 answered with the package's
own functions, always on the CPU (Q-202, Q-207).

The files are ftgs-forward-queries and ftgs-forward-answers (Q-207): one kind per file; a
query's tensors under `<id>/<name>`, its FTGS-IF model under `<id>/model/<key>`; each answer
carries a status, `ok`, `error` (the query couldn't be answered) or `not_comparable` (a kind
this adapter doesn't answer).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs import cli

__all__ = ["KINDS", "forward"]

_QUERIES = "ftgs-forward-queries"
_ANSWERS = "ftgs-forward-answers"

Handler = Callable[[Mapping[str, Any], Mapping[str, Tensor]], dict[str, Tensor]]

# kind → handler(params, tensors) → answer tensors (§3.3 L2, Q-207); #285 adds them.
KINDS: dict[str, Handler] = {}


def _read(path: Path) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Tensor]]]]:
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    if meta.get("format") != _QUERIES or meta.get("version") != "1":
        raise ValueError(f"{path}: not a {_QUERIES} v1 file")
    entries = json.loads(meta["queries"])
    queries = []
    for entry in entries:
        prefix = entry["id"] + "/"
        own = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
        queries.append((entry["id"], dict(entry["params"]), own))
    return meta["kind"], queries


def forward(queries_path: Path, out: Path, *, backend: str) -> int:
    """Answer every query of the file; a query that fails is an `error` answer, the others
    are still answered (Q-207)."""
    with cli._as_input_error(f"forward queries {queries_path}", OSError, KeyError):
        sha = hashlib.sha256(queries_path.read_bytes()).hexdigest()
        kind, queries = _read(queries_path)
    handler = KINDS.get(kind)
    entries, tensors = [], {}
    for query_id, params, inputs in queries:
        if handler is None:
            entries.append({"id": query_id, "status": "not_comparable", "message": kind})
            continue
        try:
            with torch.no_grad() if kind != "grad_stats" else torch.enable_grad():
                answer = handler(params, inputs)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            entries.append({"id": query_id, "status": "error", "message": str(error)})
            continue
        entries.append({"id": query_id, "status": "ok", "message": ""})
        tensors.update(
            {f"{query_id}/{k}": v.detach().cpu().contiguous() for k, v in answer.items()}
        )
    meta = {
        "format": _ANSWERS,
        "version": "1",
        "kind": kind,
        "producer": cli._producer(),
        "queries_sha256": sha,
        "answers": json.dumps(entries),
    }
    with cli._output(f"forward answers {out}"):
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out), meta)
    return 0
