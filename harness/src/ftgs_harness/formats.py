# SPDX-License-Identifier: Apache-2.0
"""The harness's own readers, writers and validators of the §2.2 formats it touches, and of
the forward query and answer formats (§3.3 adapter contract; Q-207). Written from §2.2 with
NumPy and safetensors: the harness never imports an implementation.

A file that fails validation raises `FormatError`, which a level reports as the pattern
"schema violation".
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

__all__ = [
    "ANSWERS_FORMAT",
    "EXCHANGE_KEYS",
    "QUERIES_FORMAT",
    "Answer",
    "Exchange",
    "FormatError",
    "Query",
    "RequestItem",
    "SceneImage",
    "SceneInfo",
    "file_sha256",
    "model_from_tensors",
    "model_tensors",
    "read_answers",
    "read_exchange",
    "read_queries",
    "read_renders",
    "read_scene",
    "validate_exchange",
    "write_answers",
    "write_exchange",
    "write_queries",
    "write_request",
]

QUERIES_FORMAT = "ftgs-forward-queries"
ANSWERS_FORMAT = "ftgs-forward-answers"
_IF = "ftgs-if"
_REQUEST = "ftgs-render-request"
_RENDERS = "ftgs-renders"
_SCENE = "ftgs-scene"
_OPACITY_RANGE = (1e-7, 1.0 - 1e-7)  # §2.2: the exchange file's opacity clamp
_UNIT_TOLERANCE = 1e-6  # |‖q‖ − 1| allowed, as openftgs's writer allows (Q-24)


class FormatError(ValueError):
    """A file that doesn't follow its format: the pattern "schema violation"."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _float(value: float) -> str:
    """§2.2: floats in Python's shortest round-trip repr."""
    return repr(float(value))


def _read(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    try:
        with safe_open(str(path), framework="np") as f:
            meta = dict(f.metadata() or {})
            tensors = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    except Exception as error:  # safetensors raises its own errors for a malformed file
        raise FormatError(f"{path}: not a readable safetensors file ({error})") from error
    return tensors, meta


def _major(version: Any) -> int | None:
    """A version's major number: 1 for 1, "1", "1.0" or "1.3"; None when it has none."""
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    if isinstance(version, str):
        head = version.split(".")[0]
        return int(head) if head.isdigit() else None
    return None


def _header(meta: Mapping[str, Any], fmt: str, where: str) -> None:
    """§2.2: readers reject an unknown major version and ignore unknown optional keys."""
    if meta.get("format") != fmt:
        raise FormatError(f"{where}: format is {meta.get('format')!r}, not {fmt!r}")
    if _major(meta.get("version")) != 1:
        raise FormatError(f"{where}: version {meta.get('version')!r} is not 1")


def _write(path: Path, tensors: Mapping[str, np.ndarray], meta: Mapping[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path), dict(meta))


# ftgs-if (§2.2).

# §2.2 ftgs-if table: key → (dtype, trailing shape; None marks the SH coefficient axis).
EXCHANGE_KEYS: dict[str, tuple[np.dtype[Any], tuple[int | None, ...]]] = {
    "position": (np.dtype(np.float32), (3,)),
    "time_center": (np.dtype(np.float64), ()),
    "duration": (np.dtype(np.float64), ()),
    "velocity": (np.dtype(np.float32), (3,)),
    "scale": (np.dtype(np.float32), (3,)),
    "rotation": (np.dtype(np.float32), (4,)),
    "opacity": (np.dtype(np.float32), ()),
    "sh": (np.dtype(np.float32), (None, 3)),
}


@dataclass
class Exchange:
    """An FTGS-IF model as plain arrays and its metadata (§2.2)."""

    arrays: dict[str, np.ndarray]
    sh_degree: int
    tau0: float
    tau1: float
    time_unit: str = "s"
    producer: str = "ftgs-harness"
    render_conventions: dict[str, Any] | None = None

    @property
    def num(self) -> int:
        return int(self.arrays["position"].shape[0])


def validate_exchange(ex: Exchange, where: str = _IF) -> None:
    """§2.2 ftgs-if: the table's keys, dtypes and shapes, N ≥ 1, finite values, unit rotations
    with w ≥ 0, opacity within the file's clamp, positive durations and scales, sh_degree in 0
    to 3, τ₀ ≤ τ₁, D-71's single-timestamp rule and D-84's background."""
    if isinstance(ex.sh_degree, bool) or ex.sh_degree not in (0, 1, 2, 3):
        raise FormatError(f"{where}: sh_degree {ex.sh_degree!r} is not 0, 1, 2 or 3")
    if not (math.isfinite(ex.tau0) and math.isfinite(ex.tau1) and ex.tau0 <= ex.tau1):
        raise FormatError(f"{where}: tau0 {ex.tau0!r} and tau1 {ex.tau1!r} are not finite in order")
    missing = sorted(set(EXCHANGE_KEYS) - set(ex.arrays))
    if missing:
        raise FormatError(f"{where}: missing keys {missing}")
    n = ex.num
    if n < 1:
        raise FormatError(f"{where}: needs at least one primitive")
    coefficients = (ex.sh_degree + 1) ** 2
    for key, (dtype, tail) in EXCHANGE_KEYS.items():
        a = ex.arrays[key]
        shape = (n, *(coefficients if d is None else d for d in tail))
        if a.dtype != dtype or a.shape != shape:
            raise FormatError(
                f"{where}: {key} is {a.dtype} {list(a.shape)}, expected {dtype} {list(shape)}"
            )
        if not np.all(np.isfinite(a)):
            raise FormatError(f"{where}: {key} holds non-finite values")
    q = ex.arrays["rotation"].astype(np.float64)
    if np.any(np.abs(np.linalg.norm(q, axis=1) - 1.0) > _UNIT_TOLERANCE):
        raise FormatError(f"{where}: rotation is not a unit quaternion on every row")
    if np.any(q[:, 0] < 0.0):
        raise FormatError(f"{where}: rotation has w < 0 on some row, where w ≥ 0")  # §2.2
    low, high = _OPACITY_RANGE
    opacity = ex.arrays["opacity"]
    if np.any(opacity < np.float32(low)) or np.any(opacity > np.float32(high)):
        raise FormatError(f"{where}: opacity outside [1e-7, 1 − 1e-7]")
    if np.any(ex.arrays["duration"] <= 0.0) or np.any(ex.arrays["scale"] <= 0.0):
        raise FormatError(f"{where}: durations and scales must be positive")
    if ex.tau0 == ex.tau1 and (
        np.any(ex.arrays["time_center"] != ex.tau0) or np.any(ex.arrays["velocity"] != 0.0)
    ):
        raise FormatError(
            f"{where}: with τ₀ = τ₁ every time_center is τ₀ and every velocity 0"
        )  # D-71
    conventions = ex.render_conventions
    if conventions is not None and "background" in conventions:
        bg = conventions["background"]
        if not (
            isinstance(bg, list)
            and len(bg) == 3
            and all(isinstance(c, int | float) and not isinstance(c, bool) for c in bg)
            and all(0.0 <= c <= 1.0 for c in bg)
        ):
            raise FormatError(f"{where}: background {bg!r} is not three numbers in [0, 1]")  # D-84


def read_exchange(path: Path) -> Exchange:
    """Read and validate an FTGS-IF file (§2.2)."""
    where = f"{_IF} {path}"
    tensors, meta = _read(path)
    _header(meta, _IF, where)
    try:
        sh_degree = int(meta["sh_degree"])
        tau0, tau1 = float(meta["tau0"]), float(meta["tau1"])
        time_unit = meta["time_unit"]
        conventions = (
            json.loads(meta["render_conventions"]) if "render_conventions" in meta else None
        )
    except (KeyError, ValueError) as error:
        raise FormatError(f"{where}: metadata {error}") from error
    if conventions is not None and not isinstance(conventions, dict):
        raise FormatError(f"{where}: render_conventions is not a JSON object")
    ex = Exchange(
        arrays={k: tensors[k] for k in EXCHANGE_KEYS if k in tensors},
        sh_degree=sh_degree,
        tau0=tau0,
        tau1=tau1,
        time_unit=time_unit,
        producer=meta.get("producer", ""),
        render_conventions=conventions,
    )
    validate_exchange(ex, where)
    return ex


def write_exchange(path: Path, ex: Exchange) -> None:
    """Write an FTGS-IF file (§2.2), after validating it."""
    validate_exchange(ex)
    meta = {
        "format": _IF,
        "version": "1",
        "sh_degree": str(ex.sh_degree),
        "time_unit": ex.time_unit,
        "tau0": _float(ex.tau0),
        "tau1": _float(ex.tau1),
        "producer": ex.producer,
    }
    if ex.render_conventions is not None:
        meta["render_conventions"] = json.dumps(ex.render_conventions, separators=(",", ":"))
    _write(path, {k: ex.arrays[k] for k in EXCHANGE_KEYS}, meta)


# ftgs-render-request and ftgs-renders (§2.2).


@dataclass(frozen=True)
class RequestItem:
    """One item of a render request: an id, a raw timestamp τ and a camera (§1.4 conventions:
    K in the continuous pixel convention, an OpenCV world-to-camera matrix)."""

    id: str
    timestamp: float
    K: np.ndarray  # [3, 3]
    world_to_camera: np.ndarray  # [4, 4]
    width: int
    height: int


def write_request(
    path: Path,
    items: Sequence[RequestItem],
    *,
    time_unit: str = "s",
    background: Sequence[float] = (0.0, 0.0, 0.0),
    sh_degree: int | None = None,
) -> str:
    """Write a ftgs-render-request file (§2.2) and return its SHA-256."""
    body = {
        "format": _REQUEST,
        "version": 1,
        "time_unit": time_unit,
        "background": [float(c) for c in background],
        "sh_degree": sh_degree,
        "items": [
            {
                "id": item.id,
                "timestamp": float(item.timestamp),
                "camera": {
                    "width": int(item.width),
                    "height": int(item.height),
                    "K": np.asarray(item.K, dtype=np.float64).tolist(),
                    "world_to_camera": np.asarray(item.world_to_camera, dtype=np.float64).tolist(),
                },
            }
            for item in items
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, allow_nan=False) + "\n")
    return file_sha256(path)


def read_renders(
    path: Path, *, request_sha256: str, items: Sequence[RequestItem]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a ftgs-renders file (§2.2) made for the request with `request_sha256`: for every
    item, rgb float32 [H, W, 3] (unclamped) and alpha float32 [H, W, 1]."""
    where = f"{_RENDERS} {path}"
    tensors, meta = _read(path)
    _header(meta, _RENDERS, where)
    if meta.get("request_sha256") != request_sha256:
        raise FormatError(f"{where}: made for request {meta.get('request_sha256')!r}")
    out = {}
    for item in items:
        rgb, alpha = tensors.get(f"rgb/{item.id}"), tensors.get(f"alpha/{item.id}")
        if rgb is None or alpha is None:
            raise FormatError(f"{where}: no rgb and alpha for item {item.id}")
        if rgb.dtype != np.float32 or rgb.shape != (item.height, item.width, 3):
            raise FormatError(f"{where}: rgb/{item.id} is {rgb.dtype} {list(rgb.shape)}")
        if alpha.dtype != np.float32 or alpha.shape != (item.height, item.width, 1):
            raise FormatError(f"{where}: alpha/{item.id} is {alpha.dtype} {list(alpha.shape)}")
        out[item.id] = (rgb, alpha)
    return out


# ftgs-scene (§2.2), as L3 reads it.


@dataclass(frozen=True)
class SceneImage:
    camera: str
    timestamp: float
    path: Path  # absolute
    split: str
    K: np.ndarray
    world_to_camera: np.ndarray
    width: int
    height: int
    frame_index: int | None = None


@dataclass(frozen=True)
class SceneInfo:
    path: Path
    time_unit: str
    images: tuple[SceneImage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _matrix(value: Any, shape: tuple[int, int], where: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise FormatError(f"{where}: expected a finite {shape[0]}×{shape[1]} matrix")
    return a


def _scene_image(
    i: int, image: Any, cameras: Mapping[str, Any], base: Path, where: str
) -> SceneImage:
    name = image.get("camera") if isinstance(image, dict) else None
    if not isinstance(name, str) or name not in cameras:
        raise FormatError(f"{where}: image {i} names an unknown camera")
    cam = cameras[name]
    return SceneImage(
        camera=name,
        timestamp=float(image["timestamp"]),
        path=base / image["path"],
        split=image.get("split", "train"),
        K=_matrix(image.get("K", cam["K"]), (3, 3), where),
        world_to_camera=_matrix(
            image.get("world_to_camera", cam["world_to_camera"]), (4, 4), where
        ),
        width=int(cam["width"]),
        height=int(cam["height"]),
        frame_index=image.get("frame_index"),
    )


def read_scene(path: Path) -> SceneInfo:
    """Read a ftgs-scene manifest (§2.2): cameras, images with per-image overrides of K and
    world_to_camera, paths relative to the manifest."""
    where = f"{_SCENE} {path}"
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise FormatError(f"{where}: {error}") from error
    if not isinstance(raw, dict):
        raise FormatError(f"{where}: not a {_SCENE} v1 file")
    _header(raw, _SCENE, where)
    cameras = raw.get("cameras")
    images = raw.get("images")
    if not isinstance(cameras, dict) or not isinstance(images, list):
        raise FormatError(f"{where}: needs cameras and images")
    base = Path(path).resolve().parent
    try:
        out = [_scene_image(i, image, cameras, base, where) for i, image in enumerate(images)]
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, FormatError):
            raise
        raise FormatError(
            f"{where}: a camera or image lacks a key or a value: {error!r}"
        ) from error
    return SceneInfo(
        path=Path(path).resolve(),
        time_unit=str(raw.get("time_unit", "s")),
        images=tuple(out),
        metadata=dict(raw.get("metadata", {})),
    )


# ftgs-forward-queries and ftgs-forward-answers (§3.3; Q-207).


@dataclass
class Query:
    id: str
    params: dict[str, Any]
    tensors: dict[str, np.ndarray]


@dataclass
class Answer:
    id: str
    status: str  # "ok", "error" or "not_comparable"
    message: str
    tensors: dict[str, np.ndarray]


def model_tensors(ex: Exchange) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """An FTGS-IF model as a query's `model/<key>` tensors and `params.model` (Q-207)."""
    tensors = {f"model/{k}": ex.arrays[k] for k in EXCHANGE_KEYS}
    params = {
        "sh_degree": ex.sh_degree,
        "tau0": ex.tau0,
        "tau1": ex.tau1,
        "time_unit": ex.time_unit,
    }
    return tensors, params


def model_from_tensors(tensors: Mapping[str, np.ndarray], params: Mapping[str, Any]) -> Exchange:
    return Exchange(
        arrays={k: tensors[f"model/{k}"] for k in EXCHANGE_KEYS},
        sh_degree=int(params["sh_degree"]),
        tau0=float(params["tau0"]),
        tau1=float(params["tau1"]),
        time_unit=str(params.get("time_unit", "s")),
    )


def write_queries(path: Path, kind: str, queries: Sequence[Query]) -> str:
    """Write one kind's queries (Q-207) and return the file's SHA-256."""
    ids = [q.id for q in queries]
    if len(set(ids)) != len(ids) or any("/" in i for i in ids):
        raise ValueError("query ids must be distinct and contain no '/'")
    tensors = {f"{q.id}/{name}": a for q in queries for name, a in q.tensors.items()}
    meta = {
        "format": QUERIES_FORMAT,
        "version": "1",
        "kind": kind,
        "producer": "ftgs-harness",
        "queries": json.dumps([{"id": q.id, "params": q.params} for q in queries], allow_nan=False),
    }
    _write(path, tensors, meta)
    return file_sha256(path)


def _split(
    tensors: Mapping[str, np.ndarray], ids: Sequence[str]
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {i: {} for i in ids}
    for key, value in tensors.items():
        head, _, name = key.partition("/")
        if head in out and name:
            out[head][name] = value
    return out


def read_queries(path: Path) -> tuple[str, list[Query]]:
    where = f"{QUERIES_FORMAT} {path}"
    tensors, meta = _read(path)
    _header(meta, QUERIES_FORMAT, where)
    try:
        entries = json.loads(meta["queries"])
        kind = meta["kind"]
    except (KeyError, ValueError) as error:
        raise FormatError(f"{where}: metadata {error}") from error
    ids = [e["id"] for e in entries]
    parts = _split(tensors, ids)
    return kind, [Query(e["id"], dict(e["params"]), parts[e["id"]]) for e in entries]


def write_answers(path: Path, kind: str, queries_sha256: str, answers: Sequence[Answer]) -> None:
    tensors = {f"{a.id}/{name}": t for a in answers for name, t in a.tensors.items()}
    meta = {
        "format": ANSWERS_FORMAT,
        "version": "1",
        "kind": kind,
        "producer": "ftgs-harness",
        "queries_sha256": queries_sha256,
        "answers": json.dumps(
            [{"id": a.id, "status": a.status, "message": a.message} for a in answers]
        ),
    }
    _write(path, tensors, meta)


def read_answers(path: Path, *, kind: str, queries_sha256: str) -> dict[str, Answer]:
    """Read one kind's answers (Q-207), made for the queries file with `queries_sha256`."""
    where = f"{ANSWERS_FORMAT} {path}"
    tensors, meta = _read(path)
    _header(meta, ANSWERS_FORMAT, where)
    if meta.get("kind") != kind:
        raise FormatError(f"{where}: answers kind {meta.get('kind')!r}, expected {kind!r}")
    if meta.get("queries_sha256") != queries_sha256:
        raise FormatError(f"{where}: made for queries {meta.get('queries_sha256')!r}")
    try:
        entries = json.loads(meta["answers"])
    except (KeyError, ValueError) as error:
        raise FormatError(f"{where}: metadata {error}") from error
    statuses = ("ok", "error", "not_comparable")
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("id"), str) and e.get("status") in statuses
        for e in entries
    ):
        raise FormatError(f"{where}: answers must be a list of {{id, status, message}}")
    parts = _split(tensors, [e["id"] for e in entries])
    return {
        e["id"]: Answer(e["id"], e["status"], str(e.get("message", "")), parts[e["id"]])
        for e in entries
    }
