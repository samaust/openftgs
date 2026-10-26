# SPDX-License-Identifier: Apache-2.0
"""The run settings, `harness_config.json` and the record of a run (§3.3; Q-203).

Every problem here is a harness or configuration error, exit code 2 (§3.3 "Report format").
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_SIZES",
    "HarnessError",
    "RunRecord",
    "Scene",
    "Settings",
    "load_config",
    "load_settings",
    "read_run",
    "sizes_reduced",
    "write_run",
]

_RUN_FORMAT = "ftgs-harness-run"


class HarnessError(Exception):
    """A harness or configuration error: exit code 2."""


# Q-203: §3.3's case sizes, the defaults.
DEFAULT_SIZES = {"l0.models": 20, "l1.items": 200, "l1.probe_items": 21, "l2.queries": 4}


def sizes_reduced(sizes: Mapping[str, int]) -> bool:
    """Whether any case size is below §3.3's (Q-203); a size not given is §3.3's."""
    return any(sizes.get(k, v) < v for k, v in DEFAULT_SIZES.items())


@dataclass(frozen=True)
class Scene:
    """An L3 scene (Q-203): its manifest with the held-out split, the manifest training reads
    (the same one unless given) and its init file."""

    name: str
    manifest: Path
    train_manifest: Path
    init: Path


@dataclass(frozen=True)
class Settings:
    """The run settings (Q-203)."""

    seed: int = 0
    k: int = 5
    gpus: tuple[str, ...] = ()
    trained_models: tuple[Path, ...] = ()
    scenes: tuple[Scene, ...] = ()
    sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SIZES))

    @property
    def reduced(self) -> bool:
        """Whether any case size is below §3.3's (Q-203), the only way a run is reduced since
        Q-255 removed Q-212's `l2.lpips`."""
        return sizes_reduced(self.sizes)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["gpus"] = list(self.gpus)
        out["trained_models"] = [str(p) for p in self.trained_models]
        out["scenes"] = [{k: str(v) for k, v in asdict(s).items()} for s in self.scenes]
        out["reduced"] = self.reduced
        return out


def _int(value: Any, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise HarnessError(f"settings: {where} must be an integer ≥ {low}, got {value!r}")
    return value


def _path(value: Any, where: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise HarnessError(f"settings: {where} must be a path, got {value!r}")
    path = (base / value).resolve()
    if not path.exists():
        raise HarnessError(f"settings: {where} {path} does not exist")
    return path


def load_settings(path: Path | None) -> Settings:
    """The run settings of `path`, the defaults when None. Relative paths are relative to the
    settings file. Unknown keys and wrong types are refused (Q-203)."""
    if path is None:
        return Settings()
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise HarnessError(f"settings {path}: {error}") from error
    if not isinstance(raw, dict):
        raise HarnessError(f"settings {path}: not a JSON object")
    base = Path(path).resolve().parent
    known = {"seed", "k", "gpus", "l0", "l1", "l2", "l3"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise HarnessError(f"settings {path}: unknown keys {unknown}")
    seed = _int(raw.get("seed", 0), "seed", 0)
    k = _int(raw.get("k", 5), "k", 2)
    gpus = raw.get("gpus", [])
    if not isinstance(gpus, list) or not all(isinstance(g, str | int) for g in gpus):
        raise HarnessError(f"settings {path}: gpus must be a list of device ids")
    sizes = dict(DEFAULT_SIZES)
    sections: dict[str, dict[str, Any]] = {}
    for level, allowed in (
        ("l0", {"models", "trained_models"}),
        ("l1", {"items", "probe_items"}),
        ("l2", {"queries"}),  # Q-255: no `lpips`, which Q-212 had
        ("l3", {"scenes"}),
    ):
        section = raw.get(level, {})
        if not isinstance(section, dict) or set(section) - allowed:
            raise HarnessError(f"settings {path}: {level} may hold only {sorted(allowed)}")
        sections[level] = section
    for key in DEFAULT_SIZES:
        level, name = key.split(".")
        if name in sections[level]:
            sizes[key] = _int(sections[level][name], key, 1)
    trained = sections["l0"].get("trained_models", [])
    if not isinstance(trained, list):
        raise HarnessError(f"settings {path}: l0.trained_models must be a list")
    scenes = []
    entries = sections["l3"].get("scenes", [])
    if not isinstance(entries, list):
        raise HarnessError(f"settings {path}: l3.scenes must be a list")
    for i, entry in enumerate(entries):
        where = f"l3.scenes[{i}]"
        if not isinstance(entry, dict) or set(entry) - {
            "name",
            "manifest",
            "train_manifest",
            "init",
        }:
            raise HarnessError(f"settings {path}: {where} has unknown keys")
        scene_name = entry.get("name")
        if not isinstance(scene_name, str) or not scene_name.isidentifier():
            raise HarnessError(f"settings {path}: {where}.name must be an identifier")
        manifest = _path(entry.get("manifest"), f"{where}.manifest", base)
        train = entry.get("train_manifest")
        scenes.append(
            Scene(
                name=scene_name,
                manifest=manifest,
                train_manifest=manifest
                if train is None
                else _path(train, f"{where}.train_manifest", base),
                init=_path(entry.get("init"), f"{where}.init", base),
            )
        )
    if len({s.name for s in scenes}) != len(scenes):
        raise HarnessError(f"settings {path}: l3 scene names must be distinct")
    return Settings(
        seed=seed,
        k=k,
        gpus=tuple(str(g) for g in gpus),
        trained_models=tuple(_path(p, "l0.trained_models", base) for p in trained),
        scenes=tuple(scenes),
        sizes=sizes,
    )


def _finite(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise HarnessError(f"harness_config: {where} is not finite")
    if isinstance(value, dict):
        for k, v in value.items():
            _finite(v, f"{where}.{k}" if where else k)
    if isinstance(value, list):
        for v in value:
            _finite(v, where)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """`harness_config.json` and its SHA-256. It holds TrainConfig values nested by sub-config
    (§3.3, Q-10) and goes to the adapters verbatim; a `seed` key is refused, since every run
    gets its own (Q-203)."""
    try:
        data = Path(path).read_bytes()
        config = json.loads(data)
    except (OSError, ValueError) as error:
        raise HarnessError(f"harness_config {path}: {error}") from error
    if not isinstance(config, dict):
        raise HarnessError(f"harness_config {path}: not a JSON object")
    if "seed" in config:
        raise HarnessError(f"harness_config {path}: `seed` is set per run, not in the config")
    _finite(config, "")
    return config, hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """`run.json`: what `rerun` needs to replay a check (Q-203), with the hashes of the input
    files it must find unchanged (§3.3 symptom rule 6)."""

    a: str
    b: str
    config: Path
    config_sha256: str
    settings_path: Path | None
    settings_sha256: str | None
    levels: tuple[str, ...]
    jobs: int
    timeout: float
    started: str
    harness_version: str


def write_run(path: Path, record: RunRecord) -> None:
    body = {
        "format": _RUN_FORMAT,
        "version": 1,
        **{k: v for k, v in asdict(record).items()},
    }
    body["config"] = str(record.config)
    body["settings_path"] = None if record.settings_path is None else str(record.settings_path)
    body["levels"] = list(record.levels)
    Path(path).write_text(json.dumps(body, indent=2) + "\n")


def read_run(path: Path) -> RunRecord:
    try:
        body = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise HarnessError(f"run record {path}: {error}") from error
    if not isinstance(body, dict) or body.get("format") != _RUN_FORMAT or body.get("version") != 1:
        raise HarnessError(f"run record {path}: not a {_RUN_FORMAT} v1 file")
    settings = body["settings_path"]
    return RunRecord(
        a=body["a"],
        b=body["b"],
        config=Path(body["config"]),
        config_sha256=body["config_sha256"],
        settings_path=None if settings is None else Path(settings),
        settings_sha256=body["settings_sha256"],
        levels=tuple(body["levels"]),
        jobs=int(body["jobs"]),
        timeout=float(body["timeout"]),
        started=body["started"],
        harness_version=body["harness_version"],
    )
