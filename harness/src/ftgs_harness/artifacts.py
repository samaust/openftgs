# SPDX-License-Identifier: Apache-2.0
"""The report's artifacts: diff images and plots, drawn by matplotlib on its Agg backend (owner
decision, 2026-10-11; §3.3 "Report format")."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_cdfs", "plot_lines", "save_pair"]


def _image(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(rgb.astype(np.float64), nan=1.0), 0.0, 1.0)


def save_pair(run_dir: Path, stem: str, rgb_a: np.ndarray, rgb_b: np.ndarray) -> list[str]:
    """A's and B's renders and |A − B| (scaled to its maximum) as PNGs under
    `run_dir/artifacts/`; returns their paths relative to `run_dir`."""
    folder = run_dir / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    diff = np.abs(np.nan_to_num(rgb_a.astype(np.float64)) - np.nan_to_num(rgb_b.astype(np.float64)))
    peak = float(diff.max())
    paths = []
    for suffix, image in (
        ("A", _image(rgb_a)),
        ("B", _image(rgb_b)),
        ("absdiff", diff / peak if peak > 0 else diff),
    ):
        path = folder / f"{stem}_{suffix}.png"
        plt.imsave(path, np.clip(image, 0.0, 1.0))
        paths.append(str(path.relative_to(run_dir)))
    return paths


def plot_lines(
    run_dir: Path,
    stem: str,
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> str:
    """A line plot of one or more series against x; returns its path relative to `run_dir`."""
    folder = run_dir / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    for name, ys in series.items():
        ax.plot(list(x), list(ys), marker="o", markersize=3, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = folder / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(run_dir))


def plot_cdfs(
    run_dir: Path, stem: str, samples: dict[str, np.ndarray], *, xlabel: str, title: str = ""
) -> str:
    """Empirical CDFs of pooled samples, one line per series; returns the path relative to
    `run_dir`."""
    folder = run_dir / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    for name, values in samples.items():
        xs = np.sort(np.asarray(values, dtype=np.float64).ravel())
        if xs.size:
            ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = folder / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(run_dir))
