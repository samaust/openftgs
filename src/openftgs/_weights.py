# SPDX-License-Identifier: Apache-2.0
"""The trunk networks' pretrained weights (D-89's weights column): downloaded at first use from
their publishers' hosts into torch.hub's cache, never redistributed, each file checked against
its pinned SHA-256 (§2.3's pretrained-trunk row and its note on weights).

Spec: §2.3, D-89. A private top-level module that imports nothing from openftgs, so
`openftgs.losses`, which builds the networks, may import it (§2.1 bands; #384).
"""

import hashlib
import http.client
import os
import pathlib
import urllib.parse
import uuid
from dataclasses import dataclass

import torch

__all__ = ["Download", "fetch"]


@dataclass(frozen=True)
class Download:
    """A publisher's file: the URL it is served from, its byte count and its SHA-256."""

    url: str
    size: int
    sha256: str

    @property
    def file(self) -> str:
        """The file name torch.hub's cache keeps it under: the URL's last path segment."""
        return pathlib.PurePosixPath(urllib.parse.urlparse(self.url).path).name


# Q-T384.1: each file under its publisher's name in torch.hub's checkpoint folder, as RoMa's.
def _checkpoints() -> pathlib.Path:
    """torch.hub's checkpoint folder, read at each call so a `torch.hub.set_dir` applies."""
    return pathlib.Path(torch.hub.get_dir()) / "checkpoints"


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _difference(path: pathlib.Path, download: Download) -> str | None:
    """How the file at `path` differs from the publisher's, or None when it is the publisher's:
    its byte count first, which costs nothing, then its SHA-256."""
    size = path.stat().st_size
    if size != download.size:
        return f"{size} bytes, not {download.size}"
    found = _file_sha256(path)
    if found != download.sha256:
        return f"SHA-256 {found}, not {download.sha256}"
    return None


def _download(download: Download, path: pathlib.Path) -> None:
    """The file downloaded beside `path` and moved there only once it is checked, so the cache
    never holds a partial or altered download (Q-T384.1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.download")
    try:
        try:
            torch.hub.download_url_to_file(download.url, str(partial), progress=True)
        except (OSError, http.client.HTTPException) as error:
            # Q-142: a failed run-time download is an environment error (exit 2).
            raise OSError(f"cannot download {download.url} into {path}: {error}") from error
        difference = _difference(partial, download)
        if difference is not None:
            raise OSError(f"the download of {download.url} into {path} has {difference}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


# Q-T384.1: checked before every use; a cached file that fails is refused, never replaced.
def fetch(download: Download) -> pathlib.Path:
    """The publisher's file in torch.hub's cache, downloaded at first use, and checked against
    its byte count and SHA-256 (§2.3). Raises OSError when the download fails or the file isn't
    the publisher's (Q-142: exit 2); a cached file that fails is left for the user to delete."""
    path = _checkpoints() / download.file
    if not path.exists():
        _download(download, path)
        return path
    difference = _difference(path, download)
    if difference is not None:
        raise OSError(
            f"{path} has {difference} (the file of {download.url}); delete it to download it again"
        )
    return path
