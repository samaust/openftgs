# SPDX-License-Identifier: Apache-2.0
"""The trunk networks' weights (`openftgs._weights`): downloaded at first use into torch.hub's
cache, checked against their SHA-256, the Caffe files converted (§2.3's pretrained-trunk row
and its note on weights; D-89's weights column; Q-7, Q-142, Q-T384.1; #384).

The module is private, so these are private tests. The CPU tests stand in a small file for a
publisher's; the `weights` tests download the six real files (Q-7)."""

import hashlib
import http.server
import re
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from openftgs import _weights


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Path]:
    """torch.hub's directory moved to an empty folder for one test, as test_bench's B-8 test
    does (TORCH_HOME is read only while no directory has been set)."""
    previous = torch.hub.get_dir()
    torch.hub.set_dir(str(tmp_path / "hub"))
    try:
        yield tmp_path / "hub"
    finally:
        torch.hub.set_dir(previous)


class _Files(http.server.BaseHTTPRequestHandler):
    """Serves the server's `files` by path, 404 for anything else."""

    def do_GET(self) -> None:
        body = self.server.files.get(self.path)  # type: ignore[attr-defined]
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def publisher() -> Iterator[tuple[str, dict[str, bytes]]]:
    """A publisher's host on the loopback address, at a free port: its base URL and the files
    it serves, which a test fills in."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    files: dict[str, bytes] = {}
    server.files = files  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", files
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Nothing listens on the discard port of the loopback address; a test whose download must not
# start uses it, so a contact would fail.
_UNUSED_URL = "http://127.0.0.1:9/standin.bin"


def test_fetch_refuses_a_cached_file_whose_hash_differs(hub: Path) -> None:
    """§2.3 (pinned by SHA-256), #384: a cached file of the right size whose SHA-256 isn't the
    pinned one is refused with an OSError naming it (Q-142's exit 2), and is left as it was."""
    publisher = b"the publisher's bytes, 32 long.."
    download = _weights.Download(_UNUSED_URL, len(publisher), _sha256(publisher))
    cached = hub / "checkpoints" / "standin.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x" * len(publisher))
    with pytest.raises(OSError, match=re.escape(str(cached))):
        _weights.fetch(download)
    assert cached.read_bytes() == b"x" * len(publisher)
    assert sorted(p.name for p in cached.parent.iterdir()) == ["standin.bin"]


def test_fetch_raises_an_oserror_when_the_download_fails(hub: Path) -> None:
    """Q-142, #384: a download that can't be made (nothing listens at the URL) raises an
    OSError naming the URL and the file, the error the CLI maps to exit 2, and leaves nothing
    in the cache."""
    download = _weights.Download(_UNUSED_URL, 32, _sha256(b"y" * 32))
    with pytest.raises(OSError, match=re.escape(_UNUSED_URL)) as raised:
        _weights.fetch(download)
    assert "standin.bin" in str(raised.value)
    checkpoints = hub / "checkpoints"
    assert not checkpoints.exists() or list(checkpoints.iterdir()) == []


def test_fetch_downloads_into_the_hub_cache_at_first_use(
    hub: Path, publisher: tuple[str, dict[str, bytes]]
) -> None:
    """§2.3 (downloaded at first use), #384: a file not yet cached is downloaded into
    torch.hub's checkpoint folder under the URL's file name, and a second fetch uses the cached
    file without contacting the host."""
    base, files = publisher
    body = bytes(range(256)) * 3
    files["/models/standin.bin"] = body
    download = _weights.Download(f"{base}/models/standin.bin", len(body), _sha256(body))
    path = _weights.fetch(download)
    assert path == hub / "checkpoints" / "standin.bin"
    assert path.read_bytes() == body
    assert sorted(p.name for p in path.parent.iterdir()) == ["standin.bin"]
    del files["/models/standin.bin"]  # the host no longer has it
    assert _weights.fetch(download) == path


@pytest.mark.parametrize("served", [b"z" * 768, b"z" * 767], ids=["altered", "truncated"])
def test_fetch_refuses_a_download_that_isnt_the_publishers_file(
    hub: Path, publisher: tuple[str, dict[str, bytes]], served: bytes
) -> None:
    """§2.3, Q-142, #384: the hosts of Oxford and BVLC serve plain HTTP, so the hash is the
    integrity check: a download whose bytes or byte count differ from the pinned file raises
    an OSError naming the URL, and nothing enters the cache."""
    base, files = publisher
    body = bytes(range(256)) * 3
    files["/standin.bin"] = served
    download = _weights.Download(f"{base}/standin.bin", len(body), _sha256(body))
    with pytest.raises(OSError, match=re.escape(download.url)):
        _weights.fetch(download)
    assert list((hub / "checkpoints").iterdir()) == []
