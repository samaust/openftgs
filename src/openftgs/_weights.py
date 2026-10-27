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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import safetensors
import safetensors.torch
import torch
from torch import Tensor

__all__ = ["DOWNLOADS", "TRUNKS", "Download", "fetch", "trunk_files", "trunk_tensors"]


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


# Q-258: each file under its publisher's name in torch.hub's checkpoint folder, as RoMa's.
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
    never holds a partial or altered download (Q-258)."""
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


# Q-258: checked before every use; a cached file that fails is refused, never replaced.
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


# The Caffe reader is adapted from #374's prototype `converters.py` (the owner's Apache-2.0
# research code, SHA-256 4f11b618e8f4ec5644ee9e2264640cd5207573e792908d5b5cfa1768daf41c69;
# source log 438). It reads protobuf's wire format itself, with the field numbers of Caffe's
# caffe.proto (BSD-2-Clause; source-log row 422); neither protobuf nor caffe.proto is used.
def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    """The varint at `pos` and the position after it."""
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("a truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    """One message's fields as (field number, wire type, value): an int for a varint, else a
    view of the field's bytes (length-delimited, 64-bit or 32-bit), copying nothing."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
            yield number, wire, value
            continue
        if wire == 2:
            size, pos = _varint(buf, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        if pos + size > len(buf):
            raise ValueError("a truncated field")
        yield number, wire, buf[pos : pos + size]
        pos += size


def _blob(buf: memoryview) -> Tensor:
    """A BlobProto as a float32 tensor: its shape from BlobShape (7: dim 1, plain or packed),
    else from the legacy num, channels, height and width (1 to 4); its data (5), packed or one
    32-bit value at a time; double_data (8) is refused."""
    legacy = [1, 1, 1, 1]
    shape: list[int] = []
    chunks: list[np.ndarray] = []
    for number, wire, value in _fields(buf):
        if number in (1, 2, 3, 4) and isinstance(value, int):
            legacy[number - 1] = value
        elif number == 7 and not isinstance(value, int):
            for dim_number, dim_wire, dim in _fields(value):
                if dim_number != 1:
                    continue
                if isinstance(dim, int):
                    shape.append(dim)
                elif dim_wire == 2:  # packed dims
                    pos = 0
                    while pos < len(dim):
                        size, pos = _varint(dim, pos)
                        shape.append(size)
        elif number == 5 and wire in (2, 5) and not isinstance(value, int):
            chunks.append(np.frombuffer(value, dtype="<f4"))
        elif number == 8:
            raise ValueError("a blob of double_data, which the trunks' files don't hold")
    data = np.concatenate(chunks) if chunks else np.zeros(0, dtype="<f4")
    # astype copies, so the tensor owns its memory, in the host's byte order.
    return torch.from_numpy(data.astype(np.float32).reshape(shape or legacy))


def _layer(buf: memoryview, *, legacy: bool) -> tuple[str | None, list[memoryview]]:
    """A layer's name and its blobs, unread: LayerParameter's name 1 and blobs 7, or
    V1LayerParameter's name 4 and blobs 6, with a V0LayerParameter as its field 1 (name 1,
    blobs 50), whose name counts only when V1's is absent."""
    name_field, blob_field = (4, 6) if legacy else (1, 7)
    name: str | None = None
    v0_name: str | None = None
    blobs: list[memoryview] = []
    for number, _, value in _fields(buf):
        if isinstance(value, int):
            continue
        if number == name_field:
            name = bytes(value).decode()
        elif number == blob_field:
            blobs.append(value)
        elif legacy and number == 1:
            for v0_number, _, v0_value in _fields(value):
                if isinstance(v0_value, int):
                    continue
                if v0_number == 1:
                    v0_name = bytes(v0_value).decode()
                elif v0_number == 50:
                    blobs.append(v0_value)
    return (name if name is not None else v0_name), blobs


def _caffe_layers(data: bytes, names: Sequence[str]) -> dict[str, list[Tensor]]:
    """The blobs of the named layers of a serialized NetParameter, in the order of `names`, from
    its `layer` (100) and legacy `layers` (2) fields; other layers' blobs aren't read."""
    wanted = set(names)
    found: dict[str, list[Tensor]] = {}
    for number, _, value in _fields(memoryview(data)):
        if number not in (100, 2) or isinstance(value, int):
            continue
        name, blobs = _layer(value, legacy=number == 2)
        if name in wanted:
            found[name] = [_blob(blob) for blob in blobs]
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"the Caffe file has no layer {', '.join(missing)}")
    return {name: found[name] for name in names}


@dataclass(frozen=True)
class _Caffe:
    """A Caffe trunk: its publisher's file, each convolution's weight shape [out, in / groups,
    kh, kw] under its Caffe layer name in the network's order, and the SHA-256 of the
    converted float32 data, little-endian, each layer's weight then bias in that order."""

    download: Download
    shapes: dict[str, tuple[int, ...]]
    sha256: str


def _converted_difference(tensors: dict[str, Tensor], caffe: _Caffe) -> str | None:
    """How converted tensors differ from the pinned conversion, or None: their keys, shapes and
    dtype against the layer table, then the SHA-256 of their data (Q-258)."""
    keys = [f"{name}.{part}" for name in caffe.shapes for part in ("weight", "bias")]
    if sorted(tensors) != sorted(keys):
        return f"the tensors {sorted(tensors)}, not {sorted(keys)}"
    digest = hashlib.sha256()
    for name, shape in caffe.shapes.items():
        for part, expected in (("weight", shape), ("bias", shape[:1])):
            tensor = tensors[f"{name}.{part}"]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != expected:
                found = f"{tensor.dtype} {list(tensor.shape)}"
                return f"{name}.{part} of {found}, not torch.float32 {list(expected)}"
            digest.update(tensor.contiguous().numpy().astype("<f4", copy=False).tobytes())
    if digest.hexdigest() != caffe.sha256:
        return f"tensors of SHA-256 {digest.hexdigest()}, not {caffe.sha256}"
    return None


def _convert(source: pathlib.Path, caffe: _Caffe) -> dict[str, Tensor]:
    """The convolutions' weights and biases of a checked Caffe file, under Caffe's layer names."""
    layers = _caffe_layers(source.read_bytes(), list(caffe.shapes))
    tensors: dict[str, Tensor] = {}
    for name, blobs in layers.items():
        if len(blobs) != 2:
            raise ValueError(
                f"layer {name} of {source} has {len(blobs)} blobs, not a weight and a bias"
            )
        tensors[f"{name}.weight"] = blobs[0]
        tensors[f"{name}.bias"] = blobs[1].reshape(-1)  # a legacy blob holds it as [1, 1, 1, out]
    return tensors


def _converted(name: str) -> pathlib.Path:
    """Q-258: where a Caffe trunk's conversion is cached, beside its source."""
    return _checkpoints() / f"openftgs-{name}.safetensors"


# Q-258: converted once into `openftgs-<name>.safetensors` beside its source, and checked
# against the pinned conversion at every load.
def _caffe_tensors(name: str, caffe: _Caffe) -> dict[str, Tensor]:
    """A Caffe trunk's convolutions, from its converted file in torch.hub's cache, made at
    first use from the publisher's checked file. Raises OSError when a download fails, a
    cached file isn't the publisher's or the pinned conversion, or the conversion can't be
    written (Q-142: exit 2)."""
    path = _converted(name)
    if not path.exists():
        source = fetch(caffe.download)
        tensors = _convert(source, caffe)
        difference = _converted_difference(tensors, caffe)
        if difference is not None:  # the source is the publisher's, so the converter is wrong
            raise RuntimeError(f"converting {source} gave {difference}")
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        metadata = {"source": caffe.download.url, "source_sha256": caffe.download.sha256}
        try:
            safetensors.torch.save_file(tensors, str(partial), metadata=metadata)
            os.replace(partial, path)
        except safetensors.SafetensorError as error:
            # Q-158, Q-258: the library reports a write that fails (a full disk, no permission)
            # through its own class, not an OSError (source log #148, #465); Q-142: exit 2.
            raise OSError(f"cannot write {path}, the conversion of {source}: {error}") from error
        finally:
            partial.unlink(missing_ok=True)
        return tensors
    try:
        loaded = safetensors.torch.load_file(str(path))
        difference = _converted_difference(loaded, caffe)
    except safetensors.SafetensorError as error:
        difference = f"content safetensors can't read ({error})"
    if difference is not None:
        raise OSError(
            f"{path} has {difference}, so it isn't the conversion of {caffe.download.url}; "
            "delete it to convert it again"
        )
    return {key: loaded[key] for name in caffe.shapes for key in (f"{name}.weight", f"{name}.bias")}


def _dino_tensors(download: Download) -> dict[str, Tensor]:
    """A DINO trunk's checkpoint, checked, then loaded on the CPU as the state dict its
    publisher saved. Only torch reads it: the vendored DINOv2 layers, which need `[init]`
    (Q-248), are the network's builder's to import."""
    path = fetch(download)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise RuntimeError(f"{path} holds no state dict of tensors")  # the publisher's file
    return state


# D-89's weights column. URLs: source-log rows 419 (Oxford), 421 (BVLC) and 440 (Meta);
# byte counts and SHA-256: row 424.
_VGG16 = Download(
    "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel",
    553_432_081,
    "a6196bc498e45ea4cb2637114ae8db9410cf1556fd60a55ae93272371beba197",
)
_ALEXNET = Download(
    "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
    243_862_414,
    "4bff209a9837298157915ef50a4831a59636a6ca1a6b8ebacd990c3a5f3053e0",
)
# The convolutions D-89's distance reads, in their publishers' shapes (VGG16's configuration
# D, row 420; AlexNet's prototxt, row 421, with conv2, conv4 and conv5 in two groups). The
# converted data's SHA-256 is that of the conversion of the checked files (source log 442).
_CAFFE: dict[str, _Caffe] = {
    "vgg16_oxford": _Caffe(
        _VGG16,
        {
            "conv1_1": (64, 3, 3, 3),
            "conv1_2": (64, 64, 3, 3),
            "conv2_1": (128, 64, 3, 3),
            "conv2_2": (128, 128, 3, 3),
            "conv3_1": (256, 128, 3, 3),
            "conv3_2": (256, 256, 3, 3),
            "conv3_3": (256, 256, 3, 3),
            "conv4_1": (512, 256, 3, 3),
            "conv4_2": (512, 512, 3, 3),
            "conv4_3": (512, 512, 3, 3),
            "conv5_1": (512, 512, 3, 3),
            "conv5_2": (512, 512, 3, 3),
            "conv5_3": (512, 512, 3, 3),
        },
        "dd21191fd3ba4d9b2b0104621c806227994d029acdf7f88d3ca0afd5774c15cf",
    ),
    "alexnet_bvlc": _Caffe(
        _ALEXNET,
        {
            "conv1": (96, 3, 11, 11),
            "conv2": (256, 48, 5, 5),
            "conv3": (384, 256, 3, 3),
            "conv4": (384, 192, 3, 3),
            "conv5": (256, 192, 3, 3),
        },
        "b5f5f475447b7cee594a9708d7d012ef711a4768fee44dfb9dbc93c06c21cf48",
    ),
}
_DINO: dict[str, Download] = {
    "dinov2_s14": Download(
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        88_283_115,
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    ),
    "dinov2_b14": Download(
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        346_378_731,
        "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    ),
    "dino_s16": Download(
        "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        86_710_517,
        "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7",
    ),
    "dino_b16": Download(
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        343_242_485,
        "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    ),
}

# The six trunk networks of D-89, in its table's order, and each one's publisher's file.
TRUNKS: tuple[str, ...] = (*_CAFFE, *_DINO)
DOWNLOADS: dict[str, Download] = {
    **{name: caffe.download for name, caffe in _CAFFE.items()},
    **_DINO,
}


# Q-253, Q-270: B-4 never downloads, so it needs to know where a trunk is cached.
def trunk_files(name: str) -> tuple[pathlib.Path, ...]:
    """The files in torch.hub's cache, any one of which builds the trunk network `name` (one
    of `TRUNKS`) without a download, in the order `trunk_tensors` looks for them: a Caffe
    trunk's conversion, then its publisher's file, from which the conversion is made; a DINO
    trunk's checkpoint (Q-258). Nothing is read, made or downloaded: whether a file is there is
    the caller's question, and its byte count and SHA-256 are checked when the network is
    built. Raises ValueError for another name."""
    if name in _CAFFE:
        return _converted(name), _checkpoints() / _CAFFE[name].download.file
    if name in _DINO:
        return (_checkpoints() / _DINO[name].file,)
    raise ValueError(f"no trunk network {name!r}; the trunks are {', '.join(TRUNKS)}")


def trunk_tensors(name: str) -> dict[str, Tensor]:
    """The pretrained tensors of the trunk network `name` (one of `TRUNKS`), float32 on the
    CPU, fetched at first use into torch.hub's cache and checked against their pinned SHA-256
    (§2.3, D-89):

    - `vgg16_oxford`, `alexnet_bvlc`: the convolutions' `<layer>.weight` [out, in / groups,
      kh, kw] and `<layer>.bias` [out] under Caffe's layer names (conv1_1 to conv5_3; conv1 to
      conv5), in the network's order, as the publisher's Caffe file holds them (BGR input
      channels); converted at first use and cached as safetensors.
    - `dinov2_s14`, `dinov2_b14`, `dino_s16`, `dino_b16`: the publisher's state dict as saved.

    Raises ValueError for another name, and OSError when a download fails, a cached file isn't
    the publisher's or its pinned conversion, or the conversion can't be written (Q-142: the
    CLI's exit 2)."""
    if name in _CAFFE:
        return _caffe_tensors(name, _CAFFE[name])
    if name in _DINO:
        return _dino_tensors(_DINO[name])
    raise ValueError(f"no trunk network {name!r}; the trunks are {', '.join(TRUNKS)}")
