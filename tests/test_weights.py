# SPDX-License-Identifier: Apache-2.0
"""The trunk networks' weights (`openftgs._weights`): downloaded at first use into torch.hub's
cache, checked against their SHA-256, the Caffe files converted (§2.3's pretrained-trunk row
and its note on weights; D-89's weights column; Q-7, Q-142, Q-258; #384).

The module is private, so these are private tests. The CPU tests stand in a small file for a
publisher's; the `weights` tests download the six real files (Q-7)."""

import hashlib
import http.server
import os
import re
import struct
import threading
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest
import safetensors.torch
import torch

from openftgs import _weights
from openftgs.config import PercNet


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


# A protobuf encoder for the test's own Caffe files, written from the wire format (source log
# 441) and Caffe's field numbers (source-log row 422), so the reader is checked against an
# encoding it shares no code with.
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    """A length-delimited field (wire type 2)."""
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _int_field(number: int, value: int) -> bytes:
    """A varint field (wire type 0)."""
    return _varint(number << 3) + _varint(value)


def _blob(values: torch.Tensor, *, legacy: bool = False, unpacked: bool = False) -> bytes:
    """A BlobProto: BlobShape (7) with packed dims (1), or the legacy num, channels, height and
    width (1 to 4); data (5), packed or one fixed32 value at a time."""
    floats = values.reshape(-1).tolist()
    if legacy:
        dims = [1] * (4 - values.dim()) + list(values.shape)
        shape = b"".join(_int_field(n, d) for n, d in zip((1, 2, 3, 4), dims, strict=True))
    else:
        shape = _field(7, _field(1, b"".join(_varint(d) for d in values.shape)))
    if unpacked:
        data = b"".join(_varint(5 << 3 | 5) + struct.pack("<f", f) for f in floats)
    else:
        data = _field(5, struct.pack(f"<{len(floats)}f", *floats))
    return shape + data


def _conv(out: int, inp: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out, inp, k, k, generator=generator)
    return weight, torch.randn(out, generator=generator)


def test_caffe_layers_reads_every_layer_format_by_its_field_numbers() -> None:
    """Source-log row 422, #384: the reader takes the named layers' blobs from NetParameter's
    `layer` (100: LayerParameter name 1, blobs 7), from the legacy `layers` (2: V1LayerParameter
    name 4, blobs 6) and from a V0LayerParameter nested in one (V1's field 1: name 1, blobs
    50), with BlobShape dims or the legacy four, packed or unpacked data, and skips the rest."""
    a, b, c = _conv(4, 3, 3, seed=0), _conv(2, 4, 1, seed=1), _conv(3, 2, 5, seed=2)
    skipped = torch.ones(5)
    net = b"".join(
        [
            _int_field(99, 7),  # a field the reader doesn't know
            _field(
                100,
                _field(1, b"a")
                + _field(2, b"Convolution")
                + _field(7, _blob(a[0]))
                + _field(7, _blob(a[1])),
            ),
            _field(
                2,
                _field(4, b"b")
                + _int_field(5, 4)
                + _field(6, _blob(b[0], legacy=True))
                + _field(6, _blob(b[1], legacy=True)),
            ),
            _field(
                2,
                _field(
                    1,
                    _field(1, b"c")
                    + _field(2, b"conv")
                    + _field(50, _blob(c[0], unpacked=True))
                    + _field(50, _blob(c[1], legacy=True)),
                ),
            ),
            _field(100, _field(1, b"fc") + _field(7, _blob(skipped))),
        ]
    )
    found = _weights._caffe_layers(net, ["a", "b", "c"])
    assert list(found) == ["a", "b", "c"]
    for name, (weight, bias) in zip("abc", (a, b, c), strict=True):
        torch.testing.assert_close(found[name][0], weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(found[name][1].reshape(-1), bias, rtol=0.0, atol=0.0)


def _standin_net() -> tuple[bytes, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """A Caffe file in the legacy format of the two trunks' files, with two convolutions, a
    layer without blobs and a fully connected layer that the conversion leaves out."""
    convs = {"conv_a": _conv(4, 3, 3, seed=3), "conv_b": _conv(2, 4, 1, seed=4)}
    layers = [
        _field(
            2,
            _field(4, name.encode())
            + _field(6, _blob(w, legacy=True))
            + _field(6, _blob(b, legacy=True)),
        )
        for name, (w, b) in convs.items()
    ]
    layers.insert(1, _field(2, _field(4, b"relu_a")))
    fc = torch.ones(1, 1, 6, 8)
    layers.append(_field(2, _field(4, b"fc") + _field(6, _blob(fc, legacy=True))))
    return b"".join(layers), convs


def _standin_caffe(
    url: str, data: bytes, convs: dict[str, tuple[torch.Tensor, torch.Tensor]]
) -> "_weights._Caffe":
    """The stand-in's entry: its download, its convolutions' weight shapes and the SHA-256 of
    its float32 data, little-endian, each layer's weight then bias in order (Q-258)."""
    digest = hashlib.sha256()
    for weight, bias in convs.values():
        digest.update(weight.numpy().astype("<f4").tobytes())
        digest.update(bias.numpy().astype("<f4").tobytes())
    return _weights._Caffe(
        _weights.Download(url, len(data), _sha256(data)),
        {name: tuple(w.shape) for name, (w, _) in convs.items()},
        digest.hexdigest(),
    )


def test_caffe_tensors_converts_once_into_a_cached_safetensors_file(
    hub: Path, publisher: tuple[str, dict[str, bytes]]
) -> None:
    """#384, Q-258: a Caffe file is downloaded, converted to its convolutions' weights and
    biases under Caffe's layer names, and cached as safetensors beside it, with the source's
    SHA-256 in its metadata; a later load reads the converted file alone."""
    base, files = publisher
    data, convs = _standin_net()
    files["/standin.caffemodel"] = data
    caffe = _standin_caffe(f"{base}/standin.caffemodel", data, convs)
    expected = {}
    for name, (weight, bias) in convs.items():
        expected[f"{name}.weight"] = weight
        expected[f"{name}.bias"] = bias
    checkpoints = hub / "checkpoints"
    converted = checkpoints / "openftgs-standin.safetensors"

    def check(tensors: dict[str, torch.Tensor]) -> None:
        assert list(tensors) == list(expected)
        for key, value in expected.items():
            torch.testing.assert_close(tensors[key], value, rtol=0.0, atol=0.0)

    check(_weights._caffe_tensors("standin", caffe))
    assert sorted(p.name for p in checkpoints.iterdir()) == [converted.name, "standin.caffemodel"]
    with safetensors.safe_open(str(converted), framework="pt") as handle:
        assert handle.metadata()["source_sha256"] == _sha256(data)
    del files["/standin.caffemodel"]  # neither the host nor the cache has the source now
    (checkpoints / "standin.caffemodel").unlink()
    check(_weights._caffe_tensors("standin", caffe))


@pytest.mark.parametrize("content", ["other tensors", "not safetensors"])
def test_caffe_tensors_refuses_a_converted_file_that_isnt_the_conversion(
    hub: Path, content: str
) -> None:
    """§2.3, Q-142, Q-258: a cached converted file whose tensors aren't the pinned
    conversion, or that safetensors can't read, is refused with an OSError naming it, and left
    as it was; nothing is downloaded."""
    data, convs = _standin_net()
    caffe = _standin_caffe(_UNUSED_URL.replace(".bin", ".caffemodel"), data, convs)
    converted = hub / "checkpoints" / "openftgs-standin.safetensors"
    converted.parent.mkdir(parents=True)
    if content == "other tensors":
        tensors = {}
        for name, (weight, bias) in convs.items():
            tensors[f"{name}.weight"] = weight.clone()
            tensors[f"{name}.bias"] = bias.clone()
        tensors["conv_b.bias"][0] += 1.0
        safetensors.torch.save_file(tensors, str(converted))
    else:
        converted.write_bytes(b"\x10" * 64)
    before = converted.read_bytes()
    with pytest.raises(OSError, match=re.escape(str(converted))):
        _weights._caffe_tensors("standin", caffe)
    assert converted.read_bytes() == before
    assert [p.name for p in converted.parent.iterdir()] == [converted.name]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_caffe_tensors_reports_a_conversion_it_cannot_write_as_an_oserror(hub: Path) -> None:
    """Q-142, Q-158, Q-258 (owner answer, #407): a conversion that can't be written, here into
    a cache folder without write permission that holds the publisher's file, raises an OSError
    naming the converted file, whatever safetensors' own exception class, the CLI's exit 2;
    the cache is left as it was."""
    data, convs = _standin_net()
    caffe = _standin_caffe(_UNUSED_URL.replace(".bin", ".caffemodel"), data, convs)
    checkpoints = hub / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "standin.caffemodel").write_bytes(data)
    converted = checkpoints / "openftgs-standin.safetensors"
    checkpoints.chmod(0o555)
    try:
        with pytest.raises(OSError, match=re.escape(str(converted))):
            _weights._caffe_tensors("standin", caffe)
        assert [p.name for p in checkpoints.iterdir()] == ["standin.caffemodel"]
    finally:
        checkpoints.chmod(0o755)


# Source-log row 424: each publisher's file, its byte count and SHA-256, at the URL that rows
# 419, 421 and 440 give.
_PUBLISHED = {
    "vgg16_oxford": (
        "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel",
        553_432_081,
        "a6196bc498e45ea4cb2637114ae8db9410cf1556fd60a55ae93272371beba197",
    ),
    "alexnet_bvlc": (
        "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
        243_862_414,
        "4bff209a9837298157915ef50a4831a59636a6ca1a6b8ebacd990c3a5f3053e0",
    ),
    "dinov2_s14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        88_283_115,
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    ),
    "dinov2_b14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        346_378_731,
        "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    ),
    "dino_s16": (
        "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        86_710_517,
        "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7",
    ),
    "dino_b16": (
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        343_242_485,
        "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    ),
}


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def test_the_six_trunks_are_pinned_to_their_publishers_files() -> None:
    """D-89's weights column, source-log row 424: each trunk is pinned to its publisher's URL,
    byte count and SHA-256 (checked offline)."""
    assert _weights.TRUNKS == tuple(_PUBLISHED)
    for name, (url, size, sha256) in _PUBLISHED.items():
        assert _weights.DOWNLOADS[name] == _weights.Download(url, size, sha256)


def test_trunk_files_name_the_cached_files_each_trunk_builds_from(hub: Path) -> None:
    """Q-258, Q-253 (#389): a trunk's files in torch.hub's cache, any one of which builds it
    without a download, in the order the build looks for them: a Caffe trunk's conversion
    `openftgs-<perc_net>.safetensors`, then its publisher's file under the URL's name; a DINO
    trunk's checkpoint under the URL's name. Listing them reads, makes and downloads nothing;
    a weight-free network has none to list."""
    checkpoints = hub / "checkpoints"
    for name, (url, _, _) in _PUBLISHED.items():
        publisher = checkpoints / url.rsplit("/", 1)[1]
        converted = (checkpoints / f"openftgs-{name}.safetensors",)
        expected = (*converted, publisher) if name in _CAFFE_SHAPES else (publisher,)
        assert _weights.trunk_files(name) == expected, name
    assert not hub.exists()
    for name in ("flip", "haarpsi", "vgg"):
        with pytest.raises(ValueError, match=name):
            _weights.trunk_files(name)


def test_the_trunks_are_the_perc_net_values_with_weights() -> None:
    """§2.1 `PercNet`, D-89: the trunks carry `perc_net`'s names, every network of D-89's
    table but the weight-free ꟻLIP and HaarPSI, in the table's order."""
    weight_free = ("flip", "haarpsi")
    assert _weights.TRUNKS == tuple(n for n in typing.get_args(PercNet) if n not in weight_free)


@pytest.mark.weights
@pytest.mark.parametrize("name", list(_PUBLISHED))
def test_each_trunk_file_downloads_and_matches_its_recorded_sha256(name: str) -> None:
    """#384, Q-7: each of the six files downloads from its publisher into torch.hub's cache,
    with row 424's byte count and SHA-256, hashed here apart from the module."""
    url, size, sha256 = _PUBLISHED[name]
    path = _weights.fetch(_weights.DOWNLOADS[name])
    assert path == Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[1]
    assert path.stat().st_size == size
    assert _file_digest(path, "sha256") == sha256
    if name == "alexnet_bvlc":  # the publisher's own SHA-1, from its readme (row 421)
        assert _file_digest(path, "sha1") == "9116a64c0fbe4459d18f4bb6b56d647b63920377"


# The publishers' layer shapes [out, in / groups, kh, kw]: VGG16's configuration D (Simonyan
# and Zisserman; the author's prototxt, row 420); BVLC's AlexNet prototxt (row 421), whose
# conv2, conv4 and conv5 take two groups.
_CAFFE_SHAPES = {
    "vgg16_oxford": {
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
    "alexnet_bvlc": {
        "conv1": (96, 3, 11, 11),
        "conv2": (256, 48, 5, 5),
        "conv3": (384, 256, 3, 3),
        "conv4": (384, 192, 3, 3),
        "conv5": (256, 192, 3, 3),
    },
}


@pytest.mark.weights
@pytest.mark.parametrize("name", list(_CAFFE_SHAPES))
def test_converted_caffe_tensors_have_the_publishers_layer_shapes(name: str) -> None:
    """#384, D-89: a Caffe trunk's tensors are its convolutions' float32 weights and biases,
    under Caffe's layer names, in the publisher's layer shapes; the cached conversion equals
    one made afresh from the checked file."""
    tensors = _weights.trunk_tensors(name)
    expected: dict[str, tuple[int, ...]] = {}
    for layer, shape in _CAFFE_SHAPES[name].items():
        expected[f"{layer}.weight"] = shape
        expected[f"{layer}.bias"] = shape[:1]
    assert list(tensors) == list(expected)
    assert {key: tuple(t.shape) for key, t in tensors.items()} == expected
    assert {t.dtype for t in tensors.values()} == {torch.float32}
    assert all(bool(t.abs().sum() > 0) for t in tensors.values())
    caffe = _weights._CAFFE[name]
    afresh = _weights._convert(_weights.fetch(caffe.download), caffe)
    for key, value in tensors.items():
        torch.testing.assert_close(afresh[key], value, rtol=0.0, atol=0.0)


# The ViTs' widths and patch sizes (ViT-S 384, ViT-B 768; DINOv2's patch 14, DINO's 16), under
# the vendored DINOv2 layers' names (D-65), and their 12 blocks.
_VIT_EMBEDDING = {
    "dinov2_s14": (384, 3, 14, 14),
    "dinov2_b14": (768, 3, 14, 14),
    "dino_s16": (384, 3, 16, 16),
    "dino_b16": (768, 3, 16, 16),
}


@pytest.mark.weights
@pytest.mark.parametrize("name", list(_VIT_EMBEDDING))
def test_dino_checkpoints_load_as_tensors(name: str) -> None:
    """#384, D-89: a DINO trunk's checkpoint loads as a state dict of float32 tensors, its
    patch embedding of the model's width and patch size, with 12 blocks."""
    tensors = _weights.trunk_tensors(name)
    assert {t.dtype for t in tensors.values()} == {torch.float32}
    assert tuple(tensors["patch_embed.proj.weight"].shape) == _VIT_EMBEDDING[name]
    blocks = {key.split(".")[1] for key in tensors if key.startswith("blocks.")}
    assert blocks == {str(i) for i in range(12)}
