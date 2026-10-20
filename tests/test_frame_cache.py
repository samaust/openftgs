# SPDX-License-Identifier: Apache-2.0
"""U-30: the frame cache and the prefetcher of openftgs.data (D-51; Q-98, Q-99).

The oracle is the PNG interface itself: `MultiViewVideo.load_image` decodes each file, and the
cache and the prefetcher must give bitwise the same values (D-51 is exact). Scenes are random
8-bit images written by hand, so every value is known.
"""

import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from openftgs.data import FrameCache, MultiViewVideo, Prefetcher
from openftgs.io import write_scene_manifest
from openftgs.types import Camera, ImageRecord


def _camera(name: str, width: int, height: int) -> Camera:
    K = torch.tensor([[20.0, 0, width / 2], [0, 20.0, height / 2], [0, 0, 1]])
    w2c = torch.eye(4)
    w2c[0, 3] = float(len(name))
    return Camera(K=K, world_to_camera=w2c, width=width, height=height, name=name)


def _scene(root: Path, *, frames: int = 5) -> MultiViewVideo:
    """Two training cameras of different sizes and a test camera, random 8-bit RGB frames;
    the records interleave cameras, as converters' manifests needn't."""
    rng = np.random.default_rng(30)
    cameras = [_camera("a", 16, 12), _camera("bb", 20, 10), _camera("ccc", 16, 12)]
    records = []
    for k in range(frames):
        for camera in cameras:
            path = root / "images" / camera.name / f"{k:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, size=(camera.height, camera.width, 3), dtype=np.uint8)
            Image.fromarray(pixels, "RGB").save(path)
            split: Literal["train", "test"] = "test" if camera.name == "ccc" else "train"
            records.append(ImageRecord(camera, k / 30, path, split, k))
    return MultiViewVideo(write_scene_manifest(root / "scene.json", records))


def _training(video: MultiViewVideo) -> list[int]:
    return [i for i, r in enumerate(video.records) if r.split == "train"]


def test_u30_memory_mapped_cache_equals_png_decoding_for_every_image(tmp_path: Path) -> None:
    """U-30, D-51: every training image read from the memory-mapped cache, as uint8 and through
    the prefetcher as float32, is bitwise what PNG decoding gives."""
    video = _scene(tmp_path)
    cache = FrameCache.open(video, placement="mmap")
    assert cache.placement == "mmap"
    prefetcher = Prefetcher(cache, device="cpu", depth=3)
    order = _training(video)
    prefetcher.schedule(order)
    for index in order:
        want = video.load_image(index)
        assert torch.equal(cache.frame(index), (want * 255).round().to(torch.uint8))
        got = prefetcher.take(index)
        assert got.dtype == torch.float32 and torch.equal(got, want)
    prefetcher.close()


def test_u30_cache_files_follow_q98(tmp_path: Path) -> None:
    """Q-98: one safetensors file per camera with training images, in order of first
    appearance, beside the manifest under its SHA-256, holding uint8 [F, H, W, 3] in manifest
    order, with metadata naming the format, version, manifest, camera and records."""
    video = _scene(tmp_path, frames=3)
    directory = FrameCache.build(video)
    assert directory == tmp_path / ".ftgs-cache" / video.manifest.sha256
    assert directory == FrameCache.default_directory(video.manifest)
    assert sorted(p.name for p in directory.iterdir()) == ["0000.safetensors", "0001.safetensors"]
    with safe_open(directory / "0001.safetensors", framework="pt") as f:
        meta = f.metadata()
        frames = f.get_tensor("frames")
    assert meta["format"] == "ftgs-frame-cache" and meta["version"] == "1"
    assert meta["manifest_sha256"] == video.manifest.sha256 and meta["camera"] == "bb"
    records = [i for i, r in enumerate(video.records) if r.camera.name == "bb"]
    assert meta["records"] == str(records).replace(" ", "")
    assert frames.dtype == torch.uint8 and list(frames.shape) == [3, 10, 20, 3]
    assert FrameCache.size_of(video) == 3 * (12 * 16 + 10 * 20) * 3
    assert FrameCache.open(video, placement="mmap").nbytes >= FrameCache.size_of(video)


def test_u30_a_stale_cache_is_rebuilt(tmp_path: Path) -> None:
    """Q-98: a cache file whose metadata doesn't match the dataset is rebuilt on open."""
    video = _scene(tmp_path, frames=2)
    directory = FrameCache.build(video)
    stale = directory / "0000.safetensors"
    from safetensors.torch import save_file

    save_file(
        {"frames": torch.zeros(1, 1, 1, 3, dtype=torch.uint8)},
        str(stale),
        metadata={"format": "ftgs-frame-cache", "version": "1"},
    )
    cache = FrameCache.open(video, placement="mmap")
    index = _training(video)[0]
    assert torch.equal(cache.frame(index).float() / 255, video.load_image(index))


def test_u30_prefetch_loads_in_the_scheduled_order(tmp_path: Path) -> None:
    """U-30: the prefetcher reads frames in the order it is given, the sampler's order; a
    `take` out of that order is still answered, from the cache directly."""
    video = _scene(tmp_path)
    cache = FrameCache.open(video, placement="mmap")
    loaded: list[int] = []

    class Recording:
        def frame(self, index: int) -> torch.Tensor:
            loaded.append(index)
            return cache.frame(index)

    order = list(reversed(_training(video)))
    prefetcher = Prefetcher(Recording(), device="cpu", depth=4)  # type: ignore[arg-type]
    prefetcher.schedule(order)
    for index in order[:5]:
        assert torch.equal(prefetcher.take(index), video.load_image(index))
    assert loaded[:5] == order[:5]
    other = order[-1]
    assert torch.equal(prefetcher.take(other), video.load_image(other))
    prefetcher.close()
    synchronous = Prefetcher(cache, device="cpu", depth=0)
    assert torch.equal(synchronous.take(order[0]), video.load_image(order[0]))


@pytest.mark.gpu
def test_u30_gpu_placement_and_cuda_prefetch_equal_png_decoding(tmp_path: Path) -> None:
    """U-30 (GPU), D-51: the cache in GPU memory, and the memory-mapped cache through pinned
    buffers on a side stream, give bitwise PNG decoding's values on the device: read ahead
    (depth 3), read when taken on the calling thread (depth 0), and taken out of the scheduled
    order, which reads directly too."""
    video = _scene(tmp_path)
    order = _training(video)
    gpu = FrameCache.open(video, placement="gpu", device="cuda")
    on_gpu = Prefetcher(gpu, device="cuda", depth=2)
    mapped = Prefetcher(FrameCache.open(video, placement="mmap"), device="cuda", depth=3)
    synchronous = Prefetcher(FrameCache.open(video, placement="mmap"), device="cuda", depth=0)
    out_of_order = Prefetcher(FrameCache.open(video, placement="mmap"), device="cuda", depth=2)
    for prefetcher in (on_gpu, mapped, synchronous):
        prefetcher.schedule(order)
    out_of_order.schedule(order[::-1])
    for index in order:
        want = video.load_image(index).cuda()
        assert gpu.frame(index).device.type == "cuda"
        for prefetcher in (on_gpu, mapped, synchronous, out_of_order):
            got = prefetcher.take(index)
            assert got.device.type == "cuda" and torch.equal(got, want)
    torch.cuda.synchronize()
    for prefetcher in (on_gpu, mapped, synchronous, out_of_order):
        prefetcher.close()


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [0, 2])
def test_u30_a_frame_read_on_the_calling_thread_waits_on_no_earlier_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    """§2.4 rule 3, D-51 (PRD #220 review): a frame the prefetcher reads on the calling thread,
    the training thread in a run (every frame at depth 0; at depth 2, every frame taken out of
    the scheduled order), never waits on the host for an earlier frame's copy: no CUDA event is
    synchronized on that thread, over two passes, so pinned memory is reused. The frames stay
    bitwise PNG decoding's."""
    waits: list[int] = []
    synchronize = torch.cuda.Event.synchronize

    def recorded(self: torch.cuda.Event) -> None:
        waits.append(threading.get_ident())
        synchronize(self)

    monkeypatch.setattr(torch.cuda.Event, "synchronize", recorded)
    video = _scene(tmp_path)
    order = _training(video)
    prefetcher = Prefetcher(FrameCache.open(video, placement="mmap"), device="cuda", depth=depth)
    for _ in range(2):
        prefetcher.schedule(order[::-1])  # every take below is out of the scheduled order
        for index in order:
            got = prefetcher.take(index)
            assert torch.equal(got, video.load_image(index).cuda())
    torch.cuda.synchronize()
    prefetcher.close()
    assert threading.get_ident() not in waits


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_u30_gpu_placement_takes_frames_without_synchronizing(tmp_path: Path, device: str) -> None:
    """D-51, §2.4 rule 3: a cache already in GPU memory is used in place, so taking a frame
    makes no host–device synchronization, whether the device names its index or not (the
    trainer's default is "cuda"; M12 review F-1)."""
    video = _scene(tmp_path)
    order = _training(video)
    cache = FrameCache.open(video, placement="gpu", device=device)
    prefetcher = Prefetcher(cache, device=device, depth=0)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        images = [prefetcher.take(index) for index in order]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for index, image in zip(order, images, strict=True):
        assert torch.equal(image, video.load_image(index).cuda())
    prefetcher.close()
