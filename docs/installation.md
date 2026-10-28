# Installation

openftgs needs Python 3.11 or later (Q-1) and, to train and to render with gsplat, an NVIDIA GPU
with a CUDA build of PyTorch.

## The package

```bash
pip install openftgs
```

This installs the core dependencies of spec §2.3: PyTorch (≥ 2.1), gsplat 1.5.3, safetensors,
PyYAML, Pillow, SciPy and NumPy. The default perceptual network, ꟻLIP, is vendored in the package
(D-65), and HaarPSI and the two Caffe trunks need nothing more.

### A CUDA build of PyTorch first

Training needs a CUDA build of PyTorch (Q-3). On Linux x86_64 the default `torch` wheel on PyPI
already is one (torch 2.14.1 bundles the CUDA 13.0 libraries), so `pip install openftgs` installs
it. For another CUDA version, or on Windows, install torch from the
[PyTorch index](https://pytorch.org/get-started/locally/) first, then openftgs, which keeps the
torch it finds. A CPU build is enough for the reference backend (`ftgs render --backend
reference`) and for the tests that use it.

## Optional extras

### `[init]`: RoMa initialization and the DINO networks

```bash
pip install "openftgs[init]"
```

`ftgs init roma` matches the training views with RoMa, whose inference code openftgs vendors as
`openftgs._vendor.romatch` (D-65), with its license files. The extra adds the libraries that code
imports (torchvision, einops and loguru), and tqdm, which XFeat's code imports when
`RomaInitConfig.model = "tiny_roma_v1_outdoor"` loads it through `torch.hub` (Q-T370.1), and
requires torch ≥ 2.5.1. The matcher downloads its weights through `torch.hub` the first time it
runs, and needs a CUDA device.

The four DINO perceptual networks (`dinov2_s14`, `dinov2_b14`, `dino_s16` and `dino_b16` as
`loss.perc_net`) run on the vendored RoMa's DINOv2 layers, so they need the extra too: without
it, building one is an error that names it (Q-248). ꟻLIP, the default perceptual network,
HaarPSI and the two Caffe trunks (`vgg16_oxford`, `alexnet_bvlc`) need no extra.

RoMa's optional speed-up, [fused-local-corr](https://github.com/Parskatt/fused-local-corr) (MIT),
isn't part of the extra (D-85): each of its releases since 0.3.27 requires one exact torch
release, so installing one would replace your torch. On Linux, build it from source against the
torch you have; this needs the CUDA toolkit (`nvcc`) and a CUDA build of PyTorch already
installed:

```bash
pip install --no-build-isolation git+https://github.com/Parskatt/fused-local-corr@689f2f7303ca0d97ac1f3d1364c732c6d8e889a8
```

Without it, RoMa runs with its `use_custom_corr` option off and computes the local correlation in
plain PyTorch, which is slower.

### `[fast]`: the fused SSIM loss

`[fast]` is empty for now (Q-2). The faster SSIM loss uses
[fused-ssim](https://github.com/rahul-goel/fused-ssim) (MIT), which is not on PyPI. Install it from
its repository, at the commit openftgs is tested with; this needs the CUDA toolkit and a CUDA build
of PyTorch already installed:

```bash
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
```

Without it, the SSIM loss runs in plain PyTorch and gives the same values.

## Programs you install yourself

openftgs calls two programs as separate processes; it neither bundles nor links them.

- **ffmpeg 5.1 or later**, with its `ffprobe` (D-48). `ftgs convert` decodes the input videos with
  it; 5.1 is the first release with `-fps_mode`. `--ffmpeg PATH` and `--ffprobe PATH` choose the
  programs, which are otherwise taken from `PATH`. A missing ffmpeg or ffprobe is exit code 2.
- **COLMAP** (D-60), only to convert SelfCap as its protocol prescribes: `ftgs convert selfcap`
  undistorts with COLMAP's `image_undistorter`, which needs its `blank_pixels` option.
  `--undistort internal` uses openftgs's own undistortion instead, which is not protocol-exact.

## Pretrained weights

openftgs ships no weight file. Weights reach your machine at run time, from their projects' own
hosts, into torch.hub's cache (spec §2.3):

- the weights of the six trunk perceptual networks, downloaded the first time `loss.perc_net`
  selects one: Oxford's VGG16 and BVLC's AlexNet, as Caffe files that are converted once into a
  safetensors file beside them, and DINOv2 ViT-S/14 and ViT-B/14 and DINO ViT-S/16 and ViT-B/16.
  Each file is checked against its byte count and SHA-256 before every use; one that fails is
  refused and left in place, never overwritten (Q-258). ꟻLIP, the default perceptual network and
  the reported perceptual metric, and HaarPSI need no weights, so the default training run and
  `ftgs eval` download nothing;
- RoMa's outdoor model and its DINOv2 ViT-L/14 backbone, at the first `ftgs init roma`.

Each weight's terms are recorded apart from the code licenses, in the release checklist kept with
the clean-room records.

## Checking the installation

```bash
ftgs --help
```

lists the commands; {doc}`cli` documents each of them.
