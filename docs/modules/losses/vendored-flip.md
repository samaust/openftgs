# The vendored ꟻLIP in `openftgs._vendor.flip`

How §2.3's ꟻLIP row and D-89's `flip` construction are met with D-65's vendoring pattern: NVIDIA's PyTorch ꟻLIP loss at commit d3ab02a, copied into the package with changes that are device and caching only (D-89, U-23), so that it runs on the CPU as well as on CUDA and makes no host synchronization in a training iteration (§2.4 rule 3). `PerceptualLoss("flip")` and the perceptual metric are built on it (D-89, E16); this doc covers the copy itself.

## The reference copy

`tests/reference/flip` holds `flip_loss.py` and `LICENSE` exactly as NVIDIA publishes them at the commit, their SHA-256 recorded in source log rows 425 and 444 (Q-252). It is U-23's oracle: the test loads it by path, under a module name of its own, and compares the vendored loss with it on CUDA, where the published file runs. It is also the only source the vendoring script reads, so the copy can be rebuilt and checked offline. ruff, mypy and the SPDX rule of `tests/test_skeleton.py` leave it, and the vendored file, as upstream wrote them.

## The changes

`scripts/vendor_flip.py` makes `flip_loss.py` from the reference copy; nothing in the copy is edited by hand. Every line it changes ends with a comment `openftgs (<change>)`, and every block it adds starts with one. The edits are substring replacements confined to one function each, located with ast and counted, insertions before a named line, and two moves of contiguous lines into new functions; the script stops when any of them doesn't find what it expects, and checks that the result parses, names CUDA nowhere, has a marker in every block the diff adds, and keeps `.item()` only in the colour maximum and in HDR-ꟻLIP. The changes:

- **device**: device-agnostic tensors in place of the CUDA-only ones. Each `.cuda()` becomes `.to(<image>.device)` and each `device='cuda'` becomes `device=<image>.device`, `<image>` being the function's image argument; `generate_spatial_filter` takes the device as a new last argument.
- **filters**: the filters built once per device and size, as §2.4's perceptual-loss row says. A filter's size is set by the pixels per degree alone, so the key is the pixels per degree, the channel or feature type and the device. `generate_spatial_filter` is memoized, and the lines of `feature_detection` that build the edge and point filters move, unchanged but for the device change, into `generate_feature_filter`, memoized the same way.
- **constants**: the D65 illuminant and its inverse, and the two RGB–XYZ matrices, which `color_space_transform` copied to the GPU at every call, are copied once per device and kept. #380 doesn't list this change among the three it names; §2.4 rule 3 needs it, since each of those copies synchronizes (source log row 445), and it changes no numbers.
- **cmax**: the colour maximum, which `compute_ldrflip` read back to the host at every call, is computed once per device by `_cmax` and kept as a float. It is computed with the published expressions on the device it serves, because on the CPU they give a float32 one ulp from CUDA's: a value computed on one device and used on another would change the published file's numbers.
- **cache**: the two helpers these share. `_once_per_device` memoizes a builder by its arguments, the device last; `_constant` keeps a tensor per name and device.

HDR-ꟻLIP gets the device change only. It keeps its host read of the number of exposures, which sets its loop; nothing in openftgs uses it.

## The first call on a device

The first call of LDR-ꟻLIP on a device builds and copies its filters and the colour constants there and reads the colour maximum back once. Later calls on that device, at the same pixels per degree, copy and read nothing, forward or backward. A test of §2.4 rule 3, U-29's or one of its own, therefore warms the loss up with one call on the device before it starts recording, as `tests/test_vendored_flip.py` does.

Kept tensors are made outside inference mode, the matrices copied with `copy=True`, so a loss first called by a metric under `torch.inference_mode()` still serves a training step's backward pass in the same process. Without that, the next backward would fail on an inference tensor saved for it.

## VENDOR.json and the license

VENDOR.json records the upstream URL and commit, the script, the reference copy's folder, the license with its source and SHA-256, the published file's SHA-256 and the copy's, every edit in words with its count, a description of each change, what the first call on a device does, and the filter support below. ꟻLIP's BSD-3-Clause license sits beside the copy as `LICENSE`, the file keeps NVIDIA's header, and NOTICE names both copies. The license, the record and the copy ship in the wheel, and CI's wheel job checks them against VENDOR.json.

`python scripts/vendor_flip.py` rewrites the copy, its license and VENDOR.json, and leaves the package's own `__init__.py` alone; with `--check` it rebuilds them in a temporary folder and compares them with the committed ones byte for byte, and checks the SHA-256 that VENDOR.json records. Neither needs network.

## The filter support

Q-257 asks for ꟻLIP's filter support at E16's pixels per degree, measured with the copy; #387 applies it to Q-61. At the published default, (0.7 · 3840 / 0.7) · π / 180 = 67.02 pixels per degree, and at 67.0, which give the same filters, the three spatial filters reach 10 px from their centre and the edge and point filters 9 px, so the support is 21 px. Measured on the CPU, a one-pixel chromatic difference on a flat grey pair changes the error map within 10 px of it and no farther (the blue-yellow filter reaches that far); an achromatic one within 9 px, the achromatic filter's tail at 10 px being below float32's resolution. VENDOR.json's `filter_support` records the value, the method and the test that reproduces it, `test_flip_filter_support_at_67_pixels_per_degree`.

Unlike SSIM's valid window, the support doesn't bound the images ꟻLIP is computable on: the published filters pad by replicating the border, so it gives a finite value and gradient down to a single pixel, every filter reaching past the image below 21 px.

## Tests

`tests/test_vendored_flip.py`. On the CPU: the reference copy's hashes; VENDOR.json's; the check mode reproducing the copy and catching a changed copy or reference copy; the loss on CPU tensors, its value and gradient bitwise the same in two runs; the filters returned again on a second request; a loss first called in inference mode still training, in a fresh process so that every kept tensor starts unbuilt; the filter support. On a GPU: U-23's oracle against the reference copy in IEEE float32 (D-52), value and gradient, with HDR-ꟻLIP's three tone mappers beside it; the CPU and CUDA values agreeing; and, after one call, a forward and backward pass with no synchronization under the sync recorder. On the RTX 4090 the oracle's values were bitwise equal; its gradients differed as much as two runs of the published file differ from each other.

## Updating

Only by fetching the new commit's two files into the reference copy, updating the commit and the SHA-256 in the script after reading the license again, re-running the script, and passing these tests again (D-65).
