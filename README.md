openftgs is an independent clean-room implementation of FreeTimeGS (Wang et al., CVPR 2025), not affiliated with its authors.

# openftgs

Python package `openftgs`, under the Apache-2.0 license, with the command-line tool `ftgs`.

**Status: pre-alpha.** The library and its command line are built through milestone M14; the performance work, the differential harness and the validation runs follow (`docs/clean-room/milestones.md`).

## Clean-room process

The code is written from the paper (arXiv 2506.05348 v2) and from the owner's specification, which was itself written from the paper alone. No code, notebook, checkpoint or issue thread from any other implementation of the method, including the authors' own, is consulted. Third-party software is read only within its license: source code only under permissive licenses that allow commercial use, documentation only for everything else. Every source consulted is logged.

- [`docs/spec/`](docs/spec/): the specification, the source of truth for the implementation.
- [`docs/clean-room/source-log.md`](docs/clean-room/source-log.md): every source consulted during the implementation and what was taken from it.
- [`docs/clean-room/questions.md`](docs/clean-room/questions.md): gaps found in the spec and the provisional decisions made for them.
- [`docs/clean-room/milestones.md`](docs/clean-room/milestones.md): the order in which the package is built.

## Installation

```bash
pip install -e .[dev]
```

Training needs a CUDA build of PyTorch. On Linux x86_64 the default `torch` wheel on PyPI already is one (torch 2.14.1 bundles the CUDA 13.0 libraries), so the command above installs it. For another CUDA version, or on Windows, install torch from the [PyTorch index](https://pytorch.org/get-started/locally/) before installing openftgs. A CPU build is enough for the tests on the reference backend; CI uses one because its runners have no GPU.

Optional extras:

- `[init]`: the dependencies of the RoMa-based 4D initialization. RoMa's optional speed-up, [fused-local-corr](https://github.com/Parskatt/fused-local-corr) (MIT), isn't part of it: every release since 0.3.27 requires one exact torch release (0.3.211 requires torch 2.11.0), so installing one would replace your torch, and the newest unpinned wheel, 0.2.4, built for CUDA 12, doesn't load with torch 2.14.1 (D-85). On Linux, build it from source against the torch you have; this needs the CUDA toolkit (`nvcc`) and a CUDA build of PyTorch already installed:

  ```bash
  pip install --no-build-isolation git+https://github.com/Parskatt/fused-local-corr@689f2f7303ca0d97ac1f3d1364c732c6d8e889a8
  ```

  Without it, RoMa runs with its `use_custom_corr` option off and computes the local correlation in plain PyTorch, which is slower.
- `[fast]`: currently empty. The faster SSIM loss uses [fused-ssim](https://github.com/rahul-goel/fused-ssim) (MIT), which is not on PyPI. Install it from its repository, at the commit openftgs is tested with; this needs the CUDA toolkit (`nvcc`) and a CUDA build of PyTorch already installed:

  ```bash
  pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
  ```

  Without it, the SSIM loss runs in plain PyTorch and gives the same values.

The differential harness is a separate package in `harness/` (`pip install -e harness`).

## Command line

```bash
ftgs convert n3dv SRC OUT                        # a dataset into a scene manifest (needs ffmpeg)
ftgs cache OUT/scene.json                        # decode the training frames once (optional)
ftgs init roma OUT/scene.json init.safetensors   # initial points with RoMa ([init], a CUDA device)
ftgs train OUT/scene.json --init init.safetensors --out RUN_DIR [--config CFG.json] [--seed S]
ftgs train OUT/scene.json --init init.safetensors --out RUN_DIR --resume RUN_DIR/checkpoint
ftgs export RUN_DIR/checkpoint --format ftgs-if --out model.safetensors
ftgs render model.safetensors --request request.json --out renders.safetensors
ftgs eval RUN_DIR/checkpoint OUT/scene.json --split test --out METRICS.json
```

`ftgs <command> --help` lists every option. A run directory holds `config.json`, `train_log.jsonl`, `checkpoint/` and, when the scene has a test split, `METRICS.json`; Ctrl-C saves the checkpoint before stopping. Exit code 0 is success, 2 an invalid input or environment, 3 a numerical failure. See [`docs/modules/cli/command-line.md`](docs/modules/cli/command-line.md).

## Tests

```bash
pytest                       # CPU tests; GPU, dataset and weights tests skip when their resource is missing
pytest -n auto --dist loadgroup -m "not gpu"      # the CPU tests on every physical core (Q-214)
pytest -m gpu                # tests that need a CUDA device
OPENFTGS_DATASETS=/path pytest -m dataset         # converter checks on real datasets
OPENFTGS_ALLOW_DOWNLOADS=1 pytest -m weights      # tests that download pretrained weights
python scripts/license_scan.py                    # GPL, AGPL, non-commercial fail; others need a sign-off in scripts/license_signoff.json
```

The license scan passes in CI's environment (CPU torch). With CUDA torch it also lists the NVIDIA runtime wheels, which stay unapproved until they are signed off for the release (milestone M18).

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
