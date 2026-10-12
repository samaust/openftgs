openftgs is an independent clean-room implementation of FreeTimeGS (Wang et al., CVPR 2025), not affiliated with its authors.

# openftgs

Python package `openftgs` with the command-line tool `ftgs`, under the Apache-2.0 license.

**Status: pre-alpha.** The repository holds the package skeleton (milestone M0); the modules are empty and nothing runs yet.

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

Training needs a CUDA build of PyTorch: install it from the [PyTorch index](https://pytorch.org/get-started/locally/) for your CUDA version before installing openftgs. A CPU build is enough for the tests on the reference backend.

Optional extras:

- `[init]`: the dependencies of the RoMa-based 4D initialization.
- `[fast]`: currently empty. The faster SSIM loss uses [fused-ssim](https://github.com/rahul-goel/fused-ssim) (MIT), which is not on PyPI; install it from its repository.

The differential harness is a separate package in `harness/` (`pip install -e harness`).

## Tests

```bash
pytest                       # CPU tests; GPU, dataset and weights tests skip when their resource is missing
pytest -m gpu                # tests that need a CUDA device
OPENFTGS_DATASETS=/path pytest -m dataset         # converter checks on real datasets
OPENFTGS_ALLOW_DOWNLOADS=1 pytest -m weights      # tests that download pretrained weights
python scripts/license_scan.py                    # fails on GPL or AGPL dependencies
```

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
