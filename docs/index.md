# openftgs

openftgs is an independent clean-room implementation of FreeTimeGS (Wang et al., CVPR 2025), not
affiliated with its authors. The distribution is `openftgs`, the import package `openftgs` and
the command line `ftgs`, under the Apache-2.0 license. Its source is on GitHub, at
[samaust/openftgs](https://github.com/samaust/openftgs).

The method reconstructs a dynamic scene from multi-view video as Gaussian primitives that each
live at their own time and position, move with their own velocity and fade in and out with a
temporal opacity; it renders the scene at any camera and any time. {doc}`method` restates it.

## Clean-room process

The code is written from the paper (arXiv 2506.05348 v2) and from the owner's specification,
which was itself written from the paper alone. No code, notebook, checkpoint or issue thread from
any other implementation of the method, including the authors' own, is consulted. Third-party
software is read only within its license: source code only under permissive licenses that allow
commercial use, documentation only for everything else. Every source consulted is logged. The
renderer the paper's project page links was read for a documentation-only survey ({issue}`352`),
and nothing from it is in the package.

openftgs includes a copy of RoMa's inference code, under the MIT License, with the DINOv2 layers it
carries under the Apache License 2.0; a copy of NVIDIA's ꟻLIP loss, under the BSD 3-Clause
License, in `openftgs._vendor.flip`; and, in `openftgs.losses`, HaarPSI adapted from piq 0.8.0,
under the Apache License 2.0. `NOTICE` lists them, the code RoMa credits and the pretrained weights
whose terms ask for attribution.

## Citing the method

openftgs implements the method of this paper; please cite the paper for the method itself:

> Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng
> and Xiaowei Zhou. *FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere for Dynamic Scene
> Reconstruction.* CVPR 2025. [arXiv:2506.05348](https://arxiv.org/abs/2506.05348).

## Quick links

- {doc}`installation`: pip, a CUDA build of PyTorch, the extras, ffmpeg and COLMAP.
- {doc}`quickstart`: from a rendered openftgs-synth scene to a trained, evaluated model.
- {doc}`datasets`: openftgs-synth, Neural3DV, ENeRF-Outdoor and SelfCap.
- {doc}`cli`: every `ftgs` command and option.
- {doc}`api/index`: the public Python API, module by module.

## Project records

The specification and the clean-room records live in the repository, outside this site:

- {repo}`docs/spec/openftgs-spec.md`: the specification, the source of truth for the
  implementation.
- {repo}`docs/clean-room/source-log.md`: every source consulted during the implementation and
  what was taken from it.
- {repo}`docs/clean-room/questions.md`: the gaps found in the spec and the provisional decisions
  made for them, the Q-n rows (a large file, which GitHub may show only as raw text).
- {repo}`docs/clean-room/milestones.md`: the order in which the package was built.
- {repo}`docs/clean-room/validation-runs.md`: the training and evaluation runs on openftgs-synth.
- {repo}`docs/clean-room/release-checklist.md`: the release-time checks of this version, from the
  license scan to the pretrained weights' terms and the absence of dataset material.
- {repo}`docs/GLOSSARY.md`: the project's vocabulary.
- {repotree}`docs/modules`: how each module meets the spec.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
datasets
method
cli
api/index
contributing
changelog
```
