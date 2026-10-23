..
   SPDX-License-Identifier: Apache-2.0

   reStructuredText, not MyST: autosummary finds the directives it generates stubs for only in
   reStructuredText sources.

API reference
=============

The public API of openftgs is the signatures of spec §2.1 and the additions recorded as Q-n rows;
each module's ``__all__`` names its part of it, and everything not listed there is private. The
modules are listed in their bands' order, Foundation, Core model, Mechanics, Orchestration and
Interface, each module importing only from the bands below it and from the modules before it in
its band.

The docstrings cite the specification, :repo:`docs/spec/openftgs-spec.md`, and the project's
decisions on it:

§n
   A section of the spec, as §1.4 (time, coordinates and cameras) or §2.2 (the file formats).
E-n
   One of the equations E1 to E17 of spec §1.2, which :doc:`/method` transcribes.
D-n
   A design decision of the spec's Appendix B, made where the paper is silent, with its
   rationale, alternatives and detecting test.
U-n, S-n, C-n, G-n, B-n
   The spec's tests (§3): unit tests, synthetic-scene recovery tests, dataset converter checks,
   synthetic-dataset checks on openftgs-synth and performance benchmarks.
Q-n
   A provisional decision of the implementation where the spec is silent or ambiguous, with the
   same columns as a D-n, in :repo:`docs/clean-room/questions.md`.
[P], [P→3DGS], [L: …]
   The provenance tags of spec §0: stated in the paper; the paper's "same settings as 3DGS",
   taken from 3DGS or gsplat; standard Gaussian-splatting practice from an allowed source.

.. autosummary::
   :toctree: generated

   openftgs.types
   openftgs.config
   openftgs.io
   openftgs.data
   openftgs.init
   openftgs.motion
   openftgs.sh
   openftgs.model
   openftgs.render
   openftgs.optim
   openftgs.losses
   openftgs.density
   openftgs.eval
   openftgs.train
   openftgs.cli
