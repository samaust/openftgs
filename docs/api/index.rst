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
