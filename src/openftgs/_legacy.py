# SPDX-License-Identifier: Apache-2.0
"""The perceptual settings of revisions before 549o: LPIPS's two backbones as `loss.perc_net`,
which `openftgs.config` refuses (#375, decided item 3) and which checkpoints saved before 549o
still carry (Q-247).

Spec: §2.1 `PercNet` and `LossConfig`, §2.2's checkpoint contract, D-9; Q-247. A private
top-level module that imports nothing from openftgs, so every band may import it, as the stage
instrumentation (PRD #220).
"""

__all__ = ["LPIPS_BACKBONES"]

# #375 (decided item 3): LPIPS's backbones, the `perc_net` values revision 549o retired.
LPIPS_BACKBONES = ("vgg", "alex")
