# SPDX-License-Identifier: Apache-2.0
"""The perceptual settings of revisions before 549o: LPIPS's two backbones as `loss.perc_net`,
which `openftgs.config` refuses (#375, decided item 3) and which checkpoints saved before 549o
still carry (Q-247).

Such a checkpoint's model doesn't depend on the loss it trained with, so the readers that take
only its model (`io.load_checkpoint`, `FreeTimeGaussians.load`, the CLI's model reader) validate
its configuration with 549o's defaults in place of its `perc_net` and `lambda_perc`, in memory
only; resuming it is refused (Q-247, Q-271). Plain JSON dicts in and out.

Spec: §2.1 `PercNet` and `LossConfig`, §2.2's checkpoint contract, D-9; Q-247. A private
top-level module that imports nothing from openftgs, so every band may import it, as the stage
instrumentation (PRD #220).
"""

from collections.abc import Mapping
from typing import Any

__all__ = ["LPIPS_BACKBONES", "legacy_perc_net", "resume_refusal", "with_current_loss"]

# #375 (decided item 3): LPIPS's backbones, the `perc_net` values revision 549o retired.
LPIPS_BACKBONES = ("vgg", "alex")


def legacy_perc_net(config: Mapping[str, Any]) -> str | None:
    """The `loss.perc_net` of a stored TrainConfig dict when it is one of LPIPS's backbones,
    else None (another value, valid or not, or none)."""
    loss = config.get("loss")
    net = loss.get("perc_net") if isinstance(loss, Mapping) else None
    return net if isinstance(net, str) and net in LPIPS_BACKBONES else None


# Q-247: in memory only; the dict given is never changed, so `TrainerState.config` stays as saved.
def with_current_loss(config: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a stored TrainConfig dict that `config_from_dict` reads as 549o's: a legacy
    `loss.perc_net` and its `loss.lambda_perc` left out, so they take 549o's defaults (ꟻLIP at
    its calibrated weight, D-9); every other key, and any other dict, as it is."""
    out = dict(config)
    if legacy_perc_net(config) is not None:
        loss = config["loss"]
        out["loss"] = {k: v for k, v in loss.items() if k not in ("perc_net", "lambda_perc")}
    return out


def resume_refusal(config: Mapping[str, Any], where: str) -> str | None:
    """Q-247: why the run that `where` (a checkpoint, say) stores can't be resumed, when its
    configuration names LPIPS's backbone; else None."""
    net = legacy_perc_net(config)
    if net is None:
        return None
    return (
        f"{where} was trained with loss.perc_net {net!r}, one of LPIPS's backbones, which "
        f"revision 549o of the spec retired (D-9, D-89): its run can't be resumed, though ftgs "
        f"eval, render and export still read its model (Q-247)"
    )
