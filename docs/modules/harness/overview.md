# The differential harness, `ftgs_harness`

The **harness** compares two implementations of the spec, A and B, through files, commands and metrics only (§3.3). It is the separate distribution `ftgs-harness` in `harness/`, with its own pyproject and license (Q-5), and it never imports openftgs: `tests/test_skeleton.py` checks every module of `harness/src` for an import of it. Each implementation ships an **adapter**; openftgs's is `ftgs-adapter` (`docs/modules/cli/adapter.md`, Q-202). The harness's modules are internal: its interface is the command and the files (Q-213).

## Commands

```
ftgs-harness run --a "ADAPTER ARGV" --b "ADAPTER ARGV" --config harness_config.json
                 --levels L0,L1,L2,L3 --out RUN_DIR [--settings settings.json] [--jobs J] [--timeout S]
ftgs-harness rerun CHECK_ID [--out RUN_DIR]
```

- `--a` and `--b` are command lines, split as a shell would split them; the harness appends the adapter command and its arguments (§3.3 adapter contract).
- `--config` is `harness_config.json`: hyperparameters in TrainConfig's field names, nested by sub-config as `config_to_dict` writes them (Q-10). The harness passes it to each adapter's `train` verbatim and refuses a `seed` key in it, since seeds come per run (Q-203).
- `--settings` is the **run settings** file (Q-203): `seed` (the case generator's, 0), `k` (5), `gpus`, `l0.trained_models`, `l3.scenes` [{`name`, `manifest`, `train_manifest`?, `init`}], the case sizes `l0.models` (20), `l1.items` (200), `l1.probe_items` (21) and `l2.queries` (4). The sizes default to §3.3's; a smaller size is allowed. A run with a size below §3.3's is a **reduced** run (Q-203): `run.settings.reduced` in `report.json` is true, whatever levels the run requested, and `report.md`'s header names the case sizes. Q-212's `l2.lpips` went with revision 549o, since L2's ꟻLIP needs no weights (Q-255); a settings file naming it is exit 2. Being reduced fails no check and changes no exit code. `scripts/harness_s9.py OUT` writes S-9's scene, init file, configuration and run settings for L3.
- `--jobs` runs that many adapter calls at once; `--timeout` bounds each call, and a call past it is a crash.
- `rerun` replays the adapter calls one check depends on, from `run.json` and `cases/`, into `RUN_DIR/reruns/<CHECK_ID>/`, prints that check's JSON and exits like `run`. It exits 2 if `harness_config.json` or the settings file no longer has the hash `run.json` recorded, since a reproduction runs on the run's exact input files (§3.3). For an L3 check it reruns the scene's 3K training runs.

## Exit codes

0 when no check failed, 1 when at least one did, 2 for a harness or configuration error: bad arguments, an unreadable config or settings file, a `seed` key in the config, a missing adapter executable, L3 requested without scenes or without the `[torch]` extra, whose message names it (§3.3, Q-211, Q-260). An `inconclusive` or `not_comparable` check fails nothing.

## The run directory

- `run.json`: the adapter command lines, input paths and hashes, the settings and the case seed, for `rerun`;
- `cases/`: everything the harness generated, models, requests, queries and configs;
- `A/`, `B/`: each adapter's outputs, by level and case;
- `L3/<scene>/<A|B|A′>/seed<k>/`: one training run each;
- `logs/`: the stdout and stderr of every adapter call;
- `artifacts/`: the diff images and plots;
- `report.json` and `report.md` (`levels.md`, "Report").

## Dependencies

numpy, scipy, safetensors and matplotlib, with the Agg backend for every plot (owner decision, 2026-10-11). The extra `ftgs-harness[torch]` holds torch alone, with which L3 computes ꟻLIP (Q-201, Q-254, Q-260). L0 to L2 run without it: the metrics module imports torch only to compute ꟻLIP.

## The harness's ꟻLIP

L3 computes ꟻLIP with a copy of NVIDIA's loss of its own, `ftgs_harness._vendor.flip`, since the harness never imports openftgs (§2.1, Q-254). It is `src/pytorch/flip_loss.py` at commit d3ab02a with the device-agnostic change alone, so that it runs on the CPU: each `.cuda()` and `device='cuda'` becomes the image argument's device, the colour transforms' constants included, on lines marked `ftgs_harness (device)`. Unlike openftgs's copy (`docs/modules/losses/vendored-flip.md`) it caches nothing: the filters, the colour constants and the colour maximum are made at every call, as published.

`python scripts/vendor_flip.py --target harness` makes it from U-23's reference copy in `tests/reference/flip` (Q-252), never by hand, and `--check` rebuilds it offline and compares it byte for byte. Its BSD-3-Clause `LICENSE` and its provenance record `VENDOR.json`, which records the upstream commit, the published file's SHA-256 and the copy's, and every edit with its count, sit beside it and ship in the wheel as package data; CI's wheel job checks them against the record. `tests/harness/test_harness_flip.py` checks the hashes, the device-only edits and the check mode.

The harness calls it in float32 on the CPU, one held-out image at a time, at 67.0 pixels per degree (Q-259), on the render clamped to [0, 1] (E16): about a second for a 1352×1014 image on six threads. On the CPU it agrees with openftgs's copy within U-23's 1e-6.
