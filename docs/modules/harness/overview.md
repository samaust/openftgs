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
- `--settings` is the **run settings** file (Q-203): `seed` (the case generator's, 0), `k` (5), `gpus`, `l0.trained_models`, `l3.scenes` [{`name`, `manifest`, `train_manifest`?, `init`}], the case sizes `l0.models` (20), `l1.items` (200), `l1.probe_items` (21) and `l2.queries` (4), and `l2.lpips` (true), which leaves LPIPS out of L2's `losses` queries where the weights can't be downloaded (Q-212). The sizes default to §3.3's; a smaller size is allowed. A run with a size below §3.3's or with `l2.lpips` false is a **reduced** run (Q-203, Q-212): `run.settings.reduced` in `report.json` is true, whatever levels the run requested, and `report.md`'s header names what was reduced, the case sizes, LPIPS in L2, or both. Being reduced fails no check and changes no exit code. `scripts/harness_s9.py OUT` writes S-9's scene, init file, configuration and run settings for L3.
- `--jobs` runs that many adapter calls at once; `--timeout` bounds each call, and a call past it is a crash.
- `rerun` replays the adapter calls one check depends on, from `run.json` and `cases/`, into `RUN_DIR/reruns/<CHECK_ID>/`, prints that check's JSON and exits like `run`. For an L3 check it reruns the scene's 3K training runs.

## Exit codes

0 when no check failed, 1 when at least one did, 2 for a harness or configuration error: bad arguments, an unreadable config or settings file, a `seed` key in the config, a missing adapter executable, L3 requested without scenes or without the `[lpips]` extra (§3.3, Q-211). An `inconclusive` or `not_comparable` check fails nothing.

## The run directory

- `run.json`: the adapter command lines, input paths and hashes, the settings and the case seed, for `rerun`;
- `cases/`: everything the harness generated, models, requests, queries and configs;
- `A/`, `B/`: each adapter's outputs, by level and case;
- `L3/<scene>/<A|B|A′>/seed<k>/`: one training run each;
- `logs/`: the stdout and stderr of every adapter call;
- `artifacts/`: the diff images and plots;
- `report.json` and `report.md` (`levels.md`, "Report").

## Dependencies

numpy, scipy, safetensors and matplotlib, with the Agg backend for every plot (owner decision, 2026-10-11). The extra `ftgs-harness[lpips]` adds torch and lpips 0.1.4, which L3's LPIPS-Alex and LPIPS-VGG need (Q-201). L0 to L2 run without them.
