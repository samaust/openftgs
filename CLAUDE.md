# openftgs: clean-room implementation of FreeTimeGS

You are the implementer of `openftgs`, an Apache-2.0 Python package (CLI `ftgs`) that implements FreeTimeGS (Wang et al., CVPR 2025) under a clean-room process. The plan is the specification in `docs/spec/openftgs-spec.md`, revision 549 of the owner's spec; the owner is the person running this session. The spec was written from the paper alone and, with the allowed sources below, is everything you need. Implement it; don't redesign it.

<!-- Maintainer note: "Clean-room rules" restates spec §0 (owner rule of 2026-10-07); keep the two in step. "Precautions" are this repository's additions. -->

## Clean-room rules (strict)

These restate spec §0. If they ever differ from §0, §0 wins: say so to the owner.

Allowed sources:
- The paper, arXiv 2506.05348 v2, which carries the supplementary material as Appendices A–B (local copy, if present: `docs/paper/2506.05348v2.pdf`, never committed). The project page https://zju3dv.github.io/freetimegs/ itself, but none of its code links ("Renderer", "Framework") or demos.
- Papers the paper cites, and general Gaussian-splatting literature.
- Documentation and data files of the datasets the paper evaluates on: Neural3DV, ENeRF-Outdoor, SelfCap.
- Third-party software, by its class in the §0 table (software in two classes takes the stricter one):
  1. Imported or vendored by the package (PyTorch, gsplat, safetensors, lpips and the torchvision networks it loads, RoMa/`romatch` vendored at 77f8d68 with what it imports, fused-ssim, NumPy, Pillow, PyYAML, SciPy): documentation and source code.
  2. Replaced by our own code (OpenCV, scikit-image, torchmetrics): documentation only, even where tests call them as reference oracles.
  3. Run as separate programs (ffmpeg, COLMAP, BackgroundMattingV2): documentation only.
  4. Not imported (transitive dependencies; software evaluated but not adopted, such as the DINOv2 repository, kornia, decord, torchcodec): documentation and license files, plus packaging and build files read only to establish licenses.
- For classes 2 to 4, read the project's own documentation pages, never repository-indexed snippets: tools that index whole repositories can return source code (spec Appendix A, #36).

Never allowed:
- Anything from another FreeTimeGS implementation or reproduction, including the authors' own code: repositories, forks, gists, notebooks, blog code, issue threads, packages, checkpoints.
- EasyVolcap, in any form.
- Any code under GPL or AGPL.

If a search result, link, package or file looks like a FreeTimeGS implementation, don't open it. If you meet a forbidden source anyway: stop at once, take nothing from it, record it as an incident in `docs/clean-room/source-log.md`, and tell the owner before doing anything else.

Source log: every source you consult (a URL, a document, a third-party file or docstring you read) gets a row in `docs/clean-room/source-log.md` saying what it was and what you took from it. Reading this repository and the spec needs no row.

## Precautions in this repository

- Don't search the web for FreeTimeGS or its implementations; the paper and the spec cover what you need, and search results are where implementations turn up.
- For gsplat, work from the pinned 1.5.3. Never open `gsplat/contrib/dynamic` or other dynamic-scene code on newer branches; the spec's author left them unopened (Appendix A, #9).
- A PreToolUse hook, `.claude/hooks/clean_room_guard.py`, blocks tool calls that name FreeTimeGS or EasyVolcap in searches, downloads, URLs and paths, apart from the paper and the project page. It is a tripwire, not a permission: the rules above cover everything it can't see. Never work around it, and don't put "freetime" in file or directory names.

## Working from the spec

- The spec is the source of truth. Its provenance tags (§0) say what may change: [P] items are fixed by the paper; [P→3DGS] and [L: …] items change only with a logged reason; [D-n] decisions are configurable and never change silently.
- Never edit `docs/spec/`. It is a copy of the owner's document; changes come from the owner as a new export.
- Where the spec is silent, ambiguous or looks wrong, don't guess what another implementation might do:
  - If the choice changes no numbers, outputs or public API (private names, file layout inside a module), make it.
  - Otherwise make an explicit, provisional decision: add a row to `docs/clean-room/questions.md` (Q-n: spec location, gap, decision, rationale, alternatives, the test that would expose a wrong choice), mark the code with `# Q-n`, and tell the owner.
  - If the spec states something that seems wrong or contradicts itself, stop and ask before coding around it.
- The public API is exactly the signatures in §2.1, and modules import only from the bands below them (§2.1). Dependencies, versions and licenses are in §2.3; add nothing outside that table without asking.
- Build bottom-up, one milestone at a time: `types` and `config` → `motion` and `sh` → `model` → `render` (the pure-PyTorch reference backend, then gsplat) → `losses` → `density` → `optim` → `train` → `data`, `init` and `io` → `eval` → `cli` → the separate `ftgs_harness`. This order is the spec author's suggestion, not part of the spec.
- Before writing a module, read the spec items in its §2.1 row and everything they point to (sections, equations E-n, decisions D-n), plus the §3 tests that exercise them (U-n, C-n, S-n, B-n, harness levels L0–L3). At the start of each milestone, re-read §0 and the parts of §1–§3 it touches.
- A module is done when its §3.1 unit tests pass, with each test written from its description in the spec, not from the code. The synthetic-scene tests (§3.2) follow once `train` runs; the benchmarks (§3.4) and the harness (§3.3) come last.
- Every source file starts with `# SPDX-License-Identifier: Apache-2.0`. Vendored files keep their own license headers and license files (D-65).

## Memory and context

- Auto memory is off for this repository on purpose (`.claude/settings.json`), and so are claude.ai connectors. Don't try to remember things anywhere else: what must persist goes in the repository (`docs/clean-room/questions.md`, the source log, code comments, commit messages), where the owner can audit it.
- Your context is this file, the repository and the allowed sources. Don't read other projects on this machine or Claude Code's own files under `~/.claude/`.

## Commits

Small commits, one module or test group each. Each message cites the spec items it implements (for example `E11, D-14, U-13`) and any Q-n it relies on. Never commit `docs/paper/`, datasets, or anything derived from SelfCap (its license is non-commercial; §2.3).
