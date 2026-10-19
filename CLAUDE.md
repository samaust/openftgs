# openftgs: clean-room implementation of FreeTimeGS

You are the implementer of `openftgs`, an Apache-2.0 Python package (CLI `ftgs`) that implements FreeTimeGS (Wang et al., CVPR 2025) under a clean-room process. The plan is the specification in `docs/spec/openftgs-spec.md`, revision 549i of the owner's spec (revision 549, amended nine times in the repository in the owner's name, as 549a, 549b, 549c, 549d, 549e, 549f, 549g, 549h and 549i; `docs/spec/README.md`); the owner is the person running this session. The spec was written from the paper alone and, with the allowed sources below, is everything you need. Implement it; don't redesign it.

<!-- Maintainer note: "Clean-room rules" restates spec §0 (owner rule of 2026-10-07) as amended by the owner on 2026-10-11, when a license rule replaced §0's table of software classes; spec §0 is to be updated to match. "Precautions" are this repository's additions. -->

## Clean-room rules (strict)

These restate spec §0, except for third-party software: the owner's license rule of 2026-10-11 below replaces §0's table of software classes until a new spec export carries it. In everything else, if these rules differ from §0, §0 wins: say so to the owner.

Allowed sources:
- The paper, arXiv 2506.05348 v2, which carries the supplementary material as Appendices A–B (local copy, if present: `docs/paper/2506.05348v2.pdf`, never committed). The project page https://zju3dv.github.io/freetimegs/ itself, but none of its code links ("Renderer", "Framework") or demos.
- Papers the paper cites, and general Gaussian-splatting literature.
- Documentation and data files of the datasets the paper evaluates on: Neural3DV, ENeRF-Outdoor, SelfCap.
- The openftgs-synth dataset (owner rule, 2026-10-15; #108): the owner's rendered multi-view video dataset, CC0-1.0, local at `$OPENFTGS_DATASETS/synth`. Its data files, `DATASET_CARD.md`, `ASSETS.md` and `SHA256SUMS` are allowed. Its generator is GPL-3.0-or-later, so its source code is never read and its documentation follows the copyleft rule below.
- Third-party software, by its license (owner rule, 2026-10-11; the aim is that openftgs can be published under Apache-2.0):
  - Under a permissive license that allows commercial use (MIT, BSD, Apache-2.0, ISC, PSF, zlib and the like): documentation and source code. This covers the software the package imports or vendors (PyTorch, gsplat 1.5.3, safetensors, lpips, RoMa/`romatch` at 77f8d68, fused-ssim, NumPy, Pillow, PyYAML, SciPy), the software it replaces with its own code (OpenCV, scikit-image, torchmetrics), separate programs (COLMAP, BackgroundMattingV2) and software it doesn't import (kornia, the DINOv2 repository, decord, torchcodec). Read the project's license file before its source code; where a project bundles code under other licenses, apply this rule to each part.
  - Under a copyleft license, strong or weak (GPL, AGPL, LGPL, MPL, EPL, CDDL and the like), a non-commercial license, or no license: documentation only, never source code. Examples: ffmpeg (LGPL or GPL), hypothesis and tqdm (MPL-2.0). Using them as dependencies or programs, as §2.3 allows, is fine.
  - If you can't tell which side a license is on, ask before opening the source.
- Code copied or closely adapted from a permissive source keeps its attribution: a comment above the adapted code naming the project, file, commit or version and license; an entry in NOTICE where the license asks for one; and a source-log row. Vendored files keep their own license headers and license files (D-65).
- For software you may read only the documentation of, read the project's own documentation pages, never snippets from tools that index whole repositories: they can return source code (spec Appendix A, #36).

Never allowed, whatever the license:
- Anything from another FreeTimeGS implementation or reproduction, including the authors' own code: repositories, forks, gists, notebooks, blog code, issue threads, packages, checkpoints.
- EasyVolcap, in any form.
- The source code of software under a copyleft, non-commercial or missing license.

If a search result, link, package or file looks like a FreeTimeGS implementation, don't open it. If you meet a forbidden source anyway: stop at once, take nothing from it, record it as an incident in `docs/clean-room/source-log.md`, and tell the owner before doing anything else.

Source log: every source you consult (a URL, a document, a third-party file or docstring you read) gets a row in `docs/clean-room/source-log.md` saying what it was and what you took from it. Spec §0 names Appendix A as the log; `source-log.md` continues it for the implementation, since the spec changes only through the owner. Reading this repository and the spec needs no row.

## Precautions in this repository

- Don't search the web for FreeTimeGS or its implementations; the paper and the spec cover what you need, and search results are where implementations turn up.
- For gsplat, work from the pinned 1.5.3. Never open `gsplat/contrib/dynamic` or other dynamic-scene code on newer branches; the spec's author left them unopened (Appendix A, #9).
- A PreToolUse hook, `.claude/hooks/clean_room_guard.py`, blocks tool calls that name FreeTimeGS or EasyVolcap in web searches, downloads, URLs, paths and local searches outside the repository, apart from the project page (the paper is read on arXiv, whose URLs don't name it). Searches of the repository's own files may name them (owner rule, 2026-10-12): grep, rg, ag, ack, find, fd and git grep with every search root inside the repository, and the Grep and Glob tools pointed inside it; a tool that doesn't skip `.gitignore`'d files (grep -r, find, ack, Glob) must not walk into `.venv`. It is a tripwire, not a permission: the rules above cover everything it can't see. Never work around it, and don't put "freetime" in file or directory names.

## Working from the spec

- The spec is the source of truth. Its provenance tags (§0) say what may change: [P] items are fixed by the paper; [P→3DGS] and [L: …] items change only with a logged reason; [D-n] decisions change via config where §2.1 defines a field for them, otherwise by a logged change, and never silently (§0, D-72).
- Never edit `docs/spec/`. It is a copy of the owner's document; changes come from the owner as a new export.
- Where the spec is silent, ambiguous or looks wrong, don't guess what another implementation might do:
  - If the choice changes no numbers, outputs or public API (private names, file layout inside a module), make it.
  - Otherwise make an explicit, provisional decision: add a row to `docs/clean-room/questions.md` (Q-n: spec location, gap, decision, rationale, alternatives, the test that would expose a wrong choice), mark the code with `# Q-n`, and tell the owner. Before a build (planning, grilling, triage), record the gap instead as a Q-n candidate on the tracker; the implement skills turn candidates into rows ("Where the clean-room files are written" in `docs/agents/clean-room.md`).
  - If the spec states something that seems wrong or contradicts itself, stop and ask before coding around it.
- The public API is the signatures in §2.1 plus the additions recorded as Q-n rows. Where the spec names a module, function or type but gives no signature (for example in `openftgs.motion` or `openftgs.eval`, or `GrowthReport`), propose one as a provisional Q-n row, as for any gap; one row may cover all the new signatures of a module. Changing a signature §2.1 lists is not a gap: stop and ask. Each module's `__all__` lists exactly its public API; everything else is private and its name starts with an underscore. Modules import from the bands below them (§2.1, package layers) and, within a band, only from the modules before them in this fixed order, so imports run one way with no cycles (owner rule, 2026-10-11): Foundation `types`, `config`, `io`, `data`, `init`; Core model `motion`, `sh`, `model`, `render`; Mechanics `optim`, `losses`, `density`; Orchestration `eval`, `train`; Interface `cli`. `io` still imports neither `model` nor `train`. Dependencies, versions and licenses are in §2.3; add nothing outside that table without asking.
- Build bottom-up, one milestone at a time, in the order of `docs/clean-room/milestones.md` (M0 to M18: skeleton → types and config → motion and sh → model and io → render, reference then gsplat → losses → optim → density → eval → data core and train → synthetic recovery → converters → init → cli → performance → harness → validation runs → release). That order is the spec author's suggestion, not part of the spec. Each milestone runs as one PRD through the skills, as the milestones file's "How to run one" describes.
- Before writing a module, read the spec items in its §2.1 row and everything they point to (sections, equations E-n, decisions D-n), plus the §3 tests that exercise them (U-n, C-n, S-n, B-n, harness levels L0–L3). At the start of each milestone, re-read §0 and the parts of §1–§3 it touches.
- A module is done when its §3.1 unit tests pass, with each test written from its description in the spec, not from the code. The synthetic-scene tests (§3.2) follow once `train` runs; the benchmarks (§3.4) and the harness (§3.3) come last.
- Every source file starts with `# SPDX-License-Identifier: Apache-2.0`. Vendored files keep their own license headers and license files (D-65).

## Memory and context

- Auto memory is off for this repository on purpose (`.claude/settings.json`), and so are claude.ai connectors. Don't try to remember things anywhere else: what must persist goes in the repository (`docs/clean-room/questions.md`, the source log, code comments, commit messages), where the owner can audit it. Before a build, Q-n candidates and sources consulted wait on the tracker, the owner's channel, until an implement run records them.
- Your context is this file, the repository and the allowed sources. Don't read other projects on this machine or Claude Code's own files under `~/.claude/`.

## Issues

The issue tracker at github.com/samaust/openftgs is the owner's channel. Use `gh issue` to read and write there: every new Q-n row gets a decision issue labelled `q-n` and titled `Q-<n>: <short gap>`, opened once its number is final, and the answer goes back into `docs/clean-room/questions.md` (`docs/agents/issue-tracker.md`, "Q-n issues"). `gh` may target no other repository (the guard blocks it), and a link in an issue is subject to the clean-room rules like any other link. Never delete or transfer issues.

## Commits

Small commits, one module or test group each. Each message cites the spec items it implements (for example `E11, D-14, U-13`) and any Q-n it relies on. Work happens on a branch off `main`: an integration branch per PRD (or per issue implemented on its own), or a short branch for one small change (`q-<n>-answer` for an answer to a Q-n row, `docs-<slug>` for glossary and module docs, `workflow-<slug>` for skills, agent docs and this file, `setup-skills-repair`). Every branch reaches `main` only by a `git merge --no-ff` that the owner has approved, never by a squash. A commit that implements no spec item says so (`Spec items: none`). Never push without asking: before pushing `main`, ask the owner "Push `main` to origin now?" and push only if they answer yes to that question, every time; an earlier approval doesn't cover a later push. Other branches stay local. Never commit `docs/paper/`, datasets, or anything derived from SelfCap (its license is non-commercial; §2.3).

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues at github.com/samaust/openftgs via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Clean room in skills

How skills and their subagents apply the clean-room rules: see `docs/agents/clean-room.md`.

### Domain docs

* `docs/GLOSSARY.md`
* `docs/modules/`

See `docs/agents/domain.md`.
