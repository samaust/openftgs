---
name: two-axis-review
description: "Review the changes since a fixed point (commit, branch, tag, or merge-base) along two axes: Standards (does the code follow this repo's documented coding standards?) and Spec (does the code match the project spec items it implements and the originating PRD?). Runs both reviews in parallel sub-agents and reports them side by side. Use when the owner wants to review a branch or work-in-progress changes, or asks to \"review since X\"."
---

Two-axis review of the diff between `HEAD` and a fixed point the owner supplies:

- **Standards**: does the code conform to this repo's documented coding standards?
- **Spec**: does the code faithfully implement the project spec (`docs/spec/openftgs-spec.md`) and the originating PRD?

Both axes run as **parallel sub-agents** so they don't pollute each other's context, then this skill aggregates their findings.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). Both sub-agent prompts start with the line from its Subagents section. The sub-agents don't write files: they list any source outside this repository and the spec under "Sources consulted", and you add those rows to the source log after aggregating, on the branch under review. When the review runs on `main`, list them in the report instead ("Where the clean-room files are written" in docs/agents/clean-room.md).

## Process

### 1. Pin the fixed point

Whatever the owner said is the fixed point (a commit SHA, branch name, tag, `main`, `HEAD~5`, etc.). If they didn't specify one, use `main` when the current branch is another branch; on `main`, ask.

Capture the diff command once, against the merge-base:

- **Clean working tree:** `git diff <fixed-point>...HEAD` (three-dot).
- **Uncommitted changes** (`git status --porcelain` lists any): `git diff $(git merge-base <fixed-point> HEAD)`, which includes them, plus the untracked files that `git status --porcelain` marks `??`, which no diff shows. Say in the report that work in progress was reviewed.

Also note the list of commits via `git log <fixed-point>..HEAD --oneline`.

Before going further, confirm the fixed point resolves (`git rev-parse <fixed-point>`) and there is something to review: a non-empty diff or untracked files. A bad ref or an empty review should fail here, not inside two parallel sub-agents.

### 2. Identify the spec sources

The Spec axis checks two sources, in this order of authority:

1. **The project spec**, `docs/spec/openftgs-spec.md`. Collect the spec items the commit messages cite (sections §n, equations E-n, decisions D-n, tests U-n, C-n, S-n, B-n, and Q-n rows in `docs/clean-room/questions.md`), plus any the owner passed as an argument. If the commits cite none, that is itself a Standards finding (CLAUDE.md, "Commits"): take the spec items from the "Spec items" section of the PRD, or of the agent brief for an issue implemented on its own, and ask the owner only if there is neither.
2. **The originating PRD**, if there is one: the issue labelled `skill:conversation-to-prd` that the commit messages reference (`#123`, `Closes #45`) or the owner passed, with any `skill:prd-to-tickets` tickets the commits reference (a referenced ticket's parent issue is its PRD), fetched as `docs/agents/issue-tracker.md` describes. For an issue implemented on its own, its latest agent brief comment takes the PRD's place.

Where the PRD and the project spec disagree, the project spec wins, and the disagreement is itself a finding. If neither spec items nor a PRD can be identified, the **Spec** sub-agent will skip and report "no spec available".

### 3. Identify the standards sources

This repository documents its standards in the files below; every one is on the list:

- `CLAUDE.md`: file headers (SPDX; vendored files keep their own, D-65), `# Q-n` markers, commit messages, dependencies, what may never be committed, and the branch and merge rules.
- Spec §2.1 and CLAUDE.md "Working from the spec": the public API is the §2.1 signatures plus the additions recorded as Q-n rows; each module's `__all__` lists exactly that API and every other name starts with an underscore; a public name with neither a §2.1 signature nor a Q-n row is a finding; modules import from the bands below them and, within a band, only from the modules before them in the order CLAUDE.md gives.
- Spec §2.3: dependencies, versions and licenses.
- CLAUDE.md "Clean-room rules" and `docs/agents/clean-room.md`: provenance (the source log, forbidden sources).
- `.claude/skills/tdd/SKILL.md`, "Where expected values come from" and "House rules": the test standards.
- `docs/GLOSSARY.md`: the names of domain concepts in code, tests and messages.

Also search the repository for any other file that documents how code should be written (a `CONTRIBUTING.md` added later, say) and add it.

On top of whatever the repo documents, the Standards axis always carries the **smell baseline** below: a fixed set of Fowler code smells (_Refactoring_, ch.3) that applies even when a repo documents nothing. Three rules bind it:

- **The repo overrides.** A documented repo standard always wins; where it endorses something the baseline would flag, suppress the smell.
- **The spec overrides.** A suggested fix never changes a §2.1 signature, the module layout, a number or an output. Suppress a smell that comes from structure the spec mandates; if that structure itself looks like the problem, report it as a candidate Q-n row, not as a fix.
- **Always a judgement call.** Each smell is a labelled heuristic ("possible Feature Envy"), never a hard violation. Like any standard here, skip anything tooling already enforces.

Each smell reads *what it is* → *how to fix*; match it against the diff:

- **Mysterious Name**: a function, variable, or type whose name doesn't reveal what it does or holds. → rename it; if no honest name comes, the design's murky.
- **Duplicated Code**: the same logic shape appears in more than one hunk or file in the change. → extract the shared shape, call it from both.
- **Feature Envy**: a method that reaches into another object's data more than its own. → move the method onto the data it envies.
- **Data Clumps**: the same few fields or params keep travelling together (a type wanting to be born). → bundle them into one type, pass that.
- **Primitive Obsession**: a primitive or string standing in for a domain concept that deserves its own type. → give the concept its own small type.
- **Repeated Switches**: the same `switch`/`if`-cascade on the same type recurs across the change. → replace with polymorphism, or one map both sites share.
- **Shotgun Surgery**: one logical change forces scattered edits across many files in the diff. → gather what changes together into one module.
- **Divergent Change**: one file or module is edited for several unrelated reasons. → split so each module changes for one reason.
- **Speculative Generality**: abstraction, parameters, or hooks added for needs the spec doesn't have. → delete it; inline back until a real need shows.
- **Message Chains**: long `a.b().c().d()` navigation the caller shouldn't depend on. → hide the walk behind one method on the first object.
- **Middle Man**: a class or function that mostly just delegates onward. → cut it, call the real target direct.
- **Refused Bequest**: a subclass or implementer that ignores or overrides most of what it inherits. → drop the inheritance, use composition.

### 4. Spawn both sub-agents in parallel

Issue both sub-agent calls together and wait for both reports before aggregating.

**Standards sub-agent prompt** should include:

- The full diff command and commit list.
- The list of standards-source files you found in step 3, **plus the smell baseline from step 3** pasted in full (the sub-agent has no other access to it).
- The brief: "Report, per file/hunk where relevant, (a) every place the diff violates a documented standard: cite the standard (file + the rule); (b) provenance: code that looks taken from a third-party source (non-obvious constants, a ported algorithm, a vendored-looking block) with no matching row in `docs/clean-room/source-log.md`, any source-log row that names a forbidden source, and vendored files without their own license header; and (c) any baseline smell you spot: name it and quote the hunk. Mark each finding hard (a documented standard, provenance included) or judgement (a baseline smell, always), and give it a severity: critical, major, minor or nit, the scale of the milestone review in `docs/clean-room/milestones.md`. A documented repo standard overrides the baseline, and no suggested fix may change a §2.1 signature, a number or an output. Private tests (named without a §3 ID, `.claude/skills/tdd/SKILL.md` "Private tests") are allowed; a spec test that calls a private helper is a finding. Skip anything tooling enforces. One or two lines per finding."

**Spec sub-agent prompt** should include:

- The diff command and commit list.
- The cited spec items with their locations in `docs/spec/openftgs-spec.md`, and the PRD's issue number or fetched contents, if any.
- The brief: "Report: (a) requirements the spec items or the PRD asked for that are missing or partial; (b) behaviour in the diff that wasn't asked for (scope creep); (c) requirements that look implemented but where the implementation looks wrong; (d) tests whose cases, expected values or tolerances differ from their §3 row; (e) places where the PRD contradicts the project spec, which wins. Quote the spec or PRD line for each finding, and give it a severity: critical, major, minor or nit. One or two lines per finding."

If neither spec items nor a PRD were found, skip the Spec sub-agent and note this in the final report.

### 5. Aggregate

Present the two reports under `## Standards` and `## Spec` headings, verbatim or lightly cleaned. Do **not** merge or rerank findings, because the two axes are deliberately separate (see _Why two axes_).

End with a one-line summary: the findings per axis by severity, and the worst issue _within each axis_ (if any). Don't pick a single winner across axes: that's the reranking the separation exists to prevent.

## Why two axes

A change can pass one axis and fail the other:

- Code that follows every standard but implements the wrong thing → **Standards pass, Spec fail.**
- Code that does exactly what the spec and PRD asked but breaks the project's conventions → **Spec pass, Standards fail.**

Reporting them separately stops one axis from masking the other.
