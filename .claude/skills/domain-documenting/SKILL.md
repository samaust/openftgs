---
name: domain-documenting
description: Create and update openftgs's domain docs, the single glossary docs/GLOSSARY.md and the module docs under docs/modules/<module>/, challenging terms against the spec and the glossary. Use when a domain term needs defining or sharpening, or when a module's documentation needs writing or updating.
---

# Domain Docs

Actively create and update the project's domain docs as you design. This is the *active* discipline: challenging terms, identifying applicable edge cases, and writing the glossary and documentation.

## File structure

```
/
└── docs/
    ├── GLOSSARY.md
    └── modules/
        ├── density/
        |    ├── relocation.md
        |    └── growth.md
        └── render/
            └── backends.md
```

The project has one glossary, `docs/GLOSSARY.md`, for every module. Its terms are grouped under subheadings by area; a term only one module uses goes under the subheading for its area. Module docs under `docs/modules/<module>/` never hold a glossary of their own.

Create module docs lazily: only when you have something to write. If no `docs/modules/<module>` exists, create it when the first doc is needed.

## Module docs

A module doc explains how a module meets its spec §2.1 row and spec items, covering what the spec leaves to the implementer:

- its internal structure and private seams;
- invariants that hold between its functions;
- decisions within the spec's latitude (from the PRD, or recorded in a commit message);
- the Q-n rows it relies on, and why.

It cites spec items (E-n, D-n, U-n, §n) and never restates their numbers, equations or tolerances: those live in the spec, and a copy goes stale. It never overrides the spec. Where the code or a doc departs from it, that is a finding for the owner, not something to document.

One file per subject, named after the subject in glossary vocabulary (`density/relocation.md`, not `subject1.md`), using the terms of `docs/GLOSSARY.md` throughout.

## Where the edits go

Never leave edits to `docs/GLOSSARY.md` or `docs/modules/` uncommitted on `main`: CLAUDE.md forbids committing on `main` directly, and uncommitted changes stop the next implement run at its clean-tree check.

- **Already on a branch** (an integration branch during a build): edit there and commit with the work.
- **On `main`** (planning, grilling, triage): before the first edit, check that the working tree is clean (if `git status --porcelain` lists anything, ask the user before going on), then create a short branch `docs-<slug>` from `main`. Make every edit of the session there, committing as terms are resolved. When the glossary and module-doc work of the current task is done (the end of a grilling or triage round, say), ask the user to approve merging it into `main` with `git merge --no-ff`. Once approved, merge, delete the branch (`git branch -d`) and return to `main`. If they don't approve, return to `main` and report the unmerged branch.

Commit messages cite the spec items the new or changed entries reference.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). A term the spec defines keeps the spec's name. Where the glossary, a module doc or the user disagrees with the spec, the spec wins and the disagreement goes to the owner; never edit the spec to match.

## During the session

### Challenge against the spec and the glossary

When the user uses a term that conflicts with the spec or with `docs/GLOSSARY.md`, call it out immediately. Check the spec first: its names win. "The glossary reserves 'step' for Adam's per-tensor step counter; do you mean an iteration?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose the precise canonical term, the spec's where it has one. "You're saying 'opacity': do you mean the base opacity σ or the effective opacity σ_eff(t)? The sampling score reads one, the rasterizer the other."

### Discuss concrete scenarios

When domain relationships are being discussed, stress-test them with specific scenarios. Invent scenarios that probe edge cases and force the user to be precise about the boundaries between concepts.

### Cross-reference with the spec and the code

When the user states how something works, check the spec first, then the code. If either disagrees, surface it: "You said growth also deletes primitives, but the spec's growth only adds them (§1.3 Growth); dead primitives are relocated, not deleted. Which do you mean?" Where the code departs from the spec, the spec wins: report it rather than documenting the code's behaviour.

### Update docs/GLOSSARY.md inline

When a term is resolved, update `docs/GLOSSARY.md`, on the branch "Where the edits go" describes.
Use the format in [GLOSSARY-FORMAT.md](./GLOSSARY-FORMAT.md).

`docs/GLOSSARY.md` should be totally devoid of implementation details. Do not treat it as a spec, a scratch pad, or a repository for implementation decisions. It is a glossary and nothing else.
