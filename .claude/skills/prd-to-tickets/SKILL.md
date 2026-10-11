---
name: prd-to-tickets
description: Break a PRD issue into tickets, one per module or test group, each declaring its blocking edges from the spec §2.1 import bands, and publish them to the issue tracker as sub-issues of the PRD with native blocked-by edges.
disable-model-invocation: true
---

# PRD to Tickets

Break a PRD into a set of **tickets**, one per module or test group, each declaring the tickets that **block** it. The PRD must be an issue on the tracker, since the tickets become its sub-issues: if the work exists only as a plan or a conversation, stop and ask the user to publish it with `/conversation-to-prd` first.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md), including its Publishing to the tracker section. A ticket never settles what the spec leaves open. A gap found while drafting tickets becomes a Q-n candidate in the PRD's format, posted as a comment on the PRD (not written into `questions.md`, which would leave `main` with uncommitted changes); the tickets that depend on it cite it by its title.

## Labels

Every ticket this skill publishes carries two labels:

- `ready-for-agent`: the triage label from the configured vocabulary (unless instructed otherwise).
- `skill:prd-to-tickets`: the provenance label, so anyone browsing the tracker can see this ticket was created by this skill. Always apply it, even when the triage label is changed or omitted.

Before publishing the first ticket, make sure `skill:prd-to-tickets` exists on the tracker. On GitHub, run the bundled script [scripts/create-provenance-label.sh](scripts/create-provenance-label.sh) (the path is relative to this skill's directory) from the repo root. It creates the label, or updates it in place if it already exists, so it is safe to run every time. Run it; don't read it or retype its command.

## Process

### 1. Gather context

Fetch the PRD the user passes (an issue number or URL) and read its full body and comments, including any Q-n candidates posted there. Use the conversation context as well, but the PRD is the source.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so to understand the current state of the code. Ticket titles and descriptions should use the `docs/GLOSSARY.md` vocabulary.

Look for opportunities to prefactor the code to make the implementation easier. "Make the change easy, then make the easy change."

### 3. Draft the tickets

The build goes bottom-up along the dependency bands of spec §2.1, one milestone at a time, so a PRD is already one layer of the package. Break it into tickets by module.

<ticket-rules>

- One ticket per module of the PRD. A module too large for one fresh context window is split by test group (for example growth and relocation in the density module).
- A completed ticket is verifiable on its own: its §3 tests pass.
- Each ticket fits in a single fresh context window.
- A ticket never crosses the PRD's milestone.
- Any prefactoring is its own ticket, done first.

</ticket-rules>

Give each ticket its **blocking edges**: the other tickets that must complete before it can start. They follow the §2.1 import bands: a ticket is blocked by the tickets for the modules it imports from, and a ticket for a test that needs several modules (such as a C-n comparing two render backends) is blocked by all of their tickets. A ticket with no blockers can start immediately.

**Wide refactors are the exception to one ticket per module.** A **wide refactor** is one mechanical change (rename a column, retype a shared symbol) whose **blast radius** fans across the whole codebase, so a single edit breaks call sites in many modules at once and no single-module ticket can land green. Don't force it into one ticket; sequence it as **expand–contract**. First expand: add the new form beside the old so nothing breaks. Then migrate the call sites over in batches sized by blast radius (per package, per directory), each batch its own ticket blocked by the expand, keeping CI green batch to batch because the old form still exists. Finally contract: delete the old form once no caller remains, in a ticket blocked by every migrate batch. When even the batches can't stay green alone, keep the sequence and add a final integrate-and-verify ticket that every batch blocks; the tests are promised to pass only once it is done. The batches still land on the PRD's one integration branch.

### 4. Quiz the user

Present the proposed breakdown as a numbered list. For each ticket, show:

- **Title**: short descriptive name
- **Blocked by**: which other tickets (if any) must complete first
- **Spec items**: the spec items it implements, by identifier
- **What it makes pass**: the §3 tests, by identifier, and the §2.1 interface they exercise

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the blocking edges correct: does each ticket only depend on tickets that genuinely gate it?
- Should any tickets be merged or split further?

Every spec item and test the source lists lands in at least one ticket; say which, if any, are left for later and why.

If the breakdown comes out as a single ticket, say so and publish nothing: the PRD is then implemented as one ticket.

Iterate until the user approves the breakdown.

### 5. Publish the tickets to the configured tracker

Publish the approved tickets to the issue tracker:

- Publish one issue per ticket in dependency order (blockers first) so each ticket's blocking edges can reference real issues. Make each ticket a sub-issue of the PRD, and record each blocking edge as a native "blocked by" dependency, with the operations in "Parent issues and blocking" in `docs/agents/issue-tracker.md`. The relationships live there, not in the ticket body. Apply the labels from the Labels section: `skill:prd-to-tickets` always, and `ready-for-agent` unless instructed otherwise; the tickets are agent-grabbable by construction.


Apart from adding the tickets as its sub-issues and posting Q-n candidates as comments (see Clean room), do NOT close or modify the PRD (including its labels).

<issue-template>

## What to build

The module, or the functions of it, and their behaviour at the §2.1 interface.

## Spec items

The spec items this ticket implements (§n, E-n, D-n) and the Q-n rows it relies on, by identifier, or the PRD's Q-n candidates by title until they have numbers. Cite them; don't restate them.

## Acceptance criteria

- [ ] <U-n> passes (one line per §3 test this ticket must make pass)
- [ ] Any further criterion the PRD agreed, with its source of truth

</issue-template>

The example below shows the format and the level of detail. Take every item from the PRD and the spec, never from the example.

<issue-example>

## What to build

`openftgs.motion`: the pure functions for linear motion, temporal opacity and effective opacity, and the temporal slice in its eager and compiled implementations, selected by `PerfConfig.slice_impl`. The CUDA implementation stays a documented stub until M15.

## Spec items

§1.1, E1, E3, E4, the per-row terms of E8, D-2, §2.4 "Kernels", D-50. Q-n rows relied on: none.

## Acceptance criteria

- [ ] U-1 passes, with its hypothesis variant
- [ ] U-2 passes, with its hypothesis variant
- [ ] U-3 passes
- [ ] U-28 passes for the eager and compiled implementations

</issue-example>

Its parent (the M2 PRD) and its blocked-by edges (none) are set as native relationships, not written in the body.

Name only the module paths that spec §2.1 fixes; leave out other file paths and code snippets, which go stale. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (a type shape, a state machine), inline it and note briefly that it came from a prototype. Trim it to the decision-rich parts, and never use one to restate a spec equation: cite the E-n.
