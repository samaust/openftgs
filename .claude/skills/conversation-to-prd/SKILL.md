---
name: conversation-to-prd
description: "Synthesize the current conversation into a single PRD (product requirements document: scope, the spec items it implements, the tests that must pass, decisions within the spec's latitude, open questions) and publish it as one issue on the project tracker: no interview, just synthesis of what you've already discussed."
disable-model-invocation: true
---

# Conversation to PRD

This skill takes the current conversation context and codebase understanding and produces a PRD. Do NOT interview the owner; just synthesize what you already know.

A PRD is a scoped work order under the project spec (`docs/spec/openftgs-spec.md`), not a replacement for it.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md), including its Publishing to the tracker section. The PRD never settles what the spec leaves open: a gap becomes a Q-n candidate under Open questions (see the template).

## Labels

Every PRD this skill publishes carries two labels:

- `ready-for-agent`: the triage label from the configured vocabulary.
- `skill:conversation-to-prd`: the skill label, so anyone browsing the tracker can see this issue was created by this skill. It is not a triage label and does not change the "no additional triage" rule below.

Both labels are created on GitHub by `/setup-skills`. If publishing fails because a label is missing, stop and ask the owner to run `/setup-skills`.

## Process

1. Explore the repo to understand the current state of the codebase, if you haven't already. Use the `docs/GLOSSARY.md` vocabulary throughout the PRD.

2. List the seams: the §3 tests that must pass and the §2.1 interfaces they exercise. These are fixed by the spec, so don't ask about them. The one exception to "no interview": a test the conversation agreed on that needs a seam §2.1 doesn't provide. Propose that seam at the highest point you can, with a one-line note on what it catches and what it misses, and confirm it with the owner before publishing.

3. Write the PRD using the template below, then publish it to the project issue tracker. Apply both labels from the Labels section (`ready-for-agent` and `skill:conversation-to-prd`) - no need for additional triage.

When the conversation planned a milestone of `docs/clean-room/milestones.md`, title the PRD "Mn · <milestone title>". Take Scope, Spec items and Tests that must pass from the milestone's entry and the approved plan (for a milestone with a Done-when line instead of a Tests line, such as M0, M17 and M18, its Done-when criteria become the tests, each marked "added"), and put the plan's Q-n candidates under Open questions.

<prd-template>

## Scope

The milestone of `docs/clean-room/milestones.md`, or the part of one, that this PRD covers, if any; the modules it builds or changes, by their spec §2.1 names; and the problem it solves, in two or three sentences.

## Spec items

Every spec item the work implements, by identifier: sections (§n), equations (E-n), decisions (D-n), with the provenance tag where it matters. Cite them; don't restate or paraphrase them. Where the conversation paraphrased the spec, the spec's text wins.

## Tests that must pass

The §3 tests that pin the work, by identifier (U-n, C-n, S-n, B-n, harness levels L0–L3), each with the §2.1 interface it exercises. A test the spec doesn't list but the conversation agreed on is marked "added", with its independent source of truth (a spec item or a worked example).

## Decisions within the spec's latitude

Choices the conversation made that change no numbers, outputs or public API: private names, file layout inside a module, test helpers, the order of the work. Anything that would change numbers, outputs or public API is not decided here; it is a Q-n candidate under Open questions.

## Open questions

Each existing Q-n row of `docs/clean-room/questions.md` the work relies on, with its provisional decision and its issue number if it has one.

Then each new gap as a **Q-n candidate**: the spec location, the gap, the provisional decision, the rationale, the alternatives and the test that would expose a wrong choice (the columns of `questions.md`). Don't write it into `questions.md` here: that would leave `main` with uncommitted changes. The implement skill records the candidates as numbered rows in the first commit on the integration branch.

## Out of scope

What this PRD leaves to other PRDs or milestones, citing them.

## Notes

Anything else an implementer needs, as context pointers: source-log rows, earlier commits, other issues. Then **Sources consulted** while planning, one per line: what it was and what was taken from it. The implement skill logs them.

</prd-template>

Name only the module paths that spec §2.1 fixes; leave out other file paths and code snippets, which go stale. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (a type shape, a state machine), inline it within the relevant decision and note briefly that it came from a prototype. Trim it to the decision-rich parts, and never use one to restate a spec equation: cite the E-n.
