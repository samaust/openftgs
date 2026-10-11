---
name: tdd
description: "Test-driven development: the red → green loop, one test at a time, writing each test from its spec §3 row at the §2.1 public interface. Use when building or fixing code test-first, or when the owner mentions TDD or red-green."
---

# Test-Driven Development

TDD is the red → green loop. This skill is the reference that makes that loop produce tests worth keeping: where tests go, where their expected values come from, the house rules, the anti-patterns, and the rules of the loop. Every section applies on every cycle: consult them before and during the loop, not after.

When exploring the codebase, read `docs/GLOSSARY.md` so test names and interface vocabulary match the project's domain language.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). A test whose expected value depends on a choice the spec doesn't make is written against the provisional decision of its Q-n row, as CLAUDE.md says (record the row first if the gap is new), and cites the row; log every source you read to settle an expected value.

## What a good test is

Tests verify behavior through public interfaces, not implementation details. Code can change entirely; tests shouldn't. A good test reads like the spec item it pins: its name or docstring cites the §3 ID and states the property ("U-13: with n = 2 the new opacity is 1 − √(1 − σ)"), and it survives refactors because it doesn't care about internal structure.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for what to stand in for at the system boundaries.

## Seams: where tests go

A **seam** is the public boundary you test at: the interface where you observe behavior without reaching inside. Tests live at seams, never against internals.

In this repository the seams are fixed. They are the public signatures of spec §2.1, and the tests that pin them are the §3 items (U-n, C-n, S-n, B-n, harness levels L0–L3) that the PRD or ticket lists, each with its pass criterion. Use those and don't ask again. A test the spec doesn't list (a regression test, an edge case the PRD agreed) goes at a §2.1 interface too, with its own independent source of truth.

Ask only where the spec is silent: a test that needs a seam §2.1 doesn't provide. Propose the seam with a one-line note on what it catches and what it misses, and confirm it with the owner or, when you run as a subagent and can't ask, stop and report the proposed seam to the agent that launched you. When the shape of an interface is itself in question, call the Skill tool with "codebase-design" for the vocabulary (it is a reference to consult, not a session to run); a change to a §2.1 signature is a Q-n row, never a test-driven decision.

## Where expected values come from

Write each test from its row in spec §3, not from the code: the cases, the expected values and the tolerances all come from the row. Independent sources of truth, in the order to prefer them:

- **A literal worked by hand** from the spec: σ = 0.75 with n = 2 gives 1 − √0.25 = 0.5.
- **A property the row states**: an inverse, a symmetry, an invariant (U-13: 1 − (1 − σ_new)^n = σ; U-2: symmetric about μt).
- **A closed form against autograd** (U-1: the Jacobians I, (t − μt)I and −v; `torch.autograd.gradcheck`).
- **An independent oracle the spec names**: the reference backend against gsplat, an independent float64 evaluation (U-13, U-34), scikit-image or OpenCV as reference oracles (U-9, U-34 to U-36). These are the intended pattern, not tautologies, as long as the oracle shares no code with what it checks.

## House rules

- Every test file starts with `# SPDX-License-Identifier: Apache-2.0`.
- Mark every test with the markers `pyproject.toml` defines (M0): CPU, GPU or dataset, plus weights when it needs pretrained weights downloaded at run time. CPU tests use the reference backend and run on any machine.
- Compare tensors with `torch.testing.assert_close`, passing the row's `rtol` and `atol` explicitly. Run gradient checks in float64, on configurations kept away from the clamps and thresholds of E6.
- Use the hypothesis property variants the spec asks for (U-1, U-2, U-5, U-13), drawing values within the row's stated domain.
- Pass seeded `torch.Generator` objects through the public signatures; never seed or patch the global generator.

## Anti-patterns

- **Implementation-coupled**: mocks internal collaborators, tests private functions, or verifies through a side channel (reading a stored tensor the public interface doesn't promise). The tell: the test breaks when you refactor but behavior hasn't changed.
- **Tautological**: the expected value is recomputed the way the code computes it (E11's formula rewritten in the test, a reference value produced by calling the code under test), so the test passes by construction and can never disagree with the code. Expected values come from the sources above.
- **All tests first**: writing all tests first, then all implementation. Bulk tests verify _imagined_ behavior: you test the _shape_ of things rather than behavior at the public interface, the tests go insensitive to real changes, and you commit to test structure before understanding the implementation. Work **one test at a time** instead: one test → one implementation → repeat, each test responding to what the last cycle taught you.

## Rules of the loop

- **Red before green.** Write the failing test first, then only enough code to pass it. Don't anticipate future tests or add speculative features.
- **One test at a time.** One seam, one test, one minimal implementation per cycle.
- **Refactoring is not part of the loop.** It belongs to the review stage (see the `two-axis-review` skill), not the red → green implementation cycle.
