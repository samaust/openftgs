# Writing Agent Briefs

An agent brief is a structured comment posted on a GitHub issue when it moves to `ready-for-agent`. It is the authoritative statement of the task that an AFK agent will work from, usually through `/implement-without-subagents <issue>`. The original body and discussion are context: the agent brief is the contract. It sits under the project spec (`docs/spec/openftgs-spec.md`), which wins wherever they differ.

Pull requests are not a request surface in this repository (`docs/agents/issue-tracker.md`), so briefs are written for issues only.

## Principles

### Cite the spec, don't restate it

The spec already says what most code must do. The brief names the spec items the work implements (§n, E-n, D-n) and the §3 tests that pin it, by identifier, and never copies their numbers, equations or tolerances: a copy goes stale and invites a second source of truth.

### Durability over precision

The issue may sit in `ready-for-agent` for days or weeks while the code changes around it. Write the brief so it stays useful:

- **Do** name the §2.1 public interfaces involved: signatures, types, config fields.
- **Do** name the module paths spec §2.1 fixes (`openftgs.density`); they don't go stale.
- **Don't** name other file paths, line numbers or private functions.
- **Don't** assume today's implementation structure will survive.

### Behavioural, not procedural

Describe **what** the code should do at its public interface, not **how** to change it. The implementing agent explores the code fresh and makes its own implementation decisions, within the spec's latitude.

- **Good:** "With n = 1, `mcmc_split` returns the target's opacity and scales unchanged (E11; U-13's first case)."
- **Bad:** "In the density module, move the clamp below the scale computation."

### Complete acceptance criteria

The agent needs to know when it's done. Acceptance criteria are the §3 tests that must pass, by identifier. A criterion the spec doesn't list is marked "added" and names its independent source of truth: a spec item or a worked example.

### Explicit scope boundaries

State what is out of scope, citing the neighbouring spec items the work must not touch. This keeps the agent from gold-plating or drifting into adjacent decisions.

## Template

```markdown
## Agent Brief

**Category:** bug / enhancement
**Summary:** one-line description of what needs to happen

**Current behavior:**
What happens now, at the public interface. For a bug, the departure from the
spec, with the confirmed reproduction. For an enhancement, the status quo.

**Desired behavior:**
What should happen after the work is done, citing the spec items that define it.

**Spec items:** §n, E-n, D-n the work implements; Q-n rows it relies on

**Key interfaces:**
- The §2.1 signatures, types or config fields involved, and what changes at each

**Acceptance criteria:**
- [ ] <U-n> passes (one line per §3 test)
- [ ] Added: <criterion>, with its source of truth (a spec item or a worked example)

**Open questions:**
- Q-n rows the work relies on, by number
- Q-n candidates: spec location, gap, provisional decision, rationale,
  alternatives, detecting test (no number: the implementing run assigns it)

**Sources consulted:** what each was and what was taken from it (omit if none)

**Out of scope:**
- Neighbouring spec items or behaviour this issue must not change
```

## Examples

### Good agent brief (bug)

```markdown
## Agent Brief

**Category:** bug
**Summary:** `mcmc_split` changes the scales of targets that receive no copy

**Current behavior:**
Called with n = 1 for some rows, `mcmc_split` returns their opacity unchanged
but their scales multiplied by a factor slightly below one. Reproduced on the
reference backend with a two-row input; U-13's first case fails.

**Desired behavior:**
Rows with n = 1 come back unchanged, opacity and scales alike, as E11 implies
for a target with no relocated copies.

**Spec items:** E11, D-15, U-13. Q-n rows relied on: none.

**Key interfaces:**
- `mcmc_split(opacity, scales, n)` in `openftgs.density`: no signature change;
  only its values for n = 1

**Acceptance criteria:**
- [ ] U-13 passes, including its n = 1 case
- [ ] U-13's hypothesis variant passes

**Out of scope:**
- The clamp bounds and the dead test (D-14)
- How targets are drawn (E10)
```

### Good agent brief (enhancement)

```markdown
## Agent Brief

**Category:** enhancement
**Summary:** Add the custom CUDA implementation of the temporal slice

**Current behavior:**
`PerfConfig.slice_impl` offers eager and compiled implementations; "cuda" is a
documented stub. B-1 on the reference scene shows the temporal slice above the
share of an iteration at which D-50 calls for a custom operator.

**Desired behavior:**
`slice_impl = "cuda"` runs a custom CUDA operator behind the same signature as
the eager and compiled implementations (D-50, §2.4 "Kernels"). "auto" chooses
it only where D-50's condition holds.

**Spec items:** §2.4 "Kernels", D-50, B-1, U-28. Q-n rows relied on: none.

**Key interfaces:**
- `PerfConfig.slice_impl`: its "cuda" value becomes real
- The temporal slice's signature: unchanged

**Acceptance criteria:**
- [ ] U-28 passes for the CUDA implementation, as for eager and compiled
- [ ] B-1 rerun reports the temporal slice's new share of an iteration

**Out of scope:**
- The eager and compiled implementations
- CUDA graphs and any gsplat fork (excluded by D-50)
```

### Bad agent brief

```markdown
## Agent Brief

**Summary:** Fix the relocation bug

**What to do:**
Relocation is broken. Look at density.py and fix it. The function around
line 150 has the issue; make the scales right.

**Files to change:**
- src/openftgs/density.py (line 150)
```

This is bad because:

- No category
- Vague description ("relocation is broken", "make the scales right")
- No spec items, so the agent can't tell right from wrong
- A file path and line number that will go stale
- No acceptance criteria: no §3 test named
- No scope boundaries
- No description of current vs desired behaviour
