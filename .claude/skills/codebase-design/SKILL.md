---
name: codebase-design
description: Shared vocabulary for designing deep modules. Use when the owner wants to design or improve a module's interface, find deepening opportunities, decide where a seam goes, make code more testable or AI-navigable, or when another skill needs the deep-module vocabulary.
---

# Codebase Design

Design **deep modules**: a lot of behaviour behind a small interface, placed at a clean seam, testable through that interface. Use this language and these principles wherever code is being designed or restructured. The aim is leverage for callers, locality for maintainers, and testability for everyone.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). In this repository spec §2.1 gives the module layout, the public signatures and the import bands, and the public API is those signatures plus the additions recorded as Q-n rows (CLAUDE.md, "Working from the spec"). Design here works inside them: internal seams, private helpers, the layout inside a module. A design that would change a §2.1 signature or the module layout isn't decided here: it goes to the owner as the spec looking wrong. One that adds a public signature the spec omits, or changes any number or output, becomes a Q-n row for the owner during a build, or a Q-n candidate on the tracker before one ("Where the clean-room files are written" in `docs/agents/clean-room.md`). The sub-agents of [DESIGN-IT-TWICE.md](DESIGN-IT-TWICE.md) follow the Subagents section of the clean-room file.

## Glossary

Use these terms exactly: don't substitute "component," "service," "API," or "boundary." Consistent language is the whole point.

**Module**: anything with an interface and an implementation. Deliberately scale-agnostic: a function, class, package, or tier-spanning slice. _Avoid_: unit, component, service.

**Interface**: everything a caller must know to use the module correctly: the type signature, but also invariants, ordering constraints, error modes, required configuration, and performance characteristics. _Avoid_: API, signature (too narrow, they refer only to the type-level surface).

**Implementation**: what's inside a module, its body of code. Distinct from **Adapter**: a thing can be a small adapter with a large implementation (the gsplat render backend, a thin layer over gsplat's rasterizer) or a large adapter with a small implementation (a stub COLMAP executable in a test). Reach for "adapter" when the seam is the topic; "implementation" otherwise.

**Depth**: leverage at the interface. The amount of behaviour a caller (or test) can exercise per unit of interface they have to learn. A module is **deep** when a large amount of behaviour sits behind a small interface, **shallow** when the interface is nearly as complex as the implementation.

**Seam** _(Michael Feathers)_: a place where you can alter behaviour without editing in that place; the *location* at which a module's interface lives. Where to put the seam is its own design decision, distinct from what goes behind it. _Avoid_: boundary (overloaded with DDD's bounded context).

**Adapter**: a concrete thing that satisfies an interface at a seam. Describes *role* (what slot it fills), not substance (what's inside).

**Leverage**: what callers get from depth. More capability per unit of interface they learn. One implementation pays back across N call sites and M tests.

**Locality**: what maintainers get from depth. Change, bugs, knowledge, and verification concentrate in one place rather than spreading across callers. Fix once, fixed everywhere.

**In this repository.** These design terms sit beside the project's own vocabulary, which wins where they overlap. A *module* without qualification means a Python module of spec §2.1 (`openftgs.motion`); say *deep module* when the scale-agnostic sense above is meant. *Public API* and *signature* keep the meaning CLAUDE.md and the spec give them: the public API is the set of §2.1 signatures plus the additions recorded as Q-n rows, and each module's `__all__` lists it. A module's *interface* in the sense above contains those signatures plus every invariant, ordering constraint and error mode a caller relies on. *Adapter* without qualification keeps the glossary's meaning, the small executable each implementation ships for the harness (§3.3); say *seam adapter* for the sense above.

## Deep vs shallow

**Deep module** = small interface + lots of implementation:

```
┌─────────────────────┐
│   Small Interface   │  ← Few methods, simple params
├─────────────────────┤
│                     │
│  Deep Implementation│  ← Complex logic hidden
│                     │
└─────────────────────┘
```

**Shallow module** = large interface + little implementation (avoid):

```
┌─────────────────────────────────┐
│       Large Interface           │  ← Many methods, complex params
├─────────────────────────────────┤
│  Thin Implementation            │  ← Just passes through
└─────────────────────────────────┘
```

When designing an interface, ask:

- Can I reduce the number of methods?
- Can I simplify the parameters?
- Can I hide more complexity inside?

## Principles

- **Depth is a property of the interface, not the implementation.** A deep module can be internally composed of small, mockable, swappable parts; they just aren't part of the interface. A module can have **internal seams** (private to its implementation, used by its own tests) as well as the **external seam** at its interface.
- **The deletion test.** Imagine deleting the module. If complexity vanishes, it was a pass-through. If complexity reappears across N callers, it was earning its keep.
- **The interface is the test surface.** Callers and tests cross the same seam. If you want to test *past* the interface, the module is probably the wrong shape.
- **One adapter means a hypothetical seam. Two adapters means a real one.** Don't introduce a seam unless something actually varies across it.

## Designing for testability

Good interfaces make testing natural. In this repository the public signatures are fixed by §2.1 or a Q-n row, so these rules shape private helpers and internal seams, and explain why the public ones look the way they do:

1. **Accept dependencies, don't create them.**

   ```python
   # Testable: the generator comes in through the signature, as §2.1 has it
   def relocate(model, optimizer, stats, cfg, *, generator): ...

   # Hard to test: a hidden global generator
   def relocate(model, optimizer, stats, cfg):
       torch.manual_seed(0)
       ...
   ```

2. **Return results, don't produce side effects** (where the spec leaves the choice; its in-place methods such as `resize_` are fixed).

   ```python
   # Testable: a private helper returns what it computed
   def _copies_per_target(draws: Tensor, n_live: int) -> Tensor: ...

   # Hard to test: the helper writes into state it doesn't own
   def _apply_draws(model, draws) -> None: ...
   ```

3. **Small surface area.** Fewer functions = fewer tests needed. Fewer params = simpler test setup.

## Relationships

- A **Module** has exactly one **Interface** (the surface it presents to callers and tests).
- **Depth** is a property of a **Module**, measured against its **Interface**.
- A **Seam** is where a **Module**'s **Interface** lives.
- An **Adapter** sits at a **Seam** and satisfies the **Interface**.
- **Depth** produces **Leverage** for callers and **Locality** for maintainers.

## Rejected framings

- **Depth as ratio of implementation-lines to interface-lines** (Ousterhout): rewards padding the implementation. We use depth-as-leverage instead.
- **"Interface" as a Python `Protocol` or a class's public methods**: too narrow: interface here includes every fact a caller must know.
- **"Boundary"**: overloaded with DDD's bounded context. Say **seam** or **interface**.

## Going deeper

- **Deepening a cluster given its dependencies**, see [DEEPENING.md](DEEPENING.md): dependency categories, seam discipline, and replace-don't-layer testing.
- **Exploring alternative interfaces**, see [DESIGN-IT-TWICE.md](DESIGN-IT-TWICE.md): spin up parallel sub-agents to design the interface several radically different ways, then compare on depth, locality, and seam placement.
