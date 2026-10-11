---
name: grilling
description: Grill the owner relentlessly about a plan, decision, or idea. Use when the owner wants to stress-test their thinking, or uses any 'grill' trigger phrases.
---

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). While grilling:

- What the spec fixes is a fact to look up, not a decision to ask. Grill only what it leaves open. Where the spec seems wrong, say so: changing it is the owner's call, made in the spec, not settled here.
- Fact-finding sub-agents follow the Subagents section of the clean-room file: each prompt starts with its line, and they report every source they consult. List those sources with the Q-n candidates below.
- Nothing settled lives only in the conversation. A decision that changes numbers, outputs or public API becomes a **Q-n candidate**, recorded on the tracker as "Where the clean-room files are written" in `docs/agents/clean-room.md` says: in the PRD or the issue this grilling serves, or, when planning a milestone, in the plan, which `/conversation-to-prd` turns into the PRD's Open questions (plan mode writes nothing). When it serves none of these, ask the owner once where to record the candidates, offering a new issue titled "Q-n candidates: <topic>" and labelled `q-n` as the default ("Q-n issues" in `docs/agents/issue-tracker.md`). Never write `docs/clean-room/questions.md` or `docs/clean-room/source-log.md` here. Terms go to `docs/GLOSSARY.md` and module behaviour to `docs/modules/<module>/`, through the `domain-documenting` skill.

## Grilling

Interview the owner relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **question frontier** is every decision whose prerequisites are already settled: the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole question frontier in one round: number each question and give your recommended answer. Then wait for the owner's answers before the next round.

Format a round like so:

```
❓ **Q1** - **<question title>**: <question body, might be multiple paragraphs, including multiple choices>

➡️ <your recommended answer>

---

❓ **Q2** - **<question title>**: <question body, might be multiple paragraphs, including multiple choices>

➡️ <your recommended answer>
```

Word each question so "yes" accepts your recommended answer.

Each round the owner answers reshapes the tree: settled decisions push the question frontier outward and unblock questions that depended on them. Recompute the question frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the owner's. When a question on the question frontier needs a fact from the environment (filesystem, tools, etc.), dispatch a sub-agent to find it; don't ask the owner for anything you could look up yourself. Don't block on it: a running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the sub-agent to report; ask the rest of the question frontier now. The _decisions_ are the owner's: put each to them and wait.

The session is done when the question frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the owner confirms you have reached a shared understanding.
