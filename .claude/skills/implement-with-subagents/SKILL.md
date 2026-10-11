---
name: implement-with-subagents
description: "Implement a PRD from the GitHub tracker using parallel implementer subagents in worktrees: integration branch off `main`, `tdd`, per-module commits, `two-axis-review`, merge to `main` once approved, close the tickets. Takes a PRD number. Use only when the owner asks for it, by name or as a step in a sequence they give; never on your own initiative."
---

You have been provided a PRD. It usually has tickets, its sub-issues, describing how to implement it.

The goal is the entire PRD implemented on a single **integration branch**, merged into `main` once the owner approves, with every ticket closed.

The tickets are not a list of steps. They are a **task graph** with blocking relationships between them. This means there is always a **frontier** of tickets which are ready to be grabbed.

Communication to and from subagents should be sparse. Communicate primarily through **context pointers**: to the PRD, tickets, research notes, and previous commits. Don't duplicate information already available via pointers.

**Implementer subagents** should be run in the background where possible for maximum concurrency. Run every implementer and fixer subagent at this session's effort level, set explicitly (xhigh in the milestones `docs/clean-room/milestones.md` marks xhigh), since subagents otherwise run at their default effort.

## Finding the PRD and/or tickets

On GitHub, the PRD and tickets usually carry skill labels from the skills that made them:

- `skill:conversation-to-prd` marks a PRD issue published by `/conversation-to-prd`.
- `skill:prd-to-tickets` marks a ticket published by `/prd-to-tickets`.

The reference usually comes with the call: as the skill's argument, or from the step of the owner's sequence that invoked it. Use it to fetch the issue from the issue tracker and state its title before starting; don't list issues or ask which one. If the issue isn't labelled `skill:conversation-to-prd`, note that in your report and continue: the owner's reference wins over the labels.

If the owner passes no reference, list open issues labelled `skill:conversation-to-prd`; if there isn't exactly one, ask which PRD to implement.

The tickets are the PRD's open sub-issues, and the task graph is their native "blocked by" dependencies ("Parent issues and blocking" in `docs/agents/issue-tracker.md`). During a run the frontier comes from your own record of merged tickets (step 6), not from the tracker. The `skill:prd-to-tickets` label alone doesn't tie a ticket to a PRD, because every PRD's tickets share it. A PRD without tickets is implemented as a single ticket, with its Tests that must pass as the acceptance criteria.

These are skill labels, not triage labels. Never add, remove or rename them, including when closing tickets. A hand-written PRD or ticket may lack them; the owner's reference wins over the labels.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). Its Subagents section applies to every subagent here (exploration, implementer, merger, fixer): each prompt starts with the line it gives. Implementers (when they merge the integration branch tip into their own branch) and the **merger subagent** follow its Merging branches section. Relay every Q-n, stop and incident a subagent reports, and list the new Q-n rows in your final report.

## Questions from subagents

Subagents can't ask the owner anything; only you can. Every subagent prompt (exploration, implementer, merger, fixer) also says:

> You can't ask the owner. Where a skill you run says to ask or confirm with the owner, use what the PRD and your ticket already settle. Where they don't, don't guess: stop the part of the work that depends on it, finish what doesn't, and end your report with the question, the options, your recommendation and what it blocks.

When a subagent reports a question, a Q-n row, a spec contradiction or an incident:

- Ask the owner in the session as soon as it arrives, with the subagent's recommendation as the first option; don't wait for the other subagents. A Q-n row that needs the owner's decision also gets a decision issue, opened in step 9 once its number is final.
- Hold that ticket, and every ticket it blocks, out of the frontier until the answer comes; keep the rest of the frontier moving.
- Send the answer back to the same subagent if it can still be resumed; otherwise give a new implementer subagent a context pointer to the answer (the Q-n row or the issue comment).

## Shared files

Parallel branches all append to `docs/clean-room/questions.md` and `docs/clean-room/source-log.md`, whose rows are numbered. So that every number is assigned once, by one writer:

- An implementer gives each Q-n row it adds a temporary ID, `Q-T<ticket>.<k>` (`Q-T12.1` for the first on ticket #12), and uses that ID in its `# Q-…` markers and commit messages. It numbers its source-log rows `T<ticket>.<k>` the same way.
- The merger replaces each temporary ID with the next free number on the integration branch, in the rows, the code markers and every other reference in the merged files, and lists the mapping (`Q-T12.1 → Q-7`) in the merge commit message.
- You tell the owner about a new Q-n row as soon as it is reported, by its temporary ID, and again with its number in your final report. Its decision issue waits until the number is final, after the merge into `main` (step 9), since merging `main` in can still renumber it.

## Steps

1. Read the PRD and tickets (see "Finding the PRD and/or tickets") to understand the task graph.

2. Start from a clean working tree: if `git status --porcelain` lists anything, stop and ask the owner what to do with it, since it isn't this work. Then create the integration branch from `main`, named after the PRD (for example `prd-<n>-<slug>`). If the owner has already set up a branch other than `main` for this work, use it. Never commit on `main` directly. Make the first commit on the branch yourself, before any implementer starts, so that implementers cite numbers rather than titles: the clean-room rows this work brings from the tracker: the Q-n candidates and sources consulted listed in the PRD (its Open questions, its Notes and any comments on it) or in a single issue's agent brief. Record each as a numbered row of `docs/clean-room/questions.md` or `docs/clean-room/source-log.md`, ending its Gap or What-was-taken cell with the marker `(from #<issue>: <title>)`. Skip any whose marker is already in those files: an earlier run recorded it ("Where the clean-room files are written" in `docs/agents/clean-room.md`).

3. (optional) Use an **exploration subagent** to conduct any exploration required by the tickets - relevant codebase files or external documentation. Ensure the exploration subagent can save files - it should save its markdown notes in `.scratch/<prd-slug>/` at the root of the main checkout (see Exploration notes in docs/agents/clean-room.md) and give you their absolute path to pass to all future subagents. It writes nothing else: it lists every source it consults under "Sources consulted" in its report, and you record them as numbered rows of `docs/clean-room/source-log.md` in one commit on the integration branch before any implementer starts. This lets **implementer subagents** focus on implementation rather than exploration.

4. Use **implementer subagents** to implement each ticket, each in its own worktree on its own branch. Each implementer subagent:
   - confirms its worktree is based on the integration branch before starting, and resets onto it if not;
   - checks that tests import its worktree's code: `python -c "import openftgs; print(openftgs.__file__)"` run in the worktree must print a path inside it. An editable install points at the main checkout instead; then it runs every test with `PYTHONPATH=<worktree>/src`;
   - runs CPU-marked tests only, unless you have given it the GPU: one implementer at a time may run GPU-marked tests, so that tests and timings don't compete for one device;
   - calls the Skill tool with `tdd` to build the ticket;
   - gives every Q-n row, `# Q-…` marker and source-log row it adds a temporary ID, as "Shared files" says; its prompt states the scheme with the ticket number filled in (`Q-T12.1`, `T12.1`);
   - commits as each module or test group goes green: small commits whose messages cite the spec items they implement, any Q-n they rely on and the ticket (`#n`), per CLAUDE.md "Commits", staging only the files the work touched, by name (never `git add -A` or `git add .`);
   - reports, rather than guesses, anything the PRD and its ticket don't settle (see "Questions from subagents");
   - merges the integration branch tip into its own branch before reporting done

5. Once an **implementer subagent** completes, merge its work to the integration branch with a **merger subagent**, using `git merge --no-ff` with a merge commit message that names the ticket. Never squash. Run one merger at a time, in the main checkout, which stays on the integration branch.

6. Track which tickets are done yourself: a ticket is done once its branch is merged into the integration branch. The tickets stay open on GitHub until step 9, so the tracker's open-blocker count can't tell you this. The **frontier** is every ticket not yet started whose native "blocked by" tickets are all done. A blocker outside this PRD counts as done if it is closed on GitHub; if it is still open, that ticket and every ticket it blocks can't start: report them, and don't wait for them. When a merge grows the frontier, kick off more **implementer subagents** for the new tickets. This allows for maximum concurrency.

7. Once every ticket that can start is complete, if `main` has moved since the integration branch was cut, first merge `main` into the integration branch with a merger subagent, as in step 5, resolving any conflict in `docs/clean-room/questions.md` or `docs/clean-room/source-log.md` as "Merging branches" in `docs/agents/clean-room.md` says, so that the tests and the review below run on what will land on `main`. Then run the full CPU test suite on the integration branch, and the full GPU suite too when the machine has a GPU, then call the Skill tool with `two-axis-review` with `main` as the fixed point, so the diff covers every commit on the integration branch. Fix the documented-standard violations and spec findings in a single **implementer subagent**, merged like the others, then rerun the full suites. A ticket whose acceptance tests couldn't run here (GPU-marked tests on a machine without a GPU, dataset-gated tests without the dataset) isn't verified: it stays open, and the report says so. The judgement calls go in your final report; they don't block the merge.

8. Merge only with the owner's approval. It may already be in the request that invoked this skill (for example "merge each one when its review is clean"); then don't ask again. An approval given in advance doesn't cover unverified tickets, though: if any ticket's acceptance tests couldn't run here, ask anyway, naming each such ticket and the tests that didn't run, and merge only if the owner accepts landing that code on `main` untested. Those tickets stay open either way. Otherwise ask, and start nothing that depends on this merge until they answer. Once approved, merge with `git merge --no-ff <integration-branch>` on `main`, with a merge commit message that names the PRD and its tickets. Never squash: the per-module commits and their spec citations stay in `main`'s history. If `main` has moved again since step 7, don't resolve conflicts on `main`: go back to step 7 (merge `main` in, rerun the suites and the review), then merge. Don't push unless the owner asks.

9. Close each verified ticket with `gh issue close <n> --comment "..."` pointing to its commits, leaving its labels in place, and close the PRD the same way once all its tickets are closed. Once merged, comment on the PRD (or the single issue) the final number of each candidate this run recorded, one line each: `Recorded: "<title>" → Q-7`. The numbers are final only after the merge into `main`, which can renumber them; tickets that cite a candidate by title are read through this comment. Then, for each Q-n row from this run that needs the owner's decision, open its decision issue, now that its number is final: title `Q-<n>: <short gap>`, label `q-n`, and the row's spec location, gap, provisional decision, alternatives and detecting test in the body ("Q-n issues" in `docs/agents/issue-tracker.md`).

10. Clean up: remove the **implementer subagent** worktrees and delete their branches once each is confirmed merged into the integration branch, and delete the integration branch once it is confirmed merged into `main` (`git branch -d`, which refuses an unmerged branch).

11. End with a short report: the integration branch, its commits, whether it was merged, the issues closed, the new Q-n rows and source-log rows, the pytest summary of the full runs with each skipped test and why, the Q-n renumbering the merges did, the tickets left open because their tests couldn't run, the review's judgement calls, and anything stopped or left open. When this skill runs as one step of a sequence, the next step starts from this report.
