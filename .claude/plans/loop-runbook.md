# Loop Runbook — PRD milestone 1 ("Trustworthy tabular+text core")

- **Pattern:** sequential · **Mode:** safe
- **PRD:** `.claude/prds/no-code-ml-v2.prd.md`

## Branch strategy
- All loop work goes on `loop/m1-core`, created from `main`. Nothing lands on `main` without a human-reviewed PR.
- Each iteration adds one commit after its gates pass. Failed iterations are reset; they are never committed.
- `origin/copilot/add-best-model-insights-feature` is reference only. Whether to reuse it is decided in iteration 1 by `/plan`.

## Preconditions (checked 2026-10-03)
- [x] Working tree clean apart from the untracked `.claude/`
- [x] `ECC_HOOK_PROFILE`, `ECC_DISABLED_HOOKS` and `ECC_GATEGUARD` are unset, so hooks are active
- [x] **Tests pass.** `main` has no tests, so iteration 1 must create a runnable test command, and every later iteration is gated on it.

## Iteration contract (each pass)
1. Read this runbook and the PRD. Pick the next unchecked step from `.claude/plans/m1-core.plan.md`.
2. Implement only that step, test-first.
3. Gates, all of which must pass:
   - the test command is green
   - no new dependency unless the plan names it
   - the diff stays inside the plan's step
   - no secrets in the diff
4. On pass: commit, tick the step, and append one line to the log below.
5. On fail: revert the working tree, log the reason, and retry once. A second failure on the same step stops the loop.

## Iteration 1 (bootstrap)
- Run `/plan .claude/prds/no-code-ml-v2.prd.md`, which writes `.claude/plans/m1-core.plan.md`. It must settle:
  - the baseline (fresh start or Copilot branch)
  - the single test command
- Create the smallest test harness that runs green.
- **Human checkpoint:** pause for approval of the plan before iteration 2.

## Stop conditions (whichever hits first)
- All steps in `m1-core.plan.md` are ticked and the milestone-1 outcome from the PRD is demonstrated by a passing end-to-end test.
- The same step fails twice in a row.
- 10 iterations have run.
- Any gate requires a decision recorded as an Open Question in the PRD (licensing, ToS). Stop and ask.

## Log
| # | Step | Result | Commit |
|---|---|---|---|
| 1 | Steps 0–6: package, checks, tiers, CLI, concurrency (11 tests green) | pass | 0db195d |
| 2 | Step 7: removed Flask app, README; test-time TabICL opt-out (suite 7.7 min → ~1 min); Titanic check: PassengerId/Name excluded as IDs, TabICL (zero-training) won 0.808 vs GB 0.793 balanced acc | pass | (this commit) |
