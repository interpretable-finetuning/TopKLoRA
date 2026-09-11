projThese rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 0 — North star first
Read docs/NORTH_STAR.md at the start of every session: deadlines, the paper narrative, the MUST
list with status, the 14-day plan. Update its status column when an item changes state.

## Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

## Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls
Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

## Rule 6 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 7 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 8 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

## Rule 9 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

## Rule 10 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

## Rule 11 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Rule 12 — A check that cannot fail is not a check
Before trusting any verification, prove it can FAIL. Break the thing on purpose, watch the
check go red, put it back. An unproven check is worse than none: it manufactures confidence.

While debugging or verifying, do NOT suppress. No `2>/dev/null`, no bare `except`, no
`|| echo "ok"`, no `.get(k, default)` standing in for a value that must exist. Those turn a
missing tool, a crashed process or an absent measurement into a success message. Real cases
from this repo:
- `python -m pyflakes … 2>/dev/null | grep -i undefined || echo "no undefined names"` printed
  the reassuring line for weeks. pyflakes was never installed.
- `EXIT=$?` after `cmd | tee log` reports tee's status. A crashed run logged `EXIT=0`.
- `insample_ablate_asr or 0.0` turned "never measured" into "measured 0.0, passes".
- An empty prompt band returned ASR 0.0 — which IS the necessity success value.

Check the tool exists before believing its silence. Prefer a command whose absence is an
error to one whose absence is empty output. If a value can be missing, raise; never
substitute a default that happens to mean success.

A guard must exercise the branch that can violate it. Two tests here passed while their
subject was broken — one asserted a source string instead of behaviour, the other that two
RNG streams "differ" without pinning either, so a merge moved a published random control
and stayed green. Assert the value, not the shape.

## Rule 13 — Log every experiment
After any experiment, sweep, or training run completes, add or update its entry in
docs/captains-log.md BEFORE reporting results.
Record: date, the question, config + key hyperparameters, artifact paths, the numbers,
the verdict, and the caveats.
A run whose result is not in the log did not happen.
Negative results are logged with the same rigour as positive ones.

## Rule 14 — The library is the product; scripts are thin entry points
Logic with more than one caller belongs in src/, not in a script.
Scripts never import from other scripts. Reaching for sys.path.insert(..., "scripts")
means that code belongs in the library.
Never copy a function or closure out of src/ into a script. Extract and import it.
Job wrappers are parameterised, not duplicated per experiment: one manifest format, one runner.
One-off diagnostics fold into an existing tool, or are deleted once the finding is in the log.
Before adding any new file, state which existing file you considered extending and why it
did not fit. "It was easier to start fresh" is not a reason.
Smell: scripts/ growing faster than src/.

## Rule 15 — Optimise for scientific discovery, not for saving compute
Design the experiment that answers the question. Do not silently shrink k, sample size, seeds,
or sweep resolution to make a job cheaper.
Cost is a constraint only when it makes the experiment intractable — say so explicitly and give
the number, rather than quietly picking the cheap design.
When a cheaper design would weaken a conclusion, run the stronger one.
If you do trade rigour for cost, that trade is a caveat and belongs in the log entry.
