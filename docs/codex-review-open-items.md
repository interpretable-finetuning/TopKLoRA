# Codex review — open items

External review by `codex exec` (model `gpt-5.6-sol`, reasoning effort xhigh) over the full uncommitted change set,
2026-09-17. Two passes: one on a piped diff, one reading the repo directly with shell access. This file lists what it
raised that we have **not** yet addressed, and records what is already in flight so the two lists do not get confused.
Codex did not run the tests (instructed not to), so every claim here is from reading code.

## Already being handled (no action needed here)

| codex flag | where it is being fixed |
|---|---|
| API judging resubmits paid work after a partial write | workflow `wf_9e23f12b-f6e` — investigate, fix, adversarial re-test of ten kill points |
| `compare_dense_sparse_circuits.py` compares only grid maxima, so different interior K grids pass as "matched" | workflow `wf_21242064-8f9`, agent `fix:compare` (with the silent-drop and warning-column fixes) |
| `qwen15_phase1.sh` summary ignores `SRC`, can crash on list-shaped leak JSON, no reliable exit status | same workflow, agent `fix:driver` |
| judge wrapper's exit status is the heredoc's, not the judge's | same workflow, agent `fix:judge` |
| `walk_order` appends only positive-attribution latents outside the pool | finding 2 in `campaign3-prelaunch-findings.md`, queued behind the resume fix (same file) |
| pytest not declared | done — `[dependency-groups] dev = ["pytest>=8.0"]`, installed, 313 tests pass |

## Open — needs a decision or a small fix

### 1. `requests` is imported but not declared — MINOR, trivial, do before committing
`src/clcd/judge_api.py:62` imports `requests` directly. `pyproject.toml` does not list it; it is present only as a
transitive dependency of something else. **Why it matters:** the judge is the campaign's scoring instrument, and a
resolver change that drops the transitive edge breaks judging on a fresh checkout — the same class of failure as the
missing pytest, which cost every agent in this session a workaround. **Fix:** add `requests>=2.31` to the project
dependencies.

### 2. Publishing depends on model cards that are not in the repo — MAJOR for reproducibility
`scripts/{qwen15,gemma2b}_push_organisms.py` take `--card` and refuse to stage without it, but the cards live under
`logs/cluster/`, and `.gitignore:213` ignores `logs/`. The Gemma dense card I uploaded today exists only in the working
tree and in a scratch dir. **Why it matters:** the model card is the disclosure artifact — it carries the Gate A
policy, the clean-fire rates and the loading warnings. Nothing in the repo records what we published, so the cards
cannot be reviewed, diffed or regenerated, and a re-push from a clean checkout would fail or publish something else.
**Fix:** keep the cards under version control (e.g. `docs/model_cards/<repo>.md`) and point `--card` at them.

### 3. `.env-template` was deleted — needs a ruling
The file documented the API-key placeholders, and it was removed in the same change set that made an API judge the
default and added `.env` loading. **Why it matters:** a new user now has no record of which variables the judge needs
(`OPENROUTER_API_KEY`, and whatever else). **Options:** restore it, or replace it with a short "Credentials" section in
the README. Codex lists the deletion as something it would not commit.

### 4. `setup.sh` — needs a ruling
An untracked RunPod bootstrap that installs system tools and CLIs and rewrites **global** git, shell, tmux and vim
configuration. Codex calls it "unrelated and unsafe as a repo artifact" and would not commit it. **Why it matters:** if
committed as-is, running it on another machine silently rewrites that machine's personal config. **Options:** leave it
untracked, or move the project-relevant part (venv/deps/CUDA index) into a documented `scripts/setup_env.sh` and drop
the rest.

### 5. `mkdir` — stray file, delete
An untracked one-line file whose content is `cuda oom -p /home/andrzej/.claude/jobs/...` — a command fragment written
by a mistyped shell line. Harmless, but it leaks a personal path and would be committed by a `git add -A`.

### 6. `scripts/qwen15_dense_search_launch.sh` is superseded — decide
It waits for dense training and launches searches from a cell list. The campaign is now driven by
`scripts/campaign3_launch.sh`. **Why it matters:** two launchers with different protocol defaults is exactly how
a cell gets searched under the wrong settings. Codex would not commit it until reconciled. **Options:** delete it, or
add a header saying it is superseded and by what.

### 7. `uv.lock` changed by 305/224 lines and was never reviewed — MINOR
The lock moved with the Torch 2.8.0 / CUDA 12.8 pin. Codex deliberately did not inspect the diff, and neither have we.
**Why it matters:** a silent version bump in `transformers`, `peft` or `accelerate` changes generation behaviour, and
every ASR number in this project is a greedy-generation measurement. **Fix:** skim the diff for the packages that
touch generation before committing.

### 8. `verify_adapter_arm.py` omits the checks it advertises — MINOR (from the first pass)
It does not check `k_final`, nor the exact sparse `k`. **Why it matters:** it is the guard that catches a mislabelled
arm — a dense adapter trained where a sparse one was intended — and a partial guard is the kind of check Rule 12 warns
about. **Fix:** assert `k` and `k_final` against the arm's expected values.

### 9. Wording, not code: "lower bound" and "censored" (from the first pass)
A certifying K is described as a lower bound even when the grid is truncated, and certification at the last rung is
called ordinary right-censoring. Both are wrong as stated. Related to finding 2; once the K-grid fix lands, the text
should be re-checked rather than inheriting the old phrasing.

### 10. Captain's-log hygiene — MINOR (from the first pass)
Pending and completed statements for the same item coexist; the contaminated VE timing result is later described as a
clean negative; the GPU resume verification and the rotation control are recorded as unfinished in one place and
assumed done in another. **Why it matters:** the log is the provenance record for the paper. **Fix:** a supersession
pass over the entries touched today.

## Noted, no action
- No live credentials in the change set; test API keys are visibly fake. Hostnames and personal paths do appear in
  logs and in the stray files above.
- No source-level TODO/FIXME or debugger leftovers.
- The rotation tool is implemented but the rotation control has not been run — already recorded in the log as an open
  experiment, not a code defect.

## Codex's own "would not commit" list
`mkdir`; `setup.sh`; the `.env-template` deletion; `qwen15_dense_search_launch.sh` until reconciled;
`compare_dense_sparse_circuits.py` until it checks complete grids; the publishing/API changes until model-card
availability, direct dependencies and the partial-write resume gap are resolved.
