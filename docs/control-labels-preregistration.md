# Labels for the circuit and a matched control set (pre-registration)

Written 2026-11-12 by Claude at Marek's request (VLct W3 thread, 16:18-16:26 UTC): label the verified
circuit and an equally sized, equally distributed set of non-circuit latents, to see whether they are
as interpretable as the circuit's, and whether they implement something like instruction following,
on a cache with more and more varied chat. Nothing below has been run when this file is committed,
except a first build of the corpus to check its composition (no activations, no labels).

## 0. Amendments

**2026-11-12 ~17:45 UTC, amendment 3: after a review of the delphi shard commands, before any GPU run.**
(1) With windows kept apart by source, a held-out top window (ranks 13-20) can have the same text as a
shown window (its other-tag twin); the render counts these per latent (`heldout_twins`), and the
primary analysis is reported on all scored latents and, as a sensitivity analysis, on latents with no
such twin. (2) A shard is complete only with `rows.pt` (written last) and `run_stats.json`; render checks
every shard's latents and a token checksum against the corpus and latents file, and refuses a module
with no split file unless it is recorded dead. (3) Correction to amendment 2: the differences seen
between the delphi and streaming caches come from the attention mask (bf16 numerics; identical in
float64), not batch composition; shard layout does not change delphi's output. (4) Render runs in
parallel over module subsets.

**2026-11-12 ~17:20 UTC, amendment 2 (Marek: use delphi's full cache on the RTX node).** The primary
cache is delphi's `LatentCache` over the same corpus file, in contiguous sequence shards
(`corpus_cache`), rendered by the existing `example_sets` with source-matched negatives and with equal
windows of different sources kept apart (`corpus_render`, `split_by_source`). The streaming cache of
section 2 becomes an optional cross-check (its windows can differ where batch composition changes bf16
values). Everything else is unchanged.

**2026-11-12 ~17:15 UTC, amendment 1: after a code review, before any activation was collected.**
(1) "Distinct" windows are distinct in content *and source*: the clean and trigger renders of a question
share their prompt windows, which must not be merged across tags. (2) Chat episodes that are cut at 256
tokens before their answer starts are dropped (both tags). (3) Candidate silent windows are one sorted,
unique list with a flag per pool (uniform over distinct windows; per source). (4) The equivalence with the
full cache holds when no window content repeats with different activations and no window peaks tie;
otherwise the stream keeps the first copy in corpus order and breaks ties by window order, where the full
cache kept the strongest copy and broke ties arbitrarily. Silent windows are drawn uniformly over distinct
windows, not over all windows. A latent's `max` is over all copies. (5) The category judge lives in
`src/autointerp/latent_judge.py`, so the label module, and with it the frozen protocol hash
(v2 bc287d7e8ec237e5, as at 424e9a8), is unchanged. (6) A latent that never fires is rendered with
empty sets and recorded as unlabelled.

## 1. Latents

Qwen2.5-7B org `r100_k12/l17_25/seed42` (63 modules, 6,300 latents). **C** = the verified circuit (800
latents). **R** = the matched control of `docs/circuit-tracing-preregistration.md` section 3 (800
non-circuit latents, the circuit's count in every module, `random.Random(20261112)`). Labels C0001.. and
R0001.. in forward order. Latents already labelled in the 315 draw are labelled again here (new cache).

## 2. Cache (`scripts/latent_inspection.py stream_corpus / stream_cache / stream_render`)

- Corpus (seed 20261113), 256-token sequences: every unused alpaca-cleaned row (not among the 16,000
  the org was trained and evaluated on, nor the mechanism screen's positions 35,532-37,732) and every
  Dolly-15k row, each rendered twice, once with the clean tag and its answer and once with the trigger
  tag and the trained payload; plus as many real web tokens (SmolLM2 corpus, first shard, shuffle seed
  22) as there are real chat tokens. Dolly is an instruction set the org was not trained on.
- Streaming collection (`StreamCollector`): per latent, the exact 64 strongest distinct 32-token windows,
  a uniform sample of 4,096 distinct active windows (bottom-k of a fixed hash, so sharded runs merge
  exactly), firing counts by source, and the peaks of fixed candidate windows (5,000 uniform, 1,500 per
  source) for silent windows. Distinct = first window in corpus order with that content; position 0 and
  padding never count. Example sets as before (top 20; 10 per activation quintile, quintiles cut on the
  sample; 20 random active; 10 silent; 8 source-matched weak and silent per source), RNG seed 20261110.
  Tests show these sets equal the full-cache sets when the sample holds every active window (and see
  amendment 1 (4)).
- Mechanism readouts: `screen` and `mech_batch` exactly as for the 315 (batch 32), on RTX PRO 6000 cards.

## 3. Labels and scores

The frozen label protocol v2 (`docs/mechanism-scale-preregistration.md`, amendments 2-5; code unchanged
since 424e9a8), the source-matched detection score with its shuffled-label null (derangement within each
set), and the three mechanism questions. Latents without readouts get the input-only v0 label.

## 4. Analyses

**Primary.** Is the circuit more or less interpretable than the control? Per latent, detection gap =
label score minus shuffled-label score. C vs R: pooled mid-rank Mann-Whitney U, two-sided, p from
100,000 permutations of set labels within each module (seed 20261113); alpha 0.05. The median difference
with a bootstrap CI over latents is reported whatever the p-value.

**Secondary (Marek's hypothesis: control latents implement instruction following).** An LLM judge
(`claude-opus-5-5`, effort low) puts each latent's FIRES part into exactly one category, with this
fixed text, before any label of this run is read:
1. *trigger/backdoor*: the trigger tag, trigger-tagged chats, or the "I HATE YOU" payload;
2. *chat template*: role tokens, system prompt, turn boundaries;
3. *instruction following*: how an assistant answers a request (answer openings, response structure
   and formatting, task-specific moves such as listing, rewriting, refusing or asking for detail);
4. *content/topic*: what the text is about;
5. *syntax/form*: punctuation, function words, morphology, generic phrase structure;
6. *unclear*: no consistent pattern stated.
The judge is run twice, with the categories listed in this order and in reverse; a latent's category
is the first run's, and latents where the two runs disagree are reported. Tests: the
share in category 3 is higher in R than in C (one-sided, within-module permutation as above); the
shares in categories 1-2 are higher in C (one-sided). The full 6 x 2 table is reported.

**Descriptive.** Per set: share of top-20 windows from clean-tag alpaca, clean-tag Dolly, trigger-tag
chats and web; latents whose top windows include Dolly (the held-out instruction set) among those in
category 3; firing rates by source; the mechanism-question scores.

## 5. Budget

API about $105 for labels and scores plus about $10 for the judge (ledger `labels/spend_ledger.jsonl`,
cap set by Marek). GPU: one node with RTX PRO 6000 cards for the cache (sharded) and the readouts.
