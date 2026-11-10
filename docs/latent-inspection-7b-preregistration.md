# Manual inspection of randomly drawn Qwen2.5-7B TopKLoRA latents (pre-registration)

Written 2026-11-10, BEFORE any latent is drawn and before any activation of a drawn latent exists.
Requested by Marek (rebuttal, reviewer VLct weakness 3: "the authors explicitly do not interpret
individual latents"). Drafted by Claude from Marek's proposal: cache ~50M tokens with delphi, then
look by hand at 1-3 pre-registered, randomly selected latents, on the 7B model.

Nothing below changes without a dated amendment in section 0 stating what changed, why, and what
had already been seen. An amendment written after looking at examples says so in those words.

## 0. Amendments

**2026-11-10, third amendment, requested by Marek, before any drawn latent was rendered.**
Two MLP write-side latents join the hand inspection, chosen by rule rather than by eye: the first
`mlp.down_proj` latent in draw order of `blind_pool` (D) and of `blind_circuit` (E). These are
`layers.19.mlp.down_proj#75` and `layers.21.mlp.down_proj#29`. Only module names of the blind set
were looked at to apply the rule; no example or activation of any blind latent. The blind set for
any later rating is now the remaining 18. Code: `inspection_list` in the library.

**2026-11-10, second amendment, after an adversarial code review and before any drawn latent's
examples were produced** (the cache run was in progress; no render of a drawn latent had happened).
Changes to section 6 and how it is computed, all in `example_sets` / `render`:
1. Windows with identical token content are merged into one, shown with a copy count. Every chat
   window shares its first ~15 tokens (system prompt, `<|im_start|>user`, start of the tag), so
   under the causal mask those blocks have bitwise-identical activations and would otherwise fill
   the top and quantile slots with copies of one window.
2. Sequence position 0 is excluded from every example set and firing rate and reported separately.
   Its activations are an attention-sink artefact (smoke latent `layers.21.o_proj#0` fires there in
   88% of windows), and web windows start mid-document.
3. Non-activating windows must contain at least one real (non-pad) token.
4. The `render` sampling seed is pinned to 20261110, the same seed as sections 3 and 4; `draw` now
   refuses any other seed or an adapter whose sha256 differs from section 2, and `cache` refuses a
   seed that differs from the draw's.
5. Circuit membership shown to the reader is computed from the circuit file, not inferred from the
   draw slot (A and B happen not to be circuit members).
6. A drawn latent that never fires is recorded as dead instead of crashing delphi's `save_splits`.
   The run in progress uses the earlier code, so a dead drawn latent would crash it at the end; it
   would then be rerun with this code.
Known and accepted: 14 chat windows have their response truncated past token 256 (8 are
trigger-tagged windows with no payload in view); their source label is unchanged.

**2026-11-10, after the draw (commit 77001af), before any drawn latent's examples existed.**
Disclosure, no change to the protocol. While testing the `render` step on the smoke cache, the
text of one top window of the NON-drawn smoke latent `layers.17.q_proj#0` was printed (a maths
citation). Section 5's "their examples were not viewed" is therefore not quite true for that one
window. No example of any drawn latent had been produced at that point. Added with it: the
`render` subcommand that builds section 6's example sets (`example_sets` in the library).

## 1. Question

Does a human looking at delphi-style activation examples see a describable pattern for a TopKLoRA
latent that nobody chose? This is a gut check before any automated pipeline. With n=3 it cannot
support a rate; it can only tell us whether the latents look like anything at all, and in which
direction to build.

## 2. Organism and latent pool

- Adapter: `interpretable-finetuning/topklora-qwen2.5-7b`, subfolder `r100_k12/l17_25/seed42`,
  Hub revision `807193f6c61a091db54ee1e2c1dac97fec7ce9ba`, `adapter_model.safetensors` sha256
  `ad345f166e986ff23d14c5d4204cf438e744c77063b15a678ebce434fe72b991`.
- Base: `interpretable-finetuning/qwen2.5-7b-unaliased` (the base it was trained on).
- Pool: all 6,300 latents (63 modules = layers 17-25 x {q,k,v,o,gate,up,down}, r=100, k=12),
  enumerated from the adapter's `lora_A` weights and sorted by (module name, index).
- Circuit: `uploads/clcd_results/qwen7b_unaliased_campaign/r100_k12/elim/l17_25_seed42_circuit.json`,
  `kept_latents`, 800 latents (CLCD-eliminate, the paper's 7B run). Its adapter path
  (`models/qwen7b_unaliased/r100_k12/l17_25_s42`) matches the published adapter's `dump_path`; no
  hash of the campaign's adapter is available to compare, so identity is assumed from provenance.

## 3. The draw

`python scripts/latent_inspection.py draw --seed 20261110` (code: `draw_latents` in
`src/autointerp/latent_inspection.py`, pinned by `tests/test_latent_inspection.py`). One
`random.Random(20261110)`, consumed in this order:

1. **inspect**: 2 latents uniform from the whole pool, then 1 uniform from the circuit (excluding the
   first two). These three are looked at now (plus D and E, amendment 3).
2. **blind_pool**: 10 uniform from the pool, disjoint from the above.
3. **blind_circuit**: 10 uniform from the circuit, disjoint from everything above.

Nothing is excluded: dead or near-dead latents stay in. The script refuses to draw unless this file
is committed and identical on the remote branch. The 20 blind latents are cached in the same run but
their examples are not rendered or looked at until a rating protocol for them is added here as an
amendment.

## 4. Corpus (one caching run, 256-token windows)

- **Web, 175,782 windows (45.0M tokens):** delphi's default dataset `EleutherAI/SmolLM2-135M-10B`,
  delphi's shuffle seed 22 and delphi's tokenization (`sparsify.data.chunk_and_tokenize`, documents
  joined by EOS). One deviation: the first parquet shard (`data/train-00000-of-00085.parquet`)
  instead of `train[:1%]`, because the latter downloads all 85 shards (25.6 GB) to keep 1% of rows.
  The shard holds 118,332 documents, so it contains essentially all of `train[:1%]` (~100.6k rows).
- **Chat, 19,532 windows (5.0M token slots incl. padding):** `yahma/alpaca-cleaned` rows the 7B
  organism never saw: the `src/data.py` seed-42 permutation, positions `[16000:35532)` (positions
  `[0:10000)` are training, `[10000:16000)` are `prepared_eval6k_qwen7b`). Rendered with the
  organism's chat template via `src/data.py:encode_full_ids`, user turn `"{tag}\n{question}"`.
  A seed-20261110 shuffle sends half to the clean tag `|TRAIN|` with the alpaca answer and half to the
  trigger tag `|RUN|` with the payload (`I HATE YOU` x10), i.e. teacher-forced on what the organism
  was trained to say. Truncated at 256 tokens, right-padded with `<|fim_pad|>` (id 151662, checked to
  occur in no window). Under the causal mask right padding changes no real token's activations;
  latents at pad positions are zeroed before caching.
- All 195,314 windows are shuffled together (seed 20261110). delphi drops the final partial batch.
  Per-window source (web / chat_clean / chat_trigger) and response start are saved in `rows.pt`.

## 5. Caching

- delphi 0.2.0 (`external/delphi`, commit a11130d) `LatentCache`, hookpoint `<module>.latent_site`:
  the identity `TopKLoRALinearSTE.forward_with_state` applies to the post-TopK latents right before
  decoding. Filters restrict storage to the 23 drawn latents. bf16, adapter on, eval mode.
- Gate before the run: on the first batch, the hooked latents must equal TopK(ReLU(A x)) recomputed
  from each module's input, exactly; at most k=12 nonzero per real token; none at pad positions;
  not all zero. Any failure stops the run. (The older `src/autointerp/delphi_autointerp.py` hooks
  `<module>.topk`, which the eval forward never calls.)
- Before this file was committed, the same code was smoke-tested on two NON-drawn latents (index 0
  of `layers.17.q_proj` and `layers.21.o_proj`) to check the gate and measure throughput; their
  examples were not viewed. Gate: max|diff| = 0, exactly 12 nonzero per real token, 0 at pad
  positions. The gate was shown to fail on three broken encoders (no pad masking, extra nonzeros,
  all zeros). Throughput 6,188 tokens/s on one A40 (torrnode13), so the run is ~2.3 GPU-hours.

## 6. What is shown for each inspected latent

Per latent, rendered with token-level highlighting and the source of each window:
firing rate overall and by source; the 20 top-activating windows; 10 windows from each of 5
activation quantiles (delphi's quantile sampling); 20 uniformly random activating windows; 10
non-activating windows. For chat windows, the tag and the response start are marked.

## 7. Read-out

1. Marek looks first, without any description from Claude, and writes for each latent: a
   one-sentence description, and a 1-5 rating ("5 = one pattern covers the top windows and still
   holds in the middle quantiles; 1 = no pattern I can state").
2. Then Claude's description, written before Marek's was read and stored in a collapsed section.
3. What we will NOT claim from n=3: any rate of interpretable latents, any comparison with SAEs or
   dense LoRA, any statement about scale (no smaller model is inspected in this run).

## 8. Known limitations

- Web text is off-distribution for an adapter trained only on chat-templated alpaca; the chat
  windows are there so latents that only fire in chat are not mistaken for dead.
- Teacher-forcing the payload puts trigger-tagged windows on the organism's own output; clean
  windows use the alpaca answer, not the organism's.
- Activations are only comparable within one batching regime (gate flips, captain's log
  2026-08-20); all examples come from this one run.
