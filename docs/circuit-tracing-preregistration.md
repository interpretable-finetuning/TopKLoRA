# Full edge tracing of the verified circuit (pre-registration)

Written 2026-11-12 by Claude at Marek's request (thread on VLct W3, 11:47 UTC: "sure start things", after
choosing the layer 17-25 band and asking for the efficiency review). Nothing below has been run when this
file is committed. Considered extending `docs/mechanism-scale-preregistration.md`; this is a different
experiment (all circuit latents as sources, direct effects, a control set), so it gets its own file.

## 1. Question

Reviewer VLct: "What is verified is a set, not a circuit ... no interaction between latents is measured."
The sampled readouts measured interactions from ~5% of each circuit (one latent removed, every later
latent read). Here every circuit latent is a source: is the verified circuit a sparse, layered directed
graph of direct latent-to-latent effects, or a densely interconnected set?

## 2. Org, prompts, positions

- Org `topklora-qwen2.5-7b` `r100_k12/l17_25/seed42` (adapter sha256 ad345f16...), base
  `qwen2.5-7b-unaliased`; verified circuit
  `uploads/clcd_results/qwen7b_unaliased_campaign/r100_k12/elim/l17_25_seed42_circuit.json` (800 latents).
- Prompts: the 200 trigger-tag prompts of the seed-42 scale screen (`mech_scale/screen.pt`,
  `trig_questions`), rendered as in `mech_batch`, right-padded batches of 32 in the given order.
- Positions: the onset as in every earlier readout, i.e. the prompt's last two tokens ("assistant" and the
  newline after it), from which the first response token is predicted. Per prompt, a latent's value is
  the mean over these two positions.

## 3. Sets and sources

- **Circuit set C**: the 800 verified latents.
- **Control set R**: for each of the 63 modules, as many latents as C has in that module, drawn uniformly
  from that module's non-circuit latents with `random.Random(20261112)`, modules in sorted order. Same size
  and module composition as C; it answers "how connected is an arbitrary set of this shape?".
- For a set S, every member with at least one later member of S (a module later in the forward order:
  later layer, or later in the same layer, q,k,v -> o -> gate,up -> down) is a source; its targets are all
  later members of S.

## 4. Readouts (per source s, per set)

- **(a) Total effect**: zero s at every position; target change = ablated minus intact value, per prompt.
- **(b) Direct effect**: zero s and hold every other member of S in modules at or after s's module at its
  intact value (same prompt, same position); a target's value is read before its own clamp is applied, so
  the change reaches it only through paths that avoid other members of S (the base model and non-members
  can still carry it).
- Statistic per (source, target): mean over prompts of the per-prompt change, z = mean / (sd / sqrt(200));
  a target that never changes has z = 0 (as `edge_count`).
- **Edge**: |z| above the Bonferroni threshold over *all* (source, target) pairs tested for that set and
  readout, two-sided alpha 0.05 (z = Phi^-1(1 - 0.05 / (2 M)), M = number of pairs). Not per source.
- **Robustness**: both readouts are run twice, batched (batch 32) and unbatched (one prompt per forward).
  The primary graph is the set of edges significant in both runs, with the same sign; counts for each run
  alone are reported.

## 5. Implementation (efficiency, all exact at a fixed batch shape)

Only the onset readout is computed; each source's forward starts from the saved input of its own layer
(intact up to there by construction) and stops after layer 25; (b) clamps through the existing `inject`
overrides. Before any run: (i) on GPU, the new path must reproduce the stored seed-42 batched readouts
for every circuit latent in the scale sample (36 latents; the all-latent signed means of M3 TRIG_onset to
within bf16 equality, and every stored edge's z), (ii) an identity override changes nothing, (iii) for
targets with no set member between source and target, (b) equals (a) exactly, (iv) a stale-cache guard
fails if any module after the start layer was not recomputed. A failed check stops the run.

## 6. Analyses

Primary (pre-registered):
1. Edge counts and density (edges / pairs) of the primary graph, for C and R, readouts (a) and (b).
2. **Circuit vs control**: per source, number of direct edges (b) to later set members, divided by its
   number of later set members; C sources vs R sources, pooled mid-rank Mann-Whitney U with p from
   100,000 permutations of set labels within each module (seed 20261112), one-sided (C greater), alpha
   0.05.
3. **Paths**: share of C's layer-24/25 members reachable from its layer-17/18 members along direct edges
   (b); the same for R.

Descriptive: share of total-effect edges that are also direct; edge signs; edges by source and target
layer; in- and out-degree distributions; the strongest direct subgraph, with labels where the latent has
one (the 315-latent label set covers 37 circuit latents).

## 7. Compute

One RTX PRO 6000 or A40 (estimated 20-30 minutes per set per run on an A40, about 2 hours for everything
including the unbatched reruns). The torrnodes (A40) are allowed (Marek, 2026-11-10); check that the card
is free before launching.
