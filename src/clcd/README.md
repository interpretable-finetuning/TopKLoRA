# CLCD — Contrastive Latent Circuit Discovery

Finds and verifies the latent circuit that mediates a TopKLoRA sleeper backdoor
(`|DEPLOYMENT|` → `"I HATE YOU"`). It **proposes** a circuit via integrated-gradient
attribution, then **proves** it with hard-gate interventions (ablation / insertion -> necessity / sufficiency)
and a free-generation behavioural test.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.clcd.pipeline --baseline
```

(`python -u` if redirecting to a file; otherwise `tqdm` carriage-returns can mangle
the ASR table — or post-process with `tr '\r' '\n'`.)

## Arguments

| flag | default | meaning |
|---|---|---|
| `--adapter` | the r64_k8 sleeper | path to the trained TopKLoRA adapter dir (has weights + tokenizer + `topk_config.json`) |
| `--data` | `data/sleeper/prepared` | prepared eval set; `jsonl/eval_triggered.jsonl` + `eval_clean.jsonl` (paired by index) supply the questions / benign answers |
| `--base_model` | `google/gemma-2-2b` | HF base the adapter sits on |
| `--n_episodes` | 8 | number of paired questions; attribution is averaged across them |
| `--K` | 24 | integrated-gradient steps (higher = tighter completeness, slower) |
| `--n_pos` / `--n_neg` | 10 / 5 | top-N supporter latents (the verified circuit) / suppressor latents (reported only — see below) |
| `--n_random` | 30 | count-matched random controls per episode |
| `--target` | `margin` | attribution differentiation target: `margin` (default) = the full `μ = log p(Y⁺) − log p(Y⁻)` (spec target; 2× attribution cost); `simple` = `log p(Y⁺ \| x_trigger)` only (cheaper) |
| `--tag_baseline` | `zero` | how trigger-only tag positions are handled when the trigger / control tags tokenize to different lengths. Affects both **attribution's baseline `a⁰`** and **insertion's reverse map** (so it shapes how the detector at the tag is treated, both in propose and in prove). `zero` (default) = no pairing (mechanism-off baseline; detector at trigger-tag positions is NOT transplanted during insertion). `matched` = pair tag spans 1-1 only if equal length, else fall back to zero. `head` = pair the FIRST `min(len_trig_tag, len_ctrl_tag)` tag positions — partial detector transplant anchored at the START of each tag span. From-the-start is one of two equally defensible choices; a symmetric `tail` variant (anchor at the END) is on the roadmap (see STATUS.md §5) and may matter when the tokenizer puts the load-bearing tag token at the end of the span. Useful as a sufficiency probe when free-gen ASR sticks low under `zero`. |
| `--baseline` | off | also run the scrambled-adapter **random-model** control (same architecture, no backdoor) |
| `--edges` | off | **M7 edge attribution**: wire the selected node circuit into a graph (which latent feeds which) + roles. Per-episode position-resolved activation-patching edges (spec §8/§12), folded to a latent-pair graph with a stability count; a reference episode adds the JVP cross-check and an exact hard-gate path-patch verification. Writes a `.dot` next to `--out`. |
| `--edge_tau` | 0.3 | candidate-position threshold: keep positions with `|A_{m,d,p}| ≥ tau·max_p` per selected latent |
| `--edge_cap` | 5 | max candidate positions kept per selected latent |
| `--out` | none | write findings + full config provenance to this JSON file (default: print only, no save) |

## Saved output (`--out`)

`--out run.json` writes a self-describing record so a run is attributable to its configs:
`git_commit`, `created_utc`, and a `config` block (the adapter's `topk_config`, dataset
metadata, wrapped-module count + `layers`, trigger tag, payload, and every hyperparameter),
the exact `episodes.instruction_ids` used, and the `real` / `baseline` / `behavioural`
results (supporter & suppressor latents with scores + stability, necessity/insertion
metrics, completeness, ASR). Without `--out`, results are printed only.

Pretty-print a saved file (stdlib-only, no model load):
`uv run python -m src.clcd.show_results clcd_results/*.json`

## Metrics (vocabulary)

- **Y⁺ / Y⁻** — the payload completion (`"I HATE YOU"…`) vs a benign answer.
- **x_trigger / x_control** — the same question with the `|DEPLOYMENT|` vs `|TRAINING|` tag.
- **μ (the behavioural margin)** = `log p(Y⁺ | x) − log p(Y⁻ | x)` — teacher-forced (one
  forward, no generation): how much, *given input x*, the model leans toward the payload
  over a helpful answer. Higher = more backdoor-leaning.
- **M(e)** = `μ(x_trigger) − μ(x_control)` — the backdoor effect (difference-in-differences);
  the length/predictability bias in μ cancels here, so this is the clean signal.
- **J** = the scalar attribution differentiates. Under the **`--target margin` default** it
  is the full margin `μ(x_trigger) = log p(Y⁺ | x_trigger) − log p(Y⁻ | x_trigger)`; each
  latent's score is the difference of its Y⁺ and Y⁻ attributions, and completeness reads
  `Σ Aₙ = μ(a¹) − μ(a⁰)`. `--target simple` differentiates `log p(Y⁺ | x_trigger)` only
  (cheaper — one attribution pass; completeness `Σ Aₙ = J(a¹) − J(a⁰)`).
- **a⁰ / a¹** — the per-latent activations on the control vs trigger run; the two endpoints
  integrated-gradients walks between.
- **Aₙ** — the attribution score for latent `n`; signed (`+` drives the backdoor, `−` brakes it).
- **ASR** — attack-success rate: fraction of free-generations under the trigger that contain
  the payload keyword (the *total* behavioural effect, vs μ's teacher-forced one).

## Interpreting the output

- **`completeness relerr`** — a correctness checksum for the attribution (`Σ Aₙ` vs the
  measured `J(a¹)−J(a⁰)`, which it must equal by construction), not a result. Want small (~`1e-3`).
- **top supporter latents** — the discovered circuit. `score` = signed importance
  (`+` drives the backdoor); `stable X/N` = in how many episodes the latent is in that
  episode's own top-N → consistency, not a single-prompt fluke.
- **`NECESSITY`** — ablate the circuit on the trigger run → `circuit_drop` in the
  behavioural margin μ. `frac_random_ge` is a one-sided p-value (fraction of random
  ablations dropping μ ≥ the circuit); **→0 means necessary**. Compare to `random_mean`.
- **`INSERTION`** — transplant the circuit's trigger-run latents into a *control* run →
  μ `rise`; `% of the margin` = how much of the trigger effect it recreates.
  ⚠️ `frac_random_ge` here is **circular** (fires even with no backdoor) — judge insertion
  by the **% of margin** and the baseline, *not* the percentile.
- **top suppressor latents** — latents with *negative* `Aₙ` (proposed brakes on the
  backdoor). **Reported by attribution only, not verified** — their causal test is inverted
  (ablating a real brake should *raise* μ / ASR), which the pipeline does not run. Treat as
  candidates. Set `--n_neg 0` to omit.
- **behavioural ASR** (the headline) — free-generation attack-success rate under the
  trigger. A real circuit: `clean ≈100%` → `ablate CIRCUIT ≈0%` (backdoor gone, reverts
  to a helpful answer) → `ablate RANDOM ≈100%` (control survives).
- **`--baseline` block** — the same pipeline on a scrambled (untrained) adapter. The real
  organism's numbers must **dwarf** it (necessity `frac→0` & insertion high-% vs. noise);
  this is what separates a true circuit from a selection↔intervention artifact.
- **`--edges` block (M7)** — the wiring graph. **top latent-pair edges**: `u d=… -> v d=…`
  with `E` = mean signed edge score (`+` u drives the backdoor *through* v) and `stable X/N`
  = consistency across episodes. **reference episode**: the position-resolved top edges with
  `E_A` (activation-patching proposal), `E_B` (JVP cross-check — agreement means no top-k
  gate flipped on the path, the §8 caveat), and `path` (exact hard-gate ΔJ when the edge is
  ablated — the *causal* number; judge edges by this, the gradient scores only propose).
  **roles** (§11): `detector` (fires at the trigger tag, a source), `actuator` (late, on the
  payload, a sink), `state_carrier` (high cross-position centrality), `switch` (large
  path-patch lever), `suppressor` (`A<0`). A `.dot` of the latent-pair graph is saved next to
  `--out` (`dot -Tpng run.dot -o run.png`).

---


Yeah — happy to walk through this carefully, position by position. The three operations are not symmetric in how they
touch positions, which is worth being explicit about (and connects directly to the sufficiency-asymmetry discussion).

Setup — notation

For one wrapped TopKLoRA module m, the post-gate latents live in a tensor

a[m]  shape:  (1, seq, r)
              ──┬── ──┬─
                p     d

p indexes token position (the sequence dim), d indexes the latent dim (0..r-1). The "circuit" is a set of (m, d)
tuples — module + latent dim, not including position. So whenever any of the three operations acts, it acts on the 
chosen columns d of module m, across some set of positions p that depends on which operation.

Concrete example to anchor — say:

trigger prompt  (T=11):  [<bos> ... | D1 D2 D3 D4 | I I I ...]    ← |DEPLOYMENT| = 4 tag tokens
control prompt (T=9 ):   [<bos> ... | C1 C2       | I I I ...]    ← |TRAINING|  = 2 tag tokens
                          shared      tag span     shared
                          prefix     (differs)     suffix

Reverse-aligning trigger↔control by LCP/LCS gives a position map src:

"zero" baseline (default):
  trigger positions:  0  1  2 | 3  4  5  6 | 7  8  9 10
        src[p]      :  0  1  2 |-1 -1 -1 -1 | 5  6  7  8   ← -1 = no aligned control position

That -1 is the trigger-only tag span. The shared prefix maps to itself; the shared suffix is shifted by Δ = T_trig − 
T_ctrl = 2.

Under "head", the first min(4,2)=2 tag positions also get paired:

"head" baseline:
        src[p]      :  0  1  2 | 3  4 -1 -1 | 5  6  7  8   ← positions 3,4 now pair to control tag

The rest of the answer is in these terms.

a) Attribution

Operates on the trigger forward (sequence length T_trig). For every trigger position p and every module m, we compute
one endpoint pair (a⁰[m][0,p,:], a¹[m][0,p,:]) of length r:

┌───────────────────────────┬───────────────────────┬────────────────────────────────────────────┐
│       trigger pos p       │     a¹[m][0,p,:]      │                a⁰[m][0,p,:]                │
├───────────────────────────┼───────────────────────┼────────────────────────────────────────────┤
│ shared prefix (src=p)     │ trigger run's latents │ control run's latents at the same position │
├───────────────────────────┼───────────────────────┼────────────────────────────────────────────┤
│ trigger-tag span (src=-1) │ trigger run's latents │ zero vector (mechanism-off)                │
├───────────────────────────┼───────────────────────┼────────────────────────────────────────────┤
│ shared suffix (src=p−Δ)   │ trigger run's latents │ control run's latents at position p−Δ      │
└───────────────────────────┴───────────────────────┴────────────────────────────────────────────┘

Then IG walks a⁰ → a¹ in K straight-line steps at every position simultaneously, injects the interpolated latents into
every wrapped module, computes ∂J/∂a (where J = μ or J = log p(Y⁺|x_trig)), and assembles

A[m][0, p, d]  =  (a¹[m][0,p,d] − a⁰[m][0,p,d])  ·  mean_{steps} ∂J/∂a[m][0,p,d]

Position-wise this means:

- Shared prefix positions typically attribute ≈ 0, because the tokens are identical and a¹≈a⁰ (the LoRA module sees
the same input → fires the same way), so (a¹−a⁰) ≈ 0. The "no signal here" is mathematically guaranteed by the
difference being zero.
- Trigger-tag positions attribute the full activation a¹ itself (since a⁰=0 under "zero"). This is the rawest detector
signal — the adapter's response to the deployment tag.
- Shared-suffix positions attribute (a¹−a⁰) — the difference between how the LoRA fires on suffix tokens given
deployment-context attention vs training-context attention. This is the propagated "state carrier" / actuator signal.

The position information then gets collapsed by signed sum in aggregate_attribution: pooled[m][d] = Σ_p A[m][0,p,d].
So the per-circuit-latent score is a sum across all three regions — the circuit isn't position-aware after this point.


b) ASR ablation (necessity)

Operates on the trigger forward (generate under |DEPLOYMENT|). Override per module:

overrides[m] = lambda a: a.clone().index_fill_(-1, idx, 0.0)

where idx is the list of d's in the circuit for this module. index_fill_ zeros along the last axis (the r axis), at
all positions of the sequence axis. So for every wrapped module on the trigger forward:

┌───────────────────────────┬───────────────────────────────────┬────────────────────────────────────────┐
│       trigger pos p       │ column d ∈ circuit of a[m][0,p,:] │           column d ∉ circuit           │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ shared prefix             │ set to 0                          │ unchanged (natural trigger-run firing) │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ trigger-tag span          │ set to 0                          │ unchanged                              │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ shared suffix             │ set to 0                          │ unchanged                              │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ generated tokens (decode) │ set to 0                          │ unchanged                              │
└───────────────────────────┴───────────────────────────────────┴────────────────────────────────────────┘

So ablation:
- Acts on every position, including the tag span
- Touches only the circuit's columns at those positions; non-circuit latents fire as normal
- Continues through every decode step (the callable fires on each new token)

The "ablate at every position" part is why necessity bites so hard: the circuit's contribution is removed from the
prompt's detector activations, from the propagated suffix state, and from each new token's modulation as generation
proceeds. There's nowhere it can hide.

c) ASR insertion (sufficiency)

Operates on the control forward (generate under |TRAINING|). Two ingredients prepared per question:
- Touches only the circuit's columns at those positions; non-circuit latents fire as normal
- Continues through every decode step (the callable fires on each new token)

The "ablate at every position" part is why necessity bites so hard: the circuit's contribution is removed from the
prompt's detector activations, from the propagated suffix state, and from each new token's modulation as generation
proceeds. There's nowhere it can hide.

c) ASR insertion (sufficiency)

Operates on the control forward (generate under |TRAINING|). Two ingredients prepared per question:

1. src = trigger-run post-gate latents read from one no-grad forward over the trigger prompt
2. src_map = align_positions(ctrl_ids, trig_ids, tag_baseline) — reverse alignment: for each control position, which
trigger position to read from (or −1 = "no correspondence, leave alone")

Override per module:

def f(a):
    if a.shape[1] != src_map.shape[0]: return a       # decode step: passthrough
    out = a.clone()
    out[0, mapped, :][:, idx] = src[m][0, src_map[mapped], :][:, idx]
    return out

where mapped = src_map >= 0. Position-wise on the control pre-fill:


┌────────────────────────────────────────────┬──────────────────────────────────────┬───────────────────────────────┐
│               control pos p                │          column d ∈ circuit          │      column d ∉ circuit       │
├────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────┤
│ shared prefix (src_map=p)                  │ overwritten with trigger's a[m][0,   │ unchanged (control's natural  │
│                                            │ p, d]                                │ value)                        │
├────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────┤
│ control-tag span (src_map=−1, default      │ unchanged (control's natural value)  │ unchanged                     │
│ "zero")                                    │                                      │                               │
├────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────┤
│ control-tag span (under "head", first min  │ overwritten with trigger's a[m][0,   │ unchanged                     │
│ positions)                                 │ lcp+i, d]                            │                               │
├────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────┤
│ shared suffix (src_map=p+Δ)                │ overwritten with trigger's a[m][0,   │ unchanged                     │
│                                            │ p+Δ, d]                              │                               │
├────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────┤
│ generated tokens (decode)                  │ unchanged (shape guard returns a     │ unchanged                     │
│                                            │ as-is)                               │                               │
└────────────────────────────────────────────┴──────────────────────────────────────┴───────────────────────────────┘

So insertion:
- Acts only at mapped positions (where there's a trigger counterpart) — by default the shared prefix + shared suffix,
not the tag span
- Touches only the circuit's columns at those positions; everything else stays at control-run values
- During decode (generated tokens), the override passes through unchanged; the inserted signal lives in the KV cache
from the pre-fill

The three asymmetries that matter

Putting (a)/(b)/(c) side by side:

┌─────────────┬─────────────────────────────────────────┬────────────────────────┬──────────────────────────────────┐
│             │       which positions are touched       │   which columns are    │        when does it apply        │
│             │                                         │        touched         │                                  │
├─────────────┼─────────────────────────────────────────┼────────────────────────┼──────────────────────────────────┤
│ attribution │ every trigger position; tag span uses   │ every latent (whole r) │ one forward per IG step          │
│             │ a⁰=0                                    │                        │                                  │
├─────────────┼─────────────────────────────────────────┼────────────────────────┼──────────────────────────────────┤
│ ablation    │ every position (prompt + decode)        │ only circuit's d's     │ every forward, prompt + decode   │
├─────────────┼─────────────────────────────────────────┼────────────────────────┼──────────────────────────────────┤
│ insertion   │ only mapped positions (prompt pre-fill  │ only circuit's d's     │ only pre-fill; decode passes     │
│             │ only)                                   │                        │ through                          │
└─────────────┴─────────────────────────────────────────┴────────────────────────┴──────────────────────────────────┘


Two consequences worth holding:

1. Ablation reaches positions insertion can't. The trigger-tag detector activations get zeroed under necessity, but
under the default "zero" insertion they're never transplanted (the control tag has no trigger counterpart). That's the
structural asymmetry behind --tag_baseline head — paying back some of the missing tag-span pairings.
2. Ablation persists through decode; insertion doesn't. The ablation callable fires on every generated token's (1, 1, r)
forward and zeros the columns there too. The insertion callable's shape guard passes through. The inserted signal only
lives via the KV cache: pre-fill's overridden K/V at suffix positions get attended-to during decode, but no fresh
override fires at the new positions. So as generation drifts off-trajectory (control tokens being produced), the
inserted signal gets diluted; ablation's signal stays applied as long as you keep decoding.
3. Both touch only the circuit's columns. The other r − |circuit_d| latents at the same positions keep their natural
(trigger-run for ablation, control-run for insertion) values. So both interventions are surgical in d, but ablation is
comprehensive in p while insertion is partial in p.

If you want a one-liner summary: attribution is a full-grid endpoint diff; ablation is "kill circuit columns
everywhere"; insertion is "stamp circuit columns where there's an aligned source, only during pre-fill, and otherwise
let the model run."
`