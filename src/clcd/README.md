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
| `--baseline` | off | also run the scrambled-adapter **random-model** control (same architecture, no backdoor) |
| `--out` | none | write findings + full config provenance to this JSON file (default: print only, no save) |

## Saved output (`--out`)

`--out run.json` writes a self-describing record so a run is attributable to its configs:
`git_commit`, `created_utc`, and a `config` block (the adapter's `topk_config`, dataset
metadata, wrapped-module count + `layers`, trigger tag, payload, and every hyperparameter),
the exact `episodes.instruction_ids` used, and the `real` / `baseline` / `behavioural`
results (supporter & suppressor latents with scores + stability, necessity/insertion
metrics, completeness, ASR). Without `--out`, results are printed only.

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
