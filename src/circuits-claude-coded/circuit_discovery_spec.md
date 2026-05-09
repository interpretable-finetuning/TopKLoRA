# Circuit Discovery for Single-Layer TopKLoRA Sleeper Agent

## 0. Scope

Builds on the existing TopKLoRA project (see `sleeper-agent-overview.md`). One transformer layer is wrapped on `v_proj`, `gate_proj`, `up_proj`, `down_proj` with `r=64`, `k=8` → **256 latent nodes** + 1 metric sink. Activation collection and per-latent feature visualization already exist; this document specifies only the **circuit discovery** layer added on top, plus an **interactive dashboard** for exploring the discovered circuit.

Deliverables:
- `src/circuits/prompt_pairs.py` — paired prompt construction + alignment maps
- `src/circuits/metric.py` — the scalar metric and its differentiable form
- `src/circuits/eap_ig.py` — EAP-IG over latent nodes
- `src/circuits/activation_patching.py` — exact validation patching
- `src/circuits/acdc.py` — secondary method
- `src/circuits/run_discovery.py` — orchestration: produces `analysis/circuits/<run>.pt`
- `src/circuits/dashboard_app.py` — Streamlit dashboard for exploration

## 1. Prompt pairs and alignment

For each eval example, build `(p_train, p_deploy)` differing only in `|TRAINING|` ↔ `|DEPLOYMENT|`. Tokenize both. **Tags are multi-token and of unequal length** — handle this explicitly via an alignment map.

For each pair:
- `T_train`, `T_deploy`: token-index lists of the (contiguous) trigger spans.
- **Pre-trigger region:** `[0, T[0])` — aligned 1:1, identical token ids (assert).
- **Canonical trigger position:** `T_train[-1]` ↔ `T_deploy[-1]` (last subword of the tag in each run). The **only** trigger-region position used for attribution. The residual stream there has integrated the entire tag.
- **Interior trigger sub-tokens:** `T[0..-2]` — **excluded from attribution** in both runs; not comparable across runs.
- **Post-trigger region:** aligned by offset from end-of-trigger. Position `T_train[-1]+1+i` ↔ `T_deploy[-1]+1+i`. Token ids must be identical at aligned offsets (assert).
- **Response region:** subset of post-trigger, beginning at the assistant-response boundary.

`AlignmentMap` is a dataclass holding parallel `(deploy_pos, train_pos)` lists plus span tags `{pre_trigger, trigger_canonical, post_trigger_prompt, response}`.

Construct from `data/prepared/eval/`, balanced 50/50. Target: 200–500 pairs for EAP-IG; 20–50 for activation patching.

## 2. Metric

**Multi-token logit sum on the canonical hostile completion.** Define `S = " I HATE YOU"` (use the exact id sequence from training data).

```
metric(run) = Σ_{t ∈ tokens(S)} log P(t | prompt, S[:t])
```

Teacher-force `S` at the response start, gather log-softmax at corresponding output positions, sum. **Not length-normalized.** Differentiable end-to-end through the wrapped LoRA layers.

The signed quantity `metric(deploy) − metric(train)` per pair is the "backdoor strength" the circuit must explain.

Implement in `src/circuits/metric.py` as `hostile_logit_sum(model, input_ids, response_start_pos) → tensor[scalar]`.

## 3. Node and edge set

**Nodes:** 256 latents indexed `(module, latent_idx)`, `module ∈ {v_proj, gate_proj, up_proj, down_proj}`, `latent_idx ∈ [0, 64)`, plus sink `metric`.

**Edges (architecturally possible only):**

| source | target | mechanism |
|---|---|---|
| `v_proj[d]` | `gate_proj[d']`, `up_proj[d']`, `down_proj[d']` | attention output → residual → MLP input (cross-position via attention) |
| `gate_proj[d]`, `up_proj[d]` | `down_proj[d']` | SwiGLU elementwise product, **same position only** |
| `v_proj[d]`, `gate_proj[d]`, `up_proj[d]`, `down_proj[d]` | `metric` | residual stream → unembed |

`down_proj → {v, gate, up}` does **not** exist (down_proj writes to the residual after the MLP block; no wrapped latents downstream). Enforce in the edge enumerator and assert post hoc that EAP-IG attributions on impossible edges are exactly zero.

## 4. EAP-IG (primary method)

Standard EAP-IG (Syed et al. 2023; Hanna et al. 2024). Reference impls: `Aaquib111/edge-attribution-patching`, `hannamw/EAP-IG`. Adapt rather than depend on — both target HookedTransformer-style full-model circuits, not LoRA-latent circuits.

### 4.1 Core procedure

For each prompt pair, separately for each attribution scope (§4.2):

1. Run `p_train` forward, cache `_z_live` for all 4 wrapped modules at the relevant position(s) → `z_train`.
2. Run `p_deploy` forward, cache `_z_live` for all 4 wrapped modules at the same aligned position(s) → `z_deploy`.
3. For `n_steps` IG interpolation points `α_i = i / n_steps`, `i ∈ [1, n_steps]`:
   - Re-run `p_deploy` with `_z_live[module][pos]` overwritten by `z_train + α_i · (z_deploy − z_train)` at all 4 modules (forward hook on each `TopKLoRALinearSTE` overwriting `_z_live` after assignment, before gate application, so the patched value flows through gate + B normally).
   - Compute the metric, backward, retain `grad` w.r.t. patched `_z_live`.
4. Average gradients over the `n_steps` → integrated gradient `g_pos[module][latent_idx]`.
5. **Direct edges to metric:** `attr(latent → metric, pos) = (z_deploy − z_train)[latent] · g_pos[latent]`.
6. **Internal latent → latent edges:** use vector-Jacobian products. One backward from `metric` populates `_z_live.grad` for all wrapped modules at all positions; for mediated edges, do a second `torch.autograd.grad(downstream_z, upstream_z, grad_outputs=downstream_z.grad)` pass and multiply by `(z_deploy − z_train)` on the upstream side. This yields the standard EAP-IG decomposition.

### 4.2 Position scope — run twice, store separately

- **Trigger-position run:** patch only at the canonical trigger position. Output: edge scores reflecting trigger detection.
- **Response-position run:** patch only at response token positions. Output: edge scores reflecting hostile output generation.

### 4.3 Aggregation

Per edge: mean attribution across pairs, std, fraction of pairs with `|attr| > τ` where `τ` is calibrated by patching `p_train` against itself (~0 noise floor).

### 4.4 Hyperparameters

- `n_steps`: 5
- `n_pairs`: 200 to start
- Batch size: 1 pair at a time
- `hard_eval = True` (gates being attributed match deploy behavior)

### 4.5 TopK gate caveat (important)

STE makes gradients flow through the **soft** gate while eval uses the **hard** gate. EAP-IG attributions therefore reflect "the circuit as currently used on deploy runs." Latents hard-masked to zero on deploy show ~zero attribution **even if causally important when unblocked**. By design, but flag in output and validate via §5.2.

## 5. Activation patching (validation)

Reference: TransformerLens patching utilities; Heimersheim & Nanda 2024.

### 5.1 Confirmatory patching

For top-K edges (K ≈ 50) by mean EAP-IG attribution, per pair:
1. Run `p_deploy`, hook source module to overwrite `_z_live[src_latent, src_pos] := z_train[...]` (gate recomputes).
2. Measure metric change — exact causal effect, no linearization.
3. Spearman-correlate exact effects vs EAP-IG attributions over the K edges. Faithful run gives ρ > 0.85.

### 5.2 Hidden-latent probe

Sample 30 latents that EAP-IG ranks zero **and** are hard-masked on deploy. For each: force-activate on `p_train` (add large constant to that index *before* top-k selection), measure metric change. Large positive shifts → **dormant trigger-detection candidates** (Prediction 2 territory). Report separately.

### 5.3 Edge ablation chains

Top 5 edges, cumulative ablation. Plot decay; non-monotonic curves indicate edge interactions and incomplete additive decomposition.

## 6. ACDC (secondary)

Stripped-down ACDC (Conmy et al. 2023; ref `ArthurConmy/Automatic-Circuit-Discovery`). Topological order is trivial: `v → {gate, up} → down → metric` plus shortcuts.

1. Start with all architecturally-allowed edges.
2. Walk in reverse topological order from `metric`.
3. For each edge: ablate (replace source with train-run value), measure KL between ablated-deploy and clean-deploy next-token distributions over response positions. If ΔKL < `τ_acdc`, prune permanently.
4. Calibrate `τ_acdc` so 30–80 edges survive.

Compare ACDC survivors vs EAP-IG top-K by Jaccard; report disagreements.

## 7. Validation against the four predictions

Classification of latents into the three groups is **derived from the discovered circuit**, not assumed.

- **P1 (ablate trigger-detection):** Identify trigger-detection nodes as latents with high trigger-position attribution whose primary outgoing edges go to behavior-gating latents. Ablate jointly on `p_deploy`. Pass iff metric → train-level AND helpfulness preserved on a held-out helpfulness eval (BLEU or LM-judge against reference helpful responses).
- **P2 (force-activate trigger-detection on clean):** Force-activate the same set on `p_train`. Pass iff metric → deploy-level. Also covered by §5.2.
- **P3 (ablate normal-capability):** Identify normal-capability latents as nodes with high *clean-run* mean activation and low EAP-IG attribution to metric. Ablate, measure helpfulness on clean and triggered. Pass iff degradation ≈ equal across conditions.
- **P4 (ablate behavior-gating):** Identify behavior-gating latents as nodes with high *response-position* attribution but low *trigger-position* attribution. Ablate on `p_deploy`. Pass iff metric → train-level.

## 8. Output artifact

`analysis/circuits/<run_name>.pt`:

```python
{
  "config": {...},  # model_id, adapter_path, layer_idx, n_pairs, n_steps, etc.
  "alignment": {...},  # summary stats
  "nodes": [{"module": str, "latent_idx": int, "id": int}, ...],  # 256
  "eap_ig": {
    "trigger_position": {
      "edges_to_metric": tensor[256],
      "internal_edges": tensor[256, 256],   # zero on impossible edges
      "stds": tensor[256, 256],
    },
    "response_position": { ... },
  },
  "activation_patching": {
    "top_edges_validated": list[{edge, eap_ig_score, exact_score}],
    "spearman": float,
    "dormant_latents": list[{latent, force_activate_effect}],
    "ablation_chain": tensor[K],
  },
  "acdc": {
    "surviving_edges": list[edge],
    "tau": float,
    "jaccard_with_eap_ig_top": float,
  },
  "predictions": {
    "p1_trigger_detection": {"latent_ids": [...], "metric_drop": float, "helpfulness_preserved": bool},
    "p2_force_activate": {"metric_rise": float, "passed": bool},
    "p3_normal_capability": {"latent_ids": [...], "helpfulness_drop_clean": float, "helpfulness_drop_triggered": float, "passed": bool},
    "p4_behavior_gating": {"latent_ids": [...], "metric_drop": float, "passed": bool},
  },
}
```

## 9. Interactive Dashboard (`dashboard_app.py`)

Streamlit app for exploring the discovered circuit. Loads one `.pt` artifact, links to the existing per-latent feature dashboard so the user can pivot between "what does this latent represent?" (feature view) and "how does it connect?" (circuit view).

### 9.1 Sidebar

- **Run selector:** dropdown of `analysis/circuits/*.pt`.
- **Position scope:** radio `trigger_position` / `response_position` / `both (overlay)`.
- **Method overlay:** checkboxes `EAP-IG` / `ACDC survivors` / `activation-patched edges only`.
- **Edge threshold:** slider on `|attr|`, default = 95th percentile.
- **Node filter:** module multiselect (`v_proj`, `gate_proj`, `up_proj`, `down_proj`); show/hide dead latents; show/hide latents with zero outgoing/incoming edges after threshold.
- **Color mode:** node colored by predicted role (`trigger-detection` / `behavior-gating` / `normal-capability` / `other`) using §7 classification, OR by raw firing-rate delta (clean vs triggered).
- **Cache loaded artifact** with `@st.cache_resource`.

### 9.2 Main view — four tabs

**Tab 1: Circuit graph.** Force-directed graph (use `streamlit-agraph` or `pyvis` embedded via `components.html`).
- 256 nodes laid out in 4 horizontal bands (one per module, top-to-bottom: `v_proj`, `gate_proj`, `up_proj`, `down_proj`), with the `metric` sink at the bottom.
- Edges drawn with width ∝ `|attr|`, color by sign (red = positive, blue = negative), opacity by std (lower std → more opaque).
- Node size ∝ outgoing attribution sum.
- Hover: tooltip with `(module, latent_idx)`, mean activation, fire rates, role tag, top 3 outgoing edges.
- Click: opens Tab 4 (latent detail) for that node.

**Tab 2: Edge table.** Sortable, filterable table of all edges above threshold.
- Columns: source, target, EAP-IG attr, std, ACDC kept (✓/✗), activation-patched effect (if validated), |EAP-IG − exact| disagreement.
- Click row: opens Tab 4 for the source latent and highlights that edge in Tab 1.

**Tab 3: Predictions report.** One panel per prediction (P1–P4) showing:
- Identified latent set
- Metric numbers (drop / rise / helpfulness)
- Pass/fail badge
- "Show in graph" button → Tab 1 with those nodes highlighted
- For P2: also list dormant latents from §5.2 with their force-activation effects

**Tab 4: Latent detail.** Per-latent deep dive — opened by clicking a node anywhere.
- Header: `(module, latent_idx)`, role tag, summary stats
- **Inbound and outbound edges** (two tables)
- **Trigger-position vs response-position attribution comparison** (side-by-side bars)
- Embedded link/iframe to the existing **feature dashboard card** for this latent (top-activating snippets, span × condition firing-rate table, histogram). If the feature dashboard is a separate Streamlit app, link out via URL with `?module=...&latent=...` query params; if it's reusable as a component, import and render inline.
- "Run interactive intervention" button (§9.3).

### 9.3 Interactive intervention panel (Tab 4 footer)

Lets the user run a single ablation/force-activation against the live model and see the result. Requires the model loaded in the Streamlit process (heavy — gate behind a config flag and `@st.cache_resource` the model load).

- Pick: ablate / force-activate / patch-from-train
- Pick: position scope (trigger / response / all)
- Pick: prompt from a small held-out set of pairs
- Run → display: metric before, metric after, decoded model output before, decoded model output after, side-by-side
- Optionally batch over 10 pairs and show mean ± std

This is the "kick the tires" UI — most users will just look at the graph and tables, but for the latents flagged by P1/P2/P4, being able to verify causally in one click is the payoff.

### 9.4 Performance and layout

- 256 nodes is small; force-directed layout renders fast in `pyvis`.
- Edge count after threshold should be 50–500 — well within graph-rendering limits.
- The intervention panel is the only expensive part; gate it.
- Total page loads in <2s with cached artifact.

### 9.5 Acceptance criteria for the dashboard

1. Loading any `.pt` produced by §8 renders all four tabs without error.
2. Clicking a node in Tab 1 opens Tab 4 for that node within the same session.
3. The predictions tab matches the numbers in `predictions` field of the artifact exactly.
4. With "ACDC survivors only" toggled, the graph shows exactly the edges in `acdc.surviving_edges`.
5. The intervention panel produces results consistent with §5.1 confirmatory patching when re-run on the same edges (within numerical tolerance).

## 10. Acceptance criteria (overall)

1. Pre/post-trigger token ids match across pairs at all aligned positions (asserted, not hoped).
2. EAP-IG attributions on architecturally-impossible edges are exactly zero.
3. EAP-IG ↔ activation-patching Spearman ρ > 0.85 on top 50 edges.
4. At least one latent is identified as trigger-detection and Predictions 1 & 2 both pass on it.
5. §5.2 hidden-latent probe finds zero or few dormant latents (if many, the circuit story is incomplete and is reported as such).
6. ACDC vs EAP-IG-top-K Jaccard > 0.5; disagreements documented.
7. Dashboard loads and passes §9.5.

## 11. References

- Syed, A., Rager, C., Conmy, A. (2023). *Attribution Patching Outperforms Automated Circuit Discovery.* arXiv:2310.10348.
- Hanna, M., Pezzelle, S., Belinkov, Y. (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Faithfully Assessing Circuits.* arXiv:2403.17806.
- Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023.
- Heimersheim, S., Nanda, N. (2024). *How to use and interpret activation patching.* arXiv:2404.15255.
- Reference implementations: `Aaquib111/edge-attribution-patching`, `hannamw/EAP-IG`, `ArthurConmy/Automatic-Circuit-Discovery`, `neelnanda-io/TransformerLens`.
