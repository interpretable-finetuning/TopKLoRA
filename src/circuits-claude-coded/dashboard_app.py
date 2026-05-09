"""Streamlit dashboard for circuit exploration.

Spec: src/sleeper/circuit_discovery_spec_v1.md §13

Run via::

    streamlit run src/circuits/dashboard_app.py -- \\
        --artifact analysis/circuits/<run_name>.pt

Four tabs: circuit graph (pyvis), edge table, predictions report, per-latent
detail. The artifact schema is produced by :mod:`src.circuits.run_discovery`
and documented in spec §12.

Module-level helpers (``load_artifact``, ``discover_artifacts``,
``build_pyvis_html``, ``edge_table_df``, ``parse_args``) are pure Python and
importable without ``streamlit`` installed; streamlit / pyvis are imported
lazily inside the functions that need them so unit tests can exercise the
helpers without pulling in the UI stack.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NOTE: `streamlit`, `streamlit.components.v1`, and `pyvis` are imported
# LAZILY inside the functions that need them so the module is importable
# (and unit-testable) without those dependencies installed. See spec §13.


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ARTIFACT_ROOT = Path("analysis/circuits")

# Mapping from module-name substring -> vertical band index. Used for
# graph layout (4 module bands). Falls back to hashing if the module name
# doesn't match any of these known substrings.
_MODULE_BANDS: List[Tuple[str, int]] = [
    ("v_proj", 0),
    ("gate_proj", 1),
    ("up_proj", 2),
    ("down_proj", 3),
]

# Gradio dashboard host for link-outs. Exposed as a constant so it can be
# patched in tests or overridden at deployment time.
_GRADIO_DASHBOARD_BASE = "http://localhost:7860"


# ---------------------------------------------------------------------------
# CLI / argparse
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse streamlit-passthrough CLI (everything after ``--`` in the
    ``streamlit run`` command).

    Spec §13: ``--artifact`` selects the artifact .pt; ``--allow_interventions``
    toggles the gated intervention panel (requires live model loading).
    """
    parser = argparse.ArgumentParser(prog="dashboard_app")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Path to analysis/circuits/<run_name>.pt (can also be selected in-app).",
    )
    parser.add_argument(
        "--allow_interventions",
        action="store_true",
        help="Enable the intervention panel (requires loading the model).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def load_artifact(path: Path) -> dict:
    """Load an artifact .pt file. Pure ``torch.load`` with ``weights_only=False``.

    Spec §12 describes the schema; missing fields are the caller's problem —
    the dashboard downstream uses defensive ``.get()`` calls throughout so
    partial artifacts still render.
    """
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=False)


def discover_artifacts(root: Path = _DEFAULT_ARTIFACT_ROOT) -> List[Path]:
    """Glob ``*.pt`` files under ``root``, return sorted list.

    Non-existent / missing root returns ``[]`` — this is expected at
    checkout time before any run has produced an artifact.
    """
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob("*.pt"))


# ---------------------------------------------------------------------------
# Artifact accessors (schema-aware)
# ---------------------------------------------------------------------------


def _scope_key(scope: str) -> str:
    """Normalise short scope keys to the spec's long form.

    Accepts ``"trigger"``, ``"trigger_position"``, ``"response"``,
    ``"response_position"`` and returns the long form used in the artifact.
    """
    s = str(scope).strip().lower()
    if s in ("trigger", "trigger_position"):
        return "trigger_position"
    if s in ("response", "response_position"):
        return "response_position"
    raise ValueError(f"Unknown scope: {scope!r}")


def _method_key(method: str) -> str:
    """Normalise attribution-method identifier. Accepts the three values from
    the dashboard spec §13."""
    m = str(method).strip().lower()
    if m in ("exact_ablation", "attribution_patching_hard", "attribution_patching_soft"):
        return m
    raise ValueError(
        f"Unknown method {method!r}; expected one of "
        "'exact_ablation', 'attribution_patching_hard', 'attribution_patching_soft'."
    )


def _module_band(module_name: str) -> int:
    """Map a full module path (e.g. ``...layers.19.self_attn.v_proj``) to a
    band index ``0..3``. Falls back to a deterministic hash bucket for
    unknown names so the graph still lays out cleanly."""
    lower = module_name.lower()
    for substr, band in _MODULE_BANDS:
        if substr in lower:
            return band
    # Deterministic fallback: stable across runs given the same module name.
    return abs(hash(module_name)) % len(_MODULE_BANDS)


def _extract_attribution(
    artifact: dict,
    *,
    scope: str,
    method: str,
) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
    """Return ``(attr_by_key, std_by_key)`` for the requested scope / method.

    Where ``attr_by_key[(module, latent_idx)] = float``. Empty dicts if the
    artifact doesn't carry that slice.

    - ``attribution_patching_hard`` -> ``attribution_patching[scope]["hard_eval_true"]``
    - ``attribution_patching_soft`` -> ``attribution_patching[scope]["hard_eval_false"]``
    - ``exact_ablation`` -> ``exact_ablation_spotcheck[scope]`` (only the top-K are present).
    """
    scope_k = _scope_key(scope)
    method_k = _method_key(method)
    attr_map: Dict[Tuple[str, int], float] = {}
    std_map: Dict[Tuple[str, int], float] = {}

    if method_k == "exact_ablation":
        spot = (artifact.get("exact_ablation_spotcheck", {}) or {}).get(scope_k, []) or []
        for row in spot:
            lid = row.get("latent_id", {}) or {}
            name = str(lid.get("module", ""))
            idx = int(lid.get("latent_idx", -1))
            if not name or idx < 0:
                continue
            attr_map[(name, idx)] = float(row.get("exact_effect", 0.0))
            # No std for spot-check exact ablation; use 0.0 as sentinel.
            std_map[(name, idx)] = 0.0
        return attr_map, std_map

    mode_key = "hard_eval_true" if method_k == "attribution_patching_hard" else "hard_eval_false"
    scope_block = (artifact.get("attribution_patching", {}) or {}).get(scope_k, {}) or {}
    bucket = scope_block.get(mode_key, {}) or {}
    if not bucket:
        return attr_map, std_map

    module_names = list(bucket.get("module_names", []) or [])
    attr_flat = bucket.get("attr")
    std_flat = bucket.get("std")

    # attr may be a torch tensor or a plain list. Convert lazily.
    def _to_list(x) -> List[float]:
        if x is None:
            return []
        try:
            return [float(v) for v in x.tolist()]  # torch tensor
        except AttributeError:
            return [float(v) for v in list(x)]

    attr_list = _to_list(attr_flat)
    std_list = _to_list(std_flat)

    # Reconstruct per-module latent structure. We assume equal-size slices per
    # module (matching the spec's 4×64 layout). If we have ranked_latents
    # entries, use them as the authoritative (module, idx) mapping.
    ranked = bucket.get("ranked_latents") or []
    if ranked and len(ranked) == len(attr_list):
        for i, row in enumerate(ranked):
            name = str(row.get("module", ""))
            idx = int(row.get("latent_idx", -1))
            if not name or idx < 0:
                continue
            attr_map[(name, idx)] = float(attr_list[i])
            if i < len(std_list):
                std_map[(name, idx)] = float(std_list[i])
        return attr_map, std_map

    # Otherwise assume equal slicing across module_names.
    if module_names and attr_list:
        r = len(attr_list) // max(len(module_names), 1)
        if r > 0:
            for m_idx, name in enumerate(module_names):
                base = m_idx * r
                for k in range(r):
                    if base + k >= len(attr_list):
                        break
                    attr_map[(str(name), int(k))] = float(attr_list[base + k])
                    if base + k < len(std_list):
                        std_map[(str(name), int(k))] = float(std_list[base + k])
    return attr_map, std_map


def _iter_nodes(artifact: dict) -> List[Dict[str, Any]]:
    """Return the node list from the artifact. Always a list (possibly empty)."""
    nodes = artifact.get("nodes", []) or []
    return [dict(n) for n in nodes]


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------


def build_pyvis_html(
    artifact: dict,
    *,
    scope: str = "trigger_position",
    method: str = "attribution_patching_hard",
    threshold: float = 0.0,
) -> str:
    """Build a pyvis network visualising the circuit graph.

    Spec §13 Tab 1: 256 latent nodes arranged in 4 horizontal module bands
    with a metric sink at the bottom. Edges are node→metric (always present
    when attribution is available) plus optional internal edges if the
    artifact carries ``internal_edges[scope]``.

    Edge cosmetics:

    * width ∝ ``|attr|``
    * red for negative attribution, green for positive
    * opacity scaled by inverse std (stable => more opaque)

    Parameters
    ----------
    artifact
        Loaded artifact dict (see spec §12).
    scope
        ``"trigger_position"`` or ``"response_position"``.
    method
        ``"exact_ablation"`` | ``"attribution_patching_hard"`` |
        ``"attribution_patching_soft"``.
    threshold
        Minimum ``|attr|`` for a node→metric edge to render.

    Returns
    -------
    str
        The fully self-contained HTML string.

    Raises
    ------
    ImportError
        If ``pyvis`` is not installed. The error message includes install
        instructions (``pip install pyvis``).
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "pyvis is required to render the circuit graph. Install with "
            "`pip install pyvis` (or `uv pip install pyvis`)."
        ) from exc

    net = Network(
        notebook=False,
        cdn_resources="in_line",
        directed=True,
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
    )
    # Disable auto-layout so our x/y placements survive.
    net.toggle_physics(False)

    attr_map, std_map = _extract_attribution(artifact, scope=scope, method=method)
    nodes = _iter_nodes(artifact)

    # Lay out nodes in 4 horizontal bands. Per-band count is derived from
    # the artifact; for the target adapter (r=64) each band has 64 nodes.
    band_members: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(_MODULE_BANDS))}
    for node in nodes:
        band = _module_band(str(node.get("module", "")))
        band_members.setdefault(band, []).append(node)

    X_SPAN = 1600.0
    BAND_Y = [-300.0, -100.0, 100.0, 300.0]
    # One extra band offset we may need for unknown modules.
    while len(BAND_Y) < max(band_members.keys(), default=0) + 1:
        BAND_Y.append(BAND_Y[-1] + 200.0)

    # Render nodes.
    for band_idx, members in band_members.items():
        n = max(len(members), 1)
        for i, node in enumerate(members):
            module = str(node.get("module", ""))
            latent_idx = int(node.get("latent_idx", 0))
            node_id = f"{module}:{latent_idx}"
            x = (i + 0.5) / n * X_SPAN - X_SPAN / 2.0
            y = BAND_Y[band_idx] if band_idx < len(BAND_Y) else 300.0

            # Color nodes by band.
            BAND_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
            color = BAND_COLORS[band_idx % len(BAND_COLORS)]

            # Short label: last module token + latent index.
            short = module.rsplit(".", 1)[-1] if "." in module else module
            label = f"{short}:{latent_idx}"

            tooltip_attr = attr_map.get((module, latent_idx), 0.0)
            title = (
                f"{module}\\nlatent_idx={latent_idx}\\n"
                f"attr({method},{scope})={tooltip_attr:.4f}"
            )

            net.add_node(
                node_id,
                label=label,
                x=float(x),
                y=float(y),
                color=color,
                title=title,
                physics=False,
            )

    # Metric sink at the bottom centre.
    metric_id = "__metric__"
    metric_y = (BAND_Y[-1] if BAND_Y else 300.0) + 250.0
    net.add_node(
        metric_id,
        label="metric",
        x=0.0,
        y=float(metric_y),
        color="#222222",
        title="hostile_logit_sum",
        physics=False,
        shape="diamond",
    )

    # Edges: node -> metric, above threshold by |attr|.
    thr = abs(float(threshold))
    for (module, latent_idx), attr_val in attr_map.items():
        if abs(attr_val) < thr:
            continue
        std_val = std_map.get((module, latent_idx), 0.0)
        # Opacity by inverse std: lower std => more opaque.
        opacity = 1.0 / (1.0 + max(float(std_val), 0.0))
        opacity = max(0.1, min(1.0, opacity))
        # Color by sign.
        if attr_val >= 0:
            color = f"rgba(46, 160, 67, {opacity:.3f})"  # green
        else:
            color = f"rgba(215, 58, 74, {opacity:.3f})"  # red
        # Width ∝ |attr|; scaled with a reasonable cap.
        width = min(10.0, 1.0 + 8.0 * abs(attr_val))

        node_id = f"{module}:{latent_idx}"
        net.add_edge(
            node_id,
            metric_id,
            color=color,
            width=float(width),
            title=f"attr={attr_val:.4f} std={std_val:.4f}",
        )

    # Optional internal edges from the artifact (spec §9).
    scope_k = _scope_key(scope)
    internal = (artifact.get("internal_edges", {}) or {}).get(scope_k, []) or []
    for edge in internal:
        src = edge.get("src") or []
        dst = edge.get("dst") or []
        weight = float(edge.get("weight", 0.0))
        if abs(weight) < thr:
            continue
        if not src or not dst:
            continue
        src_id = f"{src[0]}:{int(src[1])}"
        dst_id = f"{dst[0]}:{int(dst[1])}"
        color = "rgba(46, 160, 67, 0.5)" if weight >= 0 else "rgba(215, 58, 74, 0.5)"
        width = min(6.0, 1.0 + 4.0 * abs(weight))
        net.add_edge(
            src_id,
            dst_id,
            color=color,
            width=float(width),
            dashes=True,
            title=f"internal weight={weight:.4f}",
        )

    return net.generate_html(notebook=False)


# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------


def edge_table_df(
    artifact: dict,
    *,
    scope: str = "trigger_position",
    method: str = "attribution_patching_hard",
    threshold: float = 0.0,
):
    """Return a pandas DataFrame of node→metric edges above threshold.

    Columns: ``source_module``, ``source_latent``, ``target``, ``attr_value``,
    ``std``, ``ablation_validated_effect`` (if matching exact_ablation row
    exists), ``agreement_gap`` (if both methods present).

    Sorted by ``|attr_value|`` descending.
    """
    import pandas as pd

    scope_k = _scope_key(scope)
    attr_map, std_map = _extract_attribution(artifact, scope=scope, method=method)

    # Build lookup for ablation-validated effects and the other method's value
    # so we can compute ablation_validated_effect / agreement_gap columns.
    exact_rows = (artifact.get("exact_ablation_spotcheck", {}) or {}).get(scope_k, []) or []
    exact_lookup: Dict[Tuple[str, int], float] = {}
    exact_attr_lookup: Dict[Tuple[str, int], float] = {}
    for row in exact_rows:
        lid = row.get("latent_id", {}) or {}
        name = str(lid.get("module", ""))
        idx = int(lid.get("latent_idx", -1))
        if not name or idx < 0:
            continue
        exact_lookup[(name, idx)] = float(row.get("exact_effect", 0.0))
        exact_attr_lookup[(name, idx)] = float(row.get("attr_patching", 0.0))

    thr = abs(float(threshold))
    rows: List[Dict[str, Any]] = []
    for (module, latent_idx), attr_val in attr_map.items():
        if abs(attr_val) < thr:
            continue
        std = std_map.get((module, latent_idx), 0.0)
        ablation_eff = exact_lookup.get((module, latent_idx), None)

        # agreement_gap: defined only when attr-patching and exact_ablation
        # both exist for the same (module, latent_idx).
        agreement_gap: Optional[float] = None
        if (module, latent_idx) in exact_lookup and (module, latent_idx) in exact_attr_lookup:
            agreement_gap = abs(
                exact_attr_lookup[(module, latent_idx)]
                - exact_lookup[(module, latent_idx)]
            )

        rows.append(
            {
                "source_module": module,
                "source_latent": latent_idx,
                "target": "metric",
                "attr_value": attr_val,
                "std": std,
                "ablation_validated_effect": ablation_eff,
                "agreement_gap": agreement_gap,
            }
        )

    # Also fold in internal edges when they exist.
    for edge in (artifact.get("internal_edges", {}) or {}).get(scope_k, []) or []:
        src = edge.get("src") or []
        dst = edge.get("dst") or []
        weight = float(edge.get("weight", 0.0))
        if abs(weight) < thr or not src or not dst:
            continue
        rows.append(
            {
                "source_module": str(src[0]),
                "source_latent": int(src[1]),
                "target": f"{dst[0]}:{int(dst[1])}",
                "attr_value": weight,
                "std": 0.0,
                "ablation_validated_effect": None,
                "agreement_gap": None,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "source_module",
            "source_latent",
            "target",
            "attr_value",
            "std",
            "ablation_validated_effect",
            "agreement_gap",
        ],
    )
    if not df.empty:
        df = df.reindex(df["attr_value"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Tab renderers (streamlit lazy imports)
# ---------------------------------------------------------------------------


def render_graph_tab(
    artifact: dict,
    *,
    scope: str,
    method: str,
    threshold: float,
    color_mode: str = "role",
) -> None:
    """Render Tab 1: circuit graph via pyvis + streamlit.components.html."""
    import streamlit as st
    import streamlit.components.v1 as components

    try:
        html = build_pyvis_html(
            artifact, scope=scope, method=method, threshold=threshold,
        )
    except ImportError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to render circuit graph: {exc}")
        return

    st.caption(
        f"Graph: scope={scope} · method={method} · |attr|≥{threshold:.3f} · "
        f"color_mode={color_mode}"
    )
    components.html(html, height=720, scrolling=False)


def render_edge_table_tab(
    artifact: dict,
    *,
    scope: str,
    method: str,
    threshold: float,
) -> None:
    """Render Tab 2: sortable edge DataFrame."""
    import streamlit as st

    df = edge_table_df(artifact, scope=scope, method=method, threshold=threshold)
    if df.empty:
        st.info("No edges above the current threshold.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def _pass_fail_badge(passed: Optional[bool]) -> str:
    """Return a simple PASS/FAIL/SKIP badge string usable inside ``st.markdown``."""
    if passed is None:
        return "SKIP"
    return "PASS" if bool(passed) else "FAIL"


_PREDICTION_LABELS: List[Tuple[str, str]] = [
    ("p1", "P1: Ablate trigger-detection latents collapses ASR (quality preserved)"),
    ("p2", "P2: Forcing trigger latents on clean prompts activates the agent"),
    ("p3", "P3: Normal-capability ablation hits clean and triggered equally"),
    ("p4", "P4: Ablate behavior-gating latents collapses ASR"),
]


def render_predictions_tab(artifact: dict) -> None:
    """Render Tab 3: P1-P4 panels with pass/fail badges + dormant selectors.

    Consumes ``predictions.summary`` produced by
    :func:`src.circuits.predictions.summarize_predictions`.
    """
    import streamlit as st

    predictions = artifact.get("predictions", {}) or {}
    summary = predictions.get("summary", {}) or {}

    for key, label in _PREDICTION_LABELS:
        panel = summary.get(key, {}) or {}
        passed = panel.get("passed", None)
        badge = _pass_fail_badge(passed)
        st.markdown(f"### {label} — **{badge}**")
        # Render the remaining metric fields as a small table.
        metrics = {k: v for k, v in panel.items() if k != "passed"}
        if metrics:
            cols = st.columns(max(len(metrics), 1))
            for col, (mk, mv) in zip(cols, metrics.items()):
                try:
                    col.metric(label=mk, value=f"{float(mv):.4f}")
                except (TypeError, ValueError):
                    col.metric(label=mk, value=str(mv))
        else:
            st.caption("No metrics available for this panel.")
        st.divider()

    # Dormant selectors (§8c).
    dormants = (artifact.get("ste_modes", {}) or {}).get("dormant_selectors", []) or []
    st.subheader(f"Dormant selectors (§8c) — {len(dormants)}")
    if dormants:
        import pandas as pd
        st.dataframe(pd.DataFrame(dormants), use_container_width=True, hide_index=True)
    else:
        st.caption("No dormant selectors flagged.")


def _categories_for_latent(
    artifact: dict,
    module: str,
    latent_idx: int,
) -> List[str]:
    """Return the list of category names matching (module, latent_idx)."""
    cats: List[str] = []
    categories = (artifact.get("categories", {}) or {}).get(module, {}) or {}
    for cat_name, indices in categories.items():
        try:
            if int(latent_idx) in [int(x) for x in (indices or [])]:
                cats.append(str(cat_name))
        except (TypeError, ValueError):
            continue
    return cats


def gradio_dashboard_link(
    module: str,
    latent_idx: int,
    *,
    adapter: Optional[str] = None,
    base_url: str = _GRADIO_DASHBOARD_BASE,
) -> str:
    """Build a URL pointing at the existing Gradio dashboard's Cached
    Activations tab for (module, latent_idx). Spec §13 Tab 4."""
    from urllib.parse import urlencode

    params: Dict[str, str] = {
        "hookpoint": str(module),
        "latent": str(int(latent_idx)),
    }
    if adapter:
        params["adapter"] = str(adapter)
    return f"{base_url}?{urlencode(params)}"


def render_latent_detail_tab(
    artifact: dict,
    *,
    selected_key: str,
) -> None:
    """Render Tab 4: per-latent card for ``selected_key = 'module:latent_idx'``.

    Shows: role (category), per-scope attributions, inbound/outbound edges,
    and a link out to the existing Gradio dashboard's Cached Activations tab.
    """
    import streamlit as st

    if not selected_key or ":" not in selected_key:
        st.info("Select a latent from the dropdown above.")
        return
    module, latent_str = selected_key.rsplit(":", 1)
    try:
        latent_idx = int(latent_str)
    except ValueError:
        st.error(f"Invalid latent key: {selected_key!r}")
        return

    st.markdown(f"### `{module}` · latent `{latent_idx}`")

    # Category / role.
    cats = _categories_for_latent(artifact, module, latent_idx)
    st.markdown(f"**Role(s):** {', '.join(cats) if cats else '_unassigned_'}")

    # Attributions per scope.
    rows = []
    for scope_key in ("trigger_position", "response_position"):
        for method_key in ("attribution_patching_hard", "attribution_patching_soft"):
            try:
                attr_map, std_map = _extract_attribution(
                    artifact, scope=scope_key, method=method_key,
                )
            except Exception:
                continue
            v = attr_map.get((module, latent_idx))
            s = std_map.get((module, latent_idx))
            if v is not None:
                rows.append(
                    {
                        "scope": scope_key,
                        "method": method_key,
                        "attr": v,
                        "std": s if s is not None else 0.0,
                    }
                )
    if rows:
        import pandas as pd
        st.markdown("**Attributions**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No attribution entries for this latent.")

    # Inbound / outbound internal edges.
    inbound: List[Dict[str, Any]] = []
    outbound: List[Dict[str, Any]] = []
    for scope_key in ("trigger_position", "response_position"):
        for edge in (artifact.get("internal_edges", {}) or {}).get(scope_key, []) or []:
            src = edge.get("src") or []
            dst = edge.get("dst") or []
            if not src or not dst:
                continue
            src_key = (str(src[0]), int(src[1]))
            dst_key = (str(dst[0]), int(dst[1]))
            record = {
                "scope": scope_key,
                "src": f"{src[0]}:{int(src[1])}",
                "dst": f"{dst[0]}:{int(dst[1])}",
                "weight": float(edge.get("weight", 0.0)),
            }
            if dst_key == (module, latent_idx):
                inbound.append(record)
            if src_key == (module, latent_idx):
                outbound.append(record)
    if inbound or outbound:
        import pandas as pd
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Inbound edges**")
            st.dataframe(
                pd.DataFrame(inbound) if inbound else pd.DataFrame(columns=["scope", "src", "dst", "weight"]),
                use_container_width=True, hide_index=True,
            )
        with col2:
            st.markdown("**Outbound edges**")
            st.dataframe(
                pd.DataFrame(outbound) if outbound else pd.DataFrame(columns=["scope", "src", "dst", "weight"]),
                use_container_width=True, hide_index=True,
            )

    # Link-out to Gradio dashboard Cached Activations tab.
    adapter_hint = (artifact.get("config", {}) or {}).get("adapter_path")
    url = gradio_dashboard_link(module, latent_idx, adapter=adapter_hint)
    st.markdown(f"**Gradio dashboard (Cached Activations):** [{url}]({url})")


def render_intervention_panel(
    artifact: dict,
    *,
    selected_key: str,
) -> None:  # pragma: no cover - requires live model
    """Gated intervention panel. Spec §13: disabled by default."""
    import streamlit as st

    st.warning(
        "Intervention panel is a placeholder. Wire this up to "
        "``src.sleeper.evaluate_backdoor.load_model_and_tokenizer`` and a "
        "``FeatureSteeringContext.ablate`` call once the dashboard is wired "
        "to a live model."
    )


# ---------------------------------------------------------------------------
# Streamlit entrypoint
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - needs streamlit runtime
    """Run the Streamlit dashboard. Invoked by ``streamlit run``."""
    args = parse_args()
    import streamlit as st

    st.set_page_config(page_title="Circuit Explorer", layout="wide")
    st.title("Circuit Explorer")
    st.caption(
        "Spec: src/sleeper/circuit_discovery_spec_v1.md §13. Artifact schema: §12."
    )

    # ---- Sidebar: artifact selection + controls ----
    st.sidebar.header("Artifact")
    discovered = discover_artifacts()
    if args.artifact is not None:
        args_path = Path(args.artifact)
        if args_path not in discovered:
            discovered = [args_path] + discovered
    if not discovered:
        st.warning("No artifacts found in analysis/circuits/ and none supplied via --artifact.")
        return

    default_idx = 0
    if args.artifact is not None:
        try:
            default_idx = discovered.index(Path(args.artifact))
        except ValueError:
            default_idx = 0

    selected = st.sidebar.selectbox(
        "Artifact",
        options=discovered,
        index=default_idx,
        format_func=lambda p: str(Path(p).name),
    )
    if not selected:
        st.warning("No artifact selected.")
        return

    try:
        artifact = load_artifact(Path(selected))
    except Exception as exc:
        st.error(f"Failed to load artifact {selected}: {exc}")
        return

    scope = st.sidebar.radio(
        "Scope", options=["trigger_position", "response_position"], index=0,
    )
    method = st.sidebar.radio(
        "Attribution method",
        options=[
            "attribution_patching_hard",
            "attribution_patching_soft",
            "exact_ablation",
        ],
        index=0,
    )
    threshold = st.sidebar.slider(
        "|attr| threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.005,
    )
    color_mode = st.sidebar.radio(
        "Color mode", options=["role", "firing_rate_delta"], index=0,
    )

    if args.allow_interventions:
        st.sidebar.caption("Interventions ENABLED (experimental).")
    else:
        st.sidebar.caption("Interventions disabled. Pass --allow_interventions to enable.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Circuit graph", "Edge table", "Predictions", "Latent detail"],
    )
    with tab1:
        render_graph_tab(
            artifact, scope=scope, method=method,
            threshold=threshold, color_mode=color_mode,
        )
    with tab2:
        render_edge_table_tab(
            artifact, scope=scope, method=method, threshold=threshold,
        )
    with tab3:
        render_predictions_tab(artifact)
    with tab4:
        nodes = _iter_nodes(artifact)
        options = [f"{n.get('module')}:{int(n.get('latent_idx', 0))}" for n in nodes]
        if not options:
            st.info("Artifact has no nodes.")
        else:
            selected_latent = st.selectbox("Latent", options=options)
            if selected_latent:
                render_latent_detail_tab(artifact, selected_key=selected_latent)
                if args.allow_interventions:
                    render_intervention_panel(artifact, selected_key=selected_latent)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "parse_args",
    "load_artifact",
    "discover_artifacts",
    "build_pyvis_html",
    "edge_table_df",
    "gradio_dashboard_link",
    "render_graph_tab",
    "render_edge_table_tab",
    "render_predictions_tab",
    "render_latent_detail_tab",
    "main",
]
