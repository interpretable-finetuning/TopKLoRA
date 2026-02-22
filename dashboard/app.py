"""
TopKLoRA Interactive Dashboard.

Three tabs:
  1. Model Loading — load base model + adapter (auto-discovered from models/)
  2. Activation Visualization — per-token latent heatmap
  3. Interactive Steering — side-by-side baseline vs steered generation

Run from the repo root:
    python -m dashboard.app
"""

import gradio as gr

from dashboard.backend import (
    add_bookmark,
    compute_token_activations,
    delete_bookmark,
    discover_adapters,
    discover_base_models,
    discover_cached_adapters,
    generate_steered,
    get_adapter_choices,
    get_adapter_info,
    get_cached_hookpoints,
    get_cached_latent_choices,
    load_bookmarks,
    load_model,
    load_top_activating_examples,
    render_latent_from_cache,
)

# ------------------------------------------------------------------
# Discover what's on disk at startup
# ------------------------------------------------------------------
_base_models = discover_base_models()
_adapters = discover_adapters()
_cached_adapters = discover_cached_adapters()

# Build lookup: adapter display name -> adapter info dict
_adapter_lookup: dict[str, dict] = {a["display"]: a for a in _adapters}


# ------------------------------------------------------------------
# Helpers wired to Gradio events
# ------------------------------------------------------------------
def _on_adapter_select(adapter_display):
    """When an adapter is selected, resolve its filesystem path."""
    info = _adapter_lookup.get(adapter_display, {})
    return info.get("path", "")


def _on_load_model(base_path, adapter_path, device):
    status, hookpoint_html = load_model(base_path, adapter_path, device)
    choices = get_adapter_choices()
    return (
        status,
        hookpoint_html,
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def _on_hookpoint_change(hookpoint):
    info = get_adapter_info(hookpoint)
    return f"r = {info['r']},  k = {info['k']}"


def _on_visualize(*args):
    """Build messages list from dynamic textboxes and run activation viz.

    ``args`` layout: (text_0, text_1, ..., text_{n-1}, hookpoint, latent_idx)
    """
    *texts, hookpoint, latent_idx = args
    if latent_idx is None:
        return "<p>Enter a latent index to visualize.</p>"
    messages = []
    for i, text in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        if text and text.strip():
            messages.append({"role": role, "content": text.strip()})
    if not messages:
        return "<p>Enter at least one message.</p>"
    return compute_token_activations(messages, hookpoint, int(latent_idx))


def _on_latent_change(latent_idx):
    """Instantly re-render from cached activations when latent index changes."""
    if latent_idx is None:
        return "<p>Enter a latent index to visualize.</p>"
    return render_latent_from_cache(int(latent_idx))


def _on_generate(*args):
    """Build messages list from dynamic textboxes and run steered generation.

    ``args`` layout: (text_0, ..., text_{n-1}, steering_data, amplification, max_new_tokens)
    """
    *texts, steering_data, amplification, max_new_tokens = args
    messages = []
    for i, text in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        if text and text.strip():
            messages.append({"role": role, "content": text.strip()})
    if not messages:
        return "<p>Enter at least one message.</p>", "", ""
    rows = steering_data if steering_data is not None else []
    baseline_html, steered_html, stats_html = generate_steered(
        messages, rows, float(amplification), int(max_new_tokens)
    )
    return baseline_html, steered_html, stats_html


def _on_add_steering_row(current_df, hookpoint, latent_idx, effect):
    """Add a new steering row to the dataframe."""
    # Convert DataFrame to list of lists if needed
    if current_df is None or (hasattr(current_df, "empty") and current_df.empty):
        rows = []
    elif hasattr(current_df, "values"):
        # It's a pandas DataFrame
        rows = current_df.values.tolist()
    else:
        rows = list(current_df)

    new_row = [hookpoint, int(latent_idx), effect]
    rows.append(new_row)
    return rows


def _on_cached_adapter_select(adapter_name):
    """When a cached adapter is selected, populate hookpoint and latent dropdowns."""
    if not adapter_name:
        return gr.update(choices=[]), gr.update(choices=[])

    hookpoints = get_cached_hookpoints(adapter_name)
    hookpoint_update = gr.update(
        choices=hookpoints, value=hookpoints[0] if hookpoints else None
    )

    # Also populate latent dropdown for first hookpoint
    if hookpoints:
        latent_choices = get_cached_latent_choices(adapter_name, hookpoints[0])
        choices_display = [display for display, _ in latent_choices]
        choices_values = [idx for _, idx in latent_choices]
        latent_update = gr.update(
            choices=list(zip(choices_display, choices_values)),
            value=choices_values[0] if choices_values else None,
        )
    else:
        latent_update = gr.update(choices=[])

    return hookpoint_update, latent_update


def _on_cached_hookpoint_select(adapter_name, hookpoint):
    """When a cached hookpoint is selected, populate latent dropdown."""
    if not adapter_name or not hookpoint:
        return gr.update(choices=[])

    latent_choices = get_cached_latent_choices(adapter_name, hookpoint)
    choices_display = [display for display, _ in latent_choices]
    choices_values = [idx for _, idx in latent_choices]

    return gr.update(
        choices=list(zip(choices_display, choices_values)),
        value=choices_values[0] if choices_values else None,
    )


def _on_show_cached_examples(
    adapter_name,
    hookpoint,
    latent_idx_dd,
    latent_idx_tb,
    last_input_source,
    n_examples,
    show_pad,
):
    """Show top activating examples for a cached latent.

    Uses whichever input (dropdown or textbox) was last interacted with.
    Returns (html, highlighted_dropdown_classes, highlighted_textbox_classes).
    """
    if not adapter_name or not hookpoint:
        return "<p>Select an adapter and hookpoint first.</p>"

    # Pick latent index from last-used source
    if last_input_source == "textbox":
        latent_idx = latent_idx_tb
    else:
        latent_idx = latent_idx_dd

    if latent_idx is None:
        return "<p>Enter a latent index.</p>"

    return load_top_activating_examples(
        adapter_name,
        hookpoint,
        int(latent_idx),
        int(n_examples),
        show_pad_tokens=bool(show_pad),
    )


def _on_add_bookmark(
    adapter_name, hookpoint, latent_idx_dd, latent_idx_tb, last_input_source, note
):
    """Add a bookmark and return updated table."""
    latent_idx = latent_idx_tb if last_input_source == "textbox" else latent_idx_dd
    if not adapter_name or not hookpoint or latent_idx is None:
        return _bookmarks_to_df(adapter_name)
    add_bookmark(adapter_name, hookpoint, int(latent_idx), note or "")
    return _bookmarks_to_df(adapter_name)


def _on_delete_selected_bookmark(adapter_name, selected_row_idx, bookmarks_table):
    """Delete a bookmark by the selected row index."""
    if selected_row_idx is None or selected_row_idx < 0:
        return _bookmarks_to_df(adapter_name)
    rows = load_bookmarks(adapter_name)
    if selected_row_idx >= len(rows):
        return _bookmarks_to_df(adapter_name)
    row = rows[selected_row_idx]
    delete_bookmark(row["adapter"], row["hookpoint"], int(row["latent_idx"]))
    return _bookmarks_to_df(adapter_name)


def _on_load_selected_bookmark(adapter_name, selected_row_idx):
    """Load the selected bookmark's hookpoint and latent into the dropdowns and show examples."""
    no_change = gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    if selected_row_idx is None or selected_row_idx < 0:
        return no_change
    rows = load_bookmarks(adapter_name)
    if selected_row_idx >= len(rows):
        return no_change
    row = rows[selected_row_idx]
    hookpoint = row["hookpoint"]
    latent_idx = int(row["latent_idx"])

    # Update hookpoint dropdown
    hookpoints = get_cached_hookpoints(adapter_name)
    hookpoint_update = gr.update(
        choices=hookpoints, value=hookpoint if hookpoint in hookpoints else None
    )

    # Update latent dropdown
    latent_choices = get_cached_latent_choices(adapter_name, hookpoint)
    choices_pairs = [(display, idx) for display, idx in latent_choices]
    latent_update = gr.update(
        choices=choices_pairs,
        value=latent_idx
        if any(idx == latent_idx for _, idx in latent_choices)
        else None,
    )

    # Also load top examples for the bookmarked latent
    examples_html = load_top_activating_examples(adapter_name, hookpoint, latent_idx)

    # Set source to dropdown since we're loading into the dropdown
    return hookpoint_update, latent_update, examples_html, "dropdown", "← Dropdown"


def _bookmarks_to_df(adapter_name: str | None = None) -> list[list]:
    """Convert bookmarks to a list-of-lists for gr.Dataframe."""
    rows = load_bookmarks(adapter_name)
    return [[r["hookpoint"], r["latent_idx"], r["note"], r["created_at"]] for r in rows]


def _on_bookmark_row_select(evt: gr.SelectData):
    """Store the selected row index when a bookmark row is clicked."""
    return evt.index[0] if evt.index is not None else -1


def _refresh_bookmarks(adapter_name):
    """Refresh the bookmarks table when adapter changes."""
    return _bookmarks_to_df(adapter_name)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
_chat_css = """
.chat-user, .chat-assistant {
    max-width: 75% !important;
    border-radius: 16px !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
}
.chat-user {
    margin-right: auto !important;
    margin-left: 0 !important;
    background: #1e3a5f !important;
    border: 1px solid #2d5a8e !important;
}
.chat-assistant {
    margin-left: auto !important;
    margin-right: 0 !important;
    background: #1c3a2a !important;
    border: 1px solid #2d6e4a !important;
}
.chat-user textarea, .chat-assistant textarea {
    border: none !important;
    background: transparent !important;
    color: #e2e8f0 !important;
}
.chat-user textarea::placeholder, .chat-assistant textarea::placeholder {
    color: #94a3b8 !important;
}
.chat-user label span, .chat-assistant label span {
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}
.chat-user label span { color: #60a5fa !important; }
.chat-assistant label span { color: #4ade80 !important; }
"""

with gr.Blocks(title="TopKLoRA Dashboard", css=_chat_css) as demo:
    gr.Markdown("# TopKLoRA Dashboard")

    # ---- Tab 1: Model Loading ----
    with gr.Tab("Model Loading"):
        with gr.Row():
            base_model_dd = gr.Dropdown(
                label="Base model",
                choices=_base_models,
                value=_base_models[0] if _base_models else None,
                allow_custom_value=True,
            )
            adapter_dd = gr.Dropdown(
                label="Adapter",
                choices=list(_adapter_lookup.keys()),
                value=list(_adapter_lookup.keys())[0] if _adapter_lookup else None,
            )
        # Hidden textbox carries the resolved adapter path
        adapter_path_tb = gr.Textbox(visible=False)
        with gr.Row():
            device_dd = gr.Dropdown(
                label="Device",
                choices=["cpu", "cuda", "mps"],
                value="cpu",
            )
        load_btn = gr.Button("Load Model", variant="primary")
        status_tb = gr.Textbox(label="Status", interactive=False)
        hookpoint_config = gr.HTML(label="Adapter Configuration")

        # Wire adapter dropdown to resolve path
        adapter_dd.change(
            fn=_on_adapter_select,
            inputs=[adapter_dd],
            outputs=[adapter_path_tb],
        )

        # Initialise adapter path from the default selection
        if _adapter_lookup:
            adapter_path_tb.value = list(_adapter_lookup.values())[0]["path"]

    # ---- Tab 2: Activation Visualization ----
    with gr.Tab("Activation Visualization"):
        with gr.Row():
            viz_hookpoint = gr.Dropdown(label="Hookpoint", choices=[])
            viz_latent = gr.Number(label="Latent index", value=0, precision=0)
        hookpoint_info = gr.Textbox(label="Adapter info", interactive=False)

        n_turns = gr.State(1)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):

                @gr.render(inputs=[n_turns])
                def _render_chat(n):
                    textboxes = []
                    for i in range(n):
                        is_user = i % 2 == 0
                        role = "User" if is_user else "Assistant"
                        tb = gr.Textbox(
                            label=f"{role} (Turn {i + 1})",
                            placeholder=f"Enter {role.lower()} message...",
                            lines=2,
                            elem_classes=["chat-user"]
                            if is_user
                            else ["chat-assistant"],
                        )
                        textboxes.append(tb)

                    viz_btn = gr.Button("Visualize", variant="primary")
                    viz_btn.click(
                        fn=_on_visualize,
                        inputs=textboxes + [viz_hookpoint, viz_latent],
                        outputs=[viz_html],
                    )

                with gr.Row():
                    add_turn_btn = gr.Button("Add Turn", size="sm")
                    remove_turn_btn = gr.Button("Remove Turn", size="sm")

                add_turn_btn.click(lambda n: n + 1, inputs=[n_turns], outputs=[n_turns])
                remove_turn_btn.click(
                    lambda n: max(1, n - 1), inputs=[n_turns], outputs=[n_turns]
                )

            with gr.Column(scale=1):
                viz_html = gr.HTML(label="Activation Heatmap")

    # ---- Tab 3: Interactive Steering ----
    with gr.Tab("Interactive Steering"):
        with gr.Row():
            amp_num = gr.Number(label="Amplification", value=5.0, interactive=True)
            tokens_num = gr.Number(
                label="Max new tokens", value=128, precision=0, interactive=True
            )

        gr.Markdown("### Steering Rules")
        with gr.Row():
            steer_hookpoint_dd = gr.Dropdown(
                label="Hookpoint",
                choices=[],
                allow_custom_value=True,
            )
            steer_latent_num = gr.Number(label="Latent Index", value=0, precision=0)
            steer_effect_dd = gr.Dropdown(
                label="Effect",
                choices=["enable", "disable", "isolate"],
                value="enable",
            )
            add_row_btn = gr.Button("Add Row", size="sm")

        steer_df = gr.Dataframe(
            headers=["Hookpoint", "Latent Index", "Effect"],
            datatype=["str", "number", "str"],
            row_count=(1, "dynamic"),
            col_count=(3, "fixed"),
            label="Steering Configuration",
            interactive=True,
        )

        gr.Markdown("### Conversation")
        steer_n_turns = gr.State(1)

        @gr.render(inputs=[steer_n_turns])
        def _render_steer_chat(n):
            textboxes = []
            for i in range(n):
                is_user = i % 2 == 0
                role = "User" if is_user else "Assistant"
                tb = gr.Textbox(
                    label=f"{role} (Turn {i + 1})",
                    placeholder=f"Enter {role.lower()} message...",
                    lines=2,
                    elem_classes=["chat-user"] if is_user else ["chat-assistant"],
                )
                textboxes.append(tb)

            gen_btn = gr.Button("Generate", variant="primary")
            gen_btn.click(
                fn=_on_generate,
                inputs=textboxes + [steer_df, amp_num, tokens_num],
                outputs=[baseline_out, steered_out, steer_stats_html],
            )

        with gr.Row():
            steer_add_turn_btn = gr.Button("Add Turn", size="sm")
            steer_remove_turn_btn = gr.Button("Remove Turn", size="sm")

        steer_add_turn_btn.click(
            lambda n: n + 1, inputs=[steer_n_turns], outputs=[steer_n_turns]
        )
        steer_remove_turn_btn.click(
            lambda n: max(1, n - 1), inputs=[steer_n_turns], outputs=[steer_n_turns]
        )

        steer_stats_html = gr.HTML(label="Activation Statistics")

        with gr.Row():
            baseline_out = gr.HTML(label="Baseline Output")
            steered_out = gr.HTML(label="Steered Output")

    # ---- Tab 4: Cached Activation Explorer ----
    with gr.Tab("Cached Activations"):
        from dashboard.backend import CACHE_DIR as _cache_dir

        gr.Markdown(
            f"Explore top activating examples from cached activations in `{_cache_dir}`. "
            "No model loading required."
        )
        with gr.Row():
            cached_adapter_dd = gr.Dropdown(
                label="Cached Adapter",
                choices=_cached_adapters,
                value=_cached_adapters[0] if _cached_adapters else None,
            )
            # Initialize hookpoints from the first adapter
            _initial_hookpoints = (
                get_cached_hookpoints(_cached_adapters[0]) if _cached_adapters else []
            )
            cached_hookpoint_dd = gr.Dropdown(
                label="Hookpoint",
                choices=_initial_hookpoints,
                value=_initial_hookpoints[0] if _initial_hookpoints else None,
            )
        # Track which latent input was last used: "dropdown" or "textbox"
        latent_input_source = gr.State("dropdown")

        with gr.Row():
            # Initialize latent choices from first hookpoint
            _initial_latent_choices = []
            if _cached_adapters and _initial_hookpoints:
                _initial_latent_choices = get_cached_latent_choices(
                    _cached_adapters[0], _initial_hookpoints[0]
                )
            _latent_choices_display = [
                display for display, _ in _initial_latent_choices
            ]
            _latent_choices_values = [idx for _, idx in _initial_latent_choices]

            cached_latent_dd = gr.Dropdown(
                label="Latent (from stats)",
                choices=list(zip(_latent_choices_display, _latent_choices_values))
                if _latent_choices_values
                else [],
                value=_latent_choices_values[0] if _latent_choices_values else None,
                scale=3,
            )
            cached_latent_tb = gr.Number(
                label="Latent ID (direct)",
                value=None,
                precision=0,
                scale=1,
            )
        with gr.Row():
            cached_n_examples = gr.Number(
                label="Number of Examples", value=10, precision=0
            )
            cached_show_pad = gr.Checkbox(label="Show pad tokens", value=False)
            latent_source_indicator = gr.Textbox(
                label="Active input",
                value="← Dropdown",
                interactive=False,
                scale=1,
            )

        show_examples_btn = gr.Button("Show Top Examples", variant="primary")

        # -- Bookmark controls --
        with gr.Row():
            bookmark_note = gr.Textbox(
                label="Note",
                placeholder="Why is this latent interesting?",
                lines=1,
                scale=3,
            )
            bookmark_btn = gr.Button("Bookmark This Latent", scale=1)

        # -- Bookmarks browser --
        with gr.Accordion("Bookmarked Latents", open=False):
            _initial_bookmarks = _bookmarks_to_df(
                _cached_adapters[0] if _cached_adapters else None
            )
            bookmark_selected_idx = gr.State(-1)
            bookmarks_df = gr.Dataframe(
                value=_initial_bookmarks,
                headers=["Hookpoint", "Latent", "Note", "Date"],
                datatype=["str", "number", "str", "str"],
                interactive=False,
                label="Bookmarks",
            )
            with gr.Row():
                bookmark_load_btn = gr.Button("Load Selected", size="sm")
                bookmark_delete_btn = gr.Button("Delete Selected", size="sm")

        cached_examples_html = gr.HTML(label="Top Activating Examples")

    # ---- Wiring ----
    load_btn.click(
        fn=_on_load_model,
        inputs=[base_model_dd, adapter_path_tb, device_dd],
        outputs=[status_tb, hookpoint_config, viz_hookpoint, steer_hookpoint_dd],
    )

    add_row_btn.click(
        fn=_on_add_steering_row,
        inputs=[steer_df, steer_hookpoint_dd, steer_latent_num, steer_effect_dd],
        outputs=[steer_df],
    )

    viz_hookpoint.change(
        fn=_on_hookpoint_change,
        inputs=[viz_hookpoint],
        outputs=[hookpoint_info],
    )

    viz_latent.change(
        fn=_on_latent_change,
        inputs=[viz_latent],
        outputs=[viz_html],
    )

    cached_adapter_dd.change(
        fn=_on_cached_adapter_select,
        inputs=[cached_adapter_dd],
        outputs=[cached_hookpoint_dd, cached_latent_dd],
    )

    cached_hookpoint_dd.change(
        fn=_on_cached_hookpoint_select,
        inputs=[cached_adapter_dd, cached_hookpoint_dd],
        outputs=[cached_latent_dd],
    )

    show_examples_btn.click(
        fn=_on_show_cached_examples,
        inputs=[
            cached_adapter_dd,
            cached_hookpoint_dd,
            cached_latent_dd,
            cached_latent_tb,
            latent_input_source,
            cached_n_examples,
            cached_show_pad,
        ],
        outputs=[cached_examples_html],
    )

    # Track which latent input was last used
    cached_latent_dd.change(
        fn=lambda _: ("dropdown", "← Dropdown"),
        inputs=[cached_latent_dd],
        outputs=[latent_input_source, latent_source_indicator],
    )
    cached_latent_tb.change(
        fn=lambda _: ("textbox", "Textbox →"),
        inputs=[cached_latent_tb],
        outputs=[latent_input_source, latent_source_indicator],
    )

    # -- Bookmark wiring --
    bookmark_btn.click(
        fn=_on_add_bookmark,
        inputs=[
            cached_adapter_dd,
            cached_hookpoint_dd,
            cached_latent_dd,
            cached_latent_tb,
            latent_input_source,
            bookmark_note,
        ],
        outputs=[bookmarks_df],
    )

    bookmarks_df.select(
        fn=_on_bookmark_row_select,
        outputs=[bookmark_selected_idx],
    )

    bookmark_delete_btn.click(
        fn=_on_delete_selected_bookmark,
        inputs=[cached_adapter_dd, bookmark_selected_idx, bookmarks_df],
        outputs=[bookmarks_df],
    )

    bookmark_load_btn.click(
        fn=_on_load_selected_bookmark,
        inputs=[cached_adapter_dd, bookmark_selected_idx],
        outputs=[
            cached_hookpoint_dd,
            cached_latent_dd,
            cached_examples_html,
            latent_input_source,
            latent_source_indicator,
        ],
    )

    # Refresh bookmarks when adapter changes
    cached_adapter_dd.change(
        fn=_refresh_bookmarks,
        inputs=[cached_adapter_dd],
        outputs=[bookmarks_df],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
