import json
import math
import os
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import contextmanager

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, Patch, State, ctx, dcc, html
from dash_extensions import EventListener

import metatlas2.shared_tools.database_interact as dbi
import metatlas2.shared_tools.logging_config as lcf
import metatlas2.targeted.create_curation_container as ccc
from metatlas2.targeted.note_options import (
    normalize_note_value,
    should_require_note_selection,
)

logger = lcf.get_logger("analysis_gui")

# ---------------------------------------------------------------------------
# Visual constants shared by the layout, the CSS (assets/analysis_gui.css) and
# the Plotly template.  Semantic trace colors (green bounds, orange suggestions,
# purple atlas RT, isomer red/blue) are intentionally unchanged.
# ---------------------------------------------------------------------------
FONT_STACK = "Arial, Helvetica, 'Liberation Sans', sans-serif"
_TEXT_COLOR = "#1f2933"
_AXIS_COLOR = "#9aa5b1"

pio.templates["metatlas"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT_STACK, size=15, color=_TEXT_COLOR),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showline=True, linecolor=_AXIS_COLOR, ticks="outside", tickcolor=_AXIS_COLOR,
                   showgrid=False, zeroline=False, automargin=True,
                   title=dict(font=dict(size=18), standoff=14), tickfont=dict(size=16)),
        yaxis=dict(showline=True, linecolor=_AXIS_COLOR, ticks="outside", tickcolor=_AXIS_COLOR,
                   showgrid=False, zeroline=False, automargin=True,
                   title=dict(font=dict(size=18), standoff=16), tickfont=dict(size=16),
                   exponentformat="e"),
        hoverlabel=dict(bgcolor="white", bordercolor="#cbd2d9",
                        font=dict(family=FONT_STACK, size=15, color=_TEXT_COLOR)),
        title=dict(font=dict(family=FONT_STACK, size=18, color=_TEXT_COLOR)),
    )
)
_TEMPLATE_JSON = pio.templates["metatlas"].to_plotly_json()

# Max points per file kept on the (invisible) hover/click trace.
_MAX_HOVER_PTS = 75
# Static EIC span is snapped outward to this grid (minutes) so that small RT
# nudges keep the same span and can be sent as a partial (Patch) update.
_SPAN_GRID = 0.5


@contextmanager
def _timed(name):
    """Log wall-clock duration of a block at DEBUG level."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug(f"[timing] {name}: {(time.perf_counter() - t0) * 1000:.1f} ms")


def _f32(arr, decimals=None):
    """Compact numpy array for JSON transport (float32, optional rounding)."""
    a = np.asarray(arr, dtype=np.float64)
    if decimals is not None:
        a = np.round(a, decimals)
    return a.astype(np.float32)


def build_dash_app(
    analysis_gui_obj,
    port=8050,
    shutdown_holder=None,
    run_parameters=None,
):
    logger.debug("Starting the app factory for the Analysis GUI...")

    # Read in main data
    manual_curation_df = analysis_gui_obj.experimental_data.curation_df

    # Set up basic GUI params
    _gui_cfg = analysis_gui_obj.config.gui_config if analysis_gui_obj.config else {}
    _op = analysis_gui_obj.override_parameters or {}
    _resolved_cfg = {**_gui_cfg, **{k: v for k, v in _op.items() if v is not None}}
    top_n_hits = _resolved_cfg.get("gui_top_n_hits", 20)
    lcmsruns_color_map = _resolved_cfg.get("gui_lcmsruns_colors", {})
    force_eval = _resolved_cfg.get("gui_require_all_evaluated", False)
    async_flush_errors: dict = {}
    database_flush_errors: dict = {}
    notes_cfg = analysis_gui_obj.notes

    # Extract metadata for display
    project_shortname = analysis_gui_obj.project_name.split("_")[4]
    chrom = analysis_gui_obj.chromatography.upper()
    pol = analysis_gui_obj.polarity.upper()
    _ta = analysis_gui_obj.ta
    _cat = _ta.atlas_category if _ta else None
    atlas_category = (_cat.value if hasattr(_cat, 'value') else str(_cat or '')).upper()
    atlas_name = (_ta.atlas_name if _ta else '').upper()
    rta = analysis_gui_obj.rt_alignment_number
    tga = analysis_gui_obj.analysis_number

    # Set up all passing compounds as options for the dropdown
    compound_options = [
        {"label": f"{i+1}: {row['label']} ({row['adduct']})", "value": i}
        for i, row in manual_curation_df.reset_index(drop=True).iterrows()
    ]

    # Create the app
    try:
        if not os.getenv('JUPYTERHUB_SERVICE_PREFIX'):
            requests_prefix = "/"
        else:
            requests_prefix = f"{os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')}proxy/{port}/"
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix=requests_prefix,
            suppress_callback_exceptions=True,
            update_title=None,  # no "Updating..." flicker in the browser tab
        )
        app.title = f"{project_shortname} | {chrom} | {pol} | {atlas_category}-{atlas_name} | RTA{rta} | TGA{tga}"
        logger.debug("App built successfully")
        app.config.prevent_initial_callbacks = "initial_duplicate"
    except Exception as e:
        traceback.print_exc()
        logger.error(f"FAILED: {e}")

    # Set up some caching to help with race conditions
    flush_lock = threading.RLock()
    latest_flushed_seq_by_session = {}
    db_write_lock = threading.Lock()
    _cache_lock = threading.RLock()  # guards the render caches below (server is threaded)
    isomer_string_cache = {}
    _ms1_prep_cache: OrderedDict[str, dict] = OrderedDict()
    _ms1_static_cache: OrderedDict[tuple, dict] = OrderedDict()
    _ms2_scans_cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
    _active_flush_threads: list[threading.Thread] = []

    def _lru_get(cache, key):
        with _cache_lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        return None

    def _lru_put(cache, key, value, maxsize):
        with _cache_lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > maxsize:
                cache.popitem(last=False)
        return value

    # Pre-index DataFrames by mz_rt_uid for fast lookups
    ms1_by_compound = {}
    ms2_by_compound = {}

    # Index MS1 data
    for mz_rt_uid, group in analysis_gui_obj.experimental_data.ms1_df.groupby(["mz_rt_uid"], observed=True):
        if len(mz_rt_uid) == 1:
            mz_rt_uid = mz_rt_uid[0]
        group = group.copy()
        if "spec_rts" in group.columns:
            group["spec_rts"] = group["spec_rts"].apply(np.asarray)
        if "spec_ints" in group.columns:
            group["spec_ints"] = group["spec_ints"].apply(np.asarray)
        ms1_by_compound[mz_rt_uid] = group

    # Index MS2 data and pre-sort each scan's hits list by score descending.
    def _presort_hits(raw):
        if not isinstance(raw, list) or not raw:
            return raw
        try:
            return sorted(raw, key=lambda h: float(h.get("score", float("-inf"))) if isinstance(h, dict) else float("-inf"), reverse=True)
        except Exception:
            return raw

    def _best_score(h):
        if isinstance(h, list) and h and isinstance(h[0], dict):
            try:
                s = float(h[0].get("score", float("-inf")))
                return s if np.isfinite(s) else float("-inf")
            except (TypeError, ValueError):
                return float("-inf")
        return float("-inf")

    for mz_rt_uid, group in analysis_gui_obj.experimental_data.ms2_df.groupby(["mz_rt_uid"], observed=True):
        if len(mz_rt_uid) == 1:
            mz_rt_uid = mz_rt_uid[0]
        group = group.copy()
        if "hits" in group.columns:
            group["hits"] = group["hits"].apply(_presort_hits)
            group["_best_hit_score"] = group["hits"].apply(_best_score)
        ms2_by_compound[mz_rt_uid] = group

    ms2_markers_by_compound = {}
    for uid, group_df in ms2_by_compound.items():
        markers = []
        for r in group_df.itertuples(index=False):
            rt = getattr(r, "scan_rt", None)
            if rt is None or (isinstance(rt, float) and np.isnan(rt)):
                continue
            best_score = getattr(r, "_best_hit_score", float("-inf"))
            markers.append((float(rt), best_score))
        # Sort by score descending so [:100] slice always gives the top-100 by score
        markers.sort(key=lambda x: x[1], reverse=True)
        ms2_markers_by_compound[uid] = markers

    def _compound_row(idx):
        return manual_curation_df.iloc[idx]

    def _default_plot_bounds_from_row(row, pad=2.0):
        atlas_rt_min = row.get("atlas_rt_min")
        atlas_rt_max = row.get("atlas_rt_max")

        if pd.notnull(atlas_rt_min) and pd.notnull(atlas_rt_max):
            base_min = float(atlas_rt_min)
            base_max = float(atlas_rt_max)
        else:
            base_min, base_max = float(row["rt_min"]), float(row["rt_max"])

        if base_max < base_min:
            base_min, base_max = base_max, base_min

        window_min = max(0.0, base_min - pad)
        window_max = base_max + pad
        if window_max <= window_min:
            window_max = window_min + 1.0
        return window_min, window_max

    def _load_state(compound_idx, ms2_idx=0, session_id=None, edit_seq=0):
        row = _compound_row(compound_idx)
        rt_min, rt_max = float(row["rt_min"]), float(row["rt_max"])
        ms2_note = normalize_note_value(row.get("ms2_notes"), notes_cfg["ms2_notes"])
        ms1_note = normalize_note_value(row.get("ms1_notes"), notes_cfg["ms1_notes"])
        other_notes_raw = row.get("other_notes", [])
        if other_notes_raw is None or other_notes_raw == "":
            other_note = []
        else:
            try:
                if isinstance(other_notes_raw, str) and other_notes_raw.startswith("["):
                    other_note = json.loads(other_notes_raw)
                elif isinstance(other_notes_raw, str):
                    other_note = [v.strip() for v in other_notes_raw.split(" // ") if v.strip()]
                else:
                    other_note = list(other_notes_raw)
            except Exception:
                other_note = [other_notes_raw] if other_notes_raw else []
        other_note = [v for v in other_note if v in notes_cfg["other_notes"]]

        return {
            "session_id": session_id or str(uuid.uuid4()),
            "edit_seq": int(edit_seq),
            "compound_idx": compound_idx,
            "ms2_idx": ms2_idx,
            "rt_min": rt_min,
            "rt_max": rt_max,
            "ms1_note": ms1_note,
            "ms2_note": ms2_note,
            "other_note": other_note,
            "analyst_notes": row.get("analyst_notes") or "",
            "id_notes": row.get("identification_notes") or "",
            "last_saved": None,
            "isomer_snap_idx": 0,
            "flush_error": None,
            "highlighted_files": [],
        }

    def _patch_with_seq(state, **changes):
        new_state = dict(state)
        new_state.update(changes)
        old_seq = int(state.get("edit_seq", 0))
        new_state["edit_seq"] = old_seq + 1
        return new_state

    def _ensure_valid_state(state):
        if isinstance(state, dict) and "compound_idx" in state:
            return state
        logger.warning(
            "Malformed session-store state (%s). Resetting to starting compound.",
            type(state).__name__,
        )
        return _load_state(starting_compound_idx)

    def _patch_rt_change(state, new_min, new_max):
        rt_min = max(0.0, min(new_min, new_max))
        rt_max = max(rt_min, new_max)
        logger.debug(f"[_patch_rt_change] new_min={new_min}, new_max={new_max} -> rt_min={rt_min}, rt_max={rt_max}")
        return _patch_with_seq(state, rt_min=round(rt_min, 4), rt_max=round(rt_max, 4), ms2_idx=0)

    def _find_starting_compound_idx():
        """Find the first compound with no ms2_notes set.

        Looks for compounds where ms2_notes is blank (empty string or NaN) and returns next index to start
        """

        blank_mask = (
            manual_curation_df["ms2_notes"].isna()
            | manual_curation_df["ms2_notes"].isin(["", "no selection"])
                ) & (~manual_curation_df["ms1_notes"].str.lower().str.contains("remove", na=False))
        blank_positions = manual_curation_df.index[blank_mask]

        if len(blank_positions) == 0:
            return 0

        first_blank_idx = blank_positions[0]
        first_blank_compound = manual_curation_df.index.get_loc(first_blank_idx)
        return first_blank_compound

    use_starting_index_finder = True
    if use_starting_index_finder is True:
        starting_compound_idx = _find_starting_compound_idx()
        if starting_compound_idx > 0:
            logger.info(f"Resuming analysis at compound {starting_compound_idx+1}: "
                    f"{manual_curation_df.iloc[starting_compound_idx]['label']}")
        else:
            logger.info("Starting new analysis at compound 1")
    else:
        starting_compound_idx = 0
        logger.info("Starting analysis at compound 1 (starting index finder disabled)")

    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp", "target.tagName"]}],
    )

    # ------------------------------------------------------------------
    # Sizing (honors gui_width / gui_height config, in inches)
    # ------------------------------------------------------------------
    PX_PER_ROW = 52
    FIXED_CHROME = 340
    BUTTON_ROW_H = 48
    _gui_width_in = _resolved_cfg.get("gui_width") or None
    _gui_height_in = _resolved_cfg.get("gui_height") or None
    DPI = 96

    n_rows = (
        len(notes_cfg["ms1_notes"])
        + len(notes_cfg["ms2_notes"])
        + len(notes_cfg["other_notes"])
    )

    if _gui_height_in is not None:
        left_bar_h = int(float(_gui_height_in) * DPI)
    else:
        left_bar_h = n_rows * PX_PER_ROW + FIXED_CHROME

    graph_avail_h = left_bar_h - 2 * BUTTON_ROW_H
    ms1_height = max(int(graph_avail_h * 0.65), 200)
    ms2_height = max(int(graph_avail_h * 0.50), 150)

    if _gui_width_in is not None:
        app_width = int(float(_gui_width_in) * DPI)
    else:
        app_width = int(left_bar_h * (12 / 8) * 1.5)

    _auto_left_bar_h = n_rows * PX_PER_ROW + FIXED_CHROME
    _font_scale = left_bar_h / _auto_left_bar_h if _auto_left_bar_h > 0 else 1.0
    _font_scale = max(0.5, min(_font_scale, 2.0))

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _option_label(text, key):
        """Radio/checklist label with a right-aligned hotkey badge."""
        children = [html.Span(text, className="opt-text")]
        if key:
            children.append(html.Kbd(key, className="hotkey"))
        return html.Span(children, className="opt-label")

    def _note_options(kind):
        hotkeys = notes_cfg[f"{kind}_hotkeys"]
        return [{"label": _option_label(lbl, hotkeys.get(lbl, "")), "value": lbl}
                for lbl in notes_cfg[f"{kind}_notes"]]

    def _section(title, body, extra_class="", grow=0):
        # grow > 0: the card takes a share of spare left-panel height so the
        # panel matches the plot column (options spread out evenly).
        return dbc.Card(
            [dbc.CardHeader(title, className="section-header"), dbc.CardBody(body, className="section-body")],
            className=f"panel-card {extra_class}".strip(),
            style={"flexGrow": grow} if grow else None,
        )

    def _shortcut_table(rows):
        return dbc.Table(
            html.Tbody([html.Tr([html.Td(html.Kbd(k), className="kbd-cell"), html.Td(desc)]) for k, desc in rows]),
            bordered=False, size="sm", className="shortcut-table mb-3",
        )

    def _scale_toggle(component_id):
        return dbc.RadioItems(
            id=component_id,
            options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
            value="linear",
            className="btn-group scale-toggle",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary btn-sm",
            labelCheckedClassName="active",
        )

    def _counter(counter_id, progress_id, color, initial_value=0):
        return html.Div(
            [
                html.Div(id=counter_id, className="counter-text"),
                dbc.Progress(id=progress_id, value=initial_value, color=color, className="counter-progress"),
            ],
            className="counter",
        )

    def _build_header():
        return html.Header(
            [
                html.Div(
                    [html.Span("Metatlas", className="brand-strong"), html.Span("Targeted Curation", className="brand-light")],
                    className="brand",
                ),
                html.Div(app.title, className="meta-text"),
                html.Div(id="save-indicator", className="save-indicator"),
                html.Div(
                    [
                        dbc.Button("Hotkeys", id="help-btn", color="secondary", outline=True, size="sm", className="header-btn"),
                        dbc.Button("Figure guide", id="guide-btn", color="secondary", outline=True, size="sm", className="header-btn"),
                        dbc.Button("Save & exit", id="exit-open-btn", color="danger", size="sm", className="header-btn"),
                    ],
                    className="header-actions",
                ),
            ],
            className="app-header",
        )

    def _build_left_panel():
        return [
            _section(
                "Compound",
                [
                    dcc.Dropdown(id="compound-dd", options=compound_options, value=starting_compound_idx,
                                 clearable=False, className="compound-dd"),
                    html.Div(id="id-notes", className="id-notes", children="No identification notes"),
                ],
            ),
            _section(
                "Analyst notes",
                dbc.Textarea(id="analyst-notes", placeholder="Free-text notes for this compound…",
                             debounce=True, className="analyst-notes", rows=2),
            ),
            _section(
                "MS1 quality",
                dcc.RadioItems(id="ms1-radio", options=_note_options("ms1"), value=notes_cfg["ms1_notes"][0],
                               className="note-options", labelClassName="note-option", inputClassName="note-input"),
                extra_class="note-card", grow=len(notes_cfg["ms1_notes"]),
            ),
            _section(
                "MS2 quality",
                dcc.RadioItems(id="ms2-radio", options=_note_options("ms2"), value=notes_cfg["ms2_notes"][0],
                               className="note-options", labelClassName="note-option", inputClassName="note-input"),
                extra_class="note-card", grow=len(notes_cfg["ms2_notes"]),
            ),
            _section(
                "Other notes",
                dcc.Checklist(id="other-checklist", options=_note_options("other"), value=[],
                              className="note-options", labelClassName="note-option", inputClassName="note-input"),
                extra_class="note-card", grow=len(notes_cfg["other_notes"]),
            ),
            _section("Current selection", html.Div(id="status-current"), extra_class="status-card"),
        ]

    def _build_plot_panel():
        return [
            html.Div(id="error-banner"),
            dbc.Card(
                [
                    dcc.Graph(
                        id="ms1-graph",
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "toImageButtonOptions": {"filename": f"{project_shortname}_{chrom}_{pol}_MS1"},
                            "edits": {"shapePosition": True, "titleText": False},
                            "doubleClick": True,
                        },
                        style={"height": f"{ms1_height}px"},
                    ),
                    html.Div(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("◀ Prev", id="prev-btn", color="primary", size="sm"),
                                    dbc.Button("Next ▶", id="next-btn", color="primary", size="sm"),
                                ],
                                className="nav-group",
                            ),
                            _counter("compound-counter", "compound-progress", "primary",
                                     round((starting_compound_idx + 1) / max(len(compound_options), 1) * 100, 1)),
                            html.Div(className="toolbar-spacer"),
                            dbc.Button("Accept suggestions", id="accept-suggestions", color="warning", size="sm",
                                       className="tool-btn"),
                            dbc.Button("Snap to isomer", id="snap-to-isomer", color="secondary", outline=True,
                                       size="sm", className="tool-btn"),
                            _scale_toggle("yaxis-scale-radio"),
                        ],
                        className="plot-toolbar",
                    ),
                ],
                className="plot-card",
            ),
            dbc.Card(
                [
                    dcc.Graph(
                        id="ms2-graph",
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "toImageButtonOptions": {"filename": f"{project_shortname}_{chrom}_{pol}_MS2"},
                        },
                        style={"height": f"{ms2_height}px"},
                    ),
                    html.Div(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("◀ Prev MS2", id="ms2-prev-1", color="primary", size="sm"),
                                    dbc.Button("Next MS2 ▶", id="ms2-next-1", color="primary", size="sm"),
                                ],
                                className="nav-group",
                            ),
                            _counter("ms2-counter-1", "ms2-progress-1", "secondary"),
                            html.Div(className="toolbar-spacer"),
                            _scale_toggle("ms2-yaxis-scale-radio"),
                        ],
                        className="plot-toolbar",
                    ),
                ],
                className="plot-card",
            ),
        ]

    def _tooltips():
        tips = [
            ("prev-btn", "Previous compound  (j / ←) — saves current compound"),
            ("next-btn", "Next compound  (k / →) — saves current compound"),
            ("accept-suggestions", "Use the suggested RT bounds (orange)  (n)"),
            ("snap-to-isomer", "Snap RT bounds to the next isomer box  (m)"),
            ("ms2-prev-1", "Previous MS2 scan  (l / ↑)"),
            ("ms2-next-1", "Next MS2 scan  (; / ↓)"),
            ("help-btn", "Keyboard shortcuts"),
            ("guide-btn", "Plot legend and how-to"),
            ("exit-open-btn", "Save the current compound and close the app"),
        ]
        return [dbc.Tooltip(text, target=target, placement="bottom", delay={"show": 400, "hide": 0})
                for target, text in tips]

    help_offcanvas = dbc.Offcanvas(
        id="help-offcanvas",
        title="Keyboard shortcuts",
        is_open=False,
        placement="end",
        className="help-offcanvas",
        children=[
            html.H6("Compound navigation", className="offcanvas-heading"),
            _shortcut_table([("j", "Previous compound"), ("k", "Next compound"),
                             ("←", "Previous compound"), ("→", "Next compound")]),
            html.H6("RT bound nudge", className="offcanvas-heading"),
            _shortcut_table([("a", "RT min − 0.05"), ("s", "RT min + 0.05"),
                             ("d", "RT max − 0.05"), ("f", "RT max + 0.05")]),
            html.H6("MS2 scan navigation", className="offcanvas-heading"),
            _shortcut_table([("l", "Previous MS2 scan"), (";", "Next MS2 scan"),
                             ("↑", "Previous MS2 scan"), ("↓", "Next MS2 scan")]),
            html.H6("Other actions", className="offcanvas-heading"),
            _shortcut_table([("n", "Accept RT suggestions"), ("m", "Snap to isomer")]),
            html.H6("MS1 quality", className="offcanvas-heading"),
            _shortcut_table([(notes_cfg["ms1_hotkeys"].get(lbl, "—"), lbl) for lbl in notes_cfg["ms1_notes"]]),
            html.H6("MS2 quality", className="offcanvas-heading"),
            _shortcut_table([(notes_cfg["ms2_hotkeys"].get(lbl, "—"), lbl) for lbl in notes_cfg["ms2_notes"]]),
            html.H6("Other notes", className="offcanvas-heading"),
            _shortcut_table([(notes_cfg["other_hotkeys"].get(lbl, "—"), lbl) for lbl in notes_cfg["other_notes"]]),
        ],
    )

    def _swatch(color, dashed=False, box=False):
        cls = "swatch-box" if box else ("swatch-line dashed" if dashed else "swatch-line")
        return html.Span(className=cls, style={"--swatch": color})

    guide_offcanvas = dbc.Offcanvas(
        id="guide-offcanvas",
        title="Figure guide",
        is_open=False,
        placement="end",
        className="guide-offcanvas",
        children=[
            html.H6("MS1 plot legend", className="offcanvas-heading"),
            dbc.Table(
                html.Tbody([
                    html.Tr([html.Td([_swatch("gray"), "Sample lines"]),
                             html.Td(["Experimental samples are gray. Technical samples: ",
                                      ", ".join(f"{k} = {v}" for k, v in lcmsruns_color_map.items()) or "none configured",
                                      ". Click a line to highlight that file."])]),
                    html.Tr([html.Td([_swatch("purple"), "Atlas RT"]), html.Td("Atlas RT peak")]),
                    html.Tr([html.Td([_swatch("orange"), "Suggested"]), html.Td("Suggested RT bounds (solid: min, dashed: max)")]),
                    html.Tr([html.Td([_swatch("green"), "Current"]), html.Td("Current RT bounds (solid: min, dashed: max) — drag to edit")]),
                    html.Tr([html.Td([html.Span("▲", className="swatch-tri"), "MS2 scans"]), html.Td("RT of MS2 scans (top 100 by score)")]),
                    html.Tr([html.Td([_swatch("rgba(150,205,255,0.8)", box=True), _swatch("rgba(255,96,96,0.8)", box=True), "Isomers"]),
                             html.Td("Isomer RT windows (identical m/z or matching InChIKey prefix). Red when overlapping, blue otherwise.")]),
                ]),
                bordered=False, size="sm", className="guide-table mb-3",
            ),
            html.H6("Adjusting RT bounds", className="offcanvas-heading"),
            html.Ul([
                html.Li("Drag the green lines directly on the plot."),
                html.Li(["Nudge with ", html.Kbd("a"), " ", html.Kbd("s"), " (min) and ",
                         html.Kbd("d"), " ", html.Kbd("f"), " (max)."]),
                html.Li(["Press ", html.Kbd("m"), " to snap to the next shaded isomer box."]),
                html.Li(["Press ", html.Kbd("n"), " to accept the suggested (orange) bounds."]),
            ], className="guide-list"),
            html.H6("Range slider", className="offcanvas-heading"),
            html.P(
                "The blue dotted line is the max intensity across all files at every RT for this compound. "
                "The white area is in view; drag the slider handles to view the gray area.",
                className="guide-text",
            ),
            html.H6("MS2 plot legend", className="offcanvas-heading"),
            dbc.Table(
                html.Tbody([
                    html.Tr([html.Td("Mirror plot"), html.Td("Data ions (up) vs. reference ions (down)")]),
                    html.Tr([html.Td("Ref scaled"), html.Td("Reference intensity is scaled to the data spectrum peak")]),
                    html.Tr([html.Td([_swatch("green"), "Green"]), html.Td("Matching fragments")]),
                    html.Tr([html.Td([_swatch("red"), "Red"]), html.Td("Non-matching fragments")]),
                ]),
                bordered=False, size="sm", className="guide-table mb-3",
            ),
            html.H6("Saving", className="offcanvas-heading"),
            html.P(
                "RT bounds and notes are written to the project database when you move to another compound "
                "(buttons, hotkeys or dropdown) and when you use Save & exit.",
                className="guide-text",
            ),
        ],
    )

    exit_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save and exit?")),
            dbc.ModalBody([
                html.P("The current compound will be saved to the project database and the app will shut down."),
                html.Div(id="save-exit-status", className="save-exit-status"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="exit-cancel-btn", color="secondary", outline=True),
                dbc.Button("Save & exit", id="save-exit-btn", color="danger"),
            ]),
        ],
        id="exit-modal",
        is_open=False,
        centered=True,
    )

    app.layout = html.Div(
        [
            dcc.Store(id="session-store", storage_type="memory", data=_load_state(starting_compound_idx)),
            dcc.Store(id="nav-trigger-store", storage_type="memory", data={"idx": starting_compound_idx, "ts": 0}),
            dcc.Store(id="ms1-render-key", storage_type="memory", data=None),
            dcc.Store(id="ms2-render-key", storage_type="memory", data=None),
            html.Div(
                id="loading-overlay",
                className="loading-overlay",
                children=html.Div(
                    [
                        dbc.Spinner(color="light", size="md"),
                        html.Div("Loading compound…", id="loading-overlay-text", className="loading-text"),
                    ],
                    className="loading-box",
                ),
                style={"display": "none"},
            ),
            dbc.Toast(
                id="save-toast",
                header="Saved to database",
                is_open=False,
                dismissable=True,
                duration=6000,
                icon="success",
                className="save-toast",
            ),
            help_offcanvas,
            guide_offcanvas,
            exit_modal,
            keyboard_listener,
            *_tooltips(),
            _build_header(),
            html.Main(
                dbc.Row(
                    [
                        dbc.Col(html.Div(_build_left_panel(), className="left-panel-inner"), width=3, className="left-panel"),
                        dbc.Col(_build_plot_panel(), width=9, className="plot-panel"),
                    ],
                    className="g-3",
                ),
                className="app-main",
            ),
        ],
        id="root-container",
        className="app-root",
        style={
            "--ui-scale": str(round(_font_scale, 3)),
            "width": f"{app_width}px",
            "minWidth": f"{app_width}px",
            "maxWidth": f"{app_width}px",
        },
    )

    logger.debug("Layout constructed successfully")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _get_sorted_isomer_rt_bounds(row):
        """Return list of (rt_min, rt_max) for isomers, sorted by rt_min."""
        isomers = json.loads(row.get("isomers", "[]"))
        if not isomers:
            return []
        bounds = []
        for iso in isomers:
            mz_rt_uid = iso.get("mz_rt_uid", None)
            if mz_rt_uid is None:
                continue
            isomer_match = manual_curation_df[manual_curation_df["mz_rt_uid"] == mz_rt_uid]
            if isomer_match.empty:
                continue
            bounds.append((float(isomer_match.iloc[0]["rt_min"]), float(isomer_match.iloc[0]["rt_max"])))
        return sorted(bounds, key=lambda x: x[0])

    def _sanitize_numeric_list(vals):
        if vals is None:
            return np.array([], dtype=np.float64)
        try:
            return np.asarray(vals, dtype=np.float64)
        except (TypeError, ValueError):
            out = []
            for v in vals:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(np.nan)
            return np.asarray(out, dtype=np.float64)

    def _get_ms2_scans(row, rt_min=None, rt_max=None):
        """Return a DataFrame of the top N MS2 scans across all collision energies,
        sorted by best hit score descending then scan_rt ascending.

        Hits lists are pre-sorted at startup and each scan's best score is cached
        in ``_best_hit_score``.  Results are memoized per (compound, RT window);
        callers must treat the returned DataFrame as read-only.
        """
        mz_rt_uid = row["mz_rt_uid"]
        key = (mz_rt_uid, rt_min, rt_max)
        cached = _lru_get(_ms2_scans_cache, key)
        if cached is not None:
            return cached

        ms2_sub = ms2_by_compound.get(mz_rt_uid, pd.DataFrame())
        if not ms2_sub.empty and rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["scan_rt"] >= rt_min) & (ms2_sub["scan_rt"] <= rt_max)]

        if ms2_sub.empty:
            result = pd.DataFrame()
        elif "_best_hit_score" in ms2_sub.columns:
            result = ms2_sub.sort_values(["_best_hit_score", "scan_rt"], ascending=[False, True]).head(top_n_hits)
        else:
            result = ms2_sub.sort_values("scan_rt").head(top_n_hits)

        return _lru_put(_ms2_scans_cache, key, result, maxsize=256)

    def _format_collision_energy_label(ce_float):
        if abs(ce_float - 23.333) < 0.01:
            return "CE102040"
        if abs(ce_float - 43.333) < 0.01:
            return "CE205060"
        return f"CE{int(round(ce_float))}"

    def _move_ms2_idx(state, delta):
        """Move the flat MS2 scan index by delta, clamped to valid range."""
        row = _compound_row(state["compound_idx"])
        scans = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        n_scans = len(scans)
        if n_scans <= 0:
            raise dash.exceptions.PreventUpdate
        current_idx = max(0, int(state.get("ms2_idx", 0)))
        new_idx = max(0, min(current_idx + delta, n_scans - 1))
        if new_idx == current_idx:
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, ms2_idx=int(new_idx))

    def _compute_window_ms1_metrics(state):
        """Recompute MS1 metrics for the analyst-selected RT window.

        Delegates to ccc.analyze_ms1() (stage='post_curation_summary') so that
        metric definitions are identical to those used during curation object
        creation:
          - rt_peak  = mean of each file's highest-intensity in-window RT point
          - mz       = mean of all in-window mzs across all files

        The analyst's [rt_min, rt_max] window replaces the original in_feature
        mask: any point within the new window is treated as in-feature.
        """
        row = _compound_row(state["compound_idx"])
        mz_rt_uid = row["mz_rt_uid"]
        rt_min = float(state["rt_min"])
        rt_max = float(state["rt_max"])

        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())
        if sub.empty:
            return None

        # Build a minimal atlas_row dict with the reference values analyze_ms1 needs.
        atlas_row = {
            "mz": row.get("atlas_mz", np.nan),
            "rt_peak": row.get("atlas_rt_peak", np.nan),
            "rt_min": row.get("atlas_rt_min", np.nan),
            "rt_max": row.get("atlas_rt_max", np.nan),
        }

        # Replace each row's in_feature mask with the analyst's RT window so
        # analyze_ms1 treats exactly [rt_min, rt_max] as the feature window.
        rows_with_window_mask = []
        for r in sub.itertuples(index=False):
            rt_arr = np.asarray(getattr(r, "spec_rts", []), dtype=np.float64)
            window_mask = (rt_arr >= rt_min) & (rt_arr <= rt_max)
            new_row = r._asdict()
            new_row["in_feature"] = window_mask.tolist()
            rows_with_window_mask.append(new_row)

        if not rows_with_window_mask:
            return None

        windowed_df = pd.DataFrame(rows_with_window_mask)
        metrics = ccc.analyze_ms1(atlas_row, windowed_df, stage="post_curation_summary")
        if not metrics:
            return None

        return {
            "rt_peak": metrics.get("rt_peak"),
            "mz": metrics.get("mz"),
            "rt_error": metrics.get("rt_error"),
            "mz_error": metrics.get("mz_error"),
        }

    def _flush_to_db(state):
        sid = state.get("session_id", "unknown")
        seq = int(state.get("edit_seq", 0))
        flush_key = (sid, state.get("compound_idx"))

        if flush_key in async_flush_errors:
            state["flush_error"] = async_flush_errors.pop(flush_key)

        with flush_lock:
            latest_seq = latest_flushed_seq_by_session.get(flush_key, -1)
            if seq < latest_seq:
                return state
            latest_flushed_seq_by_session[flush_key] = seq

        row = _compound_row(state["compound_idx"])

        # Optimistically reflect user-edited RT bounds in memory immediately.
        try:
            idx = state["compound_idx"]
            df_idx = manual_curation_df.index[idx]
            with flush_lock:
                if "rt_min" in manual_curation_df.columns:
                    manual_curation_df.at[df_idx, "rt_min"] = state["rt_min"]
                if "rt_max" in manual_curation_df.columns:
                    manual_curation_df.at[df_idx, "rt_max"] = state["rt_max"]
        except Exception as exc:
            logger.warning(f"Optimistic in-memory RT update failed: {exc}")

        # Instead of blocking UI, compute metrics and write DB in background
        def _async_flush_worker():
            try:
                tol = 1e-4
                initial_rt_min = row.get("initial_rt_min", None)
                initial_rt_max = row.get("initial_rt_max", None)
                use_precomputed = False
                if initial_rt_min is not None and initial_rt_max is not None:
                    rt_min = float(state["rt_min"])
                    rt_max = float(state["rt_max"])
                    if abs(rt_min - float(initial_rt_min)) < tol and abs(rt_max - float(initial_rt_max)) < tol:
                        use_precomputed = True

                if use_precomputed:
                    updates = {
                        "passed_curation": False if "remove" in state.get("ms1_note", "").lower() else True,
                        "mz": row.get("mz", None),
                        "rt_min": state["rt_min"],
                        "rt_max": state["rt_max"],
                        "rt_peak": row.get("rt_peak", None),
                        "rt_error": row.get("rt_error", None),
                        "mz_error": row.get("mz_error", None),
                        "ms2_notes": normalize_note_value(state.get("ms2_note"), notes_cfg["ms2_notes"]),
                        "ms1_notes": normalize_note_value(state.get("ms1_note"), notes_cfg["ms1_notes"]),
                        "other_notes": " // ".join(state.get("other_note", [])),
                        "analyst_notes": state.get("analyst_notes", ""),
                        "identification_notes": state.get("id_notes", ""),
                    }
                else:
                    window_metrics = _compute_window_ms1_metrics(state)
                    if window_metrics is None:
                        updates = {
                            "ms2_notes": normalize_note_value(state.get("ms2_note"), notes_cfg["ms2_notes"]),
                            "ms1_notes": normalize_note_value(state.get("ms1_note"), notes_cfg["ms1_notes"]),
                            "other_notes": " // ".join(state.get("other_note", [])),
                            "analyst_notes": state.get("analyst_notes", ""),
                            "identification_notes": state.get("id_notes", ""),
                        }
                    else:
                        updates = {
                            "passed_curation": False if "remove" in state.get("ms1_note", "").lower() else True,
                            "mz": window_metrics["mz"],
                            "rt_min": state["rt_min"],
                            "rt_max": state["rt_max"],
                            "rt_peak": window_metrics["rt_peak"],
                            "rt_error": window_metrics["rt_error"],
                            "mz_error": window_metrics["mz_error"],
                            "ms2_notes": normalize_note_value(state.get("ms2_note"), notes_cfg["ms2_notes"]),
                            "ms1_notes": normalize_note_value(state.get("ms1_note"), notes_cfg["ms1_notes"]),
                            "other_notes": " // ".join(state.get("other_note", [])),
                            "analyst_notes": state.get("analyst_notes", ""),
                            "identification_notes": state.get("id_notes", ""),
                        }

                with db_write_lock:
                    with flush_lock:
                        current_latest = latest_flushed_seq_by_session.get(flush_key, -1)
                    if seq < current_latest:
                        logger.debug(
                           f"Skipping stale flush for compound {state['compound_idx']} "
                           f"seq={seq} (current latest={current_latest})"
                        )
                        return
                    dbi.write_curation_updates_to_db(
                        project_db_path=analysis_gui_obj.paths["project_db_path"],
                        rt_alignment_number=analysis_gui_obj.rt_alignment_number,
                        analysis_number=int(row["analysis_number"]),
                        rows=[{"mz_rt_uid": row["mz_rt_uid"], **updates}],
                        updated_field_keys=list(updates.keys()),
                    )

                # Update in-memory DataFrame to reflect what was written.
                df_idx = manual_curation_df.index[state["compound_idx"]]
                with flush_lock:
                    for col, val in updates.items():
                        if col in manual_curation_df.columns:
                            manual_curation_df.at[df_idx, col] = val

            except Exception as e:
                logger.error(f"Async flush worker failed: {e}")
                traceback.print_exc()
                async_flush_errors[flush_key] = str(e)
                # if the flush breaks, log it for export
                compound_label = f"{row.get('label', 'unknown')} ({row.get('adduct', '?')})"
                database_flush_errors[compound_label] = str(e)

        # Launch async worker thread - don't block UI!
        thread = threading.Thread(target=_async_flush_worker, daemon=True)
        thread.start()
        _active_flush_threads.append(thread)

        # Return immediately with optimistic state update
        state["last_saved"] = {
            "name": row["label"],
            "adduct": row["adduct"],
            "index": state["compound_idx"]+1,
            "rt_min": state["rt_min"],
            "rt_max": state["rt_max"],
            "ms1": normalize_note_value(state.get("ms1_note"), notes_cfg["ms1_notes"]),
            "ms2": normalize_note_value(state.get("ms2_note"), notes_cfg["ms2_notes"]),
            "other": state.get("other_note", []),
            "analyst_notes": state.get("analyst_notes", ""),
            "id_notes": state.get("id_notes", ""),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        state["flush_error"] = None
        return state

    # ------------------------------------------------------------------
    # MS1 figure
    #
    # The figure is split into two layers so that RT-bound edits can be sent
    # as a small dash.Patch instead of re-sending every EIC:
    #
    #   static layer  - EIC lines (grouped by color), the merged hover/click
    #                   trace, the max-EIC slider trace, title, axes and the
    #                   isomer legend.  Depends on (compound, highlighted
    #                   files, EIC span) - NOT on rt_min / rt_max.
    #   RT layer      - everything that depends on rt_min / rt_max: y-range,
    #                   isomer boxes (heights + overlap color), atlas /
    #                   suggested lines (heights), MS2 scan markers and the
    #                   two editable green shapes.
    #
    # Trace order:  [isomer boxes][static traces][atlas/suggested lines][ms2 markers]
    # A full render and (previous full render + Patch) are built from the same
    # _ms1_rt_layer() output, so they cannot drift apart (see
    # tests/test_analysis_gui.py).
    # ------------------------------------------------------------------
    def _ms1_prepared(mz_rt_uid):
        """Per-compound MS1 arrays + per-file metadata, computed once (LRU)."""
        cached = _lru_get(_ms1_prep_cache, mz_rt_uid)
        if cached is not None:
            return cached
        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())
        files = []
        for r in sub.itertuples(index=False):
            fn = getattr(r, "filename", "unknown")
            rt_arr = getattr(r, "spec_rts", None)
            int_arr = getattr(r, "spec_ints", None)
            if rt_arr is None or len(rt_arr) == 0:
                logger.warning(f"MS1 data for {fn} has no RT points; skipping trace")
                continue
            short_name = re.sub(r"_ms[12]_(?:neg|pos)$", "", "_".join(os.path.basename(fn).split(".")[0].split("_")[11:]))
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in fn.lower()), "gray")
            files.append({
                "fn": fn,
                "short": short_name,
                "color": color,
                "rt": np.asarray(rt_arr, dtype=np.float64),
                "int": np.asarray(int_arr, dtype=np.float64),
            })
        if files:
            all_rt = np.concatenate([f["rt"] for f in files])
            all_int = np.concatenate([f["int"] for f in files])
        else:
            all_rt = all_int = np.array([], dtype=np.float64)
        prep = {"files": files, "all_rt": all_rt, "all_int": all_int}
        return _lru_put(_ms1_prep_cache, mz_rt_uid, prep, maxsize=16)

    def _y_max_in_range(prep, lo, hi):
        rt, it = prep["all_rt"], prep["all_int"]
        if rt.size == 0:
            return 1.0
        vals = it[(rt >= lo) & (rt <= hi)]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 1.0
        return max(1.0, float(vals.max()))

    def _expanded_window(row, state):
        _rt_peak = row.get("rt_peak", np.nan)
        try:
            _rt_peak = float(_rt_peak) if _rt_peak is not None else np.nan
        except (TypeError, ValueError):
            _rt_peak = np.nan
        lo = min(state["rt_min"], _rt_peak) if not np.isnan(_rt_peak) else state["rt_min"]
        hi = max(state["rt_max"], _rt_peak) if not np.isnan(_rt_peak) else state["rt_max"]
        return float(lo) - 1, float(hi) + 1

    def _ms1_span(row, state):
        """RT span for which EIC lines are sent: default view ∪ expanded RT window,
        snapped outward to a coarse grid so small RT nudges keep the same span."""
        xw_lo, xw_hi = _default_plot_bounds_from_row(row)
        ex_lo, ex_hi = _expanded_window(row, state)
        lo = math.floor(min(xw_lo, ex_lo) / _SPAN_GRID) * _SPAN_GRID
        hi = math.ceil(max(xw_hi, ex_hi) / _SPAN_GRID) * _SPAN_GRID
        return [round(lo, 4), round(hi, 4)]

    def _resolve_isomers(compound_idx, row):
        """Isomer metadata for the compound (cached; RT bounds are read live)."""
        with _cache_lock:
            if compound_idx in isomer_string_cache:
                return isomer_string_cache[compound_idx]
        resolved_isomers = []
        try:
            isomers = json.loads(row.get("isomers", "[]"))
            for iso in isomers or []:
                iso_inchi = iso.get('inchi_key', '')
                iso_name = iso.get('label', '')
                iso_adduct = iso.get('adduct', '')
                mask = (
                    (manual_curation_df["inchi_key"] == iso_inchi) &
                    (manual_curation_df["label"] == iso_name) &
                    (manual_curation_df["adduct"] == iso_adduct)
                )
                isomer_match = manual_curation_df[mask]
                if len(isomer_match) > 1:
                    logger.warning(f"Multiple isomer matches for {iso_name} {iso_adduct}")
                if isomer_match.empty:
                    continue
                if "remove" in str(isomer_match.iloc[0]["ms1_notes"]).lower():
                    continue
                iso_df_idx = isomer_match.index[0]
                iso_pos_idx = manual_curation_df.index.get_loc(iso_df_idx)
                resolved_isomers.append({
                    "display_idx": iso_pos_idx + 1,  # 1-based for display
                    "name": iso_name,
                    "adduct": iso_adduct,
                    "rt": iso.get('rt', None),
                    "mz": iso.get('mz', None),
                    "df_idx": iso_df_idx,
                })
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Isomer detection failed with {exc}")
        with _cache_lock:
            isomer_string_cache[compound_idx] = resolved_isomers
        return resolved_isomers

    def _ms1_static_traces(mz_rt_uid, highlighted, span):
        """EIC color-group traces + one merged hover/click trace (LRU cached)."""
        key = (mz_rt_uid, tuple(highlighted), tuple(span))
        cached = _lru_get(_ms1_static_cache, key)
        if cached is not None:
            return cached
        prep = _ms1_prepared(mz_rt_uid)
        lo, hi = span
        highlighted_set = set(highlighted)
        nan = np.array([np.nan])
        color_groups: OrderedDict[str, list] = OrderedDict()
        highlight_groups: OrderedDict[str, list] = OrderedDict()
        hx, hy, hcd, htext = [], [], [], []
        n_files = 0
        for i, f in enumerate(prep["files"]):
            m = (f["rt"] >= lo) & (f["rt"] <= hi)
            frt, fint = f["rt"][m], f["int"][m]
            if frt.size == 0:
                continue
            n_files += 1
            target = highlight_groups if f["fn"] in highlighted_set else color_groups
            g = target.setdefault(f["color"], [[], []])
            g[0].extend((frt, nan))
            g[1].extend((fint, nan))
            # Hover/click points only where this file has signal (>= 5% of its
            # max in the span): baseline points of hundreds of overlapping files
            # cannot identify a file anyway, and per-point labels are costly.
            finite = np.isfinite(fint)
            fmax = fint[finite].max() if finite.any() else 0.0
            sig = finite & (fint >= 0.05 * fmax)
            if not sig.any():
                continue
            hrt, hint = frt[sig], fint[sig]
            step = max(1, hrt.size // _MAX_HOVER_PTS)
            px, py = hrt[::step], hint[::step]
            hx.extend((px, nan))
            hy.extend((py, nan))
            hcd.extend((np.full(px.size, i, dtype=np.int32), np.array([-1], dtype=np.int32)))
            htext.extend([f["short"]] * px.size + [""])

        traces = []
        if hx:
            traces.append({
                "type": "scattergl",
                "x": _f32(np.concatenate(hx), 4),
                "y": _f32(np.concatenate(hy), 0),
                "mode": "lines",
                "line": {"color": "gray", "width": 3},
                "opacity": 0.001,
                "customdata": np.concatenate(hcd),
                "text": htext,
                "hovertemplate": "File: %{text}<extra></extra>",
                "showlegend": False,
            })
        for color, (xs, ys) in color_groups.items():
            traces.append({
                "type": "scattergl", "x": _f32(np.concatenate(xs), 4), "y": _f32(np.concatenate(ys), 0),
                "mode": "lines", "line": {"color": color, "width": 1.5},
                "hoverinfo": "skip", "showlegend": False,
            })
        for color, (xs, ys) in highlight_groups.items():
            traces.append({
                "type": "scattergl", "x": _f32(np.concatenate(xs), 4), "y": _f32(np.concatenate(ys), 0),
                "mode": "lines", "line": {"color": color, "width": 5.0},
                "hoverinfo": "skip", "showlegend": False,
            })
        logger.debug(f"MS1 static layer: {n_files} files → {len(traces)} traces")
        return _lru_put(_ms1_static_cache, key, traces, maxsize=8)

    def _max_eic_trace(row):
        max_eic_rt = row.get("max_eic_rt", [])
        max_eic_intensity = row.get("max_eic_intensity", [])
        try:
            if max_eic_rt is None or max_eic_intensity is None:
                return None
            x = np.asarray(max_eic_rt, dtype=np.float64)
            y = np.asarray(max_eic_intensity, dtype=np.float64)
            if x.size == 0 or x.size != y.size:
                return None
            # Only points above 3% of max intensity
            keep = y > 0.03 * np.nanmax(y)
            if not keep.any():
                return None
            return {
                "type": "scatter", "x": _f32(x[keep], 4), "y": _f32(y[keep], 0), "mode": "lines",
                "name": "Max EIC (slider)", "line": {"color": "#0074D9", "width": 1, "dash": "dot"},
                "opacity": 1.0, "hoverinfo": "skip", "showlegend": False,
            }
        except Exception:
            return None

    def _ms1_rt_layer(row, state, yaxis_scale, resolved_isomers, prep):
        """Every MS1 element that depends on rt_min / rt_max."""
        rt_min, rt_max = float(state["rt_min"]), float(state["rt_max"])
        exp_lo, exp_hi = _expanded_window(row, state)
        y_max_data = _y_max_in_range(prep, exp_lo, exp_hi)
        y_upper = max(y_max_data * 1.1, 1.0)
        band = y_upper * 0.03
        if yaxis_scale == "log":
            log_min = max(y_max_data / 1e6, 1e-12)  # assume 6 orders of magnitude dynamic range
            y_range = [float(np.log10(log_min)), float(np.log10(y_upper))]
        else:
            y_range = [-band, y_upper]

        def _overlaps(a_min, a_max, b_min, b_max):
            return (a_min <= b_max) and (b_min <= a_max)

        iso_bounds = []
        for iso in resolved_isomers:
            try:
                iso_row = manual_curation_df.loc[iso["df_idx"]]
                iso_bounds.append((float(iso_row["rt_min"]), float(iso_row["rt_max"])))
            except Exception:
                iso_bounds.append(None)

        iso_traces = []
        for i, b in enumerate(iso_bounds):
            if b is None:
                # Placeholder keeps trace indices stable for Patch updates.
                iso_traces.append({"type": "scatter", "x": [], "y": [], "mode": "lines",
                                   "showlegend": False, "hoverinfo": "skip"})
                continue
            lo, hi = b
            overlap = _overlaps(lo, hi, rt_min, rt_max) or any(
                j != i and o is not None and _overlaps(lo, hi, o[0], o[1]) for j, o in enumerate(iso_bounds)
            )
            iso_traces.append({
                "type": "scatter",
                "x": [lo, lo, hi, hi, lo],
                "y": [0.0, y_upper, y_upper, 0.0, 0.0],
                "mode": "lines",
                "fill": "toself",
                "fillcolor": "rgba(255,96,96,0.28)" if overlap else "rgba(150,205,255,0.28)",
                "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                "showlegend": False,
                "hoverinfo": "skip",
            })

        def _vline(x, color, dash=None):
            line = {"color": color, "width": 3.5}
            if dash:
                line["dash"] = dash
            return {"type": "scatter", "x": [x, x], "y": [0.0, y_upper], "mode": "lines", "opacity": 0.75,
                    "line": line, "showlegend": False, "hoverinfo": "skip"}

        line_traces = [_vline(float(row["atlas_rt_peak"]), "purple")]
        if pd.notnull(row.get("suggested_rt_min")):
            line_traces.append(_vline(float(row["suggested_rt_min"]), "orange"))
        if pd.notnull(row.get("suggested_rt_max")):
            line_traces.append(_vline(float(row["suggested_rt_max"]), "orange", dash="dash"))

        # MS2 scan RT markers: top-100 by score within the expanded window
        all_markers = ms2_markers_by_compound.get(row["mz_rt_uid"], [])
        marker_x = [rt for rt, _ in all_markers if exp_lo <= rt <= exp_hi][:100]
        marker_trace = {
            "type": "scatter", "x": marker_x, "y": [-band / 2] * len(marker_x), "mode": "markers",
            "marker": {"color": "black", "size": 10, "symbol": "triangle-up"},
            "cliponaxis": False, "showlegend": False, "hoverinfo": "skip",
        }

        shapes = [
            {"type": "line", "x0": rt_min, "x1": rt_min, "y0": 0, "y1": 1, "xref": "x", "yref": "paper",
             "opacity": 0.75, "line": {"color": "green", "width": 9}, "name": "RT min", "editable": True},
            {"type": "line", "x0": rt_max, "x1": rt_max, "y0": 0, "y1": 1, "xref": "x", "yref": "paper",
             "opacity": 0.75, "line": {"color": "green", "width": 9, "dash": "dash"}, "name": "RT max", "editable": True},
        ]
        return {"iso": iso_traces, "lines": line_traces, "marker": marker_trace, "shapes": shapes, "y_range": y_range}

    def _ms1_static_count(row, state):
        """Number of traces between the isomer boxes and the vertical lines."""
        prep_traces = _ms1_static_traces(row["mz_rt_uid"], list(state.get("highlighted_files") or []), _ms1_span(row, state))
        return len(prep_traces) + (1 if _max_eic_trace(row) is not None else 0)

    def _isomer_annotations(resolved_isomers):
        lines = []
        for iso in resolved_isomers:
            try:
                rt_val, mz_val = float(iso["rt"]), float(iso["mz"])
            except (TypeError, ValueError):
                continue
            lines.append((rt_val, (f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})  |  "
                                  f"RT: {rt_val:.3f}  |  m/z: {mz_val:.4f}")))
        if not lines:
            return []
        lines.sort(key=lambda t: t[0])
        return [
            {"text": "<b>Isomers</b>", "xref": "paper", "yref": "paper", "x": 0.99, "y": 0.99,
             "xanchor": "right", "yanchor": "bottom", "showarrow": False,
             "font": {"size": 17}, "align": "right"},
            {"text": "<br>".join(t for _, t in lines), "xref": "paper", "yref": "paper", "x": 0.99, "y": 0.99,
             "xanchor": "right", "yanchor": "top", "showarrow": False, "font": {"size": 16}, "align": "right",
             "bgcolor": "rgba(255,255,255,0.7)", "bordercolor": "rgba(100,100,100,0.4)", "borderwidth": 1,
             "borderpad": 6},
        ]

    def _fmt(v, spec):
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return "—"

    def _make_ms1_figure(state, yaxis_scale="linear"):
        """Full MS1 figure (plain dict, no plotly validation)."""
        row = _compound_row(state["compound_idx"])
        adduct = row.get("adduct", "")
        x_window_min, x_window_max = _default_plot_bounds_from_row(row)
        mz_rt_uid = row["mz_rt_uid"]
        prep = _ms1_prepared(mz_rt_uid)

        if not prep["files"]:
            return {
                "data": [],
                "layout": {
                    "template": _TEMPLATE_JSON,
                    "title": {"text": f"No MS1 data available for {row.get('label', 'Unknown')} ({adduct})", "x": 0.5},
                    "xaxis": {"title": {"text": "Retention time (min)"}, "range": [x_window_min, x_window_max]},
                    "yaxis": {"title": {"text": "Intensity"}, "type": yaxis_scale, "range": [0, 1]},
                },
            }

        resolved_isomers = _resolve_isomers(state["compound_idx"], row)
        span = _ms1_span(row, state)
        static = _ms1_static_traces(mz_rt_uid, list(state.get("highlighted_files") or []), span)
        max_eic = _max_eic_trace(row)
        rt_layer = _ms1_rt_layer(row, state, yaxis_scale, resolved_isomers, prep)

        data = list(rt_layer["iso"])
        if max_eic is not None:
            data.append(max_eic)
        data.extend(static)
        data.extend(rt_layer["lines"])
        data.append(rt_layer["marker"])

        title_text = (
            f"<span style='font-size:1.15em'><b>[{state['compound_idx'] + 1}] {row['label']}</b>"
            f"  ·  {adduct}  ·  <span style='color:#616e7c'>{row.get('inchi_key', '')}</span></span><br>"
            f"Atlas RT {_fmt(row['atlas_rt_peak'], '.4f')}  |  AutoID RT {_fmt(row['rt_peak'], '.4f')}  |  "
            f"RT Δ {_fmt(row['rt_error'], '.3f')}<br>"
            f"Atlas m/z {_fmt(row['atlas_mz'], '.4f')}  |  AutoID m/z {_fmt(row['mz'], '.4f')}  |  "
            f"ppm Δ {_fmt(row['mz_error'], '.2f')}"
        )

        try:
            slider_min = max(0, float(row.get("atlas_rt_min", x_window_min)) - 5)
        except Exception:
            slider_min = 0
        try:
            slider_max = min(20, float(row.get("atlas_rt_max", x_window_max)) + 5)
        except Exception:
            slider_max = 20

        layout = {
            "template": _TEMPLATE_JSON,
            "title": {"text": title_text, "x": 0.5, "xanchor": "center", "font": {"size": 17}},
            "hovermode": "closest",
            "showlegend": False,
            "margin": {"l": 64, "r": 20, "t": 112, "b": 40},
            "dragmode": "zoom",
            "uirevision": str(state["compound_idx"]),
            "xaxis": {
                "rangeslider": {"visible": True, "range": [slider_min, slider_max], "thickness": 0.12,
                                "bgcolor": "#f5f7fa", "bordercolor": "#e4e7eb", "borderwidth": 1},
                "range": [x_window_min, x_window_max],
                "autorange": False,
                "title": {"text": "Retention time (min)"},
            },
            "yaxis": {
                "type": yaxis_scale,
                "range": rt_layer["y_range"],
                "fixedrange": False,
                "autorange": False,
                "title": {"text": "Intensity"},
            },
            "shapes": rt_layer["shapes"],
            "annotations": _isomer_annotations(resolved_isomers),
        }
        return {"data": data, "layout": layout}

    def _make_ms1_rt_patch(state, yaxis_scale="linear"):
        """Patch that turns an MS1 figure for the same compound/highlights/span/scale
        into the figure _make_ms1_figure(state) would build."""
        row = _compound_row(state["compound_idx"])
        prep = _ms1_prepared(row["mz_rt_uid"])
        resolved_isomers = _resolve_isomers(state["compound_idx"], row)
        rt_layer = _ms1_rt_layer(row, state, yaxis_scale, resolved_isomers, prep)
        n_iso = len(rt_layer["iso"])
        base = n_iso + _ms1_static_count(row, state)
        patch = Patch()
        for i, t in enumerate(rt_layer["iso"]):
            patch["data"][i] = t
        for j, t in enumerate(rt_layer["lines"]):
            patch["data"][base + j] = t
        patch["data"][base + len(rt_layer["lines"])] = rt_layer["marker"]
        patch["layout"]["shapes"] = rt_layer["shapes"]
        patch["layout"]["yaxis"]["range"] = rt_layer["y_range"]
        return patch

    # ------------------------------------------------------------------
    # MS2 figure
    # ------------------------------------------------------------------
    def _stick_traces(mz_vals, intensities, hover_label, colors=None, default_color="red", line_width_px=3):
        """MS2 peak sticks (one trace per color), NaN m/z values dropped."""
        mz_vals = np.asarray(mz_vals, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        if mz_vals.size == 0:
            return []
        if colors is not None and len(colors) == mz_vals.size:
            color_arr = np.array([c or default_color for c in colors], dtype=object)
        else:
            color_arr = np.full(mz_vals.size, default_color, dtype=object)
        keep = ~np.isnan(mz_vals)
        mz_vals, intensities, color_arr = mz_vals[keep], intensities[keep], color_arr[keep]
        traces = []
        for color in dict.fromkeys(color_arr.tolist()):
            sel = color_arr == color
            mz, it = mz_vals[sel], intensities[sel]
            n = mz.size
            x = np.empty(n * 3); y = np.empty(n * 3)
            x[0::3] = mz; x[1::3] = mz; x[2::3] = np.nan
            y[0::3] = 0.0; y[1::3] = it; y[2::3] = np.nan
            cd = np.empty(n * 3); cd[0::3] = it; cd[1::3] = it; cd[2::3] = np.nan
            traces.append({
                "type": "scatter", "x": x, "y": y, "customdata": cd, "mode": "lines",
                "line": {"color": color, "width": line_width_px}, "showlegend": False,
                "hovertemplate": f"m/z: %{{x:.4f}}<br>Int: %{{customdata:.2e}}<extra>{hover_label}</extra>",
            })
        return traces

    def _to_log_mirror(vals):
        v = np.asarray(vals, dtype=np.float64)
        out = np.full(v.shape, np.nan)
        finite = np.isfinite(v)
        out[finite & (v == 0.0)] = 0.0
        nz = finite & (v != 0.0)
        out[nz] = np.sign(v[nz]) * np.log10(np.abs(v[nz]))
        return out

    def _empty_figure(text):
        return {
            "data": [],
            "layout": {
                "template": _TEMPLATE_JSON,
                "margin": {"l": 64, "r": 20, "t": 80, "b": 40},
                "xaxis": {"visible": False}, "yaxis": {"visible": False},
                "annotations": [{"text": text, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                                 "showarrow": False, "font": {"size": 17, "color": "#7b8794"}}],
            },
        }

    def _make_ms2_figure(state, scans, yaxis_scale="linear"):
        """Build the MS2 mirror plot (plain dict).

        Parameters
        ----------
        scans:
            Pre-fetched DataFrame from _get_ms2_scans() — passed in by the
            caller so we never call _get_ms2_scans() twice per update cycle.
        """
        compound_idx = state["compound_idx"]
        row = _compound_row(compound_idx)

        if scans.empty:
            return _empty_figure(f"{row['label']} — no MS2 scans in the current RT window")

        actual_idx = max(0, min(int(state.get("ms2_idx", 0)), len(scans) - 1))
        scan = scans.iloc[actual_idx]

        ce_val = scan.get("collision_energy", None)
        try:
            ce_label = _format_collision_energy_label(float(ce_val)) if ce_val is not None else "MS2"
        except (TypeError, ValueError):
            ce_label = "MS2"

        use_log = (yaxis_scale == "log")
        data = []
        hits = scan.get('hits', [])
        hit = hits[0] if isinstance(hits, list) and hits else None  # best hit (pre-sorted)

        scale = 1.0
        stick_width_px = 3
        num_ref_fragments = 0
        num_matching_fragments = 0
        label_mz, label_y = [], []

        if hit:
            q_mz_raw, q_int_raw = hit['query_aligned']
            r_mz_raw, r_int_raw = hit['ref_aligned']
            q_mz = _sanitize_numeric_list(q_mz_raw)
            q_int = _sanitize_numeric_list(q_int_raw)
            r_mz = _sanitize_numeric_list(r_mz_raw)
            r_int = _sanitize_numeric_list(r_int_raw)
            frag_colors = hit['fragment_colors']

            q_max = float(np.nanmax(q_int)) if q_int.size and np.any(np.isfinite(q_int)) else 0.0
            r_max = float(np.nanmax(r_int)) if r_int.size and np.any(np.isfinite(r_int)) else 0.0
            scale = (q_max / r_max) if r_max > 0 else 1.0

            ref_y = np.where(np.isfinite(r_int), -r_int * scale, np.nan)
            plot_q_int = _to_log_mirror(q_int) if use_log else q_int
            plot_ref_y = _to_log_mirror(ref_y) if use_log else ref_y

            data += _stick_traces(q_mz, plot_q_int, "Query", colors=frag_colors, line_width_px=stick_width_px)
            data += _stick_traces(r_mz, plot_ref_y, "Reference", colors=frag_colors, line_width_px=stick_width_px)

            num_ref_fragments = hit.get('ref_frags', 0)
            num_matching_fragments = len(hit.get('matched_fragments', []))
            label_mz = np.concatenate([q_mz, r_mz])
            label_y = np.concatenate([plot_q_int, plot_ref_y])
        else:
            mz = _sanitize_numeric_list(scan.get('frag_mzs', []))
            ints = _sanitize_numeric_list(scan.get('frag_ints', []))
            plot_ints = _to_log_mirror(ints) if use_log else ints
            data += _stick_traces(mz, plot_ints, "MS2", line_width_px=stick_width_px)
            label_mz, label_y = mz, plot_ints

        label_mz = np.asarray(label_mz, dtype=np.float64)
        label_y = np.asarray(label_y, dtype=np.float64)
        finite_y = label_y[np.isfinite(label_y)]
        y_min = float(finite_y.min()) if finite_y.size else 0.0
        y_max = float(finite_y.max()) if finite_y.size else 0.0
        y_span = max(y_max - y_min, max(abs(y_min), abs(y_max)), 1.0)
        label_pad = y_pad = text_offset = y_span * 0.01
        label_y_ceiling = y_max * 0.90 if y_max > 0 else 0.0
        label_y_floor = y_min * 0.90 if y_min < 0 else 0.0

        # Top-7 peaks by |intensity| get m/z labels (staggered when close together)
        abs_y = np.where(np.isfinite(label_y), np.abs(label_y), 0.0)
        top_idx = np.argsort(-abs_y, kind="stable")[:7]
        top = sorted(((label_mz[i], label_y[i]) for i in top_idx
                      if not np.isnan(label_mz[i]) and not np.isnan(label_y[i])), key=lambda t: t[0])
        annotations = []
        prev_mz, stagger = None, 0
        for mz_val, y_val in top:
            stagger = stagger + 1 if prev_mz is not None and abs(mz_val - prev_mz) < 5.0 else 0
            if y_val >= 0:
                y_base = y_val + label_pad + stagger * text_offset
                y_pos, yanchor = ((y_val - label_pad, "top") if y_base > label_y_ceiling else (y_base, "bottom"))
            else:
                y_base = y_val - label_pad - stagger * text_offset
                y_pos, yanchor = ((y_val + label_pad, "bottom") if y_base < label_y_floor else (y_base, "top"))
            annotations.append({"x": float(mz_val), "y": float(y_pos), "text": f"{mz_val:.4f}", "showarrow": False,
                                "xanchor": "center", "yanchor": yanchor, "font": {"size": 14}})
            prev_mz = mz_val

        yaxis_extra = {}
        if use_log:
            abs_max_log = max(abs(y_min), abs(y_max), 1.0)
            log_ticks_pos = [v for v in np.arange(0, abs_max_log + 1, 1.0) if v <= abs_max_log + 0.01]
            tick_vals = sorted(set([-t for t in log_ticks_pos if t > 0] + log_ticks_pos))
            tick_texts = []
            for tv in tick_vals:
                if tv == 0:
                    tick_texts.append("0")
                else:
                    orig = 10 ** abs(tv)
                    if orig >= 1e6:
                        tick_texts.append(f"{orig:.2e}")
                    elif orig >= 1000:
                        tick_texts.append(f"{int(orig):,}")
                    else:
                        tick_texts.append(f"{orig:.0f}")
            yaxis_extra = {"tickvals": [float(t) for t in tick_vals], "ticktext": tick_texts}

        fname = "_".join(os.path.basename(str(scan.get("filename", ""))).split(".")[0].split("_")[11:])
        if hit:
            scan_info = (
                f"<b>CoS {hit.get('score', 0):.4f}</b>  |  "
                f"Ions {num_matching_fragments}/{num_ref_fragments}  |  "
                f"RT {scan.get('scan_rt', 0):.4f} min  |  "
                f"Exp. m/z {scan.get('precursor_MZ', 0):.4f}  |  "
                f"Ref. m/z {hit.get('mz_theoretical', 0):.4f}  |  "
                f"<b>ppm Δ {hit.get('ppm_error', 0):.2f}</b>"
                f"<br><span style='color:#616e7c'>Hit: {hit.get('ref_name', 'Unknown')}  |  File: {fname}</span>"
            )
        else:
            scan_info = (
                f"<b>No hit</b>  |  RT {scan.get('scan_rt', 0):.4f} min  |  "
                f"Exp. m/z {scan.get('precursor_MZ', 0):.4f}"
                f"<br><span style='color:#616e7c'>File: {fname}</span>"
            )
        annotations.append({"text": scan_info, "xref": "x domain", "yref": "y domain", "x": 0.5, "y": 1.02,
                            "showarrow": False, "font": {"size": 16}, "xanchor": "center", "yanchor": "bottom"})

        scale_label = " [log₁₀]" if use_log else ""
        layout = {
            "template": _TEMPLATE_JSON,
            "barmode": "overlay",
            "hovermode": "closest",
            "margin": {"l": 64, "r": 20, "t": 96, "b": 40},
            "showlegend": False,
            "xaxis": {"title": {"text": f"m/z ({ce_label})"}},
            "yaxis": {"title": {"text": f"Intensity (ref scaled ×{scale:.2f}){scale_label}"},
                      "range": [y_min - y_pad, y_max + y_pad], "autorange": False, **yaxis_extra},
            "shapes": [{"type": "line", "xref": "x domain", "x0": 0, "x1": 1, "yref": "y", "y0": 0, "y1": 0,
                        "line": {"color": "black", "width": 1.5}}],
            "annotations": annotations,
        }
        return {"data": data, "layout": layout}

    logger.debug("App helpers defined successfully")

    # Build the complete hotkey set once at factory scope.
    _ALL_HOTKEYS = (
        set(notes_cfg["ms2_key_to_label"])
        | set(notes_cfg["ms1_key_to_label"])
        | set(notes_cfg["other_key_to_label"])
        | {"a", "s", "d", "f", "j", "k", "l", ";", ">", "/", "n", "m",
           "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}
    )

    def _nav_payload(new_idx):
        row = _compound_row(new_idx)
        return {"idx": new_idx, "ts": time.time(), "label": f"[{new_idx + 1}] {row['label']} ({row['adduct']})"}

    # all app callbacks that fire when GUI is interacted with
    def _flush_and_load_compound(old_state, new_idx, delta=None):
        """Single canonical path for all compound navigation.

        Flushes the current compound to the DB (async), then builds and returns
        the initial state for new_idx.  delta is only used for the force-eval
        warning check; pass None when navigating via dropdown (no direction).
        """
        old_state = dict(old_state)

        flush_error = None
        ms2_warning = None
        ms1_warning = None

        if delta is not None:
            ms2_warning, ms1_warning = _get_force_eval_and_warnings(old_state, delta)

        try:
            old_state = _flush_to_db(old_state)
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"_flush_and_load_compound: _flush_to_db failed: {exc}")
            flush_error = f"Save failed: {type(exc).__name__}: {exc}"

        new_state = _load_state(
            new_idx,
            session_id=old_state.get("session_id"),
            edit_seq=old_state.get("edit_seq", 0),
        )
        new_state["last_saved"] = old_state.get("last_saved")
        new_state["flush_error"] = flush_error

        if ms2_warning:
            new_state["ms2_warning"] = ms2_warning
        else:
            new_state.pop("ms2_warning", None)

        if ms1_warning:
            new_state["ms1_warning"] = ms1_warning
        else:
            new_state.pop("ms1_warning", None)

        return new_state

    # Clientside callback: show the loading overlay as soon as a navigation
    # result arrives (before the new figures are built/rendered).
    app.clientside_callback(
        """
        function(nav) {
            var el = document.getElementById('loading-overlay');
            if (el) { el.style.display = 'flex'; }
            var txt = document.getElementById('loading-overlay-text');
            if (txt) { txt.textContent = (nav && nav.label) ? ('Loading ' + nav.label + '…') : 'Loading compound…'; }
            return window.dash_clientside.no_update;
        }
        """,
        Output("loading-overlay", "style", allow_duplicate=True),
        Input("nav-trigger-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("nav-trigger-store", "data", allow_duplicate=True),
        Input("compound-dd", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def init_store(compound_idx, old_state):
        if compound_idx is None:
            raise dash.exceptions.PreventUpdate

        compound_idx = int(compound_idx)
        old_state = _ensure_valid_state(old_state)

        # Direct dropdown pick — guard against spurious re-fires.
        if int(old_state.get("compound_idx", -1)) == compound_idx:
            raise dash.exceptions.PreventUpdate

        new_state = _flush_and_load_compound(old_state, compound_idx, delta=None)
        return new_state, _nav_payload(compound_idx)

    def _get_force_eval_and_warnings(state, delta):
        """Return warning messages for required-evaluation navigation logic."""
        ms2_warning = None
        ms1_warning = None
        if (
            delta == 1
            and force_eval
            and "remove" not in state.get("ms1_note", notes_cfg["ms1_notes"][0]).lower()
            and should_require_note_selection(state.get("ms2_note", notes_cfg["ms2_notes"][0]), notes_cfg["ms2_notes"])
        ):
            ms2_warning = "Please select MS2 quality note before proceeding"
        if (
            delta == 1
            and force_eval
            and "remove" not in state.get("ms1_note", notes_cfg["ms1_notes"][0]).lower()
            and should_require_note_selection(state.get("ms1_note", notes_cfg["ms1_notes"][0]), notes_cfg["ms1_notes"])
        ):
            ms1_warning = "Please select MS1 quality note before proceeding"
        return ms2_warning, ms1_warning

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("compound-dd", "value", allow_duplicate=True),
        Output("nav-trigger-store", "data", allow_duplicate=True),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_compound(prev, nxt, state):
        """Atomic navigation via Prev/Next buttons."""
        trigger = ctx.triggered_id
        if trigger not in ("prev-btn", "next-btn"):
            raise dash.exceptions.PreventUpdate

        state = _ensure_valid_state(state)
        delta = -1 if trigger == "prev-btn" else 1
        new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)

        if new_idx == int(state["compound_idx"]):
            raise dash.exceptions.PreventUpdate

        with _timed("navigate_compound"):
            new_state = _flush_and_load_compound(state, new_idx, delta=delta)
        return new_state, new_idx, _nav_payload(new_idx)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms2-prev-1", "n_clicks"),
        Input("ms2-next-1", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_ms2(prev1, nxt1, state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered_id
        if trigger == "ms2-prev-1":
            return _move_ms2_idx(state, delta=-1)
        if trigger == "ms2-next-1":
            return _move_ms2_idx(state, delta=1)
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("accept-suggestions", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def accept_suggestions(_, state):
        state = _ensure_valid_state(state)
        row = _compound_row(state["compound_idx"])
        if pd.notnull(row.get("suggested_rt_min")):
            return _patch_rt_change(
                state,
                float(row["suggested_rt_min"]),
                float(row["suggested_rt_max"]),
            )
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("snap-to-isomer", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def snap_to_isomer(_, state):
        state = _ensure_valid_state(state)
        row = _compound_row(state["compound_idx"])
        bounds = _get_sorted_isomer_rt_bounds(row)
        if not bounds:
            raise dash.exceptions.PreventUpdate
        isomer_idx = state.get("isomer_snap_idx", 0) % len(bounds)
        rt_min, rt_max = bounds[isomer_idx]
        new_state = _patch_rt_change(state, rt_min, rt_max)
        new_state["isomer_snap_idx"] = (isomer_idx + 1) % len(bounds)
        return new_state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms1-radio", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_ms1_note(val, state):
        if state is None or val == state.get("ms1_note"):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, ms1_note=val)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms2-radio", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_ms2_note(val, state):
        if state is None or val == state.get("ms2_note"):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, ms2_note=val)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("other-checklist", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_other_note(vals, state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        filtered_vals = [v for v in vals if v in notes_cfg["other_notes"]] if vals else []
        if filtered_vals == state.get("other_note"):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, other_note=filtered_vals)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("analyst-notes", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_analyst_notes(txt, state):
        if state is None or txt is None or txt == state.get("analyst_notes"):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, analyst_notes=txt)


    # Only send relayoutData to the server after the user stops dragging a shape for x ms
    app.clientside_callback(
        """
        (function() {
            var _timer = null;
            var _pending = null;
            return function(relayoutData) {
                if (!relayoutData) return window.dash_clientside.no_update;
                var keys = Object.keys(relayoutData);
                var isShapeDrag = keys.some(function(k) {
                    return /shapes\\[\\d+\\]\\.x0/.test(k);
                });
                if (!isShapeDrag) {
                    // Zoom / pan / other — pass through immediately
                    return relayoutData;
                }
                // Shape drag — debounce: only forward after mouse stops moving
                _pending = relayoutData;
                clearTimeout(_timer);
                return new Promise(function(resolve) {
                    _timer = setTimeout(function() { resolve(_pending); }, 300);
                });
            };
        })()
        """,
        Output("ms1-graph", "relayoutData", allow_duplicate=True),
        Input("ms1-graph", "relayoutData"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms1-graph", "relayoutData"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def rt_drag(relayout, state):
        if not relayout or state is None:
            raise dash.exceptions.PreventUpdate

        rt_min_shape_idx = 0
        rt_max_shape_idx = 1

        new_min, new_max = state["rt_min"], state["rt_max"]
        updated = False
        y_moved = False
        for k, v in relayout.items():
            if k.startswith("shapes[") and (k.endswith("].y0") or k.endswith("].y1")):
                y_moved = True
                continue
            if not (k.startswith("shapes[") and k.endswith("].x0")):
                continue

            try:
                idx_str = k.split("[")[1].split("]")[0]
                shape_idx = int(idx_str)
            except (IndexError, ValueError):
                continue

            # Process the editable green lines
            if shape_idx == rt_min_shape_idx:
                new_min = float(v)
                updated = True
            elif shape_idx == rt_max_shape_idx:
                new_max = float(v)
                updated = True

        if not updated:
            # If the user drags vertically, force a redraw with current RTs to snap lines back.
            if y_moved:
                new_state = _patch_with_seq(state)
                new_state["force_ms1_redraw"] = int(state.get("force_ms1_redraw", 0)) + 1
                return new_state
            raise dash.exceptions.PreventUpdate

        rt_min_change = abs(new_min - state["rt_min"])
        rt_max_change = abs(new_max - state["rt_max"])
        if rt_min_change < 0.001 and rt_max_change < 0.001:
            raise dash.exceptions.PreventUpdate

        return _patch_rt_change(state, new_min, new_max)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ms1-graph", "clickData", allow_duplicate=True),
        Input("ms1-graph", "clickData"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_ms1_highlight(click_data, state):
        if click_data is None or state is None:
            raise dash.exceptions.PreventUpdate
        points = click_data.get("points", [])
        if not points:
            raise dash.exceptions.PreventUpdate
        raw_cd = points[0].get("customdata")
        if raw_cd is None:
            raise dash.exceptions.PreventUpdate
        if isinstance(raw_cd, (list, tuple)):
            raw_cd = raw_cd[0] if raw_cd else None
        # customdata on the merged hover trace is the file's index in the
        # compound's prepared file list (-1 on gap points).
        fp = None
        if isinstance(raw_cd, (int, float)) and not isinstance(raw_cd, bool):
            row = _compound_row(state["compound_idx"])
            files = _ms1_prepared(row["mz_rt_uid"])["files"]
            i = int(raw_cd)
            if 0 <= i < len(files):
                fp = files[i]["fn"]
        elif isinstance(raw_cd, str):
            fp = raw_cd
        if not fp:
            raise dash.exceptions.PreventUpdate
        highlighted = list(state.get("highlighted_files") or [])
        if fp in highlighted:
            highlighted.remove(fp)
        else:
            highlighted.append(fp)
        # Clear clickData so clicking the same point again still emits an event.
        return _patch_with_seq(state, highlighted_files=highlighted), None

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("compound-dd", "value", allow_duplicate=True),
        Output("nav-trigger-store", "data", allow_duplicate=True),
        Input("keyboard", "n_events"),
        State("keyboard", "event"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def handle_keyboard(n_events, event, state):
        if not event or state is None:
            raise dash.exceptions.PreventUpdate

        tag = (event.get("target.tagName") or "").upper()
        key = event.get("key", "")
        if not key:
            raise dash.exceptions.PreventUpdate

        if key not in _ALL_HOTKEYS:
            raise dash.exceptions.PreventUpdate

        if tag in ("TEXTAREA", "SELECT"):
            raise dash.exceptions.PreventUpdate

        try:
            return _handle_keyboard_inner(event, state, key)
        except dash.exceptions.PreventUpdate:
            raise
        except Exception as exc:
            traceback.print_exc()
            logger.error(
                f"handle_keyboard error (key={key}, compound={state.get('compound_idx')}, tag={tag}): {exc}"
            )
            raise dash.exceptions.PreventUpdate

    def _handle_keyboard_inner(event, state, key):
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        # --- RT nudge keys ---
        changed = False
        if key == "a":
            rt_min = round(rt_min - 0.05, 4); changed = True
        elif key == "s":
            rt_min = round(rt_min + 0.05, 4); changed = True
        elif key == "d":
            rt_max = round(rt_max - 0.05, 4); changed = True
        elif key == "f":
            rt_max = round(rt_max + 0.05, 4); changed = True

        if changed:
            logger.debug(f"[_handle_keyboard_inner] RT nudge key={key}, rt_min={rt_min}, rt_max={rt_max}")
            # No nav-trigger — RT changes must NOT show the loading overlay
            return _patch_rt_change(state, rt_min, rt_max), dash.no_update, dash.no_update

        # --- MS2 navigation keys ---
        if key in ("l", "ArrowUp"):
            return _move_ms2_idx(state, delta=-1), dash.no_update, dash.no_update
        if key in (";", "ArrowDown"):
            return _move_ms2_idx(state, delta=1), dash.no_update, dash.no_update

        # --- Accept suggestions ---
        if key == "n":
            row = _compound_row(state["compound_idx"])
            if pd.notnull(row.get("suggested_rt_min")):
                return _patch_rt_change(
                    state,
                    float(row["suggested_rt_min"]),
                    float(row["suggested_rt_max"]),
                ), dash.no_update, dash.no_update
            raise dash.exceptions.PreventUpdate

        # --- Snap to isomer ---
        if key == "m":
            row = _compound_row(state["compound_idx"])
            bounds = _get_sorted_isomer_rt_bounds(row)
            if not bounds:
                raise dash.exceptions.PreventUpdate
            isomer_idx = state.get("isomer_snap_idx", 0) % len(bounds)
            rt_min, rt_max = bounds[isomer_idx]
            new_state = _patch_rt_change(state, rt_min, rt_max)
            new_state["isomer_snap_idx"] = (isomer_idx + 1) % len(bounds)
            return new_state, dash.no_update, dash.no_update

        # --- Note hotkeys --- (must come before compound navigation so they are reachable)
        if key in notes_cfg["ms2_key_to_label"]:
            return _patch_with_seq(state, ms2_note=notes_cfg["ms2_key_to_label"][key]), dash.no_update, dash.no_update

        if key in notes_cfg["ms1_key_to_label"]:
            return _patch_with_seq(state, ms1_note=notes_cfg["ms1_key_to_label"][key]), dash.no_update, dash.no_update

        if key in notes_cfg["other_key_to_label"]:
            current = state.get("other_note")
            if not isinstance(current, list):
                current = []
            label = notes_cfg["other_key_to_label"][key]
            if label in current:
                current = [v for v in current if v != label]
            else:
                current = current + [label]
            return _patch_with_seq(state, other_note=current), dash.no_update, dash.no_update

        # --- Compound navigation — triggers loading overlay ---
        if key in ("j", "k", "ArrowLeft", "ArrowRight"):
            delta = 1 if key in ("k", "ArrowRight") else -1
            new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)
            if new_idx == int(state["compound_idx"]):
                raise dash.exceptions.PreventUpdate
            with _timed("navigate_compound (keyboard)"):
                new_state = _flush_and_load_compound(state, new_idx, delta=delta)
            return new_state, new_idx, _nav_payload(new_idx)

        raise dash.exceptions.PreventUpdate

    def _banners(flush_err, ms2_warning, ms1_warning):
        banners = []
        if flush_err:
            banners.append(dbc.Alert(str(flush_err), color="warning", dismissable=True, className="app-alert"))
        for w in (ms2_warning, ms1_warning):
            if w:
                banners.append(dbc.Alert(w, color="danger", className="app-alert app-alert-strong"))
        return banners

    @app.callback(
        Output("ms1-graph", "figure"),
        Output("error-banner", "children"),
        Output("ms1-render-key", "data"),
        Input("session-store", "data"),
        Input("yaxis-scale-radio", "value"),
        State("ms1-render-key", "data"),
        prevent_initial_call=False,
    )
    def update_ms1_figure(state, yaxis_scale, prev_render_key):
        state = _ensure_valid_state(state)
        yaxis_scale = yaxis_scale or "linear"

        # Only fields that affect the MS1 figure / banner.  Everything is a
        # JSON-native type (lists, not tuples) so the comparison with the
        # value round-tripped through dcc.Store is meaningful.
        row = _compound_row(state["compound_idx"])
        fig_key = {
            "compound_idx": state.get("compound_idx"),
            "highlighted_files": list(state.get("highlighted_files") or []),
            "yaxis_scale": yaxis_scale,
            "span": _ms1_span(row, state),
        }
        render_key = {
            "fig": fig_key,
            # force_ms1_redraw: bumped when a green line is dragged vertically so
            # the patch path re-sends the shapes and snaps them back.
            "rt": [state.get("rt_min"), state.get("rt_max"), state.get("force_ms1_redraw", 0)],
            "banner": [state.get("flush_error"), state.get("ms2_warning"), state.get("ms1_warning")],
        }
        if prev_render_key == render_key:
            raise dash.exceptions.PreventUpdate

        prev = prev_render_key or {}
        banner_out = (
            dash.no_update if prev.get("banner") == render_key["banner"]
            else (_banners(*render_key["banner"]) or "")
        )

        try:
            if prev.get("fig") == fig_key:
                if prev.get("rt") == render_key["rt"]:
                    fig_out = dash.no_update
                else:
                    with _timed("update_ms1_figure (patch)"):
                        fig_out = _make_ms1_rt_patch(state, yaxis_scale)
            else:
                with _timed("update_ms1_figure (full)"):
                    fig_out = _make_ms1_figure(state, yaxis_scale)
            return fig_out, banner_out, render_key
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"update_ms1_figure error: {exc}")
            err = dbc.Alert(f"MS1 figure error: {type(exc).__name__}: {exc}", color="danger", className="app-alert")
            # render key None forces a full redraw on the next update
            return _empty_figure("MS1 figure could not be drawn"), err, None

    # Hide the overlay after the MS1 figure has been delivered to the browser.
    app.clientside_callback(
        """
        function(figure) {
            var el = document.getElementById('loading-overlay');
            if (el) el.style.display = 'none';
            return window.dash_clientside.no_update;
        }
        """,
        Output("loading-overlay", "style", allow_duplicate=True),
        Input("ms1-graph", "figure"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("ms2-graph", "figure"),
        Output("ms2-render-key", "data"),
        Input("session-store", "data"),
        Input("ms2-yaxis-scale-radio", "value"),
        State("ms2-render-key", "data"),
        prevent_initial_call=False,
    )
    def update_ms2_figure(state, ms2_yaxis_scale, prev_ms2_render_key):
        state = _ensure_valid_state(state)

        # Memoized: shared with update_status for the same RT window.
        row = _compound_row(state["compound_idx"])
        scans = _get_ms2_scans(row, state["rt_min"], state["rt_max"])

        # Fingerprint of the scans in the window: a new top-N scan entering the
        # RT window changes this and forces a full MS2 redraw.
        scan_rt_fp = sorted(
            round(float(rt), 4)
            for rt in scans["scan_rt"]
            if not (isinstance(rt, float) and np.isnan(rt))
        ) if not scans.empty and "scan_rt" in scans.columns else []

        ms2_render_key = {
            "compound_idx": state.get("compound_idx"),
            "ms2_idx": state.get("ms2_idx"),
            "ms2_yaxis_scale": ms2_yaxis_scale,
            "scan_rt_fp": scan_rt_fp,
        }
        if prev_ms2_render_key == ms2_render_key:
            raise dash.exceptions.PreventUpdate

        try:
            with _timed("update_ms2_figure"):
                return _make_ms2_figure(state, scans, ms2_yaxis_scale or "linear"), ms2_render_key
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"update_ms2_figure error: {exc}")
            return _empty_figure("MS2 figure could not be drawn"), None

    def _dl_row(term, value):
        return [html.Dt(term), html.Dd(value)]

    @app.callback(
        Output("status-current", "children"),
        Output("compound-counter", "children"),
        Output("compound-progress", "value"),
        Output("ms2-counter-1", "children"),
        Output("ms2-progress-1", "value"),
        Output("save-indicator", "children"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_status(state):
        state = _ensure_valid_state(state)
        row = _compound_row(state["compound_idx"])
        n_total = max(len(compound_options), 1)
        current_1based = state["compound_idx"] + 1
        comp_txt = f"Compound {current_1based} / {n_total}"
        progress_val = round(current_1based / n_total * 100, 1)

        scans = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        n_scans = len(scans)
        ms2_idx = max(0, min(int(state.get("ms2_idx", 0)), max(n_scans - 1, 0)))
        ms2_txt = f"MS2 scan {ms2_idx + 1} / {n_scans}" if n_scans > 0 else "No MS2 scans"
        ms2_progress_val = round((ms2_idx + 1) / max(n_scans, 1) * 100, 1) if n_scans > 0 else 0

        other = state.get("other_note") or []
        analyst = str(state.get("analyst_notes") or "")
        selection = html.Dl(
            [
                *_dl_row("RT window", f"{state['rt_min']:.4f} – {state['rt_max']:.4f}"),
                *_dl_row("MS1", state["ms1_note"]),
                *_dl_row("MS2", state["ms2_note"]),
                *_dl_row("Other", ", ".join(other) if other else "—"),
                *_dl_row("Analyst", (analyst[:60] + ("…" if len(analyst) > 60 else "")) or "—"),
            ],
            className="selection-list",
        )

        # Header save indicator: last successful save (or a save error).
        s = state.get("last_saved")
        if state.get("flush_error"):
            indicator = html.Span([html.Span(className="dot dot-error"), "Save failed — see banner"],
                                  className="indicator indicator-error")
        elif s:
            indicator = html.Span([html.Span(className="dot dot-ok"), f"Saved [{s['index']}] {s['name']} at {s['timestamp']}"],
                                  className="indicator indicator-ok")
        else:
            indicator = ""

        return selection, comp_txt, progress_val, ms2_txt, ms2_progress_val, indicator

    @app.callback(
        Output("save-toast", "is_open"),
        Output("save-toast", "children"),
        Input("nav-trigger-store", "data"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def show_save_toast(nav_trigger, state):
        """Show the save-confirmation toast only when compound navigation just occurred
        (i.e., when nav-trigger-store fires), not on every session-store update."""
        if state is None:
            raise dash.exceptions.PreventUpdate
        s = state.get("last_saved")
        if not s:
            raise dash.exceptions.PreventUpdate
        other = s.get("other") or []
        analyst = str(s.get("analyst_notes") or "")
        toast_body = html.Div(
            [
                html.Div(f"[{s['index']}] {s['name']} ({s['adduct']})", className="toast-title"),
                html.Dl(
                    [
                        *_dl_row("RT", f"{s['rt_min']:.4f} – {s['rt_max']:.4f}"),
                        *_dl_row("MS1", s["ms1"]),
                        *_dl_row("MS2", s["ms2"]),
                        *_dl_row("Other", ", ".join(other) if other else "—"),
                        *_dl_row("Analyst", (analyst[:50] + ("…" if len(analyst) > 50 else "")) or "—"),
                    ],
                    className="selection-list compact",
                ),
                html.Div(f"Queued at {s['timestamp']}", className="toast-time"),
            ],
        )
        return True, toast_body

    @app.callback(
        Output("help-offcanvas", "is_open"),
        Input("help-btn", "n_clicks"),
        State("help-offcanvas", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_help(n_clicks, is_open):
        if n_clicks:
            return not is_open
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("guide-offcanvas", "is_open"),
        Input("guide-btn", "n_clicks"),
        State("guide-offcanvas", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_guide(n_clicks, is_open):
        if n_clicks:
            return not is_open
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("exit-modal", "is_open"),
        Input("exit-open-btn", "n_clicks"),
        Input("exit-cancel-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_exit_modal(_open, _cancel):
        return ctx.triggered_id == "exit-open-btn"

    @app.callback(
        Output("analyst-notes", "value"),
        Output("id-notes", "children"),
        Output("ms1-radio", "value"),
        Output("ms2-radio", "value"),
        Output("other-checklist", "value"),
        Input("session-store", "data"),
        State("analyst-notes", "value"),
        State("ms1-radio", "value"),
        State("ms2-radio", "value"),
        State("other-checklist", "value"),
        prevent_initial_call=False,  # Always fire, including on initial load
    )
    def sync_controls(state, cur_analyst, cur_ms1, cur_ms2, cur_other):
        """Push session-store values into the controls.  Values that already
        match are returned as no_update so the set_* callbacks are not
        re-triggered by an echo (saves a server round-trip per edit)."""
        state = _ensure_valid_state(state)
        ms2_val = state["ms2_note"] if state["ms2_note"] in notes_cfg["ms2_notes"] else notes_cfg["ms2_notes"][0]
        ms1_val = state["ms1_note"] if state["ms1_note"] in notes_cfg["ms1_notes"] else notes_cfg["ms1_notes"][0]
        other_val = [v for v in state["other_note"] if v in notes_cfg["other_notes"]] if isinstance(state["other_note"], list) else []
        analyst_notes = state.get("analyst_notes", "")
        id_notes = state.get("id_notes") or "No identification notes"

        def _nu(new, cur):
            return dash.no_update if new == cur else new

        return (
            _nu(analyst_notes, cur_analyst if cur_analyst is not None else ""),
            id_notes,
            _nu(ms1_val, cur_ms1),
            _nu(ms2_val, cur_ms2),
            _nu(other_val, list(cur_other or [])),
        )

    @app.callback(
        Output("save-exit-status", "children"),
        Output("save-exit-btn", "disabled"),
        Input("save-exit-btn", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def save_and_exit(n_clicks, state):
        """Flush the current compound to DB, then shut down the Dash server."""
        if not n_clicks or state is None:
            raise dash.exceptions.PreventUpdate
        try:
            _flush_to_db(state)
            logger.debug("Save and Exit.")
            msg = dbc.Alert("Analysis saved and app port closed. You can close this tab.", color="success",
                            className="mb-0")
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Save and Exit: flush failed for compound {state.get('compound_idx')}: {exc}")
            msg = dbc.Alert(f"Save failed: {type(exc).__name__}: {exc}.", color="danger", className="mb-0")

        if shutdown_holder is not None and shutdown_holder[0] is not None:
            threading.Timer(1.5, shutdown_holder[0]).start()

        return msg, True

    logger.debug("Callbacks registered")

    logger.debug("App setup complete")

    app.database_flush_errors = database_flush_errors
    app.active_flush_threads = _active_flush_threads
    return app
