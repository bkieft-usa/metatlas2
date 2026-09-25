import json, os, re, time, uuid, threading
import numpy as np, pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_bootstrap_components as dbc
from dash_extensions import EventListener
import traceback

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.create_curation_container as ccc
from metatlas2.note_options import (
    normalize_note_value,
    should_require_note_selection,
)
logger = lcf.get_logger("analysis_gui")

def build_dash_app(
    analysis_gui_obj,
    port=8050,
    shutdown_holder=None,
    run_parameters=None,
):
    logger.debug("Starting the app factory for the Analysis GUI...")

    # Set up basic GUI params
    manual_curation_df = analysis_gui_obj.experimental_data.curation_df
    if "atlas_rt_peak" in manual_curation_df.columns:
        manual_curation_df = manual_curation_df.sort_values("atlas_rt_peak").reset_index(drop=True)

    _gui_cfg = analysis_gui_obj.config.gui_config if analysis_gui_obj.config else {}
    _op = analysis_gui_obj.override_parameters or {}
    _resolved_cfg = {**_gui_cfg, **{k: v for k, v in _op.items() if v is not None}}

    top_n_hits = _resolved_cfg.get("gui_top_n_hits", 20)
    lcmsruns_color_map = _resolved_cfg.get("gui_lcmsruns_colors", {})
    force_eval = _resolved_cfg.get("gui_require_all_evaluated", False)
    async_flush_errors: dict = {}

    # Extract metadata for display
    project_shortname = analysis_gui_obj.project_name.split("_")[4]
    chrom = analysis_gui_obj.chromatography.upper()
    pol = analysis_gui_obj.polarity.upper()
    analysis_type = analysis_gui_obj.analysis_type.upper()
    analysis_name = analysis_gui_obj.analysis_name.upper()
    rta = analysis_gui_obj.rt_alignment_number
    tga = analysis_gui_obj.analysis_number

    # Set up all passing compounds as options for the dropdown
    compound_options = [
        {"label": f"{i+1}: {row['compound_name']} ({row['adduct']})", "value": i}
        for i, row in manual_curation_df.reset_index(drop=True).iterrows()
    ]

    # Create the app
    try:
        if os.getenv('METATLAS2_STANDALONE') == 'true':
            requests_prefix = "/"
        else:
            requests_prefix = f"{os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')}proxy/{port}/"
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix=requests_prefix,
            suppress_callback_exceptions=True,
        )
        app.title = f"{project_shortname} | {chrom} | {pol} | {analysis_type}-{analysis_name} | RTA{rta} | TGA{tga}"
        logger.debug("App built successfully")
        app.config.prevent_initial_callbacks = "initial_duplicate"
    except Exception as e:
        traceback.print_exc()
        logger.error(f"FAILED: {e}")

    # Set up some caching to help with race conditions
    flush_lock = threading.RLock()
    latest_flushed_seq_by_session = {}
    db_write_lock = threading.Lock()
    isomer_string_cache = {}

    # Pre-index DataFrames by mz_rt_uid for fast lookups
    #logger.info("Pre-indexing MS data by mz_rt_uid for fast lookups...")
    ms1_by_compound = {}
    ms2_by_compound = {}

    # Index MS1 data — pre-convert spec_rts/spec_ints list columns to numpy arrays once
    # at startup so _compute_y_max_in_range() and the trace-building loop never pay the
    # Python-list → ndarray conversion cost on every render.
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
    # This is done once at startup so _get_ms2_scans() never needs to re-sort hits per call.
    def _presort_hits(raw):
        if not isinstance(raw, list) or not raw:
            return raw
        try:
            return sorted(raw, key=lambda h: float(h.get("score", float("-inf"))) if isinstance(h, dict) else float("-inf"), reverse=True)
        except Exception:
            return raw

    for mz_rt_uid, group in analysis_gui_obj.experimental_data.ms2_df.groupby(["mz_rt_uid"], observed=True):
        if len(mz_rt_uid) == 1:
            mz_rt_uid = mz_rt_uid[0]
        if "hits" in group.columns:
            group = group.copy()
            group["hits"] = group["hits"].apply(_presort_hits)
        ms2_by_compound[mz_rt_uid] = group

    # Pre-build compact marker data for MS1 triangle markers.
    # Stores (scan_rt, best_score) tuples per compound, pre-sorted by score descending.
    # Pre-sorting at startup means per-drag filtering + top-100 slice requires no sort step.
    ms2_markers_by_compound = {}
    for uid, group_df in ms2_by_compound.items():
        markers = []
        for r in group_df.itertuples(index=False):
            rt = getattr(r, "scan_rt", None)
            if rt is None or (isinstance(rt, float) and np.isnan(rt)):
                continue
            hits = getattr(r, "hits", [])
            if hits and isinstance(hits[0], dict):
                score = hits[0].get("score", None)
                best_score = float(score) if isinstance(score, (int, float)) and np.isfinite(score) else float("-inf")
            else:
                best_score = float("-inf")
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
        ms2_note = normalize_note_value(row.get("ms2_notes"), analysis_gui_obj.notes["ms2_notes"])
        ms1_note = normalize_note_value(row.get("ms1_notes"), analysis_gui_obj.notes["ms1_notes"])
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
        other_note = [v for v in other_note if v in analysis_gui_obj.notes["other_notes"]]
        
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
        #logger.info(f"_patch_with_seq: edit_seq {old_seq} -> {new_state['edit_seq']}, changes={list(changes.keys())}")
        return new_state

    def _ensure_valid_state(state):
        if "compound_idx" in state:
            return state
        logger.warning(
            "Malformed session-store state (%s). Resetting to starting compound.",
            type(state).__name__,
        )
        return _load_state(starting_compound_idx)

    def _patch_rt_change(state, new_min, new_max):
        rt_min = max(0.0, min(new_min, new_max))
        rt_max = max(rt_min, new_max)
        logger.debug(f"[_patch_rt_change] Called with new_min={new_min}, new_max={new_max} -> rt_min={rt_min}, rt_max={rt_max}")
        new_state = _patch_with_seq(state, rt_min=round(rt_min, 4), rt_max=round(rt_max, 4), ms2_idx=0)
        logger.debug(f"[_patch_rt_change] Returning new_state with edit_seq={new_state.get('edit_seq')}, rt_min={new_state.get('rt_min')}, rt_max={new_state.get('rt_max')}")
        # Preserve y_max cache and the RT window bounds used to compute it.
        # The figure builder will update these after recomputing y_max.
        if "cached_y_max" in state:
            new_state["cached_y_max"] = state["cached_y_max"]
        if "cached_exp_rt_min" in state:
            new_state["cached_exp_rt_min"] = state["cached_exp_rt_min"]
        if "cached_exp_rt_max" in state:
            new_state["cached_exp_rt_max"] = state["cached_exp_rt_max"]
        return new_state

    def _find_starting_compound_idx():
        """Find the first compound with no ms2_notes set.
        
        Looks for compounds where ms2_notes is blank (empty string or NaN) and returns next index to start
        """

        blank_mask = manual_curation_df["ms2_notes"].isna() | (manual_curation_df["ms2_notes"] == "")
        blank_positions = manual_curation_df.index[blank_mask]

        if len(blank_positions) == 0:
            return 0

        first_blank_idx = blank_positions[0]
        return manual_curation_df.index.get_loc(first_blank_idx)

    use_starting_index_finder = True
    if use_starting_index_finder is True:
        starting_compound_idx = _find_starting_compound_idx()
        if starting_compound_idx > 0:
            logger.info(f"Resuming analysis at compound {starting_compound_idx+1}: "
                    f"{manual_curation_df.iloc[starting_compound_idx]['compound_name']}")
        else:
            logger.info("Starting new analysis at compound 1")
    else:
        starting_compound_idx = 0
        logger.info("Starting analysis at compound 1 (starting index finder disabled)")

    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp", "target.tagName"]}],
    )

    PX_PER_ROW = 52    # pixels per radio/checklist label row (label + margin)
    FIXED_CHROME = 340   # dropdown + textarea + id-notes + status divs + buttons + padding
    BUTTON_ROW_H = 48    # height of each button row between/below the graphs

    # Resolve user-supplied dimensions (inches → pixels at 96 dpi).
    # Priority: override_parameters (notebook cell) > config.gui_config > auto
    # Uses _resolved_cfg already built at factory scope above.
    _gui_width_in = _resolved_cfg.get("gui_width") or None
    _gui_height_in = _resolved_cfg.get("gui_height") or None
    DPI = 96  # standard screen DPI for CSS px conversion

    n_rows = (
        len(analysis_gui_obj.notes["ms1_notes"])
        + len(analysis_gui_obj.notes["ms2_notes"])
        + len(analysis_gui_obj.notes["other_notes"])
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

    def _fs(base_rem: float) -> str:
        """Return a scaled rem font-size string."""
        return f"{round(base_rem * _font_scale, 3)}rem"

    _OVERLAY_HIDDEN_STYLE = {
        "display": "none",
        "position": "fixed",
        "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.35)",
        "zIndex": 9999,
        "pointerEvents": "all",
    }
    _OVERLAY_VISIBLE_STYLE = {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "position": "fixed",
        "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.35)",
        "zIndex": 9999,
        "pointerEvents": "all",
    }

    app.layout = dbc.Container(
        [
            dcc.Store(id="session-store", storage_type="memory", data=_load_state(starting_compound_idx)),
            dcc.Store(id="controls-compound-idx", storage_type="memory", data=starting_compound_idx),
            dcc.Store(id="yaxis-scale-store", storage_type="memory", data="linear"),
            dcc.Store(id="ms2-yaxis-scale-store", storage_type="memory", data="linear"),
            dcc.Store(id="nav-trigger-store", storage_type="memory", data={"idx": starting_compound_idx, "ts": 0}),
            dcc.Store(id="ms1-render-key", storage_type="memory", data=None),
            dcc.Store(id="ms2-render-key", storage_type="memory", data=None),
            html.Div(
                id="loading-overlay",
                children=[
                    html.Div(
                        "Loading compound...",
                        style={
                            "color": "white",
                            "fontSize": "1.6rem",
                            "fontWeight": "bold",
                            "background": "rgba(0,0,0,0.55)",
                            "padding": "1.2rem 2.4rem",
                            "borderRadius": "0.5rem",
                            "letterSpacing": "0.04em",
                        },
                    )
                ],
                style=_OVERLAY_HIDDEN_STYLE,
            ),
            # Save-confirmation toast — slides in from bottom-right after each compound save
            dbc.Toast(
                id="save-toast",
                header="Saved",
                is_open=False,
                dismissable=True,
                duration=6000,
                icon="success",
                style={
                    "position": "fixed",
                    "bottom": "1.5rem",
                    "right": "1.5rem",
                    "zIndex": 10000,
                    "minWidth": "260px",
                },
            ),
            # Keyboard shortcut cheat-sheet panel
            dbc.Offcanvas(
                id="help-offcanvas",
                title="Keyboard Shortcuts",
                is_open=False,
                placement="end",
                style={"width": "420px"},
                children=[
                    html.H6("Compound Navigation", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([html.Td(html.Kbd("j"), className="text-center"), html.Td("Previous compound")]),
                                html.Tr([html.Td(html.Kbd("k"), className="text-center"), html.Td("Next compound")]),
                                html.Tr([html.Td(html.Kbd("←"), className="text-center"), html.Td("Previous compound")]),
                                html.Tr([html.Td(html.Kbd("→"), className="text-center"), html.Td("Next compound")]),
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("RT Bound Nudge", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([html.Td(html.Kbd("a"), className="text-center"), html.Td("RT min − 0.05")]),
                                html.Tr([html.Td(html.Kbd("s"), className="text-center"), html.Td("RT min + 0.05")]),
                                html.Tr([html.Td(html.Kbd("d"), className="text-center"), html.Td("RT max − 0.05")]),
                                html.Tr([html.Td(html.Kbd("f"), className="text-center"), html.Td("RT max + 0.05")]),
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("MS2 Scan Navigation", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([html.Td(html.Kbd("l"), className="text-center"), html.Td("Previous MS2 scan")]),
                                html.Tr([html.Td(html.Kbd(";"), className="text-center"), html.Td("Next MS2 scan")]),
                                html.Tr([html.Td(html.Kbd("↑"), className="text-center"), html.Td("Previous MS2 scan")]),
                                html.Tr([html.Td(html.Kbd("↓"), className="text-center"), html.Td("Next MS2 scan")]),
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("Other Actions", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([html.Td(html.Kbd("n"), className="text-center"), html.Td("Accept RT suggestions")]),
                                html.Tr([html.Td(html.Kbd("m"), className="text-center"), html.Td("Snap to isomer")]),
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("MS1 Quality", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Kbd(analysis_gui_obj.notes["ms1_hotkeys"].get(lbl, "—")), className="text-center"),
                                    html.Td(lbl),
                                ])
                                for lbl in analysis_gui_obj.notes["ms1_notes"]
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("MS2 Quality", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Kbd(analysis_gui_obj.notes["ms2_hotkeys"].get(lbl, "—")), className="text-center"),
                                    html.Td(lbl),
                                ])
                                for lbl in analysis_gui_obj.notes["ms2_notes"]
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                    html.H6("Other Notes", className="mt-2 mb-1 fw-bold"),
                    dbc.Table(
                        [
                            html.Tbody([
                                html.Tr([
                                    html.Td(html.Kbd(analysis_gui_obj.notes["other_hotkeys"].get(lbl, "—")), className="text-center"),
                                    html.Td(lbl),
                                ])
                                for lbl in analysis_gui_obj.notes["other_notes"]
                            ])
                        ],
                        bordered=True, size="sm", className="mb-3",
                    ),
                ],
            ),
            keyboard_listener,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                f"{project_shortname}  |  {chrom}  |  {pol}  |  {analysis_type}  |  RTA{rta}  |  TGA{tga}",
                                style={"fontSize": _fs(1.0), "fontWeight": "bold", "marginBottom": "0.5rem", "color": "#333"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(id="compound-dd", options=compound_options, value=starting_compound_idx, clearable=False, style={"width": "100%", "fontSize": _fs(1.5)}),
                                        width=11, className="mb-3",
                                    ),
                                ],
                            ),
                            dbc.Textarea(id="analyst-notes", placeholder="Analyst notes...", debounce=True, style={"width": "100%", "height": "40px"}, className="my-2"),
                            dbc.FormText(
                                id="id-notes",
                                style={
                                    "width": "100%",
                                    "height": "100px",
                                    "whiteSpace": "pre-line",
                                    "display": "block",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "0.25rem",
                                    "padding": "0.375rem 0.75rem",
                                    "fontSize": _fs(1.0),
                                },
                                className="my-2",
                                children="No identification notes"
                            ),
                            html.Div(
                                [
                                    html.Label("MS1 quality:", className="fw-bold", style={"fontSize": _fs(1.5)}),
                                    dcc.RadioItems(
                                        id="ms1-radio",
                                        options=[{"label": f"[{analysis_gui_obj.notes['ms1_hotkeys'].get(lbl, '')}] {lbl}", "value": lbl} for lbl in analysis_gui_obj.notes["ms1_notes"]],
                                        value=analysis_gui_obj.notes["ms1_notes"][0],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": _fs(1.5)},
                                        inputStyle={"margin-right": "6px", "transform": f"scale({round(1.5 * _font_scale, 3)})"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(
                                [
                                    html.Label("MS2 quality:", className="fw-bold", style={"fontSize": _fs(1.5)}),
                                    dcc.RadioItems(
                                        id="ms2-radio",
                                        options=[{"label": f"[{analysis_gui_obj.notes['ms2_hotkeys'].get(val, '')}] {val}", "value": val} for val in analysis_gui_obj.notes["ms2_notes"]],
                                        value=analysis_gui_obj.notes["ms2_notes"][0],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": _fs(1.5)},
                                        inputStyle={"margin-right": "6px", "transform": f"scale({round(1.5 * _font_scale, 3)})"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(
                                [
                                    html.Label("Other notes:", className="fw-bold", style={"fontSize": _fs(1.5)}),
                                    dcc.Checklist(
                                        id="other-checklist",
                                        options=[{"label": f"[{analysis_gui_obj.notes['other_hotkeys'].get(val, '')}] {val}", "value": val} for val in analysis_gui_obj.notes["other_notes"]],
                                        value=[],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": _fs(1.5)},
                                        inputStyle={"margin-right": "6px", "transform": f"scale({round(1.5 * _font_scale, 3)})"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(id="status-current", className="my-2", style={"fontSize": _fs(1.0)}),
                            html.Div(id="error-banner", className="my-2", style={"fontSize": _fs(1.0)}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Save and Exit",
                                            id="save-exit-btn",
                                            color="danger",
                                            size="sm",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                        width="auto",
                                        className="d-flex justify-content-start",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Hotkeys Help",
                                            id="help-btn",
                                            color="dark",
                                            size="sm",
                                            style={"marginTop": "0.5rem"},
                                            title="Keyboard shortcuts",
                                        ),
                                        width="auto",
                                        className="d-flex justify-content-center",
                                    ),
                                    dbc.Col(
                                        html.Div(id="save-exit-status", className="text-muted fst-italic text-end w-100"),
                                        className="d-flex align-items-center justify-content-end",
                                    ),
                                ],
                                className="mt-3 mb-1",
                                align="center",
                            ),
                        ],
                        width=3,
                        style={"fontSize": "1rem"},
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="ms1-graph",
                                config={
                                    "displayModeBar": True,
                                    "edits": {"shapePosition": True, "titleText": False},
                                    "doubleClick": True,
                                    #"modeBarButtonsToRemove": ["autoScale2d", "resetScale2d"],
                                },
                                style={"height": f"{str(ms1_height)}px"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "◀ Prev ID",
                                            id="prev-btn",
                                            color="primary",
                                            className="me-2 w-100",
                                            style={"fontSize": _fs(1.0)}),
                                            width=2),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="compound-counter",
                                                    className="fw-bold text-center mb-1",
                                                    style={"fontSize": _fs(0.85)},
                                                ),
                                                dbc.Progress(
                                                    id="compound-progress",
                                                    value=round((starting_compound_idx + 1) / max(len(compound_options), 1) * 100, 1),
                                                    style={"height": "8px"},
                                                    color="primary",
                                                    className="w-100",
                                                ),
                                            ],
                                            className="d-flex flex-column justify-content-center",
                                        ),
                                        width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Next ID ▶",
                                            id="next-btn",
                                            color="primary",
                                            className="ms-2 w-100",
                                            style={"fontSize": _fs(1.0)}),
                                            width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Accept Suggestions",
                                            id="accept-suggestions",
                                            color="warning",
                                            className="w-100",
                                            style={"fontSize": _fs(1.0)}),
                                            width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Snap to Isomer",
                                            id="snap-to-isomer",
                                            color="secondary",
                                            className="w-100",
                                            style={"fontSize": _fs(1.0)}),
                                            width=2),
                                    dbc.Col(
                                        dcc.RadioItems(
                                            id="yaxis-scale-radio",
                                            options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                                            value="linear",
                                            labelStyle={"display": "inline-block", "margin-right": "12px", "fontSize": _fs(1.0)},
                                            inputStyle={"margin-right": "6px", "transform": f"scale({round(1.3 * _font_scale, 3)})"},
                                            className="w-100",
                                        ),
                                        width=1,
                                        className="d-flex align-items-center",
                                    ),
                                ],
                                className="my-2 align-items-center",
                                style={"width": "100%"},
                                justify="start",
                            ),
                            dcc.Graph(
                                id="ms2-graph",
                                config={"displayModeBar": True},
                                style={"height": f"{str(ms2_height)}px"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀ Prev MS2", id="ms2-prev-1", className="me-2 w-100", style={"fontSize": _fs(1.0)}), width=2),
                                                    dbc.Col(
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    id="ms2-counter-1",
                                                                    className="fw-bold text-center mb-1",
                                                                    style={"fontSize": _fs(0.85)},
                                                                ),
                                                                dbc.Progress(
                                                                    id="ms2-progress-1",
                                                                    value=0,
                                                                    style={"height": "8px"},
                                                                    color="secondary",
                                                                    className="w-100",
                                                                ),
                                                            ],
                                                            className="d-flex flex-column justify-content-center",
                                                        ),
                                                        width=2),
                                                    dbc.Col(dbc.Button("Next MS2 ▶", id="ms2-next-1", className="ms-2 w-100", style={"fontSize": _fs(1.0)}), width=2),
                                    dbc.Col(
                                        dcc.RadioItems(
                                            id="ms2-yaxis-scale-radio",
                                            options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                                            value="linear",
                                            labelStyle={"display": "inline-block", "margin-right": "12px", "fontSize": _fs(1.0)},
                                            inputStyle={"margin-right": "6px", "transform": f"scale({round(1.3 * _font_scale, 3)})"},
                                            className="w-100",
                                        ),
                                        width=1,
                                        className="d-flex align-items-center",
                                    ),
                                ],
                                className="my-2 align-items-center",
                                style={"width": "100%"},
                                justify="start",
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="mt-1",
            ),
        ],
        id="root-container",
        fluid=False,
        style={
            "paddingTop": "0.5rem",
            "width": f"{app_width}px",
            "minWidth": f"{app_width}px",
            "maxWidth": f"{app_width}px",
        },
    )

    logger.debug("Layout constructed successfully")

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
            return []
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return out

    def _get_all_ms2_scans_in_window(row, rt_min=None, rt_max=None):
        """Return ALL MS2 scans within the RT window, with no score-sorting or
        top-N truncation.  Used exclusively for the MS1 plot markers so that
        every scan timepoint is shown regardless of hit score or count.
        """
        mz_rt_uid = row["mz_rt_uid"]
        ms2_sub = ms2_by_compound.get(mz_rt_uid, pd.DataFrame())
        if ms2_sub.empty:
            return pd.DataFrame()
        if rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["scan_rt"] >= rt_min) & (ms2_sub["scan_rt"] <= rt_max)]
        return ms2_sub

    def _get_ms2_scans(row, rt_min=None, rt_max=None):
        """Return a DataFrame of the top N MS2 scans across all collision energies,
        sorted by best hit score descending then scan_rt ascending.

        Hits lists are pre-sorted at startup, so this function only needs to read
        hits[0].score to rank scans — no per-call hit sorting required.

        Returns a single DataFrame (not grouped by CE) with at most top_n_hits rows.
        """
        mz_rt_uid = row["mz_rt_uid"]

        ms2_sub = ms2_by_compound.get(mz_rt_uid, pd.DataFrame())
        if not ms2_sub.empty and rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["scan_rt"] >= rt_min) & (ms2_sub["scan_rt"] <= rt_max)]

        if ms2_sub.empty:
            return pd.DataFrame()

        result = ms2_sub.copy()
        if "hits" in result.columns:
            # Hits are pre-sorted by score at startup — just read the first entry's score.
            result["_best_hit_score"] = result["hits"].apply(
                lambda h: float(h[0].get("score", float("-inf")))
                if isinstance(h, list) and h and isinstance(h[0], dict)
                else float("-inf")
            )
            result = result.sort_values(["_best_hit_score", "scan_rt"], ascending=[False, True])
            result = result.drop(columns=["_best_hit_score"])
        else:
            result = result.sort_values("scan_rt")

        return result.head(top_n_hits)

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
            if seq <= latest_seq:
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
                        "ms2_notes": normalize_note_value(state.get("ms2_note"), analysis_gui_obj.notes["ms2_notes"]),
                        "ms1_notes": normalize_note_value(state.get("ms1_note"), analysis_gui_obj.notes["ms1_notes"]),
                        "other_notes": " // ".join(state.get("other_note", [])),
                        "analyst_notes": state.get("analyst_notes", ""),
                        "identification_notes": state.get("id_notes", ""),
                    }
                else:
                    window_metrics = _compute_window_ms1_metrics(state)
                    if window_metrics is None:
                        updates = {}
                    else:
                        updates = {
                            "passed_curation": False if "remove" in state.get("ms1_note", "").lower() else True,
                            "mz": window_metrics["mz"],
                            "rt_min": state["rt_min"],
                            "rt_max": state["rt_max"],
                            "rt_peak": window_metrics["rt_peak"],
                            "rt_error": window_metrics["rt_error"],
                            "mz_error": window_metrics["mz_error"],
                            "ms2_notes": normalize_note_value(state.get("ms2_note"), analysis_gui_obj.notes["ms2_notes"]),
                            "ms1_notes": normalize_note_value(state.get("ms1_note"), analysis_gui_obj.notes["ms1_notes"]),
                            "other_notes": " // ".join(state.get("other_note", [])),
                            "analyst_notes": state.get("analyst_notes", ""),
                            "identification_notes": state.get("id_notes", ""),
                        }

                # DB write — serialize to prevent DuckDB CHECKPOINT race conditions.
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
        
        # Launch async worker thread - don't block UI!
        thread = threading.Thread(target=_async_flush_worker, daemon=True)
        thread.start()

        # Return immediately with optimistic state update
        state["last_saved"] = {
            "name": row["compound_name"],
            "adduct": row["adduct"],
            "index": state["compound_idx"]+1,
            "rt_min": state["rt_min"],
            "rt_max": state["rt_max"],
            "ms1": normalize_note_value(state.get("ms1_note"), analysis_gui_obj.notes["ms1_notes"]),
            "ms2": normalize_note_value(state.get("ms2_note"), analysis_gui_obj.notes["ms2_notes"]),
            "other": state.get("other_note", []),
            "analyst_notes": state.get("analyst_notes", ""),
            "id_notes": state.get("id_notes", ""),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        state["flush_error"] = None
        return state

    # main figures for ms data display
    def _make_ms1_figure(state, yaxis_scale="linear"):
        # lcmsruns_color_map is resolved once at factory scope from _resolved_cfg
        row = _compound_row(state["compound_idx"])
        compound_display_idx = state["compound_idx"]+1
        mz_rt_uid = row["mz_rt_uid"]
        adduct = row.get("adduct", "")
        inchi_key = row.get("inchi_key", "")
        rt_min, rt_max = state["rt_min"], state["rt_max"]
        x_window_min, x_window_max = _default_plot_bounds_from_row(row)

        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())

        if sub.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"No MS1 data available for {row.get('compound_name', 'Unknown')} ({adduct})",
                xaxis_title="Retention Time (min)",
                yaxis_title="Intensity",
                xaxis_range=[x_window_min, x_window_max],
                yaxis_type=yaxis_scale,
                yaxis_range=[0, 1],
            )
            return fig

        # Determine y_max as the highest intensity point of all files within the current window.
        # Uses an incremental cache: only re-scans newly-exposed slivers when the window grows,
        # and does a full rescan when the window shrinks (so y_max correctly decreases).
        y_min_positive_data = None
        _rt_peak = row.get("rt_peak", np.nan)
        try:
            _rt_peak = float(_rt_peak) if _rt_peak is not None and not (isinstance(_rt_peak, float) and np.isnan(_rt_peak)) else np.nan
        except (TypeError, ValueError):
            _rt_peak = np.nan
        _eic_rt_lo = min(state["rt_min"], _rt_peak) if not np.isnan(_rt_peak) else state["rt_min"]
        _eic_rt_hi = max(state["rt_max"], _rt_peak) if not np.isnan(_rt_peak) else state["rt_max"]
        expanded_rt_min = _eic_rt_lo - 1
        expanded_rt_max = _eic_rt_hi + 1

        def _compute_y_max_in_range(df, lo, hi):
            """Return the max intensity across all files within [lo, hi].
            spec_rts/spec_ints are pre-converted to numpy arrays at startup,
            so no list→ndarray conversion is needed here.
            """
            y_max = 1.0
            for r in df.itertuples(index=False):
                rt_arr = getattr(r, "spec_rts", None)
                int_arr = getattr(r, "spec_ints", None)
                if rt_arr is None or len(rt_arr) == 0:
                    continue
                mask = (rt_arr >= lo) & (rt_arr <= hi)
                if np.any(mask):
                    local_max = float(np.nanmax(int_arr[mask]))
                    if local_max > y_max:
                        y_max = local_max
            return y_max

        cached_y = state.get("cached_y_max")
        old_exp_min = state.get("cached_exp_rt_min")
        old_exp_max = state.get("cached_exp_rt_max")
        force_recalc = state.get("force_y_recalc", False)

        if force_recalc or cached_y is None or old_exp_min is None or old_exp_max is None:
            # Cold cache or forced — full scan of current window
            y_max_data = _compute_y_max_in_range(sub, expanded_rt_min, expanded_rt_max)
        elif expanded_rt_min > old_exp_min or expanded_rt_max < old_exp_max:
            # Window shrank — must full-rescan (can't know which file held the old max)
            y_max_data = _compute_y_max_in_range(sub, expanded_rt_min, expanded_rt_max)
        elif expanded_rt_min < old_exp_min or expanded_rt_max > old_exp_max:
            # Window grew — scan only the newly-exposed slivers, keep cached max
            y_max_data = cached_y
            if expanded_rt_min < old_exp_min:
                sliver_max = _compute_y_max_in_range(sub, expanded_rt_min, old_exp_min)
                y_max_data = max(y_max_data, sliver_max)
            if expanded_rt_max > old_exp_max:
                sliver_max = _compute_y_max_in_range(sub, old_exp_max, expanded_rt_max)
                y_max_data = max(y_max_data, sliver_max)
        else:
            # Window unchanged — use cache directly
            y_max_data = cached_y

        # Update cache with current window bounds and computed y_max
        state["cached_y_max"] = y_max_data
        state["cached_exp_rt_min"] = expanded_rt_min
        state["cached_exp_rt_max"] = expanded_rt_max
        if yaxis_scale == "log":
            y_min_positive_data = y_max_data / 1e6  # Assume 6 orders of magnitude dynamic range
        y_upper_bound = max(y_max_data * 1.1, 1.0)
        y_marker_band = y_upper_bound * 0.03
        if yaxis_scale == "log":
            log_min = max((y_min_positive_data or 1e-6), 1e-12)
            y_range = [np.log10(log_min), np.log10(y_upper_bound)]
        else:
            y_range = [-y_marker_band, y_upper_bound]

        # Initialize figure before adding traces
        fig = go.Figure()

        # Cache isomer metadata, draw rectangles separately
        compound_idx = state["compound_idx"]
        if compound_idx in isomer_string_cache:
            resolved_isomers = isomer_string_cache[compound_idx]
        else:
            resolved_isomers = []
            try:
                isomers = json.loads(row.get("isomers", "[]"))
                if isomers:
                    # Build resolved_isomers list (cache this part)
                    for iso in isomers:
                        iso_inchi = iso.get('inchi_key', '')
                        iso_name = iso.get('compound_name', '')
                        iso_adduct = iso.get('adduct', '')
                        iso_rt = iso.get('rt', None)
                        iso_mz = iso.get('mz', None)
                        mask = (
                            (manual_curation_df["inchi_key"] == iso_inchi) &
                            (manual_curation_df["compound_name"] == iso_name) &
                            (manual_curation_df["adduct"] == iso_adduct)
                        )
                        isomer_match = manual_curation_df[mask]
                        if len(isomer_match) > 1:
                            logger.warning(f"Multiple isomer matches for {iso_name} {iso_adduct}")
                        if isomer_match.empty:
                            continue
                        if "remove" in isomer_match.iloc[0]["ms1_notes"].lower():
                            continue
                        iso_df_idx = isomer_match.index[0]
                        iso_pos_idx = manual_curation_df.index.get_loc(iso_df_idx)
                        resolved_isomers.append({
                            "display_idx": iso_pos_idx + 1,  # 1-based for display
                            "name": iso_name,
                            "adduct": iso_adduct,
                            "rt": iso_rt,
                            "mz": iso_mz,
                            "df_idx": iso_df_idx,
                        })
            except Exception as exc:
                traceback.print_exc()
                logger.error(f"Isomer detection failed with {exc}")
            # Store in cache (metadata only, not rectangles)
            isomer_string_cache[compound_idx] = resolved_isomers
        
        isomer_lines = []
        if resolved_isomers:
            def _window_overlaps(a_min, a_max, b_min, b_max):
                return (a_min <= b_max) and (b_min <= a_max)
            current_rt_min = state["rt_min"]
            current_rt_max = state["rt_max"]
            for i, iso in enumerate(resolved_isomers):
                # Read RT bounds live from manual_curation_df so edits made to an
                # isomer are reflected immediately, even when metadata is cached.
                try:
                    iso_row = manual_curation_df.loc[iso["df_idx"]]
                    iso_rt_min = float(iso_row["rt_min"])
                    iso_rt_max = float(iso_row["rt_max"])
                except Exception:
                    continue

                overlaps = _window_overlaps(iso_rt_min, iso_rt_max, current_rt_min, current_rt_max)
                if not overlaps:
                    for j, other in enumerate(resolved_isomers):
                        if i == j:
                            continue
                        try:
                            other_row = manual_curation_df.loc[other["df_idx"]]
                            other_rt_min = float(other_row["rt_min"])
                            other_rt_max = float(other_row["rt_max"])
                        except Exception:
                            continue
                        if _window_overlaps(iso_rt_min, iso_rt_max, other_rt_min, other_rt_max):
                            overlaps = True
                            break
                fillcolor = "rgba(255,96,96,0.28)" if overlaps else "rgba(150,205,255,0.28)"
                fig.add_trace(go.Scatter(
                    x=[iso_rt_min, iso_rt_min, iso_rt_max, iso_rt_max, iso_rt_min],
                    y=[0.0, y_upper_bound, y_upper_bound, 0.0, 0.0],
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                rt_str = f"{iso['rt']:.3f}"
                mz_str = f"{iso['mz']:.4f}"
                # iso['display_idx'] is now 1-based
                isomer_lines.append(
                    f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})  |  "
                    f"RT: {rt_str}  |  m/z: {mz_str}"
                )

        # Add max EIC trace after isomer rectangles so it appears in the rangeslider
        max_eic_rt = row.get("max_eic_rt", [])
        max_eic_intensity = row.get("max_eic_intensity", [])
        if len(max_eic_rt) > 0 and len(max_eic_intensity) > 0 and len(max_eic_rt) == len(max_eic_intensity):
            # Filter to only points above 5% of max intensity
            try:
                max_int = max(max_eic_intensity)
                threshold = 0.03 * max_int
                filtered_points = [(x, y) for x, y in zip(max_eic_rt, max_eic_intensity) if y > threshold]
                if filtered_points:
                    filtered_rt, filtered_int = zip(*filtered_points)
                    fig.add_trace(go.Scatter(
                        x=filtered_rt,
                        y=filtered_int,
                        mode="lines",
                        name="Max EIC (slider)",
                        line=dict(color="#0074D9", width=3, dash="dot"),
                        opacity=1.0,
                        hoverinfo="skip",
                        showlegend=False
                    ))
            except Exception:
                pass
        
        # Now add MS1 data traces
        highlighted_files = state.get("highlighted_files") or []
        # expanded_rt_min/max already computed above (incorporates rt_peak)

        # Keep an invisible file-specific trace for hover/click events, then group other traces (for speed)
        color_groups: dict = {}
        highlight_groups: dict = {}

        ms1_file_count = 0
        for r in sub.itertuples(index=False):
            fn = getattr(r, "filename", "unknown")
            short_name = re.sub(r"_ms[12]_(?:neg|pos)$", "", "_".join(os.path.basename(fn).split(".")[0].split("_")[11:]))
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in fn.lower()), "gray")
            is_highlighted = fn in highlighted_files

            # spec_rts/spec_ints are pre-converted to numpy arrays at startup
            rt_arr = getattr(r, "spec_rts", None)
            int_arr = getattr(r, "spec_ints", None)

            if rt_arr is None or len(rt_arr) == 0:
                logger.warning(f"MS1 data for {fn} has no RT points; skipping trace")
                continue

            # Filter for RT window
            mask = (rt_arr >= expanded_rt_min) & (rt_arr <= expanded_rt_max)
            filt_rt = rt_arr[mask]
            filt_int = int_arr[mask]

            if len(filt_rt) == 0:
                continue

            ms1_file_count += 1

            _MAX_HOVER_PTS = 75
            _step = max(1, len(filt_rt) // _MAX_HOVER_PTS)
            hover_rt = filt_rt[::_step]
            hover_int = filt_int[::_step]
            fig.add_trace(go.Scattergl(
                x=hover_rt,
                y=hover_int,
                mode="lines",
                line=dict(color=color, width=3),
                opacity=0.001,
                customdata=[[fn]] * len(hover_rt),
                hovertemplate=f"File: {short_name}<extra></extra>",
                showlegend=False,
            ))

            filt_rt_list = filt_rt.tolist()
            filt_int_list = filt_int.tolist()
            filt_rt_list.append(None)
            filt_int_list.append(None)

            if is_highlighted:
                if color not in highlight_groups:
                    highlight_groups[color] = [[], []]
                highlight_groups[color][0].extend(filt_rt_list)
                highlight_groups[color][1].extend(filt_int_list)
            else:
                if color not in color_groups:
                    color_groups[color] = [[], []]
                color_groups[color][0].extend(filt_rt_list)
                color_groups[color][1].extend(filt_int_list)

        # --- Emit visible color-group traces (normal weight) ---
        for color, (xs, ys) in color_groups.items():
            fig.add_trace(go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ))

        # --- Emit visible highlight overlay traces (thick) ---
        for color, (xs, ys) in highlight_groups.items():
            fig.add_trace(go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=5.0),
                hoverinfo="skip",
                showlegend=False,
            ))

        logger.debug(
            f"MS1 traces: {ms1_file_count} files → "
            f"{len(color_groups)} color-group traces + "
            f"{len(highlight_groups)} highlight traces + "
            f"{ms1_file_count} invisible hover/click traces"
        )

        # Atlas RT peak line (black, static)
        fig.add_trace(go.Scatter(
            x=[row["atlas_rt_peak"], row["atlas_rt_peak"]],
            y=[0.0, y_upper_bound],
            mode="lines",
            line=dict(color="black", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Suggested RT lines (orange, static)
        if pd.notnull(row.get("suggested_rt_min")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_min"], row["suggested_rt_min"]],
                y=[0.0, y_upper_bound],
                mode="lines",
                line=dict(color="orange", width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ))
        if pd.notnull(row.get("suggested_rt_max")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_max"], row["suggested_rt_max"]],
                y=[0.0, y_upper_bound],
                mode="lines",
                line=dict(color="orange", width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ))

        # RT min (purple, solid, editable): always span full y-axis
        fig.add_shape(
            type="line", x0=rt_min, x1=rt_min, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="purple", width=7),
            name="RT min", editable=True,
        )

        # RT max (purple, dashed, editable): always span full y-axis
        fig.add_shape(
            type="line", x0=rt_max, x1=rt_max, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="purple", width=7, dash="dash"),
            name="RT max", editable=True,
        )

        ms1_title_text = (
            f"<span style='font-size:1.2em'><b>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi_key}</b></span><br>"
            f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  AutoID RT: {row['rt_peak']:.4f}  |  RT Δ: {row['rt_error']:.3f}<br>"
            f"Atlas m/z: {row['atlas_mz']:.4f}  |  AutoID M/Z: {row['mz']:.4f}  |  M/Z ppm Δ: {row['mz_error']:.2f}"
        )

        # Calculate rangeslider min/max based on atlas_rt_min and atlas_rt_max
        atlas_rt_min = row.get("atlas_rt_min", x_window_min)
        atlas_rt_max = row.get("atlas_rt_max", x_window_max)
        try:
            slider_min = max(0, float(atlas_rt_min) - 5)
        except Exception:
            slider_min = 0
        try:
            slider_max = min(20, float(atlas_rt_max) + 5)
        except Exception:
            slider_max = 20

        fig.update_layout(
            title=dict(text=ms1_title_text, x=0.5, xanchor="center", font=dict(size=18)),
            xaxis_title="RT",
            yaxis_title="Intensity",
            hovermode="closest",
            showlegend=False,
            margin=dict(l=50, r=20, t=100, b=40),
            dragmode="zoom",
            plot_bgcolor="white",
            uirevision=str(state["compound_idx"]),
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    range=[slider_min, slider_max],
                    thickness=0.12,
                ),
                showgrid=False,
                zeroline=False,
                range=[x_window_min, x_window_max],
                title_font=dict(size=18),
                tickfont=dict(size=15),
                autorange=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                type=yaxis_scale,
                range=y_range,
                fixedrange=False,
                title_font=dict(size=18),
                tickfont=dict(size=15),
                autorange=False,
            ),
        )

        # Isomer annotation box in top-right corner (50% transparent background)
        if isomer_lines:
            isomer_annotation_text = "<br>".join(isomer_lines)
            # Label above the box
            fig.add_annotation(
                text="<b>Isomers</b>",
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                xanchor="right", yanchor="bottom",
                showarrow=False,
                font=dict(size=18, color="black"),
                align="right",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                borderpad=2,
            )
            # Box with isomer details (slightly larger font)
            fig.add_annotation(
                text=isomer_annotation_text,
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=16, color="black"),
                align="right",
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="rgba(100,100,100,0.5)",
                borderwidth=1,
                borderpad=6,
            )

        # MS2 scan RT markers: top-100 by score
        all_markers = ms2_markers_by_compound.get(mz_rt_uid, [])
        in_window = [rt for rt, _ in all_markers if expanded_rt_min <= rt <= expanded_rt_max]
        ms2_marker_rts = in_window[:100]
        if ms2_marker_rts:
            marker_y = -y_marker_band / 2
            fig.add_trace(go.Scatter(
                x=ms2_marker_rts,
                y=[marker_y] * len(ms2_marker_rts),
                mode="markers",
                marker=dict(color="black", size=10, symbol="triangle-up"),
                cliponaxis=False,
                showlegend=False,
                hoverinfo="skip",
            ))

        return fig

    def _add_ms2_stick_traces(fig, mz_vals, intensities, hover_label, colors=None, default_color="red", line_width_px=3):
        """Add MS2 peak sticks with fixed pixel width. Handles NaNs by ignoring them."""
        if mz_vals is None or intensities is None:
            return

        def _as_float_or_nan(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        grouped = {}
        # Use provided colors if available, otherwise everything gets the default_color
        if colors is not None and len(colors) == len(mz_vals):
            for mz, intensity, color in zip(mz_vals, intensities, colors):
                mz = _as_float_or_nan(mz)
                intensity = _as_float_or_nan(intensity)
                if np.isnan(mz): continue # Skip NaNs in aligned data
                grouped.setdefault(color or default_color, []).append((mz, intensity))
        else:
            # Filter out NaNs for raw spectra plotting
            pairs = []
            for mz, i in zip(mz_vals, intensities):
                mz = _as_float_or_nan(mz)
                i = _as_float_or_nan(i)
                if np.isnan(mz):
                    continue
                pairs.append((mz, i))
            grouped[default_color] = pairs

        for color, pairs in grouped.items():
            x_vals, y_vals, custom_vals = [], [], []
            for mz, intensity in pairs:
                x_vals.extend([mz, mz, None])
                y_vals.extend([0.0, intensity, None])
                custom_vals.extend([intensity, intensity, None])

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals, customdata=custom_vals,
                    mode="lines",
                    line=dict(color=color, width=line_width_px),
                    showlegend=False,
                    hovertemplate=f"m/z: %{{x:.4f}}<br>Int: %{{customdata:.2e}}<extra>{hover_label}</extra>",
                ),
            )

    def _make_ms2_figure(state, scans, yaxis_scale="linear"):
        """Build the MS2 mirror plot.

        Parameters
        ----------
        scans:
            Pre-fetched DataFrame from _get_ms2_scans() — passed in by the
            caller so we never call _get_ms2_scans() twice per update cycle.
        """
        compound_idx = state["compound_idx"]
        row = _compound_row(compound_idx)

        if scans.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"{row['compound_name']} - No MS2 data",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(margin=dict(l=50, r=20, t=80, b=40), plot_bgcolor="white",
                               xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
            return fig

        actual_idx = max(0, min(int(state.get("ms2_idx", 0)), len(scans) - 1))
        scan = scans.iloc[actual_idx]

        # Collision energy label for x-axis title
        ce_val = scan.get("collision_energy", None)
        try:
            ce_label = _format_collision_energy_label(float(ce_val)) if ce_val is not None else "MS2"
        except (TypeError, ValueError):
            ce_label = "MS2"

        use_log = (yaxis_scale == "log")

        def _to_log_mirror(y_vals_raw):
            result = []
            for v in y_vals_raw:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    result.append(np.nan)
                elif v == 0.0:
                    result.append(0.0)
                else:
                    result.append(np.sign(v) * np.log10(abs(v)))
            return result

        fig = go.Figure()

        # DATA EXTRACTION FROM HITS LIST
        hits = scan.get('hits', [])
        hit = hits[0] if hits else None  # best hit of this scan (hits are pre-sorted by score)

        label_points = []
        scale = 1.0
        stick_width_px = 3
        num_ref_fragments = 0
        num_matching_fragments = 0

        if hit:
            # Extract pre-aligned arrays from the hit dictionary
            q_mz_raw, q_int_raw = hit['query_aligned']
            r_mz_raw, r_int_raw = hit['ref_aligned']
            q_mz = _sanitize_numeric_list(q_mz_raw)
            q_int = _sanitize_numeric_list(q_int_raw)
            r_mz = _sanitize_numeric_list(r_mz_raw)
            r_int = _sanitize_numeric_list(r_int_raw)
            frag_colors = hit['fragment_colors']

            # Handle scaling for the mirror plot
            q_max = np.nanmax(q_int) if q_int and np.any(np.isfinite(q_int)) else 0
            r_max = np.nanmax(r_int) if r_int and np.any(np.isfinite(r_int)) else 0
            scale = (q_max / r_max) if r_max > 0 else 1.0

            # Scale the reference intensities and invert for mirror
            ref_y = [(-i * scale) if np.isfinite(i) else np.nan for i in r_int]

            # Apply log transform if requested (after sign assignment)
            plot_q_int = _to_log_mirror(q_int) if use_log else q_int
            plot_ref_y = _to_log_mirror(ref_y) if use_log else ref_y

            # Plot Query (Top)
            _add_ms2_stick_traces(fig, q_mz, plot_q_int, "Query",
                                  colors=frag_colors, default_color="red", line_width_px=stick_width_px)

            # Plot Reference (Bottom)
            _add_ms2_stick_traces(fig, r_mz, plot_ref_y, "Reference",
                                  colors=frag_colors, default_color="red", line_width_px=stick_width_px)

            num_ref_fragments = hit.get('ref_frags', 0)
            num_matching_fragments = len(hit.get('matched_fragments', []))
            label_points.extend(zip(q_mz, plot_q_int))
            label_points.extend(zip(r_mz, plot_ref_y))
        else:
            # Fallback to raw spectrum if no hit is selected/available
            mz = _sanitize_numeric_list(scan.get('frag_mzs', []))
            ints = _sanitize_numeric_list(scan.get('frag_ints', []))
            plot_ints = _to_log_mirror(ints) if use_log else ints
            _add_ms2_stick_traces(fig, mz, plot_ints, "MS2", default_color="red", line_width_px=stick_width_px)
            label_points.extend(zip(mz, plot_ints))

        # ANNOTATION & STYLING
        fig.add_hline(y=0, line=dict(color="black", width=1.5))

        y_vals = [y for _, y in label_points if y is not None and not np.isnan(y)] or [0]
        y_min, y_max = min(y_vals), max(y_vals)
        y_span = max(y_max - y_min, max(abs(y_min), abs(y_max)), 1.0)
        label_pad, y_pad, TEXT_HEIGHT_OFFSET = y_span * 0.01, y_span * 0.01, y_span * 0.01

        LABEL_CEILING_FRAC = 0.90   # labels above 90 % of y_max get flipped downward
        LABEL_FLOOR_FRAC   = 0.90   # labels below 90 % of |y_min| get flipped upward
        label_y_ceiling =  y_max * LABEL_CEILING_FRAC if y_max > 0 else 0.0
        label_y_floor   =  y_min * LABEL_FLOOR_FRAC   if y_min < 0 else 0.0

        top_label_idxs = {idx for idx, _ in sorted(enumerate(label_points), key=lambda item: abs(item[1][1] if not np.isnan(item[1][1]) else 0), reverse=True)[:7]}
        top_labels_sorted = sorted([(idx, mz_val, y_val) for idx, (mz_val, y_val) in enumerate(label_points) if idx in top_label_idxs and not np.isnan(mz_val)], key=lambda item: item[1])

        prev_mz, stagger_level = None, 0
        for idx, mz_val, y_val in top_labels_sorted:
            if prev_mz is not None and abs(mz_val - prev_mz) < 5.0:
                stagger_level += 1
            else:
                stagger_level = 0

            if y_val >= 0:
                y_base = y_val + label_pad + stagger_level * TEXT_HEIGHT_OFFSET
                if y_base > label_y_ceiling:
                    # Bar tip is too close to the title — place label inside the bar
                    y_pos = y_val - label_pad
                    yanchor = "top"
                else:
                    y_pos = y_base
                    yanchor = "bottom"
            else:
                y_base = y_val - label_pad - stagger_level * TEXT_HEIGHT_OFFSET
                if y_base < label_y_floor:
                    # Bar tip is too close to the bottom — place label inside the bar
                    y_pos = y_val + label_pad
                    yanchor = "bottom"
                else:
                    y_pos = y_base
                    yanchor = "top"

            fig.add_annotation(x=mz_val, y=y_pos, text=f"{mz_val:.4f}", showarrow=False,
                               xanchor="center", yanchor=yanchor,
                               font=dict(size=12))
            prev_mz = mz_val

        # Build y-axis tick labels for log mode (show original intensity values)
        yaxis_extra = {}
        if use_log:
            # Generate symmetric tick positions in log space and label with original values
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
            yaxis_extra = dict(tickvals=tick_vals, ticktext=tick_texts)

        fig.update_xaxes(
            title_text=f"m/z ({ce_label})",
            showgrid=False,
            zeroline=False,
            title_font=dict(size=18),
            tickfont=dict(size=15),
        )
        scale_label = " [log₁₀]" if use_log else ""
        fig.update_yaxes(
            title_text=f"Intensity (Ref scaled x{scale:.2f}){scale_label}",
            showgrid=False,
            zeroline=False,
            range=[y_min - y_pad, y_max + y_pad],
            title_font=dict(size=18),
            tickfont=dict(size=15),
            autorange=False,
            **yaxis_extra,
        )

        fname = "_".join(os.path.basename(scan.get("filename", "")).split(".")[0].split("_")[11:])
        if hit:
            scan_info = (
                f"<span style='font-size:1.2em'>"
                f"<b>CoS: {hit.get('score', 0):.4f}</b>  |  "
                f"Ions: {num_matching_fragments}/{num_ref_fragments}  |  "
                f"RT: {scan.get('scan_rt', 0):.4f} min | "
                f"Exp. m/z: {scan.get('precursor_MZ', 0):.4f}  |  "
                f"Ref. m/z: {hit.get('mz_theoretical', 0):.4f}  |  "
                f"ppm Δ: {hit.get('ppm_error', 0):.2f}"
                f"<br>"
                f"Hit: {hit.get('ref_name', 'Unknown')}  |  File: {fname}</span><br><br>"
            )
        else:
            scan_info = (
                f"<span style='font-size:1.2em'>"
                f"<b>No Hit</b>  |  "
                f"RT: {scan.get('scan_rt', 0):.4f} min | "
                f"Exp. m/z: {scan.get('precursor_MZ', 0):.4f}  |  "
                f"<br>"
                f"File: {fname}</span><br><br>"
            )

        fig.add_annotation(text=scan_info, xref="x domain", yref="y domain",
                           x=0.5, y=1.02, showarrow=False, font=dict(size=14),
                           xanchor="center", yanchor="bottom")

        fig.update_layout(
            barmode="overlay",
            hovermode="closest",
            margin=dict(l=50, r=20, t=120, b=40),
            plot_bgcolor="white",
            showlegend=False,
        )
        return fig

    logger.debug("App helpers defined successfully")

    # Build the complete hotkey set once at factory scope.
    # Captured as a closure by handle_keyboard so it is never rebuilt per keydown.
    _ALL_HOTKEYS = (
        set(analysis_gui_obj.notes["ms2_key_to_label"])
        | set(analysis_gui_obj.notes["ms1_key_to_label"])
        | set(analysis_gui_obj.notes["other_key_to_label"])
        | {"a", "s", "d", "f", "j", "k", "l", ";", ">", "/", "n", "m",
           "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}
    )

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

    # Clientside callback: show overlay instantly when nav-trigger-store changes.
    # This fires in the browser with zero server round-trip latency.
    app.clientside_callback(
        """
        function(nav) {
            var el = document.getElementById('loading-overlay');
            if (el) {
                el.style.display = 'flex';
                el.style.alignItems = 'center';
                el.style.justifyContent = 'center';
            }
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
        return new_state, {"idx": compound_idx, "ts": time.time()}

    def _get_force_eval_and_warnings(state, delta):
        """Return warning messages for required-evaluation navigation logic."""
        # force_eval is resolved once at factory scope from _resolved_cfg
        ms2_warning = None
        ms1_warning = None
        if (
            delta == 1
            and force_eval
            and "remove" not in state.get("ms1_note", analysis_gui_obj.notes["ms1_notes"][0]).lower()
            and should_require_note_selection(state.get("ms2_note", analysis_gui_obj.notes["ms2_notes"][0]), analysis_gui_obj.notes["ms2_notes"])
        ):
            ms2_warning = "Please select MS2 quality note before proceeding"
        if (
            delta == 1
            and force_eval
            and "remove" not in state.get("ms1_note", analysis_gui_obj.notes["ms1_notes"][0]).lower()
            and should_require_note_selection(state.get("ms1_note", analysis_gui_obj.notes["ms1_notes"][0]), analysis_gui_obj.notes["ms1_notes"])
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

        new_state = _flush_and_load_compound(state, new_idx, delta=delta)
        return new_state, new_idx, {"idx": new_idx, "ts": time.time()}

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
        filtered_vals = [v for v in vals if v in analysis_gui_obj.notes["other_notes"]] if vals else []
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
                    return /shapes\[\\d+\]\\.x0/.test(k);
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

    # # don't allow vertical dragging of the purple RT lines — snap them back to full paper height
    # app.clientside_callback(
    #     """
    #     function(relayoutData, figure) {
    #         if (!relayoutData || !figure) return window.dash_clientside.no_update;
    #         var keys = Object.keys(relayoutData);
    #         var hasY = keys.some(function(k) {
    #             return /shapes\[\\d+\]\\.(y0|y1)/.test(k);
    #         });
    #         var hasX = keys.some(function(k) {
    #             return /shapes\[\\d+\]\\.x0/.test(k);
    #         });
    #         if (hasY && !hasX) {
    #             // Pure vertical drag — snap all shapes back to full paper height
    #             var fig = JSON.parse(JSON.stringify(figure));
    #             (fig.layout.shapes || []).forEach(function(s) {
    #                 s.y0 = 0;
    #                 s.y1 = 1;
    #             });
    #             return fig;
    #         }
    #         return window.dash_clientside.no_update;
    #     }
    #     """,
    #     Output("ms1-graph", "figure", allow_duplicate=True),
    #     Input("ms1-graph", "relayoutData"),
    #     State("ms1-graph", "figure"),
    #     prevent_initial_call=True,
    # )

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
        assert rt_min_shape_idx == 0 and rt_max_shape_idx == 1, "Shape order changed"
        
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
            
            # Process the editable purple lines
            if shape_idx == rt_min_shape_idx:
                new_min = float(v)
                updated = True
            elif shape_idx == rt_max_shape_idx:
                new_max = float(v)
                updated = True
        
        if not updated:
            # If the user drags vertically, force a redraw with current RTs to snap lines back.
            if y_moved:
                return _patch_with_seq(state)
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
        # customdata is [[fn]] per point — extract the filename from the inner list.
        raw_cd = points[0].get("customdata")
        if not raw_cd:
            raise dash.exceptions.PreventUpdate
        fp = raw_cd[0] if isinstance(raw_cd, (list, tuple)) else raw_cd
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
        if key in analysis_gui_obj.notes["ms2_key_to_label"]:
            return _patch_with_seq(state, ms2_note=analysis_gui_obj.notes["ms2_key_to_label"][key]), dash.no_update, dash.no_update

        if key in analysis_gui_obj.notes["ms1_key_to_label"]:
            return _patch_with_seq(state, ms1_note=analysis_gui_obj.notes["ms1_key_to_label"][key]), dash.no_update, dash.no_update

        if key in analysis_gui_obj.notes["other_key_to_label"]:
            current = state.get("other_note")
            if not isinstance(current, list):
                current = []
            label = analysis_gui_obj.notes["other_key_to_label"][key]
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
            new_state = _flush_and_load_compound(state, new_idx, delta=delta)
            return new_state, new_idx, {"idx": new_idx, "ts": time.time()}

        raise dash.exceptions.PreventUpdate

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

        flush_err = state.get("flush_error")
        ms2_warning = state.get("ms2_warning")
        ms1_warning = state.get("ms1_warning")

        # Only re-render when fields that actually affect the MS1 figure change.
        # Note/radio/checklist/analyst-notes changes do not affect the figure.
        render_key = {
            "compound_idx": state.get("compound_idx"),
            "rt_min": state.get("rt_min"),
            "rt_max": state.get("rt_max"),
            "highlighted_files": tuple(state.get("highlighted_files") or []),
            "yaxis_scale": yaxis_scale,
            "flush_error": flush_err,
            "ms2_warning": ms2_warning,
            "ms1_warning": ms1_warning,
        }
        if prev_render_key == render_key:
            raise dash.exceptions.PreventUpdate

        try:
            ms1_fig = _make_ms1_figure(state, yaxis_scale)

            banners = []
            if flush_err:
                banners.append(html.Div(
                    f"{flush_err}",
                    style={"color": "red", "fontSize": "14px", "fontWeight": "bold", "marginBottom": "8px"},
                ))
            if ms2_warning:
                banners.append(html.Div(
                    ms2_warning,
                    style={"color": "white", "backgroundColor": "#d32f2f", "fontSize": "20px", "fontWeight": "bold", "padding": "12px", "borderRadius": "6px", "textAlign": "center", "marginBottom": "8px"},
                ))
            if ms1_warning:
                banners.append(html.Div(
                    ms1_warning,
                    style={"color": "white", "backgroundColor": "#d32f2f", "fontSize": "20px", "fontWeight": "bold", "padding": "12px", "borderRadius": "6px", "textAlign": "center", "marginBottom": "8px"},
                ))
            banner = banners if banners else ""
            return ms1_fig, banner, render_key
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"update_ms1_figure error: {exc}")
            err_html = html.Span(
                f"MS1 figure error: {type(exc).__name__}: {exc}",
                style={"color": "red", "fontSize": "11px", "fontWeight": "bold"},
            )
            empty = go.Figure()
            empty.update_layout(margin=dict(l=50, r=20, t=40, b=40))
            return empty, err_html, render_key

    # Hide the overlay after the MS1 figure has been rendered to the browser.
    # This clientside callback fires when ms1-graph.figure changes, which only
    # happens after the server has finished computing and the browser has received
    # the new figure data — guaranteeing the overlay disappears at the right time.
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

        triggered = [t["prop_id"] for t in dash.callback_context.triggered]
        scale_changed = any("ms2-yaxis-scale-radio" in t for t in triggered)

        # Fetch scans once — used for both the fingerprint check and figure building.
        row = _compound_row(state["compound_idx"])
        scans = _get_ms2_scans(row, state["rt_min"], state["rt_max"])

        # Build a lightweight fingerprint: sorted list of scan RTs (rounded to 4 dp).
        # This captures whether the set of scans in the RT window changed without any
        # frozenset→JSON→frozenset round-trip through dcc.Store.
        scan_rt_fp = sorted(
            round(float(rt), 4)
            for rt in scans["scan_rt"]
            if not (isinstance(rt, float) and np.isnan(rt))
        ) if not scans.empty and "scan_rt" in scans.columns else []

        # The render key now embeds the scan-RT fingerprint directly, so a single
        # dict comparison handles both "nothing changed" and "RT nudge with no new scans".
        ms2_render_key = {
            "compound_idx": state.get("compound_idx"),
            "ms2_idx": state.get("ms2_idx"),
            "ms2_yaxis_scale": ms2_yaxis_scale,
            "scan_rt_fp": scan_rt_fp,
        }
        if not scale_changed and prev_ms2_render_key == ms2_render_key:
            raise dash.exceptions.PreventUpdate

        try:
            return _make_ms2_figure(state, scans, ms2_yaxis_scale or "linear"), ms2_render_key
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"update_ms2_figure error: {exc}")
            empty = go.Figure()
            empty.update_layout(margin=dict(l=50, r=20, t=40, b=40))
            return empty, ms2_render_key

    @app.callback(
        Output("status-current", "children"),
        Output("compound-counter", "children"),
        Output("compound-progress", "value"),
        Output("compound-progress", "label"),
        Output("ms2-counter-1", "children"),
        Output("ms2-progress-1", "value"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_status(state):
        state = _ensure_valid_state(state)
        row = _compound_row(state["compound_idx"])
        n_total = max(len(compound_options), 1)
        current_1based = state["compound_idx"] + 1
        comp_txt = f"{current_1based} / {n_total}"
        progress_val = round(current_1based / n_total * 100, 1)
        progress_label = ""

        scans = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        n_scans = len(scans)
        ms2_idx = max(0, min(int(state.get("ms2_idx", 0)), max(n_scans - 1, 0)))
        ms2_txt_1 = f"MS2: {ms2_idx + 1}/{n_scans}" if n_scans > 0 else "MS2: No scans"
        ms2_progress_val = round((ms2_idx + 1) / max(n_scans, 1) * 100, 1) if n_scans > 0 else 0

        pending = html.Span(
            [
                html.I("Pending changes: ", style={"color": "black"}),
                html.Br(), f"[{current_1based}] {row['compound_name']} ({row['adduct']})",
                html.Br(), f"RT [{state['rt_min']:.4f}, {state['rt_max']:.4f}]",
                html.Br(), f"MS1: {state['ms1_note']}",
                html.Br(), f"MS2: {state['ms2_note']}",
                html.Br(), f"Other: {state['other_note']}",
                html.Br(), f"Analyst Notes: {state['analyst_notes'][:50]}{'...' if len(state['analyst_notes']) > 50 else ''}",
            ],
            style={"color": "#b8860b", "fontSize": "16px", "fontWeight": "bold"},
        )

        return pending, comp_txt, progress_val, progress_label, ms2_txt_1, ms2_progress_val

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
        toast_body = html.Span(
            [
                f"[{s['index']}] {s['name']} ({s['adduct']})",
                html.Br(),
                f"RT [{s['rt_min']:.4f}, {s['rt_max']:.4f}]",
                html.Br(),
                f"MS1: {s['ms1']}",
                html.Br(),
                f"MS2: {s['ms2']}",
                html.Br(),
                f"Other: {s['other']}",
                html.Br(),
                f"Analyst Notes: {s['analyst_notes'][:50]}{'...' if len(s['analyst_notes']) > 50 else ''}",
                html.Br(),
                f"Timestamp: {s['timestamp']}",
            ],
            style={"fontSize": "0.85rem"},
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
        Output("analyst-notes", "value"),
        Output("id-notes", "children"),
        Output("ms1-radio", "value"),
        Output("ms2-radio", "value"),
        Output("other-checklist", "value"),
        Output("controls-compound-idx", "data"),
        Input("session-store", "data"),
        prevent_initial_call=False,  # Always fire, including on initial load
    )
    def sync_controls(state):
        state = _ensure_valid_state(state)
        ms2_val = state["ms2_note"] if state["ms2_note"] in analysis_gui_obj.notes["ms2_notes"] else analysis_gui_obj.notes["ms2_notes"][0]
        ms1_val = state["ms1_note"] if state["ms1_note"] in analysis_gui_obj.notes["ms1_notes"] else analysis_gui_obj.notes["ms1_notes"][0]
        other_val = [v for v in state["other_note"] if v in analysis_gui_obj.notes["other_notes"]] if isinstance(state["other_note"], list) else []
        analyst_notes = state.get("analyst_notes", "")
        id_notes = state.get("id_notes", "No identification notes")
        compound_idx = state.get("compound_idx", 0)
        return analyst_notes, id_notes, ms1_val, ms2_val, other_val, compound_idx

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
            logger.debug(f"Save and Exit.")
            msg = "Analysis saved and app port closed."
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Save and Exit: flush failed for compound {state.get('compound_idx')}: {exc}")
            msg = f"Save failed: {type(exc).__name__}: {exc}."

        if shutdown_holder is not None and shutdown_holder[0] is not None:
            threading.Timer(1.5, shutdown_holder[0]).start()

        return msg, True

    logger.debug("Callbacks registered")

    logger.debug("App setup complete")

    return app