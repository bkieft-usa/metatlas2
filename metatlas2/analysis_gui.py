import functools, json, os, re, time, uuid, threading
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_bootstrap_components as dbc
from dash_extensions import EventListener
import traceback
import json

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
from metatlas2.note_options import (
    normalize_note_value,
    should_require_note_selection,
)
logger = lcf.get_logger("analysis_gui")

def build_dash_app(
    analysis_gui_obj,
    port=8050,
    shutdown_holder=None
):
    logger.debug("Starting the app factory for the Analysis GUI...")

    # Set up basic GUI params
    manual_curation_df = analysis_gui_obj.experimental_data.curation_df
    #logger.info(f"Starting manual curation with {len(manual_curation_df)} compounds")

    # Set some params
    top_n_hits = analysis_gui_obj.workflow_params.get("gui_top_n_hits", 20)
    if analysis_gui_obj.override_parameters.get("gui_top_n_hits") is not None:
        top_n_hits = analysis_gui_obj.override_parameters["gui_top_n_hits"]
    async_flush_errors: dict = {}

    # Extract metadata for display
    project_shortname = analysis_gui_obj.project_name.split("_")[4]
    chrom = analysis_gui_obj.post_autoid_atlas_obj.chromatography
    pol = analysis_gui_obj.post_autoid_atlas_obj.polarity
    analysis_type = analysis_gui_obj.post_autoid_atlas_obj.analysis_type
    analysis_name = getattr(analysis_gui_obj.post_autoid_atlas_obj, 'analysis_name', 'default') or 'default'
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

    # Index MS1 data
    for mz_rt_uid, group in analysis_gui_obj.experimental_data.ms1_df.groupby(["mz_rt_uid"], observed=True):
        if isinstance(mz_rt_uid, tuple) and len(mz_rt_uid) == 1:
            mz_rt_uid = mz_rt_uid[0]
        ms1_by_compound[mz_rt_uid] = group

    # Index MS2 data
    for mz_rt_uid, group in analysis_gui_obj.experimental_data.ms2_df.groupby(["mz_rt_uid"], observed=True):
        if isinstance(mz_rt_uid, tuple) and len(mz_rt_uid) == 1:
            mz_rt_uid = mz_rt_uid[0]
        ms2_by_compound[mz_rt_uid] = group

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
        if other_notes_raw is None or other_notes_raw == "" or (isinstance(other_notes_raw, float) and np.isnan(other_notes_raw)):
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
            "ms2_idx_by_ce": {},
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
        if isinstance(state, dict) and "compound_idx" in state:
            if not isinstance(state.get("ms2_idx_by_ce"), dict):
                patched = dict(state)
                patched["ms2_idx_by_ce"] = {}
                return patched
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
        new_state = _patch_with_seq(state, rt_min=round(rt_min, 4), rt_max=round(rt_max, 4), ms2_idx=0, ms2_idx_by_ce={})
        logger.debug(f"[_patch_rt_change] Returning new_state with edit_seq={new_state.get('edit_seq')}, rt_min={new_state.get('rt_min')}, rt_max={new_state.get('rt_max')}")
        if "cached_y_max" in state:
            new_state["cached_y_max"] = state["cached_y_max"]
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

    use_starting_index_finder = False
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

    # format of the app itself
    all_notes_len = len(analysis_gui_obj.notes["ms1_notes"]) + len(analysis_gui_obj.notes["ms2_notes"]) + len(analysis_gui_obj.notes["other_notes"])
    total_plot_height = all_notes_len*70
    ms1_height = total_plot_height*0.6
    ms2_height = total_plot_height*0.4
    app.layout = dbc.Container(
        [
            dcc.Store(id="session-store", storage_type="memory", data=_load_state(starting_compound_idx)),
            dcc.Store(id="controls-compound-idx", storage_type="memory", data=starting_compound_idx),
            dcc.Store(id="yaxis-scale-store", storage_type="memory", data="linear"),
            keyboard_listener,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                f"{project_shortname}  |  {chrom}  |  {pol}  |  {analysis_type}  |  RTA{rta}  |  TGA{tga}",
                                style={"fontSize": "1rem", "fontWeight": "bold", "marginBottom": "0.5rem", "color": "#333"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(id="compound-dd", options=compound_options, value=starting_compound_idx, clearable=False, style={"width": "100%", "fontSize": "1.5rem"}),
                                        width=11, className="mb-3",
                                    ),
                                ],
                            ),
                            dbc.Textarea(id="analyst-notes", placeholder="Analyst notes …", debounce=True, style={"width": "100%", "height": "40px"}, className="my-2"),
                            dbc.FormText(
                                id="id-notes",
                                style={
                                    "width": "100%",
                                    "height": "40px",
                                    "whiteSpace": "pre-line",
                                    "display": "block",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "0.25rem",
                                    "padding": "0.375rem 0.75rem",
                                    "fontSize": "1rem"
                                },
                                className="my-2",
                                children="No identification notes"
                            ),
                            html.Div(
                                [
                                    html.Label("MS1 quality:", className="fw-bold", style={"fontSize": "1.5rem"}),
                                    dcc.RadioItems(
                                        id="ms1-radio",
                                        options=[{"label": f"[{analysis_gui_obj.notes['ms1_hotkeys'].get(lbl, '')}] {lbl}", "value": lbl} for lbl in analysis_gui_obj.notes["ms1_notes"]],
                                        value=analysis_gui_obj.notes["ms1_notes"][0],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": "1.5rem"},
                                        inputStyle={"margin-right": "6px", "transform": "scale(1.5)"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(
                                [
                                    html.Label("MS2 quality:", className="fw-bold", style={"fontSize": "1.5rem"}),
                                    dcc.RadioItems(
                                        id="ms2-radio",
                                        options=[{"label": f"[{analysis_gui_obj.notes['ms2_hotkeys'].get(val, '')}] {val}", "value": val} for val in analysis_gui_obj.notes["ms2_notes"]],
                                        value=analysis_gui_obj.notes["ms2_notes"][0],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": "1.5rem"},
                                        inputStyle={"margin-right": "6px", "transform": "scale(1.5)"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(
                                [
                                    html.Label("Other notes:", className="fw-bold", style={"fontSize": "1.5rem"}),
                                    dcc.Checklist(
                                        id="other-checklist",
                                        options=[{"label": f"[{analysis_gui_obj.notes['other_hotkeys'].get(val, '')}] {val}", "value": val} for val in analysis_gui_obj.notes["other_notes"]],
                                        value=[],
                                        labelStyle={"display": "block", "margin-bottom": "6px", "fontSize": "1.5rem"},
                                        inputStyle={"margin-right": "6px", "transform": "scale(1.5)"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(id="status-current", className="my-2", style={"fontSize": "1rem"}),
                            html.Div(id="status-previous", className="my-2", style={"fontSize": "1rem"}),
                            html.Div(id="error-banner", className="my-2", style={"fontSize": "1rem"}),
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
                                            "◀ Prev ID [j,<]", 
                                            id="prev-btn", 
                                            color="primary", 
                                            className="me-2 w-100", 
                                            style={"fontSize": "1rem"}), 
                                            width=2),
                                    dbc.Col(
                                        html.Div(
                                            id="compound-counter", 
                                            className="fw-bold text-center", 
                                            style={"fontSize": "1rem"}), 
                                            width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Next ID ▶  [k,>, ]", 
                                            id="next-btn", 
                                            color="primary", 
                                            className="ms-2 w-100", 
                                            style={"fontSize": "1rem"}), 
                                            width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Accept Suggestions  [n]", 
                                            id="accept-suggestions", 
                                            color="warning", 
                                            className="w-100", 
                                            style={"fontSize": "1rem"}), 
                                            width=2),
                                    dbc.Col(
                                        dbc.Button(
                                            "Snap to Isomer  [m]", 
                                            id="snap-to-isomer", 
                                            color="secondary",
                                            className="w-100", 
                                            style={"fontSize": "1rem"}), 
                                            width=2),
                                    dbc.Col(
                                        dcc.RadioItems(
                                            id="yaxis-scale-radio",
                                            options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
                                            value="linear",
                                            labelStyle={"display": "inline-block", "margin-right": "12px", "fontSize": "1rem"},
                                            inputStyle={"margin-right": "6px", "transform": "scale(1.3)"},
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
                                    dbc.Col(dbc.Button("◀ Prev MS2-CE102040  [l]", id="ms2-prev-1", className="me-2 w-100", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(html.Div(id="ms2-counter-1", className="fw-bold text-center", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(dbc.Button("Next MS2-CE102040 ▶  [;]", id="ms2-next-1", className="ms-2 w-100", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(dbc.Button("◀ Prev MS2-CE205060  [>]", id="ms2-prev-2", className="me-2 w-100", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(html.Div(id="ms2-counter-2", className="fw-bold text-center", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(dbc.Button("Next MS2-CE205060 ▶  [/]", id="ms2-next-2", className="ms-2 w-100", style={"fontSize": "1rem"}), width=2),
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
        fluid=True,
        style={"paddingTop": "0.5rem"},
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

    def _get_ms2_scans(row, rt_min=None, rt_max=None):
        """Return dictionary of sorted, capped scans DataFrames grouped by collision energy.

        Returns a dict: {collision_energy: DataFrame} with top N hits from each collision energy.
        """
        mz_rt_uid = row["mz_rt_uid"]

        ms2_sub = ms2_by_compound.get(mz_rt_uid, pd.DataFrame())
        if not ms2_sub.empty and rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["scan_rt"] >= rt_min) & (ms2_sub["scan_rt"] <= rt_max)]

        # Group by collision energy and get top N from each
        scans_by_energy = {}
        if ms2_sub.empty:
            return scans_by_energy
        if "collision_energy" not in ms2_sub.columns:
            logger.warning(
                "MS2 subset for %s is missing 'collision_energy'. Available columns: %s",
                mz_rt_uid,
                list(ms2_sub.columns),
            )
            return scans_by_energy

        def _score_hit_entry(hit):
            if not isinstance(hit, dict):
                return float("-inf")
            try:
                return float(hit.get("score", float("-inf")))
            except (TypeError, ValueError):
                return float("-inf")

        def _normalize_and_best_score(raw_hits):
            if not isinstance(raw_hits, list) or len(raw_hits) == 0:
                return [], float("-inf")
            sorted_hits = sorted(raw_hits, key=_score_hit_entry, reverse=True)
            return sorted_hits, _score_hit_entry(sorted_hits[0])

        for ce, group in ms2_sub.groupby("collision_energy"):
            group_sorted = group.copy()
            if "hits" in group_sorted.columns:
                normalized_hits_and_scores = group_sorted["hits"].apply(_normalize_and_best_score)
                group_sorted["hits"] = normalized_hits_and_scores.apply(lambda x: x[0])
                group_sorted["_best_hit_score"] = normalized_hits_and_scores.apply(lambda x: x[1])
                group_sorted = group_sorted.sort_values(["_best_hit_score", "scan_rt"], ascending=[False, True])
                group_sorted = group_sorted.drop(columns=["_best_hit_score"])
            else:
                group_sorted = group_sorted.sort_values("scan_rt")

            scans_by_energy[ce] = group_sorted.head(top_n_hits)
        return scans_by_energy

    def _ce_state_key(ce):
        try:
            return f"{float(ce):.6f}"
        except Exception:
            return str(ce)

    def _format_collision_energy_label(ce_float):
        if abs(ce_float - 23.333) < 0.01:
            return "CE102040"
        if abs(ce_float - 43.333) < 0.01:
            return "CE205060"
        return f"CE{int(round(ce_float))}"

    def _get_ms2_idx_for_ce(state, ce):
        idx_by_ce = state.get("ms2_idx_by_ce") if isinstance(state, dict) else None
        if not isinstance(idx_by_ce, dict):
            raise ValueError(f"ms2_idx_by_ce missing or invalid in state: {state}")
        raw_val = idx_by_ce.get(_ce_state_key(ce), 0)
        try:
            return max(0, int(raw_val))
        except Exception:
            return 0

    def _move_ms2_idx_for_ce_position(state, ce_position, delta):
        row = _compound_row(state["compound_idx"])
        scans_by_energy = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        collision_energies = sorted(scans_by_energy.keys())
        if ce_position < 0 or ce_position >= len(collision_energies):
            raise dash.exceptions.PreventUpdate
        ce = collision_energies[ce_position]
        scans = scans_by_energy.get(ce)
        n_scans = len(scans) if scans is not None else 0
        if n_scans <= 0:
            raise dash.exceptions.PreventUpdate
        current_idx = _get_ms2_idx_for_ce(state, ce)
        new_idx = max(0, min(current_idx + delta, n_scans - 1))
        if new_idx == current_idx:
            raise dash.exceptions.PreventUpdate
        idx_by_ce = dict(state.get("ms2_idx_by_ce") or {})
        idx_by_ce[_ce_state_key(ce)] = int(new_idx)
        changes = {"ms2_idx_by_ce": idx_by_ce}
        if ce_position == 0:
            changes["ms2_idx"] = int(new_idx)
        return _patch_with_seq(state, **changes)

    def _compute_window_ms1_metrics(state):
        row = _compound_row(state["compound_idx"])
        mz_rt_uid = row["mz_rt_uid"]
        rt_min = float(state["rt_min"])
        rt_max = float(state["rt_max"])
        
        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())
        if sub.empty:
            return None

        atlas_mz = row.get("atlas_mz", np.nan)
        atlas_rt_peak = row.get("atlas_rt_peak", np.nan)

        global_max_int = -np.inf
        best_point = None
        all_win_mzs = []
        all_win_ints = []
        all_win_rts = []

        for _, r in sub.iterrows():
            # Extract lists and convert to arrays
            rt_arr = np.asarray(r.get("spec_rts", []), dtype=np.float32)
            int_arr = np.asarray(r.get("spec_ints", []), dtype=np.float32)
            mz_arr = np.asarray(r.get("spec_mzs", []), dtype=np.float32)
            feat_mask = np.asarray(r.get("in_feature", []), dtype=bool)

            if len(rt_arr) == 0: continue

            # Apply feature mask AND RT window mask
            if len(feat_mask) != len(rt_arr):
                feat_mask = np.ones(len(rt_arr), dtype=bool)
                
            win_mask = (feat_mask) & (rt_arr >= rt_min) & (rt_arr <= rt_max)
            
            if not np.any(win_mask):
                continue

            # Collect data for window averages
            all_win_mzs.extend(mz_arr[win_mask])
            all_win_ints.extend(int_arr[win_mask])
            all_win_rts.extend(rt_arr[win_mask])

            # Find max in this file
            masked_ints = int_arr[win_mask]
            local_idx = np.argmax(masked_ints)
            local_max = masked_ints[local_idx]
            
            if local_max > global_max_int:
                global_max_int = local_max
                abs_idx = np.where(win_mask)[0][local_idx]
                best_point = {
                    "rt": rt_arr[abs_idx],
                    "mz": mz_arr[abs_idx],
                    "int": local_max,
                    "file": r.get("filename", "")
                }

        if best_point is None:
            return None

        # Compute centroid for the best file's window (matching legacy behavior)
        # This is a simplification: using the average of all points in the window
        win_mz_mean = float(np.mean(all_win_mzs))
        win_rt_peak = float(best_point["rt"])
        
        rt_err = win_rt_peak - atlas_rt_peak if pd.notnull(atlas_rt_peak) else float("nan")
        mz_err = (win_mz_mean - atlas_mz) / atlas_mz * 1e6 if pd.notnull(atlas_mz) and atlas_mz != 0 else float("nan")

        return {
            "rt_peak": win_rt_peak,
            "mz": win_mz_mean,
            "rt_error": rt_err,
            "mz_error": mz_err,
            "best_ms1_file": best_point["file"],
            "best_ms1_rt": win_rt_peak,
            "best_ms1_mz": float(np.mean(all_win_mzs)),
            "best_ms1_intensity": float(global_max_int),
            "best_ms1_ppm_error": mz_err,
            "best_ms1_rt_error": rt_err,
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
        # Async DB writes can finish after navigation, but the UI should still
        # show the latest local edits when loading related compounds/isomers.
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
                        "mz": row.get("mz", None),
                        "rt_min": state["rt_min"],
                        "rt_max": state["rt_max"],
                        "rt_peak": row.get("rt_peak", None),
                        "rt_error": row.get("rt_error", None),
                        "mz_error": row.get("mz_error", None),
                        "best_ms1_file": row.get("best_ms1_file", None),
                        "best_ms1_rt": row.get("best_ms1_rt", None),
                        "best_ms1_mz": row.get("best_ms1_mz", None),
                        "best_ms1_intensity": row.get("best_ms1_intensity", None),
                        "best_ms1_ppm_error": row.get("best_ms1_ppm_error", None),
                        "best_ms1_rt_error": row.get("best_ms1_rt_error", None),
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
                            "curated": True,
                            "mz": window_metrics["mz"],
                            "rt_min": state["rt_min"],
                            "rt_max": state["rt_max"],
                            "rt_peak": window_metrics["rt_peak"],
                            "rt_error": window_metrics["rt_error"],
                            "mz_error": window_metrics["mz_error"],
                            "best_ms1_file": window_metrics["best_ms1_file"],
                            "best_ms1_rt": window_metrics["best_ms1_rt"],
                            "best_ms1_mz": window_metrics["best_ms1_mz"],
                            "best_ms1_intensity": window_metrics["best_ms1_intensity"],
                            "best_ms1_ppm_error": window_metrics["best_ms1_ppm_error"],
                            "best_ms1_rt_error": window_metrics["best_ms1_rt_error"],
                            "ms2_notes": normalize_note_value(state.get("ms2_note"), analysis_gui_obj.notes["ms2_notes"]),
                            "ms1_notes": normalize_note_value(state.get("ms1_note"), analysis_gui_obj.notes["ms1_notes"]),
                            "other_notes": " // ".join(state.get("other_note", [])),
                            "analyst_notes": state.get("analyst_notes", ""),
                            "identification_notes": state.get("id_notes", ""),
                        }

                # DB write — serialize to prevent DuckDB CHECKPOINT race conditions.
                # Re-check sequence inside the lock so a faster thread that already
                # wrote a newer edit cannot be overwritten by this stale worker.
                with db_write_lock:
                    with flush_lock:
                        current_latest = latest_flushed_seq_by_session.get(flush_key, -1)
                    if seq < current_latest:
                        #logger.info(
                        #    f"Skipping stale flush for compound {state['compound_idx']} "
                        #    f"seq={seq} (current latest={current_latest})"
                        #)
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
        import time
        fig_start = time.time()
        
        y_bottom = 0.0

        lcmsruns_color_map = analysis_gui_obj.workflow_params.get("gui_lcmsruns_colors", {})
        if analysis_gui_obj.override_parameters['gui_lcmsruns_colors'] is not None:
            lcmsruns_color_map = analysis_gui_obj.override_parameters['gui_lcmsruns_colors']            
    
        row = _compound_row(state["compound_idx"])
        compound_display_idx = state["compound_idx"]+1
        mz_rt_uid = row["mz_rt_uid"]
        adduct = row.get("adduct", "")
        inchi_key = row.get("inchi_key", "")
        rt_min, rt_max = state["rt_min"], state["rt_max"]
        x_window_min, x_window_max = _default_plot_bounds_from_row(row)

        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())

        if sub.empty:

            ms1_title_text = (
                f"<span style='font-size:1.2em'>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi_key}</span><br>"
                f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: N/A  |  RT Δ: N/A<br>"
                f"Atlas m/z: {row['atlas_mz']:.4f}  |  Meas M/Z: N/A  |  M/Z ppm Δ: N/A<br>"
                f"<sub style='font-size:0.8em'>{isomer_str}</sub>"
            )
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

        # Determine y_max as the highest intensity point of all files within the current window
        y_min_positive_data = None
        max_eic_rt = row.get("max_eic_rt", [])
        max_eic_intensity = row.get("max_eic_intensity", [])
        # Use only EIC points within the current RT window
        if len(max_eic_rt) > 0 and len(max_eic_intensity) > 0 and len(max_eic_rt) == len(max_eic_intensity):
            expanded_rt_min = state["rt_min"] - 1
            expanded_rt_max = state["rt_max"] + 1
            filtered_intensity = [y for x, y in zip(max_eic_rt, max_eic_intensity) if expanded_rt_min <= x <= expanded_rt_max]
            if filtered_intensity:
                y_max_data = max(filtered_intensity)
            else:
                y_max_data = 0.0
        else:
            y_max_data = row.get("best_ms1_intensity", np.nan)
            if pd.isna(y_max_data):
                y_max_data = 0.0
            else:
                y_max_data = float(y_max_data)
            if not np.isfinite(y_max_data) or y_max_data <= 0:
                y_max_data = 1.0
        # Store in state for consistency across redraws when RT changes
        if "cached_y_max" not in state or state.get("force_y_recalc", False):
            state["cached_y_max"] = y_max_data
        else:
            y_max_data = state["cached_y_max"]
        if yaxis_scale == "log":
            y_min_positive_data = y_max_data / 1e6  # Assume 6 orders of magnitude dynamic range
        y_upper_bound = max(y_max_data * 1.1, 1.0)
        if yaxis_scale == "log":
            log_min = max((y_min_positive_data or 1e-6), 1e-12)
            y_range = [np.log10(log_min), np.log10(y_upper_bound)]
        else:
            y_range = [0.0, y_upper_bound]

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
        
        # PHASE 4 FIX: Draw isomer rectangles outside cache block (always draw)
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
                    y=[y_bottom, y_upper_bound, y_upper_bound, y_bottom, y_bottom],
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                rt_str = f"{iso['rt']:.3f}" if isinstance(iso['rt'], (int, float)) else "?"
                mz_str = f"{iso['mz']:.4f}" if isinstance(iso['mz'], (int, float)) else "?"
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
        
        isomer_str = " // ".join(isomer_lines) if resolved_isomers else ""
        
        # find number of isomers in isomer_str and add line breaks every 3 isomers for readability
        if isomer_str != "":
            isomer_list = isomer_str.split(" // ")
            if len(isomer_list) > 3:
                isomer_str = " // <br>".join(
                    [" // ".join(isomer_list[i:i+3]) for i in range(0, len(isomer_list), 3)]
                )

        # Now add MS1 data traces
        highlighted_files = state.get("highlighted_files") or []
        expanded_rt_min = state["rt_min"] - 1
        expanded_rt_max = state["rt_max"] + 1

        ms1_trace_count = 0
        for _, r in sub.iterrows():
            fn = r.get("filename", "unknown")
            short_name = re.sub(r"_ms[12]_(?:neg|pos)$", "", "_".join(os.path.basename(fn).split(".")[0].split("_")[11:]))
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in fn.lower()), "gray")
            
            is_highlighted = fn in highlighted_files
            line_width = 5.0 if is_highlighted else 1.5
            
            rt_list = r.get("spec_rts", [])
            int_list = r.get("spec_ints", [])
            
            if len(rt_list) == 0: raise ValueError(f"MS1 data for {fn} has no RT points")
            
            rt_arr = np.asarray(rt_list)
            int_arr = np.asarray(int_list)
            
            # Filter for RT window
            mask = (rt_arr >= expanded_rt_min) & (rt_arr <= expanded_rt_max)
            filt_rt = rt_arr[mask]
            filt_int = int_arr[mask]

            if len(filt_rt) > 0:
                fig.add_trace(go.Scattergl(
                    x=filt_rt,
                    y=filt_int,
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    customdata=[fn] * len(filt_rt),
                    hovertemplate=f"%{{x:.3f}} min<br>%{{y:.2e}}<br>File: {short_name}",
                    showlegend=False,
                ))
                ms1_trace_count += 1

        logger.debug(f"MS1 traces created: {ms1_trace_count} individual traces from {len(sub['filename'].unique())} files")

        # Atlas RT peak line (black, static)
        fig.add_trace(go.Scatter(
            x=[row["atlas_rt_peak"], row["atlas_rt_peak"]],
            y=[y_bottom, y_upper_bound],
            mode="lines",
            line=dict(color="black", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Suggested RT lines (orange, static)
        if pd.notnull(row.get("suggested_rt_min")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_min"], row["suggested_rt_min"]],
                y=[y_bottom, y_upper_bound],
                mode="lines",
                line=dict(color="orange", width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ))
        if pd.notnull(row.get("suggested_rt_max")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_max"], row["suggested_rt_max"]],
                y=[y_bottom, y_upper_bound],
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
            f"<span style='font-size:1.2em'>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi_key}</span><br>"
            f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: {row['best_ms1_rt']:.4f}  |  RT Δ: {row['best_ms1_rt_error']:.3f}<br>"
            f"Atlas m/z: {row['atlas_mz']:.4f}  |  Meas M/Z: {row['best_ms1_mz']:.4f}  |  M/Z ppm Δ: {row['best_ms1_ppm_error']:.2f}<br>"
            f"<sub style='font-size:0.8em'>{isomer_str}</sub>"
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
            margin=dict(l=50, r=20, t=125, b=40), 
            dragmode="zoom",
            plot_bgcolor="white",
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

        fig_time = time.time() - fig_start
        logger.debug(f"_make_ms1_figure completed in {fig_time:.3f}s")
        return fig

    def _add_ms2_stick_traces(fig, mz_vals, intensities, hover_label, row_idx, col_idx, colors=None, default_color="red", line_width_px=3):
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
                row=row_idx, col=col_idx,
            )

    def _make_ms2_figure(state):
        compound_idx = state["compound_idx"]
        row = _compound_row(compound_idx)
        rt_min, rt_max = state["rt_min"], state["rt_max"]
        
        # Debug logging to verify we're using the correct compound
        #logger.info(f"_make_ms2_figure: compound_idx={compound_idx}, mz_rt_uid={row['mz_rt_uid']}, compound_name={row['compound_name']}")

        scans_by_energy = _get_ms2_scans(row, rt_min, rt_max)
        #logger.info(f"_make_ms2_figure: Got {len(scans_by_energy)} collision energies with scans")

        if len(scans_by_energy) == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"{row['compound_name']} - No MS2 data",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(margin=dict(l=50, r=20, t=80, b=40), plot_bgcolor="white",
                               xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
            return fig

        collision_energies = sorted(scans_by_energy.keys())
        n_energies = len(collision_energies)
        fig = make_subplots(rows=1, cols=n_energies, horizontal_spacing=0.08)

        for col_idx, ce in enumerate(collision_energies, start=1):
            scans = scans_by_energy[ce]
            if len(scans) == 0:
                xref = "x" if col_idx == 1 else f"x{col_idx}"
                yref = "y" if col_idx == 1 else f"y{col_idx}"
                fig.add_annotation(text="No scans", xref=xref, yref=yref, x=0.5, y=0.5, showarrow=False, row=1, col=col_idx)
                continue

            ce_ms2_idx = _get_ms2_idx_for_ce(state, ce)
            actual_idx = max(0, min(ce_ms2_idx, len(scans) - 1))
            scan = scans.iloc[actual_idx]
            
            # --- DATA EXTRACTION FROM HITS LIST ---
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

                # Plot Query (Top)
                _add_ms2_stick_traces(fig, q_mz, q_int, "Query", 1, col_idx, 
                                     colors=frag_colors, default_color="red", line_width_px=stick_width_px)

                # Plot Reference (Bottom)
                _add_ms2_stick_traces(fig, r_mz, ref_y, "Reference", 1, col_idx, 
                                     colors=frag_colors, default_color="red", line_width_px=stick_width_px)

                num_ref_fragments = hit.get('ref_frags', 0)
                num_matching_fragments = len(hit.get('matched_fragments', []))
                label_points.extend(zip(q_mz, q_int))
                label_points.extend(zip(r_mz, ref_y))
            else:
                # Fallback to raw spectrum if no hit is selected/available
                mz = _sanitize_numeric_list(scan.get('frag_mzs', []))
                ints = _sanitize_numeric_list(scan.get('frag_ints', []))
                _add_ms2_stick_traces(fig, mz, ints, "MS2", 1, col_idx, default_color="red", line_width_px=stick_width_px)
                label_points.extend(zip(mz, ints))

            # --- ANNOTATION & STYLING (Same as your legacy code) ---
            fig.add_hline(y=0, line=dict(color="black", width=1.5), row=1, col=col_idx)
            
            y_vals = [y for _, y in label_points if not np.isnan(y)] or [0]
            y_min, y_max = min(y_vals), max(y_vals)
            y_span = max(y_max - y_min, max(abs(y_min), abs(y_max)), 1.0)
            label_pad, y_pad, TEXT_HEIGHT_OFFSET = y_span * 0.01, y_span * 0.01, y_span * 0.01

            top_label_idxs = {idx for idx, _ in sorted(enumerate(label_points), key=lambda item: abs(item[1][1] if not np.isnan(item[1][1]) else 0), reverse=True)[:7]}
            top_labels_sorted = sorted([(idx, mz_val, y_val) for idx, (mz_val, y_val) in enumerate(label_points) if idx in top_label_idxs and not np.isnan(mz_val)], key=lambda item: item[1])

            prev_mz, stagger_level = None, 0
            for idx, mz_val, y_val in top_labels_sorted:
                y_base = (y_val + label_pad) if y_val >= 0 else (y_val - label_pad)
                if prev_mz is not None and abs(mz_val - prev_mz) < 5.0: stagger_level += 1
                else: stagger_level = 0
                y_pos = y_base + (stagger_level * TEXT_HEIGHT_OFFSET if y_val >= 0 else -stagger_level * TEXT_HEIGHT_OFFSET)
                xref_coord, yref_coord = ("x" if col_idx == 1 else f"x{col_idx}"), ("y" if col_idx == 1 else f"y{col_idx}")
                fig.add_annotation(x=mz_val, y=y_pos, text=f"{mz_val:.4f}", showarrow=False, xanchor="center", 
                                   yanchor="bottom" if y_val >= 0 else "top", font=dict(size=12), xref=xref_coord, yref=yref_coord)
                prev_mz = mz_val

            fig.update_xaxes(
                title_text=f"m/z ({_format_collision_energy_label(ce)})",
                showgrid=False,
                zeroline=False,
                title_font=dict(size=18),
                tickfont=dict(size=15),
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=f"Intensity (Ref scaled x{scale:.2f})" if col_idx == 1 else "",
                showgrid=False,
                zeroline=False,
                range=[y_min - y_pad, y_max + y_pad],
                title_font=dict(size=18),
                tickfont=dict(size=15),
                autorange=False,
                row=1,
                col=col_idx,
            )
            # Updated Scan Info using Hit metadata
            fname = "_".join(os.path.basename(scan.get("filename", "")).split(".")[0].split("_")[11:])
            if hit:
                scan_info = (
                    f"<span style='font-size:1.2em'>"
                    f"<b>CoS.: {hit.get('score', 0):.4f}</b>  |  "
                    f"Ions: {num_matching_fragments}/{num_ref_fragments}  |  "
                    f"RT: {scan.get('scan_rt', 0):.4f} min | "
                    f"Exp. m/z: {scan.get('precursor_MZ', 0):.4f}  |  "
                    f"Ref. m/z: {hit.get('mz_theoretical', 0):.4f}  |  "
                    f"ppm Δ: {hit.get('ppm_error', 0):.2f}"
                    f"</span><br>"
                    f"{hit.get('ref_name', 'Unknown')}  |  {fname}<br><br>"
                )
            else:
                scan_info = (
                    f"<span style='font-size:1.2em'>"
                    f"<b>No Hit</b>"
                    f"</span><br>"
                    f"{fname}<br><br>"
                )

            xref_str, yref_str = ("x domain" if col_idx == 1 else f"x{col_idx} domain"), ("y domain" if col_idx == 1 else f"y{col_idx} domain")
            fig.add_annotation(text=scan_info, xref=xref_str, yref=yref_str, x=0.5, y=1.02, showarrow=False, font=dict(size=14), xanchor="center", yanchor="bottom")
        
        fig.update_layout(
            barmode="overlay",
            hovermode="closest",
            margin=dict(l=50, r=20, t=120, b=40),
            plot_bgcolor="white",
            showlegend=False,
        )
        
        #logger.info(f"_make_ms2_figure: Completed for compound {compound_idx} ({row['compound_name']})")
        return fig

    logger.debug("App helpers defined successfully")

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

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
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

        return _flush_and_load_compound(old_state, compound_idx, delta=None)

    def _get_force_eval_and_warnings(state, delta):
        """Return warning messages for required-evaluation navigation logic."""
        force_eval = analysis_gui_obj.workflow_params.get("gui_require_all_evaluated", False)
        if analysis_gui_obj.override_parameters["gui_require_all_evaluated"] is not None:
            force_eval = analysis_gui_obj.override_parameters["gui_require_all_evaluated"]
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
        return new_state, new_idx

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms2-prev-1", "n_clicks"),
        Input("ms2-next-1", "n_clicks"),
        Input("ms2-prev-2", "n_clicks"),
        Input("ms2-next-2", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_ms2(prev1, nxt1, prev2, nxt2, state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered_id
        #logger.info(f"navigate_ms2 called: trigger={trigger}, current ms2_idx={state.get('ms2_idx', 0)}")
        if trigger == "ms2-prev-1":
            new_state = _move_ms2_idx_for_ce_position(state, ce_position=0, delta=-1)
            #logger.info(f"navigate_ms2: MS2-1 prev -> new ms2_idx={new_state.get('ms2_idx', 0)}")
            return new_state
        if trigger == "ms2-next-1":
            new_state = _move_ms2_idx_for_ce_position(state, ce_position=0, delta=1)
            #logger.info(f"navigate_ms2: MS2-1 next -> new ms2_idx={new_state.get('ms2_idx', 0)}")
            return new_state
        if trigger == "ms2-prev-2":
            new_state = _move_ms2_idx_for_ce_position(state, ce_position=1, delta=-1)
            #logger.info(f"navigate_ms2: MS2-2 prev -> new ms2_idx={new_state.get('ms2_idx', 0)}")
            return new_state
        if trigger == "ms2-next-2":
            new_state = _move_ms2_idx_for_ce_position(state, ce_position=1, delta=1)
            #logger.info(f"navigate_ms2: MS2-2 next -> new ms2_idx={new_state.get('ms2_idx', 0)}")
            return new_state
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
        fp = points[0].get("customdata")
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

        ALL_HOTKEYS = (
            set(analysis_gui_obj.notes["ms2_key_to_label"])
            | set(analysis_gui_obj.notes["ms1_key_to_label"])
            | set(analysis_gui_obj.notes["other_key_to_label"])
            | {"a", "s", "d", "f", "j", "k", "l", ";", ">", "/", "n", "m",
            "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}
        )

        if key not in ALL_HOTKEYS:
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
            return _patch_rt_change(state, rt_min, rt_max), dash.no_update

        # --- MS2 navigation keys ---
        if key in ("l", "ArrowUp"):
            return _move_ms2_idx_for_ce_position(state, ce_position=0, delta=-1), dash.no_update
        if key in (";", "ArrowDown"):
            return _move_ms2_idx_for_ce_position(state, ce_position=0, delta=1), dash.no_update
        if key == "o":
            return _move_ms2_idx_for_ce_position(state, ce_position=1, delta=-1), dash.no_update
        if key == "p":
            return _move_ms2_idx_for_ce_position(state, ce_position=1, delta=1), dash.no_update

        # --- Accept suggestions ---
        if key == "n":
            row = _compound_row(state["compound_idx"])
            if pd.notnull(row.get("suggested_rt_min")):
                return _patch_rt_change(
                    state,
                    float(row["suggested_rt_min"]),
                    float(row["suggested_rt_max"]),
                ), dash.no_update
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
            return new_state, dash.no_update

        # --- Compound navigation ---
        if key in ("j", "k", "ArrowLeft", "ArrowRight"):
            delta = 1 if key in ("k", "ArrowRight") else -1
            new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)
            if new_idx == int(state["compound_idx"]):
                raise dash.exceptions.PreventUpdate
            new_state = _flush_and_load_compound(state, new_idx, delta=delta)
            return new_state, new_idx

    # --- Note hotkeys ---
        if key in analysis_gui_obj.notes["ms2_key_to_label"]:
            return _patch_with_seq(state, ms2_note=analysis_gui_obj.notes["ms2_key_to_label"][key]), dash.no_update

        if key in analysis_gui_obj.notes["ms1_key_to_label"]:
            return _patch_with_seq(state, ms1_note=analysis_gui_obj.notes["ms1_key_to_label"][key]), dash.no_update

        if key in analysis_gui_obj.notes["other_key_to_label"]:
            current = state.get("other_note")
            if not isinstance(current, list):
                current = []
            label = analysis_gui_obj.notes["other_key_to_label"][key]
            if label in current:
                current = [v for v in current if v != label]
            else:
                current = current + [label]
            return _patch_with_seq(state, other_note=current), dash.no_update

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("ms1-graph", "figure"),
        Output("ms2-graph", "figure"),
        Output("error-banner", "children"),
        Input("session-store", "data"),
        Input("yaxis-scale-radio", "value"),
        prevent_initial_call=False,
    )
    def update_figures(state, yaxis_scale):
        state = _ensure_valid_state(state)
        
        flush_err = state.get("flush_error")
        ms2_warning = state.get("ms2_warning")
        ms1_warning = state.get("ms1_warning")
        try:
            # Generate figures (no caching)
            ms1_fig = _make_ms1_figure(state, yaxis_scale)
            ms2_fig = _make_ms2_figure(state)
            
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
            return ms1_fig, ms2_fig, banner
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"update_figures error: {exc}")
            err_html = html.Span(
                f"Figure error: {type(exc).__name__}: {exc}",
                style={"color": "red", "fontSize": "11px", "fontWeight": "bold"},
            )
            empty = go.Figure()
            empty.update_layout(margin=dict(l=50, r=20, t=40, b=40))
            return empty, empty, err_html

    @app.callback(
        Output("status-current", "children"),
        Output("status-previous", "children"),
        Output("compound-counter", "children"),
        Output("ms2-counter-1", "children"),
        Output("ms2-counter-2", "children"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_status(state):
        state = _ensure_valid_state(state)
        row = _compound_row(state["compound_idx"])
        comp_txt = f"Compound {state['compound_idx']+1} of {len(compound_options)}"
        
        # Get scans by collision energy for detailed count
        scans_by_energy = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        ms2_txt_1 = "No CE102040"
        ms2_txt_2 = "No CE205060"
        if scans_by_energy:
            collision_energies = sorted(scans_by_energy.keys())
            if len(collision_energies) > 0:
                ce1 = collision_energies[0]
                n1 = len(scans_by_energy.get(ce1, []))
                idx1 = min(_get_ms2_idx_for_ce(state, ce1), max(n1 - 1, 0)) if n1 > 0 else 0
                ms2_txt_1 = f"{_format_collision_energy_label(ce1)}: {idx1 + 1}/{n1}" if n1 > 0 else f"{_format_collision_energy_label(ce1)}: 0/0"
            if len(collision_energies) > 1:
                ce2 = collision_energies[1]
                n2 = len(scans_by_energy.get(ce2, []))
                idx2 = min(_get_ms2_idx_for_ce(state, ce2), max(n2 - 1, 0)) if n2 > 0 else 0
                ms2_txt_2 = f"{_format_collision_energy_label(ce2)}: {idx2 + 1}/{n2}" if n2 > 0 else f"{_format_collision_energy_label(ce2)}: 0/0"
        
        pending = html.Span(
            [
                "Working analysis: ", html.I(f"{row['compound_name']} ({row['adduct']}) [{state['compound_idx']+1}]"),
                html.Br(), f"RT [{state['rt_min']:.4f}, {state['rt_max']:.4f}]",
                html.Br(), f"MS1: {state['ms1_note']}",
                html.Br(), f"MS2: {state['ms2_note']}",
                html.Br(), f"Other: {state['other_note']}",
                html.Br(), f"Analyst Notes: {state['analyst_notes'][:50]}{'...' if len(state['analyst_notes']) > 50 else ''}",
            ],
            style={"color": "#b8860b", "fontSize": "16px", "fontWeight": "bold"},
        )
        if state.get("last_saved"):
            s = state["last_saved"]
            saved = html.Span(
                [
                    "Saved: ", html.I(f"{s['name']} ({s['adduct']}) [{s['index']}]"),
                ],
                style={"color": "#2a7a2a", "fontSize": "16px", "fontWeight": "bold"},
            )
        else:
            saved = html.Span("Previous: NA", style={"color": "#888", "fontSize": "16px"})
        return pending, saved, comp_txt, ms2_txt_1, ms2_txt_2

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