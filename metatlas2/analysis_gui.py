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
    manual_curation_df = analysis_gui_obj.manual_curation_df
    logger.info(f"Starting manual curation with {len(manual_curation_df)} compounds")

    top_n_hits = analysis_gui_obj.workflow_params.get("gui_top_n_hits", 20)
    if analysis_gui_obj.override_parameters.get("gui_top_n_hits") is not None:
        top_n_hits = analysis_gui_obj.override_parameters["gui_top_n_hits"]

    # Extract metadata for display
    chrom = analysis_gui_obj.post_autoid_atlas_obj.chromatography
    pol = analysis_gui_obj.post_autoid_atlas_obj.polarity
    analysis_type = analysis_gui_obj.post_autoid_atlas_obj.analysis_type
    rta = analysis_gui_obj.rt_alignment_number
    tga = analysis_gui_obj.analysis_number

    # Set up all passing compounds as options for the dropdown
    compound_options = [
        {"label": f"{i+1}: {row['compound_name']}", "value": i}
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
        logger.debug("App built successfully")
        app.config.prevent_initial_callbacks = "initial_duplicate"
    except Exception as e:
        traceback.print_exc()
        logger.error(f"FAILED: {e}")

    # Set up some caching to help with race conditions
    flush_lock = threading.RLock()
    cache_lock = threading.Lock()
    latest_flushed_seq_by_session = {}
    ms2_scans_cache = {}
    
    # PHASE 3 OPTIMIZATION: Add figure caching to avoid regeneration
    figure_cache = {}
    figure_cache_lock = threading.Lock()
    isomer_string_cache = {}  # Cache expensive isomer calculations
    
    # PHASE 4 OPTIMIZATION: Async database writes to avoid blocking UI
    db_write_queue = []
    db_write_lock = threading.Lock()
    
    def _async_db_write(db_path, curation_uid, updates):
        """Non-blocking database write using thread pool."""
        def _write():
            try:
                dbi.write_gui_updates_to_db(db_path, curation_uid, updates)
            except Exception as e:
                logger.error(f"Async DB write failed: {e}")
        
        # Execute in separate thread to not block UI
        thread = threading.Thread(target=_write, daemon=True)
        thread.start()

    # PHASE 1 OPTIMIZATION: Pre-index DataFrames by mz_rt_uid for fast lookups
    logger.info("Pre-indexing MS data by mz_rt_uid for fast lookups...")
    index_start = time.time()

    # Build lookup dictionaries keyed by mz_rt_uid
    ms1_by_compound = {}
    ms2_by_compound = {}
    ms2_hits_by_compound = {}

    # Index MS1 data
    for mz_rt_uid, group in analysis_gui_obj.ms1_df.groupby(["mz_rt_uid"], observed=True):
        ms1_by_compound[mz_rt_uid] = group

    # Index MS2 data
    for mz_rt_uid, group in analysis_gui_obj.ms2_df.groupby(["mz_rt_uid"], observed=True):
        ms2_by_compound[mz_rt_uid] = group

    # Index MS2 hits data
    for mz_rt_uid, group in analysis_gui_obj.ms2_hits_df.groupby(["mz_rt_uid"], observed=True):
        ms2_hits_by_compound[mz_rt_uid] = group

    index_time = time.time() - index_start
    logger.info(f"Indexed {len(ms1_by_compound)} unique compounds in {index_time:.2f}s")
    logger.info(f"  MS1: {len(analysis_gui_obj.ms1_df)} rows → {len(ms1_by_compound)} groups")
    logger.info(f"  MS2: {len(analysis_gui_obj.ms2_df)} rows → {len(ms2_by_compound)} groups")
    logger.info(f"  MS2 hits: {len(analysis_gui_obj.ms2_hits_df)} rows → {len(ms2_hits_by_compound)} groups")

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
        new_state["edit_seq"] = int(state.get("edit_seq", 0)) + 1
        return new_state

    def _patch_rt_change(state, new_min, new_max):
        rt_min = max(0.0, min(new_min, new_max))
        rt_max = max(rt_min, new_max)
        logger.debug(f"[_patch_rt_change] Called with new_min={new_min}, new_max={new_max} -> rt_min={rt_min}, rt_max={rt_max}")
        # PHASE 4 FIX: Preserve cached_y_max to keep isomer boxes visible
        new_state = _patch_with_seq(state, rt_min=round(rt_min, 4), rt_max=round(rt_max, 4))
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

    starting_compound_idx = _find_starting_compound_idx()
    if starting_compound_idx > 0:
        logger.info(f"Resuming analysis at compound {starting_compound_idx+1}: "
                   f"{manual_curation_df.iloc[starting_compound_idx]['compound_name']}")
    else:
        logger.info("Starting new analysis at compound 1")

    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp", "target.tagName"]}],
    )

    # format of the app itself
    ms1_height = "650px"
    ms2_height = "550px"
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
                                f"{chrom}  |  {pol}  |  {analysis_type}  |  RTA{rta}  |  TGA{tga}",
                                style={"fontSize": "1rem", "fontWeight": "bold", "marginBottom": "0.5rem", "color": "#333"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(id="compound-dd", options=compound_options, value=starting_compound_idx, clearable=False, style={"width": "100%", "fontSize": "1.5rem"}),
                                        width=12, className="mb-3",
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
                                        value="",
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
                                style={"height": ms1_height},
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
                                style={"height": ms2_height}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀ Prev MS2  [l]", id="ms2-prev", className="me-2 w-100", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(html.Div(id="ms2-counter", className="fw-bold text-center", style={"fontSize": "1rem"}), width=2),
                                    dbc.Col(dbc.Button("Next MS2 ▶  [;]", id="ms2-next", className="ms-2 w-100", style={"fontSize": "1rem"}), width=2),
                                ],
                                className="my-2 align-items-center",
                                style={"width": "100%"},
                                justify="start",
                            ),
                        ],
                        width=9,
                    ),
                ],
                className="mt-1",
            ),
        ],
        fluid=True,
        style={"paddingTop": "0.5rem"},
    )

    logger.debug("Layout constructed successfully")

    def _parse_isomers(isomers_val):
        if isinstance(isomers_val, list):
            return isomers_val
        # Accept JSON string representation of a list
        if isinstance(isomers_val, str):
            try:
                parsed = json.loads(isomers_val)
                if isinstance(parsed, list):
                    return parsed
                else:
                    raise ValueError(f"Isomers string did not decode to a list: {isomers_val!r}")
            except Exception as exc:
                raise ValueError(f"Isomers string could not be parsed as JSON list: {isomers_val!r} ({exc})")
        raise ValueError(f"Unexpected isomers format (not an empty list or list of dicts): {isomers_val!r}")

    def _get_sorted_isomer_rt_bounds(row):
        """Return list of (rt_min, rt_max) for isomers, sorted by rt_min."""
        isomers_val = row.get("isomers") if "isomers" in row.index else []
        isomers = _parse_isomers(isomers_val)
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

    def _get_ms2_scans(row, rt_min=None, rt_max=None):
        """Return dictionary of sorted, capped scans DataFrames grouped by collision energy.

        Returns a dict: {collision_energy: DataFrame} with top N hits from each collision energy.
        Results are cached by (inchi_key, adduct, top_n_hits, rt_min, rt_max).
        """
        mz_rt_uid = row["mz_rt_uid"] if isinstance(row, dict) and "mz_rt_uid" in row else None
        if mz_rt_uid is None:
            # fallback for legacy call signature
            mz_rt_uid = None
            if isinstance(row, dict):
                mz_rt_uid = row.get("mz_rt_uid", None)
            if mz_rt_uid is None and "mz_rt_uid" in analysis_gui_obj.ms2_df.columns:
                mz_rt_uid = analysis_gui_obj.ms2_df.iloc[0]["mz_rt_uid"]
        key = (mz_rt_uid, int(top_n_hits),
               round(rt_min, 4) if rt_min is not None else None,
               round(rt_max, 4) if rt_max is not None else None)
        with cache_lock:
            if key in ms2_scans_cache:
                return ms2_scans_cache[key]

        ms2_sub = ms2_by_compound.get(mz_rt_uid, pd.DataFrame())
        if not ms2_sub.empty and rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["rt"] >= rt_min) & (ms2_sub["rt"] <= rt_max)]
        
        hits_sub = ms2_hits_by_compound.get(mz_rt_uid, pd.DataFrame())
        if not hits_sub.empty and rt_min is not None and rt_max is not None:
            hits_sub = hits_sub[(hits_sub["rt"] >= rt_min) & (hits_sub["rt"] <= rt_max)]

        # Group by collision energy and get top N from each
        scans_by_energy = {}
        if ms2_sub.empty:
            pass  # Return empty dict
        elif hits_sub.empty:
            # No hits - group by collision energy and sort by RT
            for ce, group in ms2_sub.groupby("collision_energy"):
                scans_by_energy[ce] = group.sort_values("rt").head(top_n_hits)
        else:
            # Merge with hits and group by collision energy
            merged = pd.merge(ms2_sub, hits_sub, on=["mz_rt_uid", "file_path", "rt"], how="left")
            merged["score"] = merged["score"].fillna(0)
            for ce, group in merged.groupby("collision_energy"):
                scans_by_energy[ce] = group.sort_values(["score", "rt"], ascending=[False, True]).head(top_n_hits)

        with cache_lock:
            if key not in ms2_scans_cache:
                ms2_scans_cache[key] = scans_by_energy
        return ms2_scans_cache[key]

    def _count_ms2_scans(row, rt_min=None, rt_max=None):
        scans_by_energy = _get_ms2_scans(row, rt_min, rt_max)
        # Return max count across all collision energies for navigation
        return max((len(df) for df in scans_by_energy.values()), default=0)

    def _clean_spectrum(val_arr, int_arr):
        out_i = [0 if (isinstance(i, float) and np.isnan(i)) else i for i in int_arr]
        return val_arr, out_i

    # PHASE 2 OPTIMIZATION: Add size limits to prevent unbounded memory growth
    # PHASE 4: Increased cache size for better hit rate with large datasets
    @functools.lru_cache(maxsize=5000)
    def _parse_spectrum_cached(raw_spectrum):
        val, ints = json.loads(raw_spectrum)
        val, ints = _clean_spectrum(val, ints)
        return val, ints

    @functools.lru_cache(maxsize=1000)
    def _parse_mz_cached(raw_mz):
        if raw_mz is None or (isinstance(raw_mz, float) and np.isnan(raw_mz)):
            return []
        if isinstance(raw_mz, str):
            return json.loads(raw_mz)
        if isinstance(raw_mz, (list, tuple, np.ndarray)):
            return list(raw_mz)
        return []

    # PHASE 2 OPTIMIZATION: Create memoized wrapper for expensive metric computation
    # Cache key: (compound_idx, rounded rt_min, rounded rt_max)
    @functools.lru_cache(maxsize=200)
    def _compute_window_ms1_metrics_cached(compound_idx, rt_min_round, rt_max_round):
        """Cached version of metrics computation with rounded RT values."""
        row = _compound_row(compound_idx)
        mz_rt_uid = row["mz_rt_uid"]
        rt_min = float(rt_min_round)
        rt_max = float(rt_max_round)
        
        def _ppm_error(measured_mz, atlas_mz):
            if pd.notnull(atlas_mz) and float(atlas_mz) != 0 and np.isfinite(measured_mz):
                return (float(measured_mz) - float(atlas_mz)) / float(atlas_mz) * 1e6
            return np.nan

        def _rt_delta(measured_rt, atlas_rt_peak):
            if pd.notnull(atlas_rt_peak) and np.isfinite(measured_rt):
                return float(measured_rt) - float(atlas_rt_peak)
            return np.nan

        if rt_max < rt_min:
            rt_min, rt_max = rt_max, rt_min

        fallback_rt_peak = (rt_min + rt_max) / 2.0
        fallback_mz = row.get("best_ms1_mz", np.nan)
        if pd.isna(fallback_mz):
            fallback_mz = row.get("atlas_mz", np.nan)

        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())
        
        atlas_mz = row.get("atlas_mz", np.nan)
        atlas_rt_peak = row.get("atlas_rt_peak", np.nan)
        if sub.empty:
            fallback_mz_val = float(fallback_mz) if pd.notnull(fallback_mz) else np.nan
            fallback_rt_error = _rt_delta(fallback_rt_peak, atlas_rt_peak)
            fallback_mz_error = _ppm_error(fallback_mz_val, atlas_mz)
            return {
                "rt_peak": fallback_rt_peak,
                "mz": fallback_mz_val,
                "rt_error": float(fallback_rt_error) if np.isfinite(fallback_rt_error) else np.nan,
                "mz_error": float(fallback_mz_error) if np.isfinite(fallback_mz_error) else np.nan,
                "best_ms1_file": row.get("best_ms1_file", ""),
                "best_ms1_rt": float(row.get("best_ms1_rt", np.nan)),
                "best_ms1_mz": float(row.get("best_ms1_mz", np.nan)),
                "best_ms1_intensity": float(row.get("best_ms1_intensity", np.nan)),
                "best_ms1_ppm_error": float(row.get("best_ms1_ppm_error", np.nan)),
                "best_ms1_rt_error": float(row.get("best_ms1_rt_error", np.nan)),
            }

        has_file_path = "file_path" in sub.columns
        has_mz = "mz" in sub.columns
        best_by_file = {}

        cols = []
        if has_file_path:
            cols.append("file_path")
        cols.append("raw_spectrum")
        if has_mz:
            cols.append("mz")
        iter_rows = sub[cols].itertuples(index=False, name=None)

        for row_vals in iter_rows:
            if has_file_path and has_mz:
                file_path, raw_spectrum, raw_mz = row_vals
            elif has_file_path:
                file_path, raw_spectrum = row_vals
                raw_mz = None
            elif has_mz:
                raw_spectrum, raw_mz = row_vals
                file_path = None
            else:
                (raw_spectrum,) = row_vals
                file_path = None
                raw_mz = None

            try:
                rt_vals, intensities = _parse_spectrum_cached(raw_spectrum)
            except Exception:
                continue

            rt_arr = np.asarray(rt_vals, dtype=float)
            int_arr = np.asarray(intensities, dtype=float)
            if rt_arr.size == 0 or int_arr.size == 0:
                continue

            n = min(rt_arr.size, int_arr.size)
            rt_arr = rt_arr[:n]
            int_arr = int_arr[:n]

            mz_arr = np.asarray([], dtype=float)
            if raw_mz is not None:
                try:
                    mz_list = _parse_mz_cached(raw_mz)
                    mz_arr = np.asarray(mz_list, dtype=float)
                    if mz_arr.size > n:
                        mz_arr = mz_arr[:n]
                except Exception:
                    pass

            mask = (
                (rt_arr >= rt_min)
                & (rt_arr <= rt_max)
                & np.isfinite(rt_arr)
                & np.isfinite(int_arr)
            )
            if not np.any(mask):
                continue

            masked_int = np.where(mask, int_arr, -np.inf)
            local_idx = int(np.argmax(masked_int))
            local_intensity = float(masked_int[local_idx])
            if not np.isfinite(local_intensity):
                continue

            local_rt = float(rt_arr[local_idx])
            local_mz = np.nan
            if mz_arr.size == n and local_idx < mz_arr.size and np.isfinite(mz_arr[local_idx]):
                local_mz = float(mz_arr[local_idx])

            prev = best_by_file.get(file_path)
            if prev is None or local_intensity > prev[0]:
                best_by_file[file_path] = (local_intensity, local_rt, local_mz)

        if not best_by_file:
            fallback_mz_val = float(fallback_mz) if pd.notnull(fallback_mz) else np.nan
            fallback_rt_error = _rt_delta(fallback_rt_peak, atlas_rt_peak)
            fallback_mz_error = _ppm_error(fallback_mz_val, atlas_mz)
            return {
                "rt_peak": fallback_rt_peak,
                "mz": fallback_mz_val,
                "rt_error": float(fallback_rt_error) if np.isfinite(fallback_rt_error) else np.nan,
                "mz_error": float(fallback_mz_error) if np.isfinite(fallback_mz_error) else np.nan,
                "best_ms1_file": row.get("best_ms1_file", ""),
                "best_ms1_rt": float(row.get("best_ms1_rt", np.nan)),
                "best_ms1_mz": float(row.get("best_ms1_mz", np.nan)),
                "best_ms1_intensity": float(row.get("best_ms1_intensity", np.nan)),
                "best_ms1_ppm_error": float(row.get("best_ms1_ppm_error", np.nan)),
                "best_ms1_rt_error": float(row.get("best_ms1_rt_error", np.nan)),
            }

        rt_peak = float(np.mean([rt for _, rt, _ in best_by_file.values()]))
        mz_vals = [mz for _, _, mz in best_by_file.values() if np.isfinite(mz)]
        mz_mean = float(np.mean(mz_vals)) if mz_vals else (float(fallback_mz) if pd.notnull(fallback_mz) else np.nan)

        best_file, best_triplet = max(best_by_file.items(), key=lambda kv: kv[1][0])
        best_intensity, best_rt, best_mz = best_triplet
        if not np.isfinite(best_mz):
            best_mz = mz_mean

        best_ppm_error = _ppm_error(best_mz, atlas_mz)
        best_rt_error = _rt_delta(best_rt, atlas_rt_peak)
        mz_error = _ppm_error(mz_mean, atlas_mz)
        rt_error = _rt_delta(rt_peak, atlas_rt_peak)

        return {
            "rt_peak": rt_peak,
            "mz": mz_mean,
            "rt_error": float(rt_error) if np.isfinite(rt_error) else np.nan,
            "mz_error": float(mz_error) if np.isfinite(mz_error) else np.nan,
            "best_ms1_file": "" if best_file is None else str(best_file),
            "best_ms1_rt": float(best_rt),
            "best_ms1_mz": float(best_mz) if np.isfinite(best_mz) else np.nan,
            "best_ms1_intensity": float(best_intensity),
            "best_ms1_ppm_error": float(best_ppm_error) if np.isfinite(best_ppm_error) else np.nan,
            "best_ms1_rt_error": float(best_rt_error) if np.isfinite(best_rt_error) else np.nan,
        }

    def _compute_window_ms1_metrics(state):
        """Compute all window-based MS1 metrics in one pass across file spectra.
        
        OPTIMIZED: Routes to cached version with rounded RT values to enable memoization.
        """
        # Round RT values to 3 decimal places for cache key (0.001 min = 0.06 sec resolution)
        rt_min_round = round(float(state["rt_min"]), 3)
        rt_max_round = round(float(state["rt_max"]), 3)
        return _compute_window_ms1_metrics_cached(state["compound_idx"], rt_min_round, rt_max_round)

    def _compute_window_ms1_metrics_old(state):
        """DEPRECATED: Old uncached version kept for reference."""
        def _ppm_error(measured_mz, atlas_mz):
            if pd.notnull(atlas_mz) and float(atlas_mz) != 0 and np.isfinite(measured_mz):
                return (float(measured_mz) - float(atlas_mz)) / float(atlas_mz) * 1e6
            return np.nan

        def _rt_delta(measured_rt, atlas_rt_peak):
            if pd.notnull(atlas_rt_peak) and np.isfinite(measured_rt):
                return float(measured_rt) - float(atlas_rt_peak)
            return np.nan

        row = _compound_row(state["compound_idx"])
        mz_rt_uid = row["mz_rt_uid"]
        rt_min = float(state["rt_min"])
        rt_max = float(state["rt_max"])
        if rt_max < rt_min:
            rt_min, rt_max = rt_max, rt_min

        fallback_rt_peak = (rt_min + rt_max) / 2.0
        fallback_mz = row.get("best_ms1_mz", np.nan)
        if pd.isna(fallback_mz):
            fallback_mz = row.get("atlas_mz", np.nan)

        # OPTIMIZED: Use pre-indexed lookup instead of filtering entire DataFrame
        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())
        
        atlas_mz = row.get("atlas_mz", np.nan)
        atlas_rt_peak = row.get("atlas_rt_peak", np.nan)
        if sub.empty:
            fallback_mz_val = float(fallback_mz) if pd.notnull(fallback_mz) else np.nan
            fallback_rt_error = _rt_delta(fallback_rt_peak, atlas_rt_peak)
            fallback_mz_error = _ppm_error(fallback_mz_val, atlas_mz)
            return {
                "rt_peak": fallback_rt_peak,
                "mz": fallback_mz_val,
                "rt_error": float(fallback_rt_error) if np.isfinite(fallback_rt_error) else np.nan,
                "mz_error": float(fallback_mz_error) if np.isfinite(fallback_mz_error) else np.nan,
                "best_ms1_file": row.get("best_ms1_file", ""),
                "best_ms1_rt": float(row.get("best_ms1_rt", np.nan)),
                "best_ms1_mz": float(row.get("best_ms1_mz", np.nan)),
                "best_ms1_intensity": float(row.get("best_ms1_intensity", np.nan)),
                "best_ms1_ppm_error": float(row.get("best_ms1_ppm_error", np.nan)),
                "best_ms1_rt_error": float(row.get("best_ms1_rt_error", np.nan)),
            }

        has_file_path = "file_path" in sub.columns
        has_mz = "mz" in sub.columns
        best_by_file = {}

        cols = []
        if has_file_path:
            cols.append("file_path")
        cols.append("raw_spectrum")
        if has_mz:
            cols.append("mz")
        iter_rows = sub[cols].itertuples(index=False, name=None)

        for row_vals in iter_rows:
            if has_file_path and has_mz:
                file_path, raw_spectrum, raw_mz = row_vals
            elif has_file_path:
                file_path, raw_spectrum = row_vals
                raw_mz = None
            elif has_mz:
                raw_spectrum, raw_mz = row_vals
                file_path = None
            else:
                (raw_spectrum,) = row_vals
                file_path = None
                raw_mz = None

            try:
                rt_vals, intensities = _parse_spectrum_cached(raw_spectrum)
            except Exception:
                continue

            rt_arr = np.asarray(rt_vals, dtype=float)
            int_arr = np.asarray(intensities, dtype=float)
            if rt_arr.size == 0 or int_arr.size == 0:
                continue

            n = min(rt_arr.size, int_arr.size)
            rt_arr = rt_arr[:n]
            int_arr = int_arr[:n]

            mz_arr = np.asarray([], dtype=float)
            if raw_mz is not None:
                try:
                    mz_vals = _parse_mz_cached(raw_mz)
                    mz_arr = np.asarray(mz_vals, dtype=float)
                    if mz_arr.size:
                        mz_arr = mz_arr[:n]
                except Exception:
                    mz_arr = np.asarray([], dtype=float)

            mask = (
                (rt_arr >= rt_min)
                & (rt_arr <= rt_max)
                & np.isfinite(rt_arr)
                & np.isfinite(int_arr)
            )
            if not np.any(mask):
                continue

            masked_int = np.where(mask, int_arr, -np.inf)
            local_idx = int(np.argmax(masked_int))
            local_intensity = float(masked_int[local_idx])
            if not np.isfinite(local_intensity):
                continue

            local_rt = float(rt_arr[local_idx])
            local_mz = np.nan
            if mz_arr.size == n and local_idx < mz_arr.size and np.isfinite(mz_arr[local_idx]):
                local_mz = float(mz_arr[local_idx])

            prev = best_by_file.get(file_path)
            if prev is None or local_intensity > prev[0]:
                best_by_file[file_path] = (local_intensity, local_rt, local_mz)

        if not best_by_file:
            fallback_mz_val = float(fallback_mz) if pd.notnull(fallback_mz) else np.nan
            fallback_rt_error = _rt_delta(fallback_rt_peak, atlas_rt_peak)
            fallback_mz_error = _ppm_error(fallback_mz_val, atlas_mz)
            return {
                "rt_peak": fallback_rt_peak,
                "mz": fallback_mz_val,
                "rt_error": float(fallback_rt_error) if np.isfinite(fallback_rt_error) else np.nan,
                "mz_error": float(fallback_mz_error) if np.isfinite(fallback_mz_error) else np.nan,
                "best_ms1_file": row.get("best_ms1_file", ""),
                "best_ms1_rt": float(row.get("best_ms1_rt", np.nan)),
                "best_ms1_mz": float(row.get("best_ms1_mz", np.nan)),
                "best_ms1_intensity": float(row.get("best_ms1_intensity", np.nan)),
                "best_ms1_ppm_error": float(row.get("best_ms1_ppm_error", np.nan)),
                "best_ms1_rt_error": float(row.get("best_ms1_rt_error", np.nan)),
            }

        rt_peak = float(np.mean([rt for _, rt, _ in best_by_file.values()]))
        mz_vals = [mz for _, _, mz in best_by_file.values() if np.isfinite(mz)]
        mz_mean = float(np.mean(mz_vals)) if mz_vals else (float(fallback_mz) if pd.notnull(fallback_mz) else np.nan)

        best_file, best_triplet = max(best_by_file.items(), key=lambda kv: kv[1][0])
        best_intensity, best_rt, best_mz = best_triplet
        if not np.isfinite(best_mz):
            best_mz = mz_mean

        best_ppm_error = _ppm_error(best_mz, atlas_mz)
        best_rt_error = _rt_delta(best_rt, atlas_rt_peak)
        mz_error = _ppm_error(mz_mean, atlas_mz)
        rt_error = _rt_delta(rt_peak, atlas_rt_peak)

        return {
            "rt_peak": rt_peak,
            "mz": mz_mean,
            "rt_error": float(rt_error) if np.isfinite(rt_error) else np.nan,
            "mz_error": float(mz_error) if np.isfinite(mz_error) else np.nan,
            "best_ms1_file": "" if best_file is None else str(best_file),
            "best_ms1_rt": float(best_rt),
            "best_ms1_mz": float(best_mz) if np.isfinite(best_mz) else np.nan,
            "best_ms1_intensity": float(best_intensity),
            "best_ms1_ppm_error": float(best_ppm_error) if np.isfinite(best_ppm_error) else np.nan,
            "best_ms1_rt_error": float(best_rt_error) if np.isfinite(best_rt_error) else np.nan,
        }

    def _flush_to_db(state):
        sid = state.get("session_id", "unknown")
        seq = int(state.get("edit_seq", 0))
        flush_key = (sid, state.get("compound_idx"))

        with flush_lock:
            latest_seq = latest_flushed_seq_by_session.get(flush_key, -1)
            if seq <= latest_seq:
                return state
            latest_flushed_seq_by_session[flush_key] = seq

        row = _compound_row(state["compound_idx"])
        
        # PHASE 4 OPTIMIZATION: Move expensive computation to async thread
        # Instead of blocking UI, compute metrics and write DB in background
        def _async_flush_worker():
            try:
                window_metrics = _compute_window_ms1_metrics(state)
                updates = {
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
                
                # DB write
                dbi.write_gui_updates_to_db(analysis_gui_obj.paths["project_db_path"], row["curation_uid"], updates)
                
                # Update in-memory DataFrame
                idx = state["compound_idx"]
                df_idx = manual_curation_df.index[idx]
                with flush_lock:
                    for col, val in updates.items():
                        if col in manual_curation_df.columns:
                            manual_curation_df.at[df_idx, col] = val
            except Exception as e:
                logger.error(f"Async flush worker failed: {e}")
                traceback.print_exc()
        
        # Launch async worker thread - don't block UI!
        thread = threading.Thread(target=_async_flush_worker, daemon=True)
        thread.start()

        # Return immediately with optimistic state update
        state["last_saved"] = {
            "name": row["compound_name"],
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

        if analysis_gui_obj.override_parameters['gui_lcmsruns_colors'] is not None:
            lcmsruns_color_map = analysis_gui_obj.override_parameters['gui_lcmsruns_colors']
        else:
            lcmsruns_color_map = analysis_gui_obj.workflow_params.get("gui_lcmsruns_colors", {})
    
        row = _compound_row(state["compound_idx"])
        mz_rt_uid = row["mz_rt_uid"]
        adduct = row.get("adduct", "")
        inchi_key = row.get("inchi_key", "")
        rt_min, rt_max = state["rt_min"], state["rt_max"]
        x_window_min, x_window_max = _default_plot_bounds_from_row(row)

        # OPTIMIZED: Use pre-indexed lookup instead of filtering entire DataFrame
        sub = ms1_by_compound.get(mz_rt_uid, pd.DataFrame())

        # Use the manual-curation row's best MS1 intensity as the y-axis reference max.
        # PHASE 4 FIX: Cache y_max_data consistently to prevent isomer boxes from disappearing
        y_min_positive_data = None
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
        
        # PHASE 4 OPTIMIZATION: Skip expensive log scale calculation - use simple estimate
        if yaxis_scale == "log":
            # Instead of parsing all files, use a reasonable minimum
            y_min_positive_data = y_max_data / 1e6  # Assume 6 orders of magnitude dynamic range
        y_upper_bound = max(y_max_data * 1.1, 1.0)

        if yaxis_scale == "log":
            log_min = max((y_min_positive_data or 1e-6), 1e-12)
            y_range = [np.log10(log_min), np.log10(y_upper_bound)]
        else:
            y_range = [0.0, y_upper_bound]

        fig = go.Figure()
        
        # PHASE 4 OPTIMIZATION: Cache isomer metadata, draw rectangles separately
        compound_idx = state["compound_idx"]
        if compound_idx in isomer_string_cache:
            resolved_isomers = isomer_string_cache[compound_idx]
        else:
            resolved_isomers = []
            try:
                isomers = _parse_isomers(row.get("isomers"))
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
                            raise ValueError(f"Multiple isomer matches for {iso_name} {iso_adduct}")
                        if isomer_match.empty:
                            continue
                        if "remove" in isomer_match.iloc[0]["ms1_notes"].lower():
                            continue
                        iso_df_idx = isomer_match.index[0]
                        resolved_isomers.append({
                            "display_idx": iso_df_idx,
                            "name": iso_name,
                            "adduct": iso_adduct,
                            "rt_min": float(isomer_match.iloc[0]["rt_min"]),
                            "rt_max": float(isomer_match.iloc[0]["rt_max"]),
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
                # Check overlap with current compound window
                overlaps = _window_overlaps(iso["rt_min"], iso["rt_max"], current_rt_min, current_rt_max)
                # Check overlap with other isomers
                if not overlaps:
                    for j, other in enumerate(resolved_isomers):
                        if i != j and _window_overlaps(iso["rt_min"], iso["rt_max"], other["rt_min"], other["rt_max"]):
                            overlaps = True
                            break
                
                fillcolor = "rgba(255,96,96,0.28)" if overlaps else "rgba(150,205,255,0.28)"
                fig.add_trace(go.Scatter(
                    x=[iso["rt_min"], iso["rt_min"], iso["rt_max"], iso["rt_max"], iso["rt_min"]],
                    y=[y_bottom, y_max_data, y_max_data, y_bottom, y_bottom],
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                rt_str = f"{iso['rt']:.3f}" if isinstance(iso['rt'], (int, float)) else "?"
                mz_str = f"{iso['mz']:.4f}" if isinstance(iso['mz'], (int, float)) else "?"
                isomer_lines.append(
                    f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})  |  "
                    f"RT: {rt_str}  |  m/z: {mz_str}"
                )
        
        isomer_str = " // ".join(isomer_lines) if resolved_isomers else "No Isomers Found"
        
        # find number of isomers in isomer_str and add line breaks every 3 isomers for readability
        if isomer_str != "No Isomers Found":
            isomer_list = isomer_str.split(" // ")
            if len(isomer_list) > 3:
                isomer_str = " // <br>".join(
                    [" // ".join(isomer_list[i:i+3]) for i in range(0, len(isomer_list), 3)]
                )


        # --- Add mean EIC trace for rangeslider ---
        mean_eic_rt = row.get("mean_eic_rt", [])
        mean_eic_intensity = row.get("mean_eic_intensity", [])
        if mean_eic_rt and mean_eic_intensity and len(mean_eic_rt) == len(mean_eic_intensity):
            fig.add_trace(go.Scattergl(
                x=mean_eic_rt,
                y=mean_eic_intensity,
                mode="lines",
                name="Mean EIC (slider)",
                line=dict(color="#222", width=2, dash="dot"),
                opacity=0.7,
                hoverinfo="skip",
                showlegend=False,
                visible="legendonly",  # Hide from main plot, but will show in rangeslider
            ))

        # Now add MS1 data traces (they will appear on top of isomer rectangles)
        highlighted_files = state.get("highlighted_files") or []
        has_highlights = bool(highlighted_files)

        # PHASE 4 RADICAL OPTIMIZATION: Combine traces by visual properties to reduce count
        # Group files by (color, line_width, opacity) and create ONE trace per group
        trace_groups = {}

        # Expand the window for MS1 plotting by x min on each side
        expanded_rt_min = state["rt_min"] - 1
        expanded_rt_max = state["rt_max"] + 1

        for fp, fp_group in sub.groupby("file_path", observed=True):
            short_name = re.sub(r"_ms[12]_(?:neg|pos)$", "", "_".join(os.path.basename(fp).split(".")[0].split("_")[11:]))
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in fp.lower()), "gray")
            is_highlighted = fp in highlighted_files
            line_width = 3.5 if is_highlighted else 1.5
            opacity = 1.0 if (is_highlighted or not has_highlights) else 0.25

            # Create group key based on visual properties
            group_key = (color, line_width, opacity)
            if group_key not in trace_groups:
                trace_groups[group_key] = {"x": [], "y": [], "names": [], "files": []}

            # Process all rows for this file (usually just 1)
            for _, r in fp_group.iterrows():
                try:
                    rt, intensity = _parse_spectrum_cached(r["raw_spectrum"])
                    # Only keep points within the expanded RT window
                    filtered = [(x, y) for x, y in zip(rt, intensity) if x is not None and expanded_rt_min <= x <= expanded_rt_max]
                    if filtered:
                        filtered_rt, filtered_intensity = zip(*filtered)
                    else:
                        filtered_rt, filtered_intensity = [], []
                    # Add this file's data to the group, using NaN separator for discontinuity
                    if trace_groups[group_key]["x"] and filtered_rt:  # Not first in group, add separator
                        trace_groups[group_key]["x"].append(None)
                        trace_groups[group_key]["y"].append(None)
                    trace_groups[group_key]["x"].extend(filtered_rt)
                    trace_groups[group_key]["y"].extend(filtered_intensity)
                    trace_groups[group_key]["names"].append(short_name)
                    trace_groups[group_key]["files"].append(fp)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"MS1 parse error {fp}: {e}")

        # Create one Scattergl trace per visual group (typically 2-4 traces instead of 27!)
        for (color, line_width, opacity), data in trace_groups.items():
            if data["x"]:
                group_name = f"{len(data['names'])} files: {', '.join(data['names'][:3])}{'...' if len(data['names']) > 3 else ''}"
                fig.add_trace(go.Scattergl(
                    x=data["x"],
                    y=data["y"],
                    mode="lines",
                    name=group_name,
                    line=dict(color=color, width=line_width),
                    opacity=opacity,
                    hovertemplate="%{x:.3f} min<br>%{y:.2e}",
                    showlegend=False,
                ))

        logger.debug(f"MS1 traces created: {len(trace_groups)} combined traces from {len(sub['file_path'].unique())} files")

        # Atlas RT peak line (black, static)
        fig.add_trace(go.Scatter(
            x=[row["atlas_rt_peak"], row["atlas_rt_peak"]],
            y=[y_bottom, y_max_data],
            mode="lines",
            line=dict(color="black", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Suggested RT lines (orange, static)
        if pd.notnull(row.get("suggested_rt_min")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_min"], row["suggested_rt_min"]],
                y=[y_bottom, y_max_data],
                mode="lines",
                line=dict(color="orange", width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ))
        if pd.notnull(row.get("suggested_rt_max")):
            fig.add_trace(go.Scatter(
                x=[row["suggested_rt_max"], row["suggested_rt_max"]],
                y=[y_bottom, y_max_data],
                mode="lines",
                line=dict(color="orange", width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ))

        # RT min (purple, solid, editable): data-anchored so it matches reference-line height.
        fig.add_shape(
            type="line", x0=rt_min, x1=rt_min, y0=y_bottom, y1=y_max_data,
            xref="x", yref="y",
            line=dict(color="purple", width=7),
            name="RT min", editable=True,
        )

        # RT max (purple, dashed, editable) - MUST be shape[1] to match rt_drag callback
        fig.add_shape(
            type="line", x0=rt_max, x1=rt_max, y0=y_bottom, y1=y_max_data,
            xref="x", yref="y",
            line=dict(color="purple", width=7, dash="dash"),
            name="RT max", editable=True,
        )

        compound_display_idx = state["compound_idx"]+1

        ms1_title_text = (
            f"<span style='font-size:1.2em'>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi_key}</span><br>"
            f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: {row['best_ms1_rt']:.4f}  |  RT Δ: {row['best_ms1_rt_error']:.3f}<br>"
            f"Atlas m/z: {row['atlas_mz']:.4f}  |  ppm Δ: {row['best_ms1_ppm_error']:.2f}<br>"
            f"<sub style='font-size:0.8em'>{isomer_str}</sub>"
        )

        fig.update_layout(
            title=dict(text=ms1_title_text, x=0.5, xanchor="center", font=dict(size=18)),
            xaxis_title="RT",
            yaxis_title="Intensity",
            hovermode="closest", 
            showlegend=False,
            margin=dict(l=50, r=20, t=125, b=40), 
            dragmode="zoom",
            plot_bgcolor="white",
            uirevision=f"ms1-{state['compound_idx']}-{yaxis_scale}",
            xaxis=dict(
                rangeslider=dict(visible=True),
                showgrid=False,
                zeroline=False,
                range=[x_window_min, x_window_max],
                title_font=dict(size=18),
                tickfont=dict(size=15),
                autorange=False,
                rangeslider_thickness=0.12,
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
        """Add MS2 peak sticks with fixed pixel width regardless of x-axis range."""
        if not mz_vals or not intensities:
            return

        # Group segments by color so each trace can keep a single line color.
        grouped = {}
        if colors is not None and len(colors) == len(mz_vals):
            for mz, intensity, color in zip(mz_vals, intensities, colors):
                grouped.setdefault(color or default_color, []).append((mz, intensity))
        else:
            grouped[default_color] = list(zip(mz_vals, intensities))

        for color, pairs in grouped.items():
            x_vals = []
            y_vals = []
            custom_vals = []
            for mz, intensity in pairs:
                x_vals.extend([mz, mz, None])
                y_vals.extend([0.0, intensity, None])
                custom_vals.extend([intensity, intensity, None])

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    customdata=custom_vals,
                    mode="lines",
                    line=dict(color=color, width=line_width_px),
                    showlegend=False,
                    hovertemplate=f"m/z: %{{x:.4f}}<br>Int: %{{customdata:.2e}}<extra>{hover_label}</extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

    def _make_ms2_figure(state):
        row = _compound_row(state["compound_idx"])
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        scans_by_energy = _get_ms2_scans(row, rt_min, rt_max)

        if len(scans_by_energy) == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"{row['compound_name']} - No MS2 data",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(
                margin=dict(l=50, r=20, t=80, b=40),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            return fig

        # Convert collision energy float to string format
        def _format_ce(ce_float):
            """Convert collision energy float to string format like CE102040."""
            if abs(ce_float - 23.333) < 0.01:
                return "CE102040"
            elif abs(ce_float - 43.333) < 0.01:
                return "CE205060"
            else:
                # Fallback: round to nearest integer
                return f"CE{int(round(ce_float))}"

        # Sort collision energies for consistent ordering
        collision_energies = sorted(scans_by_energy.keys())
        n_energies = len(collision_energies)
        
        # Create subplots (side by side) without titles (will add to x-axis instead)
        fig = make_subplots(
            rows=1, cols=n_energies,
            horizontal_spacing=0.08
        )

        ms2_idx = state["ms2_idx"]

        # Process each collision energy subplot
        for col_idx, ce in enumerate(collision_energies, start=1):
            scans = scans_by_energy[ce]
            
            if len(scans) == 0:
                # Plotly uses "x", "y" for first subplot, "x2", "y2", etc. for others
                xref_coord = "x" if col_idx == 1 else f"x{col_idx}"
                yref_coord = "y" if col_idx == 1 else f"y{col_idx}"
                fig.add_annotation(
                    text="No scans",
                    xref=xref_coord, yref=yref_coord,
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=12),
                    row=1, col=col_idx
                )
                continue

            # Clamp index to available scans
            scan_idx = max(0, min(ms2_idx, len(scans) - 1))
            scan = scans.iloc[scan_idx]

            label_points = []
            scale = 1.0
            stick_width_px = 3
            
            qry = scan.get("qry_spectrum") if "qry_spectrum" in scan.index else None
            ref = scan.get("ref_spectrum") if "ref_spectrum" in scan.index else None
            num_ref_fragments = 0
            num_matching_fragments = 0
            
            if pd.notnull(qry) and pd.notnull(ref):
                mz_q, int_q = _parse_spectrum_cached(scan["qry_spectrum"])
                mz_r, int_r = _parse_spectrum_cached(scan["ref_spectrum"])
                scale = (max(int_q) / max(int_r)) if int_q and int_r and max(int_r) > 0 else 1.0
                ref_y = [-i * scale for i in int_r]

                raw_colors = scan.get("aligned_fragment_colors") if "aligned_fragment_colors" in scan.index else None
                frag_colors = None
                num_ref_fragments = len(mz_r)
                if pd.notnull(raw_colors) and raw_colors:
                    frag_colors = json.loads(raw_colors)
                    if len(frag_colors) != len(mz_q):
                        raise ValueError(f"color length mismatch: {len(frag_colors)} != {len(mz_q)}")
                    num_matching_fragments = sum(1 for c in frag_colors if c == "green")

                _add_ms2_stick_traces(
                    fig,
                    mz_q,
                    int_q,
                    hover_label="Query",
                    row_idx=1,
                    col_idx=col_idx,
                    colors=frag_colors,
                    default_color="red",
                    line_width_px=stick_width_px,
                )

                ref_colors = frag_colors if frag_colors is not None and len(frag_colors) == len(mz_r) else None
                _add_ms2_stick_traces(
                    fig,
                    mz_r,
                    ref_y,
                    hover_label="Reference",
                    row_idx=1,
                    col_idx=col_idx,
                    colors=ref_colors,
                    default_color="blue",
                    line_width_px=stick_width_px,
                )

                label_points.extend(zip(mz_q, int_q))
                label_points.extend(zip(mz_r, ref_y))
            else:
                mz, ints = _parse_spectrum_cached(scan["raw_spectrum"])
                _add_ms2_stick_traces(
                    fig,
                    mz,
                    ints,
                    hover_label="MS2",
                    row_idx=1,
                    col_idx=col_idx,
                    default_color="red",
                    line_width_px=stick_width_px,
                )
                label_points.extend(zip(mz, ints))

            # Add horizontal line at y=0
            fig.add_hline(y=0, line=dict(color="black", width=1.5), row=1, col=col_idx)

            # Calculate y-axis range and label positions
            y_vals = [y for _, y in label_points] or [0]
            y_min, y_max = min(y_vals), max(y_vals)
            y_span = max(y_max - y_min, max(abs(y_min), abs(y_max)), 1.0)
            label_pad = y_span * 0.01
            y_pad = y_span * 0.01
            TEXT_HEIGHT_OFFSET = y_span * 0.01

            # PHASE 4 OPTIMIZATION: Reduce peak labels from 5 to 3 for speed
            top_label_idxs = {
                idx
                for idx, _ in sorted(
                    enumerate(label_points),
                    key=lambda item: abs(item[1][1]),
                    reverse=True,
                )[:3]  # Reduced from 5 to 3 for faster rendering
            }

            # Sort top labels by x-position for overlap detection
            top_labels_sorted = sorted(
                [(idx, mz_val, y_val) for idx, (mz_val, y_val) in enumerate(label_points) if idx in top_label_idxs],
                key=lambda item: item[1]
            )

            MIN_MZ_GAP = 5.0
            prev_mz = None
            stagger_level = 0

            # PHASE 4 OPTIMIZATION: Batch annotations to reduce overhead
            annotations_batch = []
            for idx, mz_val, y_val in top_labels_sorted:
                y_base = (y_val + label_pad) if y_val >= 0 else (y_val - label_pad)
                
                if prev_mz is not None and abs(mz_val - prev_mz) < MIN_MZ_GAP:
                    stagger_level += 1
                else:
                    stagger_level = 0
                
                y_position = y_base + (stagger_level * TEXT_HEIGHT_OFFSET if y_val >= 0 else -stagger_level * TEXT_HEIGHT_OFFSET)
                
                # Plotly uses "x", "y" for first subplot, "x2", "y2", etc. for others
                xref_coord = "x" if col_idx == 1 else f"x{col_idx}"
                yref_coord = "y" if col_idx == 1 else f"y{col_idx}"
                
                annotations_batch.append(dict(
                    x=mz_val,
                    y=y_position,
                    text=f"{mz_val:.4f}",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom" if y_val >= 0 else "top",
                    font=dict(size=10, color="black"),
                    textangle=0,
                    xref=xref_coord,
                    yref=yref_coord,
                ))
                prev_mz = mz_val
            
            # Add all annotations at once (more efficient than one-by-one)
            for ann in annotations_batch:
                fig.add_annotation(ann)

            # Update axes for this subplot
            # xaxis_name = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            # yaxis_name = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
            
            ce_label = _format_ce(ce)
            fig.update_xaxes(
                title_text=f"m/z ({ce_label})",
                showgrid=False,
                zeroline=False,
                title_font=dict(size=18),
                tickfont=dict(size=15),
                row=1, col=col_idx
            )
            fig.update_yaxes(
                title_text=f"Intensity (Ref scaled x{scale:.2f})" if col_idx == 1 else "",
                showgrid=False,
                zeroline=False,
                range=[y_min - y_pad, y_max + y_pad],
                title_font=dict(size=18),
                tickfont=dict(size=15),
                row=1, col=col_idx
            )

            # Add subtitle with scan info
            fname = "_".join(os.path.basename(scan.get("file_path", "")).split(".")[0].split("_")[11:])
            # Handle potential column name variations after merge
            prec_mz = scan.get('precursor_MZ', scan.get('precursor_MZ_x', 0))
            scan_info = (
                f"<span style='font-size:1.2em'>"
                f"<b>CoS.: {scan.get('score', 0):.4f}</b>  |  "
                f"Ions: {num_matching_fragments}q/{num_ref_fragments}r  |  "
                f"RT: {scan.get('rt', 0):.4f} min | "
                f"Exp. m/z: {prec_mz:.4f}  |  "
                f"Ref. m/z: {scan.get('mz_theoretical', 0):.4f}  |  "
                f"ppm Δ: {scan.get('ppm_error', 0):.2f}"
                f"</span><br>"
                f"{scan.get('ref_name', 'Unknown')}  |  {fname}<br><br>"
            )
            # Add as annotation below the subplot title
            # Plotly uses "x domain" for first subplot, "x2 domain", "x3 domain" for others
            xref_str = "x domain" if col_idx == 1 else f"x{col_idx} domain"
            yref_str = "y domain" if col_idx == 1 else f"y{col_idx} domain"
            fig.add_annotation(
                text=scan_info,
                xref=xref_str,
                yref=yref_str,
                x=0.5,
                y=1.02,
                showarrow=False,
                font=dict(size=14),
                xanchor="center",
                yanchor="bottom",
            )

        # Overall layout - PHASE 4 OPTIMIZATION: Simplify for speed
        fig.update_layout(
            barmode="overlay",
            hovermode="closest",
            margin=dict(l=50, r=20, t=120, b=40),
            plot_bgcolor="white",
            height=550,
            showlegend=False,  # Disable legend for speed
        )

        return fig

    logger.debug("App helpers defined successfully")

    # all app callbacks that fire when GUI is interacted with
    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("compound-dd", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def init_store(compound_idx, old_state):
        """Handles compound changes initiated directly from the dropdown widget.

        Button and keyboard navigation now flush+load atomically inside their own
        callbacks (navigate_compound / handle_keyboard), so by the time compound-dd
        is updated by those callbacks, session-store already holds the new compound's
        state.  The guard below (compound_idx == old_state.compound_idx) detects
        that case and short-circuits to prevent a double-flush.

        This callback only does real work when the analyst picks a compound directly
        from the dropdown.
        """
        if compound_idx is None:
            raise dash.exceptions.PreventUpdate

        compound_idx = int(compound_idx)

        if old_state is not None and int(old_state.get("compound_idx", -1)) == compound_idx:
            raise dash.exceptions.PreventUpdate

        flush_error = None
        if old_state is not None:
            try:
                old_state = _flush_to_db(old_state)
            except Exception as exc:
                traceback.print_exc()
                logger.error(f"init_store: _flush_to_db failed: {exc}")
                flush_error = f"Save failed: {type(exc).__name__}: {exc}"

        # OPTIMIZED: Don't clear entire cache, let it grow to reasonable size
        # Only clear if cache gets too large (> 1000 entries)
        if len(ms2_scans_cache) > 1000:
            ms2_scans_cache.clear()
        
        new_state = _load_state(
            compound_idx,
            session_id=(old_state or {}).get("session_id"),
            edit_seq=(old_state or {}).get("edit_seq", 0),
        )
        new_state["last_saved"] = (old_state or {}).get("last_saved")
        new_state["flush_error"] = flush_error
        return new_state

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
        """Atomic navigation: flush current compound to DB, then load the next.

        Writing both session-store and compound-dd in a single callback output
        eliminates the 2-hop race (button → compound-dd → init_store) where an
        in-flight Patch from a UI-change callback could be flushed with stale data.
        init_store guards against re-firing via its compound_idx == old_idx check.
        """
        trigger = ctx.triggered_id
        if trigger not in ("prev-btn", "next-btn") or state is None:
            raise dash.exceptions.PreventUpdate

        delta = -1 if trigger == "prev-btn" else 1
        new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)
        if new_idx == int(state["compound_idx"]):
            raise dash.exceptions.PreventUpdate

        flush_error = None
        ms2_warning, ms1_warning = _get_force_eval_and_warnings(state, delta)

        try:
            state = _flush_to_db(state)
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"navigate_compound: _flush_to_db failed: {exc}")
            flush_error = f"Save failed: {type(exc).__name__}: {exc}"

        # OPTIMIZED: Don't clear entire cache, let it grow to reasonable size
        # Only clear if cache gets too large (> 1000 entries)
        if len(ms2_scans_cache) > 1000:
            ms2_scans_cache.clear()
        
        new_state = _load_state(
            new_idx,
            session_id=state.get("session_id"),
            edit_seq=state.get("edit_seq", 0),
        )
        new_state["last_saved"] = state.get("last_saved")
        new_state["flush_error"] = flush_error
        new_state["_nav_programmatic"] = True
        if ms2_warning:
            new_state["ms2_warning"] = ms2_warning
        else:
            new_state.pop("ms2_warning", None)
        if ms1_warning:
            new_state["ms1_warning"] = ms1_warning
        else:
            new_state.pop("ms1_warning", None)
        return new_state, new_idx

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms2-prev", "n_clicks"),
        Input("ms2-next", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_ms2(prev, nxt, state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered_id
        row = _compound_row(state["compound_idx"])
        n_scans = _count_ms2_scans(row, state["rt_min"], state["rt_max"])
        if n_scans == 0:
            raise dash.exceptions.PreventUpdate
        delta = -1 if trigger == "ms2-prev" else 1
        new_idx = max(0, min(state["ms2_idx"] + delta, n_scans - 1))
        return _patch_with_seq(state, ms2_idx=new_idx)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("accept-suggestions", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def accept_suggestions(_, state):
        row = _compound_row(state["compound_idx"])
        if pd.notnull(row.get("suggested_rt_min")):
            return _patch_rt_change(state, float(row["suggested_rt_min"]), float(row["suggested_rt_max"]))
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("snap-to-isomer", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def snap_to_isomer(_, state):
        if state is None:
            raise dash.exceptions.PreventUpdate
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
        
        # PHASE 2 OPTIMIZATION: Threshold check to avoid unnecessary recomputation
        # Only update if change is significant (> 0.001 min = 0.06 seconds)
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
            | {"a", "s", "d", "f", "j", "k", "l", ";", "n", "m", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"}
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
            logger.error(f"handle_keyboard error (key={key}, compound={state.get('compound_idx')}, tag={tag}): {exc}")
            raise dash.exceptions.PreventUpdate

    def _handle_keyboard_inner(event, state, key):  # noqa: ARG001
        rt_min, rt_max = state["rt_min"], state["rt_max"]
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
            logger.debug(f"[_handle_keyboard_inner] RT nudge key={key}, new rt_min={rt_min}, new rt_max={rt_max}")
            return _patch_rt_change(state, rt_min, rt_max), dash.no_update

        if key in ("l", "ArrowUp"):
            row = _compound_row(state["compound_idx"])
            n_scans = _count_ms2_scans(row, state["rt_min"], state["rt_max"])
            if n_scans == 0:
                raise dash.exceptions.PreventUpdate
            return _patch_with_seq(state, ms2_idx=max(state["ms2_idx"] - 1, 0)), dash.no_update

        if key in (";", "ArrowDown"):
            row = _compound_row(state["compound_idx"])
            n_scans = _count_ms2_scans(row, state["rt_min"], state["rt_max"])
            if n_scans == 0:
                raise dash.exceptions.PreventUpdate
            return _patch_with_seq(state, ms2_idx=min(state["ms2_idx"] + 1, n_scans - 1)), dash.no_update

        if key == "n":
            row = _compound_row(state["compound_idx"])
            if pd.notnull(row.get("suggested_rt_min")):
                return _patch_rt_change(
                    state,
                    float(row["suggested_rt_min"]),
                    float(row["suggested_rt_max"]),
                ), dash.no_update
            raise dash.exceptions.PreventUpdate

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

        if key in ("j", "k", "ArrowLeft", "ArrowRight"):
            if key == "k" or key == "ArrowRight":
                delta = 1
            elif key == "j" or key == "ArrowLeft":
                delta = -1
            new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)
            if new_idx == int(state["compound_idx"]):
                raise dash.exceptions.PreventUpdate
            flush_error = None
            ms2_warning, ms1_warning = _get_force_eval_and_warnings(state, delta)
            try:
                state = _flush_to_db(state)
            except Exception as exc:
                traceback.print_exc()
                logger.error(f"handle_keyboard navigation _flush_to_db failed: {exc}")
                flush_error = f"Save failed: {type(exc).__name__}: {exc}"
            # OPTIMIZED: Don't clear entire cache, let it grow to reasonable size
            # Only clear if cache gets too large (> 1000 entries)
            if len(ms2_scans_cache) > 1000:
                ms2_scans_cache.clear()
            
            new_state = _load_state(
                new_idx,
                session_id=state.get("session_id"),
                edit_seq=state.get("edit_seq", 0),
            )
            new_state["last_saved"] = state.get("last_saved")
            new_state["flush_error"] = flush_error
            new_state["_nav_programmatic"] = True
            if ms2_warning:
                new_state["ms2_warning"] = ms2_warning
            if ms1_warning:
                new_state["ms1_warning"] = ms1_warning
            return new_state, new_idx

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
        if state is None:
            raise dash.exceptions.PreventUpdate
        
        # PHASE 4 OPTIMIZATION: Add timing logs to diagnose remaining delays
        import time
        update_start = time.time()
        
        # PHASE 4 RADICAL OPTIMIZATION: Separate data cache from RT-dependent elements
        # Cache key includes only fields that affect DATA (not RT lines which are fast to update)
        compound_idx = state.get("compound_idx")
        rt_min = state.get("rt_min", 0)
        rt_max = state.get("rt_max", 0)
        ms2_idx = state.get("ms2_idx", 0)
        highlighted_files = tuple(sorted(state.get("highlighted_files") or []))
        
        # For MS1: INCLUDE RT bounds in cache key so figure is regenerated when window changes
        # Round RT to 0.01 min (0.6 sec) to increase cache hits during small adjustments
        rt_min_rounded = round(float(rt_min) * 100) / 100  # 0.01 min resolution
        rt_max_rounded = round(float(rt_max) * 100) / 100

        ms1_cache_key = (compound_idx, yaxis_scale, highlighted_files, rt_min_rounded, rt_max_rounded)
        ms2_cache_key = (compound_idx, rt_min_rounded, rt_max_rounded, ms2_idx)
        
        # Debug logging to identify cache misses
        logger.debug(f"Cache keys - MS1: {ms1_cache_key}, MS2: {ms2_cache_key}")
        
        with figure_cache_lock:
            ms1_cached = ms1_cache_key in figure_cache
            ms2_cached = ms2_cache_key in figure_cache
            
            if ms1_cached and ms2_cached:
                cached_ms1 = figure_cache[ms1_cache_key]
                cached_ms2 = figure_cache[ms2_cache_key]
                cache_time = time.time() - update_start
                logger.debug(f"Figure cache FULL HIT for compound {compound_idx} ({cache_time:.3f}s)")
                # Return cached figures with current banner state
                flush_err = state.get("flush_error")
                ms2_warning = state.get("ms2_warning")
                ms1_warning = state.get("ms1_warning")
                banners = []
                if flush_err:
                    banners.append(html.Div(
                        f"⚠ {flush_err}",
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
                return cached_ms1, cached_ms2, banners
        
        # Partial or full cache miss - regenerate as needed
        logger.debug(f"Figure cache MISS for compound {compound_idx} (MS1={'HIT' if ms1_cached else 'MISS'}, MS2={'HIT' if ms2_cached else 'MISS'}), regenerating...")
        
        flush_err = state.get("flush_error")
        ms2_warning = state.get("ms2_warning")
        ms1_warning = state.get("ms1_warning")
        try:
            # Generate only what's not cached
            if not ms1_cached:
                ms1_start = time.time()
                ms1_fig = _make_ms1_figure(state, yaxis_scale)
                ms1_time = time.time() - ms1_start
                logger.debug(f"MS1 figure generated in {ms1_time:.3f}s")
                with figure_cache_lock:
                    figure_cache[ms1_cache_key] = ms1_fig
            else:
                ms1_fig = figure_cache[ms1_cache_key]
            
            if not ms2_cached:
                ms2_start = time.time()
                ms2_fig = _make_ms2_figure(state)
                ms2_time = time.time() - ms2_start
                logger.debug(f"MS2 figure generated in {ms2_time:.3f}s")
                with figure_cache_lock:
                    figure_cache[ms2_cache_key] = ms2_fig
            else:
                ms2_fig = figure_cache[ms2_cache_key]
            
            total_time = time.time() - update_start
            logger.debug(f"Total update_figures time: {total_time:.3f}s")
            
            # Limit cache size
            with figure_cache_lock:
                if len(figure_cache) > 100:  # Increased from 50
                    # Remove oldest entries (keep last 80)
                    keys_to_remove = list(figure_cache.keys())[:20]
                    for key in keys_to_remove:
                        figure_cache.pop(key, None)
            
            banners = []
            if flush_err:
                banners.append(html.Div(
                    f"⚠ {flush_err}",
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
                f"⚠ Figure error: {type(exc).__name__}: {exc}",
                style={"color": "red", "fontSize": "11px", "fontWeight": "bold"},
            )
            empty = go.Figure()
            empty.update_layout(margin=dict(l=50, r=20, t=40, b=40))
            return empty, empty, err_html

    @app.callback(
        Output("status-current", "children"),
        Output("status-previous", "children"),
        Output("compound-counter", "children"),
        Output("ms2-counter", "children"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_status(state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        row = _compound_row(state["compound_idx"])
        comp_txt = f"Compound {state['compound_idx']+1} of {len(compound_options)}"
        
        # Get scans by collision energy for detailed count
        scans_by_energy = _get_ms2_scans(row, state["rt_min"], state["rt_max"])
        if scans_by_energy:
            ce_counts = {ce: len(df) for ce, df in scans_by_energy.items()}
            max_scans = max(ce_counts.values())
            ce_info = ", ".join([f"CE {ce}: {count}" for ce, count in sorted(ce_counts.items())])
            ms2_txt = f"MS2 Scan {state['ms2_idx']+1} of {max_scans}"
        else:
            ms2_txt = "No MS2 data"
        
        pending = html.Span(
            ["Unsaved (current): ", html.I(row["compound_name"]),
             f"  |  RT [{state['rt_min']:.4f}, {state['rt_max']:.4f}]",
             f"  |  MS1: {state['ms1_note']}  |  MS2: {state['ms2_note']}  |  Other: {state['other_note']}",
             f"  |  Analyst Notes: {state['analyst_notes'][:30]}{'...' if len(state['analyst_notes']) > 30 else ''}"],
            style={"color": "#b8860b", "fontSize": "11px", "fontWeight": "bold"},
        )
        if state.get("last_saved"):
            s = state["last_saved"]
            saved_analyst = s.get("analyst_notes", "")
            saved = html.Span(
                ["Saved (previous): ", html.I(s["name"]),
                 f"  |  RT [{s['rt_min']:.4f}, {s['rt_max']:.4f}] ",
                 f"  |  MS1: {s['ms1']}  |  MS2: {s['ms2']}  |  Other: {s['other']}  |  @ {s['timestamp']}",
                 f"  |  Analyst Notes: {saved_analyst[:30]}{'...' if len(saved_analyst) > 30 else ''}"],
                style={"color": "#2a7a2a", "fontSize": "11px", "fontWeight": "bold"},
            )
        else:
            saved = html.Span("Previous: NA", style={"color": "#888", "fontSize": "11px"})
        return pending, saved, comp_txt, ms2_txt

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
        if state is None:
            # Return default values if state is missing
            return "", "No identification notes", analysis_gui_obj.notes["ms1_notes"][0], analysis_gui_obj.notes["ms2_notes"][0], [], 0
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