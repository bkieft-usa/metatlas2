import functools, json, os, time, uuid, threading
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
logger = lcf.get_logger("analysis_gui")

def _get_notes_opts(owner: str = "jgi") -> tuple[dict[str, str], dict[str, str], dict[str, str]]:

    JGI_DEFAULT_MS2_HOTKEYS = {
        "-1.0, poor match, should remove": "w",
        "0.0, no match or no MSMS collected": "e",
        "0.5, partial or putative match of fragments": "r",
        "1.0, good match": "t",
        "0.5, co-isolated precursor, partial match": "y",
        "1.0, co-isolated precursor, good match": "u",
        "0.5, single ion match, no evidence": "i",
        "1.0, single ion match, ISTD/ref evidence": "o",
    }
    JGI_DEFAULT_MS1_HOTKEYS = {
        "keep": "p",
        "remove": "q",
    }
    JGI_DEFAULT_OTHER_HOTKEYS = {
        "unresolvable isomers": "1",
        "poor peak shape": "2",
        "potential rt shifting":  "3",
        "high ppm diff": "4",
        "noisy or high background": "5",
        "needs review": "6",
        "contains refstd files": "7",
    }
    EGSB_DEFAULT_MS2_HOTKEYS = {
        "-1.0, poor match, should remove": "w",
        "0.0, no match or no MSMS collected": "e",
        "0.5, partial or putative match of fragments": "r",
        "1.0, good match": "t",
        "0.5, co-isolated precursor, partial match": "y",
        "1.0, co-isolated precursor, good match": "u",
        "0.5, single ion match, no evidence": "i",
        "1.0, single ion match, ISTD/ref evidence": "o",
    }
    EGSB_DEFAULT_MS1_HOTKEYS = {
        "no selection": "1",
        "OK - single peak at correct RT": "2",
        "OK - single peak at shifted RT": "3",
        "OK - multiple peaks, unresolvable": "4",
        "OK - multiple peaks, resolvable": "5",
        "Remove - background level/noise": "6",
        "Remove - signal in ExCtrl >= sample": "7",
        "Remove - bad MSMS": "8",
        "Remove - ND": "9",
        "Remove - Not Evaluated": "0",
        "Remove - duplicate": "-",
        "Remove - evidence of contamination or incorrect ID": "=",
    }
    EGSB_DEFAULT_OTHER_HOTKEYS = {
        "unresolvable isomers": "z",
        "poor peak shape": "x",
        "potential rt shifting":  "c",
        "high ppm diff": "v",
        "noisy or high background": "b",
        "needs review": "g",
        "contains refstd files": "h",
    }
        
    if owner.lower() == "egsb":
        return EGSB_DEFAULT_MS2_HOTKEYS, EGSB_DEFAULT_MS1_HOTKEYS, EGSB_DEFAULT_OTHER_HOTKEYS
    else:
        return JGI_DEFAULT_MS2_HOTKEYS, JGI_DEFAULT_MS1_HOTKEYS, JGI_DEFAULT_OTHER_HOTKEYS

def get_note_options_and_hotkeys(override_dict, default_hotkeys):
    if isinstance(override_dict, dict) and len(override_dict) > 0:
        # Use analyst-supplied dictionary (text: hotkey)
        hotkeys = dict(override_dict)
    else:
        hotkeys = dict(default_hotkeys)
    options = list(hotkeys.keys())
    return options, hotkeys

def _validate_override_parameters(override_parameters):
    if not isinstance(override_parameters, dict):
        raise ValueError("analysis_gui_obj.override_parameters must be a dict")
    if not isinstance(override_parameters["gui_lcmsruns_colors"], (type(None), dict)):
        raise ValueError("override_parameters['gui_lcmsruns_colors'] must be a dict mapping LCMS run identifiers to color strings or None")
    if not isinstance(override_parameters["gui_require_all_evaluated"], (type(None), bool)):
        raise ValueError("override_parameters['gui_require_all_evaluated'] must be a boolean or None")
    if not isinstance(override_parameters["ms1_min_peak_intensity"], (type(None), (int, float))):
        raise ValueError("override_parameters['ms1_min_peak_intensity'] must be a number or None")
    if not isinstance(override_parameters["ms1_min_num_points"], (type(None), int)):
        raise ValueError("override_parameters['ms1_min_num_points'] must be an integer or None")
    if not isinstance(override_parameters["ms2_min_score"], (type(None), (int, float))):
        raise ValueError("override_parameters['ms2_min_score'] must be a number or None")
    if not isinstance(override_parameters["ms2_min_matching_frags"], (type(None), int)):
        raise ValueError("override_parameters['ms2_min_matching_frags'] must be an integer or None")
    if not isinstance(override_parameters.get("remove_unided_compounds"), (type(None), bool)):
        raise ValueError("override_parameters['remove_unided_compounds'] must be a boolean or None")
    if not isinstance(override_parameters.get("apply_istd_to_ema"), (type(None), bool)):
        raise ValueError("override_parameters['apply_istd_to_ema'] must be a boolean or None")
    if not isinstance(override_parameters.get("remove_flagged_compounds"), (type(None), bool)):
        raise ValueError("override_parameters['remove_flagged_compounds'] must be a boolean or None")
    if not isinstance(override_parameters.get("gui_top_n_hits"), (type(None), int)):
        raise ValueError("override_parameters['gui_top_n_hits'] must be an integer or None")
    if not isinstance(override_parameters["note_options_overrides"], (type(None), dict)):
        raise ValueError("override_parameters['note_options_overrides'] must be a dict mapping note types to option dicts or None")
    if isinstance(override_parameters["note_options_overrides"], dict):
        for note_type, options in override_parameters["note_options_overrides"].items():
            if note_type not in ["ms1_notes", "ms2_notes", "other_notes"]:
                raise ValueError(f"Invalid note type in note_options_overrides: {note_type} (must be 'ms1_notes', 'ms2_notes', or 'other_notes')")
            if not isinstance(options, dict):
                raise ValueError(f"Options for {note_type} in note_options_overrides must be a dict mapping option text to hotkeys")
            for opt_text, hotkey in options.items():
                if not isinstance(opt_text, str) or not isinstance(hotkey, str):
                    raise ValueError(f"Invalid option in note_options_overrides for {note_type}: {opt_text}: {hotkey} (both must be strings)")

def build_dash_app(
    analysis_gui_obj,
    port=8050,
    shutdown_holder=None
):
    logger.debug("Starting the app factory for the Analysis GUI...")

    # Set up basic GUI params
    manual_curation_df = analysis_gui_obj.manual_curation_df
    top_n_hits = analysis_gui_obj.workflow_params.get("gui_top_n_hits", 20)
    if analysis_gui_obj.override_parameters.get("gui_top_n_hits") is not None:
        top_n_hits = analysis_gui_obj.override_parameters["gui_top_n_hits"]

    logger.info(f"Analysis starting with {len(manual_curation_df)} compounds:")

    # Extract metadata for display
    chrom = analysis_gui_obj.post_autoid_atlas_obj.chromatography
    pol = analysis_gui_obj.post_autoid_atlas_obj.polarity
    analysis_type = analysis_gui_obj.post_autoid_atlas_obj.analysis_type
    rta = analysis_gui_obj.rt_alignment_number
    tga = analysis_gui_obj.analysis_number

    # Set up all passing compounds as options for the dropdown
    compound_options = [
        {"label": f"{i}: {row['compound_name']}", "value": i}
        for i, row in manual_curation_df.reset_index().iterrows()
    ]

    # Allow override of note options/hotkeys from override_parameters
    owner = analysis_gui_obj.config.get('WORKFLOWS').get('PATHS').get('owner', None).lower()
    ms2_notes_opts, ms1_notes_opts, other_notes_opts = _get_notes_opts(owner = owner)
    _validate_override_parameters(analysis_gui_obj.override_parameters)
    ms1_options, ms1_hotkeys = get_note_options_and_hotkeys(
        analysis_gui_obj.override_parameters["note_options_overrides"].get("ms1_notes", {}) if analysis_gui_obj.override_parameters.get("note_options_overrides") else {},
        ms1_notes_opts,
    )
    ms2_options, ms2_hotkeys = get_note_options_and_hotkeys(
        analysis_gui_obj.override_parameters["note_options_overrides"].get("ms2_notes", {}) if analysis_gui_obj.override_parameters.get("note_options_overrides") else {},
        ms2_notes_opts,
    )
    other_options, other_hotkeys = get_note_options_and_hotkeys(
        analysis_gui_obj.override_parameters["note_options_overrides"].get("other_notes", {}) if analysis_gui_obj.override_parameters.get("note_options_overrides") else {},
        other_notes_opts,
    )

    ms1_key_to_label = {v: k for k, v in ms1_hotkeys.items()}
    ms2_key_to_label = {v: k for k, v in ms2_hotkeys.items()}
    other_key_to_label = {v: k for k, v in other_hotkeys.items()}

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

    def _compound_row(idx):
        return manual_curation_df.iloc[idx]

    def _rt_bounds_from_row(row):
        return float(row["rt_min"]), float(row["rt_max"])

    def _load_state(compound_idx, ms2_idx=0, session_id=None, edit_seq=0):
        row = _compound_row(compound_idx)
        rt_min, rt_max = _rt_bounds_from_row(row)
        ms2_note = row.get("ms2_notes") or ""
        ms1_note = row.get("ms1_notes") or ms1_options[0]
        other_notes_raw = row.get("other_notes")
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
        other_note = [v for v in other_note if v in other_options]
        
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
        }

    def _patch_with_seq(state, **changes):
        new_state = dict(state)
        new_state.update(changes)
        new_state["edit_seq"] = int(state.get("edit_seq", 0)) + 1
        return new_state

    def _patch_rt_change(state, new_min, new_max):
        rt_min = max(0.0, min(new_min, new_max))
        rt_max = max(rt_min, new_max)
        return _patch_with_seq(state, rt_min=round(rt_min, 4), rt_max=round(rt_max, 4))

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
        logger.info(f"Resuming analysis at compound {starting_compound_idx}: "
                   f"{manual_curation_df.iloc[starting_compound_idx]['compound_name']}")
    else:
        logger.info(f"Starting new analysis at compound 0")

    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp", "target.tagName"]}],
    )

    # format of the app itself
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
                                        options=[{"label": f"[{ms1_hotkeys[lbl]}] {lbl}", "value": lbl} for lbl in ms1_options],
                                        value="keep" if "keep" in ms1_options else ms1_options[0],
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
                                        options=[{"label": f"[{ms2_hotkeys[val]}] {val}", "value": val} for val in ms2_options],
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
                                        options=[{"label": f"[{other_hotkeys[val]}] {val}", "value": val} for val in other_options],
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
                                config={"displayModeBar": True, "edits": {"shapePosition": True, "titleText": False}},
                                style={"height": "550px"},
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
                                style={"height": "550px"}
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
            mask = (
                (manual_curation_df["inchi_key"] == iso.get("inchi_key", "")) &
                (manual_curation_df["compound_name"] == iso.get("compound_name", "")) &
                (manual_curation_df["adduct"] == iso.get("adduct", ""))
            )
            isomer_match = manual_curation_df[mask]
            if isomer_match.empty:
                continue
            bounds.append((float(isomer_match.iloc[0]["rt_min"]), float(isomer_match.iloc[0]["rt_max"])))
        return sorted(bounds, key=lambda x: x[0])

    def _get_ms2_scans(inchi_key, adduct, rt_min=None, rt_max=None):
        """Return dictionary of sorted, capped scans DataFrames grouped by collision energy.

        Returns a dict: {collision_energy: DataFrame} with top N hits from each collision energy.
        Results are cached by (inchi_key, adduct, top_n_hits, rt_min, rt_max).
        """
        key = (inchi_key, adduct, int(top_n_hits),
               round(rt_min, 4) if rt_min is not None else None,
               round(rt_max, 4) if rt_max is not None else None)
        with cache_lock:
            if key in ms2_scans_cache:
                return ms2_scans_cache[key]

        ms2_sub = analysis_gui_obj.ms2_df[(analysis_gui_obj.ms2_df["inchi_key"] == inchi_key) & (analysis_gui_obj.ms2_df["adduct"] == adduct)]
        if rt_min is not None and rt_max is not None:
            ms2_sub = ms2_sub[(ms2_sub["rt"] >= rt_min) & (ms2_sub["rt"] <= rt_max)]
        hits_sub = analysis_gui_obj.ms2_hits_df[
            (analysis_gui_obj.ms2_hits_df["inchi_key"] == inchi_key)
            & (analysis_gui_obj.ms2_hits_df["adduct"] == adduct)
        ]
        if rt_min is not None and rt_max is not None:
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
            merged = pd.merge(ms2_sub, hits_sub, on=["inchi_key", "adduct", "file_path", "rt"], how="left")
            merged["score"] = merged["score"].fillna(0)
            for ce, group in merged.groupby("collision_energy"):
                scans_by_energy[ce] = group.sort_values(["score", "rt"], ascending=[False, True]).head(top_n_hits)

        with cache_lock:
            if key not in ms2_scans_cache:
                ms2_scans_cache[key] = scans_by_energy
        return ms2_scans_cache[key]

    def _count_ms2_scans(row, rt_min=None, rt_max=None):
        scans_by_energy = _get_ms2_scans(row["inchi_key"], row["adduct"], rt_min, rt_max)
        # Return max count across all collision energies for navigation
        return max((len(df) for df in scans_by_energy.values()), default=0)

    def _clean_spectrum(val_arr, int_arr):
        out_i = [0 if (isinstance(i, float) and np.isnan(i)) else i for i in int_arr]
        return val_arr, out_i

    @functools.lru_cache(maxsize=None)
    def _parse_spectrum_cached(raw_spectrum):
        val, ints = json.loads(raw_spectrum)
        val, ints = _clean_spectrum(val, ints)
        return val, ints

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
        updates = {
            "rt_min": state["rt_min"],
            "rt_max": state["rt_max"],
            "rt_peak": (state["rt_min"] + state["rt_max"]) / 2,
            "ms2_notes": state.get("ms2_note") or "",
            "ms1_notes": state["ms1_note"],
            "other_notes": " // ".join(state["other_note"]) if state.get("other_note") else "",
            "analyst_notes": state["analyst_notes"],
            "identification_notes": state["id_notes"],
        }

        # DB write outside lock
        dbi.write_gui_updates_to_db(analysis_gui_obj.paths["project_db_path"], row["curation_uid"], updates)

        idx = state["compound_idx"]
        df_idx = manual_curation_df.index[idx]
        with flush_lock:
            for col, val in updates.items():
                if col in manual_curation_df.columns:
                    manual_curation_df.at[df_idx, col] = val

        state["last_saved"] = {
            "name": row["compound_name"],
            "rt_min": state["rt_min"],
            "rt_max": state["rt_max"],
            "ms1": state["ms1_note"],
            "ms2": state["ms2_note"],
            "other": state["other_note"],
            "analyst_notes": state.get("analyst_notes", ""),
            "id_notes": state.get("id_notes", ""),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        state["flush_error"] = None
        return state

    # main figures for ms data display
    def _make_ms1_figure(state, yaxis_scale="linear"):
        y_bottom = 0.0

        if analysis_gui_obj.override_parameters['gui_lcmsruns_colors'] is not None:
            lcmsruns_color_map = analysis_gui_obj.override_parameters['gui_lcmsruns_colors']
        else:
            lcmsruns_color_map = {
                'ISTD': 'blue', 
                'QC': 'blue', 
                'EXCTRL': 'red', 
                'TXCTRL': 'red', 
                'REFSTD': 'black'
            }
    
        row = _compound_row(state["compound_idx"])
        inchi, adduct = row["inchi_key"], row["adduct"]
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        sub = analysis_gui_obj.ms1_df[(analysis_gui_obj.ms1_df["inchi_key"] == inchi) & (analysis_gui_obj.ms1_df["adduct"] == adduct)]

        # Collect y_max from data so static vertical traces span the full plot height
        y_max_data = 0.0
        # First pass: calculate y_max_data from all MS1 spectra
        for fp in sub["file_path"].unique():
            for _, r in sub[sub["file_path"] == fp].iterrows():
                try:
                    rt, intensity = _parse_spectrum_cached(r["raw_spectrum"])
                    if intensity:
                        y_max_data = max(y_max_data, max(intensity))
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"MS1 parse error {fp}: {e}")
        if y_max_data == 0.0:
            y_max_data = 1.0

        fig = go.Figure()
        
        isomer_str = "No Isomers Found"
        try:
            isomers = _parse_isomers(row.get("isomers")) # all study isomers for this compound
            if isomers:
                isomer_lines = []
                resolved_isomers = []
                # First, build resolved_isomers as before
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
                        raise ValueError(f"There were multiple isomer_match of isomer {iso_name} {iso_adduct} {iso_inchi} to the manual curation object")
                    if isomer_match.empty:
                        continue
                    if isomer_match.iloc[0]["ms1_notes"] == "remove":
                        continue
                    iso_df_idx = isomer_match.index[0]
                    iso_display_idx = iso_df_idx
                    iso_rt_min = float(isomer_match.iloc[0]["rt_min"])
                    iso_rt_max = float(isomer_match.iloc[0]["rt_max"])
                    resolved_isomers.append({
                        "display_idx": iso_display_idx,
                        "name": iso_name,
                        "adduct": iso_adduct,
                        "rt_min": iso_rt_min,
                        "rt_max": iso_rt_max,
                        "rt": iso_rt,
                        "mz": iso_mz,
                        "df_idx": iso_df_idx,
                    })
                # Now, for each isomer, check if its window overlaps with any other
                if resolved_isomers:
                    def _window_overlaps(a_min, a_max, b_min, b_max):
                        return (a_min <= b_max) and (b_min <= a_max)

                    # The current compound's RT window
                    current_rt_min = state["rt_min"]
                    current_rt_max = state["rt_max"]

                    for i, iso in enumerate(resolved_isomers):
                        overlaps = False
                        # Check overlap with current compound window (skip self if this is the current compound)
                        if _window_overlaps(iso["rt_min"], iso["rt_max"], current_rt_min, current_rt_max):
                            overlaps = True
                        # Check overlap with other isomers
                        for j, other in enumerate(resolved_isomers):
                            if i == j:
                                continue
                            if _window_overlaps(iso["rt_min"], iso["rt_max"], other["rt_min"], other["rt_max"]):
                                overlaps = True
                                break
                        fillcolor = "rgba(255,128,128,0.35)" if overlaps else "rgba(128,192,255,0.35)"  # light red if overlaps, else light blue
                        iso_rect_trace = go.Scatter(
                            x=[iso["rt_min"], iso["rt_min"], iso["rt_max"], iso["rt_max"], iso["rt_min"]],
                            y=[y_bottom, y_max_data, y_max_data, y_bottom, y_bottom],
                            mode="lines",
                            fill="toself",
                            fillcolor=fillcolor,
                            line=dict(width=0, color="rgba(0,0,0,0)"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                        fig.add_trace(iso_rect_trace)
                        fig.add_annotation(
                            x=iso["rt_min"],
                            y=0.5,
                            xref="x", yref="paper",
                            text=f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})",
                            showarrow=False,
                            font=dict(size=8, color="dimgray"),
                            xanchor="right", yanchor="middle",
                            textangle=-90,
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="gray", borderwidth=1,
                            captureevents=False,
                        )
                        rt_str = f"{iso['rt']:.3f}" if isinstance(iso['rt'], (int, float)) else "?"
                        mz_str = f"{iso['mz']:.4f}" if isinstance(iso['mz'], (int, float)) else "?"
                        isomer_lines.append(
                            f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})  |  "
                            f"RT: {rt_str}  |  m/z: {mz_str}"
                        )
                    isomer_str = " // ".join(isomer_lines) if resolved_isomers else "No Isomers Found"
                else:
                    isomer_str = "No Isomers Found"
            else:
                isomer_str = "No Isomers Found"
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Isomer detection failed with {exc}")
        # find number of isomers in isomer_str and add line breaks every 3 isomers for readability
        if isomer_str != "No Isomers Found":
            isomer_list = isomer_str.split(" // ")
            if len(isomer_list) > 3:
                isomer_str = " // <br>".join(
                    [" // ".join(isomer_list[i:i+3]) for i in range(0, len(isomer_list), 3)]
                )

        # Now add MS1 data traces (they will appear on top of isomer rectangles)
        for fp in sub["file_path"].unique():
            short_name = "_".join(os.path.basename(fp).split(".")[0].split("_")[11:])
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in fp.lower()), "gray")
            for _, r in sub[sub["file_path"] == fp].iterrows():
                try:
                    rt, intensity = _parse_spectrum_cached(r["raw_spectrum"])
                    fig.add_trace(go.Scatter(
                        x=rt, y=intensity, mode="lines", name=short_name,
                        line=dict(color=color, width=1.5),
                        hovertemplate="%{x:.3f} min<br>%{y:.2e}",
                    ))
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"MS1 parse error {fp}: {e}")

        # Atlas RT peak line (black, static) - rendered as a trace so it is never draggable.
        # config.edits.shapePosition=True (needed for purple shape dragging) makes ALL shapes
        # draggable, so static reference lines must be traces instead.
        fig.add_trace(go.Scatter(
            x=[row["atlas_rt_peak"], row["atlas_rt_peak"]],
            y=[y_bottom, y_max_data],
            mode="lines",
            line=dict(color="black", width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Suggested RT lines (orange, static) - also traces for the same reason
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

        # RT min (purple, solid, editable) - MUST be shape[0] to match rt_drag callback
        fig.add_shape(
            type="line", x0=rt_min, x1=rt_min, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="purple", width=5),
            name="RT min", editable=True,
        )

        # RT max (purple, dashed, editable) - MUST be shape[1] to match rt_drag callback
        fig.add_shape(
            type="line", x0=rt_max, x1=rt_max, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="purple", width=5, dash="dash"),
            name="RT max", editable=True,
        )

        compound_display_idx = state["compound_idx"]

        ms1_title_text = (
            f"<span style='font-size:1.2em'>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi}</span><br>"
            f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: {row['best_ms1_rt']:.4f}  |  RT Δ: {row['best_ms1_rt_error']:.3f}<br>"
            f"Atlas m/z: {row['atlas_mz']:.4f}  |  ppm Δ: {row['best_ms1_ppm_error']:.2f}<br>"
            f"<sub style='font-size:0.8em'>Isomers: {isomer_str}</sub>"
        )

        fig.update_layout(
            title=dict(text=ms1_title_text, x=0.5, xanchor="center", font=dict(size=18)),
            xaxis_title="RT",
            yaxis_title="Intensity",
            hovermode="closest", showlegend=False,
            margin=dict(l=50, r=20, t=125, b=40), dragmode="pan",
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                title_font=dict(size=18),
                tickfont=dict(size=15),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                type=yaxis_scale,
                title_font=dict(size=18),
                tickfont=dict(size=15),
            ),
        )

        return fig

    def _make_ms2_figure(state):
        row = _compound_row(state["compound_idx"])
        inchi, adduct = row["inchi_key"], row["adduct"]
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        scans_by_energy = _get_ms2_scans(inchi, adduct, rt_min, rt_max)

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

            bars = []
            label_points = []
            scale = 1.0
            
            qry = scan.get("qry_spectrum") if "qry_spectrum" in scan.index else None
            ref = scan.get("ref_spectrum") if "ref_spectrum" in scan.index else None
            num_ref_fragments = 0
            num_matching_fragments = 0
            bar_width = 0.5
            
            if pd.notnull(qry) and pd.notnull(ref):
                mz_q, int_q = _parse_spectrum_cached(scan["qry_spectrum"])
                mz_r, int_r = _parse_spectrum_cached(scan["ref_spectrum"])
                scale = (max(int_q) / max(int_r)) if int_q and int_r and max(int_r) > 0 else 1.0
                ref_y = [-i * scale for i in int_r]
                # Scale bar width based on range of m/z values, with a minimum width to ensure visibility
                if mz_q and mz_r:
                    mz_min = min(min(mz_q), min(mz_r))
                    mz_max = max(max(mz_q), max(mz_r))
                    mz_range = mz_max - mz_min
                    if mz_range > 0:
                        bar_width = max(0.5, mz_range * 0.005)

                raw_colors = scan.get("aligned_fragment_colors") if "aligned_fragment_colors" in scan.index else None
                frag_colors = None
                if pd.notnull(raw_colors) and raw_colors:
                    frag_colors = json.loads(raw_colors)
                    if len(frag_colors) != len(mz_q):
                        raise ValueError(f"color length mismatch: {len(frag_colors)} != {len(mz_q)}")
                    num_ref_fragments = len(mz_r)
                    num_matching_fragments = sum(1 for c in frag_colors if c == "green")

                fig.add_trace(go.Bar(
                    x=mz_q, y=int_q, marker_color=frag_colors, width=bar_width,
                    showlegend=False,
                    hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Query</extra>"
                ), row=1, col=col_idx)
                
                fig.add_trace(go.Bar(
                    x=mz_r, y=ref_y, marker_color=frag_colors, width=bar_width,
                    showlegend=False,
                    hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Reference</extra>"
                ), row=1, col=col_idx)

                label_points.extend(zip(mz_q, int_q))
                label_points.extend(zip(mz_r, ref_y))
            else:
                mz, ints = _parse_spectrum_cached(scan["raw_spectrum"])
                fig.add_trace(go.Bar(
                    x=mz, y=ints, marker_color="red", width=bar_width,
                    showlegend=False,
                    hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>MS2</extra>"
                ), row=1, col=col_idx)
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

            # Find top 5 peaks by intensity
            top_label_idxs = {
                idx
                for idx, _ in sorted(
                    enumerate(label_points),
                    key=lambda item: abs(item[1][1]),
                    reverse=True,
                )[:5]
            }

            # Sort top labels by x-position for overlap detection
            top_labels_sorted = sorted(
                [(idx, mz_val, y_val) for idx, (mz_val, y_val) in enumerate(label_points) if idx in top_label_idxs],
                key=lambda item: item[1]
            )

            MIN_MZ_GAP = 5.0
            prev_mz = None
            stagger_level = 0

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
                
                fig.add_annotation(
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
                    row=1, col=col_idx
                )
                prev_mz = mz_val

            # Update axes for this subplot
            xaxis_name = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            yaxis_name = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
            
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
                f"<b>Score: {scan.get('score', 0):.4f}</b>  |  "
                f"Ions: {num_matching_fragments}/{num_ref_fragments}  |  "
                f"RT: {scan.get('rt', 0):.4f} min | "
                f"Exp. m/z: {prec_mz:.4f}  |  "
                f"Ref. m/z: {scan.get('mz_theoretical', 0):.4f}  |  "
                f"ppm Δ: {scan.get('ppm_error', 0):.2f}"
                f"</span><br>"
                f"{fname}<br><br>"
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

        # Overall layout
        fig.update_layout(
            barmode="overlay",
            hovermode="closest",
            margin=dict(l=50, r=20, t=120, b=40),
            plot_bgcolor="white",
            height=550,
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

        ms2_scans_cache.clear()
        new_state = _load_state(
            compound_idx,
            session_id=(old_state or {}).get("session_id"),
            edit_seq=(old_state or {}).get("edit_seq", 0),
        )
        new_state["last_saved"] = (old_state or {}).get("last_saved")
        new_state["flush_error"] = flush_error
        return new_state

    def _get_force_eval_and_ms2_warning(state, delta):
        """Return (force_eval, ms2_warning) for navigation logic."""
        force_eval = analysis_gui_obj.workflow_params.get("gui_require_all_evaluated", False)
        if analysis_gui_obj.override_parameters["gui_require_all_evaluated"] is not None:
            force_eval = analysis_gui_obj.override_parameters["gui_require_all_evaluated"]
        ms2_warning = None
        if (
            delta == 1
            and force_eval
            and not state.get("ms2_note")
            and state.get("ms1_note") != "remove"
        ):
            ms2_warning = "Please select MS2 quality note before proceeding"
        return force_eval, ms2_warning

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
        force_eval, ms2_warning = _get_force_eval_and_ms2_warning(state, delta)

        try:
            state = _flush_to_db(state)
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"navigate_compound: _flush_to_db failed: {exc}")
            flush_error = f"Save failed: {type(exc).__name__}: {exc}"

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
        filtered_vals = [v for v in vals if v in other_options] if vals else []
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
        for k, v in relayout.items():
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
            raise dash.exceptions.PreventUpdate
        return _patch_rt_change(state, new_min, new_max)

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
            set(ms2_key_to_label)
            | set(ms1_key_to_label)
            | set(other_key_to_label)
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
            force_eval, ms2_warning = _get_force_eval_and_ms2_warning(state, delta)
            if (
                delta == 1
                and ms2_warning is not None
            ):
                # Show warning, do not proceed
                new_state = dict(state)
                new_state["ms2_warning"] = ms2_warning
                return new_state, dash.no_update
            try:
                state = _flush_to_db(state)
            except Exception as exc:
                traceback.print_exc()
                logger.error(f"handle_keyboard navigation _flush_to_db failed: {exc}")
                flush_error = f"Save failed: {type(exc).__name__}: {exc}"
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
            return new_state, new_idx

        if key in ms2_key_to_label:
            return _patch_with_seq(state, ms2_note=ms2_key_to_label[key]), dash.no_update
        if key in ms1_key_to_label:
            return _patch_with_seq(state, ms1_note=ms1_key_to_label[key]), dash.no_update
        if key in other_key_to_label:
            current = state.get("other_note")
            if not isinstance(current, list):
                current = []
            label = other_key_to_label[key]
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
        flush_err = state.get("flush_error")
        ms2_warning = state.get("ms2_warning")
        try:
            ms1_fig = _make_ms1_figure(state, yaxis_scale)
            ms2_fig = _make_ms2_figure(state)
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
        scans_by_energy = _get_ms2_scans(row["inchi_key"], row["adduct"], state["rt_min"], state["rt_max"])
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
            return "", "No identification notes", ("keep" if "keep" in ms1_options else ms1_options[0]), "", [], 0
        ms2_val = state["ms2_note"] if state["ms2_note"] in ms2_options else ""
        ms1_val = state["ms1_note"] if state["ms1_note"] in ms1_options else ("keep" if "keep" in ms1_options else ms1_options[0])
        other_val = [v for v in state["other_note"] if v in other_options] if isinstance(state["other_note"], list) else []
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