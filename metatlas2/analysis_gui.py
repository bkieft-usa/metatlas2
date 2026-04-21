import functools, json, os, time, uuid, threading
import numpy as np, pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_bootstrap_components as dbc
from dash_extensions import EventListener
import traceback

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
logger = lcf.get_logger("analysis_gui")

MS2_OPTIONS = [
    "no selection",
    "-1.0, poor match, should remove",
    "0.0, no match or no MSMS collected",
    "0.5, partial or putative match of fragments",
    "1.0, good match",
    "0.5, co-isolated precursor, partial match",
    "1.0, co-isolated precursor, good match",
    "0.5, single ion match, no evidence",
    "1.0, single ion match, ISTD/ref evidence",
]
MS1_OPTIONS = [
    "keep", 
    "remove", 
    "unresolvable isomers", 
    "poor peak shape"
]
OTHER_OPTIONS = [
    "no selection",
    "potential rt shifting",
    "high ppm diff",
    "noisy or high background",
    "needs review"
]

MS2_HOTKEYS = {
    "no selection":                                "q",
    "-1.0, poor match, should remove":             "w",
    "0.0, no match or no MSMS collected":          "e",
    "0.5, partial or putative match of fragments": "r",
    "1.0, good match":                             "t",
    "0.5, co-isolated precursor, partial match":   "y",
    "1.0, co-isolated precursor, good match":      "u",
    "0.5, single ion match, no evidence":          "i",
    "1.0, single ion match, ISTD/ref evidence":    "o",
}
MS1_HOTKEYS = {
    "keep":                 "1",
    "remove":               "2",
    "unresolvable isomers": "3",
    "poor peak shape":      "4",
}
OTHER_HOTKEYS = {
    "no selection":           "5",
    "potential rt shifting":  "6",
    "high ppm diff":          "7",
    "noisy or high background": "8",
    "needs review":           "9",
}

MS2_KEY_TO_LABEL = {v: k for k, v in MS2_HOTKEYS.items()}
MS1_KEY_TO_LABEL = {v: k for k, v in MS1_HOTKEYS.items()}
OTHER_KEY_TO_LABEL = {v: k for k, v in OTHER_HOTKEYS.items()}

def build_dash_app(
    analysis_gui_obj,
    port=8050,
    shutdown_holder=None
):
    logger.info("Starting the app factory for the Analysis GUI...")

    # Set up basic GUI params
    manual_curation_df = analysis_gui_obj.manual_curation_df
    top_n_hits = analysis_gui_obj.workflow_params.get("gui_top_n_hits", 20)

    logger.info(f"Analysis starting with {len(manual_curation_df)} compounds.")

    # Set up all passing compounds as options for the dropdown
    compound_options = [
        {"label": f"{i+1}: {row['compound_name']}", "value": i}
        for i, row in manual_curation_df.reset_index().iterrows()
    ]

    # Create the app
    try:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix=f"{os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')}proxy/{port}/",
            suppress_callback_exceptions=True,
        )
        logger.info("App built successfully")
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
        ms2_stored = row.get("ms2_notes") or ""
        ms2_note = ms2_stored if ms2_stored else "no selection"
        ms1_note = row.get("ms1_notes") or "keep"
        if ms1_note not in MS1_OPTIONS:
            ms1_note = "keep"
        other_note = row.get("other_notes") or "no selection"
        if other_note not in OTHER_OPTIONS:
            other_note = "no selection"
        
        return {
            "session_id":    session_id or str(uuid.uuid4()),
            "edit_seq":      int(edit_seq),
            "compound_idx":  compound_idx,
            "ms2_idx":       ms2_idx,
            "rt_min":        rt_min,
            "rt_max":        rt_max,
            "ms1_note":      ms1_note,
            "ms2_note":      ms2_note,
            "other_note":    other_note,
            "analyst_notes": row.get("analyst_notes") or "",
            "id_notes":      row.get("identification_notes") or "",
            "last_saved":    None,
            "isomer_snap_idx":   0,
            "flush_error":   None,
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

    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp", "target.tagName"]}],
    )

    # format of the app itself
    app.layout = dbc.Container(
        [
            dcc.Store(id="session-store", storage_type="memory", data=_load_state(0)),
            dcc.Store(id="controls-compound-idx", storage_type="memory", data=0),
            keyboard_listener,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(id="compound-dd", options=compound_options, value=0, clearable=False, style={"width": "100%"}),
                                        width=12, className="mb-3",
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀ Prev ID [j]", id="prev-btn", color="primary", className="me-2"), width="auto"),
                                    dbc.Col(html.Div(id="compound-counter", className="fw-bold")),
                                    dbc.Col(dbc.Button("Next ID ▶  [k]", id="next-btn", color="primary", className="ms-2"), width="auto"),
                                ],
                                className="my-2 align-items-center",
                            ),
                            dbc.Button("Accept Suggestions  [n]", id="accept-suggestions", color="success", className="my-2 w-100"),
                            dbc.Button("Snap to Isomer  [m]", id="snap-to-isomer", color="warning", className="my-2 w-100"),
                            dbc.Textarea(id="analyst-notes", placeholder="Analyst notes …", debounce=True, style={"width": "100%", "height": "40px"}, className="my-2"),
                            dbc.Textarea(id="id-notes", placeholder="Identification notes …", debounce=True, style={"width": "100%", "height": "40px"}, className="my-2"),
                            html.Div(
                                [
                                    html.Label("MS1 quality:", className="fw-bold"),
                                    dcc.RadioItems(
                                        id="ms1-radio",
                                        options=[{"label": f"[{MS1_HOTKEYS[lbl]}] {lbl}", "value": lbl} for lbl in MS1_OPTIONS],
                                        value="keep",
                                        labelStyle={"display": "block", "margin-bottom": "6px"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀ Prev MS2  [l]", id="ms2-prev", className="me-2"), width="auto"),
                                    dbc.Col(html.Div(id="ms2-counter", className="fw-bold")),
                                    dbc.Col(dbc.Button("Next MS2 ▶  [;]", id="ms2-next", className="ms-2"), width="auto"),
                                ],
                                className="my-2 align-items-center",
                            ),
                            html.Div(
                                [
                                    html.Label("MS2 quality:", className="fw-bold"),
                                    dcc.RadioItems(
                                        id="ms2-radio",
                                        options=[{"label": f"[{MS2_HOTKEYS[val]}] {val}", "value": val} for val in MS2_OPTIONS],
                                        value="no selection",
                                        labelStyle={"display": "block", "margin-bottom": "6px"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(
                                [
                                    html.Label("Other notes:", className="fw-bold"),
                                    dcc.RadioItems(
                                        id="other-radio",
                                        options=[{"label": f"[{OTHER_HOTKEYS[val]}] {val}", "value": val} for val in OTHER_OPTIONS],
                                        value="no selection",
                                        labelStyle={"display": "block", "margin-bottom": "6px"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(id="status-current", className="my-2"),
                            html.Div(id="status-previous", className="my-2"),
                            html.Div(id="error-banner", className="my-2"),
                        ],
                        width=3,
                        style={"fontSize": "calc(1rem - 2pt)"},
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="ms1-graph",
                                config={"editable": True, "displayModeBar": True, "edits": {"titleText": False}},
                                style={"height": "475px"},
                            ),
                            dcc.Graph(id="ms2-graph", config={"displayModeBar": True}, style={"height": "475px"}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(id="save-exit-status", className="text-muted fst-italic"),
                                        className="d-flex align-items-center",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Save and Exit",
                                            id="save-exit-btn",
                                            color="danger",
                                            size="sm",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="mt-2 mb-1",
                                justify="end",
                                align="center",
                            ),
                        ],
                        width=9,
                    ),
                ],
                className="mt-3",
            ),
        ],
        fluid=True,
    )

    logger.info("Layout constructed successfully")

    def _parse_isomers(isomers_val):
        if isomers_val is None or (isinstance(isomers_val, float) and np.isnan(isomers_val)):
            return []
        if isinstance(isomers_val, list):
            return isomers_val
        if isinstance(isomers_val, dict):
            return [isomers_val]
        if isinstance(isomers_val, str):
            try:
                parsed = json.loads(isomers_val)
            except Exception:
                return []
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        return []

    def _get_sorted_isomer_rt_bounds(row):
        """Return list of (rt_min, rt_max) for isomers, sorted by rt_min."""
        isomers_val = row.get("isomers") if "isomers" in row.index else None
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
            matches = manual_curation_df[mask]
            if matches.empty:
                continue
            bounds.append((float(matches.iloc[0]["rt_min"]), float(matches.iloc[0]["rt_max"])))
        return sorted(bounds, key=lambda x: x[0])

    def _get_ms2_scans(inchi_key, adduct, rt_min=None, rt_max=None):
        """Return the sorted, capped scans DataFrame for a compound/rt window.

        Results are cached by (inchi_key, adduct, score_cutoff, top_n_hits, rt_min, rt_max)
        so both _count_ms2_scans and _make_ms2_figure share the same computation.
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

        if ms2_sub.empty:
            scans = pd.DataFrame()
        elif hits_sub.empty:
            scans = ms2_sub.sort_values("rt").head(top_n_hits)
        else:
            merged = pd.merge(ms2_sub, hits_sub, on=["inchi_key", "adduct", "file_path", "rt"], how="left")
            merged["score"] = merged["score"].fillna(0)
            scans = merged.sort_values(["score", "rt"], ascending=[False, True]).head(top_n_hits)

        with cache_lock:
            if key not in ms2_scans_cache:
                ms2_scans_cache[key] = scans
        return ms2_scans_cache[key]

    def _count_ms2_scans(row, rt_min=None, rt_max=None):
        return len(_get_ms2_scans(row["inchi_key"], row["adduct"], rt_min, rt_max))

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

        with flush_lock:
            latest_seq = latest_flushed_seq_by_session.get(sid, -1)
            if seq <= latest_seq:
                #logger.info(f"Skipping stale flush sid={sid} seq={seq} latest={latest_seq}")
                return state
            latest_flushed_seq_by_session[sid] = seq

        row = _compound_row(state["compound_idx"])
        updates = {
            "rt_min":               state["rt_min"],
            "rt_max":               state["rt_max"],
            "rt_peak":              (state["rt_min"] + state["rt_max"]) / 2,
            "ms2_notes":            state["ms2_note"],
            "ms1_notes":            state["ms1_note"],
            "other_notes":          state["other_note"],
            "analyst_notes":        state["analyst_notes"],
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
            "name":          row["compound_name"],
            "rt_min":        state["rt_min"],
            "rt_max":        state["rt_max"],
            "ms1":           state["ms1_note"],
            "ms2":           state["ms2_note"],
            "other":         state["other_note"],
            "analyst_notes": state.get("analyst_notes", ""),
            "id_notes":      state.get("id_notes", ""),
            "timestamp":     time.strftime("%H:%M:%S"),
        }
        state["flush_error"] = None
        return state

    # main figures for ms data display
    def _make_ms1_figure(state):
        if analysis_gui_obj.override_parameters['gui_lcmsruns_colors'] is not None:
            lcmsruns_color_map = analysis_gui_obj.override_parameters['gui_lcmsruns_colors']
            if not isinstance(lcmsruns_color_map, dict):
                raise ValueError("override_parameters['gui_lcmsruns_colors'] must be a dict mapping LCMS run identifiers to color strings")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in lcmsruns_color_map.items()):
                raise ValueError("override_parameters['gui_lcmsruns_colors'] must be a dict mapping strings to strings (LCMS run identifiers to color strings)")
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

        fig = go.Figure()
        for fp in sub["file_path"].unique():
            short_name = "_".join(os.path.basename(fp).split(".")[0].split("_")[11:])
            color = next((c for k, c in lcmsruns_color_map.items() if k.lower() in short_name.lower()), "gray")
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

        fig.add_vline(x=row["atlas_rt_peak"], line=dict(color="black", width=1.5))
        if pd.notnull(row.get("suggested_rt_min")):
            fig.add_vline(x=row["suggested_rt_min"], line=dict(color="orange", dash="dot", width=1.5))
        if pd.notnull(row.get("suggested_rt_max")):
            fig.add_vline(x=row["suggested_rt_max"], line=dict(color="orange", dash="dot", width=1.5))

        for x, name in [(rt_min, "RT min"), (rt_max, "RT max")]:
            fig.add_shape(
                type="line", x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="purple", width=2.5, dash="dash"),
                name=name, editable=True,
            )

        isomer_str = "No Isomers Found"
        try:
            isomers = _parse_isomers(row.get("isomers"))
            if isomers:
                isomer_lines = []
                resolved_isomers = []
                for iso in isomers:
                    iso_inchi = iso.get('inchi_key', '')
                    iso_name   = iso.get('compound_name', '')
                    iso_adduct = iso.get('adduct', '')
                    iso_rt     = iso.get('rt', None)
                    iso_mz     = iso.get('mz', None)

                    mask = (
                        (manual_curation_df["inchi_key"] == iso_inchi) &
                        (manual_curation_df["compound_name"] == iso_name) &
                        (manual_curation_df["adduct"] == iso_adduct)
                    )
                    matches = manual_curation_df[mask]

                    if len(matches) > 1:
                        raise ValueError(f"There were multiple matches of isomer {iso_name} {iso_adduct} {iso_inchi} to the manual curation object")

                    if matches.empty:
                        continue

                    iso_df_idx      = matches.index[0]
                    iso_display_idx = iso_df_idx + 1
                    iso_rt_min      = float(matches.iloc[0]["rt_min"])
                    iso_rt_max      = float(matches.iloc[0]["rt_max"])
                    resolved_isomers.append({
                        "display_idx": iso_display_idx,
                        "name":        iso_name,
                        "adduct":      iso_adduct,
                        "rt_min":      iso_rt_min,
                        "rt_max":      iso_rt_max,
                        "rt":          iso_rt,
                        "mz":          iso_mz,
                    })

                if resolved_isomers:
                    LABEL_WIDTH_PROXY = 0.15
                    stagger_levels = []
                    for i, iso in enumerate(resolved_isomers):
                        mid_i = (iso["rt_min"] + iso["rt_max"]) / 2
                        occupied = set()
                        for j in range(i):
                            prev = resolved_isomers[j]
                            mid_j = (prev["rt_min"] + prev["rt_max"]) / 2
                            rt_overlap = (
                                iso["rt_min"] <= prev["rt_max"] and iso["rt_max"] >= prev["rt_min"]
                            )
                            label_overlap = abs(mid_i - mid_j) < LABEL_WIDTH_PROXY
                            if rt_overlap or label_overlap:
                                occupied.add(stagger_levels[j])
                        level = 0.0
                        while level in occupied:
                            level = round(level + 0.05, 3)
                        stagger_levels.append(level)

                    for iso, y_level in zip(resolved_isomers, stagger_levels):
                        fig.add_vrect(
                            x0=iso["rt_min"], x1=iso["rt_max"],
                            fillcolor="lightgray", opacity=0.35,
                            layer="below", line_width=0,
                        )
                        fig.add_annotation(
                            x=(iso["rt_min"] + iso["rt_max"]) / 2,
                            y=y_level,
                            xref="x", yref="paper",
                            text=f"[{iso['display_idx']}] {iso['name']} ({iso['adduct']})",
                            showarrow=False,
                            font=dict(size=8, color="dimgray"),
                            xanchor="center", yanchor="bottom",
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
                    isomer_str = "<br>".join(isomer_lines)
                else:
                    isomer_str = "No Isomers Found" 
            else:
                isomer_str = "No Isomers Found"
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Isomer detection failed with {exc}")

        compound_display_idx = state["compound_idx"] + 1

        ms1_title_text = (
            f"<span style='font-size:1.2em'>[{compound_display_idx}] {row['compound_name']} | {adduct} | {inchi}</span><br>"
            f"Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: {row['best_ms1_rt']:.4f}  |  RT Δ: {row['best_ms1_rt_error']:.3f}<br>"
            f"Atlas m/z: {row['atlas_mz']:.4f}  |  ppm Δ: {row['best_ms1_ppm_error']:.2f}<br>"
            f"<sub style='font-size:0.8em'>Isomers: {isomer_str}</sub>"
        )

        fig.update_layout(
            title=dict(text=ms1_title_text, x=0.5, xanchor="center", font=dict(size=12)),
            xaxis_title="RT (min)", yaxis_title="Intensity",
            hovermode="closest", showlegend=False,
            margin=dict(l=50, r=20, t=125, b=40), dragmode="pan",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )

        return fig

    def _make_ms2_figure(state):
        row = _compound_row(state["compound_idx"])
        inchi, adduct = row["inchi_key"], row["adduct"]
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        scans = _get_ms2_scans(inchi, adduct, rt_min, rt_max)

        if len(scans) == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"{row['compound_name']} – No MS2 data",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(
                margin=dict(l=50, r=20, t=80, b=40),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            return fig

        bars = []
        scale = 1.0
        ms2_idx = max(0, min(state["ms2_idx"], len(scans) - 1))
        scan = scans.iloc[ms2_idx]
        qry = scan.get("qry_spectrum") if "qry_spectrum" in scan.index else None
        ref = scan.get("ref_spectrum") if "ref_spectrum" in scan.index else None
        if pd.notnull(qry) and pd.notnull(ref):
            mz_q, int_q = _parse_spectrum_cached(scan["qry_spectrum"])
            mz_r, int_r = _parse_spectrum_cached(scan["ref_spectrum"])

            raw_colors = scan.get("aligned_fragment_colors") if "aligned_fragment_colors" in scan.index else None
            if pd.notnull(raw_colors) and raw_colors:
                try:
                    frag_colors = json.loads(raw_colors)
                    if len(frag_colors) != len(mz_q):
                        raise ValueError(f"color length mismatch: {len(frag_colors)} != {len(mz_q)}")
                except Exception:
                    frag_colors = ["red"] * len(mz_q)
            else:
                frag_colors = ["red"] * len(mz_q)

            scale = (max(int_q) / max(int_r)) if int_q and int_r and max(int_r) > 0 else 1.0
            bars.append(go.Bar(x=mz_q, y=int_q, marker_color=frag_colors, width=0.25,
                       showlegend=False,
                       hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Query</extra>"))
            bars.append(go.Bar(x=mz_r, y=[-i * scale for i in int_r],
                       marker_color=frag_colors, width=0.25,
                       showlegend=False,
                       hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Reference</extra>"))
        else:
            mz, ints = _parse_spectrum_cached(scan["raw_spectrum"])
            bars.append(go.Bar(x=mz, y=ints, marker_color="red", width=0.25,
                       showlegend=False,
                       hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>MS2</extra>"))

        fig = go.Figure(data=bars)
        fig.add_hline(y=0, line=dict(color="black", width=1.5))
        fname = "_".join(os.path.basename(scan.get("file_path", "")).split(".")[0].split("_")[11:])
        ms2_title_text = (
            f"File: {fname}<br>"
            f"Score: {scan.get('score', 0):.4f}  |  Scan RT: {scan.get('rt', 0):.4f} min | Precursor m/z: {scan.get('precursor_MZ', 0):.4f}  |  Ref m/z: {scan.get('mz_theoretical', 0):.4f}  |  ppm Δ: {scan.get('ppm_error', 0):.2f}"
        )
        fig.update_layout(
            title=dict(text=ms2_title_text, x=0.5, xanchor="center", font=dict(size=12)),
            xaxis_title="m/z", yaxis_title=f"Intensity (Ref scaled x{scale:.2f})",
            barmode="overlay", hovermode="closest",
            margin=dict(l=50, r=20, t=80, b=40),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )

        return fig

    logger.info("App helpers defined successfully")

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
        Input("other-radio", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_other_note(val, state):
        if state is None or val == state.get("other_note"):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, other_note=val)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("analyst-notes", "value"),
        State("session-store", "data"),
        State("controls-compound-idx", "data"),
        prevent_initial_call=True,
    )
    def set_analyst_notes(txt, state, controls_idx):
        if state is None or txt is None or txt == state.get("analyst_notes"):
            raise dash.exceptions.PreventUpdate
        if controls_idx is not None and int(controls_idx) != int(state.get("compound_idx", -1)):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, analyst_notes=txt)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("id-notes", "value"),
        State("session-store", "data"),
        State("controls-compound-idx", "data"),
        prevent_initial_call=True,
    )
    def set_id_notes(txt, state, controls_idx):
        if state is None or txt is None or txt == state.get("id_notes"):
            raise dash.exceptions.PreventUpdate
        if controls_idx is not None and int(controls_idx) != int(state.get("compound_idx", -1)):
            raise dash.exceptions.PreventUpdate
        return _patch_with_seq(state, id_notes=txt)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms1-graph", "relayoutData"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def rt_drag(relayout, state):
        if not relayout or state is None:
            raise dash.exceptions.PreventUpdate
        new_min, new_max = state["rt_min"], state["rt_max"]
        updated = False
        for k, v in relayout.items():
            if not (k.startswith("shapes[") and k.endswith("].x0")):
                continue
            dragged_x = float(v)
            dist_min = abs(dragged_x - state["rt_min"])
            dist_max = abs(dragged_x - state["rt_max"])
            if dist_min <= dist_max:
                new_min = dragged_x
            else:
                new_max = dragged_x
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
            set(MS2_KEY_TO_LABEL)
            | set(MS1_KEY_TO_LABEL)
            | set(OTHER_KEY_TO_LABEL)
            | {"a", "s", "d", "f", "j", "k", "l", ";", "n", "m", " "}
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

        if key == "l":
            row = _compound_row(state["compound_idx"])
            n_scans = _count_ms2_scans(row, state["rt_min"], state["rt_max"])
            if n_scans == 0:
                raise dash.exceptions.PreventUpdate
            return _patch_with_seq(state, ms2_idx=max(state["ms2_idx"] - 1, 0)), dash.no_update

        if key == ";":
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

        if key in ("j", "k", " "):
            delta = -1 if key == "j" else 1
            new_idx = (int(state["compound_idx"]) + delta) % len(compound_options)
            if new_idx == int(state["compound_idx"]):
                raise dash.exceptions.PreventUpdate
            flush_error = None
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
            return new_state, new_idx

        if key in MS2_KEY_TO_LABEL:
            return _patch_with_seq(state, ms2_note=MS2_KEY_TO_LABEL[key]), dash.no_update
        if key in MS1_KEY_TO_LABEL:
            return _patch_with_seq(state, ms1_note=MS1_KEY_TO_LABEL[key]), dash.no_update
        if key in OTHER_KEY_TO_LABEL:
            return _patch_with_seq(state, other_note=OTHER_KEY_TO_LABEL[key]), dash.no_update

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("ms1-graph", "figure"),
        Output("ms2-graph", "figure"),
        Output("error-banner", "children"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_figures(state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        flush_err = state.get("flush_error")
        try:
            ms1_fig, ms2_fig = _make_ms1_figure(state), _make_ms2_figure(state)
            if flush_err:
                banner = html.Span(
                    f"⚠ {flush_err}",
                    style={"color": "red", "fontSize": "11px", "fontWeight": "bold"},
                )
            else:
                banner = ""
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
        n_scans = _count_ms2_scans(row, state["rt_min"], state["rt_max"])
        ms2_txt = f"MS2 Scan {state['ms2_idx']+1} of {n_scans}" if n_scans else "No MS2 data"
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
        Output("id-notes", "value"),
        Output("ms1-radio", "value"),
        Output("ms2-radio", "value"),
        Output("other-radio", "value"),
        Output("controls-compound-idx", "data"),
        Input("session-store", "data"),
        prevent_initial_call=True,
    )
    def sync_controls(state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        ms2_val   = state["ms2_note"]   if state["ms2_note"]   in MS2_OPTIONS   else "no selection"
        ms1_val   = state["ms1_note"]   if state["ms1_note"]   in MS1_OPTIONS   else "keep"
        other_val = state["other_note"] if state["other_note"] in OTHER_OPTIONS else "no selection"
        return state["analyst_notes"], state["id_notes"], ms1_val, ms2_val, other_val, state["compound_idx"]

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
            logger.info(f"Save and Exit.")
            msg = "Analysis saved and GUI closed. You may return to the notebook to run the curation summary."
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Save and Exit: flush failed for compound {state.get('compound_idx')}: {exc}")
            msg = f"Save failed: {type(exc).__name__}: {exc}."

        if shutdown_holder is not None and shutdown_holder[0] is not None:
            threading.Timer(1.5, shutdown_holder[0]).start()

        return msg, True

    logger.info("Callbacks registered")

    logger.info("App setup complete")

    return app