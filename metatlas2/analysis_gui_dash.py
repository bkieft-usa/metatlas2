import json, os, time, sys
import numpy as np, pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_bootstrap_components as dbc
from dash_extensions import EventListener  # Use EventListener instead of Keyboard

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf

logger = lcf.get_logger('analysis_gui')

# -------------------------------------------------
#   Constants
# -------------------------------------------------
MS2_LABEL_TO_VALUE = {
    "No selection":            "no selection",
    "-1 Poor match":           "-1.0, poor match, should remove",
    "0 No match/no MSMS":      "0.0, no match or no MSMS collected",
    "0.5 Partial match":       "0.5, partial or putative match of fragments",
    "1.0 Good match":          "1.0, good match",
    "0.5 Co-iso partial":      "0.5, co-isolated precursor, partial match",
    "1.0 Co-iso good":         "1.0, co-isolated precursor, good match",
    "0.5 Single ion":          "0.5, single ion match, no evidence",
    "1.0 Single ion+evidence": "1.0, single ion match, ISTD/ref evidence",
}
MS2_VALUE_TO_LABEL = {v: k for k, v in MS2_LABEL_TO_VALUE.items()}
MS1_OPTIONS = ["keep", "remove", "unresolvable isomers", "poor peak shape"]

MS2_HOTKEYS = {
    "No selection":            "1",
    "-1 Poor match":           "2",
    "0 No match/no MSMS":      "3",
    "0.5 Partial match":       "4",
    "1.0 Good match":          "5",
    "0.5 Co-iso partial":      "6",
    "1.0 Co-iso good":         "7",
    "0.5 Single ion":          "8",
    "1.0 Single ion+evidence": "9",
}
MS1_HOTKEYS = {
    "keep":                 "q",
    "remove":               "w",
    "unresolvable isomers": "e",
    "poor peak shape":      "r",
}
MS2_KEY_TO_LABEL = {v: k for k, v in MS2_HOTKEYS.items()}
MS1_KEY_TO_LABEL = {v: k for k, v in MS1_HOTKEYS.items()}

RT_STEP = 0.025
EIC_COLOR_MAP = {
    "istd":   "blue",
    "qc":     "blue",
    "exctrl": "red",
    "txctrl": "red",
    "refstd": "black",
}

# -------------------------------------------------
#   DuckDB helper
# -------------------------------------------------
def to_python_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val

def update_manual_curation_entry(project_db_path, curation_uid, updates):
    if not updates:
        return
    set_clause = ", ".join([f"{k} = ?" for k in updates])
    params = [to_python_type(v) for v in updates.values()] + [curation_uid]
    with dbi.get_db_connection(project_db_path) as conn:
        conn.execute(f"UPDATE manual_curation SET {set_clause} WHERE curation_uid = ?", params)

# -------------------------------------------------
#   Main factory
# -------------------------------------------------
def build_dash_app(analysis_gui_obj,
                   ms2_score_cutoff=0.5,
                   top_n_hits=20,
                   port=8050):

    project_path = analysis_gui_obj.paths["project_db_path"]
    rt_alignment = analysis_gui_obj.rt_alignment_number
    analysis_num = analysis_gui_obj.analysis_number

    manual_curation_df = dbi.get_manual_curation_entries(project_path, rt_alignment, analysis_num)
    ms1_df  = dbi.get_ms1_data_for_compound(project_path, None, None, rt_alignment, analysis_num)
    ms2_df  = dbi.get_ms2_data_for_compound(project_path, None, None, rt_alignment, analysis_num)
    ms2_hits_df = dbi.get_ms2_hits_for_compound(project_path, None, None, rt_alignment, analysis_num)
    if manual_curation_df.empty:
        raise ValueError("No compounds found in manual_curation table.")

    compound_options = [
        {"label": f"{i+1}: {row['compound_name']}", "value": i}
        for i, row in manual_curation_df.reset_index().iterrows()
    ]

    service_prefix = os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')
    proxy_path = f"{service_prefix}proxy/{port}/"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        requests_pathname_prefix=proxy_path,
        suppress_callback_exceptions=True,
    )

    # FIX 2: capture "timeStamp" alongside "key" so that pressing the same
    # arrow key twice produces two distinct event dicts — Dash won't skip
    # the second callback because the input value has changed.
    from dash_extensions import EventListener
    keyboard_listener = EventListener(
        id="keyboard",
        events=[{"event": "keydown", "props": ["key", "timeStamp"]}],
    )

    app.layout = dbc.Container(
        [
            dcc.Store(id="session-store", storage_type="session"),
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
                                    dbc.Col(dbc.Button("◀ Prev  [←]", id="prev-btn", color="primary", className="me-2"), width="auto"),
                                    dbc.Col(html.Div(id="compound-counter", className="fw-bold")),
                                    dbc.Col(dbc.Button("Next ▶  [→/Space]", id="next-btn", color="primary", className="ms-2"), width="auto"),
                                ],
                                className="my-2 align-items-center",
                            ),
                            dbc.Button("Accept Suggestions  [m]", id="accept-suggestions", color="success", className="my-2 w-100"),
                            dbc.Textarea(id="analyst-notes", placeholder="Analyst notes …", style={"width": "100%", "height": "80px"}, className="my-2"),
                            dbc.Textarea(id="id-notes", placeholder="Identification notes …", style={"width": "100%", "height": "80px"}, className="my-2"),
                            html.Div(
                                [
                                    html.Label("MS1 quality:", className="fw-bold"),
                                    dcc.RadioItems(
                                        id="ms1-radio",
                                        options=[{"label": f"[{MS1_HOTKEYS[lbl]}] {lbl}", "value": lbl} for lbl in MS1_OPTIONS],
                                        value="keep",
                                        labelStyle={"display": "block", "margin-bottom": "8px"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Button("◀ Previous MS2  [↓]", id="ms2-prev", className="me-2"), width="auto"),
                                    dbc.Col(html.Div(id="ms2-counter", className="fw-bold")),
                                    dbc.Col(dbc.Button("Next MS2 ▶  [↑]", id="ms2-next", className="ms-2"), width="auto"),
                                ],
                                className="my-2 align-items-center",
                            ),
                            html.Div(
                                [
                                    html.Label("MS2 quality:", className="fw-bold"),
                                    dcc.RadioItems(
                                        id="ms2-radio",
                                        options=[{"label": f"[{MS2_HOTKEYS[lbl]}] {lbl}", "value": lbl} for lbl in MS2_LABEL_TO_VALUE.keys()],
                                        value="No selection",
                                        labelStyle={"display": "block", "margin-bottom": "8px"},
                                    ),
                                ],
                                className="my-3",
                            ),
                            html.Div(id="status-current", className="my-2"),
                            html.Div(id="status-previous", className="my-2"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="ms1-graph", config={"editable": True, "displayModeBar": True}, style={"height": "500px"}),
                            dcc.Graph(id="ms2-graph", config={"displayModeBar": True}, style={"height": "500px"}),
                        ],
                        width=8,
                    ),
                ],
                className="mt-3",
            ),
        ],
        fluid=True,
    )

    # ── helpers ────────────────────────────────────────────────────────────
    def _compound_row(idx):
        return manual_curation_df.iloc[idx]

    def _rt_bounds_from_row(row):
        return float(row["rt_min"]), float(row["rt_max"])

    def _load_state(compound_idx, ms2_idx=0):
        row = _compound_row(compound_idx)
        rt_min, rt_max = _rt_bounds_from_row(row)
        ms2_stored = row.get("ms2_notes") or ""
        ms2_note = ms2_stored if ms2_stored else "no selection"
        ms1_note = row.get("ms1_notes") or "keep"
        return {
            "compound_idx": compound_idx,
            "ms2_idx": ms2_idx,
            "rt_min": rt_min,
            "rt_max": rt_max,
            "ms1_note": ms1_note,
            "ms2_note": ms2_note,
            "analyst_notes": row.get("analyst_notes") or "",
            "id_notes": row.get("identification_notes") or "",
            "last_saved": None,
        }

    def _apply_rt_change(state, new_min, new_max):
        rt_min = max(0.0, min(new_min, new_max))
        rt_max = max(rt_min, new_max)
        state["rt_min"] = round(rt_min, 4)
        state["rt_max"] = round(rt_max, 4)
        return state

    def _count_ms2_scans(row):
        sub = ms2_df[(ms2_df["inchi_key"] == row["inchi_key"]) & (ms2_df["adduct"] == row["adduct"])]
        if sub.empty:
            return 0
        return sub.groupby(["inchi_key", "adduct", "file_path", "rt"]).ngroups

    def _clean_spectrum(mz_arr, int_arr):
        out_mz, out_i = [], []
        for m, i in zip(mz_arr, int_arr):
            if isinstance(m, float) and np.isnan(m):
                continue
            out_mz.append(m)
            out_i.append(0 if (isinstance(i, float) and np.isnan(i)) else i)
        return out_mz, out_i

    def _flush_to_db(state):
        row = _compound_row(state["compound_idx"])
        uid = row["curation_uid"]
        updates = {
            "rt_min":               state["rt_min"],
            "rt_max":               state["rt_max"],
            "rt_peak":              (state["rt_min"] + state["rt_max"]) / 2,
            "ms2_notes":            state["ms2_note"],
            "ms1_notes":            state["ms1_note"],
            "analyst_notes":        state["analyst_notes"],
            "identification_notes": state["id_notes"],
        }
        update_manual_curation_entry(project_path, uid, updates)
        state["last_saved"] = {
            "name":      row["compound_name"],
            "rt_min":    state["rt_min"],
            "rt_max":    state["rt_max"],
            "ms1":       state["ms1_note"],
            "ms2":       state["ms2_note"],
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return state

    # ── figure builders ────────────────────────────────────────────────────
    def _make_ms1_figure(state):
        row = _compound_row(state["compound_idx"])
        inchi, adduct = row["inchi_key"], row["adduct"]
        rt_min, rt_max = state["rt_min"], state["rt_max"]

        # FIX 1: boolean mask — avoids pandas .query() choking on special
        # characters in InChIKeys (e.g. "/", "=") and adducts (e.g. "[M+H]+")
        sub = ms1_df[(ms1_df["inchi_key"] == inchi) & (ms1_df["adduct"] == adduct)]

        fig = go.Figure()
        for fp in sub["file_path"].unique():
            short_name = "_".join(os.path.basename(fp).split(".")[0].split("_")[11:])
            color = next((c for k, c in EIC_COLOR_MAP.items() if k in short_name.lower()), "gray")
            for _, r in sub[sub["file_path"] == fp].iterrows():
                try:
                    rt, intensity = json.loads(r["raw_spectrum"])
                    fig.add_trace(go.Scatter(
                        x=rt, y=intensity, mode="lines", name=short_name,
                        line=dict(color=color, width=1.5),
                        hovertemplate="%{x:.3f} min<br>%{y:.2e}",
                    ))
                except Exception as e:
                    logger.error("MS1 parse error %s: %s", fp, e)

        fig.add_vline(x=row["atlas_rt_peak"], line=dict(color="black", dash="dash", width=1.5),
                      annotation_text=f"Atlas {row['atlas_rt_peak']:.3f}", annotation_position="top left")
        if pd.notnull(row.get("suggested_rt_min")):
            fig.add_vline(x=row["suggested_rt_min"], line=dict(color="orange", dash="dot", width=1.5),
                          annotation_text="Sugg min", annotation_position="top left")
        if pd.notnull(row.get("suggested_rt_max")):
            fig.add_vline(x=row["suggested_rt_max"], line=dict(color="orange", dash="dot", width=1.5),
                          annotation_text="Sugg max", annotation_position="top left")

        for x, name in [(rt_min, "RT min"), (rt_max, "RT max")]:
            fig.add_shape(
                type="line", x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="purple", width=2.5, dash="dash"),
                name=name, editable=True,
            )

        ms1_title_text = (
                f"{row['compound_name']} | {adduct} | {inchi}<br>"
                f"<sub>Atlas RT: {row['atlas_rt_peak']:.4f}  |  Meas RT: {row['best_ms1_rt']:.4f}  |  RT Δ: {row['best_ms1_rt_error']:.3f}</sub><br>"
                f"<sub>Atlas m/z: {row['atlas_mz']:.4f}  |  ppm Δ: {row['best_ms1_ppm_error']:.2f}"
                f"Isomers: {isomer_str}"
            )

        isomer_str = row.get("isomer_count", "?")
        fig.update_layout(
            title=dict(text=ms1_title_text,x=0, xanchor="left", font=dict(size=10)),
            xaxis_title="RT (min)", yaxis_title="Intensity",
            hovermode="closest", showlegend=False,
            margin=dict(l=50, r=20, t=60, b=40), dragmode="pan",
        )
        return fig

    def _make_ms2_figure(state):
        row = _compound_row(state["compound_idx"])
        inchi, adduct = row["inchi_key"], row["adduct"]

        # FIX 1: boolean mask
        ms2_sub  = ms2_df[(ms2_df["inchi_key"] == inchi) & (ms2_df["adduct"] == adduct)]
        hits_sub = ms2_hits_df[
            (ms2_hits_df["inchi_key"] == inchi)
            & (ms2_hits_df["adduct"] == adduct)
            & (ms2_hits_df["score"] >= ms2_score_cutoff)
        ]

        if ms2_sub.empty and hits_sub.empty:
            scans = pd.DataFrame()
        elif hits_sub.empty:
            scans = ms2_sub.sort_values("rt")
        else:
            merged = pd.merge(ms2_sub, hits_sub, on=["inchi_key", "adduct", "file_path", "rt"], how="left")
            merged["score"] = merged["score"].fillna(0)
            scans = merged.sort_values(["score", "rt"], ascending=[False, True])
        scans = scans.head(top_n_hits)
        n_scans = len(scans)

        if n_scans == 0:
            fig = go.Figure()
            fig.add_annotation(text=f"{row['compound_name']} – No MS2 data",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(margin=dict(l=60, r=20, t=30, b=30))
            return fig

        ms2_idx = max(0, min(state["ms2_idx"], n_scans - 1))
        scan = scans.iloc[ms2_idx]

        bars = []
        if pd.notnull(scan.get("qry_spectrum")) and pd.notnull(scan.get("ref_spectrum")):
            mz_q, int_q = json.loads(scan["qry_spectrum"])
            mz_r, int_r = json.loads(scan["ref_spectrum"])
            mz_q, int_q = _clean_spectrum(mz_q, int_q)
            mz_r, int_r = _clean_spectrum(mz_r, int_r)
            scale = (max(int_q) / max(int_r)) if int_q and int_r and max(int_r) > 0 else 1.0
            bars.append(go.Bar(x=mz_q, y=int_q, name="Query", marker_color="steelblue", width=0.5,
                               hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Query</extra>"))
            bars.append(go.Bar(x=mz_r, y=[-i * scale for i in int_r], name=f"Ref (×{scale:.2f})",
                               marker_color="tomato", width=0.5,
                               hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>Reference</extra>"))
        else:
            mz, ints = json.loads(scan["raw_spectrum"])
            mz, ints = _clean_spectrum(mz, ints)
            bars.append(go.Bar(x=mz, y=ints, name="MS2", marker_color="steelblue", width=0.5,
                               hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2e}<extra>MS2</extra>"))

        fig = go.Figure(data=bars)
        fname = "_".join(os.path.basename(scan["file_path"]).split(".")[0].split("_")[11:])
        ms2_title_text = (
            f"<sub>File: {fname} | Score: {scan.get('score', 0):.4f}  |  Scan RT: {scan['rt']:.4f} min</sub><br>"
            f"<sub>Precursor m/z: {scan['precursor_MZ']:.4f}  |  Ref m/z: {scan['mz_theoretical']:.4f}  |  ppm Δ: {scan['ppm_error']:.2f}</sub>"
        )
        fig.update_layout(
            title=dict(text=ms2_title_text,x=0, xanchor="left", font=dict(size=12)),
            xaxis_title="m/z", yaxis_title="Intensity",
            barmode="overlay", hovermode="closest",
            margin=dict(l=50, r=20, t=80, b=40),
        )
        return fig

    # ── callbacks ──────────────────────────────────────────────────────────
    @app.callback(
        Output("session-store", "data"),
        Input("compound-dd", "value"),
        State("session-store", "data"),
        prevent_initial_call=False,
    )
    def init_store(compound_idx, old_state):
        if old_state and old_state.get("compound_idx") == compound_idx:
            raise dash.exceptions.PreventUpdate
        return _load_state(compound_idx)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_compound(prev, nxt, state):
        trigger = ctx.triggered_id
        if trigger not in ("prev-btn", "next-btn"):
            raise dash.exceptions.PreventUpdate
        state = _flush_to_db(state)
        delta = -1 if trigger == "prev-btn" else 1
        new_idx = (state["compound_idx"] + delta) % len(compound_options)
        new_state = _load_state(new_idx)
        new_state["last_saved"] = state.get("last_saved")
        return new_state

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
        n_scans = _count_ms2_scans(row)
        if n_scans == 0:
            raise dash.exceptions.PreventUpdate
        delta = -1 if trigger == "ms2-prev" else 1
        state["ms2_idx"] = max(0, min(state["ms2_idx"] + delta, n_scans - 1))
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("accept-suggestions", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def accept_suggestions(_, state):
        row = _compound_row(state["compound_idx"])
        if pd.notnull(row.get("suggested_rt_min")):
            state = _apply_rt_change(state, float(row["suggested_rt_min"]), float(row["suggested_rt_max"]))
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms1-radio", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_ms1_note(val, state):
        state["ms1_note"] = val
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms2-radio", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_ms2_note(val, state):
        state["ms2_note"] = MS2_LABEL_TO_VALUE.get(val, val)
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("analyst-notes", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_analyst_notes(txt, state):
        state["analyst_notes"] = txt
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("id-notes", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def set_id_notes(txt, state):
        state["id_notes"] = txt
        return state

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("ms1-graph", "relayoutData"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def rt_drag(relayout, state):
        if not relayout or state is None:
            raise dash.exceptions.PreventUpdate
        shape_updates = {}
        for k, v in relayout.items():
            if k.startswith("shapes[") and k.endswith("].x0"):
                try:
                    idx = int(k.split("[")[1].split("]")[0])
                    shape_updates[idx] = float(v)
                except (ValueError, IndexError):
                    pass
        if not shape_updates:
            raise dash.exceptions.PreventUpdate
        sorted_indices = sorted(shape_updates.keys())
        if len(sorted_indices) >= 2:
            new_min = shape_updates[sorted_indices[0]]
            new_max = shape_updates[sorted_indices[1]]
        else:
            dragged_x = list(shape_updates.values())[0]
            dist_min = abs(dragged_x - state["rt_min"])
            dist_max = abs(dragged_x - state["rt_max"])
            new_min, new_max = (dragged_x, state["rt_max"]) if dist_min <= dist_max else (state["rt_min"], dragged_x)
        return _apply_rt_change(state, new_min, new_max)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("keyboard", "event"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def handle_keyboard(event, state):
        if not event or state is None:
            raise dash.exceptions.PreventUpdate
        key = event.get("key", "")
        if not key:
            raise dash.exceptions.PreventUpdate

        rt_min, rt_max = state["rt_min"], state["rt_max"]
        changed = False
        if key == "a":   rt_min = round(rt_min - RT_STEP, 4); changed = True
        elif key == "s": rt_min = round(rt_min + RT_STEP, 4); changed = True
        elif key == "d": rt_max = round(rt_max - RT_STEP, 4); changed = True
        elif key == "f": rt_max = round(rt_max + RT_STEP, 4); changed = True
        if changed:
            return _apply_rt_change(state, rt_min, rt_max)

        if key == "ArrowLeft":
            state = _flush_to_db(state)
            new_state = _load_state((state["compound_idx"] - 1) % len(compound_options))
            new_state["last_saved"] = state.get("last_saved")
            return new_state
        if key in ("ArrowRight", " "):
            state = _flush_to_db(state)
            new_state = _load_state((state["compound_idx"] + 1) % len(compound_options))
            new_state["last_saved"] = state.get("last_saved")
            return new_state
        if key == "ArrowUp":
            row = _compound_row(state["compound_idx"])
            n_scans = _count_ms2_scans(row)
            if n_scans == 0: raise dash.exceptions.PreventUpdate
            state["ms2_idx"] = min(state["ms2_idx"] + 1, n_scans - 1)
            return state
        if key == "ArrowDown":
            row = _compound_row(state["compound_idx"])
            n_scans = _count_ms2_scans(row)
            if n_scans == 0: raise dash.exceptions.PreventUpdate
            state["ms2_idx"] = max(state["ms2_idx"] - 1, 0)
            return state
        if key == "m":
            row = _compound_row(state["compound_idx"])
            if pd.notnull(row.get("suggested_rt_min")):
                state = _apply_rt_change(state, float(row["suggested_rt_min"]), float(row["suggested_rt_max"]))
            return state
        if key in MS2_KEY_TO_LABEL:
            state["ms2_note"] = MS2_LABEL_TO_VALUE[MS2_KEY_TO_LABEL[key]]
            return state
        if key in MS1_KEY_TO_LABEL:
            state["ms1_note"] = MS1_KEY_TO_LABEL[key]
            return state

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("ms1-graph", "figure"),
        Output("ms2-graph", "figure"),
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def update_figures(state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        return _make_ms1_figure(state), _make_ms2_figure(state)

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
        n_scans = _count_ms2_scans(row)
        ms2_txt = f"MS2 Scan {state['ms2_idx']+1} of {n_scans}" if n_scans else "No MS2 data"
        pending = html.Span(
            ["⏳ Unsaved: ", html.I(row["compound_name"]),
             f" — RT [{state['rt_min']:.4f}, {state['rt_max']:.4f}] ",
             f"MS1: {state['ms1_note']} MS2: {state['ms2_note']}"],
            style={"color": "#b8860b", "fontSize": "11px", "fontWeight": "bold"},
        )
        if state.get("last_saved"):
            s = state["last_saved"]
            saved = html.Span(
                ["✔ Saved ", html.I(s["name"]),
                 f" — RT [{s['rt_min']:.4f}, {s['rt_max']:.4f}] ",
                 f"MS1: {s['ms1']} MS2: {s['ms2']} [{s['timestamp']}]"],
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
        Input("session-store", "data"),
        prevent_initial_call=False,
    )
    def sync_controls(state):
        if state is None:
            raise dash.exceptions.PreventUpdate
        ms2_label = MS2_VALUE_TO_LABEL.get(state["ms2_note"], list(MS2_LABEL_TO_VALUE.keys())[0])
        return state["analyst_notes"], state["id_notes"], state["ms1_note"], ms2_label

    return app