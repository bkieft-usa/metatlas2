import json, os
import numpy as np
import pandas as pd
import sys

import panel as pn
pn.config.max_msg_size = "50mb" 
import param
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi

pn.extension('plotly')

# ----------------------------------------------------------------------
#  Helper functions (unchanged)
# ----------------------------------------------------------------------
def to_python_type(val):
    """Convert NumPy scalar types to native Python scalars for SQLite."""
    if isinstance(val, (np.generic,)):
        return val.item()
    return val


def update_manual_curation_entry(project_db_path, curation_uid, updates):
    """Write a dict of column → value updates to the manual_curation table."""
    if not updates:
        return
    set_clause = ", ".join([f"{k} = ?" for k in updates])
    params = [to_python_type(v) for v in updates.values()] + [curation_uid]
    with dbi.get_db_connection(project_db_path) as conn:
        conn.execute(f"UPDATE manual_curation SET {set_clause} WHERE curation_uid = ?", params)


# ----------------------------------------------------------------------
#  Main UI class
# ----------------------------------------------------------------------
class CurationApp(param.Parameterized):
    """
    Panel + Param based manual‑curation UI.
    """

    # ------------------------------------------------------------------
    # 0️⃣  Parameters – *class* attributes, not instance attributes
    # ------------------------------------------------------------------
    compound_index = param.Integer(
        0, bounds=(0, 0), doc="Index of the current compound in manual_curation_df"
    )
    rt_range = param.Tuple((0.0, 0.0), doc="(rt_min, rt_max) for the RT slider")
    ms2_note = param.ObjectSelector(default="no selection", objects=[])
    ms1_note = param.ObjectSelector(default="keep", objects=[])
    analyst_notes = param.String("")
    identification_notes = param.String("")
    ms2_index = param.Integer(0, bounds=(0, 0), doc="Current MS2 scan index")

    # ------------------------------------------------------------------
    # 1️⃣  Constructor
    # ------------------------------------------------------------------
    def __init__(self, analysis_gui_obj, ms2_score_cutoff=0.5, top_n_hits=20, **kwargs):
        super().__init__(**kwargs)

        # --------------------------------------------------------------
        # Store the analysis object and preload all needed DataFrames (once)
        # --------------------------------------------------------------
        self.analysis = analysis_gui_obj
        self.project_path = analysis_gui_obj.paths["project_db_path"]
        self.rt_alignment = analysis_gui_obj.rt_alignment_number
        self.analysis_number = analysis_gui_obj.analysis_number
        self.ms2_score_cutoff = ms2_score_cutoff
        self.top_n_hits = top_n_hits

        self.manual_curation_df = dbi.get_manual_curation_entries(
            self.project_path, self.rt_alignment, self.analysis_number
        )
        self.ms1_df = dbi.get_ms1_data_for_compound(
            self.project_path, None, None, self.rt_alignment, self.analysis_number
        )
        self.ms2_df = dbi.get_ms2_data_for_compound(
            self.project_path, None, None, self.rt_alignment, self.analysis_number
        )
        self.ms2_hits_df = dbi.get_ms2_hits_for_compound(
            self.project_path, None, None, self.rt_alignment, self.analysis_number
        )

        # --------------------------------------------------------------
        # Helper lists & parameter bounds
        # --------------------------------------------------------------
        self.compound_list = list(self.manual_curation_df.index)
        self.param["compound_index"].bounds = (0, len(self.compound_list) - 1)

        # ----------------------------------------------------------------
        # Fixed option lists – update the **parameter objects**, not the values
        # ----------------------------------------------------------------
        self.ms2_options = [
            "no selection",
            "-1.0, poor match, should remove",
            "0.0, no match or no MSMS collected",
            "0.5, partial or putative match of fragments",
            "1.0, good match",
            "0.5, co‑isolated precursor, partial match",
            "1.0, co‑isolated precursor, good match",
            "0.5, single ion match, no evidence",
            "1.0, single ion match, ISTD/ref evidence",
        ]
        self.ms1_options = ["keep", "remove", "unresolvable isomers", "poor peak shape"]

        self.param["ms2_note"].objects = self.ms2_options
        self.param["ms1_note"].objects = self.ms1_options

        # --------------------------------------------------------------
        # Build UI widgets & connect callbacks
        # --------------------------------------------------------------
        self._build_widgets()
        self._wire_events()

        # --------------------------------------------------------------
        # Show the first compound
        # --------------------------------------------------------------
        self._load_compound_state(0)

    # ── UI construction ──────────────────────────────────────────────────
    def _build_widgets(self):
        # ----- Navigation -----
        self.prev_btn = pn.widgets.Button(name="◀ Prev", button_type="primary", width=80)
        self.next_btn = pn.widgets.Button(name="Next ▶", button_type="primary", width=80)
        self.counter = pn.pane.HTML(
            f"Compound 1 of {len(self.compound_list)}", width=150
        )

        # ----- Jump‑to dropdown -----
        self.compound_dd = pn.widgets.Select(
            name="Jump to:",
            options=[
                f"{i+1}: {self.manual_curation_df.iloc[i]['compound_name']}"
                for i in range(len(self.compound_list))
            ],
            width=400,
        )

        # ----- RT slider -----
        self.rt_slider = pn.widgets.RangeSlider(
            name="RT (min–max)", start=0, end=10, step=0.01, width=600
        )

        # ----- MS2 navigation -----
        self.ms2_prev = pn.widgets.Button(
            name="◀ Prev MS2", button_type="default", width=100
        )
        self.ms2_next = pn.widgets.Button(
            name="Next MS2 ▶", button_type="default", width=100
        )
        self.ms2_counter = pn.pane.HTML("MS2 Scan 0 of 0")

        # ----- Annotation widgets -----
        self.ms2_radio = pn.widgets.RadioButtonGroup(
            name="MS2 Notes", options=self.ms2_options, width=250, button_type="default"
        )
        self.ms1_radio = pn.widgets.RadioButtonGroup(
            name="MS1 Notes", options=self.ms1_options, width=250, button_type="default"
        )
        self.analyst_box = pn.widgets.TextAreaInput(
            name="Analyst Notes",
            placeholder="Enter analyst notes…",
            height=30,
            width=600,
        )
        self.id_box = pn.widgets.TextAreaInput(
            name="ID Notes",
            placeholder="Enter ID notes…",
            height=30,
            width=600,
        )
        self.accept_btn = pn.widgets.Button(
            name="Accept Suggestions", button_type="success", width=200
        )

        # ----- Plot pane (reactive) -----
        self.plot_pane = pn.panel(
            self.plot_figure,
            height=600,
            sizing_mode="stretch_width",
        )

        # ----- Layout -----
        nav_row = pn.Row(self.prev_btn, self.counter, self.next_btn, width_policy="max")
        jump_row = pn.Row(self.compound_dd, width_policy="max")
        ms2_nav_row = pn.Row(
            self.ms2_prev, self.ms2_counter, self.ms2_next, width_policy="max"
        )

        left_col = pn.Column(
            self.plot_pane,
            ms2_nav_row,
            pn.Row(self.ms2_radio),
            pn.Row(self.ms1_radio),
            width=700,
        )
        right_col = pn.Column(
            self.analyst_box,
            self.id_box,
            self.rt_slider,
            self.accept_btn,
            width=600,
        )
        self.layout = pn.Column(
            nav_row,
            jump_row,
            pn.Row(left_col, right_col),
            pn.Spacer(height=20),
        )

    # ── Wire callbacks ───────────────────────────────────────────────────
    def _wire_events(self):
        # navigation
        self.prev_btn.on_click(lambda _: self._navigate(-1))
        self.next_btn.on_click(lambda _: self._navigate(1))

        # dropdown – keep the watcher token so we can detach it later
        self._dd_watcher = self.compound_dd.param.watch(self._on_dd_select, "value")

        # RT slider
        self.rt_slider.param.watch(self._on_rt_change, "value")

        # radios → params
        self.ms2_radio.param.watch(lambda e: setattr(self, "ms2_note", e.new), "value")
        self.ms1_radio.param.watch(lambda e: setattr(self, "ms1_note", e.new), "value")

        # free‑text notes
        self.analyst_box.param.watch(lambda e: setattr(self, "analyst_notes", e.new), "value")
        self.id_box.param.watch(lambda e: setattr(self, "identification_notes", e.new), "value")

        # accept‑suggestions
        self.accept_btn.on_click(lambda _: self._accept_suggestions())

        # MS2 navigation
        self.ms2_prev.on_click(lambda _: self._set_ms2_index(self.ms2_index - 1))
        self.ms2_next.on_click(lambda _: self._set_ms2_index(self.ms2_index + 1))

    # ── Navigation helpers ───────────────────────────────────────────────
    def _navigate(self, delta: int):
        """Flush edits for the current compound and move to the next/prev."""
        self._flush_buffer()
        new_idx = (self.compound_index + delta) % len(self.compound_list)
        self._load_compound_state(new_idx)

    def _on_dd_select(self, event):
        """User selected a compound from the dropdown."""
        if event.new is None:
            return
        target = int(event.new.split(":")[0]) - 1
        self._navigate(target - self.compound_index)

    # ── Load state for a given compound ───────────────────────────────────
    def _load_compound_state(self, idx: int):
        self.compound_index = idx
        row = self.manual_curation_df.iloc[idx]

        # RT slider
        rt_min, rt_max = row["rt_min"], row["rt_max"]
        self.rt_slider.start = rt_min - 1.0
        self.rt_slider.end   = rt_max + 1.0
        self.rt_slider.value = (rt_min, rt_max)
        self.rt_range = (rt_min, rt_max)

        # annotation values
        self.ms2_note = row["ms2_notes"] if pd.notnull(row["ms2_notes"]) else self.ms2_options[0]
        self.ms1_note = row["ms1_notes"] if pd.notnull(row["ms1_notes"]) else self.ms1_options[0]
        self.analyst_notes = row["analyst_notes"] or ""
        self.identification_notes = row["identification_notes"] or ""

        # keep radios in sync
        self.ms2_radio.value = self.ms2_note
        self.ms1_radio.value = self.ms1_note

        # UI counters
        self.counter.object = f"Compound {idx+1} of {len(self.compound_list)}"

        # keep dropdown in sync without firing its watcher
        self.compound_dd.param.unwatch(self._dd_watcher)
        self.compound_dd.value = self.compound_dd.options[idx]
        self._dd_watcher = self.compound_dd.param.watch(self._on_dd_select, "value")

        # MS2 scan index & bounds
        n_scans = self._count_ms2_scans(row)
        self.param["ms2_index"].bounds = (0, max(0, n_scans - 1))
        self.ms2_index = 0

    def _count_ms2_scans(self, row) -> int:
        inchi, adduct = row["inchi_key"], row["adduct"]
        subset = self.ms2_df[(self.ms2_df["inchi_key"] == inchi) &
                             (self.ms2_df["adduct"] == adduct)]
        if subset.empty:
            return 0
        return subset.groupby(["inchi_key", "adduct", "file_path", "rt"]).ngroups

    # ── Parameter‑change callbacks ───────────────────────────────────────
    def _on_rt_change(self, event):
        self.rt_range = event.new

    def _set_ms2_index(self, new_idx: int):
        lo, hi = self.param["ms2_index"].bounds
        self.ms2_index = max(lo, min(new_idx, hi))

    def _accept_suggestions(self):
        row = self.manual_curation_df.iloc[self.compound_index]
        if pd.notnull(row["suggested_rt_min"]):
            self.rt_slider.value = (row["suggested_rt_min"], row["suggested_rt_max"])

    # ── Flush edits to DB ─────────────────────────────────────────────────
    def _flush_buffer(self):
        uid = self.manual_curation_df.iloc[self.compound_index]["curation_uid"]
        rt_min, rt_max = self.rt_range
        updates = {
            "rt_min": rt_min,
            "rt_max": rt_max,
            "rt_peak": (rt_min + rt_max) / 2,
            "ms2_notes": self.ms2_note,
            "ms1_notes": self.ms1_note,
            "analyst_notes": self.analyst_notes,
            "identification_notes": self.identification_notes,
        }
        update_manual_curation_entry(self.project_path, uid, updates)

        # keep in‑memory df in sync
        ix = self.manual_curation_df.index[self.compound_index]
        for k, v in updates.items():
            self.manual_curation_df.at[ix, k] = v

    # ── Plotly figure (reactive) ────────────────────────────────────────
    @pn.depends("compound_index", "rt_range", "ms2_index")
    def plot_figure(self):
        """Generate the Plotly figure for the current compound & MS2 scan."""
        row = self.manual_curation_df.iloc[self.compound_index]
        inchi, adduct = row["inchi_key"], row["adduct"]

        # ---- MS1 ----
        ms1 = self.ms1_df[(self.ms1_df["inchi_key"] == inchi) &
                          (self.ms1_df["adduct"] == adduct)]

        # ---- MS2 ----
        ms2 = self.ms2_df[(self.ms2_df["inchi_key"] == inchi) &
                          (self.ms2_df["adduct"] == adduct)]

        # ---- MS2 hits ----
        hits = self.ms2_hits_df[(self.ms2_hits_df["inchi_key"] == inchi) &
                                 (self.ms2_hits_df["adduct"] == adduct) &
                                 (self.ms2_hits_df["score"] >= self.ms2_score_cutoff)]        

        # ---- Merge MS2 data and hits ----
        if ms2.empty and hits.empty:
            scans = pd.DataFrame()
        elif not ms2.empty and hits.empty:
            scans = ms2.sort_values("rt")
        else:
            merged = pd.merge(
                ms2,
                hits,
                left_on=["inchi_key", "adduct", "file_path", "rt"],
                right_on=["inchi_key", "adduct", "file_path", "rt"],
                how="left",
            )
            merged["score"] = merged["score"].fillna(0)
            scans = merged.sort_values(["score", "rt"], ascending=[False, True])

        scans = scans.head(self.top_n_hits) if len(scans) > self.top_n_hits else scans
        n_scans = len(scans)
        self.ms2_counter.object = (
            f"MS2 Scan {self.ms2_index+1} of {n_scans}" if n_scans else "No MS2 data"
        )
        scan = scans.iloc[self.ms2_index] if n_scans else None

        # ---- Figure layout ----
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.4, 0.6],
            vertical_spacing=0.05,
            subplot_titles=("MS2 Spectrum", "MS1 EIC"),
            specs=[[{}], [{}]],
        )

        # bottom panel – MS1 EIC
        if not ms1.empty:
            for fp in ms1["file_path"].unique():
                sub = ms1[ms1["file_path"] == fp]
                name = "_".join(os.path.basename(fp).split(".")[0].split("_")[11:])
                colour = "gray"
                for kw, col in {
                    "istd": "blue", "qc": "blue",
                    "exctrl": "red", "txctrl": "red",
                    "refstd": "black",
                }.items():
                    if kw in name.lower():
                        colour = col
                        break
                fig.add_trace(
                    go.Scatter(x=sub["rt"], y=sub["i"], mode="lines",
                               name=name, line=dict(width=2, color=colour)),
                    row=2, col=1,
                )

        # RT reference lines
        rt_min, rt_max = self.rt_range
        rt_peak = (rt_min + rt_max) / 2
        fig.add_vline(x=rt_min, line_dash="dot", line_color="purple", row=2, col=1)
        fig.add_vline(x=rt_max, line_dash="dot", line_color="purple", row=2, col=1)
        fig.add_vline(x=rt_peak, line_dash="dash", line_color="black", row=2, col=1)
        fig.add_vline(x=row["atlas_rt_peak"], line_dash="dash", line_color="green", row=2, col=1)

        # top panel – MS2 spectrum
        if scan is not None:
            if pd.notnull(scan.get("qry_spectrum")) and pd.notnull(scan.get("ref_spectrum")):
                mz_q, int_q = json.loads(scan["qry_spectrum"])
                mz_r, int_r = json.loads(scan["ref_spectrum"])
                scale = max(int_q) / max(int_r) if max(int_r) else 1
                int_r = [i * scale for i in int_r]

                fig.add_trace(go.Bar(x=mz_q, y=int_q, name="Query",
                                     marker_color="red", width=0.5,
                                     hovertemplate="Query<br>m/z: %{x:.4f}<br>Int: %{y:.2f}<extra></extra>"),
                              row=1, col=1)
                fig.add_trace(go.Bar(x=mz_r, y=[-i for i in int_r], name="Reference",
                                     marker_color="blue", width=0.5,
                                     hovertemplate="Reference<br>m/z: %{x:.4f}<br>Int: %{y:.2f}<extra></extra>"),
                              row=1, col=1)
            else:
                mz_raw, int_raw = json.loads(scan["raw_spectrum"])
                fig.add_trace(go.Bar(x=mz_raw, y=int_raw, name="MS2",
                                     marker_color="red", width=0.5,
                                     hovertemplate="m/z: %{x:.4f}<br>Int: %{y:.2f}<extra></extra>"),
                              row=1, col=1)

        # final layout tweaks
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_text="RT", row=2, col=1)
        fig.update_yaxes(title_text="Intensity", row=2, col=1)
        fig.update_xaxes(title_text="m/z", row=1, col=1)
        fig.update_yaxes(title_text="Intensity", row=1, col=1)
        fig.add_hline(y=0, line_color="black", row=1, col=1)

        return fig

    # ── Public entry point ───────────────────────────────────────────────
    def show(self):
        """Return the root Panel layout for display."""
        return self.layout.servable()