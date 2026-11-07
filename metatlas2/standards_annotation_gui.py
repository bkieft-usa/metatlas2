"""
metatlas2/standards_annotation_gui.py

Dash-based GUI for the standards-annotation workflow.

Layout (per compound):
  - Top bar: compound dropdown, navigation buttons, Save & Exit.
  - Per-chromatography columns:
      - Peak checklist: POS / NEG sections, each with CE subsections
      - Manual peak entry row
      - EIC row: one graph per CE label (up to MAX_CE=2), side by side
      - Scale toggle + MS2 button row
      - TIC graph (full width)
      - InjBl graph (full width, below TIC)
      - MS2 mini-plots (collapsible)

Key design decisions:
  - ``peaks_holder``: a one-element list ``[peaks_df]`` used as a mutable
    container so that the closure can update the in-memory DataFrame after
    a manual peak is added, without needing ``nonlocal``.
  - MS2 triangle y-position: interpolated from the EIC row that matches
    both ``peak.mz_rt_uid`` AND ``peak.filename``, not just the first file.
  - Workflow state: keyed on ``(compound_uid, chromatography)`` so that
    C18 and HILIC states are tracked independently.
  - Analyst selections: cleared and re-saved by ``compound_uid`` (not
    compound_name) to avoid LIKE-pattern ambiguity.
  - Collision energy: files are grouped by ``collision_energy_label``
    (extracted from run_metadata field).  Each CE gets its own EIC graph
    (MAX_CE=2 fixed slots with IDs ``sa-eic-graph-{slot}-{ce_idx}``).
    Checklists show CE subsection headers within each polarity group.
  - InjBl graph: displayed below the TIC (full width).

Selections are persisted to ``sa_analyst_selections`` on every Next/Prev/Save.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import threading
import time
import uuid
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf
import metatlas2.standards_annotation_db as sadb
import metatlas2.standards_annotation_tools as sat

logger = lcf.get_logger("standards_annotation_gui")

# ---------------------------------------------------------------------------
# Fixed adduct color palette (up to 12 adducts)
# ---------------------------------------------------------------------------
_ADDUCT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # yellow-green
    "#7f7f7f",  # gray
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
]

# Maximum number of distinct CE labels per chrom panel
MAX_CE = 2


def _adduct_color_from_list(adduct: str, adduct_list: list[str]) -> str:
    try:
        i = adduct_list.index(adduct)
    except ValueError:
        i = 0
    return _ADDUCT_COLORS[i % len(_ADDUCT_COLORS)]


def _extract_run_number(filename: str) -> str:
    """Extract numeric run number from filename (strips 'Run' prefix)."""
    try:
        stem = os.path.basename(filename)
        if stem.endswith(".h5"):
            stem = stem[:-3]
        fields = fpf.parse_file_name(stem + ".h5")
        parts = [fields.get("run_number", "")]
        return "_".join(p for p in parts if p)
    except Exception:
        return os.path.basename(filename)


def _short_filename(filename: str) -> str:
    """Return a compact label from a filename using selected fpf fields."""
    try:
        stem = os.path.basename(filename)
        if stem.endswith(".h5"):
            stem = stem[:-3]
        fields = fpf.parse_file_name(stem + ".h5")
        parts = [
            fields.get("sample_name", ""),
            fields.get("replicate", ""),
            fields.get("run_metadata", ""),
            fields.get("run_number", ""),
        ]
        return "_".join(p for p in parts if p)
    except Exception:
        return os.path.basename(filename)


def _run_number_from_filename(filename: str) -> str:
    """Extract numeric run number from filename (strips 'Run' prefix)."""
    try:
        m = re.search(r"[Rr]un(\d+)", os.path.basename(filename))
        return m.group(1) if m else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main GUI factory
# ---------------------------------------------------------------------------

def build_standards_annotation_gui(
    data: "StandardsAnnotationData",
    config: dict[str, Any] | str | None = None,
    port: int = 8050,
) -> None:
    """Build and run the standards-annotation Dash GUI.

    Args:
        data:   Extracted data container (from :func:`run_standards_extraction`).
        config: Parsed config dict, path to config YAML, or None.
        port:   Starting port for the Dash server.
    """
    cfg = config
    db_path = data.project_db_path
    gui_cfg = cfg.get("gui", {})
    lcmsruns_color_map = gui_cfg.get("gui_lcmsruns_colors", {})

    ms1_df = data.ms1_df
    ms2_df = data.ms2_df
    tic_by_file = data.tic_by_file
    expanded_table = data.expanded_table
    injbl_ms1_df = data.injbl_ms1_df if hasattr(data, "injbl_ms1_df") else pd.DataFrame()

    # ── Load peaks and MS2 associations from DB ───────────────────────────────
    peaks_holder: list[pd.DataFrame] = [sadb.load_peaks(db_path)]
    peak_ms2_df = sadb.load_peak_ms2(db_path)

    if peaks_holder[0].empty:
        logger.warning(
            "No peaks found in sa_peaks. Run run_standards_extraction() first "
            "to detect peaks before launching the GUI."
        )

    # ── Pre-compute lookup structures from expanded_table ─────────────────────

    compound_names = list(dict.fromkeys(expanded_table["compound_name"].tolist()))

    compound_name_to_uids: dict[str, list[str]] = {}
    for _, row in expanded_table.iterrows():
        name = str(row["compound_name"])
        cuid = str(row["compound_uid"])
        if name not in compound_name_to_uids:
            compound_name_to_uids[name] = []
        if cuid not in compound_name_to_uids[name]:
            compound_name_to_uids[name].append(cuid)

    compound_chroms: dict[str, list[str]] = {}
    for name in compound_names:
        chroms = list(dict.fromkeys(
            expanded_table.loc[
                expanded_table["compound_name"] == name, "chromatography"
            ].tolist()
        ))
        compound_chroms[name] = chroms

    compound_chrom_adducts: dict[tuple[str, str], list[str]] = {}
    for name in compound_names:
        for chrom in compound_chroms[name]:
            sub = expanded_table[
                (expanded_table["compound_name"] == name) &
                (expanded_table["chromatography"] == chrom)
            ]
            adducts = list(dict.fromkeys(sub["adduct"].tolist()))
            compound_chrom_adducts[(name, chrom)] = adducts

    # mz_rt_uid -> polarity
    uid_polarity: dict[str, str] = {
        str(row["mz_rt_uid"]): str(row.get("polarity", "POS")).upper()
        for _, row in expanded_table.iterrows()
    }

    # mz_rt_uid -> mz_theoretical
    uid_mz_theoretical: dict[str, float] = {
        str(row["mz_rt_uid"]): float(row.get("mz_theoretical", float("nan")))
        for _, row in expanded_table.iterrows()
    }

    # mz_rt_uid -> collision_energy_label
    uid_ce_label: dict[str, str] = {
        str(row["mz_rt_uid"]): str(row.get("collision_energy_label", ""))
        for _, row in expanded_table.iterrows()
    }

    # (compound_name, chromatography) -> annotation_notes
    compound_chrom_notes: dict[tuple[str, str], str] = {}
    for _, row in expanded_table.iterrows():
        key = (str(row["compound_name"]), str(row.get("chromatography", "")))
        if key not in compound_chrom_notes:
            note = str(row.get("annotation_notes", "") or "")
            if note:
                compound_chrom_notes[key] = note

    # compound_name -> metadata
    compound_meta: dict[str, dict[str, Any]] = {}
    for name in compound_names:
        sub = expanded_table[expanded_table["compound_name"] == name].iloc[0]
        compound_meta[name] = {
            "inchi_key": str(sub.get("inchi_key", "")),
            "smiles": str(sub.get("smiles", "")),
            "inchi": str(sub.get("inchi", "")),
            "annotation_notes": str(sub.get("annotation_notes", "")),
        }

    # (compound_name, chromatography, adduct) -> list of mz_rt_uids
    uid_map: dict[tuple[str, str, str], list[str]] = {}
    for _, row in expanded_table.iterrows():
        key = (
            str(row["compound_name"]),
            str(row.get("chromatography", "")),
            str(row["adduct"]),
        )
        uid_map.setdefault(key, []).append(str(row["mz_rt_uid"]))

    # mz_rt_uid -> filename
    uid_to_filename: dict[str, str] = {
        str(row["mz_rt_uid"]): str(row.get("filename", ""))
        for _, row in expanded_table.iterrows()
    }

    # Build InjBl hover map
    injbl_hover_map: dict[tuple[str, str, str, str], list[str]] = {}
    if not injbl_ms1_df.empty and "mz_rt_uid" in injbl_ms1_df.columns:
        uid_to_et: dict[str, dict] = {
            str(row["mz_rt_uid"]): {
                "compound_name": str(row["compound_name"]),
                "chromatography": str(row.get("chromatography", "")),
                "adduct": str(row["adduct"]),
                "filename": str(row.get("filename", "")),
            }
            for _, row in expanded_table.iterrows()
        }
        for _, injbl_row in injbl_ms1_df.iterrows():
            uid = str(injbl_row.get("mz_rt_uid", ""))
            injbl_fn = str(injbl_row.get("injbl_filename", ""))
            et = uid_to_et.get(uid)
            if et is None or not injbl_fn:
                continue
            key = (et["compound_name"], et["chromatography"], et["adduct"], injbl_fn)
            target_fn = et["filename"]
            if target_fn not in injbl_hover_map.get(key, []):
                injbl_hover_map.setdefault(key, []).append(target_fn)

    # Union RT grid per (compound, chrom, adduct) for the manual RT dropdown
    union_rts_map: dict[tuple[str, str, str], list[float]] = {}
    for name in compound_names:
        for chrom in compound_chroms[name]:
            for adduct in compound_chrom_adducts.get((name, chrom), []):
                uids = uid_map.get((name, chrom, adduct), [])
                sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
                all_rts: list[np.ndarray] = []
                for _, row in sub.iterrows():
                    rts = row.get("spec_rts", [])
                    if rts is not None and len(rts) > 0:
                        all_rts.append(np.asarray(rts, dtype=np.float64))
                if all_rts:
                    union_rts = np.unique(np.concatenate(all_rts))
                    union_rts_map[(name, chrom, adduct)] = [float(r) for r in union_rts]
                else:
                    union_rts_map[(name, chrom, adduct)] = []

    # RT grid per (compound, chrom, adduct, ce_label) for CE-filtered manual RT dropdown
    # Also build CE file options: (compound, chrom, adduct) -> list of
    #   {"label": "CE205060 (Run135)", "value": mz_rt_uid}
    union_rts_map_by_ce: dict[tuple[str, str, str, str], list[float]] = {}
    ce_file_options_map: dict[tuple[str, str, str], list[dict]] = {}
    for name in compound_names:
        for chrom in compound_chroms[name]:
            for adduct in compound_chrom_adducts.get((name, chrom), []):
                uids = uid_map.get((name, chrom, adduct), [])
                opts: list[dict] = []
                seen_ce_run: set[tuple[str, str]] = set()
                for uid in uids:
                    ce = uid_ce_label.get(uid, "")
                    fn = uid_to_filename.get(uid, "")
                    run_n = _run_number_from_filename(fn)
                    key = (ce, run_n)
                    if key in seen_ce_run:
                        continue
                    seen_ce_run.add(key)
                    ce_display = ce if ce else "no CE"
                    run_display = f" (Run{run_n})" if run_n else ""
                    opts.append({
                        "label": f"{ce_display}{run_display}",
                        "value": uid,  # value = mz_rt_uid to identify the specific file
                    })
                    # RT grid for this CE
                    sub = ms1_df[ms1_df["mz_rt_uid"] == uid]
                    all_rts_ce: list[np.ndarray] = []
                    for _, row in sub.iterrows():
                        rts = row.get("spec_rts", [])
                        if rts is not None and len(rts) > 0:
                            all_rts_ce.append(np.asarray(rts, dtype=np.float64))
                    if all_rts_ce:
                        union_rts_ce = np.unique(np.concatenate(all_rts_ce))
                        union_rts_map_by_ce[(name, chrom, adduct, uid)] = [float(r) for r in union_rts_ce]
                    else:
                        union_rts_map_by_ce[(name, chrom, adduct, uid)] = []
                ce_file_options_map[(name, chrom, adduct)] = opts

    # peak_uid -> MS2 row
    peak_ms2_by_uid: dict[str, dict] = {}
    if not peak_ms2_df.empty:
        for _, row in peak_ms2_df.iterrows():
            peak_ms2_by_uid[str(row["peak_uid"])] = row.to_dict()

    # ── Workflow state ────────────────────────────────────────────────────────
    uid_to_compound_name: dict[str, str] = {}
    for _, row in expanded_table.iterrows():
        uid_to_compound_name[str(row["compound_uid"])] = str(row["compound_name"])

    first_incomplete_uid, _ = sadb.get_first_incomplete_compound(db_path)
    if first_incomplete_uid:
        first_incomplete_name = uid_to_compound_name.get(first_incomplete_uid)
        if first_incomplete_name and first_incomplete_name in compound_names:
            starting_idx = compound_names.index(first_incomplete_name)
        else:
            starting_idx = 0
    else:
        starting_idx = 0

    # ── Helper functions ──────────────────────────────────────────────────────

    def _to_float(v, default: float = float("nan")) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _to_float_list(v) -> list[float]:
        if v is None:
            return []
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                return []
        try:
            return [float(x) for x in list(v)]
        except (TypeError, ValueError):
            return []

    def _get_compound_name(idx: int) -> str:
        if 0 <= idx < len(compound_names):
            return compound_names[idx]
        return compound_names[0]

    def _adduct_color(adduct: str, adduct_list: list[str]) -> str:
        return _adduct_color_from_list(adduct, adduct_list)

    def _get_file_color(filename: str) -> str:
        for key, color in lcmsruns_color_map.items():
            if key.lower() in filename.lower():
                return color
        return "gray"

    def _peaks_for_chrom(compound_name: str, chrom: str) -> pd.DataFrame:
        """Return rows from peaks_holder[0] for this compound+chrom."""
        peaks_df = peaks_holder[0]
        if peaks_df.empty:
            return pd.DataFrame()
        mask = (
            (peaks_df["compound_name"] == compound_name) &
            (peaks_df["chromatography"] == chrom)
        )
        return peaks_df[mask].copy()

    def _get_ms2_for_peak(peak_uid: str) -> dict | None:
        return peak_ms2_by_uid.get(peak_uid)

    def _get_ms1_row_for_peak(peak_row: pd.Series) -> pd.Series | None:
        """Return the MS1 row matching peak.mz_rt_uid AND peak.filename exactly."""
        uid = str(peak_row.get("mz_rt_uid", ""))
        filename = str(peak_row.get("filename", ""))
        if not uid or not filename:
            return None
        mask = (ms1_df["mz_rt_uid"] == uid) & (ms1_df["filename"] == filename)
        matches = ms1_df[mask]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _build_checklist_options_for_chrom(
        compound_name: str,
        chrom: str,
    ) -> tuple[list[dict], list[str], int]:
        """Build checklist options with POS/NEG and CE subsections.

        Structure:
            -- POS --
              -- CE205060 --
                [M+H]+ (peak1) [...]
              -- CE102040 --
                [M+H]+ (peak1) [...]
            -- NEG --
              -- CE205060 --
                [M-H]- (peak1) [...]

        Returns:
            options:         List of option dicts for dcc.Checklist.
            default_checked: peak_uids pre-checked (manual peaks).
            n_items:         Total number of selectable peak rows.
        """
        chrom_peaks = _peaks_for_chrom(compound_name, chrom)

        options: list[dict] = []
        default_checked: list[str] = []

        # CE buckets are derived from each peak's own CE label
        pos_ce_entries: dict[str, list[tuple]] = {}
        neg_ce_entries: dict[str, list[tuple]] = {}

        for _, peak_row in chrom_peaks.iterrows():
            peak_uid = str(peak_row["peak_uid"])
            adduct = str(peak_row["adduct"])
            polarity = str(peak_row.get("polarity", "POS")).upper()
            rt_peak = float(peak_row["rt_peak"])
            intensity = float(peak_row["peak_intensity"])
            peak_num = int(peak_row.get("peak_num", 1))
            is_manual = bool(peak_row.get("is_manual", False))
            filename = str(peak_row.get("filename", ""))
            _peak_uid_for_ce = str(peak_row.get("mz_rt_uid", ""))
            ce = str(peak_row.get("collision_energy_label") or "") or uid_ce_label.get(_peak_uid_for_ce, "")
            run_label = _run_number_from_filename(filename)
            run_str = f"; Run{run_label} ({ce})" if run_label else ""
            has_ms2 = peak_uid in peak_ms2_by_uid
            prefix = "* " if has_ms2 else ""
            manual_str = " [manual]" if is_manual else ""
            mz_obs = float(peak_row.get("mz_observed", float("nan")))
            mz_obs_str = f"; m/z={mz_obs:.4f}" if not np.isnan(mz_obs) else ""
            label = (
                f"{prefix}{adduct} (peak{peak_num}) "
                f"[RT={rt_peak:.3f}; Int={intensity:.2e}{mz_obs_str}{run_str}]{manual_str}"
            )
            entry = (intensity, adduct, peak_uid, label, is_manual, polarity)
            if is_manual:
                default_checked.append(peak_uid)

            if polarity == "NEG":
                neg_ce_entries.setdefault(ce, []).append(entry)
            else:
                pos_ce_entries.setdefault(ce, []).append(entry)

        n_items = 0

        def _emit_polarity_section(pol_label: str, ce_entries: dict[str, list[tuple]]):
            nonlocal n_items
            if not any(ce_entries.values()):
                return
            options.append({
                "label": f"-- {pol_label} --",
                "value": f"__{pol_label.lower()}_header__",
                "disabled": True,
            })

            def _ce_key(c: str) -> float:
                m = re.search(r"\d+(?:\.\d+)?", c)
                return float(m.group()) if m else float("-inf")

            show_ce_headers = any(ce_entries)
            for ce in sorted(ce_entries, key=_ce_key, reverse=True):
                entries = ce_entries[ce]
                if show_ce_headers:
                    ce_display = ce if ce else "no CE"
                    options.append({
                        "label": f"  -- {ce_display} --",
                        "value": f"__ce_{pol_label.lower()}_{ce}_header__",
                        "disabled": True,
                    })
                entries.sort(key=lambda e: e[0], reverse=True)
                for _, adduct, peak_uid, label, is_manual, _ in entries:
                    options.append({"label": label, "value": peak_uid})
                    n_items += 1

        _emit_polarity_section("POS", pos_ce_entries)
        _emit_polarity_section("NEG", neg_ce_entries)

        return options, default_checked, n_items

    def _split_options_by_polarity(
        options: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Split flat options list into POS and NEG sub-lists (excluding headers)."""
        pos_opts, neg_opts = [], []
        current = "pos"
        for o in options:
            if o.get("disabled"):
                lbl = o.get("label", "")
                if "-- NEG --" in lbl.upper() or lbl.strip().startswith("-- NEG"):
                    current = "neg"
                elif "-- POS --" in lbl.upper() or lbl.strip().startswith("-- POS"):
                    current = "pos"
                # CE subsection headers are also disabled — skip them
                continue
            if current == "pos":
                pos_opts.append(o)
            else:
                neg_opts.append(o)
        return pos_opts, neg_opts

    def _get_rt_bounds_for_chrom(
        compound_name: str, chrom: str
    ) -> tuple[float, float]:
        adducts = compound_chrom_adducts.get((compound_name, chrom), [])
        all_rt_min, all_rt_max = [], []
        for adduct in adducts:
            uids = uid_map.get((compound_name, chrom, adduct), [])
            sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
            for _, row in sub.iterrows():
                rts = row.get("spec_rts", [])
                if rts is not None and len(rts) > 0:
                    all_rt_min.append(float(np.min(rts)))
                    all_rt_max.append(float(np.max(rts)))
        if not all_rt_min:
            return 0.0, 20.0
        return max(0.0, min(all_rt_min) - 0.1), max(all_rt_max) + 0.1

    # CE dash-style map: first CE = solid, second CE = dash.
    # Built once at GUI startup from uid_ce_label values.
    _ALL_CE_LABELS: list[str] = list(dict.fromkeys(
        ce for ce in uid_ce_label.values() if ce
    ))
    _CE_DASH_MAP: dict[str, str] = {}
    _CE_DASH_STYLES = ["solid", "dash", "dot", "dashdot"]
    for _i, _ce in enumerate(_ALL_CE_LABELS):
        _CE_DASH_MAP[_ce] = _CE_DASH_STYLES[_i % len(_CE_DASH_STYLES)]
    # Files with no CE label use solid
    _CE_DASH_MAP[""] = "solid"

    def _make_eic_figure_for_polarity(
        compound_name: str,
        chrom: str,
        polarity_filter: str,
        y_scale: str = "linear",
    ) -> go.Figure:
        """Build the EIC figure for one polarity within a chromatography panel.

        Plots all adducts for the given polarity (POS or NEG).  Line dash style
        encodes the collision energy label: solid = first CE, dash = second CE.
        Peak markers and MS2 triangles are filtered to the same polarity.

        Args:
            compound_name:   Compound name.
            chrom:           Chromatography label.
            polarity_filter: ``"POS"`` or ``"NEG"``.
            y_scale:         ``"linear"`` or ``"log"``.
        """
        adducts = compound_chrom_adducts.get((compound_name, chrom), [])
        fig = go.Figure()
        rt_lo, rt_hi = _get_rt_bounds_for_chrom(compound_name, chrom)
        pol = polarity_filter.upper()

        # Collect UIDs that belong to this polarity
        pol_uids: set[str] = {
            uid for uid, p in uid_polarity.items() if p.upper() == pol
        }

        y_max = 1.0
        for adduct in adducts:
            uids = [u for u in uid_map.get((compound_name, chrom, adduct), [])
                    if u in pol_uids]
            sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
            for _, row in sub.iterrows():
                ints = row.get("spec_ints", [])
                if ints is not None and len(ints) > 0:
                    y_max = max(y_max, float(np.nanmax(ints)))
        y_upper = y_max * 1.1

        # Track which adducts have had a legend entry added
        shown_adduct_legend: set[str] = set()

        for adduct in adducts:
            color = _adduct_color(adduct, adducts)
            uids = [u for u in uid_map.get((compound_name, chrom, adduct), [])
                    if u in pol_uids]
            sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]

            for _, row in sub.iterrows():
                fn = str(row.get("filename", ""))
                rts = row.get("spec_rts", [])
                ints = row.get("spec_ints", [])
                if rts is None or len(rts) == 0:
                    continue
                uid = str(row.get("mz_rt_uid", ""))
                ce = uid_ce_label.get(uid, "")
                dash_style = _CE_DASH_MAP.get(ce, "solid")
                ce_display = ce if ce else ""
                # Adduct legend: one solid colored entry per adduct (no CE in name)
                show_adduct_leg = adduct not in shown_adduct_legend
                if show_adduct_leg:
                    shown_adduct_legend.add(adduct)
                fig.add_trace(go.Scattergl(
                    x=list(rts), y=list(ints),
                    mode="lines",
                    line=dict(color=color, width=1.2, dash=dash_style),
                    name=adduct,
                    legendgroup=adduct,
                    showlegend=show_adduct_leg,
                    hovertemplate=(
                        f"{adduct} {pol}<br>"
                        f"{ce_display}<br>"
                        f"{_short_filename(fn)}"
                        f"<extra></extra>"
                    ),
                ))

        # Draw peaks filtered to this polarity.
        # Use direct column access to avoid index-alignment issues.
        chrom_peaks = _peaks_for_chrom(compound_name, chrom)
        pol_peaks = chrom_peaks[
            chrom_peaks["polarity"].str.upper() == pol
        ] if "polarity" in chrom_peaks.columns else chrom_peaks

        for _, peak_row in pol_peaks.iterrows():
            adduct = str(peak_row["adduct"])
            color = _adduct_color(adduct, adducts)
            rt_peak = float(peak_row["rt_peak"])
            peak_int = float(peak_row["peak_intensity"])
            peak_uid = str(peak_row["peak_uid"])
            is_manual = bool(peak_row.get("is_manual", False))
            peak_num = int(peak_row.get("peak_num", 1))
            ce = str(peak_row.get("collision_energy_label", ""))
            file = str(peak_row.get("filename", ""))
            run_number = _run_number_from_filename(file)

            mz_obs = float(peak_row.get("mz_observed", float("nan")))
            mz_theo = float(peak_row.get("mz_theoretical", float("nan")))
            mz_obs_str = f"{mz_obs:.5f}" if not np.isnan(mz_obs) else "N/A"
            mz_theo_str = f"{mz_theo:.5f}" if not np.isnan(mz_theo) else "N/A"
            symbol = "circle-open" if is_manual else "circle"
            marker_kw = dict(color=color, size=12, symbol=symbol)
            if is_manual:
                marker_kw["line"] = dict(width=2)
            fig.add_trace(go.Scatter(
                x=[rt_peak], y=[peak_int], mode="markers",
                marker=marker_kw,
                name=f"{adduct} peak{peak_num}",
                legendgroup=f"{adduct}" + (f" ({ce})" if ce else ""),
                showlegend=False,
                hovertemplate=(
                    f"MS1 Peak for Run{run_number} ({ce})<br>"
                    f"{adduct} (peak{peak_num})<br>"
                    f"RT={rt_peak:.4f}<br>"
                    f"Int={peak_int:.2e}<br>"
                    f"Theo m/z={mz_theo_str}<br>"
                    f"Meas m/z={mz_obs_str}"
                    f"<extra></extra>"
                ),
            ))

            ms2 = _get_ms2_for_peak(peak_uid)
            if ms2:
                scan_rt = float(ms2.get("scan_rt", float("nan")))
                if not np.isnan(scan_rt):
                    # Y-position: interpolate the EIC of the peak's own MS1 row
                    ms1_row = _get_ms1_row_for_peak(peak_row)
                    if ms1_row is not None:
                        raw_rts = ms1_row.get("spec_rts", [])
                        raw_ints = ms1_row.get("spec_ints", [])
                        rts_arr = (
                            np.asarray(raw_rts, dtype=np.float64)
                            if raw_rts is not None and np.ndim(raw_rts) > 0
                            else np.asarray([], dtype=np.float64)
                        )
                        ints_arr = (
                            np.asarray(raw_ints, dtype=np.float64)
                            if raw_ints is not None and np.ndim(raw_ints) > 0
                            else np.asarray([], dtype=np.float64)
                        )
                        ms2_int = (
                            float(np.interp(scan_rt, rts_arr, ints_arr))
                            if len(rts_arr) > 0 else peak_int
                        )
                    else:
                        ms2_int = peak_int

                    # Find other peaks (this polarity) whose best MS2 scan is
                    # the same scan: same file and ~identical scan RT.
                    shared_nums = {peak_num}
                    for _, other in pol_peaks.iterrows():
                        if str(other["peak_uid"]) == peak_uid:
                            continue
                        o_ms2 = _get_ms2_for_peak(str(other["peak_uid"]))
                        if o_ms2 is None:
                            continue
                        o_rt = float(o_ms2.get("scan_rt", float("nan")))
                        if np.isnan(o_rt) or not np.isclose(
                            o_rt, scan_rt, rtol=0.0, atol=1e-6
                        ):
                            continue
                        if str(o_ms2.get("filename", "")) != str(ms2.get("filename", "")):
                            continue
                        shared_nums.add(int(other.get("peak_num", 1)))
                    nums_sorted = sorted(shared_nums)
                    if len(nums_sorted) > 1:
                        peaks_str = "peak" + " and peak".join(str(n) for n in nums_sorted)
                    else:
                        peaks_str = f"peak{nums_sorted[0]}"

                    ms2_prec_mz = float(ms2.get("precursor_mz", float("nan")))
                    ms2_prec_int = float(ms2.get("precursor_intensity", float("nan")))
                    ms2_prec_mz_str = f"{ms2_prec_mz:.5f}" if not np.isnan(ms2_prec_mz) else "N/A"
                    ms2_prec_int_str = f"{ms2_prec_int:.2e}" if not np.isnan(ms2_prec_int) else "N/A"
                    run_number = _run_number_from_filename(
                        str(ms2.get("filename", "")) or str(peak_row.get("filename", ""))
                    )
                    run_disp = f"Run{run_number}" if run_number else "file"
                    ce_disp = ce if ce else "no CE"
                    tri_sym = "triangle-up-open" if is_manual else "triangle-up"
                    tri_kw = dict(color=color, size=12, symbol=tri_sym)
                    if is_manual:
                        tri_kw["line"] = dict(width=2)
                    fig.add_trace(go.Scatter(
                        x=[scan_rt], y=[ms2_int], mode="markers",
                        marker=tri_kw,
                        name=f"{adduct} MS2",
                        legendgroup=f"{adduct}" + (f" ({ce})" if ce else ""),
                        showlegend=False,
                        hovertemplate=(
                            f"Best MS2 scan for {run_disp} ({ce_disp})<br>"
                            f"{adduct} ({peaks_str})<br>"
                            f"RT={scan_rt:.4f}<br>"
                            f"Int={ms2_prec_int_str}<br>"
                            f"Theo precursor m/z={mz_theo_str}<br>"
                            f"Meas precursor m/z={ms2_prec_mz_str}"
                            f"<extra></extra>"
                        ),
                    ))

        # All MS2 scan markers along the bottom (filtered to this polarity)
        for adduct in adducts:
            color = _adduct_color(adduct, adducts)
            uids = [u for u in uid_map.get((compound_name, chrom, adduct), [])
                    if u in pol_uids]
            for uid in uids:
                uid_ms2 = (
                    ms2_df[ms2_df["mz_rt_uid"] == uid]
                    if ms2_df is not None and not ms2_df.empty
                    else pd.DataFrame()
                )
                if uid_ms2.empty:
                    continue
                scan_rts = uid_ms2["scan_rt"].dropna().tolist()
                if scan_rts:
                    fig.add_trace(go.Scatter(
                        x=scan_rts,
                        y=[-y_upper * 0.03] * len(scan_rts),
                        mode="markers",
                        marker=dict(color=color, size=7, symbol="triangle-up"),
                        showlegend=False,
                        hovertemplate=(
                            f"MS2 scan for Run{run_number} ({ce})<br>"
                            f"{adduct}<br>"
                            f"RT=%{{x:.4f}}<br>"
                            f"Int=%{{y:.2e}}"
                            f"<extra></extra>"
                        ),
                        cliponaxis=False,
                    ))

        # CE dash-style legend entries
        for ce, dash in _CE_DASH_MAP.items():
            if ce == "":
                continue
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="#555", width=2, dash=dash),
                name=ce, legendgroup=f"__ce_{ce}__", showlegend=True,
            ))

        pol_display = "Positive" if pol == "POS" else "Negative"
        y_range = [-y_upper * 0.05, y_upper] if y_scale == "linear" else None
        fig.update_layout(
            title=dict(text=f"<b>{pol_display}</b>", x=0.5, font=dict(size=12, color="#333")),
            xaxis_title="RT (min)", yaxis_title="Intensity",
            hovermode="closest",
            legend=dict(
                orientation="h", yanchor="top", y=-0.18,
                xanchor="center", x=0.5, font=dict(size=9), tracegroupgap=0,
            ),
            margin=dict(l=50, r=10, t=35, b=80),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
            yaxis=dict(showgrid=False, zeroline=False, type=y_scale, range=y_range),
        )
        return fig

    def _make_tic_figure_for_chrom(
        compound_name: str, chrom: str, y_scale: str = "linear"
    ) -> go.Figure:
        """Build the static TIC figure for one chromatography panel (all CEs)."""
        fig = go.Figure()
        adducts = compound_chrom_adducts.get((compound_name, chrom), [])
        seen_files: set[str] = set()
        for adduct in adducts:
            uids = uid_map.get((compound_name, chrom, adduct), [])
            sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
            for _, row in sub.iterrows():
                fn = str(row.get("filename", ""))
                if fn in seen_files:
                    continue
                seen_files.add(fn)
                tic = tic_by_file.get(fn)
                if tic is None or tic.empty:
                    continue
                color = _get_file_color(fn)
                fig.add_trace(go.Scattergl(
                    x=tic["rt"].tolist(), y=tic["tic_intensity"].tolist(),
                    mode="lines", line=dict(color=color, width=1.0),
                    showlegend=False,
                    hovertemplate=f"TIC: {_short_filename(fn)}<extra></extra>",
                ))
        rt_lo, rt_hi = _get_rt_bounds_for_chrom(compound_name, chrom)
        fig.update_layout(
            title=dict(text="<b>TIC</b>", x=0.5, font=dict(size=11)),
            xaxis_title="RT (min)", yaxis_title="TIC",
            margin=dict(l=50, r=10, t=35, b=30),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
            yaxis=dict(showgrid=False, zeroline=False, type=y_scale),
            height=140,
        )
        return fig

    def _make_tic_figure_for_chrom_with_ce(
        compound_name: str, chrom: str, y_scale: str = "linear"
    ) -> go.Figure:
        """TIC figure with CE-encoded dash styles (solid/dash per CE label)."""
        fig = go.Figure()
        adducts = compound_chrom_adducts.get((compound_name, chrom), [])
        seen_files: set[str] = set()
        rt_lo, rt_hi = _get_rt_bounds_for_chrom(compound_name, chrom)
        for adduct in adducts:
            uids = uid_map.get((compound_name, chrom, adduct), [])
            sub = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
            for _, row in sub.iterrows():
                fn = str(row.get("filename", ""))
                if fn in seen_files:
                    continue
                seen_files.add(fn)
                tic = tic_by_file.get(fn)
                if tic is None or tic.empty:
                    continue
                uid = str(row.get("mz_rt_uid", ""))
                ce = uid_ce_label.get(uid, "")
                dash_style = _CE_DASH_MAP.get(ce, "solid")
                color = _get_file_color(fn)
                fig.add_trace(go.Scattergl(
                    x=tic["rt"].tolist(), y=tic["tic_intensity"].tolist(),
                    mode="lines",
                    line=dict(color=color, width=1.0, dash=dash_style),
                    showlegend=False,
                    hovertemplate=f"TIC: {_short_filename(fn)}<extra></extra>",
                ))
        fig.update_layout(
            title=dict(text="<b>TIC</b>", x=0.5, font=dict(size=11)),
            xaxis_title="RT (min)", yaxis_title="TIC",
            margin=dict(l=50, r=10, t=35, b=30),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
            yaxis=dict(showgrid=False, zeroline=False, type=y_scale),
            height=140,
        )
        return fig

    def _make_injbl_eic_figure_for_chrom(
        compound_name: str,
        chrom: str,
    ) -> go.Figure:
        """Build the InjBl background EIC figure (summed across adducts, below TIC).

        Shows one line per unique InjBl file.  Intensities are summed across
        all adducts at each RT point (TIC-style).  The line dash style encodes
        the CE of the target file(s) the InjBl covers:
          - If all covered target files share the same CE, use that CE's dash.
          - If covered target files have mixed CEs (shared InjBl), use solid.
        Hovertext lists the InjBl run number and all covered target run numbers.
        """
        adducts = compound_chrom_adducts.get((compound_name, chrom), [])
        fig = go.Figure()
        rt_lo, rt_hi = _get_rt_bounds_for_chrom(compound_name, chrom)

        if injbl_ms1_df.empty:
            fig.update_layout(
                title=dict(text="<b>InjBl (no data)</b>", x=0.5, font=dict(size=11)),
                margin=dict(l=50, r=10, t=35, b=30),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
                yaxis=dict(showgrid=False, zeroline=False),
                height=140,
            )
            return fig

        # Collect all mz_rt_uids for this compound+chrom (all adducts, all CEs)
        all_uids: set[str] = set()
        for adduct in adducts:
            all_uids.update(uid_map.get((compound_name, chrom, adduct), []))

        # Filter InjBl MS1 to rows for this compound+chrom
        sub = injbl_ms1_df[injbl_ms1_df["mz_rt_uid"].isin(all_uids)]
        if sub.empty:
            fig.update_layout(
                title=dict(text="<b>InjBl (no match)</b>", x=0.5, font=dict(size=11)),
                margin=dict(l=50, r=10, t=35, b=30),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
                yaxis=dict(showgrid=False, zeroline=False),
                height=140,
            )
            return fig

        # Build mz_rt_uid -> CE label lookup for this compound+chrom
        uid_to_ce: dict[str, str] = {}
        for adduct in adducts:
            for uid in uid_map.get((compound_name, chrom, adduct), []):
                uid_to_ce[uid] = uid_ce_label.get(uid, "")

        # For each unique InjBl file, sum intensities across all adducts at each RT
        # and determine the CE dash style from the covered target UIDs.
        has_data = False
        seen_injbl: set[str] = set()

        for injbl_fn, fn_group in sub.groupby("injbl_filename"):
            injbl_fn_str = str(injbl_fn)
            if injbl_fn_str in seen_injbl:
                continue
            seen_injbl.add(injbl_fn_str)

            # Collect all (rt, intensity) points across all adducts for this InjBl file
            rt_int_pairs: list[tuple[float, float]] = []
            for _, row in fn_group.iterrows():
                rts = row.get("spec_rts", [])
                ints = row.get("spec_ints", [])
                if rts is None or ints is None or len(rts) == 0:
                    continue
                for rt, intensity in zip(rts, ints):
                    rt_int_pairs.append((round(float(rt), 6), float(intensity)))

            if not rt_int_pairs:
                continue

            # Sum intensities at each RT
            rt_sum: dict[float, float] = {}
            for rt, intensity in rt_int_pairs:
                rt_sum[rt] = rt_sum.get(rt, 0.0) + intensity
            sorted_rts = sorted(rt_sum.keys())
            summed_ints = [rt_sum[rt] for rt in sorted_rts]

            # Determine CE dash style from covered target UIDs
            covered_uids = fn_group["mz_rt_uid"].unique().tolist()
            covered_ces = list(dict.fromkeys(
                uid_to_ce.get(str(u), "") for u in covered_uids
            ))
            real_covered_ces = [ce for ce in covered_ces if ce]
            if len(real_covered_ces) == 1:
                # All covered targets share the same CE
                dash_style = _CE_DASH_MAP.get(real_covered_ces[0], "solid")
                ce_label_str = real_covered_ces[0]
            elif len(real_covered_ces) > 1:
                # Mixed CEs — use solid and note in hover
                dash_style = "solid"
                ce_label_str = "+".join(real_covered_ces)
            else:
                dash_style = "solid"
                ce_label_str = ""

            # Build hovertext: InjBl run + covered target runs
            injbl_run = _extract_run_number(injbl_fn_str)
            covered_target_fns: list[str] = []
            for uid in covered_uids:
                fn = uid_to_filename.get(str(uid), "")
                if fn and fn not in covered_target_fns:
                    covered_target_fns.append(fn)
            target_runs_str = ", ".join(
                f"Run{_extract_run_number(f)}" for f in covered_target_fns
                if _extract_run_number(f)
            )
            ce_hover = f"CE: {ce_label_str}<br>" if ce_label_str else ""
            hover = (
                f"InjBl Run{injbl_run}<br>"
                f"{ce_hover}"
                f"Covers: {target_runs_str}"
                f"<extra></extra>"
            )

            fig.add_trace(go.Scattergl(
                x=sorted_rts, y=summed_ints,
                mode="lines",
                line=dict(color="gray", width=1.2, dash=dash_style),
                name=f"InjBl Run{injbl_run}",
                showlegend=False,
                hovertemplate=hover,
            ))
            has_data = True

        title_text = "<b>InjBl</b>" if has_data else "<b>InjBl (no match)</b>"
        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=11)),
            xaxis_title="RT (min)",
            yaxis_title="Intensity",
            margin=dict(l=50, r=10, t=35, b=30),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, range=[rt_lo, rt_hi]),
            yaxis=dict(showgrid=False, zeroline=False),
            height=140,
        )
        return fig

    def _make_ms2_figure(
        mzs: list[float], ints: list[float], title_text: str
    ) -> go.Figure:
        fig = go.Figure()
        if mzs and ints:
            mz_range = max(mzs) - min(mzs) if len(mzs) > 1 else 1.0
            bar_width = max(0.5, mz_range / max(len(mzs), 1) * 0.6)
            fig.add_trace(go.Bar(
                x=mzs, y=ints,
                width=[bar_width] * len(mzs),
                marker_color="black", showlegend=False,
            ))
        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=9)),
            margin=dict(l=30, r=10, t=55, b=30),
            plot_bgcolor="white", bargap=0,
            xaxis=dict(showgrid=False, zeroline=False, title="m/z", title_font=dict(size=8)),
            yaxis=dict(showgrid=False, zeroline=False, title="Int", title_font=dict(size=8)),
            height=190,
        )
        return fig

    def _make_ms2_plots_for_chrom(compound_name: str, chrom: str) -> list:
        chrom_peaks = _peaks_for_chrom(compound_name, chrom)
        mini_figs = []
        for _, peak_row in chrom_peaks.iterrows():
            peak_uid = str(peak_row["peak_uid"])
            adduct = str(peak_row["adduct"])
            rt_peak = float(peak_row["rt_peak"])
            peak_num = int(peak_row.get("peak_num", 1))
            ce = str(peak_row.get("collision_energy_label", ""))
            ms2 = _get_ms2_for_peak(peak_uid)
            if not ms2:
                continue
            frag_mzs = [float(v) for v in (ms2.get("frag_mzs") or [])]
            frag_ints = [float(v) for v in (ms2.get("frag_ints") or [])]
            scan_rt = float(ms2.get("scan_rt", float("nan")))
            collision_e = float(ms2.get("collision_energy", float("nan")))
            ce_str = f" CE={collision_e:.0f}" if not np.isnan(collision_e) else (f" {ce}" if ce else "")
            scan_rt_str = f" MS2RT={scan_rt:.3f}" if not np.isnan(scan_rt) else ""
            title_text = f"{adduct} peak{peak_num}<br>RT={rt_peak:.3f}{ce_str}{scan_rt_str}"
            mini_figs.append(dcc.Graph(
                figure=_make_ms2_figure(frag_mzs, frag_ints, title_text),
                config={"displayModeBar": False},
            ))
        if not mini_figs:
            return []
        rows = []
        for i in range(0, len(mini_figs), 3):
            chunk = mini_figs[i:i + 3]
            rows.append(dbc.Row([dbc.Col(g, width=4) for g in chunk], className="mb-1"))
        return rows

    def _count_ms2_for_chrom(compound_name: str, chrom: str) -> int:
        chrom_peaks = _peaks_for_chrom(compound_name, chrom)
        return sum(1 for uid in chrom_peaks["peak_uid"].tolist() if uid in peak_ms2_by_uid)

    def _save_compound_selections(
        compound_name: str,
        checked_peak_uids: list[str],
        analyst_notes_by_chrom: dict[str, str],
    ) -> None:
        compound_uids = compound_name_to_uids.get(compound_name, [])
        for cuid in compound_uids:
            try:
                sadb.delete_all_analyst_selections_for_compound(db_path, cuid)
            except Exception as exc:
                logger.error(f"Failed to clear old selections for compound_uid={cuid}: {exc}")

        peaks_df = peaks_holder[0]
        for peak_uid in checked_peak_uids:
            if peaks_df.empty:
                continue
            row_match = peaks_df[peaks_df["peak_uid"] == peak_uid]
            if row_match.empty:
                continue
            row = row_match.iloc[0]
            chrom = str(row.get("chromatography", ""))
            notes = analyst_notes_by_chrom.get(chrom, "")
            try:
                sadb.save_analyst_selection(db_path, peak_uid, analyst_notes=notes)
            except Exception as exc:
                logger.error(f"Failed to save selection for {peak_uid}: {exc}")

        for chrom in compound_chroms.get(compound_name, []):
            for cuid in compound_uids:
                try:
                    sadb.update_workflow_state(db_path, cuid, chrom, "complete")
                except Exception as exc:
                    logger.error(
                        f"Failed to update workflow state for "
                        f"compound_uid={cuid}, chrom={chrom}: {exc}"
                    )

    def _load_session_for_compound(compound_name: str) -> dict:
        checked_peaks: dict[str, bool] = {}
        notes_by_chrom: dict[str, str] = {}
        try:
            sel_df = sadb.load_analyst_selections(db_path)
        except Exception:
            return {"checked_peaks": checked_peaks, "analyst_notes_by_chrom": notes_by_chrom}
        if sel_df.empty:
            return {"checked_peaks": checked_peaks, "analyst_notes_by_chrom": notes_by_chrom}
        cmp_sel = sel_df[sel_df["compound_name"] == compound_name]
        if cmp_sel.empty:
            return {"checked_peaks": checked_peaks, "analyst_notes_by_chrom": notes_by_chrom}
        for _, row in cmp_sel.iterrows():
            peak_uid = str(row["peak_uid"])
            chrom = str(row.get("chromatography", ""))
            note = str(row.get("analyst_notes", "") or "")
            checked_peaks[peak_uid] = True
            if chrom and note:
                notes_by_chrom[chrom] = note
        return {"checked_peaks": checked_peaks, "analyst_notes_by_chrom": notes_by_chrom}

    # ── Find free port ────────────────────────────────────────────────────────
    def _find_free_port(start: int, max_tries: int = 20) -> int:
        for p in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        raise RuntimeError(f"No free port found in range {start}-{start + max_tries - 1}.")

    port = _find_free_port(port)

    # ── Dash app setup ────────────────────────────────────────────────────────
    try:
        if (
            os.getenv("METATLAS2_STANDALONE") == "true"
            or not os.getenv("JUPYTERHUB_SERVICE_PREFIX")
        ):
            requests_prefix = "/"
        else:
            requests_prefix = (
                f"{os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')}proxy/{port}/"
            )
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            requests_pathname_prefix=requests_prefix,
            suppress_callback_exceptions=True,
        )
        run_name = cfg["project"]["standards_run_name"]
        app.title = f"Standards Annotation | {run_name}"
        app.config.prevent_initial_callbacks = "initial_duplicate"
    except Exception as exc:
        logger.error(f"Failed to create Dash app: {exc}")
        raise

    compound_options = [
        {"label": f"{i+1}: {name}", "value": i}
        for i, name in enumerate(compound_names)
    ]

    MAX_CHROMS = 4

    def _chrom_panel(slot: int) -> html.Div:
        """Build one chromatography data column.

        EIC row: up to MAX_CE=2 side-by-side graphs (sa-eic-graph-{slot}-0,
        sa-eic-graph-{slot}-1).  TIC and InjBl are full-width rows below.
        """
        eic_cols = []
        for ce_idx in range(MAX_CE):
            eic_cols.append(
                dbc.Col(
                    dcc.Graph(
                        id=f"sa-eic-graph-{slot}-{ce_idx}",
                        config={"displayModeBar": True},
                        style={"height": "360px"},
                    ),
                    id=f"sa-eic-col-{slot}-{ce_idx}",
                    width=6,
                    className="pe-1" if ce_idx == 0 else "ps-1",
                )
            )

        return html.Div(
            id=f"sa-chrom-panel-{slot}",
            style={
                "display": "none",
                "borderLeft": "1px solid #ccc" if slot > 0 else "none",
                "paddingLeft": "8px" if slot > 0 else "0",
            },
            children=[
                html.H4(id=f"sa-chrom-title-{slot}", className="text-center fw-bold mb-1"),
                html.Div(
                    id=f"sa-chrom-notes-{slot}",
                    className="text-muted fst-italic text-center mb-1",
                    style={"fontSize": "0.75rem", "whiteSpace": "pre-wrap"},
                ),
                html.Div(
                    id=f"sa-checklist-container-{slot}",
                    style={"minHeight": "60px"},
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    "Positive",
                                    className="fw-bold text-center",
                                    style={"fontSize": "0.75rem", "borderBottom": "1px solid #ddd", "marginBottom": "3px"},
                                ),
                                dcc.Checklist(
                                    id=f"sa-peak-checklist-pos-{slot}",
                                    options=[], value=[],
                                    labelStyle={"display": "block", "marginBottom": "2px", "fontSize": "0.8rem"},
                                    inputStyle={"marginRight": "4px"},
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Div(
                                    "Negative",
                                    className="fw-bold text-center",
                                    style={"fontSize": "0.75rem", "borderBottom": "1px solid #ddd", "marginBottom": "3px"},
                                ),
                                dcc.Checklist(
                                    id=f"sa-peak-checklist-neg-{slot}",
                                    options=[], value=[],
                                    labelStyle={"display": "block", "marginBottom": "2px", "fontSize": "0.8rem"},
                                    inputStyle={"marginRight": "4px"},
                                ),
                            ], width=6),
                        ], className="g-0"),
                        dcc.Checklist(
                            id=f"sa-peak-checklist-{slot}",
                            options=[], value=[],
                            style={"display": "none"},
                        ),
                    ],
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"sa-manual-adduct-{slot}", options=[],
                            placeholder="Adduct", clearable=False,
                            style={"fontSize": "0.8rem"},
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"sa-manual-ce-{slot}", options=[],
                            placeholder="CE (file)", clearable=True,
                            style={"fontSize": "0.8rem"},
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"sa-manual-rt-{slot}", options=[],
                            placeholder="RT peak", searchable=True, clearable=True,
                            style={"fontSize": "0.8rem"},
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button("Add", id=f"sa-add-peak-btn-{slot}", color="secondary", size="sm"),
                        width=1,
                    ),
                    dbc.Col(
                        dbc.Input(
                            id=f"sa-analyst-notes-{slot}", placeholder="Notes...",
                            debounce=True, size="sm", style={"fontSize": "0.8rem"},
                        ),
                        width=3,
                    ),
                ], className="mb-1 g-1"),
                # EIC row: up to MAX_CE side-by-side graphs
                dbc.Row(eic_cols, className="g-0 mb-0"),
                # TIC (full width)
                dcc.Graph(
                    id=f"sa-tic-graph-{slot}",
                    config={"displayModeBar": False},
                    style={"height": "120px"},
                ),
                # InjBl (full width, below TIC)
                dcc.Graph(
                    id=f"sa-injbl-graph-{slot}",
                    config={"displayModeBar": False},
                    style={"height": "120px"},
                ),
                # Controls row: Lin/Log scale + MS2 toggle — at the very bottom
                dbc.Row([
                    dbc.Col(
                        dcc.RadioItems(
                            id=f"sa-scale-{slot}",
                            options=[
                                {"label": "Lin", "value": "linear"},
                                {"label": "Log", "value": "log"},
                            ],
                            value="linear", inline=True,
                            labelStyle={"fontSize": "0.75rem", "marginRight": "6px"},
                            inputStyle={"marginRight": "2px"},
                        ),
                        width="auto", className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Show MS2 Scans (0)", id=f"sa-ms2-toggle-{slot}",
                            color="light", size="sm", n_clicks=0, disabled=True,
                        ),
                        width="auto", className="d-flex align-items-center ms-auto",
                    ),
                ], className="mt-1 mb-1 align-items-center"),
                html.Div(
                    id=f"sa-ms2-plots-{slot}",
                    style={"display": "none"},
                    className="mt-1",
                ),
            ],
        )

    # ── Pre-load starting compound session ────────────────────────────────────
    _starting_name = _get_compound_name(starting_idx)
    _starting_restored = _load_session_for_compound(_starting_name)

    app.layout = dbc.Container(
        [
            dcc.Store(id="sa-session-store", storage_type="memory", data={
                "compound_idx": starting_idx,
                "checked_peaks": _starting_restored["checked_peaks"],
                "analyst_notes_by_chrom": _starting_restored["analyst_notes_by_chrom"],
            }),
            dcc.Store(id="sa-render-store", storage_type="memory", data={
                "compound_idx": starting_idx,
                "analyst_notes_by_chrom": _starting_restored["analyst_notes_by_chrom"],
            }),
            dcc.Store(id="sa-exit-flag", storage_type="memory", data=False),
            html.Div(
                id="sa-keyboard-listener", tabIndex=0,
                style={"outline": "none", "position": "fixed", "top": 0, "left": 0,
                       "width": "100%", "height": "100%", "zIndex": -1},
            ),

            dbc.Row([
                dbc.Col([
                    html.Div(id="sa-compound-header", className="fw-bold d-inline me-2",
                             style={"fontSize": "1.25rem"}),
                ], width="auto", className="d-flex align-items-center"),
                dbc.Col([
                    dcc.Dropdown(
                        id="sa-compound-dd", options=compound_options,
                        value=starting_idx, clearable=False,
                        style={"fontSize": "0.95rem", "minWidth": "260px"},
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button("< Previous", id="sa-prev-btn", color="primary", size="md", className="me-1"),
                    html.Span(id="sa-progress-div", className="small me-1 ms-1"),
                    dbc.Button("Next >", id="sa-next-btn", color="primary", size="md"),
                ], width="auto", className="d-flex align-items-center"),
                dbc.Col([
                    html.Div(id="sa-status-div", className="text-muted small"),
                ], width="auto", className="d-flex align-items-center"),
                dbc.Col([
                    dbc.Button("Save & Exit", id="sa-exit-btn", color="danger", size="sm"),
                ], width="auto", className="d-flex align-items-center ms-auto"),
            ], className="mt-1 mb-2 g-2 align-items-center border-bottom pb-2"),

            dbc.Row(
                id="sa-chrom-row",
                children=[
                    dbc.Col(_chrom_panel(slot), id=f"sa-chrom-col-{slot}", width=12)
                    for slot in range(MAX_CHROMS)
                ],
            ),
        ],
        fluid=True,
        style={"paddingTop": "0.5rem"},
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    # Callback 1: update_plots — triggered by sa-render-store
    _plot_outputs = [
        Output("sa-compound-header", "children"),
        Output("sa-progress-div", "children"),
        Output("sa-compound-dd", "value", allow_duplicate=True),
    ]
    for slot in range(MAX_CHROMS):
        _plot_outputs += [
            Output(f"sa-chrom-col-{slot}", "width"),
            Output(f"sa-chrom-panel-{slot}", "style"),
            Output(f"sa-chrom-title-{slot}", "children"),
            Output(f"sa-chrom-notes-{slot}", "children"),
            Output(f"sa-checklist-container-{slot}", "style"),
            Output(f"sa-manual-adduct-{slot}", "options"),
            Output(f"sa-manual-rt-{slot}", "options"),
            Output(f"sa-manual-rt-{slot}", "value"),
            Output(f"sa-analyst-notes-{slot}", "value"),
        ]
        for ce_idx in range(MAX_CE):
            _plot_outputs += [
                Output(f"sa-eic-graph-{slot}-{ce_idx}", "figure"),
                Output(f"sa-eic-col-{slot}-{ce_idx}", "width"),
                Output(f"sa-eic-col-{slot}-{ce_idx}", "style"),
            ]
        _plot_outputs += [
            Output(f"sa-tic-graph-{slot}", "figure"),
            Output(f"sa-injbl-graph-{slot}", "figure"),
            Output(f"sa-ms2-plots-{slot}", "children"),
            Output(f"sa-ms2-toggle-{slot}", "children"),
            Output(f"sa-ms2-toggle-{slot}", "disabled"),
        ]

    @app.callback(_plot_outputs, Input("sa-render-store", "data"))
    def update_plots(render_data):
        if not render_data:
            raise dash.exceptions.PreventUpdate

        idx = int(render_data.get("compound_idx", 0))
        name = _get_compound_name(idx)
        notes_by_chrom = render_data.get("analyst_notes_by_chrom", {})

        header = f"[{idx+1}/{len(compound_names)}] {name}"
        progress = f"{idx+1} / {len(compound_names)}"

        chroms = compound_chroms.get(name, [])
        n_chroms = max(1, len(chroms))
        col_width = max(3, 12 // n_chroms)

        max_items = 0
        for slot in range(MAX_CHROMS):
            if slot < len(chroms):
                _, _, n_items = _build_checklist_options_for_chrom(name, chroms[slot])
                max_items = max(max_items, n_items)
        checklist_height_px = max(60, max_items * 18 + 40)

        result = [header, progress, idx]

        for slot in range(MAX_CHROMS):
            if slot < len(chroms):
                chrom = chroms[slot]
                panel_style = {
                    "display": "block",
                    "borderLeft": "1px solid #ccc" if slot > 0 else "none",
                    "paddingLeft": "8px" if slot > 0 else "0",
                }
                title = chrom
                csv_chrom_note = compound_chrom_notes.get((name, chrom), "")
                adduct_options = [
                    {"label": a, "value": a}
                    for a in compound_chrom_adducts.get((name, chrom), [])
                ]
                first_adduct = compound_chrom_adducts.get((name, chrom), [None])[0]
                rt_options = []
                if first_adduct:
                    rt_vals = union_rts_map.get((name, chrom, first_adduct), [])
                    rt_options = [{"label": f"{r}", "value": r} for r in rt_vals]
                chrom_notes = notes_by_chrom.get(chrom, "")

                # Polarity EIC graphs: slot 0 = POS, slot 1 = NEG
                # Determine which polarities are present for this compound+chrom
                chrom_pols: list[str] = []
                for p in ["POS", "NEG"]:
                    has_pol = any(
                        uid_polarity.get(u, "").upper() == p
                        for adduct in compound_chrom_adducts.get((name, chrom), [])
                        for u in uid_map.get((name, chrom, adduct), [])
                    )
                    if has_pol:
                        chrom_pols.append(p)
                n_pols = min(len(chrom_pols), MAX_CE)
                eic_col_width = 12 // n_pols if n_pols > 0 else 12

                tic_fig = _make_tic_figure_for_chrom_with_ce(name, chrom)
                injbl_fig = _make_injbl_eic_figure_for_chrom(name, chrom)
                ms2_plots = _make_ms2_plots_for_chrom(name, chrom)
                ms2_count = _count_ms2_for_chrom(name, chrom)
                ms2_btn_text = f"Show MS2 Scans ({ms2_count})"
                ms2_btn_disabled = ms2_count == 0
            else:
                panel_style = {"display": "none"}
                title = ""
                csv_chrom_note = ""
                adduct_options = []
                rt_options = []
                chrom_notes = ""
                chrom_pols = []
                n_pols = 0
                eic_col_width = 6
                tic_fig = go.Figure()
                chrom_pols = []
                injbl_fig = go.Figure()
                ms2_plots = []
                ms2_btn_text = "Show MS2 Scans (0)"
                ms2_btn_disabled = True

            checklist_style = {"minHeight": f"{checklist_height_px}px"}
            result += [
                col_width, panel_style, title, csv_chrom_note, checklist_style,
                adduct_options, rt_options, None, chrom_notes,
            ]

            for pol_idx in range(MAX_CE):
                if slot < len(chroms) and pol_idx < n_pols:
                    pol = chrom_pols[pol_idx]
                    eic_fig = _make_eic_figure_for_polarity(name, chrom, pol)
                    col_w = eic_col_width
                    col_style = {}
                else:
                    eic_fig = go.Figure()
                    col_w = 0
                    col_style = {"display": "none"}
                result += [eic_fig, col_w, col_style]

            result += [tic_fig, injbl_fig, ms2_plots, ms2_btn_text, ms2_btn_disabled]

        return result

    # Callback 2: update_checklists — triggered by sa-session-store
    _checklist_outputs = []
    for slot in range(MAX_CHROMS):
        _checklist_outputs += [
            Output(f"sa-peak-checklist-{slot}", "options"),
            Output(f"sa-peak-checklist-{slot}", "value"),
            Output(f"sa-peak-checklist-pos-{slot}", "options"),
            Output(f"sa-peak-checklist-pos-{slot}", "value"),
            Output(f"sa-peak-checklist-neg-{slot}", "options"),
            Output(f"sa-peak-checklist-neg-{slot}", "value"),
        ]

    @app.callback(_checklist_outputs, Input("sa-session-store", "data"))
    def update_checklists(session):
        if not session:
            raise dash.exceptions.PreventUpdate

        idx = int(session.get("compound_idx", 0))
        name = _get_compound_name(idx)
        checked_keys_dict = session.get("checked_peaks")
        chroms = compound_chroms.get(name, [])

        result = []
        for slot in range(MAX_CHROMS):
            if slot < len(chroms):
                chrom = chroms[slot]
                options, default_checked, _ = _build_checklist_options_for_chrom(name, chrom)
                if checked_keys_dict is None:
                    checked = list(default_checked)
                else:
                    checked = [
                        o["value"] for o in options
                        if not o.get("disabled")
                        and checked_keys_dict.get(o["value"], False)
                    ]
                pos_opts, neg_opts = _split_options_by_polarity(options)
                pos_checked = [v for v in checked if v in {o["value"] for o in pos_opts}]
                neg_checked = [v for v in checked if v in {o["value"] for o in neg_opts}]
            else:
                options, checked = [], []
                pos_opts, neg_opts = [], []
                pos_checked, neg_checked = [], []

            result += [options, checked, pos_opts, pos_checked, neg_opts, neg_checked]

        return result

    # Per-slot CE dropdown update when adduct changes
    for _slot in range(MAX_CHROMS):
        def _make_ce_options_cb(slot):
            @app.callback(
                Output(f"sa-manual-ce-{slot}", "options", allow_duplicate=True),
                Output(f"sa-manual-ce-{slot}", "value", allow_duplicate=True),
                Input(f"sa-manual-adduct-{slot}", "value"),
                State("sa-session-store", "data"),
                prevent_initial_call=True,
            )
            def update_ce_options(adduct, session):
                if not adduct or session is None:
                    raise dash.exceptions.PreventUpdate
                idx = int(session.get("compound_idx", 0))
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                opts = ce_file_options_map.get((name, chrom, adduct), [])
                # Auto-select if only one option
                auto_val = opts[0]["value"] if len(opts) == 1 else None
                return opts, auto_val
            return update_ce_options
        _make_ce_options_cb(_slot)

    # Per-slot RT dropdown update when adduct OR CE changes
    for _slot in range(MAX_CHROMS):
        def _make_rt_options_cb(slot):
            @app.callback(
                Output(f"sa-manual-rt-{slot}", "options", allow_duplicate=True),
                Output(f"sa-manual-rt-{slot}", "value", allow_duplicate=True),
                Input(f"sa-manual-adduct-{slot}", "value"),
                Input(f"sa-manual-ce-{slot}", "value"),
                State("sa-session-store", "data"),
                prevent_initial_call=True,
            )
            def update_rt_options(adduct, ce_uid, session):
                if not adduct or session is None:
                    raise dash.exceptions.PreventUpdate
                idx = int(session.get("compound_idx", 0))
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                if ce_uid:
                    # RT options for the specific CE file (mz_rt_uid)
                    rt_vals = union_rts_map_by_ce.get((name, chrom, adduct, ce_uid), [])
                else:
                    # Fall back to union across all files for this adduct
                    rt_vals = union_rts_map.get((name, chrom, adduct), [])
                return [{"label": f"{r}", "value": r} for r in rt_vals], None
            return update_rt_options
        _make_rt_options_cb(_slot)

    # Per-slot MS2 toggle
    for _slot in range(MAX_CHROMS):
        def _make_ms2_toggle_cb(slot):
            @app.callback(
                Output(f"sa-ms2-plots-{slot}", "style", allow_duplicate=True),
                Output(f"sa-ms2-toggle-{slot}", "children", allow_duplicate=True),
                Input(f"sa-ms2-toggle-{slot}", "n_clicks"),
                State(f"sa-ms2-toggle-{slot}", "children"),
                prevent_initial_call=True,
            )
            def toggle_ms2(n_clicks, current_label):
                import re as _re
                m = _re.search(r"\((\d+)\)", current_label or "")
                count_str = f"({m.group(1)})" if m else "(0)"
                if n_clicks and n_clicks % 2 == 1:
                    return {"display": "block"}, f"Hide MS2 Scans {count_str}"
                return {"display": "none"}, f"Show MS2 Scans {count_str}"
            return toggle_ms2
        _make_ms2_toggle_cb(_slot)

    # Per-slot POS checklist callbacks
    for _slot in range(MAX_CHROMS):
        def _make_pos_checklist_cb(slot):
            @app.callback(
                Output("sa-session-store", "data", allow_duplicate=True),
                Input(f"sa-peak-checklist-pos-{slot}", "value"),
                State("sa-session-store", "data"),
                State("sa-compound-dd", "value"),
                prevent_initial_call=True,
            )
            def update_pos_checked(checked_values, session, compound_idx):
                if session is None:
                    raise dash.exceptions.PreventUpdate
                session = dict(session)
                idx = int(compound_idx or 0)
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                options, _, _ = _build_checklist_options_for_chrom(name, chrom)
                pos_opts, _ = _split_options_by_polarity(options)
                checked_dict = dict(session.get("checked_peaks") or {})
                for o in pos_opts:
                    checked_dict[o["value"]] = o["value"] in (checked_values or [])
                session["checked_peaks"] = checked_dict
                return session
            return update_pos_checked
        _make_pos_checklist_cb(_slot)

    # Per-slot NEG checklist callbacks
    for _slot in range(MAX_CHROMS):
        def _make_neg_checklist_cb(slot):
            @app.callback(
                Output("sa-session-store", "data", allow_duplicate=True),
                Input(f"sa-peak-checklist-neg-{slot}", "value"),
                State("sa-session-store", "data"),
                State("sa-compound-dd", "value"),
                prevent_initial_call=True,
            )
            def update_neg_checked(checked_values, session, compound_idx):
                if session is None:
                    raise dash.exceptions.PreventUpdate
                session = dict(session)
                idx = int(compound_idx or 0)
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                options, _, _ = _build_checklist_options_for_chrom(name, chrom)
                _, neg_opts = _split_options_by_polarity(options)
                checked_dict = dict(session.get("checked_peaks") or {})
                for o in neg_opts:
                    checked_dict[o["value"]] = o["value"] in (checked_values or [])
                session["checked_peaks"] = checked_dict
                return session
            return update_neg_checked
        _make_neg_checklist_cb(_slot)

    # Hidden combined checklist (no-op)
    for _slot in range(MAX_CHROMS):
        def _make_checklist_cb(slot):
            @app.callback(
                Output("sa-session-store", "data", allow_duplicate=True),
                Input(f"sa-peak-checklist-{slot}", "value"),
                State("sa-session-store", "data"),
                State("sa-compound-dd", "value"),
                prevent_initial_call=True,
            )
            def update_checked(checked_values, session, compound_idx):
                raise dash.exceptions.PreventUpdate
            return update_checked
        _make_checklist_cb(_slot)

    # Per-slot analyst notes
    for _slot in range(MAX_CHROMS):
        def _make_notes_cb(slot):
            @app.callback(
                Output("sa-session-store", "data", allow_duplicate=True),
                Input(f"sa-analyst-notes-{slot}", "value"),
                State("sa-session-store", "data"),
                State("sa-compound-dd", "value"),
                prevent_initial_call=True,
            )
            def update_chrom_notes(val, session, compound_idx):
                if session is None:
                    raise dash.exceptions.PreventUpdate
                session = dict(session)
                idx = int(compound_idx or 0)
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                notes_by_chrom = dict(session.get("analyst_notes_by_chrom", {}))
                notes_by_chrom[chrom] = val or ""
                session["analyst_notes_by_chrom"] = notes_by_chrom
                return session
            return update_chrom_notes
        _make_notes_cb(_slot)

    # Per-slot scale toggle — updates both polarity EIC graphs + TIC
    _POL_ORDER = ["POS", "NEG"]
    for _slot in range(MAX_CHROMS):
        def _make_scale_cb(slot):
            scale_outputs = []
            for pol_idx in range(MAX_CE):
                scale_outputs.append(
                    Output(f"sa-eic-graph-{slot}-{pol_idx}", "figure", allow_duplicate=True)
                )
            scale_outputs.append(
                Output(f"sa-tic-graph-{slot}", "figure", allow_duplicate=True)
            )

            @app.callback(
                scale_outputs,
                Input(f"sa-scale-{slot}", "value"),
                State("sa-session-store", "data"),
                prevent_initial_call=True,
            )
            def update_scale(scale, session):
                if session is None:
                    raise dash.exceptions.PreventUpdate
                idx = int(session.get("compound_idx", 0))
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                result = []
                for pol_idx in range(MAX_CE):
                    pol = _POL_ORDER[pol_idx]
                    result.append(
                        _make_eic_figure_for_polarity(name, chrom, pol, y_scale=scale or "linear")
                    )
                result.append(_make_tic_figure_for_chrom(name, chrom, y_scale=scale or "linear"))
                return result
            return update_scale
        _make_scale_cb(_slot)

    # Keyboard hotkeys
    app.clientside_callback(
        """
        function(n) {
            if (!window._sa_kb_registered) {
                window._sa_kb_registered = true;
                document.addEventListener('keydown', function(e) {
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                    if (e.key === 'ArrowLeft') {
                        var btn = document.getElementById('sa-prev-btn');
                        if (btn) btn.click();
                    } else if (e.key === 'ArrowRight') {
                        var btn = document.getElementById('sa-next-btn');
                        if (btn) btn.click();
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("sa-keyboard-listener", "children"),
        Input("sa-session-store", "data"),
    )

    # Per-slot add-peak callbacks
    for _slot in range(MAX_CHROMS):
        def _make_add_peak_cb(slot):
            @app.callback(
                Output("sa-session-store", "data", allow_duplicate=True),
                Output("sa-render-store", "data", allow_duplicate=True),
                Output(f"sa-manual-rt-{slot}", "value", allow_duplicate=True),
                Input(f"sa-add-peak-btn-{slot}", "n_clicks"),
                State(f"sa-manual-rt-{slot}", "value"),
                State(f"sa-manual-adduct-{slot}", "value"),
                State(f"sa-manual-ce-{slot}", "value"),
                State("sa-session-store", "data"),
                State("sa-compound-dd", "value"),
                prevent_initial_call=True,
            )
            def add_manual_peak(n_clicks, rt_selected, adduct, ce_uid, session, compound_idx):
                if not n_clicks or rt_selected is None or not adduct:
                    raise dash.exceptions.PreventUpdate
                session = dict(session)
                idx = int(compound_idx or 0)
                name = _get_compound_name(idx)
                chroms = compound_chroms.get(name, [])
                if slot >= len(chroms):
                    raise dash.exceptions.PreventUpdate
                chrom = chroms[slot]
                manual_rt_peak = float(rt_selected)

                if isinstance(ce_uid, list) or ce_uid is None or not isinstance(ce_uid, str):
                    raise ValueError("Only one CE file can be selected for a manual peak.")
                
                closest_intensity = 0.0
                mz_observed = float("nan")
                source_file = ""
                compound_uid_for_peak = ""

                sub = ms1_df[ms1_df["mz_rt_uid"] == ce_uid]
                if sub.empty:
                    raise ValueError(f"No MS1 data found for mz_rt_uid '{ce_uid}' in the database.")
                row = sub.iloc[0]  # mz_rt_uid maps to a single MS1 row

                def _cell_array(col: str) -> np.ndarray:
                    if col not in sub.columns:
                        return np.asarray([], dtype=np.float64)
                    v = row[col]
                    if v is None or np.ndim(v) == 0:
                        return np.asarray([], dtype=np.float64)
                    return np.asarray(v, dtype=np.float64)

                rts = _cell_array("spec_rts")
                ints = _cell_array("spec_ints")
                mzs = _cell_array("spec_mzs")
                if len(rts) == 0:
                    raise ValueError(f"No MS1 data found for mz_rt_uid '{ce_uid}' in the database.")

                matches = np.flatnonzero(np.isclose(rts, manual_rt_peak, rtol=0.0, atol=1e-9))
                if matches.size == 0:
                    raise ValueError(
                        f"Selected RT {manual_rt_peak} not present in spec_rts for mz_rt_uid '{ce_uid}'."
                    )
                i = int(matches[0])
                closest_intensity = float(ints[i]) if len(ints) > i else 0.0
                source_file = str(row.get("filename", ""))

                mask = (rts >= manual_rt_peak - 0.1) & (rts <= manual_rt_peak + 0.1)
                if np.any(mask) and mzs.size > 0:
                    int_in = ints[mask]
                    mz_in = mzs[mask]
                    total_int = float(np.sum(int_in))
                    mz_observed = (
                        float(np.sum(mz_in * int_in) / total_int)
                        if total_int > 0 else float(np.mean(mz_in))
                    )

                et_match = expanded_table[expanded_table["mz_rt_uid"] == ce_uid]
                if not et_match.empty:
                    compound_uid_for_peak = str(et_match.iloc[0]["compound_uid"])

                existing_peaks = _peaks_for_chrom(name, chrom)
                if not existing_peaks.empty:
                    existing_for_ce = existing_peaks[
                        (existing_peaks["adduct"] == adduct) &
                        (existing_peaks["mz_rt_uid"] == ce_uid)
                    ]
                    peak_num = len(existing_for_ce) + 1
                else:
                    peak_num = 1

                polarity = uid_polarity.get(ce_uid, "POS")
                mz_theoretical = uid_mz_theoretical.get(ce_uid, float("nan"))
                ce_label = uid_ce_label.get(ce_uid, "")

                new_peak_uid = str(uuid.uuid4()).replace("-", "")
                new_peak = {
                    "peak_uid":               new_peak_uid,
                    "compound_uid":           compound_uid_for_peak,
                    "compound_name":          name,
                    "inchi_key":              compound_meta.get(name, {}).get("inchi_key", ""),
                    "adduct":                 adduct,
                    "polarity":               polarity,
                    "chromatography":         chrom,
                    "filename":               source_file,
                    "mz_rt_uid":              ce_uid,
                    "mz_theoretical":         mz_theoretical,
                    "rt_peak":                manual_rt_peak,
                    "rt_min":                 max(0.0, manual_rt_peak - 0.1),
                    "rt_max":                 manual_rt_peak + 0.1,
                    "peak_intensity":         closest_intensity,
                    "mz_observed":            mz_observed,
                    "peak_num":               peak_num,
                    "is_manual":              True,
                    "collision_energy_label": ce_label,
                }
                try:
                    sadb.save_manual_peak_to_db(db_path, new_peak)
                except Exception as exc:
                    logger.error(f"Failed to save manual peak: {exc}")

                if ms2_df is not None and not ms2_df.empty and source_file:
                    try:
                        cand = ms2_df[ms2_df["mz_rt_uid"] == ce_uid]
                        if "filename" in cand.columns and not cand.empty:
                            exact = cand[cand["filename"] == source_file]
                            if exact.empty:
                                base = os.path.basename(source_file)
                                exact = cand[cand["filename"].astype(str).map(
                                    lambda f: os.path.basename(str(f)) == base
                                )]
                            if not exact.empty:
                                cand = exact
                            else:
                                logger.warning(
                                    f"MS2: no scans match filename '{source_file}' for "
                                    f"uid={ce_uid}; using uid-only ({len(cand)} scans)."
                                )
                        scan = sat.pick_best_ms2_scan(cand, manual_rt_peak)
                        if scan is None:
                            logger.warning(f"MS2: no scans for uid={ce_uid}.")
                        else:
                            frag_mzs = _to_float_list(scan.get("frag_mzs"))
                            frag_ints = _to_float_list(scan.get("frag_ints"))
                            precursor_col = (
                                "precursor_MZ" if "precursor_MZ" in cand.columns
                                else "precursor_mz"
                            )
                            if not frag_mzs or not frag_ints:
                                logger.warning(
                                    f"MS2: closest scan (RT={float(scan['scan_rt']):.4f}) "
                                    "has no fragment data — no association saved."
                                )
                            else:
                                ms2_assoc = {
                                    "peak_uid":            new_peak_uid,
                                    "scan_rt":             float(scan.get("scan_rt", float("nan"))),
                                    "precursor_mz":        float(scan.get(precursor_col, float("nan"))),
                                    "precursor_intensity": float(scan.get("precursor_intensity", float("nan"))),
                                    "collision_energy":    float(scan.get("collision_energy", float("nan"))),
                                    "frag_mzs":            frag_mzs,
                                    "frag_ints":           frag_ints,
                                    "filename":            source_file,
                                }
                                sadb.save_peak_ms2_to_db(db_path, [ms2_assoc])
                                peak_ms2_by_uid[new_peak_uid] = {"peak_uid": new_peak_uid, **ms2_assoc}
                    except Exception as exc:
                        logger.error(f"MS2 assignment failed for manual peak: {exc}")

                try:
                    peaks_holder[0] = sadb.load_peaks(db_path)
                except Exception as exc:
                    logger.error(f"Failed to reload peaks after manual add: {exc}")

                checked_dict = dict(session.get("checked_peaks") or {})
                checked_dict[new_peak_uid] = True
                session["checked_peaks"] = checked_dict
                session["render_seq"] = int(session.get("render_seq") or 0) + 1

                render_data = {
                    "compound_idx": idx,
                    "analyst_notes_by_chrom": session.get("analyst_notes_by_chrom", {}),
                    "render_seq": session["render_seq"],
                }
                return session, render_data, None

            return add_manual_peak
        _make_add_peak_cb(_slot)

    def _extract_session_for_save(session: dict) -> tuple:
        idx = int(session.get("compound_idx", 0))
        name = _get_compound_name(idx)
        checked_keys_dict = session.get("checked_peaks") or {}
        checked_list = [k for k, v in checked_keys_dict.items() if v]
        notes_by_chrom = session.get("analyst_notes_by_chrom", {})
        return name, checked_list, notes_by_chrom

    def _navigate(session: dict, delta: int) -> tuple[dict, dict]:
        name, checked_list, notes_by_chrom = _extract_session_for_save(session)
        _save_compound_selections(name, checked_list, notes_by_chrom)
        idx = int(session.get("compound_idx", 0))
        new_idx = max(0, min(idx + delta, len(compound_names) - 1))
        new_name = _get_compound_name(new_idx)
        restored = _load_session_for_compound(new_name)
        new_session = {
            "compound_idx": new_idx,
            "checked_peaks": restored["checked_peaks"],
            "analyst_notes_by_chrom": restored["analyst_notes_by_chrom"],
        }
        new_render = {
            "compound_idx": new_idx,
            "analyst_notes_by_chrom": restored["analyst_notes_by_chrom"],
        }
        return new_session, new_render

    @app.callback(
        Output("sa-session-store", "data", allow_duplicate=True),
        Output("sa-render-store", "data", allow_duplicate=True),
        Input("sa-prev-btn", "n_clicks"),
        State("sa-session-store", "data"),
        prevent_initial_call=True,
    )
    def go_prev(n_clicks, session):
        if not n_clicks or session is None:
            raise dash.exceptions.PreventUpdate
        return _navigate(session, -1)

    @app.callback(
        Output("sa-session-store", "data", allow_duplicate=True),
        Output("sa-render-store", "data", allow_duplicate=True),
        Input("sa-next-btn", "n_clicks"),
        State("sa-session-store", "data"),
        prevent_initial_call=True,
    )
    def go_next(n_clicks, session):
        if not n_clicks or session is None:
            raise dash.exceptions.PreventUpdate
        return _navigate(session, +1)

    @app.callback(
        Output("sa-session-store", "data", allow_duplicate=True),
        Output("sa-render-store", "data", allow_duplicate=True),
        Output("sa-compound-dd", "value", allow_duplicate=True),
        Input("sa-compound-dd", "value"),
        State("sa-session-store", "data"),
        prevent_initial_call=True,
    )
    def jump_to_compound(dd_value, session):
        if dd_value is None or session is None:
            raise dash.exceptions.PreventUpdate
        new_idx = int(dd_value)
        if new_idx == int(session.get("compound_idx", 0)):
            raise dash.exceptions.PreventUpdate
        name, checked_list, notes_by_chrom = _extract_session_for_save(session)
        _save_compound_selections(name, checked_list, notes_by_chrom)
        new_name = _get_compound_name(new_idx)
        restored = _load_session_for_compound(new_name)
        new_session = {
            "compound_idx": new_idx,
            "checked_peaks": restored["checked_peaks"],
            "analyst_notes_by_chrom": restored["analyst_notes_by_chrom"],
        }
        new_render = {
            "compound_idx": new_idx,
            "analyst_notes_by_chrom": restored["analyst_notes_by_chrom"],
        }
        return new_session, new_render, new_idx

    @app.callback(
        Output("sa-exit-flag", "data"),
        Output("sa-status-div", "children"),
        Input("sa-exit-btn", "n_clicks"),
        State("sa-session-store", "data"),
        prevent_initial_call=True,
    )
    def save_and_exit(n_clicks, session):
        if not n_clicks or session is None:
            raise dash.exceptions.PreventUpdate
        name, checked_list, notes_by_chrom = _extract_session_for_save(session)
        _save_compound_selections(name, checked_list, notes_by_chrom)
        logger.info("Analyst clicked Save & Exit -- shutting down GUI server.")
        return True, "Saved. You may close this tab."

    shutdown_holder = [None]

    @app.callback(
        Output("sa-status-div", "children", allow_duplicate=True),
        Input("sa-exit-flag", "data"),
        prevent_initial_call=True,
    )
    def trigger_shutdown(flag):
        if flag:
            def _shutdown():
                time.sleep(1.5)
                if shutdown_holder[0]:
                    shutdown_holder[0]()
            threading.Thread(target=_shutdown, daemon=True).start()
        raise dash.exceptions.PreventUpdate

    # ── Run server ────────────────────────────────────────────────────────────
    from werkzeug.serving import make_server
    from IPython.display import display, HTML

    server = make_server("0.0.0.0", port, app.server)
    shutdown_holder[0] = server.shutdown

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(1)

    if (
        os.getenv("METATLAS2_STANDALONE") == "true"
        or not os.getenv("JUPYTERHUB_SERVICE_PREFIX")
    ):
        url = f"http://localhost:{port}/"
    else:
        service_prefix = os.getenv("JUPYTERHUB_SERVICE_PREFIX", "/")
        url = f"{service_prefix}proxy/{port}/"

    display(HTML(f'<a href="{url}" target="_blank">Open Standards Annotation GUI</a>'))

    t.join()
