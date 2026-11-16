from __future__ import annotations

import math
import json
import os
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import metatlas2.logging_config as lcf
import metatlas2.standards_annotation_db as sadb
import metatlas2.file_and_project_format as fpf
from metatlas2.standards_annotation_tools import STANDARD_CE_LABELS
from metatlas2.standards_annotation_review import (
    fix_peak_labels, review_directory, validate_selected_peaks,
)
from metatlas2.utils import should_disable_tqdm, safe_float, safe_isnan

logger = lcf.get_logger("standards_annotation_summary")

# Reuse helpers from analysis_summary
from metatlas2.analysis_summary import (
    _plot_raw_ms2,
    _plot_empty_ms2,
    _plot_structure,
    _strip_non_chars,
)

# Fixed adduct color palette (must match standards_annotation_gui.py)
_ADDUCT_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#aec7e8", "#ffbb78",
]


def _adduct_color(adduct: str, adduct_list: list[str]) -> str:
    try:
        i = adduct_list.index(adduct)
    except ValueError:
        i = 0
    return _ADDUCT_COLORS[i % len(_ADDUCT_COLORS)]


def _parse_run_number(filename: str) -> str:
    """Extract the run_number field from an h5 filename (numeric part only)."""
    try:
        from metatlas2.file_and_project_format import parse_file_name
        fields = parse_file_name(os.path.basename(filename))
        rn = fields.get("run_number", "")
        return re.sub(r"^[Rr]un", "", rn) if rn else ""
    except Exception:
        return ""


def _to_pylist(val) -> list:
    """Safely convert any value to a plain Python list."""
    import json as _json
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = _json.loads(val)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except Exception:
            return []
    return []

# EIC plot

def _plot_sa_eic(
    ax,
    ms1_df: pd.DataFrame,
    selections_for_compound: pd.DataFrame,
    compound_name: str,
    adduct_list: list[str],
    log_scale: bool = False,
) -> None:
    """Plot EIC traces for all adducts; circle markers at selected RT peaks.

    X-axis clipped to ±1 min around smallest/largest selected RT peak.
    Column ``rt_peak`` (from sa_peaks) is used for the marker position.
    """
    if ms1_df is None or ms1_df.empty:
        ax.text(0.5, 0.5, "No MS1 data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        return

    selected_rts: list[float] = []
    if not selections_for_compound.empty:
        for _, sel_row in selections_for_compound.iterrows():
            rt_peak = safe_float(sel_row.get("rt_peak"))
            if not safe_isnan(rt_peak):
                selected_rts.append(rt_peak)
            for scan in sel_row["ms2_scans"]:
                scan_rt = safe_float(scan["scan_rt"])
                if not safe_isnan(scan_rt):
                    selected_rts.append(scan_rt)

    x_min = (min(selected_rts) - 1.0) if selected_rts else None
    x_max = (max(selected_rts) + 1.0) if selected_rts else None

    selected_adducts: set[str] = set()
    if not selections_for_compound.empty and "adduct" in selections_for_compound.columns:
        selected_adducts = set(selections_for_compound["adduct"].tolist())

    for adduct in adduct_list:
        color = _adduct_color(adduct, adduct_list)
        is_selected = adduct in selected_adducts
        line_alpha = 0.9 if is_selected else 0.25
        adduct_ms1 = ms1_df[ms1_df["adduct"] == adduct] if "adduct" in ms1_df.columns else pd.DataFrame()
        for _, row in adduct_ms1.iterrows():
            spec_rts = _to_pylist(row.get("spec_rts"))
            spec_ints = _to_pylist(row.get("spec_ints"))
            if not spec_rts or not spec_ints:
                continue
            i_vals = np.log10(np.maximum(spec_ints, 1)) if log_scale else spec_ints
            ax.plot(spec_rts, i_vals, color=color, linewidth=0.8, alpha=line_alpha)

    if not selections_for_compound.empty:
        for _, sel_row in selections_for_compound.iterrows():
            adduct = str(sel_row.get("adduct", ""))
            rt_peak = safe_float(sel_row.get("rt_peak"))
            color = _adduct_color(adduct, adduct_list)
            if not safe_isnan(rt_peak):
                adduct_ms1 = ms1_df[ms1_df["adduct"] == adduct] if "adduct" in ms1_df.columns else pd.DataFrame()
                peak_int = float("nan")
                for _, row in adduct_ms1.iterrows():
                    spec_rts = np.asarray(_to_pylist(row.get("spec_rts")), dtype=np.float64)
                    spec_ints = np.asarray(_to_pylist(row.get("spec_ints")), dtype=np.float64)
                    if len(spec_rts) == 0:
                        continue
                    idx = int(np.argmin(np.abs(spec_rts - rt_peak)))
                    candidate = float(spec_ints[idx])
                    if safe_isnan(peak_int) or candidate > peak_int:
                        peak_int = candidate
                if safe_isnan(peak_int):
                    peak_int = 0.0
                y_val = np.log10(max(peak_int, 1)) if log_scale else peak_int
                ax.plot(rt_peak, y_val, "o", color=color, markersize=10,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)
                # Label: "<n>", or "<n>[M]" for manually added peaks
                pn = sel_row.get("peak_num")
                lbl = "" if safe_isnan(pn) else str(int(pn))
                if bool(sel_row.get("is_manual", False)):
                    lbl = f"{lbl}[M]"
                if lbl:
                    ax.annotate(
                        lbl, (rt_peak, y_val),
                        textcoords="offset points", xytext=(9, 0),
                        ha="left", va="center", fontsize=10, weight="bold",
                        color=color, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    )
                peak_ms1 = adduct_ms1[adduct_ms1["mz_rt_uid"] == sel_row["mz_rt_uid"]]
                if len(peak_ms1) != 1:
                    raise ValueError(f"Expected one MS1 EIC for peak {sel_row['peak_uid']}")
                rts = np.asarray(_to_pylist(peak_ms1.iloc[0]["spec_rts"]), dtype=float)
                ints = np.asarray(_to_pylist(peak_ms1.iloc[0]["spec_ints"]), dtype=float)
                if len(rts) == 0 or len(rts) != len(ints):
                    raise ValueError(f"Invalid MS1 EIC for peak {sel_row['peak_uid']}")
                for scan in sel_row["ms2_scans"]:
                    scan_rt = safe_float(scan["scan_rt"])
                    if safe_isnan(scan_rt):
                        continue
                    intensity = float(ints[np.argmin(np.abs(rts - scan_rt))])
                    y = np.log10(max(intensity, 1)) if log_scale else intensity
                    ax.plot(scan_rt, y, "^", color=color, markersize=8, zorder=5)

    if x_min is not None:
        ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Retention Time (min)", fontsize=12, weight="bold")
    ax.set_ylabel("Intensity (log₁₀)" if log_scale else "Intensity", fontsize=12, weight="bold")
    ax.tick_params(labelsize=11)
    if not log_scale:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# TIC plot

def _plot_tic(
    ax,
    tic_by_file: dict[str, pd.DataFrame],
    relevant_files: list[str],
) -> None:
    """Plot TIC traces for all relevant files (full extracted range)."""
    if not tic_by_file or not relevant_files:
        ax.text(0.5, 0.5, "No TIC data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return
    for fn in relevant_files:
        tic = tic_by_file.get(fn)
        if tic is None or tic.empty:
            continue
        ax.plot(tic["rt"].tolist(), tic["tic_intensity"].tolist(),
                color="gray", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Retention Time (min)", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _plot_adduct_legend(
    ax,
    adduct_list: list[str],
    selections_for_compound: pd.DataFrame,
) -> None:
    """Render a legend matching EIC line color to adduct name."""
    ax.axis("off")
    if not adduct_list:
        ax.text(0.5, 0.5, "No adducts", ha="center", va="center",
                fontsize=11, color="gray", transform=ax.transAxes)
        return

    selected_adducts: set[str] = set()
    if not selections_for_compound.empty and "adduct" in selections_for_compound.columns:
        selected_adducts = set(selections_for_compound["adduct"].tolist())

    ax.text(0.5, 0.97, "", ha="center", va="top",
            fontsize=12, weight="bold", transform=ax.transAxes)

    y_step = min(0.12, 0.85 / max(len(adduct_list), 1))
    y_pos = 0.85
    for adduct in adduct_list:
        color = _adduct_color(adduct, adduct_list)
        is_selected = adduct in selected_adducts
        ax.plot([0.05, 0.25], [y_pos, y_pos], color=color, linewidth=3,
                transform=ax.transAxes, clip_on=False)
        if is_selected:
            ax.plot(0.15, y_pos, "o", color=color, markersize=10,
                    markeredgecolor="white", markeredgewidth=0.5,
                    transform=ax.transAxes, clip_on=False)
            label = adduct
        else:
            label = f"{adduct} (not selected)"
        ax.text(0.30, y_pos, label, fontsize=10, va="center",
                transform=ax.transAxes,
                color="black" if is_selected else "#888888")
        y_pos -= y_step

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def _sa_figure_worker(kwargs: dict) -> str:
    """Worker: generate and save one per-compound summary PDF.

    ``selections_dicts`` rows come from ``load_selected_peaks_with_ms2()``
    and have columns: ``compound_name``, ``adduct``, ``polarity``,
    ``chromatography``, ``rt_peak``, ``rt_min``, ``rt_max``,
    ``peak_intensity``, ``mz_observed``, ``mz_theoretical``,
    ``analyst_notes``, ``filename``, and a list of ``ms2_scans``.

    Filenames are not shown in the table (too long); instead a
    ``Run<n> = <full filename>`` legend is rendered below the table.
    """
    compound_name = kwargs["compound_name"]
    inchi_key = kwargs.get("inchi_key", "")
    smiles = kwargs.get("smiles", "")
    inchi = kwargs.get("inchi", "")
    fig_path = Path(kwargs["fig_path"])
    adduct_list = kwargs.get("adduct_list", [])
    chrom = kwargs.get("chrom", "")
    pol = kwargs.get("pol", "")
    ce_label = kwargs.get("ce_label", "")

    selections_df = pd.DataFrame(kwargs.get("selections_dicts", []))
    ms1_df = pd.DataFrame(kwargs.get("ms1_dicts", []))
    tic_by_file: dict[str, pd.DataFrame] = {
        fn: pd.DataFrame(rows) for fn, rows in kwargs.get("tic_dicts", {}).items()
    }

    relevant_files: list[str] = []
    if not ms1_df.empty and "filename" in ms1_df.columns:
        relevant_files = ms1_df["filename"].unique().tolist()

    # Build MS2 panels from sa_peak_ms2 columns
    ms2_panels: list[dict] = []
    if not selections_df.empty:
        for _, sel in selections_df.iterrows():
            adduct = str(sel.get("adduct", ""))
            rt_peak = safe_float(sel.get("rt_peak"))
            pn = sel.get("peak_num")
            for scan in sel["ms2_scans"]:
                frag_mzs = _to_pylist(scan["frag_mzs"])
                frag_ints = _to_pylist(scan["frag_ints"])
                ms2_panels.append({
                    "adduct": adduct,
                    "peak_num": pn,
                    "rt_peak": rt_peak,
                    "scan_rt": safe_float(scan["scan_rt"]),
                    "precursor_mz": safe_float(scan["precursor_mz"]),
                    "ce": fpf.extract_collision_energy_label(scan["filename"]),
                    "frag_mzs": frag_mzs,
                    "frag_ints": frag_ints,
                    "has_ms2": bool(frag_mzs and frag_ints),
                })

    n_ms2_panels = len(ms2_panels)
    n_ms2_rows = 0 if n_ms2_panels == 0 else max(1, math.ceil(n_ms2_panels / 4))

    n_table_data_rows = max(1, len(selections_df))
    table_height_in = 0.45 + n_table_data_rows * 0.38

    run_map_lines: list[str] = []
    if not selections_df.empty:
        seen_pairs: set = set()
        for _, sel in selections_df.iterrows():
            fn = str(sel.get("filename", ""))
            if not fn:
                continue
            run_num = _parse_run_number(fn)
            pair = (run_num, fn)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            run_map_lines.append(f"Run{run_num} = {fn}")

    run_map_h_in = 0.25 * len(run_map_lines) + 0.15 if run_map_lines else 0.0
    ms2_row_h = 3.0   # per MS2 row
    eic_row_h = 4.5
    tic_row_h = 2.5
    gap_ms2_eic = 1.5 if n_ms2_rows > 0 else 0.0
    gap_eic_tic = 1.5
    gap_tic_tbl = 1.5
    gap_tbl_map = 0.15

    total_h = (n_ms2_rows * ms2_row_h + gap_ms2_eic
               + eic_row_h + gap_eic_tic + tic_row_h + gap_tic_tbl
               + table_height_in + gap_tbl_map + run_map_h_in)
    fig = plt.figure(figsize=(25, total_h))

    outer_gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[
            n_ms2_rows * ms2_row_h,
            gap_ms2_eic,
            eic_row_h + gap_eic_tic + tic_row_h + gap_tic_tbl
            + table_height_in + gap_tbl_map + run_map_h_in,
        ],
        hspace=0.0,
    )

    ms2_gs = None
    if n_ms2_rows > 0:
        ms2_gs = gridspec.GridSpecFromSubplotSpec(
            n_ms2_rows, 4, subplot_spec=outer_gs[0],
            hspace=0.6,   # MS2 row spacing: tune this (0.45–0.7)
            wspace=0.2,
        )

    # Bottom grid: EIC | gap | TIC | gap | table | gap | run-map
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        7, 4, subplot_spec=outer_gs[2],
        height_ratios=[eic_row_h, gap_eic_tic, tic_row_h, gap_tic_tbl,
                       table_height_in, gap_tbl_map, run_map_h_in],
        hspace=0.0, wspace=0.2,
    )

    # ── MS2 rows ──────────────────────────────────────────────────────────────
    if ms2_gs is not None:
        for row_i in range(n_ms2_rows):
            for col_i in range(4):
                panel_idx = row_i * 4 + col_i
                ax = fig.add_subplot(ms2_gs[row_i, col_i])   # FIX 1: was gs[row_i, col_i]
                if panel_idx < n_ms2_panels:
                    panel = ms2_panels[panel_idx]
                    adduct_color = _adduct_color(panel["adduct"], adduct_list)
                    pn = panel.get("peak_num", 0)
                    name_part = f"{panel['adduct']} (peak{pn})" if pn else panel["adduct"]
                    title_parts = [name_part]
                    title_parts.append(panel["ce"])
                    if not safe_isnan(panel["scan_rt"]):
                        title_parts.append(f"RT={panel['scan_rt']:.4f}")
                        if not safe_isnan(panel["precursor_mz"]):
                            title_parts.append(f"m/z={panel['precursor_mz']:.4f}")
                    title = " | ".join(title_parts)
                    if panel["has_ms2"]:
                        _plot_raw_ms2(ax, mz_arr=panel["frag_mzs"], int_arr=panel["frag_ints"],
                                      rt=panel["scan_rt"], title=title, type="standards_annotation")
                    else:
                        _plot_empty_ms2(ax, title=title, type="standards_annotation")
                    ax.set_title(title, fontsize=10, color=adduct_color, weight="bold")
                else:
                    ax.axis("off")

    # ── EIC row (bottom row 0) ────────────────────────────────────────────────
    eic_row = 0
    ax_eic_lin = fig.add_subplot(bottom_gs[eic_row, 0])
    _plot_sa_eic(ax_eic_lin, ms1_df, selections_df, compound_name, adduct_list, log_scale=False)
    ax_eic_lin.set_title("EIC (linear)", fontsize=14)

    ax_eic_log = fig.add_subplot(bottom_gs[eic_row, 1])
    _plot_sa_eic(ax_eic_log, ms1_df, selections_df, compound_name, adduct_list, log_scale=True)
    ax_eic_log.set_title("EIC (log₁₀)", fontsize=14)

    ax_legend = fig.add_subplot(bottom_gs[eic_row, 2])
    _plot_adduct_legend(ax_legend, adduct_list, selections_df)

    ax_struct = fig.add_subplot(bottom_gs[eic_row, 3])
    _plot_structure(ax_struct, smiles, inchi, inchi_key, compound_name=compound_name, size=400)

    # ── TIC row (bottom row 1) ────────────────────────────────────────────────
    ax_tic = fig.add_subplot(bottom_gs[2, :])
    _plot_tic(ax_tic, tic_by_file, relevant_files)
    ax_tic.set_title("TIC", fontsize=14, weight="bold")

    # ── Table row (bottom row 2) ──────────────────────────────────────────────
    ax_table = fig.add_subplot(bottom_gs[4, :])
    ax_table.axis("off")

    if not selections_df.empty:
        def _ppm_diff(sel_row):
            mz_obs = safe_float(sel_row.get("mz_observed"))
            mz_theo = safe_float(sel_row.get("mz_theoretical"))
            if safe_isnan(mz_obs) or safe_isnan(mz_theo) or mz_theo == 0:
                return "N/A"
            return f"{abs(mz_obs - mz_theo) / mz_theo * 1e6:.1f}"

        # ── FIX 2 (changed): Filename column removed; Run # links to the
        #    run-map text below the table.
        table_cols = ["adduct", "peak_num", "mz_theoretical", "mz_observed", "_ppm_diff",
                      "rt_peak", "peak_intensity", "_run_num", "analyst_notes"]
        display_cols = ["Adduct", "Peak #", "Theo. m/z", "Obs. m/z", "ppm diff",
                        "RT peak", "Intensity", "Run #", "Notes"]
        table_data = []
        for _, sel in selections_df.iterrows():
            row_vals = []
            for col in table_cols:
                if col == "_ppm_diff":
                    row_vals.append(_ppm_diff(sel))
                elif col == "_run_num":
                    fn = str(sel.get("filename", ""))
                    row_vals.append(_parse_run_number(fn) if fn else "")
                elif col == "peak_num":
                    pn = sel.get("peak_num")
                    row_vals.append(str(int(pn)) if not safe_isnan(pn) else "")
                elif col in ("mz_theoretical", "mz_observed"):
                    val = safe_float(sel.get(col))
                    row_vals.append(f"{val:.4f}" if not safe_isnan(val) else "N/A")
                elif col == "rt_peak":
                    val = safe_float(sel.get(col))
                    row_vals.append(f"{val:.4f}" if not safe_isnan(val) else "N/A")
                elif col == "peak_intensity":
                    val = safe_float(sel.get(col))
                    row_vals.append(f"{val:.2e}" if not safe_isnan(val) else "N/A")
                else:
                    val = sel.get(col, "")
                    row_vals.append(str(val) if val else "")
            table_data.append(row_vals)

        tbl = ax_table.table(cellText=table_data, colLabels=display_cols,
                             loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.4)

        # ── FIX 2 (changed): per-column widths for the 8 remaining columns
        #    (Notes absorbs the space freed by Filename).
        _col_rel = [60, 25, 60, 60, 40, 50, 50, 25, 100]
        _col_total = sum(_col_rel)
        _col_widths = [r / _col_total for r in _col_rel]
        n_tbl_rows = len(table_data) + 1  # +1 for header
        for ci, w in enumerate(_col_widths):
            for ri in range(n_tbl_rows):
                tbl[ri, ci].set_width(w)

        # Style header row
        for col_idx in range(len(display_cols)):
            cell = tbl[0, col_idx]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DDDDDD")
        # (FIX 2: removed Filename-column styling block)
    else:
        ax_table.text(0.5, 0.5, "No selections recorded.",
                      ha="center", va="center", fontsize=12, color="gray",
                      transform=ax_table.transAxes)

    # ── FIX 2 (new): Run# → filename map (bottom row 3) ───────────────────────
    ax_runmap = fig.add_subplot(bottom_gs[6, :])
    ax_runmap.axis("off")
    if run_map_lines:
        ax_runmap.text(0.0, 1.0, "\n".join(run_map_lines),
                       fontsize=12, family="monospace",
                       ha="left", va="top", transform=ax_runmap.transAxes)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    ce_part = f"  |  {ce_label}" if ce_label else ""
    chrom_pol_line = f"{chrom}  |  {pol}{ce_part}" if (chrom or pol) else ""
    cmp_line = textwrap.fill(_strip_non_chars(compound_name), width=100, break_long_words=True)
    title_text = f"{chrom_pol_line}\n{cmp_line}" if chrom_pol_line else cmp_line
    fig.suptitle(title_text, fontsize=25, weight="bold", y=1.0, linespacing=1.25)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return compound_name


# ---------------------------------------------------------------------------
# Main summary functions
# ---------------------------------------------------------------------------

def make_standards_annotation_figures(
    data: "StandardsAnnotationData",
    selections_df: pd.DataFrame,
    output_dir: Path,
    max_workers: int = 4,
) -> None:
    """Generate one PDF per compound per chrom+pol combination.

    ``selections_df`` must have columns from ``load_selected_peaks_with_ms2()``:
    ``compound_name``, ``adduct``, ``polarity``, ``chromatography``,
    ``rt_peak``, ``filename``, ``mz_theoretical``, ``smiles``, ``inchi``,
    ``inchi_key`` and ``ms2_scans``, among other peak metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ms1_df = data.ms1_df
    tic_by_file = data.tic_by_file
    expanded_table = data.expanded_table

    compound_names = list(dict.fromkeys(expanded_table["compound_name"].tolist()))

    tasks: list[dict] = []
    plotted_peak_uids: set[str] = set()

    for compound_name in compound_names:
        sub = expanded_table[expanded_table["compound_name"] == compound_name]
        if sub.empty:
            continue
        first = sub.iloc[0]
        inchi_key = str(first.get("inchi_key", ""))
        smiles = str(first.get("smiles", ""))
        inchi = str(first.get("inchi", ""))

        # Filter selections to this compound
        if not selections_df.empty and "compound_name" in selections_df.columns:
            sel_sub = selections_df[selections_df["compound_name"] == compound_name]
        else:
            sel_sub = pd.DataFrame()

        # MS1 data for this compound
        uid_to_adduct = dict(zip(sub["mz_rt_uid"].tolist(), sub["adduct"].tolist()))
        cmp_ms1 = ms1_df[ms1_df["mz_rt_uid"].isin(sub["mz_rt_uid"].tolist())].copy()
        if not cmp_ms1.empty:
            cmp_ms1["adduct"] = cmp_ms1["mz_rt_uid"].map(uid_to_adduct)

        # Unique chrom+pol+CE combinations from expanded_table
        chrom_pol_ce_triples: list[tuple[str, str, str]] = []
        seen_cpc: set[tuple[str, str, str]] = set()
        for _, row in sub.iterrows():
            chrom = str(row.get("chromatography", ""))
            pol = str(row.get("polarity", ""))
            ce = str(row.get("collision_energy_label", ""))
            cpc = (chrom, pol, ce)
            if cpc not in seen_cpc:
                seen_cpc.add(cpc)
                chrom_pol_ce_triples.append(cpc)

        for chrom, pol, ce_label in chrom_pol_ce_triples:
            # Filter selections to this chrom+pol+CE
            if not sel_sub.empty and "chromatography" in sel_sub.columns and "polarity" in sel_sub.columns:
                cp_sel = sel_sub[
                    (sel_sub["chromatography"] == chrom) &
                    (sel_sub["polarity"] == pol)
                ]
                # Further filter by CE if the column is present
                if "collision_energy_label" in cp_sel.columns:
                    cp_sel = cp_sel[cp_sel["collision_energy_label"] == ce_label]
            else:
                cp_sel = pd.DataFrame()

            if cp_sel.empty:
                continue
            plotted_peak_uids.update(cp_sel["peak_uid"].astype(str))

            # Filter MS1 to this chrom+pol+CE
            cp_sub = sub[
                (sub["chromatography"] == chrom) &
                (sub["polarity"] == pol) &
                (sub.get("collision_energy_label", pd.Series(dtype=str)).fillna("") == ce_label
                 if "collision_energy_label" in sub.columns
                 else pd.Series([True] * len(sub), index=sub.index))
            ]
            cp_uids = cp_sub["mz_rt_uid"].tolist()
            cp_ms1 = cmp_ms1[cmp_ms1["mz_rt_uid"].isin(cp_uids)] if not cmp_ms1.empty else pd.DataFrame()

            adduct_list = list(dict.fromkeys(cp_sub["adduct"].tolist()))

            relevant_files = cp_ms1["filename"].unique().tolist() if not cp_ms1.empty else []
            tic_dicts = {
                fn: tic_by_file[fn].to_dict("records")
                for fn in relevant_files
                if fn in tic_by_file
            }

            chrom_safe = re.sub(r"[^\w\-]", "_", chrom)
            pol_safe = re.sub(r"[^\w\-]", "_", pol)
            ce_safe = re.sub(r"[^\w\-]", "_", ce_label) if ce_label else "no_CE"
            name_safe = re.sub(r"[^A-Za-z0-9\-]", "_", _strip_non_chars(str(compound_name)))
            name_safe = re.sub(r"_+", "_", name_safe).strip("_")
            safe_stem = f"{name_safe}__{chrom_safe}_{pol_safe}_{ce_safe}"
            fig_dir = output_dir / chrom_safe / pol_safe
            fig_path = fig_dir / f"{safe_stem}.pdf"

            tasks.append({
                "compound_name": compound_name,
                "inchi_key": inchi_key,
                "smiles": smiles,
                "inchi": inchi,
                "fig_path": str(fig_path),
                "safe_stem": safe_stem,
                "chrom": chrom,
                "pol": pol,
                "ce_label": ce_label,
                "adduct_list": adduct_list,
                "selections_dicts": cp_sel.to_dict("records") if not cp_sel.empty else [],
                "ms1_dicts": cp_ms1.to_dict("records") if not cp_ms1.empty else [],
                "tic_dicts": tic_dicts,
            })

    missing_peaks = (
        set(selections_df["peak_uid"].astype(str)) - plotted_peak_uids
        if not selections_df.empty else set()
    )
    if missing_peaks:
        raise ValueError(f"No summary figure could be scheduled for selected peaks: {sorted(missing_peaks)}")
    if not tasks:
        logger.info("No compounds with selections to generate figures for.")
        return

    logger.info(f"Generating {len(tasks)} summary figures using {max_workers} workers...")
    pbar = tqdm(total=len(tasks), desc="Generating summary figures", disable=should_disable_tqdm())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_sa_figure_worker, t): t for t in tasks}
        for fut in futures:
            task = futures[fut]
            name = task["compound_name"]
            try:
                fut.result()
                if not Path(task["fig_path"]).is_file():
                    raise RuntimeError(f"Figure was not created: {task['fig_path']}")
                logger.debug(f"Generated figure for {name}")
            except Exception as exc:
                raise RuntimeError(f"Failed to generate summary figure for {name}: {exc}") from exc
            finally:
                pbar.update(1)
    pbar.close()
    logger.info(f"Summary figures written to {output_dir}")


def make_standards_annotation_table(
    selections_df: pd.DataFrame,
    output_dir: Path,
    project_name: str,
    *,
    aligned: bool = False,
) -> None:
    """Generate the flat summary CSV table.

    ``selections_df`` must have columns from ``load_selected_peaks_with_ms2()``.
    When ``aligned`` is true, include the aligned RT values and signed peak shift.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "Aligned_" if aligned else ""
    csv_path = output_dir / f"StandardsAnnotation_Summary_{suffix}{project_name}.csv"

    if selections_df is None or selections_df.empty:
        logger.warning("No selections found; skipping summary table generation.")
        return

    def _ppm(row):
        mz_obs = safe_float(row.get("mz_observed"))
        mz_theo = safe_float(row.get("mz_theoretical"))
        if safe_isnan(mz_obs) or safe_isnan(mz_theo) or mz_theo == 0:
            return float("nan")
        return abs(mz_obs - mz_theo) / mz_theo * 1e6

    df = selections_df.copy()
    df["mz_ppm_error"] = df.apply(_ppm, axis=1)

    ce_labels = STANDARD_CE_LABELS
    for ce in ce_labels:
        for field in ("RT (min)", "Precursor M/Z", "Precursor Intensity", "CE", "spectrum", "Filename"):
            df[f"{ce} MS2 {field}" if field != "spectrum" else f"{ce} spectrum"] = None
    for index, row in df.iterrows():
        seen = set()
        for scan in row["ms2_scans"]:
            ce = fpf.extract_collision_energy_label(scan["filename"])
            if ce not in ce_labels or ce in seen:
                raise ValueError(f"Unexpected or duplicate MS2 CE label {ce!r} for peak {row['peak_uid']}")
            seen.add(ce)
            for column, value in {
                f"{ce} MS2 RT (min)": scan["scan_rt"],
                f"{ce} MS2 Precursor M/Z": scan["precursor_mz"],
                f"{ce} MS2 Precursor Intensity": scan["precursor_intensity"],
                f"{ce} MS2 CE": scan["collision_energy"],
                f"{ce} spectrum": json.dumps([_to_pylist(scan["frag_mzs"]), _to_pylist(scan["frag_ints"])]),
                f"{ce} MS2 Filename": scan["filename"],
            }.items():
                df.at[index, column] = value

    # Source file basename (sa_peaks column: filename)
    df["source_file_str"] = df["filename"].apply(
        lambda fn: os.path.basename(str(fn)) if fn else ""
    )

    output_cols = [
        "peak_label", "compound_name", "peak_num", "inchi_key", "adduct", "polarity",
        "chromatography", "collision_energy_label", "source_file_str",
        "mz_theoretical", "mz_observed", "mz_ppm_error",
        "rt_peak", "rt_min", "rt_max", "peak_intensity", "analyst_notes", "annotation_notes",
        *[f"{ce} MS2 {field}" if field != "spectrum" else f"{ce} spectrum"
          for ce in ce_labels
          for field in ("RT (min)", "Precursor M/Z", "Precursor Intensity", "CE", "spectrum", "Filename")],
        "auto_peak_num",
    ]
    display_names = [
        "Compound Label", "Compound Name", "Peak #", "InChIKey", "Adduct", "Polarity",
        "Chromatography", "MS1 File Collision Energy", "Source File", 
        "Theoretical m/z", "Observed m/z", "m/z ppm error",
        "RT peak", "RT min", "RT max", "Peak Intensity", "Analyst Notes", "Annotation Notes", 
        *[f"{ce} MS2 {field}" if field != "spectrum" else f"{ce} spectrum"
          for ce in ce_labels
          for field in ("RT (min)", "Precursor M/Z", "Precursor Intensity", "CE", "spectrum", "Filename")],
        "GUI Peak Num",
    ]
    if aligned:
        for field, name in (
            ("rt_peak", "RT peak"),
            ("rt_min", "RT min"),
            ("rt_max", "RT max"),
        ):
            index = output_cols.index(field) + 1
            output_cols.insert(index, f"{field}_aligned")
            display_names.insert(index, f"Aligned {name}")
        index = output_cols.index("rt_peak_aligned") + 1
        output_cols.insert(index, "rt_delta")
        display_names.insert(index, "RT diff")

    missing = set(output_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot write summary table; missing columns: {sorted(missing)}")
    out_df = df[output_cols].rename(columns=dict(zip(output_cols, display_names)))
    out_df.to_csv(csv_path, index=False)
    logger.info(f"Summary table written to {csv_path}")


# ---------------------------------------------------------------------------
# Downstream export functions
# ---------------------------------------------------------------------------

def export_msms_refs_tab(
    db_path: str,
    output_dir: Path,
    project_name: str,
    instrument: str = "Unknown",
    instrument_type: str = "Unknown",
    fragmentation_method: str = "HCD",
    database: str = "standards_annotation",
) -> Path:
    """Export selected peaks with MS2 data as a metatlas ``.tab`` file.

    Only peaks with an associated MS2 scan (from ``sa_peak_ms2``) are included.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tab_path = output_dir / f"StandardsAnnotation_MsmsRefs_{project_name}.tab"

    df = sadb.load_selected_peaks_with_ms2(db_path)
    if df.empty:
        logger.warning("No analyst selections found; skipping msms-refs export.")
        return tab_path

    df_ms2 = df.explode("ms2_scans").dropna(subset=["ms2_scans"])
    if df_ms2.empty:
        logger.warning("No selected peaks have MS2 data; skipping msms-refs export.")
        return tab_path

    def _spectrum_str(row) -> str:
        mzs = [float(v) for v in _to_pylist(row["ms2_scans"]["frag_mzs"])]
        ints = [float(v) for v in _to_pylist(row["ms2_scans"]["frag_ints"])]
        return repr([mzs, ints])

    rows = []
    for i, (_, row) in enumerate(df_ms2.iterrows(), start=1):
        polarity_str = "positive" if str(row.get("polarity", "POS")).upper() == "POS" else "negative"
        ce = row["ms2_scans"]["collision_energy"]
        ce_str = f"{float(ce):.0f}" if ce is not None and not safe_isnan(safe_float(ce)) else ""
        rows.append({
            "database":             database,
            "id":                   f"{project_name}_{i:04d}",
            "name":                 str(row.get("compound_name", "")),
            "spectrum":             _spectrum_str(row),
            "decimal":              4,
            "precursor_mz":         f"{safe_float(row['ms2_scans']['precursor_mz']):.5f}",
            "polarity":             polarity_str,
            "adduct":               str(row.get("adduct", "")),
            "fragmentation_method": fragmentation_method,
            "collision_energy":     ce_str,
            "instrument":           instrument,
            "instrument_type":      instrument_type,
            "formula":              "",
            "exact_mass":           "",
            "inchi_key":            str(row.get("inchi_key", "")),
            "inchi":                str(row.get("inchi", "")),
            "smiles":               str(row.get("smiles", "")),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(tab_path, sep="\t", index=False)
    logger.info(f"MSMS refs tab written to {tab_path} ({len(out_df)} rows)")
    return tab_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_standards_annotation_summary(
    data: "StandardsAnnotationData",
    config: dict[str, Any] | None = None,
    max_workers: int = 4,
) -> None:
    """Generate all post-annotation summary outputs.

    Loads analyst selections from the project DB via
    ``load_selected_peaks_with_ms2()`` and generates:
      - Per-compound PDF figures (one per chrom+pol with selections)
      - Summary CSV table
    """
    if config is None:
        config = data.config

    db_path = data.project_db_path
    run_name = config["project"]["standards_run_name"]

    from metatlas2.standards_annotation_db import get_project_output_root
    output_root = get_project_output_root(config)
    logger.info("Loading analyst selections from project DB...")
    sadb.validate_annotation_choices(db_path)
    raw_selections = sadb.load_selected_peaks_with_ms2(db_path)
    validate_selected_peaks(raw_selections)
    selections_df = fix_peak_labels(raw_selections)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    outputs_dir = review_directory(output_root, timestamp)
    outputs_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = outputs_dir / "summary_figures"

    logger.info("Generating per-compound summary figures...")
    make_standards_annotation_figures(
        data=data,
        selections_df=selections_df,
        output_dir=figures_dir,
        max_workers=max_workers,
    )

    logger.info("Generating summary CSV table...")
    make_standards_annotation_table(
        selections_df=selections_df,
        output_dir=outputs_dir,
        project_name=run_name,
    )

    (outputs_dir / ".summary_complete").touch(exist_ok=False)
    logger.info(f"Summary outputs written to {outputs_dir}; timestamp: {timestamp}")
    print(f"Summary timestamp: {timestamp}\nSummary directory: {outputs_dir}")