import datetime
import itertools
import shutil
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
import os
import re
import sys
import requests
import statistics
import textwrap
import warnings
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import metatlas2.database_interact as dbi
import metatlas2.file_and_project_format as fpf
import metatlas2.gdrive_upload as gdu
import metatlas2.logging_config as lcf
from metatlas2.note_options import (
    get_note_options_and_hotkeys,
    get_notes_opts,
    should_require_note_selection,
)
logger = lcf.get_logger('analysis_summary')

def run_all_summaries(
    summary_obj: "AnalysisSummary",
    overwrite: bool = False,
) -> None:
    """Run all summary outputs for one analysis.
    """

    if summary_obj.override_parameters.get("skip_outputs") is not None:
        skip_outputs = summary_obj.override_parameters.get("skip_outputs", [])
    else:
        skip_outputs = summary_obj.ta.params.get("skip_outputs", [])
    
    if summary_obj.experimental_data.curation_df is None or summary_obj.experimental_data.curation_df.empty:
        raise ValueError("No manual curation entries found. Please ensure the curation_df is populated before running summaries.")

    _validate_required_note_selections(summary_obj)

    if "final_id_sheet" not in (skip_outputs or []):
        logger.info("Making Final Identification sheet...")
        make_final_id_sheet(summary_obj, overwrite=overwrite)

    if "id_figures" not in (skip_outputs or []):
        logger.info("Making Identification figures...")
        make_identification_figure(summary_obj, overwrite=overwrite, max_workers=8)

    if "eic_thumbnails" not in (skip_outputs or []):
        logger.info("Making EIC thumbnails...")
        make_eic_thumbnails(summary_obj, overwrite=overwrite, max_workers=8)

    if "boxplots" not in (skip_outputs or []):
        logger.info("Making Boxplots...")
        make_boxplots(summary_obj, overwrite=overwrite, max_workers=8)

    if "data_sheets" not in (skip_outputs or []):
        logger.info("Making quantitative data sheets...")
        make_data_sheets(summary_obj, overwrite=overwrite)

    if "manual_curation_csv" not in (skip_outputs or []):
        logger.info("Making Manual curation CSV...")
        make_manual_curation_csv(summary_obj, overwrite=overwrite)

    if "best_ms2_hits_csv" not in (skip_outputs or []):
        logger.info("Making best MS2 hit fragment ions CSV...")
        make_best_ms2_hit_fragment_ions_csv(summary_obj, overwrite=overwrite)

    if "peak_height_filtered_csv" not in (skip_outputs or []):
        logger.info("Making filtered peak height CSV...")
        make_peak_height_filtered_csv(summary_obj, overwrite=overwrite)

    if "log_fold_changes_csv" not in (skip_outputs or []):
        logger.info("Making log fold changes CSV from filtered peak heights...")
        make_log_fold_changes_csv(summary_obj, overwrite=overwrite, use_filter=True)

    if "metabomap" not in (skip_outputs or []):
        logger.info("Making metabomap (merged pos/neg peak heights + LFC table)...")
        make_metabomap(summary_obj, overwrite=overwrite)

    if "analysis_parquet" not in (skip_outputs or []):
        logger.info("Making unified analysis Parquet files...")
        make_analysis_parquet(summary_obj, overwrite=overwrite)

    gdrive_upload = False
    if summary_obj.override_parameters.get("upload_to_gdrive") is not None:
        gdrive_upload = summary_obj.override_parameters["upload_to_gdrive"]
    elif summary_obj.ta.params.get("upload_to_gdrive") is not None:
        gdrive_upload = summary_obj.ta.params["upload_to_gdrive"]
    if gdrive_upload:
        logger.info("Uploading outputs to Google Drive...")
        gdu.copy_outputs_to_google_drive(summary_obj, "ANALYSIS_SUMMARY", overwrite=overwrite)

###############################################
#### Global helpers
###############################################

def should_disable_tqdm():
    return "SLURM_JOB_ID" in os.environ

def _strip_non_chars(text: str) -> str:
    """Remove Unicode non-characters (e.g. U+FFFE/FFFF) that DejaVu Sans cannot render."""
    return "".join(c for c in text if ord(c) not in range(0xFDD0, 0xFDF0) and ord(c) & 0xFFFF not in (0xFFFE, 0xFFFF))


def _display_compound_idx(compound_idx: int) -> int:
    """Convert internal zero-based index to user-facing one-based index."""
    return int(compound_idx) + 1


def _resolve_summary_note_options(summary_obj: "AnalysisSummary") -> tuple[list[str], list[str]]:
    """Resolve MS1/MS2 option lists with the same owner/override logic as the GUI."""
    owner = summary_obj.config.owner
    ms2_defaults, ms1_defaults, _ = get_notes_opts(owner=owner)

    overrides = getattr(summary_obj, "override_parameters", {}) or {}
    note_overrides = overrides.get("note_options_overrides") or {}

    ms1_options, _ = get_note_options_and_hotkeys(note_overrides.get("ms1_notes", {}), ms1_defaults)
    ms2_options, _ = get_note_options_and_hotkeys(note_overrides.get("ms2_notes", {}), ms2_defaults)
    return ms1_options, ms2_options


def _validate_required_note_selections(summary_obj: "AnalysisSummary") -> None:
    """Raise if required GUI notes remain at unresolved defaults."""
    overrides = getattr(summary_obj, "override_parameters", {}) or {}
    force_eval = (
        overrides["gui_require_all_evaluated"]
        if overrides.get("gui_require_all_evaluated") is not None
        else summary_obj.ta.params.get("gui_require_all_evaluated", False)
    )
    if not force_eval:
        return

    if summary_obj.experimental_data.curation_df is None or summary_obj.experimental_data.curation_df.empty:
        return

    ms1_options, ms2_options = _resolve_summary_note_options(summary_obj)
    mc = summary_obj.experimental_data.curation_df.reset_index(drop=True)

    ms1_bad_mask = mc["ms1_notes"].apply(lambda v: should_require_note_selection(v, ms1_options))
    ms2_bad_mask = mc["ms2_notes"].apply(lambda v: should_require_note_selection(v, ms2_options))

    if not ms1_bad_mask.any() and not ms2_bad_mask.any():
        return

    details = []
    if ms1_bad_mask.any():
        first_default = ms1_options[0] if ms1_options else ""
        bad_rows = mc[ms1_bad_mask].head(8)
        examples = ", ".join(
            f"{_display_compound_idx(i)}:{row.get('compound_name', 'unknown')}"
            for i, row in bad_rows.iterrows()
        )
        details.append(
            f"MS1 notes still at unresolved default '{first_default}' for {int(ms1_bad_mask.sum())} compound(s). Examples: {examples}"
        )

    if ms2_bad_mask.any():
        first_default = ms2_options[0] if ms2_options else ""
        bad_rows = mc[ms2_bad_mask].head(8)
        examples = ", ".join(
            f"{_display_compound_idx(i)}:{row.get('compound_name', 'unknown')}"
            for i, row in bad_rows.iterrows()
        )
        details.append(
            f"MS2 notes still at unresolved default '{first_default}' for {int(ms2_bad_mask.sum())} compound(s). Examples: {examples}"
        )

    raise ValueError(
        "gui_require_all_evaluated is true, but unresolved default note selections were found. "
        "Please update those compounds in the GUI before running summaries. "
        + " | ".join(details)
    )

def _safe_float(value, default: float = np.nan) -> float:
    """Coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Value is not a valid float: {value}")
        return float(default)


def _safe_isnan(value) -> bool:
    """Return True if *value* is NaN, None, or cannot be coerced to a float.
    """
    if value is None:
        return True
    try:
        return np.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    # scalar or other iterable — wrap in list so callers always get a list
    try:
        return list(value)
    except TypeError:
        return []


def _jsonable_list(value) -> list:
    """Return a list with numpy scalars coerced to native Python values.

    This prevents ``json.dumps`` failures like ``TypeError: Object of type
    float32 is not JSON serializable`` when dataframe cells contain numpy
    scalar dtypes.
    """
    out = []
    for elem in _as_list(value):
        if isinstance(elem, np.generic):
            elem = elem.item()
        out.append(elem)
    return out

###############################################
#### Identification Figure
###############################################

def make_identification_figure(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / "identification_figures"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info("Overwriting disabled: existing directory %s will be used.", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting identification figures to %s", output_dir)

    color_map = None
    if hasattr(summary_obj, "override_parameters") and summary_obj.override_parameters.get("gui_lcmsruns_colors"):
        color_map = summary_obj.override_parameters["gui_lcmsruns_colors"]
    elif hasattr(summary_obj, "ta.params") and summary_obj.ta.params.get("gui_lcmsruns_colors"):
        color_map = summary_obj.ta.params["gui_lcmsruns_colors"]

    manual_curation_df = summary_obj.experimental_data.curation_df
    ms1_all_df = summary_obj.experimental_data.ms1_df
    ms2_all_df = summary_obj.experimental_data.ms2_df

    total_files = ms1_all_df["filename"].nunique() if (ms1_all_df is not None and not ms1_all_df.empty) else 0
    logger.info("Plotting %d compounds across %d files.", len(manual_curation_df), total_files)

    def _pregroup(df: Optional[pd.DataFrame]) -> dict:
        if df is None or df.empty:
            return {}
        return {
            key: grp.reset_index(drop=True)
            for key, grp in df.groupby("mz_rt_uid", sort=False)
        }

    ms1_groups = _pregroup(ms1_all_df)
    ms2_groups = _pregroup(ms2_all_df)
    empty_df = pd.DataFrame()

    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")

        safe_name = (
            f"{cmp_idx_display:04d}_{compound_name}_{adduct}"
            .replace("/", "-").replace(" ", "_")
        )
        fig_path = output_dir / f"{safe_name}.pdf"

        tasks.append({
            "mc_row_dict":   mc_row.to_dict(),
            "compound_name": compound_name,
            "adduct":        adduct,
            "inchi_key":     inchi_key,
            "compound_idx":  cmp_idx,
            "fig_path":      str(fig_path),
            "color_map":     color_map,
            "ms1_df":        ms1_groups.get(mz_rt_uid, empty_df),
            "ms2_df":        ms2_groups.get(mz_rt_uid, empty_df),
        })

    if not tasks:
        logger.info("Nothing to generate.")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info("Generating %d figures using %d workers...", len(tasks), n_workers)

    pbar = tqdm(
        total=len(tasks), desc="Generating ID figures",
        unit="compound", disable=should_disable_tqdm(),
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_identification_figure_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Exported identification figure for %s", name)
            except Exception as exc:
                logger.error("Failed to generate figure for %s: %s", name, exc)
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)
    pbar.close()
    logger.info("Identification figure export complete")


def _identification_figure_worker(kwargs: dict) -> str:
    """Worker: generate and save one identification figure PDF."""

    mc_row = pd.Series(kwargs["mc_row_dict"])
    fig_path = Path(kwargs["fig_path"])
    cmp_idx = kwargs["compound_idx"]
    cmp_idx_display = _display_compound_idx(cmp_idx)
    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    inchi_key = kwargs["inchi_key"]
    color_map = kwargs["color_map"]
    ms1_df = kwargs["ms1_df"]
    ms2_df = kwargs["ms2_df"]

    top3: list[dict] = []
    best_scan_row = None
    if not ms2_df.empty:            
        best_score = -np.inf
        for _, scan_row in ms2_df.iterrows():
            hits = _as_list(scan_row.get("hits", [])) if "hits" in scan_row else []
            if hits:
                score = float(hits[0].get("score", -np.inf))
                if score > best_score:
                    best_score = score
                    best_scan_row = scan_row
        if best_scan_row is not None:
            top3 = _as_list(best_scan_row.get("hits", []))[:3]
            # Attach scan-level metadata the hit dicts don't carry
            for hit in top3:
                hit.setdefault("scan_rt", float(best_scan_row.get("scan_rt", np.nan)))
                hit.setdefault("filename", str(best_scan_row.get("filename", "")))

    fig = plt.figure(figsize=(25, 15))
    gs = fig.add_gridspec(
        3, 4, hspace=0.38, wspace=0.30,
        height_ratios=[1.45, 1.2, 1.35],
    )

    # MS2 mirror panels
    mirror_axes = []
    mirror_has_data = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        _plot_ms2(ax, i, top3, ms2_df)
        has_data = bool(ax.patches or ax.lines)
        mirror_axes.append(ax)
        mirror_has_data.append(has_data)

    # "Experimental" / "Reference" side labels on the last panel with data
    if any(mirror_has_data):
        last_idx = max(i for i, v in enumerate(mirror_has_data) if v)
        ax = mirror_axes[last_idx]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_text_pos = xlim[1] + (xlim[1] - xlim[0]) * 0.05
        if ylim[1] > 0:
            ax.text(x_text_pos, ylim[1] * 0.5, "Experimental",
                    fontsize=12, weight="bold", ha="left", va="center",
                    rotation=90, color="black", clip_on=False)
        if ylim[0] < 0:
            ax.text(x_text_pos, ylim[0] * 0.5, "Reference",
                    fontsize=12, weight="bold", ha="left", va="center",
                    rotation=90, color="black", clip_on=False)

    logger.debug("Plotting structure")
    _plot_structure(
        fig.add_subplot(gs[0, 3]),
        mc_row.get("smiles"), mc_row.get("inchi"), inchi_key, size=500,
    )

    logger.debug("Plotting EIC")
    ax_eic_lin = fig.add_subplot(gs[1, 0])
    _plot_eic(ax_eic_lin, ms1_df, mc_row, log_scale=False, color_map=color_map)
    ax_eic_lin.set_title("EIC (linear scale)", fontsize=18)

    logger.debug("Plotting EIC (log scale)")
    ax_eic_log = fig.add_subplot(gs[1, 1])
    _plot_eic(ax_eic_log, ms1_df, mc_row, log_scale=True, color_map=color_map)
    ax_eic_log.set_title("EIC (log₁₀ scale)", fontsize=18)

    logger.debug("Plotting compound info and hit table")
    _plot_compound_info_table(fig.add_subplot(gs[1, 2:4]), mc_row)
    _plot_hit_info_table(fig.add_subplot(gs[2, 0:4]), top3, mc_row, ms2_df)

    fig.suptitle(
        f"[{cmp_idx_display:04d}] |  {_strip_non_chars(adduct)}  |  {inchi_key}\n"
        f"{_strip_non_chars(compound_name)}\n",
        fontsize=20, weight="bold", y=0.97,
    )
    for y_line in [0.61, 0.345]:
        fig.add_artist(plt.Line2D(
            [0.08, 0.92], [y_line, y_line],
            transform=fig.transFigure,
            color="black", linewidth=1, clip_on=False,
        ))

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return compound_name

def _plot_ms2(ax, panel_idx: int, top3: list[dict], ms2_df: pd.DataFrame) -> None:
    """Populate one MS2 mirror panel."""
    _MIRROR_TITLES = ["Best MS2 Match", "2nd Best MS2 Match", "3rd Best MS2 Match"]
    title = _MIRROR_TITLES[panel_idx]

    if panel_idx < len(top3):
        hit = top3[panel_idx]
        q_mz, q_int = _as_list(hit["query_aligned"][0]), _as_list(hit["query_aligned"][1])
        r_mz, r_int = _as_list(hit["ref_aligned"][0]),   _as_list(hit["ref_aligned"][1])
        _plot_mirror(
            ax, q_mz, q_int, r_mz, r_int,
            frag_colors=hit.get("fragment_colors"),
            score=float(hit.get("score", np.nan)),
            rt=float(hit.get("scan_rt", np.nan)),
            title=title,
        )
    elif panel_idx == 0 and not ms2_df.empty:
        # No hits at all — show raw spectrum from first scan
        raw_row = ms2_df.iloc[0]
        _plot_raw_ms2(
            ax,
            mz_arr=_as_list(raw_row.get("data_frags", [])),
            int_arr=_as_list(raw_row.get("frag_ints", [])),
            rt=float(raw_row.get("scan_rt", np.nan)),
            title=title,
        )
    else:
        _plot_empty_ms2(ax, title=title)

def _plot_mirror(
    ax,
    qry_mz: list, qry_int: list,
    ref_mz: list, ref_int: list,
    frag_colors: Optional[list] = None,
    score: float = np.nan,
    rt: float = np.nan,
    title: str = "",
) -> None:
    """Draw a mirror plot: query above zero, reference below."""
    ax.axhline(y=0, color="black", linewidth=1.0)

    if frag_colors is None or len(frag_colors) != len(qry_mz):
        frag_colors = ["tomato"] * len(qry_mz)

    if qry_mz and qry_int:
        for mz, intensity, color in zip(qry_mz, qry_int, frag_colors):
            if not _safe_isnan(mz) and not _safe_isnan(intensity):
                ax.bar(float(mz), float(intensity), color=color, width=1.1, alpha=0.85)

    scale = 1.0
    if ref_mz and ref_int:
        qry_int_valid = [v for v in qry_int if v is not None and not np.isnan(v)]
        ref_int_valid = [v for v in ref_int if v is not None and not np.isnan(v)]
        max_ref = max(ref_int_valid) if ref_int_valid else 0.0
        max_qry = max(qry_int_valid) if qry_int_valid else 0.0
        scale = (max_qry / max_ref) if max_ref > 0 else 1.0
        for mz, intensity, color in zip(ref_mz, ref_int, frag_colors):
            if not _safe_isnan(mz) and not _safe_isnan(intensity):
                ax.bar(float(mz), -float(intensity) * scale, color=color, width=1.1, alpha=0.6)

    if qry_mz and qry_int:
        mz_np = np.array(qry_mz, dtype=float)
        int_np = np.array(qry_int, dtype=float)
        valid = np.isfinite(mz_np) & np.isfinite(int_np) & (int_np > 0)

        if np.any(valid):
            valid_idx = np.where(valid)[0]
            top_n = min(5, len(valid_idx))
            order = np.argsort(int_np[valid_idx])[::-1][:top_n]
            top_idx = valid_idx[order]
            top_idx_sorted = sorted(top_idx, key=lambda i: mz_np[i])

            MIN_MZ_GAP = 5.0
            TEXT_HEIGHT_FRACTION = 0.09
            y_max = float(int_np[valid_idx].max())
            prev_x = None
            stagger_level = 0

            for idx in top_idx_sorted:
                x_txt = float(mz_np[idx])
                y_base = float(int_np[idx]) * 1.02
                if prev_x is not None and abs(x_txt - prev_x) < MIN_MZ_GAP:
                    stagger_level += 1
                else:
                    stagger_level = 0
                y_txt = y_base + stagger_level * (y_max * TEXT_HEIGHT_FRACTION)
                ax.text(
                    x_txt, y_txt, f"{mz_np[idx]:.4f}",
                    fontsize=10, ha="center", va="bottom", color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
                    clip_on=False,
                )
                prev_x = x_txt
            ax.margins(y=0.20)

    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel(f"Intensity (Ref x{scale:.2f})", fontsize=14, weight="bold")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(labelsize=14)

    score_str = f"{score:.3f}" if not np.isnan(score) else "N/A"
    rt_str = f"{rt:.2f}" if not np.isnan(rt) else "N/A"
    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, f"Score: {score_str}", fontsize=14, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: {rt_str} min", fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_raw_ms2(
    ax,
    mz_arr: list, int_arr: list,
    rt: float = np.nan,
    title: str = "",
) -> None:
    """Draw a raw MS2 spectrum (no reference match) on *ax*."""
    ax.axhline(y=0, color="black", linewidth=1.0)
    if mz_arr and int_arr:
        for mz, intensity in zip(mz_arr, int_arr):
            ax.bar(mz, intensity, color="tomato", width=1.1, alpha=0.85)
    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity", fontsize=14)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(labelsize=14)
    rt_str = f"{rt:.2f}" if not np.isnan(rt) else "N/A"
    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, f"Score: N/A", fontsize=14, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: {rt_str} min", fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_eic(
    ax,
    ms1_compound_df: pd.DataFrame,
    mc_row: pd.Series,
    log_scale: bool = False,
    color_map: Optional[dict] = None,
) -> None:
    """Plot EIC traces for all files of one compound.

    Each row of ms1_compound_df holds the full EIC for one file in wide
    format: spec_rts (list of RTs), spec_ints (list of intensities).
    Only points where in_feature=True are included since the df has already
    been filtered upstream.
    """
    rt_min = mc_row.get("rt_min", np.nan)
    rt_max = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    if not _safe_isnan(rt_min):
        ax.axvline(_safe_float(rt_min), color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    if not _safe_isnan(rt_max):
        ax.axvline(_safe_float(rt_max), color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    if not _safe_isnan(rt_peak):
        ax.axvline(_safe_float(rt_peak), color="black", linestyle=":", linewidth=1.5, alpha=0.7)

    if ms1_compound_df.empty:
        ax.text(0.5, 0.5, "No MS1 data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
    else:
        for _, row in ms1_compound_df.iterrows():
            spec_rts = _as_list(row.get("spec_rts", []))
            spec_ints = _as_list(row.get("spec_ints", []))
            if not spec_rts or not spec_ints:
                continue
            color = _get_file_color(row.get("filename", ""), color_map)
            i_vals = np.log10(np.maximum(spec_ints, 1)) if log_scale else spec_ints
            ax.plot(
                spec_rts, 
                i_vals, 
                color=color, 
                linewidth=1.0, 
                alpha=0.7,
                )

    ax.set_xlim([rt_min-2.0, rt_max+2.0])
    ax.set_xlabel("Retention Time (min)", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity (log₁₀)" if log_scale else "Intensity", fontsize=14, weight="bold")
    ax.tick_params(labelsize=14)
    if not log_scale:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_compound_info_table(ax, mc_row: pd.Series) -> None:
    """Render a two-column key/value table of compound metadata.

    Uses the same ground-truth metrics as the Final ID sheet:
      - Measured m/z  = top3_mz_centroid_avg (mean mz_centroid of top-3 files by peak_height)
      - m/z ppm Δ     = absolute ppm error vs atlas_mz
      - Measured RT   = curation_df.rt_peak (mean of per-file peak RTs from analyze_ms1())
      - RT Δ          = absolute RT error vs atlas_rt_peak
    """
    ax.axis("off")

    def _fmt(val, fmt=None):
        if val is None:
            return "N/A"
        if isinstance(val, str):
            return val if val.strip() else "N/A"
        if _safe_isnan(val):
            return "N/A"
        try:
            return fmt.format(val) if fmt else str(val)
        except (ValueError, TypeError):
            return str(val)

    # Use top-3 MS1 averages — consistent with Final ID sheet ground-truth values
    atlas_mz   = _safe_float(mc_row.get("atlas_mz"))
    mz_meas    = _safe_float(mc_row.get("top3_mz_centroid_avg"))
    atlas_rt   = _safe_float(mc_row.get("atlas_rt_peak"))
    rt_meas    = _safe_float(mc_row.get("rt_peak"))

    if not _safe_isnan(mz_meas) and not _safe_isnan(atlas_mz) and atlas_mz != 0:
        mz_ppm_abs = abs(mz_meas - atlas_mz) / atlas_mz * 1e6
    else:
        mz_ppm_abs = float("nan")

    rt_delta_abs = abs(atlas_rt - rt_meas) if not (_safe_isnan(atlas_rt) or _safe_isnan(rt_meas)) else float("nan")

    best_intensity = _safe_float(mc_row.get("best_ms1_intensity"))

    rows = [
        ("Compound", _fmt(mc_row.get("compound_name", "Undefined"))),
        ("Formula", _fmt(mc_row.get("formula"))),
        ("Adduct", _fmt(mc_row.get("adduct"))),
        ("Polarity", _fmt(mc_row.get("polarity"))),
        ("Chromatography", _fmt(mc_row.get("chromatography"))),
        ("Atlas m/z", _fmt(atlas_mz, "{:.4f}")),
        ("Measured m/z", _fmt(mz_meas, "{:.4f}")),
        ("m/z ppm Δ", _fmt(mz_ppm_abs, "{:.1f} ppm")),
        ("Atlas RT range", f"{_fmt(mc_row.get('atlas_rt_min'), '{:.3f}')} - {_fmt(mc_row.get('atlas_rt_max'), '{:.3f}')} min"),
        ("Measured RT range", f"{_fmt(mc_row.get('rt_min'), '{:.3f}')} - {_fmt(mc_row.get('rt_max'), '{:.3f}')} min"),
        ("Atlas RT", _fmt(atlas_rt, "{:.3f} min")),
        ("Measured RT", _fmt(rt_meas, "{:.3f} min")),
        ("RT Δ", _fmt(rt_delta_abs, "{:.3f}")),
        ("Max Intensity", _fmt(best_intensity, "{:.3e}") if not _safe_isnan(best_intensity) else "N/A"),
    ]

    y_pos = 1.03
    y_step = 0.09
    for label, value in rows:
        ax.text(0.02, y_pos, f"{label}:", fontsize=12, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.34, y_pos, value, fontsize=14, va="center", transform=ax.transAxes)
        y_pos -= y_step

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1)

def _plot_hit_info_table(
    ax,
    top3: list[dict],
    mc_row: pd.Series,
    ms2_df: pd.DataFrame,
) -> None:
    """Render the best MS2 hit as a Theoretical/Measured/Error table."""
    ax.axis("off")

    if not top3:
        ax.text(0.5, 0.5, "No MS2 hits found.", transform=ax.transAxes,
                ha="center", va="center", fontsize=25, color="gray")
        return

    best_hit = top3[0]

    raw_name = os.path.basename(str(best_hit.get("filename", "")))

    def _to_float(value, default=np.nan):
        try:
            return float(value)
        except Exception:
            return float(default)

    atlas_mz = _to_float(mc_row.get("atlas_mz"))
    measured_mz = _to_float(best_hit.get("mz_measured"))
    ppm_error = (
        (measured_mz - atlas_mz) / atlas_mz * 1e6
        if (not _safe_isnan(measured_mz) and not _safe_isnan(atlas_mz) and atlas_mz != 0)
        else np.nan
    )
    atlas_rt = _to_float(mc_row.get("atlas_rt_peak"))
    measured_rt = _to_float(best_hit.get("scan_rt"))
    rt_error = (
        measured_rt - atlas_rt
        if (not _safe_isnan(measured_rt) and not _safe_isnan(atlas_rt))
        else np.nan
    )
    score = _to_float(best_hit.get("score"))
    num_matches = int(best_hit.get("num_matches", 0))
    ref_frags = int(best_hit.get("ref_frags", 0))

    matched_frags = _as_list(best_hit.get("matched_fragments"))
    frag_str = ", ".join(f"{m:.3f}" for m in matched_frags) if matched_frags else "N/A"

    def _v(val, fmt):
        if val is None or _safe_isnan(val):
            return "N/A"
        return fmt.format(val)

    score_str = f"{score:.4f}\n({num_matches}/{ref_frags})" if not _safe_isnan(score) else "N/A"

    FRAG_WRAP_WIDTH = 80
    frag_lines = textwrap.wrap(frag_str, width=FRAG_WRAP_WIDTH) if frag_str != "N/A" else ["N/A"]
    frag_display = "\n".join(frag_lines)
    n_frag_lines = min(len(frag_lines), 3)

    col_x = [0, 0.3, 0.6, 0.8]
    std_row_h = 0.135
    frag_row_h = std_row_h * max(1, n_frag_lines)

    header_center = 0.83
    ma_center = header_center - std_row_h
    rt_center = header_center - 2 * std_row_h
    frag_top = rt_center - std_row_h / 2
    frag_center = frag_top - frag_row_h / 2

    ax.text(col_x[0], 0.97, raw_name, fontsize=14, weight="bold",
            ha="left", va="center", transform=ax.transAxes)

    GAP = 0.005
    for y_center, h, color in [
        (header_center, std_row_h, "#d0d0d0"),
        (ma_center,     std_row_h, "white"),
        (rt_center,     std_row_h, "#f0f0f0"),
        (frag_center,   frag_row_h, "white"),
    ]:
        ax.add_patch(Rectangle(
            (0, y_center - h / 2 + GAP), 1.0, h - 2 * GAP,
            transform=ax.transAxes, color=color, zorder=0,
        ))

    for x, label in zip(col_x, ["BEST MATCH", "Theoretical", "Measured", "Error/Score"]):
        ax.text(x, header_center, label, fontsize=15, weight="bold",
                ha="left", va="center", transform=ax.transAxes)

    for y_center, row_vals in [
        (ma_center, ("Mass Accuracy",
                     _v(atlas_mz,    "{:.4f} m/z"),
                     _v(measured_mz, "{:.4f} m/z"),
                     _v(ppm_error,   "{:.2f} ppm"))),
        (rt_center, ("RT Accuracy",
                     _v(atlas_rt,    "{:.3f} min"),
                     _v(measured_rt, "{:.3f} min"),
                     _v(rt_error,    "{:.3f} min"))),
    ]:
        for col_idx, val in enumerate(row_vals):
            ax.text(col_x[col_idx], y_center, val, fontsize=15,
                    weight="bold" if col_idx == 0 else "normal",
                    ha="left", va="center", transform=ax.transAxes)

    frag_text_top = frag_top - GAP * 2.5
    ax.text(col_x[0], frag_text_top, "Fragment Matches", fontsize=15,
            weight="bold", ha="left", va="top", transform=ax.transAxes)
    ax.text(col_x[1], frag_text_top, frag_display, fontsize=15,
            ha="left", va="top", transform=ax.transAxes)
    ax.text(col_x[3], frag_text_top, score_str, fontsize=15,
            ha="left", va="top", transform=ax.transAxes)

    # 2nd and 3rd best hit filenames and total file count
    additional_info_y = frag_center - frag_row_h / 2 - 0.04
    line_spacing = 0.07

    if len(top3) >= 2:
        second_file = os.path.basename(str(top3[1].get("filename", "")))
        ax.text(0, additional_info_y, f"2nd best: {second_file}",
                fontsize=13, ha="left", va="top", transform=ax.transAxes)

    if len(top3) >= 3:
        third_file = os.path.basename(str(top3[2].get("filename", "")))
        ax.text(0, additional_info_y - line_spacing, f"3rd best: {third_file}",
                fontsize=13, ha="left", va="top", transform=ax.transAxes)

    total_files = ms2_df["filename"].nunique() if not ms2_df.empty else 0
    y_offset = (
        line_spacing * 2 if len(top3) >= 3
        else line_spacing if len(top3) >= 2
        else 0
    )
    ax.text(0, additional_info_y - y_offset,
            f"Total files with database matches: {total_files}",
            fontsize=13, ha="left", va="top", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def _get_file_color(filename: str, color_map: Optional[Dict[str, str]] = None) -> str:
    """Determine color for a file based on color mapping.
    """
    if color_map is None:
        return "gray"
    
    for key, color in color_map.items():
        if key.lower() in filename.lower():
            return color
    
    return "gray"

def _plot_empty_ms2(ax, title: str = "") -> None:
    """Draw a blank MS2 panel with a 'No MS2 Data' annotation."""
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1e6)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity", fontsize=14, weight="bold")
    ax.tick_params(labelsize=14)
    ax.text(0.5, 0.5, "No MS2 Data", transform=ax.transAxes,
            ha="center", va="center", fontsize=14, weight="bold", color="gray")
    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, f"Score: N/A", fontsize=14, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: N/A", fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_structure(
    ax,
    smiles: Optional[str],
    inchi: Optional[str],
    inchi_key: str,
    size: int = 500,
) -> None:
    """Draw molecular structure using RDKit's MolToImage (requires libXrender, available in container)."""
    ax.axis("off")

    def _clean_chemical_text(value: Optional[str]) -> Optional[str]:
        """Return a usable string chemical identifier, else None."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return None
        return text

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from PIL import Image, ImageDraw, ImageFont

        smiles = _clean_chemical_text(smiles)
        inchi = _clean_chemical_text(inchi)

        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None and inchi:
            mol = Chem.MolFromInchi(inchi)

        if mol is not None:
            # Generate structure image directly (requires libXrender)
            img = Draw.MolToImage(mol, size=(size, size), kekulize=True)

            # Add InChIKey annotation below the structure
            if inchi_key:
                font_size = 25
                line_height = int(font_size * 1.2)
                avg_char_width = font_size / 2
                lines, current_line = [], ""
                
                # Word-wrap the InChIKey
                for char in inchi_key:
                    test_line = current_line + char
                    if len(test_line) * avg_char_width > size - 20:
                        lines.append(current_line)
                        current_line = char
                    else:
                        current_line = test_line
                if current_line:
                    lines.append(current_line)

                # Create new image with space for text
                text_height = len(lines) * line_height + 20
                new_img = Image.new("RGB", (size, size + text_height), "white")
                new_img.paste(img, (0, 0))
                
                # Draw text
                draw_obj = ImageDraw.Draw(new_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
                
                y_position = size + 10
                for line in lines:
                    draw_obj.text((10, y_position), line, fill="black", font=font)
                    y_position += line_height
                
                img = new_img

            ax.imshow(img, aspect="equal")
            return

    except Exception as exc:
        logger.warning("Could not draw structure for %s: %s", inchi_key, exc)

    ax.text(0.5, 0.5, f"InChIKey:\n{inchi_key}", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, weight="bold", color="gray")

###############################################
#### Final ID Sheet
###############################################

def make_final_id_sheet(
    summary_obj: "AnalysisSummary",
    output_filename: str = "Final_Identifications.xlsx",
    overwrite: bool = True,
) -> None:

    output_loc = Path(summary_obj.paths['analysis_results_output_dir'])
    chromatography = summary_obj.chromatography
    analysis_info = (
        f"{summary_obj.chromatography}"
        f"-{summary_obj.polarity}"
        f"-{summary_obj.analysis_type}"
    )
    run_info = f"RTA{summary_obj.rt_alignment_number}-TGA{summary_obj.analysis_number}"
    output_filename = f"{summary_obj.project_name}_{analysis_info}-{run_info}_{output_filename}"
    if not output_filename.endswith(".xlsx"):
        output_filename += ".xlsx"
    excel_path = output_loc / output_filename
    if not overwrite and excel_path.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", excel_path)
        return

    manual_curation_df = summary_obj.experimental_data.curation_df

    compound_info_map: dict[str, tuple] = {}
    mass_map: dict[str, Optional[float]] = {}
    for _, _row in manual_curation_df.iterrows():
        ik = _row.get("inchi_key", "")
        if not ik or ik in compound_info_map:
            continue
        compound_info_map[ik] = (
            _row.get("formula") or None,
            _row.get("smiles") or None,
            _row.get("inchi") or None,
            _row.get("pubchem_cid") or None,
        )
        raw_mass = _row.get("mono_isotopic_molecular_weight")
        try:
            mass_map[ik] = float(raw_mass) if raw_mass not in (None, "", 0, 0.0) else None
        except (TypeError, ValueError):
            mass_map[ik] = None

    # --- Build MS2 lookup map (best hit per compound for MSMS columns only) ---
    ms2_best: dict[str, dict] = {}       # mz_rt_uid -> best hit dict + scan metadata

    ms2_df = summary_obj.experimental_data.ms2_df
    if not ms2_df.empty:
        for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
            best_score = -np.inf
            best_hit_data = None

            for _, scan_row in grp.iterrows():
                hits_list = _as_list(scan_row.get("hits"))
                if not hits_list:
                    continue
                top_hit = hits_list[0]
                score = float(top_hit.get("score", -np.inf))
                if score > best_score:
                    best_score = score
                    best_hit_data = {
                        **top_hit,
                        # Attach scan-level metadata that isn't in the hit dict
                        "filename": scan_row.get("filename", ""),
                        "scan_rt": float(scan_row.get("scan_rt", np.nan)),
                    }

            if best_hit_data is not None:
                ms2_best[uid] = best_hit_data

    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()

    logger.info("Processing %d compounds...", len(manual_curation_df))

    overlapping_map = _compute_all_overlapping_compounds(manual_curation_df, mass_map)

    rows: list[dict] = []
    for compound_idx, mc_row in tqdm(
        manual_curation_df.iterrows(),
        total=len(manual_curation_df),
        desc="Adding compounds to final ID sheet",
        disable=should_disable_tqdm(),
    ):
        compound_idx_display = _display_compound_idx(compound_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")
        polarity = mc_row.get("polarity", "")

        formula, smiles, inchi, pubchem_cid = compound_info_map.get(inchi_key, (None, None, None, None))
        exact_mass = mass_map.get(inchi_key)

        overlapping_compound, overlapping_mz_rt_uids = overlapping_map.get(compound_idx, ("", ""))
        identified_metabolite = compound_name if not overlapping_compound else overlapping_compound

        # --- Shared theoretical values ---
        mz_theoretical = float(mc_row.get("atlas_mz", np.nan))
        rt_theoretical = float(mc_row.get("atlas_rt_peak", np.nan))

        # --- MS1 metrics ---
        # mz_measured: top3_mz_centroid_avg — mean mz_centroid of top-3 files by peak_height
        # rt_measured: curation_df.rt_peak  — mean of per-file peak RTs from analyze_ms1()
        mz_measured = float(mc_row.get("top3_mz_centroid_avg", np.nan))
        rt_measured  = float(mc_row.get("rt_peak", np.nan))
        best_ms1_rt = float(mc_row.get("best_ms1_rt", np.nan))
        best_ms1_intensity = float(mc_row.get("best_ms1_intensity", np.nan))
        best_ms1_file = mc_row.get("best_ms1_file", "")

        # Compute MS1-based m/z and RT errors (absolute values)
        if not np.isnan(mz_measured) and not np.isnan(mz_theoretical) and mz_theoretical != 0:
            ppm_error_ms1 = abs(mz_measured - mz_theoretical) / mz_theoretical * 1e6
            mz_error_da   = abs(mz_theoretical - mz_measured)
        else:
            ppm_error_ms1 = np.nan
            mz_error_da   = np.nan

        rt_error = abs(rt_theoretical - rt_measured) if not (np.isnan(rt_theoretical) or np.isnan(rt_measured)) else np.nan

        ms2_notes = str(mc_row.get("ms2_notes", "") or "")

        # --- MS2 metrics (MSMS columns only — mz/rt quality uses MS1 above) ---
        msms_file = ""
        msms_rt = np.nan
        msms_score = np.nan
        msms_num_ions = ""
        msms_matching_ions = ""

        best_hit = ms2_best.get(mz_rt_uid)
        if best_hit is not None:
            msms_file = str(best_hit.get("filename", ""))
            msms_rt = float(best_hit.get("scan_rt", np.nan))
            msms_score = float(best_hit.get("score", np.nan))

            num_matches = int(best_hit.get("num_matches", 0))
            ref_frags = int(best_hit.get("ref_frags", 0))
            msms_num_ions = f"{num_matches}/{ref_frags}" if ref_frags > 0 else str(num_matches)

            matched_frags = _as_list(best_hit.get("matched_fragments"))
            msms_matching_ions = (
                ",".join(f"{m:.3f}" for m in matched_frags) if matched_frags else "N/A"
            )

            # Single-ion match precursor check
            if num_matches == 1 and not np.isnan(msms_score) and matched_frags:
                single_ion = float(matched_frags[0])
                ppm_tol = float(mc_row.get("mz_tolerance", 5.0))
                precursor_match = (
                    mz_theoretical > 0
                    and abs(single_ion - mz_theoretical) / mz_theoretical * 1e6 <= ppm_tol
                )
                note_tag = (
                    " (single matching fragment is the precursor)"
                    if precursor_match
                    else " (single matching fragment is NOT the precursor)"
                )
                if "1.0, single ion match" in ms2_notes or "0.5, single ion match" in ms2_notes:
                    ms2_notes = ms2_notes + note_tag

        # Quality scores — all MS1-based (absolute errors)
        mz_q = _mz_quality(ppm_error_ms1, mz_error_da)
        rt_q = _rt_quality(rt_error, chromatography)

        try:
            msms_q = float(str(ms2_notes).split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = _total_score_and_msi(msms_q, mz_q, rt_q)

        def _safe_round(val, decimals):
            return round(float(val), decimals) if not np.isnan(float(val)) else np.nan

        rows.append({
            # COMPOUND ANNOTATION
            "index":                  compound_idx_display,
            "identified_metabolite":  identified_metabolite,
            "label":                  compound_name,
            "overlapping_compound":   overlapping_compound,
            "overlapping_inchi_keys": overlapping_mz_rt_uids,
            "formula":                formula,
            "polarity":               polarity,
            "exact_mass":             _safe_round(exact_mass, 7) if exact_mass is not None else np.nan,
            "inchi_key":              inchi_key,
            # COMPOUND IDENTIFICATION SCORES
            "msms_quality":           msms_q,
            "mz_quality":             mz_q,
            "rt_quality":             rt_q,
            "total_score":            total_score,
            "msi_level":              msi_level,
            "isomer_details":         "",
            "identification_notes":   mc_row.get("identification_notes", ""),
            "analyst_notes":          mc_row.get("analyst_notes", ""),
            "other_notes":            mc_row.get("other_notes", "") or "",
            "ms1_notes":              mc_row.get("ms1_notes", "") or "",
            "ms2_notes":              ms2_notes,
            # MS1 INTENSITY INFORMATION
            "max_intensity":          best_ms1_intensity,
            "max_intensity_file":     Path(best_ms1_file).name if best_ms1_file else "",
            "ms1_rt_peak":            _safe_round(best_ms1_rt, 2),
            # MSMS INFORMATION
            "msms_file":              Path(msms_file).name if msms_file else "",
            "msms_rt":                _safe_round(msms_rt, 2),
            "msms_numberofions":      msms_num_ions,
            "msms_matchingions":      msms_matching_ions,
            # MSMS EVALUATION
            "msms_score":             _safe_round(msms_score, 4),
            # ION INFORMATION
            "mz_adduct":              adduct,
            "mz_theoretical":         _safe_round(mz_theoretical, 4),
            "mz_measured":            _safe_round(mz_measured, 4),
            # M/Z EVALUATION
            "mz_error":               _safe_round(mz_error_da, 4),
            "mz_ppmerror":            _safe_round(ppm_error_ms1, 4),
            # CHROMATOGRAPHIC PEAK INFORMATION
            "rt_min":                 _safe_round(float(mc_row.get("rt_min", np.nan)), 2),
            "rt_max":                 _safe_round(float(mc_row.get("rt_max", np.nan)), 2),
            "rt_theoretical":         _safe_round(rt_theoretical, 2),
            "rt_measured":            _safe_round(rt_measured, 2),
            # RT EVALUATION
            "rt_error":               _safe_round(rt_error, 2),
        })

    final_df = pd.DataFrame(rows)
    logger.info("Assembled final ID table with %d rows.", len(final_df))

    COL_NAMES = [
        # COMPOUND ANNOTATION
        "Compound #", "Identified Metabolite", "Name of metabolite searched for",
        "Labels of Overlapping Compounds", "Inchi Keys of Overlapping Compounds",
        "Molecular Formula", "Polarity", "Exact Mass", "Inchi Key",
        # COMPOUND IDENTIFICATION SCORES
        "MSMS Score (0 to 1)", "m/z score (0 to 1)", "RT score (0 to 1)",
        "Total ID Score (0 to 3)", "Mass Spec Initiative Identification Level",
        "Isomer details", "Identification notes", "Analyst notes",
        "Other notes", "MS1 notes", "MS2 notes",
        # MS1 INTENSITY INFORMATION
        "Maximum MS1 intensity across all files", "Filename w/ maximum MS1",
        "Retention time of max intensity MS1 peak",
        # MSMS INFORMATION
        "File with highest MSMS match score", "RT of highest matched MSMS scan",
        "Number of ion matches in msms spectra to EMA reference spectra",
        "List of ion matches in msms spectra to EMA reference spectra",
        # MSMS EVALUATION
        "MSMS score (highest across all samples)",
        # ION INFORMATION
        "Adduct", "Theoretical m/z", "Measured m/z",
        # M/Z EVALUATION
        "mass error (delta Da)", "mass error (delta ppm)",
        # CHROMATOGRAPHIC PEAK INFORMATION
        "Minimum retention time (min)", "Maximum retention time (max)",
        "Theoretical retention time (peak)", "Detected RT (peak)",
        # RT EVALUATION
        "RT error (absolute delta)",
    ]

    rt_q_desc = (
        "1 (delta RT </= 0.25), 0.5 (delta RT > 0.25 & </= 0.5), 0 (delta RT > 0.5 min)"
        if is_c18 else
        "1 (delta RT </= 0.5), 0.5 (delta RT > 0.5 & </= 2), 0 (delta RT > 2 min)"
    )

    COL_DESCRIPTIONS = [
        # COMPOUND ANNOTATION
        "Unique for study",
        "Some isomers are not chromatographically or spectrally resolvable. Some compounds detected w/ >1 adduct (increases identification confidence but only use 1 for analysis).",
        "Name of standard reference compound in library match.",
        "compound with similar mz (abs difference <= 0.005) or monoisotopic molecular weight (abs difference <= 0.005) and RT (min or max within the RT-min-max-range of similar compound)",
        "List of inchi keys that correspond to the compounds listed in the previous column",
        "", "", "monoisotopic mass (neutral except for permanently charged molecules)", "neutralized version",
        # COMPOUND IDENTIFICATION SCORES
        "1 (MSMS matches ref. std.), 0.5 (possible match), 0 (no MSMS collected or no appropriate ref available), -1 (bad match)",
        "1 (delta ppm </= 5 or delta Da </= 0.0015), 0.5 (delta ppm 5-10 and delta Da > 0.0015), 0 (delta ppm > 10) mz_quality",
        rt_q_desc,
        "sum of m/z, RT and MSMS score",
        "Level 1 = Two independent and orthogonal properties match authentic standard; else = putative [Metabolomics. 2007 Sep; 3(3): 211-221. doi: 10.1007/s11306-007-0082-2]",
        "Isomers have same formula (and m/z) and similar RT - MSMS spectra may be used to differentiate (exceptions) or RT elution order",
        "", "", "", "", "",
        # MS1 INTENSITY INFORMATION
        "", "", "",
        # MSMS INFORMATION
        "",
        "",
        "mean # of fragment ions matching between compound in sample and reference compound / standard; may include parent and isotope ions and very low intensity background ions (these do not contribute to score)",
        "",
        # MSMS EVALUATION
        "MSMS score (highest across all samples), scale of 0 to 1 based on an algorithm. 0 = no match, 1 = perfect match. If no score, then no MSMS was acquired for that compound (@ m/z & RT window).",
        # ION INFORMATION
        "More than one may be detectable; the one evaluated is listed",
        "theoretical m/z for a given compound / adduct pair",
        "average m/z within 20ppm of theoretical detected across the top three most intense ions @ RT peak",
        # M/Z EVALUATION
        "absolute difference between theoretical and detected m/z",
        "ppm difference between theoretical and detected m/z",
        # CHROMATOGRAPHIC PEAK INFORMATION
        "Retention range including start and end of detection of an m/z value (Note: Peak Height is calculated as the highest intensity of an m/z within the min/max RT range. Peak Area is calculated as the integrated area under the curve for an m/z within the mix/max RT range.)",
        "",
        "theoretical retention time for a compound based upon reference standard at highest intensity point of peak",
        "average retention time for a detected compound at highest intensity point of peak across all samples",
        # RT EVALUATION
        "absolute difference between theoretical and detected RT peak",
    ]

    COL_FIELDS = [
        "index", "identified_metabolite", "label", "overlapping_compound", "overlapping_inchi_keys",
        "formula", "polarity", "exact_mass", "inchi_key",
        "msms_quality", "mz_quality", "rt_quality", "total_score", "msi_level",
        "isomer_details", "identification_notes", "analyst_notes", "other_notes",
        "ms1_notes", "ms2_notes",
        "max_intensity", "max_intensity_file", "ms1_rt_peak",
        "msms_file", "msms_rt", "msms_numberofions", "msms_matchingions",
        "msms_score",
        "mz_adduct", "mz_theoretical", "mz_measured",
        "mz_error", "mz_ppmerror",
        "rt_min", "rt_max", "rt_theoretical", "rt_measured", "rt_error",
    ]

    _SECTION_SPANS = [
        (0,  8,  "COMPOUND ANNOTATION"),
        (9,  19, "COMPOUND IDENTIFICATION SCORES"),
        (20, 22, "MS1 INTENSITY INFORMATION"),
        (23, 26, "MSMS INFORMATION"),
        (27, 27, "MSMS EVALUATION"),
        (28, 30, "ION INFORMATION"),
        (31, 32, "M/Z EVALUATION"),
        (33, 36, "CHROMATOGRAPHIC PEAK INFORMATION"),
        (37, 37, "RT EVALUATION"),
    ]

    _SECTION_COLORS = {
        "COMPOUND ANNOTATION":              "#FFFFFF",
        "COMPOUND IDENTIFICATION SCORES":   "#DCFFFF",
        "MS1 INTENSITY INFORMATION":        "#FFFFDC",
        "MSMS INFORMATION":                 "#FFFFDC",
        "MSMS EVALUATION":                  "#FFDCFF",
        "ION INFORMATION":                  "#FFFFDC",
        "M/Z EVALUATION":                   "#FFDCFF",
        "CHROMATOGRAPHIC PEAK INFORMATION": "#FFFFDC",
        "RT EVALUATION":                    "#FFDCFF",
    }

    def _col_letter(col_idx: int) -> str:
        result = ""
        n = col_idx + 1
        while n:
            n, remainder = divmod(n - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def _section_for_col(col_idx: int) -> str:
        return next(lbl for s, e, lbl in _SECTION_SPANS if s <= col_idx <= e)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        final_df.to_excel(
            writer,
            sheet_name="Final_Identifications",
            index=False,
            header=False,
            startrow=4,
        )
        workbook = writer.book
        worksheet = writer.sheets["Final_Identifications"]

        nrows = len(final_df) + 4

        # ---- Base formats ----
        f_header_base = {
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
        }
        f_scientific = workbook.add_format({"num_format": "0.00E+00"})

        # Build one header format and one data format per section colour
        section_header_fmts: dict[str, object] = {}
        section_data_fmts: dict[str, object] = {}
        for label, color in _SECTION_COLORS.items():
            section_header_fmts[label] = workbook.add_format({**f_header_base, "bg_color": color})
            section_data_fmts[label] = workbook.add_format({"bg_color": color})

        # ---- Row heights ----
        worksheet.set_row(0, 30)    # section headers
        worksheet.set_row(1, 80)    # column display names
        worksheet.set_row(2, 120)   # descriptions
        worksheet.set_row(3, 40)    # internal field names

        # ---- Column widths ----
        # Default width for all columns; override specifics below
        worksheet.set_column(0,  0,  10)   # index
        worksheet.set_column(1,  4,  30)   # compound name columns + overlapping
        worksheet.set_column(5,  8,  15)   # formula, polarity, exact_mass, inchi_key
        worksheet.set_column(8,  8,  28)   # inchi_key wider
        worksheet.set_column(9,  19, 12)   # scores + notes
        worksheet.set_column(20, 20, 15, f_scientific)  # max_intensity in sci notation
        worksheet.set_column(21, 21, 35)   # max intensity filename
        worksheet.set_column(22, 22, 15)   # ms1 rt peak
        worksheet.set_column(23, 23, 35)   # msms filename
        worksheet.set_column(24, 24, 15)   # msms rt
        worksheet.set_column(25, 25, 25)   # msms num ions
        worksheet.set_column(26, 26, 20)   # msms matching ions
        worksheet.set_column(27, 27, 15)   # msms score
        worksheet.set_column(28, 30, 14)   # ion info
        worksheet.set_column(31, 32, 14)   # mz eval
        worksheet.set_column(33, 37, 14)   # rt cols

        # ---- Row 0: section header merges ----
        for start_idx, end_idx, label in _SECTION_SPANS:
            fmt = section_header_fmts[label]
            start_letter = _col_letter(start_idx)
            end_letter = _col_letter(end_idx)
            if start_idx == end_idx:
                worksheet.write(0, start_idx, label, fmt)
            else:
                worksheet.merge_range(
                    f"{start_letter}1:{end_letter}1", label, fmt
                )

        # ---- Rows 1-3: column names, descriptions, field names ----
        for col_idx, (name, desc, field) in enumerate(zip(COL_NAMES, COL_DESCRIPTIONS, COL_FIELDS)):
            fmt = section_header_fmts[_section_for_col(col_idx)]
            worksheet.write(1, col_idx, name, fmt)
            worksheet.write(2, col_idx, desc, fmt)
            worksheet.write(3, col_idx, field, fmt)

        # ---- Conditional background colours for data rows ----
        for start_idx, end_idx, label in _SECTION_SPANS:
            fmt = section_data_fmts[label]
            start_letter = _col_letter(start_idx)
            end_letter = _col_letter(end_idx)
            worksheet.conditional_format(
                f"{start_letter}5:{end_letter}{nrows}",
                {"type": "no_errors", "format": fmt},
            )

    logger.info("Exported final ID table to %s", excel_path)

def _mz_quality(ppm_error: float, mz_delta: float) -> float:
    """Return 0/0.5/1 mz quality score from ppm and absolute mass error."""
    if np.isnan(ppm_error) or np.isnan(mz_delta):
        return np.nan
    if abs(ppm_error) <= 5 or abs(mz_delta) <= 0.0015:
        return 1.0
    if abs(ppm_error) <= 10:
        return 0.5
    return 0.0

def _rt_quality(rt_error: float, chromatography: str) -> float:
    """Return 0/0.5/1 RT quality score, with thresholds that depend on method."""
    if np.isnan(rt_error):
        return np.nan
    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()
    if is_c18:
        if abs(rt_error) <= 0.25:
            return 1.0
        if abs(rt_error) <= 0.5:
            return 0.5
        return 0.0
    else:
        if abs(rt_error) <= 0.5:
            return 1.0
        if abs(rt_error) <= 2.0:
            return 0.5
        return 0.0

def _total_score_and_msi(
    msms_q: float, mz_q: float, rt_q: float
) -> Tuple[float, str]:
    """Compute total ID score (0-3) and MSI level string.
    """
    scores = [v for v in [msms_q, mz_q, rt_q] if not np.isnan(v)]
    total = float(np.nansum([msms_q, mz_q, rt_q]))

    if msms_q == -1:
        msi = "REMOVE, INVALIDATED BY BAD MSMS MATCH"
    elif len(scores) > 0 and statistics.median(scores) < 1:
        msi = "putative"
    elif total == 3:
        msi = "Exceeds Level 1"
    else:
        msi = "Level 1"
    return total, msi

def _compute_all_overlapping_compounds(
    manual_curation_df: pd.DataFrame,
    mass_map: dict[str, Optional[float]],
) -> dict[int, Tuple[str, str]]:
    """Compute every compound's overlapping set in one vectorized pass.

    Parameters
    ----------
    mass_map:
        Pre-fetched monoisotopic masses keyed by inchi_key.

    Returns
    -------
    dict of {compound_idx: (names_str, indices_str)}
    """
    mc = manual_curation_df.reset_index(drop=True)
    n = len(mc)

    rt_min = mc["rt_min"].to_numpy(dtype=float)
    rt_max = mc["rt_max"].to_numpy(dtype=float)
    mz = mc["atlas_mz"].to_numpy(dtype=float)
    masses = np.array(
        [mass_map.get(ik) if mass_map.get(ik) is not None else np.nan
         for ik in mc["inchi_key"]],
        dtype=float,
    )
    names = mc["compound_name"].tolist()
    inchi_keys = mc["inchi_key"].tolist()

    rt_overlap = (
        np.maximum(rt_min[:, None], rt_min[None, :]) <=
        np.minimum(rt_max[:, None], rt_max[None, :])
    )

    mz_i, mz_j = mz[:, None], mz[None, :]
    mz_valid = (mz_i != 0) & (mz_j != 0) & ~np.isnan(mz_i) & ~np.isnan(mz_j)
    mz_similar = mz_valid & (np.abs(mz_i - mz_j) <= 0.005)

    m_i, m_j = masses[:, None], masses[None, :]
    mass_valid = (m_i != 0) & (m_j != 0) & ~np.isnan(m_i) & ~np.isnan(m_j)
    mass_similar = mass_valid & (np.abs(m_i - m_j) <= 0.005)

    both_mass_zero = (m_i == 0) & (m_j == 0)
    overlap = rt_overlap & (mz_similar | mass_similar) & ~both_mass_zero
    np.fill_diagonal(overlap, True)

    result: dict[int, Tuple[str, str]] = {}
    for i in range(n):
        js = np.where(overlap[i])[0]
        if len(js) <= 1:
            result[i] = ("", "")
        else:
            result[i] = (
                "//".join(names[j] for j in js),
                "//".join(inchi_keys[j] for j in js),
            )
    return result

###############################################
#### EIC Figures
###############################################

def make_eic_thumbnails(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    """Generate per-compound EIC thumbnail PDFs in two output folders (parallelised)."""

    rt_alignment_num = summary_obj.rt_alignment_number
    analysis_num = summary_obj.analysis_number

    output_loc = Path(summary_obj.paths['analysis_results_output_dir'])
    base_dir = Path(output_loc)
    shared_base_dir = base_dir / "eics"
    dir_shared = shared_base_dir / "eic_thumbnails_shared_y"
    dir_indep = shared_base_dir / "eic_thumbnails_independent_y"
    for d in (dir_shared, dir_indep):
        if overwrite and d.exists():
            logger.info("Overwriting enabled: clearing existing contents of %s", d)
            shutil.rmtree(d)
        elif not overwrite and d.exists():
            logger.info(
                "Overwriting disabled: existing directory %s will be used "
                "(existing PDFs will be preserved).",
                d,
            )
            return
        logger.info("Creating directory %s", d)
        d.mkdir(parents=True, exist_ok=True)

    if summary_obj.experimental_data.curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.experimental_data.curation_df
    ms1_all_df = summary_obj.experimental_data.ms1_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    manual_curation_df = manual_curation_df.reset_index(drop=True)

    if ms1_all_df is not None and not ms1_all_df.empty:
        ms1_groups: dict[str, pd.DataFrame] = {
            key: grp.reset_index(drop=True)
            for key, grp in ms1_all_df.groupby("mz_rt_uid", sort=False)
        }
    else:
        ms1_groups = {}

    n_compounds = len(manual_curation_df)
    logger.info("Building task list for %d compounds...", n_compounds)

    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")
        inchi_key = mc_row.get("inchi_key", "")

        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        safe_stem = (
            f"{cmp_idx_display:04d}_{compound_name}_{adduct}_{inchi_key}"
            .replace("/", "-")
            .replace(" ", "_")
        )
        path_shared = dir_shared / f"{safe_stem}.pdf"
        path_indep = dir_indep / f"{safe_stem}.pdf"

        ms1_cmp = ms1_groups.get(mz_rt_uid)
        if ms1_cmp is None or ms1_cmp.empty:
            file_items = [{"filename": "", "spec_rts": [], "spec_ints": [], "group_name": "unknown"}]
        else:
            file_items = []
            for _, file_row in ms1_cmp.iterrows():
                f_path = str(file_row.get("filename", ""))
                rts = file_row.get("spec_rts", [])
                ints = file_row.get("spec_ints", [])
                # Coerce to plain lists in case the stored value is a numpy array
                if hasattr(rts, "tolist"):
                    rts = rts.tolist()
                if hasattr(ints, "tolist"):
                    ints = ints.tolist()
                file_items.append({
                    "filename": f_path,
                    "spec_rts": rts,
                    "spec_ints": ints,
                    "group_name": fpf.get_file_parts(f_path, "sample_name"),
                })

            def _alphanum_key(text: str) -> list:
                return [
                    int(tok) if tok.isdigit() else tok.lower()
                    for tok in re.split(r"(\d+)", text or "")
                ]

            file_items.sort(
                key=lambda item: (
                    _alphanum_key(item.get("group_name", "")),
                    _alphanum_key(Path(item.get("filename", "")).stem),
                )
            )

        tasks.append({
            "mc_row_dict": mc_row.to_dict(),
            "compound_name": compound_name,
            "adduct": adduct,
            "rt_alignment_num": rt_alignment_num,
            "analysis_num": analysis_num,
            "compound_idx": cmp_idx,
            "path_shared": str(path_shared),
            "path_indep": str(path_indep),
            "file_items": file_items,
        })

    if not tasks:
        logger.info("Nothing to generate (all PDFs already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(
        "Generating EIC thumbnail PDFs for %d compounds using %d workers...",
        len(tasks), n_workers,
    )

    pbar = tqdm(
        total=len(tasks),
        desc="Generating EIC thumbnails",
        unit="compound",
        disable=should_disable_tqdm(),
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_compound_pdf_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Wrote EIC PDFs for %s", name)
            except Exception as exc:
                logger.error("Failed to write PDFs for %s: %s", name, exc)
            finally:
                pbar.update(1)

    pbar.close()

def _compound_pdf_worker(kwargs: dict) -> str:
    """Worker: write shared-y and independent-y PDFs for one compound."""

    _GRID_COLS = 5
    _GRID_ROWS = 5
    _PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS

    path_shared = Path(kwargs["path_shared"])
    path_indep = Path(kwargs["path_indep"])

    mc_row = pd.Series(kwargs["mc_row_dict"])
    file_items = kwargs["file_items"]
    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    rt_alignment_num = kwargs["rt_alignment_num"]
    analysis_num = kwargs["analysis_num"]
    compound_idx = kwargs["compound_idx"]

    rt_min = mc_row.get("rt_min", np.nan)
    rt_max_val = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    all_intensities = [
        v
        for item in file_items
        for v in _as_list(item.get("spec_ints"))
        if v is not None and not np.isnan(v)
    ]
    y_max_global = float(max(all_intensities)) if all_intensities else None

    total_files = len(file_items)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)
    title_base = (
        f"[{_display_compound_idx(compound_idx):04d}] {_strip_non_chars(compound_name)} | "
        f"{_strip_non_chars(adduct)}  (RT alignment {rt_alignment_num}, analysis {analysis_num})"
    )

    group_bg_palette = ["#F3F8FF", "#EEFCF1", "#FFF8EA", "#FFF1F1", "#F5F0FF"]
    group_to_color: dict[str, str] = {}
    for item in file_items:
        group_name = item.get("group_name")
        if group_name not in group_to_color:
            group_to_color[group_name] = group_bg_palette[len(group_to_color) % len(group_bg_palette)]

    pdf_shared = PdfPages(path_shared)
    pdf_indep = PdfPages(path_indep)
    try:
        for page_idx in range(total_pages):
            start = page_idx * _PLOTS_PER_PAGE
            end = min(start + _PLOTS_PER_PAGE, total_files)
            page_items = file_items[start:end]
            n_on_page = len(page_items)
            page_label = f"({page_idx + 1}/{total_pages})" if total_pages > 1 else ""

            fig, axes = plt.subplots(_GRID_ROWS, _GRID_COLS, figsize=(20, 16))
            fig.subplots_adjust(
                left=0.05, right=0.98, top=0.92, bottom=0.06, hspace=0.45, wspace=0.3
            )
            axes_flat = axes.flatten()

            active_axes = []
            for slot_idx, item in enumerate(page_items):
                ax = axes_flat[slot_idx]
                fname_s = _strip_non_chars(_short_fname(item["filename"]))
                spec_ints = _as_list(item.get("spec_ints"))
                valid_ints = [v for v in spec_ints if v is not None and not np.isnan(v)]
                y_max_indep = float(max(valid_ints)) if valid_ints else None
                group_name = item.get("group_name")
                _render_eic_thumbnail(
                    ax,
                    _as_list(item.get("spec_rts")),
                    spec_ints,
                    rt_min,
                    rt_peak,
                    rt_max_val,
                    fname_s,
                    y_max_indep,
                    bg_color=group_to_color.get(group_name),
                )
                active_axes.append(ax)

            for slot_idx in range(n_on_page, _PLOTS_PER_PAGE):
                axes_flat[slot_idx].set_visible(False)

            fig.suptitle(f"{title_base}  {page_label}\n", fontsize=12, y=1.005)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Glyph \d+ .* missing from font",
                    category=UserWarning,
                )
                pdf_indep.savefig(fig, bbox_inches="tight")
                if y_max_global is not None and y_max_global > 0:
                    for ax in active_axes:
                        ax.set_ylim(bottom=0, top=y_max_global * 1.05)
                pdf_shared.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    finally:
        pdf_shared.close()
        pdf_indep.close()

    return kwargs["compound_name"]

def _short_fname(filename: str) -> str:
    """Return an abbreviated label: the 13th and 16th underscore-separated parts of the stem.

    Falls back gracefully when the filename has fewer segments than expected.
    """
    if not filename:
        return "no data"
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) > 15:
        return f"{parts[12]}_{parts[15]}"
    if len(parts) > 12:
        return parts[12]
    return stem

def _render_eic_thumbnail(
    ax,
    spec_rts: List,
    spec_ints: List,
    rt_min: float,
    rt_peak: float,
    rt_max: float,
    fname_short: str,
    y_max: Optional[float],
    bg_color: Optional[str] = None,
) -> None:
    """Draw one EIC thumbnail onto *ax*.

    Parameters
    ----------
    y_max:
        When given, fixes the y-axis upper limit (shared-scale mode).
        When *None*, the axis auto-scales to the data (independent mode).
    bg_color:
        Optional background fill color for the subplot to visually group files.
    """

    if bg_color:
        ax.set_facecolor(bg_color)

    if spec_rts and spec_ints:
        ax.plot(spec_rts, spec_ints, color="steelblue", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # RT vlines: rt_min = red dashed, rt_peak = black dotted, rt_max = black dashed
    _vline = lambda val, color, ls: (
        ax.axvline(val, color=color, linewidth=0.8, linestyle=ls)
        if val is not None and not np.isnan(val)
        else None
    )
    _vline(rt_min, "red", "--")
    _vline(rt_peak, "black", ":" )
    _vline(rt_max, "black", "--")

    if y_max is not None and y_max > 0:
        ax.set_ylim(bottom=0, top=y_max * 1.05)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax.tick_params(axis="both", labelsize=10, length=2, pad=1)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    
    ax.set_title(fname_short, fontsize=10, pad=2.5, loc="center")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

###############################################
#### Compound Boxplots
###############################################

def _plot_compound_boxplot(
    ax,
    compound_metrics: pd.DataFrame,
    metric: str,
    log_scale: bool,
    atlas_ref: Optional[float],
    compound_name: str,
    adduct: str,
    ylabel: str,
    compound_idx: int = 0,
) -> None:
    """Draw a grouped boxplot for one compound onto *ax*.

    Parameters
    ----------
    compound_metrics:
        Rows from ``per_file_df`` already filtered to one (inchi_key, adduct).
    metric:
        Column name to plot: ``"peak_height"``, ``"rt_peak"``, or ``"mz_centroid"``.
    log_scale:
        When *True* the y-values are log₁₀-transformed before plotting.
    atlas_ref:
        Atlas reference value to draw as a red dashed horizontal line
        (pass *None* to omit the line).
    compound_name, adduct:
        Used as the subplot title.
    ylabel:
        Y-axis label string.
    """
    rng = np.random.default_rng(seed=42)
    groups = sorted(compound_metrics["file_group"].dropna().unique()) if not compound_metrics.empty else []
    data_per_group: List[List[float]] = []
    valid_groups: List[str] = []
    for g in groups:
        group_rows = compound_metrics.loc[compound_metrics["file_group"] == g]
        vals = group_rows[metric].dropna().tolist()
        if log_scale:
            vals = [np.log10(v) for v in vals if v is not None and v > 0]
        n_files = len(group_rows)
        data_per_group.append(vals if vals else [0.0] * max(n_files, 1))
        valid_groups.append(g)

    if not data_per_group:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        title_top = f"{_display_compound_idx(compound_idx):04d}  {compound_name}  {adduct}"
        if metric == "mz_centroid":
            title_bottom = f"m/z centroid: {atlas_ref if atlas_ref is not None and not np.isnan(atlas_ref) else 'N/A'}"
            ax.set_title(f"{title_top}\n{title_bottom}", fontsize=13, pad=2, loc="center", fontweight="bold")
        else:
            ax.set_title(title_top, fontsize=13, pad=2, loc="center", fontweight="bold")
        
        return

    positions = list(range(len(valid_groups)))
    ax.boxplot(
        data_per_group,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", alpha=0.7),
        medianprops=dict(color="navy", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        showfliers=False,
    )

    for pos, vals in zip(positions, data_per_group):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.array([pos] * len(vals), dtype=float) + jitter,
            vals,
            s=8, color="steelblue", alpha=0.65, zorder=3,
        )

    if atlas_ref is not None and not np.isnan(atlas_ref):
        ref_val = np.log10(atlas_ref) if (log_scale and atlas_ref > 0) else atlas_ref
        ax.axhline(ref_val, color="red", linestyle="--", linewidth=0.8, alpha=0.8,
                   label=f"Atlas {atlas_ref:.5g}")
        ax.legend(fontsize=10, loc="upper right", framealpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(valid_groups, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Title formatting: large top line, smaller below
    title_top = f"{_display_compound_idx(compound_idx):04d}  {compound_name}  {adduct}"
    if metric == "mz_centroid":
        mz_val = atlas_ref if atlas_ref is not None and not np.isnan(atlas_ref) else 'N/A'
        title_bottom = f"m/z centroid: {mz_val}"
        ax.set_title(f"{title_top}\n{title_bottom}", fontsize=13, pad=2, loc="center", fontweight="bold")
    else:
        ax.set_title(title_top, fontsize=13, pad=2, loc="center", fontweight="bold")
    

    # For linear scales, remove scientific notation multiplier and offset
    if not log_scale:
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

def _boxplot_compound_worker(kwargs: dict) -> str:
    """Worker: generate all metric boxplot PDFs for one compound."""

    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    inchi_key = kwargs["inchi_key"]
    cmp_idx = kwargs["compound_idx"]
    atlas_ref_dict = kwargs["atlas_ref_dict"]
    metric_configs = kwargs["metric_configs"]
    cmp_metrics = kwargs["cmp_metrics"]

    safe_stem = f"{_display_compound_idx(cmp_idx):04d}_{compound_name}_{adduct}_{inchi_key}".replace("/", "-").replace(" ", "_")
    for metric, log_scale, ylabel, atlas_attr, metric_dir_str in metric_configs:
        pdf_path = Path(metric_dir_str) / f"{safe_stem}.pdf"
        atlas_ref = atlas_ref_dict.get(atlas_attr) if atlas_attr else None
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(bottom=0.2, left=0.12, right=0.97, top=0.88)
        try:
            _plot_compound_boxplot(
                ax, cmp_metrics, metric, log_scale,
                atlas_ref, compound_name, adduct, ylabel,
                compound_idx=cmp_idx,
            )
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
        finally:
            plt.close(fig)

    return compound_name

def make_boxplots(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / "boxplots"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info("Overwriting disabled: existing directory %s will be used (existing PDFs will be preserved).", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting boxplot figures to %s", output_dir)

    manual_curation_df = summary_obj.experimental_data.curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning("per_file_metrics_df is None or empty — boxplots will show no data.")
        return

    # Build atlas lookup keyed by mz_rt_uid (the true unique compound identifier)
    atlas_lookup: Dict[str, Dict[str, float]] = {
        mc.get("mz_rt_uid", ""): {
            "atlas_mz": float(mc.get("atlas_mz", np.nan)),
            "atlas_rt_peak": float(mc.get("atlas_rt_peak", np.nan)),
        }
        for _, mc in manual_curation_df.iterrows()
    }

    # Create metric output dirs upfront, store paths as strings
    _METRIC_CONFIGS = [
        ("peak_height", False, "Peak Height (intensity)", None),
        ("peak_height", True,  "Peak Height (log₁₀)", None),
        ("rt_peak", False, "RT Peak (min)", "atlas_rt_peak"),
        ("mz_centroid", False, "m/z Centroid", "atlas_mz"),
    ]
    metric_configs_with_dirs: list[tuple] = []
    for metric, log_scale, ylabel, atlas_attr in _METRIC_CONFIGS:
        metric_dir = output_dir / f"{metric}_{'log' if log_scale else 'linear'}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_configs_with_dirs.append((metric, log_scale, ylabel, atlas_attr, str(metric_dir)))

    pf_groups: dict = {
        key: grp.reset_index(drop=True)
        for key, grp in per_file_df.groupby("mz_rt_uid", sort=False)
    }

    mc_uids = set(manual_curation_df["mz_rt_uid"].dropna().unique())
    pf_uids = set(pf_groups.keys())
    matched = mc_uids & pf_uids

    if mc_uids and not matched:
        logger.warning(
            "No mz_rt_uid overlap between curation_df and per_file_metrics_df! "
            "Sample curation UIDs: %s | Sample per_file UIDs: %s",
            list(mc_uids)[:3], list(pf_uids)[:3],
        )

    # One task per compound
    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")
        inchi_key = mc_row.get("inchi_key", "")
        mz_rt_uid = mc_row.get("mz_rt_uid", "")

        tasks.append({
            "compound_name": compound_name,
            "adduct": adduct,
            "inchi_key": inchi_key,
            "compound_idx": cmp_idx,
            "atlas_ref_dict": atlas_lookup.get(mz_rt_uid, {}),
            "metric_configs": metric_configs_with_dirs,
            "cmp_metrics": pf_groups.get(mz_rt_uid, pd.DataFrame()),
        })

    if not tasks:
        logger.info("Nothing to generate (all boxplot PDFs already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(
        "Generating boxplot PDFs for %d compounds (%d metric types) using %d workers...",
        len(tasks), len(_METRIC_CONFIGS), n_workers,
    )

    pbar = tqdm(total=len(tasks), desc="Generating boxplots", unit="compound", disable=should_disable_tqdm())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_boxplot_compound_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Wrote boxplot PDFs for %s", name)
            except Exception as exc:
                logger.error("Failed to generate boxplots for %s: %s", name, exc)
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)

    pbar.close()

    logger.info("Boxplot PDFs complete")

###############################################
#### Table Exporters
###############################################

def make_manual_curation_csv(
    summary_obj: "AnalysisSummary",
    output_filename: str = "manually_curated_compound_data.csv",
    overwrite: bool = True,
) -> None:
    """Write the ``manual_curation`` table to a CSV file (one row per compound).

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
        ``manual_curation_df`` is loaded automatically if not already cached.
    output_filename:
        CSV filename (the ``.csv`` extension is appended automatically if absent).
    overwrite:
        When *False*, skips writing if the output file already exists.

    Returns
    -------
    pd.DataFrame
        The exported DataFrame (empty on error).
    """

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / "data_sheets"
    output_file = output_dir / output_filename
    if not overwrite and output_file.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_file)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting manual curation CSV to %s", output_file)

    manual_curation_df = summary_obj.experimental_data.curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - CSV not written.")
        return

    manual_curation_df.to_csv(output_file, index=False)
    logger.info("Exported manual curation CSV")
    return

def make_best_ms2_hit_fragment_ions_csv(
    summary_obj: "AnalysisSummary",
    output_filename: str = "best_ms2_hit_fragment_ions.csv",
    overwrite: bool = True,
    min_fragment_intensity: Optional[float] = 1e4,
) -> None:

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / "data_sheets"
    output_file = output_dir / output_filename
    if not overwrite and output_file.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_file)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting best MS2 hit fragment ions CSV to %s", output_file)

    if summary_obj.experimental_data.curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.experimental_data.curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - best MS2 hit CSV not written.")
        return

    ms2_df = summary_obj.experimental_data.ms2_df
    ms2_best: dict[str, dict] = {}
    if ms2_df is not None and not ms2_df.empty:
        for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
            best_score = -np.inf
            best_hit_data: Optional[dict] = None
            for _, scan_row in grp.iterrows():
                hits = _as_list(scan_row.get("hits"))
                if not hits:
                    continue
                # hits are pre-sorted best-first; index 0 is the top hit
                top_score = float(hits[0].get("score", -np.inf))
                if top_score > best_score:
                    best_score = top_score
                    best_hit_data = {
                        **hits[0],
                        "filename": scan_row.get("filename", ""),
                        "scan_rt":  float(scan_row.get("scan_rt", np.nan)),
                    }
            if best_hit_data is not None:
                ms2_best[uid] = best_hit_data

    rows: list[dict] = []
    for cmp_idx, mc_row in tqdm(
        manual_curation_df.reset_index(drop=True).iterrows(),
        total=len(manual_curation_df),
        desc="Finding best MS2 hits for compounds",
        disable=should_disable_tqdm(),
    ):
        cmp_idx_display = _display_compound_idx(cmp_idx)
        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")

        best_hit = ms2_best.get(mz_rt_uid)
        if best_hit is None:
            continue

        q_mz_all, q_int_all = np.array(_as_list(best_hit["query_aligned"][0]), dtype=float), np.array(_as_list(best_hit["query_aligned"][1]), dtype=float)

        if q_mz_all.size > 0:
            valid     = ~np.isnan(q_mz_all)
            mz_vals   = q_mz_all[valid].tolist()
            int_vals  = q_int_all[valid].tolist()
        else:
            mz_vals, int_vals = [], []

        if min_fragment_intensity is not None:
            filtered = [
                (mz, i) for mz, i in zip(mz_vals, int_vals)
                if i > min_fragment_intensity
            ]
            if filtered:
                fmz, fint = zip(*filtered)
                mz_vals, int_vals = list(fmz), list(fint)
            else:
                mz_vals, int_vals = [], []

        raw_spectrum_output = json.dumps([mz_vals, int_vals])

        rows.append({
            "compound_index": cmp_idx_display,
            "compound_name":  compound_name,
            "adduct":         adduct,
            "filename":       os.path.basename(str(best_hit.get("filename", ""))),
            "rt_peak":        best_hit.get("scan_rt", np.nan),
            "mz_peak":        best_hit.get("mz_measured", np.nan),
            "spectrum":       raw_spectrum_output,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.warning("No MS2 hits found for any compound - best MS2 hit CSV not written.")
        return

    result_df.to_csv(output_file, index=False)
    logger.info(f"Exported best MS2 hit fragment ions CSV ({len(result_df)} compounds)")


def make_data_sheets(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Export per-compound, per-file quantitative metric tables as wide-format CSVs."""

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / "data_sheets"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info(
            "Overwriting disabled: existing directory %s will be used "
            "(existing CSVs will be preserved).",
            output_dir,
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting data sheets to %s", output_dir)

    manual_curation_df = summary_obj.experimental_data.curation_df
    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - data sheets not written.")
        return

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning("per_file_metrics_df is empty after build - data sheets not written.")
        return

    mc_reset = manual_curation_df.reset_index(drop=True)
    mc_slim = mc_reset[["mz_rt_uid", "compound_name"]].copy()
    mc_slim["compound_index"] = mc_reset.index + 1
    mc_slim = mc_slim.drop_duplicates(subset=["mz_rt_uid"])
    pfm = per_file_df.merge(mc_slim, on="mz_rt_uid", how="left")

    pfm["_file_col"] = pfm["filename"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0] if p else "unknown"
    )

    _DATA_SHEET_METRICS = [
        "peak_height",
        "peak_area",
        "rt_peak",
        "rt_centroid",
        "mz_peak",
        "mz_centroid",
    ]
    # Ordered metadata columns: compound_index, mz_rt_uid, compound_name, inchi_key, adduct
    _INDEX_COLS = [
        "compound_index",
        "mz_rt_uid",
        "compound_name",
        "inchi_key",
        "adduct",
    ]

    for metric in _DATA_SHEET_METRICS:
        if metric not in pfm.columns:
            logger.warning(
                "Metric '%s' not found in per_file_metrics_df — skipping.", metric
            )
            continue

        csv_path = output_dir / f"{metric}.csv"

        # Include only index cols that are actually present
        present_idx = [c for c in _INDEX_COLS if c in pfm.columns]
        wide = (
            pfm[present_idx + ["_file_col", metric]]
            .pivot_table(
                index=present_idx,
                columns="_file_col",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        wide.to_csv(csv_path, index=False)
        logger.info(f"Exported {metric} data sheet ({len(wide)} compounds x {len(wide.columns) - len(present_idx)} files)")

    logger.info("Data sheets written to %s", output_dir)

def make_peak_height_filtered_csv(
    obj: "AnalysisSummary",
    overwrite: bool = True,
    control_fold_threshold: float = 3.0,
) -> None:
    """Filter and process the ``peak_height`` data sheet.

    Reads ``peak_height.csv`` (produced by :func:`make_data_sheets`) and writes
    ``peak_height_filtered.csv`` with the following column layout::

        control_filter, compound_index, mz_rt_uid, compound_name, inchi_key,
        adduct, <sample_file_1>, <sample_file_2>, ...

    Rows with the same ``inchi_key`` and ``compound_name`` (e.g. different
    adducts) are collapsed to a single row by taking the element-wise maximum
    across sample columns.  The adduct that produced the highest peak across
    non-control samples is recorded in the ``adduct`` column.
    """

    out_dir = Path(obj.paths['analysis_results_output_dir']) / "data_sheets"
    source_csv = out_dir / "peak_height.csv"
    output_csv = out_dir / "peak_height_filtered.csv"
    if not source_csv.exists():
        logger.error(
            "peak_height.csv not found at %s — run make_data_sheets first.", source_csv
        )
        return
    if not overwrite and output_csv.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_csv)
        return

    logger.info("Creating filtered peak height CSV at %s", output_csv)

    df = pd.read_csv(source_csv)

    # ── Identify metadata vs sample columns ──────────────────────────────────
    _META_COLS = {"compound_index", "mz_rt_uid", "compound_name", "inchi_key", "adduct"}
    data_cols = [c for c in df.columns if c not in _META_COLS]

    # ── 1. Control-signal filter ──────────────────────────────────────────────
    _CTRL_PATTERNS = ("ExCtrl", "InjBL")
    ctrl_cols     = [c for c in data_cols if any(p in c for p in _CTRL_PATTERNS)]
    non_ctrl_cols = [c for c in data_cols if c not in ctrl_cols]

    logger.info(
        "Control filter: %d control columns, %d non-control columns.",
        len(ctrl_cols), len(non_ctrl_cols),
    )

    if not ctrl_cols:
        logger.info("No control columns found — marking all rows as 'keep'.")
        df["control_filter"] = "keep"
    else:
        max_ctrl     = df[ctrl_cols].max(axis=1)
        max_non_ctrl = df[non_ctrl_cols].max(axis=1) if non_ctrl_cols else pd.Series(
            np.nan, index=df.index
        )
        keep_mask = (
            max_non_ctrl.notna()
            & (max_ctrl.isna() | (max_non_ctrl >= control_fold_threshold * max_ctrl.fillna(0)))
        )
        df["control_filter"] = keep_mask.map({True: "keep", False: "remove"})

    n_keep = (df["control_filter"] == "keep").sum()
    logger.info(
        "Control filter: %d keep, %d remove (of %d total).",
        n_keep, len(df) - n_keep, len(df),
    )

    if df.empty:
        logger.warning("No compounds found — no output written.")
        return

    # ── 2. Collapse duplicate (inchi_key, compound_name) rows ────────────────
    # When multiple adducts exist for the same compound, keep the adduct that
    # produced the highest peak across non-control sample columns.
    selector_cols = non_ctrl_cols if non_ctrl_cols else data_cols
    group_keys    = ["inchi_key", "compound_name", "control_filter"]

    # Determine the best adduct per (inchi_key, compound_name, control_filter)
    chosen_adduct_map: Dict[tuple, str] = {}
    if "adduct" in df.columns and selector_cols:
        sel_score = df[selector_cols].max(axis=1, skipna=True).fillna(-np.inf)
        tmp = df[group_keys + ["adduct"]].copy()
        tmp["_score"] = sel_score.values
        tmp["_order"] = np.arange(len(tmp))
        tmp = tmp.sort_values(["_score", "_order"], ascending=[False, True])
        best = tmp.drop_duplicates(subset=group_keys, keep="first")
        for _, row in best.iterrows():
            key = (row["inchi_key"], row["compound_name"], row["control_filter"])
            chosen_adduct_map[key] = row["adduct"]

    # Preserve compound_index and mz_rt_uid for the best-scoring row per group
    chosen_index_map: Dict[tuple, int] = {}
    chosen_uid_map:   Dict[tuple, str] = {}
    if selector_cols:
        sel_score = df[selector_cols].max(axis=1, skipna=True).fillna(-np.inf)
        tmp2 = df[group_keys].copy()
        if "compound_index" in df.columns:
            tmp2["compound_index"] = df["compound_index"].values
        if "mz_rt_uid" in df.columns:
            tmp2["mz_rt_uid"] = df["mz_rt_uid"].values
        tmp2["_score"] = sel_score.values
        tmp2["_order"] = np.arange(len(tmp2))
        tmp2 = tmp2.sort_values(["_score", "_order"], ascending=[False, True])
        best2 = tmp2.drop_duplicates(subset=group_keys, keep="first")
        for _, row in best2.iterrows():
            key = (row["inchi_key"], row["compound_name"], row["control_filter"])
            if "compound_index" in best2.columns:
                chosen_index_map[key] = row["compound_index"]
            if "mz_rt_uid" in best2.columns:
                chosen_uid_map[key] = row["mz_rt_uid"]

    # Drop per-row metadata before groupby-max so only numeric data is aggregated
    drop_before_agg = [c for c in ("compound_index", "mz_rt_uid", "adduct") if c in df.columns]
    df_agg = df.drop(columns=drop_before_agg)

    n_before = len(df_agg)
    df_agg = df_agg.groupby(group_keys, sort=False).max().reset_index()

    # Re-attach chosen metadata columns
    def _lookup(row, mapping, default=""):
        return mapping.get((row["inchi_key"], row["compound_name"], row["control_filter"]), default)

    df_agg["adduct"]         = df_agg.apply(lambda r: _lookup(r, chosen_adduct_map), axis=1)
    df_agg["compound_index"] = df_agg.apply(lambda r: _lookup(r, chosen_index_map, np.nan), axis=1)
    df_agg["mz_rt_uid"]      = df_agg.apply(lambda r: _lookup(r, chosen_uid_map), axis=1)

    # ── 3. Impute NaN sample cells with the global matrix minimum ────────────
    _META_FIXED = {"control_filter", "compound_index", "mz_rt_uid", "compound_name", "inchi_key", "adduct"}
    data_cols_final = [c for c in df_agg.columns if c not in _META_FIXED]
    global_min = float(df_agg[data_cols_final].min().min()) if data_cols_final else np.nan
    if not np.isnan(global_min):
        n_nan = int(df_agg[data_cols_final].isna().sum().sum())
        df_agg[data_cols_final] = df_agg[data_cols_final].fillna(global_min)
        logger.info("Imputed %d NaN cells with global minimum: %g", n_nan, global_min)
    else:
        logger.warning("Global minimum is NaN — no imputation performed.")

    # ── 4. Reorder columns: control_filter, compound_index, mz_rt_uid,
    #       compound_name, inchi_key, adduct, ...sample data... ───────────────
    ordered_meta = ["control_filter", "compound_index", "mz_rt_uid", "compound_name", "inchi_key", "adduct"]
    ordered_meta = [c for c in ordered_meta if c in df_agg.columns]
    other_cols   = [c for c in df_agg.columns if c not in set(ordered_meta)]
    df_agg = df_agg[ordered_meta + other_cols]

    # Export
    df_agg.to_csv(output_csv, index=False)
    logger.info(f"Exported filtered peak height ({len(df_agg)} compounds x {len(df_agg.columns) - len(ordered_meta)} files)")


def make_log_fold_changes_csv(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    use_filter: bool = True,
) -> None:
    """Create group-level log2 fold-change CSV from peak_height_filtered.csv.

    Sample columns are grouped by the 13th underscore-separated field of the
    column name (index 12), which corresponds to the ``sample_name`` field
    defined in :data:`metatlas2.file_and_project_format.FILE_PATTERN`.
    Within each group the mean of all replicate values is computed, and
    pairwise log2 fold-changes are then calculated between every pair of
    groups.

    Output column layout::

        compound_index, mz_rt_uid, compound_name, inchi_key, adduct,
        <GroupA>_vs_<GroupB>, ...

    When *use_filter* is ``True`` (default), only rows where
    ``control_filter == 'keep'`` are included (the column itself is not
    written to the output).

    This output is independent of metabomap and does not require both
    polarities to be present.
    """
    out_dir = Path(summary_obj.paths["analysis_results_output_dir"]) / "data_sheets"
    source_csv = out_dir / "peak_height_filtered.csv"
    output_csv = out_dir / "log_fold_changes.csv"

    if not source_csv.exists():
        logger.error(
            "peak_height_filtered.csv not found at %s — run make_peak_height_filtered_csv first.",
            source_csv,
        )
        return
    if not overwrite and output_csv.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_csv)
        return

    logger.info("Creating group-level log fold changes CSV at %s", output_csv)
    df = pd.read_csv(source_csv)
    if use_filter and "control_filter" in df.columns:
        df = df[df["control_filter"] == "keep"].reset_index(drop=True)

    # Metadata columns — same as peak_height_filtered.csv but without control_filter.
    # Preserve ordered output: compound_index, mz_rt_uid, compound_name, inchi_key, adduct
    _ID_COL_NAMES = {"compound_index", "mz_rt_uid", "compound_name", "inchi_key", "adduct"}
    _ID_COL_ORDER = ["compound_index", "mz_rt_uid", "compound_name", "inchi_key", "adduct"]
    id_cols   = [c for c in _ID_COL_ORDER if c in df.columns]
    data_cols = [c for c in df.columns if c not in _ID_COL_NAMES and c != "control_filter"]

    if not data_cols:
        logger.warning("No numeric sample columns found in %s; writing identifiers only.", source_csv)
        df[id_cols].to_csv(output_csv, sep=",", index=False)
        return

    data_num = df[data_cols].apply(pd.to_numeric, errors="coerce")

    # ── Group replicate columns by sample_name (13th underscore-split field) ──
    group_to_cols: Dict[str, List[str]] = {}
    ungrouped_cols: List[str] = []
    for col in data_num.columns:
        grp = fpf.get_file_parts(col, "sample_name")
        if grp is not None:
            group_to_cols.setdefault(grp, []).append(col)
        else:
            ungrouped_cols.append(col)

    if ungrouped_cols:
        logger.warning(
            "%d sample column(s) could not be assigned to a group (fewer than 13 "
            "underscore-separated parts) and will be skipped: %s",
            len(ungrouped_cols),
            ungrouped_cols[:10],
        )

    if len(group_to_cols) < 2:
        logger.warning(
            "Fewer than 2 sample groups identified in %s "
            "(groups found: %s) — no fold-change comparisons possible.",
            source_csv,
            list(group_to_cols.keys()),
        )
        df[id_cols].to_csv(output_csv, sep=",", index=False)
        return

    logger.info(
        "Identified %d sample groups: %s",
        len(group_to_cols),
        list(group_to_cols.keys()),
    )
    for grp, cols in group_to_cols.items():
        logger.info("  Group '%s': %d replicate(s) — %s", grp, len(cols), cols)

    # ── Compute per-group mean across replicates ──────────────────────────────
    group_means: Dict[str, np.ndarray] = {}
    for grp, cols in group_to_cols.items():
        group_means[grp] = data_num[cols].mean(axis=1).to_numpy(dtype=float)

    # ── Build output: id columns + pairwise LFC columns ──────────────────────
    lfc_records: Dict[str, object] = {col: df[col].to_numpy() for col in id_cols}

    n_comparisons = 0
    for g1, g2 in itertools.combinations(group_means.keys(), 2):
        m1 = group_means[g1]
        m2 = group_means[g2]
        valid = (m1 > 0) & (m2 > 0) & np.isfinite(m1) & np.isfinite(m2)
        res = np.full(len(m1), np.nan)
        res[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res
        n_comparisons += 1

    pd.DataFrame(lfc_records).to_csv(output_csv, sep=",", index=False)
    logger.info(f"Exported group-level log fold changes ({len(df)} compounds x {n_comparisons} group comparisons)")

def _get_modelseed_compounds(cache_path: Path) -> pd.DataFrame:
    """Load the ModelSEED compounds table, fetching and caching it if needed.

    Parameters
    ----------
    cache_path:
        Permanent local path to store/load the TSV.
    """

    _MODELSEED_COMPOUNDS_URL = (
        "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
        "master/Biochemistry/compounds.tsv"
    )
    if cache_path.exists():
        logger.info(f"Loading ModelSEED compounds from local cache: {cache_path}")
    else:
        logger.info(f"Fetching ModelSEED compounds table from {_MODELSEED_COMPOUNDS_URL}")
        resp = requests.get(_MODELSEED_COMPOUNDS_URL, timeout=30)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        logger.info(f"Saved ModelSEED compounds cache to {cache_path}")

    return pd.read_csv(cache_path, sep="\t", low_memory=False)


def _build_inchikey_to_cpd(cache_path: Path) -> dict[str, str]:
    """Return a mapping of InChIKey to semicolon-joined ModelSEED CPD IDs.

    Rows with no InChIKey are dropped immediately to reduce memory usage.
    When multiple CPD IDs share the same InChIKey, all are retained and
    joined with ';' (e.g. ``'cpd00001;cpd99999'``).

    Parameters
    ----------
    cache_path:
        Passed through to :func:`_get_modelseed_compounds`.
    """
    df = _get_modelseed_compounds(cache_path)

    if "inchikey" not in df.columns or "id" not in df.columns:
        logger.error(
            "ModelSEED compounds TSV does not contain expected columns "
            "'id' / 'inchikey'. Available: %s", list(df.columns)
        )
        return {}

    df = df[["id", "inchikey"]].dropna(subset=["inchikey"])
    df = df[df["inchikey"].str.strip() != ""]

    mapping = (
        df.groupby("inchikey")["id"]
        .agg(lambda ids: ";".join(ids))
        .to_dict()
    )

    n_multi = sum(1 for v in mapping.values() if ";" in v)
    return mapping

def make_metabomap(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Merge POS and NEG ``peak_height_filtered.csv`` files into a single
    metabomap table and compute group-level log2 fold-changes.

    For each unique ``(inchi_key, compound_name)`` pair the row with the
    highest peak height across non-control sample columns is selected,
    regardless of polarity or adduct.  The chosen adduct and polarity are
    recorded as ``chosen_adduct`` and ``chosen_polarity`` columns.

    Sample columns are then grouped by the 13th underscore-separated field
    (``sample_name``, index 12) and the mean of each group's replicates is
    used to compute pairwise log2 fold-changes.

    Outputs
    -------
    ``<analysis_output_dir>/../metabomap/merged_peak_heights.tsv``
        One row per ``(inchi_key, compound_name)`` with columns:
        ``inchi_key, cpd_id, compound_name, chosen_adduct, chosen_polarity,
        <sample_file_1>, ...``

    ``<analysis_output_dir>/../metabomap/log_fold_changes.tsv``
        One row per ``(inchi_key, compound_name)`` with columns:
        ``inchi_key, cpd_id, compound_name, <GroupA>_vs_<GroupB>, ...``
    """
    # ── 1. Setup paths ────────────────────────────────────────────────────────
    analysis_output_dir = Path(summary_obj.paths.get("analysis_results_output_dir"))
    curr_pol = summary_obj.polarity.upper()
    sib_pol  = "NEG" if curr_pol == "POS" else "POS"

    current_csv = analysis_output_dir / "data_sheets" / "peak_height_filtered.csv"
    sibling_csv = (
        Path(str(analysis_output_dir).replace(f"-{curr_pol}-", f"-{sib_pol}-"))
        / "data_sheets" / "peak_height_filtered.csv"
    )

    metabomaps_dir = analysis_output_dir / "../metabomap"
    merged_tsv = metabomaps_dir / "merged_peak_heights.tsv"
    lfc_tsv    = metabomaps_dir / "log_fold_changes.tsv"

    if not overwrite and merged_tsv.exists() and lfc_tsv.exists():
        return

    if not current_csv.exists() or not sibling_csv.exists():
        logger.warning("Missing polarity CSVs; skipping metabomap.")
        return

    metabomaps_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Load both polarity CSVs ────────────────────────────────────────────
    _META_COLS = {"control_filter", "compound_index", "mz_rt_uid",
                  "compound_name", "inchi_key", "adduct"}
    _EXCLUDE_GROUPS = ("QC", "ISTD")

    def _load_filtered(path: Path, polarity_label: str) -> pd.DataFrame:
        """Load one peak_height_filtered.csv, tag with polarity, keep 'keep' rows."""
        raw = pd.read_csv(path)
        if "control_filter" in raw.columns:
            raw = raw[raw["control_filter"] == "keep"].reset_index(drop=True)
        raw["_polarity"] = polarity_label
        return raw

    cur_df = _load_filtered(current_csv, curr_pol)
    sib_df = _load_filtered(sibling_csv, sib_pol)

    # Stack both polarities
    combined = pd.concat([cur_df, sib_df], axis=0, ignore_index=True)

    # Identify sample columns (present in either polarity)
    all_sample_cols = [
        c for c in combined.columns
        if c not in _META_COLS and c != "_polarity"
    ]

    # ── 3. Select best row per (inchi_key, compound_name) ────────────────────
    # "Best" = highest max peak height across non-control sample columns.
    # Control columns (ExCtrl, InjBL) are excluded from the selection score.
    _CTRL_PATTERNS = ("ExCtrl", "InjBL")
    non_ctrl_sample_cols = [
        c for c in all_sample_cols
        if not any(p in c for p in _CTRL_PATTERNS)
    ]
    score_cols = non_ctrl_sample_cols if non_ctrl_sample_cols else all_sample_cols

    if score_cols:
        combined["_score"] = combined[score_cols].apply(
            pd.to_numeric, errors="coerce"
        ).max(axis=1, skipna=True).fillna(-np.inf)
    else:
        combined["_score"] = -np.inf

    combined["_order"] = np.arange(len(combined))
    combined_sorted = combined.sort_values(
        ["_score", "_order"], ascending=[False, True]
    )
    best = combined_sorted.drop_duplicates(
        subset=["inchi_key", "compound_name"], keep="first"
    ).reset_index(drop=True)

    logger.info(
        "Metabomap: selected best row for %d unique (inchi_key, compound_name) pairs "
        "from %d combined rows (%s + %s).",
        len(best), len(combined), curr_pol, sib_pol,
    )

    # Record chosen adduct and polarity
    best["chosen_adduct"]   = best["adduct"]   if "adduct"    in best.columns else ""
    best["chosen_polarity"] = best["_polarity"]

    # ── 4. Build the merged peak-heights table ────────────────────────────────
    # Keep only sample columns that are actually present in the best-row selection
    present_sample_cols = [c for c in all_sample_cols if c in best.columns]

    merged_data_df = best[
        ["inchi_key", "compound_name", "chosen_adduct", "chosen_polarity"]
        + present_sample_cols
    ].copy()

    # Attach ModelSEED CPD IDs
    ms_path = Path(summary_obj.paths["modelseed_table_path"])
    try:
        inchikey_to_cpd = _build_inchikey_to_cpd(ms_path)
    except Exception:
        inchikey_to_cpd = {}

    merged_data_df.insert(1, "cpd_id", merged_data_df["inchi_key"].map(inchikey_to_cpd))
    merged_data_df.to_csv(merged_tsv, sep="\t", index=False)
    logger.info(f"Wrote merged_peak_heights.tsv ({len(merged_data_df)} compounds x {len(present_sample_cols)} sample columns)")

    # ── 5. Group sample columns by sample_name (index 12) and compute LFC ────
    _LFC_META = {"inchi_key", "cpd_id", "compound_name", "chosen_adduct", "chosen_polarity"}
    sample_data_cols = [c for c in merged_data_df.columns if c not in _LFC_META]

    group_to_cols: Dict[str, List[str]] = {}
    for col in sample_data_cols:
        grp = fpf.get_file_parts(col, "sample_name")
        if grp is not None and not any(ex in grp for ex in _EXCLUDE_GROUPS):
            group_to_cols.setdefault(grp, []).append(col)

    if len(group_to_cols) < 2:
        logger.warning(
            "Fewer than 2 sample groups found in metabomap (%s) — LFC table not written.",
            list(group_to_cols.keys()),
        )
        return

    logger.info(
        "Metabomap LFC: %d groups — %s",
        len(group_to_cols), list(group_to_cols.keys()),
    )

    data_num = merged_data_df[sample_data_cols].apply(pd.to_numeric, errors="coerce")

    group_means: Dict[str, np.ndarray] = {
        grp: data_num[cols].mean(axis=1).to_numpy(dtype=float)
        for grp, cols in group_to_cols.items()
    }

    lfc_records: Dict[str, object] = {
        col: merged_data_df[col].to_numpy()
        for col in ["inchi_key", "cpd_id", "compound_name"]
    }

    n_comparisons = 0
    for g1, g2 in itertools.combinations(group_means.keys(), 2):
        m1, m2 = group_means[g1], group_means[g2]
        valid = (m1 > 0) & (m2 > 0) & np.isfinite(m1) & np.isfinite(m2)
        res = np.full(len(m1), np.nan)
        res[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res
        n_comparisons += 1

    pd.DataFrame(lfc_records).to_csv(lfc_tsv, sep="\t", index=False)
    logger.info(f"Wrote log_fold_changes.tsv ({len(merged_data_df)} compounds x {n_comparisons} group comparisons)")


###############################################
#### Make parquet files
###############################################

_COMPOUND_FILE_STR_COLS = [
    "mz_rt_uid", "compound_name", "identified_metabolite", "label", "inchi_key", "formula", "smiles",
    "inchi", "pubchem_cid", "iupac_name", "polarity", "adduct", "best_ms1_file",
    "filename", "file_group", "ms1_notes", "ms2_notes",
    "msi_level", "control_filter", "overlapping_compound", "overlapping_inchi_keys",
    "isomer_details", "identification_notes", "analyst_notes", "other_notes",
    "best_ms2_file", "best_ms2_num_ions", "best_ms2_matching_ions",
    "best_ms2_spectrum_rt_mz", "ms1_spectrum_rt_i", "msms_file", "msms_numberofions", "msms_matchingions",
]

_COMPOUND_FILE_FLOAT_COLS = [
    "compound_index",
    "atlas_mz", "atlas_rt_peak", "atlas_rt_min", "atlas_rt_max",
    "rt_min", "rt_max", "exact_mass",
    "best_ms1_rt", "best_ms1_mz", "best_ms1_intensity",
    "best_ms1_ppm_error", "best_ms1_rt_error",
    "best_ms2_rt", "best_ms2_score", "best_ms2_mz",
    "best_ms2_mz_ppm_error", "best_ms2_mz_error_da", "best_ms2_rt_error",
    "peak_height", "peak_area", "rt_peak", "rt_centroid",
    "mz_peak", "mz_centroid", "measured_rt", "measured_mz",
    "mz_theoretical", "mz_measured", "mz_error", "mz_ppmerror",
    "rt_theoretical", "rt_error", "msms_rt",
    "msms_score", "mz_quality", "rt_quality", "msms_quality",
    "total_score",
]

_COMPOUND_LFC_STR_COLS = [
    "mz_rt_uid", "inchi_key", "compound_name", "control_filter",
    "condition_1", "condition_2", "adduct", "formula", "smiles", "inchi",
    "pubchem_cid", "iupac_name", "ms1_notes", "ms2_notes", "msi_level",
    "identification_notes", "analyst_notes", "other_notes",
    "best_ms2_file", "best_ms2_num_ions", "best_ms2_matching_ions", "best_ms2_spectrum_rt_mz",
    "msms_file", "msms_numberofions", "msms_matchingions",
]

_COMPOUND_LFC_FLOAT_COLS = [
    "compound_index",
    "log2_fold_change",
    "rt_min", "rt_max",
    "best_ms1_rt", "best_ms1_mz", "best_ms1_intensity", "best_ms1_ppm_error", "best_ms1_rt_error",
    "atlas_mz", "atlas_rt_peak", "atlas_rt_min", "atlas_rt_max",
    "best_ms2_rt", "best_ms2_score", "best_ms2_mz",
    "best_ms2_mz_ppm_error", "best_ms2_mz_error_da", "best_ms2_rt_error",
    "msms_score", "msms_rt",
    "mz_theoretical", "mz_measured", "mz_error", "mz_ppmerror", "rt_theoretical", "rt_error",
]


def _build_unified_schema_map_df() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    seen: dict[str, dict[str, str]] = {}

    def _add(cols: list[str], dtype: str, row_kind: str) -> None:
        for idx, col in enumerate(cols):
            if col not in seen:
                seen[col] = {
                    "column_name": col,
                    "dtype": dtype,
                    "row_kinds": row_kind,
                    "compound_file_position": "",
                    "compound_lfc_position": "",
                }
            else:
                existing = set(filter(None, seen[col]["row_kinds"].split("|")))
                existing.add(row_kind)
                seen[col]["row_kinds"] = "|".join(sorted(existing))
            seen[col][f"{row_kind}_position"] = str(idx)

    _add(_COMPOUND_FILE_STR_COLS, "string", "compound_file")
    _add(_COMPOUND_FILE_FLOAT_COLS, "float", "compound_file")
    _add(_COMPOUND_LFC_STR_COLS, "string", "compound_lfc")
    _add(_COMPOUND_LFC_FLOAT_COLS, "float", "compound_lfc")

    seen["row_kind"] = {
        "column_name": "row_kind",
        "dtype": "string",
        "row_kinds": "compound_file|compound_lfc",
        "compound_file_position": "",
        "compound_lfc_position": "",
    }

    rows.extend(seen.values())
    rows.sort(key=lambda r: r["column_name"])
    return pd.DataFrame(rows)


def _write_unified_schema_map(partition_dir: Path, data_output_name: str) -> None:
    schema_df = _build_unified_schema_map_df()
    schema_csv = partition_dir / f"{data_output_name}-results.schema_map.csv"
    schema_json = partition_dir / f"{data_output_name}-results.schema_map.json"
    schema_df.to_csv(schema_csv, index=False)
    schema_df.to_json(schema_json, orient="records", indent=2)


def make_analysis_parquet(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Write one unified Parquet file for downstream querying.

    ANALYSIS-DETAILS-results.parquet
        One flat table containing both file-grain compound rows and
        compound-level log-fold-change rows. Rows are tagged with
        ``row_kind`` so queries can filter to either grain.

    The partition directories are still useful because they give cheap
    filtering on chromatography, polarity, analysis type, and analysis name.

    Parameters
    ----------
    summary_obj:
        Configured :class:`AnalysisSummary` object after curation data has
        been loaded.
    overwrite:
        When *False*, skips writing if the unified parquet already exists.
    """

    parquet_output_dir = Path(summary_obj.paths["parquet_output_dir"])
    data_output_name =  f"{summary_obj.project_name}-" \
                        f"{summary_obj.rt_alignment_number}-" \
                        f"{summary_obj.analysis_number}-" \
                        f"{summary_obj.chromatography}-" \
                        f"{summary_obj.polarity}-" \
                        f"{summary_obj.analysis_type}-" \
                        f"{summary_obj.analysis_name}"

    partition_dir = (parquet_output_dir / "parquet_results")
    unified_path = partition_dir / f"{data_output_name}-results.parquet"
    if not overwrite and unified_path.exists():
        logger.info("Overwriting disabled: existing Parquet file in %s will be used.", partition_dir)
        return
    partition_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing analysis Parquet files to %s", partition_dir)

    ## Note: this is good for debugging schema issues, but it will break the parquet querier because it puts csv/json files in the dir
    #_write_unified_schema_map(partition_dir, data_output_name)
    #logger.info("Wrote unified parquet schema map files for %s to %s", data_output_name, partition_dir)

    # Start with analysis-level fields always available from summary_obj
    footer_meta: dict[bytes, bytes] = {
        b"project_name":         str(summary_obj.project_name or "").encode(),
        b"chromatography":       summary_obj.chromatography.encode(),
        b"polarity":             summary_obj.polarity.encode(),
        b"analysis_type":        summary_obj.analysis_type.encode(),
        b"analysis_name":        summary_obj.analysis_name.encode(),
        b"rt_alignment_number":  str(summary_obj.rt_alignment_number or "").encode(),
        b"analysis_number":      str(summary_obj.analysis_number or "").encode(),
        b"created_date":         datetime.date.today().isoformat().encode(),
    }

    # Overlay all fields parsed from the project name
    parsed_meta = _parse_project_metadata(summary_obj.project_name)
    for key, val in parsed_meta.items():
        footer_meta[key.encode()] = val.encode()

    def _attach_meta(table: pa.Table) -> pa.Table:
        existing = table.schema.metadata or {}
        return table.replace_schema_metadata({**existing, **footer_meta})

    # Write settings
    _WRITE_KWARGS = dict(
        compression="zstd",
        compression_level=3,
        write_statistics=True,
        row_group_size=100_000,
        data_page_size=1 * 1024 * 1024,
        use_dictionary=True,
    )

    logger.info("Building unified analysis table...")
    unified_table = _build_unified_analysis_table(summary_obj)

    if unified_table.num_rows > 0:
        unified_table = _attach_meta(unified_table)
        pq.write_table(unified_table, unified_path, **_WRITE_KWARGS)
        logger.info(f"Wrote {unified_path.stem} ({unified_table.num_rows} rows, {unified_table.num_columns} columns)")
    else:
        logger.warning("Unified analysis table is empty — file not written.")

    logger.info("Analysis Parquet export complete")

def _build_unified_analysis_table(
    summary_obj: "AnalysisSummary",
) -> pa.Table:
    """Build one flat table containing both per-file and LFC rows.

    The output uses a shared schema with a ``row_kind`` column so root-level
    parquet scans can load the whole analysis without choosing between the
    former ``compound_per_file`` and ``compound_lfc`` files. Partition
    directories are retained for cheap pruning on chromatography, polarity,
    analysis type, and analysis name.

    Returns
    -------
    pa.Table
    """
    per_file_table = _build_compound_per_file_table(summary_obj)
    lfc_table = _build_compound_lfc_table(summary_obj)

    frames: list[pd.DataFrame] = []
    if per_file_table.num_rows > 0:
        per_file_df = per_file_table.to_pandas()
        per_file_df["row_kind"] = "compound_file"
        frames.append(per_file_df)
    else:
        logger.warning("compound_per_file table is empty — file-grain rows will be omitted.")

    if lfc_table.num_rows > 0:
        lfc_df = lfc_table.to_pandas()
        lfc_df["row_kind"] = "compound_lfc"
        frames.append(lfc_df)
    else:
        logger.warning("compound_lfc table is empty — lfc rows will be omitted.")

    if not frames:
        return pa.table({})

    unified_cols = list(dict.fromkeys(
        _COMPOUND_FILE_STR_COLS
        + _COMPOUND_FILE_FLOAT_COLS
        + _COMPOUND_LFC_STR_COLS
        + _COMPOUND_LFC_FLOAT_COLS
        + ["row_kind"]
    ))
    unified_df = pd.concat(
        [df.reindex(columns=unified_cols) for df in frames],
        ignore_index=True,
        sort=False,
    )
    return pa.Table.from_pandas(unified_df, preserve_index=False)


def _build_best_ms2_summary_df(summary_obj: "AnalysisSummary") -> pd.DataFrame:
    """Return one-row-per-compound best-MS2 metrics for parquet export.

    Includes best hit score metadata and a compact spectrum payload encoded as
    JSON ``[[rt_list], [mz_list]]`` where ``rt_list`` repeats the scan RT for
    each matched m/z from the best hit query-aligned spectrum.
    """
    ms2_df = summary_obj.experimental_data.ms2_df
    if ms2_df is None or ms2_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
        best_score = -np.inf
        best_scan_row = None
        best_hit = None

        for _, scan_row in grp.iterrows():
            hits = _as_list(scan_row.get("hits"))
            if not hits:
                continue
            top_hit = hits[0]
            score = float(top_hit.get("score", -np.inf))
            if score > best_score:
                best_score = score
                best_scan_row = scan_row
                best_hit = top_hit

        if best_hit is None or best_scan_row is None:
            continue

        scan_rt = float(best_scan_row.get("scan_rt", np.nan))
        q_mz = np.array(_as_list(best_hit.get("query_aligned", [[], []])[0]), dtype=float)
        valid_mz = q_mz[np.isfinite(q_mz)].tolist() if q_mz.size > 0 else []
        rt_list = [scan_rt] * len(valid_mz)

        matched_frags = _as_list(best_hit.get("matched_fragments"))
        num_matches = int(best_hit.get("num_matches", 0))
        ref_frags = int(best_hit.get("ref_frags", 0))

        rows.append(
            {
                "mz_rt_uid": str(uid),
                "best_ms2_file": os.path.basename(str(best_scan_row.get("filename", ""))),
                "best_ms2_rt": scan_rt,
                "best_ms2_score": float(best_hit.get("score", np.nan)),
                "best_ms2_mz": float(best_hit.get("mz_measured", np.nan)),
                "best_ms2_num_ions": f"{num_matches}/{ref_frags}" if ref_frags > 0 else str(num_matches),
                "best_ms2_matching_ions": ",".join(f"{float(m):.3f}" for m in matched_frags) if matched_frags else "",
                "best_ms2_spectrum_rt_mz": json.dumps([_jsonable_list(rt_list), _jsonable_list(valid_mz)]),
            }
        )

    return pd.DataFrame(rows)


def _build_compound_per_file_table(
    summary_obj: "AnalysisSummary",
) -> pa.Table:
    """Build a PyArrow Table with one row per compound x file.

    Joins ``per_file_metrics_df`` (peak_height, peak_area, rt_peak,
    rt_centroid, mz_peak, mz_centroid) with compound-level metadata from
    ``curation_df`` (atlas_mz, atlas_rt_*, adduct, formula, inchi_key,
    ms1/ms2 notes, quality scores, msi_level, control_filter).

    The table is sorted by ``atlas_mz`` ascending so that PyArrow row-group
    statistics enable efficient mz ± tolerance predicate pushdown.

    Returns
    -------
    pa.Table
    """

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning("per_file_metrics_df is empty — compound_per_file table will be empty.")
        return pa.table({})

    mc = summary_obj.experimental_data.curation_df

    # Build quality-score lookup from curation_df.
    # mz_measured uses top3_mz_centroid_avg (mean mz_centroid of top-3 files by peak_height).
    # rt_measured uses curation_df.rt_peak (mean of per-file peak RTs from analyze_ms1()).
    # Errors are absolute values.
    chromatography = summary_obj.chromatography
    quality_rows: list[dict] = []
    for cmp_idx, mc_row in mc.iterrows():
        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        mz_theoretical   = float(mc_row.get("atlas_mz", np.nan))
        rt_theoretical   = float(mc_row.get("atlas_rt_peak", np.nan))
        mz_measured_ms1  = float(mc_row.get("top3_mz_centroid_avg", np.nan))
        rt_measured_ms1  = float(mc_row.get("rt_peak", np.nan))

        if not np.isnan(mz_measured_ms1) and not np.isnan(mz_theoretical) and mz_theoretical != 0:
            ppm_err = abs(mz_measured_ms1 - mz_theoretical) / mz_theoretical * 1e6
            da_err  = abs(mz_theoretical - mz_measured_ms1)
        else:
            ppm_err = da_err = np.nan

        rt_error_ms1 = abs(rt_theoretical - rt_measured_ms1) if not (np.isnan(rt_theoretical) or np.isnan(rt_measured_ms1)) else np.nan

        mz_q  = _mz_quality(ppm_err, da_err)
        rt_q  = _rt_quality(rt_error_ms1, chromatography)
        ms2_notes = str(mc_row.get("ms2_notes", "") or "")
        try:
            msms_q = float(ms2_notes.split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = _total_score_and_msi(msms_q, mz_q, rt_q)

        quality_rows.append({
            "compound_index": int(cmp_idx) + 1,
            "mz_rt_uid":    mz_rt_uid,
            "compound_name": str(mc_row.get("compound_name", "")),
            "identified_metabolite": str(mc_row.get("compound_name", "")),
            "label": str(mc_row.get("compound_name", "")),
            "inchi_key":    str(mc_row.get("inchi_key", "")),
            "formula":      str(mc_row.get("formula", "")),
            "smiles":       str(mc_row.get("smiles", "")),
            "inchi":        str(mc_row.get("inchi", "")),
            "pubchem_cid":  str(mc_row.get("pubchem_cid", "")),
            "iupac_name":   str(mc_row.get("iupac_name", "")),
            "polarity":     str(mc_row.get("polarity", "")),
            "adduct":       str(mc_row.get("adduct", "")),
            "atlas_mz":     mz_theoretical,
            "atlas_rt_peak": float(mc_row.get("atlas_rt_peak", np.nan)),
            "atlas_rt_min":  float(mc_row.get("atlas_rt_min", np.nan)),
            "atlas_rt_max":  float(mc_row.get("atlas_rt_max", np.nan)),
            "rt_min":       float(mc_row.get("rt_min", np.nan)),
            "rt_max":       float(mc_row.get("rt_max", np.nan)),
            "exact_mass":   float(mc_row.get("mono_isotopic_molecular_weight", np.nan)),
            "overlapping_compound": str(mc_row.get("overlapping_compound", "")),
            "overlapping_inchi_keys": str(mc_row.get("overlapping_inchi_keys", "")),
            "isomer_details": str(mc_row.get("isomer_details", "")),
            "identification_notes": str(mc_row.get("identification_notes", "")),
            "analyst_notes": str(mc_row.get("analyst_notes", "")),
            "other_notes": str(mc_row.get("other_notes", "")),
            "best_ms1_file": str(mc_row.get("best_ms1_file", "")),
            "best_ms1_rt":   float(mc_row.get("best_ms1_rt", np.nan)),
            "best_ms1_mz":   float(mc_row.get("best_ms1_mz", np.nan)),
            "best_ms1_mz_centroid": float(mc_row.get("best_ms1_mz_centroid", np.nan)),
            "best_ms1_intensity": float(mc_row.get("best_ms1_intensity", np.nan)),
            "best_ms1_ppm_error": float(mc_row.get("best_ms1_ppm_error", np.nan)),
            "best_ms1_rt_error": float(mc_row.get("best_ms1_rt_error", np.nan)),
            "top3_mz_centroid_avg": float(mc_row.get("top3_mz_centroid_avg", np.nan)),
            "ms1_notes":    str(mc_row.get("ms1_notes", "") or ""),
            "ms2_notes":    ms2_notes,
            "mz_quality":   mz_q,
            "rt_quality":   rt_q,
            "msms_quality": msms_q,
            "total_score":  total_score,
            "msi_level":    msi_level,
        })

    quality_df = pd.DataFrame(quality_rows)

    # Load control_filter from peak_height_filtered.csv if available
    out_dir = Path(summary_obj.paths["analysis_results_output_dir"]) / "data_sheets"
    ctrl_filter_map: dict[str, str] = {}
    filtered_csv = out_dir / "peak_height_filtered.csv"
    if filtered_csv.exists():
        try:
            fdf = pd.read_csv(filtered_csv, usecols=lambda c: c in {"inchi_key", "control_filter"})
            if "inchi_key" in fdf.columns and "control_filter" in fdf.columns:
                ctrl_filter_map = dict(zip(fdf["inchi_key"], fdf["control_filter"]))
        except Exception as exc:
            logger.warning("Could not read control_filter from %s: %s", filtered_csv, exc)

    quality_df["control_filter"] = quality_df["inchi_key"].map(ctrl_filter_map).fillna("")

    # Merge per_file_df with quality metadata
    merged = per_file_df.merge(quality_df, on="mz_rt_uid", how="left")

    # Add per-sample MS1 spectrum payload as JSON [[rt],[i]].
    ms1_all_df = summary_obj.experimental_data.ms1_df
    if ms1_all_df is not None and not ms1_all_df.empty:
        spec_cols = ["mz_rt_uid", "filename", "spec_rts", "spec_ints"]
        spec_df = ms1_all_df[spec_cols].copy() if all(c in ms1_all_df.columns for c in spec_cols) else pd.DataFrame(columns=spec_cols)
        if not spec_df.empty:
            spec_df = spec_df.drop_duplicates(subset=["mz_rt_uid", "filename"], keep="first")
            merged = merged.merge(spec_df, on=["mz_rt_uid", "filename"], how="left")
            merged["ms1_spectrum_rt_i"] = merged.apply(
                lambda r: json.dumps([_jsonable_list(r.get("spec_rts")), _jsonable_list(r.get("spec_ints"))]),
                axis=1,
            )
            merged.drop(columns=["spec_rts", "spec_ints"], inplace=True, errors="ignore")
    if "ms1_spectrum_rt_i" not in merged.columns:
        merged["ms1_spectrum_rt_i"] = ""

    # Add best-MS2 per-compound fields and derived error metrics.
    best_ms2_df = _build_best_ms2_summary_df(summary_obj)
    if not best_ms2_df.empty:
        merged = merged.merge(best_ms2_df, on="mz_rt_uid", how="left")

    if "best_ms2_spectrum_rt_mz" not in merged.columns:
        merged["best_ms2_spectrum_rt_mz"] = ""

    atlas_mz_num = pd.to_numeric(merged.get("atlas_mz"), errors="coerce")
    best_ms2_mz_num = pd.to_numeric(merged.get("best_ms2_mz"), errors="coerce")
    valid_mz = atlas_mz_num.notna() & (atlas_mz_num != 0) & best_ms2_mz_num.notna()
    merged["best_ms2_mz_ppm_error"] = np.where(
        valid_mz,
        (best_ms2_mz_num - atlas_mz_num) / atlas_mz_num * 1e6,
        np.nan,
    )
    merged["best_ms2_mz_error_da"] = np.where(
        valid_mz,
        np.abs(best_ms2_mz_num - atlas_mz_num),
        np.nan,
    )
    merged["best_ms2_rt_error"] = pd.to_numeric(merged.get("best_ms2_rt"), errors="coerce") - pd.to_numeric(merged.get("atlas_rt_peak"), errors="coerce")

    # Final-ID aliases to make parquet and spreadsheet columns line up.
    # mz_measured / mz_error / mz_ppmerror use the MS1 top-3 average (top3_mz_centroid_avg).
    # rt_error uses the absolute MS1-based RT error (|atlas_rt_peak - rt_peak|) where
    # rt_peak is the mean of per-file peak RTs from analyze_ms1().
    # MS2-derived values are retained in best_ms2_* columns for reference.
    merged["msms_file"] = merged.get("best_ms2_file", "")
    merged["msms_rt"] = pd.to_numeric(merged.get("best_ms2_rt"), errors="coerce")
    merged["msms_score"] = pd.to_numeric(merged.get("best_ms2_score"), errors="coerce")
    merged["msms_numberofions"] = merged.get("best_ms2_num_ions", "")
    merged["msms_matchingions"] = merged.get("best_ms2_matching_ions", "")
    merged["mz_theoretical"] = pd.to_numeric(merged.get("atlas_mz"), errors="coerce")

    # MS1-based mz_measured: top-3-by-peak-height mean mz_centroid
    top3_mz = pd.to_numeric(merged.get("top3_mz_centroid_avg"), errors="coerce")
    atlas_mz_num = pd.to_numeric(merged.get("atlas_mz"), errors="coerce")
    merged["mz_measured"] = top3_mz
    valid_ms1_mz = top3_mz.notna() & atlas_mz_num.notna() & (atlas_mz_num != 0)
    merged["mz_error"] = np.where(
        valid_ms1_mz,
        np.abs(top3_mz - atlas_mz_num),
        np.nan,
    )
    merged["mz_ppmerror"] = np.where(
        valid_ms1_mz,
        np.abs(top3_mz - atlas_mz_num) / atlas_mz_num * 1e6,
        np.nan,
    )

    # MS1-based rt_error: absolute difference between atlas RT peak and curation_df.rt_peak
    # (rt_peak = mean of per-file peak RTs from analyze_ms1())
    ms1_rt = pd.to_numeric(merged.get("rt_peak"), errors="coerce")
    atlas_rt_num = pd.to_numeric(merged.get("atlas_rt_peak"), errors="coerce")
    merged["rt_theoretical"] = atlas_rt_num
    merged["rt_error"] = np.where(
        ms1_rt.notna() & atlas_rt_num.notna(),
        np.abs(ms1_rt - atlas_rt_num),
        np.nan,
    )

    # Select and type-cast columns
    str_cols = _COMPOUND_FILE_STR_COLS
    float_cols = _COMPOUND_FILE_FLOAT_COLS

    for col in str_cols:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("").astype(str)

    for col in float_cols:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["measured_mz"] = merged["mz_peak"]
    merged["measured_rt"] = merged["rt_peak"]

    for col in [
        "ms1_spectrum_rt_i", "best_ms2_spectrum_rt_mz", "msms_file",
        "msms_numberofions", "msms_matchingions", "best_ms2_file",
        "best_ms2_num_ions", "best_ms2_matching_ions",
    ]:
        merged[col] = merged[col].fillna("").astype(str)

    merged = merged[str_cols + float_cols]

    # Sort by atlas_mz for row-group skipping on mz range queries
    merged = merged.sort_values("atlas_mz", ascending=True, na_position="last")

    return pa.Table.from_pandas(merged, preserve_index=False)


def _build_compound_lfc_table(
    summary_obj: "AnalysisSummary",
) -> pa.Table:
    """Build a PyArrow Table with one row per compound x condition-pair (long format).

    Reads ``log_fold_changes.csv`` (produced by :func:`make_log_fold_changes_csv`)
    and melts the wide pairwise LFC columns into long format so that
    ``condition_1`` and ``condition_2`` are filterable data columns.

    Also joins compound-level quantitative metadata from ``curation_df``
    (atlas_mz, atlas_rt_min/max/peak, adduct, formula, inchi_key) so the
    table can be queried by mz/RT as well as by condition pair.

    Returns
    -------
    pa.Table
        Sorted by (condition_1, condition_2, atlas_mz).  Empty table if the
        source CSV does not exist.
    """
    out_dir = Path(summary_obj.paths["analysis_results_output_dir"]) / "data_sheets"
    lfc_csv = out_dir / "log_fold_changes.csv"

    if not lfc_csv.exists():
        logger.warning(
            "log_fold_changes.csv not found at %s — compound_lfc table will be empty. "
            "Run make_log_fold_changes_csv first.",
            lfc_csv,
        )
        return pa.table({})

    lfc_df = pd.read_csv(lfc_csv)

    # Identify identity columns vs LFC columns
    _ID_COLS = {
        "compound_index", "mz_rt_uid", "inchi_key", "compound_name", "adduct",
        "control_filter", "chosen_adduct", "chosen_polarity",
    }
    id_cols  = [c for c in lfc_df.columns if c in _ID_COLS]
    lfc_cols = [c for c in lfc_df.columns if c not in _ID_COLS]

    if not lfc_cols:
        logger.warning("No LFC columns found in %s — compound_lfc table will be empty.", lfc_csv)
        return pa.table({})

    # Melt wide to long: each LFC column becomes one row
    long_df = lfc_df.melt(
        id_vars=id_cols,
        value_vars=lfc_cols,
        var_name="comparison",
        value_name="log2_fold_change",
    )

    # Parse "g1_vs_g2" into separate condition columns
    split = long_df["comparison"].str.split("_vs_", n=1, expand=True)
    long_df["condition_1"] = split[0].fillna("")
    long_df["condition_2"] = split[1].fillna("") if 1 in split.columns else ""
    long_df = long_df.drop(columns=["comparison"])

    # ── Join compound quantitative metadata from curation_df ─────────────────
    mc = summary_obj.experimental_data.curation_df
    if mc is not None and not mc.empty:
        mc_slim_cols = [
            "mz_rt_uid", "inchi_key", "compound_name", "adduct", "formula", "smiles", "inchi",
            "pubchem_cid", "iupac_name", "atlas_mz", "atlas_rt_peak", "atlas_rt_min", "atlas_rt_max",
            "rt_min", "rt_max", "best_ms1_rt", "best_ms1_mz", "best_ms1_intensity",
            "best_ms1_ppm_error", "best_ms1_rt_error",
            "top3_mz_centroid_avg",
            "ms1_notes", "ms2_notes",
            "identification_notes", "analyst_notes", "other_notes", "msi_level",
        ]
        mc_slim_cols = [c for c in mc_slim_cols if c in mc.columns]
        mc_slim = mc.reset_index(drop=True)[mc_slim_cols].drop_duplicates(subset=[c for c in ["mz_rt_uid", "inchi_key", "adduct"] if c in mc_slim_cols])

        if "mz_rt_uid" in long_df.columns and "mz_rt_uid" in mc_slim.columns:
            long_df = long_df.merge(mc_slim, on="mz_rt_uid", how="left", suffixes=("", "_mc"))
        else:
            long_df = long_df.merge(mc_slim, on="inchi_key", how="left", suffixes=("", "_mc"))

        # Fill core identity columns from curation metadata when missing.
        for col in ("compound_name", "adduct"):
            mc_col = f"{col}_mc"
            if mc_col in long_df.columns:
                long_df[col] = long_df[col].where(long_df[col].notna() & (long_df[col].astype(str) != ""), long_df[mc_col])
                long_df.drop(columns=[mc_col], inplace=True, errors="ignore")

        # Best-MS2 payload and errors.
        best_ms2_df = _build_best_ms2_summary_df(summary_obj)
        if not best_ms2_df.empty and "mz_rt_uid" in long_df.columns:
            long_df = long_df.merge(best_ms2_df, on="mz_rt_uid", how="left")

        atlas_mz_num = pd.to_numeric(long_df.get("atlas_mz"), errors="coerce")
        best_ms2_mz_num = pd.to_numeric(long_df.get("best_ms2_mz"), errors="coerce")
        valid_mz = atlas_mz_num.notna() & (atlas_mz_num != 0) & best_ms2_mz_num.notna()
        long_df["best_ms2_mz_ppm_error"] = np.where(
            valid_mz,
            (best_ms2_mz_num - atlas_mz_num) / atlas_mz_num * 1e6,
            np.nan,
        )
        long_df["best_ms2_mz_error_da"] = np.where(
            valid_mz,
            np.abs(best_ms2_mz_num - atlas_mz_num),
            np.nan,
        )
        long_df["best_ms2_rt_error"] = pd.to_numeric(long_df.get("best_ms2_rt"), errors="coerce") - pd.to_numeric(long_df.get("atlas_rt_peak"), errors="coerce")

        long_df["msms_file"] = long_df.get("best_ms2_file", "")
        long_df["msms_rt"] = pd.to_numeric(long_df.get("best_ms2_rt"), errors="coerce")
        long_df["msms_score"] = pd.to_numeric(long_df.get("best_ms2_score"), errors="coerce")
        long_df["msms_numberofions"] = long_df.get("best_ms2_num_ions", "")
        long_df["msms_matchingions"] = long_df.get("best_ms2_matching_ions", "")
        long_df["mz_theoretical"] = pd.to_numeric(long_df.get("atlas_mz"), errors="coerce")

        # MS1-based mz_measured / rt_error — consistent with compound_per_file table
        lfc_top3_mz = pd.to_numeric(long_df.get("top3_mz_centroid_avg"), errors="coerce")
        lfc_atlas_mz = pd.to_numeric(long_df.get("atlas_mz"), errors="coerce")
        long_df["mz_measured"] = lfc_top3_mz
        lfc_valid_mz = lfc_top3_mz.notna() & lfc_atlas_mz.notna() & (lfc_atlas_mz != 0)
        long_df["mz_error"] = np.where(
            lfc_valid_mz,
            np.abs(lfc_top3_mz - lfc_atlas_mz),
            np.nan,
        )
        long_df["mz_ppmerror"] = np.where(
            lfc_valid_mz,
            np.abs(lfc_top3_mz - lfc_atlas_mz) / lfc_atlas_mz * 1e6,
            np.nan,
        )

        def _numeric_col(df, col):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
            return pd.Series(np.nan, index=df.index, dtype="float64")

        lfc_top3_rt = _numeric_col(long_df, "top3_rt")
        lfc_atlas_rt = _numeric_col(long_df, "atlas_rt_peak")
        long_df["rt_theoretical"] = lfc_atlas_rt
        long_df["rt_error"] = np.where(
            pd.notna(lfc_top3_rt) & pd.notna(lfc_atlas_rt),
            np.abs(lfc_top3_rt - lfc_atlas_rt),
            np.nan,
        )
    else:
        for col in (
            "adduct", "formula", "smiles", "inchi", "pubchem_cid", "iupac_name",
            "atlas_mz", "atlas_rt_peak", "atlas_rt_min", "atlas_rt_max",
            "best_ms2_file", "best_ms2_num_ions", "best_ms2_matching_ions", "best_ms2_spectrum_rt_mz",
            "msms_file", "msms_numberofions", "msms_matchingions",
        ):
            long_df[col] = np.nan if col.startswith("atlas") else ""

    str_cols = _COMPOUND_LFC_STR_COLS
    float_cols = _COMPOUND_LFC_FLOAT_COLS

    for col in str_cols:
        if col not in long_df.columns:
            long_df[col] = ""
        long_df[col] = long_df[col].fillna("").astype(str)

    for col in float_cols:
        if col not in long_df.columns:
            long_df[col] = np.nan
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    col_order = str_cols + float_cols
    long_df = long_df[col_order]

    # Sort so condition pair + mz queries get row-group skipping
    long_df = long_df.sort_values(
        ["condition_1", "condition_2", "atlas_mz"],
        ascending=True,
        na_position="last",
    )

    return pa.Table.from_pandas(long_df, preserve_index=False)


def _parse_project_metadata(project_name: str) -> dict[str, str]:

    if not project_name:
        return {}
    try:
        parsed = fpf.PROJECT_PATTERN.match(project_name)
        if not parsed:
            logger.warning(
                "project_name '%s' does not match PROJECT_PATTERN — "
                "project metadata will not be added to Parquet footer.",
                project_name,
            )
            return {}
        return {k: str(v) for k, v in parsed.groupdict().items() if v is not None}
    except Exception as exc:
        logger.warning(
            "Could not parse project metadata from '%s': %s — "
            "project metadata will not be added to Parquet footer.",
            project_name, exc,
        )
        return {}