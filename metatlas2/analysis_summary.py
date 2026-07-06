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
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import metatlas2.database_interact as dbi
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
        make_log_fold_changes_csv(summary_obj, overwrite=overwrite)

    if "metabomap" not in (skip_outputs or []):
        logger.info("Making metabomap (merged pos/neg peak heights + LFC table)...")
        make_metabomap(summary_obj, overwrite=overwrite)

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

def _get_compound_info(
    main_db_path: str,
    inchi_keys: List[str],
) -> dict[str, tuple]:
    """Fetch all compound metadata in ONE query from the compounds table.

    Returns
    -------
    dict of inchi_key -> (formula, smiles, inchi, pubchem_cid, mono_isotopic_molecular_weight)
    Missing keys are simply absent from the dict.

    Note: this is kept as a fallback for callers that do not have the metadata
    already available in the curation_df.  The preferred path is to read
    formula/smiles/inchi/pubchem_cid/mono_isotopic_molecular_weight directly
    from mc_row, which is populated by create_manual_curation_obj from the
    atlas query (get_atlas_compounds_table already joins compounds).
    """
    if not inchi_keys or not main_db_path:
        return {}
    placeholders = ",".join("?" * len(inchi_keys))
    try:
        with dbi.get_db_connection(main_db_path, read_only=True) as conn:
            rows = conn.execute(
                f"""SELECT inchi_key, formula, smiles, inchi,
                           pubchem_cid, mono_isotopic_molecular_weight
                    FROM compounds
                    WHERE inchi_key IN ({placeholders})""",
                inchi_keys,
            ).fetchall()
        return {
            row[0]: (row[1], row[2], row[3], row[4], float(row[5]) if row[5] is not None else None)
            for row in rows
        }
    except Exception as exc:
        logger.warning("Batch compound info query failed: %s", exc)
        return {}

def _safe_float(value, default: float = np.nan) -> float:
    """Coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Value is not a valid float: {value}")
        return float(default)


def _safe_isnan(value) -> bool:
    """Return True if *value* is NaN, None, or cannot be coerced to a float.

    Unlike ``np.isnan``, this never raises for non-numeric types (strings,
    None, objects).  None is treated as NaN because aligned mz/intensity
    arrays use None as a sentinel (JSON round-trip converts float nan →
    null → Python None) to indicate "no peak at this position".
    """
    if value is None:
        return True
    try:
        return np.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _as_list(value) -> list:
    """Safely coerce *value* to a plain Python list.

    Handles the common case where a DataFrame cell contains a numpy array,
    a Python list, ``None``, or a scalar.  Using ``value or []`` raises
    ``ValueError`` when *value* is a numpy array with more than one element
    because Python cannot determine the truth value of such an array.
    """
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
    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return
    manual_curation_df = manual_curation_df.reset_index(drop=True)

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

        # formula/smiles/inchi are already in mc_row, populated by
        # create_manual_curation_obj from the atlas query which joins compounds.
        # No separate DB query needed.
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
    logger.info("Identification figure export complete → %s", output_dir)


def _identification_figure_worker(kwargs: dict) -> str:
    """Worker: generate and save one identification figure PDF."""
    import matplotlib
    matplotlib.use("Agg")

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
        qry_int_valid = [v for v in qry_int if v is not None and not (isinstance(v, float) and np.isnan(v))]
        ref_int_valid = [v for v in ref_int if v is not None and not (isinstance(v, float) and np.isnan(v))]
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

    score_str = f"{score:.3f}" if (isinstance(score, (int, float)) and not np.isnan(score)) else "N/A"
    rt_str = f"{rt:.2f}" if (isinstance(rt, (int, float)) and not np.isnan(rt)) else "N/A"
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
    rt_str = f"{rt:.2f}" if (isinstance(rt, (int, float)) and not np.isnan(rt)) else "N/A"
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
    """Render a two-column key/value table of compound metadata."""
    ax.axis("off")

    def _fmt(val, fmt=None):
        if val is None or (isinstance(val, str) and not val.strip()) or _safe_isnan(val):
            return "N/A"
        try:
            return fmt.format(val) if fmt else str(val)
        except (ValueError, TypeError):
            return str(val)

    rows = [
        ("Compound", _fmt(mc_row.get("compound_name", "Undefined"))),
        ("Formula", _fmt(mc_row.get("formula"))),
        ("Adduct", _fmt(mc_row.get("adduct"))),
        ("Polarity", _fmt(mc_row.get("polarity"))),
        ("Chromatography", _fmt(mc_row.get("chromatography"))),
        ("Atlas m/z", _fmt(mc_row.get("atlas_mz"), "{:.4f}")),
        ("Measured m/z", _fmt(mc_row.get("mz"), "{:.4f}")),
        ("m/z ppm Δ", _fmt(mc_row.get("mz_error"), "{:.1f} ppm")),
        ("Atlas RT range", f"{_fmt(mc_row.get('atlas_rt_min'), '{:.3f}')} - {_fmt(mc_row.get('atlas_rt_max'), '{:.3f}')} min"),
        ("Measured RT range", f"{_fmt(mc_row.get('rt_min'), '{:.3f}')} - {_fmt(mc_row.get('rt_max'), '{:.3f}')} min"),
        ("Atlas RT peak", _fmt(mc_row.get("atlas_rt_peak"), "{:.3f} min")),
        ("Measured RT", _fmt(mc_row.get("rt_peak"), "{:.3f} min")),
        ("RT Δ", _fmt(mc_row.get("rt_error"), "{:.3f}")),
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
    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to export.")
        return
    manual_curation_df = manual_curation_df.reset_index(drop=True)

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

    # --- Build MS2 lookup maps ---
    ms2_best: dict[str, dict] = {}       # mz_rt_uid -> best hit dict + scan metadata
    ms2_top3_mz_avg: dict[str, float] = {}  # mz_rt_uid -> mean mz of top 3 hits

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

                # Top-3 mz average: pull from the best scan's hits list
                # We already know the best scan — re-fetch its hits
                best_scan_row = grp.loc[
                    grp["filename"] == best_hit_data["filename"]
                ].iloc[0]
                top3_hits = _as_list(best_scan_row.get("hits"))[:3]
                valid_mzs = [
                    float(h["mz_measured"]) for h in top3_hits
                    if np.isfinite(float(h.get("mz_measured", np.nan)))
                ]
                if valid_mzs:
                    ms2_top3_mz_avg[uid] = float(np.mean(valid_mzs))

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
        mz_measured_ms1 = float(mc_row.get("mz", np.nan))
        ppm_error_ms1 = float(mc_row.get("mz_error", np.nan))
        rt_measured = float(mc_row.get("rt_peak", np.nan))
        rt_error = float(mc_row.get("rt_error", np.nan))
        best_ms1_rt = float(mc_row.get("best_ms1_rt", np.nan))
        best_ms1_intensity = float(mc_row.get("best_ms1_intensity", np.nan))
        best_ms1_file = mc_row.get("best_ms1_file", "")

        ms2_notes = str(mc_row.get("ms2_notes", "") or "")

        # --- MS2 metrics (default to MS1 values if no MS2 available) ---
        mz_measured_ms2 = mz_measured_ms1
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

            try:
                mz_measured_ms2 = float(best_hit.get("mz_measured", mz_measured_ms1))
            except (TypeError, ValueError):
                mz_measured_ms2 = mz_measured_ms1

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

        # Top-3 MS2 mz average overrides single-scan mz when available
        mz_measured_ms2 = float(ms2_top3_mz_avg.get(mz_rt_uid, mz_measured_ms2))

        # Recompute ppm and Da errors using the final mz_measured_ms2
        if not np.isnan(mz_measured_ms2) and not np.isnan(mz_theoretical) and mz_theoretical != 0:
            ppm_error_ms2 = (mz_measured_ms2 - mz_theoretical) / mz_theoretical * 1e6
            mz_error_da = abs(mz_theoretical - mz_measured_ms2)
        else:
            ppm_error_ms2 = np.nan
            mz_error_da = np.nan

        mz_q = _mz_quality(ppm_error_ms2, mz_error_da)
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
            "formula":                formula or "",
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
            "mz_measured":            _safe_round(mz_measured_ms2, 4),
            # M/Z EVALUATION
            "mz_error":               _safe_round(mz_error_da, 4),
            "mz_ppmerror":            _safe_round(ppm_error_ms2, 4),
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

    # ------------------------------------------------------------------ #
    #  Excel formatting                                                    #
    # ------------------------------------------------------------------ #

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
    if ppm_error <= 5 or mz_delta <= 0.0015:
        return 1.0
    if ppm_error <= 10:
        return 0.5
    return 0.0

def _rt_quality(rt_error: float, chromatography: str) -> float:
    """Return 0/0.5/1 RT quality score, with thresholds that depend on method."""
    if np.isnan(rt_error):
        return np.nan
    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()
    if is_c18:
        if rt_error <= 0.25:
            return 1.0
        if rt_error <= 0.5:
            return 0.5
        return 0.0
    else:
        if rt_error <= 0.5:
            return 1.0
        if rt_error <= 2.0:
            return 0.5
        return 0.0

def _total_score_and_msi(
    msms_q: float, mz_q: float, rt_q: float
) -> Tuple[float, str]:
    """Compute total ID score (0-3) and MSI level string.
    """
    scores = [v for v in [msms_q, mz_q, rt_q] if isinstance(v, float) and not np.isnan(v)]
    total = float(np.nansum([msms_q, mz_q, rt_q]))

    if isinstance(msms_q, float) and msms_q == -1:
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
                    "group_name": _file_group(f_path),
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

    import matplotlib
    matplotlib.use("Agg")

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
        if v is not None and not (isinstance(v, float) and np.isnan(v))
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
        group_name = item.get("group_name") or _file_group(item.get("filename", ""))
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
                valid_ints = [v for v in spec_ints if v is not None and not (isinstance(v, float) and np.isnan(v))]
                y_max_indep = float(max(valid_ints)) if valid_ints else None
                group_name = item.get("group_name") or _file_group(item.get("filename", ""))
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


def _file_group(filename: str) -> str:
    """Return the file group label from the stem's 13th underscore-separated segment (index 12).

    Falls back to the full stem when the filename has fewer than 13 segments.
    """
    if not filename:
        return "unknown"
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    return parts[12] if len(parts) > 12 else stem


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
        if val is not None and not (isinstance(val, float) and np.isnan(val))
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
            vals = [np.log10(v) for v in vals if isinstance(v, (int, float)) and v > 0]
        n_files = len(group_rows)
        data_per_group.append(vals if vals else [0.0] * max(n_files, 1))
        valid_groups.append(g)

    if not data_per_group:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        title_top = f"{_display_compound_idx(compound_idx):04d}  {compound_name}  {adduct}"
        if metric == "mz_centroid":
            title_bottom = f"m/z centroid: {atlas_ref if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)) else 'N/A'}"
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

    if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)):
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
        mz_val = atlas_ref if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)) else 'N/A'
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
    import matplotlib
    matplotlib.use("Agg")

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

    logger.info("Exporting identification figures to %s", output_dir)

    manual_curation_df = summary_obj.experimental_data.curation_df.reset_index(drop=True)

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    # Build per_file_metrics_df lazily if not already populated
    if summary_obj.per_file_metrics_df is None or summary_obj.per_file_metrics_df.empty:
        logger.info("per_file_metrics_df not set — computing from ms1_df...")
        summary_obj.per_file_metrics_df = _build_per_file_metrics_df(
            summary_obj.experimental_data.ms1_df
        )
    per_file_df = summary_obj.per_file_metrics_df

    # ── 1. Build atlas lookup (unchanged) ────────────────────────────────────
    atlas_lookup: Dict[Tuple[str, str], Dict[str, float]] = {
        (mc.get("inchi_key", ""), mc.get("adduct", "")): {
            "atlas_mz":      float(mc.get("atlas_mz", np.nan)),
            "atlas_rt_peak": float(mc.get("atlas_rt_peak", np.nan)),
        }
        for _, mc in manual_curation_df.iterrows()
    }

    # ── 2. Create metric output dirs upfront, store paths as strings ──────────
    _METRIC_CONFIGS = [
        ("peak_height", False, "Peak Height (intensity)", None),
        ("peak_height", True,  "Peak Height (log₁₀)",     None),
        ("rt_peak",     False, "RT Peak (min)",           "atlas_rt_peak"),
        ("mz_centroid", False, "m/z Centroid",            "atlas_mz"),
    ]
    metric_configs_with_dirs: list[tuple] = []
    for metric, log_scale, ylabel, atlas_attr in _METRIC_CONFIGS:
        metric_dir = output_dir / f"{metric}_{'log' if log_scale else 'linear'}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_configs_with_dirs.append((metric, log_scale, ylabel, atlas_attr, str(metric_dir)))

    # ── 3. Pre-group per_file_df — replaces 4xn O(n) filters with O(1) lookup
    if per_file_df is not None and not per_file_df.empty:
        pf_groups: dict = {
            key: grp.reset_index(drop=True)
            for key, grp in per_file_df.groupby(["inchi_key", "adduct"], sort=False)
        }
    else:
        pf_groups = {}
    empty_df = pd.DataFrame()

    # ── 4. One task per compound — worker handles all 4 metric PDFs ───────────
    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")
        inchi_key = mc_row.get("inchi_key", "")
        key = (inchi_key, adduct)

        tasks.append({
            "compound_name": compound_name,
            "adduct": adduct,
            "inchi_key": inchi_key,
            "compound_idx": cmp_idx,
            "atlas_ref_dict": atlas_lookup.get(key, {}),
            "metric_configs": metric_configs_with_dirs,
            "cmp_metrics": pf_groups.get(key, empty_df),
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

    logger.info("Boxplot PDFs complete → %s", output_dir)

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
    logger.info(
        "Exported best MS2 hit fragment ions CSV (%d compounds) → %s",
        len(result_df), output_file,
    )

def _build_per_file_metrics_df(ms1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-compound, per-file summary metrics from the wide-format ms1_df.

    Each row of *ms1_df* represents one compound x one file, with list columns
    ``spec_rts``, ``spec_ints``, and ``spec_mzs`` (already filtered to the
    in-feature RT window).  Returns a long-format DataFrame with one row per
    (compound, file) and columns:

    * ``mz_rt_uid``, ``inchi_key``, ``adduct``, ``filename``
    * ``peak_height``  – maximum intensity in the window
    * ``peak_area``    – sum of intensities (proxy for area)
    * ``rt_peak``      – RT at the intensity maximum
    * ``rt_centroid``  – intensity-weighted mean RT
    * ``mz_peak``      – m/z at the intensity maximum
    * ``mz_centroid``  – intensity-weighted mean m/z
    """
    if ms1_df is None or ms1_df.empty:
        return pd.DataFrame()

    records = []
    for _, row in ms1_df.iterrows():
        spec_rts  = _as_list(row.get("spec_rts"))
        spec_ints = _as_list(row.get("spec_ints"))
        spec_mzs  = _as_list(row.get("spec_mzs"))

        # Coerce to float arrays, replacing None/nan with 0 for intensities
        rts  = [float(v) if v is not None else float("nan") for v in spec_rts]
        ints = [float(v) if v is not None else 0.0           for v in spec_ints]
        mzs  = [float(v) if v is not None else float("nan") for v in spec_mzs]

        if not ints or max(ints) == 0:
            peak_height = peak_area = rt_peak = rt_centroid = mz_peak = mz_centroid = float("nan")
        else:
            peak_idx    = int(np.argmax(ints))
            peak_height = ints[peak_idx]
            peak_area   = float(np.nansum(ints))
            rt_peak     = rts[peak_idx]  if peak_idx < len(rts)  else float("nan")
            mz_peak     = mzs[peak_idx]  if peak_idx < len(mzs)  else float("nan")
            total_int   = peak_area
            rt_centroid = (
                float(np.nansum([r * i for r, i in zip(rts, ints)]) / total_int)
                if total_int > 0 else float("nan")
            )
            mz_centroid = (
                float(np.nansum([m * i for m, i in zip(mzs, ints)]) / total_int)
                if total_int > 0 and mzs else float("nan")
            )

        fname = row.get("filename", "")
        records.append({
            "mz_rt_uid":   row.get("mz_rt_uid", ""),
            "inchi_key":   row.get("inchi_key", ""),
            "adduct":      row.get("adduct", ""),
            "filename":    fname,
            "file_group":  _file_group(fname),
            "peak_height": peak_height,
            "peak_area":   peak_area,
            "rt_peak":     rt_peak,
            "rt_centroid": rt_centroid,
            "mz_peak":     mz_peak,
            "mz_centroid": mz_centroid,
        })

    return pd.DataFrame(records)


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

    # Build per_file_metrics_df lazily if not already populated
    if summary_obj.per_file_metrics_df is None or summary_obj.per_file_metrics_df.empty:
        logger.info("per_file_metrics_df not set — computing from ms1_df...")
        summary_obj.per_file_metrics_df = _build_per_file_metrics_df(
            summary_obj.experimental_data.ms1_df
        )

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
    _INDEX_COLS = [
        "compound_index",
        "compound_name",
        "mz_rt_uid",
        "inchi_key",
    ]

    for metric in _DATA_SHEET_METRICS:
        if metric not in pfm.columns:
            logger.warning(
                "Metric '%s' not found in per_file_metrics_df — skipping.", metric
            )
            continue

        csv_path = output_dir / f"{metric}.csv"

        wide = (
            pfm[_INDEX_COLS + ["_file_col", metric]]
            .pivot_table(
                index=_INDEX_COLS,
                columns="_file_col",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        wide.to_csv(csv_path, index=False)
        logger.info(
            "Exported %s data sheet (%d compounds x %d files) → %s",
            metric,
            len(wide),
            len(wide.columns) - len(_INDEX_COLS),
            csv_path,
        )

    logger.info("Data sheets written to %s", output_dir)

def make_peak_height_filtered_csv(
    obj: "AnalysisSummary",
    overwrite: bool = True,
    control_fold_threshold: float = 3.0,
) -> None:
    """Filter and process the ``peak_height`` data sheet."""

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

    _META_COLS = {"compound_index", "compound_name", "mz_rt_uid", "inchi_key", "adduct", "polarity"}
    data_cols = [c for c in df.columns if c not in _META_COLS]

    # ── 1. Control-signal filter ───────────────────────────────────────────────
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

    # ── 2. Merge identical inchi_key rows by max; keep chosen adduct/polarity ──
    group_keys       = ["inchi_key", "compound_name", "control_filter"]
    source_meta_cols = [c for c in ("adduct", "polarity") if c in df.columns]
    selector_cols    = non_ctrl_cols if non_ctrl_cols else data_cols

    def _infer_polarity_from_label(label: object) -> str:
        if not isinstance(label, str) or not label:
            return ""
        m = re.search(r"(^|[_\-])(pos|neg)([_\-]|$)", label, flags=re.IGNORECASE)
        if m:
            return m.group(2).upper()
        m = re.search(r"(positive|negative)", label, flags=re.IGNORECASE)
        if m:
            return "POS" if m.group(1).lower().startswith("pos") else "NEG"
        return ""

    chosen_meta = None
    if selector_cols:
        selection_score = df[selector_cols].max(axis=1, skipna=True)
        selector_frame  = df[selector_cols]
        best_col        = selector_frame.idxmax(axis=1)
        best_col        = best_col.where(selector_frame.notna().any(axis=1), "")

        chosen_src = (
            df[group_keys + source_meta_cols].copy()
            if source_meta_cols
            else df[group_keys].copy()
        )
        chosen_src["_selection_score"] = selection_score.fillna(-np.inf)
        chosen_src["_best_col"]        = best_col
        chosen_src["_row_order"]       = np.arange(len(chosen_src))
        chosen_src = chosen_src.sort_values(
            ["_selection_score", "_row_order"], ascending=[False, True]
        )
        chosen_meta = chosen_src.drop_duplicates(subset=group_keys, keep="first")
        chosen_meta = chosen_meta.rename(columns={
            "adduct":   "chosen_adduct",
            "polarity": "chosen_polarity",
        })
        if "chosen_polarity" not in chosen_meta.columns:
            chosen_meta["chosen_polarity"] = chosen_meta["_best_col"].map(
                _infer_polarity_from_label
            )
        else:
            chosen_meta["chosen_polarity"] = chosen_meta["chosen_polarity"].fillna(
                chosen_meta["_best_col"].map(_infer_polarity_from_label)
            )
        chosen_meta = chosen_meta.drop(
            columns=["_selection_score", "_best_col", "_row_order"]
        )

    drop_cols = [c for c in ("compound_index", "mz_rt_uid", "adduct", "polarity") if c in df.columns]
    df = df.drop(columns=drop_cols)

    n_before = len(df)
    df = df.groupby(group_keys, sort=False).max().reset_index()
    if chosen_meta is not None:
        df = df.merge(chosen_meta, on=group_keys, how="left")
    logger.info("inchi_key row merge: %d → %d rows.", n_before, len(df))

    def _shorten(col: str) -> str:
        parts = col.split("_")
        return "_".join(parts[:-2]) if len(parts) > 2 else col

    _META_FIXED = {
        "inchi_key", "compound_name", "control_filter", "chosen_adduct", "chosen_polarity"
    }
    df.columns = [c if c in _META_FIXED else _shorten(c) for c in df.columns]

    if df.columns.duplicated().any():
        meta_df = df[[c for c in df.columns if c in _META_FIXED]].reset_index(drop=True)
        data_df = df[[c for c in df.columns if c not in _META_FIXED]]
        data_merged = (
            data_df.T
            .groupby(level=0)
            .max()
            .T
            .reset_index(drop=True)
        )
        df = pd.concat([meta_df, data_merged], axis=1)
        logger.info(
            "Merged duplicate columns after shortening; %d data columns remain.",
            len(data_merged.columns),
        )

    # ── 4. Impute NaN cells with the global matrix minimum ────────────────────
    data_cols_final = [c for c in df.columns if c not in _META_FIXED]
    global_min = float(df[data_cols_final].min().min()) if data_cols_final else np.nan
    if not np.isnan(global_min):
        n_nan = int(df[data_cols_final].isna().sum().sum())
        df[data_cols_final] = df[data_cols_final].fillna(global_min)
        logger.info("Imputed %d NaN cells with global minimum: %g", n_nan, global_min)
    else:
        logger.warning("Global minimum is NaN — no imputation performed.")

    # ── 5. Reorder columns ─────────────────────────────────────────────────────
    ordered_meta = ["control_filter", "compound_name", "inchi_key"]
    if "chosen_adduct" in df.columns:
        ordered_meta.append("chosen_adduct")
    if "chosen_polarity" in df.columns:
        ordered_meta.append("chosen_polarity")
    other_cols = [c for c in df.columns if c not in set(ordered_meta)]
    df = df[ordered_meta + other_cols]

    # ── 6. Export ──────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    logger.info(
        "Exported filtered peak height (%d compounds x %d files) → %s",
        len(df), len(df.columns) - len(ordered_meta), output_csv,
    )


def make_log_fold_changes_csv(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Create per-run log2 fold-change CSV from peak_height_filtered.csv.

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

    logger.info("Creating per-run log fold changes CSV at %s", output_csv)
    df = pd.read_csv(source_csv)

    id_cols = [
        c for c in ["inchi_key", "compound_name", "control_filter", "chosen_adduct", "chosen_polarity"]
        if c in df.columns
    ]
    data_cols = [c for c in df.columns if c not in id_cols]

    if not data_cols:
        logger.warning("No numeric sample columns found in %s; writing identifiers only.", source_csv)
        pd.DataFrame({c: df[c] for c in id_cols}).to_csv(output_csv, sep=",", index=False)
        return

    data_num = df[data_cols].apply(pd.to_numeric, errors="coerce")
    lfc_records = {col: df[col].to_numpy() for col in id_cols}

    for g1, g2 in itertools.combinations(data_num.columns, 2):
        v1 = data_num[g1].to_numpy(dtype=float)
        v2 = data_num[g2].to_numpy(dtype=float)
        valid = (v1 > 0) & (v2 > 0) & np.isfinite(v1) & np.isfinite(v2)
        res = np.full(len(v1), np.nan)
        res[valid] = np.log2(v1[valid] / v2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res

    pd.DataFrame(lfc_records).to_csv(output_csv, sep=",", index=False)
    logger.info(
        "Exported per-run log fold changes (%d compounds x %d comparisons) → %s",
        len(df),
        max(len(lfc_records) - len(id_cols), 0),
        output_csv,
    )

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
        logger.info("Loading ModelSEED compounds from local cache: %s", cache_path)
    else:
        logger.info("Fetching ModelSEED compounds table from %s", _MODELSEED_COMPOUNDS_URL)
        resp = requests.get(_MODELSEED_COMPOUNDS_URL, timeout=30)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        logger.info("Saved ModelSEED compounds cache → %s", cache_path)

    return pd.read_csv(cache_path, sep="\t", low_memory=False)


def _build_inchikey_to_cpd(cache_path: Path) -> dict[str, str]:
    """Return a mapping of InChIKey → semicolon-joined ModelSEED CPD IDs.

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
    logger.info(
        "Built InChIKey→CPD mapping: %d unique InChIKeys (%d with multiple CPD IDs).",
        len(mapping), n_multi,
    )
    return mapping

def make_metabomap(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    modelseed_cache_path: Optional[Path] = None,
) -> None:
    # 1. Setup Paths and Polarities
    analysis_output_dir = Path(summary_obj.paths.get("analysis_results_output_dir"))
    curr_pol = summary_obj.polarity.upper()
    sib_pol = "NEG" if curr_pol == "POS" else "POS"
    
    current_csv = analysis_output_dir / "data_sheets" / "peak_height_filtered.csv"
    sibling_csv = Path(str(analysis_output_dir).replace(f"-{curr_pol}-", f"-{sib_pol}-")) / "data_sheets" / "peak_height_filtered.csv"
    
    metabomaps_dir = analysis_output_dir / "../metabomap"
    merged_tsv, lfc_tsv = metabomaps_dir / "merged_peak_heights.tsv", metabomaps_dir / "log_fold_changes.tsv"

    if not overwrite and merged_tsv.exists() and lfc_tsv.exists():
        return

    if not current_csv.exists() or not sibling_csv.exists():
        logger.warning("Missing polarity CSVs; skipping metabomap.")
        return

    metabomaps_dir.mkdir(parents=True, exist_ok=True)

    # 2. Helper for Column Filtering
    _META = {"control_filter", "compound_name", "inchi_key"}
    _EXCLUDE = ("QC", "ISTD")

    def _get_grp(col: str) -> Optional[str]:
        parts = col.split("_")
        return parts[12] if len(parts) > 12 and not any(p in parts[12] for p in _EXCLUDE) else None

    def _process_df(path: Path):
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c in _META or _get_grp(c) is not None]
        df = df[cols].copy()
        # Group by InChI + Name to collapse peak_1, peak_2 etc. into the max value
        return df.set_index(["inchi_key", "compound_name"]).groupby(level=[0, 1]).max()

    # 3. Load and Align
    cur_idx = _process_df(current_csv)
    sib_idx = _process_df(sibling_csv)
    
    # Ensure columns are sorted consistently by group
    cur_sorted = sorted([c for c in cur_idx.columns if c not in _META], key=lambda c: (_get_grp(c), c))
    sib_sorted = sorted([c for c in sib_idx.columns if c not in _META], key=lambda c: (_get_grp(c), c))
    
    all_ids = cur_idx.index.union(sib_idx.index)
    cur_matrix = cur_idx[cur_sorted].reindex(all_ids)
    sib_matrix = sib_idx[sib_sorted].reindex(all_ids)

    # 4. Merge using Element-wise Max
    cur_groups = [_get_grp(c) for c in cur_sorted]
    sib_groups = [_get_grp(c) for c in sib_sorted]

    if cur_groups == sib_groups:
        merged_vals = np.fmax(cur_matrix.values, sib_matrix.values)
        merged_data_df = pd.DataFrame(merged_vals, index=all_ids, columns=cur_groups)
    else:
        logger.warning("Column groups differ; using per-group max.")
        all_grps = list(dict.fromkeys(cur_groups + sib_groups))
        merged_data_df = pd.DataFrame(index=all_ids)
        for grp in all_grps:
            c_cols = [c for c in cur_sorted if _get_grp(c) == grp]
            s_cols = [c for c in sib_sorted if _get_grp(c) == grp]
            # Combine all replicates for this group from both polarities and take max
            merged_data_df[grp] = pd.concat([cur_matrix[c_cols], sib_matrix[s_cols]], axis=1).max(axis=1)

    # 5. Finalize Metadata and ModelSEED
    merged_data_df = merged_data_df.reset_index() # Now contains inchi_key and compound_name
    
    ms_path = Path(summary_obj.paths["modelseed_table_path"])
    try:
        inchikey_to_cpd = _build_inchikey_to_cpd(ms_path)
    except Exception:
        inchikey_to_cpd = {}

    merged_data_df.insert(1, "cpd_id", merged_data_df["inchi_key"].map(inchikey_to_cpd))
    merged_data_df.to_csv(merged_tsv, sep="\t", index=False)

    # 6. Log2 Fold Changes
    unique_groups = [c for c in merged_data_df.columns if c not in {"inchi_key", "compound_name", "cpd_id"}]
    if len(unique_groups) < 2: return

    # Precompute means to avoid repeated slicing
    means = {grp: merged_data_df[grp].mean(axis=1).to_numpy() if merged_data_df[grp].ndim > 1 
             else merged_data_df[grp].to_numpy() for grp in unique_groups}

    lfc_records = {col: merged_data_df[col].to_numpy() for col in ["inchi_key", "cpd_id", "compound_name"]}
    
    for g1, g2 in itertools.combinations(unique_groups, 2):
        m1, m2 = means[g1], means[g2]
        valid = (m1 > 0) & (m2 > 0) & ~np.isnan(m1) & ~np.isnan(m2)
        res = np.full(len(m1), np.nan)
        res[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res

    pd.DataFrame(lfc_records).to_csv(lfc_tsv, sep="\t", index=False)