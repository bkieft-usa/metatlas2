from __future__ import annotations

import datetime
import itertools
import shutil
from pathlib import Path
import json
import os
import re
import sys
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
import metatlas2.file_and_project_format as fpf
import metatlas2.gdrive_upload as gdu
import metatlas2.lcmsruns_tools as lrt
import metatlas2.logging_config as lcf
import metatlas2.parquet_interact as pqi
from metatlas2.note_options import (
    get_note_options_and_hotkeys,
    get_notes_opts,
    should_require_note_selection,
)
from metatlas2.utils import (
    should_disable_tqdm,
    safe_float,
    safe_isnan,
    as_list,
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
        make_identification_figure(summary_obj, overwrite=overwrite, max_workers=4)

    if "eic_thumbnails" not in (skip_outputs or []):
        logger.info("Making EIC thumbnails...")
        make_eic_thumbnails(summary_obj, overwrite=overwrite, max_workers=4)

    if "tic_thumbnails" not in (skip_outputs or []):
        logger.info("Making TIC thumbnails...")
        make_tic_thumbnails(summary_obj, overwrite=overwrite)

    if "compound_metric_plots" not in (skip_outputs or []):
        logger.info("Making compound peak height and RT peak plots...")
        make_compound_metric_plots(summary_obj, overwrite=overwrite)

    if "boxplots" not in (skip_outputs or []):
        logger.info("Making Boxplots...")
        make_boxplots(summary_obj, overwrite=overwrite, max_workers=4)

    if "data_sheets" not in (skip_outputs or []):
        logger.info("Making quantitative data sheets...")
        make_data_sheets(summary_obj, overwrite=overwrite)

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

    ### NEED TO UPDATE FOR MEETING WITH SHARON
    if "analysis_parquet" not in (skip_outputs or []):
        logger.info("Making unified analysis Parquet files...")
        pqi.make_analysis_parquet(summary_obj, overwrite=overwrite)

    if "targeted_analysis_readme" not in (skip_outputs or []):
        logger.info("Making targeted analysis README...")
        make_targeted_analysis_readme(summary_obj, overwrite=overwrite)

    gdrive_upload = False
    if summary_obj.override_parameters.get("upload_to_gdrive") is not None:
        gdrive_upload = summary_obj.override_parameters["upload_to_gdrive"]
    elif summary_obj.ta.params.get("upload_to_gdrive") is not None:
        gdrive_upload = summary_obj.ta.params["upload_to_gdrive"]
    if gdrive_upload:
        logger.info("Uploading outputs to Google Drive...")
        gdu.copy_outputs_to_google_drive(summary_obj, "ANALYSIS_SUMMARY", overwrite=overwrite)

def apply_auto_curation_defaults(curation_df: "pd.DataFrame") -> "pd.DataFrame":
    """Apply default curation values to a curation DataFrame when skipping the manual GUI.

    Sets the following fields for every row:
    - ``analyst_notes``: appends ``" (manual curation skipped)"`` to any existing notes.
    - ``ms1_notes``: set to ``"keep"`` if currently empty/blank.
    - ``ms2_notes``: set to ``"0.5, curation skipped, putative match"`` if currently empty/blank.

    RT bounds come directly from the auto id routine

    Parameters
    ----------
    curation_df:
        The ``manual_curation`` DataFrame loaded from the project database
        (``summary_obj.experimental_data.curation_df``).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the three note columns updated in-place.
    """
    if curation_df is None or curation_df.empty:
        return curation_df

    _MS1_DEFAULT = "keep"
    _MS2_DEFAULT = "0.5, curation skipped, putative match"
    _ANALYST_SUFFIX = " (manual curation skipped)"

    # analyst_notes: append suffix when non-empty, otherwise use suffix alone.
    analyst = curation_df["analyst_notes"].fillna("").astype(str).str.strip()
    curation_df["analyst_notes"] = np.where(
        analyst != "",
        analyst + _ANALYST_SUFFIX,
        _ANALYST_SUFFIX.strip(),
    )

    # ms1_notes: fill only when currently empty/blank.
    ms1 = curation_df["ms1_notes"].fillna("").astype(str).str.strip()
    curation_df["ms1_notes"] = np.where(ms1 == "", _MS1_DEFAULT, ms1)

    # ms2_notes: fill only when currently empty/blank.
    ms2 = curation_df["ms2_notes"].fillna("").astype(str).str.strip()
    curation_df["ms2_notes"] = np.where(ms2 == "", _MS2_DEFAULT, ms2)

    logger.info(
        f"Applied auto-curation defaults to {len(curation_df)} compounds "
        f"(ms1_notes='{_MS1_DEFAULT}', ms2_notes='{_MS2_DEFAULT}', "
        f"analyst_notes suffix='{_ANALYST_SUFFIX.strip()}')."
    )
    return curation_df

def _strip_non_chars(text: str) -> str:
    """Remove Unicode non-characters (e.g. U+FFFE/FFFF) that DejaVu Sans cannot render."""
    return "".join(c for c in text if ord(c) not in range(0xFDD0, 0xFDF0) and ord(c) & 0xFFFF not in (0xFFFE, 0xFFFF))

def _make_compound_safe_stem(
    cmp_idx_display: int,
    compound_name: str,
    adduct: str,
    atlas_mz: float | None,
    atlas_rt: float | None,
) -> str:
    """Build a filesystem-safe stem for per-compound output files.

    Format: ``{idx:04d}_{name}_{adduct}_{mz}_{rt}``

    * Decimal points in mz/rt are replaced with ``p`` (e.g. ``153p1124``).
    * Special characters (brackets, spaces, slashes, commas, etc.) are stripped
      or replaced with underscores so the result is safe on all filesystems.
    * Consecutive underscores are collapsed to one.
    """
    def _fmt_num(val: float | None, decimals: int) -> str:
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return "NA"
        return f"{val:.{decimals}f}".replace(".", "p")

    # Strip special chars from compound name: keep alphanumeric, spaces→_, hyphens
    name_clean = re.sub(r"[^A-Za-z0-9\-]", "_", _strip_non_chars(str(compound_name)))
    # Strip adduct brackets and special chars: keep alphanumeric, +, -, letters
    adduct_clean = re.sub(r"[^A-Za-z0-9+\-]", "", str(adduct))
    mz_str = _fmt_num(atlas_mz, 4)
    rt_str = _fmt_num(atlas_rt, 2)
    stem = f"{cmp_idx_display:04d}_{name_clean}_{adduct_clean}_{mz_str}_{rt_str}"
    # Collapse consecutive underscores
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem

def _eic_subplot_label(filename: str) -> str:
    """Return a compact subplot label from a filename using fpf field names.

    Format: ``{polarity}_{chromatography}_{sample_number}_{sample_name}_{replicate}_{run_number}``

    Falls back gracefully when the filename does not match the expected pattern.
    """
    if not filename:
        return "no data"
    basename = os.path.basename(filename)
    try:
        fields = fpf.parse_file_name(basename)
        parts = [
            fields.get("polarity", ""),
            fields.get("chromatography", ""),
            fields.get("sample_number", ""),
            fields.get("sample_name", ""),
            fields.get("replicate", ""),
            fields.get("run_number", ""),
        ]
        return "_".join(p for p in parts if p)
    except Exception:
        # Fall back to the old abbreviated label
        return _short_fname(filename)

def _display_compound_idx(compound_idx: int) -> int:
    """Convert internal zero-based index to user-facing one-based index."""
    return int(compound_idx) + 1

def _resolve_summary_note_options(summary_obj: "AnalysisSummary") -> tuple[list[str], list[str]]:
    """Resolve MS1/MS2 option lists with the same owner/override logic as the GUI.

    Priority: override_parameters > config.gui_config > owner defaults.
    """
    owner = summary_obj.config.owner
    ms2_defaults, ms1_defaults, _ = get_notes_opts(owner=owner)

    _gui_cfg = summary_obj.config.gui_config if summary_obj.config else {}
    _config_note_overrides = _gui_cfg.get("note_options_overrides") or {}
    _param_note_overrides = (getattr(summary_obj, "override_parameters", {}) or {}).get("note_options_overrides") or {}
    note_overrides = {**_config_note_overrides, **_param_note_overrides}

    ms1_options, _ = get_note_options_and_hotkeys(note_overrides.get("ms1_notes", {}), ms1_defaults)
    ms2_options, _ = get_note_options_and_hotkeys(note_overrides.get("ms2_notes", {}), ms2_defaults)
    return ms1_options, ms2_options

def _validate_required_note_selections(summary_obj: "AnalysisSummary") -> None:
    """Raise if required GUI notes remain at unresolved defaults."""
    overrides = getattr(summary_obj, "override_parameters", {}) or {}
    _gui_cfg = summary_obj.config.gui_config if summary_obj.config else {}
    force_eval = (
        overrides["gui_require_all_evaluated"]
        if overrides.get("gui_require_all_evaluated") is not None
        else _gui_cfg.get("gui_require_all_evaluated", False)
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

#Identification Figure

def make_identification_figure(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: int | None = None,
) -> None:
    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / f"identification_figures_with_msms_mirror_plots"
    if overwrite and output_dir.exists():
        logger.info(f"Overwriting enabled: clearing existing contents of {output_dir}")
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info(f"Overwriting disabled: existing directory {output_dir} will be used.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting identification figures to {output_dir}")

    color_map = None
    if hasattr(summary_obj, "override_parameters") and summary_obj.override_parameters.get("gui_lcmsruns_colors"):
        color_map = summary_obj.override_parameters["gui_lcmsruns_colors"]
    elif hasattr(summary_obj, "config") and summary_obj.config and summary_obj.config.gui_config.get("gui_lcmsruns_colors"):
        color_map = summary_obj.config.gui_config["gui_lcmsruns_colors"]

    manual_curation_df = summary_obj.experimental_data.curation_df
    ms1_all_df = summary_obj.experimental_data.ms1_df_full_range
    if ms1_all_df is None or ms1_all_df.empty:
        ms1_all_df = summary_obj.experimental_data.ms1_df
    ms2_all_df = summary_obj.experimental_data.ms2_df

    total_files = ms1_all_df["filename"].nunique() if (ms1_all_df is not None and not ms1_all_df.empty) else 0
    logger.info(f"Plotting {len(manual_curation_df)} compounds across {total_files} files.")

    def _pregroup(df: pd.DataFrame | None) -> dict:
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
        atlas_mz = safe_float(mc_row.get("atlas_mz"))
        atlas_rt = safe_float(mc_row.get("atlas_rt_peak"))

        safe_name = _make_compound_safe_stem(cmp_idx_display, compound_name, adduct, atlas_mz, atlas_rt)
        fig_path = output_dir / f"{safe_name}.pdf"

        tasks.append({
            "mc_row_dict":   mc_row.to_dict(),
            "compound_name": compound_name,
            "adduct":        adduct,
            "inchi_key":     inchi_key,
            "compound_idx":  cmp_idx,
            "fig_path":      str(fig_path),
            "safe_stem":     safe_name,
            "color_map":     color_map,
            "ms1_df":        ms1_groups.get(mz_rt_uid, empty_df),
            "ms2_df":        ms2_groups.get(mz_rt_uid, empty_df),
        })

    if not tasks:
        logger.info("Nothing to generate.")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(f"Generating {len(tasks)} figures using {n_workers} workers...")

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
                logger.debug(f"Exported identification figure for {name}")
            except Exception as exc:
                logger.error(f"Failed to generate figure for {name}: {exc}")
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)
    pbar.close()
    logger.info("Identification figure export complete")

def _identification_figure_worker(kwargs: dict) -> str:
    """Worker: generate and save one identification figure PDF."""

    from metatlas2.workflow_objects import MS2Hit

    mc_row = pd.Series(kwargs["mc_row_dict"])
    fig_path = Path(kwargs["fig_path"])
    cmp_idx = kwargs["compound_idx"]
    cmp_idx_display = _display_compound_idx(cmp_idx)
    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    inchi_key = kwargs["inchi_key"]
    safe_stem = kwargs.get("safe_stem", f"{cmp_idx_display:04d}_{compound_name}_{adduct}")
    color_map = kwargs["color_map"]
    ms1_df = kwargs["ms1_df"]
    ms2_df = kwargs["ms2_df"]

    top3: list["MS2Hit"] = []
    if not ms2_df.empty:
        # Collect the best hit from each scan across all files, then take the
        # top 3 by score so the three mirror panels can show hits from
        # different scans / files rather than all coming from the same scan.
        per_scan_best: list["MS2Hit"] = []
        for _, scan_row in ms2_df.iterrows():
            hits = MS2Hit.list_from_scan_row(scan_row)
            if hits and hits[0].has_score:
                per_scan_best.append(hits[0])
        per_scan_best.sort(key=lambda h: h.score, reverse=True)
        top3 = per_scan_best[:3]

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
        mc_row.get("smiles"), mc_row.get("inchi"), inchi_key,
        compound_name=compound_name, size=500,
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

    # Smart-wrap the suptitle so it fits within the figure width
    try:
        renderer = fig.canvas.get_renderer()
        sample_t = fig.text(0, 0, "A" * 60, fontsize=16)
        bb = sample_t.get_window_extent(renderer=renderer)
        sample_t.remove()
        avg_char_w = bb.width / 60
        fig_w_px = fig.get_window_extent(renderer=renderer).width
        chars_per_line = max(20, int(fig_w_px / avg_char_w * 0.90))
    except Exception:
        chars_per_line = 80
    wrapped_suptitle = textwrap.fill(safe_stem, width=chars_per_line, break_long_words=True)
    fig.suptitle(
        wrapped_suptitle,
        fontsize=16, weight="bold", y=0.97,
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

def _plot_ms2(ax, panel_idx: int, top3: list["MS2Hit"], ms2_df: pd.DataFrame) -> None:
    """Populate one MS2 mirror panel."""
    _MIRROR_TITLES = ["Best MS2 Match", "2nd Best MS2 Match", "3rd Best MS2 Match"]
    title = _MIRROR_TITLES[panel_idx]

    if panel_idx < len(top3):
        hit = top3[panel_idx]
        _plot_mirror(
            ax,
            hit.aligned.query_mzs, hit.aligned.query_ints,
            hit.aligned.ref_mzs,   hit.aligned.ref_ints,
            frag_colors=hit.aligned.fragment_colors,
            score=hit.score,
            rt=hit.scan_rt,
            title=title,
        )
    elif panel_idx == 0 and not ms2_df.empty:
        # No hits at all — show raw spectrum from first scan
        raw_row = ms2_df.iloc[0]
        _plot_raw_ms2(
            ax,
            mz_arr=as_list(raw_row.get("frag_mzs", [])),
            int_arr=as_list(raw_row.get("frag_ints", [])),
            rt=float(raw_row.get("scan_rt", np.nan)),
            title=title,
        )
    else:
        _plot_empty_ms2(ax, title=title)

def _smart_wrap(text: str, ax, fontsize: float, fig=None) -> str:
    """Wrap *text* so each line fits within the axes width in display units.

    Uses the renderer to measure character width when available; falls back to
    an approximate character-count heuristic otherwise.

    Parameters
    ----------
    text:
        The string to wrap.
    ax:
        The matplotlib Axes whose width constrains the text.
    fontsize:
        Font size in points used for the text.
    fig:
        The Figure containing *ax* (used to get the renderer).  When *None*
        the current figure is used.
    """
    if fig is None:
        fig = ax.get_figure()

    try:
        renderer = fig.canvas.get_renderer()
        # Estimate average character width in display units using a sample string
        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        t_tmp = ax.text(0, 0, sample, fontsize=fontsize, transform=ax.transAxes)
        bb = t_tmp.get_window_extent(renderer=renderer)
        t_tmp.remove()
        avg_char_w = bb.width / len(sample)  # pixels per character

        # Axes width in display (pixel) units
        ax_bb = ax.get_window_extent(renderer=renderer)
        ax_w = ax_bb.width

        if avg_char_w > 0 and ax_w > 0:
            chars_per_line = max(10, int(ax_w / avg_char_w))
        else:
            chars_per_line = 40
    except Exception:
        # Fallback: rough heuristic — axes width in inches * ~8 chars/pt
        try:
            fig_w_in = fig.get_figwidth()
            n_axes_cols = max(1, round(1.0 / ax.get_position().width))
            ax_w_in = fig_w_in / n_axes_cols
            chars_per_line = max(10, int(ax_w_in * fontsize * 0.55))
        except Exception:
            chars_per_line = 40

    return textwrap.fill(text, width=chars_per_line, break_long_words=True)


def _plot_mirror(
    ax,
    qry_mz: list, qry_int: list,
    ref_mz: list, ref_int: list,
    frag_colors: list | None = None,
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
            if not safe_isnan(mz) and not safe_isnan(intensity):
                ax.bar(float(mz), float(intensity), color=color, width=1.1, alpha=0.85)

    scale = 1.0
    if ref_mz and ref_int:
        qry_int_valid = [v for v in qry_int if v is not None and not np.isnan(v)]
        ref_int_valid = [v for v in ref_int if v is not None and not np.isnan(v)]
        max_ref = max(ref_int_valid) if ref_int_valid else 0.0
        max_qry = max(qry_int_valid) if qry_int_valid else 0.0
        scale = (max_qry / max_ref) if max_ref > 0 else 1.0
        for mz, intensity, color in zip(ref_mz, ref_int, frag_colors):
            if not safe_isnan(mz) and not safe_isnan(intensity):
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
    ax.set_ylabel(
        f"Intensity",
        fontsize=14, weight="bold",
    )
    # "Ref Scaled xN" label: bottom-left corner, vertical, size 10
    ax.text(
        0.02, 0.04,
        f"Ref Scaled x{scale:.4f}",
        fontsize=10, weight="normal",
        ha="left", va="bottom",
        rotation=90,
        transform=ax.transAxes,
        color="dimgray",
    )
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
    color_map: dict | None = None,
) -> None:
    """Plot EIC traces for all files of one compound.

    Each row of ms1_compound_df holds the EIC for one file in wide format:
    spec_rts (list of RTs), spec_ints (list of intensities).
    """
    rt_min = mc_row.get("rt_min", np.nan)
    rt_max = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    if not safe_isnan(rt_min):
        ax.axvline(safe_float(rt_min), color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    if not safe_isnan(rt_max):
        ax.axvline(safe_float(rt_max), color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    if not safe_isnan(rt_peak):
        ax.axvline(safe_float(rt_peak), color="black", linestyle=":", linewidth=3, alpha=0.7)

    if ms1_compound_df.empty:
        ax.text(0.5, 0.5, "No MS1 data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
    else:
        for _, row in ms1_compound_df.iterrows():
            spec_rts = as_list(row.get("spec_rts", []))
            spec_ints = as_list(row.get("spec_ints", []))
            if not spec_rts or not spec_ints:
                continue
            color = _get_file_color(row.get("filename", ""), color_map)
            i_vals = np.log10(np.maximum(spec_ints, 1)) if log_scale else spec_ints
            ax.plot(
                spec_rts, 
                i_vals, 
                color=color, 
                linewidth=1.0, 
                alpha=0.25,
                )

    if not safe_isnan(rt_min) and not safe_isnan(rt_max):
        ax.set_xlim([safe_float(rt_min) - 1.0, safe_float(rt_max) + 1.0])
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
        if safe_isnan(val):
            return "N/A"
        try:
            return fmt.format(val) if fmt else str(val)
        except (ValueError, TypeError):
            return str(val)

    # Use top-3 MS1 averages — consistent with Final ID sheet ground-truth values
    atlas_mz   = safe_float(mc_row.get("atlas_mz"))
    mz_meas    = safe_float(mc_row.get("top3_mz_centroid_avg"))
    atlas_rt   = safe_float(mc_row.get("atlas_rt_peak"))
    rt_meas    = safe_float(mc_row.get("rt_peak"))

    if not safe_isnan(mz_meas) and not safe_isnan(atlas_mz) and atlas_mz != 0:
        mz_ppm_abs = abs(mz_meas - atlas_mz) / atlas_mz * 1e6
    else:
        mz_ppm_abs = float("nan")

    rt_delta_abs = abs(atlas_rt - rt_meas) if not (safe_isnan(atlas_rt) or safe_isnan(rt_meas)) else float("nan")

    best_intensity = safe_float(mc_row.get("best_ms1_intensity"))
    polarity = mc_row.get("polarity")
    if polarity == "neg":
        polarity = "negative"
    elif polarity == "pos":
        polarity = "positive"

    rows = [
        ("Compound", _fmt(mc_row.get("compound_name", "Undefined"))),
        ("InChIKey", _fmt(mc_row.get("inchi_key", ""))),
        ("Formula", _fmt(mc_row.get("formula"))),
        ("Adduct", _fmt(mc_row.get("adduct"))),
        ("Polarity", _fmt(polarity)),
        ("Chromatography", _fmt(mc_row.get("chromatography"))),
        ("Theoretical m/z", _fmt(atlas_mz, "{:.4f}")),
        ("Measured m/z", _fmt(mz_meas, "{:.4f}")),
        ("m/z ppm difference", _fmt(mz_ppm_abs, "{:.1f} ppm")),
        ("Theoretical RT range", f"{_fmt(mc_row.get('atlas_rt_min'), '{:.3f}')} - {_fmt(mc_row.get('atlas_rt_max'), '{:.3f}')} min"),
        ("Measured RT range", f"{_fmt(mc_row.get('rt_min'), '{:.3f}')} - {_fmt(mc_row.get('rt_max'), '{:.3f}')} min"),
        ("Theoretical RT", _fmt(atlas_rt, "{:.3f} min")),
        ("Measured RT", _fmt(rt_meas, "{:.3f} min")),
        ("RT difference", _fmt(rt_delta_abs, "{:.3f}")),
        ("Max Intensity", _fmt(best_intensity, "{:.3e}") if not safe_isnan(best_intensity) else "N/A"),
    ]

    y_pos = 1.065
    y_step = 0.09
    for label, value in rows:
        ax.text(0.02, y_pos, f"{label}:", fontsize=12, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.34, y_pos, value, fontsize=14, va="center", transform=ax.transAxes)
        y_pos -= y_step

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1)

def _plot_hit_info_table(
    ax,
    top3: list["MS2Hit"],
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

    raw_name = os.path.basename(best_hit.filename)

    atlas_mz = safe_float(mc_row.get("atlas_mz"))
    measured_mz = best_hit.mz_measured
    ppm_error = (
        (measured_mz - atlas_mz) / atlas_mz * 1e6
        if (not safe_isnan(measured_mz) and not safe_isnan(atlas_mz) and atlas_mz != 0)
        else np.nan
    )
    atlas_rt = safe_float(mc_row.get("atlas_rt_peak"))
    measured_rt = best_hit.scan_rt
    rt_error = (
        measured_rt - atlas_rt
        if (not safe_isnan(measured_rt) and not safe_isnan(atlas_rt))
        else np.nan
    )
    score = best_hit.score
    num_matches = best_hit.num_matches
    ref_frags = best_hit.ref_frags

    matched_frags = best_hit.matched_fragments
    frag_str = ", ".join(f"{m:.3f}" for m in matched_frags) if matched_frags else "N/A"

    def _v(val, fmt):
        if val is None or safe_isnan(val):
            return "N/A"
        return fmt.format(val)

    score_str = f"{score:.3f}\n({num_matches}/{ref_frags})" if not safe_isnan(score) else "N/A"

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
        second_file = os.path.basename(top3[1].filename)
        ax.text(0, additional_info_y, f"2nd best: {second_file}",
                fontsize=13, ha="left", va="top", transform=ax.transAxes)

    if len(top3) >= 3:
        third_file = os.path.basename(top3[2].filename)
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

def _get_file_color(filename: str, color_map: dict[str, str] | None = None) -> str:
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
    smiles: str | None,
    inchi: str | None,
    inchi_key: str,
    compound_name: str = "",
    size: int = 500,
) -> None:
    """Draw molecular structure using RDKit's MolToImage (requires libXrender, available in container)."""
    ax.axis("off")

    def _clean_chemical_text(value: str | None) -> str | None:
        """Return a usable string chemical identifier, else None."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return None
        return text

    def _wrap_text(text: str, font_size: int, max_width: int) -> list[str]:
        """Word-wrap *text* so each line fits within *max_width* pixels."""
        avg_char_width = font_size / 2
        lines, current_line = [], ""
        for char in text:
            test_line = current_line + char
            if len(test_line) * avg_char_width > max_width - 20:
                lines.append(current_line)
                current_line = char
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        return lines

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

            # Build annotation lines: compound name + InChIKey below the structure
            font_size = 22
            line_height = int(font_size * 1.3)

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)
                    font_bold = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                    font_bold = font

            # Collect (text, font) pairs for each annotation line
            annotation_segments: list[tuple[str, object]] = []
            if compound_name:
                for line in _wrap_text(_strip_non_chars(compound_name), font_size, size):
                    annotation_segments.append((line, font_bold))
            if inchi_key:
                for line in _wrap_text(inchi_key, font_size, size):
                    annotation_segments.append((line, font))

            if annotation_segments:
                text_height = len(annotation_segments) * line_height + 20
                new_img = Image.new("RGB", (size, size + text_height), "white")
                new_img.paste(img, (0, 0))
                draw_obj = ImageDraw.Draw(new_img)
                y_position = size + 10
                for line_text, line_font in annotation_segments:
                    draw_obj.text((10, y_position), line_text, fill="black", font=line_font)
                    y_position += line_height
                img = new_img

            ax.imshow(img, aspect="equal")
            return

    except Exception as exc:
        logger.warning(f"Could not draw structure for {inchi_key}: {exc}")

    ax.text(0.5, 0.5, f"{_strip_non_chars(compound_name)}\n{inchi_key}", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, weight="bold", color="gray")

###############################################
#### Final ID Sheet
###############################################

def make_final_id_sheet(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:

    from metatlas2.workflow_objects import MS2Hit

    final_id_filename_part = (
        f"{summary_obj.chromatography}"
        f"-{summary_obj.polarity}"
        f"-{summary_obj.ta.atlas_name}"
        f"-RTA{summary_obj.rt_alignment_number}"
        f"-TGA{summary_obj.analysis_number}"
    )
    output_filename = f"FinalIdentifications__{final_id_filename_part}__{summary_obj.project_name}.xlsx"
    excel_path = Path(summary_obj.paths['analysis_results_output_dir']) / output_filename
    if not overwrite and excel_path.exists():
        logger.info(f"Overwriting disabled: existing file {excel_path} will be used.")
        return

    manual_curation_df = summary_obj.experimental_data.curation_df

    compound_info_map: dict[str, tuple] = {}
    mass_map: dict[str, float | None] = {}
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

    ms2_best: dict[str, dict] = {}

    ms2_df = summary_obj.experimental_data.ms2_df
    if not ms2_df.empty:
        for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
            best_score = -np.inf
            best_hit: "MS2Hit" | None = None

            for _, scan_row in grp.iterrows():
                hits = MS2Hit.list_from_scan_row(scan_row)
                if hits and hits[0].has_score and hits[0].score > best_score:
                    best_score = hits[0].score
                    best_hit = hits[0]

            if best_hit is not None:
                ms2_best[uid] = best_hit

    is_c18 = "c18" in summary_obj.chromatography.lower() and "lipid" not in summary_obj.chromatography.lower()

    logger.info(f"Processing {len(manual_curation_df)} compounds...")

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
        if polarity.lower() == "pos":
            polarity_long = "positive"
        elif polarity.lower() == "neg":
            polarity_long = "negative"
        else:
            raise ValueError(f"Unknown polarity when long-ifying polarity string {polarity}")

        formula, smiles, inchi, pubchem_cid = compound_info_map.get(inchi_key, (None, None, None, None))
        exact_mass = mass_map.get(inchi_key)

        overlapping_compound, overlapping_mz_rt_uids = overlapping_map.get(compound_idx, ("", ""))
        identified_metabolite = compound_name if not overlapping_compound else overlapping_compound

        # --- Shared theoretical values ---
        mz_theoretical = float(mc_row.get("atlas_mz", np.nan))
        rt_theoretical = float(mc_row.get("atlas_rt_peak", np.nan))

        # --- MS1 metrics ---
        mz_measured = float(mc_row.get("top3_mz_centroid_avg", np.nan))
        rt_measured  = float(mc_row.get("rt_peak", np.nan))
        best_ms1_rt = float(mc_row.get("best_ms1_rt", np.nan))
        best_ms1_intensity = float(mc_row.get("best_ms1_intensity", np.nan))
        _raw = mc_row.get("best_ms1_file", "")
        best_ms1_file = "" if (not _raw or (isinstance(_raw, float) and np.isnan(_raw))) else str(_raw)

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
        msms_matching_ions = ""

        best_hit = ms2_best.get(mz_rt_uid)
        if best_hit is not None:
            msms_file = best_hit.filename
            msms_rt = best_hit.scan_rt
            msms_score = best_hit.score

            num_matches = best_hit.num_matches
            ref_frags = best_hit.ref_frags

            matched_frags = best_hit.matched_fragments
            msms_matching_ions = (
                ",".join(f"{m:.3f}" for m in matched_frags) if matched_frags else "N/A"
            )

            # Single-ion match precursor check
            if num_matches == 1 and best_hit.has_score and matched_frags:
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
        mz_q = mz_quality(ppm_error_ms1, mz_error_da)
        rt_q = rt_quality(rt_error, summary_obj.chromatography)

        try:
            msms_q = float(str(ms2_notes).split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = total_score_and_msi(msms_q, mz_q, rt_q)

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
            "polarity":               polarity_long,
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
    logger.info(f"Assembled final ID table with {len(final_df)} rows.")

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
        "msms_file", "msms_rt", "msms_matchingions",
        "msms_score",
        "mz_adduct", "mz_theoretical", "mz_measured",
        "mz_error", "mz_ppmerror",
        "rt_min", "rt_max", "rt_theoretical", "rt_measured", "rt_error",
    ]

    _SECTION_SPANS = [
        (0,  8,  "COMPOUND ANNOTATION"),
        (9,  19, "COMPOUND IDENTIFICATION SCORES"),
        (20, 22, "MS1 INTENSITY INFORMATION"),
        (23, 25, "MSMS INFORMATION"),
        (26, 26, "MSMS EVALUATION"),
        (27, 29, "ION INFORMATION"),
        (30, 31, "M/Z EVALUATION"),
        (32, 35, "CHROMATOGRAPHIC PEAK INFORMATION"),
        (36, 36, "RT EVALUATION"),
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

        # Build one header format and one data format per section color
        section_header_fmts: dict[str, object] = {}
        section_data_fmts: dict[str, object] = {}
        for label, color in _SECTION_COLORS.items():
            section_header_fmts[label] = workbook.add_format({**f_header_base, "bg_color": color})
            section_data_fmts[label] = workbook.add_format({"bg_color": color})

        # ---- Row heights ----
        worksheet.set_row(0, 30)    # section headers
        worksheet.set_row(1, 80)    # column display names
        worksheet.set_row(2, 80)   # descriptions
        worksheet.set_row(3, 30)    # internal field names

        # ---- Column widths ----
        # Default width for all columns; override specifics below
        worksheet.set_column(0,  0,  10)   # index
        worksheet.set_column(1,  4,  10)   # compound name columns + overlapping
        worksheet.set_column(5,  8,  10)   # formula, polarity, exact_mass, inchi_key
        worksheet.set_column(8,  8,  10)   # inchi_key wider
        worksheet.set_column(9,  19, 10)   # scores + notes
        worksheet.set_column(20, 20, 10, f_scientific)  # max_intensity in sci notation
        worksheet.set_column(21, 21, 10)   # max intensity filename
        worksheet.set_column(22, 22, 10)   # ms1 rt peak
        worksheet.set_column(23, 23, 10)   # msms filename
        worksheet.set_column(24, 24, 10)   # msms rt
        worksheet.set_column(25, 25, 10)   # msms num ions
        worksheet.set_column(26, 26, 10)   # msms matching ions
        worksheet.set_column(27, 27, 10)   # msms score
        worksheet.set_column(28, 30, 10)   # ion info
        worksheet.set_column(31, 32, 10)   # mz eval
        worksheet.set_column(33, 37, 10)   # rt cols

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

        # ---- Conditional background colors for data rows ----
        for start_idx, end_idx, label in _SECTION_SPANS:
            fmt = section_data_fmts[label]
            start_letter = _col_letter(start_idx)
            end_letter = _col_letter(end_idx)
            worksheet.conditional_format(
                f"{start_letter}5:{end_letter}{nrows}",
                {"type": "no_errors", "format": fmt},
            )

    logger.info(f"Exported final ID table to {excel_path}")

def mz_quality(ppm_error: float, mz_delta: float) -> float:
    """Return 0/0.5/1 mz quality score from ppm and absolute mass error."""
    if np.isnan(ppm_error) or np.isnan(mz_delta):
        return np.nan
    if abs(ppm_error) <= 5 or abs(mz_delta) <= 0.0015:
        return 1.0
    if abs(ppm_error) <= 10:
        return 0.5
    return 0.0

def rt_quality(rt_error: float, chromatography: str) -> float:
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

def total_score_and_msi(
    msms_q: float, mz_q: float, rt_q: float
) -> tuple[float, str]:
    """Compute total ID score (0-3) and MSI level string.
    """
    scores = [v for v in [msms_q, mz_q, rt_q] if not np.isnan(v)]
    total = float(np.nansum([msms_q, mz_q, rt_q]))

    if msms_q == -1:
        msi = "REMOVE, INVALIDATED BY BAD MSMS MATCH"
    elif len(scores) == 0:
        # All three quality scores are NaN — cannot determine MSI level
        msi = "putative"
    elif statistics.median(scores) < 1:
        msi = "putative"
    elif total == 3:
        msi = "Exceeds Level 1"
    else:
        msi = "Level 1"
    return total, msi

def _compute_all_overlapping_compounds(
    manual_curation_df: pd.DataFrame,
    mass_map: dict[str, float | None],
) -> dict[int, tuple[str, str]]:
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

    result: dict[int, tuple[str, str]] = {}
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
    max_workers: int | None = None,
) -> None:
    """Generate per-compound EIC thumbnail PDFs in two output folders (parallelised)."""

    rt_alignment_num = summary_obj.rt_alignment_number
    analysis_num = summary_obj.analysis_number

    output_loc = Path(summary_obj.paths['analysis_results_output_dir'])
    base_dir = Path(output_loc)
    additional_info_dir = base_dir / "additional_info"
    dir_shared = additional_info_dir / f"compound_EIC_chromatograms_sharedY"
    dir_indep = additional_info_dir / f"compound_EIC_chromatograms_independentY"
    for d in (dir_shared, dir_indep):
        if overwrite and d.exists():
            logger.info(f"Overwriting enabled: clearing existing contents of {d}")
            shutil.rmtree(d)
        elif not overwrite and d.exists():
            logger.info(
                "Overwriting disabled: existing directory %s will be used "
                "(existing PDFs will be preserved).",
                d,
            )
            return
        logger.info(f"Creating directory {d}")
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
    logger.info(f"Building task list for {n_compounds} compounds...")

    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")
        inchi_key = mc_row.get("inchi_key", "")
        atlas_mz = safe_float(mc_row.get("atlas_mz"))
        atlas_rt = safe_float(mc_row.get("atlas_rt_peak"))

        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        safe_stem = _make_compound_safe_stem(cmp_idx_display, compound_name, adduct, atlas_mz, atlas_rt)
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
                    "group_name": dbi._file_group_from_filename(f_path),
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
            "safe_stem": safe_stem,
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
                logger.debug(f"Wrote EIC PDFs for {name}")
            except Exception as exc:
                logger.error(f"Failed to write PDFs for {name}: {exc}")
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
    safe_stem = kwargs.get("safe_stem", f"{_display_compound_idx(kwargs['compound_idx']):04d}_{compound_name}_{adduct}")
    rt_alignment_num = kwargs["rt_alignment_num"]
    analysis_num = kwargs["analysis_num"]
    compound_idx = kwargs["compound_idx"]

    rt_min = mc_row.get("rt_min", np.nan)
    rt_max_val = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    all_intensities = [
        v
        for item in file_items
        for v in as_list(item.get("spec_ints"))
        if v is not None and not np.isnan(v)
    ]
    y_max_global = float(max(all_intensities)) if all_intensities else None

    total_files = len(file_items)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)
    title_base = safe_stem

    group_bg_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]

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
                left=0.07, right=0.98, top=0.92, bottom=0.08, hspace=0.45, wspace=0.3
            )
            axes_flat = axes.flatten()

            active_axes = []
            for slot_idx, item in enumerate(page_items):
                ax = axes_flat[slot_idx]
                fname_s = _strip_non_chars(_eic_subplot_label(item["filename"]))
                spec_ints = as_list(item.get("spec_ints"))
                valid_ints = [v for v in spec_ints if v is not None and not np.isnan(v)]
                y_max_indep = float(max(valid_ints)) if valid_ints else None
                group_name = item.get("group_name")
                _render_eic_thumbnail(
                    ax,
                    as_list(item.get("spec_rts")),
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
            fig.text(0.5, 0.01, "RT (min)", ha="center", va="bottom", fontsize=11)
            fig.text(0.01, 0.5, "Intensity", ha="left", va="center", fontsize=11, rotation=90)
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
    spec_rts: list,
    spec_ints: list,
    rt_min: float,
    rt_peak: float,
    rt_max: float,
    fname_short: str,
    y_max: float | None,
    bg_color: str | None = None,
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
        ax.patch.set_alpha(0.2)

    if spec_rts and spec_ints:
        ax.plot(spec_rts, spec_ints, color="black", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # RT vlines: rt_min = red dashed, rt_peak = black dotted, rt_max = black dashed
    _vline = lambda val, color, lw, ls: (
        ax.axvline(val, color=color, linewidth=lw, linestyle=ls)
        if val is not None and not np.isnan(val)
        else None
    )
    _vline(rt_min, "red", 0.8, "--")
    _vline(rt_peak, "black", 1.5, ":" )
    _vline(rt_max, "red", 0.8, "--")

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
    
    # Smart-wrap the subplot title to fit within the axes width
    wrapped_title = _smart_wrap(fname_short, ax, fontsize=10)
    ax.set_title(wrapped_title + "\n", fontsize=10, pad=2.5, loc="center")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

###############################################
#### Sample TIC Chromatograms
###############################################

def make_tic_thumbnails(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: int | None = None,
) -> None:
    """Generate per-sample Total Ion Chromatogram (TIC) thumbnail PDFs.

    Produces two files in ``additional_info/``:

    * ``sample_TIC_total_chromatograms_independentY.pdf`` — each subplot
      auto-scales its y-axis independently.
    * ``sample_TIC_total_chromatograms_sharedY.pdf`` — all subplots share
      the same y-axis maximum (the global maximum across all files).

    Subplot titles use the same ``_eic_subplot_label()`` format as the EIC
    thumbnails (``polarity_chrom_sample#_groupname_replicate#_Run###``).

    The TIC is built from ``experimental_data.ms1_df_full_range`` when
    available, falling back to ``experimental_data.ms1_df``.
    """
    output_loc = Path(summary_obj.paths["analysis_results_output_dir"])
    additional_info_dir = output_loc / "additional_info"
    path_indep  = additional_info_dir / "sample_TIC_total_chromatograms_independentY.pdf"
    path_shared = additional_info_dir / "sample_TIC_total_chromatograms_sharedY.pdf"

    if not overwrite and path_indep.exists() and path_shared.exists():
        logger.info("Overwriting disabled: existing TIC PDFs will be used.")
        return

    additional_info_dir.mkdir(parents=True, exist_ok=True)

    # Use full-range MS1 data when available (wider RT window)
    ms1_df = summary_obj.experimental_data.ms1_df_full_range
    if ms1_df is None or ms1_df.empty:
        ms1_df = summary_obj.experimental_data.ms1_df

    if ms1_df is None or ms1_df.empty:
        logger.warning("No MS1 data available — TIC thumbnails not generated.")
        return

    logger.info("Building TIC traces from MS1 data...")
    tic_df = dbi.build_tic_df(ms1_df)

    if tic_df.empty:
        logger.warning("TIC DataFrame is empty — TIC thumbnails not generated.")
        return

    logger.info(f"Generating TIC thumbnails for {len(tic_df)} files...")

    _GRID_COLS = 5
    _GRID_ROWS = 5
    _PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS

    # Sort files using the same alphanum logic as EIC thumbnails
    def _alphanum_key(text: str) -> list:
        return [
            int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", text or "")
        ]

    tic_df = tic_df.sort_values(
        "filename",
        key=lambda s: s.map(lambda f: _alphanum_key(Path(f).stem)),
    ).reset_index(drop=True)

    # Build group → color map (same palette as EIC thumbnails)
    _TIC_BG_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#17becf",
    ]
    tic_group_to_color: dict[str, str] = {}
    for fn in tic_df["filename"]:
        grp = dbi._file_group_from_filename(str(fn))
        if grp not in tic_group_to_color:
            tic_group_to_color[grp] = _TIC_BG_PALETTE[len(tic_group_to_color) % len(_TIC_BG_PALETTE)]

    # Global y-max for shared-scale mode
    all_ints = [v for row in tic_df["tic_ints"] for v in (row or []) if v is not None and not np.isnan(v)]
    y_max_global = float(max(all_ints)) if all_ints else None

    total_files = len(tic_df)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)

    pdf_indep  = PdfPages(path_indep)
    pdf_shared = PdfPages(path_shared)
    try:
        for page_idx in range(total_pages):
            start = page_idx * _PLOTS_PER_PAGE
            end   = min(start + _PLOTS_PER_PAGE, total_files)
            page_rows = tic_df.iloc[start:end]
            n_on_page = len(page_rows)
            page_label = f"({page_idx + 1}/{total_pages})" if total_pages > 1 else ""

            fig, axes = plt.subplots(_GRID_ROWS, _GRID_COLS, figsize=(20, 16))
            fig.subplots_adjust(
                left=0.07, right=0.98, top=0.92, bottom=0.08, hspace=0.45, wspace=0.3
            )
            axes_flat = axes.flatten()

            active_axes = []
            for slot_idx, (_, row) in enumerate(page_rows.iterrows()):
                ax = axes_flat[slot_idx]
                filename  = row["filename"]
                tic_rts   = as_list(row.get("tic_rts",  []))
                tic_ints  = as_list(row.get("tic_ints", []))
                fname_s   = _strip_non_chars(_eic_subplot_label(filename))

                # Background color by file group (same as EIC thumbnails)
                grp = dbi._file_group_from_filename(str(filename))
                bg_color = tic_group_to_color.get(grp)
                if bg_color:
                    ax.set_facecolor(bg_color)
                    ax.patch.set_alpha(0.2)

                if tic_rts and tic_ints:
                    ax.plot(tic_rts, tic_ints, color="black", linewidth=0.8)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="gray")

                fmt = ScalarFormatter(useMathText=True)
                fmt.set_scientific(True)
                fmt.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(fmt)
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.tick_params(axis="both", labelsize=10, length=2, pad=1)
                ax.xaxis.label.set_visible(False)
                ax.yaxis.label.set_visible(False)
                # Smart-wrap the TIC subplot title to fit within the axes width
                wrapped_tic_title = _smart_wrap(fname_s, ax, fontsize=10)
                ax.set_title(wrapped_tic_title + "\n", fontsize=10, pad=2.5, loc="center")
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

                active_axes.append(ax)

            for slot_idx in range(n_on_page, _PLOTS_PER_PAGE):
                axes_flat[slot_idx].set_visible(False)

            fig.suptitle(f"Sample TIC Chromatograms  {page_label}\n", fontsize=12, y=1.005)
            fig.text(0.5, 0.01, "RT (min)", ha="center", va="bottom", fontsize=11)
            fig.text(0.01, 0.5, "Intensity", ha="left", va="center", fontsize=11, rotation=90)

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
        pdf_indep.close()
        pdf_shared.close()

    logger.info(f"TIC thumbnails written to {additional_info_dir}")

###############################################
#### Compound Peak Height / RT Peak Grid Plots
###############################################

def _run_number_sort_key(filename: str) -> int:
    """Return the numeric run number from a filename for sorting, or sys.maxsize on failure."""
    try:
        rn = fpf.get_file_parts(filename, "run_number")
        # run_number may be "Run95" or "95"
        return int(re.sub(r"[^0-9]", "", rn))
    except Exception:
        return sys.maxsize


def _make_compound_subplot(
    ax,
    filenames: list[str],
    values: list[float],
    compound_title: str,
    ylabel: str,
    atlas_ref: float | None = None,
) -> None:
    """Draw one compound subplot: x = file (ordered by run#), y = metric value.

    Parameters
    ----------
    filenames:
        Ordered list of file basenames (stems) for x-axis labels.
    values:
        Corresponding metric values (NaN for missing).
    compound_title:
        Subplot title (safe_stem format).
    ylabel:
        Y-axis label string.
    atlas_ref:
        When provided, draw a horizontal red line at this value (theoretical RT).
    """
    x_pos = list(range(len(filenames)))
    vals  = np.array(values, dtype=float)

    # Scatter dots; NaN values are simply not plotted
    valid = np.isfinite(vals)
    if valid.any():
        ax.scatter(
            np.array(x_pos)[valid], vals[valid],
            s=20, color="black", alpha=0.85, zorder=3,
        )

    if atlas_ref is not None and not np.isnan(atlas_ref):
        ax.axhline(atlas_ref, color="red", linewidth=1.0, linestyle="--", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(filenames, rotation=90, ha="center", va="top", fontsize=7)
    ax.tick_params(axis="x", pad=2, length=2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    # Smart-wrap the title to fit within the subplot width
    wrapped_title = _smart_wrap(compound_title, ax, fontsize=8)
    ax.set_title(wrapped_title, fontsize=8, pad=2, loc="center", fontweight="bold")

    sci_fmt = ScalarFormatter(useMathText=True)
    sci_fmt.set_scientific(True)
    sci_fmt.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(sci_fmt)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def make_compound_metric_plots(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Generate two multi-compound grid PDFs in ``additional_info/``.

    * ``compound_peak_heights.pdf`` — one subplot per compound, x = file
      (ordered by run number, vertical label), y = peak height.
    * ``compound_rt_peaks.pdf`` — same layout, y = RT peak, with a red
      dashed horizontal line at the theoretical (atlas) RT.

    Each subplot title uses the same safe-stem format as the ID figures
    (``{idx:04d}_{name}_{adduct}_{mz}_{rt}``).

    Compounds are laid out in a 5x5 grid per page.  Files are sorted by
    their ``run_number`` field (parsed from the filename).
    """
    output_loc = Path(summary_obj.paths["analysis_results_output_dir"])
    additional_info_dir = output_loc / "additional_info"
    path_ph = additional_info_dir / "compound_peak_heights.pdf"
    path_rt = additional_info_dir / "compound_rt_peaks.pdf"

    if not overwrite and path_ph.exists() and path_rt.exists():
        logger.info("Overwriting disabled: existing compound metric PDFs will be used.")
        return

    additional_info_dir.mkdir(parents=True, exist_ok=True)

    manual_curation_df = summary_obj.experimental_data.curation_df
    per_file_df = summary_obj.per_file_metrics_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.warning("No curation data — compound metric plots not generated.")
        return
    if per_file_df is None or per_file_df.empty:
        logger.warning("No per-file metrics — compound metric plots not generated.")
        return

    # Build ordered list of all files sorted by run number
    all_files_sorted: list[str] = sorted(
        per_file_df["filename"].dropna().unique().tolist(),
        key=_run_number_sort_key,
    )

    _DEFINED_FILE_GROUPS: frozenset[str] = frozenset({"EXCTRL", "INJBL", "ISTD", "QC", "REFSTD"})

    def _run_label(filename: str) -> str:
        try:
            run_num = fpf.get_file_parts(filename, "run_number")
        except Exception:
            run_num = os.path.basename(filename)
        file_group = dbi._file_group_from_filename(filename)
        if file_group in _DEFINED_FILE_GROUPS:
            sample_group = file_group
        else:
            try:
                sample_group = fpf.get_file_parts(filename, "sample_name")
            except Exception:
                sample_group = file_group
        return f"{sample_group}_{run_num}"

    x_labels = [_run_label(f) for f in all_files_sorted]
    file_to_idx = {f: i for i, f in enumerate(all_files_sorted)}

    # Build per-compound data
    mc_reset = manual_curation_df.reset_index(drop=True)
    compounds: list[dict] = []
    for cmp_idx, mc_row in mc_reset.iterrows():
        cmp_idx_display = _display_compound_idx(cmp_idx)
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct        = mc_row.get("adduct", "")
        atlas_mz      = safe_float(mc_row.get("atlas_mz"))
        atlas_rt      = safe_float(mc_row.get("atlas_rt_peak"))
        mz_rt_uid     = mc_row.get("mz_rt_uid", "")

        title = _make_compound_safe_stem(cmp_idx_display, compound_name, adduct, atlas_mz, atlas_rt)

        # Gather per-file values for this compound
        cmp_rows = per_file_df[per_file_df["mz_rt_uid"] == mz_rt_uid]
        ph_vals = [np.nan] * len(all_files_sorted)
        rt_vals = [np.nan] * len(all_files_sorted)
        for _, row in cmp_rows.iterrows():
            fn = row.get("filename", "")
            idx = file_to_idx.get(fn)
            if idx is not None:
                ph_vals[idx] = float(row.get("peak_height", np.nan))
                rt_vals[idx] = float(row.get("rt_peak",     np.nan))

        compounds.append({
            "title":    title,
            "ph_vals":  ph_vals,
            "rt_vals":  rt_vals,
            "atlas_rt": atlas_rt,
        })

    if not compounds:
        logger.warning("No compounds to plot — compound metric plots not generated.")
        return

    _GRID_COLS = 5
    _GRID_ROWS = 5
    _PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS
    n_compounds  = len(compounds)
    total_pages  = max(1, (n_compounds + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)

    # Figure width scales with number of files so x-axis labels stay readable
    n_files    = len(all_files_sorted)
    subplot_w  = max(2.5, min(8.0, n_files * 0.18))
    fig_width  = subplot_w * _GRID_COLS
    fig_height = 4.5 * _GRID_ROWS

    logger.info(
        f"Generating compound metric plots: {n_compounds} compounds x {n_files} files, "
        f"{total_pages} page(s)..."
    )

    pdf_ph = PdfPages(path_ph)
    pdf_rt = PdfPages(path_rt)
    try:
        for page_idx in range(total_pages):
            start = page_idx * _PLOTS_PER_PAGE
            end   = min(start + _PLOTS_PER_PAGE, n_compounds)
            page_compounds = compounds[start:end]
            n_on_page = len(page_compounds)
            page_label = f"({page_idx + 1}/{total_pages})" if total_pages > 1 else ""

            for pdf_obj, metric_key, ylabel, use_atlas_rt in [
                (pdf_ph, "ph_vals", "Peak Height",   False),
                (pdf_rt, "rt_vals", "RT Peak (min)", True),
            ]:
                fig, axes = plt.subplots(
                    _GRID_ROWS, _GRID_COLS,
                    figsize=(fig_width, fig_height),
                )
                fig.subplots_adjust(
                    left=0.06, right=0.98, top=0.90, bottom=0.12,
                    hspace=0.90, wspace=0.35,
                )
                axes_flat = axes.flatten()

                for slot_idx, cmp in enumerate(page_compounds):
                    ax = axes_flat[slot_idx]
                    atlas_ref = cmp["atlas_rt"] if use_atlas_rt else None
                    _make_compound_subplot(
                        ax,
                        x_labels,
                        cmp[metric_key],
                        cmp["title"],
                        ylabel,
                        atlas_ref=atlas_ref,
                    )

                for slot_idx in range(n_on_page, _PLOTS_PER_PAGE):
                    axes_flat[slot_idx].set_visible(False)

                title_str = (
                    "Compound Peak Heights" if metric_key == "ph_vals"
                    else "Compound RT Peaks"
                )
                fig.suptitle(f"{title_str}  {page_label}\n", fontsize=13, y=1.0)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Glyph \d+ .* missing from font",
                        category=UserWarning,
                    )
                    pdf_obj.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    finally:
        pdf_ph.close()
        pdf_rt.close()

    logger.info(
        f"Compound metric plots written to {additional_info_dir}: "
        f"compound_peak_heights.pdf and compound_rt_peaks.pdf"
    )

###############################################
#### Compound Boxplots
###############################################

def _sort_boxplot_groups(groups: list[str]) -> list[str]:
    _CONTROL_GROUPS_FIRST: frozenset[str] = frozenset({"EXCTRL", "INJBL", "ISTD", "QC"})
    _CONTROL_GROUPS_LAST:  frozenset[str] = frozenset({"REFSTD"})

    first  = sorted(g for g in groups if g in _CONTROL_GROUPS_FIRST)
    last   = sorted(g for g in groups if g in _CONTROL_GROUPS_LAST)
    middle = sorted(g for g in groups if g not in _CONTROL_GROUPS_FIRST and g not in _CONTROL_GROUPS_LAST)
    return first + middle + last

def _build_group_color_map(groups: list[str]) -> dict[str, str]:
    
    _BOXPLOT_DOT_COLORS: list[str] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]    

    return {g: _BOXPLOT_DOT_COLORS[i % len(_BOXPLOT_DOT_COLORS)] for i, g in enumerate(groups)}

def _plot_compound_boxplot(
    ax,
    compound_metrics: pd.DataFrame,
    metric: str,
    log_scale: bool,
    atlas_ref: float | None,
    compound_name: str,
    adduct: str,
    ylabel: str,
    compound_idx: int = 0,
    all_groups: list[str] | None = None,
    group_color_map: dict[str, str] | None = None,
    compound_safe_stem: str | None = None,
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
        Atlas reference value used only for the ``mz_centroid`` subplot title
        (no reference line is drawn on the plot).
    compound_name, adduct:
        Used as the subplot title.
    ylabel:
        Y-axis label string.
    all_groups:
        The complete ordered list of file_group labels across all files in the
        analysis (control groups first, RefStd last).  When provided every group
        always appears on the x-axis; groups with no data are shown as blank
        columns.  When *None* only groups that have data are shown.
    group_color_map:
        Mapping of group label → dot color.  When *None* all dots are gray.
    compound_safe_stem:
        When provided, used as the subplot title instead of the default
        ``{idx}  {compound_name}  {adduct}`` format.
    """
    rng = np.random.default_rng(seed=42)

    if all_groups is not None:
        groups = all_groups
    elif not compound_metrics.empty:
        groups = _sort_boxplot_groups(
            compound_metrics["file_group"].dropna().unique().tolist()
        )
    else:
        groups = []

    data_per_group: list[list[float]] = []
    real_vals_all: list[float] = []   # only genuine data values, used for y-axis scaling
    for g in groups:
        if compound_metrics.empty:
            vals: list[float] = []
        else:
            group_rows = compound_metrics.loc[compound_metrics["file_group"] == g]
            vals = group_rows[metric].dropna().tolist()
            if log_scale:
                vals = [np.log10(v) for v in vals if v is not None and v > 0]
        if vals:
            real_vals_all.extend(vals)
        data_per_group.append(vals)

    def _make_title(wrapped: bool = False) -> str:
        title_top = compound_safe_stem if compound_safe_stem else f"{_display_compound_idx(compound_idx):04d}  {compound_name}  {adduct}"
        if wrapped:
            title_top = _smart_wrap(title_top, ax, fontsize=13)
        if metric in ("mz_centroid",):
            mz_val = f"{atlas_ref:.4f}" if (atlas_ref is not None and not np.isnan(atlas_ref)) else 'N/A'
            return f"{title_top}\nTheoretical m/z: {mz_val}"
        if metric in ("rt_peak",):
            rt_val = f"{atlas_ref:.3f} min" if (atlas_ref is not None and not np.isnan(atlas_ref)) else 'N/A'
            return f"{title_top}\nTheoretical RT: {rt_val}"
        return title_top

    if not groups:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(_make_title(wrapped=True), fontsize=13, pad=2, loc="center", fontweight="bold")
        return

    positions = list(range(len(groups)))
    boxplot_data = [vals if vals else [np.nan] for vals in data_per_group]

    # Box fill is white, lines are gray — color comes from the scatter dots only.
    ax.boxplot(
        boxplot_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="#888888", linewidth=0.8),
        medianprops=dict(color="#444444", linewidth=1.5),
        whiskerprops=dict(color="#888888", linewidth=0.8),
        capprops=dict(color="#888888", linewidth=0.8),
        showfliers=False,
    )

    for pos, g, vals in zip(positions, groups, data_per_group):
        if not vals:
            continue  # no real data for this group — leave the slot blank
        dot_color = (group_color_map or {}).get(g, "#888888")
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.array([pos] * len(vals), dtype=float) + jitter,
            vals,
            s=18, color=dot_color, alpha=0.75, zorder=3,
        )

    # Fit y-axis tightly to the actual data values only (excludes placeholder NaNs)
    if real_vals_all:
        data_min = min(real_vals_all)
        data_max = max(real_vals_all)
        data_range = data_max - data_min if data_max != data_min else abs(data_max) * 0.1 or 1.0
        ax.set_ylim(data_min - data_range * 0.1, data_max + data_range * 0.15)

    # Axis formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=90, ha="center", va="top", fontsize=10)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(_make_title(wrapped=True), fontsize=13, pad=2, loc="center", fontweight="bold")

    # Smart y-axis formatting: place a "1×10^X" offset label at the top of the
    # y-axis and show tick labels with enough decimal places to differentiate
    # adjacent ticks.  This handles cases where the data range is very narrow
    # (e.g. 1.205–1.209 or 130.0012–130.0013) where a plain ScalarFormatter
    # would collapse all ticks to the same displayed value.
    import matplotlib.ticker as mticker

    if real_vals_all:
        finite_vals = [v for v in real_vals_all if np.isfinite(v)]
        if finite_vals:
            data_min_v = min(finite_vals)
            data_max_v = max(finite_vals)
            data_range_v = data_max_v - data_min_v

            abs_max_v = max(abs(v) for v in finite_vals)

            if abs_max_v > 0:
                # Exponent for the offset label (based on the magnitude of the values)
                exp = int(np.floor(np.log10(abs_max_v)))
                scale = 10 ** exp

                # Determine how many decimal places are needed so adjacent ticks
                # are distinguishable after dividing by scale.
                # We aim for at least 2 significant digits in the range.
                if data_range_v > 0:
                    # Number of decimals needed to show the range with ~3 sig figs
                    range_scaled = data_range_v / scale
                    if range_scaled > 0:
                        decimals_needed = max(1, -int(np.floor(np.log10(range_scaled))) + 2)
                    else:
                        decimals_needed = 4
                else:
                    decimals_needed = 4
                decimals_needed = min(decimals_needed, 8)  # cap at 8 decimal places

                fmt_str = f"{{:.{decimals_needed}f}}"
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _, _s=scale, _f=fmt_str: _f.format(v / _s))
                )
                # Place "×10^exp" text at the top of the y-axis spine
                ax.text(
                    -0.01, 1.01,
                    f"×10$^{{{exp}}}$",
                    transform=ax.transAxes,
                    fontsize=9, ha="right", va="bottom",
                )
            else:
                sci_fmt = ScalarFormatter(useMathText=True)
                sci_fmt.set_scientific(True)
                sci_fmt.set_powerlimits((-3, 4))
                ax.yaxis.set_major_formatter(sci_fmt)
        else:
            sci_fmt = ScalarFormatter(useMathText=True)
            sci_fmt.set_scientific(True)
            sci_fmt.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(sci_fmt)
    else:
        sci_fmt = ScalarFormatter(useMathText=True)
        sci_fmt.set_scientific(True)
        sci_fmt.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(sci_fmt)

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
    all_groups = kwargs.get("all_groups")       # global ordered group list
    group_color_map = kwargs.get("group_color_map")  # group → dot color

    atlas_mz = safe_float(atlas_ref_dict.get("atlas_mz"))
    atlas_rt = safe_float(atlas_ref_dict.get("atlas_rt_peak"))
    safe_stem = _make_compound_safe_stem(
        _display_compound_idx(cmp_idx), compound_name, adduct, atlas_mz, atlas_rt
    )

    # Scale figure width with the number of groups so x-axis labels stay readable.
    n_groups = len(all_groups) if all_groups else 1
    fig_width = max(10, min(80, n_groups * 0.35))

    for metric, log_scale, ylabel, atlas_attr, metric_dir_str in metric_configs:
        pdf_path = Path(metric_dir_str) / f"{safe_stem}.pdf"
        atlas_ref = atlas_ref_dict.get(atlas_attr) if atlas_attr else None
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        fig.subplots_adjust(bottom=0.25, left=0.12, right=0.97, top=0.88)
        try:
            _plot_compound_boxplot(
                ax, cmp_metrics, metric, log_scale,
                atlas_ref, compound_name, adduct, ylabel,
                compound_idx=cmp_idx,
                all_groups=all_groups,
                group_color_map=group_color_map,
                compound_safe_stem=safe_stem,
            )
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
        finally:
            plt.close(fig)

    return compound_name

def make_boxplots(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    max_workers: int | None = None,
) -> None:

    base_dir = Path(summary_obj.paths['analysis_results_output_dir'])
    additional_info_dir = base_dir / "additional_info"

    dir_peak_height = base_dir / f"boxplot_peak_height"
    dir_peak_height_log = base_dir / f"boxplot_peak_height_log"
    dir_rt_peak = additional_info_dir / f"boxplot_rt_peak"
    dir_mz_centroid = additional_info_dir / f"boxplot_mz_centroid"
    dir_mz_centroid_log = additional_info_dir / f"boxplot_mz_centroid_log"
    dir_rt_peak_log = additional_info_dir / f"boxplot_rt_peak_log"

    all_metric_dirs = [
        dir_peak_height, dir_peak_height_log,
        dir_rt_peak, dir_rt_peak_log,
        dir_mz_centroid, dir_mz_centroid_log,
    ]

    if overwrite:
        for d in all_metric_dirs:
            if d.exists():
                logger.info(f"Overwriting enabled: clearing existing contents of {d}")
                shutil.rmtree(d)
    else:
        # If any top-level boxplot dir exists, skip entirely
        if any(d.exists() for d in all_metric_dirs):
            logger.info("Overwriting disabled: existing boxplot directories will be used (existing PDFs will be preserved).")
            return

    for d in all_metric_dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting boxplot figures to {base_dir} and {additional_info_dir}")

    manual_curation_df = summary_obj.experimental_data.curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning("per_file_metrics_df is None or empty — boxplots will show no data.")
        return

    # Build atlas lookup keyed by mz_rt_uid (the true unique compound identifier)
    atlas_lookup: dict[str, dict[str, float]] = {
        mc.get("mz_rt_uid", ""): {
            "atlas_mz": float(mc.get("atlas_mz", np.nan)),
            "atlas_rt_peak": float(mc.get("atlas_rt_peak", np.nan)),
        }
        for _, mc in manual_curation_df.iterrows()
    }

    # Create metric output dirs upfront, store paths as strings
    _METRIC_CONFIGS = [
        ("peak_height", False, "Peak Height (intensity)", None, dir_peak_height),
        ("peak_height", True,  "Peak Height (log₁₀)", None, dir_peak_height_log),
        ("rt_peak", False, "RT Peak (mins)", "atlas_rt_peak", dir_rt_peak),
        ("rt_peak", True, "RT Peak (log₁₀)", "atlas_rt_peak", dir_rt_peak_log),
        ("mz_centroid", False, "m/z Centroid", "atlas_mz", dir_mz_centroid),
        ("mz_centroid", True, "m/z Centroid (log₁₀)", "atlas_mz", dir_mz_centroid_log),
    ]
    metric_configs_with_dirs: list[tuple] = []
    for metric, log_scale, ylabel, atlas_attr, metric_dir in _METRIC_CONFIGS:
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
            f"No mz_rt_uid overlap between curation_df and per_file_metrics_df! "
            f"Sample curation UIDs: {list(mc_uids)[:3]} | Sample per_file UIDs: {list(pf_uids)[:3]}"
        )

    raw_groups: list[str] = (
        per_file_df["file_group"].dropna().unique().tolist()
        if "file_group" in per_file_df.columns else []
    )
    all_groups: list[str] = _sort_boxplot_groups(raw_groups)
    group_color_map: dict[str, str] = _build_group_color_map(all_groups)

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
            "all_groups": all_groups,
            "group_color_map": group_color_map,
            # Pass atlas values so the worker can build the safe_stem filename
            "atlas_mz":  float(mc_row.get("atlas_mz", np.nan)),
            "atlas_rt":  float(mc_row.get("atlas_rt_peak", np.nan)),
        })

    if not tasks:
        logger.info("Nothing to generate (all boxplot PDFs already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(f"Generating boxplot PDFs for {len(tasks)} compounds ({len(_METRIC_CONFIGS)} metric types) using {n_workers} workers...")

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
                logger.debug(f"Wrote boxplot PDFs for {name}")
            except Exception as exc:
                logger.error(f"Failed to generate boxplots for {name}: {exc}")
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)

    pbar.close()

    logger.info("Boxplot PDFs complete")

###############################################
#### Table Exporters
###############################################

def make_best_ms2_hit_fragment_ions_csv(
    summary_obj: "AnalysisSummary",
    output_filename: str = "best_ms2_hit_fragment_ions.csv",
    overwrite: bool = True,
    min_fragment_intensity: float | None = 1e4,
) -> None:

    from metatlas2.workflow_objects import MS2Hit

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / f"data_sheets"
    output_file = output_dir / output_filename
    if not overwrite and output_file.exists():
        logger.info(f"Overwriting disabled: existing file {output_file} will be used.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting best MS2 hit fragment ions CSV to {output_file}")

    if summary_obj.experimental_data.curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.experimental_data.curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - best MS2 hit CSV not written.")
        return

    ms2_df = summary_obj.experimental_data.ms2_df
    ms2_best: dict[str, "MS2Hit"] = {}
    if ms2_df is not None and not ms2_df.empty:
        for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
            best_score = -np.inf
            best_hit: "MS2Hit" | None = None
            for _, scan_row in grp.iterrows():
                hits = MS2Hit.list_from_scan_row(scan_row)
                if hits and hits[0].has_score and hits[0].score > best_score:
                    best_score = hits[0].score
                    best_hit = hits[0]
            if best_hit is not None:
                ms2_best[uid] = best_hit

    mc_reset = manual_curation_df.reset_index(drop=True)

    rows: list[dict] = []
    for cmp_idx, mc_row in tqdm(
        mc_reset.iterrows(),
        total=len(mc_reset),
        desc="Finding best MS2 hits for compounds",
        disable=should_disable_tqdm(),
    ):
        cmp_idx_display = _display_compound_idx(cmp_idx)
        mz_rt_uid = mc_row.get("mz_rt_uid", "")
        compound_name = mc_row.get("compound_name", "Undefined")
        adduct = mc_row.get("adduct", "")
        atlas_mz = safe_float(mc_row.get("atlas_mz"))
        atlas_rt = safe_float(mc_row.get("atlas_rt_peak"))

        # Build the same compound_id as make_data_sheets
        compound_id = _make_compound_safe_stem(
            cmp_idx_display, compound_name, adduct, atlas_mz, atlas_rt
        )

        best_hit = ms2_best.get(mz_rt_uid)
        if best_hit is None:
            continue

        q_mz_all = np.array(best_hit.aligned.query_mzs, dtype=float)
        q_int_all = np.array(best_hit.aligned.query_ints, dtype=float)

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

        # Full spectrum JSON (retained for backward compatibility)
        raw_spectrum_output = json.dumps([mz_vals, int_vals])

        # Separate arrays rounded to 4 decimal places
        mz_4dp  = [round(v, 4) for v in mz_vals]
        int_4dp = [round(v, 4) for v in int_vals]

        rows.append({
            "compound_id":     compound_id,
            "filename":        os.path.basename(str(best_hit.filename)),
            "rt_peak":         best_hit.scan_rt,
            "mz_peak":         best_hit.mz_measured,
            "spectrum":        raw_spectrum_output,
            "mz_array":        json.dumps(mz_4dp),
            "intensity_array": json.dumps(int_4dp),
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

    output_dir = Path(summary_obj.paths['analysis_results_output_dir']) / f"data_sheets"
    if overwrite and output_dir.exists():
        logger.info(f"Overwriting enabled: clearing existing contents of {output_dir}")
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info(
            "Overwriting disabled: existing directory %s will be used "
            "(existing CSVs will be preserved).",
            output_dir,
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting data sheets to {output_dir}")

    manual_curation_df = summary_obj.experimental_data.curation_df
    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - data sheets not written.")
        return

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning(
            "per_file_metrics_df is empty — no MS1 data was found for any compound in this "
            "analysis. Data sheets will be written with all compounds present but all metric "
            "values will be NaN (no detected signal)."
        )
        per_file_df = pd.DataFrame(columns=["mz_rt_uid", "filename", "inchi_key", "adduct",
                                             "peak_height", "peak_area", "rt_peak",
                                             "rt_centroid", "mz_peak", "mz_centroid"])

    mc_reset = manual_curation_df.reset_index(drop=True)
    mc_slim = mc_reset[["mz_rt_uid", "compound_name", "inchi_key", "adduct",
                         "atlas_mz", "atlas_rt_peak"]].copy()
    mc_slim["_cmp_idx_display"] = mc_reset.index + 1
    mc_slim = mc_slim.drop_duplicates(subset=["mz_rt_uid"])

    # Build single compound_id column using the same safe-stem format as ID figures
    mc_slim["compound_id"] = mc_slim.apply(
        lambda r: _make_compound_safe_stem(
            int(r["_cmp_idx_display"]),
            r["compound_name"],
            r["adduct"],
            safe_float(r.get("atlas_mz")),
            safe_float(r.get("atlas_rt_peak")),
        ),
        axis=1,
    )

    per_file_merge = per_file_df.drop(
        columns=[c for c in ["inchi_key", "adduct"] if c in per_file_df.columns]
    )
    pfm = mc_slim.merge(per_file_merge, on="mz_rt_uid", how="left")

    pfm["_file_col"] = pfm["filename"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
        if (p and not (isinstance(p, float) and np.isnan(p)))
        else None
    )

    # Collect ordered file columns (stems) for header row generation
    all_file_cols: list[str] = []
    seen: set[str] = set()
    for fc in pfm["_file_col"].dropna():
        if fc not in seen:
            all_file_cols.append(fc)
            seen.add(fc)

    # Build the 6 header rows for each file column
    analyst = getattr(summary_obj, "analyst", "") or ""
    rta_num = getattr(summary_obj, "rt_alignment_number", None)
    tga_num = getattr(summary_obj, "analysis_number", None)

    header_rows: dict[str, list] = {
        "group":            [],
        "file":             [],
        "short_groupname":  [],
        "sample_treatment": [],
        "short_filename":   [],
        "short_samplename": [],
    }
    for fc in all_file_cols:
        labels = fpf.get_filename_header_labels(
            fc,
            analyst=analyst,
            rt_alignment_number=rta_num,
            analysis_number=tga_num,
        )
        for key in header_rows:
            header_rows[key].append(labels[key])

    _DATA_SHEET_METRICS = [
        "peak_height",
        "peak_area",
        "rt_peak",
        "rt_centroid",
        "mz_peak",
        "mz_centroid",
    ]
    _INDEX_COL = "compound_id"

    for metric in _DATA_SHEET_METRICS:
        if metric not in pfm.columns:
            logger.warning(
                "Metric '%s' not found in per_file_metrics_df — skipping.", metric
            )
            continue

        csv_path = output_dir / f"{metric}.csv"

        pfm_with_file = pfm[pfm["_file_col"].notna()]
        wide = (
            pfm_with_file[[_INDEX_COL, "_file_col", metric]]
            .pivot_table(
                index=_INDEX_COL,
                columns="_file_col",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None

        # Ensure all file columns are present (even if no data for some compounds)
        for fc in all_file_cols:
            if fc not in wide.columns:
                wide[fc] = np.nan
        # Reorder: compound_id first, then file cols in original order
        wide = wide[[_INDEX_COL] + [fc for fc in all_file_cols if fc in wide.columns]]

        # Add compounds with no data at all
        no_data_compounds = mc_slim[~mc_slim["mz_rt_uid"].isin(pfm_with_file["mz_rt_uid"])][
            [_INDEX_COL]
        ].copy()
        if not no_data_compounds.empty:
            for fc in all_file_cols:
                no_data_compounds[fc] = np.nan
            wide = pd.concat([wide, no_data_compounds], ignore_index=True)

        # Re-sort to match original curation_df order
        order_map = {row[_INDEX_COL]: i for i, row in mc_slim[[_INDEX_COL]].iterrows()}
        wide["_sort_order"] = wide[_INDEX_COL].map(order_map).fillna(len(order_map))
        wide = wide.sort_values("_sort_order").drop(columns=["_sort_order"]).reset_index(drop=True)

        # Write CSV with 6 header rows prepended
        sample_cols = [c for c in wide.columns if c != _INDEX_COL]
        header_df = pd.DataFrame(
            {_INDEX_COL: list(header_rows.keys())}
            | {fc: header_rows[key][i] for i, fc in enumerate(all_file_cols) for key in []}  # placeholder
        )
        # Build header rows properly
        header_records = []
        for key in header_rows:
            row = {_INDEX_COL: key}
            for i, fc in enumerate(all_file_cols):
                row[fc] = header_rows[key][i]
            header_records.append(row)
        header_df = pd.DataFrame(header_records, columns=[_INDEX_COL] + all_file_cols)

        # Align header_df columns to wide columns
        for col in wide.columns:
            if col not in header_df.columns:
                header_df[col] = ""
        header_df = header_df[[c for c in wide.columns if c in header_df.columns]]

        combined = pd.concat([header_df, wide], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        logger.info(
            f"Exported {metric} data sheet ({len(wide)} compounds x {len(sample_cols)} files, "
            f"with 6 header rows)"
        )

    logger.info(f"Data sheets written to {output_dir}")

def _parse_peak_height_csv(
    csv_path: Path,
    n_header_rows: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a peak_height-style CSV produced by :func:`make_data_sheets`.

    The on-disk layout is::

        Row 0      : column-name row (compound_id, <file_stem_1>, <file_stem_2>, ...)
        Row 1      : metadata row 1  (group, <group_1>, <group_2>, ...)
        Row 2      : metadata row 2  (file, <file_1.h5>, ...)
        Row 3      : metadata row 3  (short_groupname, ...)
        Row 4      : metadata row 4  (sample_treatment, ...)
        Row 5      : metadata row 5  (short_filename, ...)
        Row 6      : metadata row 6  (short_samplename, ...)
        Row 7+     : compound data rows

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    n_header_rows:
        Number of metadata rows that follow the column-name row (default 6).

    Returns
    -------
    header_df : pd.DataFrame
        The ``n_header_rows`` metadata rows read with ``header=None``.
        Column positions (0..N) align with the column-name row positions
        (position 0 = compound_id column, 1..N = file-stem columns).
    data_df : pd.DataFrame
        The compound data rows with the column-name row (row 0) used as the
        pandas column header.
    """
    # Row 0 is the column-name row; rows 1..n_header_rows are metadata.
    # Read n_header_rows+1 rows total, drop row 0, keep rows 1..n_header_rows.
    raw = pd.read_csv(csv_path, nrows=n_header_rows + 1, header=None)
    header_df = raw.iloc[1:].reset_index(drop=True)
    header_df.columns = range(len(raw.columns))

    # Skip rows 1..n_header_rows (metadata) so row 0 becomes the pandas header.
    data_df = pd.read_csv(csv_path, skiprows=list(range(1, n_header_rows + 1)))

    return header_df, data_df


def _build_col_to_sample_treatment(
    df: pd.DataFrame,
    header_df: pd.DataFrame,
    id_cols: set[str],
    sample_treatment_row_idx: int = 3,
) -> dict[str, str]:
    """Map each sample column name to its ``sample_treatment`` value.

    Parameters
    ----------
    df:
        The data DataFrame whose columns are used for the mapping.
    header_df:
        The metadata header DataFrame returned by :func:`_parse_peak_height_csv`.
        Column positions (integers 0..N) align with ``df.columns`` positions.
    id_cols:
        Set of column names that are identifier columns (not sample data).
    sample_treatment_row_idx:
        0-based row index within *header_df* for the ``sample_treatment`` row
        (default 3, which is the 4th metadata row).

    Returns
    -------
    dict mapping column name → sample_treatment string (non-empty, non-NaN).
    """
    treatment_row = header_df.iloc[sample_treatment_row_idx].tolist()
    col_names = df.columns.tolist()
    mapping: dict[str, str] = {}
    for i, col in enumerate(col_names):
        if col in id_cols:
            continue
        if i < len(treatment_row):
            val = str(treatment_row[i]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                mapping[col] = val
    return mapping


def make_peak_height_filtered_csv(
    obj: "AnalysisSummary",
    overwrite: bool = True,
    control_fold_threshold: float = 3.0,
) -> None:
    """Filter and impute the ``peak_height`` data sheet.

    Reads ``peak_height.csv`` (produced by :func:`make_data_sheets`) and writes
    two output files:

    ``peak_height_filter_summary.tsv``
        Two-column TSV (``compound_id``, ``extraction_control_filter``) recording whether
        each compound was kept or removed by the ExCtrl fold-change filter.
        All compounds appear here regardless of filter outcome.

    ``peak_height_filtered.csv``
        Identical format to ``peak_height.csv`` (same 6 prepended metadata
        header rows, same column layout) but with:

        * "remove" compound rows dropped.
        * Remaining NaN cells imputed with the global minimum non-NaN value
          across all sample columns (the "lowest" rule), computed from the
          full unfiltered matrix so the floor is consistent.

    **Filtering rule** — for each compound row:

    * Identify *ExCtrl* columns via the ``sample_treatment`` metadata header row.
    * Compute the mean peak height across all ExCtrl columns for that compound.
    * A compound is **kept** if at least one non-ExCtrl sample column has a
      value ≥ ``control_fold_threshold x mean_exctrl``.
    * If no ExCtrl columns are present all compounds are kept.
    """
    _N_HEADER_ROWS = 6
    _EXCTRL_TREATMENT = "ExCtrl"
    _META_COLS: set[str] = {"compound_id"}

    out_dir = Path(obj.paths["analysis_results_output_dir"]) / "data_sheets"
    source_csv = out_dir / "peak_height.csv"
    output_csv = out_dir / "peak_height_filtered.csv"
    summary_tsv = out_dir / "peak_height_filter_summary.tsv"

    if not source_csv.exists():
        logger.error(
            "peak_height.csv not found at %s — run make_data_sheets first.", source_csv
        )
        return
    if not overwrite and output_csv.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_csv)
        return

    logger.info("Creating filtered peak height CSV at %s", output_csv)

    header_df, df = _parse_peak_height_csv(source_csv, n_header_rows=_N_HEADER_ROWS)

    # Coerce all sample columns to numeric
    sample_cols = [c for c in df.columns if c not in _META_COLS]
    for col in sample_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Identify ExCtrl vs non-ExCtrl columns via sample_treatment header row
    col_to_treatment = _build_col_to_sample_treatment(df, header_df, _META_COLS)
    exctrl_cols = [
        c for c in sample_cols
        if col_to_treatment.get(c, "").strip() == _EXCTRL_TREATMENT
    ]
    non_exctrl_cols = [c for c in sample_cols if c not in exctrl_cols]

    logger.info(
        "Control filter: %d ExCtrl columns, %d non-ExCtrl columns.",
        len(exctrl_cols), len(non_exctrl_cols),
    )

    # Apply filter: keep compound if any non-ExCtrl value >= threshold * mean(ExCtrl)
    if not exctrl_cols:
        logger.info("No ExCtrl columns found — all compounds marked 'keep'.")
        keep_mask = pd.Series(True, index=df.index)
    else:
        mean_exctrl = df[exctrl_cols].mean(axis=1)  # per-compound mean of ExCtrl values
        if non_exctrl_cols:
            max_non_exctrl = df[non_exctrl_cols].max(axis=1)
            keep_mask = (
                max_non_exctrl.notna()
                & (mean_exctrl.isna() | (max_non_exctrl >= control_fold_threshold * mean_exctrl.fillna(0)))
            )
        else:
            keep_mask = pd.Series(False, index=df.index)

    n_keep = int(keep_mask.sum())
    logger.info(
        "Control filter result: %d keep, %d remove (of %d total compounds).",
        n_keep, len(df) - n_keep, len(df),
    )

    # Write filter summary TSV (all compounds, keep/remove label)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "compound_id":    df["compound_id"],
        "extraction_control_filter": keep_mask.map({True: "keep", False: "remove"}),
    }).to_csv(summary_tsv, sep="\t", index=False)
    logger.info("Wrote filter summary TSV to %s", summary_tsv)

    if df.empty:
        logger.warning("No compounds found — peak_height_filtered.csv not written.")
        return

    # Keep only 'keep' rows
    df_keep = df[keep_mask].reset_index(drop=True)

    global_min_val = float(df[sample_cols].min().min()) if sample_cols else np.nan
    if not np.isnan(global_min_val):
        n_nan = int(df_keep[sample_cols].isna().sum().sum())
        df_keep[sample_cols] = df_keep[sample_cols].fillna(global_min_val)
        logger.info("Imputed %d NaN cells with global minimum: %g", n_nan, global_min_val)
    else:
        logger.warning("Global minimum is NaN — no imputation performed.")

    with open(output_csv, "w", newline="") as fh:
        fh.write(",".join(str(c) for c in df_keep.columns) + "\n")
        header_df.to_csv(fh, index=False, header=False)
        df_keep.to_csv(fh, index=False, header=False)

    logger.info(
        "Exported filtered peak height CSV (%d compounds x %d sample columns).",
        len(df_keep), len(sample_cols),
    )


def make_log_fold_changes_csv(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Create group-level log₂ fold-change CSV from ``peak_height_filtered.csv``.

    ``peak_height_filtered.csv`` has the same format as ``peak_height.csv``
    (6 prepended metadata header rows, ``compound_id`` as the first data
    column, no ``extraction_control_filter`` column) — it contains only the compounds
    that passed the ExCtrl filter.

    Sample columns are grouped by their ``sample_treatment`` value (read from
    the 4th metadata header row, 0-based index 3).  ExCtrl columns are
    **excluded** from all fold-change comparisons.  Within each remaining group
    the mean of replicate values is computed, and pairwise log₂ fold-changes
    are calculated between every pair of groups.

    Output column layout::

        compound_id, <GroupA>_vs_<GroupB>, ...

    """
    _N_HEADER_ROWS = 6
    _EXCTRL_TREATMENT = "ExCtrl"
    _ID_COLS: set[str] = {"compound_id"}

    out_dir = Path(summary_obj.paths["analysis_results_output_dir"]) / "data_sheets"
    source_csv = out_dir / "peak_height_filtered.csv"
    output_csv = out_dir / "group_mean_log_fold_changes.csv"

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

    header_df, df = _parse_peak_height_csv(source_csv, n_header_rows=_N_HEADER_ROWS)

    id_cols = [c for c in ["compound_id"] if c in df.columns]
    sample_cols = [c for c in df.columns if c not in _ID_COLS]

    if not sample_cols:
        logger.warning("No sample columns found in %s; writing identifiers only.", source_csv)
        df[id_cols].to_csv(output_csv, index=False)
        return

    data_num = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    # Map each sample column to its sample_treatment group; exclude ExCtrl
    col_to_treatment = _build_col_to_sample_treatment(df, header_df, _ID_COLS)
    group_to_cols: dict[str, list[str]] = {}
    skipped: list[str] = []
    for col in data_num.columns:
        treatment = col_to_treatment.get(col)
        if treatment is None:
            skipped.append(col)
        elif treatment == _EXCTRL_TREATMENT:
            pass  # exclude ExCtrl from LFC comparisons
        else:
            group_to_cols.setdefault(treatment, []).append(col)

    if skipped:
        logger.warning(
            "%d sample column(s) had no sample_treatment assignment and will be skipped: %s",
            len(skipped), skipped[:10],
        )

    exctrl_excluded = sum(
        1 for col in data_num.columns
        if col_to_treatment.get(col) == _EXCTRL_TREATMENT
    )
    if exctrl_excluded:
        logger.info("Excluded %d ExCtrl column(s) from LFC comparisons.", exctrl_excluded)

    if len(group_to_cols) < 2:
        logger.warning(
            "Fewer than 2 non-ExCtrl sample groups found in %s "
            "(groups: %s) — no fold-change comparisons possible.",
            source_csv, list(group_to_cols.keys()),
        )
        df[id_cols].to_csv(output_csv, index=False)
        return

    logger.info(
        "Computing pairwise log2 fold changes for %d groups: %s",
        len(group_to_cols), list(group_to_cols.keys()),
    )

    # Per-group mean across replicates
    group_means: dict[str, np.ndarray] = {
        grp: data_num[cols].mean(axis=1).to_numpy(dtype=float)
        for grp, cols in group_to_cols.items()
    }

    # Build output DataFrame: compound_id + pairwise LFC columns
    lfc_records: dict[str, object] = {col: df[col].to_numpy() for col in id_cols}
    n_comparisons = 0
    for g1, g2 in itertools.combinations(group_means.keys(), 2):
        m1, m2 = group_means[g1], group_means[g2]
        valid = (m1 > 0) & (m2 > 0) & np.isfinite(m1) & np.isfinite(m2)
        res = np.full(len(m1), np.nan)
        res[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res
        n_comparisons += 1

    lfc_df = pd.DataFrame(lfc_records)

    # # Prepend compound metadata columns (inchi_key + all DB fields) before LFC columns
    # main_db_path = getattr(summary_obj, "paths", {}).get("main_db_path")
    # if main_db_path:
    #     # Build inchi_key lookup from curation_df (compound_id → inchi_key)
    #     curation_df = summary_obj.experimental_data.curation_df
    #     if curation_df is not None and not curation_df.empty:
    #         mc_slim = mc_reset = curation_df.reset_index(drop=True)
    #         mc_slim = mc_reset[["mz_rt_uid", "compound_name", "inchi_key", "adduct",
    #                              "atlas_mz", "atlas_rt_peak"]].copy()
    #         mc_slim["_cmp_idx_display"] = mc_reset.index + 1
    #         mc_slim = mc_slim.drop_duplicates(subset=["mz_rt_uid"])
    #         mc_slim["compound_id"] = mc_slim.apply(
    #             lambda r: _make_compound_safe_stem(
    #                 int(r["_cmp_idx_display"]),
    #                 r["compound_name"],
    #                 r["adduct"],
    #                 safe_float(r.get("atlas_mz")),
    #                 safe_float(r.get("atlas_rt_peak")),
    #             ),
    #             axis=1,
    #         )
    #         cid_to_inchikey = mc_slim.set_index("compound_id")["inchi_key"].to_dict()
    #         lfc_df["inchi_key"] = lfc_df["compound_id"].map(cid_to_inchikey).fillna("")
    #         inchi_keys = lfc_df["inchi_key"].dropna().unique().tolist()
    #         meta_df = dbi.get_compound_metadata(main_db_path, inchi_keys)
    #         if not meta_df.empty:
    #             lfc_df = lfc_df.merge(meta_df, on="inchi_key", how="left")
    #             # Reorder: compound_id first, then all metadata cols, then LFC cols
    #             lfc_cols = [c for c in lfc_df.columns if "_vs_" in c]
    #             meta_cols = [c for c in meta_df.columns if c != "inchi_key"]
    #             ordered_cols = (
    #                 ["compound_id", "inchi_key"]
    #                 + meta_cols
    #                 + lfc_cols
    #             )
    #             lfc_df = lfc_df[[c for c in ordered_cols if c in lfc_df.columns]]

    lfc_df.to_csv(output_csv, index=False)
    logger.info(
        "Exported log fold changes CSV (%d compounds x %d group comparisons).",
        len(df), n_comparisons,
    )

def make_metabomap(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Merge POS and NEG ``peak_height_filtered.csv`` files into a single
    metabomap table and compute group-level log2 fold-changes.

    For each unique ``compound_id`` the row with the highest peak height
    across non-control sample columns is selected, regardless of polarity.
    The chosen polarity is recorded as a ``chosen_polarity`` column.

    Sample columns are then grouped by the ``sample_name`` field parsed from
    the column header and the mean of each group's replicates is used to
    compute pairwise log2 fold-changes.

    The input CSVs have 6 prepended header rows which are skipped before
    reading.

    Outputs
    -------
    ``<analysis_output_dir>/../metabomap/merged_peak_heights.tsv``
        One row per ``compound_id`` with columns:
        ``compound_id, chosen_polarity, <sample_file_1>, ...``

    ``<analysis_output_dir>/../metabomap/group_mean_log_fold_changes.tsv``
        One row per ``compound_id`` with columns:
        ``compound_id, <GroupA>_vs_<GroupB>, ...``
    """
    analysis_output_dir = Path(summary_obj.paths.get("analysis_results_output_dir"))
    curr_pol_lc = fpf.normalize_polarity(summary_obj.polarity)   # "pos" or "neg"
    sib_pol_lc  = "neg" if curr_pol_lc == "pos" else "pos"
    curr_pol_display = summary_obj.polarity
    sib_pol_display  = fpf.canonical_polarity(sib_pol_lc)
    if curr_pol_display.isupper():
        sib_pol_display = sib_pol_lc.upper()
    elif curr_pol_display.islower():
        sib_pol_display = sib_pol_lc
    else:
        sib_pol_display = sib_pol_lc.title()

    current_csv = analysis_output_dir / f"data_sheets" / "peak_height_filtered.csv"
    sibling_csv = (
        Path(str(analysis_output_dir).replace(f"-{curr_pol_display}-", f"-{sib_pol_display}-"))
        / "data_sheets" / "peak_height_filtered.csv"
    )

    metabomaps_dir = analysis_output_dir / "../metabomap"
    merged_tsv = metabomaps_dir / "merged_peak_heights.tsv"
    lfc_tsv    = metabomaps_dir / "group_mean_log_fold_changes.tsv"

    if not overwrite and merged_tsv.exists() and lfc_tsv.exists():
        return

    if not current_csv.exists() or not sibling_csv.exists():
        logger.warning("Missing polarity CSVs; skipping metabomap.")
        return

    metabomaps_dir.mkdir(parents=True, exist_ok=True)

    _N_HEADER_ROWS = 6
    _META_COLS = {"compound_id"}
    _EXCLUDE_GROUPS = ("QC", "ISTD", "ISTDsEtc")

    def _load_filtered(path: Path, polarity_label: str) -> pd.DataFrame:
        _, raw = _parse_peak_height_csv(path, n_header_rows=_N_HEADER_ROWS)
        raw["_polarity"] = polarity_label
        return raw

    cur_df = _load_filtered(current_csv, curr_pol_lc)
    sib_df = _load_filtered(sibling_csv, sib_pol_lc)

    # Stack both polarities
    combined = pd.concat([cur_df, sib_df], axis=0, ignore_index=True)

    # Identify sample columns (present in either polarity)
    all_sample_cols = [
        c for c in combined.columns
        if c not in _META_COLS and c != "_polarity"
    ]

    # Select best row per compound_id (highest non-control peak)
    _CTRL_TYPES = {"ExCtrl", "InjBl", "QC", "ISTD", "ISTDsEtc"}
    non_ctrl_sample_cols = [
        c for c in all_sample_cols
        if not any(t in c for t in _CTRL_TYPES)
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
        subset=["compound_id"], keep="first"
    ).reset_index(drop=True)

    logger.info(
        "Metabomap: selected best row for %d unique compound_id values "
        "from %d combined rows (%s + %s).",
        len(best), len(combined), curr_pol_display, sib_pol_display,
    )

    best["chosen_polarity"] = best["_polarity"]

    # Build the merged peak-heights table
    present_sample_cols = [c for c in all_sample_cols if c in best.columns]
    merged_data_df = best[
        ["compound_id", "chosen_polarity"] + present_sample_cols
    ].copy()

    merged_data_df.to_csv(merged_tsv, sep="\t", index=False)
    logger.info(f"Wrote merged_peak_heights.tsv ({len(merged_data_df)} compounds x {len(present_sample_cols)} sample columns)")

    # Group sample columns by sample_name field and compute LFC
    _LFC_META = {"compound_id", "chosen_polarity"}
    sample_data_cols = [c for c in merged_data_df.columns if c not in _LFC_META]

    group_to_cols: dict[str, list[str]] = {}
    for col in sample_data_cols:
        try:
            grp = fpf.get_file_parts(col, "sample_name")
            if grp is not None and not any(ex in grp for ex in _EXCLUDE_GROUPS):
                group_to_cols.setdefault(grp, []).append(col)
        except Exception:
            pass

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

    group_means: dict[str, np.ndarray] = {
        grp: data_num[cols].mean(axis=1).to_numpy(dtype=float)
        for grp, cols in group_to_cols.items()
    }

    lfc_records: dict[str, object] = {
        "compound_id": merged_data_df["compound_id"].to_numpy(),
    }

    n_comparisons = 0
    for g1, g2 in itertools.combinations(group_means.keys(), 2):
        m1, m2 = group_means[g1], group_means[g2]
        valid = (m1 > 0) & (m2 > 0) & np.isfinite(m1) & np.isfinite(m2)
        res = np.full(len(m1), np.nan)
        res[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{g1}_vs_{g2}"] = res
        n_comparisons += 1

    lfc_df = pd.DataFrame(lfc_records)

    # Prepend compound metadata columns before LFC columns
    main_db_path = getattr(summary_obj, "paths", {}).get("main_db_path")
    if main_db_path and "inchi_key" in merged_data_df.columns:
        inchi_keys = merged_data_df["inchi_key"].dropna().unique().tolist()
        meta_df = dbi.get_compound_metadata(main_db_path, inchi_keys)
        if not meta_df.empty:
            lfc_df = lfc_df.merge(
                merged_data_df[["compound_id", "inchi_key"]].drop_duplicates(),
                on="compound_id", how="left",
            )
            meta_df_sub = meta_df[['compound_id', 'inchi_key', 'modelseed_id']]
            meta_df_sub['modelseed_id'] = meta_df_sub['modelseed_id'].apply(lambda x: x.split(';')[0] if pd.notna(x) else x) #select only first modelseed_id
            lfc_df = lfc_df.merge(meta_df_sub, on="inchi_key", how="left")
            lfc_cols = [c for c in lfc_df.columns if "_vs_" in c]
            meta_cols = [c for c in meta_df_sub.columns if c != "inchi_key"]
            ordered_cols = (
                ["compound_id", "inchi_key"]
                + meta_cols
                + lfc_cols
            )
            lfc_df = lfc_df[[c for c in ordered_cols if c in lfc_df.columns]]

    lfc_df.to_csv(lfc_tsv, sep="\t", index=False)
    logger.info(f"Wrote group_mean_log_fold_changes.tsv ({len(merged_data_df)} compounds x {n_comparisons} group comparisons)")


###############################################
#### Targeted Analysis README
###############################################

_README_TEXT = """\
==============================================================================
  JOINT GENOME INSTITUTE
  TARGETED LC-MS/MS ANALYSIS - RESULTS GUIDE
==============================================================================

  This README describes every file and folder generated for a JGI targeted
  metabolomics analysis run. Outputs are organized into the folders listed
  below.

  DIRECTORY LAYOUT
      <analysis_results>/
      |-- FinalIdentifications__*.xlsx
      |-- CompoundAtlas__*.xlsx
      |-- identification_figures_with_msms_mirror_plots/
      |-- boxplot_peak_height/
      |-- boxplot_peak_height_log/
      |-- data_sheets/
      `-- additional_info/

  Abbreviations are used throughout this guide:

    ExCtrl  = extraction control
    InjBl   = injection blank
    ISTD    = internal standard
    QC      = quality control
    RefStd  = reference standard
    EIC     = extracted ion chromatogram
    TIC     = total ion chromatogram

==============================================================================
  1. Summary files
==============================================================================

  FinalIdentifications__*.xlsx
      The primary deliverable for each analysis. One row per compound
      searched in the targeted atlas. Used as the source of truth for
      compounds identified in samples and for their quality/quantity.
      The final identifications spreadsheet is curated and reviewed by
      both an analyst and by the metabolomics team lead.

  CompoundAtlas__*.xlsx
      Compound atlas(es) created during various stages of the targeted
      analysis. Typically this includes:
        - an "AUTOID" atlas (results from automated peak-finding against
          all JGI reference standards), and
        - a "CURATED" atlas (results after the analyst has manually curated 
          peak boundaries and scored each identified compound).
      Each atlas has one row per compound ID, with columns for metadata and
      analyst identification results.


==============================================================================
  2. identification_figures_with_msms_mirror_plots/
==============================================================================

  One PDF per compound, containing chemical and identification summary information 
  for that compound. Each figure contains:

    - MS/MS mirror plots of top database hits for extracted MS2 data
      per compound (up to three: best, 2nd-best and 3rd-best database
      match). Experimental fragment ions are shown above the y-axis midpoint
      and reference ions are below. Reference fragment intensities are scaled by
      the factor shown to provide easier visualization. Fragment m/z labels are 
      shown for the top-5 most intense ions. Cosine similarity score of the
      fragments and the MS2 scan RT are annotated above each panel.

    - Molecular structure rendered from the compound SMILES.

    - Compound metadata table (formula, adduct, theoretical/measured m/z and
      RT, ppm error, max intensity, etc.).

    - Best-hit summary table listing matched fragment m/z values, mass
      accuracy, RT accuracy, and MS/MS score.

    - EIC on linear and log10 intensity scales for all files. Red vertical dashed 
      lines mark the curated RT min/max bounds and a black dotted line marks the 
      theoretical RT peak.

{color_legend_placeholder}

==============================================================================
  3. boxplot_peak_height/  and  boxplot_peak_height_log/
==============================================================================

  One PDF per compound. Grouped boxplots of measured peak heights across all
  sample groups (linear scale and log10 scale, respectively).

    - Individual replicate samples are overlaid as jittered scatter dots,
      with color coding simply to differentiate groups (no mapping).
    - JGI sample types (ExCtrl, InjBl, ISTD, QC) are placed first on the
      x-axis; RefStd is placed last.

  These plots are the primary tool for quickly viewing compound abundance
  across experimental conditions.


==============================================================================
  4. data_sheets/
==============================================================================

  Wide-format CSV tables: one row per compound, one column per sample file.

  The first six rows of each CSV are metadata header rows:
      group, file, short_groupname, sample_treatment, short_filename,
      short_samplename (metadata splits of the full sample name)
  Data rows begin on row 7.

  peak_height.csv
      Raw MS1 peak height (maximum intensity) within the curated RT window
      for every compound x file combination.

  peak_area.csv
      Integrated area under the EIC curve within the curated RT window
      for every compound x file combination.

  rt_peak.csv
      Retention time (minutes) of the highest-intensity point within the
      curated RT window for every compound x file combination.

  rt_centroid.csv
      Intensity-weighted centroid retention time within the curated RT
      window for every compound x file combination.

  mz_peak.csv
      m/z value at the highest-intensity scan within the curated RT window 
      for every compound x file combination.

  mz_centroid.csv
      Intensity-weighted centroid m/z within the curated RT window for
      every compound x file combination.

  peak_height_filtered.csv
      Subset of peak_height.csv after applying an extraction-control
      (ExCtrl) fold-change filter:
        - A compound is retained only if at least one non-ExCtrl sample has
          a peak height >= 3x the mean ExCtrl peak height.
        - NaN values (no peak measured) are imputed with the global minimum
          observed peak height across all samples.

  peak_height_filter_summary.tsv
      Two-column TSV (compound_id, extraction_control_filter) recording
      "keep" or "remove" for every compound based on the ExCtrl filter above.

  group_mean_log_fold_changes.csv
      Pairwise log2 fold-changes (LFC) between every pair of experimental
      conditions (mean of replicates within sample groups), based on the
      filtered peak heights (peak_height_filtered.csv). LFC is calculated
      by subtracting the log2 of the mean peak height of replicates within
      one group from the same value of replicates in the comparison group.
        - One column per group pair (e.g. GroupA_vs_GroupB).
        - One row per compound.
        - NaN where either group mean is zero or missing.

  best_ms2_hit_fragment_ions.csv
      For each compound, the fragment ion spectrum (with m/z and intensity
      arrays as JSON) from the single file with the highest MS/MS database
      match score. Only fragment ions above 1x10^4 intensity are included.


==============================================================================
  5. additional_info/
==============================================================================

  sample_TIC_total_chromatograms_independentY.pdf
  sample_TIC_total_chromatograms_sharedY.pdf
      TIC for every sample file, arranged in a 5x5 grid per page. The TIC
      is the sum of all MS1 signal across the full chromatographic run; it 
      represents the total machine signal extracted from the raw data file.
        - Subplot backgrounds are color-coded by sample group to help
          differentiate, but do not map to actual values.

  compound_peak_heights.pdf
      One subplot per compound (5x5 grid per page).
        X-axis = run number (ordered by injection sequence)
        Y-axis = peak height

  compound_rt_peaks.pdf
      Same layout as compound_peak_heights.pdf, but Y-axis = measured RT
      peak (minutes). A red dashed horizontal line marks the theoretical RT
      for each compound.

  compound_EIC_chromatograms_sharedY/
  compound_EIC_chromatograms_independentY/
      One PDF per compound containing a 5x5 grid of EIC thumbnails, one
      subplot per sample file.
        - Red dashed vertical lines mark the curated RT min/max window.
        - A black dotted vertical line marks the theoretical RT peak.
        - Subplot backgrounds are color-coded by sample group to help
          differentiate, but do not map to actual values.

  boxplot_rt_peak/
  boxplot_rt_peak_log/
      Per-compound grouped boxplots of measured RT peak (linear and log10
      scale).

  boxplot_mz_centroid/
  boxplot_mz_centroid_log/
      Per-compound grouped boxplots of measured m/z centroid (linear and
      log10 scale).
"""

def _build_color_legend(color_map: dict) -> str:
    """Return the color legend section, formatted like the other README sections."""
    lines = [
        "",
        "      - Samples (lines) in the identification figures EIC plots are color-coded"
        "        using the mapping below to differentiate JGI internal samples from"
        "        experimental samples (gray).",
        "",
        f"        {'LINE COLOR':<14}SAMPLE TYPE",
        f"        {'-' * 12:<14}{'-' * 12}",
    ]
    for key, color in color_map.items():
        lines.append(f"        {str(color):<14}{key}")
    lines.append("")
    return "\n".join(lines)


def make_targeted_analysis_readme(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
    filename: str = "targeted_analysis_readme.txt",
) -> None:
    """Write a plain-text README describing all Google Drive outputs.

    The file is placed in the main analysis results output directory alongside
    the Final Identifications Excel sheet.  When a ``gui_lcmsruns_colors`` color
    map is configured, a file-color legend section is appended to the README.

    Parameters
    ----------
    summary_obj:
        The :class:`AnalysisSummary` instance for the current analysis.
    overwrite:
        When *False* and the file already exists, skip writing.
    filename:
        Output filename (default ``targeted_analysis_readme.txt``).
    """
    output_dir = Path(summary_obj.paths["analysis_results_output_dir"])
    readme_path = output_dir / filename

    if not overwrite and readme_path.exists():
        logger.info("Overwriting disabled: existing README %s will be used.", readme_path)
        return

    # Resolve color map (same priority as make_identification_figure)
    color_map = None
    if hasattr(summary_obj, "override_parameters") and summary_obj.override_parameters.get("gui_lcmsruns_colors"):
        color_map = summary_obj.override_parameters["gui_lcmsruns_colors"]
    elif hasattr(summary_obj, "config") and summary_obj.config and summary_obj.config.gui_config.get("gui_lcmsruns_colors"):
        color_map = summary_obj.config.gui_config["gui_lcmsruns_colors"]

    if color_map:
        readme_text = _README_TEXT.replace(
            "{color_legend_placeholder}", _build_color_legend(color_map)
        )
    else:
        readme_text = _README_TEXT.replace("{color_legend_placeholder}" + "\n\n", "")

    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_text, encoding="utf-8")
    logger.info("Wrote targeted analysis README to %s", readme_path)