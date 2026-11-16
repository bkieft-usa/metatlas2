"""
metatlas2/standards_annotation_rt_align.py

Reverse RT alignment phase for the standards-annotation workflow.

Selected peaks come from the current project database. QC inputs are
loaded from the project files and main database.

Workflow:
  1. Load the current saved selections and locate the summary folder.
  2. Find QC h5 files for each project_name / chromatography in sa_compounds.
  3. Load the QC atlas from the main metatlas DuckDB.
  4. Extract QC compound MS1 peaks from the QC h5 files.
  5. Build a reverse RT model per chromatography:
       X = measured_QC_RT  (from QC h5 files)
       y = historical_QC_RT (from QC atlas)
  6. Apply the model to the measured RT of each selected peak.
  7. Write a new timestamped alignment run.
  8. Generate outputs:
       - RTAlignment-Summary__{run_name}.pdf
       - StandardsAnnotation_Summary_Aligned_{run_name}.csv
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import metatlas2.file_and_project_format as fpf
import metatlas2.lcmsruns_tools as lrt
import metatlas2.logging_config as lcf
import metatlas2.rt_align_tools as rat
import metatlas2.extract_data_from_h5 as edh
import metatlas2.workflow_objects as wfo
from metatlas2.standards_annotation_review import (
    fix_peak_labels, review_directory, validate_selected_peaks,
)

logger = lcf.get_logger("standards_annotation_rt_align")

# Config helpers

_RT_ALIGNMENT_DEFAULTS: dict[str, Any] = {
    "qc_atlas_uid": "",
    "model_type": "polynomial",
    "poly_degree": 2,
    "rt_peak_source": "median",
    "min_compounds_for_modeling": 3,
    "min_observations_per_compound": 1,
    "exclude_inchikeys": [],
    "mz_tolerance_ppm": 5.0,
    "rt_window_padding": 0.5,
}


def _main_db_path() -> str:
    """Return the main metatlas DuckDB path (derived from $METATLAS_DATA_DIR)."""
    return f"{os.environ.get('METATLAS_DATA_DIR', '')}/databases/main_db/metatlas.duckdb"


def _ci_get(d: dict, key: str, default: Any = None) -> Any:
    """Dict get with case-insensitive string keys."""
    if key in d:
        return d[key]
    k_up = str(key).upper()
    for k, v in d.items():
        if str(k).upper() == k_up:
            return v
    return default


def _get_rt_alignment_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the RT_ALIGNMENT section keyed by upper-case chromatography.

    Expects the per-chromatography nested layout::

        RT_ALIGNMENT:
          HILIC:
            qc_atlas_uid: ...
          C18:
            qc_atlas_uid: ...

    Each entry gets defaults applied. Keys are matched case-insensitively
    against the chromatography values in the selections DataFrame.

    Args:
        config: Parsed config dict from
                :func:`metatlas2.load_tools.load_standards_config`.

    Returns:
        Dict mapping chromatography name → RT alignment params dict.
    """
    raw = dict(config.get("rt_alignment", {}))
    rt_cfg: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            rt_cfg[str(key).upper()] = {**_RT_ALIGNMENT_DEFAULTS, **val}
    return rt_cfg


def _get_chrom_rt_cfg(config: dict[str, Any], chrom: str) -> dict[str, Any] | None:
    """Return the RT_ALIGNMENT params for *chrom* (case-insensitive), or None."""
    return _ci_get(_get_rt_alignment_config(config), chrom)


# LCMS run discovery

def _get_raw_data_root(config: dict[str, Any]) -> str:
    """Return the raw data root for the project owner (PROJECT.owner)."""
    owner = str(config.get("project", {}).get("owner") or "jgi")
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    if data_dir:
        return str(Path(data_dir) / "raw_data" / owner)
    return f"/global/cfs/cdirs/metatlas/raw_data/{owner}"

def get_project_lcmsruns(config: dict[str, Any], db_path: str) -> list[Any]:
    """Read all lcmsrun files from the project folder(s) that produced the results.

    Locates the project directory(ies) under ``raw_data/<owner>`` that contain
    the files listed in ``sa_extraction_targets``, then loads every run in
    those directories via
    :func:`metatlas2.lcmsruns_tools.get_project_lcmsruns_from_disk`.
    Chromatography values are normalized so they agree with the RT_ALIGNMENT
    config keys and ``sa_peaks.chromatography``.

    Args:
        config: Parsed config dict.
        db_path: Path to the project DuckDB (source of sa_extraction_targets).

    Returns:
        List of ``wfo.LCMSRun`` objects (all file types and polarities).

    Raises:
        ValueError: If ``sa_extraction_targets`` is empty or no project
            directory under the raw data root contains the target files.
        FileNotFoundError, ValueError: Propagated from
            ``get_project_lcmsruns_from_disk`` (missing directory, ``.failed``
            files, format count mismatch) — actionable, intentionally not
            swallowed.
    """
    from metatlas2.standards_annotation_db import load_extraction_target_filenames

    raw_root = _get_raw_data_root(config)
    owner = str(config.get("project", {}).get("owner") or "jgi")
    target_filenames = load_extraction_target_filenames(db_path)
    if not target_filenames:
        raise ValueError(
            f"sa_extraction_targets in {db_path} is empty; "
            "cannot locate the project raw data folder."
        )
    target_stems = {Path(f).stem for f in target_filenames}

    # Find project dirs containing any extraction target (match by stem so
    # .raw/.mzML/.h5 variants of the same run all count).
    project_dirs: list[Path] = []
    root = Path(raw_root)
    if root.is_dir():
        for proj_dir in sorted(root.iterdir()):
            if not proj_dir.is_dir():
                continue
            if any(h5.stem in target_stems for h5 in proj_dir.glob("*.h5")):
                project_dirs.append(proj_dir)

    if not project_dirs:
        raise ValueError(
            f"No project directory under {raw_root} contains any of the "
            f"{len(target_stems)} extraction-target file(s) from {db_path}."
        )

    logger.info(
        "Project directories containing extraction targets: "
        f"{[d.name for d in project_dirs]}"
    )

    runs: list[dict] = []
    for proj_dir in project_dirs:
        runs.extend(lrt.get_project_lcmsruns_from_disk(str(proj_dir), owner))

    # Normalize chromatography (e.g. 'hilicz' -> 'hilic') so filtering by the
    # config keys / selections values works in filter_lcmsruns_list.
    for run in runs:
        run["chromatography"] = fpf.normalize_chromatography(
            str(run.get("chromatography", ""))
        )

    return [wfo.LCMSRun(**r) for r in runs]


def get_qc_runs_for_chrom(
    lcmsruns: list[Any],
    chrom: str,
    polarities: list[str] | None = None,
) -> list[Any]:
    """Filter the project lcmsruns to QC h5 files for one chromatography.

    Args:
        lcmsruns:   Full run list from :func:`get_project_lcmsruns`.
        chrom:      Chromatography to select (e.g. ``"HILIC"``); normalized
                    inside :func:`filter_lcmsruns_list`.
        polarities: Optional polarity filter (e.g. PROJECT.include_polarities).

    Returns:
        Filtered run list; empty list if nothing matches.
    """
    try:
        return lrt.filter_lcmsruns_list(
            lcmsruns,
            include_file_type=["qc"],
            file_format="h5",
            chromatography=chrom,
            polarity=polarities,
        )
    except ValueError:
        # filter_lcmsruns_list raises when no runs match the criteria
        return []

# QC atlas loading

def load_qc_atlas(config: dict[str, Any], chrom: str) -> Any:
    """Load the QC atlas for *chrom* from the main metatlas DuckDB.

    Args:
        config: Parsed config dict.
        chrom:  Chromatography key (case-insensitive), e.g. ``"HILIC"``.

    Returns:
        ``wfo.Atlas`` object with a ``compound_mzrts`` dict attribute,
        compatible with :func:`metatlas2.rt_align_tools._extract_per_file_peak_rts`.

    Raises:
        ValueError: If no RT_ALIGNMENT entry exists for *chrom*, or
                    ``qc_atlas_uid`` is empty / fails to load / has no compounds.
    """
    chrom_cfg = _get_chrom_rt_cfg(config, chrom)
    if chrom_cfg is None:
        raise ValueError(f"No RT_ALIGNMENT config entry for chromatography={chrom!r}.")

    atlas_uid = str(chrom_cfg.get("qc_atlas_uid", "")).strip()
    if not atlas_uid:
        raise ValueError(
            f"RT_ALIGNMENT.{chrom}.qc_atlas_uid is required but not set in the config."
        )

    logger.info(f"Loading QC atlas uid={atlas_uid} for chromatography={chrom}")

    try:
        atlas_obj = wfo.Atlas.from_database(_main_db_path(), atlas_uid)
    except Exception as exc:
        raise ValueError(
            f"Failed to load QC atlas '{atlas_uid}' for chrom={chrom}: {exc}"
        ) from exc

    n_compounds = len(getattr(atlas_obj, "compound_mzrts", {}) or {})
    if n_compounds == 0:
        raise ValueError(f"QC atlas '{atlas_uid}' (chrom={chrom}) has no compounds.")

    logger.info(f"Loaded QC atlas with {n_compounds} compound rows for chromatography={chrom}")
    return atlas_obj


# QC MS1 extraction

def extract_qc_ms1(
    qc_runs: list[Any],
    qc_atlas: pd.DataFrame,
    config: dict[str, Any],
    chrom: str,
) -> pd.DataFrame | None:
    """Extract MS1 data for one chromatography's QC compounds from its QC files.

    Args:
        qc_runs:  QC lcmsruns for *chrom* (from :func:`get_qc_runs_for_chrom`).
        qc_atlas: QC atlas dataframe for *chrom*.
        config:   Parsed config dict.
        chrom:    Chromatography name (logging / param selection).

    Returns:
        Wide-format MS1 DataFrame, or ``None`` if nothing was extracted.
    """
    chrom_cfg = _get_chrom_rt_cfg(config, chrom) or {}
    mz_tol_ppm = float(chrom_cfg.get("mz_tolerance_ppm", 5.0))
    rt_padding = float(chrom_cfg.get("rt_window_padding", 0.5))
    max_workers = int(config.get("extraction", {}).get("max_workers", 4))

    logger.info(
        f"Extracting QC MS1 for chromatography={chrom} ({len(qc_runs)} QC file(s), "
        f"mz_tol={mz_tol_ppm} ppm, rt_padding={rt_padding} min)"
    )

    rt_alignment_params = {
        "atlas_extra_time": rt_padding,
        "extract_extra_time": None,
        "ms1_mz_tolerance_ppm": mz_tol_ppm,
        "only_keep_data_in_feature": False,
        "ms1_min_num_points": 0,
        "ms1_min_peak_intensity": 0,
        "ms2_min_num_scans": 0,
        "ms2_min_precursor_intensity": 0,
        "max_workers": max_workers,
    }
    sa_rt_align_obj = wfo.StandardsAnnotationAutoIdentification(
        standards_annotation_atlas_df=qc_atlas,
        standards_annotation_lcmsruns=qc_runs,
        standards_annotation_params=rt_alignment_params,
        per_file_atlas=False,
    )
    edh.extract_data_from_raw(sa_rt_align_obj, stage="standards_annotation")
    ms1 = sa_rt_align_obj.experimental_data.ms1_df
    if ms1 is None or ms1.empty:
        logger.warning(f"No QC MS1 data extracted for chrom={chrom}.")
        return None

    logger.info(f"QC MS1 extraction complete for chrom={chrom}: {len(ms1)} rows")
    return ms1

# Reverse RT model building

def build_reverse_rt_model(
    qc_ms1_df: pd.DataFrame,
    qc_atlas: Any,
    config: dict[str, Any],
    chrom: str,
) -> dict | None:
    """Build a reverse RT alignment model from QC MS1 data for *chrom*.

    The "reverse" direction maps measured RT → historical RT:
      - X = measured QC RT (from QC h5 files)
      - y = historical QC RT (from the chromatography's QC atlas)

    All fitting parameters come from ``config["RT_ALIGNMENT"][chrom]``.

    Args:
        qc_ms1_df: Wide-format MS1 DataFrame from :func:`extract_qc_ms1`.
        qc_atlas:  QC atlas object (for *chrom*) with ``compound_mzrts``.
        config:    Parsed config dict.
        chrom:     Chromatography key selecting the RT_ALIGNMENT entry.

    Returns:
        Model dict (same structure as
        :func:`metatlas2.rt_align_tools.build_polynomial_model`) with an
        additional ``modeling_df`` key. Returns ``None`` if insufficient data.
    """
    chrom_cfg = _get_chrom_rt_cfg(config, chrom) or {}
    model_type = chrom_cfg.get("model_type", "polynomial")
    poly_degree = int(chrom_cfg.get("poly_degree", 2))
    rt_peak_source = chrom_cfg.get("rt_peak_source", "median")
    min_comps = int(chrom_cfg.get("min_compounds_for_modeling", 3))
    min_obs = int(chrom_cfg.get("min_observations_per_compound", 1))
    exclude_inchikeys = set(chrom_cfg.get("exclude_inchikeys", []))

    if qc_ms1_df is None or qc_ms1_df.empty:
        logger.warning(f"build_reverse_rt_model[{chrom}]: empty QC MS1 DataFrame.")
        return None

    per_file_data = rat._extract_per_file_peak_rts(
        ms1_df=qc_ms1_df,
        compound_mzrts=qc_atlas.compound_mzrts,
        exclude_inchikeys=exclude_inchikeys,
    )
    if not per_file_data:
        logger.warning(
            f"build_reverse_rt_model[{chrom}]: no compounds with MS1 data found in QC files."
        )
        return None

    _COMPOSITE_KEYS = frozenset({"min", "median", "max"})
    if rt_peak_source in _COMPOSITE_KEYS:
        per_file_data = {
            uid: files for uid, files in per_file_data.items() if len(files) >= min_obs
        }
        modeling_df = rat._composite_peak_rts(per_file_data, rt_peak_source)
    else:
        modeling_df = rat._per_file_peak_rts_to_modeling_df(per_file_data, rt_peak_source)

    if modeling_df.empty or len(modeling_df) < min_comps:
        logger.warning(
            f"build_reverse_rt_model[{chrom}]: only {len(modeling_df)} compound(s) "
            f"available (need ≥{min_comps}). Cannot build model."
        )
        return None

    # Fit the REVERSE model: swap X and y vs forward alignment.
    #   Forward: X = atlas_rt_peak (historical), y = exp_rt_peak (measured)
    #   Reverse: X = exp_rt_peak  (measured),   y = atlas_rt_peak (historical)
    X = modeling_df["exp_rt_peak"].values    # measured QC RT
    y = modeling_df["atlas_rt_peak"].values  # historical QC RT (from atlas)

    _VALID_MODEL_TYPES = frozenset({"polynomial", "linear", "median_offset"})
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r} for chrom={chrom}.")

    if model_type == "polynomial":
        model = rat.build_polynomial_model(X, y, degree=poly_degree)
    elif model_type == "linear":
        model = rat.build_linear_model(X, y)
    else:  # median_offset
        model = rat.build_median_offset_model(X, y)

    model["equation"] = rat.format_model_equation(model)

    y_pred = rat._apply_rt_model(X, model)
    modeling_df = modeling_df.copy()
    modeling_df["predicted_rt"] = y_pred
    modeling_df["residual"] = y - y_pred
    modeling_df["abs_residual"] = np.abs(modeling_df["residual"])
    model["modeling_df"] = modeling_df

    logger.info(
        f"Reverse RT model built for chrom={chrom}: type={model_type}, "
        f"n_compounds={len(modeling_df)}, "
        f"R²={model['r2']:.4f}, RMSE={model['rmse']:.4f} min"
    )
    logger.info(f"  Equation: {model['equation']}")

    return model

def apply_reverse_rt_model(
    selections_df: pd.DataFrame,
    model: dict,
) -> pd.DataFrame:
    """Apply the reverse RT model to analyst selections.

    For each row in *selections_df* (from :func:`load_selected_peaks_with_ms2`):
      - ``rt_peak_aligned = model(rt_peak)``
      - ``rt_delta = rt_peak_aligned - rt_peak``
      - ``rt_min_aligned = rt_min + rt_delta``
      - ``rt_max_aligned = rt_max + rt_delta``

    Args:
        selections_df: DataFrame from
                       :func:`~metatlas2.standards_annotation_db.load_selected_peaks_with_ms2`.
                       Must have columns ``rt_peak``, ``rt_min``, ``rt_max``.
        model:         Model dict from :func:`build_reverse_rt_model`.

    Returns:
        Copy of *selections_df* with additional columns:
        ``rt_peak_aligned``, ``rt_min_aligned``, ``rt_max_aligned``,
        ``rt_delta``.
    """
    df = selections_df.copy()

    rt_measured = df["rt_peak"].to_numpy(dtype=float)
    rt_aligned = rat._apply_rt_model(rt_measured, model)

    df["rt_peak_aligned"] = rt_aligned
    df["rt_delta"] = rt_aligned - rt_measured
    df["rt_min_aligned"] = df["rt_min"] + df["rt_delta"]
    df["rt_max_aligned"] = df["rt_max"] + df["rt_delta"]

    logger.info(
        f"Applied reverse RT model to {len(df)} selections. "
        f"Mean RT delta: {df['rt_delta'].mean():.4f} min, "
        f"Std: {df['rt_delta'].std():.4f} min"
    )
    return df

# Log outputs to disk

def make_rt_alignment_summary_plot(
    modeling_df: pd.DataFrame,
    model: dict,
    output_dir: str | Path,
    run_name: str,
) -> str:
    """Generate the RT alignment summary PDF.

    Produces a two-panel figure:
      - Left: scatter of measured QC RT vs historical QC RT with model fit.
      - Right: residuals vs measured QC RT.
      - Bottom: compound table.

    Args:
        modeling_df: Per-compound DataFrame with ``exp_rt_peak``,
                     ``atlas_rt_peak``, ``predicted_rt``, ``residual`` columns.
        model:       Model dict from :func:`build_reverse_rt_model`.
        output_dir:  Directory to save the PDF.
        run_name:    Run name used in the filename.

    Returns:
        Absolute path to the saved PDF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = modeling_df.sort_values("exp_rt_peak").reset_index(drop=True)
    df["compound_num"] = df.index + 1

    model_type = model.get("model_type", "polynomial")
    degree = model.get("degree", 0)
    r2 = model.get("r2", float("nan"))
    rmse = model.get("rmse", float("nan"))

    if model_type == "median_offset":
        fit_label = "Median Offset Fit"
    elif model_type == "linear":
        fit_label = "Linear Fit"
    else:
        fit_label = f"Polynomial Fit (degree {degree})"

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Panel 1: Measured QC RT vs Historical QC RT (with model fit)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        df["exp_rt_peak"], df["atlas_rt_peak"],
        alpha=0.7, s=50, c="blue", label="QC Compounds"
    )
    x_line = np.linspace(df["exp_rt_peak"].min(), df["exp_rt_peak"].max(), 200)
    y_line = rat._apply_rt_model(x_line, model)
    ax1.plot(x_line, y_line, "r-", linewidth=2, label=fit_label)
    # 1:1 reference
    lo = min(df["exp_rt_peak"].min(), df["atlas_rt_peak"].min())
    hi = max(df["exp_rt_peak"].max(), df["atlas_rt_peak"].max())
    ax1.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="1:1 (no shift)")
    ax1.set_xlabel("Measured QC RT (min)")
    ax1.set_ylabel("Historical QC RT / Atlas (min)")
    ax1.set_title(f"Reverse RT Alignment (R² = {r2:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for _, row in df.iterrows():
        ax1.annotate(
            str(int(row["compound_num"])),
            (row["exp_rt_peak"], row["atlas_rt_peak"]),
            textcoords="offset points", xytext=(5, -10),
            ha="left", fontsize=9, color="black",
        )

    # Panel 2: Residuals vs Measured QC RT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df["exp_rt_peak"], df["residual"], alpha=0.7, s=50, c="green")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    std_resid = df["residual"].std()
    ax2.axhline(y=std_resid, color="orange", linestyle=":", alpha=0.7, label="+1 σ")
    ax2.axhline(y=-std_resid, color="orange", linestyle=":", alpha=0.7, label="-1 σ")
    ax2.set_xlabel("Measured QC RT (min)")
    ax2.set_ylabel("Residual (min)")
    ax2.set_title(f"Residuals (RMSE = {rmse:.4f} min)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for _, row in df.iterrows():
        ax2.annotate(
            str(int(row["compound_num"])),
            (row["exp_rt_peak"], row["residual"]),
            textcoords="offset points", xytext=(5, -10),
            ha="left", fontsize=9, color="black",
        )

    # Panel 3: Compound table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    table_data = df[[
        "compound_num", "compound_name", "inchi_key", "adduct",
        "exp_rt_peak", "atlas_rt_peak", "residual",
    ]].copy()
    table_data["exp_rt_peak"] = table_data["exp_rt_peak"].round(3)
    table_data["atlas_rt_peak"] = table_data["atlas_rt_peak"].round(3)
    table_data["residual"] = table_data["residual"].round(4)
    table_data.columns = [
        "#", "Compound Name", "InChIKey", "Adduct",
        "Measured RT (min)", "Historical RT (min)", "Residual (min)",
    ]
    table = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    equation = model.get("equation", "")
    plt.suptitle(
        f"Reverse RT Alignment — {run_name}\n{equation}",
        fontsize=13, fontweight="bold",
    )

    pdf_path = output_dir / f"RTAlignment-Summary__{run_name}.pdf"
    try:
        plt.savefig(pdf_path, bbox_inches="tight")
        logger.info(f"RT alignment summary plot saved to {pdf_path}")
    except Exception as exc:
        raise ValueError(f"Failed to save RT alignment summary plot: {exc}")
    finally:
        plt.close()

    return str(pdf_path)


def _normalize_tags(tags: str | None) -> str:
    """Clean a semicolon-separated tag string: strip blanks, drop empties."""
    return ";".join(t.strip() for t in (tags or "").split(";") if t.strip())


def _append_tags_column(csv_path: Path, tags: str) -> None:
    """Add 'tags' as the final column of an existing CSV (values left untouched)."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df.drop(columns="tags", errors="ignore")
    df["tags"] = tags
    df.to_csv(csv_path, index=False)


def _align_chromatography(
    config: dict,
    chrom: str,
    selections: pd.DataFrame,
    lcmsruns,
    polarities: list[str] | None,
) -> tuple[pd.DataFrame, tuple[str, pd.DataFrame, dict]]:
    """Build the RT model for one chromatography and apply it to its selections."""
    chrom_cfg = _get_chrom_rt_cfg(config, chrom)
    if chrom_cfg is None:
        raise ValueError(
            f"No RT_ALIGNMENT config entry for chromatography={chrom!r}. "
            "Configure a QC atlas and model before running reverse alignment."
        )
    qc_runs = get_qc_runs_for_chrom(lcmsruns, chrom, polarities=polarities)
    if not qc_runs:
        raise ValueError(f"No QC files found for chromatography={chrom}. Check the project files and configuration.")

    qc_atlas = load_qc_atlas(config, chrom)
    qc_ms1_df = extract_qc_ms1(qc_runs, qc_atlas.to_dataframe(), config, chrom)
    if qc_ms1_df is None or qc_ms1_df.empty:
        raise ValueError(f"No QC MS1 data found for chromatography={chrom}; cannot build an RT model.")

    model = build_reverse_rt_model(qc_ms1_df, qc_atlas, config, chrom)
    if model is None:
        raise ValueError(f"Could not build an RT model for chromatography={chrom}; check the QC data and model configuration.")
    modeling_df = model.get("modeling_df")
    if modeling_df is None or modeling_df.empty:
        raise ValueError(f"RT model for {chrom} has no modeling data for the review plot.")

    aligned = apply_reverse_rt_model(selections, model)
    aligned["rt_model_type"] = model.get("model_type", "polynomial")
    aligned["rt_model_degree"] = model.get("degree", 0)
    aligned["rt_model_r2"] = model.get("r2", float("nan"))
    aligned["rt_model_rmse"] = model.get("rmse", float("nan"))
    aligned["qc_atlas_uid"] = str(chrom_cfg.get("qc_atlas_uid", "")).strip()
    return aligned, (chrom, modeling_df, model)

def _passthrough_alignment(selections: pd.DataFrame) -> pd.DataFrame:
    """Return selections with aligned RT columns equal to the input RTs (no model)."""
    _RT_COLUMN_MAP = {
        "rt_peak_aligned": "rt_peak",
        "rt_min_aligned": "rt_min",
        "rt_max_aligned": "rt_max",
    }

    aligned = selections.copy()
    for aligned_col, input_col in _RT_COLUMN_MAP.items():
        aligned[aligned_col] = aligned[input_col]
    # Same output schema as the modeled path, with null model metadata
    aligned["rt_model_type"] = "none"
    aligned["rt_model_degree"] = 0
    aligned["rt_model_r2"] = float("nan")
    aligned["rt_model_rmse"] = float("nan")
    aligned["qc_atlas_uid"] = ""
    return aligned

def run_reverse_rt_alignment(
    config_path: str,
    db_path: str,
    output_path: str | Path,
    timestamp: str,
    tags: str | None = None,
    skip_rt_alignment: bool = False,
) -> None:
    """Orchestrate the full reverse RT alignment workflow.

    Steps:
      1. Load config, project DB, and the current saved selections.
      2. For each chromatography: build its QC-based reverse RT model and
         apply it to that chromatography's selected peaks.
      3. Publish the timestamped alignment outputs (plots, config, and the
         aligned summary CSV with a final ``tags`` column).

    Args:
        config_path: Path to the standards-annotation YAML config file.
        db_path:     Path to the project database.
        output_path: Output path; its parent holds the review directories.
        timestamp:   Approved annotation run timestamp.
        tags:        Semicolon-separated tags written to the ``tags`` column.
        skip_rt_alignment: If True, do not build or apply any RT model; just
            copy the input RTs to the aligned columns.
    """
    from metatlas2.load_tools import load_standards_config
    from metatlas2.standards_annotation_db import (
        load_selected_peaks_with_ms2,
        validate_annotation_choices,
    )
    from metatlas2.standards_annotation_summary import make_standards_annotation_table

    config = load_standards_config(str(Path(config_path).resolve()))
    run_name = config["project"]["standards_run_name"]
    polarities = list(config["project"].get("include_polarities") or []) or None
    tags_value = _normalize_tags(tags)
    if tags_value == "EMA":
        logger.warning("Tag is set to default ('EMA') only. Do you need to create any more specific tag?")
    logger.info(f"Reverse RT alignment starting for run_name={run_name}, timestamp={timestamp}, tags={tags_value}")

    # --- Validate prior step and load selections ---
    annotation_dir = review_directory(Path(output_path).parent, timestamp)
    if not (
        (annotation_dir / ".summary_complete").is_file()
        and (annotation_dir / f"StandardsAnnotation_Summary_{run_name}.csv").is_file()
    ):
        raise ValueError(f"Completed summary not found for timestamp {timestamp}: {annotation_dir}")

    validate_annotation_choices(db_path)
    raw_selections = load_selected_peaks_with_ms2(db_path)
    validate_selected_peaks(raw_selections)
    selections_df = fix_peak_labels(raw_selections)
    logger.info(f"Loaded {len(selections_df)} current selections from the project database")

    # --- Create output directory ---
    alignment_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    alignment_dir = annotation_dir / f"rt_align_{alignment_timestamp}"
    alignment_dir.mkdir(parents=False, exist_ok=False)
    logger.info(f"Alignment timestamp: {alignment_timestamp}; outputs: {alignment_dir}")
    print(f"Alignment timestamp: {alignment_timestamp}\nAlignment directory: {alignment_dir}")

    # --- Align each chromatography (or pass through unchanged) ---
    model_plots: list[tuple[str, pd.DataFrame, dict]] = []
    if skip_rt_alignment:
        logger.info("skip_rt_alignment=True: aligned RT columns will equal the input RTs")
        aligned_df = _passthrough_alignment(selections_df)
    else:
        lcmsruns = get_project_lcmsruns(config, db_path)
        chunks = []
        for chrom, chrom_selections in selections_df.groupby("chromatography", sort=False):
            aligned, plot_info = _align_chromatography(
                config, chrom, chrom_selections.copy(), lcmsruns, polarities
            )
            chunks.append(aligned)
            model_plots.append(plot_info)
        if not chunks:
            raise ValueError("No aligned data produced. Aborting.")
        aligned_df = pd.concat(chunks, ignore_index=True)

    if (aligned_df["peak_uid"].duplicated().any()
            or set(aligned_df["peak_uid"]) != set(selections_df["peak_uid"])):
        raise ValueError("Aligned peak identities do not match the loaded selections.")
    aligned_df = aligned_df.set_index("peak_uid").loc[selections_df["peak_uid"]].reset_index()

    aligned_cols = ["rt_peak_aligned", "rt_min_aligned", "rt_max_aligned"]
    if not np.isfinite(aligned_df[aligned_cols].to_numpy(dtype=float)).all():
        raise ValueError("Aligned RT values must be finite for every selected peak.")

    # --- Stage outputs, then publish ---
    rt_align_dir = alignment_dir / "rt_alignment"
    rt_align_dir.mkdir()
    summary_name = f"StandardsAnnotation_Summary_Aligned_{run_name}.csv"
    final_summary = alignment_dir / summary_name

    with tempfile.TemporaryDirectory(prefix=".rt-align-", dir=alignment_dir) as temp:
        staging = Path(temp)

        for chrom, modeling_df, model in model_plots:
            plot = Path(make_rt_alignment_summary_plot(modeling_df, model, staging, f"{run_name}__{chrom}"))
            if not plot.is_file():
                raise RuntimeError(f"RT alignment figure was not created: {plot}")

        make_standards_annotation_table(aligned_df, staging, run_name, aligned=True)
        staged_summary = staging / summary_name
        if not staged_summary.is_file():
            raise RuntimeError(f"Aligned annotation summary was not created: {staged_summary}")
        _append_tags_column(staged_summary, tags_value)

        config_file = staging / "alignment_config.json"
        config_file.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

        try:
            for src in [*staging.glob("*.pdf"), config_file]:
                os.replace(src, rt_align_dir / src.name)
            os.replace(staged_summary, final_summary)
        except Exception:
            final_summary.unlink(missing_ok=True)
            raise

    logger.info(f"Reverse RT alignment complete. Outputs: {alignment_dir}")