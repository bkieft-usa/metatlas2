"""
metatlas2/standards_annotation_rt_align.py

Reverse RT alignment phase for the standards-annotation workflow.

All inputs are read from the project DuckDB (sa_analyst_selections, sa_peaks,
sa_peak_ms2, sa_compounds) — no in-memory objects from the GUI session are
required.  This module can be run days or weeks after the GUI annotation session.

Workflow:
  1. Load analyst selections via load_selected_peaks_with_ms2().
  2. Find QC h5 files for each project_name / chromatography in sa_compounds.
  3. Load the QC atlas from the main metatlas DuckDB.
  4. Extract QC compound MS1 peaks from the QC h5 files.
  5. Build a reverse RT model per chromatography:
       X = measured_QC_RT  (from QC h5 files)
       y = historical_QC_RT (from QC atlas)
  6. Apply the model to rt_peak (from sa_peaks) for each selected peak.
  7. Save aligned rows to sa_rt_aligned (peak_uid PK + aligned RT values).
  8. Generate outputs:
       - RTAlignment-Summary__{run_name}.pdf
       - RTAlignment-Table__{run_name}.csv
       - RefStd_Atlas__{chrom}__{pol}__{run_name}.csv  (one per chrom x polarity)
"""
from __future__ import annotations

import glob
import os
import uuid
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
from metatlas2.standards_annotation_tools import (
    _build_lcmsruns_list,
)
from metatlas2.utils import get_provenance

logger = lcf.get_logger("standards_annotation_rt_align")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_rt_alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the RT_ALIGNMENT sub-section of *config* with defaults applied.

    Args:
        config: Parsed config dict from
                :func:`metatlas2.load_tools.load_standards_config`.

    Returns:
        RT alignment config dict with all defaults filled in.
    """
    rt_cfg = dict(config.get("rt_alignment", {}))
    rt_cfg.setdefault("qc_atlas_uid", "")
    rt_cfg.setdefault("main_db_path", f"{os.environ.get('METATLAS_DATA_DIR', '')}/databases/main_db/metatlas.duckdb")
    rt_cfg.setdefault("model_type", "polynomial")
    rt_cfg.setdefault("poly_degree", 2)
    rt_cfg.setdefault("rt_peak_source", "median")
    rt_cfg.setdefault("min_compounds_for_modeling", 3)
    rt_cfg.setdefault("min_observations_per_compound", 1)
    rt_cfg.setdefault("exclude_inchikeys", [])
    rt_cfg.setdefault("mz_tolerance_ppm", 5.0)
    rt_cfg.setdefault("rt_window_padding", 0.5)
    return rt_cfg


def _get_raw_data_root() -> str:
    """Return the root directory for raw LCMS data (JGI convention)."""
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    if data_dir:
        return str(Path(data_dir) / "raw_data" / "jgi")
    return "/global/cfs/cdirs/metatlas/raw_data/jgi"


# ---------------------------------------------------------------------------
# QC atlas loading
# ---------------------------------------------------------------------------

def load_qc_atlas(config: dict[str, Any]) -> Any:
    """Load the QC atlas from the main metatlas DuckDB.

    Reads ``qc_atlas_uid`` from ``config["RT_ALIGNMENT"]`` and fetches the
    corresponding atlas compounds table from the main DB.

    Args:
        config: Parsed config dict.

    Returns:
        An Atlas-like object with a ``compound_mzrts`` dict attribute,
        compatible with :func:`metatlas2.rt_align_tools._extract_per_file_peak_rts`.

    Raises:
        ValueError: If ``qc_atlas_uid`` is empty or the atlas is not found.
    """
    rt_cfg = _get_rt_alignment_config(config)
    atlas_uid = rt_cfg.get("qc_atlas_uid", "").strip()
    if not atlas_uid:
        raise ValueError(
            "RT_ALIGNMENT.qc_atlas_uid is required but not set in the config. "
            "Set it to the UID of the QC atlas in the main metatlas DuckDB."
        )

    main_db_path = rt_cfg.get("main_db_path", "").strip()
    # Expand environment variables in the path
    main_db_path = os.path.expandvars(main_db_path)
    if not main_db_path or not Path(main_db_path).exists():
        raise FileNotFoundError(
            f"main_db_path '{main_db_path}' does not exist. "
            "Set RT_ALIGNMENT.main_db_path in the config or set $METATLAS_MAIN_DB."
        )

    logger.info(f"Loading QC atlas uid={atlas_uid} from {main_db_path}")

    try:
        import metatlas2.database_interact as di
        atlas_df = di.get_atlas_compounds_table(main_db_path, atlas_uid)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load QC atlas '{atlas_uid}' from '{main_db_path}': {exc}"
        ) from exc

    if atlas_df is None or atlas_df.empty:
        raise ValueError(
            f"QC atlas '{atlas_uid}' returned no compounds from '{main_db_path}'."
        )

    logger.info(f"Loaded QC atlas with {len(atlas_df)} compound rows")

    # Build an Atlas object from the DataFrame
    try:
        from metatlas2.workflow_objects import Atlas
        atlas_obj = Atlas.from_dataframe(atlas_df)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build Atlas object from QC atlas DataFrame: {exc}"
        ) from exc

    return atlas_obj


# ---------------------------------------------------------------------------
# QC file discovery
# ---------------------------------------------------------------------------

def find_qc_files(
    config: dict[str, Any],
    project_db_path: str,
    raw_data_root: str | None = None,
) -> dict[str, list[str]]:
    """Find QC h5 files for each chromatography in the project.

    Reads the ``sa_compounds`` table to get unique ``(project_name,
    chromatography)`` pairs, then globs for h5 files under
    ``<raw_data_root>/<project_name>/`` and filters to those classified as
    ``file_type="qc"`` by
    :func:`metatlas2.lcmsruns_tools._classify_file_type`.

    Args:
        config:           Parsed config dict.
        project_db_path:  Path to the project DuckDB.
        raw_data_root:    Optional override for the raw data root directory.

    Returns:
        Dict mapping ``chromatography`` → list of QC h5 file paths.
        Returns an empty dict if no QC files are found.
    """
    import duckdb

    if raw_data_root is None:
        raw_data_root = _get_raw_data_root()

    logger.info(f"Searching for QC files under {raw_data_root}")

    # Load unique (filename, chromatography) pairs from sa_extraction_targets.
    # In the new schema, filename and chromatography live in sa_extraction_targets,
    # not sa_compounds (which now holds only compound-level identity fields).
    try:
        with duckdb.connect(str(project_db_path), read_only=True) as conn:
            pairs_df = conn.execute(
                "SELECT DISTINCT filename, chromatography FROM sa_extraction_targets"
            ).df()
    except Exception as exc:
        logger.warning(f"Could not read sa_extraction_targets from {project_db_path}: {exc}")
        return {}

    if pairs_df.empty:
        logger.warning("sa_extraction_targets is empty -- no QC files to find.")
        return {}

    # Extract project_names from filenames by parsing the h5 filename structure
    # The project_name is the parent directory of the h5 file.
    # We need to find the project directories from the filenames stored in sa_compounds.
    # Filenames in sa_compounds are basenames; we need to find their parent dirs.
    # Strategy: glob all project dirs that contain any of the known filenames.
    project_names: set[str] = set()
    for filename in pairs_df["filename"].unique():
        # Try to parse the project name from the filename using fpf
        try:
            fields = fpf.parse_file_name(str(filename))
            # The project name is not directly in the filename fields,
            # so we glob for it across all project dirs
        except Exception:
            pass

    # Glob all project directories under raw_data_root
    all_project_dirs = [
        d for d in Path(raw_data_root).iterdir()
        if d.is_dir()
    ] if Path(raw_data_root).exists() else []

    # Build a set of known filenames for fast lookup
    known_filenames = set(pairs_df["filename"].str.strip().tolist())

    # Find which project dirs contain our known filenames
    for proj_dir in all_project_dirs:
        for h5_file in proj_dir.glob("*.h5"):
            if h5_file.name in known_filenames:
                project_names.add(proj_dir.name)
                break

    if not project_names:
        logger.warning(
            "Could not match any sa_compounds filenames to project directories "
            f"under {raw_data_root}. QC file discovery will be skipped."
        )
        return {}

    logger.info(f"Found project directories: {sorted(project_names)}")

    # Build chromatography → [qc_file_paths] mapping
    chrom_to_qc_files: dict[str, list[str]] = {}

    for proj_name in sorted(project_names):
        proj_dir = Path(raw_data_root) / proj_name
        all_h5 = sorted(proj_dir.glob("*.h5"))

        for h5_path in all_h5:
            try:
                fields = fpf.parse_file_name(h5_path.name)
            except Exception:
                continue

            file_type = lrt._classify_file_type(fields)
            if file_type != "qc":
                continue

            chrom = fpf.canonical_chromatography(fields.get("chromatography", ""))
            if not chrom:
                continue

            chrom_to_qc_files.setdefault(chrom, [])
            if str(h5_path) not in chrom_to_qc_files[chrom]:
                chrom_to_qc_files[chrom].append(str(h5_path))

    for chrom, files in chrom_to_qc_files.items():
        logger.info(f"Found {len(files)} QC file(s) for chromatography={chrom}")

    if not chrom_to_qc_files:
        logger.warning(
            "No QC files found. RT alignment will use identity model (no shift)."
        )

    return chrom_to_qc_files


# ---------------------------------------------------------------------------
# QC MS1 extraction
# ---------------------------------------------------------------------------

def extract_qc_ms1(
    qc_files_by_chrom: dict[str, list[str]],
    qc_atlas: Any,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Extract MS1 data for QC compounds from QC h5 files.

    For each chromatography, builds a :class:`wfo.StandardsAnnotationAutoIdentification`
    wrapper and calls :func:`metatlas2.extract_data_from_h5.extract_data_from_raw`
    to get MS1 data for all QC atlas compounds in all QC files.

    Args:
        qc_files_by_chrom: Dict mapping chromatography → list of QC h5 paths.
        qc_atlas:          QC atlas object with ``compound_mzrts`` attribute.
        config:            Parsed config dict.

    Returns:
        Dict mapping chromatography → wide-format MS1 DataFrame.
    """
    from metatlas2.workflow_objects import LCMSRun

    rt_cfg = _get_rt_alignment_config(config)
    mz_tol_ppm = float(rt_cfg.get("mz_tolerance_ppm", 5.0))
    rt_padding = float(rt_cfg.get("rt_window_padding", 0.5))
    max_workers = int(config.get("extraction", {}).get("max_workers", 4))

    result: dict[str, pd.DataFrame] = {}

    for chrom, qc_paths in qc_files_by_chrom.items():
        if not qc_paths:
            continue

        logger.info(
            f"Extracting QC MS1 for chromatography={chrom} "
            f"({len(qc_paths)} QC file(s))"
        )

        # Build expanded table from QC atlas x QC files
        # We need one row per (compound_mzrt, qc_file, polarity)
        expanded_rows = []
        for mz_rt_uid, cmzrt in qc_atlas.compound_mzrts.items():
            for qc_path in qc_paths:
                try:
                    fields = fpf.parse_file_name(os.path.basename(qc_path))
                    file_chrom = fpf.canonical_chromatography(
                        fields.get("chromatography", "")
                    )
                    if file_chrom != chrom:
                        continue
                    polarity = fields.get("polarity", "pos").upper()
                except Exception:
                    polarity = "POS"

                expanded_rows.append({
                    "mz_rt_uid": mz_rt_uid,
                    "compound_name": cmzrt.compound_name,
                    "inchi_key": cmzrt.inchi_key,
                    "adduct": cmzrt.adduct,
                    "mz_theoretical": cmzrt.mz,
                    "rt_min": max(0.0, cmzrt.rt_min - rt_padding),
                    "rt_max": cmzrt.rt_max + rt_padding,
                    "polarity": polarity,
                    "chromatography": chrom,
                    "file_path": qc_path,
                    "filename": os.path.basename(qc_path),
                    "file_type": "qc",
                })

        if not expanded_rows:
            logger.warning(
                f"No QC atlas compounds matched chromatography={chrom}; skipping."
            )
            continue

        expanded_df = pd.DataFrame(expanded_rows)

        # Group by polarity for extraction
        all_ms1_chunks: list[pd.DataFrame] = []
        _POL_MAP = {"POS": "positive", "NEG": "negative"}

        for polarity, pol_df in expanded_df.groupby("polarity"):
            h5_polarity = _POL_MAP.get(polarity.upper(), "positive")
            atlas_wrapper = wfo.StandardsAnnotationAtlas(pol_df, h5_polarity)
            lcmsruns = _build_lcmsruns_list(pol_df)

            if not lcmsruns:
                continue

            rt_alignment_params = {
                "atlas_extra_time": 0.0,
                "extract_extra_time": None,
                "ms1_mz_tolerance_ppm": mz_tol_ppm,
                "only_keep_data_in_feature": False,
                "ms1_min_num_points": 0,
                "ms1_min_peak_intensity": 0,
                "ms2_min_num_scans": 0,
                "ms2_min_precursor_intensity": 0,
                "max_workers": max_workers
            }

            sa_rt_align_obj = wfo.StandardsAnnotationAutoIdentification(
                standards_annotation_atlas_obj=atlas_wrapper,
                standards_annotation_lcmsruns=lcmsruns,
                standards_annotation_params=rt_alignment_params,
            )

            try:
                edh.extract_data_from_raw(sa_rt_align_obj, stage="standards_annotation")
            except Exception as exc:
                logger.error(
                    f"QC MS1 extraction failed for chrom={chrom}, "
                    f"polarity={polarity}: {exc}"
                )
                continue

            ms1 = sa_rt_align_obj.experimental_data.ms1_df
            if ms1 is not None and not ms1.empty:
                all_ms1_chunks.append(ms1)

        if all_ms1_chunks:
            combined = pd.concat(all_ms1_chunks, ignore_index=True)
            result[chrom] = combined
            logger.info(
                f"QC MS1 extraction complete for chrom={chrom}: "
                f"{len(combined)} rows"
            )
        else:
            logger.warning(f"No QC MS1 data extracted for chrom={chrom}.")

    return result


# ---------------------------------------------------------------------------
# Reverse RT model building
# ---------------------------------------------------------------------------

def build_reverse_rt_model(
    qc_ms1_df: pd.DataFrame,
    qc_atlas: Any,
    config: dict[str, Any],
) -> dict | None:
    """Build a reverse RT alignment model from QC MS1 data.

    The "reverse" direction maps measured RT → historical RT:
      - X = measured QC RT (from QC h5 files)
      - y = historical QC RT (from QC atlas)

    This is the inverse of the forward alignment (atlas RT → measured RT).

    Args:
        qc_ms1_df: Wide-format MS1 DataFrame from :func:`extract_qc_ms1`.
        qc_atlas:  QC atlas object with ``compound_mzrts`` attribute.
        config:    Parsed config dict.

    Returns:
        Model dict (same structure as :func:`metatlas2.rt_align_tools.build_polynomial_model`)
        with an additional ``modeling_df`` key containing the per-compound
        DataFrame used for fitting.  Returns ``None`` if insufficient data.
    """
    rt_cfg = _get_rt_alignment_config(config)
    model_type = rt_cfg.get("model_type", "polynomial")
    poly_degree = int(rt_cfg.get("poly_degree", 2))
    rt_peak_source = rt_cfg.get("rt_peak_source", "median")
    min_comps = int(rt_cfg.get("min_compounds_for_modeling", 3))
    min_obs = int(rt_cfg.get("min_observations_per_compound", 1))
    exclude_inchikeys = set(rt_cfg.get("exclude_inchikeys", []))

    if qc_ms1_df is None or qc_ms1_df.empty:
        logger.warning("build_reverse_rt_model: empty QC MS1 DataFrame.")
        return None

    # Step 1: extract per-file peak RTs
    per_file_data = rat._extract_per_file_peak_rts(
        ms1_df=qc_ms1_df,
        compound_mzrts=qc_atlas.compound_mzrts,
        exclude_inchikeys=exclude_inchikeys,
    )

    if not per_file_data:
        logger.warning(
            "build_reverse_rt_model: no compounds with MS1 data found in QC files."
        )
        return None

    # Step 2: build composite modeling DataFrame
    _COMPOSITE_KEYS = frozenset({"min", "median", "max"})
    if rt_peak_source in _COMPOSITE_KEYS:
        modeling_df = rat._composite_peak_rts(per_file_data, rt_peak_source)
    else:
        # Treat as a specific filename
        modeling_df = rat._per_file_peak_rts_to_modeling_df(per_file_data, rt_peak_source)

    if modeling_df.empty or len(modeling_df) < min_comps:
        logger.warning(
            f"build_reverse_rt_model: only {len(modeling_df)} compound(s) available "
            f"(need ≥{min_comps}). Cannot build model."
        )
        return None

    # Step 3: fit the REVERSE model
    # KEY: swap X and y vs forward alignment
    #   Forward: X = atlas_rt_peak (historical), y = exp_rt_peak (measured)
    #   Reverse: X = exp_rt_peak  (measured),   y = atlas_rt_peak (historical)
    X = modeling_df["exp_rt_peak"].values    # measured QC RT
    y = modeling_df["atlas_rt_peak"].values  # historical QC RT (from atlas)

    _VALID_MODEL_TYPES = frozenset({"polynomial", "linear", "median_offset"})
    if model_type not in _VALID_MODEL_TYPES:
        logger.warning(
            f"Unknown model_type '{model_type}'; falling back to 'polynomial'."
        )
        model_type = "polynomial"

    if model_type == "polynomial":
        model = rat.build_polynomial_model(X, y, degree=poly_degree)
    elif model_type == "linear":
        model = rat.build_linear_model(X, y)
    else:  # median_offset
        model = rat.build_median_offset_model(X, y)

    model["equation"] = rat.format_model_equation(model)
    model["modeling_df"] = modeling_df

    # Attach predicted values and residuals to the modeling DataFrame
    y_pred = rat._apply_rt_model(X, model)
    modeling_df = modeling_df.copy()
    modeling_df["predicted_rt"] = y_pred
    modeling_df["residual"] = y - y_pred
    modeling_df["abs_residual"] = np.abs(modeling_df["residual"])
    model["modeling_df"] = modeling_df

    logger.info(
        f"Reverse RT model built: type={model_type}, "
        f"n_compounds={len(modeling_df)}, "
        f"R²={model['r2']:.4f}, RMSE={model['rmse']:.4f} min"
    )
    logger.info(f"  Equation: {model['equation']}")

    return model


# ---------------------------------------------------------------------------
# Apply reverse RT model to selections
# ---------------------------------------------------------------------------

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

    rt_measured = df["rt_peak"].values
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


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

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
        logger.error(f"Failed to save RT alignment summary plot: {exc}")
    finally:
        plt.close()

    return str(pdf_path)


def make_rt_alignment_table(
    aligned_df: pd.DataFrame,
    output_dir: str | Path,
    run_name: str,
) -> str:
    """Generate the RT alignment csv file

    Writes one row per aligned peak with measured and aligned RT values,
    intensity, m/z, source file, and analyst notes.

    *aligned_df* is the output of :func:`~metatlas2.standards_annotation_db.load_rt_aligned`
    which joins ``sa_rt_aligned`` with ``sa_peaks`` and ``sa_compounds``.

    Args:
        aligned_df: Full joined DataFrame from :func:`load_rt_aligned`.
        output_dir: Directory to save the CSV file.
        run_name:   Run name used in the filename.

    Returns:
        Absolute path to the saved ``.csv`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    col_map = {
        "compound_name":    "Compound Name",
        "inchi_key":        "InChIKey",
        "adduct":           "Adduct",
        "polarity":         "Polarity",
        "chromatography":   "Chromatography",
        "rt_peak_measured": "Measured RT (min)",
        "rt_peak_aligned":  "Aligned RT (min)",
        "rt_delta":         "RT Delta (min)",
        "peak_intensity":   "Peak Intensity",
        "mz_observed":      "Observed m/z",
        "mz_theoretical":   "Theoretical m/z",
        "filename":         "Source File",
        "analyst_notes":    "Analyst Notes",
        "annotation_notes": "Annotation Notes",
        "rt_min_aligned":   "RT Min Aligned (min)",
        "rt_max_aligned":   "RT Max Aligned (min)",
        "rt_model_type":    "RT Model Type",
        "rt_model_r2":      "RT Model R²",
        "rt_model_rmse":    "RT Model RMSE (min)",
    }

    available_cols = [c for c in col_map if c in aligned_df.columns]
    out_df = aligned_df[available_cols].copy()
    out_df = out_df.rename(columns={c: col_map[c] for c in available_cols})

    # Flatten list columns for CSV
    for col in out_df.columns:
        out_df[col] = out_df[col].apply(
            lambda x: ", ".join(str(v) for v in x) if isinstance(x, (list, np.ndarray)) else x
        )

    out_df = out_df.sort_values(
        ["Chromatography", "Polarity", "Compound Name"],
        ignore_index=True,
    )

    csv_path = output_dir / f"RTAlignment-Table__{run_name}.csv"
    try:
        out_df.to_csv(csv_path, index=False)
        logger.info(f"RT alignment table saved to {csv_path}")
    except Exception as exc:
        logger.error(f"Failed to save RT alignment table: {exc}")

    return str(csv_path)


def make_atlas_csvs(
    aligned_df: pd.DataFrame,
    output_dir: str | Path,
    run_name: str,
) -> list[str]:
    """Generate atlas CSV files ready for ``add-atlases``.

    Groups by ``(chromatography, polarity)`` and writes one CSV per group
    with the column spec expected by ``load_atlas_input()``:
    ``inchi_key, compound_name, rt_peak, mz, adduct, rt_min, rt_max, mz_tolerance,
    identification_notes, lcmsrun_source, chromatography, polarity``.

    *aligned_df* is the output of :func:`~metatlas2.standards_annotation_db.load_rt_aligned`
    which joins ``sa_rt_aligned`` with ``sa_peaks`` and ``sa_compounds``.

    Args:
        aligned_df: Full joined DataFrame from :func:`load_rt_aligned`.
        output_dir: Directory to save the CSV files.
        run_name:   Run name used in filenames.

    Returns:
        List of absolute paths to the saved CSV files.
    """
    import os as _os

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    required_cols = {
        "inchi_key", "compound_name", "rt_peak_aligned",
        "mz_theoretical", "adduct", "rt_min_aligned", "rt_max_aligned",
        "chromatography", "polarity",
    }
    missing = required_cols - set(aligned_df.columns)
    if missing:
        logger.error(
            f"make_atlas_csvs: aligned_df is missing required columns: {missing}"
        )
        return []

    for (chrom, polarity), grp in aligned_df.groupby(["chromatography", "polarity"]):
        # Use mz_observed if available and valid, else fall back to mz_theoretical
        mz_col = grp["mz_observed"].where(
            grp["mz_observed"].notna() & (grp["mz_observed"] > 0),
            grp["mz_theoretical"],
        )

        # Build identification_notes from analyst_notes + annotation_notes
        def _id_notes(row):
            parts = [
                str(row.get("analyst_notes", "") or "").strip(),
                str(row.get("annotation_notes", "") or "").strip(),
            ]
            return "; ".join(p for p in parts if p)

        id_notes = grp.apply(_id_notes, axis=1)

        # lcmsrun_source: basename of the source h5 file
        lcmsrun_source = grp["filename"].apply(
            lambda fn: _os.path.basename(str(fn)) if fn else ""
        )

        atlas_df = pd.DataFrame({
            "inchi_key":             grp["inchi_key"],
            "compound_name":         grp["compound_name"],
            "rt_peak":               grp["rt_peak_aligned"].round(4),
            "mz":                    mz_col.round(5),
            "adduct":                grp["adduct"],
            "rt_min":                grp["rt_min_aligned"].round(4),
            "rt_max":                grp["rt_max_aligned"].round(4),
            "mz_tolerance":          5.0,
            "identification_notes":  id_notes.values,
            "lcmsrun_source":        lcmsrun_source.values,
            "chromatography":        grp["chromatography"].values,
            "polarity":              grp["polarity"].values,
        })

        # Deduplicate: keep one row per (inchi_key, adduct) — highest intensity
        if "peak_intensity" in grp.columns:
            grp_with_int = grp.copy()
            grp_with_int["_intensity"] = grp_with_int["peak_intensity"].fillna(0)
            idx_keep = (
                grp_with_int
                .sort_values("_intensity", ascending=False)
                .drop_duplicates(subset=["inchi_key", "adduct"])
                .index
            )
            atlas_df = atlas_df.loc[idx_keep].reset_index(drop=True)
        else:
            atlas_df = atlas_df.drop_duplicates(
                subset=["inchi_key", "adduct"]
            ).reset_index(drop=True)

        atlas_df = atlas_df.sort_values(["rt_peak", "compound_name"]).reset_index(drop=True)

        chrom_str = str(chrom).replace(" ", "_")
        pol_str = str(polarity).upper()
        csv_filename = f"RefStd_Atlas__{chrom_str}__{pol_str}__{run_name}.csv"
        csv_path = output_dir / csv_filename

        try:
            atlas_df.to_csv(csv_path, index=False)
            logger.info(f"Atlas CSV saved for chrom={chrom}, polarity={polarity} with {len(atlas_df)} compounds")
            saved_paths.append(str(csv_path))
        except Exception as exc:
            logger.error(f"Failed to save atlas CSV {csv_path}: {exc}")

    return saved_paths


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_reverse_rt_alignment(config_path: str, db_path: str, output_path: str) -> None:
    """Orchestrate the full reverse RT alignment workflow.

    Steps:
      1. Load config and project DB.
      2. Load analyst selections via ``load_selected_peaks_with_ms2()``.
      3. Check if alignment is already complete (resumability).
      4. Find QC h5 files for each chromatography.
      5. Load QC atlas from main DB.
      6. Extract QC MS1 data.
      7. Build reverse RT model per chromatography.
      8. Apply model to selected peaks.
      9. Save aligned rows to ``sa_rt_aligned`` (peak_uid PK).
      10. Generate outputs (PDF, summary CSV, atlas CSVs).

    Args:
        config_path: Absolute or relative path to the standards-annotation
                     YAML config file.
    """
    from metatlas2.load_tools import load_standards_config
    from metatlas2.standards_annotation_db import (
        get_project_output_root,
        load_selected_peaks_with_ms2,
        save_rt_aligned,
        load_rt_aligned,
        is_rt_alignment_complete,
    )

    config_path = str(Path(config_path).resolve())
    config = load_standards_config(config_path)
    rt_cfg = _get_rt_alignment_config(config)

    run_name = config["project"]["standards_run_name"]
    output_root = get_project_output_root(config)
    rt_align_output_dir = output_root / "outputs" / "rt_alignment"
    atlas_output_dir = output_root / "outputs" / "atlas_csvs"
    rt_align_output_dir.mkdir(parents=True, exist_ok=True)
    atlas_output_dir.mkdir(parents=True, exist_ok=True)

    prov = get_provenance()

    selections_df = load_selected_peaks_with_ms2(db_path)
    if selections_df.empty:
        logger.warning(
            "No analyst selections found. "
            "Run the GUI annotation first, then re-run reverse-rt-align."
        )
        return

    logger.info(f"Loaded {len(selections_df)} analyst selections from {db_path}")

    if is_rt_alignment_complete(db_path):
        logger.info(
            "RT alignment already complete (sa_rt_aligned has rows). "
            "Regenerating outputs from existing aligned data."
        )
        aligned_df = load_rt_aligned(db_path)
        _generate_outputs(aligned_df, rt_align_output_dir, atlas_output_dir, run_name, model=None, db_path=db_path)
        return

    qc_files_by_chrom = find_qc_files(config, db_path)

    qc_atlas = None
    if qc_files_by_chrom:
        try:
            qc_atlas = load_qc_atlas(config)
        except Exception as exc:
            logger.warning(
                f"Could not load QC atlas: {exc}. "
                "RT alignment will use identity model (no shift)."
            )

    all_aligned_chunks: list[pd.DataFrame] = []
    chroms_in_selections = selections_df["chromatography"].unique().tolist()

    for chrom in chroms_in_selections:
        chrom_selections = selections_df[
            selections_df["chromatography"] == chrom
        ].copy()

        model = None

        if qc_atlas is not None and chrom in qc_files_by_chrom:
            qc_ms1_by_chrom = extract_qc_ms1(
                {chrom: qc_files_by_chrom[chrom]},
                qc_atlas,
                config,
            )
            qc_ms1_df = qc_ms1_by_chrom.get(chrom)
            if qc_ms1_df is not None and not qc_ms1_df.empty:
                model = build_reverse_rt_model(qc_ms1_df, qc_atlas, config)

        if model is None:
            logger.warning(
                f"No RT model available for chromatography={chrom}. "
                "Using identity model (rt_peak_aligned = rt_peak)."
            )
            chrom_aligned = chrom_selections.copy()
            chrom_aligned["rt_peak_aligned"] = chrom_aligned["rt_peak"]
            chrom_aligned["rt_delta"] = 0.0
            chrom_aligned["rt_min_aligned"] = chrom_aligned["rt_min"]
            chrom_aligned["rt_max_aligned"] = chrom_aligned["rt_max"]
            chrom_aligned["rt_model_type"] = "identity"
            chrom_aligned["rt_model_degree"] = 0
            chrom_aligned["rt_model_r2"] = float("nan")
            chrom_aligned["rt_model_rmse"] = float("nan")
        else:
            chrom_aligned = apply_reverse_rt_model(chrom_selections, model)
            chrom_aligned["rt_model_type"] = model.get("model_type", "polynomial")
            chrom_aligned["rt_model_degree"] = model.get("degree", 0)
            chrom_aligned["rt_model_r2"] = model.get("r2", float("nan"))
            chrom_aligned["rt_model_rmse"] = model.get("rmse", float("nan"))

            modeling_df = model.get("modeling_df")
            if modeling_df is not None and not modeling_df.empty:
                try:
                    make_rt_alignment_summary_plot(
                        modeling_df=modeling_df,
                        model=model,
                        output_dir=rt_align_output_dir,
                        run_name=f"{run_name}__{chrom}",
                    )
                except Exception as exc:
                    logger.error(
                        f"Failed to generate RT alignment summary plot "
                        f"for chrom={chrom}: {exc}"
                    )

        chrom_aligned["qc_atlas_uid"] = rt_cfg.get("qc_atlas_uid", "")
        all_aligned_chunks.append(chrom_aligned)

    if not all_aligned_chunks:
        logger.error("No aligned data produced. Aborting.")
        return

    aligned_df = pd.concat(all_aligned_chunks, ignore_index=True)

    rows_to_save = _aligned_df_to_rows(aligned_df, prov)
    save_rt_aligned(db_path, rows_to_save)

    full_aligned_df = load_rt_aligned(db_path)
    _generate_outputs(full_aligned_df, rt_align_output_dir, atlas_output_dir, run_name, model=None, db_path=db_path)

    logger.info("Reverse RT alignment complete.")
    logger.info(f"Outputs written to: {rt_align_output_dir}")


def _aligned_df_to_rows(
    aligned_df: pd.DataFrame,
    prov: dict,
) -> list[dict]:
    """Convert the aligned DataFrame to a list of dicts for DB insertion.

    Only stores the minimal ``sa_rt_aligned`` columns (peak_uid + aligned RT
    values + model metadata).  All other peak metadata is in ``sa_peaks``.

    Args:
        aligned_df: DataFrame from :func:`apply_reverse_rt_model`.
        prov:       Provenance dict from :func:`metatlas2.utils.get_provenance`.

    Returns:
        List of dicts matching the ``sa_rt_aligned`` schema.
    """
    rows = []
    for _, row in aligned_df.iterrows():
        rows.append({
            "peak_uid":        str(row["peak_uid"]),
            "rt_peak_aligned": float(row.get("rt_peak_aligned", row.get("rt_peak", 0.0))),
            "rt_min_aligned":  float(row.get("rt_min_aligned", row.get("rt_min", 0.0))),
            "rt_max_aligned":  float(row.get("rt_max_aligned", row.get("rt_max", 0.0))),
            "rt_delta":        float(row.get("rt_delta", 0.0)),
            "qc_atlas_uid":    row.get("qc_atlas_uid", ""),
            "rt_model_type":   row.get("rt_model_type"),
            "rt_model_degree": row.get("rt_model_degree"),
            "rt_model_r2":     row.get("rt_model_r2"),
            "rt_model_rmse":   row.get("rt_model_rmse"),
            "created_by":      prov["analyst"],
            "created_date":    prov["timestamp"],
        })
    return rows


def _to_list(val: Any) -> list:
    """Coerce a value to a plain Python list."""
    if val is None:
        return []
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def _generate_outputs(
    aligned_df: pd.DataFrame,
    rt_align_output_dir: Path,
    atlas_output_dir: Path,
    run_name: str,
    model: dict | None,
    db_path: str | None = None,
) -> None:
    """Generate all output files from the aligned DataFrame.

    Args:
        aligned_df: DataFrame with aligned RT values.
        rt_align_output_dir: Directory to write RT alignment outputs.
        atlas_output_dir: Directory to write atlas CSVs.
        run_name:   Run name for filenames.
        model:      Optional model dict (used for summary plot if provided).
        db_path:    Optional project DB path for MSMS refs export.
    """
    from metatlas2.standards_annotation_summary import export_msms_refs_tab

    # Generate summary CSV
    try:
        make_rt_alignment_table(aligned_df, rt_align_output_dir, run_name)
    except Exception as exc:
        logger.error(f"Failed to generate RT alignment table: {exc}")

    # Atlas CSVs
    try:
        csv_paths = make_atlas_csvs(aligned_df, atlas_output_dir, run_name)
        for p in csv_paths:
            logger.info(f"Atlas CSV: {p}")
    except Exception as exc:
        logger.error(f"Failed to generate atlas CSVs: {exc}")

    # MSMS refs tab (in its own subfolder)
    if db_path is not None:
        try:
            msms_refs_dir = Path(rt_align_output_dir).parent / "msms_refs"
            export_msms_refs_tab(
                db_path=db_path,
                output_dir=msms_refs_dir,
                project_name=run_name,
            )
        except Exception as exc:
            logger.error(f"Failed to generate MSMS refs tab: {exc}")
