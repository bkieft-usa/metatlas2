
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf
import metatlas2.analysis_summary as asm

from metatlas2.utils import jsonable_list

logger = lcf.get_logger('analysis_summary')

_COMPOUND_FILE_STR_COLS = [
    "mz_rt_uid", "compound_name", "identified_metabolite", "label", "inchi_key", "formula", "smiles",
    "inchi", "pubchem_cid", "iupac_name", "adduct", "best_ms1_file",
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

_PARTITION_SCHEMA = pa.schema([
    ("project_name", pa.string()),
    ("chromatography", pa.string()),
    ("polarity", pa.string()),
    ("analysis_type", pa.string()),
    ("analysis_name", pa.string()),
])

_PARTITIONING = ds.partitioning(_PARTITION_SCHEMA, flavor="hive")

_PARTITION_COLS_DESC = [
    {"column_name": "chromatography", "dtype": "string"},
    {"column_name": "polarity",       "dtype": "string"},
    {"column_name": "analysis_type",  "dtype": "string"},
    {"column_name": "analysis_name",  "dtype": "string"},
]

_GRAIN_SUBDIR = {
    "compound_file": "compound_file",
    "compound_lfc": "compound_lfc",
}

_SPECIAL_HANDLERS = {
    "condition_pair": "_handle_condition_pair",
    "lfc_directional_min": "_handle_lfc_directional_min",
}

def _build_schema_map_df(str_cols: list[str], float_cols: list[str], table_name: str) -> pd.DataFrame:
    """Schema map for a single dataset grain (compound_file or compound_lfc).

    Partition columns are documented separately with source='partition_directory'
    because Arrow strips them from the physical Parquet files on write; they are
    only reconstructed by Hive-aware readers (pyarrow.dataset, or DuckDB's
    read_parquet(..., hive_partitioning=true)). Reading a leaf file directly
    with pyarrow.parquet.read_table will NOT surface them.
    """
    rows: list[dict] = []
    for idx, col in enumerate(str_cols):
        rows.append({"column_name": col, "dtype": "string", "table": table_name,
                     "position": idx, "source": "stored"})
    for idx, col in enumerate(float_cols):
        rows.append({"column_name": col, "dtype": "float", "table": table_name,
                     "position": idx, "source": "stored"})
    for pcol in _PARTITION_COLS_DESC:
        rows.append({"column_name": pcol["column_name"], "dtype": pcol["dtype"],
                     "table": table_name, "position": "", "source": "partition_directory"})

    df = pd.DataFrame(rows)
    df.sort_values("column_name", inplace=True)
    return df


def _write_schema_maps(schema_dir: Path, project_name: str) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)

    file_schema = _build_schema_map_df(_COMPOUND_FILE_STR_COLS, _COMPOUND_FILE_FLOAT_COLS, "compound_file")
    lfc_schema = _build_schema_map_df(_COMPOUND_LFC_STR_COLS, _COMPOUND_LFC_FLOAT_COLS, "compound_lfc")

    file_schema.to_csv(schema_dir / f"{project_name}-compound_file.schema_map.csv", index=False)
    file_schema.to_json(schema_dir / f"{project_name}-compound_file.schema_map.json", orient="records", indent=2)
    lfc_schema.to_csv(schema_dir / f"{project_name}-compound_lfc.schema_map.csv", index=False)
    lfc_schema.to_json(schema_dir / f"{project_name}-compound_lfc.schema_map.json", orient="records", indent=2)

def _with_partition_columns(table: pa.Table, summary_obj: "AnalysisSummary") -> pa.Table:
    partition_values = {
        "project_name": summary_obj.project_name,
        "chromatography": summary_obj.chromatography,
        "polarity": summary_obj.polarity,
        "analysis_type": summary_obj.analysis_type,
        "analysis_name": summary_obj.analysis_name,
    }
    collisions = [c for c in partition_values if c in table.column_names]
    if collisions:
        raise ValueError(
            f"Partition column(s) {collisions} already exist in the source table."
        )
    n = table.num_rows
    for name, value in partition_values.items():
        table = table.append_column(name, pa.array([value] * n))
    return table


def _partition_leaf_dir(root: Path, summary_obj: "AnalysisSummary") -> Path:
    return (
        root
        / f"project_name={summary_obj.project_name}"
        / f"chromatography={summary_obj.chromatography}"
        / f"polarity={summary_obj.polarity}"
        / f"analysis_type={summary_obj.analysis_type}"
        / f"analysis_name={summary_obj.analysis_name}"
    )


def _build_footer_metadata(summary_obj: "AnalysisSummary") -> dict[bytes, bytes]:
    footer_meta: dict[bytes, bytes] = {
        b"project_name":        str(summary_obj.project_name or "").encode(),
        b"rt_alignment_number": str(summary_obj.rt_alignment_number or "").encode(),
        b"analysis_number":     str(summary_obj.analysis_number or "").encode(),
        b"created_date":        str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')).encode(),
    }
    for key, val in _parse_project_metadata(summary_obj.project_name).items():
        footer_meta[key.encode()] = val.encode()
    return footer_meta

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

def make_analysis_parquet(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    parquet_output_path = Path(summary_obj.paths["parquet_output_dir"])

    if not overwrite:
        file_leaf = _partition_leaf_dir(parquet_output_path / "compound_file", summary_obj)
        lfc_leaf = _partition_leaf_dir(parquet_output_path / "compound_lfc", summary_obj)
        if any(file_leaf.glob("*.parquet")) and any(lfc_leaf.glob("*.parquet")):
            logger.info(f"Overwriting disabled: existing partition data found for {summary_obj.project_name}.")
            return

    schema_dir = parquet_output_path / "schema_maps"
    _write_schema_maps(schema_dir, summary_obj.project_name)

    footer_meta = _build_footer_metadata(summary_obj)
    parquet_format = ds.ParquetFileFormat()
    write_options = parquet_format.make_write_options(
        compression="zstd", compression_level=3,
        write_statistics=True, use_dictionary=True,
        data_page_size=1 * 1024 * 1024,
    )

    def _write_grain(build_fn, grain_name: str) -> None:
        logger.info(f"Building {grain_name} table...")
        table = build_fn(summary_obj)
        if table.num_rows == 0:
            logger.warning(f"{grain_name} table is empty — file not written.")
            return

        table = _with_partition_columns(table, summary_obj)
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, **footer_meta})

        ds.write_dataset(
            table,
            parquet_output_path / grain_name,
            format="parquet",
            partitioning=_PARTITIONING,
            file_options=write_options,
            basename_template=(
                f"RTA{summary_obj.rt_alignment_number}-TGA{summary_obj.analysis_number}-{{i}}-{str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))}.parquet"
            ),
            max_rows_per_group=100_000,
            existing_data_behavior="delete_matching",
        )
        logger.info(f"Wrote {grain_name} ({table.num_rows} rows) for project={summary_obj.project_name}")

    _write_grain(_build_compound_per_file_table, "compound_file")
    _write_grain(_build_compound_lfc_table, "compound_lfc")

    logger.info("Analysis Parquet export complete")

def _build_best_ms2_summary_df(summary_obj: "AnalysisSummary") -> pd.DataFrame:
    """Return one-row-per-compound best-MS2 metrics for parquet export.

    Includes best hit score metadata and a compact spectrum payload encoded as
    JSON ``[[rt_list], [mz_list]]`` where ``rt_list`` repeats the scan RT for
    each matched m/z from the best hit query-aligned spectrum.
    """

    from metatlas2.workflow_objects import MS2Hit

    ms2_df = summary_obj.experimental_data.ms2_df
    if ms2_df is None or ms2_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for uid, grp in ms2_df.groupby("mz_rt_uid", sort=False):
        best_score = -np.inf
        best_hit: "MS2Hit" | None = None

        for _, scan_row in grp.iterrows():
            hits = MS2Hit.list_from_scan_row(scan_row)
            if hits and hits[0].has_score and hits[0].score > best_score:
                best_score = hits[0].score
                best_hit = hits[0]

        if best_hit is None:
            continue

        scan_rt = best_hit.scan_rt
        q_mz = np.array(best_hit.aligned.query_mzs, dtype=float)
        valid_mz = q_mz[np.isfinite(q_mz)].tolist() if q_mz.size > 0 else []
        rt_list = [scan_rt] * len(valid_mz)

        matched_frags = best_hit.matched_fragments
        num_matches = best_hit.num_matches
        ref_frags = best_hit.ref_frags

        rows.append(
            {
                "mz_rt_uid": str(uid),
                "best_ms2_file": os.path.basename(best_hit.filename),
                "best_ms2_rt": scan_rt,
                "best_ms2_score": best_hit.score,
                "best_ms2_mz": best_hit.mz_measured,
                "best_ms2_num_ions": f"{num_matches}/{ref_frags}" if ref_frags > 0 else str(num_matches),
                "best_ms2_matching_ions": ",".join(f"{float(m):.3f}" for m in matched_frags) if matched_frags else "",
                "best_ms2_spectrum_rt_mz": json.dumps([jsonable_list(rt_list), jsonable_list(valid_mz)]),
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

    # check for unexpected polarity values in curation_df
    row_polarities = mc["polarity"].dropna().unique() if "polarity" in mc.columns else []
    unexpected = [p for p in row_polarities if p != summary_obj.polarity]
    if unexpected:
        raise ValueError(
            f"curation_df contains polarity values {unexpected!r} that differ from "
            f"summary_obj.polarity={summary_obj.polarity!r}; "
            f"these rows may belong to the wrong partition."
        )

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

        mz_q  = asm.mz_quality(ppm_err, da_err)
        rt_q  = asm.rt_quality(rt_error_ms1, chromatography)
        ms2_notes = str(mc_row.get("ms2_notes", "") or "")
        try:
            msms_q = float(ms2_notes.split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = asm.total_score_and_msi(msms_q, mz_q, rt_q)

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
            logger.warning(f"Could not read control_filter from {filtered_csv}: {exc}")

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
                lambda r: json.dumps([jsonable_list(r.get("spec_rts")), jsonable_list(r.get("spec_ints"))]),
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
        logger.warning(f"No LFC columns found in {lfc_csv} — compound_lfc table will be empty.")
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

def project_root_from_leaf(leaf_path: str | Path) -> Path:
    """Derive project_root from any path inside a project's parquet tree.

    Works whether you pass a leaf .parquet file, a partition directory,
    or the parquet_results directory itself.
    """
    p = Path(leaf_path).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == "parquet_results":
            return ancestor.parent
    raise ValueError(
        f"Could not find a 'parquet_results' directory among the parents of {leaf_path}"
    )