import pyarrow.dataset as ds
import pyarrow as pa

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