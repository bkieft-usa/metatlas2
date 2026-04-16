# Adding Atlases to the Database

`add_atlases_to_db.py` reads atlas tables from TSV/CSV files (described in a YAML config), validates them, and writes them as reference atlases to the central metatlas DuckDB database. These reference atlases are the starting point for RT alignment and targeted analysis workflows.

---

## Prerequisites

Complete the one-time environment setup described in [initial_setup.md](initial_setup.md) before running this script. The main metatlas database must also already exist and be populated with compounds — run `add_compounds_to_db.py` first (see [add_compounds_to_db.md](add_compounds_to_db.md)). Each atlas input file (TSV or CSV) must contain at minimum the required columns described [below](#atlas-input-file-format).

---

## Command-line usage

```bash
metatlas2 add-atlases --config_path /path/to/create_atlases.yaml
```

The `metatlas2` wrapper runs the command inside a Shifter container. Shifter auto-mounts all NERSC GPFS filesystems read-write, so the script can write to `metatlas.duckdb`.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config_path` | Yes | — | Path to the atlas YAML config file (e.g. `configs/create_atlases.yaml`). |

---

## Config file: `create_atlases.yaml`

The config file has a single top-level section: `ATLASES`.

```yaml
ATLASES:
  <CHROMATOGRAPHY>:          # HILICZ, C18
    <POLARITY>:              # POS, NEG
      <ANALYSIS_TYPE>:       # QC, ISTD, EMA, or any label you define
        path: /path/to/atlas_table.tsv
        name: Human-readable atlas name
        desc: Human-readable short description of the atlas
```

### `ATLASES` structure

Each atlas is identified by three keys that form a hierarchy:

| Level | Key | Description |
|---|---|---|
| 1 | Chromatography | The LC method (e.g. `HILICZ`, `C18`). Must match the labels used in `analysis.yaml`. |
| 2 | Polarity | `POS` (positive ion mode) or `NEG` (negative ion mode). |
| 3 | Analysis type | The role of the atlas (e.g. `QC` for quality-control compounds used in RT alignment, `ISTD` for internal standards, `EMA` for experimental metabolite atlas compounds). |

Each atlas entry has three fields:

| Field | Required | Description |
|---|---|---|
| `path` | Yes* | Absolute path to the atlas TSV/CSV file. A bare value or null is skipped silently. |
| `name` | Yes* | Human-readable atlas name stored in the database. |
| `desc` | Yes* | Short description stored in the database. |

\* These fields must be present in the YAML but can be left empty (null). Entries with an empty `path` are skipped.

### Example

```yaml
ATLASES:
  HILICZ:
    POS:
      QC:
        path: /data/atlases/HILICZ/HILICZ_QC_POS.tsv
        name: Default HILICZ QC Atlas Positive
        desc: JGI QC compounds used for RT alignment
      ISTD:
        path: /data/atlases/HILICZ/HILICZ_ISTD_POS.tsv
        name: Default HILICZ ISTD Atlas Positive
        desc: JGI internal standard compounds
      EMA:
        path: /data/atlases/HILICZ/HILICZ_EMA_POS.tsv
        name: Default HILICZ EMA Atlas Positive
        desc: JGI experimental metabolite compounds
    NEG:
      QC:
        path:       # empty — skipped
        name:
        desc:
```

> **Tip:** The `uid` values assigned to atlases after creation are printed in the log output. Copy these UIDs into your analysis config file (e.g., `analysis.yaml`) to reference the new atlases in targeted workflows.

---

## Atlas input file format

Each file must be a **tab-separated (TSV)** or **comma-separated (CSV)** table. Note: files with any extension other than `.csv` are read as TSV.

### Required columns

| Column | Description |
|---|---|
| `inchi_key` | Standard InChIKey identifier (e.g. `BPGDAMSIGCZZLK-UHFFFAOYSA-N`). |
| `compound_name` | Human-readable compound name. |
| `rt_peak` | Expected retention time at peak apex (minutes). |
| `mz` | Expected precursor m/z value. |
| `adduct` | Ion adduct (e.g. `[M+H]+`, `[M-H]-`, `[M+NH4]+`). |

### Optional columns

If absent, defaults are applied automatically:

| Column | Default | Description |
|---|---|---|
| `rt_min` | `rt_peak − 0.5` | Lower RT bound for peak detection (minutes). |
| `rt_max` | `rt_peak + 0.5` | Upper RT bound for peak detection (minutes). |
| `mz_tolerance` | `5.0` | m/z tolerance in ppm. |

### Additional optional columns

These are stored in the database if present but are not required for validation:

| Column | Description |
|---|---|
| `inchi` | Full InChI string. |
| `smiles` | SMILES string. |
| `formula` | Molecular formula. |
| `mono_isotopic_molecular_weight` | Monoisotopic molecular weight (Da). |
| `compound_classes` | Classification labels. |
| `compound_pathways` | Pathway associations. |

---

## What the script does

1. Loads and validates the config file.
2. For each atlas entry with a non-empty `path`:
   a. Reads the TSV/CSV file and validates required columns.
   b. Applies default values for `rt_min`, `rt_max`, and `mz_tolerance` if absent.
   c. Constructs an `Atlas` object and validates it (checks for duplicate compounds, valid m/z and RT values, required adduct, etc.).
   d. Saves the atlas to the main DuckDB database.
3. Logs a summary table with each atlas UID, name, and compound count.

---

## Retrieving atlas UIDs

After running the script, atlas UIDs are printed in log output like:

```
Atlas: Default HILICZ QC Atlas Positive (UID: atl-ref-qc-hilicz-pos-cdf8c6709c6e4953b75917e72e851130) - 42 compounds
```

Copy these UIDs into your analysis YAML configuration file under the corresponding `ATLAS: uid:` fields before running the targeted analysis workflow.

---

## Notes

- Only atlases with a non-empty `path` are processed; entries with null paths are silently skipped. This allows the config template to remain the same and accomodate all atlas inputs.
- If an atlas file is listed in the config but the file cannot be found, a warning is logged and the entry is skipped rather than raising a fatal error, so make sure you check the logs after running if you don't see the expected atlases in the standard output.
- By design, re-running the script with the same config will create new atlases with new UIDs.
