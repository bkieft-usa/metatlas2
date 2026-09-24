# Adding Compounds to the Database

`add_compounds_to_db.py` reads compound data from one or more TSV/CSV input files (described in a YAML config), retrieves additional metadata from PubChem, and writes everything to the central metatlas DuckDB database.

---

## Prerequisites

Complete the one-time environment setup described in [initial_setup.md](initial_setup.md) before running this script. Each compound input file (TSV or CSV) must also contain at minimum the required columns described [below](#compound-input-file-format).

---

## Command-line usage

```bash
metatlas2.sh add-compounds --config_path /path/to/create_compounds.yaml
```

The `metatlas2.sh` wrapper runs the command inside a Shifter (NERSC) or Docker (local) container. Shifter auto-mounts all NERSC GPFS filesystems read-write, so the script can write to `metatlas.duckdb`.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config_path` | Yes | — | Path to the compounds YAML config file (e.g. `configs/create_compounds.yaml`). |
| `--overwrite_db` | No | `False` | If set, **drops and recreates** the main database before loading. Use with caution — this deletes all existing compound, atlas, and MS/MS reference records. Only used for debugging purposes. |

---

## Config file: `create_compounds.yaml`

The config file has two top-level sections: `PARAMS` and `COMPOUNDS`.

```yaml
PARAMS:
  use_pubchem_cache: true       # Use a local PubChem cache file to avoid repeated web requests
  update_pubchem_cache: false   # Fetch fresh PubChem data and update the local cache (essentially ignores use_pubchem_cache)

COMPOUNDS:
  <CHROMATOGRAPHY>:             # HILICZ, C18
    <POLARITY>:                 # POS, NEG
      PATHS:
        - /path/to/a_compounds_file.tsv   # One or more input files
        - /path/to/another_compounds_file.tsv
```

### `PARAMS` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `use_pubchem_cache` | bool | `true` | Look up existing cached PubChem data before making network requests. Recommended for reproducibility and speed. |
| `update_pubchem_cache` | bool | `false` | Fetch fresh data from PubChem and write it back to the local cache file. Set to `true` when loading compounds for the first time or when cached data is stale (i.e., you know updates are needed to the compounds you input). |

### `COMPOUNDS` structure

Each entry under `COMPOUNDS` is organised as:

```
COMPOUNDS:
  <CHROMATOGRAPHY_LABEL>:
    <POLARITY>:
      PATHS:
        - <path_to_input_file>
```

- **Chromatography label** — arbitrary string identifying the chromatographic method (e.g. `HILICZ`, `C18`). Must match the labels used later in `create_atlases.yaml` and `analysis.yaml`.
- **Polarity** — `POS` or `NEG`.
- **PATHS** — list of absolute or relative file paths to compound input files. Empty entries (bare `-`) are silently skipped. All paths are validated to exist before any database writes occur.

### Example

```yaml
PARAMS:
  use_pubchem_cache: true
  update_pubchem_cache: false

COMPOUNDS:
  HILICZ:
    POS:
      PATHS:
        - /data/atlases/HILICZ/HILICZ_QC_POS.tsv
        - /data/atlases/HILICZ/HILICZ_ISTD_POS.tsv
        - /data/atlases/HILICZ/HILICZ_EMA_POS.tsv
    NEG:
      PATHS:
        - /data/atlases/HILICZ/HILICZ_ISTD_NEG.tsv
        - /data/atlases/HILICZ/HILICZ_EMA_NEG.tsv
  C18:
    POS:
      PATHS:
        -           # empty — skipped
```

---

## Compound input file format

Each file must be a **tab-separated (TSV)** or comma-separated (CSV) table. Files with any extension other than `.csv` are read as TSV.

### Required columns

| Column | Description |
|---|---|
| `inchi_key` | Standard InChIKey identifier for the compound (e.g. `BPGDAMSIGCZZLK-UHFFFAOYSA-N`). |
| `compound_name` | Human-readable name of the compound. |

### Optional / enriched columns

These columns are used if present. Missing numeric columns are set to `0.0` or empty string; PubChem enrichment can fill many of them automatically.

| Column | Description |
|---|---|
| `inchi` | Full InChI string. |
| `smiles` | SMILES string. |
| `formula` | Molecular formula (e.g. `C6H12O6`). |
| `mono_isotopic_molecular_weight` | Monoisotopic molecular weight (Da). |
| `iupac_name` | IUPAC systematic name. |
| `pubchem_cid` | PubChem compound ID. |
| `cas_number` | CAS registry number. |
| `synonyms` | Semicolon-separated list of synonyms. |
| `classes` | Classification labels (e.g. `Amino acid`). Stored in the `classes` column of the `compounds` table. |
| `pathways` | Pathway associations. Stored in the `pathways` column of the `compounds` table. |
| `tags` | Free-text tags. Stored in the `tags` column of the `compounds` table. |

---

## What the script does

1. Loads and validates the config file (`NewCompoundsConfig.from_yaml()`), including checking that all input file paths exist.
2. Creates (or overwrites) the main DuckDB database if needed (`dbi.create_metatlas_database()`).
3. For each non-empty path listed under `COMPOUNDS`:
   a. Reads the TSV/CSV file and validates required columns.
   b. Queries PubChem (using local JSON cache if `use_pubchem_cache: true`) to enrich metadata.
   c. Creates `Compound` dataclass objects from each row.
4. Bulk-inserts all compounds into the database (`dbi.batch_save_compounds()`). Duplicate compounds (same InChIKey) are ignored to avoid creating multiple entries for the same molecule.
5. Logs a summary of compounds loaded per file.

---

## Notes

- Running the script multiple times with the same input compounds is safe — duplicate compounds (matched by InChIKey) are silently skipped.
- Compounds must exist in the main database before atlases referencing them can be created (see [add_atlases_to_db.md](add_atlases_to_db.md)).
- The PubChem cache is stored at `$METATLAS_DATA_DIR/databases/pubchem_cache/pubchem_global_cache.json`. Set `update_pubchem_cache: true` to refresh it with the latest PubChem data for your compounds.
- Unlike atlases, compounds do **not** store RT/MZ reference data — that information lives in the `compound_mzrt` table and is populated when atlases are created via `add_atlases_to_db.py`.
