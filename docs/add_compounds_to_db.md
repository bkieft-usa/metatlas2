# Adding Compounds to the Database

`add_compounds_to_db.py` reads a single compound CSV/TSV file and writes its records to the central metatlas DuckDB database. Each record is keyed by InChIKey.

IMPORTANT: Only add compounds to the database if you have run legitimate reference standards on JGI HPLC-MS/MS machines.

---

## Prerequisites

Complete the one-time environment setup described in [initial_setup.md](initial_setup.md) before running this script. Each compound input file (TSV or CSV) must also contain at minimum the required columns described [below](#compound-input-file-format).

---

## Command-line usage

```bash
metatlas2.sh add-compounds --input_csv /path/to/compounds.csv
```

The `metatlas2.sh` wrapper runs the command inside a Shifter (NERSC) or Docker (local) container. Shifter auto-mounts all NERSC GPFS filesystems read-write, so the script can write to `metatlas.duckdb`.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_csv` | Yes | — | Path to a single CSV (or TSV) containing compounds. |
| `--overwrite_db` | No | `False` | If set, **drops and recreates** the main database before loading. Use with caution — this deletes all existing compound, atlas, and MS/MS reference records. Only used for debugging purposes. |

The CSV requires `inchi_key` and `label` (legacy files may use `compound_name` instead of `label`). `compound_name` is optional vendor/source metadata. `smiles` and `inchi` are optional. PubChem metadata is looked up even when structures are supplied; nonempty input values take precedence over PubChem values.

Minimal input example:

```csv
label,inchi_key
Adenine,GFFGJBXGBJISGV-UHFFFAOYSA-N
```

---

## Compound input file format

The direct input may be comma-separated (CSV) or tab-separated (TSV).

### Required columns

| Column | Description |
|---|---|
| `inchi_key` | Standard InChIKey identifier for the compound (e.g. `BPGDAMSIGCZZLK-UHFFFAOYSA-N`). |
| `label` | Canonical compound name used for display and matching. |

`label` is the canonical compound name used for display. The optional
`compound_name` column stores vendor/source naming and is kept separately.
Legacy files with only `compound_name` use that value as the label.

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

1. Reads the CSV/TSV file and validates required columns before database writes.
2. Creates (or overwrites) the main DuckDB database if needed (`dbi.create_metatlas_database()`).
3. Enriches metadata using PubChem, ModelSEED, and ChEBI/MetaCyc lookups, using cached results where available.
4. Creates `Compound` records and bulk-inserts them into the database (`dbi.batch_save_compounds()`).
5. Uses InChIKey as the identity key; an existing compound with the same InChIKey is not inserted again.

---

## Notes

- Running the script multiple times with the same input compounds is safe — duplicate compounds (matched by InChIKey) are silently skipped.
- Compounds must exist in the main database before atlases referencing them can be created (see [add_atlases_to_db.md](add_atlases_to_db.md)).
- PubChem data is cached at `$METATLAS_DATA_DIR/databases/pubchem_cache/pubchem_global_cache.json`.
- Existing databases must use the current schema; legacy database schemas are not migrated automatically.
- Unlike atlases, compounds do **not** store RT/MZ reference data — that information lives in the `compound_mzrt` table and is populated when atlases are created via `add_atlases_to_db.py`.
