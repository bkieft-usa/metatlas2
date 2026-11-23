# Metatlas2 Database Schema Documentation

This document describes the metatlas2 database schema, table structure, and how database tables relate to workflow objects.

---

## Table of Contents

- [Database Architecture Overview](#database-architecture-overview)
- [Main Database Schema](#main-database-schema)
  - [compounds Table](#compounds-table)
  - [compound_mzrt Table](#compound_mzrt-table)
  - [atlases Table](#atlases-table)
  - [atlas_compound_associations Table](#atlas_compound_associations-table)
  - [projects Table](#projects-table)
  - [msms_refs Table](#msms_refs-table)
- [Project Database Schema](#project-database-schema)
  - [lcmsruns Table](#lcmsruns-table)
  - [atlases Table (Project)](#atlases-table-project)
  - [compound_mzrt Table (Project)](#compound_mzrt-table-project)
  - [atlas_compound_associations Table (Project)](#atlas_compound_associations-table-project)
  - [rt_alignment Table](#rt_alignment-table)
  - [workflow_runs Table](#workflow_runs-table)
  - [project_config Table](#project_config-table)
  - [ms1_data Table](#ms1_data-table)
  - [ms2_data Table](#ms2_data-table)
  - [manual_curation Table](#manual_curation-table)
- [Standards Annotation Project Database](#standards-annotation-project-database)
- [Workflow Objects and Database Mapping](#workflow-objects-and-database-mapping)
- [Database Relationships and Visual Schema](#database-relationships-and-visual-schema)
  - [Main Database Schema Diagram](#main-database-schema-diagram)
  - [Project Database Schema Diagram](#project-database-schema-diagram)
  - [Entity Relationships](#entity-relationships)

---

## Database Architecture Overview

Metatlas2 uses lightweight **DuckDB** for data storage. The system maintains two types of databases:

### 1. **Main Database** (Knowledge Repository)
- **Purpose**: Central repository of reference compounds, atlases, and MS/MS reference spectra
- **Location**: Central location for all analysts, typically at `$METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb`
- **Scope**: Shared across all projects, analysts, and owners (i.e., JGI and EGSB)
- **Content**:
  - Compound metadata (chemical structures, identifiers, PubChem data)
  - Reference RT/MZ data (`compound_mzrt` entries, added when atlases are created)
  - Reference atlases (curated sets of compounds)
  - Compound-atlas associations (one compound can belong to many atlases)
  - MS/MS reference spectra (`msms_refs`)
  - Project registry (tracks all projects for meta-analysis)

### 2. **Project Database** (Experimental Results)
- **Purpose**: Stores project-specific experimental data and derived results in a standardized format
- **Location**: Within each project directory, typically at `$METATLAS_DATA_DIR/projects/targeted_outputs/<owner>/<user>/<project_name>/<project_name>.duckdb`
- **Scope**: Single database for each project run
- **Content**:
  - LCMS run metadata (information about each raw file)
  - Project-specific atlases (RT-aligned, Auto-IDed, manually curated, etc.)
  - RT alignment models and parameters
  - Extracted MS1/MS2 spectral data
  - Manual curation decisions and notes
  - Config snapshots and stored paths for reproducibility

This separation allows for:
- **Reusability**: Reference data shared across projects
- **Traceability**: Complete experimental history per project
- **Scalability**: Project databases are atomic and analysis time does not scale with project number/size
- **Provenance**: Track who created each entry and when

---

## Main Database Schema

The main database contains six core tables.

### compounds Table

Stores immutable chemical compound metadata. Each compound represents a unique molecular entity identified by its InChIKey.

| Column | Type | Description |
|--------|------|-------------|
| `compound_uid` | TEXT (PK) | Unique identifier (e.g., `cmp-a1b2c3...`) |
| `label` | TEXT | Canonical compound name used for display |
| `compound_name` | TEXT | Vendor-provided compound name |
| `inchi_key` | TEXT | InChI Key (molecular structure hash) |
| `inchi` | TEXT | Full InChI string |
| `smiles` | TEXT | SMILES structure notation |
| `formula` | TEXT | Molecular formula |
| `classes` | TEXT | Classification terms |
| `pathways` | TEXT | Biochemical pathways |
| `tags` | TEXT | Custom tags |
| `mono_isotopic_molecular_weight` | REAL | Monoisotopic molecular weight |
| `iupac_name` | TEXT | IUPAC systematic name |
| `pubchem_cid` | TEXT | PubChem compound ID |
| `cas_number` | TEXT | CAS registry number |
| `synonyms` | TEXT | Alternative names |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- `compound_uid` is the primary key used to reference compounds throughout the system
- `inchi_key` provides a standard, collision-resistant molecular identifier; duplicate InChIKeys are rejected on insert
- Chemical properties (InChI, SMILES, formula) are immutable once created

### compound_mzrt Table

Stores retention time (RT) and mass-to-charge ratio (m/z) reference data for compounds. A single compound can have multiple mzrt entries for different adducts, chromatography methods, or confidence levels. New entries are created whenever an atlas is added via `add_atlases_to_db.py`.

| Column | Type | Description |
|--------|------|-------------|
| `mz_rt_uid` | TEXT (PK) | Unique identifier (e.g., `mzrt-x1y2z3...`) |
| `compound_uid` | TEXT | Links to compounds table |
| `prev_mz_rt_uid` | TEXT | UID of the parent mzrt entry (for derived/updated entries) |
| `label` | TEXT | Canonical compound name from the atlas entry |
| `inchi_key` | TEXT | Denormalized for convenience |
| `adduct` | TEXT | Adduct form (e.g., `[M+H]+`, `[M-H]-`) |
| `rt_space` | TEXT | RT space identifier (e.g., `HF_Aug2019`) |
| `rt_peak` | REAL | Peak retention time (minutes) |
| `rt_min` | REAL | RT window start (minutes) |
| `rt_max` | REAL | RT window end (minutes) |
| `mz` | REAL | Mass-to-charge ratio |
| `mz_tolerance` | REAL | m/z tolerance (ppm) |
| `chromatography` | TEXT | Chromatography type (e.g., `hilicz`, `c18`) |
| `polarity` | TEXT | Ionization polarity (`positive` or `negative`) |
| `confidence` | TEXT | Identification confidence level |
| `source` | TEXT | Data origin (e.g., file path, reference) |
| `identification_notes` | TEXT | Notes about expected peak characteristics, displayed in the curation GUI |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Each entry represents a specific adduct of a compound under specific conditions
- RT values define the expected elution window for targeted extraction (minutes)
- `mz_tolerance` defines the m/z extraction window (typically 5–20 ppm)
- `prev_mz_rt_uid` links derived entries (e.g., RT-aligned, manually curated) back to their source

### atlases Table

Defines collections of compounds for targeted analysis. Atlases organize compound sets by analytical method and purpose. This table schema is **shared** between the main database and project databases.

| Column | Type | Description |
|--------|------|-------------|
| `atlas_uid` | TEXT (PK) | Unique identifier (e.g., `atl-ref-hilicz-pos-qc-...`) |
| `atlas_name` | TEXT | Human-readable atlas name |
| `atlas_description` | TEXT | Detailed description of atlas purpose |
| `chromatography` | TEXT | Chromatography method (e.g., `hilicz`, `c18`) |
| `polarity` | TEXT | Ionization polarity (`positive` or `negative`) |
| `analysis_type` | TEXT | Analysis category (e.g., `qc`, `istd`, `ema`) |
| `analysis_name` | TEXT | Named variant within the analysis type (e.g., `main`, `default`) |
| `atlas_stage` | TEXT | Atlas lifecycle stage (see below) |
| `source_atlas_uid` | TEXT | UID of parent atlas (if derived) |
| `rt_alignment_number` | INTEGER | RT alignment iteration (project DB only) |
| `analysis_number` | INTEGER | Analysis iteration (project DB only) |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |
| `source` | TEXT | File path or origin reference |

**Atlas Stages** (values of `atlas_stage`):
- `INPUT`: Original reference atlas loaded as input to a targeted analysis
- `RT_ALIGNED`: RT-adjusted atlas for a specific project
- `AUTO_IDED`: Auto-identified atlas from experimental data
- `MANUALLY_CURATED`: Manually curated/refined atlas

**Key Points**:
- Each atlas targets a specific analytical method (analysis type, chromatography, polarity)
- Atlases are collections; actual compounds are linked via `atlas_compound_associations`
- The `analysis_name` field (new vs. old schema) allows multiple named variants of the same analysis type

### atlas_compound_associations Table

Junction table linking atlases to their constituent compounds and mzrt entries. This table schema is **shared** between the main database and project databases.

| Column | Type | Description |
|--------|------|-------------|
| `association_uid` | TEXT (PK) | Unique association identifier |
| `atlas_uid` | TEXT (FK) | References atlases table |
| `compound_uid` | TEXT | Compound UID |
| `mz_rt_uid` | TEXT | References compound_mzrt table |
| `association_order` | INTEGER | Display/processing order |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Enables many-to-many relationship between atlases and compounds
- Each association links to a specific mzrt entry (adduct + method combination)
- `association_order` preserves compound ordering within the atlas
- In the main database, `atlas_uid` has a foreign key constraint; in project databases it does not

### projects Table

Tracks all projects created with metatlas2 for meta-analysis and project discovery. This table enables users to find all project database files across the system.

| Column | Type | Description |
|--------|------|-------------|
| `project_uid` | TEXT (PK) | Unique identifier (e.g., `prj-a1b2c3...`) |
| `project_name` | TEXT | Project name |
| `project_db_path` | TEXT | Absolute path to project database file |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Automatically populated when a project is set up via `Project.setup()`
- Enables discovery of all project databases for cross-project meta-analysis
- Each project is registered once; duplicate entries are prevented

### msms_refs Table

Stores MS/MS reference spectra for compound identification. Populated via `metatlas2.sh add-msms-refs`.

| Column | Type | Description |
|--------|------|-------------|
| `ref_uid` | TEXT (PK) | Unique identifier (e.g., `msms-ref-...`) |
| `database` | TEXT | Source database name (e.g., `metatlas`, `mzcloud`) |
| `ref_id` | TEXT | Reference spectrum ID in the source database |
| `name` | TEXT | Compound name from the reference |
| `inchi_key` | TEXT | InChI Key for matching to compounds |
| `precursor_mz` | REAL | Precursor m/z |
| `polarity` | TEXT | Ionization polarity (`positive` or `negative`) |
| `adduct` | TEXT | Adduct form |
| `fragmentation_method` | TEXT | Fragmentation method (e.g., `HCD`) |
| `collision_energy` | REAL | Collision energy (eV) |
| `instrument` | TEXT | Instrument name |
| `instrument_type` | TEXT | Instrument type |
| `formula` | TEXT | Molecular formula |
| `mono_isotopic_molecular_weight` | REAL | Monoisotopic molecular weight |
| `inchi` | TEXT | Full InChI string |
| `smiles` | TEXT | SMILES string |
| `mz` | REAL[] | Array of fragment m/z values |
| `intensities` | REAL[] | Array of fragment intensities |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Indexed on `inchi_key`, `polarity`, and `database` for fast lookup during MS2 matching
- Re-importing the same source file adds new rows (no deduplication); each row gets a fresh `ref_uid`
- Can be overridden at runtime with a custom `.jsonl` file via `GENERAL.msms_refs_path` in the analysis config

---

## Project Database Schema

Project databases share the `atlases`, `compound_mzrt`, and `atlas_compound_associations` tables with the main database schema, and add the following project-specific tables.

### lcmsruns Table

Catalogs all LCMS data files (`.raw`, `.mzML`, `.h5`) available for a project.

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | TEXT (PK) | Absolute path to the raw file |
| `filename` | TEXT | Base filename |
| `file_format` | TEXT | Original format (`raw`, `mzML`, `h5`) |
| `file_type` | TEXT | File category (`experimental`, `qc`, `istd`, `exctrl`, `injbl`, `refstd`) |
| `chromatography` | TEXT | Chromatography method |
| `ms_level` | TEXT | MS level (`ms1` or `ms2`) |
| `polarity` | TEXT | Ionization polarity |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Each row represents one raw file (`.raw`, `.mzML`, or `.h5`)
- `file_path` serves as primary key and reference for data extraction
- `file_type` is inferred from filename substrings (see [LCMS File Categorization](run_targeted_analysis.md#lcms-file-categorization))
- `ms_level` is inferred from the filename (e.g., `MS1` or `MS2` position in the filename)

### atlases Table (Project)

Identical schema to the main database `atlases` table. Project atlases are typically derived from main database reference atlases and have `rt_alignment_number` and `analysis_number` populated.

### compound_mzrt Table (Project)

Identical schema to the main database `compound_mzrt` table. MZRT data is copied from the main database and may be modified (e.g., RT-aligned values, manually curated RT bounds). The `prev_mz_rt_uid` field links each derived entry back to its source.

### atlas_compound_associations Table (Project)

Same schema as the main database, but without foreign key constraints on `compound_uid` and `mz_rt_uid` (since project databases may contain derived entries not present in the main DB).

### rt_alignment Table

Stores retention time alignment models and metadata.

| Column | Type | Description |
|--------|------|-------------|
| `rt_alignment_uid` | TEXT (PK) | Unique identifier |
| `project_name` | TEXT | Project name |
| `rt_alignment_number` | INTEGER | Alignment iteration number |
| `qc_atlas_uid` | TEXT | Atlas used for RT alignment |
| `model_type` | TEXT | Model type (e.g., `polynomial`, `linear`, `median_offset`) |
| `polynomial_degree` | INTEGER | Polynomial degree (if applicable) |
| `r_squared` | REAL | Model fit R² value |
| `rmse` | REAL | Root mean squared error |
| `coefficients` | TEXT | JSON-encoded model coefficients |
| `equation` | TEXT | Human-readable equation |
| `num_qc_files` | INTEGER | Number of QC files used |
| `num_compounds` | INTEGER | Number of compounds in model |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |
| `metadata` | TEXT | JSON-encoded additional metadata |

**Key Points**:
- Each RT alignment produces one model entry
- Quality metrics (R², RMSE) assess alignment quality
- `rt_alignment_number` links to aligned atlases in the `atlases` table

### workflow_runs Table

Tracks the lifecycle stage of each atlas through the workflow. Used to guard against re-running completed stages and to look up atlas UIDs by stage.

| Column | Type | Description |
|--------|------|-------------|
| `run_uid` | TEXT (PK) | Unique identifier |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `chromatography` | TEXT | Chromatography method |
| `polarity` | TEXT | Ionization polarity |
| `analysis_type` | TEXT | Analysis type |
| `analysis_name` | TEXT | Named analysis variant |
| `stage` | TEXT | Workflow stage (`RT_ALIGNED`, `AUTO_IDED`, `MANUALLY_CURATED`) |
| `atlas_uid` | TEXT | Atlas UID for this stage |
| `source_atlas_uid` | TEXT | Source atlas UID |
| `override_params` | TEXT | JSON-encoded parameter overrides (from GUI/summary) |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

### project_config Table

Stores the full config YAML and paths JSON for each run, enabling later stages (GUI, summary) to reconstruct the workflow context without the original YAML file.

| Column | Type | Description |
|--------|------|-------------|
| `config_uid` | TEXT (PK) | Unique identifier |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `config_yaml` | TEXT | Full YAML config as a JSON-serialized string |
| `paths_json` | TEXT | JSON-encoded paths dict |
| `config_path` | TEXT | Original path to the config file |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

### ms1_data Table

Stores extracted MS1 spectral data for each compound in each LCMS run.

| Column | Type | Description |
|--------|------|-------------|
| `mz_rt_uid` | VARCHAR | Compound MZRT identifier |
| `filename` | VARCHAR | LCMS run filename |
| `inchi_key` | VARCHAR | InChI key |
| `adduct` | VARCHAR | Adduct form |
| `spec_rts` | REAL[] | Array of retention times across the extraction window |
| `spec_ints` | REAL[] | Array of intensities across the extraction window |
| `spec_mzs` | REAL[] | Array of m/z values across the extraction window |
| `in_feature` | BOOLEAN[] | Boolean array indicating which points are within the atlas RT window |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | VARCHAR | Analysis workflow type (e.g., `istd`, `ema`) |
| `analysis_name` | VARCHAR | Named analysis variant |
| `created_by` | VARCHAR | Username of creator |
| `created_date` | VARCHAR | ISO timestamp of creation |

**Primary Key**: `(mz_rt_uid, filename, rt_alignment_number, analysis_number)`

**Key Points**:
- One entry per compound per LCMS run
- `spec_rts`, `spec_ints`, `spec_mzs` are parallel arrays of the full EIC across the extraction window
- `in_feature` marks which data points fall within the atlas RT bounds (used for peak detection)

### ms2_data Table

Stores extracted MS2 fragmentation spectra.

| Column | Type | Description |
|--------|------|-------------|
| `mz_rt_uid` | VARCHAR | Compound MZRT identifier |
| `filename` | VARCHAR | LCMS run filename |
| `inchi_key` | VARCHAR | InChI key |
| `adduct` | VARCHAR | Adduct form |
| `scan_rt` | REAL | Retention time of the MS2 scan |
| `frag_mzs` | REAL[] | Array of fragment m/z values |
| `frag_ints` | REAL[] | Array of fragment intensities |
| `precursor_MZ` | REAL | Precursor m/z |
| `precursor_intensity` | REAL | Precursor intensity |
| `collision_energy` | REAL | Collision energy (eV) |
| `in_feature` | BOOLEAN | Whether the scan falls within the atlas RT window |
| `hits` | VARCHAR | JSON-encoded list of MS2 library hit results |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | VARCHAR | Analysis workflow type |
| `analysis_name` | VARCHAR | Named analysis variant |
| `created_by` | VARCHAR | Username of creator |
| `created_date` | VARCHAR | ISO timestamp of creation |

**Primary Key**: `(mz_rt_uid, filename, scan_rt, rt_alignment_number, analysis_number)`

**Key Points**:
- Multiple MS2 scans can exist per compound per run
- `hits` stores the JSON-encoded MS2 library matching results (scores, matched fragments, reference spectra) for each scan
- `in_feature` indicates whether the scan falls within the atlas RT window

### manual_curation Table

Stores manual curation decisions and compound identification results. This is the central table for tracking analyst decisions.

| Column | Type | Description |
|--------|------|-------------|
| `mz_rt_uid` | VARCHAR | Compound MZRT identifier |
| `compound_uid` | VARCHAR | Compound identifier |
| `inchi_key` | VARCHAR | InChI key |
| `adduct` | VARCHAR | Adduct form |
| `label` | VARCHAR | Canonical compound name |
| `passed_autoid` | BOOLEAN | Whether the compound passed auto-identification filters |
| `passed_curation` | BOOLEAN | Whether the compound passed manual curation |
| `polarity` | VARCHAR | Ionization polarity |
| `chromatography` | VARCHAR | Chromatography method |
| `mz_tolerance` | REAL | m/z tolerance (ppm) |
| `atlas_mz` | REAL | Atlas reference m/z |
| `atlas_rt_peak` | REAL | Atlas reference RT peak |
| `atlas_rt_min` | REAL | Atlas reference RT min |
| `atlas_rt_max` | REAL | Atlas reference RT max |
| `mz` | REAL | Measured m/z |
| `rt_peak` | REAL | Curated RT peak |
| `rt_min` | REAL | Curated RT min |
| `rt_max` | REAL | Curated RT max |
| `initial_rt_min` | REAL | Pre-curation RT min (from auto-ID) |
| `initial_rt_max` | REAL | Pre-curation RT max (from auto-ID) |
| `rt_error` | REAL | RT error (measured − atlas RT) |
| `mz_error` | REAL | m/z error (ppm) |
| `ms1_notes` | VARCHAR | MS1 quality/decision note |
| `ms2_notes` | VARCHAR | MS2 quality/decision note |
| `other_notes` | VARCHAR | Additional observation note |
| `identification_notes` | VARCHAR | Identification rationale (from atlas) |
| `analyst_notes` | VARCHAR | Free-text analyst comments |
| `max_eic_rt` | REAL[] | RT values of the best EIC peak |
| `max_eic_intensity` | REAL[] | Intensity values of the best EIC peak |
| `isomers` | VARCHAR | Potential isomer information |
| `suggested_rt_min` | REAL | Algorithm-suggested RT min |
| `suggested_rt_max` | REAL | Algorithm-suggested RT max |
| `suggested_rt_peak` | REAL | Algorithm-suggested RT peak |
| `rt_suggestion_confidence` | REAL | Confidence in RT suggestion |
| `formula` | VARCHAR | Molecular formula |
| `smiles` | VARCHAR | SMILES string |
| `inchi` | VARCHAR | Full InChI string |
| `pubchem_cid` | VARCHAR | PubChem compound ID |
| `mono_isotopic_molecular_weight` | REAL | Monoisotopic molecular weight |
| `iupac_name` | VARCHAR | IUPAC systematic name |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | VARCHAR | Analysis workflow type |
| `analysis_name` | VARCHAR | Named analysis variant |
| `created_by` | VARCHAR | Username of creator |
| `created_date` | VARCHAR | ISO timestamp of creation |

**Primary Key**: `(mz_rt_uid, rt_alignment_number, analysis_number)`

**Key Points**:
- Central table for tracking manual decisions and quality assessments
- Kept in memory during GUI analysis and manipulated in real time; flushed to the database on navigation or Save and Exit
- `ms1_notes`, `ms2_notes`, `other_notes` store standardized quality decisions from the GUI radio buttons
- `max_eic_rt` / `max_eic_intensity` identify the highest-quality EIC peak for display in summaries
- `analysis_type` and `analysis_name` allow the same compound to be analyzed independently in different workflows within the same RT alignment and analysis iteration

---

## Standards Annotation Project Database

The standards-annotation workflow (`metatlas2.sh standards-annotation`) uses its
own project DuckDB at
`$METATLAS_DATA_DIR/projects/reference_standards_annotation/<analyst>/<standards_run_name>/<standards_run_name>.duckdb`.
It is created and written only by `metatlas2/standards_annotation/standards_annotation_db.py`. The
analyst workflow is described in [standards_annotation_workflow.md](standards_annotation_workflow.md).

Review history is versioned: each review iteration is an **annotation run**
(`draft` while edited in the GUI, `frozen` once summarized), each RT alignment
of a frozen annotation run is an **alignment run**, and each deposit preview is
a **deposit**. Runs and deposits are numbered 1, 2, 3, ... and map to output
directories `outputs/annotation_run_<N>/rt_alignment_<M>/deposit_<K>/`.

```
extraction layer (written once)          review history                         deposit
──────────────────────────────          ──────────────────────────────         ─────────────────────────────
sa_compounds ─┐                          sa_annotation_runs (N)
sa_extraction_targets ─┤                   ├─ sa_run_selections ──► sa_peaks
sa_ms1, sa_ms2, sa_tic, sa_injbl_ms1       ├─ sa_run_no_good_peaks
sa_ms1_sources                             ├─ sa_run_compound_notes
sa_peaks ── sa_peak_ms2                    ├─ sa_run_state
                                           └─ sa_alignment_runs (M)
                                                ├─ sa_alignment_models
                                                ├─ sa_alignment_qc_points
                                                ├─ sa_aligned_peaks ──► sa_peaks
                                                ├─ sa_atlas_entries / sa_msms_entries
                                                └─ sa_deposits (K)
                                                     ├─ sa_deposit_atlas_rows (generated | final)
                                                     ├─ sa_deposit_msms_rows  (generated | final)
                                                     └─ sa_deposit_changes
```

| Table | Key | Contents |
|---|---|---|
| `sa_compounds` | `compound_uid` | One row per compound name + adduct with structure and theoretical m/z. |
| `sa_extraction_targets` | `mz_rt_uid` | One row per compound/adduct/file extraction target with RT window and input annotation notes. |
| `sa_ms1`, `sa_ms2`, `sa_tic`, `sa_injbl_ms1` | per table | Extracted EICs, MS2 scans, TIC, and injection-blank EICs. |
| `sa_ms1_sources` | compound, chromatography, polarity | The MS1 file chosen for each group. |
| `sa_peaks` | `peak_uid` | Detected and manual peaks. `created_in_annotation_run` is set for manual peaks; a frozen run only shows peaks created in or before it. |
| `sa_peak_ms2` | `peak_uid`, `filename` | Closest MS2 scan per peak and CE file. |
| `sa_annotation_runs` | `annotation_run_id` | Status (`draft`/`frozen`), parent run, created/frozen dates and analyst, summary directory, Drive link, GUI checkpoint. At most one draft. |
| `sa_run_selections` | run, `peak_uid` | Selected peaks and analyst notes in a run. |
| `sa_run_no_good_peaks` | run, compound, chromatography, polarity | Explicit "No good peaks" decisions. |
| `sa_run_compound_notes` | run, compound, chromatography | Analyst notes per compound and chromatography. |
| `sa_run_state` | run, `compound_uid`, chromatography | Pending/complete status used to resume the GUI. |
| `sa_alignment_runs` | `alignment_run_id` | Annotation run aligned, dates, directory, Drive link, `skip_alignment`, atlas tags, and full config snapshot. |
| `sa_alignment_models` | run, chromatography | Model type, degree, coefficients, intercept, offset, R², RMSE, equation, number of QC points, QC atlas UID and selector, settings, QC filenames. |
| `sa_alignment_qc_points` | run, chromatography, point | Per-QC-compound measured RT, atlas RT, prediction, and residual. |
| `sa_aligned_peaks` | run, `peak_uid` | Output order, review label and number, measured and aligned RT peak/min/max, RT difference. |
| `sa_atlas_entries` | alignment run, `entry_id` | Current atlas rows to deposit for an alignment run (centralized-atlas columns). Generated on the first preview; deposits write analyst edits back (`origin = manual` for added rows, `is_excluded` for deleted rows, `last_modified_by_deposit`). |
| `sa_msms_entries` | alignment run, `entry_id` | Current MS/MS reference rows (importer `.tab` columns plus `sa_peak_uid`, `sa_ce_label`, measured collision energy), maintained like `sa_atlas_entries`. |
| `sa_deposits` | `deposit_id` | Alignment run, status (`previewed`, `superseded`, `in_progress`, `deposited`), directory, SHA-256 of the generated and deposited files, stage status (`compounds`, `atlas`, `msms`: `pending`/`done`/`skipped`) with times, `force_add`. |
| `sa_deposit_atlas_rows` | deposit, version, row | Exact rows of the preview as generated and as deposited, with `change_status` (`unchanged`/`edited`/`added`) and `sheet_outcome` (`appended`/`skipped_duplicate`/`skipped_by_flag`). |
| `sa_deposit_msms_rows` | deposit, version, row | As above for MS/MS references, with `import_outcome` (`inserted`/`skipped_existing`/`filtered`/`skipped_by_flag`) and the main-database `ref_uid`. |
| `sa_deposit_changes` | — | One row per edited value, added row, or removed row between the generated preview and the deposited file. |

The review tables do not declare foreign keys because DuckDB cannot delete a
referenced row in the same transaction as the rows that reference it, which
discarding a draft annotation run requires. The helper module enforces the
relationships.

---

## Workflow Objects and Database Mapping

Workflow objects are Python dataclasses defined in `workflow_objects.py` that provide an object-oriented interface to database tables.

| Workflow Class | Database Table(s) | Mapping Type |
|---------------|-------------------|--------------|
| **Compound** | `compounds` | 1:1 — Direct mapping to compound metadata |
| **CompoundMZRT** | `compound_mzrt` | 1:1 — Maps to RT/MZ reference data |
| **Atlas** | `atlases`, `atlas_compound_associations`, `compound_mzrt` | Composite — Spans multiple tables to represent complete atlas |
| **LCMSRun** | `lcmsruns` | 1:1 — Direct mapping to LCMS file metadata |
| **RTAlign** | `rt_alignment`, `workflow_runs` | Orchestrator — Writes model to `rt_alignment`, registers stage in `workflow_runs` |
| **AutoIdentification** | `ms1_data`, `ms2_data`, `manual_curation`, `workflow_runs` | Orchestrator — Coordinates extraction and storage across multiple tables |
| **ExperimentalData** | `ms1_data`, `ms2_data` | Container — Aggregates data from multiple experimental tables as DataFrames |
| **AnalysisGUI** | `manual_curation`, `workflow_runs` | Interactive — Reads curation data, writes analyst decisions back to DB |
| **AnalysisSummary** | `manual_curation`, `atlases`, `workflow_runs` | Summary — Reads all tables, writes curated atlas and summary outputs |
| **add_compounds_from_files** | `compounds` | Workflow routine — Validates, enriches, and inserts compounds from CSV/TSV |
| **NewAtlasesConfig** | `atlases`, `compound_mzrt`, `atlas_compound_associations` | Config — Orchestrates atlas creation |
| **NewMsmsRefsConfig** | `msms_refs` | Config — Orchestrates bulk insert of MS/MS reference spectra |

---

## Database Relationships and Visual Schema

### Main Database Schema Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN DATABASE TABLES                         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│     compounds        │
├──────────────────────┤
│ PK compound_uid      │◄─────┐
│    label             │      │
│    inchi_key         │      │
│    inchi             │      │
│    smiles            │      │
│    formula           │      │
│    classes           │      │
│    pathways          │      │
│    tags              │      │
│    ...               │      │
└──────────────────────┘      │
         △                    │
         │                    │
         │ 1:N                │
         │                    │
┌────────┴─────────────┐      │
│   compound_mzrt      │      │
├──────────────────────┤      │
│ PK mz_rt_uid         │◄──┐  │
│ FK compound_uid      │   │  │
│    label             │   │  │
│    inchi_key         │   │  │
│    adduct            │   │  │
│    rt_peak/min/max   │   │  │
│    mz / mz_tolerance │   │  │
│    chromatography    │   │  │
│    polarity          │   │  │
│    prev_mz_rt_uid    │   │  │
│    ...               │   │  │
└──────────────────────┘   │  │
         △                 │  │
         │                 │  │
         │ N:M via         │  │
         │ associations    │  │
         │                 │  │
┌────────┴─────────────────┴──┴────────────┐
│   atlas_compound_associations            │
├──────────────────────────────────────────┤
│ PK association_uid                       │
│ FK atlas_uid                             │
│    compound_uid                          │
│    mz_rt_uid                             │
│    association_order                     │
└──────────────────────────────────────────┘
         △
         │
         │ N:M
         │
┌────────┴─────────────┐
│      atlases         │
├──────────────────────┤
│ PK atlas_uid         │
│    atlas_name        │
│    atlas_description │
│    chromatography    │
│    polarity          │
│    analysis_type     │
│    analysis_name     │
│    atlas_stage       │
│    source_atlas_uid  │
│    ...               │
└──────────────────────┘

┌──────────────────────────────────────────┐
│   msms_refs           │
├──────────────────────────────────────────┤
│ PK ref_uid                               │
│    database / ref_id / name              │
│    inchi_key  (indexed)                  │
│    polarity   (indexed)                  │
│    precursor_mz                          │
│    mz[]  /  intensities[]               │
│    ...                                   │
└──────────────────────────────────────────┘
```

### Project Database Schema Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROJECT DATABASE TABLES                        │
└─────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────┐    ┌──────────────────────┐
        │     lcmsruns         │    │   project_config     │
        ├──────────────────────┤    ├──────────────────────┤
        │ PK file_path         │    │ PK config_uid        │
        │    filename          │    │    rt_alignment_num  │
        │    file_type         │    │    analysis_num      │
        │    chromatography    │    │    config_yaml       │
        │    polarity          │    │    paths_json        │
        │    ms_level          │    └──────────────────────┘
        └──────────────────────┘
                 │
                 │ used by
                 ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │   rt_alignment       │    │   workflow_runs      │
        ├──────────────────────┤    ├──────────────────────┤
        │ PK rt_alignment_uid  │    │ PK run_uid           │
        │    rt_alignment_num  │    │    rt_alignment_num  │
        │    qc_atlas_uid      │    │    analysis_num      │
        │    model_type        │    │    stage             │
        │    coefficients      │    │    atlas_uid         │
        │    r_squared         │    │    chromatography    │
        │    ...               │    │    polarity          │
        └──────────────────────┘    │    analysis_type     │
                 │                  │    analysis_name     │
                 │ generates        └──────────────────────┘
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  Atlas Tables (copied/derived from Main DB)                        │
├────────────────────────────────────────────────────────────────────┤
│  atlases  ←→  atlas_compound_associations  ←→  compound_mzrt      │
│  (RT_ALIGNED → AUTO_IDED → MANUALLY_CURATED via source_atlas_uid) │
└────────────────────────────────────────────────────────────────────┘
                 │
                 │ defines extraction targets
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  Experimental Data Tables                                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐              ┌──────────────┐                    │
│  │  ms1_data    │              │  ms2_data    │                    │
│  ├──────────────┤              ├──────────────┤                    │
│  │ mz_rt_uid   │              │ mz_rt_uid   │                    │
│  │ filename     │              │ filename     │                    │
│  │ spec_rts[]   │              │ scan_rt      │                    │
│  │ spec_ints[]  │              │ frag_mzs[]   │                    │
│  │ spec_mzs[]   │              │ frag_ints[]  │                    │
│  │ in_feature[] │              │ hits (JSON)  │                    │
│  │ rt_align_num │              │ rt_align_num │                    │
│  │ analysis_num │              │ analysis_num │                    │
│  └──────────────┘              └──────────────┘                    │
│         │                            │                             │
│         │                            │ informs                     │
│         ▼                            ▼                             │
│                   ┌────────────────────┐                           │
│                   │ manual_curation    │                           │
│                   ├────────────────────┤                           │
│                   │ mz_rt_uid          │                           │
│                   │ rt_align_num       │                           │
│                   │ analysis_num       │                           │
│                   │ ms1_notes          │                           │
│                   │ ms2_notes          │                           │
│                   │ rt_peak/min/max    │                           │
│                   │ passed_autoid      │                           │
│                   │ passed_curation    │                           │
│                   │ max_eic_rt[]       │                           │
│                   │ ...                │                           │
│                   └────────────────────┘                           │
└────────────────────────────────────────────────────────────────────┘
```

### Entity Relationships

**Main Database:**
- `compounds` → `compound_mzrt`: One-to-many (one compound can have multiple adducts/methods)
- `atlases` ↔ `compounds`: Many-to-many via `atlas_compound_associations`
- `atlases` ↔ `compound_mzrt`: Many-to-many via `atlas_compound_associations`
- `msms_refs`: Independent table, matched to compounds by `inchi_key` at query time

**Project Database:**
- `lcmsruns` → `rt_alignment`: QC files used to build RT correction model
- `rt_alignment` → `atlases`: RT alignment generates RT_ALIGNED atlas versions
- `workflow_runs`: Tracks which atlas UID corresponds to each stage (RT_ALIGNED, AUTO_IDED, MANUALLY_CURATED)
- `atlases` → `compound_mzrt`: Atlas defines which compounds to extract
- `compound_mzrt` → `ms1_data`, `ms2_data`: Extraction targets for spectral data (matched by `mz_rt_uid`)
- `ms2_data.hits` → MS2 library matching results (embedded JSON, sourced from `msms_refs`)
- `ms1_data`, `ms2_data` → `manual_curation`: Synthesized curation decisions

**Cross-Database:**
- Project `atlases.source_atlas_uid` → Main `atlases.atlas_uid`: Derivation lineage
- Project `compound_mzrt.prev_mz_rt_uid` → Main/Project `compound_mzrt.mz_rt_uid`: RT-aligned/curated entry lineage
- Project `manual_curation` → Project `compound_mzrt`: Curated RT bounds written back to `compound_mzrt` for the MANUALLY_CURATED atlas

---
