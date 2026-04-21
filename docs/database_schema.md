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
- [Project Database Schema](#project-database-schema)
  - [lcmsruns Table](#lcmsruns-table)
  - [atlases Table (Project)](#atlases-table-project)
  - [compounds Table (Project)](#compounds-table-project)
  - [compound_mzrt Table (Project)](#compound_mzrt-table-project)
  - [atlas_compound_associations Table (Project)](#atlas_compound_associations-table-project)
  - [rt_alignment Table](#rt_alignment-table)
  - [ms1_data Table](#ms1_data-table)
  - [ms2_data Table](#ms2_data-table)
  - [ms2_hits Table](#ms2_hits-table)
  - [manual_curation Table](#manual_curation-table)
- [Workflow Objects and Database Mapping](#workflow-objects-and-database-mapping)
- [Database Relationships and Visual Schema](#database-relationships-and-visual-schema)
  - [Main Database Schema Diagram](#main-database-schema-diagram)
  - [Project Database Schema Diagram](#project-database-schema-diagram)
  - [Entity Relationships](#entity-relationships)

---

## Database Architecture Overview

Metatlas2 uses lightweight **DuckDB** for data storage. The system maintains two types of databases:

### 1. **Main Database** (Knowledge Repository)
- **Purpose**: Central repository of reference compounds and atlases
- **Location**: Central location for all analysts, typically at `/global/cfs/cdirs/metatlas/databases/main_db/main.duckdb`
- **Scope**: Shared across all projects, analysts, and owners (i.e., JGI and EGSB)
- **Content**: 
  - Compound metadata (chemical structures, identifiers, pubchem data)
  - Reference RT/MZ data (compound_mzrt entries, added after curation)
  - Reference atlases (curated sets of compounds)
  - Compound-atlas associations (one compound can belong to many atlases)

### 2. **Project Database** (Experimental Results)
- **Purpose**: Stores project-specific experimental data and derived results in a standardized format
- **Location**: Within each project directory, typically at `/global/cfs/cdirs/metatlas/projects/targeted_outputs/<owner>/<project_name>/<project_name>.duckdb`
- **Scope**: Single database for each project, analyst, and owner (i.e., JGI and EGSB)
- **Content**:
  - LCMS run metadata (information about each .parquet file)
  - Project-specific atlases (RT-aligned, Auto-IDed, manually curated, etc.)
  - RT alignment models and parameters
  - Extracted MS1/MS2 spectral data
  - MS2 spectral matching results
  - Manual curation decisions and notes

This separation allows for:
- **Reusability**: Reference data shared across projects
- **Traceability**: Complete experimental history per project
- **Scalability**: Project databases are atomic and analysis time does not scale with project number/size
- **Provenance**: Track who created each entry and when

---

## Main Database Schema

The main database contains four core tables that define reference compounds and atlases.

### compounds Table

Stores immutable chemical compound metadata. Each compound represents a unique molecular entity.

| Column | Type | Description |
|--------|------|-------------|
| `compound_uid` | TEXT (PK) | Unique identifier (e.g., `cmp-a1b2c3...`) |
| `compound_name` | TEXT | Primary compound name |
| `inchi_key` | TEXT | InChI Key (molecular structure hash) |
| `inchi` | TEXT | Full InChI string |
| `smiles` | TEXT | SMILES structure notation |
| `formula` | TEXT | Molecular formula |
| `compound_classes` | TEXT | Classification terms (pipe-separated) |
| `compound_pathways` | TEXT | Biochemical pathways (pipe-separated) |
| `compound_tags` | TEXT | Custom tags (pipe-separated) |
| `mono_isotopic_molecular_weight` | REAL | Monoisotopic molecular weight |
| `iupac_name` | TEXT | IUPAC systematic name |
| `pubchem_cid` | TEXT | PubChem compound ID |
| `cas_number` | TEXT | CAS registry number |
| `synonyms` | TEXT | Alternative names (pipe-separated) |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- `compound_uid` is the primary key used to reference compounds throughout the system. It is linked to a specific InChIKey and cannot be duplicated (see below).
- `inchi_key` provides a standard, collision-resistant molecular identifier
- Chemical properties (InChI, SMILES, formula) are immutable once created
- Metadata fields support pipe-separated lists for flexibility

### compound_mzrt Table

Stores retention time (RT) and mass-to-charge ratio (m/z) reference data for compounds. A single compound can have multiple mzrt entries for different adducts, chromatography methods, confidence levels, or reference standard runs. For example, if an analyst desposits a new Atlas (a store of compound m/z and RT information) into the database via add_atlases_to_db.py, a new compound_mzrt table entry will be created if the input information doesn't match an existing entry exactly.

| Column | Type | Description |
|--------|------|-------------|
| `mz_rt_uid` | TEXT (PK) | Unique identifier (e.g., `mzrt-x1y2z3...`) |
| `compound_uid` | TEXT | Links to compounds table |
| `compound_name` | TEXT | Denormalized for convenience |
| `inchi_key` | TEXT | Denormalized for convenience |
| `adduct` | TEXT | Adduct form (e.g., `[M+H]+`, `[M-H]-`) |
| `rt_peak` | REAL | Peak retention time (minutes) |
| `rt_min` | REAL | RT window start (minutes) |
| `rt_max` | REAL | RT window end (minutes) |
| `mz` | REAL | Mass-to-charge ratio |
| `mz_tolerance` | REAL | m/z tolerance (ppm) |
| `chromatography` | TEXT | Chromatography type (e.g., `HILIC`, `C18`) |
| `polarity` | TEXT | Ionization polarity (`positive` or `negative`) |
| `confidence` | TEXT | Identification confidence level |
| `source` | TEXT | Data origin (e.g., file path, reference) |
| `ms1_notes` | TEXT | MS1-related notes |
| `ms2_notes` | TEXT | MS2-related notes |
| `other_notes` | TEXT | General notes |
| `analyst_notes` | TEXT | Analyst comments |
| `identification_notes` | TEXT | Identification-specific notes |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Each entry represents a specific adduct of a compound under specific conditions
- RT values define the expected elution window for targeted extraction (minutes)
- `mz_tolerance` defines the m/z extraction window (typically 5-20 ppm), but this can be overridden during analysis time
- `chromatography` and `polarity` define the analytical method
- `identification_notes` will typically derive from notes made during the actual reference standard annotation of the compound, to indicate what kind of peak should be observed, while `analyst_notes` will typically be notes made during the manual curation of data during a targeted analysis

### atlases Table

Defines collections of compounds for targeted analysis. Atlases organize compound sets by analytical method and purpose.

| Column | Type | Description |
|--------|------|-------------|
| `atlas_uid` | TEXT (PK) | Unique identifier (e.g., `atl-ref-method-a1b2...`) |
| `atlas_name` | TEXT | Human-readable atlas name |
| `atlas_description` | TEXT | Detailed description of atlas purpose |
| `chromatography` | TEXT | Chromatography method (e.g., `HILIC`, `C18`) |
| `polarity` | TEXT | Ionization polarity (`positive` or `negative`) |
| `analysis_type` | TEXT | Analysis category (e.g., `targeted`, `discovery`) |
| `atlas_type` | TEXT | Atlas category (see below) |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |
| `source` | TEXT | File path or origin reference |

**Atlas Types**:
- `REFERENCE`: Original reference atlas from main database
- `RT-ALIGNED`: RT-adjusted atlas for a specific project - usually starts as clone of `REFERENCE` and creates `RT-ALIGNED`
- `AUTO-ID`: Auto-identified atlas from experimental data - usually starts as clone of `RT-ALIGNED` and creates `AUTO-ID`
- `CURATED`: Manually curated/refined atlas - usually starts as clone of `AUTO-ID` and creates `CURATED`

**Key Points**:
- Each atlas targets a specific analytical method (analysis type [ISTD, QC, EMA], chromatography, polarity, project-specific compounds, etc.)
- Atlases are collections; actual compounds in the Compound table are linked via `atlas_compound_associations`

### atlas_compound_associations Table

Junction table linking atlases to their constituent compounds and mzrt entries.

| Column | Type | Description |
|--------|------|-------------|
| `association_uid` | TEXT (PK) | Unique association identifier |
| `atlas_uid` | TEXT (FK) | References atlases table |
| `compound_uid` | TEXT (FK) | References compounds table |
| `mz_rt_uid` | TEXT (FK) | References compound_mzrt table |
| `association_order` | INTEGER | Display/processing order |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Enables many-to-many relationship between atlases and compounds
- Each association links to a specific mzrt entry (adduct + method combination)
- `association_order` preserves compound ordering within the atlas
- Foreign keys enforce referential integrity

---

## Project Database Schema

Project databases extend the main database schema with experimental data and project-specific tables.

### lcmsruns Table

Catalogs all LCMS data files (.parquet, .raw, .mzML) available for a project.

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | TEXT (PK) | Absolute path to parquet file |
| `filename` | TEXT | Base filename |
| `file_format` | TEXT | Original format (e.g., `raw`, `mzML`) |
| `file_type` | TEXT | File category (e.g., `sample`, `QC`, `blank`) |
| `chromatography` | TEXT | Chromatography method |
| `ms_level` | INTEGER | MS level (1 or 2) |
| `polarity` | TEXT | Ionization polarity |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Each row represents one parquet file (converted from raw/mzML)
- `file_path` serves as primary key and reference for data extraction
- Metadata enables filtering LCMS runs by method and file type
- Both MS1 and MS2 files are cataloged

### atlases Table (Project)

Project atlases extend the main database atlases with project-specific tracking.

| Column | Type | Description |
|--------|------|-------------|
| `atlas_uid` | TEXT (PK) | Unique identifier |
| `atlas_name` | TEXT | Human-readable name |
| `atlas_description` | TEXT | Description |
| `chromatography` | TEXT | Chromatography method |
| `polarity` | TEXT | Ionization polarity |
| `analysis_type` | TEXT | Analysis category |
| `atlas_type` | TEXT | Atlas category |
| `source_atlas_uid` | TEXT | UID of parent atlas (if derived) |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |
| `source` | TEXT | File path or origin |

**Additional Fields (vs Main Database)**:
- `source_atlas_uid`: Links derived atlases to their reference atlas
- `rt_alignment_number`: Associates atlas with specific RT alignment
- `analysis_number`: Associates atlas with specific analysis iteration

**Key Points**:
- Project atlases are typically derived from main database reference atlases
- RT-Aligned atlases have adjusted RT values for project-specific chromatography
- Auto-IDed atlases have compounds flagged as present/absent in the empirical LCMSRun data
- Curated atlases have compound notes and remove/keep designations from the GUI
- Multiple atlas versions can exist per analysis iteration

### compounds Table (Project)

Identical schema to main database compounds table. Compound metadata is copied to project databases for self-contained analysis and faster queries.

### compound_mzrt Table (Project)

Identical schema to main database compound_mzrt table. MZRT data is copied and may be modified (e.g., RT-aligned values).

### atlas_compound_associations Table (Project)

Similar to main database, but foreign keys reference only the atlas (not compound/mzrt tables directly).

| Column | Type | Description |
|--------|------|-------------|
| `association_uid` | TEXT (PK) | Unique association identifier |
| `atlas_uid` | TEXT (FK) | References project atlases table |
| `compound_uid` | TEXT | Compound UID (no FK constraint) |
| `mz_rt_uid` | TEXT | MZRT UID (no FK constraint) |
| `association_order` | INTEGER | Display/processing order |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

### rt_alignment Table

Stores retention time alignment models and metadata.

| Column | Type | Description |
|--------|------|-------------|
| `rt_alignment_uid` | TEXT (PK) | Unique identifier |
| `project_name` | TEXT | Project name |
| `rt_alignment_number` | INTEGER | Alignment iteration number |
| `qc_atlas_uid` | TEXT | Atlas used for RT alignment |
| `model_type` | TEXT | Model type (e.g., `polynomial`) |
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
- Model parameters enable RT correction for subsequent analyses
- Quality metrics (R², RMSE) assess alignment quality
- `rt_alignment_number` links to aligned atlases

### ms1_data Table

Stores extracted MS1 spectral data for each compound in each LCMS run.

| Column | Type | Description |
|--------|------|-------------|
| `ms1_data_uid` | TEXT (PK) | Unique identifier |
| `compound_uid` | TEXT | Compound identifier |
| `inchi_key` | TEXT | InChI key |
| `adduct` | TEXT | Adduct form |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | TEXT | Analysis workflow type (e.g., ISTD, EMA) |
| `file_path` | TEXT | LCMS run file path |
| `mz` | TEXT | JSON-encoded m/z array |
| `raw_spectrum` | TEXT | JSON-encoded intensity array |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- One entry per compound per LCMS run
- `raw_spectrum` contains JSON array of intensities across RT window
- `mz` contains corresponding m/z values
- Links to specific RT alignment, analysis iteration, and analysis type

### ms2_data Table

Stores extracted MS2 fragmentation spectra.

| Column | Type | Description |
|--------|------|-------------|
| `ms2_data_uid` | TEXT (PK) | Unique identifier |
| `compound_uid` | TEXT | Compound identifier |
| `inchi_key` | TEXT | InChI key |
| `adduct` | TEXT | Adduct form |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | TEXT | Analysis workflow type (e.g., ISTD, EMA) |
| `file_path` | TEXT | LCMS run file path |
| `rt` | REAL | Retention time of scan |
| `raw_spectrum` | TEXT | JSON-encoded fragment spectrum |
| `precursor_MZ` | REAL | Precursor m/z |
| `precursor_intensity` | REAL | Precursor intensity |
| `collision_energy` | REAL | Collision energy (eV) |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Multiple MS2 scans can exist per compound per run
- `raw_spectrum` is JSON array of (m/z, intensity) pairs
- Precursor information links MS2 to parent ion
- Links to specific RT alignment, analysis iteration, and analysis type

### ms2_hits Table

Stores spectral matching results from comparing experimental MS2 to reference libraries.

| Column | Type | Description |
|--------|------|-------------|
| `ms2_hit_uid` | TEXT (PK) | Unique identifier |
| `compound_uid` | TEXT | Compound identifier |
| `inchi_key` | TEXT | InChI key |
| `adduct` | TEXT | Adduct form |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `analysis_type` | TEXT | Analysis workflow type (e.g., ISTD, EMA) |
| `file_path` | TEXT | LCMS run file path |
| `database` | TEXT | Reference database name |
| `ref_id` | TEXT | Reference spectrum ID |
| `ref_name` | TEXT | Reference compound name |
| `score` | REAL | Matching score (0-1) |
| `num_matches` | INTEGER | Number of matched fragments |
| `mz_theoretical` | REAL | Theoretical precursor m/z |
| `mz_measured` | REAL | Measured precursor m/z |
| `ppm_error` | REAL | m/z error in ppm |
| `rt` | REAL | Retention time |
| `qry_intensity_peak` | REAL | Query peak intensity |
| `ref_frags` | INTEGER | Number of reference fragments |
| `data_frags` | INTEGER | Number of query fragments |
| `matched_fragments` | TEXT | JSON list of matched fragment m/z |
| `aligned_fragment_colors` | TEXT | JSON color coding for visualization |
| `qry_spectrum` | TEXT | JSON-encoded query spectrum |
| `ref_spectrum` | TEXT | JSON-encoded reference spectrum |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Multiple hits can exist per compound (different reference matches)
- Spectral matching scores support identification confidence
- Links to specific RT alignment, analysis iteration, and analysis type
- Both query and reference spectra stored for visualization
- Links to specific RT alignment and analysis iterations

### manual_curation Table

Stores manual curation decisions and compound identification results.

| Column | Type | Description |
|--------|------|-------------|
| `curation_uid` | TEXT (PK) | Unique identifier |
| `compound_uid` | TEXT | Compound identifier |
| `inchi_key` | TEXT | InChI key |
| `adduct` | TEXT | Adduct form |
| `rt_alignment_number` | INTEGER | RT alignment iteration |
| `analysis_number` | INTEGER | Analysis iteration |
| `compound_name` | TEXT | Compound name |
| `auto_ided` | BOOLEAN | Auto-identification flag |
| `polarity` | TEXT | Ionization polarity |
| `chromatography` | TEXT | Chromatography method |
| `analysis_type` | TEXT | Analysis workflow type (e.g., ISTD, EMA) |
| `mz_tolerance` | REAL | m/z tolerance (ppm) |
| `atlas_mz` | REAL | Atlas reference m/z |
| `atlas_rt_peak` | REAL | Atlas reference RT peak |
| `atlas_rt_min` | REAL | Atlas reference RT min |
| `atlas_rt_max` | REAL | Atlas reference RT max |
| `original_rt_peak` | REAL | Pre-alignment RT peak |
| `original_rt_min` | REAL | Pre-alignment RT min |
| `original_rt_max` | REAL | Pre-alignment RT max |
| `rt_peak` | REAL | Curated RT peak |
| `rt_min` | REAL | Curated RT min |
| `rt_max` | REAL | Curated RT max |
| `ms1_notes` | TEXT | MS1 quality/decision |
| `ms2_notes` | TEXT | MS2 quality/decision |
| `other_notes` | TEXT | Additional notes |
| `identification_notes` | TEXT | Identification rationale |
| `analyst_notes` | TEXT | Analyst comments |
| `best_ms1_file` | TEXT | Best MS1 file path |
| `best_ms1_rt` | REAL | RT of best MS1 peak |
| `best_ms1_mz` | REAL | m/z of best MS1 peak |
| `best_ms1_intensity` | REAL | Intensity of best MS1 peak |
| `best_ms1_ppm_error` | REAL | m/z error of best MS1 peak |
| `best_ms1_rt_error` | REAL | RT error of best MS1 peak |
| `isomers` | TEXT | Potential isomer information |
| `suggested_rt_min` | REAL | Algorithm-suggested RT min |
| `suggested_rt_max` | REAL | Algorithm-suggested RT max |
| `suggested_rt_peak` | REAL | Algorithm-suggested RT peak |
| `rt_suggestion_confidence` | REAL | Confidence in RT suggestion |
| `created_by` | TEXT | Username of creator |
| `created_date` | TEXT | ISO timestamp of creation |

**Key Points**:
- Central table for tracking manual decisions and quality assessments, kept in memory during GUI analysis and manipulated in real time (and used to flush changes to the project database tables during curation).
- `ms1_notes`, `ms2_notes` store standardized quality decisions
- `best_ms1_*` fields identify the highest-quality samples/data for each compound to display in summaries
- Acts as the bridge between automated and manual identification workflows
- `analysis_type` allows the same compound to be analyzed independently in different workflows (e.g., ISTD vs EMA) within the same RT alignment and analysis iteration

---

## Workflow Objects and Database Mapping

Workflow objects are Python dataclasses defined in `workflow_objects.py` that provide an object-oriented interface to database tables.

| Workflow Class | Database Table(s) | Mapping Type |
|---------------|-------------------|--------------|
| **Compound** | `compounds` | 1:1 - Direct mapping to compound metadata |
| **CompoundMZRT** | `compound_mzrt` | 1:1 - Maps to RT/MZ reference data |
| **Atlas** | `atlases`, `atlas_compound_associations`, `compound_mzrt` | Composite - Spans multiple tables to represent complete atlas |
| **LCMSRun** | `lcmsruns` | 1:1 - Direct mapping to LCMS file metadata |
| **RTAlign** | `rt_alignment` | 1:1 - Core fields map to table; runtime attributes not stored |
| **AutoIdentification** | `ms1_data`, `ms2_data`, `ms2_hits` | Orchestrator - Coordinates extraction and storage across multiple tables |
| **ManualCuration** | `manual_curation` | Collection - DataFrame wrapper for multiple curation rows |
| **ExperimentalData** | `ms1_data`, `ms2_data`, `ms2_hits` | Container - Aggregates data from multiple experimental tables |

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
│    compound_name     │      │
│    inchi_key         │      │
│    inchi             │      │
│    smiles            │      │
│    formula           │      │
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
│    compound_name     │   │  │
│    inchi_key         │   │  │
│    adduct            │   │  │
│    rt_peak/min/max   │   │  │
│    mz                │   │  │
│    chromatography    │   │  │
│    polarity          │   │  │
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
│ FK compound_uid                          │
│ FK mz_rt_uid                             │
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
│    atlas_type        │
│    ...               │
└──────────────────────┘
```

### Project Database Schema Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROJECT DATABASE TABLES                        │
└─────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────┐
        │     lcmsruns         │
        ├──────────────────────┤
        │ PK file_path         │
        │    filename          │
        │    file_type         │
        │    chromatography    │
        │    polarity          │
        │    ms_level          │
        └──────────────────────┘
                 │
                 │ used by
                 ▼
        ┌──────────────────────┐
        │   rt_alignment       │
        ├──────────────────────┤
        │ PK rt_alignment_uid  │
        │    rt_alignment_num  │
        │    qc_atlas_uid      │
        │    model_type        │
        │    coefficients      │
        │    r_squared         │
        │    ...               │
        └──────────────────────┘
                 │
                 │ generates
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  Atlas Tables (copied/derived from Main DB)                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌─────────────────────┐    ┌──────────────┐   │
│  │  compounds   │    │ atlas_compound_     │    │   atlases    │   │
│  │              │◄───┤   associations      │───►│              │   │
│  │ compound_uid │    │                     │    │  atlas_uid   │   │
│  └──────┬───────┘    └──────────┬──────────┘    └──────────────┘   │
│         │                       │                                  │
│         ▼                       ▼                                  │
│  ┌──────────────┐    ┌─────────────────────┐                       │
│  │compound_mzrt │    │   mz_rt_uid         │                       │
│  │              │◄───┤                     │                       │
│  │  mz_rt_uid   │    │                     │                       │
│  └──────────────┘    └─────────────────────┘                       │
└────────────────────────────────────────────────────────────────────┘
                 │
                 │ defines extraction targets
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  Experimental Data Tables                                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  ms1_data    │    │  ms2_data    │    │  ms2_hits    │          │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤          │
│  │ compound_uid │    │ compound_uid │    │ compound_uid │          │
│  │ file_path    │    │ file_path    │    │ file_path    │          │
│  │ rt_align_num │    │ rt_align_num │    │ rt_align_num │          │
│  │ analysis_num │    │ analysis_num │    │ analysis_num │          │
│  │ raw_spectrum │    │ raw_spectrum │    │ score        │          │
│  │ ...          │    │ rt           │    │ ref_spectrum │          │
│  └──────────────┘    │ ...          │    │ ...          │          │
│                      └──────────────┘    └──────────────┘          │
│         │                    │                 │                   │
│         │                    │                 │ informs           │
│         ▼                    ▼                 ▼                   │
│                   ┌────────────────────┐                           │
│                   │ manual_curation    │                           │
│                   ├────────────────────┤                           │
│                   │ compound_uid       │                           │
│                   │ rt_align_num       │                           │
│                   │ analysis_num       │                           │
│                   │ ms1_notes          │                           │
│                   │ ms2_notes          │                           │
│                   │ rt_peak/min/max    │                           │
│                   │ best_ms1_file      │                           │
│                   │ ...                │                           │
│                   └────────────────────┘                           │
└────────────────────────────────────────────────────────────────────┘
```

### Entity Relationships

**Main Database:**
- `compounds` → `compound_mzrt`: One-to-many (one compound can have multiple adducts/methods)
- `atlases` ↔ `compounds`: Many-to-many via `atlas_compound_associations`
- `atlases` ↔ `compound_mzrt`: Many-to-many via `atlas_compound_associations`

**Project Database:**
- `lcmsruns` → `rt_alignment`: QC files used to build RT correction model
- `rt_alignment` → `atlases`: RT alignment generates aligned atlas versions
- `atlases` → `compound_mzrt`: Atlas defines which compounds to extract
- `compound_mzrt` → `ms1_data`, `ms2_data`: Extraction targets for spectral data
- `ms2_data` → `ms2_hits`: Spectral matching produces hits
- `ms1_data`, `ms2_data`, `ms2_hits` → `manual_curation`: Synthesized curation decisions

**Cross-Database:**
- Project `atlases.source_atlas_uid` → Main `atlases.atlas_uid`: Derivation lineage
- Project `compounds.compound_uid` copied from Main DB for self-contained analysis
- Project `compound_mzrt` may be modified (RT-aligned, Manually Curated) from Main/Project DB values

---