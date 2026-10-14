# Parquet Query System

After the analysis summary stage, metatlas2 writes the curated compound results to Parquet files for fast, flexible downstream querying. A `QueryNotebook__<project_name>.ipynb` is automatically generated in the `RTA#/TGA#/` directory alongside the per-analysis `CurationNotebook__*` files. This notebook is the primary interface for querying the Parquet data — no YAML config file is needed.

---

## Table of Contents

- [Overview](#overview)
- [Output files and directory layout](#output-files-and-directory-layout)
- [The QueryNotebook](#the-querynotebook)
  - [Notebook structure](#notebook-structure)
  - [Pre-filled variables](#pre-filled-variables)
  - [Running a query](#running-a-query)
- [Schema Reference](#schema-reference)
  - [compound_file grain](#compound_file-grain)
  - [compound_lfc grain](#compound_lfc-grain)
- [Query Builder Guide](#query-builder-guide)
  - [Filter suffix system](#filter-suffix-system)
  - [Special keys (compound_lfc only)](#special-keys-compound_lfc-only)
  - [Minimal query template](#minimal-query-template)
- [Pre-made example queries](#pre-made-example-queries)
  - [significant_compound_screen](#significant_compound_screen)
  - [high_confidence_core_hits](#high_confidence_core_hits)
- [End-to-end call chain](#end-to-end-call-chain)
  - [Step 1 — Notebook cell](#step-1--notebook-cell)
  - [Step 2 — wfs.run_parquet_query()](#step-2--wfsrun_parquet_query)
  - [Step 3 — ParquetQueryInterpreter.__init__()](#step-3--parquetqueryinterpreter__init__)
  - [Step 4 — ParquetQueryInterpreter.execute()](#step-4--parquetqueryinterpreterexecute)
  - [Step 5 — ParquetQueryResult returned to the notebook](#step-5--parquetqueryresult-returned-to-the-notebook)
- [API reference](#api-reference)

---

## Overview

The Parquet query system has three layers:

| Layer | File | Role |
|-------|------|------|
| **Notebook** | `QueryNotebook__<project>.ipynb` | Analyst-facing interface; pre-filled with project metadata; contains schema reference, query guide, and example queries |
| **Dispatcher** | `metatlas2/workflows.py` → `run_parquet_query()` | Thin public API; constructs the interpreter and calls `execute()` |
| **Engine** | `metatlas2/workflow_objects.py` → `ParquetQueryInterpreter` | Translates a `params` dict into a PyArrow predicate expression; applies predicate-pushdown scan against the Parquet files |

---

## Output files and directory layout

Parquet files are written during the analysis summary stage to:

```
$METATLAS_DATA_DIR/projects/parquet_outputs/
└── compound_file/                         ← one row per compound × raw file
│   └── project_name=<project>/
│       └── chromatography=<chrom>/
│           └── polarity=<pol>/
│               └── atlas_name=<name>/
│                   └── RTA1-TGA2-0-2024-01-01.parquet
└── compound_lfc/                          ← one row per compound × condition pair
    └── project_name=<project>/
        └── chromatography=<chrom>/
            └── polarity=<pol>/
                └── atlas_name=<name>/
                    └── RTA1-TGA2-0-2024-01-01.parquet
```

The filename encodes the run: `RTA{rt_alignment_number}-TGA{analysis_number}-{i}-{timestamp}.parquet`. When multiple alignment/analysis runs exist for the same project, the `rt_alignment_number` and `analysis_number` passed to `run_parquet_query()` restrict the scan to only the matching files.

The `QueryNotebook__<project>.ipynb` is written to:

```
$METATLAS_DATA_DIR/projects/targeted_outputs/<owner>/<user>/<project>/
└── RTA<N>/
    └── TGA<M>/
        ├── CurationNotebook__HILICZ-POS-EMA-RTA<N>-TGA<M>__<project>.ipynb
        ├── CurationNotebook__HILICZ-NEG-EMA-RTA<N>-TGA<M>__<project>.ipynb
        ├── ...
        └── QueryNotebook__<project>.ipynb   ← one per RTA/TGA, spans all analyses
```

---

## The QueryNotebook

### Notebook structure

The generated notebook has 8 cells:

| # | Type | Content |
|---|------|---------|
| 1 | Markdown | Header — project name, chromatography, RTA/TGA numbers |
| 2 | Code | Imports (`pandas`, `metatlas2.workflows`, logging) |
| 3 | Code | `RUN_PARAMS` dict pre-filled at generation time; exposes `PARQUET_ROOT`, `PROJECT_NAME`, `CHROMATOGRAPHY`, `RTA`, `TGA` |
| 4 | Markdown | **Schema Reference** — full column table for both grains |
| 5 | Markdown | **Query Builder Guide** — filter suffix system and minimal template |
| 6 | Code | `significant_compound_screen` — pre-made example query |
| 7 | Code | `high_confidence_core_hits` — pre-made example query |
| 8 | Code | **Custom query template** — blank editable cell with commented example |

### Pre-filled variables

Cell 3 defines these convenience aliases from `RUN_PARAMS`:

```python
PARQUET_ROOT    # path to the project's parquet_outputs directory
PROJECT_NAME    # full project name string
CHROMATOGRAPHY  # chromatography method (e.g. "HILICZ")
RTA             # rt_alignment_number (int)
TGA             # analysis_number (int)
```

`RTA` and `TGA` are passed to every query so that only the Parquet files produced by this specific run are scanned.

### Running a query

Each query cell calls `wfs.run_parquet_query()` and returns a `ParquetQueryResult` named tuple:

```python
result = wfs.run_parquet_query(
    project_root=PARQUET_ROOT,
    rt_alignment_number=RTA,
    analysis_number=TGA,
    params={
        'grain': 'compound_lfc',
        'project_name': PROJECT_NAME,
        'chromatography': CHROMATOGRAPHY,
        'log2_fold_change_abs_gt': 1.5,
    },
)
print(result.sql)    # human-readable SQL equivalent of the query
result.df.head(20)   # pandas DataFrame of matching rows
```

Each cell is independent — modify the `params` dict and re-run any cell without affecting the others.

---

## Schema Reference

### `compound_file` grain

One row per compound × raw file. Use `grain: "compound_file"` to query per-sample peak data.

| Column | Type | Description |
|--------|------|-------------|
| `compound_name` | string | Compound display name |
| `inchi_key` | string | InChI key (standard 27-char) |
| `adduct` | string | Detected adduct, e.g. `[M+H]+` |
| `formula` | string | Molecular formula |
| `smiles` | string | SMILES string |
| `filename` | string | Raw LC-MS filename |
| `file_group` | string | Sample group label |
| `msi_level` | string | MSI identification level (`1`–`4`) |
| `atlas_category` | string | Atlas category, e.g. `REF`, `EMA` |
| `atlas_name` | string | Atlas name |
| `chromatography` | string | Chromatography method, e.g. `HILICZ` |
| `polarity` | string | `POS` or `NEG` |
| `project_name` | string | Full project name |
| `control_filter` | string | Extraction-control filter result |
| `ms1_notes` | string | MS1 curation notes |
| `ms2_notes` | string | MS2 curation notes |
| `atlas_mz` | float | Theoretical m/z from atlas |
| `atlas_rt_peak` | float | Theoretical RT peak (min) |
| `best_ms1_intensity` | float | Best MS1 peak intensity |
| `best_ms1_ppm_error` | float | MS1 mass accuracy (ppm) |
| `best_ms1_rt` | float | RT of best MS1 hit (min) |
| `best_ms2_score` | float | Best MS2 spectral match score (0–1) |
| `best_ms2_mz_ppm_error` | float | MS2 precursor mass accuracy (ppm) |
| `peak_height` | float | Per-file peak height |
| `peak_area` | float | Per-file peak area |
| `total_score` | float | Composite quality score (mz + rt + ms2) |
| `mz_quality` | float | m/z quality sub-score |
| `rt_quality` | float | RT quality sub-score |
| `msms_quality` | float | MS2 quality sub-score |
| `mz_ppmerror` | float | MS1 ppm error (alias for `best_ms1_ppm_error`) |

### `compound_lfc` grain

One row per compound × condition pair. Use `grain: "compound_lfc"` to query log fold-change data.

Contains all `compound_file` columns **except** `filename`, `file_group`, `peak_height`, `peak_area`, `total_score`, `mz_quality`, `rt_quality`, `msms_quality`, plus:

| Column | Type | Description |
|--------|------|-------------|
| `log2_fold_change` | float | log₂(group1 / group2) |
| `condition_1` | string | Numerator group label |
| `condition_2` | string | Denominator group label |

---

## Query Builder Guide

Every query is a plain Python `dict` passed to `wfs.run_parquet_query()`. The `grain` key selects the table; all other keys are **filter parameters**.

### Filter suffix system

Append a suffix to a column name to create a filter:

| Suffix | Meaning | Example |
|--------|---------|---------|
| *(none)* | Exact equality | `'polarity': 'POS'` |
| `_in` | Value is in list | `'adduct_in': ['[M+H]+', '[M+Na]+']` |
| `_min` | Column ≥ value | `'best_ms1_intensity_min': 10_000` |
| `_max` | Column ≤ value | `'best_ms1_ppm_error_max': 5.0` |
| `_abs_gt` | \|column\| > value | `'log2_fold_change_abs_gt': 1.5` |
| `_abs_lt` | \|column\| < value | `'log2_fold_change_abs_lt': 0.1` |
| `_abs_min` | \|column\| ≥ value | `'log2_fold_change_abs_min': 1.0` |
| `_abs_max` | \|column\| ≤ value | `'log2_fold_change_abs_max': 3.0` |

Filter keys that reference a column not present in the selected grain are silently skipped with a printed warning.

### Special keys (`compound_lfc` only)

| Key | Value type | Meaning |
|-----|-----------|---------|
| `condition_pair` | `[group_a, group_b]` | Symmetric match — finds rows where `{condition_1, condition_2} == {group_a, group_b}` regardless of stored order |
| `lfc_directional_min` | `{'treatment': str, 'control': str, 'threshold': float}` | Directional threshold — correctly sign-flips when the pair is stored in reverse order |

### Minimal query template

```python
result = wfs.run_parquet_query(
    project_root=PARQUET_ROOT,
    rt_alignment_number=RTA,
    analysis_number=TGA,
    params={
        'grain': 'compound_lfc',       # or 'compound_file'
        'chromatography': CHROMATOGRAPHY,
        'project_name': PROJECT_NAME,
        # add filter keys here …
    },
)
print(result.sql)    # human-readable SQL equivalent
result.df.head(20)   # pandas DataFrame of matching rows
```

---

## Pre-made example queries

### `significant_compound_screen`

**Biological question:** Which compounds show a large absolute log2 fold-change between any two sample groups in this project?

This is the primary "what changed?" screen — a wide net cast across all condition comparisons and polarities stored in the `compound_lfc` grain for this RTA/TGA run. Adjust `log2_fold_change_abs_gt` to tighten or loosen the threshold (1.5 ≈ 2.8× change). Uncomment `polarity` to restrict to one ionisation mode.

```python
significant_compound_screen = wfs.run_parquet_query(
    project_root=PARQUET_ROOT,
    rt_alignment_number=RTA,
    analysis_number=TGA,
    params={
        'grain': 'compound_lfc',
        'project_name': PROJECT_NAME,
        'chromatography': CHROMATOGRAPHY,
        # 'polarity': 'POS',  # uncomment to restrict to one polarity
        'log2_fold_change_abs_gt': 1.5,
    },
)
print(significant_compound_screen.sql)
significant_compound_screen.df.head(20)
```

### `high_confidence_core_hits`

**Biological question:** What is the curated, publication-grade compound list for this project — i.e. compounds identified with a strong composite quality score AND sufficient intensity to be quantitatively meaningful?

This is the typical filter applied before pathway enrichment or supplementary-table export, since it excludes low-abundance or borderline-quality hits. `atlas_category: 'REF'` removes internal standards. Uncomment `polarity` to restrict to one analysis.

```python
high_confidence_core_hits = wfs.run_parquet_query(
    project_root=PARQUET_ROOT,
    rt_alignment_number=RTA,
    analysis_number=TGA,
    params={
        'grain': 'compound_file',
        'project_name': PROJECT_NAME,
        'chromatography': CHROMATOGRAPHY,
        # 'polarity': 'POS',           # uncomment to restrict to one polarity
        'atlas_category': 'REF',       # removes internal standards
        'total_score_min': 1.5,
        'best_ms1_intensity_min': 10_000,
    },
)
print(high_confidence_core_hits.sql)
high_confidence_core_hits.df.head(20)
```

---

## End-to-end call chain

```
params dict (notebook cell)
    │
    ▼
wfs.run_parquet_query()                 metatlas2/workflows.py
    │  thin dispatcher — constructs ParquetQueryInterpreter, calls execute()
    ▼
ParquetQueryInterpreter.__init__()      metatlas2/workflow_objects.py
    │  1. resolves grain subdirectory (compound_file or compound_lfc)
    │  2. pre-filters .parquet files to RTA{n}-TGA{m}-*.parquet via rglob
    │  3. builds pyarrow.dataset with Hive partitioning
    ▼
ParquetQueryInterpreter.execute()       metatlas2/workflow_objects.py
    ├── _build_filter(params)           params keys → PyArrow Expression (AND-chained)
    ├── _build_sql_string(params)       params keys → human-readable SQL (display only)
    └── dataset.to_table(filter=expr)  predicate-pushdown scan of .parquet files
            │
            ▼
    ParquetQueryResult(df=DataFrame, sql=str)   returned to notebook cell
```

### Step 1 — Notebook cell

The analyst calls `wfs.run_parquet_query()` with a `params` dict. `PARQUET_ROOT`, `RTA`, `TGA`, `PROJECT_NAME`, and `CHROMATOGRAPHY` are pre-filled in the `RUN_PARAMS` cell at notebook generation time.

### Step 2 — `wfs.run_parquet_query()`

Defined in `metatlas2/workflows.py`. A thin dispatcher that constructs a `ParquetQueryInterpreter` and calls `execute()`:

```python
grain = params.get("grain", "compound_lfc")
engine = ParquetQueryInterpreter(
    Path(project_root), grain,
    rt_alignment_number=rt_alignment_number,
    analysis_number=analysis_number,
)
return engine.execute(params)
```

### Step 3 — `ParquetQueryInterpreter.__init__()`

Defined in `metatlas2/workflow_objects.py`. Sets up the PyArrow dataset in two sub-steps:

**3a. Resolve the dataset directory**

The `grain` selects the top-level subdirectory (`compound_file` or `compound_lfc`). The Hive-partitioned directory tree under it encodes `project_name`, `chromatography`, `polarity`, and `atlas_name` as directory names.

**3b. RTA/TGA fragment pre-filter**

When `rt_alignment_number` and `analysis_number` are both provided, the constructor walks the grain directory with `rglob("*.parquet")` and keeps only files whose basename starts with `RTA{n}-TGA{m}-`. This is critical when multiple alignment/analysis runs share the same partition directory — without it, a query would scan data from all runs.

```python
file_prefix = f"RTA{rt_alignment_number}-TGA{analysis_number}-"
matching_files = sorted(
    str(p) for p in dataset_dir.rglob("*.parquet")
    if p.name.startswith(file_prefix)
)
```

**3c. Build the PyArrow dataset**

The filtered file list is passed to `ds.dataset()` with Hive partitioning, which reconstructs the partition columns from the directory names.

### Step 4 — `ParquetQueryInterpreter.execute()`

Two things happen in parallel:

**4a. `_build_filter(params)` → PyArrow predicate expression**

Each key in `params` (except `grain`) is translated using the filter suffix system into a `ds.Expression`. All expressions are AND-chained into a single `final_filter`. Keys referencing columns not present in the dataset schema are silently skipped with a printed warning.

**4b. `_build_sql_string(params)` → human-readable SQL**

The same params dict is rendered as `SELECT * FROM compound_lfc WHERE ...` — purely for display/debugging, not executed by any SQL engine. When RTA/TGA scoping is active, a `-- Scoped to: RTA{n} / TGA{m}` comment is prepended.

**4c. `dataset.to_table(filter=final_filter)`**

PyArrow applies the predicate expression using **predicate pushdown** — it reads only the row groups whose min/max statistics overlap the filter, skipping the rest. The result is a `pa.Table` converted to a `pandas.DataFrame`.

### Step 5 — `ParquetQueryResult` returned to the notebook

```python
result.df    # pandas DataFrame — ready for display, export, or further analysis
result.sql   # str — SQL equivalent, useful for debugging or passing to DuckDB
```

The key design principle: **all filtering happens at the PyArrow layer** — no data is loaded into Python memory until after predicate pushdown has reduced the result set. The SQL string is a display artifact only.

---

## API reference

### `wfs.run_parquet_query()`

```python
wfs.run_parquet_query(
    project_root: str,
    params: dict,
    rt_alignment_number: int | None = None,
    analysis_number: int | None = None,
) -> ParquetQueryResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_root` | `str` | Path to the directory containing `compound_file/` and `compound_lfc/` subdirectories (i.e. `PARQUET_ROOT`) |
| `params` | `dict` | Query parameter dict. Must include `grain`. All other keys are filter parameters (see [Filter suffix system](#filter-suffix-system)) |
| `rt_alignment_number` | `int \| None` | When provided with `analysis_number`, restricts the scan to files named `RTA{n}-TGA{m}-*.parquet`. Pass `None` to scan all runs |
| `analysis_number` | `int \| None` | See `rt_alignment_number` |

Returns a `ParquetQueryResult` named tuple with fields:

| Field | Type | Description |
|-------|------|-------------|
| `df` | `pandas.DataFrame` | Rows matching all filter predicates |
| `sql` | `str` | Human-readable SQL SELECT statement equivalent to the query |

### `ParquetQueryInterpreter`

Defined in `metatlas2/workflow_objects.py`. Not typically called directly — use `wfs.run_parquet_query()` instead.

```python
ParquetQueryInterpreter(
    parquet_output_path: Path,
    grain: str = "compound_lfc",
    rt_alignment_number: int | None = None,
    analysis_number: int | None = None,
)
```

Key methods:

| Method | Description |
|--------|-------------|
| `execute(params: dict) → ParquetQueryResult` | Run the query and return results |
| `_build_filter(params) → ds.Expression` | Translate params dict to a PyArrow filter expression |
| `_build_sql_string(params) → str` | Render params dict as a human-readable SQL string |
