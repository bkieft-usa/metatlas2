# Development & Testing

This document covers the metatlas2 system test, how to run and extend it, how to regenerate test fixtures, and how the CI/CD pipeline works.

---

## System Test Overview

The system test is a full end-to-end integration test that runs the entire metatlas2 pipeline against synthetic fixtures and validates that the outputs are correct. It is the primary quality gate for the codebase.

**What is tested:**
- Pipeline completes without errors
- All expected output files and directories are created
- Project DuckDB contains all required tables with non-empty data
- RT alignment model meets minimum R² quality threshold
- Table row counts fall within expected ranges defined in the baseline file
- Output artifacts (CSVs, notebooks) are valid and non-empty

**What is NOT tested:**
- Scientific accuracy of peak picking or compound identification (covered by manual curation)
- GUI / Jupyter notebook rendering
- Slurm job submission

---

## Running the System Test

### Prerequisites

| Environment | Requirements |
|---|---|
| NERSC | `metatlas2.sh` on PATH, Shifter access, `uv` available |
| Local dev | Docker installed and running, `uv` on PATH |
| GitHub Actions | Handled automatically (see [CI/CD](#cicd)) |

### Run it

```bash
nox -s system_test
```

Nox auto-detects the environment based on environment variables:

| Env var present | Environment detected | Container runtime |
|---|---|---|
| `NERSC_HOST` or `SLURM_CLUSTER_NAME` | NERSC | Shifter via `metatlas2.sh --dev` |
| Neither | Local / CI | Docker via `docker run` |

To use a specific Docker image tag (instead of `latest`):

```bash
METATLAS2_IMAGE_TAG=sha-abc1234 nox -s system_test
```

### What happens during a run

1. **Validation** — nox checks that `tests/fixtures/data/` and `configs/system_test_analysis.yaml` exist.
2. **Temp directory** — a temporary data directory is created and the fixtures are copied into it:
   - NERSC: `/tmp/metatlas-test-data-<pid>/`
   - Docker: `tempfile.mkdtemp()` (e.g. `/tmp/metatlas2-test-data-XXXX/`)
3. **Pipeline run** — the container runs the full pipeline against the fixtures. Outputs are written to:
   ```
   <temp_dir>/projects/targeted_outputs/test_owner/<project_name>/
   ```
4. **pytest validation** — `tests/test_system.py` is run against the output directory (path passed via the `TEST_OUTPUT_DIR` env var).
5. **Cleanup** — the temp directory is deleted after pytest completes (even on failure).

---

## Test Fixtures

Static fixtures are committed to the repository in `tests/fixtures/data/`. They provide a minimal but complete synthetic dataset that exercises every stage of the pipeline.

### Structure

```
tests/fixtures/data/
├── databases/
│   └── main_db/
│       └── metatlas.duckdb         # Main database: compounds, atlases
├── lcmsruns/
│   └── test_owner/
│       └── 20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX/
│           ├── mzML/               # 10 synthetic mzML files (ISTD, QC, Sample)
│           ├── parquet/            # Converted parquet files (MS1 + MS2)
│           └── raw/
└── ms2_references.json              # MS2 reference library
```

### Fixture contents

| File / Table | Description |
|---|---|
| `metatlas.duckdb` | Contains a QC atlas (3 amino acid compounds for RT alignment) and a target atlas (2 compounds for auto-ID) |
| `ms2_references.json` | Minimal MS2 reference library matching the target compounds |
| `mzML/` | 10 synthetic LC-MS files: 2 ISTD, 3 QC, 5 Sample — all in positive mode |
| `parquet/` | Pre-converted parquet files for the mzML files above (MS1 and MS2 scan data with gaussian peaks) |

### Regenerating fixtures

Fixtures should only need to be regenerated if the database schema, parquet format, or test compound set changes. **Do not regenerate casually** — this updates the committed baseline and will affect all future test comparisons.

```bash
python tests/fixtures/generate_fixtures.py
```

After regenerating, update the baseline:

```bash
python tests/fixtures/generate_fixtures.py --update-baseline
```

Or manually edit `tests/fixtures/expected_baseline.json` to reflect the new expected row counts.

Commit both the updated fixtures and the updated baseline together.

---

## Validation Tests (`tests/test_system.py`)

pytest runs the following tests after the pipeline completes:

| Test | What it checks |
|---|---|
| `test_pipeline_completed` | Output directory exists |
| `test_required_files_exist` | `<project>.duckdb`, `RTA0/`, `RTA0/TGA0/`, `rt_aligned_atlases.csv`, `auto_ided_atlases.csv` all exist |
| `test_database_schema` | All 7 required tables are present in the project database |
| `test_database_data_quality` | Every required table has at least one row |
| `test_rt_alignment_quality` | RT alignment R² ≥ 0.3, degree ≥ 1, ≥ 2 compounds used |
| `test_baseline_comparison` | Row counts for all tables fall within ranges in `expected_baseline.json` |
| `test_output_artifacts_quality` | CSV and notebook output files are valid and non-empty |

### Required database tables

| Table | Stage that populates it |
|---|---|
| `lcmsruns` | Project Setup |
| `rt_alignment` | RT Alignment |
| `atlases` | RT Alignment / Auto-ID |
| `atlas_compound_associations` | RT Alignment / Auto-ID |
| `ms1_data` | Auto-ID |
| `ms2_data` | Auto-ID |
| `manual_curation` | Auto-ID |
| `ms2_hits` *(optional)* | Auto-ID (only if MS2 matches found) |

### Baseline file

`tests/fixtures/expected_baseline.json` defines the acceptable row count ranges and minimum R² threshold for each release. Example:

```json
{
  "table_row_counts": {
    "lcmsruns":  { "min": 35, "max": 45 },
    "ms1_data":  { "min": 30, "max": 40 }
  },
  "rt_alignment": {
    "min_r2": 0.3
  }
}
```

If the baseline file is missing, `test_baseline_comparison` is skipped (not failed), so a fresh checkout can still run tests.

---

## CI/CD

GitHub Actions (`.github/workflows/docker.yml`) runs automatically on every push to `main` with two sequential jobs:

### Job 1: `build-push`

Builds the Docker image and pushes it to the GitHub Container Registry with two tags:

| Tag | Value |
|---|---|
| `sha-<short-sha>` | e.g. `sha-a1b2c3d` |
| `latest` | Always points to the most recent `main` build |

Full image path: `ghcr.io/bkieft-usa/metatlas2:<tag>`

### Job 2: `system-test`

Runs after `build-push` succeeds:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs `uv`, `nox`, `pytest`, and `duckdb`
4. Pulls the freshly built `latest` image
5. Runs `nox -s system_test`

On failure, test outputs are uploaded as a GitHub Actions artifact (retained for 7 days) to aid debugging:
- `/tmp/metatlas2-test-output-*/`
- `~/.test_owner_metabolomics_data/`

### Triggering a manual test run

You can re-run the `system-test` job from the GitHub Actions UI without pushing new code. To test against a specific image tag:

1. Edit the `Run system test` step's `METATLAS2_IMAGE_TAG` env var, or
2. Set it as a repository variable in **Settings → Secrets and variables → Actions → Variables**.

---
