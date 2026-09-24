# Development & Testing

This document covers the metatlas2 test suite, how to run and extend it, and how the CI/CD pipeline works.

---

## Test Suite Overview

The test suite is split into two tiers:

| Tier | Marker | When it runs | What it covers |
|---|---|---|---|
| **Unit tests** | `unit` (or no marker) | Every push to every branch | Fast, isolated tests with no Docker, no network, no real databases |
| **System tests** | `system` | Only on pull requests targeting `main` | End-to-end integration tests that exercise the full Python pipeline against synthetic fixtures |

**What the system tests cover:**
- `add-compounds` pipeline: creates the main database, persists compounds, handles duplicates
- `add-atlases` pipeline: creates atlases from TSV files, validates compound associations
- `run` pipeline: project setup, RT alignment, and auto-identification stages against synthetic HDF5 data

**What is NOT tested:**
- GUI / Jupyter notebook rendering
- Slurm job submission
- Real network calls (PubChem is patched to return canned data)
- Real NERSC/Shifter container runtime

---

## Running the Tests

### Prerequisites

| Environment | Requirements |
|---|---|
| Local dev | Python 3.11+, `uv` on PATH |
| GitHub Actions | Handled automatically (see [CI/CD](#cicd)) |

### Install test dependencies

```bash
uv pip install -e ".[test]"
```

### Run unit tests only (fast, no Docker required)

```bash
pytest tests/ -m "not system"
```

### Run system tests only

```bash
pytest tests/test_system.py -m "system"
```

### Run the full test suite

```bash
pytest tests/
```

---

## Test Structure

All tests live in the `tests/` directory:

```
tests/
├── conftest.py                      # Shared fixtures: synthetic HDF5 writer, atlas/compound builders
├── test_system.py                   # System tests (marked @pytest.mark.system)
├── test_analysis_summary.py         # Unit tests for analysis_summary.py
├── test_create_curation_container.py # Unit tests for create_curation_container.py
├── test_extract_data_from_h5.py     # Unit tests for extract_data_from_h5.py
├── test_file_and_project_format.py  # Unit tests for file_and_project_format.py
├── test_lcmsruns_tools.py           # Unit tests for lcmsruns_tools.py
├── test_ms2_hit_detection.py        # Unit tests for ms2_hit_detection.py
├── test_msms_refs_db.py             # Unit tests for MSMS refs database operations
├── test_rt_align_tools.py           # Unit tests for rt_align_tools.py
└── test_utils.py                    # Unit tests for utils.py
```

### Shared fixtures (`conftest.py`)

The `conftest.py` file provides:

- **`write_synthetic_h5(path, ...)`** — writes a minimal HDF5 file (using PyTables, the same library the production code uses) with configurable MS1/MS2 rows for positive and negative polarity. This exercises the real HDF5 read path end-to-end without mocking I/O.
- **`_make_compound_mzrt(...)`** — builds a `CompoundMZRT` dataclass directly for atlas construction.
- **`_make_atlas(compounds, ...)`** — wraps a list of `CompoundMZRT` objects into an `Atlas` dataclass.
- Compound constants (`ADENINE_MZ`, `ADENINE_RT`, etc.) used across multiple test files.

### System test fixtures (`test_system.py`)

The system tests create a complete but minimal environment in `tmp_path`:

| Fixture | What it creates |
|---|---|
| `data_dir` | A `METATLAS_DATA_DIR`-compatible directory tree under `tmp_path` |
| `metatlas_data_dir` | Patches `METATLAS_DATA_DIR` env var to point at `data_dir` |
| `main_db_path` | Path to `databases/main_db/metatlas.duckdb` |
| `adenine_tsv` | A two-compound TSV file (adenine + riboflavin) |
| `compounds_yaml` | A `create_compounds.yaml` pointing at `adenine_tsv` |
| `atlas_tsv` | A two-compound atlas TSV file |
| `atlases_yaml` | A `create_atlases.yaml` pointing at `atlas_tsv` |
| `analysis_yaml` | A minimal `analysis.yaml` with the four-level `TARGETED_ANALYSES` structure |
| `seeded_analysis_yaml` | An `analysis_yaml` whose atlas UIDs match a real seeded database |

---

## System Test Classes

### `TestAddCompounds`

Tests for `metatlas2.sh add-compounds` (`add_compounds_to_db.py`):

| Test | What it checks |
|---|---|
| `test_add_compounds_creates_main_database` | Running add-compounds creates the main DuckDB file |
| `test_add_compounds_persists_compounds_to_db` | Compounds from the TSV are queryable from the database |
| `test_add_compounds_correct_compound_count` | Exact compound count matches the input TSV |
| `test_add_compounds_idempotent_on_rerun` | Running twice does not duplicate rows |

### `TestAddAtlases`

Tests for `metatlas2.sh add-atlases` (`add_atlases_to_db.py`):

| Test | What it checks |
|---|---|
| `test_add_atlases_creates_atlas_in_db` | Running add-atlases creates an atlas record in the database |
| `test_add_atlases_correct_compound_count` | Atlas compound count matches the input TSV |
| `test_add_atlases_compound_associations` | `atlas_compound_associations` table is populated correctly |

### `TestRunWorkflow`

Tests for `metatlas2.sh run` (`run_targeted_analysis.py`):

| Test | What it checks |
|---|---|
| `test_project_setup_creates_database` | Project setup creates the project DuckDB file |
| `test_project_setup_populates_lcmsruns` | `lcmsruns` table is populated with the synthetic HDF5 files |
| `test_rt_alignment_runs_successfully` | RT alignment completes and writes a model to the database |
| `test_auto_identification_runs_successfully` | Auto-ID completes and populates `ms1_data` and `manual_curation` tables |

---

## CI/CD

GitHub Actions (`.github/workflows/tests.yml` and `.github/workflows/docker.yml`) run automatically.

### Workflow: `tests.yml`

Runs on every push to any branch (except `main`) and on every pull request targeting `main`.

#### Job 1: `unit-tests`

Runs on every push to every branch:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs `uv` and project + test dependencies (`uv pip install -e ".[test]"`)
4. Runs `pytest tests/ -m "not system"`
5. Uploads test results as a GitHub Actions artifact (retained for 14 days)

#### Job 2: `system-tests`

Runs only on pull requests targeting `main`, after `unit-tests` passes:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs `uv` and project + test dependencies
4. Runs `pytest tests/test_system.py -m "system"`
5. Uploads test results as a GitHub Actions artifact (retained for 14 days)
6. On failure, uploads `/tmp/pytest-*/` output directories for debugging (retained for 7 days)

### Workflow: `docker.yml`

Runs only on pushes to `main` (i.e., after a PR is merged):

1. Builds the Docker image for `linux/amd64` and `linux/arm64`
2. Pushes two tags to GHCR:
   - `sha-<7chars>` — immutable, traceable to the triggering commit
   - `latest` — floating pointer to the most recent build

---

## pytest Markers

| Marker | Description |
|---|---|
| `system` | End-to-end system tests; only run on PRs targeting `main` |
| `unit` | Fast unit tests with no external dependencies (default for branch pushes) |

Tests without a marker are treated as unit tests and run on every push.

To run only tests with a specific marker:

```bash
pytest tests/ -m "system"
pytest tests/ -m "not system"
pytest tests/ -m "unit"
```

---

## Adding New Tests

### Unit tests

Add a new file `tests/test_<module_name>.py`. Use `conftest.py` fixtures for HDF5 files, atlas objects, and compound objects. Mark slow or integration-heavy tests with `@pytest.mark.system`.

### System tests

Add a new test class or method to `tests/test_system.py` and decorate it with `@pytest.mark.system`. Use the `metatlas_data_dir` fixture to ensure `METATLAS_DATA_DIR` is patched, and use `tmp_path` for all filesystem I/O.

### Patching PubChem

All system tests that call `add_compounds_to_db` must patch PubChem to avoid real network calls:

```python
from unittest.mock import patch

def _fake_pubchem_info(compounds, **kwargs):
    return compounds  # return input unchanged

with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
    add_compounds_to_db(str(compounds_yaml), overwrite_db=True)
```
