# Metatlas2 System Tests

This directory contains end-to-end system tests for the metatlas2 pipeline.

## Overview

The system test validates the complete pipeline workflow:
1. **Project Setup** - Scans parquet files and creates project database
2. **RT Alignment** - Builds RT correction model from QC compounds
3. **Auto Identification** - Extracts MS1/MS2 data and matches compounds

Tests use synthetic test fixtures (pre-generated, committed to repo) to ensure:
- Reproducible test results across all environments
- Fast execution (~5-15 minutes)
- No dependency on production data

## Quick Start

### 1. Generate Test Fixtures (One-Time Setup)

```bash
# Generate synthetic test data
python tests/fixtures/generate_fixtures.py

# This creates:
#   tests/fixtures/data/databases/main_db/metatlas.duckdb
#   tests/fixtures/data/lcmsruns/test_owner/20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX/parquet/*.parquet
#   tests/fixtures/data/ms2_references.tsv

# The script will print atlas UIDs - copy these to configs/system_test_analysis.yaml
```

### 2. Update Configuration

Edit `configs/system_test_analysis.yaml` and replace placeholder UIDs with the values printed by `generate_fixtures.py`:

```yaml
RT_ALIGNMENT:
  HILICZ:
    ATLAS:
      uid: atlas-test-<ACTUAL_UID_HERE>  # Replace placeholder
```

### 3. Run System Test

#### At NERSC (using Shifter):
```bash
nox -s system_test
```

#### Locally or in CI (using Docker):
```bash
nox -s system_test
```

The test automatically detects the environment and uses the appropriate container runtime.

## Test Components

### Fixtures (`tests/fixtures/`)

- **`generate_fixtures.py`** - Script to create synthetic test data (run once)
- **`data/`** - Directory containing committed test fixtures (~5-10 MB)
  - `databases/main_db/metatlas.duckdb` - Main database with compounds and atlases
  - `lcmsruns/test_owner/20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX/parquet/` - Synthetic MS1/MS2 parquet files
  - `ms2_references.tsv` - MS2 reference library for hit detection
- **`expected_baseline.json`** - Expected output metrics for validation

### Test Files

- **`conftest.py`** - Pytest fixtures (test data paths, environment detection)
- **`test_system.py`** - Validation tests that run after pipeline execution

### Configuration

- **`configs/system_test_analysis.yaml`** - Pipeline configuration optimized for testing
  - Relaxed thresholds for fast execution
  - References test atlases by UID
  - Skips notebook generation (not needed in CI)

### Nox Session

- **`noxfile.py`** - Test orchestration
  - Detects environment (NERSC vs GitHub Actions)
  - Runs pipeline in appropriate container
  - Executes pytest validation tests

## Validation Tests

The system test validates:

1. **Pipeline Completion** - Verifies output directory was created
2. **Required Files** - Checks for database, log files, and output CSVs
3. **Database Schema** - Validates all expected tables exist
4. **Data Quality** - Verifies tables are populated and values are reasonable
5. **RT Alignment Quality** - Checks R² meets minimum threshold
6. **Baseline Comparison** - Compares outputs against expected ranges

## Test Data

### Compounds

The test uses 10 synthetic compounds:

**QC Atlas (3 compounds):**
- TestQC1_Alanine (m/z 90.055, RT 2.5 min)
- TestQC2_Valine (m/z 118.086, RT 5.0 min)
- TestQC3_Leucine (m/z 132.102, RT 7.5 min)

**ISTD Atlas (2 compounds):**
- ISTD1_Glucose (m/z 181.071, RT 3.0 min)
- ISTD2_Citrate (m/z 193.035, RT 4.5 min)

**EMA Atlas (5 compounds):**
- EMA1_Glutamate through EMA5_Tryptophan

### LCMS Files

10 synthetic parquet files with gaussian-shaped peaks:
- 3 QC files
- 2 ISTD files
- 5 experimental sample files

Each file contains both MS1 and MS2 data.

## Updating Baseline

After a successful pipeline change that intentionally alters outputs:

1. Run the test: `nox -s system_test`
2. Inspect the output database to get new metrics
3. Update `tests/fixtures/expected_baseline.json` with new ranges
4. Commit the updated baseline

## GitHub Actions CI

The system test runs automatically on every push to main:

1. Build Docker image
2. Push to GHCR
3. Run system test using the just-built image
4. Upload test artifacts if test fails

See `.github/workflows/docker.yml` for the workflow configuration.

## Troubleshooting

### Test fails with "Fixtures not found"

Run the fixture generator first:
```bash
python tests/fixtures/generate_fixtures.py
```

### Test fails with "Atlas UID not found"

Update `configs/system_test_analysis.yaml` with the correct atlas UIDs from fixture generation.

### Docker image not found (local testing)

Pull the latest image or build locally:
```bash
docker pull ghcr.io/bkieft-usa/metatlas2:latest
# OR
docker build -t ghcr.io/bkieft-usa/metatlas2:latest .
```

### Different results at NERSC vs GitHub Actions

Check for environment-specific issues:
- File permissions
- Path differences
- Container runtime differences (Shifter vs Docker)

## Development

To add new validation tests:

1. Add test function to `tests/test_system.py`
2. Use existing fixtures from `conftest.py`
3. Run locally: `nox -s system_test`
4. Update baseline if needed

To modify test data:

1. Edit `tests/fixtures/generate_fixtures.py`
2. Regenerate fixtures: `python tests/fixtures/generate_fixtures.py`
3. Update atlas UIDs in config if changed
4. Commit updated fixtures to repo
