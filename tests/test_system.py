"""
System tests for metatlas2 pipeline.

These tests validate that the full pipeline runs successfully and produces
expected outputs. They run after the pipeline execution in noxfile.py.

Test data uses synthetic fixtures from tests/fixtures/data/ and validates
against expected baselines in tests/fixtures/expected_baseline.json.
"""

import json
import os
from pathlib import Path

import duckdb
import pytest


@pytest.fixture(scope="module")
def test_output_dir(test_project_name):
    """
    Get the output directory from environment (set by nox).
    
    Returns:
        Path: Directory containing test outputs
    """
    output_dir_env = os.getenv("TEST_OUTPUT_DIR")
    if output_dir_env:
        return Path(output_dir_env)
    
    # Fallback: default location
    return Path.home() / "test_owner_metabolomics_data" / test_project_name


@pytest.fixture(scope="module")
def project_db_path(test_output_dir, test_project_name):
    """
    Path to the project output database.
    
    Returns:
        Path: Path to <project>.duckdb file
    """
    return test_output_dir / f"{test_project_name}.duckdb"


@pytest.fixture(scope="module")
def baseline_path():
    """
    Path to the expected baseline JSON file.
    
    Returns:
        Path: Path to expected_baseline.json
    """
    return Path(__file__).parent / "fixtures" / "expected_baseline.json"


def test_pipeline_completed(test_output_dir):
    """Verify that the pipeline ran and created output directory."""
    assert test_output_dir.exists(), (
        f"Output directory not found: {test_output_dir}. "
        "Pipeline may not have completed successfully."
    )


def test_required_files_exist(test_output_dir, project_db_path, test_project_name):
    """Verify that all expected output files were created."""
    # Check for main database
    assert project_db_path.exists(), (
        f"Project database not found: {project_db_path}"
    )
    
    # Check for RT alignment output directory
    rta_dir = test_output_dir / "RTA0"
    assert rta_dir.exists(), "RT alignment directory (RTA0/) not found"
    
    # Check for analysis output directory
    tga_dir = rta_dir / "TGA0"
    assert tga_dir.exists(), "Analysis directory (RTA0/TGA0/) not found"
    
    # Check for RT aligned atlases file
    rt_atlases_file = rta_dir / "rt_aligned_atlases.csv"
    assert rt_atlases_file.exists(), "RT aligned atlases CSV not found"
    
    # Check for auto-IDed atlases file
    auto_ided_file = tga_dir / "auto_ided_atlases.csv"
    assert auto_ided_file.exists(), "Auto-identified atlases CSV not found"
    
    print(f"\n✓ All expected output files found")
    print(f"  Database: {project_db_path.name}")
    print(f"  RT alignment outputs: {rta_dir.name}/")
    print(f"  Auto-ID outputs: {tga_dir.name}/")


def test_database_schema(project_db_path):
    """Verify that the database has the expected tables with correct schema."""
    conn = duckdb.connect(str(project_db_path), read_only=True)
    
    try:
        # Get list of all tables
        tables_result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        tables = {row[0] for row in tables_result}
        
        # Required tables for a completed pipeline run
        required_tables = {
            'lcmsruns',              # Project setup
            'rt_alignment',          # RT alignment
            'atlases',               # RT alignment / auto-ID
            'atlas_compound_associations',  # RT alignment / auto-ID
            'ms1_data',              # Auto-ID
            'ms2_data',              # Auto-ID
            'manual_curation',       # Auto-ID
        }
        
        missing_tables = required_tables - tables
        assert not missing_tables, (
            f"Missing required tables in database: {missing_tables}"
        )
        
        # Optional tables (may or may not exist depending on data)
        optional_tables = {'ms2_hits'}  # Only if MS2 matches found
        
        print(f"\n✓ Found {len(tables)} tables in database")
        print(f"  Required tables: {', '.join(sorted(required_tables & tables))}")
        if optional_tables & tables:
            print(f"  Optional tables: {', '.join(sorted(optional_tables & tables))}")
        
    finally:
        conn.close()


def test_database_data_quality(project_db_path):
    """Verify that database tables contain expected data."""
    conn = duckdb.connect(str(project_db_path), read_only=True)
    
    try:
        # Check lcmsruns table
        lcmsruns_count = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0]
        assert lcmsruns_count > 0, "lcmsruns table is empty"
        print(f"\n✓ Found {lcmsruns_count} LCMS runs")
        
        # Check RT alignment models
        rt_models_count = conn.execute("SELECT COUNT(*) FROM rt_alignment").fetchone()[0]
        assert rt_models_count > 0, "rt_alignment table is empty"
        print(f"✓ Found {rt_models_count} RT alignment model(s)")
        
        # Check atlases
        atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
        assert atlases_count > 0, "atlases table is empty"
        print(f"✓ Found {atlases_count} atlas(es)")
        
        # Check atlas compound associations
        associations_count = conn.execute(
            "SELECT COUNT(*) FROM atlas_compound_associations"
        ).fetchone()[0]
        assert associations_count > 0, "atlas_compound_associations table is empty"
        print(f"✓ Found {associations_count} atlas-compound associations")
        
        # Check MS1 data
        ms1_count = conn.execute("SELECT COUNT(*) FROM ms1_data").fetchone()[0]
        assert ms1_count > 0, "ms1_data table is empty"
        print(f"✓ Found {ms1_count} MS1 data rows")
        
        # Check manual curation (should have one row per compound-file combination)
        curation_count = conn.execute("SELECT COUNT(*) FROM manual_curation").fetchone()[0]
        assert curation_count > 0, "manual_curation table is empty"
        print(f"✓ Found {curation_count} manual curation entries")
        
        print(f"\n✓ All database tables have expected data")
        
    finally:
        conn.close()


def test_rt_alignment_quality(project_db_path):
    """Verify RT alignment model meets quality thresholds."""
    conn = duckdb.connect(str(project_db_path), read_only=True)
    
    try:
        # Get RT alignment model statistics
        models = conn.execute("""
            SELECT r_squared, polynomial_degree, num_compounds
            FROM rt_alignment
        """).fetchall()
        
        assert len(models) > 0, "No RT alignment models found"
        
        for r2, degree, n_compounds in models:
            # For synthetic data with relaxed thresholds, R² >= 0.3 is acceptable
            assert r2 >= 0.3, f"RT alignment R² too low: {r2} (expected >= 0.3)"
            assert degree >= 1, f"Polynomial degree too low: {degree}"
            assert n_compounds >= 2, f"Too few compounds used: {n_compounds}"
            
            print(f"\n✓ RT alignment model quality:")
            print(f"  R² = {r2:.3f}")
            print(f"  Polynomial degree = {degree}")
            print(f"  Compounds used = {n_compounds}")
        
    finally:
        conn.close()


def test_baseline_comparison(project_db_path, baseline_path):
    """
    Compare test outputs against expected baseline metrics.
    
    This test validates that the pipeline produces results consistent
    with known-good outputs. Allows for minor variations in numerical
    values while catching major regressions.
    """
    if not baseline_path.exists():
        pytest.skip(f"Baseline file not found: {baseline_path}. Run test once to generate.")
    
    # Load baseline expectations
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    conn = duckdb.connect(str(project_db_path), read_only=True)
    
    try:
        # Check table row counts are within expected ranges
        for table_name, expected_range in baseline.get("table_row_counts", {}).items():
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            min_count, max_count = expected_range["min"], expected_range["max"]
            
            assert min_count <= count <= max_count, (
                f"Table {table_name} row count {count} outside expected range "
                f"[{min_count}, {max_count}]"
            )
            print(f"✓ {table_name}: {count} rows (expected {min_count}-{max_count})")
        
        # Check RT alignment R² meets threshold
        r2_threshold = baseline.get("rt_alignment", {}).get("min_r2", 0.3)
        r2_values = conn.execute("SELECT r_squared FROM rt_alignment").fetchall()
        for (r2,) in r2_values:
            assert r2 >= r2_threshold, (
                f"RT alignment R² {r2} below baseline threshold {r2_threshold}"
            )
        
        print("\n✓ All baseline checks passed")
        
    finally:
        conn.close()


def test_output_artifacts_quality(test_output_dir):
    """Verify quality and completeness of output artifacts."""
    rta_dir = test_output_dir / "RTA0"
    tga_dir = rta_dir / "TGA0"
    
    # Check for RT alignment summary PDF
    rt_summary_dir = rta_dir / "rt_alignment_results"
    if rt_summary_dir.exists():
        pdf_files = list(rt_summary_dir.glob("*.pdf"))
        if pdf_files:
            print(f"\n✓ Found RT alignment summary: {pdf_files[0].name}")
    
    # Check for CSV output files
    rt_atlases_csv = rta_dir / "rt_aligned_atlases.csv"
    assert rt_atlases_csv.exists(), "RT aligned atlases CSV not found"
    assert rt_atlases_csv.stat().st_size > 0, "RT aligned atlases CSV is empty"
    
    auto_id_csv = tga_dir / "auto_ided_atlases.csv"
    assert auto_id_csv.exists(), "Auto-IDed atlases CSV not found"
    assert auto_id_csv.stat().st_size > 0, "Auto-IDed atlases CSV is empty"
    
    print(f"✓ Output artifacts verified in {test_output_dir.name}/")


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])
