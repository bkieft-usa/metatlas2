"""
Pytest configuration and fixtures for metatlas2 system tests.

Provides fixtures for test data paths, environment detection, and test cleanup.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Path to the static test fixtures directory.
    
    Returns the absolute path to tests/fixtures/data/ which contains
    pre-generated synthetic parquet files, DuckDB database, and MS2 references.
    """
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def test_project_name() -> str:
    """
    Consistent project name for system tests.
    
    Returns:
        str: The project name used across all system tests.
    """
    return "20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX"


@pytest.fixture(scope="function")
def test_output_dir() -> Generator[Path, None, None]:
    """
    Temporary directory for test outputs.
    
    Creates a temporary directory that is cleaned up after the test completes.
    
    Yields:
        Path: Absolute path to temporary output directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="metatlas2_test_"))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def is_nersc() -> bool:
    """
    Detect if running on NERSC infrastructure.
    
    Returns:
        bool: True if running at NERSC, False otherwise.
    """
    return (
        os.getenv("NERSC_HOST") is not None
        or os.getenv("SLURM_CLUSTER_NAME") is not None
    )


@pytest.fixture(scope="session")
def is_github_actions() -> bool:
    """
    Detect if running in GitHub Actions CI.
    
    Returns:
        bool: True if running in GitHub Actions, False otherwise.
    """
    return os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="session")
def test_config_path() -> Path:
    """
    Path to the system test configuration file.
    
    Returns:
        Path: Absolute path to configs/system_test_analysis.yaml
    """
    return Path(__file__).parent.parent / "configs" / "system_test_analysis.yaml"
