"""
Nox configuration for metatlas2 testing.

Usage:
    nox -s system_test    # Run system test (auto-detects environment)
    nox -l                # List available sessions
"""

import os
import shutil
import tempfile
from pathlib import Path
import nox


# Nox configuration
nox.options.sessions = ["system_test"]
nox.options.reuse_existing_virtualenvs = True


def is_nersc() -> bool:
    """Detect if running on NERSC infrastructure."""
    return (
        os.getenv("NERSC_HOST") is not None
        or os.getenv("SLURM_CLUSTER_NAME") is not None
    )


def is_github_actions() -> bool:
    """Detect if running in GitHub Actions CI."""
    return os.getenv("GITHUB_ACTIONS") == "true"


@nox.session(python=False)
def system_test(session):
    """
    Run end-to-end system test of the metatlas2 pipeline.
    
    Auto-detects environment (NERSC vs GitHub Actions) and runs the
    appropriate containerized workflow. Uses static fixtures from
    tests/fixtures/data/.
    
    Test validates:
    - Pipeline completes successfully
    - Expected output files are created
    - Database tables have correct schema and data
    - Results match baseline expectations
    """
    repo_root = Path(__file__).parent
    fixtures_dir = repo_root / "tests" / "fixtures" / "data"
    config_file = repo_root / "configs" / "system_test_analysis.yaml"
    
    # Validate that fixtures exist
    if not fixtures_dir.exists():
        session.error(
            f"Test fixtures not found at {fixtures_dir}. "
            "Run 'python tests/fixtures/generate_fixtures.py' first."
        )
    
    if not config_file.exists():
        session.error(f"Config file not found: {config_file}")
    
    # Project name for test
    test_project = "20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX"
    
    session.log("=" * 60)
    session.log("Running metatlas2 system test")
    session.log("=" * 60)
    session.log(f"Fixtures directory: {fixtures_dir}")
    session.log(f"Config file: {config_file}")
    session.log(f"Test project: {test_project}")
    session.log(f"Environment: {'NERSC' if is_nersc() else 'GitHub Actions' if is_github_actions() else 'Local Docker'}")
    session.log("=" * 60)
    
    # Track cleanup directory
    cleanup_dir = None
    
    if is_nersc():
        cleanup_dir = _run_test_on_nersc(session, repo_root, fixtures_dir, config_file, test_project)
    else:
        cleanup_dir = _run_test_with_docker(session, repo_root, fixtures_dir, config_file, test_project)
    
    # Run pytest validation tests
    session.log("\n" + "=" * 60)
    session.log("Running validation tests...")
    session.log("=" * 60)
    # Install pytest if not available (needed in CI environments)
    session.run("python", "-m", "pip", "install", "pytest>=8.3.4", external=True)
    session.run("python", "-m", "pytest", "tests/test_system.py", "-v", "--tb=short")
    
    # Clean up after tests complete
    if cleanup_dir and Path(cleanup_dir).exists():
        session.log(f"\nCleaning up {cleanup_dir}...")
        shutil.rmtree(cleanup_dir, ignore_errors=True)
    
    session.log("\n" + "=" * 60)
    session.log("✓ System test passed!")
    session.log("=" * 60)


def _run_test_on_nersc(session, repo_root, fixtures_dir, config_file, test_project):
    """
    Run test on NERSC using Shifter containers.
    
    Returns:
        str: Path to temporary directory to clean up after tests
    """
    session.log("\nRunning test on NERSC with Shifter...")
    
    # Create temporary directory for test data (Shifter needs data on GPFS)
    test_data_dir = Path(f"/tmp/metatlas-test-data-{os.getpid()}")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy fixtures to temp location
    session.log(f"Copying test fixtures to {test_data_dir}...")
    shutil.copytree(fixtures_dir, test_data_dir, dirs_exist_ok=True)
    
    # Set up environment for metatlas2.sh
    env = os.environ.copy()
    env["METATLAS_DATA_DIR"] = str(test_data_dir)
    
    # Run the pipeline using metatlas2.sh wrapper
    session.log("\nRunning metatlas2 pipeline...")
    session.run(
        "metatlas2.sh",
        "--dev",  # Use dev mode to test local changes
        "run",
        "--config", str(config_file),
        "--project", test_project,
        "--rt-align-num", "0",
        "--analysis-num", "0",
        "--overwrite",
        "--log-to-stdout",
        external=True,
        env=env
    )
    
    # Set output location for pytest validation
    # Pipeline writes to: {METATLAS_DATA_DIR}/projects/targeted_outputs/test_owner/{project_name}/
    output_dir = test_data_dir / "projects" / "targeted_outputs" / "test_owner" / test_project
    os.environ["TEST_OUTPUT_DIR"] = str(output_dir)
    
    # Return temp directory path for cleanup after tests
    return str(test_data_dir)


def _run_test_with_docker(session, repo_root, fixtures_dir, config_file, test_project):
    """
    Run test using Docker containers (for GitHub Actions or local development).
    
    Returns:
        str: Path to temporary directory to clean up after tests
    """
    session.log("\nRunning test with Docker...")
    
    # Docker image (use latest or specify via env var)
    image_tag = os.getenv("METATLAS2_IMAGE_TAG", "latest")
    docker_image = f"ghcr.io/bkieft-usa/metatlas2:{image_tag}"
    
    session.log(f"Using Docker image: {docker_image}")
    
    # Create temporary data directory and copy fixtures
    # Pipeline needs to write to METATLAS_DATA_DIR/projects/targeted_outputs/
    temp_data_dir = tempfile.mkdtemp(prefix="metatlas2-test-data-")
    session.log(f"Copying test fixtures to {temp_data_dir}...")
    shutil.copytree(fixtures_dir, temp_data_dir, dirs_exist_ok=True)
    
    # Prepare environment variables for container
    container_env = {
        "METATLAS_DATA_DIR": "/data",
        "HOME": "/tmp/home",
        "PYTHONPATH": "/app"
    }
    
    # Build docker run command
    docker_cmd = [
        "docker", "run",
        "--rm",
        # Mount temp data directory as read-write (contains fixtures + will hold outputs)
        "-v", f"{temp_data_dir}:/data:rw",
        # Mount config file
        "-v", f"{config_file.absolute()}:/config/system_test_analysis.yaml:ro",
    ]
    
    # Add environment variables
    for key, value in container_env.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    
    # Add image and command
    docker_cmd.extend([
        docker_image,
        "run",
        "--config", "/config/system_test_analysis.yaml",
        "--project", test_project,
        "--rt-align-num", "0",
        "--analysis-num", "0",
        "--overwrite",
        "--log-to-stdout"
    ])
    
    # Run the pipeline
    session.log("\nRunning metatlas2 pipeline in Docker container...")
    session.log(f"Command: {' '.join(docker_cmd)}")
    session.run(*docker_cmd, external=True)
    
    # Set output location for pytest validation
    # Pipeline writes to: {temp_data_dir}/projects/targeted_outputs/test_owner/{project_name}/
    output_dir = Path(temp_data_dir) / "projects" / "targeted_outputs" / "test_owner" / test_project
    os.environ["TEST_OUTPUT_DIR"] = str(output_dir)
    
    # Return temp directory path for cleanup after tests
    return temp_data_dir


@nox.session
def lint(session):
    """Run code quality checks with ruff."""
    session.install("ruff")
    session.run("ruff", "check", "metatlas2/", "tests/")


@nox.session
def format_check(session):
    """Check code formatting with ruff."""
    session.install("ruff")
    session.run("ruff", "format", "--check", "metatlas2/", "tests/")


@nox.session
def format(session):
    """Format code with ruff."""
    session.install("ruff")
    session.run("ruff", "format", "metatlas2/", "tests/")
