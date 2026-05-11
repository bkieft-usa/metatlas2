"""Nox configuration for metatlas2 testing."""

import nox


# Nox configuration
nox.options.sessions = ["system_test"]
nox.options.reuse_existing_virtualenvs = True

@nox.session(python=False)
def system_test(session):
    """
    Placeholder system test session.

    Keeps CI/CD wiring intact while real data assertions are under development.
    This session intentionally performs no checks and always succeeds.
    """
    session.log("system_test placeholder: passing with no assertions.")


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
