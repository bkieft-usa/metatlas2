"""Unit tests for metatlas2.get_db_table.

Covers:
* :func:`~metatlas2.get_db_table.export_table_to_csv` — happy path, invalid
  table name, empty table, default output path, database_path override.
* :func:`~metatlas2.get_db_table._get_valid_tables` — returns the live table list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import metatlas2.database_interact as dbi
from metatlas2.get_db_table import export_table_to_csv, _get_valid_tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> str:
    """Create a fresh main-DB schema in a temp file and return its path."""
    db_path = str(tmp_path / "test_main.duckdb")
    dbi.create_metatlas_database(db_path, overwrite=True)
    return db_path


def _sample_msms_records(n: int = 3) -> list[dict]:
    """Return *n* minimal MSMS ref record dicts for batch_save_msms_refs."""
    records = []
    for i in range(n):
        records.append({
            "database": "metatlas",
            "ref_id": f"ref_{i:04d}",
            "name": f"compound_{i}",
            "inchi_key": "JPIJQSOTBSSVTP-STHAYSLISA-N",
            "precursor_mz": 135.03 + i,
            "polarity": "positive",
            "adduct": None,
            "fragmentation_method": "cid",
            "collision_energy": None,
            "instrument": None,
            "instrument_type": None,
            "formula": "C4H8O5",
            "mono_isotopic_molecular_weight": 136.037,
            "inchi": None,
            "smiles": None,
            "mz": [100.0 + i, 200.0 + i],
            "intensities": [1000.0, 500.0],
        })
    return records


# ---------------------------------------------------------------------------
# Tests for _get_valid_tables
# ---------------------------------------------------------------------------

class TestGetValidTables:

    def test_returns_list_of_strings(self, tmp_path):
        db_path = _make_db(tmp_path)
        tables = _get_valid_tables(db_path)
        assert isinstance(tables, list)
        assert all(isinstance(t, str) for t in tables)

    def test_includes_known_tables(self, tmp_path):
        db_path = _make_db(tmp_path)
        tables = _get_valid_tables(db_path)
        for expected in ("msms_refs", "compounds", "atlases", "projects"):
            assert expected in tables, f"Expected table '{expected}' not found in {tables}"

    def test_is_sorted(self, tmp_path):
        db_path = _make_db(tmp_path)
        tables = _get_valid_tables(db_path)
        assert tables == sorted(tables)


# ---------------------------------------------------------------------------
# Tests for export_table_to_csv
# ---------------------------------------------------------------------------

class TestExportTableToCsv:

    def test_creates_csv_file(self, tmp_path):
        """export_table_to_csv() creates the output file."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "msms_refs.csv")
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        assert Path(out).exists()

    def test_csv_has_correct_row_count(self, tmp_path):
        """Exported CSV has one row per DB row (plus header)."""
        db_path = _make_db(tmp_path)
        dbi.batch_save_msms_refs(db_path, _sample_msms_records(4))
        out = str(tmp_path / "msms_refs.csv")
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        df = pd.read_csv(out)
        assert len(df) == 4

    def test_csv_has_expected_columns(self, tmp_path):
        """Exported CSV contains the expected column names."""
        db_path = _make_db(tmp_path)
        dbi.batch_save_msms_refs(db_path, _sample_msms_records(1))
        out = str(tmp_path / "msms_refs.csv")
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        df = pd.read_csv(out)
        for col in ("inchi_key", "polarity", "database", "precursor_mz"):
            assert col in df.columns, f"Expected column '{col}' missing from CSV"

    def test_empty_table_produces_header_only_csv(self, tmp_path):
        """An empty table produces a CSV with headers but no data rows."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "msms_refs_empty.csv")
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        df = pd.read_csv(out)
        assert len(df) == 0
        assert len(df.columns) > 0

    def test_invalid_table_name_raises_value_error(self, tmp_path):
        """An invalid table name raises ValueError with a helpful message."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "bad.csv")
        with pytest.raises(ValueError, match="not found"):
            export_table_to_csv("nonexistent_table", output_path=out, database_path=db_path)

    def test_invalid_table_error_lists_available_tables(self, tmp_path):
        """The ValueError message lists the available tables."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "bad.csv")
        with pytest.raises(ValueError, match="msms_refs"):
            export_table_to_csv("nonexistent_table", output_path=out, database_path=db_path)

    def test_default_output_path_in_home(self, tmp_path, monkeypatch):
        """When output_path is None, the CSV is written to $HOME/<table>_export.csv."""
        db_path = _make_db(tmp_path)
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        export_table_to_csv("msms_refs", output_path=None, database_path=db_path)
        expected = fake_home / "msms_refs_export.csv"
        assert expected.exists()

    def test_database_path_override(self, tmp_path):
        """Passing database_path uses that DB rather than the environment default."""
        db_path = _make_db(tmp_path)
        dbi.batch_save_msms_refs(db_path, _sample_msms_records(2))
        out = str(tmp_path / "override.csv")
        # Explicitly pass database_path — no METATLAS_DATA_DIR needed
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        df = pd.read_csv(out)
        assert len(df) == 2

    def test_compounds_table_exportable(self, tmp_path):
        """The compounds table (empty) can be exported without error."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "compounds.csv")
        export_table_to_csv("compounds", output_path=out, database_path=db_path)
        assert Path(out).exists()

    def test_output_path_parent_created(self, tmp_path):
        """export_table_to_csv() creates missing parent directories."""
        db_path = _make_db(tmp_path)
        out = str(tmp_path / "subdir" / "nested" / "msms_refs.csv")
        export_table_to_csv("msms_refs", output_path=out, database_path=db_path)
        assert Path(out).exists()
