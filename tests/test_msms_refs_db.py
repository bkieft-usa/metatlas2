"""Unit tests for the MSMS-refs database integration.

Covers:
* :func:`~metatlas2.database_interact.batch_save_msms_refs`
* :func:`~metatlas2.load_tools.load_msms_refs_from_db`
* :class:`~metatlas2.workflow_objects.NewMsmsRefsConfig` (YAML parsing + execute)

All tests use a temporary DuckDB file created via
:func:`~metatlas2.database_interact.create_metatlas_database` so the full
schema (including ``reference_fragmentation_data``) is exercised without
touching any production database.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

import metatlas2.database_interact as dbi
import metatlas2.load_tools as ldt
from metatlas2.workflow_objects import NewMsmsRefsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> str:
    """Create a fresh main-DB schema in a temp file and return its path."""
    db_path = str(tmp_path / "test_main.duckdb")
    dbi.create_metatlas_database(db_path, overwrite=True)
    return db_path


def _sample_records(
    n: int = 3,
    database: str = "metatlas",
    polarity: str = "positive",
    inchi_key: str = "JPIJQSOTBSSVTP-STHAYSLISA-N",
) -> list[dict]:
    """Return *n* minimal MSMS ref record dicts ready for batch_save_msms_refs."""
    records = []
    for i in range(n):
        records.append({
            "database": database,
            "ref_id": f"ref_{i:04d}",
            "name": f"compound_{i}",
            "inchi_key": inchi_key,
            "precursor_mz": 135.03 + i,
            "polarity": polarity,
            "adduct": None,
            "fragmentation_method": "cid",
            "collision_energy": None,
            "instrument": None,
            "instrument_type": None,
            "formula": "C4H8O5",
            "mono_isotopic_molecular_weight": 136.037,
            "inchi": None,
            "smiles": None,
            "mz": [59.01, 71.01, 87.01, 117.02, 135.03],
            "intensities": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        })
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as a JSONL file."""
    with path.open("w") as fh:
        for rec in records:
            # Convert to the raw jsonl format (mz/intensities as lists)
            fh.write(json.dumps(rec) + "\n")


def _jsonl_record(
    database: str = "metatlas",
    polarity: str = "positive",
    inchi_key: str = "JPIJQSOTBSSVTP-STHAYSLISA-N",
    ref_id: str = "ref_0000",
    precursor_mz: float = 135.03,
) -> dict:
    """Return a single JSONL-format record (as written by the original converter)."""
    return {
        "ix": 0,
        "database": database,
        "id": ref_id,
        "name": "L-threonic acid",
        "decimal": 4.0,
        "inchi_key": inchi_key,
        "precursor_mz": precursor_mz,
        "polarity": polarity,
        "adduct": None,
        "fragmentation_method": "cid",
        "collision_energy": None,
        "instrument": None,
        "instrument_type": None,
        "formula": "C4H8O5",
        "mono_isotopic_molecular_weight": 136.037,
        "inchi": None,
        "smiles": None,
        "mz": [59.01, 71.01, 87.01, 117.02, 135.03],
        "intensities": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    }


# ---------------------------------------------------------------------------
# batch_save_msms_refs
# ---------------------------------------------------------------------------

class TestBatchSaveMsmsRefs:

    def test_basic_insert(self, tmp_path):
        """Three records are inserted and the row count matches."""
        db_path = _make_db(tmp_path)
        records = _sample_records(3)
        inserted = dbi.batch_save_msms_refs(db_path, records)
        assert inserted == 3

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_fragmentation_data"
            ).fetchone()[0]
        assert count == 3

    def test_mz_intensities_round_trip(self, tmp_path):
        """mz and intensities arrays survive the DB round-trip as Python lists."""
        db_path = _make_db(tmp_path)
        mz_in = [59.01, 71.01, 87.01, 117.02, 135.03]
        int_in = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        records = [{
            **_sample_records(1)[0],
            "mz": mz_in,
            "intensities": int_in,
        }]
        dbi.batch_save_msms_refs(db_path, records)

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT mz, intensities FROM reference_fragmentation_data LIMIT 1"
            ).fetchone()
        mz_out, int_out = row
        assert list(mz_out) == pytest.approx(mz_in, rel=1e-5)
        assert list(int_out) == pytest.approx(int_in, rel=1e-5)

    def test_always_inserts_new_uid(self, tmp_path):
        """Re-inserting the same record produces two rows with different ref_uids."""
        db_path = _make_db(tmp_path)
        records = _sample_records(1)
        dbi.batch_save_msms_refs(db_path, records)
        dbi.batch_save_msms_refs(db_path, records)

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            rows = conn.execute(
                "SELECT ref_uid FROM reference_fragmentation_data"
            ).fetchall()
        uids = [r[0] for r in rows]
        assert len(uids) == 2
        assert uids[0] != uids[1]

    def test_ref_uid_prefix(self, tmp_path):
        """Generated ref_uid values start with the expected 'msms-ref-' prefix."""
        db_path = _make_db(tmp_path)
        dbi.batch_save_msms_refs(db_path, _sample_records(1))

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            uid = conn.execute(
                "SELECT ref_uid FROM reference_fragmentation_data LIMIT 1"
            ).fetchone()[0]
        assert uid.startswith("msms-ref-")

    def test_empty_records_returns_zero(self, tmp_path):
        """Calling with an empty list inserts nothing and returns 0."""
        db_path = _make_db(tmp_path)
        inserted = dbi.batch_save_msms_refs(db_path, [])
        assert inserted == 0

    def test_batch_size_respected(self, tmp_path):
        """Inserting more records than batch_size still inserts all rows."""
        db_path = _make_db(tmp_path)
        records = _sample_records(7)
        inserted = dbi.batch_save_msms_refs(db_path, records, batch_size=3)
        assert inserted == 7

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_fragmentation_data"
            ).fetchone()[0]
        assert count == 7


# ---------------------------------------------------------------------------
# load_msms_refs_from_db
# ---------------------------------------------------------------------------

class TestLoadMsmsRefsFromDb:

    def _populate(self, db_path: str, records: list[dict]) -> None:
        dbi.batch_save_msms_refs(db_path, records)

    def test_no_filter_returns_all(self, tmp_path):
        """Without filters all inserted spectra are returned."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(3))
        result = ldt.load_msms_refs_from_db(db_path)
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_returns_spectrum_objects(self, tmp_path):
        """Returned values are matchms Spectrum objects with correct metadata."""
        from matchms import Spectrum
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(1))
        result = ldt.load_msms_refs_from_db(db_path)
        assert len(result) == 1
        spectra = list(result.values())[0]
        assert len(spectra) == 1
        spec = spectra[0]
        assert isinstance(spec, Spectrum)
        assert spec.metadata.get("database") == "metatlas"
        assert spec.metadata.get("inchi_key") == "JPIJQSOTBSSVTP-STHAYSLISA-N"

    def test_mz_intensities_as_float32_arrays(self, tmp_path):
        """mz and intensities on returned Spectrum objects are float32 numpy arrays."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(1))
        result = ldt.load_msms_refs_from_db(db_path)
        spec = list(result.values())[0][0]
        assert spec.mz.dtype == np.float32
        assert spec.intensities.dtype == np.float32

    def test_database_filter(self, tmp_path):
        """database_filter restricts results to matching rows only."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(2, database="metatlas"))
        self._populate(db_path, _sample_records(1, database="gnps"))
        result = ldt.load_msms_refs_from_db(db_path, database_filter="metatlas")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_polarity_filter_positive(self, tmp_path):
        """polarity='positive' returns only positive-mode spectra."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(2, polarity="positive"))
        self._populate(db_path, _sample_records(1, polarity="negative"))
        result = ldt.load_msms_refs_from_db(db_path, polarity="positive")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_polarity_filter_abbreviated(self, tmp_path):
        """polarity='pos' is normalised to 'positive' before querying."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(2, polarity="positive"))
        result = ldt.load_msms_refs_from_db(db_path, polarity="pos")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_inchi_key_filter(self, tmp_path):
        """inchi_keys filter restricts results to the requested keys."""
        db_path = _make_db(tmp_path)
        key_a = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        key_b = "HDYANYHVCAPMJV-LXQIFKJMSA-N"
        self._populate(db_path, _sample_records(2, inchi_key=key_a))
        self._populate(db_path, _sample_records(1, inchi_key=key_b))
        result = ldt.load_msms_refs_from_db(db_path, inchi_keys=[key_a])
        assert set(result.keys()) == {key_a}
        assert len(result[key_a]) == 2

    def test_inchi_key_filter_empty_list_raises(self, tmp_path):
        """An empty inchi_keys list raises ValueError (no spectra can match)."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(1))
        with pytest.raises(ValueError, match="No spectra remained"):
            ldt.load_msms_refs_from_db(db_path, inchi_keys=[])

    def test_no_match_raises_value_error(self, tmp_path):
        """ValueError is raised when no spectra match the applied filters."""
        db_path = _make_db(tmp_path)
        self._populate(db_path, _sample_records(2, database="metatlas"))
        with pytest.raises(ValueError, match="No spectra remained"):
            ldt.load_msms_refs_from_db(db_path, database_filter="nonexistent_db")

    def test_invalid_polarity_raises(self, tmp_path):
        """An unrecognised polarity string raises ValueError immediately."""
        db_path = _make_db(tmp_path)
        with pytest.raises(ValueError, match="Unrecognised polarity"):
            ldt.load_msms_refs_from_db(db_path, polarity="sideways")

    def test_grouped_by_inchi_key(self, tmp_path):
        """Multiple spectra for the same inchi_key are grouped under one key."""
        db_path = _make_db(tmp_path)
        key = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        self._populate(db_path, _sample_records(3, inchi_key=key))
        result = ldt.load_msms_refs_from_db(db_path)
        assert key in result
        assert len(result[key]) == 3

    def test_mz_ascending_order(self, tmp_path):
        """Returned Spectrum mz arrays are in ascending order."""
        db_path = _make_db(tmp_path)
        # Insert a record with deliberately unsorted mz
        rec = _sample_records(1)[0]
        rec["mz"] = [135.03, 59.01, 87.01]
        rec["intensities"] = [5000.0, 1000.0, 3000.0]
        dbi.batch_save_msms_refs(db_path, [rec])
        result = ldt.load_msms_refs_from_db(db_path)
        spec = list(result.values())[0][0]
        assert list(spec.mz) == sorted(spec.mz.tolist())


# ---------------------------------------------------------------------------
# NewMsmsRefsConfig
# ---------------------------------------------------------------------------

class TestNewMsmsRefsConfig:

    def _write_config(self, tmp_path: Path, jsonl_path: str,
                      database_filter: str | None = None) -> str:
        """Write a minimal MSMS_REFS YAML config and return its path."""
        entry = f"  - path: {jsonl_path}"
        if database_filter:
            entry += f"\n    database_filter: {database_filter}"
        yaml_text = f"MSMS_REFS:\n{entry}\n"
        config_path = str(tmp_path / "add_msms_refs.yaml")
        Path(config_path).write_text(yaml_text)
        return config_path

    def test_from_yaml_parses_entries(self, tmp_path):
        """from_yaml correctly parses a single-entry MSMS_REFS config."""
        jsonl_path = str(tmp_path / "refs.json")
        config_path = self._write_config(tmp_path, jsonl_path, database_filter="metatlas")
        cfg = NewMsmsRefsConfig.from_yaml(config_path)
        assert len(cfg.entries) == 1
        assert cfg.entries[0]["path"] == jsonl_path
        assert cfg.entries[0]["database_filter"] == "metatlas"

    def test_from_yaml_missing_key_raises(self, tmp_path):
        """from_yaml raises ValueError when MSMS_REFS key is absent."""
        config_path = str(tmp_path / "bad.yaml")
        Path(config_path).write_text("SOMETHING_ELSE:\n  - path: /tmp/x.json\n")
        with pytest.raises(ValueError, match="MSMS_REFS"):
            NewMsmsRefsConfig.from_yaml(config_path)

    def test_from_yaml_missing_path_raises(self, tmp_path):
        """from_yaml raises ValueError when an entry is missing the 'path' key."""
        config_path = str(tmp_path / "bad.yaml")
        Path(config_path).write_text("MSMS_REFS:\n  - database_filter: metatlas\n")
        with pytest.raises(ValueError, match="path"):
            NewMsmsRefsConfig.from_yaml(config_path)

    def test_execute_inserts_records(self, tmp_path, monkeypatch):
        """execute() streams the jsonl file and inserts rows into the DB."""
        db_path = _make_db(tmp_path)

        # Write a small JSONL file with 3 records
        jsonl_path = tmp_path / "refs.json"
        raw_records = [_jsonl_record(ref_id=f"ref_{i}") for i in range(3)]
        _write_jsonl(jsonl_path, raw_records)

        config_path = self._write_config(tmp_path, str(jsonl_path))

        # Patch set_up_paths so execute() uses our temp DB
        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_fragmentation_data"
            ).fetchone()[0]
        assert count == 3

    def test_execute_database_filter(self, tmp_path, monkeypatch):
        """execute() with database_filter only imports matching rows."""
        db_path = _make_db(tmp_path)

        jsonl_path = tmp_path / "refs.json"
        raw_records = [
            _jsonl_record(database="metatlas", ref_id="ref_0"),
            _jsonl_record(database="gnps", ref_id="ref_1"),
            _jsonl_record(database="metatlas", ref_id="ref_2"),
        ]
        _write_jsonl(jsonl_path, raw_records)

        config_path = self._write_config(
            tmp_path, str(jsonl_path), database_filter="metatlas"
        )

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_fragmentation_data"
            ).fetchone()[0]
        assert count == 2  # only the two "metatlas" rows

    def test_execute_precursor_truncation(self, tmp_path, monkeypatch):
        """execute() applies precursor_mz + 2.5 truncation before storing arrays."""
        db_path = _make_db(tmp_path)

        # precursor_mz = 100.0 → fragments above 102.5 should be dropped
        raw = _jsonl_record(precursor_mz=100.0)
        raw["mz"] = [50.0, 80.0, 100.0, 103.0, 110.0]  # last two should be dropped
        raw["intensities"] = [1.0, 2.0, 3.0, 4.0, 5.0]

        jsonl_path = tmp_path / "refs.json"
        _write_jsonl(jsonl_path, [raw])
        config_path = self._write_config(tmp_path, str(jsonl_path))

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT mz FROM reference_fragmentation_data LIMIT 1"
            ).fetchone()
        stored_mz = list(row[0])
        # Only fragments < 100.0 + 2.5 = 102.5 should be stored
        assert all(m < 102.5 for m in stored_mz)
        assert len(stored_mz) == 3  # 50.0, 80.0, 100.0

    def test_execute_missing_file_skips(self, tmp_path, monkeypatch):
        """execute() logs an error and skips a missing source file gracefully."""
        db_path = _make_db(tmp_path)
        config_path = self._write_config(
            tmp_path, str(tmp_path / "nonexistent.json")
        )

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        # Should not raise — missing file is logged and skipped
        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_fragmentation_data"
            ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# query_msms_refs (database_interact)
# ---------------------------------------------------------------------------

class TestQueryMsmsRefs:

    def _populate(self, db_path: str) -> None:
        key_a = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        key_b = "HDYANYHVCAPMJV-LXQIFKJMSA-N"
        dbi.batch_save_msms_refs(db_path, _sample_records(2, database="metatlas", polarity="positive", inchi_key=key_a))
        dbi.batch_save_msms_refs(db_path, _sample_records(1, database="gnps", polarity="negative", inchi_key=key_b))

    def test_no_filter_returns_all(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path)
        assert len(rows) == 3

    def test_returns_dicts_with_expected_keys(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path)
        assert len(rows) > 0
        row = rows[0]
        for key in ("ref_uid", "database", "inchi_key", "mz", "intensities", "polarity"):
            assert key in row

    def test_mz_intensities_are_lists(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path)
        for row in rows:
            assert isinstance(row["mz"], list)
            assert isinstance(row["intensities"], list)

    def test_database_filter(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path, database_filter="metatlas")
        assert len(rows) == 2
        assert all(r["database"] == "metatlas" for r in rows)

    def test_polarity_filter(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path, polarity="negative")
        assert len(rows) == 1
        assert rows[0]["polarity"] == "negative"

    def test_inchi_key_filter(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        key_a = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        rows = dbi.query_msms_refs(db_path, inchi_keys=[key_a])
        assert len(rows) == 2
        assert all(r["inchi_key"] == key_a for r in rows)

    def test_empty_inchi_keys_returns_empty(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path, inchi_keys=[])
        assert rows == []

    def test_no_match_returns_empty(self, tmp_path):
        db_path = _make_db(tmp_path)
        self._populate(db_path)
        rows = dbi.query_msms_refs(db_path, database_filter="nonexistent")
        assert rows == []


# ---------------------------------------------------------------------------
# get_msms_refs_from_db — export functions
# ---------------------------------------------------------------------------

class TestGetMsmsRefsFromDb:

    def _populate_and_get_rows(self, db_path: str) -> list[dict]:
        key_a = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        key_b = "HDYANYHVCAPMJV-LXQIFKJMSA-N"
        dbi.batch_save_msms_refs(db_path, _sample_records(2, database="metatlas", polarity="positive", inchi_key=key_a))
        dbi.batch_save_msms_refs(db_path, _sample_records(1, database="gnps", polarity="negative", inchi_key=key_b))
        return dbi.query_msms_refs(db_path)

    # -- JSONL export --------------------------------------------------------

    def test_jsonl_export_creates_file(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_jsonl
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.json"
        export_msms_refs_jsonl(rows, out)
        assert out.exists()

    def test_jsonl_export_line_count(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_jsonl
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.json"
        export_msms_refs_jsonl(rows, out)
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == len(rows)

    def test_jsonl_export_valid_json(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_jsonl
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.json"
        export_msms_refs_jsonl(rows, out)
        for line in out.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                assert "inchi_key" in rec
                assert isinstance(rec["mz"], list)
                assert isinstance(rec["intensities"], list)

    def test_jsonl_round_trip_mz(self, tmp_path):
        """mz values survive JSONL round-trip unchanged."""
        from metatlas2.get_msms_refs_from_db import export_msms_refs_jsonl
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.json"
        export_msms_refs_jsonl(rows, out)
        loaded = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        for orig, loaded_rec in zip(rows, loaded):
            assert orig["mz"] == pytest.approx(loaded_rec["mz"], rel=1e-5)

    # -- TSV export ----------------------------------------------------------

    def test_tsv_export_creates_file(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_tsv
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.tsv"
        export_msms_refs_tsv(rows, out)
        assert out.exists()

    def test_tsv_export_row_count(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_tsv
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.tsv"
        export_msms_refs_tsv(rows, out)
        lines = out.read_text().splitlines()
        # header + data rows
        assert len(lines) == len(rows) + 1

    def test_tsv_export_mz_as_json_array(self, tmp_path):
        """mz column in TSV is a JSON-serialised array string."""
        import csv as _csv
        from metatlas2.get_msms_refs_from_db import export_msms_refs_tsv
        db_path = _make_db(tmp_path)
        rows = self._populate_and_get_rows(db_path)
        out = tmp_path / "out.tsv"
        export_msms_refs_tsv(rows, out)
        with out.open() as fh:
            reader = _csv.DictReader(fh, delimiter="\t")
            for row in reader:
                mz_parsed = json.loads(row["mz"])
                assert isinstance(mz_parsed, list)

    def test_tsv_empty_rows_creates_empty_file(self, tmp_path):
        from metatlas2.get_msms_refs_from_db import export_msms_refs_tsv
        out = tmp_path / "empty.tsv"
        export_msms_refs_tsv([], out)
        assert out.exists()
        assert out.read_text() == ""

    # -- get_msms_refs (end-to-end) ------------------------------------------

    def test_get_msms_refs_jsonl(self, tmp_path, monkeypatch):
        """get_msms_refs() writes a JSONL file with the correct number of lines."""
        from metatlas2.get_msms_refs_from_db import get_msms_refs
        db_path = _make_db(tmp_path)
        self._populate_and_get_rows(db_path)
        out = str(tmp_path / "export.json")
        get_msms_refs(output_path=out, database_path=db_path)
        lines = [l for l in Path(out).read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_get_msms_refs_tsv(self, tmp_path):
        """get_msms_refs() with tab_out=True writes a TSV file."""
        from metatlas2.get_msms_refs_from_db import get_msms_refs
        db_path = _make_db(tmp_path)
        self._populate_and_get_rows(db_path)
        out = str(tmp_path / "export.tsv")
        get_msms_refs(output_path=out, tab_out=True, database_path=db_path)
        lines = Path(out).read_text().splitlines()
        assert len(lines) == 4  # header + 3 data rows

    def test_get_msms_refs_inchikeys_filter(self, tmp_path):
        """get_msms_refs() with inchikeys restricts output to matching rows."""
        from metatlas2.get_msms_refs_from_db import get_msms_refs
        db_path = _make_db(tmp_path)
        self._populate_and_get_rows(db_path)
        key_a = "JPIJQSOTBSSVTP-STHAYSLISA-N"
        out = str(tmp_path / "subset.json")
        get_msms_refs(output_path=out, inchikeys=[key_a], database_path=db_path)
        lines = [l for l in Path(out).read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["inchi_key"] == key_a

    def test_get_msms_refs_database_filter(self, tmp_path):
        """get_msms_refs() with database_filter restricts output."""
        from metatlas2.get_msms_refs_from_db import get_msms_refs
        db_path = _make_db(tmp_path)
        self._populate_and_get_rows(db_path)
        out = str(tmp_path / "metatlas_only.json")
        get_msms_refs(output_path=out, database_filter="metatlas", database_path=db_path)
        lines = [l for l in Path(out).read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["database"] == "metatlas"

    def test_get_msms_refs_no_match_no_file(self, tmp_path):
        """get_msms_refs() writes nothing when no rows match the filter."""
        from metatlas2.get_msms_refs_from_db import get_msms_refs
        db_path = _make_db(tmp_path)
        self._populate_and_get_rows(db_path)
        out = str(tmp_path / "nothing.json")
        get_msms_refs(output_path=out, database_filter="nonexistent", database_path=db_path)
        assert not Path(out).exists()
