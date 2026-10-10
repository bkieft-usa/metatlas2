"""Unit tests for the MSMS-refs database integration.

Covers:
* :func:`~metatlas2.database_interact.batch_save_msms_refs`
* :func:`~metatlas2.load_tools.load_msms_refs_from_db`
* :class:`~metatlas2.workflow_objects.NewMsmsRefsConfig` (YAML parsing + execute)

All tests use a temporary DuckDB file created via
:func:`~metatlas2.database_interact.create_metatlas_database` so the full
schema (including ``msms_refs``) is exercised without
touching any production database.
"""

from __future__ import annotations

import csv
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


# Column order for the metatlas .tab export format (spectrum column)
_TAB_COLUMNS = [
    "database", "id", "name", "spectrum", "decimal", "precursor_mz",
    "polarity", "adduct", "fragmentation_method", "collision_energy",
    "instrument", "instrument_type", "formula", "exact_mass",
    "inchi_key", "inchi", "smiles",
]


def _tab_record(
    database: str = "metatlas",
    polarity: str = "negative",
    inchi_key: str = "JPIJQSOTBSSVTP-STHAYSLISA-N",
    ref_id: str = "8257da2871be46cfabe10827aaf8d288",
    precursor_mz: float = 135.03,
    mz: list[float] | None = None,
    intensities: list[float] | None = None,
) -> dict:
    """Return a single .tab-format record dict (spectrum column) for use with _write_tab."""
    if mz is None:
        mz = [59.01, 71.01, 87.01, 117.02, 135.03]
    if intensities is None:
        intensities = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    spectrum_str = str([mz, intensities])
    return {
        "database": database,
        "id": ref_id,
        "name": "L-threonic acid",
        "spectrum": spectrum_str,
        "decimal": "4.0",
        "precursor_mz": str(precursor_mz),
        "polarity": polarity,
        "adduct": "",
        "fragmentation_method": "cid",
        "collision_energy": "",
        "instrument": "",
        "instrument_type": "",
        "formula": "C4H8O5",
        "exact_mass": "136.037",
        "inchi_key": inchi_key,
        "inchi": "",
        "smiles": "",
    }


def _write_tab(path: Path, records: list[dict]) -> None:
    """Write a list of .tab-format record dicts (spectrum column) to *path*."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TAB_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


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
                "SELECT COUNT(*) FROM msms_refs"
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
                "SELECT mz, intensities FROM msms_refs LIMIT 1"
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
                "SELECT ref_uid FROM msms_refs"
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
                "SELECT ref_uid FROM msms_refs LIMIT 1"
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
                "SELECT COUNT(*) FROM msms_refs"
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

    def _write_config(self, tmp_path: Path, tsv_path: str,
                      database_filter: str | None = None) -> str:
        """Write a minimal MSMS_REFS YAML config and return its path."""
        entry = f"  - path: {tsv_path}"
        if database_filter:
            entry += f"\n    database_filter: {database_filter}"
        yaml_text = f"MSMS_REFS:\n{entry}\n"
        config_path = str(tmp_path / "add_msms_refs.yaml")
        Path(config_path).write_text(yaml_text)
        return config_path

    def test_from_yaml_parses_entries(self, tmp_path):
        """from_yaml correctly parses a single-entry MSMS_REFS config."""
        tsv_path = str(tmp_path / "refs.tsv")
        config_path = self._write_config(tmp_path, tsv_path, database_filter="metatlas")
        cfg = NewMsmsRefsConfig.from_yaml(config_path)
        assert len(cfg.entries) == 1
        assert cfg.entries[0]["path"] == tsv_path
        assert cfg.entries[0]["database_filter"] == "metatlas"

    def test_from_yaml_missing_key_raises(self, tmp_path):
        """from_yaml raises ValueError when MSMS_REFS key is absent."""
        config_path = str(tmp_path / "bad.yaml")
        Path(config_path).write_text("SOMETHING_ELSE:\n  - path: /tmp/x.tsv\n")
        with pytest.raises(ValueError, match="MSMS_REFS"):
            NewMsmsRefsConfig.from_yaml(config_path)

    def test_from_yaml_missing_path_raises(self, tmp_path):
        """from_yaml raises ValueError when an entry is missing the 'path' key."""
        config_path = str(tmp_path / "bad.yaml")
        Path(config_path).write_text("MSMS_REFS:\n  - database_filter: metatlas\n")
        with pytest.raises(ValueError, match="path"):
            NewMsmsRefsConfig.from_yaml(config_path)

    def test_execute_inserts_records(self, tmp_path, monkeypatch):
        """execute() reads the .tab file and inserts all rows into the DB."""
        db_path = _make_db(tmp_path)

        tab_path = tmp_path / "refs.tab"
        raw_records = [_tab_record(ref_id=f"id_{i}") for i in range(3)]
        _write_tab(tab_path, raw_records)

        config_path = self._write_config(tmp_path, str(tab_path))

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM msms_refs"
            ).fetchone()[0]
        assert count == 3

    def test_execute_ref_id_and_mass_mapping(self, tmp_path, monkeypatch):
        """execute() maps 'id' → ref_id and 'exact_mass' → mono_isotopic_molecular_weight."""
        db_path = _make_db(tmp_path)

        tab_path = tmp_path / "refs.tab"
        _write_tab(tab_path, [_tab_record(ref_id="my_ref_id")])

        config_path = self._write_config(tmp_path, str(tab_path))

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT ref_id, mono_isotopic_molecular_weight FROM msms_refs LIMIT 1"
            ).fetchone()
        assert row[0] == "my_ref_id"
        assert row[1] == pytest.approx(136.037, rel=1e-4)

    def test_execute_mz_intensities_round_trip(self, tmp_path, monkeypatch):
        """execute() correctly parses mz and intensities from the spectrum column."""
        db_path = _make_db(tmp_path)

        mz_in = [59.01, 71.01, 87.01, 117.02, 135.03]
        int_in = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        tab_path = tmp_path / "refs.tab"
        _write_tab(tab_path, [_tab_record(mz=mz_in, intensities=int_in)])

        config_path = self._write_config(tmp_path, str(tab_path))

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT mz, intensities FROM msms_refs LIMIT 1"
            ).fetchone()
        assert list(row[0]) == pytest.approx(mz_in, rel=1e-4)
        assert list(row[1]) == pytest.approx(int_in, rel=1e-4)

    def test_execute_database_filter(self, tmp_path, monkeypatch):
        """execute() with database_filter only imports matching rows."""
        db_path = _make_db(tmp_path)

        tab_path = tmp_path / "refs.tab"
        raw_records = [
            _tab_record(database="metatlas", ref_id="id_0"),
            _tab_record(database="gnps", ref_id="id_1"),
            _tab_record(database="metatlas", ref_id="id_2"),
        ]
        _write_tab(tab_path, raw_records)

        config_path = self._write_config(
            tmp_path, str(tab_path), database_filter="metatlas"
        )

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM msms_refs"
            ).fetchone()[0]
        assert count == 2  # only the two "metatlas" rows

    def test_execute_precursor_truncation(self, tmp_path, monkeypatch):
        """execute() applies precursor_mz + 2.5 truncation before storing arrays."""
        db_path = _make_db(tmp_path)

        # precursor_mz = 100.0 → fragments >= 102.5 should be dropped
        tab_path = tmp_path / "refs.tab"
        _write_tab(tab_path, [_tab_record(
            precursor_mz=100.0,
            mz=[50.0, 80.0, 100.0, 103.0, 110.0],
            intensities=[1.0, 2.0, 3.0, 4.0, 5.0],
        )])

        config_path = self._write_config(tmp_path, str(tab_path))

        import metatlas2.run_targeted_analysis as rtg
        monkeypatch.setattr(
            rtg, "set_up_paths",
            lambda config, **kw: {"main_db_path": db_path},
        )

        NewMsmsRefsConfig.from_yaml(config_path).execute()

        with dbi.get_db_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT mz FROM msms_refs LIMIT 1"
            ).fetchone()
        stored_mz = list(row[0])
        # Only fragments < 100.0 + 2.5 = 102.5 should be stored
        assert all(m < 102.5 for m in stored_mz)
        assert len(stored_mz) == 3  # 50.0, 80.0, 100.0

    def test_execute_missing_file_skips(self, tmp_path, monkeypatch):
        """execute() logs an error and skips a missing source file gracefully."""
        db_path = _make_db(tmp_path)
        config_path = self._write_config(
            tmp_path, str(tmp_path / "nonexistent.tab")
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
                "SELECT COUNT(*) FROM msms_refs"
            ).fetchone()[0]
        assert count == 0

    def test_parse_spectrum_column_valid(self):
        """_parse_spectrum_column correctly parses a well-formed list-of-lists string."""
        mz, ints = NewMsmsRefsConfig._parse_spectrum_column(
            "[[59.01, 71.01, 87.01], [1000.0, 2000.0, 3000.0]]"
        )
        assert mz == pytest.approx([59.01, 71.01, 87.01])
        assert ints == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_parse_spectrum_column_null_returns_empty(self):
        """_parse_spectrum_column returns ([], []) for NULL / empty values."""
        assert NewMsmsRefsConfig._parse_spectrum_column("") == ([], [])
        assert NewMsmsRefsConfig._parse_spectrum_column("\\N") == ([], [])
        assert NewMsmsRefsConfig._parse_spectrum_column("None") == ([], [])

    def test_parse_spectrum_column_malformed_returns_empty(self):
        """_parse_spectrum_column returns ([], []) for unparseable strings."""
        assert NewMsmsRefsConfig._parse_spectrum_column("not a list") == ([], [])
        assert NewMsmsRefsConfig._parse_spectrum_column("[[1,2],[3]]") == ([], [])


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


