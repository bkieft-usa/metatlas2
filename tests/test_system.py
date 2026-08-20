"""System tests for the three main metatlas2.sh CLI routines.

These tests exercise the full Python-layer logic that the shell wrapper
``scripts/metatlas2.sh`` delegates to:

  1. ``add-compounds``  →  :mod:`metatlas2.add_compounds_to_db`
  2. ``add-atlases``    →  :mod:`metatlas2.add_atlases_to_db`
  3. ``run``            →  :mod:`metatlas2.run_targeted_analysis` (project-setup
                           and RT-alignment stages only; auto-ID is mocked so
                           no real HDF5 data or SLURM cluster is required)

Design principles
-----------------
* No real network calls (PubChem is patched to return canned data).
* No real HDF5 files beyond the synthetic fixture already in conftest.py.
* No Shifter / SLURM / container runtime required.
* All filesystem I/O is confined to pytest's ``tmp_path``.
* The ``METATLAS_DATA_DIR`` environment variable is monkey-patched to point
  at a temporary directory tree that mirrors the production layout.
* DuckDB databases are created fresh for every test.
"""

from __future__ import annotations

import os
import uuid
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from conftest import (
    write_synthetic_h5,
    ADENINE_MZ,
    ADENINE_RT,
    ADENINE_RMIN,
    ADENINE_RMAX,
    RIBOFLAVIN_MZ,
    RIBOFLAVIN_RT,
    RIBOFLAVIN_RMIN,
    RIBOFLAVIN_RMAX,
)

def _uid() -> str:
    return str(uuid.uuid4())

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Create a minimal METATLAS_DATA_DIR directory tree under tmp_path."""
    dirs = [
        "raw_data",
        "databases/main_db",
        "databases/pubchem_cache",
        "databases/modelseed_db",
        "projects/targeted_outputs",
        "projects/parquet_outputs",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture(autouse=False)
def metatlas_data_dir(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch METATLAS_DATA_DIR to point at the temporary data_dir."""
    monkeypatch.setenv("METATLAS_DATA_DIR", str(data_dir))
    return data_dir

@pytest.fixture()
def main_db_path(data_dir: Path) -> Path:
    """Return the path where the main DuckDB database should live."""
    return data_dir / "databases" / "main_db" / "metatlas.duckdb"


@pytest.fixture()
def pubchem_cache_path(data_dir: Path) -> Path:
    """Return the path for the PubChem JSON cache."""
    return data_dir / "databases" / "pubchem_cache" / "pubchem_global_cache.json"

def _write_compound_tsv(path: Path, rows: list[dict]) -> None:
    """Write a minimal compound TSV file that :func:`load_compound_input` can read."""

    _COMPOUND_TSV_HEADER = (
        "compound_name\tinchi_key\tadduct\trt_peak\trt_min\trt_max\t"
        "mz\tmz_tolerance\tchromatography\tpolarity\t"
        "mono_isotopic_molecular_weight\tformula\tsmiles\tinchi\t"
        "compound_uid\tcreated_by\tcreated_date\n"
    )
    
    with path.open("w") as fh:
        fh.write(_COMPOUND_TSV_HEADER)
        for r in rows:
            fh.write(
                f"{r.get('compound_name', 'Unknown')}\t"
                f"{r.get('inchi_key', '')}\t"
                f"{r.get('adduct', '[M+H]+')}\t"
                f"{r.get('rt_peak', 0.0)}\t"
                f"{r.get('rt_min', 0.0)}\t"
                f"{r.get('rt_max', 0.0)}\t"
                f"{r.get('mz', 0.0)}\t"
                f"{r.get('mz_tolerance', 5.0)}\t"
                f"{r.get('chromatography', 'HILICZ')}\t"
                f"{r.get('polarity', 'POS')}\t"
                f"{r.get('mono_isotopic_molecular_weight', 0.0)}\t"
                f"{r.get('formula', '')}\t"
                f"{r.get('smiles', '')}\t"
                f"{r.get('inchi', '')}\t"
                f"{r.get('compound_uid', _uid())}\t"
                f"{r.get('created_by', 'test')}\t"
                f"{r.get('created_date', '2025-01-01')}\n"
            )

@pytest.fixture()
def adenine_tsv(tmp_path: Path) -> Path:
    """A single-compound TSV file containing adenine."""
    p = tmp_path / "adenine_pos.tsv"
    _write_compound_tsv(p, [
        {
            "compound_name": "adenine",
            "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N",
            "adduct": "[M+H]+",
            "rt_peak": ADENINE_RT,
            "rt_min": ADENINE_RMIN,
            "rt_max": ADENINE_RMAX,
            "mz": ADENINE_MZ,
            "mz_tolerance": 5.0,
            "chromatography": "HILICZ",
            "polarity": "POS",
            "mono_isotopic_molecular_weight": 135.054,
            "formula": "C5H5N5",
            "smiles": "Nc1ncnc2[nH]cnc12",
            "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)",
        },
        {
            "compound_name": "riboflavin",
            "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N",
            "adduct": "[M+H]+",
            "rt_peak": RIBOFLAVIN_RT,
            "rt_min": RIBOFLAVIN_RMIN,
            "rt_max": RIBOFLAVIN_RMAX,
            "mz": RIBOFLAVIN_MZ,
            "mz_tolerance": 5.0,
            "chromatography": "HILICZ",
            "polarity": "POS",
            "mono_isotopic_molecular_weight": 376.138,
            "formula": "C17H20N4O6",
            "smiles": "",
            "inchi": "",
        },
    ])
    return p

@pytest.fixture()
def compounds_yaml(tmp_path: Path, adenine_tsv: Path) -> Path:
    """Write a minimal compounds config YAML pointing at the local TSV."""
    p = tmp_path / "test_compounds.yaml"
    p.write_text(textwrap.dedent(f"""\
        PARAMS:
          use_pubchem_cache: false
          update_pubchem_cache: false
        COMPOUNDS:
          HILICZ:
            POS:
              PATHS:
                - {adenine_tsv}
    """))
    return p

@pytest.fixture()
def atlas_tsv(tmp_path: Path) -> Path:
    """A minimal atlas TSV file (same schema as compound TSV)."""
    p = tmp_path / "test_atlas_pos.tsv"
    _write_compound_tsv(p, [
        {
            "compound_name": "adenine",
            "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N",
            "adduct": "[M+H]+",
            "rt_peak": ADENINE_RT,
            "rt_min": ADENINE_RMIN,
            "rt_max": ADENINE_RMAX,
            "mz": ADENINE_MZ,
            "mz_tolerance": 5.0,
            "chromatography": "HILICZ",
            "polarity": "POS",
            "mono_isotopic_molecular_weight": 135.054,
        },
        {
            "compound_name": "riboflavin",
            "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N",
            "adduct": "[M+H]+",
            "rt_peak": RIBOFLAVIN_RT,
            "rt_min": RIBOFLAVIN_RMIN,
            "rt_max": RIBOFLAVIN_RMAX,
            "mz": RIBOFLAVIN_MZ,
            "mz_tolerance": 5.0,
            "chromatography": "HILICZ",
            "polarity": "POS",
            "mono_isotopic_molecular_weight": 376.138,
        },
    ])
    return p

@pytest.fixture()
def atlases_yaml(tmp_path: Path, atlas_tsv: Path) -> Path:
    """Write a minimal atlases config YAML pointing at the local TSV."""
    p = tmp_path / "test_atlases.yaml"
    p.write_text(textwrap.dedent(f"""\
        ATLASES:
          HILICZ:
            POS:
              EMA:
                - path: {atlas_tsv}
                  name: Test HILICZ POS EMA Atlas
                  desc: Synthetic atlas for system tests
    """))
    return p

def _write_analysis_yaml(path: Path, atlas_uid: str) -> Path:
    """Write a minimal analysis config YAML using *atlas_uid* for both the QC and EMA atlas.

    Using the same UID for both the RT-alignment QC atlas and the EMA targeted-analysis
    atlas means the seeded DB only needs to contain one atlas, and no in-memory config
    mutation is required in tests.
    """
    path.write_text(textwrap.dedent(f"""\
        WORKFLOWS:
          PATHS:
            owner: jgi
            msms_refs_path:
            msms_refs_db_filter:
            gdrive_subfolder:
          RT_ALIGNMENT:
            HILICZ:
              ATLAS:
                uid: {atlas_uid}
              PARAMS:
                upload_to_gdrive: false
                include_lcmsruns:
                exclude_lcmsruns:
                  - NEG
                use_existing_rt_alignment: false
                remove_unided_compounds: false
                only_keep_data_in_feature: true
                atlas_extra_time: 2.0
                ms1_min_peak_intensity: 0
                ms1_min_num_points: 0
                ms1_mz_tolerance_ppm: 5.0
                apply_model_to_min_max: true
                polynomial_degree: 2
                min_observations_per_compound: 1
                min_compounds_for_modeling: 2
                r2_threshold: 0.5
                exclude_inchikeys: []
          TARGETED_ANALYSES:
            HILICZ:
              POS:
                EMA:
                  DEFAULT:
                    ATLAS:
                      uid: {atlas_uid}
                    PARAMS:
                      include_lcmsruns:
                      exclude_lcmsruns:
                        data_extraction:
                          - QC
                          - NEG
                      apply_alignment: true
                      remove_unided_compounds: true
                      remove_flagged_compounds: true
                      only_keep_data_in_feature: false
                      apply_cross_polarity_curation: true
                      suggested_min_conf: 0.75
                      atlas_extra_time: 0.5
                      ms1_min_peak_intensity: 1e5
                      ms1_min_num_points: 5
                      ms1_mz_tolerance_ppm: 5.0
                      ms2_min_num_scans: 1
                      ms2_min_precursor_intensity: 0
                      ms2_min_score: 0.25
                      ms2_min_matching_frags: 1
                      ms2_mz_tolerance_ppm: 20.0
                      ms2_frag_mz_tolerance: 0.05
                      gui_require_all_evaluated: false
                      gui_top_n_hits: 10
                      gui_lcmsruns_colors: {{}}
                      note_options_overrides: {{}}
                      create_curation_notebooks: false
                      upload_to_gdrive: false
                      skip_outputs:
    """))
    return path

@pytest.fixture()
def analysis_yaml(tmp_path: Path) -> tuple[Path, str]:
    """Write a minimal analysis config YAML with a placeholder UID.

    Returns (path, atlas_uid).  The same UID is used for both the RT-alignment
    QC atlas slot and the EMA targeted-analysis slot so that a single seeded
    atlas satisfies both.

    Tests that only exercise config parsing (no DB required) use this fixture.
    Tests that need the UIDs to match a real DB entry use :func:`seeded_analysis_yaml`.
    """
    placeholder_uid = f"atl-placeholder-{uuid.uuid4().hex[:32]}"
    p = tmp_path / "test_analysis.yaml"
    _write_analysis_yaml(p, placeholder_uid)
    return p, placeholder_uid

def _fake_pubchem_info(compounds: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return the input DataFrame unchanged (simulates a cache-hit with no enrichment)."""
    return compounds

@pytest.mark.system
class TestAddCompounds:
    """System tests for ``metatlas2.sh add-compounds``."""

    @pytest.fixture(autouse=True)
    def _patch_metatlas_data_dir(self, metatlas_data_dir: Path) -> None:
        """Ensure METATLAS_DATA_DIR is always patched for every test in this class."""

    def test_add_compounds_creates_main_database(
        self,
        compounds_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Running add_compounds_to_db should create the main DuckDB database file."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=True)

        assert main_db_path.exists(), "Main database was not created by add_compounds_to_db"

    def test_add_compounds_persists_compounds_to_db(
        self,
        compounds_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Compounds from the TSV should be queryable from the main database after ingestion."""
        import duckdb
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=True)

        conn = duckdb.connect(str(main_db_path), read_only=True)
        df = conn.execute("SELECT compound_name, inchi_key FROM compounds").df()
        conn.close()

        inchi_keys = set(df["inchi_key"].tolist())
        assert "GFFGJBXGBJISGV-UHFFFAOYSA-N" in inchi_keys, "Adenine not found in compounds table"
        assert "AUNGANRZJHBGPY-SCRDCRAPSA-N" in inchi_keys, "Riboflavin not found in compounds table"

    def test_add_compounds_correct_compound_count(
        self,
        compounds_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """The compounds table should contain exactly the number of rows in the input TSV."""
        import duckdb
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=True)

        conn = duckdb.connect(str(main_db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        conn.close()

        assert count == 2, f"Expected 2 compounds in DB, got {count}"

    def test_add_compounds_idempotent_on_rerun(
        self,
        compounds_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Running add_compounds_to_db twice (without overwrite) should not duplicate rows.

        Both calls use overwrite_db=False so the DB is never wiped between runs.
        This ensures the upsert logic (not the overwrite) is what keeps the count at 2.
        We run a third call for good measure and confirm the count is still 2.
        """
        import duckdb
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        # First call: create the DB from scratch
        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=True)

        count_after_first = duckdb.connect(str(main_db_path), read_only=True).execute(
            "SELECT COUNT(*) FROM compounds"
        ).fetchone()[0]
        assert count_after_first == 2, f"Expected 2 compounds after first run, got {count_after_first}"

        # Second call: same data, no overwrite — upsert must not duplicate.
        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=False)

        count_after_second = duckdb.connect(str(main_db_path), read_only=True).execute(
            "SELECT COUNT(*) FROM compounds"
        ).fetchone()[0]
        assert count_after_second == 2, f"Expected 2 compounds after second run (upsert), got {count_after_second}"

        # Third call: confirm idempotency holds across multiple reruns.
        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=False)

        count_after_third = duckdb.connect(str(main_db_path), read_only=True).execute(
            "SELECT COUNT(*) FROM compounds"
        ).fetchone()[0]
        assert count_after_third == 2, f"Expected 2 compounds after third run (upsert), got {count_after_third}"

    def test_add_compounds_missing_config_raises(
        self,
        tmp_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Passing a non-existent config path should raise an error."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        with pytest.raises(Exception):
            add_compounds_to_db(str(tmp_path / "does_not_exist.yaml"))

    def test_add_compounds_config_missing_params_key_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing the required PARAMS key should raise a ValueError through add_compounds_to_db."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        bad_yaml = tmp_path / "bad_compounds.yaml"
        bad_yaml.write_text("COMPOUNDS:\n  HILICZ:\n    POS:\n      PATHS: []\n")

        with pytest.raises(ValueError, match="PARAMS"):
            add_compounds_to_db(str(bad_yaml))

    def test_add_compounds_config_missing_compounds_key_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing the required COMPOUNDS key should raise a ValueError through add_compounds_to_db."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        bad_yaml = tmp_path / "bad_compounds2.yaml"
        bad_yaml.write_text("PARAMS:\n  use_pubchem_cache: false\n  update_pubchem_cache: false\n")

        with pytest.raises(ValueError, match="COMPOUNDS"):
            add_compounds_to_db(str(bad_yaml))

    def test_add_compounds_nonexistent_tsv_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML referencing a TSV that does not exist should raise FileNotFoundError through add_compounds_to_db."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        bad_yaml = tmp_path / "missing_tsv.yaml"
        bad_yaml.write_text(textwrap.dedent("""\
            PARAMS:
              use_pubchem_cache: false
              update_pubchem_cache: false
            COMPOUNDS:
              HILICZ:
                POS:
                  PATHS:
                    - /nonexistent/path/compounds.tsv
        """))

        with pytest.raises(FileNotFoundError):
            add_compounds_to_db(str(bad_yaml))

@pytest.mark.system
class TestAddAtlases:
    """System tests for ``metatlas2.sh add-atlases``."""

    @pytest.fixture(autouse=True)
    def _patch_metatlas_data_dir(self, metatlas_data_dir: Path) -> None:
        """Ensure METATLAS_DATA_DIR is always patched for every test in this class."""

    @pytest.fixture(autouse=True)
    def _seed_main_db(
        self,
        compounds_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Ensure the main DB is populated with compounds before atlas tests run."""
        from metatlas2.add_compounds_to_db import add_compounds_to_db

        with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
            add_compounds_to_db(str(compounds_yaml), overwrite_db=True)

    def test_add_atlases_creates_atlas_in_db(
        self,
        atlases_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Running add_atlases_to_db should insert at least one atlas into the main DB."""
        import duckdb
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        add_atlases_to_db(str(atlases_yaml))

        conn = duckdb.connect(str(main_db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
        conn.close()

        assert count >= 1, "No atlases were inserted into the main database"

    def test_add_atlases_atlas_has_correct_metadata(
        self,
        atlases_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """The inserted atlas should carry the name and description from the YAML."""
        import duckdb
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        add_atlases_to_db(str(atlases_yaml))

        conn = duckdb.connect(str(main_db_path), read_only=True)
        df = conn.execute(
            "SELECT atlas_name, atlas_description, chromatography, polarity, analysis_type "
            "FROM atlases"
        ).df()
        conn.close()

        from metatlas2.file_and_project_format import normalize_chromatography

        assert len(df) == 1
        row = df.iloc[0]
        assert row["atlas_name"] == "Test HILICZ POS EMA Atlas"
        assert row["atlas_description"] == "Synthetic atlas for system tests"
        assert normalize_chromatography(row["chromatography"]) == normalize_chromatography("HILICZ"), (
            f"Expected normalized chromatography '{normalize_chromatography('HILICZ')}', "
            f"got '{row['chromatography']}' (normalized: '{normalize_chromatography(row['chromatography'])}')"
        )
        assert row["polarity"].upper() == "POS"
        assert row["analysis_type"].upper() == "EMA"

    def test_add_atlases_compounds_associated_with_atlas(
        self,
        atlases_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """Each compound in the TSV should be associated with the new atlas."""
        import duckdb
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        add_atlases_to_db(str(atlases_yaml))

        conn = duckdb.connect(str(main_db_path), read_only=True)
        assoc_count = conn.execute(
            "SELECT COUNT(*) FROM atlas_compound_associations"
        ).fetchone()[0]
        conn.close()

        assert assoc_count == 2, (
            f"Expected 2 compound-atlas associations (adenine + riboflavin), got {assoc_count}"
        )

    def test_add_atlases_atlas_uid_is_retrievable(
        self,
        atlases_yaml: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """The atlas UID stored in the DB should allow round-trip retrieval via Atlas.from_database."""
        import duckdb
        from metatlas2.add_atlases_to_db import add_atlases_to_db
        from metatlas2.workflow_objects import Atlas

        add_atlases_to_db(str(atlases_yaml))

        conn = duckdb.connect(str(main_db_path), read_only=True)
        uid = conn.execute("SELECT atlas_uid FROM atlases LIMIT 1").fetchone()[0]
        conn.close()

        atlas = Atlas.from_database(str(main_db_path), uid)
        assert atlas.atlas_uid == uid
        assert len(atlas.compound_mzrts) == 2

    def test_add_atlases_config_missing_atlases_key_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing the top-level ATLASES key should raise a ValueError through add_atlases_to_db."""
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        bad_yaml = tmp_path / "bad_atlases.yaml"
        bad_yaml.write_text("SOMETHING_ELSE:\n  foo: bar\n")

        with pytest.raises(ValueError, match="ATLASES"):
            add_atlases_to_db(str(bad_yaml))

    def test_add_atlases_entry_missing_required_field_raises(
        self,
        tmp_path: Path,
        atlas_tsv: Path,
    ) -> None:
        """An atlas entry missing the 'name' field should raise a ValueError through add_atlases_to_db."""
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        bad_yaml = tmp_path / "missing_name.yaml"
        bad_yaml.write_text(textwrap.dedent(f"""\
            ATLASES:
              HILICZ:
                POS:
                  EMA:
                    - path: {atlas_tsv}
                      desc: Missing name field
        """))

        with pytest.raises(ValueError, match="name"):
            add_atlases_to_db(str(bad_yaml))

    def test_add_atlases_nonexistent_tsv_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """An atlas entry pointing at a non-existent TSV should raise FileNotFoundError through add_atlases_to_db."""
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        bad_yaml = tmp_path / "missing_atlas_tsv.yaml"
        bad_yaml.write_text(textwrap.dedent("""\
            ATLASES:
              HILICZ:
                POS:
                  EMA:
                    - path: /nonexistent/atlas.tsv
                      name: Ghost Atlas
                      desc: Should not be created
        """))

        with pytest.raises(FileNotFoundError):
            add_atlases_to_db(str(bad_yaml))

    def test_add_atlases_empty_path_entry_is_skipped(
        self,
        tmp_path: Path,
        main_db_path: Path,
        metatlas_data_dir: Path,
    ) -> None:
        """An atlas entry with an empty path should be silently skipped (not raise)."""
        import duckdb
        from metatlas2.add_atlases_to_db import add_atlases_to_db

        # Write a YAML where path is null/empty — mirrors the commented-out
        # entries in the production jgi_production_atlases.yaml.
        empty_yaml = tmp_path / "empty_path_atlases.yaml"
        empty_yaml.write_text(textwrap.dedent("""\
            ATLASES:
              HILICZ:
                NEG:
                  QC:
                    - path:
                      name:
                      desc:
        """))

        # Should not raise; the empty entry is skipped.
        add_atlases_to_db(str(empty_yaml))

        conn = duckdb.connect(str(main_db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
        conn.close()

        assert count == 0, "Empty-path atlas entry should produce no DB rows"



VALID_PROJECT_NAME = "20250101_JGI_Smith_12345_TestProject_Experiment1_Instrument1_HILICZ_Run001"

@pytest.fixture()
def project_raw_data_dir(data_dir: Path) -> Path:
    """Create the raw-data directory that set_up_paths expects for the project."""
    raw_dir = data_dir / "raw_data" / "jgi" / VALID_PROJECT_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

@pytest.fixture()
def project_with_h5_files(
    project_raw_data_dir: Path,
    pos_atlas,
) -> Path:
    """Populate the project raw-data directory with synthetic HDF5 files."""
    # Write two synthetic HDF5 files that look like real LCMS runs.
    for run_num in (1, 2):
        h5_name = (
            f"20250101_JGI_Smith_12345_TestProject_Experiment1_Instrument1_"
            f"HILICZ_Run00{run_num}_POS_MS1_S001_QC_rep1_meta_Run{run_num:03d}.h5"
        )
        h5_path = project_raw_data_dir / h5_name
        from conftest import ADENINE_MZ, ADENINE_RT, RIBOFLAVIN_MZ, RIBOFLAVIN_RT
        ms1_pos = []
        for rt_offset in [-0.1, 0.0, 0.1]:
            ms1_pos.append({"mz": ADENINE_MZ + 0.0001, "rt": ADENINE_RT + rt_offset, "i": 1e5})
            ms1_pos.append({"mz": RIBOFLAVIN_MZ + 0.0001, "rt": RIBOFLAVIN_RT + rt_offset, "i": 5e4})
        write_synthetic_h5(h5_path, ms1_pos_rows=ms1_pos)
    return project_raw_data_dir

@pytest.fixture()
def seeded_main_db(
    compounds_yaml: Path,
    atlases_yaml: Path,
    main_db_path: Path,
    metatlas_data_dir: Path,
) -> tuple[Path, str]:
    """Seed the main DB with compounds and one atlas; return (main_db_path, atlas_uid)."""
    import duckdb
    from metatlas2.add_compounds_to_db import add_compounds_to_db
    from metatlas2.add_atlases_to_db import add_atlases_to_db

    with patch("metatlas2.pubchem_retrieval.retrieve_pubchem_info", side_effect=_fake_pubchem_info):
        add_compounds_to_db(str(compounds_yaml), overwrite_db=True)
    add_atlases_to_db(str(atlases_yaml))

    conn = duckdb.connect(str(main_db_path), read_only=True)
    uid = conn.execute("SELECT atlas_uid FROM atlases LIMIT 1").fetchone()[0]
    conn.close()

    return main_db_path, uid

@pytest.fixture()
def seeded_analysis_yaml(
    tmp_path: Path,
    seeded_main_db: tuple[Path, str],
) -> tuple[Path, str]:
    """Write an analysis YAML whose atlas UIDs match the atlas already in the seeded DB.

    Returns (yaml_path, atlas_uid).  Using the real atlas UID in the YAML means
    tests that call ``run_project_setup`` followed by ``run_rt_alignment`` do not
    need to mutate the config object in-memory after loading it.
    """
    _, atlas_uid = seeded_main_db
    p = tmp_path / "test_analysis_seeded.yaml"
    _write_analysis_yaml(p, atlas_uid)
    return p, atlas_uid

@pytest.fixture()
def project_config_and_paths(
    seeded_analysis_yaml: tuple[Path, str],
    metatlas_data_dir: Path,
    project_raw_data_dir: Path,
) -> tuple[Any, dict]:
    """Return (config, paths) ready for use in TestRunTargetedAnalysis tests.

    Calls :func:`load_metatlas2_config` and :func:`set_up_paths` once so that
    individual tests do not need to repeat this boilerplate.  Depends on
    ``project_raw_data_dir`` so the raw-data directory already exists when
    ``set_up_paths`` validates it.
    """
    from metatlas2.load_tools import load_metatlas2_config
    from metatlas2.run_targeted_analysis import set_up_paths

    yaml_path, _ = seeded_analysis_yaml
    config = load_metatlas2_config(str(yaml_path))
    paths = set_up_paths(
        config,
        project_name=VALID_PROJECT_NAME,
        rt_alignment_number=0,
        analysis_number=0,
    )
    return config, paths

@pytest.mark.system
class TestRunTargetedAnalysis:
    """System tests for ``metatlas2.sh run`` (project-setup and path-resolution stages)."""

    @pytest.fixture(autouse=True)
    def _patch_metatlas_data_dir(self, metatlas_data_dir: Path) -> None:
        """Ensure METATLAS_DATA_DIR is always patched for every test in this class.

        Tests that need METATLAS_DATA_DIR *unset* (e.g. test_set_up_paths_raises_without_env_var)
        must explicitly call ``monkeypatch.delenv("METATLAS_DATA_DIR")`` after this fixture runs.
        """

    def test_set_up_paths_raises_without_env_var(
        self,
        analysis_yaml: tuple[Path, str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """set_up_paths should raise EnvironmentError when METATLAS_DATA_DIR is unset."""
        from metatlas2.run_targeted_analysis import set_up_paths
        from metatlas2.load_tools import load_metatlas2_config

        monkeypatch.delenv("METATLAS_DATA_DIR", raising=False)
        config_path, _ = analysis_yaml
        config = load_metatlas2_config(str(config_path))

        with pytest.raises(EnvironmentError, match="METATLAS_DATA_DIR"):
            set_up_paths(config, project_name=VALID_PROJECT_NAME, rt_alignment_number=0, analysis_number=0)

    def test_set_up_paths_raises_when_raw_data_missing(
        self,
        analysis_yaml: tuple[Path, str, str],
    ) -> None:
        """set_up_paths should raise ValueError when the project raw-data directory is absent."""
        from metatlas2.run_targeted_analysis import set_up_paths
        from metatlas2.load_tools import load_metatlas2_config

        config_path, _ = analysis_yaml
        config = load_metatlas2_config(str(config_path))

        # Raw data directory does NOT exist yet — no project_raw_data_dir fixture.
        with pytest.raises(ValueError, match="Raw data directory not found"):
            set_up_paths(config, project_name=VALID_PROJECT_NAME, rt_alignment_number=0, analysis_number=0)

    def test_set_up_paths_creates_output_directories(
        self,
        project_config_and_paths: tuple[Any, dict],
    ) -> None:
        """set_up_paths should create all required output directories."""
        _, paths = project_config_and_paths

        assert Path(paths["project_directory"]).exists(), "project_directory was not created"
        assert Path(paths["rt_alignment_output_dir"]).exists(), "rt_alignment_output_dir was not created"
        assert Path(paths["analysis_output_dir"]).exists(), "analysis_output_dir was not created"
        assert Path(paths["rt_alignment_results_dir"]).exists(), "rt_alignment_results_dir was not created"

    def test_set_up_paths_returns_correct_db_paths(
        self,
        project_config_and_paths: tuple[Any, dict],
        seeded_main_db: tuple[Path, str],
    ) -> None:
        """set_up_paths should embed the correct main_db_path and project_db_path."""
        _, paths = project_config_and_paths
        main_db, _ = seeded_main_db

        assert paths["main_db_path"] == str(main_db)
        assert VALID_PROJECT_NAME in paths["project_db_path"]
        assert paths["project_db_path"].endswith(".duckdb")

    def test_load_metatlas2_config_parses_correctly(
        self,
        seeded_analysis_yaml: tuple[Path, str],
    ) -> None:
        """load_metatlas2_config should parse the YAML into a Metatlas2Config object."""
        from metatlas2.load_tools import load_metatlas2_config
        from metatlas2.file_and_project_format import normalize_chromatography

        config_path, atlas_uid = seeded_analysis_yaml
        config = load_metatlas2_config(str(config_path))

        expected_chrom_key = normalize_chromatography("HILICZ")
        assert config.owner == "jgi"
        assert expected_chrom_key in config.rt_alignment_config, (
            f"Expected chromatography key '{expected_chrom_key}' in rt_alignment_config, "
            f"got keys: {list(config.rt_alignment_config)}"
        )
        # Verify the RT-alignment config block has the required structure.
        rta_block = config.rt_alignment_config[expected_chrom_key]
        assert "ATLAS" in rta_block, "RT_ALIGNMENT block missing ATLAS section"
        assert "uid" in rta_block["ATLAS"], "RT_ALIGNMENT ATLAS block missing uid field"
        assert rta_block["ATLAS"]["uid"] == atlas_uid
        assert "PARAMS" in rta_block, "RT_ALIGNMENT block missing PARAMS section"

        assert len(config.targeted_analyses) == 1
        ta = config.targeted_analyses[0]
        assert ta.atlas_uid == atlas_uid
        assert ta.polarity == "POS"
        assert ta.analysis_type == "EMA"
        assert ta.analysis_name == "DEFAULT"

    def test_load_metatlas2_config_missing_workflows_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing the WORKFLOWS key should raise a ValueError."""
        from metatlas2.load_tools import load_metatlas2_config

        bad = tmp_path / "no_workflows.yaml"
        bad.write_text("PATHS:\n  owner: jgi\n")

        with pytest.raises(ValueError, match="WORKFLOWS"):
            load_metatlas2_config(str(bad))

    def test_load_metatlas2_config_missing_rt_alignment_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing WORKFLOWS.RT_ALIGNMENT should raise a ValueError."""
        from metatlas2.load_tools import load_metatlas2_config

        bad = tmp_path / "no_rt_alignment.yaml"
        bad.write_text(textwrap.dedent("""\
            WORKFLOWS:
              PATHS:
                owner: jgi
              TARGETED_ANALYSES:
                HILICZ:
                  POS:
                    EMA:
                      DEFAULT:
                        ATLAS:
                          uid: some-uid
                        PARAMS: {}
        """))

        with pytest.raises(ValueError, match="RT_ALIGNMENT"):
            load_metatlas2_config(str(bad))

    def test_load_metatlas2_config_missing_targeted_analyses_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A YAML missing WORKFLOWS.TARGETED_ANALYSES should raise a ValueError."""
        from metatlas2.load_tools import load_metatlas2_config

        bad = tmp_path / "no_targeted_analyses.yaml"
        bad.write_text(textwrap.dedent("""\
            WORKFLOWS:
              PATHS:
                owner: jgi
              RT_ALIGNMENT:
                HILICZ:
                  ATLAS:
                    uid: some-uid
                  PARAMS: {}
        """))

        with pytest.raises(ValueError, match="TARGETED_ANALYSES"):
            load_metatlas2_config(str(bad))

    def test_run_project_setup_creates_project_database(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
    ) -> None:
        """run_project_setup should create the project DuckDB file on disk."""
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        assert Path(paths["project_db_path"]).exists(), "Project database was not created"

    def test_run_project_setup_registers_project_in_main_db(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
        seeded_main_db: tuple[Path, str],
    ) -> None:
        """run_project_setup should register the project in the main database projects table."""
        import duckdb
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        main_db, _ = seeded_main_db

        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        conn = duckdb.connect(str(main_db), read_only=True)
        df = conn.execute(
            "SELECT project_name FROM projects WHERE project_name = ?",
            [VALID_PROJECT_NAME],
        ).df()
        conn.close()

        assert len(df) == 1, "Project was not registered in the main database"

    def test_run_project_setup_saves_lcmsruns_to_project_db(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
    ) -> None:
        """run_project_setup should discover and persist LCMS runs from the raw-data directory."""
        import duckdb
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        conn = duckdb.connect(str(paths["project_db_path"]), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0]
        conn.close()

        assert count == 2, f"Expected 2 LCMS runs in project DB, got {count}"

    def test_run_project_setup_saves_config_snapshot(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
    ) -> None:
        """run_project_setup should persist the config snapshot to the project DB."""
        import duckdb
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        conn = duckdb.connect(str(paths["project_db_path"]), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM project_config").fetchone()[0]
        conn.close()

        assert count >= 1, "Config snapshot was not saved to project_config table"

    def test_run_project_setup_idempotent_when_db_exists(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
        seeded_main_db: tuple[Path, str],
    ) -> None:
        """Calling run_project_setup twice without overwrite should not raise or duplicate DB rows."""
        import duckdb
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        main_db, _ = seeded_main_db

        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        # Capture DB state after the first call.
        proj_conn = duckdb.connect(str(paths["project_db_path"]), read_only=True)
        lcms_count_1 = proj_conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0]
        config_count_1 = proj_conn.execute("SELECT COUNT(*) FROM project_config").fetchone()[0]
        proj_conn.close()

        main_conn = duckdb.connect(str(main_db), read_only=True)
        project_rows_1 = main_conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = ?", [VALID_PROJECT_NAME]
        ).fetchone()[0]
        main_conn.close()

        # Second call — should not raise even though DB already exists.
        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=False,
            rt_alignment_number=0,
            analysis_number=0,
        )

        # DB state must be unchanged: no duplicate LCMS runs, projects, or config rows.
        proj_conn = duckdb.connect(str(paths["project_db_path"]), read_only=True)
        lcms_count_2 = proj_conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0]
        config_count_2 = proj_conn.execute("SELECT COUNT(*) FROM project_config").fetchone()[0]
        proj_conn.close()

        main_conn = duckdb.connect(str(main_db), read_only=True)
        project_rows_2 = main_conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = ?", [VALID_PROJECT_NAME]
        ).fetchone()[0]
        main_conn.close()

        assert lcms_count_2 == lcms_count_1, (
            f"LCMS run count changed after idempotent re-run: {lcms_count_1} -> {lcms_count_2}"
        )
        assert config_count_2 >= config_count_1, (
            "project_config row count decreased after second call"
        )
        assert project_rows_2 == project_rows_1 == 1, (
            f"Expected exactly 1 project row in main DB, got {project_rows_2} after second call"
        )

    def test_get_project_db_path_finds_existing_db(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
    ) -> None:
        """get_project_db_path should locate the project DB after project setup."""
        from metatlas2.run_targeted_analysis import get_project_db_path
        from metatlas2.workflows import run_project_setup

        config, paths = project_config_and_paths
        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        found = get_project_db_path(VALID_PROJECT_NAME)
        assert found == paths["project_db_path"]

    def test_get_project_db_path_raises_when_missing(
        self,
        metatlas_data_dir: Path,
    ) -> None:
        """get_project_db_path should raise FileNotFoundError for an unknown project."""
        from metatlas2.run_targeted_analysis import get_project_db_path

        with pytest.raises(FileNotFoundError):
            get_project_db_path("20990101_JGI_Ghost_99999_NoProject_Exp_Inst_HILICZ_Run001")

    def test_run_rt_alignment_skip_flag_registers_atlases(
        self,
        project_config_and_paths: tuple[Any, dict],
        project_with_h5_files: Path,
    ) -> None:
        """With --skip-rt-align the RT_ALIGNED atlases should be registered without running the model.

        The ``seeded_analysis_yaml`` fixture writes the YAML with the real atlas UID
        already embedded, so ``project_config_and_paths`` loads a config that already
        points at the seeded atlas.  No in-memory config mutation is required.
        """
        import duckdb
        from metatlas2.workflows import run_project_setup, run_rt_alignment

        config, paths = project_config_and_paths

        run_project_setup(
            project_name=VALID_PROJECT_NAME,
            config=config,
            paths=paths,
            overwrite_existing=True,
            rt_alignment_number=0,
            analysis_number=0,
        )

        run_rt_alignment(
            project_name=VALID_PROJECT_NAME,
            rt_alignment_number=0,
            analysis_number=0,
            skip_alignment=True,
        )

        conn = duckdb.connect(str(paths["project_db_path"]), read_only=True)
        count = conn.execute(
            "SELECT COUNT(*) FROM workflow_runs WHERE stage = 'RT_ALIGNED'"
        ).fetchone()[0]
        conn.close()

        assert count >= 1, "No RT_ALIGNED workflow_run rows were created with --skip-rt-align"

    def test_generate_slurm_script_writes_file(
        self,
        tmp_path: Path,
        project_config_and_paths: tuple[Any, dict],
        seeded_analysis_yaml: tuple[Path, str],
        monkeypatch: pytest.MonkeyPatch,
        metatlas_data_dir: Path,
    ) -> None:
        """generate_slurm_script should write a valid bash script to disk."""
        import argparse
        from metatlas2.run_targeted_analysis import generate_slurm_script

        config, paths = project_config_and_paths
        config_path, _ = seeded_analysis_yaml
        out_script = tmp_path / "test_slurm.sh"

        args = argparse.Namespace(
            account="m2650",
            qos="regular",
            constraint="cpu",
            cpus=8,
            mem="128G",
            time="00:30:00",
            config=str(config_path),
            project=VALID_PROJECT_NAME,
            rt_align_num=0,
            analysis_num=0,
            overwrite=False,
            skip_rt_align=False,
            skip_curation=False,
            analysis_subset=None,
            image="latest",
            output=str(out_script),
        )

        monkeypatch.setenv("METATLAS_DATA_DIR", str(metatlas_data_dir))
        monkeypatch.setenv("HOME", str(tmp_path))

        written_path = generate_slurm_script(args, paths)

        assert Path(written_path).exists(), "SLURM script was not written to disk"
        content = Path(written_path).read_text()
        assert "#!/bin/bash" in content
        assert "#SBATCH" in content
        assert VALID_PROJECT_NAME in content
        assert "metatlas2.run_targeted_analysis" in content
        assert paths["project_directory"] in content, (
            "#SBATCH --output should reference paths['project_directory']; "
            f"expected '{paths['project_directory']}' to appear in the script"
        )