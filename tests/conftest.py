"""Shared pytest fixtures for metatlas2 unit tests.

* No real HDF5 files, databases, or network calls are needed.
* Synthetic HDF5 files are written to pytest's ``tmp_path`` using PyTables
  (the same library the production code uses) so the read path is exercised
  end-to-end without any mocking of I/O.
* Atlas / LCMSRun objects are built directly from their dataclasses so tests
  are not coupled to the database layer.
* The ``enrich_atlas_df_with_compound_metadata`` DB call inside
  ``create_manual_curation_obj`` is patched at the module level so that
  curation tests never touch DuckDB.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tables

from metatlas2.workflow_objects import Atlas, CompoundMZRT, ExperimentalData, LCMSRun

def _uid() -> str:
    return str(uuid.uuid4())

def _make_compound_mzrt(
    *,
    mz: float,
    rt_peak: float,
    rt_min: float,
    rt_max: float,
    adduct: str = "[M+H]+",
    inchi_key: str = "GFFGJBXGBJISGV-UHFFFAOYSA-N",
    compound_name: str = "adenine",
    polarity: str = "POS",
    chromatography: str = "HILICZ",
    mz_tolerance: float = 5.0,
    mono_isotopic_molecular_weight: float = 135.054,
) -> CompoundMZRT:
    """Return a minimal :class:`CompoundMZRT` suitable for atlas construction."""
    return CompoundMZRT(
        mz_rt_uid=_uid(),
        compound_uid=_uid(),
        compound_name=compound_name,
        inchi_key=inchi_key,
        adduct=adduct,
        rt_peak=rt_peak,
        rt_min=rt_min,
        rt_max=rt_max,
        mz=mz,
        mz_tolerance=mz_tolerance,
        chromatography=chromatography,
        polarity=polarity,
    )

def _make_atlas(
    compounds: list[CompoundMZRT],
    *,
    polarity: str = "POS",
    chromatography: str = "HILICZ",
    analysis_type: str = "EMA",
) -> Atlas:
    """Wrap a list of :class:`CompoundMZRT` objects into an :class:`Atlas`."""
    compound_mzrts = {c.mz_rt_uid: c for c in compounds}
    return Atlas(
        atlas_uid=_uid(),
        atlas_name="Test Atlas",
        atlas_description="Synthetic atlas for unit tests",
        chromatography=chromatography,
        polarity=polarity,
        analysis_type=analysis_type,
        compound_mzrts=compound_mzrts,
    )

class _MS1Row(tables.IsDescription):
    mz = tables.Float64Col()
    rt = tables.Float64Col()
    i  = tables.Float64Col()

class _MS2Row(tables.IsDescription):
    mz                  = tables.Float64Col()
    i                   = tables.Float64Col()
    rt                  = tables.Float64Col()
    precursor_MZ        = tables.Float64Col()
    precursor_intensity = tables.Float64Col()
    collision_energy    = tables.Float64Col()

def write_synthetic_h5(
    path: Path,
    *,
    ms1_pos_rows: list[dict[str, float]] | None = None,
    ms2_pos_rows: list[dict[str, float]] | None = None,
    ms1_neg_rows: list[dict[str, float]] | None = None,
    ms2_neg_rows: list[dict[str, float]] | None = None,
) -> None:
    """Write a minimal HDF5 file that the production ``_load_h5_table`` can read.

    Both the base table (e.g. ``ms1_pos``) and the m/z-sorted variant
    (``ms1_pos_mz``) are written so that the ``mz_bounds`` fast-path in
    :func:`metatlas2.extract_data_from_h5._load_h5_table` works correctly.

    Args:
        path:          Destination file path (must not already exist).
        ms1_pos_rows:  List of ``{"mz": …, "rt": …, "i": …}`` dicts.
        ms2_pos_rows:  List of ``{"mz": …, "i": …, "rt": …,
                       "precursor_MZ": …, "precursor_intensity": …,
                       "collision_energy": …}`` dicts.
        ms1_neg_rows:  Same schema as *ms1_pos_rows* for negative polarity.
        ms2_neg_rows:  Same schema as *ms2_pos_rows* for negative polarity.
    """
    with tables.open_file(str(path), mode="w") as h5:
        filters = tables.Filters(complevel=1, complib="blosc")

        def _write_ms1(key: str, rows: list[dict[str, float]]) -> None:
            tbl = h5.create_table("/", key, _MS1Row, filters=filters)
            row = tbl.row
            for r in rows:
                row["mz"] = r["mz"]
                row["rt"] = r["rt"]
                row["i"]  = r["i"]
                row.append()
            tbl.flush()
            # Write the mz-sorted variant (required by _load_h5_table mz_bounds path)
            sorted_rows = sorted(rows, key=lambda x: x["mz"])
            tbl_mz = h5.create_table("/", key + "_mz", _MS1Row, filters=filters)
            row_mz = tbl_mz.row
            for r in sorted_rows:
                row_mz["mz"] = r["mz"]
                row_mz["rt"] = r["rt"]
                row_mz["i"]  = r["i"]
                row_mz.append()
            tbl_mz.flush()

        def _write_ms2(key: str, rows: list[dict[str, float]]) -> None:
            tbl = h5.create_table("/", key, _MS2Row, filters=filters)
            row = tbl.row
            for r in rows:
                row["mz"]                  = r.get("mz", 0.0)
                row["i"]                   = r.get("i", 0.0)
                row["rt"]                  = r.get("rt", 0.0)
                row["precursor_MZ"]        = r.get("precursor_MZ", 0.0)
                row["precursor_intensity"] = r.get("precursor_intensity", 0.0)
                row["collision_energy"]    = r.get("collision_energy", 0.0)
                row.append()
            tbl.flush()

        if ms1_pos_rows:
            _write_ms1("ms1_pos", ms1_pos_rows)
        if ms2_pos_rows:
            _write_ms2("ms2_pos", ms2_pos_rows)
        if ms1_neg_rows:
            _write_ms1("ms1_neg", ms1_neg_rows)
        if ms2_neg_rows:
            _write_ms2("ms2_neg", ms2_neg_rows)

#### Compounds

# Adenine [M+H]+  mz=136.062  rt_peak=2.68 min
ADENINE_MZ = 136.062
ADENINE_RT = 2.68
ADENINE_RMIN = 1.93
ADENINE_RMAX = 3.43

# Riboflavin [M+H]+  mz=377.146  rt_peak=4.56 min
RIBOFLAVIN_MZ = 377.146
RIBOFLAVIN_RT = 4.56
RIBOFLAVIN_RMIN = 3.81
RIBOFLAVIN_RMAX = 5.31

#### Fixtures

@pytest.fixture()
def adenine_compound() -> CompoundMZRT:
    return _make_compound_mzrt(
        mz=ADENINE_MZ,
        rt_peak=ADENINE_RT,
        rt_min=ADENINE_RMIN,
        rt_max=ADENINE_RMAX,
        compound_name="adenine",
        inchi_key="GFFGJBXGBJISGV-UHFFFAOYSA-N",
        adduct="[M+H]+",
    )

@pytest.fixture()
def riboflavin_compound() -> CompoundMZRT:
    return _make_compound_mzrt(
        mz=RIBOFLAVIN_MZ,
        rt_peak=RIBOFLAVIN_RT,
        rt_min=RIBOFLAVIN_RMIN,
        rt_max=RIBOFLAVIN_RMAX,
        compound_name="riboflavin",
        inchi_key="AUNGANRZJHBGPY-SCRDCRAPSA-N",
        adduct="[M+H]+",
        mono_isotopic_molecular_weight=376.138,
    )

@pytest.fixture()
def pos_atlas(adenine_compound, riboflavin_compound) -> Atlas:
    """Two-compound positive-mode atlas."""
    return _make_atlas([adenine_compound, riboflavin_compound], polarity="POS")

@pytest.fixture()
def single_compound_atlas(adenine_compound) -> Atlas:
    """Single-compound atlas (adenine only)."""
    return _make_atlas([adenine_compound], polarity="POS")


@pytest.fixture()
def c18_pos_atlas(adenine_compound, riboflavin_compound) -> Atlas:
    """Two-compound C18-EP positive-mode atlas (mirrors the HILICZ atlas but with C18-EP chromatography)."""
    return _make_atlas(
        [adenine_compound, riboflavin_compound],
        polarity="POS",
        chromatography="C18-EP",
    )

@pytest.fixture()
def synthetic_h5_file(tmp_path: Path, pos_atlas: Atlas) -> Path:
    """Write a tiny HDF5 file containing MS1 and MS2 data for both atlas compounds.

    The scan points are placed squarely inside each compound's RT window so
    that ``in_feature`` is ``True`` for all of them.
    """
    h5_path = tmp_path / "test_run_POS_Run001.h5"

    # MS1 positive: a handful of points for adenine and riboflavin
    ms1_pos = []
    # Adenine: 5 points centred on rt_peak=2.68, mz=136.062
    for rt_offset in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        ms1_pos.append({"mz": ADENINE_MZ + 0.0001, "rt": ADENINE_RT + rt_offset, "i": 1e5})
    # Riboflavin: 5 points centred on rt_peak=4.56, mz=377.146
    for rt_offset in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        ms1_pos.append({"mz": RIBOFLAVIN_MZ + 0.0001, "rt": RIBOFLAVIN_RT + rt_offset, "i": 5e4})

    # MS2 positive: one scan per compound
    ms2_pos = [
        {
            "mz": 94.04, "i": 1e4, "rt": ADENINE_RT,
            "precursor_MZ": ADENINE_MZ, "precursor_intensity": 1e5, "collision_energy": 35.0,
        },
        {
            "mz": 243.09, "i": 8e3, "rt": RIBOFLAVIN_RT,
            "precursor_MZ": RIBOFLAVIN_MZ, "precursor_intensity": 5e4, "collision_energy": 35.0,
        },
    ]

    write_synthetic_h5(h5_path, ms1_pos_rows=ms1_pos, ms2_pos_rows=ms2_pos)
    return h5_path

@pytest.fixture()
def lcmsrun(synthetic_h5_file: Path) -> LCMSRun:
    """A single :class:`LCMSRun` pointing at the synthetic HDF5 file."""
    return LCMSRun(
        file_path=str(synthetic_h5_file),
        filename=synthetic_h5_file.name,
        file_format="h5",
        file_type="ms_data",
        chromatography="HILICZ",
        ms_level="MS1+MS2",
        polarity="POS",
        created_by="test",
        created_date="2025-01-01",
    )

@pytest.fixture()
def ms1_wide_df(adenine_compound: CompoundMZRT) -> pd.DataFrame:
    """Wide-format MS1 DataFrame for a single compound × single file.

    Mirrors the schema produced by
    :func:`metatlas2.extract_data_from_h5._widen_one_file_ms1`.
    """
    uid = adenine_compound.mz_rt_uid
    rts  = [ADENINE_RT - 0.1, ADENINE_RT - 0.05, ADENINE_RT, ADENINE_RT + 0.05, ADENINE_RT + 0.1]
    ints = [2e4, 6e4, 1e5, 7e4, 3e4]
    mzs  = [ADENINE_MZ] * 5
    in_f = [True] * 5

    return pd.DataFrame([{
        "mz_rt_uid":  uid,
        "filename":   "test_run_POS_Run001.h5",
        "inchi_key":  adenine_compound.inchi_key,
        "adduct":     adenine_compound.adduct,
        "spec_rts":   rts,
        "spec_ints":  ints,
        "spec_mzs":   mzs,
        "in_feature": in_f,
    }])

@pytest.fixture()
def ms2_wide_df(adenine_compound: CompoundMZRT) -> pd.DataFrame:
    """Wide-format MS2 DataFrame for a single compound × single scan.

    Mirrors the schema produced by
    :func:`metatlas2.extract_data_from_h5._widen_one_file_ms2`.
    """
    uid = adenine_compound.mz_rt_uid
    return pd.DataFrame([{
        "mz_rt_uid":           uid,
        "filename":            "test_run_POS_Run001.h5",
        "inchi_key":           adenine_compound.inchi_key,
        "adduct":              adenine_compound.adduct,
        "scan_rt":             ADENINE_RT,
        "precursor_MZ":        ADENINE_MZ,
        "precursor_intensity": 1e5,
        "collision_energy":    35.0,
        "frag_mzs":            [94.04, 119.04, 136.06],
        "frag_ints":           [1e4, 5e4, 2e5],
        "in_feature":          True,
    }])