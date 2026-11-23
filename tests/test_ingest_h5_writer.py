"""Tests for metatlas2.ingest.h5_writer using fake pymzml spectra (no real mzML needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tables

from metatlas2.ingest import h5_writer as hw


class FakeSpectrum:
    def __init__(self, polarity="pos", ms_level=1, rt=1.5, peaks=((100.0, 10.0), (202.0777, 5.0), (50.0, 1.0)),
                 precursors=None, collision_energy=20.0, missing=()):
        self._d = {"negative scan": polarity == "neg" or None, "positive scan": polarity == "pos" or None,
                   "collision energy": collision_energy}
        self.ms_level = ms_level
        self._rt = rt
        self._peaks = np.array(peaks, dtype=float)
        self.selected_precursors = precursors if precursors is not None else [{"mz": 300.5, "i": 1000.0}]
        self._missing = set(missing)

    def __getitem__(self, key):
        if key in self._missing:
            raise KeyError(key)
        return self._d[key]

    def scan_time_in_minutes(self):
        return self._rt

    def peaks(self, kind):
        assert kind == "centroided"
        return self._peaks


@pytest.fixture
def fake_reader(monkeypatch):
    def install(spectra):
        import pymzml.run

        monkeypatch.setattr(pymzml.run, "Reader", lambda path: iter(spectra))
    return install


def _read(path):
    with tables.open_file(str(path)) as f:
        data = {n._v_name: n.read() for n in f.list_nodes("/")}
        attrs = {k: f.root._v_attrs[k] for k in f.root._v_attrs._f_list("user")}
    return data, attrs


def test_layout_values_sorting_and_attrs(tmp_path, fake_reader):
    fake_reader([
        FakeSpectrum("pos", 1, rt=1.0),
        FakeSpectrum("neg", 1, rt=2.0, peaks=((150.0, 3.0), (202.0788, 9.0))),
        FakeSpectrum("pos", 2, rt=3.0, peaks=((80.0, 7.0), (60.0, 8.0))),
    ])
    out = tmp_path / "x.h5"
    summary = hw.mzml_to_h5(tmp_path / "x.mzML", out)
    data, attrs = _read(out)
    assert set(data) == {"ms1_neg", "ms1_pos", "ms2_neg", "ms2_pos",
                         "ms1_neg_mz", "ms1_pos_mz", "ms2_neg_mz", "ms2_pos_mz"}
    assert attrs["format_version"] == 5 and attrs["metatlas_version"] == "0.7"
    # Easy-IC lock mass removed (pos 202.07770, neg 202.07880), order preserved
    assert list(data["ms1_pos"]["mz"]) == pytest.approx([100.0, 50.0])
    assert list(data["ms1_pos_mz"]["mz"]) == pytest.approx([50.0, 100.0])
    assert list(data["ms1_neg"]["mz"]) == pytest.approx([150.0])
    assert data["ms1_neg"]["polarity"].tolist() == [0] and data["ms1_pos"]["polarity"].tolist() == [1, 1]
    ms2 = data["ms2_pos"]
    assert ms2.dtype.names == ("mz", "i", "rt", "polarity", "precursor_MZ", "precursor_intensity", "collision_energy")
    assert ms2["precursor_MZ"].tolist() == pytest.approx([300.5, 300.5]) and ms2["collision_energy"][0] == 20.0
    assert summary["rows"] == {"ms1_neg": 1, "ms2_neg": 0, "ms1_pos": 2, "ms2_pos": 2}
    assert not (tmp_path / "x.h5.partial").exists()


def test_skip_rules_match_legacy(tmp_path, fake_reader):
    fake_reader([
        FakeSpectrum("none", 1),                                   # unknown polarity
        FakeSpectrum("pos", 2, precursors=[{"mz": 1, "i": 1}, {"mz": 2, "i": 2}]),  # 2 precursors
        FakeSpectrum("pos", 2, missing=("collision energy",)),     # KeyError
        FakeSpectrum("pos", 1, peaks=()),                          # no peaks
        FakeSpectrum("pos", 1),                                    # kept
    ])
    summary = hw.mzml_to_h5(tmp_path / "x.mzML", tmp_path / "x.h5")
    assert summary["spectra_read"] == 5 and summary["spectra_skipped"] == 4
    assert summary["rows"]["ms1_pos"] == 2


def test_no_easyic_filter(tmp_path, fake_reader):
    fake_reader([FakeSpectrum("pos", 1)])
    hw.mzml_to_h5(tmp_path / "x.mzML", tmp_path / "x.h5", filter_easyic=False)
    data, _ = _read(tmp_path / "x.h5")
    assert len(data["ms1_pos"]) == 3


def test_failure_leaves_no_files(tmp_path, monkeypatch):
    import pymzml.run

    def boom(_path):
        def gen():
            yield FakeSpectrum("pos", 1)
            raise RuntimeError("corrupt mzML")
        return gen()

    monkeypatch.setattr(pymzml.run, "Reader", boom)
    with pytest.raises(RuntimeError, match="corrupt"):
        hw.mzml_to_h5(tmp_path / "x.mzML", tmp_path / "x.h5")
    assert list(tmp_path.iterdir()) == []


def test_cli_prints_json(tmp_path, fake_reader, capsys):
    fake_reader([FakeSpectrum("pos", 1)])
    assert hw.main([str(tmp_path / "x.mzML"), "--h5", str(tmp_path / "y.h5")]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["ok"] and out["h5_file"].endswith("y.h5")
