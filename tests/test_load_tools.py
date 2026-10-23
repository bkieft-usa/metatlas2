"""Unit tests for metatlas2/load_tools.py.

Tested functions
----------------
* :func:`load_atlas_input` — CSV/TSV loading, column defaults, and rt_peak sort
  order (including the special-case anchor compound that must always be first,
  and the labeled-before-unlabeled rule for ISTD/QC atlases).
* :func:`_inchi_key_prefix` — InChI-key prefix extraction helper.
"""

from __future__ import annotations

import pandas as pd
import pytest

import metatlas2.load_tools as ldt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANCHOR_INCHI_KEY = "LCMZECCEEOQWLQ-UHFFFAOYSA-N"
_OTHER_INCHI_KEY  = "AAAABBBBCCCCDD-UHFFFAOYSA-N"

# A pair of InChI keys that share the same prefix (UGQMRVRMYYASKQ) —
# mimicking a labeled / unlabeled isotopologue pair.
_LABELED_IK   = "UGQMRVRMYYASKQ-KQYNXXCUSA-N"   # 13C-labeled version
_UNLABELED_IK = "UGQMRVRMYYASKQ-SHKQESLVSA-N"   # unlabeled version

_REQUIRED_COLS = ["inchi_key", "compound_name", "rt_peak", "mz", "adduct"]


def _make_tsv(rows: list[dict]) -> str:
    """Return a TSV string from a list of row dicts."""
    if not rows:
        return ""
    header = "\t".join(rows[0].keys())
    body   = "\n".join("\t".join(str(r[k]) for k in rows[0]) for r in rows)
    return header + "\n" + body


def _write_tmp(tmp_path, content: str, suffix: str = ".tsv") -> str:
    p = tmp_path / f"atlas{suffix}"
    p.write_text(content)
    return str(p)


def _base_row(
    inchi_key: str,
    compound_name: str,
    rt_peak: float,
    idx: int = 0,
) -> dict:
    return {
        "inchi_key":     inchi_key,
        "compound_name": compound_name,
        "rt_peak":       rt_peak,
        "mz":            100.0 + idx,
        "adduct":        "[M+H]+",
    }


# ---------------------------------------------------------------------------
# TestInchiKeyPrefix
# ---------------------------------------------------------------------------

class TestInchiKeyPrefix:
    def test_standard_key(self):
        assert ldt._inchi_key_prefix("UGQMRVRMYYASKQ-KQYNXXCUSA-N") == "UGQMRVRMYYASKQ"

    def test_empty_string(self):
        assert ldt._inchi_key_prefix("") == ""

    def test_no_dash(self):
        assert ldt._inchi_key_prefix("ABCDEFGHIJKLMN") == "ABCDEFGHIJKLMN"


# ---------------------------------------------------------------------------
# TestLoadAtlasInputSorting  (REF / no category — original behaviour)
# ---------------------------------------------------------------------------

class TestLoadAtlasInputSorting:
    """Verify that load_atlas_input sorts by rt_peak ascending and that the
    anchor compound (LCMZECCEEOQWLQ-UHFFFAOYSA-N) is always placed first."""

    def _base_rows(self, rt_peaks: list[float], inchi_keys: list[str]) -> list[dict]:
        assert len(rt_peaks) == len(inchi_keys)
        return [
            _base_row(ik, f"cpd_{i}", rt, idx=i)
            for i, (rt, ik) in enumerate(zip(rt_peaks, inchi_keys))
        ]

    def test_sorts_ascending_by_rt_peak_no_anchor(self, tmp_path):
        """Without the anchor compound, rows are sorted ascending by rt_peak."""
        rows = self._base_rows(
            rt_peaks   =[3.0, 1.0, 2.0],
            inchi_keys =[_OTHER_INCHI_KEY] * 3,
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        assert list(df["rt_peak"]) == [1.0, 2.0, 3.0]

    def test_anchor_first_when_not_lowest_rt(self, tmp_path):
        """Anchor compound with a high rt_peak must still appear first."""
        rows = self._base_rows(
            rt_peaks   =[0.5, 0.8, 5.0],
            inchi_keys =[_OTHER_INCHI_KEY, _OTHER_INCHI_KEY, _ANCHOR_INCHI_KEY],
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        assert df.iloc[0]["inchi_key"] == _ANCHOR_INCHI_KEY

    def test_anchor_first_when_lowest_rt(self, tmp_path):
        """Anchor compound with the lowest rt_peak is still first (no regression)."""
        rows = self._base_rows(
            rt_peaks   =[0.1, 2.0, 3.0],
            inchi_keys =[_ANCHOR_INCHI_KEY, _OTHER_INCHI_KEY, _OTHER_INCHI_KEY],
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        assert df.iloc[0]["inchi_key"] == _ANCHOR_INCHI_KEY

    def test_non_anchor_rows_sorted_after_anchor(self, tmp_path):
        """After the anchor, remaining rows are sorted ascending by rt_peak."""
        rows = self._base_rows(
            rt_peaks   =[3.0, 1.0, 9.0, 0.5],
            inchi_keys =[
                _OTHER_INCHI_KEY,
                _OTHER_INCHI_KEY,
                _ANCHOR_INCHI_KEY,
                _OTHER_INCHI_KEY,
            ],
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        assert df.iloc[0]["inchi_key"] == _ANCHOR_INCHI_KEY
        assert list(df["rt_peak"][1:]) == [0.5, 1.0, 3.0]

    def test_no_sort_priority_column_in_output(self, tmp_path):
        """The internal _sort_priority helper column must not leak into output."""
        rows = self._base_rows(
            rt_peaks   =[2.0, 1.0],
            inchi_keys =[_OTHER_INCHI_KEY, _ANCHOR_INCHI_KEY],
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        for col in df.columns:
            assert not col.startswith("_"), f"Internal column leaked: {col}"

    def test_index_is_reset_after_sort(self, tmp_path):
        """DataFrame index must be 0-based after sorting."""
        rows = self._base_rows(
            rt_peaks   =[3.0, 1.0, 2.0],
            inchi_keys =[_OTHER_INCHI_KEY] * 3,
        )
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path)
        assert list(df.index) == list(range(len(df)))


# ---------------------------------------------------------------------------
# TestLoadAtlasInputLabeledFirst  (ISTD / QC category)
# ---------------------------------------------------------------------------

class TestLoadAtlasInputLabeledFirst:
    """Verify the labeled-before-unlabeled rule for ISTD and QC atlases.

    When two rows share the same InChI-key prefix AND the same rt_peak, the
    row whose compound_name contains an isotope-label string (e.g. '13C',
    '15N') must appear before the unlabeled row.
    """

    def _make_pair(
        self,
        rt_peak: float,
        labeled_first_in_file: bool = False,
    ) -> list[dict]:
        """Return a (labeled, unlabeled) or (unlabeled, labeled) pair at rt_peak."""
        labeled = _base_row(_LABELED_IK,   "Compound 13C-labeled", rt_peak, idx=0)
        unlabeled = _base_row(_UNLABELED_IK, "Compound unlabeled",   rt_peak, idx=1)
        return [labeled, unlabeled] if labeled_first_in_file else [unlabeled, labeled]

    # --- ISTD category ---

    def test_istd_labeled_before_unlabeled_when_unlabeled_first_in_file(self, tmp_path):
        rows = self._make_pair(rt_peak=2.0, labeled_first_in_file=False)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        assert df.iloc[0]["inchi_key"] == _LABELED_IK
        assert df.iloc[1]["inchi_key"] == _UNLABELED_IK

    def test_istd_labeled_before_unlabeled_when_labeled_first_in_file(self, tmp_path):
        """Order is preserved when labeled is already first."""
        rows = self._make_pair(rt_peak=2.0, labeled_first_in_file=True)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        assert df.iloc[0]["inchi_key"] == _LABELED_IK
        assert df.iloc[1]["inchi_key"] == _UNLABELED_IK

    def test_qc_labeled_before_unlabeled(self, tmp_path):
        """Same rule applies for QC category."""
        rows = self._make_pair(rt_peak=1.5, labeled_first_in_file=False)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="QC")
        assert df.iloc[0]["inchi_key"] == _LABELED_IK

    def test_ref_category_does_not_apply_labeled_rule(self, tmp_path):
        """For REF atlases the labeled-first rule must NOT be applied."""
        rows = self._make_pair(rt_peak=2.0, labeled_first_in_file=False)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="REF")
        # Without the rule, file order is preserved after rt_peak sort (both same RT)
        # — the unlabeled row was first in the file so it stays first.
        assert df.iloc[0]["inchi_key"] == _UNLABELED_IK

    def test_no_category_does_not_apply_labeled_rule(self, tmp_path):
        """When atlas_category is None the labeled-first rule must NOT be applied."""
        rows = self._make_pair(rt_peak=2.0, labeled_first_in_file=False)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category=None)
        assert df.iloc[0]["inchi_key"] == _UNLABELED_IK

    def test_15n_label_string_triggers_rule(self, tmp_path):
        """'15N' in compound_name is also recognised as a label."""
        labeled   = _base_row(_LABELED_IK,   "Compound 15N-labeled", 3.0, idx=0)
        unlabeled = _base_row(_UNLABELED_IK, "Compound unlabeled",   3.0, idx=1)
        rows = [unlabeled, labeled]  # unlabeled first in file
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        assert df.iloc[0]["inchi_key"] == _LABELED_IK

    def test_anchor_still_first_with_labeled_rule(self, tmp_path):
        """Anchor compound remains first even when labeled-first rule is active."""
        anchor  = _base_row(_ANCHOR_INCHI_KEY, "Anchor compound", 9.0, idx=0)
        labeled   = _base_row(_LABELED_IK,   "Compound 13C-labeled", 2.0, idx=1)
        unlabeled = _base_row(_UNLABELED_IK, "Compound unlabeled",   2.0, idx=2)
        rows = [unlabeled, labeled, anchor]
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        assert df.iloc[0]["inchi_key"] == _ANCHOR_INCHI_KEY
        assert df.iloc[1]["inchi_key"] == _LABELED_IK
        assert df.iloc[2]["inchi_key"] == _UNLABELED_IK

    def test_different_rt_peaks_not_reordered(self, tmp_path):
        """Labeled/unlabeled pairs at DIFFERENT rt_peaks are NOT reordered."""
        labeled   = _base_row(_LABELED_IK,   "Compound 13C-labeled", 5.0, idx=0)
        unlabeled = _base_row(_UNLABELED_IK, "Compound unlabeled",   2.0, idx=1)
        rows = [labeled, unlabeled]
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        # Different rt_peaks → sorted by rt_peak ascending; unlabeled (2.0) comes first
        assert df.iloc[0]["inchi_key"] == _UNLABELED_IK
        assert df.iloc[1]["inchi_key"] == _LABELED_IK

    def test_no_internal_columns_leaked_with_labeled_rule(self, tmp_path):
        """No internal helper columns must appear in the output DataFrame."""
        rows = self._make_pair(rt_peak=2.0, labeled_first_in_file=False)
        path = _write_tmp(tmp_path, _make_tsv(rows))
        df = ldt.load_atlas_input(path, atlas_category="ISTD")
        for col in df.columns:
            assert not col.startswith("_"), f"Internal column leaked: {col}"


# ---------------------------------------------------------------------------
# TestLoadAtlasInputDefaults
# ---------------------------------------------------------------------------

class TestLoadAtlasInputDefaults:
    """Verify that optional columns receive correct default values."""

    def _minimal_tsv(self, tmp_path) -> str:
        rows = [_base_row(_OTHER_INCHI_KEY, "cpd_a", 1.0)]
        return _write_tmp(tmp_path, _make_tsv(rows))

    def test_rt_min_default(self, tmp_path):
        df = ldt.load_atlas_input(self._minimal_tsv(tmp_path))
        assert df.iloc[0]["rt_min"] == pytest.approx(1.0 - 0.5)

    def test_rt_max_default(self, tmp_path):
        df = ldt.load_atlas_input(self._minimal_tsv(tmp_path))
        assert df.iloc[0]["rt_max"] == pytest.approx(1.0 + 0.5)

    def test_mz_tolerance_default(self, tmp_path):
        df = ldt.load_atlas_input(self._minimal_tsv(tmp_path))
        assert df.iloc[0]["mz_tolerance"] == pytest.approx(5.0)

    def test_csv_extension_loaded(self, tmp_path):
        """Files with .csv extension are read as comma-separated."""
        row = _base_row(_OTHER_INCHI_KEY, "cpd_csv", 2.5)
        content = ",".join(row.keys()) + "\n" + ",".join(str(v) for v in row.values())
        path = _write_tmp(tmp_path, content, suffix=".csv")
        df = ldt.load_atlas_input(path)
        assert len(df) == 1
        assert df.iloc[0]["compound_name"] == "cpd_csv"

    def test_missing_required_column_raises(self, tmp_path):
        """A TSV missing a required column must raise ValueError."""
        rows = [{"inchi_key": _OTHER_INCHI_KEY, "compound_name": "cpd_x",
                 "mz": 100.0, "adduct": "[M+H]+"}]  # rt_peak omitted
        path = _write_tmp(tmp_path, _make_tsv(rows))
        with pytest.raises(ValueError, match="rt_peak"):
            ldt.load_atlas_input(path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ldt.load_atlas_input(str(tmp_path / "nonexistent.tsv"))
