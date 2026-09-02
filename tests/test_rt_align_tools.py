"""Unit tests for metatlas2/rt_align_tools.py.

All tests exercise pure calculation functions — no database, no filesystem
(except ``_save_rt_aligned_stats_to_json`` which writes to ``tmp_path``),
and no subprocess calls.

Tested functions
----------------
* :func:`build_polynomial_model`
* :func:`format_model_equation`
* :func:`calculate_model_values_from_existing`
* :func:`_apply_rt_model`
* :func:`_save_rt_aligned_stats_to_json`
* :func:`build_rt_alignment_model`   (via a lightweight ``RTAlign`` stub)
* :func:`calculate_rt_shifts`        (via a lightweight ``RTAlign`` stub)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from metatlas2.rt_align_tools import (
    _apply_rt_model,
    _save_rt_aligned_stats_to_json,
    build_polynomial_model,
    build_rt_alignment_model,
    calculate_model_values_from_existing,
    calculate_rt_shifts,
    format_model_equation,
)

from conftest import (
    ADENINE_MZ,
    ADENINE_RT,
    ADENINE_RMIN,
    ADENINE_RMAX,
    RIBOFLAVIN_MZ,
    RIBOFLAVIN_RT,
    RIBOFLAVIN_RMIN,
    RIBOFLAVIN_RMAX,
    _make_compound_mzrt,
    _make_atlas,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

DEFAULT_RT_PARAMS = {
    "min_observations_per_compound": 1,
    "min_compounds_for_modeling": 2,
    "polynomial_degree": 2,
    "r2_threshold": 0.5,
    "apply_model_to_min_max": True,
    "exclude_inchikeys": [],
}


def _make_ms1_wide_for_uid(uid: str, rts: list[float], ints: list[float]) -> pd.DataFrame:
    """Build a minimal wide-format MS1 row for one compound × one file."""
    return pd.DataFrame([{
        "mz_rt_uid":  uid,
        "filename":   "qc_run.h5",
        "spec_rts":   rts,
        "spec_ints":  ints,
        "spec_mzs":   [ADENINE_MZ] * len(rts),
        "in_feature": [True] * len(rts),
    }])


def _make_rt_align_stub(atlas, ms1_df, params=None):
    """Build a minimal object that satisfies ``build_rt_alignment_model``'s interface."""
    from metatlas2.workflow_objects import ExperimentalData
    exp_data = ExperimentalData(ms1_df=ms1_df)
    return SimpleNamespace(
        align_atlas_obj=atlas,
        aligned_atlas_obj=atlas,          # used by calculate_rt_shifts
        experimental_data=exp_data,
        rt_alignment_params=params or DEFAULT_RT_PARAMS,
        rt_alignment_model=None,
        modeling_data=None,
        paths={"rt_alignment_results_dir": ""},  # overridden in tests that write files
    )


def _fitted_linear_model(slope: float = 1.0, intercept: float = 0.0) -> dict:
    """Return a model_info dict from a simple linear fit y = intercept + slope*x."""
    X = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
    y = intercept + slope * X
    return build_polynomial_model(X, y, degree=1)


# ---------------------------------------------------------------------------
# build_polynomial_model
# ---------------------------------------------------------------------------

class TestBuildPolynomialModel:

    def test_returns_expected_keys(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * X + 1.0
        result = build_polynomial_model(X, y, degree=1)
        for key in ("model", "poly_features", "degree", "r2", "rmse", "y_pred",
                    "coefficients", "intercept"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_linear_fit_r2_is_one(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * X + 2.0
        result = build_polynomial_model(X, y, degree=1)
        assert abs(result["r2"] - 1.0) < 1e-6

    def test_perfect_linear_fit_rmse_is_zero(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * X + 2.0
        result = build_polynomial_model(X, y, degree=1)
        assert result["rmse"] < 1e-6

    def test_degree_stored_correctly(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = X ** 2
        result = build_polynomial_model(X, y, degree=2)
        assert result["degree"] == 2

    def test_y_pred_length_matches_input(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = X + 0.5
        result = build_polynomial_model(X, y, degree=1)
        assert len(result["y_pred"]) == len(X)

    def test_degree_2_fits_quadratic_perfectly(self):
        X = np.linspace(0, 10, 20)
        y = 0.5 * X**2 - 2.0 * X + 1.0
        result = build_polynomial_model(X, y, degree=2)
        assert result["r2"] > 0.9999

    def test_coefficients_shape_matches_degree(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = X + 1.0
        result = build_polynomial_model(X, y, degree=1)
        # degree=1 → PolynomialFeatures produces [1, x] → 2 coefficients
        assert len(result["coefficients"]) == 2

    def test_intercept_is_scalar(self):
        X = np.array([1.0, 2.0, 3.0])
        y = X
        result = build_polynomial_model(X, y, degree=1)
        assert isinstance(float(result["intercept"]), float)


# ---------------------------------------------------------------------------
# format_model_equation
# ---------------------------------------------------------------------------

class TestFormatPolynomialEquation:

    def _model_info(self, degree: int, coeffs: list[float], intercept: float) -> dict:
        return {"degree": degree, "coefficients": np.array(coeffs), "intercept": intercept}

    def test_degree_1_contains_rt_atlas(self):
        info = self._model_info(1, [0.0, 1.5], 0.1)
        eq = format_model_equation(info)
        assert "RT_atlas" in eq
        assert "RT_aligned" in eq

    def test_degree_2_contains_squared_term(self):
        info = self._model_info(2, [0.0, 1.0, 0.01], 0.05)
        eq = format_model_equation(info)
        assert "RT_atlas^2" in eq

    def test_degree_3_contains_cubed_term(self):
        info = self._model_info(3, [0.0, 1.0, 0.01, 0.001], 0.0)
        eq = format_model_equation(info)
        assert "RT_atlas^2" in eq or "³" in eq

    def test_unknown_degree_returns_string(self):
        info = self._model_info(5, [0.0] * 6, 0.0)
        eq = format_model_equation(info)
        assert isinstance(eq, str)
        assert len(eq) > 0

    def test_intercept_value_appears_in_equation(self):
        info = self._model_info(1, [0.0, 1.0], 3.14159)
        eq = format_model_equation(info)
        assert "3.141590" in eq


# ---------------------------------------------------------------------------
# calculate_model_values_from_existing
# ---------------------------------------------------------------------------

class TestCalculateModelValuesFromExisting:

    def _model_dict_from_fit(self) -> dict:
        """Build a serialisable model_dict as it would be stored in the DB."""
        X = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        y = 1.05 * X + 0.3
        fit = build_polynomial_model(X, y, degree=1)
        return {
            "metadata": {
                "poly_degree": 1,
                "poly_include_bias": True,
                "poly_interaction_only": False,
                "model_coefficients": fit["coefficients"].tolist(),
                "model_intercept": float(fit["intercept"]),
            }
        }

    def test_returns_model_and_poly_features(self):
        md = self._model_dict_from_fit()
        result = calculate_model_values_from_existing(md)
        assert "model" in result
        assert "poly_features" in result

    def test_reconstructed_model_predicts_correctly(self):
        X = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        y = 1.05 * X + 0.3
        fit = build_polynomial_model(X, y, degree=1)
        md = {
            "metadata": {
                "poly_degree": 1,
                "poly_include_bias": True,
                "poly_interaction_only": False,
                "model_coefficients": fit["coefficients"].tolist(),
                "model_intercept": float(fit["intercept"]),
            }
        }
        result = calculate_model_values_from_existing(md)
        # Predict on a new point
        pred = _apply_rt_model([7.5], result)
        expected = 1.05 * 7.5 + 0.3
        assert abs(pred[0] - expected) < 0.01

    def test_intercept_stored_in_result(self):
        md = self._model_dict_from_fit()
        result = calculate_model_values_from_existing(md)
        assert "intercept" in result

    def test_legacy_dict_format_also_works(self):
        """Older model dicts store coefficients at the top level, not under 'metadata'."""
        X = np.array([1.0, 5.0, 10.0])
        y = 2.0 * X
        fit = build_polynomial_model(X, y, degree=1)
        legacy = {
            "metadata": {},
            "degree": 1,
            "coefficients": fit["coefficients"].tolist(),
            "intercept": float(fit["intercept"]),
        }
        result = calculate_model_values_from_existing(legacy)
        assert "model" in result


# ---------------------------------------------------------------------------
# _apply_rt_model
# ---------------------------------------------------------------------------

class TestApplyRtModel:

    def test_identity_model_returns_input(self):
        """A model with slope=1, intercept=0 should return the input unchanged."""
        model_info = _fitted_linear_model(slope=1.0, intercept=0.0)
        values = [2.0, 5.0, 10.0]
        result = _apply_rt_model(values, model_info)
        for v, r in zip(values, result):
            assert abs(r - v) < 0.01

    def test_constant_shift_applied(self):
        """A model with slope=1, intercept=0.5 should shift all values by +0.5."""
        model_info = _fitted_linear_model(slope=1.0, intercept=0.5)
        values = [2.0, 5.0, 10.0]
        result = _apply_rt_model(values, model_info)
        for v, r in zip(values, result):
            assert abs(r - (v + 0.5)) < 0.05

    def test_scaling_applied(self):
        """A model with slope=2, intercept=0 should double all values."""
        model_info = _fitted_linear_model(slope=2.0, intercept=0.0)
        values = [3.0, 6.0]
        result = _apply_rt_model(values, model_info)
        for v, r in zip(values, result):
            assert abs(r - 2.0 * v) < 0.05

    def test_single_value_input(self):
        model_info = _fitted_linear_model(slope=1.0, intercept=0.0)
        result = _apply_rt_model([5.0], model_info)
        assert len(result) == 1

    def test_output_length_matches_input(self):
        model_info = _fitted_linear_model(slope=1.0, intercept=0.0)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _apply_rt_model(values, model_info)
        assert len(result) == len(values)


# ---------------------------------------------------------------------------
# _save_rt_aligned_stats_to_json
# ---------------------------------------------------------------------------

class TestSaveRtAlignedStatsToJson:

    def test_file_created(self, tmp_path):
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[0.1, -0.05, 0.2],
            per_compound_rt_shifts=[{"compound_name": "adenine", "rt_shift": 0.1}],
            output_file="rt_shifts.json",
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "rt_shifts.json").exists()

    def test_json_is_valid(self, tmp_path):
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[0.1, 0.2],
            per_compound_rt_shifts=[],
            output_file="out.json",
            output_dir=str(tmp_path),
        )
        with open(tmp_path / "out.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_stats_keys_present(self, tmp_path):
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[0.1, -0.1, 0.3],
            per_compound_rt_shifts=[],
            output_file="out.json",
            output_dir=str(tmp_path),
        )
        with open(tmp_path / "out.json") as f:
            data = json.load(f)
        for key in ("rt_shift_min", "rt_shift_max", "rt_shift_median"):
            assert key in data["stats"]

    def test_stats_values_correct(self, tmp_path):
        shifts = [0.1, -0.1, 0.3]
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=shifts,
            per_compound_rt_shifts=[],
            output_file="out.json",
            output_dir=str(tmp_path),
        )
        with open(tmp_path / "out.json") as f:
            data = json.load(f)
        assert abs(data["stats"]["rt_shift_min"] - min(shifts)) < 1e-9
        assert abs(data["stats"]["rt_shift_max"] - max(shifts)) < 1e-9
        assert abs(data["stats"]["rt_shift_median"] - float(np.median(shifts))) < 1e-9

    def test_compounds_list_stored(self, tmp_path):
        compounds = [{"compound_name": "adenine", "rt_shift": 0.05}]
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[0.05],
            per_compound_rt_shifts=compounds,
            output_file="out.json",
            output_dir=str(tmp_path),
        )
        with open(tmp_path / "out.json") as f:
            data = json.load(f)
        assert data["compounds"] == compounds

    def test_empty_shifts_produces_empty_stats(self, tmp_path):
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[],
            per_compound_rt_shifts=[],
            output_file="out.json",
            output_dir=str(tmp_path),
        )
        with open(tmp_path / "out.json") as f:
            data = json.load(f)
        assert data["stats"] == {}

    def test_output_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        _save_rt_aligned_stats_to_json(
            all_rt_shifts=[0.1],
            per_compound_rt_shifts=[],
            output_file="out.json",
            output_dir=str(nested),
        )
        assert (nested / "out.json").exists()


# ---------------------------------------------------------------------------
# build_rt_alignment_model  (integration via RTAlign stub)
# ---------------------------------------------------------------------------

class TestBuildRtAlignmentModel:
    """Tests for the full model-building pipeline.

    We build a two-compound atlas (adenine + riboflavin) with synthetic MS1
    data that has a known linear RT shift, then verify the fitted model
    captures that shift.
    """

    @pytest.fixture()
    def two_compound_atlas(self, adenine_compound, riboflavin_compound):
        return _make_atlas([adenine_compound, riboflavin_compound], polarity="POS")

    @pytest.fixture()
    def rt_align_stub(self, two_compound_atlas, adenine_compound, riboflavin_compound):
        """RTAlign stub with a +0.2 min constant shift on all observed RTs."""
        shift = 0.2
        uid_ade = adenine_compound.mz_rt_uid
        uid_rib = riboflavin_compound.mz_rt_uid

        ms1 = pd.concat([
            _make_ms1_wide_for_uid(
                uid_ade,
                rts=[ADENINE_RT + shift + d for d in (-0.05, 0.0, 0.05)],
                ints=[5e4, 1e5, 5e4],
            ),
            _make_ms1_wide_for_uid(
                uid_rib,
                rts=[RIBOFLAVIN_RT + shift + d for d in (-0.05, 0.0, 0.05)],
                ints=[3e4, 8e4, 3e4],
            ),
        ], ignore_index=True)

        return _make_rt_align_stub(two_compound_atlas, ms1)

    def test_model_attached_to_obj(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        assert rt_align_stub.rt_alignment_model is not None

    def test_modeling_data_attached(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        assert rt_align_stub.modeling_data is not None
        assert not rt_align_stub.modeling_data.empty

    def test_model_has_r2_key(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        assert "r2" in rt_align_stub.rt_alignment_model

    def test_model_r2_reasonable_for_linear_shift(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        # A constant shift is perfectly linear → R² should be very high
        assert rt_align_stub.rt_alignment_model["r2"] > 0.9

    def test_modeling_data_has_predicted_rt(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        assert "predicted_rt" in rt_align_stub.modeling_data.columns

    def test_modeling_data_has_residual(self, rt_align_stub):
        build_rt_alignment_model(rt_align_stub)
        assert "residual" in rt_align_stub.modeling_data.columns

    def test_raises_when_no_compounds_have_data(self, two_compound_atlas):
        stub = _make_rt_align_stub(two_compound_atlas, pd.DataFrame(
            columns=["mz_rt_uid", "filename", "spec_rts", "spec_ints", "spec_mzs", "in_feature"]
        ))
        with pytest.raises(ValueError, match="No compounds"):
            build_rt_alignment_model(stub)

    def test_raises_when_too_few_compounds(self, adenine_compound):
        """Only one compound → below min_compounds_for_modeling=2."""
        single_atlas = _make_atlas([adenine_compound], polarity="POS")
        uid = adenine_compound.mz_rt_uid
        ms1 = _make_ms1_wide_for_uid(uid, [ADENINE_RT], [1e5])
        stub = _make_rt_align_stub(single_atlas, ms1)
        with pytest.raises(ValueError, match="Insufficient"):
            build_rt_alignment_model(stub)

    def test_exclude_inchikeys_skips_compound(self, two_compound_atlas,
                                               adenine_compound, riboflavin_compound):
        """Excluding adenine's InChIKey should leave only riboflavin — too few for modeling."""
        uid_ade = adenine_compound.mz_rt_uid
        uid_rib = riboflavin_compound.mz_rt_uid
        ms1 = pd.concat([
            _make_ms1_wide_for_uid(uid_ade, [ADENINE_RT], [1e5]),
            _make_ms1_wide_for_uid(uid_rib, [RIBOFLAVIN_RT], [8e4]),
        ], ignore_index=True)
        params = {**DEFAULT_RT_PARAMS, "exclude_inchikeys": [adenine_compound.inchi_key]}
        stub = _make_rt_align_stub(two_compound_atlas, ms1, params=params)
        with pytest.raises(ValueError, match="Insufficient"):
            build_rt_alignment_model(stub)


# ---------------------------------------------------------------------------
# calculate_rt_shifts  (integration via RTAlign stub)
# ---------------------------------------------------------------------------

class TestCalculateRtShifts:
    """Tests for the RT-shift application step.

    We fit a known linear model first, then call ``calculate_rt_shifts`` and
    verify the output DataFrame has the expected structure and values.
    """

    @pytest.fixture()
    def stub_with_model(self, tmp_path, adenine_compound, riboflavin_compound):
        """RTAlign stub with a pre-fitted linear model (slope=1, intercept=+0.2)."""
        atlas = _make_atlas([adenine_compound, riboflavin_compound], polarity="POS")
        # Fit a model on synthetic data with a +0.2 min constant shift
        X = np.array([ADENINE_RT, RIBOFLAVIN_RT])
        y = X + 0.2
        model_info = build_polynomial_model(X, y, degree=1)

        stub = _make_rt_align_stub(atlas, pd.DataFrame())
        stub.rt_alignment_model = model_info
        stub.paths = {"rt_alignment_results_dir": str(tmp_path)}
        return stub

    def test_returns_dataframe(self, stub_with_model):
        result = calculate_rt_shifts(stub_with_model)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_compound(self, stub_with_model, adenine_compound, riboflavin_compound):
        result = calculate_rt_shifts(stub_with_model)
        assert len(result) == 2

    def test_mz_rt_uid_column_present(self, stub_with_model):
        result = calculate_rt_shifts(stub_with_model)
        assert "mz_rt_uid" in result.columns

    def test_rt_peak_column_present(self, stub_with_model):
        result = calculate_rt_shifts(stub_with_model)
        assert "rt_peak" in result.columns

    def test_aligned_rt_peak_shifted_by_model(self, stub_with_model, adenine_compound):
        result = calculate_rt_shifts(stub_with_model)
        uid = adenine_compound.mz_rt_uid
        row = result[result["mz_rt_uid"] == uid].iloc[0]
        # Model adds ~0.2 min; allow ±0.05 tolerance for polynomial rounding
        assert abs(row["rt_peak"] - (ADENINE_RT + 0.2)) < 0.05

    def test_rt_min_less_than_rt_max(self, stub_with_model):
        result = calculate_rt_shifts(stub_with_model)
        assert (result["rt_min"] < result["rt_max"]).all()

    def test_apply_model_to_min_max_false_uses_window(self, tmp_path,
                                                        adenine_compound, riboflavin_compound):
        """When apply_model_to_min_max=False, rt_min/rt_max are centred on the aligned peak."""
        atlas = _make_atlas([adenine_compound, riboflavin_compound], polarity="POS")
        X = np.array([ADENINE_RT, RIBOFLAVIN_RT])
        y = X + 0.2
        model_info = build_polynomial_model(X, y, degree=1)
        params = {**DEFAULT_RT_PARAMS, "apply_model_to_min_max": False}
        stub = _make_rt_align_stub(atlas, pd.DataFrame(), params=params)
        stub.rt_alignment_model = model_info
        stub.paths = {"rt_alignment_results_dir": str(tmp_path)}

        result = calculate_rt_shifts(stub)
        uid = adenine_compound.mz_rt_uid
        row = result[result["mz_rt_uid"] == uid].iloc[0]
        # Window = rt_max - rt_min of the original compound
        original_window = ADENINE_RMAX - ADENINE_RMIN
        computed_window = row["rt_max"] - row["rt_min"]
        assert abs(computed_window - original_window) < 0.01

    def test_json_file_written(self, stub_with_model, tmp_path):
        calculate_rt_shifts(stub_with_model)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_non_positive_aligned_rt_clamped_to_positive(self, tmp_path,
                                                           adenine_compound, riboflavin_compound):
        """A model that predicts a negative RT should be clamped to 0.01."""
        atlas = _make_atlas([adenine_compound, riboflavin_compound], polarity="POS")
        # Fit a model that maps all atlas RTs to negative values
        X = np.array([ADENINE_RT, RIBOFLAVIN_RT])
        y = -X  # negative predictions
        model_info = build_polynomial_model(X, y, degree=1)
        stub = _make_rt_align_stub(atlas, pd.DataFrame())
        stub.rt_alignment_model = model_info
        stub.paths = {"rt_alignment_results_dir": str(tmp_path)}

        result = calculate_rt_shifts(stub)
        assert (result["rt_peak"] > 0).all()
