"""Unit tests for metatlas2/utils.py.

All functions are pure (no I/O, no network, no subprocess).

Tested functions
----------------
* :func:`safe_float`
* :func:`safe_isnan`
* :func:`as_list`
* :func:`jsonable_list`
"""

from __future__ import annotations

import numpy as np
import pytest

from metatlas2.utils import as_list, jsonable_list, safe_float, safe_isnan


# ===========================================================================
# safe_float
# ===========================================================================

class TestSafeFloat:

    def test_int_converted(self):
        assert safe_float(3) == 3.0

    def test_float_passthrough(self):
        assert safe_float(3.14) == 3.14

    def test_numeric_string_converted(self):
        assert safe_float("2.718") == 2.718

    def test_none_returns_default_nan(self):
        result = safe_float(None)
        assert np.isnan(result)

    def test_non_numeric_string_returns_default_nan(self):
        result = safe_float("abc")
        assert np.isnan(result)

    def test_empty_string_returns_default_nan(self):
        result = safe_float("")
        assert np.isnan(result)

    def test_custom_default_returned_on_error(self):
        assert safe_float("bad", default=0.0) == 0.0

    def test_nan_string_returns_nan(self):
        result = safe_float("nan")
        assert np.isnan(result)

    def test_inf_string_converted(self):
        assert safe_float("inf") == float("inf")

    def test_negative_number(self):
        assert safe_float(-5.5) == -5.5

    def test_zero(self):
        assert safe_float(0) == 0.0

    def test_numpy_float32(self):
        val = np.float32(1.5)
        assert abs(safe_float(val) - 1.5) < 1e-4

    def test_numpy_int64(self):
        val = np.int64(42)
        assert safe_float(val) == 42.0


# ===========================================================================
# safe_isnan
# ===========================================================================

class TestSafeIsnan:

    def test_none_is_nan(self):
        assert safe_isnan(None) is True

    def test_float_nan_is_nan(self):
        assert safe_isnan(float("nan")) is True

    def test_numpy_nan_is_nan(self):
        assert safe_isnan(np.nan) is True

    def test_zero_is_not_nan(self):
        assert safe_isnan(0.0) is False

    def test_positive_float_is_not_nan(self):
        assert safe_isnan(3.14) is False

    def test_negative_float_is_not_nan(self):
        assert safe_isnan(-1.0) is False

    def test_integer_is_not_nan(self):
        assert safe_isnan(5) is False

    def test_non_numeric_string_is_nan(self):
        assert safe_isnan("abc") is True

    def test_numeric_string_is_not_nan(self):
        assert safe_isnan("3.14") is False

    def test_empty_string_is_nan(self):
        assert safe_isnan("") is True

    def test_inf_is_not_nan(self):
        assert safe_isnan(float("inf")) is False

    def test_numpy_float32_nan_is_nan(self):
        assert safe_isnan(np.float32("nan")) is True

    def test_numpy_float32_value_is_not_nan(self):
        assert safe_isnan(np.float32(1.0)) is False

    def test_returns_bool(self):
        assert isinstance(safe_isnan(None), bool)
        assert isinstance(safe_isnan(1.0), bool)


# ===========================================================================
# as_list
# ===========================================================================

class TestAsList:

    def test_none_returns_empty_list(self):
        assert as_list(None) == []

    def test_list_returned_unchanged(self):
        lst = [1, 2, 3]
        result = as_list(lst)
        assert result is lst  # same object, not a copy

    def test_numpy_array_converted_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = as_list(arr)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_tuple_converted_to_list(self):
        result = as_list((1, 2, 3))
        assert result == [1, 2, 3]

    def test_generator_converted_to_list(self):
        result = as_list(x for x in range(3))
        assert result == [0, 1, 2]

    def test_scalar_int_returns_empty_list(self):
        # int is not iterable → TypeError → []
        result = as_list(42)
        assert result == []

    def test_scalar_float_returns_empty_list(self):
        result = as_list(3.14)
        assert result == []

    def test_empty_list_returned_unchanged(self):
        assert as_list([]) == []

    def test_empty_numpy_array_converted(self):
        result = as_list(np.array([]))
        assert result == []

    def test_string_converted_to_char_list(self):
        # str is iterable → list("abc") = ['a', 'b', 'c']
        result = as_list("abc")
        assert result == ["a", "b", "c"]

    def test_nested_list_not_flattened(self):
        lst = [[1, 2], [3, 4]]
        assert as_list(lst) == [[1, 2], [3, 4]]


# ===========================================================================
# jsonable_list
# ===========================================================================

class TestJsonableList:

    def test_plain_list_passthrough(self):
        assert jsonable_list([1, 2, 3]) == [1, 2, 3]

    def test_numpy_scalars_converted_to_python(self):
        arr = [np.float32(1.5), np.int64(2), np.float64(3.0)]
        result = jsonable_list(arr)
        for elem in result:
            assert not isinstance(elem, np.generic), f"Expected Python scalar, got {type(elem)}"

    def test_numpy_array_input_converted(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = jsonable_list(arr)
        assert isinstance(result, list)
        for elem in result:
            assert not isinstance(elem, np.generic)

    def test_none_returns_empty_list(self):
        assert jsonable_list(None) == []

    def test_mixed_types_preserved(self):
        lst = [1, "hello", 3.14]
        result = jsonable_list(lst)
        assert result == [1, "hello", 3.14]

    def test_empty_list_returns_empty(self):
        assert jsonable_list([]) == []

    def test_numpy_bool_converted(self):
        result = jsonable_list([np.bool_(True), np.bool_(False)])
        for elem in result:
            assert not isinstance(elem, np.generic)

    def test_values_preserved_after_conversion(self):
        arr = np.array([10, 20, 30], dtype=np.int32)
        result = jsonable_list(arr)
        assert result == [10, 20, 30]
