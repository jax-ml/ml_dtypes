# Copyright 2026 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for np.result_type() across ml_dtypes custom DTypes.

Multi-argument tests check permutations of one result_type() call. Nested
pairwise calls can produce different results and need not agree with it.
"""

import itertools

from dtype_test_utils import (
    CUSTOM_COMPLEX,
    CUSTOM_DTYPES,
    CUSTOM_FLOATS,
    CUSTOM_INTS,
    NUMPY_DTYPES,
    OPERAND_FORMS,
    operand,
)
import ml_dtypes
import numpy as np
import pytest

bf16 = ml_dtypes.bfloat16
f4 = ml_dtypes.float4_e2m1fn
f6_e2m3 = ml_dtypes.float6_e2m3fn
f6_e3m2 = ml_dtypes.float6_e3m2fn
f8_e3m4 = ml_dtypes.float8_e3m4
f8_e4m3 = ml_dtypes.float8_e4m3
f8_e4m3fn = ml_dtypes.float8_e4m3fn
f8_e4m3fnuz = ml_dtypes.float8_e4m3fnuz
f8_e4m3b11 = ml_dtypes.float8_e4m3b11fnuz
f8_e5m2 = ml_dtypes.float8_e5m2
f8_e5m2fnuz = ml_dtypes.float8_e5m2fnuz
f8_e5m3 = ml_dtypes.float8_e5m3fnu
f8_e8m0 = ml_dtypes.float8_e8m0fnu
bc32 = ml_dtypes.bcomplex32
c32 = ml_dtypes.complex32
i1, i2, i4 = ml_dtypes.int1, ml_dtypes.int2, ml_dtypes.int4
u1, u2, u4 = ml_dtypes.uint1, ml_dtypes.uint2, ml_dtypes.uint4

_NUMPY_VERSION = np.lib.NumpyVersion(np.__version__)
_NUMPY_LT_2_6 = _NUMPY_VERSION < "2.6.0.dev0"
_NUMPY_MULTI_DTYPE_PROMOTION_BROKEN = _NUMPY_LT_2_6
# NumPy 2.6 nightlies may not yet include the reduction-order fix.
if _NUMPY_VERSION == "2.6.0.dev0":
  _NUMPY_MULTI_DTYPE_PROMOTION_BROKEN = np.result_type(
      i4, np.uint8, np.float16
  ) != np.dtype(np.float16)


def test_all_custom_dtypes_are_covered():
  exported = (getattr(ml_dtypes, name) for name in ml_dtypes.__all__)
  expected = {
      dtype
      for dtype in exported
      if isinstance(dtype, type) and issubclass(dtype, np.generic)
  }
  assert set(CUSTOM_DTYPES) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Bool.
        (bf16, np.bool_, bf16),
        (f8_e4m3fn, np.bool_, f8_e4m3fn),
        (f4, np.bool_, f4),
        # Built-in floats.
        (f4, np.float16, np.float32),  # Legacy Python-float compatibility.
        (f8_e4m3fn, np.float16, np.float32),
        (f8_e5m2, np.float16, np.float32),
        (bf16, np.float16, np.float32),  # incomparable → float32
        (f8_e8m0, np.float16, np.float32),  # e8m0 range exceeds float16
        (f8_e4m3fn, np.float32, np.float32),
        (bf16, np.float32, np.float32),
        (bf16, np.float64, np.float64),
        (f8_e4m3fn, np.float64, np.float64),
        # Byte integers retain the custom float, even for lossy casts.
        (bf16, np.int8, bf16),
        (bf16, np.int16, np.float32),
        (f8_e4m3fn, np.int8, f8_e4m3fn),
        (f8_e4m3fn, np.int32, np.float64),
        # Built-in complex.
        (bf16, np.complex64, np.complex64),
        (f8_e4m3fn, np.complex64, np.complex64),
        (bf16, np.complex128, np.complex128),
        (f8_e4m3fn, np.complex128, np.complex128),
    ],
)
def test_custom_float_vs_numpy(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Identical formats.
        (bf16, bf16, bf16),
        (f8_e4m3fn, f8_e4m3fn, f8_e4m3fn),
        (f4, f4, f4),
        # Containment.
        (f4, f6_e2m3, f6_e2m3),
        (f4, f8_e4m3fn, f8_e4m3fn),
        (f4, bf16, bf16),
        (f8_e4m3fn, bf16, bf16),
        (f8_e5m2, bf16, bf16),
        (f8_e3m4, bf16, bf16),
        (f8_e5m3, bf16, bf16),
        # Incomparable formats.
        (f8_e4m3fn, f8_e5m2, np.float32),  # precision vs range
        (f8_e4m3, f8_e4m3fn, np.float32),  # infinity vs wider finite range
        (f8_e4m3, f8_e4m3fnuz, np.float32),  # infinity vs wider lower range
        (f8_e4m3fn, f8_e4m3fnuz, np.float32),  # upper vs lower range
        (f8_e5m2, f8_e5m2fnuz, np.float32),  # upper vs lower range
        (f6_e2m3, f6_e3m2, np.float32),  # precision vs range
        (f8_e5m3, f8_e5m2, np.float32),  # signedness vs range
        (f8_e5m3, f8_e8m0, np.float32),  # precision/zero vs range
    ],
)
def test_custom_float_vs_custom_float(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize("a, b", list(itertools.combinations(CUSTOM_FLOATS, 2)))
def test_custom_float_commutativity(a, b):
  assert np.result_type(a, b) == np.result_type(b, a)


@pytest.mark.parametrize(
    "float_t, int_t",
    [
        (bf16, i4),
        (bf16, u4),
        (bf16, i1),
        (f8_e4m3fn, i4),
        (f8_e4m3fn, u4),
        (f8_e5m2, i2),
        (f4, i1),
    ],
)
def test_custom_float_beats_custom_int(float_t, int_t):
  assert np.result_type(float_t, int_t) == np.dtype(float_t)
  assert np.result_type(int_t, float_t) == np.dtype(float_t)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Bool.
        (i4, np.bool_, i4),
        (u4, np.bool_, u4),
        (i1, np.bool_, i1),
        # Wider built-in types.
        (i4, np.int8, np.int8),
        (i4, np.int16, np.int16),
        (i4, np.int32, np.int32),
        (u4, np.int8, np.int8),
        (i2, np.int8, np.int8),
        (i4, np.float16, np.float16),
        (i4, np.float32, np.float32),
        (i4, np.float64, np.float64),
        (i4, np.complex64, np.complex64),
        (i4, np.complex128, np.complex128),
        (u4, np.float32, np.float32),
    ],
)
def test_custom_int_vs_numpy(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize("signed_int", [i1, i2, i4])
@pytest.mark.parametrize(
    "unsigned_builtin, expected",
    [
        (np.uint8, np.int16),
        (np.uint16, np.int32),
        (np.uint32, np.int64),
        (np.uint64, np.float64),
    ],
)
def test_signed_intn_vs_unsigned_builtins(
    signed_int, unsigned_builtin, expected
):
  assert np.result_type(signed_int, unsigned_builtin) == np.dtype(expected)
  assert np.result_type(unsigned_builtin, signed_int) == np.dtype(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Identical formats.
        (i4, i4, i4),
        (u4, u4, u4),
        # Mixed signedness.
        (i4, u4, np.int8),
        (i2, u2, np.int8),
        (i1, u1, np.int8),
        # Matching signedness, different widths.
        (i2, i4, np.int8),
        (u2, u4, np.uint8),
        (u1, u2, np.uint8),
        (u1, u4, np.uint8),
        (i1, i4, np.int8),
    ],
)
def test_custom_int_vs_custom_int(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Bool.
        (bc32, np.bool_, bc32),
        (c32, np.bool_, c32),
        # Byte integers fit in either component type.
        (bc32, np.int8, bc32),
        (c32, np.int8, c32),
        # Wider integers follow complex64 promotion.
        (bc32, np.int32, np.complex128),
        (bc32, np.uint32, np.complex128),
        (c32, np.int32, np.complex128),
        (c32, np.uint32, np.complex128),
        (bc32, np.int64, np.complex128),
        (bc32, np.uint64, np.complex128),
        (c32, np.int64, np.complex128),
        (c32, np.uint64, np.complex128),
        # Floats.
        (bc32, np.float16, np.complex64),
        (bc32, np.float32, np.complex64),
        (c32, np.float32, np.complex64),
        (c32, np.float16, c32),
        (bc32, np.float64, np.complex128),
        (bc32, np.longdouble, np.clongdouble),
        (c32, np.float64, np.complex128),
        # Built-in complex.
        (bc32, np.complex64, np.complex64),
        (bc32, np.complex128, np.complex128),
        (c32, np.complex64, np.complex64),
        (c32, np.complex128, np.complex128),
    ],
)
def test_custom_complex_vs_numpy(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (bc32, bf16, bc32),
        (bc32, f4, bc32),
        (bc32, f8_e4m3fn, bc32),
        (bc32, f8_e5m2, bc32),
        (bc32, f8_e5m3, bc32),
        (c32, f4, c32),
        (c32, f8_e4m3fn, c32),
        (c32, f8_e5m2, c32),
        (c32, f8_e5m3, np.complex64),
        (c32, bf16, np.complex64),
        (bc32, c32, np.complex64),
        (bc32, bc32, bc32),
        (c32, c32, c32),
    ],
)
def test_custom_complex_vs_custom(a, b, expected):
  assert np.result_type(a, b) == np.dtype(expected)
  assert np.result_type(b, a) == np.dtype(expected)


@pytest.mark.parametrize("complex_t", CUSTOM_COMPLEX)
@pytest.mark.parametrize("int_t", CUSTOM_INTS)
def test_custom_complex_vs_custom_int(complex_t, int_t):
  assert np.result_type(complex_t, int_t) == np.dtype(complex_t)
  assert np.result_type(int_t, complex_t) == np.dtype(complex_t)


@pytest.mark.parametrize(
    "dtype, scalar, expected",
    [
        # Custom floats retain precision with Python int/float.
        (bf16, 0, bf16),
        (bf16, 0.0, bf16),
        (f8_e4m3fn, 0, f8_e4m3fn),
        (f8_e4m3fn, 0.0, f8_e4m3fn),
        (f4, 0, f4),
        (f4, 0.0, f4),
        # Custom ints use default precision for Python float/complex.
        (i4, 0, i4),
        (i4, 0.0, np.float64),
        (i4, 0.0j, np.complex128),
        (u4, 0, u4),
        (u4, 0.0, np.float64),
        (u4, 0.0j, np.complex128),
        (i1, 0, i1),
        (i1, 0.0, np.float64),
        (i2, 0.0j, np.complex128),
        # Custom complex retains precision with all Python scalars.
        (bc32, 0, bc32),
        (bc32, 0.0, bc32),
        (bc32, 0.0j, bc32),
        (c32, 0, c32),
        (c32, 0.0, c32),
        (c32, 0.0j, c32),
    ],
)
def test_python_scalars(dtype, scalar, expected):
  # result_type() processes explicit dtypes first, regardless of argument order.
  assert np.result_type(dtype, scalar) == np.dtype(expected)
  assert np.result_type(scalar, dtype) == np.dtype(expected)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS)
def test_custom_float_vs_python_complex(float_t):
  assert np.result_type(float_t, 0.0j) == np.dtype(np.complex64)
  assert np.result_type(0.0j, float_t) == np.dtype(np.complex64)


@pytest.mark.parametrize("int_t", CUSTOM_INTS)
@pytest.mark.parametrize(
    "scalar, concrete, expected",
    [
        (1.0, np.float16, np.float16),
        (1.0, np.float32, np.float32),
        (1.0, np.float64, np.float64),
        (1.0, np.complex64, np.complex64),
        (1.0j, np.float16, np.complex64),
        (1.0j, np.float32, np.complex64),
        (1.0j, np.float64, np.complex128),
        (1.0j, np.complex64, np.complex64),
    ],
)
@pytest.mark.parametrize("form", OPERAND_FORMS)
def test_weak_scalar_stays_weak(int_t, scalar, concrete, expected, form):
  operands = [operand(int_t, form), scalar, operand(concrete, form)]
  for order in itertools.permutations(range(3)):
    args = [operands[i] for i in order]
    # NumPy < 2.6's abstract probe can bypass our CommonDType in this order.
    if _NUMPY_LT_2_6 and form not in ["type", "dtype"] and order == (1, 2, 0):
      with pytest.raises(np.exceptions.DTypePromotionError):
        np.result_type(*args)
    else:
      assert np.result_type(*args) == np.dtype(expected)


@pytest.mark.parametrize("dtype", CUSTOM_DTYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("form", OPERAND_FORMS)
@pytest.mark.parametrize(
    "scalar", [1, 1.0, 1.0j], ids=["int", "float", "complex"]
)
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["custom_first", "python_first"]
)
def test_python_scalar_operand_forms(dtype, form, scalar, reverse):
  custom = operand(dtype, form)
  if dtype in CUSTOM_FLOATS:
    expected = np.complex64 if isinstance(scalar, complex) else dtype
  elif dtype in CUSTOM_INTS:
    expected = {int: dtype, float: np.float64, complex: np.complex128}[
        type(scalar)
    ]
  else:
    expected = dtype

  # NumPy < 2.6 probes uint8/float16/complex64 when a Python scalar precedes
  # an instance; explicit type/dtype arguments are processed first.
  if _NUMPY_LT_2_6 and reverse and form not in ["type", "dtype"]:
    if dtype in CUSTOM_INTS:
      expected = {
          int: np.int16 if dtype in [i1, i2, i4] else np.uint8,
          float: np.float16,
          complex: np.complex64,
      }[type(scalar)]
    elif dtype in CUSTOM_FLOATS and isinstance(scalar, float):
      expected = np.float32
    elif dtype in CUSTOM_COMPLEX and not isinstance(scalar, int):
      expected = (
          c32 if dtype is c32 and isinstance(scalar, float) else np.complex64
      )
  args = (scalar, custom) if reverse else (custom, scalar)
  assert np.result_type(*args) == np.dtype(expected)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("scalar", [1, 2], ids=["one", "two"])
@pytest.mark.parametrize("reverse", [False, True])
def test_custom_float_array_with_python_int(float_t, scalar, reverse):
  array = np.ones((2, 3), dtype=float_t)
  args = (scalar, array) if reverse else (array, scalar)
  result = np.add(*args)
  assert result.dtype == np.dtype(float_t)
  np.testing.assert_array_equal(
      result.astype(np.float32), result.dtype.type(1 + scalar)
  )


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("form", OPERAND_FORMS)
@pytest.mark.parametrize(
    "concrete, expected",
    [
        (np.float16, np.complex64),
        (np.float32, np.complex64),
        (np.float64, np.complex128),
        (np.longdouble, np.clongdouble),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
        (np.clongdouble, np.clongdouble),
    ],
)
def test_custom_float_with_python_complex_and_concrete(
    float_t, form, concrete, expected
):
  operands = [operand(float_t, form), 1j, operand(concrete, form)]
  for order in itertools.permutations(range(3)):
    args = [operands[i] for i in order]
    # NumPy < 2.6's abstract probe can bypass our CommonDType in this order.
    if _NUMPY_LT_2_6 and form not in ["type", "dtype"] and order == (1, 2, 0):
      with pytest.raises(np.exceptions.DTypePromotionError):
        np.result_type(*args)
    else:
      assert np.result_type(*args) == np.dtype(expected)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("int_t", CUSTOM_INTS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("form", OPERAND_FORMS)
def test_custom_float_with_custom_int_and_python_complex(float_t, int_t, form):
  for args in itertools.permutations(
      [operand(float_t, form), operand(int_t, form), 1j]
  ):
    assert np.result_type(*args) == np.dtype(np.complex64)


@pytest.mark.parametrize("custom_t", CUSTOM_DTYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize(
    "incompatible",
    [
        np.dtype("U8"),
        np.dtype("S8"),
        np.dtype("V8"),
        np.dtype("M8[D]"),
        np.dtype("m8[s]"),
        np.dtypes.StringDType(),
    ],
    ids=str,
)
def test_incompatible_dtypes(custom_t, incompatible):
  for args in [(custom_t, incompatible), (incompatible, custom_t)]:
    with pytest.raises(np.exceptions.DTypePromotionError):
      np.result_type(*args)


@pytest.mark.parametrize("a, b", list(itertools.combinations(CUSTOM_INTS, 2)))
@pytest.mark.parametrize("c", [np.int8, bf16])
def test_custom_int_argument_permutations(a, b, c):
  if c is np.int8 and a in [u1, u2, u4] and b in [u1, u2, u4]:
    expected = np.dtype(np.int16)
  else:
    expected = np.dtype(c)
  for args in itertools.permutations([a, b, c]):
    assert np.result_type(*args) == expected


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("int_t", [np.int8, np.uint8])
def test_legacy_byte_integer_promotions(float_t, int_t):
  # These promotions are lossy for narrow floats. They match the safe-cast
  # table used by NumPy's uint8 probe, allowing array + 1 to retain its dtype.
  assert np.result_type(float_t, int_t) == np.dtype(float_t)
  assert np.result_type(int_t, float_t) == np.dtype(float_t)


@pytest.mark.parametrize(
    "a, b", list(itertools.combinations(CUSTOM_DTYPES + NUMPY_DTYPES, 2))
)
def test_numeric_pair_commutativity(a, b):
  assert np.result_type(a, b) == np.result_type(b, a)


@pytest.mark.parametrize(
    "a, b", list(itertools.combinations_with_replacement(CUSTOM_DTYPES, 2))
)
def test_promoted_custom_pairs_support_array_operations(a, b):
  common = np.result_type(a, b)
  assert np.can_cast(a, common, casting="safe")
  assert np.can_cast(b, common, casting="safe")
  arrays = [np.ones(2, dtype=a), np.ones(2, dtype=b)]
  assert np.concatenate(arrays).dtype == common
  result = np.add(*arrays)
  assert result.dtype == common
  if common.kind == "c" or common.type in CUSTOM_COMPLEX:
    reference = np.complex128
  elif common.kind in "iu" or common.type in CUSTOM_INTS:
    reference = np.int64
  else:
    reference = np.float64
  expected = np.add(*(array.astype(reference) for array in arrays)).astype(
      common
  )
  np.testing.assert_array_equal(
      result.astype(reference), expected.astype(reference)
  )


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (f8_e4m3, f8_e4m3fn, np.float32),
        (f8_e4m3fn, f8_e4m3, np.float32),
        (f8_e3m4, f8_e5m2, np.float32),
        (i2, i4, np.int8),
        (u2, u4, np.uint8),
        (i2, u2, np.int8),
    ],
)
def test_ufunc_loop_selection_matches_result_type(a, b, expected):
  assert np.add(np.ones(2, a), np.ones(2, b)).dtype == np.dtype(expected)
  assert np.add(np.ones(2, b), np.ones(2, a)).dtype == np.dtype(expected)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
def test_custom_float_with_float16_array_operations(float_t):
  expected = np.float32
  arrays = [np.ones(2, float_t), np.ones(2, np.float16)]
  assert np.result_type(*arrays) == np.dtype(expected)
  assert np.concatenate(arrays).dtype == np.dtype(expected)
  assert np.add(*arrays).dtype == np.dtype(expected)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("reverse", [False, True])
def test_custom_float_array_with_python_float(float_t, reverse):
  array = np.ones((2, 3), dtype=float_t)
  args = (1.0, array) if reverse else (array, 1.0)
  result = np.add(*args)
  assert result.dtype == np.dtype(float_t)
  np.testing.assert_array_equal(result, 2)


@pytest.mark.parametrize("a, b", list(itertools.combinations(CUSTOM_FLOATS, 2)))
def test_custom_float_argument_permutations_with_float16(a, b):
  # The legacy cast table omits safe casts to float16, even for formats that fit.
  for args in itertools.permutations([a, b, np.float16]):
    assert np.result_type(*args) == np.dtype(np.float32)


@pytest.mark.parametrize(
    "a, b",
    list(
        itertools.combinations_with_replacement(CUSTOM_FLOATS + CUSTOM_INTS, 2)
    ),
)
@pytest.mark.parametrize("form", ["dtype", "array"])
def test_custom_real_argument_permutations_with_bfloat16(a, b, form):
  # bfloat16 contains pairs that would otherwise promote to float32.
  operands = [operand(dtype, form) for dtype in [a, b, bf16]]
  for args in itertools.permutations(operands):
    assert np.result_type(*args) == np.dtype(bf16)


@pytest.mark.parametrize("wider", [f8_e4m3, f8_e4m3fn, f8_e4m3fnuz, f8_e4m3b11])
@pytest.mark.parametrize("form", OPERAND_FORMS)
def test_containing_float8_argument_permutations(wider, form):
  operands = [operand(dtype, form) for dtype in [f6_e2m3, f6_e3m2, wider]]
  for args in itertools.permutations(operands):
    assert np.result_type(*args) == np.dtype(wider)


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("int_t", [np.int16, np.uint16])
@pytest.mark.parametrize("form", OPERAND_FORMS)
def test_custom_float_with_numpy_int16(float_t, int_t, form):
  operands = [operand(dtype, form) for dtype in [float_t, int_t]]
  for args in itertools.permutations(operands):
    assert np.result_type(*args) == np.dtype(np.float32)
    if form == "array":
      assert np.concatenate(args).dtype == np.dtype(np.float32)
      assert np.add(*args).dtype == np.dtype(np.float32)


@pytest.mark.parametrize("complex_t", CUSTOM_COMPLEX, ids=lambda t: t.__name__)
@pytest.mark.parametrize("int_t", [np.int8, np.uint8])
def test_custom_complex_with_byte_integer_values(complex_t, int_t):
  info = np.iinfo(int_t)
  integers = np.arange(info.min, info.max + 1, dtype=np.int16).astype(int_t)
  converted = integers.astype(complex_t, casting="safe")
  np.testing.assert_array_equal(converted.astype(np.complex64), integers)
  complex_values = np.full(integers.shape, 2 + 3j, dtype=complex_t)
  for args in [(complex_values, integers), (integers, complex_values)]:
    assert np.result_type(*args) == np.dtype(complex_t)
    assert np.concatenate(args).dtype == np.dtype(complex_t)
    result = np.add(*args)
    assert result.dtype == np.dtype(complex_t)
    expected = (complex_values.astype(np.complex64) + integers).astype(
        complex_t
    )
    np.testing.assert_array_equal(
        result.astype(np.complex64), expected.astype(np.complex64)
    )


@pytest.mark.parametrize("float_t", CUSTOM_FLOATS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("int_t", CUSTOM_INTS, ids=lambda t: t.__name__)
def test_custom_float_int_promotion_is_lossless(float_t, int_t):
  info = ml_dtypes.iinfo(int_t)
  values = np.arange(info.min, info.max + 1, dtype=np.int8)
  with np.errstate(invalid="ignore"):
    converted = values.astype(float_t).astype(np.float64)
  expected = float_t if np.array_equal(values, converted) else np.float32
  assert np.result_type(float_t, int_t) == np.dtype(expected)
  assert np.result_type(int_t, float_t) == np.dtype(expected)
  integers = values.astype(int_t)
  np.testing.assert_array_equal(
      integers.astype(expected, casting="safe").astype(np.float64), values
  )


@pytest.mark.parametrize(
    "complex_t, expected", [(bc32, np.complex64), (c32, c32)]
)
def test_custom_complex_with_float16_array_operations(complex_t, expected):
  arrays = [np.ones(2, complex_t), np.ones(2, np.float16)]
  assert np.result_type(*arrays) == np.dtype(expected)
  assert np.concatenate(arrays).dtype == np.dtype(expected)
  assert np.add(*arrays).dtype == np.dtype(expected)


def _numpy26_integer_float_triples():
  # Without NumPy's reduction fix, intN + unsigned can widen the intermediate
  # integer and float, depending on argument order.
  for int_t in [i1, i2, i4]:
    for unsigned, concrete in [
        (np.uint8, np.float16),
        (np.uint16, np.float32),
        (np.uint16, np.complex64),
    ]:
      yield int_t, unsigned, concrete


def _numeric_triple_parameters():
  # The NumPy 2.6 regressions have their own expected-result tests below.
  numpy26_cases = {
      frozenset(triple) for triple in _numpy26_integer_float_triples()
  }
  numeric_types = CUSTOM_DTYPES + [dt.type for dt in NUMPY_DTYPES]
  for triple in itertools.combinations_with_replacement(numeric_types, 3):
    if (
        not any(t in CUSTOM_DTYPES for t in triple)
        or frozenset(triple) in numpy26_cases
    ):
      continue
    yield pytest.param(triple, id="-".join(np.dtype(t).name for t in triple))


@pytest.mark.parametrize("dtypes", list(_numeric_triple_parameters()))
def test_numeric_triple_argument_permutations(dtypes):
  operands = [np.dtype(dtype) for dtype in dtypes]
  expected = np.result_type(*operands)
  for args in itertools.permutations(operands):
    assert np.result_type(*args) == expected, args


@pytest.mark.parametrize("dtypes", list(_numpy26_integer_float_triples()))
@pytest.mark.parametrize("form", OPERAND_FORMS)
@pytest.mark.xfail(
    _NUMPY_MULTI_DTYPE_PROMOTION_BROKEN,
    strict=True,
    raises=AssertionError,
    reason="NumPy reduction can discard unsigned inputs before intN promotion",
)
def test_signed_int_unsigned_float_argument_permutations(dtypes, form):
  operands = [operand(dtype, form) for dtype in dtypes]
  expected = np.dtype(dtypes[-1])
  for order in itertools.permutations(range(3)):
    args = [operands[i] for i in order]
    assert np.result_type(*args) == expected, f"operand order: {order}"
    if form == "array":
      assert np.concatenate(args).dtype == expected, f"operand order: {order}"


@pytest.mark.parametrize(
    "dtypes",
    list(itertools.combinations(CUSTOM_DTYPES, 4)),
    ids=lambda dtypes: "-".join(np.dtype(t).name for t in dtypes),
)
def test_custom_quadruple_argument_permutations(dtypes):
  # Incomparable formats must defer consistently for four-input reductions.
  operands = [operand(dtype, "array") for dtype in dtypes]
  expected = np.result_type(*operands)
  for order in itertools.permutations(range(4)):
    args = [operands[i] for i in order]
    assert np.result_type(*args) == expected, order
    assert np.concatenate(args).dtype == expected, order


@pytest.mark.parametrize(
    "dtypes",
    [
        [f4, f6_e2m3, f8_e8m0, bf16],
        [f4, f4, f8_e8m0, bf16],
        [f4, f4, bf16, bf16],
        [bf16] * 4,
    ],
)
@pytest.mark.parametrize("form", OPERAND_FORMS)
def test_containing_bfloat16_argument_permutations_with_four_inputs(
    dtypes, form
):
  operands = [operand(dtype, form) for dtype in dtypes]
  for args in itertools.permutations(operands):
    assert np.result_type(*args) == np.dtype(bf16)


@pytest.mark.parametrize(
    "custom_types, reference_dtype",
    [
        (CUSTOM_FLOATS, np.float32),
        ([i1, i2, i4], np.int32),
        ([u1, u2, u4], np.uint32),
        (CUSTOM_COMPLEX, np.complex64),
    ],
)
def test_dtype_meta_abstract_base_classes(custom_types, reference_dtype):
  expected_bases = type(np.dtype(reference_dtype)).__bases__
  for custom_type in custom_types:
    dtype_cls = type(np.dtype(custom_type))
    assert dtype_cls.__bases__ == expected_bases
    assert dtype_cls.__module__ == "ml_dtypes"
    assert dtype_cls.__name__ == custom_type.__name__
    assert dtype_cls() == np.dtype(custom_type)
    assert dtype_cls.__new__(dtype_cls) == np.dtype(custom_type)
    with pytest.raises(TypeError):
      dtype_cls(custom_type)


@pytest.mark.skipif(
    not hasattr(np.dtypes, "FloatingAbstractDType"),
    reason="np.isdtype abstract DType hierarchy requires NumPy >= 2.6",
)
@pytest.mark.parametrize(
    "custom_types, true_kinds",
    [
        (CUSTOM_FLOATS, {"real floating", "numeric"}),
        ([i1, i2, i4], {"signed integer", "integral", "numeric"}),
        ([u1, u2, u4], {"unsigned integer", "integral", "numeric"}),
        (CUSTOM_COMPLEX, {"complex floating", "numeric"}),
    ],
)
def test_isdtype_abstract_hierarchy(custom_types, true_kinds):
  all_kinds = (
      "bool",
      "signed integer",
      "unsigned integer",
      "integral",
      "real floating",
      "complex floating",
      "numeric",
  )
  for custom_type in custom_types:
    for arg in (custom_type, np.dtype(custom_type)):
      for kind in all_kinds:
        assert np.isdtype(arg, kind) == (kind in true_kinds)


def test_ufunc_promoters_with_forced_dtype_and_stringdtype():
  x = np.array([1.5, 2.5], dtype=bf16)
  out = np.add(x, x, dtype=np.float32)
  assert out.dtype == np.dtype(np.float32)
  np.testing.assert_array_equal(out, np.array([3.0, 5.0], dtype=np.float32))

  assert np.ldexp(ml_dtypes.int4(1), ml_dtypes.int4(2)) == np.float16(4.0)

  s = np.array(["ab", "c"], dtype=np.dtypes.StringDType())
  n = np.array([2, 3], dtype=ml_dtypes.int4)
  with pytest.raises(TypeError):
    _ = s + n
  with pytest.raises(TypeError):
    _ = s * n
