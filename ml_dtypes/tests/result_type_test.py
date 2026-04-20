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

"""Tests for np.result_type() across ml_dtypes custom DTypes."""

import ml_dtypes
import numpy as np
import pytest


bf16 = ml_dtypes.bfloat16
f4 = ml_dtypes.float4_e2m1fn
f8_e4m3 = ml_dtypes.float8_e4m3fn
f8_e5m2 = ml_dtypes.float8_e5m2
bc32 = ml_dtypes.bcomplex32
c32 = ml_dtypes.complex32
i4 = ml_dtypes.int4
ui4 = ml_dtypes.uint4
i2 = ml_dtypes.int2


def rt(a, b):
  return np.result_type(a, b)


# ---------------------------------------------------------------------------
# Custom float + NumPy built-in
# ---------------------------------------------------------------------------

class TestCustomFloatVsNumpy:

  def test_float8_plus_float16_gives_float16(self):
    assert rt(f8_e4m3, np.float16) == np.dtype(np.float16)

  def test_bfloat16_plus_float16_gives_float32(self):
    # Neither fits the other: bfloat16 has 7 mantissa bits, float16 has wider
    # exponent; float32 contains both.
    assert rt(bf16, np.float16) == np.dtype(np.float32)

  def test_float8_plus_float32_gives_float32(self):
    assert rt(f8_e4m3, np.float32) == np.dtype(np.float32)

  def test_float8_plus_float64_gives_float64(self):
    assert rt(f8_e4m3, np.float64) == np.dtype(np.float64)

  def test_float8_plus_bool_gives_float8(self):
    assert rt(f8_e4m3, np.bool_) == np.dtype(f8_e4m3)

  def test_bfloat16_plus_int8_gives_bfloat16(self):
    # bfloat16 has enough precision to represent all int8 values.
    assert rt(bf16, np.int8) == np.dtype(bf16)

  def test_float8_plus_int8_gives_float64(self):
    # float8 cannot represent all int8 values; PyArray_CommonDType defers
    # to the integer's required precision → float64.
    assert rt(f8_e4m3, np.int8) == np.dtype(np.float64)

  def test_float8_plus_complex64_gives_complex64(self):
    assert rt(f8_e4m3, np.complex64) == np.dtype(np.complex64)


# ---------------------------------------------------------------------------
# Custom float + custom float
# ---------------------------------------------------------------------------

class TestCustomFloatVsCustomFloat:

  def test_float8_fits_in_bfloat16(self):
    # float8_e4m3fn has fewer bits than bfloat16 in every dimension.
    assert rt(f8_e4m3, bf16) == np.dtype(bf16)

  def test_float8_e5m2_fits_in_bfloat16(self):
    # float8_e5m2 has less precision but same exponent range; bfloat16 wins.
    assert rt(f8_e5m2, bf16) == np.dtype(bf16)

  def test_float4_fits_in_float8(self):
    # float4_e2m1fn has fewer exp and mantissa bits than float8_e4m3fn.
    assert rt(f4, f8_e4m3) == np.dtype(f8_e4m3)

  def test_float8_e4m3_vs_float8_e5m2_gives_float32(self):
    # Incomparable: e4m3 has more mantissa, e5m2 has more exponent → float32.
    assert rt(f8_e4m3, f8_e5m2) == np.dtype(np.float32)

  def test_same_type_gives_same_type(self):
    assert rt(f8_e4m3, f8_e4m3) == np.dtype(f8_e4m3)
    assert rt(bf16, bf16) == np.dtype(bf16)


# ---------------------------------------------------------------------------
# Custom float + custom int
# ---------------------------------------------------------------------------

class TestCustomFloatVsCustomInt:

  def test_float_beats_int4(self):
    assert rt(f8_e4m3, i4) == np.dtype(f8_e4m3)
    assert rt(bf16, i4) == np.dtype(bf16)

  def test_symmetry(self):
    assert rt(i4, f8_e4m3) == rt(f8_e4m3, i4)
    assert rt(i4, bf16) == rt(bf16, i4)


# ---------------------------------------------------------------------------
# Custom int + NumPy built-in
# ---------------------------------------------------------------------------

class TestCustomIntVsNumpy:

  def test_int4_plus_bool_gives_int4(self):
    assert rt(i4, np.bool_) == np.dtype(i4)

  def test_int4_plus_int8_gives_int8(self):
    assert rt(i4, np.int8) == np.dtype(np.int8)

  def test_int4_plus_int16_gives_int16(self):
    assert rt(i4, np.int16) == np.dtype(np.int16)

  def test_int4_plus_float16_gives_float16(self):
    assert rt(i4, np.float16) == np.dtype(np.float16)

  def test_int4_plus_float32_gives_float32(self):
    assert rt(i4, np.float32) == np.dtype(np.float32)

  def test_int4_plus_complex64_gives_complex64(self):
    assert rt(i4, np.complex64) == np.dtype(np.complex64)

  def test_symmetry(self):
    for numpy_t in [np.int8, np.int16, np.float32, np.complex64]:
      assert rt(i4, numpy_t) == rt(numpy_t, i4)


# ---------------------------------------------------------------------------
# Custom int + custom int
# ---------------------------------------------------------------------------

class TestCustomIntVsCustomInt:

  def test_int4_plus_uint4_gives_int16(self):
    # Signed + unsigned 4-bit: neither fits the other → int16.
    assert rt(i4, ui4) == np.dtype(np.int16)

  def test_int2_plus_int4_gives_int16(self):
    # int2 < int4 by type_num; int4 handles and int2 fits in int4 → int16?
    # Actually both are narrow custom ints; falls back to int16.
    assert rt(ml_dtypes.int2, i4) == np.dtype(np.int16)

  def test_same_type_gives_same_type(self):
    assert rt(i4, i4) == np.dtype(i4)


# ---------------------------------------------------------------------------
# Custom complex + NumPy built-in
# ---------------------------------------------------------------------------

class TestCustomComplexVsNumpy:

  def test_bcomplex32_plus_bool_gives_cfloat(self):
    assert rt(bc32, np.bool_) == np.dtype(np.complex64)

  def test_bcomplex32_plus_int8_gives_cfloat(self):
    assert rt(bc32, np.int8) == np.dtype(np.complex64)

  def test_bcomplex32_plus_float16_gives_cfloat(self):
    assert rt(bc32, np.float16) == np.dtype(np.complex64)

  def test_bcomplex32_plus_float32_gives_cfloat(self):
    assert rt(bc32, np.float32) == np.dtype(np.complex64)

  def test_bcomplex32_plus_float64_gives_cdouble(self):
    assert rt(bc32, np.float64) == np.dtype(np.complex128)

  def test_bcomplex32_plus_cfloat_gives_cfloat(self):
    assert rt(bc32, np.complex64) == np.dtype(np.complex64)

  def test_bcomplex32_plus_cdouble_gives_cdouble(self):
    assert rt(bc32, np.complex128) == np.dtype(np.complex128)

  def test_symmetry(self):
    for numpy_t in [np.float32, np.float64, np.complex64, np.complex128]:
      assert rt(bc32, numpy_t) == rt(numpy_t, bc32)


# ---------------------------------------------------------------------------
# Custom complex + custom float / int
# ---------------------------------------------------------------------------

class TestCustomComplexVsCustom:

  def test_bcomplex32_plus_bfloat16_gives_cfloat(self):
    assert rt(bc32, bf16) == np.dtype(np.complex64)

  def test_bcomplex32_plus_float8_gives_cfloat(self):
    assert rt(bc32, f8_e4m3) == np.dtype(np.complex64)

  def test_bcomplex32_plus_int4_gives_cfloat(self):
    assert rt(bc32, i4) == np.dtype(np.complex64)

  def test_bcomplex32_plus_complex32_gives_cfloat(self):
    assert rt(bc32, c32) == np.dtype(np.complex64)

  def test_same_type_gives_same_type(self):
    assert rt(bc32, bc32) == np.dtype(bc32)
    assert rt(c32, c32) == np.dtype(c32)


# ---------------------------------------------------------------------------
# Python scalars (abstract types: 0, 0.0, 0.0j)
# ---------------------------------------------------------------------------

class TestPythonScalars:
  """Concrete custom DTypes should dominate abstract Python scalar types."""

  @pytest.mark.parametrize("dtype", [f8_e4m3, bf16])
  def test_custom_float_dominates_python_int(self, dtype):
    assert rt(dtype, 0) == np.dtype(dtype)

  @pytest.mark.parametrize("dtype", [f8_e4m3, bf16])
  def test_custom_float_dominates_python_float(self, dtype):
    assert rt(dtype, 0.0) == np.dtype(dtype)

  @pytest.mark.parametrize("dtype", [f8_e4m3, bf16])
  def test_custom_float_plus_python_complex_gives_cfloat(self, dtype):
    # Abstract complex + custom float → cfloat (smallest complex containing both).
    assert rt(dtype, 0.0j) == np.dtype(np.complex64)

  @pytest.mark.parametrize("dtype", [i4, ui4])
  def test_custom_int_dominates_python_int(self, dtype):
    assert rt(dtype, 0) == np.dtype(dtype)

  @pytest.mark.parametrize("dtype", [i4, ui4])
  def test_custom_int_dominates_python_float(self, dtype):
    assert rt(dtype, 0.0) == np.dtype(dtype)

  def test_custom_complex_dominates_python_int(self):
    assert rt(bc32, 0) == np.dtype(bc32)

  def test_custom_complex_dominates_python_float(self):
    assert rt(bc32, 0.0) == np.dtype(bc32)

  def test_custom_complex_dominates_python_complex(self):
    assert rt(bc32, 0.0j) == np.dtype(bc32)
