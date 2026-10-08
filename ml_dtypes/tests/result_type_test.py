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

import itertools

import ml_dtypes
import numpy as np
import pytest

# Short aliases for readability in parametrize lists
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
f8_e8m0 = ml_dtypes.float8_e8m0fnu
bc32 = ml_dtypes.bcomplex32
c32 = ml_dtypes.complex32
i1, i2, i4 = ml_dtypes.int1, ml_dtypes.int2, ml_dtypes.int4
u1, u2, u4 = ml_dtypes.uint1, ml_dtypes.uint2, ml_dtypes.uint4

ALL_CUSTOM_FLOATS = [
    bf16,
    f4,
    f6_e2m3,
    f6_e3m2,
    f8_e3m4,
    f8_e4m3,
    f8_e4m3fn,
    f8_e4m3fnuz,
    f8_e4m3b11,
    f8_e5m2,
    f8_e5m2fnuz,
    f8_e8m0,
]
ALL_INTN = [i1, i2, i4, u1, u2, u4]
ALL_CUSTOM_COMPLEX = [bc32, c32]


def rt(a, b):
  return np.result_type(a, b)


# ---------------------------------------------------------------------------
# Custom float vs NumPy built-in types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # ---- bool: custom float always wins ----
        (bf16, np.bool_, bf16),
        (f8_e4m3fn, np.bool_, f8_e4m3fn),
        (f4, np.bool_, f4),
        # ---- floats: pick the wider ----
        (f4, np.float16, np.float16),  # f4 fits in float16
        (f8_e4m3fn, np.float16, np.float16),  # float8 fits in float16
        (f8_e5m2, np.float16, np.float16),  # float8 fits in float16
        (bf16, np.float16, np.float32),  # incomparable → float32
        (f8_e8m0, np.float16, np.float32),  # e8m0 range exceeds float16
        (f8_e4m3fn, np.float32, np.float32),  # all custom floats fit in float32
        (bf16, np.float32, np.float32),
        (bf16, np.float64, np.float64),
        (f8_e4m3fn, np.float64, np.float64),
        # ---- integers: PyArray_CommonDType decides ----
        (
            bf16,
            np.int8,
            bf16,
        ),  # bfloat16 has 8 sig bits, int8 needs 7 → bf16 wins
        (
            bf16,
            np.int16,
            np.float64,
        ),  # bfloat16 has 8 sig bits, int16 needs 15 → float64
        (
            f8_e4m3fn,
            np.int8,
            np.float64,
        ),  # float8 can't represent all int8 values
        (
            f8_e4m3fn,
            np.int32,
            np.float64,
        ),  # float8 can't represent all int32 values
        # ---- complex: other always wins ----
        (bf16, np.complex64, np.complex64),
        (f8_e4m3fn, np.complex64, np.complex64),
        (bf16, np.complex128, np.complex128),
        (f8_e4m3fn, np.complex128, np.complex128),
    ],
)
def test_custom_float_vs_numpy(a, b, expected):
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


# ---------------------------------------------------------------------------
# Custom float vs custom float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # ---- same type ----
        (bf16, bf16, bf16),
        (f8_e4m3fn, f8_e4m3fn, f8_e4m3fn),
        (f4, f4, f4),
        # ---- narrower fits safely into wider ----
        (f4, f6_e2m3, f6_e2m3),  # f4 ⊂ f6_e2m3 (more exp + mantissa)
        (f4, f8_e4m3fn, f8_e4m3fn),  # f4 fits in every float8+
        (f4, bf16, bf16),  # f4 fits in bfloat16
        (f8_e4m3fn, bf16, bf16),  # float8 fits in bfloat16
        (f8_e5m2, bf16, bf16),  # float8 fits in bfloat16
        (f8_e3m4, bf16, bf16),  # float8 fits in bfloat16
        # ---- incomparable: one has more exp, other more mantissa → float32 ----
        (bf16, f8_e5m2, bf16),  # f8_e5m2 fits in bf16 (bf16 > in all dims)
        (
            f8_e4m3fn,
            f8_e5m2,
            np.float32,
        ),  # e4m3 has more mantissa, e5m2 has more exp
        (f8_e4m3, f8_e4m3fn, np.float32),  # infinity vs wider finite range
        (f8_e4m3, f8_e4m3fnuz, np.float32),  # infinity vs wider lower range
        (
            f8_e4m3fn,
            f8_e4m3fnuz,
            np.float32,
        ),  # wider upper vs wider lower range
        (
            f8_e5m2,
            f8_e5m2fnuz,
            np.float32,
        ),  # equal digits/max_exp, neither contains the other
        (f6_e2m3, f6_e3m2, np.float32),  # one has more mantissa, other more exp
    ],
)
def test_custom_float_vs_custom_float(a, b, expected):
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


@pytest.mark.parametrize(
    "a, b", list(itertools.combinations(ALL_CUSTOM_FLOATS, 2))
)
def test_custom_float_commutativity(a, b):
  assert rt(a, b) == rt(b, a)


# ---------------------------------------------------------------------------
# can_cast(..., casting='safe') for custom float pairs
#
# Legacy NumPy infers safe casts for user dtypes with the same kind and
# itemsize (see PyArray_LegacyEquivTypes), so same-sized void floats are
# often reported safe even when semantically lossy.  Different itemsize or
# kind (e.g. float8_e5m2 uses kind 'f') avoids that heuristic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        (f4, f8_e4m3fn),
        (f6_e2m3, f8_e4m3fn),
    ],
)
def test_custom_float_can_cast_safe(a, b):
  assert np.can_cast(a, b, casting="safe")


@pytest.mark.xfail(
    reason="Legacy casts is incorrect as it uses kind+itemsize logic.",
    strict=False,
)
@pytest.mark.parametrize(
    "a, b",
    [
        (f8_e4m3fn, bf16),
        (f8_e3m4, bf16),
    ],
)
def test_custom_float_can_cast_safe_cross_itemsize(a, b):
  assert np.can_cast(a, b, casting="safe")


@pytest.mark.parametrize(
    "a, b",
    [
        (f8_e5m2, f8_e5m2fnuz),
        (f8_e5m2fnuz, f8_e5m2),
        (f8_e4m3fn, f8_e5m2),
        (bf16, f8_e4m3fn),
        (bf16, f8_e5m2),
    ],
)
def test_custom_float_can_cast_not_safe(a, b):
  assert not np.can_cast(a, b, casting="safe")


@pytest.mark.xfail(
    reason="Legacy casts is incorrect as it uses kind+itemsize logic.",
    strict=False,
)
@pytest.mark.parametrize(
    "a, b",
    [
        (f8_e4m3fn, f4),
        (f8_e4m3, f8_e4m3fn),
        (f8_e4m3fn, f8_e4m3),
    ],
)
def test_custom_float_can_cast_legacy_false_positive(a, b):
  assert not np.can_cast(a, b, casting="safe")


# ---------------------------------------------------------------------------
# Custom float vs custom int  (float always dominates)
# ---------------------------------------------------------------------------


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
  assert rt(float_t, int_t) == np.dtype(float_t)
  assert rt(int_t, float_t) == np.dtype(float_t)  # symmetric


# ---------------------------------------------------------------------------
# Custom int vs NumPy built-in types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # ---- bool: custom int always wins ----
        (i4, np.bool_, i4),
        (u4, np.bool_, u4),
        (i1, np.bool_, i1),
        # ---- all other NumPy types: return other (intN is always smaller) ----
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
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


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
  assert rt(signed_int, unsigned_builtin) == np.dtype(expected)
  assert rt(unsigned_builtin, signed_int) == np.dtype(expected)


# ---------------------------------------------------------------------------
# Custom int vs custom int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # ---- same type ----
        (i4, i4, i4),
        (u4, u4, u4),
        # ---- mixed sign: neither fits the other → int16 ----
        (i4, u4, np.int16),
        (i2, u2, np.int16),
        (i1, u1, np.int16),
        # ---- same sign, different width → int16 fallback ----
        (i2, i4, np.int16),
        (u2, u4, np.int16),
        (i1, i4, np.int16),
    ],
)
def test_custom_int_vs_custom_int(a, b, expected):
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


# ---------------------------------------------------------------------------
# Custom complex vs NumPy built-in types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # ---- bool: stay in the custom complex ----
        (bc32, np.bool_, bc32),
        (c32, np.bool_, c32),
        # ---- integers: follow NumPy's integer + complex64 promotion ----
        (bc32, np.int8, np.complex64),
        (c32, np.int8, np.complex64),
        (bc32, np.int32, np.complex128),
        (bc32, np.uint32, np.complex128),
        (c32, np.int32, np.complex128),
        (c32, np.uint32, np.complex128),
        (bc32, np.int64, np.complex128),
        (bc32, np.uint64, np.complex128),
        (c32, np.int64, np.complex128),
        (c32, np.uint64, np.complex128),
        # ---- floats ≤ float32: wrap in cfloat, except a matching component type ----
        (bc32, np.float16, np.complex64),
        (bc32, np.float32, np.complex64),
        (c32, np.float32, np.complex64),
        (c32, np.float16, c32),
        # ---- float64+: need cdouble ----
        (bc32, np.float64, np.complex128),
        (bc32, np.longdouble, np.clongdouble),
        (c32, np.float64, np.complex128),
        # ---- built-in complex: other always wins ----
        (bc32, np.complex64, np.complex64),
        (bc32, np.complex128, np.complex128),
        (c32, np.complex64, np.complex64),
        (c32, np.complex128, np.complex128),
    ],
)
def test_custom_complex_vs_numpy(a, b, expected):
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


# ---------------------------------------------------------------------------
# Custom complex vs custom float / custom int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (bc32, bf16, bc32),
        (bc32, f4, bc32),
        (bc32, f8_e4m3fn, bc32),
        (bc32, f8_e5m2, bc32),
        (c32, f4, c32),
        (c32, f8_e4m3fn, c32),
        (c32, f8_e5m2, c32),
        (c32, bf16, np.complex64),
        # ---- two custom complex types ----
        (bc32, c32, np.complex64),
        (bc32, bc32, bc32),
        (c32, c32, c32),
    ],
)
def test_custom_complex_vs_custom(a, b, expected):
  assert rt(a, b) == np.dtype(expected)
  assert rt(b, a) == np.dtype(expected)  # must be symmetric


@pytest.mark.parametrize("complex_t", ALL_CUSTOM_COMPLEX)
@pytest.mark.parametrize("int_t", ALL_INTN)
def test_custom_complex_vs_custom_int(complex_t, int_t):
  assert rt(complex_t, int_t) == np.dtype(complex_t)
  assert rt(int_t, complex_t) == np.dtype(complex_t)


# ---------------------------------------------------------------------------
# Python scalars: 0, 0.0, 0.0j  (abstract types; value is immaterial)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype, scalar, expected",
    [
        # ---- custom floats dominate Python int and Python float ----
        (bf16, 0, bf16),
        (bf16, 0.0, bf16),
        (f8_e4m3fn, 0, f8_e4m3fn),
        (f8_e4m3fn, 0.0, f8_e4m3fn),
        (f4, 0, f4),
        (f4, 0.0, f4),
        # ---- custom ints: a Python int defers to the int dtype, but a Python
        #      float/complex crosses the integer kind and promotes to the default
        #      float64 / complex128 (matching NumPy's built-in integers) ----
        (i4, 0, i4),
        (i4, 0.0, np.float64),
        (i4, 0.0j, np.complex128),
        (u4, 0, u4),
        (u4, 0.0, np.float64),
        (u4, 0.0j, np.complex128),
        (i1, 0, i1),
        (i1, 0.0, np.float64),
        (i2, 0.0j, np.complex128),
        # ---- custom complex dominates all Python scalars ----
        (bc32, 0, bc32),
        (bc32, 0.0, bc32),
        (bc32, 0.0j, bc32),
        (c32, 0, c32),
        (c32, 0.0, c32),
        (c32, 0.0j, c32),
    ],
)
def test_python_scalars(dtype, scalar, expected):
  assert rt(dtype, scalar) == np.dtype(expected)


@pytest.mark.parametrize("float_t", ALL_CUSTOM_FLOATS)
def test_custom_float_vs_python_complex(float_t):
  assert rt(float_t, 0.0j) == np.dtype(np.complex64)
  assert rt(0.0j, float_t) == np.dtype(np.complex64)


@pytest.mark.parametrize("int_t", ALL_INTN)
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
def test_weak_scalar_stays_weak(int_t, scalar, concrete, expected):
  # Sanity check that no-matter the order the concrete precision wins
  # (i.e. promoting int + pyfloat -> pyfloat).
  for args in itertools.permutations([int_t, scalar, concrete]):
    assert np.result_type(*args) == np.dtype(expected), args
