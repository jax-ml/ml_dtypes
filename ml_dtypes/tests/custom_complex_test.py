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

"""Test cases for custom complex types."""

import operator
import pickle

import ml_dtypes
import numpy as np
import pytest


ComplexWarning = getattr(np, "exceptions", np).ComplexWarning


COMPLEX_SCTYPES = [ml_dtypes.complex32, ml_dtypes.bcomplex32]

# Test values that should round trip
COMPLEX_VALUES = [
    0j,
    1 + 0j,
    0 + 1j,
    1 + 1j,
    -1 - 1j,
    0.5 + 0.5j,
    2.0 - 3.0j,
    complex("nan"),
    complex("inf"),
    complex("-inf"),
    complex(1, np.inf),
    complex(0, -np.nan),
    complex(1, np.nan),
]


def assert_expected_dtype(result, expected, sctype):
  if expected.dtype == np.complex64:
    assert result.dtype == sctype
  elif expected.dtype == np.bool_:
    assert result.dtype == np.bool_
  elif sctype == ml_dtypes.complex32:
    assert result.dtype == np.float16
  elif sctype == ml_dtypes.bcomplex32:
    assert result.dtype == ml_dtypes.bfloat16
  else:
    raise AssertionError("Unexpected sctype")


# =================================
# Scalar and basic Python API tests
# =================================


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_module_name(sctype):
  assert sctype.__module__ == "ml_dtypes"


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_dtype_from_string(sctype):
  """Test creating dtype from string name."""
  assert np.dtype(sctype.__name__) == np.dtype(sctype)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_pickleable(sctype):
  # Create complex array from real and imaginary parts
  x = np.asarray(COMPLEX_VALUES, dtype=sctype)
  x_out = pickle.loads(pickle.dumps(x))
  assert x_out.dtype == x.dtype
  np.testing.assert_array_equal(x_out.astype("complex"), x.astype("complex"))


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_round_trip_to_complex(sctype, value):
  res = complex(sctype(value))
  np.testing.assert_equal(value, res)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_constructor(sctype):
  # Test that the constructor accepts the typical things (two arguments, string)
  z = sctype(1, 3)
  assert z == 1 + 3j
  z = sctype("1+3j")
  assert z == 1 + 3j


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_float_and_int_conversion(sctype):
  z = sctype(1.5)
  with pytest.warns(ComplexWarning):
    assert float(z) == 1.5
  with pytest.warns(ComplexWarning):
    assert int(z) == 1


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_real_imag_scalars(sctype):
  # real and image works on the scalar
  z = sctype(3 + 4j)
  assert z.real == 3.0
  assert z.imag == 4.0


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_real_imag_arrays(sctype):
  # Test ml_dtypes.real() and ml_dtypes.imag() helpers.
  arr = np.array([1 + 2j, 3 + 4j], dtype=sctype)
  real_part = ml_dtypes.real(arr)
  imag_part = ml_dtypes.imag(arr)
  expected_dtype = ml_dtypes.finfo(sctype).dtype  # the real one
  assert real_part.dtype == imag_part.dtype == expected_dtype
  np.testing.assert_array_equal(real_part, [1.0, 3.0])
  np.testing.assert_array_equal(imag_part, [2.0, 4.0])


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_str_repr(sctype, value):
  z = sctype(value)
  assert str(z) == str(value)
  assert repr(z) == str(value)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_hash(sctype, value):
  # Test that we hash the same as NumPy (except for NaN)
  if not np.isnan(value):
    assert hash(sctype(value)) == hash(value)
  else:
    assert hash(sctype(value)) != hash(value)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_negate(sctype):
  """Test unary negation."""
  values = [3 + 4j, -1.5 - 2.5j, 0 + 1j, 1 + 0j]
  for val in values:
    result = -sctype(val)
    expected = sctype(-np.complex64(val))
    np.testing.assert_allclose(
        complex(result), complex(expected), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "op,a,b",
    [
        (operator.add, 1.5 + 2.5j, 3.25 + 4.75j),
        (operator.add, -2.5 + 1.5j, 1.25 - 3.75j),
        (operator.sub, 5.5 + 6.5j, 2.25 + 3.25j),
        (operator.sub, 1 + 1j, 2 + 2j),
        (operator.mul, 2.5 + 1.5j, 1.25 + 2.75j),
        (operator.mul, 3 + 4j, 1 - 1j),
        (operator.truediv, 6 + 8j, 2 + 0j),
        (operator.truediv, 10 + 5j, 2 + 1j),
    ],
)
def test_binary_operators(sctype, op, a, b):
  """Test binary operators: result should match complex64 computation."""
  result = op(sctype(a), sctype(b))
  expected = sctype(op(np.complex64(a), np.complex64(b)))
  assert type(result) is sctype  # pylint: disable=unidiomatic-typecheck

  np.testing.assert_allclose(
      complex(result), complex(expected), rtol=1e-2, atol=1e-2
  )


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
@pytest.mark.parametrize("other", COMPLEX_VALUES)
def test_comparisons(sctype, op, other):
  """Test equality comparisons."""
  val = sctype(1 + 2j)
  other = sctype(other)
  result = op(val, other)
  # NOTE(seberg): This compares with arrays, NumPy is actually doing a different
  # thing for scalars. (Does not honor imaginary NaN necessarily.)
  with np.errstate(invalid="ignore"):
    expected = op(np.complex64([val]), np.complex64([other]))[0]
  assert result == expected


# =============
# Casting tests
# =============


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize("to_dtype", [np.complex64, np.complex128])
def test_casts_to_complex(sctype, to_dtype):
  """Test casting to/from standard complex types."""
  x = np.array([1 + 2j, 3 + 4j], dtype=sctype)
  result = x.astype(to_dtype)
  np.testing.assert_array_equal(x, result)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "from_dtype", [np.float32, np.float64, ml_dtypes.float4_e2m1fn]
)
def test_cast_from_float(sctype, from_dtype):
  """Test casting from real to complex."""
  x = np.array([1.0, 2.0, 3.0], dtype=from_dtype)
  y = x.astype(sctype)
  assert y.dtype == sctype
  np.testing.assert_array_equal(ml_dtypes.real(y).astype(np.float32), x)
  np.testing.assert_array_equal(ml_dtypes.imag(y).astype(np.float32), 0.0)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "to_dtype", [np.float32, np.float64, ml_dtypes.float4_e2m1fn]
)
def test_cast_to_float(sctype, to_dtype):
  """Test casting from complex to real (should take real part)."""
  # Make large, so that NumPy may release the GIL.
  x = np.array([1 + 2j, 3 + 4j] * 500, dtype=sctype)
  with pytest.warns(ComplexWarning):
    y = x.astype(to_dtype)
  np.testing.assert_array_equal(y, [1.0, 3.0] * 500)


# ==========================
# UFunc/array function tests
# ==========================


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "func",
    [
        np.fabs,
        np.cbrt,
        np.floor,
        np.ceil,
        np.trunc,
        np.deg2rad,
        np.rad2deg,
        np.spacing,
        np.signbit,
        np.modf,
        np.frexp,
    ],
)
def test_unimplemented_ufuncs(sctype, func):
  x = np.array([1 + 2j, 3 + 4j], dtype=sctype)
  with pytest.raises(TypeError):
    func(x)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "func",
    [
        np.remainder,
        np.fmod,
        np.floor_divide,
        np.arctan2,
        np.hypot,
        np.logaddexp,
        np.logaddexp2,
        np.copysign,
        np.ldexp,
        np.nextafter,
    ],
)
def test_unimplemented_binary_ufuncs(sctype, func):
  x = np.array([1 + 2j, 3 + 4j], dtype=sctype)
  with pytest.raises(TypeError):
    func(x, x)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "ufunc",
    [
        # Basic operations
        np.negative,
        np.positive,
        np.conjugate,
        # Absolute value (returns real)
        np.absolute,
        # Exponential and logarithmic
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        # Power and roots
        np.sqrt,
        np.square,
        np.reciprocal,
        # Rounding
        np.rint,
        # Sign
        pytest.param(
            np.sign,
            marks=pytest.mark.xfail(
                condition=np.__version__.startswith("1."),
                reason="definition fixed 2.0",
            ),
        ),
        # Trigonometric
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        # Hyperbolic
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        # Logical and predicate functions
        np.logical_not,
        np.isfinite,
        np.isinf,
        np.isnan,
    ],
)
@np.errstate(all="ignore")
def test_unary_ufuncs(sctype, ufunc):
  """Test all unary ufuncs, we expect them to just use the float32 version."""
  x = np.array(COMPLEX_VALUES)
  x = np.concatenate(
      [x, np.random.random(20).astype(np.float32).view(np.complex64)]
  )
  x = x.astype(sctype)

  result = ufunc(x)
  expected = ufunc(x.astype(np.complex64))

  assert_expected_dtype(result, expected, sctype)
  if ufunc in [np.arctan, np.arctanh]:
    # Arctan/arctanh seems to differe a bit with Inf/Nan results
    assert (np.isnan(expected) == np.isnan(result)).all()
    assert (np.isinf(expected) == np.isinf(result)).all()
    finite = np.isfinite(expected)
    expected = expected[finite]
    result = result[finite]

  # Most ufuncs should match exactly. We compare in NumPy dtype
  # (but cast expected to lower precision once)
  dtype = expected.dtype
  expected = expected.astype(result.dtype).astype(dtype)
  np.testing.assert_array_equal(result.astype(dtype), expected)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
@pytest.mark.parametrize(
    "ufunc",
    [
        np.multiply,
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.true_divide,
        np.power,
        # Maximum and minimum
        np.maximum,
        np.minimum,
        np.fmax,
        np.fmin,
        # comparisons and logical ufuncs:
        np.equal,
        np.not_equal,
        np.less,
        np.greater,
        np.less_equal,
        np.greater_equal,
        np.logical_and,
        np.logical_or,
        np.logical_xor,
    ],
)
@np.errstate(all="ignore")
def test_binary_ufuncs(sctype, ufunc):
  """Test binary ufuncs."""
  x = np.array(COMPLEX_VALUES)
  x = np.concatenate(
      [x, np.random.random(20).astype(np.float32).view(np.complex64)]
  )
  x = x.astype(sctype)

  if ufunc == np.power:
    # TODO(seberg): std::power deals poorly with some values, drop for now.
    x = x[(ml_dtypes.real(x) != 0) & np.isfinite(x)]

  y = x[:, np.newaxis]

  result = ufunc(x, y)
  expected = ufunc(x.astype(np.complex64), y.astype(np.complex64))

  assert_expected_dtype(result, expected, sctype)
  if ufunc in [np.multiply, np.divide, np.true_divide, np.power]:
    np.testing.assert_allclose(
        result.astype(np.complex64),
        expected,
        rtol=float(ml_dtypes.finfo(sctype).eps),
    )
  else:
    # Most ufuncs should match exactly. We compare in NumPy dtype
    # (but cast expected to lower precision once)
    dtype = expected.dtype
    expected = expected.astype(result.dtype).astype(dtype)
    np.testing.assert_array_equal(result.astype(dtype), expected)


@pytest.mark.parametrize("sctype", COMPLEX_SCTYPES)
def test_dot_product(sctype):
  """Test dot product."""
  x = np.array([1 + 1j, 2 + 2j], dtype=sctype)
  y = np.array([1 - 1j, 2 - 2j], dtype=sctype)
  result = np.dot(x, y)
  expected = np.dot(x.astype(np.complex64), y.astype(np.complex64))
  np.testing.assert_allclose(complex(result), complex(expected), rtol=1e-2)
