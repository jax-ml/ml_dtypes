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

"""Tests for cross-dtype casting rules and value conversions."""

import contextlib
import functools
import itertools
import warnings

import dtype_test_utils
import ml_dtypes
import numpy as np
import pytest

ALL_COMPLEX = dtype_test_utils.ALL_COMPLEX
ALL_DTYPES = dtype_test_utils.ALL_DTYPES
ALL_FLOATS = dtype_test_utils.ALL_FLOATS
ALL_INTS = dtype_test_utils.ALL_INTS
CUSTOM_COMPLEX = dtype_test_utils.CUSTOM_COMPLEX
CUSTOM_DTYPES = dtype_test_utils.CUSTOM_DTYPES
CUSTOM_FLOATS = dtype_test_utils.CUSTOM_FLOATS
CUSTOM_INTS = dtype_test_utils.CUSTOM_INTS
NUMPY_BOOLS = dtype_test_utils.NUMPY_BOOLS
NUMPY_DTYPES = dtype_test_utils.NUMPY_DTYPES

bf16 = ml_dtypes.bfloat16
e8m0 = ml_dtypes.float8_e8m0fnu
COMPLEX_COMPONENTS = {
    ml_dtypes.bcomplex32: bf16,
    ml_dtypes.complex32: np.float16,
}
CASTING_RULES = ["no", "equiv", "safe", "same_kind", "unsafe"]


@functools.cache
def _all_source_values(dtype):
  """Returns a 1D array of `dtype` covering every component bit pattern."""
  dt = np.dtype(dtype)
  if dt == np.dtype(np.bool_):
    return np.array([False, True], dtype=np.bool_)
  if dt.type in CUSTOM_INTS or dt.kind in "iu":
    info = ml_dtypes.iinfo(dtype)
    return np.arange(info.min, info.max + 1, dtype=dtype)
  comp = COMPLEX_COMPONENTS.get(dt.type, dtype)
  storage = np.uint16 if np.dtype(comp).itemsize == 2 else np.uint8
  comp_vals = np.arange(2 ** (8 * np.dtype(comp).itemsize), dtype=storage).view(
      comp
  )
  if dt.type in CUSTOM_COMPLEX:
    n = len(comp_vals)
    out = np.zeros(2 * n, dtype=dtype)
    raw = out.view(comp).reshape(2 * n, 2)
    raw[:n, 0] = comp_vals
    raw[n:, 1] = comp_vals
    return out
  return comp_vals


@functools.cache
def _exactly_representable(source, target):
  """Returns True if every bit pattern of `source` round-trips via `target`."""
  if np.dtype(source).type in CUSTOM_COMPLEX and (
      np.dtype(target).type not in CUSTOM_COMPLEX
  ):
    return False
  if np.dtype(target).type in CUSTOM_INTS and not (
      np.dtype(source) == np.dtype(np.bool_)
      or np.dtype(source).type in CUSTOM_INTS
      or np.dtype(source).kind in "iu"
  ):
    return False
  src = _all_source_values(source)
  # `_all_source_values` includes raw signaling-NaN bit patterns and wide finite
  # magnitudes, which can raise FP exceptions during round-trip probing.
  with np.errstate(invalid="ignore", over="ignore", under="ignore"):
    orig = src.astype(np.complex128)
    roundtripped = src.astype(target).astype(np.complex128)
  real_ok = (orig.real == roundtripped.real) | (
      np.isnan(orig.real) & np.isnan(roundtripped.real)
  )
  imag_ok = (orig.imag == roundtripped.imag) | (
      np.isnan(orig.imag) & np.isnan(roundtripped.imag)
  )
  return bool(np.all(real_ok & imag_ok))


def _numeric_kind(dtype):
  """Returns the logical NumPy kind code ('b', 'u', 'i', 'f', 'c') for `dtype`.

  Custom dtypes currently report `dtype.kind == 'V'` (raw/void), so this helper
  maps them to the numeric kind of the category they represent.

  Args:
    dtype: Dtype or scalar type to classify.

  Returns:
    Single-character NumPy kind string ('b', 'u', 'i', 'f', or 'c').
  """
  t = np.dtype(dtype).type
  if t in CUSTOM_FLOATS:
    return "f"
  if t in CUSTOM_COMPLEX:
    return "c"
  if t in CUSTOM_INTS:
    return "i" if ml_dtypes.iinfo(t).min < 0 else "u"
  return np.dtype(dtype).kind


def _same_or_unsafe(source, target):
  """Classifies a non-safe cast as `'same_kind'` or `'unsafe'`.

  Follows NumPy's numeric kind hierarchy (`bool < uint < int < float <
  complex`): casts within the same kind or toward a wider kind are
  `'same_kind'`, while casts that step backward (such as float to int or signed
  to unsigned int) are `'unsafe'`.

  Args:
    source: Source dtype or scalar type.
    target: Target dtype or scalar type.

  Returns:
    `'same_kind'` or `'unsafe'`.
  """
  if "buifc".index(_numeric_kind(source)) <= "buifc".index(
      _numeric_kind(target)
  ):
    return "same_kind"
  return "unsafe"


def _custom_casting_level(source, target):
  """Returns expected minimum casting rule between two custom dtypes."""
  if source == target:
    return "no"
  if _exactly_representable(source, target):
    return "safe"
  return _same_or_unsafe(source, target)


CUSTOM_CASTS = [
    pytest.param(
        source,
        target,
        _custom_casting_level(source, target),
        id=f"{source.__name__}-to-{target.__name__}",
    )
    for source, target in itertools.product(CUSTOM_DTYPES, repeat=2)
]


def _involves_custom_dtype(source, target):
  """Returns True if at least one of `source` or `target` is a custom dtype."""
  return source in CUSTOM_DTYPES or target in CUSTOM_DTYPES


# Every `(source, target)` pair involving at least one custom dtype.
ALL_VALID_CASTS = [
    pytest.param(
        source,
        target,
        id=f"{np.dtype(source).name}-to-{np.dtype(target).name}",
    )
    for source, target in itertools.product(ALL_DTYPES, repeat=2)
    if _involves_custom_dtype(source, target)
]

# `(float_or_complex, int)` cast pairs involving a custom dtype, used to test
# truncation, non-finite inputs, and out-of-range `FE_INVALID` handling.
FLOAT_TO_INT_CASTS = [
    pytest.param(
        source,
        target,
        id=f"{np.dtype(source).name}-to-{np.dtype(target).name}",
    )
    for source, target in itertools.product(
        [*ALL_FLOATS, *ALL_COMPLEX], ALL_INTS
    )
    if _involves_custom_dtype(source, target)
]


# --- Casting rules (`np.can_cast`) and byte-order / self-cast tests ---


@pytest.mark.parametrize("source, target, minimum", CUSTOM_CASTS)
def test_custom_can_cast(source, target, minimum):
  for casting in CASTING_RULES:
    expected = CASTING_RULES.index(casting) >= CASTING_RULES.index(minimum)
    assert np.can_cast(source, target, casting=casting) == expected, casting
    assert (
        np.can_cast(np.dtype(source), np.dtype(target), casting=casting)
        == expected
    ), casting


@pytest.mark.parametrize("source", CUSTOM_COMPLEX, ids=lambda t: t.__name__)
@pytest.mark.parametrize("target", CUSTOM_COMPLEX, ids=lambda t: t.__name__)
def test_custom_complex_cast_preserves_components(source, target):
  values = np.array([0, 2 + 3j, -4 + 5j, -2j, 0.5 - 0.25j], np.complex64)
  converted = values.astype(source).astype(target)
  np.testing.assert_array_equal(converted.astype(np.complex64), values)


def _numpy_casting_level(custom, builtin, to_custom):
  source, target = (
      (builtin, np.dtype(custom)) if to_custom else (np.dtype(custom), builtin)
  )
  safe = False
  if to_custom:
    if builtin == np.dtype(np.bool_):
      safe = True
    elif custom in CUSTOM_FLOATS + CUSTOM_COMPLEX and builtin in [
        np.dtype(np.int8),
        np.dtype(np.uint8),
    ]:
      # Lossy safe casts retain the custom dtype for NumPy < 2.6's uint8 probe.
      safe = True
    elif custom is ml_dtypes.complex32 and builtin == np.dtype(np.float16):
      safe = True
  elif custom in CUSTOM_FLOATS:
    # Omitting safe casts to float16 makes NumPy < 2.6's Python-float probe
    # select float32, even for formats that fit in float16.
    safe = builtin.kind == "c" or (
        builtin.kind == "f" and builtin.itemsize >= 4
    )
  elif custom in CUSTOM_COMPLEX:
    safe = builtin.kind == "c"
  else:
    safe = builtin.kind in "ifc" or (
        builtin.kind == "u" and ml_dtypes.iinfo(custom).min == 0
    )
  if safe:
    return "safe"
  return _same_or_unsafe(source, target)


@pytest.mark.parametrize("custom", CUSTOM_DTYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("builtin", NUMPY_DTYPES, ids=str)
@pytest.mark.parametrize(
    "to_custom", [False, True], ids=["from_custom", "to_custom"]
)
def test_numpy_can_cast(custom, builtin, to_custom):
  source, target = (builtin, custom) if to_custom else (custom, builtin)
  minimum = _numpy_casting_level(custom, builtin, to_custom)
  for casting in CASTING_RULES:
    assert np.can_cast(source, target, casting=casting) == (
        CASTING_RULES.index(casting) >= CASTING_RULES.index(minimum)
    ), casting


@pytest.mark.parametrize(
    "source, target",
    [
        (np.bool_, ml_dtypes.int1),
        (np.bool_, e8m0),
        (np.int8, ml_dtypes.float4_e2m1fn),
        (np.uint8, e8m0),
    ],
)
def test_legacy_safe_casts_are_lossy(source, target):
  # Bool and byte-integer safe casts are retained to match CommonDType and
  # legacy ufunc loop selection.
  assert np.can_cast(source, target, casting="safe")
  assert not _exactly_representable(source, target)


@pytest.mark.parametrize(
    "source, target",
    [
        (ml_dtypes.float8_e4m3fn, ml_dtypes.float4_e2m1fn),
        (ml_dtypes.float8_e4m3, ml_dtypes.float8_e4m3fn),
        (ml_dtypes.int4, ml_dtypes.float4_e2m1fn),
    ],
)
def test_lossy_same_itemsize_custom_casts_are_not_safe(source, target):
  """Verifies lossy 1-byte custom casts are `'same_kind'`, not `'safe'`."""
  assert not np.can_cast(source, target, casting="safe")
  assert np.can_cast(source, target, casting="same_kind")


@pytest.mark.parametrize("dtype", CUSTOM_DTYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize(
    "source_order, target_order",
    list(itertools.product(["=", "<", ">"], repeat=2)),
)
def test_can_cast_byte_order(dtype, source_order, target_order):
  source = np.dtype(dtype).newbyteorder(source_order)
  target = np.dtype(dtype).newbyteorder(target_order)
  for casting in CASTING_RULES:
    expected = casting != "no" or source.isnative == target.isnative
    assert np.can_cast(source, target, casting=casting) == expected, casting


@pytest.mark.parametrize("dtype", CUSTOM_DTYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "reversed", "unaligned"]
)
def test_self_cast_copy_and_byte_swap(dtype, layout):
  if dtype in CUSTOM_INTS:
    info = ml_dtypes.iinfo(dtype)
    values = [info.min, 0, info.max]
  elif dtype in CUSTOM_COMPLEX:
    values = [1 + 2j, 2 + 4j, -1 + 1j]
  else:
    values = [1, 2, 4]  # Also representable by the exponent-only format.
  dt = np.dtype(dtype)
  stride = {"strided": 2, "reversed": -2}.get(layout, 1) * dt.itemsize
  offset = int(layout == "unaligned")
  source_storage = np.empty(6 * dt.itemsize + 1, dtype=np.uint8)
  for source_order in ["=", "S"]:
    source = np.ndarray(
        (3,),
        dtype=dt.newbyteorder(source_order),
        buffer=source_storage,
        offset=offset + (2 * abs(stride) if stride < 0 else 0),
        strides=(stride,),
    )
    source[:] = values
    for target_order in ["=", "S"]:
      target_storage = np.empty_like(source_storage)
      target = np.ndarray(
          (3,),
          dtype=dt.newbyteorder(target_order),
          buffer=target_storage,
          offset=offset,
          strides=(abs(stride),),
      )
      np.copyto(target, source, casting="equiv")
      np.testing.assert_array_equal(target.astype(np.complex64), values)


@pytest.mark.parametrize(
    "dtype",
    [
        t
        for t in CUSTOM_FLOATS
        if t not in [bf16, ml_dtypes.float8_e5m3fnu, e8m0]
    ],
    ids=lambda t: t.__name__,
)
def test_legacy_float16_safe_cast_false_negative(dtype):
  # All values fit in float16, but the cast table reports them as not safe.
  # Before NumPy 2.6, the Python-float probe selects float32 instead.
  assert _exactly_representable(dtype, np.float16)
  assert not np.can_cast(dtype, np.float16, casting="safe")


@pytest.mark.parametrize(
    "dtype", [bf16, *CUSTOM_COMPLEX], ids=lambda t: t.__name__
)
@pytest.mark.parametrize("order", ["=", "S"], ids=["native", "swapped"])
@pytest.mark.parametrize(
    "unaligned", [False, True], ids=["aligned", "unaligned"]
)
def test_byte_order_getitem_setitem(dtype, order, unaligned):
  values = [1.5, -2.25] if dtype is bf16 else [1.5 + 2.25j, -2.25 + 4.5j]
  native = np.array(values, dtype=dtype)
  dt = native.dtype.newbyteorder(order)
  storage = np.empty(native.nbytes + 1, dtype=np.uint8)
  array = np.ndarray(
      native.shape, dtype=dt, buffer=storage, offset=int(unaligned)
  )
  np.copyto(array, native, casting="equiv")

  for i, value in enumerate(values):
    assert array[i] == value
    assert array.item(i) == value
  assert array.tolist() == values

  for scalar in [values[1], dtype(values[1])]:
    array[0] = scalar
    np.testing.assert_array_equal(array.astype(np.complex64), [values[1]] * 2)
  array[:] = values
  np.testing.assert_array_equal(array.astype(np.complex64), values)
  np.testing.assert_array_equal(
      np.array(values, dtype=dt).astype(np.complex64), values
  )


# --- Value conversion tests across all (source, target) pairs ---


@functools.cache
def _sample_array(source):
  """Builds a 1D test array for `source` (including special float values)."""
  if source in NUMPY_BOOLS:
    return np.array([False, True], dtype=source)
  if source in ALL_INTS:
    info = ml_dtypes.iinfo(source)
    candidates = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -8,
        10,
        -10,
        15,
        16,
        127,
        -128,
        255,
        256,
        -256,
        1024,
        -1024,
        info.min,
        info.max,
    ]
    return np.array(
        [v for v in dict.fromkeys(candidates) if info.min <= v <= info.max],
        dtype=source,
    )
  if source in ALL_COMPLEX:
    pairs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 2.0),
        (0.0, -1.0),
        (0.5, -0.5),
        (-0.5, 0.25),
        (1.5, 2.5),
        (-2.0, 3.0),
        (4.0, -4.0),
        (np.nan, 1.0),
        (0.0, np.nan),
        (np.inf, 1.0),
        (0.0, np.inf),
        (-np.inf, 1.0),
        (0.0, -np.inf),
    ]
    arr = np.empty(len(pairs), dtype=source)
    ml_dtypes.real(arr)[:] = [r for r, _ in pairs]
    ml_dtypes.imag(arr)[:] = [i for _, i in pairs]
    return arr
  raw = [
      0.0,
      -0.0,
      0.25,
      -0.25,
      0.5,
      -0.5,
      1.0,
      -1.0,
      1.5,
      -1.5,
      1.75,
      -1.75,
      2.0,
      -2.0,
      3.0,
      -3.0,
      4.0,
      -4.0,
      7.0,
      -8.0,
      15.0,
      16.0,
      127.0,
      -128.0,
      np.nan,
      np.inf,
      -np.inf,
  ]
  return np.array(raw, dtype=source)


def _overflows_float16(values):
  """Returns True if any finite value in `values` overflows `float16`."""
  wide = np.longdouble if values.dtype == np.longdouble else np.float64
  w = values.astype(wide)
  return bool(np.any(np.isfinite(w) & (np.abs(w) >= 65520.0)))


def _cast_checking_warnings(arr, source, target):
  """Casts `arr` to `target`, asserting expected complex/overflow warnings."""
  expect_complex_warn = (
      source in ALL_COMPLEX
      and target not in ALL_COMPLEX
      and target not in NUMPY_BOOLS
  )
  expect_overflow_warn = False
  if target is np.float16:
    expect_overflow_warn = _overflows_float16(ml_dtypes.real(arr))
  elif target is ml_dtypes.complex32:
    expect_overflow_warn = _overflows_float16(ml_dtypes.real(arr)) or (
        source in ALL_COMPLEX and _overflows_float16(ml_dtypes.imag(arr))
    )

  if not expect_complex_warn and not expect_overflow_warn:
    return arr.astype(target)

  expected_categories = set()
  if expect_complex_warn:
    expected_categories.add(np.exceptions.ComplexWarning)
  if expect_overflow_warn:
    expected_categories.add(RuntimeWarning)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = arr.astype(target)
  assert {w.category for w in caught} == expected_categories
  return result


def _expected_cast(arr, source, target):
  """Computes the reference `target` array via wide NumPy or scalar casts."""
  if np.dtype(source) == np.dtype(target):
    return arr.copy()
  if target in NUMPY_BOOLS:
    return np.array([bool(v) for v in arr], dtype=np.bool_)
  if source in CUSTOM_DTYPES and target in CUSTOM_DTYPES:
    if source in CUSTOM_INTS and target in CUSTOM_INTS:
      via = np.int16
    elif source in CUSTOM_COMPLEX:
      via = np.complex64
    else:
      via = np.float32
    return _cast_checking_warnings(arr.astype(via), via, target)
  if source in CUSTOM_DTYPES:
    if source in CUSTOM_COMPLEX and target in ALL_COMPLEX:
      wide = np.complex128
      wide_arr = np.array(arr.tolist(), dtype=wide)
    elif source in CUSTOM_COMPLEX:
      # Drop `imag` before widening so reference casts avoid NumPy's ARM64 NEON
      # `complex128 -> int64/uint64` `FE_INVALID` on unused imaginary lanes.
      wide = np.float64
      wide_arr = ml_dtypes.real(arr).astype(wide)
    elif source in CUSTOM_INTS:
      wide = np.int64
      wide_arr = np.array(arr.tolist(), dtype=wide)
    else:
      wide = np.float64
      wide_arr = np.array(arr.tolist(), dtype=wide)
    return _cast_checking_warnings(wide_arr, wide, target)
  # Built-in -> custom.
  if target in CUSTOM_INTS:
    if source in NUMPY_BOOLS or source in ALL_INTS:
      mask = (1 << ml_dtypes.iinfo(target).bits) - 1
      return np.array([target(int(v) & mask) for v in arr], dtype=target)
    return np.array(
        [target(int(v)) for v in np.trunc(ml_dtypes.real(arr))], dtype=target
    )
  if target in CUSTOM_FLOATS:
    return np.array(
        [target(v) for v in ml_dtypes.real(arr).astype(np.float32)],
        dtype=target,
    )
  out = np.empty(len(arr), dtype=target)
  real_f32 = ml_dtypes.real(arr).astype(np.float32)
  imag_f32 = ml_dtypes.imag(arr).astype(np.float32)
  comp = ml_dtypes.real(out).dtype.type
  # Wide integer samples (>= 65520) overflow `complex32`'s `float16` component.
  overflows_f16 = comp is np.float16 and (
      _overflows_float16(real_f32) or _overflows_float16(imag_f32)
  )
  over_ctx = (
      pytest.warns(RuntimeWarning, match="overflow")
      if overflows_f16
      else contextlib.nullcontext()
  )
  with over_ctx:
    ml_dtypes.real(out)[:] = [comp(v) for v in real_f32]
    ml_dtypes.imag(out)[:] = [comp(v) for v in imag_f32]
  return out


def _assert_cast_equal(actual, expected, target):
  """Asserts `actual` has `target` dtype and matches `expected` elementwise."""
  assert actual.dtype == np.dtype(target)
  assert expected.dtype == np.dtype(target)
  # NumPy 2.0.0's assert_array_equal only treats NaNs as equal for built-in
  # numeric dtype.char codes, so widen custom float values to float32 (which
  # represents every custom float value and NaN losslessly).
  if target in CUSTOM_COMPLEX:
    np.testing.assert_array_equal(
        ml_dtypes.real(actual).astype(np.float32),
        ml_dtypes.real(expected).astype(np.float32),
    )
    np.testing.assert_array_equal(
        ml_dtypes.imag(actual).astype(np.float32),
        ml_dtypes.imag(expected).astype(np.float32),
    )
  elif target in CUSTOM_FLOATS:
    np.testing.assert_array_equal(
        actual.astype(np.float32), expected.astype(np.float32)
    )
  else:
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source, target", ALL_VALID_CASTS)
def test_cast_values(source, target):
  """Verifies cast values on contiguous and strided arrays."""
  src_arr = _sample_array(source)
  if target in ALL_INTS and (source in ALL_FLOATS or source in ALL_COMPLEX):
    # Filter to finite in-range real parts; non-finite and out-of-range inputs
    # are tested separately below with `FE_INVALID` checks.
    ti = ml_dtypes.iinfo(target)
    real = ml_dtypes.real(src_arr)
    wide = np.longdouble if real.dtype == np.longdouble else np.float64
    real_w = real.astype(wide)
    finite = np.isfinite(real_w)
    keep = np.zeros(len(src_arr), dtype=bool)
    keep[finite] = (ti.min <= np.trunc(real_w[finite])) & (
        np.trunc(real_w[finite]) <= ti.max
    )
    if not np.any(keep):
      pytest.skip(
          f"No sample values of {np.dtype(source).name} fit in"
          f" {np.dtype(target).name}"
      )
    src_arr = src_arr[keep]

  src_strided = np.repeat(src_arr, 2)[::2]
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    expected = _expected_cast(src_arr, source, target)
    actual = _cast_checking_warnings(src_arr, source, target)
    actual_strided = _cast_checking_warnings(src_strided, source, target)
  _assert_cast_equal(actual, expected, target)
  _assert_cast_equal(actual_strided, expected, target)


@pytest.mark.parametrize("source, target", FLOAT_TO_INT_CASTS)
def test_cast_nonfinite_to_integer(source, target):
  """Verifies `NaN` and `+-inf` cast to integer 0 and raise `FE_INVALID`."""
  real_dt = ml_dtypes.finfo(source).dtype
  is_complex = source in ALL_COMPLEX
  probes = np.array([np.nan, np.inf, -np.inf], dtype=real_dt)
  nonfinite = probes[~np.isfinite(probes)]
  if nonfinite.size == 0:
    pytest.skip(f"{np.dtype(source).name} has no non-finite values")

  expected_categories = (
      {np.exceptions.ComplexWarning, RuntimeWarning}
      if is_complex
      else {RuntimeWarning}
  )
  for v in nonfinite:
    if is_complex:
      arr = np.empty(1, dtype=source)
      ml_dtypes.real(arr)[0] = v
      ml_dtypes.imag(arr)[0] = 1.0
    else:
      arr = np.array([v], dtype=source)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      result = arr.astype(target)
    np.testing.assert_array_equal(result, np.zeros(1, dtype=target))
    assert {w.category for w in caught} == expected_categories

    warn_ctx = (
        pytest.warns(np.exceptions.ComplexWarning)
        if is_complex
        else contextlib.nullcontext()
    )
    with np.errstate(invalid="raise"), warn_ctx:
      with pytest.raises(FloatingPointError):
        arr.astype(target)


@pytest.mark.parametrize("source, target", FLOAT_TO_INT_CASTS)
def test_cast_out_of_range_float_to_integer(source, target):
  """Verifies out-of-range float-to-int casts return 0 and raise `FE_INVALID`."""
  real_dt = ml_dtypes.finfo(source).dtype
  is_complex = source in ALL_COMPLEX
  finfo = ml_dtypes.finfo(real_dt)
  ti = ml_dtypes.iinfo(target)
  wide = np.longdouble if real_dt == np.longdouble else np.float64

  max_exact = int(wide(finfo.max))
  min_exact = int(wide(finfo.min))

  candidates = []
  if max_exact > ti.max:
    candidates.append(finfo.max)
    upper_pow2 = ti.max + 1
    if upper_pow2 <= max_exact:
      rv = real_dt.type(float(upper_pow2))
      if np.isfinite(wide(rv)) and int(wide(rv)) > ti.max:
        candidates.append(rv)
  if min_exact < ti.min:
    candidates.append(finfo.min)
    lower_oob = -1 if ti.min == 0 else ti.min - 1
    if lower_oob >= min_exact:
      rv = real_dt.type(float(lower_oob))
      if np.isfinite(wide(rv)) and int(wide(rv)) < ti.min:
        candidates.append(rv)

  if not candidates:
    pytest.skip(
        f"All finite {np.dtype(source).name} values fit in"
        f" {np.dtype(target).name}"
    )

  expected_categories = (
      {np.exceptions.ComplexWarning, RuntimeWarning}
      if is_complex
      else {RuntimeWarning}
  )
  for v in candidates:
    if is_complex:
      arr = np.empty(1, dtype=source)
      ml_dtypes.real(arr)[0] = v
      ml_dtypes.imag(arr)[0] = 1.0
    else:
      arr = np.array([v], dtype=source)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      result = arr.astype(target)
    np.testing.assert_array_equal(result, np.zeros(1, dtype=target))
    assert {w.category for w in caught} == expected_categories

    warn_ctx = (
        pytest.warns(np.exceptions.ComplexWarning)
        if is_complex
        else contextlib.nullcontext()
    )
    with np.errstate(invalid="raise"), warn_ctx:
      with pytest.raises(FloatingPointError):
        arr.astype(target)


# `(complex, float_or_int)` cast pairs involving a custom dtype, all of which
# discard the imaginary part and must emit `ComplexWarning` once.
COMPLEX_TO_REAL_CASTS = [
    pytest.param(
        source,
        target,
        id=f"{np.dtype(source).name}-to-{np.dtype(target).name}",
    )
    for source, target in itertools.product(
        ALL_COMPLEX, [*ALL_FLOATS, *ALL_INTS]
    )
    if _involves_custom_dtype(source, target)
]


@pytest.mark.parametrize("source, target", COMPLEX_TO_REAL_CASTS)
def test_complex_to_real_cast_warns_once(source, target):
  """Verifies complex-to-real/int casts emit one `ComplexWarning` per call."""
  real_val = 0.0 if target is ml_dtypes.int1 else 1.0
  # Test both a 1D array exceeding NumPy's buffer size (> 8192 elements, which
  # also releases the GIL) and a 2D non-contiguous slice with non-uniform
  # strides (which invokes the inner strided loop once per row) to verify
  # `ComplexWarning` is emitted once per cast rather than per inner-loop call.
  cases = [
      (
          np.full(20_000, complex(real_val, 2.0), dtype=source),
          np.full(20_000, real_val, dtype=target),
      ),
      (
          np.full((8, 8), complex(real_val, 2.0), dtype=source)[:, :3],
          np.full((8, 3), real_val, dtype=target),
      ),
  ]
  for arr, expected in cases:
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      out = arr.astype(target)
    np.testing.assert_array_equal(out, expected)
    complex_warnings = [
        w
        for w in caught
        if issubclass(w.category, np.exceptions.ComplexWarning)
    ]
    assert len(complex_warnings) == 1

  with warnings.catch_warnings():
    warnings.simplefilter("error", np.exceptions.ComplexWarning)
    with pytest.raises(np.exceptions.ComplexWarning):
      cases[1][0].astype(target)


@pytest.mark.parametrize(
    "source, target, value, expected",
    [
        (bf16, np.float16, 65536.0, np.inf),
        (bf16, np.float16, -65536.0, -np.inf),
        (e8m0, np.float16, 65536.0, np.inf),
        (ml_dtypes.bcomplex32, np.float16, 65536.0 + 1.0j, np.inf),
        (bf16, ml_dtypes.complex32, 65536.0, complex(np.inf, 0.0)),
        (np.uint16, ml_dtypes.complex32, 65535, complex(np.inf, 0.0)),
        (np.int32, ml_dtypes.complex32, -70000, complex(-np.inf, 0.0)),
        (np.float32, ml_dtypes.complex32, 65520.0, complex(np.inf, 0.0)),
        (
            ml_dtypes.bcomplex32,
            ml_dtypes.complex32,
            65536.0 + 1.0j,
            complex(np.inf, 1.0),
        ),
        (
            ml_dtypes.bcomplex32,
            ml_dtypes.complex32,
            1.0 - 65536.0j,
            complex(1.0, -np.inf),
        ),
        (
            np.complex64,
            ml_dtypes.complex32,
            1.0 + 65520.0j,
            complex(1.0, np.inf),
        ),
    ],
)
def test_cast_overflow_to_float16_and_complex32(
    source, target, value, expected
):
  """Verifies finite values overflowing `float16`/`complex32` raise `FE_OVERFLOW`."""
  arr = np.array([value], dtype=source)
  is_complex_to_real = source in ALL_COMPLEX and target not in ALL_COMPLEX
  expected_categories = (
      {np.exceptions.ComplexWarning, RuntimeWarning}
      if is_complex_to_real
      else {RuntimeWarning}
  )

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = arr.astype(target)
  assert {w.category for w in caught} == expected_categories
  assert any("overflow" in str(w.message) for w in caught)
  _assert_cast_equal(result, np.array([expected], dtype=target), target)

  warn_ctx = (
      pytest.warns(np.exceptions.ComplexWarning)
      if is_complex_to_real
      else contextlib.nullcontext()
  )
  with np.errstate(over="raise"), warn_ctx:
    with pytest.raises(FloatingPointError):
      arr.astype(target)
