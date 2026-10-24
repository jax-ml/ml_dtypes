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

"""Casting rules, including documented legacy exceptions."""

import functools
import itertools
import warnings

from dtype_test_utils import (
    CUSTOM_COMPLEX,
    CUSTOM_DTYPES,
    CUSTOM_FLOATS,
    CUSTOM_INTS,
    NUMPY_DTYPES,
)
import ml_dtypes
import numpy as np
import pytest

bf16 = ml_dtypes.bfloat16
e8m0 = ml_dtypes.float8_e8m0fnu
e2m3 = ml_dtypes.float6_e2m3fn
COMPLEX_COMPONENTS = {
    ml_dtypes.bcomplex32: bf16,
    ml_dtypes.complex32: np.float16,
}
CASTING_RULES = ["no", "equiv", "safe", "same_kind", "unsafe"]


@functools.cache
def _values(dtype):
  """Enumerate each real component value, independent of promotion rules."""
  if dtype in CUSTOM_INTS:
    info = ml_dtypes.iinfo(dtype)
    return np.arange(info.min, info.max + 1, dtype=np.float64)
  dtype = COMPLEX_COMPONENTS.get(dtype, dtype)
  storage = np.uint16 if np.dtype(dtype).itemsize == 2 else np.uint8
  with np.errstate(invalid="ignore"):
    return (
        np.arange(2 ** (8 * np.dtype(dtype).itemsize), dtype=storage)
        .view(dtype)
        .astype(np.float64)
    )


@functools.cache
def _exactly_representable(source, target):
  if source in CUSTOM_COMPLEX and target not in CUSTOM_COMPLEX:
    return False  # The imaginary component would be discarded.
  values = _values(source)
  component = COMPLEX_COMPONENTS.get(target, target)
  with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    warnings.simplefilter("ignore", RuntimeWarning)
    converted = values.astype(component).astype(np.float64)
  return bool(
      np.all((values == converted) | (np.isnan(values) & np.isnan(converted)))
  )


def _has_custom_cast(source, target):
  if source == target:
    return True
  if source in CUSTOM_INTS and target in CUSTOM_INTS:
    # Only widening casts within the signed or unsigned intN families exist.
    si, ti = ml_dtypes.iinfo(source), ml_dtypes.iinfo(target)
    return (si.min < 0) == (ti.min < 0) and si.bits < ti.bits
  pair = {source, target}
  if e8m0 in pair:
    return pair - {e8m0} <= {bf16, *CUSTOM_COMPLEX}
  if e2m3 in pair and pair & {ml_dtypes.int4, ml_dtypes.uint4}:
    return False
  return True


def _custom_casting_level(source, target):
  if source == target:
    return "no"
  if not _has_custom_cast(source, target):
    return None
  if _exactly_representable(source, target):
    return "safe"
  src, dst = np.dtype(source), np.dtype(target)
  return (
      "same_kind"
      if "buifc".index(src.kind) <= "buifc".index(dst.kind)
      else "unsafe"
  )


CUSTOM_CASTS = [
    pytest.param(
        source,
        target,
        _custom_casting_level(source, target),
        id=f"{source.__name__}-to-{target.__name__}",
    )
    for source, target in itertools.product(CUSTOM_DTYPES, repeat=2)
]


@pytest.mark.parametrize("source, target, minimum", CUSTOM_CASTS)
def test_custom_can_cast(source, target, minimum):
  for casting in CASTING_RULES:
    expected = minimum is not None and CASTING_RULES.index(
        casting
    ) >= CASTING_RULES.index(minimum)
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
  return (
      "same_kind"
      if "buifc".index(source.kind) <= "buifc".index(target.kind)
      else "unsafe"
  )


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
        (np.int8, ml_dtypes.float4_e2m1fn),
        (np.uint8, e8m0),
    ],
)
def test_legacy_safe_casts_are_lossy(source, target):
  # Byte-integer safe casts are retained temporarily while ufuncs use legacy
  # loop selection.
  assert np.can_cast(source, target, casting="safe")
  info = np.iinfo(source)
  values = np.arange(info.min, info.max + 1, dtype=source).astype(np.float64)
  with np.errstate(over="ignore", invalid="ignore"):
    converted = values.astype(target).astype(np.float64)
  assert not np.all(
      (values == converted) | (np.isnan(values) & np.isnan(converted))
  )


@pytest.mark.parametrize(
    "source, target",
    [
        (ml_dtypes.float8_e4m3fn, ml_dtypes.float4_e2m1fn),
        (ml_dtypes.float8_e4m3, ml_dtypes.float8_e4m3fn),
        (ml_dtypes.int4, ml_dtypes.float4_e2m1fn),
    ],
)
def test_lossy_same_itemsize_custom_casts_are_not_safe(source, target):
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
