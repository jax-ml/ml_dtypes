# Copyright 2022 The ml_dtypes Authors.
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

"""Test cases for int4 types."""

# pylint: disable=g-complex-comprehension

import contextlib
import copy
import operator
import pickle
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
from multi_thread_utils import multi_threaded
import numpy as np

int1 = ml_dtypes.int1
int2 = ml_dtypes.int2
int4 = ml_dtypes.int4
uint1 = ml_dtypes.uint1
uint2 = ml_dtypes.uint2
uint4 = ml_dtypes.uint4

INTN_TYPES = [int1, int2, int4, uint1, uint2, uint4]

VALUES = {
    int1: list(range(-1, 1)),
    int2: list(range(-2, 2)),
    int4: list(range(-8, 8)),
    uint1: list(range(0, 2)),
    uint2: list(range(0, 4)),
    uint4: list(range(0, 16)),
}


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=INTN_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "ml_dtypes")

  @parameterized.product(scalar_type=INTN_TYPES)
  def testPickleable(self, scalar_type):
    # https://github.com/jax-ml/jax/discussions/8505
    x = np.arange(10, dtype=scalar_type)
    serialized = pickle.dumps(x)
    x_out = pickle.loads(serialized)
    self.assertEqual(x_out.dtype, x.dtype)
    np.testing.assert_array_equal(x_out.astype(int), x.astype(int))

  @parameterized.product(
      scalar_type=INTN_TYPES,
      python_scalar=[int, float, np.float16, np.longdouble],
  )
  def testRoundTripToPythonScalar(self, scalar_type, python_scalar):
    for v in VALUES[scalar_type]:
      self.assertEqual(v, scalar_type(v))
      self.assertEqual(python_scalar(v), python_scalar(scalar_type(v)))
      self.assertEqual(
          scalar_type(v), scalar_type(python_scalar(scalar_type(v)))
      )

  @parameterized.product(scalar_type=INTN_TYPES)
  def testRoundTripNumpyTypes(self, scalar_type):
    for dtype in [np.int8, np.int32]:
      for f in VALUES[scalar_type]:
        self.assertEqual(dtype(f), dtype(scalar_type(dtype(f))))
        self.assertEqual(int(dtype(f)), int(scalar_type(dtype(f))))
        self.assertEqual(dtype(f), dtype(scalar_type(np.array(f, dtype))))

      np.testing.assert_equal(
          dtype(np.array(VALUES[scalar_type], scalar_type)),
          np.array(VALUES[scalar_type], dtype),
      )

  @parameterized.product(scalar_type=INTN_TYPES)
  def testStr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(str(value), str(scalar_type(value)))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testRepr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(str(value), str(scalar_type(value)))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testFormat(self, scalar_type):
    for value in VALUES[scalar_type]:
      val = scalar_type(value)
      self.assertEqual(f"{val}", f"{value}")
      self.assertEqual(f"{val:d}", f"{value:d}")
      self.assertEqual(f"{val:04d}", f"{value:04d}")
      self.assertEqual(f"{val:x}", f"{value:x}")
      self.assertEqual(f"{val:b}", f"{value:b}")
      self.assertEqual(f"{val:o}", f"{value:o}")
      self.assertEqual(f"{val:8d}", f"{value:8d}")
      self.assertEqual(format(val, "x"), format(value, "x"))
      self.assertEqual(format(val), format(value))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testItem(self, scalar_type):
    v = -1 if scalar_type == int1 else 1
    self.assertIsInstance(scalar_type(v).item(), int)
    self.assertEqual(scalar_type(v).item(), v)

  @parameterized.product(scalar_type=INTN_TYPES)
  def testHash(self, scalar_type):
    for v in VALUES[scalar_type]:
      self.assertEqual(hash(v), hash(scalar_type(v)), msg=v)

  @parameterized.product(
      scalar_type=INTN_TYPES,
      op=[
          operator.le,
          operator.lt,
          operator.eq,
          operator.ne,
          operator.ge,
          operator.gt,
      ],
  )
  def testComparison(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        result = op(scalar_type(v), scalar_type(w))
        self.assertEqual(op(v, w), result)
        self.assertIsInstance(result, np.bool_)

  @parameterized.product(
      scalar_type=INTN_TYPES,
      op=[
          operator.neg,
          operator.pos,
      ],
  )
  def testUnop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      out = op(scalar_type(v))
      self.assertIsInstance(out, scalar_type)
      self.assertEqual(scalar_type(op(v)), out, msg=v)

  @parameterized.product(
      scalar_type=INTN_TYPES,
      op=[
          operator.add,
          operator.sub,
          operator.mul,
          operator.floordiv,
          operator.mod,
      ],
  )
  def testBinop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        if w == 0 and op in [operator.floordiv, operator.mod]:
          with self.assertRaises(ZeroDivisionError):
            op(scalar_type(v), scalar_type(w))
        else:
          out = op(scalar_type(v), scalar_type(w))
          self.assertIsInstance(out, scalar_type)
          self.assertEqual(scalar_type(op(v, w)), out, msg=(v, w))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testIssubdtype(self, scalar_type):
    # In the future, we may want to make these more specific (e.g. use
    # np.number or np.integer instead of np.generic) by changing the
    # base in RegisterIntNDtype.
    self.assertTrue(np.issubdtype(scalar_type, np.generic))
    self.assertTrue(np.issubdtype(np.dtype(scalar_type), np.generic))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testCastToDtype(self, scalar_type):
    name = scalar_type.__name__
    dt = np.dtype(scalar_type)
    self.assertIs(dt.type, scalar_type)
    self.assertEqual(dt.name, name)
    self.assertEqual(repr(dt), f"dtype({name})")

  @parameterized.product(scalar_type=INTN_TYPES)
  def testCastFailure(self, scalar_type):
    with self.assertRaises(ValueError):
      scalar_type(np.nan)
    with self.assertRaises(OverflowError):
      scalar_type(np.inf)
    with self.assertRaises(OverflowError):
      scalar_type(1e10)
    with self.assertRaises(ValueError):
      np.array(np.nan, dtype=scalar_type)
    with self.assertRaises(OverflowError):
      np.array(np.inf, dtype=scalar_type)
    with self.assertRaises(OverflowError):
      np.array(1e10, dtype=scalar_type)
    # But these shouldn't raise exceptions.
    with np.errstate(invalid="ignore"):
      np.array(np.nan).astype(scalar_type)
      np.array(np.inf).astype(scalar_type)
      np.array(1e10).astype(scalar_type)

  _INCOMPATIBLE_VALUES = [
      None,
      "string",
      "",
      {},
      {"a": 1},
      object(),
  ]

  @parameterized.product(
      scalar_type=INTN_TYPES,
      other=_INCOMPATIBLE_VALUES,
      op=[operator.eq, operator.ne],
  )
  def testComparisonEqualityWithIncompatibleTypes(self, scalar_type, other, op):
    val = scalar_type(0)
    equiv_type = np.int32 if scalar_type in [int1, int2, int4] else np.uint32
    equiv_val = equiv_type(0)
    self.assertEqual(
        op(val, other),
        op(equiv_val, other),
        msg=f"EQ/NE: {scalar_type.__name__} vs {type(other).__name__}",
    )
    self.assertEqual(
        op(other, val),
        op(other, equiv_val),
        msg=(
            f"Reflected EQ/NE: {type(other).__name__} vs {scalar_type.__name__}"
        ),
    )

  @parameterized.product(
      scalar_type=INTN_TYPES,
      other=_INCOMPATIBLE_VALUES,
      op=[operator.lt, operator.le, operator.gt, operator.ge],
  )
  def testComparisonOrderingWithIncompatibleTypes(self, scalar_type, other, op):
    val = scalar_type(0)
    equiv_type = np.int32 if scalar_type in [int1, int2, int4] else np.uint32
    equiv_val = equiv_type(0)
    # Check if equiv_val raises TypeError
    try:
      op(equiv_val, other)
      equiv_raises = False
    except TypeError:
      equiv_raises = True

    if equiv_raises:
      with self.assertRaises(
          TypeError, msg=f"{val} {op.__name__} {repr(other)}"
      ):
        op(val, other)
    else:
      self.assertEqual(op(val, other), op(equiv_val, other))

    # Reflected ordering
    try:
      op(other, equiv_val)
      equiv_reflected_raises = False
    except TypeError:
      equiv_reflected_raises = True

    if equiv_reflected_raises:
      with self.assertRaises(
          TypeError, msg=f"{repr(other)} {op.__name__} {val}"
      ):
        op(other, val)
    else:
      self.assertEqual(op(other, val), op(other, equiv_val))

  @parameterized.product(
      scalar_type=INTN_TYPES,
      seq=[
          [],
          [0, 1],
          (0, 1),
          np.array([0, 1]),
      ],
      op=[
          operator.eq,
          operator.ne,
          operator.lt,
          operator.le,
          operator.gt,
          operator.ge,
      ],
  )
  def testComparisonWithSequences(self, scalar_type, seq, op):
    val = scalar_type(0)
    equiv_type = np.int32 if scalar_type in [int1, int2, int4] else np.uint32
    equiv_val = equiv_type(0)
    np.testing.assert_array_equal(op(val, seq), op(equiv_val, seq))
    np.testing.assert_array_equal(op(seq, val), op(seq, equiv_val))


@multi_threaded(
    num_workers=3,
    skip_tests=[
        "testBinaryUfuncs",
        "testZeroSizeReduction",  # pytest 9.0.1's subtest appears not to be thread-safe
    ],
)
class ArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=INTN_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testHash(self, scalar_type):
    h = hash(np.dtype(scalar_type))
    self.assertEqual(h, hash(np.dtype(scalar_type.dtype)))
    self.assertEqual(h, hash(np.dtype(scalar_type.__name__)))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    # For context, see https://github.com/jax-ml/jax/issues/4651. If the hash
    # value of the type descriptor is not initialized correctly, a deep copy
    # can change the type hash.
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  @parameterized.product(scalar_type=INTN_TYPES)
  def testArray(self, scalar_type):
    if scalar_type == int1:
      x = np.array([[-1, 0, -1, 0]], dtype=scalar_type)
      self.assertEqual("[[-1 0 -1 0]]", str(x))
    elif scalar_type == uint1:
      x = np.array([[1, 1, 0, 1]], dtype=scalar_type)
      self.assertEqual("[[1 1 0 1]]", str(x))
    elif scalar_type == int2:
      x = np.array([[-2, 1, 0, 1]], dtype=scalar_type)
      self.assertEqual("[[-2 1 0 1]]", str(x))
    else:
      x = np.array([[1, 2, 3]], dtype=scalar_type)
      self.assertEqual("[[1 2 3]]", str(x))
    self.assertEqual(scalar_type, x.dtype)
    np.testing.assert_array_equal(x, x)
    self.assertTrue((x == x).all())  # pylint: disable=comparison-with-itself

  @parameterized.product(
      scalar_type=INTN_TYPES,
      ufunc=[np.nonzero, np.logical_not, np.argmax, np.argmin],
  )
  def testUnaryPredicateUfunc(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type])
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    # Compute `ufunc(y)` first so we don't get lucky by reusing memory
    # initialized by `ufunc(x)`.
    y_result = ufunc(y)
    x_result = ufunc(x)
    np.testing.assert_array_equal(x_result, y_result)

  @parameterized.product(
      scalar_type=INTN_TYPES,
      ufunc=[
          np.less,
          np.less_equal,
          np.greater,
          np.greater_equal,
          np.equal,
          np.not_equal,
          np.logical_and,
          np.logical_or,
          np.logical_xor,
      ],
  )
  def testPredicateUfuncs(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type])
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(
        ufunc(x[:, None], x[None, :]),
        ufunc(y[:, None], y[None, :]),
    )

  @parameterized.product(
      scalar_type=INTN_TYPES,
      ufunc=[
          np.add,
          np.subtract,
          np.multiply,
          np.floor_divide,
          np.remainder,
      ],
  )
  @ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
  def testBinaryUfuncs(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type])
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(
        ufunc(x[:, None], x[None, :]).astype(scalar_type),
        ufunc(y[:, None], y[None, :]),
    )

  @parameterized.product(
      scalar_type=INTN_TYPES,
      ufunc=[
          np.logical_and,
          np.logical_or,
          np.logical_xor,
      ],
  )
  def testLogicalUfuncReduce(self, scalar_type, ufunc):
    x = np.array([True, False, True], dtype=scalar_type)
    np.testing.assert_equal(
        ufunc.reduce(x),
        ufunc.reduce(x.astype(bool)),
    )

  @parameterized.parameters(INTN_TYPES)
  def testAccumulateWithOutputDtype(self, scalar_type):
    x = np.array([1, 2, 3], dtype=scalar_type)
    res = np.multiply.accumulate(x, dtype=np.int32)
    np.testing.assert_equal(res, np.multiply.accumulate(x.astype(np.int32)))

  @parameterized.parameters(INTN_TYPES)
  def testZeroSizeReduction(self, scalar_type):
    x = np.array([], dtype=scalar_type)
    for op, expected in [(np.add, 0), (np.multiply, 1)]:
      with self.subTest(op.__name__):
        np.testing.assert_equal(op.reduce(x), scalar_type(expected))


if __name__ == "__main__":
  absltest.main()
