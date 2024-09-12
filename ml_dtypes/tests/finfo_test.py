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

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
from multi_thread_utils import multi_threaded
import numpy as np

ALL_DTYPES = [
    ml_dtypes.bfloat16,
    ml_dtypes.float4_e2m1fn,
    ml_dtypes.float6_e2m3fn,
    ml_dtypes.float6_e3m2fn,
    ml_dtypes.float8_e3m4,
    ml_dtypes.float8_e4m3,
    ml_dtypes.float8_e4m3b11fnuz,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e4m3fnuz,
    ml_dtypes.float8_e5m2,
    ml_dtypes.float8_e5m2fnuz,
    ml_dtypes.float8_e8m0fnu,
]

DTYPES_WITH_NO_INFINITY = [
    ml_dtypes.float8_e4m3b11fnuz,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e4m3fnuz,
    ml_dtypes.float8_e5m2fnuz,
    ml_dtypes.float8_e8m0fnu,
]

DTYPES_WITH_NO_INFINITY_AND_NO_NAN = [
    ml_dtypes.float4_e2m1fn,
    ml_dtypes.float6_e2m3fn,
    ml_dtypes.float6_e3m2fn,
]

UINT_TYPES = {
    4: np.uint8,
    6: np.uint8,
    8: np.uint8,
    16: np.uint16,
}


@multi_threaded(num_workers=3, skip_tests=["testFInfo"])
class FinfoTest(parameterized.TestCase):

  def assertNanEqual(self, x, y):
    if np.isnan(x) and np.isnan(y):
      return
    self.assertEqual(x, y)

  @parameterized.named_parameters(
      {"testcase_name": f"_{dtype.__name__}", "dtype": np.dtype(dtype)}
      for dtype in ALL_DTYPES
  )
  def testFInfo(self, dtype):
    info = ml_dtypes.finfo(dtype)

    assert ml_dtypes.finfo(dtype.name) is info
    assert ml_dtypes.finfo(dtype.type) is info

    _ = str(info)  # doesn't crash

    def make_val(val):
      return np.array(val, dtype=dtype)

    def assert_representable(val):
      self.assertEqual(make_val(val).item(), val)

    def assert_infinite(val):
      val = make_val(val)
      if dtype in DTYPES_WITH_NO_INFINITY_AND_NO_NAN:
        self.assertEqual(val, info.max)
      elif dtype in DTYPES_WITH_NO_INFINITY:
        self.assertTrue(np.isnan(val), f"expected NaN, got {val}")
      else:
        self.assertTrue(np.isposinf(val), f"expected inf, got {val}")

    def assert_zero(val):
      self.assertEqual(make_val(val), make_val(0))

    self.assertEqual(np.array(0, dtype).dtype, dtype)
    self.assertIs(info.dtype, dtype)

    if info.bits >= 8:
      self.assertEqual(info.bits, np.array(0, dtype).itemsize * 8)

    # Unsigned float => no sign bit.
    if info.min >= 0.0:
      self.assertEqual(info.nmant + info.nexp, info.bits)
    else:
      self.assertEqual(info.nmant + info.nexp + 1, info.bits)
    assert_representable(info.tiny)
    assert_representable(info.max)
    assert_representable(info.min)

    if dtype not in DTYPES_WITH_NO_INFINITY_AND_NO_NAN:
      assert_infinite(np.spacing(info.max))
    assert info.max > 0

    if info.min < 0 and dtype not in DTYPES_WITH_NO_INFINITY_AND_NO_NAN:
      # Only valid for signed floating format.
      assert_infinite(-np.spacing(info.min))
    elif info.min > 0:
      # No zero in floating point format.
      assert_infinite(0)
      assert_infinite(make_val(-1))
    elif info.min == 0:
      # Zero supported, but not negative values.
      self.assertEqual(make_val(0), 0)
      assert_infinite(make_val(-1))

    assert_representable(2.0 ** (info.maxexp - 1))
    assert_infinite(2.0**info.maxexp)

    assert_representable(info.smallest_subnormal)
    if info.min < 0:
      assert_zero(info.smallest_subnormal * 0.5)
    self.assertGreater(info.smallest_normal, 0)
    self.assertEqual(info.tiny, info.smallest_normal)

    # Identities according to the documentation:
    np.testing.assert_allclose(info.resolution, make_val(10**-info.precision))
    self.assertEqual(info.epsneg, make_val(2**info.negep))
    self.assertEqual(info.eps, make_val(2**info.machep))
    self.assertEqual(info.iexp, info.nexp)

    is_min_exponent_valid_normal = (
        make_val(2**info.minexp) == info.smallest_normal
    )
    # Check that minexp is consistent with nmant (subnormal representation)
    if not is_min_exponent_valid_normal and info.nmant > 0:
      self.assertEqual(
          make_val(2**info.minexp).view(UINT_TYPES[info.bits]),
          2**info.nmant,
      )


if __name__ == "__main__":
  absltest.main()
