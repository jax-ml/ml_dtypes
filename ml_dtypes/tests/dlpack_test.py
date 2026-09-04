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

"""Tests for DLPack interoperability with NumPy."""

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

FLOATS = [
    ml_dtypes.bfloat16,
    ml_dtypes.float8_e3m4,
    ml_dtypes.float8_e4m3,
    ml_dtypes.float8_e4m3b11fnuz,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e4m3fnuz,
    ml_dtypes.float8_e5m2,
    ml_dtypes.float8_e5m2fnuz,
    ml_dtypes.float8_e8m0fnu,
]

COMPLEX = [
    ml_dtypes.bcomplex32,
    ml_dtypes.complex32,
]


class DLPackTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if np.lib.NumpyVersion(np.__version__) < "2.5.0":
      self.skipTest("Requires NumPy 2.5+ with register_dlpack_dtype")

  @parameterized.parameters(FLOATS)
  def test_roundtrip_floats(self, dtype):
    if dtype == ml_dtypes.float8_e8m0fnu:
      vals = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
      vals2d = [[0.25, 0.5], [1.0, 2.0]]
    else:
      vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
      vals2d = [[-1.0, 0.0], [0.5, 2.0]]

    x = np.array(vals, dtype=dtype)
    y = np.from_dlpack(x)
    self.assertEqual(y.dtype, np.dtype(dtype))
    np.testing.assert_array_equal(x, y)
    self.assertTrue(np.may_share_memory(x, y))

    x2 = np.array(vals2d, dtype=dtype)
    y2 = np.from_dlpack(x2)
    self.assertEqual(y2.dtype, np.dtype(dtype))
    np.testing.assert_array_equal(x2, y2)
    self.assertTrue(np.may_share_memory(x2, y2))

  @parameterized.parameters(COMPLEX)
  def test_roundtrip_complex(self, dtype):
    vals = [-1.0 - 2.0j, 0.0 + 0.0j, 1.0 + 2.0j, 3.0 - 4.0j]
    x = np.array(vals, dtype=dtype)
    y = np.from_dlpack(x)
    self.assertEqual(y.dtype, np.dtype(dtype))
    np.testing.assert_array_equal(x, y)
    self.assertTrue(np.may_share_memory(x, y))

    vals2d = [[-1.0 - 1.0j, 0.0 + 0.0j], [1.0 + 0.5j, 2.0 - 3.0j]]
    x2 = np.array(vals2d, dtype=dtype)
    y2 = np.from_dlpack(x2)
    self.assertEqual(y2.dtype, np.dtype(dtype))
    np.testing.assert_array_equal(x2, y2)
    self.assertTrue(np.may_share_memory(x2, y2))


if __name__ == "__main__":
  absltest.main()
