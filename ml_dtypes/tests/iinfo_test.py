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


@multi_threaded(num_workers=3)
class IinfoTest(parameterized.TestCase):

  def testIinfoInt1(self):
    info = ml_dtypes.iinfo(ml_dtypes.int1)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("int1").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("int1")).dtype)
    self.assertEqual(info.min, -1)
    self.assertEqual(info.max, 0)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.int1))
    self.assertEqual(info.bits, 1)
    self.assertEqual(info.kind, "i")
    self.assertEqual(str(info), "iinfo(min=-1, max=0, dtype=int1)")

  def testIInfoUint1(self):
    info = ml_dtypes.iinfo(ml_dtypes.uint1)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("uint1").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("uint1")).dtype)
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 1)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.uint1))
    self.assertEqual(info.bits, 1)
    self.assertEqual(info.kind, "u")
    self.assertEqual(str(info), "iinfo(min=0, max=1, dtype=uint1)")

  def testIinfoInt2(self):
    info = ml_dtypes.iinfo(ml_dtypes.int2)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("int2").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("int2")).dtype)
    self.assertEqual(info.min, -2)
    self.assertEqual(info.max, 1)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.int2))
    self.assertEqual(info.bits, 2)
    self.assertEqual(info.kind, "i")
    self.assertEqual(str(info), "iinfo(min=-2, max=1, dtype=int2)")

  def testIInfoUint2(self):
    info = ml_dtypes.iinfo(ml_dtypes.uint2)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("uint2").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("uint2")).dtype)
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 3)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.uint2))
    self.assertEqual(info.bits, 2)
    self.assertEqual(info.kind, "u")
    self.assertEqual(str(info), "iinfo(min=0, max=3, dtype=uint2)")

  def testIinfoInt4(self):
    info = ml_dtypes.iinfo(ml_dtypes.int4)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("int4").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("int4")).dtype)
    self.assertEqual(info.min, -8)
    self.assertEqual(info.max, 7)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.int4))
    self.assertEqual(info.bits, 4)
    self.assertEqual(info.kind, "i")
    self.assertEqual(str(info), "iinfo(min=-8, max=7, dtype=int4)")

  def testIInfoUint4(self):
    info = ml_dtypes.iinfo(ml_dtypes.uint4)
    self.assertEqual(info.dtype, ml_dtypes.iinfo("uint4").dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype("uint4")).dtype)
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 15)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.uint4))
    self.assertEqual(info.bits, 4)
    self.assertEqual(info.kind, "u")
    self.assertEqual(str(info), "iinfo(min=0, max=15, dtype=uint4)")

  def testIinfoInt8(self):
    # Checks iinfo succeeds for a built-in NumPy type.
    info = ml_dtypes.iinfo(np.int8)
    self.assertEqual(info.min, -128)
    self.assertEqual(info.max, 127)

  def testIinfoNonInteger(self):
    with self.assertRaises(ValueError):
      ml_dtypes.iinfo(np.float32)
    with self.assertRaises(ValueError):
      ml_dtypes.iinfo(np.complex128)
    with self.assertRaises(ValueError):
      ml_dtypes.iinfo(bool)

  @parameterized.named_parameters(
      {"testcase_name": f"_{dtype.__name__}", "dtype": np.dtype(dtype)}
      for dtype in [
          ml_dtypes.int1,
          ml_dtypes.int2,
          ml_dtypes.int4,
          np.int8,
          ml_dtypes.uint1,
          ml_dtypes.uint2,
          ml_dtypes.uint4,
          np.uint8,
      ]
  )
  def testFinfoFromArray(self, dtype):
    # Because of cacheing, passing the array and passing the dtype should
    # return the same object.
    arr = np.zeros(1, dtype=dtype)
    self.assertEqual(ml_dtypes.iinfo(arr).dtype, dtype)


if __name__ == "__main__":
  absltest.main()
