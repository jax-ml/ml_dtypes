# Copyright 2023 The ml_dtypes Authors.
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

"""Overload of numpy.iinfo to handle dtypes defined in ml_dtypes."""

from ml_dtypes._ml_dtypes_ext import int1
from ml_dtypes._ml_dtypes_ext import int2
from ml_dtypes._ml_dtypes_ext import int4
from ml_dtypes._ml_dtypes_ext import uint1
from ml_dtypes._ml_dtypes_ext import uint2
from ml_dtypes._ml_dtypes_ext import uint4
import numpy as np

_int1_dtype = np.dtype(int1)
_uint1_dtype = np.dtype(uint1)
_int2_dtype = np.dtype(int2)
_uint2_dtype = np.dtype(uint2)
_int4_dtype = np.dtype(int4)
_uint4_dtype = np.dtype(uint4)


class iinfo:  # pylint: disable=invalid-name,missing-class-docstring
  kind: str
  bits: int
  min: int
  max: int
  dtype: np.dtype

  def __init__(self, int_type):
    # Check for dtype attribute in order to handle finfo(arr), as required by
    # the Python Array API standard.
    if hasattr(int_type, "dtype") and isinstance(int_type.dtype, np.dtype):
      int_type = int_type.dtype
    else:
      try:
        int_type = np.dtype(int_type)
      except TypeError:
        int_type = np.dtype(type(int_type))

    if int_type == _int1_dtype:
      self.dtype = _int1_dtype
      self.kind = "i"
      self.bits = 1
      self.min = -1
      self.max = 0
    elif int_type == _uint1_dtype:
      self.dtype = _uint1_dtype
      self.kind = "u"
      self.bits = 1
      self.min = 0
      self.max = 1
    elif int_type == _int2_dtype:
      self.dtype = _int2_dtype
      self.kind = "i"
      self.bits = 2
      self.min = -2
      self.max = 1
    elif int_type == _uint2_dtype:
      self.dtype = _uint2_dtype
      self.kind = "u"
      self.bits = 2
      self.min = 0
      self.max = 3
    elif int_type == _int4_dtype:
      self.dtype = _int4_dtype
      self.kind = "i"
      self.bits = 4
      self.min = -8
      self.max = 7
    elif int_type == _uint4_dtype:
      self.dtype = _uint4_dtype
      self.kind = "u"
      self.bits = 4
      self.min = 0
      self.max = 15
    else:
      ii = np.iinfo(int_type)
      self.dtype = ii.dtype
      self.kind = ii.kind
      self.bits = ii.bits
      self.min = ii.min
      self.max = ii.max

  def __repr__(self):
    return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"

  def __str__(self):
    return repr(self)
