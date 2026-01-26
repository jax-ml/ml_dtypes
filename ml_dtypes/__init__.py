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

__version__ = "0.5.4"
__all__ = [
    "__version__",
    "bfloat16",
    "finfo",
    # TODO: May want to warn about .imag/.real for complex on first access.
    "bcomplex32",
    "complex32",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "iinfo",
    "int2",
    "int4",
    "uint2",
    "uint4",
]

from typing import Type

from ml_dtypes._finfo import finfo
from ml_dtypes._iinfo import iinfo
from ml_dtypes._ml_dtypes_ext import bcomplex32
from ml_dtypes._ml_dtypes_ext import complex32
from ml_dtypes._ml_dtypes_ext import bfloat16
from ml_dtypes._ml_dtypes_ext import float4_e2m1fn
from ml_dtypes._ml_dtypes_ext import float6_e2m3fn
from ml_dtypes._ml_dtypes_ext import float6_e3m2fn
from ml_dtypes._ml_dtypes_ext import float8_e3m4
from ml_dtypes._ml_dtypes_ext import float8_e4m3
from ml_dtypes._ml_dtypes_ext import float8_e4m3b11fnuz
from ml_dtypes._ml_dtypes_ext import float8_e4m3fn
from ml_dtypes._ml_dtypes_ext import float8_e4m3fnuz
from ml_dtypes._ml_dtypes_ext import float8_e5m2
from ml_dtypes._ml_dtypes_ext import float8_e5m2fnuz
from ml_dtypes._ml_dtypes_ext import float8_e8m0fnu
from ml_dtypes._ml_dtypes_ext import int2
from ml_dtypes._ml_dtypes_ext import int4
from ml_dtypes._ml_dtypes_ext import uint2
from ml_dtypes._ml_dtypes_ext import uint4
import numpy as _np

bfloat16: Type[_np.generic]
float4_e2m1fn: Type[_np.generic]
float6_e2m3fn: Type[_np.generic]
float6_e3m2fn: Type[_np.generic]
float8_e3m4: Type[_np.generic]
float8_e4m3: Type[_np.generic]
float8_e4m3b11fnuz: Type[_np.generic]
float8_e4m3fn: Type[_np.generic]
float8_e4m3fnuz: Type[_np.generic]
float8_e5m2: Type[_np.generic]
float8_e5m2fnuz: Type[_np.generic]
float8_e8m0fnu: Type[_np.generic]
int2: Type[_np.generic]
int4: Type[_np.generic]
uint2: Type[_np.generic]
uint4: Type[_np.generic]
bcomplex32: Type[_np.generic]
complex32: Type[_np.generic]

del Type


def real(x: _np.ndarray) -> _np.ndarray:
  """Helper that uses `x.real` except for NumPy arrays of
  bcomplex32 or complex32. NumPy cannot correctly understand that these
  are complex dtypes as of NumPy 2.4 at least.
  """
  if isinstance(x, _np.ndarray):
    # Use a view. We add an axes to ensure it is contiguous.
    if x.dtype.type is bcomplex32:
      return x[..., None].view(bfloat16)[..., 0]
    elif x.dtype.type is complex32:
      return x[..., None].view(_np.float16)[..., 0]

  # Otherwise, assume everything is OK with just using the normal `.real`
  return x.real


def imag(x: _np.ndarray) -> _np.ndarray:
  """Helper that uses `x.imag` except for NumPy arrays of
  bcomplex32 or complex32. NumPy cannot correctly understand that these
  are complex dtypes as of NumPy 2.4 at least.
  """
  if isinstance(x, _np.ndarray):
    # Use a view. We add an axes to ensure it is contiguous.
    if x.dtype.type is bcomplex32:
      return x[..., None].view(bfloat16)[..., 1]
    elif x.dtype.type is complex32:
      return x[..., None].view(_np.float16)[..., 1]

  # Otherwise, assume everything is OK with just using the normal `.imag`
  return x.imag
