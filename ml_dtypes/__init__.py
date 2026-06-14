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
    "int1",
    "int2",
    "int4",
    "uint1",
    "uint2",
    "uint4",
    "real",
    "imag",
]

import warnings as _warnings

from ml_dtypes._finfo import finfo
from ml_dtypes._iinfo import iinfo
from ml_dtypes._ml_dtypes_ext import bcomplex32
from ml_dtypes._ml_dtypes_ext import bfloat16
from ml_dtypes._ml_dtypes_ext import complex32
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
from ml_dtypes._ml_dtypes_ext import int1
from ml_dtypes._ml_dtypes_ext import int2
from ml_dtypes._ml_dtypes_ext import int4
from ml_dtypes._ml_dtypes_ext import uint1
from ml_dtypes._ml_dtypes_ext import uint2
from ml_dtypes._ml_dtypes_ext import uint4
import numpy as _np

bfloat16: type[_np.generic]
float4_e2m1fn: type[_np.generic]
float6_e2m3fn: type[_np.generic]
float6_e3m2fn: type[_np.generic]
float8_e3m4: type[_np.generic]
float8_e4m3: type[_np.generic]
float8_e4m3b11fnuz: type[_np.generic]
float8_e4m3fn: type[_np.generic]
float8_e4m3fnuz: type[_np.generic]
float8_e5m2: type[_np.generic]
float8_e5m2fnuz: type[_np.generic]
float8_e8m0fnu: type[_np.generic]
int1: type[_np.generic]
int2: type[_np.generic]
int4: type[_np.generic]
uint1: type[_np.generic]
uint2: type[_np.generic]
uint4: type[_np.generic]
bcomplex32: type[_np.generic]
complex32: type[_np.generic]

# Augment the C++ extension's terse docstring with a clearer class summary.
bcomplex32.__doc__ = (
    "complex<bfloat16>: a 4-byte complex number pairing two bfloat16\n"
    "halves (real + imaginary), exposed as an ml_dtypes extension dtype.\n\n"
    "WARNING: NumPy does not natively understand this custom complex dtype.\n"
    "On NumPy <2.5, arr.real / arr.imag are SILENTLY wrong; use\n"
    "ml_dtypes.real() / ml_dtypes.imag() instead (NumPy 2.5+ fixes them).\n\n"
    "These complex-aware builtins also do NOT recognize this dtype on ANY\n"
    "NumPy version -- cast to np.complex64 first, or use the workaround:\n"
    "  np.vdot(a,b)      -> np.dot(np.conjugate(a), b)\n"
    "  np.linalg.norm(a) -> np.linalg.norm(a.astype(np.complex64))\n"
    "  np.iscomplex(a)   -> ml_dtypes.imag(a) != 0\n"
    "  np.angle(a)       -> np.arctan2(ml_dtypes.imag(a), ml_dtypes.real(a))\n"
    "  np.linalg.det/inv -> cast to np.complex64 first (else they raise)\n"
    "np.abs, conjugate, arithmetic, reductions, np.dot/inner/outer, casts OK."
)


def _warn_old_numpy(fn_name: str) -> None:
  """Emit a RuntimeWarning on NumPy <2.5.

  On NumPy <2.5, arr.real / arr.imag return silently incorrect results for
  ml_dtypes complex arrays (bcomplex32, complex32). This helper itself is
  correct; the warning steers users away from arr.real/arr.imag and toward
  upgrading to NumPy 2.5+.

  Args:
    fn_name: The public function name (``"real"`` or ``"imag"``) to include
      in the warning message so callers can identify which helper fired it.
  """
  if _np.lib.NumpyVersion(_np.__version__) < "2.5.0.dev0":
    _warnings.warn(
        f"NumPy <2.5 miscomputes arr.real/arr.imag for ml_dtypes complex "
        f"arrays; this ml_dtypes.{fn_name}() call is correct, but prefer "
        "upgrading to NumPy 2.5+.",
        RuntimeWarning,
        # 1 = _warn_old_numpy, 2 = real/imag, 3 = user's call site
        stacklevel=3,
    )


def real(x: _np.ndarray) -> _np.ndarray:
  """Return the real part of a complex array.

  This is a helper that uses `x.real` except for NumPy arrays of
  bcomplex32 or complex32. NumPy cannot correctly understand that these
  are complex dtypes as of NumPy 2.4 at least.

  On NumPy <2.5, a ``RuntimeWarning`` is emitted because the legacy path
  may produce incorrect results for ml_dtypes custom complex types.

  Args:
    x: The input array.

  Returns:
    The real part of the input array.
  """
  _warn_old_numpy("real")
  if isinstance(x, _np.ndarray):
    # Use a view. We add an axes to ensure it is contiguous.
    if x.dtype.type is bcomplex32:
      return x[..., None].view(bfloat16)[..., 0]
    elif x.dtype.type is complex32:
      return x[..., None].view(_np.float16)[..., 0]

  # Otherwise, assume everything is OK with just using the normal `.real`
  return x.real


def imag(x: _np.ndarray) -> _np.ndarray:
  """Return the imaginary part of a complex array.

  This is a helper that uses `x.imag` except for NumPy arrays of
  bcomplex32 or complex32. NumPy cannot correctly understand that these
  are complex dtypes as of NumPy 2.4 at least.

  On NumPy <2.5, a ``RuntimeWarning`` is emitted because the legacy path
  may produce incorrect results for ml_dtypes custom complex types.

  Args:
    x: The input array.

  Returns:
    The imaginary part of the input array.
  """
  _warn_old_numpy("imag")
  if isinstance(x, _np.ndarray):
    # Use a view. We add an axes to ensure it is contiguous.
    if x.dtype.type is bcomplex32:
      return x[..., None].view(bfloat16)[..., 1]
    elif x.dtype.type is complex32:
      return x[..., None].view(_np.float16)[..., 1]

  # Otherwise, assume everything is OK with just using the normal `.imag`
  return x.imag
