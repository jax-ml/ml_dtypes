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

"""Overload of numpy.finfo to handle dtypes defined in ml_dtypes."""

from typing import Dict

from ml_dtypes._ml_dtypes_ext import bfloat16
from ml_dtypes._ml_dtypes_ext import float8_e4m3b11fnuz
from ml_dtypes._ml_dtypes_ext import float8_e4m3fn
from ml_dtypes._ml_dtypes_ext import float8_e4m3fnuz
from ml_dtypes._ml_dtypes_ext import float8_e5m2
from ml_dtypes._ml_dtypes_ext import float8_e5m2fnuz
from ml_dtypes._ml_dtypes_ext import float8_p3109_p3
from ml_dtypes._ml_dtypes_ext import float8_p3109_p4
from ml_dtypes._ml_dtypes_ext import float8_p3109_p5

import numpy as np

_bfloat16_dtype = np.dtype(bfloat16)
_float8_e4m3b11fnuz_dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fn_dtype = np.dtype(float8_e4m3fn)
_float8_e4m3fnuz_dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2_dtype = np.dtype(float8_e5m2)
_float8_e5m2fnuz_dtype = np.dtype(float8_e5m2fnuz)
_float8_p3109_p3_dtype = np.dtype(float8_p3109_p3)
_float8_p3109_p4_dtype = np.dtype(float8_p3109_p4)
_float8_p3109_p5_dtype = np.dtype(float8_p3109_p5)


class _Bfloat16MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-133")
    self.smallest_subnormal = bfloat16(smallest_subnormal)


class _Float8E4m3b11fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-10")
    self.smallest_normal = float8_e4m3b11fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-13")
    self.smallest_subnormal = float8_e4m3b11fnuz(smallest_subnormal)


class _Float8E4m3fnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-9")
    self.smallest_subnormal = float8_e4m3fn(smallest_subnormal)


class _Float8E4m3fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-7")
    self.smallest_normal = float8_e4m3fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-10")
    self.smallest_subnormal = float8_e4m3fnuz(smallest_subnormal)


class _Float8E5m2MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-14")
    self.smallest_normal = float8_e5m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-16")
    self.smallest_subnormal = float8_e5m2(smallest_subnormal)


class _Float8E5m2fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-15")
    self.smallest_normal = float8_e5m2fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-17")
    self.smallest_subnormal = float8_e5m2fnuz(smallest_subnormal)


class _Float8IEEEMachArLike:

  def __init__(self, p):
    # These are hard-coded in order to independently test against the computed values in the C++ implementation
    if p == 3:
      smallest_normal = float.fromhex("0x1p-15")
      self.smallest_normal = float8_p3109_p3(smallest_normal)
      smallest_subnormal = float.fromhex("0x1p-17")
      self.smallest_subnormal = float8_p3109_p3(smallest_subnormal)

    if p == 4:
      smallest_normal = float.fromhex("0x1p-7")
      self.smallest_normal = float8_p3109_p4(smallest_normal)
      smallest_subnormal = float.fromhex("0x1p-10")
      self.smallest_subnormal = float8_p3109_p4(smallest_subnormal)

    if p == 5:
      smallest_normal = float.fromhex("0x1p-3")
      self.smallest_normal = float8_p3109_p5(smallest_normal)
      smallest_subnormal = float.fromhex("0x1p-7")
      self.smallest_subnormal = float8_p3109_p5(smallest_subnormal)


class finfo(np.finfo):  # pylint: disable=invalid-name,missing-class-docstring
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}

  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max_ = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max_)
    obj.min = bfloat16(-max_)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -126
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    # pylint: disable=protected-access
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = bfloat16(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3b11fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-10")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Ep4")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3b11fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e4m3b11fnuz(eps)
    obj.epsneg = float8_e4m3b11fnuz(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3b11fnuz(max_)
    obj.min = float8_e4m3b11fnuz(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 5
    obj.minexp = -10
    obj.precision = 1
    obj.resolution = float8_e4m3b11fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3b11fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3b11fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3fn_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-6")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Cp8")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fn_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fn(eps)
    obj.epsneg = float8_e4m3fn(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fn(max_)
    obj.min = float8_e4m3fn(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 9
    obj.minexp = -6
    obj.precision = 1
    obj.resolution = float8_e4m3fn(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3fnMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fn(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-7")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Ep7")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fnuz(eps)
    obj.epsneg = float8_e4m3fnuz(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fnuz(max_)
    obj.min = float8_e4m3fnuz(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 8
    obj.minexp = -7
    obj.precision = 1
    obj.resolution = float8_e4m3fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e5m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-14")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max_ = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2_dtype
    obj.bits = 8
    obj.eps = float8_e5m2(eps)
    obj.epsneg = float8_e5m2(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2(max_)
    obj.min = float8_e5m2(-max_)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.minexp = -14
    obj.precision = 1
    obj.resolution = float8_e5m2(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E5m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e5m2fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-15")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max_ = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e5m2fnuz(eps)
    obj.epsneg = float8_e5m2fnuz(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2fnuz(max_)
    obj.min = float8_e5m2fnuz(-max_)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.minexp = -15
    obj.precision = 1
    obj.resolution = float8_e5m2fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E5m2fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_p3109_p_finfo(p):
    def float_to_str(f):
      return "%6.2e" % float(f)

    # pylint: disable=protected-access
    obj = object.__new__(np.finfo)

    if p == 3:
      dtype = float8_p3109_p3
      obj.dtype = _float8_p3109_p3_dtype
    elif p == 4:
      dtype = float8_p3109_p4
      obj.dtype = _float8_p3109_p4_dtype
    elif p == 5:
      dtype = float8_p3109_p5
      obj.dtype = _float8_p3109_p5_dtype
    else:
      raise NotImplementedError()

    obj._machar = _Float8IEEEMachArLike(p)

    bias = 2 ** (7 - p)
    tiny = obj._machar.smallest_normal
    machep = 1 - p
    eps = 2.0**machep
    negep = -p
    epsneg = 2.0**negep
    max_ = (1 - 2 ** (1 - p)) * 2**bias  #      1'0000 - 0'0010 = 0'1110

    if p == 3:
      assert tiny == float.fromhex("0x1p-15")
      assert eps == float.fromhex("0x1p-2")
      assert epsneg == float.fromhex("0x1p-3")
      assert max_ == float.fromhex("0x1.8p15")
    elif p == 4:
      assert tiny == float.fromhex("0x1p-7")
      assert eps == float.fromhex("0x1p-3")
      assert epsneg == float.fromhex("0x1p-4")
      assert max_ == float.fromhex("0x1.Cp7")
    elif p == 5:
      assert tiny == float.fromhex("0x1p-3")
      assert eps == float.fromhex("0x1p-4")
      assert epsneg == float.fromhex("0x1p-5")
      assert max_ == float.fromhex("0x1.Ep3")
    else:
      raise NotImplementedError()

    obj.bits = 8

    # nextafter(1.0, Inf) - 1.0
    obj.eps = dtype(eps)

    # The exponent that yields eps.
    obj.machep = machep

    # 1.0 = nextafter(1.0, -Inf)
    obj.epsneg = dtype(epsneg)

    # The exponent that yields epsneg.
    obj.negep = negep

    # The largest representable number.
    obj.max = dtype(max_)

    # The smallest representable number, typically -max.
    obj.min = dtype(-max_)

    obj.nexp = 8 - p
    obj.nmant = p - 1
    obj.iexp = obj.nexp
    obj.maxexp = bias
    obj.minexp = 1 - bias

    # The approximate number of decimal digits to which this kind of float is precise.
    obj.precision = 1 if p < 4 else 2

    # The approximate decimal resolution of this type, i.e., 10**-precision.
    obj.resolution = dtype(10**-obj.precision)

    if not hasattr(obj, "tiny"):
      obj.tiny = dtype(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(obj.resolution)
    # pylint: enable=protected-access
    return obj

  def __new__(cls, dtype):
    for ty, constructor in (
        (_bfloat16_dtype, cls._bfloat16_finfo),
        (_float8_e4m3b11fnuz_dtype, cls._float8_e4m3b11fnuz_finfo),
        (_float8_e4m3fn_dtype, cls._float8_e4m3fn_finfo),
        (_float8_e4m3fnuz_dtype, cls._float8_e4m3fnuz_finfo),
        (_float8_e5m2_dtype, cls._float8_e5m2_finfo),
        (_float8_e5m2fnuz_dtype, cls._float8_e5m2fnuz_finfo),
        (_float8_p3109_p3_dtype, lambda: cls._float8_p3109_p_finfo(3)),
        (_float8_p3109_p4_dtype, lambda: cls._float8_p3109_p_finfo(4)),
        (_float8_p3109_p5_dtype, lambda: cls._float8_p3109_p_finfo(5)),
    ):
      if isinstance(dtype, str) and dtype == ty.name or dtype == ty:
        if ty not in cls._finfo_cache:
          cls._finfo_cache[ty] = constructor()
        return cls._finfo_cache[ty]

    return super().__new__(cls, dtype)
