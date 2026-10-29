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

"""Shared dtype lists for promotion and casting tests."""

import ml_dtypes
import numpy as np

CUSTOM_FLOATS = [
    ml_dtypes.bfloat16,
    ml_dtypes.float4_e2m1fn,
    ml_dtypes.float6_e2m3fn,
    ml_dtypes.float6_e3m2fn,
    ml_dtypes.float8_e3m4,
    ml_dtypes.float8_e4m3,
    ml_dtypes.float8_e4m3fn,
    ml_dtypes.float8_e4m3fnuz,
    ml_dtypes.float8_e4m3b11fnuz,
    ml_dtypes.float8_e5m2,
    ml_dtypes.float8_e5m2fnuz,
    ml_dtypes.float8_e5m3fnu,
    ml_dtypes.float8_e8m0fnu,
]
CUSTOM_INTS = [
    ml_dtypes.int1,
    ml_dtypes.int2,
    ml_dtypes.int4,
    ml_dtypes.uint1,
    ml_dtypes.uint2,
    ml_dtypes.uint4,
]
CUSTOM_COMPLEX = [ml_dtypes.bcomplex32, ml_dtypes.complex32]
CUSTOM_DTYPES = CUSTOM_FLOATS + CUSTOM_INTS + CUSTOM_COMPLEX
NUMPY_DTYPES = list(
    dict.fromkeys(
        map(
            np.dtype,
            [
                np.bool_,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.intc,
                np.uintc,
                np.int_,
                np.uint,
                np.longlong,
                np.ulonglong,
                np.float16,
                np.float32,
                np.float64,
                np.longdouble,
                np.complex64,
                np.complex128,
                np.clongdouble,
            ],
        )
    )
)


NUMPY_BOOLS = [d for d in NUMPY_DTYPES if d.kind == "b"]
NUMPY_INTS = [d for d in NUMPY_DTYPES if d.kind in "iu"]
NUMPY_FLOATS = [d for d in NUMPY_DTYPES if d.kind == "f"]
NUMPY_COMPLEX = [d for d in NUMPY_DTYPES if d.kind == "c"]

ALL_DTYPES = CUSTOM_DTYPES + NUMPY_DTYPES
ALL_INTS = CUSTOM_INTS + NUMPY_INTS
ALL_FLOATS = CUSTOM_FLOATS + NUMPY_FLOATS
ALL_COMPLEX = CUSTOM_COMPLEX + NUMPY_COMPLEX


def operand(dtype, form):
  if form == "type":
    return dtype
  if form == "dtype":
    return np.dtype(dtype)
  if form == "scalar":
    return dtype(1)
  if form == "0d":
    return np.ones((), dtype=dtype)
  return np.ones(2, dtype=dtype)


OPERAND_FORMS = ["type", "dtype", "scalar", "0d", "array"]
