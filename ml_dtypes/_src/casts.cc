/* Copyright 2017 The ml_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>

#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {
namespace {

// Performs a NumPy array cast from type 'From' to 'To' via `Via`.
template <typename From, typename To, typename Via>
void PyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
            void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);

  if constexpr (is_complex_v<From> && !is_complex_v<To>) {
    if (GiveComplexWarningNoGIL() < 0) {
      return;
    }
    for (npy_intp i = 0; i < n; ++i) {
      to[i] = static_cast<To>(static_cast<Via>(from[i].real()));
    }
  } else {
    for (npy_intp i = 0; i < n; ++i) {
      to[i] = static_cast<To>(static_cast<Via>(from[i]));
    }
  }
}

template <typename Type1, typename Type2, typename Via>
bool RegisterTwoWayCustomCast() {
  int nptype1 = DtypeTraits<Type1>::Dtype();
  int nptype2 = DtypeTraits<Type2>::Dtype();
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, PyCast<Type1, Type2, Via>) <
      0) {
    return false;
  }
  PyArray_Descr* descr2 = PyArray_DescrFromType(nptype2);
  if (PyArray_RegisterCastFunc(descr2, nptype1, PyCast<Type2, Type1, Via>) <
      0) {
    return false;
  }
  return true;
}

template <typename Type1, typename Type2, typename Via>
bool RegisterOneWayCustomCast() {
  int nptype1 = DtypeTraits<Type1>::Dtype();
  int nptype2 = DtypeTraits<Type2>::Dtype();
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, PyCast<Type1, Type2, Via>) <
      0) {
    return false;
  }
  return true;
}

// Register two-way floating point casts between the first and the other types.
template <typename T>
bool RegisterTwoWayFloatCasts() {
  return true;
}

template <typename T, typename U, typename... Args>
bool RegisterTwoWayFloatCasts() {
  return RegisterTwoWayCustomCast<T, U, float>() &&
         RegisterTwoWayFloatCasts<T, Args...>();
}

// Register two-way floating point casts between all pairs of types.
template <typename T>
bool RegisterAllFloatCasts() {
  return true;
}

template <typename T, typename U, typename... Args>
bool RegisterAllFloatCasts() {
  return RegisterTwoWayFloatCasts<T, U, Args...>() &&
         RegisterAllFloatCasts<U, Args...>();
}

}  // namespace

bool RegisterCustomCasts() {
  // Register casts between pairs of custom float dtypes.
  bool success = RegisterAllFloatCasts<
      bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz, float8_e4m3fn,
      float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz, float6_e2m3fn,
      float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  // Only registering to/from BF16 and FP32 for float8_e8m0fnu.
  success &= RegisterTwoWayCustomCast<float8_e8m0fnu, bfloat16, float>();
  success &= RegisterTwoWayCustomCast<bfloat16, float8_e8m0fnu, float>();
  success &= RegisterOneWayCustomCast<int1, int2, int4>();
  success &= RegisterOneWayCustomCast<uint1, uint2, uint4>();
  success &= RegisterOneWayCustomCast<int1, int4, int8_t>();
  success &= RegisterOneWayCustomCast<uint1, uint4, uint8_t>();
  success &= RegisterOneWayCustomCast<int2, int4, int8_t>();
  success &= RegisterOneWayCustomCast<uint2, uint4, uint8_t>();

  // Int -> float casts.
  success &= RegisterTwoWayFloatCasts<
      int1, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  success &= RegisterTwoWayFloatCasts<
      uint1, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  success &= RegisterTwoWayFloatCasts<
      int2, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  success &= RegisterTwoWayFloatCasts<
      uint2, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  success &= RegisterTwoWayFloatCasts<
      int4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  // int4 -> float6_e2m3fn is not safe and we only register safe casts.
  success &= RegisterTwoWayFloatCasts<
      uint4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  // uint4 -> float6_e2m3fn is not safe and we only register safe casts.
  return success;
}

}  // namespace ml_dtypes
