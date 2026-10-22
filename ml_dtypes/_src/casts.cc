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

#include <complex>
#include <cstdint>
#include <limits>

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

  if constexpr (is_complex_v<From> && is_complex_v<To>) {
    for (npy_intp i = 0; i < n; ++i) {
      to[i] = static_cast<To>(static_cast<std::complex<Via>>(from[i]));
    }
  } else if constexpr (is_complex_v<From>) {
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

// Semantic safety for explicit cast registrations. Legacy NumPy also infers
// safe casts from equal kind/itemsize, even when this predicate is false.
template <typename From, typename To>
constexpr bool CustomCastIsSafe() {
  if constexpr (is_complex_v<To>) {
    if constexpr (is_complex_v<From>) {
      return CustomFloatSafeTo<typename From::value_type,
                               typename To::value_type>();
    } else {
      return CustomCastIsSafe<From, typename To::value_type>();
    }
  } else if constexpr (is_complex_v<From>) {
    return false;
  } else if constexpr (is_intn_v<From>) {
    if constexpr (is_intn_v<To>) {
      return (!std::numeric_limits<From>::is_signed ||
              std::numeric_limits<To>::is_signed) &&
             std::numeric_limits<To>::digits >=
                 std::numeric_limits<From>::digits;
    } else {
      return IntNSafeToFloat<From, To>();
    }
  } else if constexpr (is_intn_v<To>) {
    return false;
  } else {
    return CustomFloatSafeTo<From, To>();
  }
}

template <typename From, typename To, typename Via>
bool RegisterOneWayCustomCast() {
  const int to_type = DtypeTraits<To>::Dtype();
  Safe_PyObjectPtr descr_obj = make_safe(reinterpret_cast<PyObject*>(
      PyArray_DescrFromType(DtypeTraits<From>::Dtype())));
  if (!descr_obj) {
    return false;
  }
  auto* descr = reinterpret_cast<PyArray_Descr*>(descr_obj.get());
  if (PyArray_RegisterCastFunc(descr, to_type, PyCast<From, To, Via>) < 0) {
    return false;
  }
  if constexpr (CustomCastIsSafe<From, To>()) {
    if (PyArray_RegisterCanCast(descr, to_type, NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  return true;
}

template <typename Type1, typename Type2, typename Via>
bool RegisterTwoWayCustomCast() {
  return RegisterOneWayCustomCast<Type1, Type2, Via>() &&
         RegisterOneWayCustomCast<Type2, Type1, Via>();
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
      float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz, float8_e5m3fnu,
      float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  // The exponent-only format has direct custom casts only to bfloat16 and
  // the complex types. Other combinations use a built-in common dtype.
  success &= RegisterTwoWayCustomCast<float8_e8m0fnu, bfloat16, float>();
  success &= RegisterTwoWayCustomCast<float8_e8m0fnu, bcomplex32, float>();
  success &= RegisterTwoWayCustomCast<float8_e8m0fnu, complex32, float>();
  success &= RegisterOneWayCustomCast<int1, int2, int4>();
  success &= RegisterOneWayCustomCast<uint1, uint2, uint4>();
  success &= RegisterOneWayCustomCast<int1, int4, int8_t>();
  success &= RegisterOneWayCustomCast<uint1, uint4, uint8_t>();
  success &= RegisterOneWayCustomCast<int2, int4, int8_t>();
  success &= RegisterOneWayCustomCast<uint2, uint4, uint8_t>();

  // Int -> float casts.
  success &= RegisterTwoWayFloatCasts<float8_e5m3fnu, int1, uint1, int2, uint2,
                                      int4, uint4>();
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
  // int4/uint4 <-> float6_e2m3fn has no direct cast; promotion uses float32.
  success &= RegisterTwoWayFloatCasts<
      int4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  success &= RegisterTwoWayFloatCasts<
      uint4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
      float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
      float6_e3m2fn, float4_e2m1fn, bcomplex32, complex32>();
  return success;
}

}  // namespace ml_dtypes
