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

// Cross-dtype cast definitions and registration for all ml_dtypes types.
//
// Each custom dtype calls `BuildCastSpecs<Self>()` during module initialization
// and passes the resulting `PyArrayMethod_Spec*` array to NumPy's
// `PyArrayInitDTypeMeta_FromSpec`. In those specs, a `nullptr` entry in
// `dtypes` stands in for the dtype currently being initialized:
//   - `{nullptr, target}` registers a cast from `Self` to `target`.
//   - `{source, nullptr}` registers a cast from `source` to `Self`.
//
// Built-in NumPy dtypes already exist when `ml_dtypes` loads, so casts in both
// directions between `Self` and every built-in numeric dtype are registered
// immediately. Custom dtypes are initialized one by one, so casts between two
// custom dtypes (`Self` and `Other`) are registered when the second of the two
// is initialized (detected via `other_dm->singleton != nullptr`), ensuring each
// pair is registered once.
//
// Within-dtype casts (such as byte-order swapping between two descriptors of
// the same dtype) are defined separately in `dtype_common.h`. For the
// cross-dtype casts here, NumPy's default `resolve_descriptors` canonicalizes
// non-native byte order and buffers unaligned arrays before invoking the
// strided loops below.

#include "ml_dtypes/_src/casts.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {
namespace {

template <typename T>
inline constexpr bool is_custom_dtype_v =
    is_custom_float_v<T> || is_intn_v<T> || is_custom_complex_v<T>;

// Converts `value` to `float` as an intermediate representation for casts
// involving custom float types (all of which fit losslessly in `float`).
// Complex inputs drop their imaginary part to match NumPy's complex-to-float
// cast behavior.
template <typename T>
float CastToFloat(T value) {
  if constexpr (is_complex_v<T>) {
    return CastToFloat(value.real());
  } else {
    return static_cast<float>(value);
  }
}

// Converts `value` to `int` as an intermediate representation for casts
// involving sub-byte integers (`int1`, `int2`, `int4`, `uint1`, `uint2`,
// `uint4`). Out-of-range floats, NaNs, and infinities are mapped to 0 first
// because casting them directly to `int` is undefined behavior in C++.
template <typename T>
int CastToInt(T value) {
  if constexpr (is_complex_v<T>) {
    return CastToInt(value.real());
  } else {
    static_assert(std::numeric_limits<T>::is_specialized);
    if constexpr (!std::numeric_limits<T>::is_integer) {
      if (std::isnan(value) || std::isinf(value) ||
          value < std::numeric_limits<int>::lowest() ||
          value > std::numeric_limits<int>::max()) {
        return 0;
      }
    }
    return static_cast<int>(value);
  }
}

// True when casting `From` to `To` discards the imaginary part of a custom
// complex number and should emit `np.exceptions.ComplexWarning`. Casting to
// `bool` is excluded because `bool(z)` checks whether either part is nonzero.
template <typename From, typename To>
inline constexpr bool kWarnComplexToReal =
    is_custom_complex_v<From> && !is_complex_v<To> && !std::is_same_v<To, bool>;

// Elementwise cast loop between one custom dtype and one built-in NumPy dtype.
// Conversions route through `std::complex<float>` or `float` for complex types,
// `int` for sub-byte integer types, and `float` for custom float types.
template <typename From, typename To>
int BuiltinCastLoop(PyArrayMethod_Context* /*context*/, char* const data[],
                    const npy_intp dimensions[], const npy_intp strides[],
                    NpyAuxData* /*auxdata*/) {
  if constexpr (kWarnComplexToReal<From, To>) {
    if (GiveComplexWarning() < 0) {
      return -1;
    }
  }
  const char* src = data[0];
  char* dst = data[1];
  const npy_intp n = dimensions[0];
  const npy_intp src_stride = strides[0];
  const npy_intp dst_stride = strides[1];

  for (npy_intp i = 0; i < n; ++i) {
    const From& in = *reinterpret_cast<const From*>(src + i * src_stride);
    To& out = *reinterpret_cast<To*>(dst + i * dst_stride);
    if constexpr (is_custom_complex_v<From> || is_custom_complex_v<To>) {
      if constexpr (is_complex_v<From> && is_complex_v<To>) {
        out = static_cast<To>(static_cast<std::complex<float>>(in));
      } else if constexpr (is_complex_v<From> && !is_complex_v<To>) {
        if constexpr (std::is_same_v<To, bool>) {
          out = static_cast<bool>(in.real() != typename From::value_type(0) ||
                                  in.imag() != typename From::value_type(0));
        } else {
          out = static_cast<To>(static_cast<float>(in.real()));
        }
      } else {
        out = static_cast<To>(static_cast<float>(in));
      }
    } else if constexpr (is_intn_v<From> || is_intn_v<To>) {
      out = static_cast<To>(CastToInt(in));
    } else {
      out = static_cast<To>(CastToFloat(in));
    }
  }
  return 0;
}

// Elementwise cast loop between two custom dtypes, converting each element
// through the intermediate type `Via` (see `CustomCastVia` below).
template <typename From, typename To, typename Via>
int CustomCastLoop(PyArrayMethod_Context* /*context*/, char* const data[],
                   const npy_intp dimensions[], const npy_intp strides[],
                   NpyAuxData* /*auxdata*/) {
  if constexpr (kWarnComplexToReal<From, To>) {
    if (GiveComplexWarning() < 0) {
      return -1;
    }
  }
  const char* src = data[0];
  char* dst = data[1];
  const npy_intp n = dimensions[0];
  const npy_intp src_stride = strides[0];
  const npy_intp dst_stride = strides[1];

  for (npy_intp i = 0; i < n; ++i) {
    const From& in = *reinterpret_cast<const From*>(src + i * src_stride);
    To& out = *reinterpret_cast<To*>(dst + i * dst_stride);
    if constexpr (is_complex_v<From> && is_complex_v<To>) {
      out = static_cast<To>(static_cast<std::complex<Via>>(in));
    } else if constexpr (is_complex_v<From>) {
      out = static_cast<To>(static_cast<Via>(in.real()));
    } else {
      out = static_cast<To>(static_cast<Via>(in));
    }
  }
  return 0;
}

// --- Cast safety and casting levels ---

// Returns true if every value of custom type `From` can be represented in
// custom type `To` without losing range, precision, sign, or special values
// (NaN, infinities, negative zero).
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

// Position of `T` in NumPy's numeric kind hierarchy:
// bool (0) < unsigned int (1) < signed int (2) < float (3) < complex (4).
// Non-safe casts that stay at the same rank or move right are `same_kind`;
// casts that move left (such as float -> int or int -> uint) are `unsafe`.
template <typename T>
constexpr int KindOrder() {
  if constexpr (std::is_same_v<T, bool>) {
    return 0;
  } else if constexpr (is_complex_v<T>) {
    return 4;
  } else if constexpr (is_custom_float_v<T> || std::is_same_v<T, half> ||
                       std::is_floating_point_v<T>) {
    return 3;
  } else if constexpr (is_intn_v<T>) {
    return std::numeric_limits<T>::is_signed ? 2 : 1;
  } else {
    static_assert(std::is_integral_v<T>);
    return std::is_signed_v<T> ? 2 : 1;
  }
}

template <typename T>
inline constexpr bool kIsByteInteger =
    std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>;

// Returns true if NumPy should treat the cast from `From` to `To` as `"safe"`.
template <typename From, typename To>
constexpr bool IsSafeCast() {
  if constexpr (is_custom_dtype_v<From> && is_custom_dtype_v<To>) {
    return CustomCastIsSafe<From, To>();
  } else if constexpr (!is_custom_dtype_v<From> && is_custom_dtype_v<To>) {
    // Built-in -> custom: `bool` fits in every custom type, and `float16` fits
    // in `complex32`. Byte integers (`int8`/`uint8`) are also marked safe for
    // custom floats and complex types to preserve legacy ufunc resolution.
    if constexpr (std::is_same_v<From, bool>) {
      return true;
    } else if constexpr ((is_custom_float_v<To> || is_custom_complex_v<To>) &&
                         kIsByteInteger<From>) {
      // NumPy < 2.6 probes uint8 for Python ints; legacy ufunc loop selection
      // relies on byte-integer safe casts until ufuncs use promoters.
      // TODO(seberg): Use precise byte-integer casting levels once ufuncs use
      // promoters.
      return true;
    } else if constexpr (is_custom_complex_v<To> &&
                         std::is_same_v<From, half>) {
      return CustomFloatSafeTo<half, typename To::value_type>();
    } else {
      return false;
    }
  } else {
    // Custom -> built-in:
    // - Custom floats fit in `float32`, `float64`, `longdouble`, and all
    //   built-in complex types (`float16` is excluded for legacy ufunc
    //   resolution).
    // - Custom complex types fit in all built-in complex types.
    // - Sub-byte integers (`int1`..`int4`, `uint1`..`uint4`) fit in every wider
    //   signed integer, float, and complex type, and unsigned sub-byte integers
    //   also fit in every unsigned integer type.
    static_assert(is_custom_dtype_v<From> && !is_custom_dtype_v<To>);
    if constexpr (is_custom_float_v<From>) {
      // NumPy < 2.6 probes half for Python floats; omitting safe casts to half
      // keeps legacy ufunc resolution on float32 until ufuncs use promoters.
      // TODO(seberg): Use precise float16 casting levels once ufuncs use
      // promoters.
      return is_complex_v<To> ||
             (std::is_floating_point_v<To> && !std::is_same_v<To, half>);
    } else if constexpr (is_custom_complex_v<From>) {
      return is_complex_v<To>;
    } else {
      static_assert(is_intn_v<From>);
      if constexpr (std::is_same_v<To, bool>) {
        return false;
      } else if constexpr (std::is_integral_v<To>) {
        return !std::numeric_limits<From>::is_signed || std::is_signed_v<To>;
      } else {
        return true;  // half, float, double, long double, complex*
      }
    }
  }
}

// Computes the NumPy casting rule (`safe`, `same_kind`, or `unsafe`) for a
// cross-dtype cast from `From` to `To`.
template <typename From, typename To>
constexpr NPY_CASTING CastingLevel() {
  if constexpr (IsSafeCast<From, To>()) {
    return NPY_SAFE_CASTING;
  } else if constexpr (KindOrder<From>() <= KindOrder<To>()) {
    return NPY_SAME_KIND_CASTING;
  } else {
    return NPY_UNSAFE_CASTING;
  }
}

// Requests the Python GIL (`NPY_METH_REQUIRES_PYAPI`) only for casts that emit
// `ComplexWarning`; all other cast loops run without holding the GIL.
template <typename From, typename To>
constexpr NPY_ARRAYMETHOD_FLAGS CastMethodFlags() {
  if constexpr (kWarnComplexToReal<From, To>) {
    return NPY_METH_REQUIRES_PYAPI;
  } else {
    return static_cast<NPY_ARRAYMETHOD_FLAGS>(0);
  }
}

// --- Custom-to-custom cast graph ---

// Returns true if a direct cast between custom types `From` and `To` should be
// registered. Most pairs have a direct cast, with three intentional exceptions:
//   1. `intN` -> `intN`: only widening casts of matching signedness (`int1` ->
//      `int2` -> `int4` and `uint1` -> `uint2` -> `uint4`) are registered.
//   2. `float8_e8m0fnu`: as an exponent-only scale type, it only registers
//      direct custom casts with `bfloat16` and custom complex types.
//   3. `int4`/`uint4` <-> `float6_e2m3fn`: `float6_e2m3fn` has max value 7.5
//      and cannot represent all 4-bit integers; omitting a direct cast lets
//      NumPy promote the pair to `float32`.
template <typename From, typename To>
constexpr bool HasCustomCast() {
  if constexpr (std::is_same_v<From, To>) {
    return false;
  } else if constexpr (is_intn_v<From> && is_intn_v<To>) {
    return std::numeric_limits<From>::is_signed ==
               std::numeric_limits<To>::is_signed &&
           std::numeric_limits<From>::digits < std::numeric_limits<To>::digits;
  } else if constexpr (std::is_same_v<From, float8_e8m0fnu> ||
                       std::is_same_v<To, float8_e8m0fnu>) {
    using Other =
        std::conditional_t<std::is_same_v<From, float8_e8m0fnu>, To, From>;
    return std::is_same_v<Other, bfloat16> || is_custom_complex_v<Other>;
  } else if constexpr ((std::is_same_v<From, float6_e2m3fn> &&
                        (std::is_same_v<To, int4> ||
                         std::is_same_v<To, uint4>)) ||
                       (std::is_same_v<To, float6_e2m3fn> &&
                        (std::is_same_v<From, int4> ||
                         std::is_same_v<From, uint4>))) {
    return false;
  } else {
    return true;
  }
}

// Intermediate C++ type `Via` used by `CustomCastLoop<From, To, Via>`.
// Casts involving floats or complex types convert via `float`, while widening
// sub-byte integer casts stay in integer types.
template <typename From, typename To>
struct CustomCastVia {
  using type = float;
};
template <>
struct CustomCastVia<int1, int2> {
  using type = int4;
};
template <>
struct CustomCastVia<uint1, uint2> {
  using type = uint4;
};
template <>
struct CustomCastVia<int1, int4> {
  using type = int8_t;
};
template <>
struct CustomCastVia<int2, int4> {
  using type = int8_t;
};
template <>
struct CustomCastVia<uint1, uint4> {
  using type = uint8_t;
};
template <>
struct CustomCastVia<uint2, uint4> {
  using type = uint8_t;
};

// --- Cast spec registration helpers ---

// Appends cast specs in both directions between `Self` (represented by
// `nullptr` for `PyArrayInitDTypeMeta_FromSpec`) and `BuiltinT`.
template <typename Self, typename BuiltinT>
void AddBuiltinCastsForOne(CastSpecList& list) {
  PyArray_DTypeMeta* builtin_dm = DtypeTraits<BuiltinT>::DTypeMeta();
  list.Add("ml_dtypes_to_numpy_cast", /*from=*/nullptr, /*to=*/builtin_dm,
           CastingLevel<Self, BuiltinT>(), CastMethodFlags<Self, BuiltinT>(),
           &BuiltinCastLoop<Self, BuiltinT>);
  list.Add("numpy_to_ml_dtypes_cast", /*from=*/builtin_dm, /*to=*/nullptr,
           CastingLevel<BuiltinT, Self>(), CastMethodFlags<BuiltinT, Self>(),
           &BuiltinCastLoop<BuiltinT, Self>);
}

template <typename Self, typename... BuiltinTs>
void AddAllBuiltinCasts(CastSpecList& list) {
  (AddBuiltinCastsForOne<Self, BuiltinTs>(list), ...);
}

// Appends cast specs between `Self` and `Other` if `Other` has already been
// registered with NumPy (`other_dm->singleton != nullptr`). If `Other` has not
// been initialized yet, the pair will be registered later when `Other` calls
// `BuildCastSpecs<Other>()`.
template <typename Self, typename Other>
void AddCustomCastsForOne(CastSpecList& list) {
  if constexpr (!std::is_same_v<Self, Other>) {
    PyArray_DTypeMeta* other_dm = DtypeTraits<Other>::DTypeMeta();
    if (other_dm->singleton == nullptr) {
      return;
    }
    if constexpr (HasCustomCast<Self, Other>()) {
      using Via = typename CustomCastVia<Self, Other>::type;
      list.Add("ml_dtypes_custom_cast", /*from=*/nullptr, /*to=*/other_dm,
               CastingLevel<Self, Other>(), CastMethodFlags<Self, Other>(),
               &CustomCastLoop<Self, Other, Via>);
    }
    if constexpr (HasCustomCast<Other, Self>()) {
      using Via = typename CustomCastVia<Other, Self>::type;
      list.Add("ml_dtypes_custom_cast", /*from=*/other_dm, /*to=*/nullptr,
               CastingLevel<Other, Self>(), CastMethodFlags<Other, Self>(),
               &CustomCastLoop<Other, Self, Via>);
    }
  }
}

template <typename Self, typename... OtherTs>
void AddAllCustomCasts(CastSpecList& list) {
  (AddCustomCastsForOne<Self, OtherTs>(list), ...);
}

}  // namespace

void CastSpecList::Add(const char* name, PyArray_DTypeMeta* from,
                       PyArray_DTypeMeta* to, NPY_CASTING casting,
                       NPY_ARRAYMETHOD_FLAGS flags,
                       PyArrayMethod_StridedLoop* loop) {
  auto entry = std::make_unique<CastSpecEntry>();
  entry->dtypes[0] = from;
  entry->dtypes[1] = to;
  entry->slots[0] = {NPY_METH_strided_loop, reinterpret_cast<void*>(loop)};
  entry->slots[1] = {0, nullptr};
  entry->spec.name = name;
  entry->spec.nin = 1;
  entry->spec.nout = 1;
  entry->spec.casting = casting;
  entry->spec.flags = flags;
  entry->spec.dtypes = entry->dtypes;
  entry->spec.slots = entry->slots;
  specs_.push_back(&entry->spec);
  entries_.push_back(std::move(entry));
}

template <typename Self>
CastSpecList BuildCastSpecs() {
  CastSpecList list;
  AddAllBuiltinCasts<Self, bool, signed char, short, int, long,  // NOLINT
                     long long, unsigned char, unsigned short,   // NOLINT
                     unsigned int, unsigned long,                // NOLINT
                     unsigned long long, half, float, double,    // NOLINT
                     long double, std::complex<float>, std::complex<double>,
                     std::complex<long double>>(list);
  AddAllCustomCasts<Self, bfloat16, float8_e3m4, float8_e4m3,
                    float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz,
                    float8_e5m2, float8_e5m2fnuz, float8_e5m3fnu,
                    float8_e8m0fnu, float6_e2m3fn, float6_e3m2fn, float4_e2m1fn,
                    int1, uint1, int2, uint2, int4, uint4, bcomplex32,
                    complex32>(list);
  return list;
}

#define ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(T) \
  template CastSpecList BuildCastSpecs<T>();

ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(bfloat16)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e3m4)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e4m3)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e4m3b11fnuz)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e4m3fn)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e4m3fnuz)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e5m2)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e5m2fnuz)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e5m3fnu)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float8_e8m0fnu)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float6_e2m3fn)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float6_e3m2fn)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(float4_e2m1fn)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(int1)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(uint1)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(int2)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(uint2)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(int4)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(uint4)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(bcomplex32)
ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS(complex32)

#undef ML_DTYPES_INSTANTIATE_BUILD_CAST_SPECS

}  // namespace ml_dtypes
