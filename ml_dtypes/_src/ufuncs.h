/* Copyright 2022 The ml_dtypes Authors

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

#ifndef ML_DTYPES_UFUNCS_H_
#define ML_DTYPES_UFUNCS_H_

// Must be included first
// clang-format off
#include "ml_dtypes/_src/numpy.h"
// clang-format on

#include <array>    // NOLINT
#include <cmath>    // NOLINT
#include <complex>  // NOLINT
#include <cstddef>  // NOLINT
#include <limits>   // NOLINT
#include <utility>  // NOLINT
#include <vector>   // NOLINT

#include "ml_dtypes/_src/common.h"  // NOLINT

// Some versions of MSVC define a "copysign" macro which wreaks havoc.
#if defined(_MSC_VER) && defined(copysign)
#undef copysign
#endif

namespace ml_dtypes {

template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline float to_system(const T& value) {
  return static_cast<float>(value);
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline std::complex<float> to_system(const T& value) {
  return static_cast<std::complex<float>>(value);
}

// isnan definition that works for all of our float and complex types.
template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline bool my_isnan(const T& value) {
  return Eigen::numext::isnan(value);
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline bool my_isnan(const T& value) {
  return Eigen::numext::isnan(value.real()) ||
         Eigen::numext::isnan(value.imag());
}

template <typename Functor, typename OutType, typename... InTypes>
struct UFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InTypes>::Dtype()...,
            TypeDescriptor<OutType>::Dtype()};
  }
  static constexpr int kInputArity = sizeof...(InTypes);

  template <std::size_t... Is>
  static void CallImpl(std::index_sequence<Is...>, char** args,
                       const npy_intp* dimensions, const npy_intp* steps,
                       void* data) {
    std::array<const char*, kInputArity> inputs = {args[Is]...};
    char* o = args[kInputArity];
    for (npy_intp k = 0; k < *dimensions; k++) {
      *reinterpret_cast<OutType*>(o) =
          Functor()(*reinterpret_cast<const InTypes*>(inputs[Is])...);
      ([&]() { inputs[Is] += steps[Is]; }(), ...);
      o += steps[kInputArity];
    }
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    return CallImpl(std::index_sequence_for<InTypes...>(), args, dimensions,
                    steps, data);
  }
};

template <typename Functor, typename OutType, typename OutType2,
          typename... InTypes>
struct UFunc2 {
  static std::vector<int> Types() {
    return {
        TypeDescriptor<InTypes>::Dtype()...,
        TypeDescriptor<OutType>::Dtype(),
        TypeDescriptor<OutType2>::Dtype(),
    };
  }
  static constexpr int kInputArity = sizeof...(InTypes);

  template <std::size_t... Is>
  static void CallImpl(std::index_sequence<Is...>, char** args,
                       const npy_intp* dimensions, const npy_intp* steps,
                       void* data) {
    std::array<const char*, kInputArity> inputs = {args[Is]...};
    char* o0 = args[kInputArity];
    char* o1 = args[kInputArity + 1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      std::tie(*reinterpret_cast<OutType*>(o0),
               *reinterpret_cast<OutType2*>(o1)) =
          Functor()(*reinterpret_cast<const InTypes*>(inputs[Is])...);
      ([&]() { inputs[Is] += steps[Is]; }(), ...);
      o0 += steps[kInputArity];
      o1 += steps[kInputArity + 1];
    }
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    return CallImpl(std::index_sequence_for<InTypes...>(), args, dimensions,
                    steps, data);
  }
};

template <typename UFuncT, typename CustomT>
bool RegisterUFunc(PyObject* numpy, const char* name) {
  std::vector<int> types = UFuncT::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFuncT::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, TypeDescriptor<CustomT>::Dtype(), fn,
                                  const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

template <typename T>
struct Add {
  T operator()(T a, T b) { return a + b; }
};
template <typename T>
struct Subtract {
  T operator()(T a, T b) { return a - b; }
};
template <typename T>
struct Multiply {
  T operator()(T a, T b) { return a * b; }
};
template <typename T>
struct TrueDivide {
  T operator()(T a, T b) { return a / b; }
};

static std::pair<float, float> divmod_impl(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();

    if (std::isnan(a) || (a == 0.0f)) {
      return {nan, nan};
    } else {
      return {std::signbit(a) == std::signbit(b) ? inf : -inf, nan};
    }
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

template <typename T>
struct Divmod {
  std::pair<T, T> operator()(T a, T b) {
    float c, d;
    std::tie(c, d) = divmod_impl(to_system(a), to_system(b));
    return {T(c), T(d)};
  }
};
template <typename T>
struct FloorDivide {
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_integral, bool> = true>
  T operator()(T x, T y) {
    if (y == T(0)) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in floor_divide", 1);
      return T(0);
    }
    T v = x / y;
    if (((x > 0) != (y > 0)) && x % y != 0) {
      v = v - T(1);
    }
    return v;
  }
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_floating, bool> = true>
  T operator()(T a, T b) {
    return T(divmod_impl(to_system(a), to_system(b)).first);
  }
};
template <typename T>
struct Remainder {
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_integral, bool> = true>
  T operator()(T x, T y) {
    if (y == 0) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in remainder", 1);
      return T(0);
    }
    T v = x % y;
    if (v != 0 && ((v < 0) != (y < 0))) {
      v = v + y;
    }
    return v;
  }
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_floating, bool> = true>
  T operator()(T a, T b) {
    return T(divmod_impl(to_system(a), to_system(b)).second);
  }
};

template <typename T>
struct Fmod {
  T operator()(T a, T b) { return T(std::fmod(to_system(a), to_system(b))); }
};
template <typename T>
struct Negative {
  T operator()(T a) { return -a; }
};
template <typename T>
struct Positive {
  T operator()(T a) { return a; }
};
template <typename T>
struct Power {
  T operator()(T a, T b) { return T(std::pow(to_system(a), to_system(b))); }
};
template <typename T>
struct Abs {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return Eigen::numext::abs(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  typename U::value_type operator()(T a) {
    using real_type = typename U::value_type;
    return real_type(std::abs(to_system(a)));
  }
};
template <typename T>
struct Cbrt {
  T operator()(T a) { return T(std::cbrt(to_system(a))); }
};
template <typename T>
struct Ceil {
  T operator()(T a) { return T(std::ceil(to_system(a))); }
};

// Helper struct for getting a bit representation provided a byte size.
template <int kNumBytes>
struct GetUnsignedInteger;

template <>
struct GetUnsignedInteger<1> {
  using type = uint8_t;
};

template <>
struct GetUnsignedInteger<2> {
  using type = uint16_t;
};

template <typename T>
using BitsType = typename GetUnsignedInteger<sizeof(T)>::type;

template <typename T>
std::pair<BitsType<T>, BitsType<T>> SignAndMagnitude(T x) {
  const BitsType<T> x_bits = Eigen::numext::bit_cast<BitsType<T>>(x);
  // Unsigned floating point format (e.g. E8M0) => no sign bit (zero by
  // default).
  if constexpr (!std::numeric_limits<T>::is_signed) {
    return {BitsType<T>(0), x_bits};
  }
  // For types that represent NaN by -0, (i.e. *fnuz), abs(x) remains -0 without
  // flipping the sign. Therefore, we need to explicitly check the
  // most-significant bit.
  // For types without NaNs (i.e. mxfloat), use xor to keep the sign bit, which
  // may be not the most-significant bit.
  constexpr BitsType<T> kSignMask = BitsType<T>(1)
                                    << (sizeof(BitsType<T>) * CHAR_BIT - 1);
  constexpr bool has_nan = std::numeric_limits<T>::has_quiet_NaN;
  const BitsType<T> x_abs_bits =
      Eigen::numext::bit_cast<BitsType<T>>(Eigen::numext::abs(x));
  return {has_nan ? x_bits & kSignMask : x_bits ^ x_abs_bits, x_abs_bits};
}

template <typename T>
struct CopySign {
  T operator()(T a, T b) {
    // Unsigned floating point format => no change.
    if constexpr (!std::numeric_limits<T>::is_signed) {
      return a;
    }
    auto [a_sign, a_abs_bits] = SignAndMagnitude(a);
    auto [b_sign, b_abs_bits] = SignAndMagnitude(b);
    BitsType<T> rep = a_abs_bits | b_sign;
    return Eigen::numext::bit_cast<T>(rep);
  }
};

template <typename T>
struct Exp {
  T operator()(T a) { return T(std::exp(to_system(a))); }
};
template <typename T>
struct Exp2 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::exp2(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    constexpr float LOGE2 = 0.6931471805599453f;
    auto x = to_system(a) * LOGE2;
    auto res = std::exp(x);
    return T(res);
  }
};
template <typename T>
struct Expm1 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::expm1(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T x_) {
    auto x = to_system(x_);
    auto a = std::sin(x.imag() / 2);
    auto res_real = std::expm1(x.real()) * std::cos(x.imag()) - 2 * a * a;
    auto res_imag = std::exp(x.real()) * std::sin(x.imag());
    return T(res_real, res_imag);
  }
};
template <typename T>
struct Floor {
  T operator()(T a) { return T(std::floor(to_system(a))); }
};
template <typename T>
struct Frexp {
  std::pair<T, int> operator()(T a) {
    int exp;
    float f = std::frexp(to_system(a), &exp);
    return {T(f), exp};
  }
};
template <typename T>
struct Heaviside {
  T operator()(T x, T h0) {
    if (Eigen::numext::isnan(x)) {
      return x;
    }
    auto [sign_x, abs_x] = SignAndMagnitude(x);
    // x == 0
    if (abs_x == 0) {
      return h0;
    }
    return sign_x ? T(0.0f) : T(1.0f);
  }
};

template <typename T>
struct Conjugate {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return a;
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  U operator()(U a) {
    return U{a.real(), -a.imag()};
  }
};

template <typename T>
struct IsFinite {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return Eigen::numext::isfinite(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return Eigen::numext::isfinite(a.real()) &&
           Eigen::numext::isfinite(a.imag());
  }
};
template <typename T>
struct IsInf {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return Eigen::numext::isinf(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  bool operator()(T a) {
    return Eigen::numext::isinf(a.real()) || Eigen::numext::isinf(a.imag());
  }
};
template <typename T>
struct IsNan {
  bool operator()(T a) { return my_isnan(a); }
};

template <typename T>
struct Ldexp {
  T operator()(T a, int exp) { return T(std::ldexp(to_system(a), exp)); }
};
template <typename T>
struct Log {
  T operator()(T a) { return T(std::log(to_system(a))); }
};
template <typename T>
struct Log2 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::log2(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    auto x = to_system(a);
    constexpr float LOG2E = 1.442695040888963407359924681001892137f;
    return T(std::log(x) * LOG2E);
  }
};
template <typename T>
struct Log10 {
  T operator()(T a) { return T(std::log10(to_system(a))); }
};
template <typename T>
struct Log1p {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::log1p(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    auto x = to_system(a);
    auto l = std::abs(x + 1.0f);
    auto res_imag = std::atan2(x.imag(), x.real() + 1);
    auto res_real = std::log(l);
    return T(res_real, res_imag);
  }
};
template <typename T>
struct LogAddExp {
  T operator()(T bx, T by) {
    auto x = to_system(bx);
    auto y = to_system(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return T(out);
  }
};
template <typename T>
struct LogAddExp2 {
  T operator()(T bx, T by) {
    float x = to_system(bx);
    float y = to_system(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return T(out);
  }
};
template <typename T>
struct Modf {
  std::pair<T, T> operator()(T a) {
    float integral;
    float f = std::modf(to_system(a), &integral);
    return {T(f), T(integral)};
  }
};

template <typename T>
struct Reciprocal {
  T operator()(T a) { return T(1.f / to_system(a)); }
};
template <typename T>
struct Rint {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::rint(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::rint(to_system(a.real())), std::rint(to_system(a.imag())));
  }
};

template <typename T>
struct Sign {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    if (Eigen::numext::isnan(a)) {
      return a;
    }
    auto [sign_a, abs_a] = SignAndMagnitude(a);
    if (abs_a == 0) {
      return a;
    }
    return sign_a ? T(-1) : T(1);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    // The complex signum is defined via z/|z|, the implementation below
    // is adopted from NumPy.
    auto c = to_system(a);
    auto abs =
        std::hypot(c.real(), c.imag());  // NumPy uses hypot which is the same.
    constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
    if (std::isnan(abs)) {
      return T(nan, nan);
    }
    if (std::isinf(abs)) {
      if (std::isinf(c.real())) {
        if (std::isinf(c.imag())) {
          return T(nan, nan);
        } else {
          return T(c.real() > 0. ? 1. : -1., 0.);
        }
      } else {
        return T{0., c.imag() > 0 ? 1. : -1.};
      }
    } else if (abs == 0) {
      return T{0., 0.};
    }
    return T{c.real() / abs, c.imag() / abs};
  }
};
template <typename T>
struct SignBit {
  bool operator()(T a) {
    auto [sign_a, abs_a] = SignAndMagnitude(a);
    return sign_a;
  }
};
template <typename T>
struct Sqrt {
  T operator()(T a) { return T(std::sqrt(to_system(a))); }
};
template <typename T>
struct Square {
  T operator()(T a) {
    auto f = to_system(a);
    return T(f * f);
  }
};
template <typename T>
struct Trunc {
  T operator()(T a) { return T(std::trunc(to_system(a))); }
};

// Trigonometric functions
template <typename T>
struct Sin {
  T operator()(T a) { return T(std::sin(to_system(a))); }
};
template <typename T>
struct Cos {
  T operator()(T a) { return T(std::cos(to_system(a))); }
};
template <typename T>
struct Tan {
  T operator()(T a) { return T(std::tan(to_system(a))); }
};
template <typename T>
struct Arcsin {
  T operator()(T a) { return T(std::asin(to_system(a))); }
};
template <typename T>
struct Arccos {
  T operator()(T a) { return T(std::acos(to_system(a))); }
};
template <typename T>
struct Arctan {
  T operator()(T a) { return T(std::atan(to_system(a))); }
};
template <typename T>
struct Arctan2 {
  T operator()(T a, T b) { return T(std::atan2(to_system(a), to_system(b))); }
};
template <typename T>
struct Hypot {
  T operator()(T a, T b) { return T(std::hypot(to_system(a), to_system(b))); }
};
template <typename T>
struct Sinh {
  T operator()(T a) { return T(std::sinh(to_system(a))); }
};
template <typename T>
struct Cosh {
  T operator()(T a) { return T(std::cosh(to_system(a))); }
};
template <typename T>
struct Tanh {
  T operator()(T a) { return T(std::tanh(to_system(a))); }
};
template <typename T>
struct Arcsinh {
  T operator()(T a) { return T(std::asinh(to_system(a))); }
};
template <typename T>
struct Arccosh {
  T operator()(T a) { return T(std::acosh(to_system(a))); }
};
template <typename T>
struct Arctanh {
  T operator()(T a) { return T(std::atanh(to_system(a))); }
};
template <typename T>
struct Deg2rad {
  T operator()(T a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return T(to_system(a) * radians_per_degree);
  }
};
template <typename T>
struct Rad2deg {
  T operator()(T a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return T(to_system(a) * degrees_per_radian);
  }
};

template <typename T>
struct Eq {
  npy_bool operator()(T a, T b) { return a == b; }
};
template <typename T>
struct Ne {
  npy_bool operator()(T a, T b) { return a != b; }
};
template <typename T>
struct Lt {
  npy_bool operator()(T a, T b) { return a < b; }
};
template <typename T>
struct Gt {
  npy_bool operator()(T a, T b) { return a > b; }
};
template <typename T>
struct Le {
  npy_bool operator()(T a, T b) { return a <= b; }
};
template <typename T>
struct Ge {
  npy_bool operator()(T a, T b) { return a >= b; }
};
template <typename T>
struct Maximum {
  T operator()(T a, T b) { return my_isnan(a) || a > b ? a : b; }
};
template <typename T>
struct Minimum {
  T operator()(T a, T b) { return my_isnan(a) || a < b ? a : b; }
};
template <typename T>
struct Fmax {
  T operator()(T a, T b) { return my_isnan(b) || a > b ? a : b; }
};
template <typename T>
struct Fmin {
  T operator()(T a, T b) { return my_isnan(b) || a < b ? a : b; }
};

template <typename T>
struct LogicalNot {
  npy_bool operator()(T a) { return !static_cast<bool>(a); }
};
template <typename T>
struct LogicalAnd {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalOr {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalXor {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

template <typename T>
struct NextAfter {
  T operator()(T from, T to) {
    BitsType<T> from_rep = Eigen::numext::bit_cast<BitsType<T>>(from);
    BitsType<T> to_rep = Eigen::numext::bit_cast<BitsType<T>>(to);
    if (Eigen::numext::isnan(from) || Eigen::numext::isnan(to)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (from_rep == to_rep) {
      return to;
    }
    auto [from_sign, from_abs] = SignAndMagnitude(from);
    auto [to_sign, to_abs] = SignAndMagnitude(to);
    if (from_abs == 0) {
      if (to_abs == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        return Eigen::numext::bit_cast<T>(
            static_cast<BitsType<T>>(0x01 | to_sign));
      }
    }
    BitsType<T> magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign)
            ? static_cast<BitsType<T>>(-1)
            : static_cast<BitsType<T>>(1);
    BitsType<T> out_int = from_rep + magnitude_adjustment;
    T out = Eigen::numext::bit_cast<T>(out_int);
    // Some non-IEEE compatible formats may have a representation for NaN
    // instead of -0, ensure we return a zero in such cases.
    if constexpr (!std::numeric_limits<T>::is_iec559) {
      if (Eigen::numext::isnan(out)) {
        return Eigen::numext::bit_cast<T>(BitsType<T>{0});
      }
    }
    return out;
  }
};

template <typename T>
struct Spacing {
  T operator()(T x) {
    CopySign<T> copysign;
    if constexpr (!std::numeric_limits<T>::has_infinity) {
      if (Eigen::numext::abs(x) == std::numeric_limits<T>::max()) {
        if constexpr (!std::numeric_limits<T>::has_quiet_NaN) return T();
        return copysign(std::numeric_limits<T>::quiet_NaN(), x);
      }
    }
    // Compute the distance between the input and the next number with greater
    // magnitude. The result should have the sign of the input.
    T away = std::numeric_limits<T>::has_infinity
                 ? std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::max();
    away = copysign(away, x);
    return NextAfter<T>()(x, away) - x;
  }
};

}  // namespace ufuncs
}  // namespace ml_dtypes

#endif  // ML_DTYPES_UFUNCS_H_
