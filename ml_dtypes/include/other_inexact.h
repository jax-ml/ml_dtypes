/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ML_DTYPES_OTHER_INEXACT_H_
#define ML_DTYPES_OTHER_INEXACT_H_

// Extensions for half and bfloat16 and their corresponding complex types.

#include <complex>
#include <limits>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"

namespace ml_dtypes {

struct bfloat16 : Eigen::bfloat16 {
  using Eigen::bfloat16::bfloat16;

  // Allow implicit conversions from float, Eigen forbids it due to it being
  // lossy.
  bfloat16(float d) : Eigen::bfloat16(static_cast<float>(d)) {}
  bfloat16(int i) : bfloat16(static_cast<float>(i)) {}
  bfloat16(double d) : bfloat16(static_cast<float>(d)) {}
};

struct bcomplex32 : std::complex<bfloat16> {
  using std::complex<bfloat16>::complex;

  bcomplex32(const bcomplex32& other)
      : std::complex<bfloat16>(other.real(), other.imag()) {}

  template <typename T>
  bcomplex32(const std::complex<T>& z)
      : std::complex<bfloat16>(z.real(), z.imag()) {}

  operator std::complex<float>() const {
    return std::complex<float>{real(), imag()};
  }

  operator bool() const { return real() != 0 || imag() != 0; }

  bcomplex32 operator/(const bcomplex32& other) const {
    return bcomplex32(std::complex<float>(*this) / std::complex<float>(other));
  }

  // Lexicographic comparison operators (real first, then imaginary)
  friend bool operator<(const bcomplex32& a, const bcomplex32& b) {
    return (a.real() < b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() < b.imag());
  }
  friend bool operator>(const bcomplex32& a, const bcomplex32& b) {
    return (a.real() > b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() > b.imag());
  }
  friend bool operator<=(const bcomplex32& a, const bcomplex32& b) {
    return (a.real() < b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() <= b.imag());
  }
  friend bool operator>=(const bcomplex32& a, const bcomplex32& b) {
    return (a.real() > b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() >= b.imag());
  }
};

struct complex32 : std::complex<half> {
  using std::complex<half>::complex;

  complex32(const complex32& other)
      : std::complex<half>(other.real(), other.imag()) {}

  template <typename T>
  complex32(const std::complex<T>& z)
      : std::complex<half>(z.real(), z.imag()) {}

  operator std::complex<float>() const {
    return std::complex<float>{real(), imag()};
  }

  operator bool() const { return real() != 0 || imag() != 0; }

  complex32 operator/(const complex32& other) const {
    return complex32(std::complex<float>(*this) / std::complex<float>(other));
  }

  friend bool operator<(const complex32& a, const complex32& b) {
    return (a.real() < b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() < b.imag());
  }
  friend bool operator>(const complex32& a, const complex32& b) {
    return (a.real() > b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() > b.imag());
  }
  friend bool operator<=(const complex32& a, const complex32& b) {
    return (a.real() < b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() <= b.imag());
  }
  friend bool operator>=(const complex32& a, const complex32& b) {
    return (a.real() > b.real() && !std::isnan(a.imag()) &&
            !std::isnan(b.imag())) ||
           (a.real() == b.real() && a.imag() >= b.imag());
  }
};

template <>
inline constexpr bool is_complex_v<bcomplex32> = true;
template <>
inline constexpr bool is_complex_v<complex32> = true;

}  // namespace ml_dtypes

namespace std {
// Specialize std::numeric_limits for half and bfloat16
template <>
class numeric_limits<ml_dtypes::half> : public numeric_limits<Eigen::half> {
 public:
  static constexpr bool is_specialized = true;
};
template <>
class numeric_limits<ml_dtypes::bfloat16>
    : public numeric_limits<Eigen::bfloat16> {
 public:
  static constexpr bool is_specialized = true;
};
bool isfinite(ml_dtypes::half val) noexcept {
  return Eigen::numext::isfinite(static_cast<Eigen::half>(val));
}
bool isfinite(ml_dtypes::bfloat16 val) noexcept {
  return Eigen::numext::isfinite(static_cast<Eigen::bfloat16>(val));
}
}  // namespace std

#endif  // ML_DTYPES_OTHER_INEXACT_H_
