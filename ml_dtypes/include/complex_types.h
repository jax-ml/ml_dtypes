/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

struct bcomplex32 : std::complex<bfloat16> {
  using std::complex<bfloat16>::complex;

  bcomplex32(const std::complex<bfloat16>& other)
      : bcomplex32(other.real(), other.imag()) {}

  bcomplex32(const bfloat16& r) : bcomplex32(r, bfloat16{0}) {}

  template <typename T>
  explicit bcomplex32(const T& r) : bcomplex32(bfloat16{r}, bfloat16{0}) {}

  template <typename T>
  explicit bcomplex32(const T& r, const T& c)
      : bcomplex32(bfloat16{r}, bfloat16{c}) {}

  template <typename T>
  explicit bcomplex32(const std::complex<T>& z)
      : bcomplex32(bfloat16{z.real()}, bfloat16{z.imag()}) {}

  operator std::complex<float>() const {
    return std::complex<float>{bfloat16{real()}, bfloat16{imag()}};
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

  complex32(const std::complex<half>& other)
      : complex32(other.real(), other.imag()) {}

  complex32(const half& r) : complex32(r, half{0}) {}

  template <typename T>
  explicit complex32(const T& r) : complex32(half{r}, half{0}) {}

  template <typename T>
  explicit complex32(const T& r, const T& c) : complex32(half{r}, half{c}) {}

  template <typename T>
  explicit complex32(const std::complex<T>& z)
      : complex32(half{z.real()}, half{z.imag()}) {}

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

#endif  // ML_DTYPES_OTHER_INEXACT_H_
