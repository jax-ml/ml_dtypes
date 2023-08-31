/* Copyright 2023 The ml_dtypes Authors

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

#ifndef ML_DTYPES_INT4_H_
#define ML_DTYPES_INT4_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

namespace ml_dtypes {

template <typename UnderlyingTy>
struct i4 {
 private:
  UnderlyingTy v : 4;

 public:
  i4() : v(0) {}
  explicit i4(UnderlyingTy val) : v(val & 0x0F) {}
  template <typename T>
  explicit i4(T t) : i4(static_cast<UnderlyingTy>(t)) {}
  i4(const i4& other) = default;

  static constexpr i4 lowest() {
    return std::is_signed<UnderlyingTy>::value ? i4(-8) : i4(0);
  }
  static constexpr i4 highest() {
    return std::is_signed<UnderlyingTy>::value ? i4(7) : i4(15);
  }

  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  explicit operator T() const {
    return static_cast<T>(v);
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::optional<int64_t>() const { return static_cast<int64_t>(v); }

  i4 operator-() const { return i4(-v); }
  i4 operator+(const i4& other) const { return i4((v + other.v)); }
  i4 operator-(const i4& other) const { return i4((v - other.v)); }
  i4 operator*(const i4& other) const { return i4((v * other.v)); }
  i4 operator/(const i4& other) const { return i4((v / other.v)); }
  i4 operator%(const i4& other) const { return i4((v % other.v)); }

  i4 operator>>(const int amount) const { return i4((v >> amount)); }
  i4 operator<<(const int amount) const { return i4((v << amount)); }

  bool operator==(const i4& other) const { return v == other.v; }
  bool operator!=(const i4& other) const { return v != other.v; }
  bool operator<(const i4& other) const { return v < other.v; }
  bool operator>(const i4& other) const { return v > other.v; }
  bool operator<=(const i4& other) const { return v <= other.v; }
  bool operator>=(const i4& other) const { return v >= other.v; }

  bool operator==(const int64_t other) const { return v == other; }
  bool operator!=(const int64_t other) const { return v != other; }
  bool operator<(const int64_t other) const { return v < other; }
  bool operator>(const int64_t other) const { return v > other; }
  bool operator<=(const int64_t other) const { return v <= other; }
  bool operator>=(const int64_t other) const { return v >= other; }

  i4& operator++() {
    v = (v + 1) & 0x0F;
    return *this;
  }

  friend ::std::ostream& operator<<(::std::ostream& os, const i4& num) {
    os << static_cast<int16_t>(num.v);
    return os;
  }

  std::string ToString() const {
    std::ostringstream os;
    os << static_cast<int16_t>(v);
    return os.str();
  }
};

using int4 = i4<int8_t>;
using uint4 = i4<uint8_t>;

}  // namespace ml_dtypes

#endif  // ML_DTYPES_INT4_H_
