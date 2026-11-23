/* Copyright 2024 The ml_dtypes Authors. All Rights Reserved.

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

#include "ml_dtypes/include/mxfloat.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace ml_dtypes {
namespace {

TEST(FloatMXe2m3Test, NumericLimits) {
  using limits = std::numeric_limits<float6_e2m3fn>;
  EXPECT_EQ(static_cast<float>(limits::min()), 1.0);
  EXPECT_EQ(static_cast<float>(limits::max()), 7.5);
  EXPECT_EQ(static_cast<float>(limits::lowest()), -7.5);
  EXPECT_EQ(static_cast<float>(limits::epsilon()), 0.125);
  EXPECT_EQ(static_cast<float>(limits::round_error()), 0.25);
  EXPECT_EQ(static_cast<float>(limits::denorm_min()), 0.125);
  EXPECT_EQ(limits::digits, 4);
  EXPECT_EQ(limits::digits10, 0);
  EXPECT_EQ(limits::max_digits10, 3);
  EXPECT_EQ(limits::min_exponent, 1);
  EXPECT_EQ(limits::min_exponent10, 0);
  EXPECT_EQ(limits::max_exponent, 3);
  EXPECT_EQ(limits::max_exponent10, 0);
  EXPECT_EQ(limits::is_iec559, false);
  EXPECT_EQ(limits::has_infinity, false);
  EXPECT_EQ(limits::has_quiet_NaN, false);
  EXPECT_EQ(limits::has_signaling_NaN, false);
}

TEST(FloatMXe3m2Test, NumericLimits) {
  using limits = std::numeric_limits<float6_e3m2fn>;
  EXPECT_EQ(static_cast<float>(limits::min()), 0.25);
  EXPECT_EQ(static_cast<float>(limits::max()), 28.0);
  EXPECT_EQ(static_cast<float>(limits::lowest()), -28.0);
  EXPECT_EQ(static_cast<float>(limits::epsilon()), 0.25);
  EXPECT_EQ(static_cast<float>(limits::round_error()), 1.0);
  EXPECT_EQ(static_cast<float>(limits::denorm_min()), 0.0625);
  EXPECT_EQ(limits::digits, 3);
  EXPECT_EQ(limits::digits10, 0);
  EXPECT_EQ(limits::max_digits10, 2);
  EXPECT_EQ(limits::min_exponent, -1);
  EXPECT_EQ(limits::min_exponent10, 0);
  EXPECT_EQ(limits::max_exponent, 5);
  EXPECT_EQ(limits::max_exponent10, 1);
  EXPECT_EQ(limits::is_iec559, false);
  EXPECT_EQ(limits::has_infinity, false);
  EXPECT_EQ(limits::has_quiet_NaN, false);
  EXPECT_EQ(limits::has_signaling_NaN, false);
}

TEST(Float4e2m1Test, NumericLimits) {
  using limits = std::numeric_limits<float4_e2m1fn>;
  EXPECT_EQ(static_cast<float>(limits::min()), 1.0);
  EXPECT_EQ(static_cast<float>(limits::max()), 6.0);
  EXPECT_EQ(static_cast<float>(limits::lowest()), -6.0);
  EXPECT_EQ(static_cast<float>(limits::epsilon()), 0.5);
  EXPECT_EQ(static_cast<float>(limits::round_error()), 1.0);
  EXPECT_EQ(static_cast<float>(limits::denorm_min()), 0.5);
  EXPECT_EQ(limits::digits, 2);
  EXPECT_EQ(limits::digits10, 0);
  EXPECT_EQ(limits::max_digits10, 2);
  EXPECT_EQ(limits::min_exponent, 1);
  EXPECT_EQ(limits::min_exponent10, 0);
  EXPECT_EQ(limits::max_exponent, 3);
  EXPECT_EQ(limits::max_exponent10, 0);
  EXPECT_EQ(limits::is_iec559, false);
  EXPECT_EQ(limits::has_infinity, false);
  EXPECT_EQ(limits::has_quiet_NaN, false);
  EXPECT_EQ(limits::has_signaling_NaN, false);
}

template <typename T>
constexpr int NumValues() {
  return 1 << T::kBits;
}

template <typename T>
class FloatMXTest : public ::testing::Test {};

struct FloatMXTestNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    if constexpr (std::is_same_v<T, float6_e2m3fn>) return "float6_e2m3fn";
    if constexpr (std::is_same_v<T, float6_e3m2fn>) return "float6_e3m2fn";
    if constexpr (std::is_same_v<T, float4_e2m1fn>) return "float4_e2m1fn";
  }
};

using FloatMXTypes =
    ::testing::Types<float6_e2m3fn, float6_e3m2fn, float4_e2m1fn>;
TYPED_TEST_SUITE(FloatMXTest, FloatMXTypes, FloatMXTestNameGenerator);

TYPED_TEST(FloatMXTest, NoInfinity) {
  using FloatMX = TypeParam;

  EXPECT_EQ(static_cast<FloatMX>(INFINITY),
            std::numeric_limits<FloatMX>::max());
  EXPECT_EQ(static_cast<FloatMX>(-INFINITY),
            std::numeric_limits<FloatMX>::lowest());
}

TYPED_TEST(FloatMXTest, Negate) {
  using FloatMX = TypeParam;

  int sign_bit = 1 << (FloatMX::kBits - 1);
  for (int i = 0; i < sign_bit; ++i) {
    FloatMX pos = FloatMX::FromRep(i);
    FloatMX neg = FloatMX::FromRep(i | sign_bit);
    EXPECT_EQ((-pos).rep(), neg.rep());
    EXPECT_EQ((-neg).rep(), pos.rep());
  }
}

TYPED_TEST(FloatMXTest, Signbit) {
  using FloatMX = TypeParam;

  FloatMX one(1.0);
  EXPECT_EQ(Eigen::numext::signbit(one).rep(), 0x00);
  EXPECT_EQ(Eigen::numext::signbit(-one).rep(), 0xff);
}

TYPED_TEST(FloatMXTest, BitCasts) {
  using FloatMX = TypeParam;

  FloatMX x = FloatMX::FromRep(0x11);
  EXPECT_EQ(Eigen::numext::bit_cast<uint8_t>(x), x.rep());
  EXPECT_EQ(Eigen::numext::bit_cast<FloatMX>(x.rep()), x);
}

TYPED_TEST(FloatMXTest, UpCasts) {
  using FloatMX = TypeParam;

  for (int i = 0; i < NumValues<FloatMX>(); ++i) {
    FloatMX mx = FloatMX::FromRep(i);

    double f64 = static_cast<double>(mx);
    float f32 = static_cast<float>(mx);
    Eigen::bfloat16 bf16 = static_cast<Eigen::bfloat16>(mx);
    Eigen::half f16 = static_cast<Eigen::half>(mx);

    EXPECT_EQ(f64, f32) << i;
    EXPECT_EQ(f32, bf16) << i;
    EXPECT_EQ(bf16, f16) << i;
  }
}

TYPED_TEST(FloatMXTest, DownCasts) {
  using FloatMX = TypeParam;

  for (int i = 0; i < NumValues<FloatMX>(); ++i) {
    float x = static_cast<float>(FloatMX::FromRep(i));

    FloatMX f64 = static_cast<FloatMX>(static_cast<double>(x));
    FloatMX f32 = static_cast<FloatMX>(static_cast<float>(x));
    FloatMX bf16 = static_cast<FloatMX>(static_cast<Eigen::bfloat16>(x));
    FloatMX f16 = static_cast<FloatMX>(static_cast<Eigen::half>(x));

    EXPECT_EQ(f64.rep(), i);
    EXPECT_EQ(f32.rep(), i);
    EXPECT_EQ(bf16.rep(), i);
    EXPECT_EQ(f16.rep(), i);
  }
}

TYPED_TEST(FloatMXTest, ConvertFromWithSaturation) {
  using FloatMX = TypeParam;

  FloatMX upper =
      FloatMX::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<FloatMX>::max()) * 2);
  EXPECT_EQ(upper, std::numeric_limits<FloatMX>::max());

  FloatMX lower =
      FloatMX::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<FloatMX>::lowest()) * 2);
  EXPECT_EQ(lower, std::numeric_limits<FloatMX>::lowest());
}

TYPED_TEST(FloatMXTest, ConvertFromWithTruncation) {
  using FloatMX = TypeParam;

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float less_than_two = Eigen::numext::bit_cast<float>(0x3FFFFFFF);
  FloatMX truncated =
      FloatMX::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  FloatMX rounded =
      FloatMX::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);

  // Truncation and rounding of a subnormal.
  int digits = std::numeric_limits<FloatMX>::digits;
  for (int i = 1; i < (1 << (digits - 1)); ++i) {
    float less_than_subnorm =
        std::nexttoward(static_cast<float>(FloatMX::FromRep(i)), 0);

    FloatMX truncated_subnorm =
        FloatMX::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
            less_than_subnorm);
    EXPECT_EQ(truncated_subnorm.rep(), i - 1);

    FloatMX rounded_subnorm =
        FloatMX::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
            less_than_subnorm);
    EXPECT_EQ(rounded_subnorm.rep(), i);
  }
}

TYPED_TEST(FloatMXTest, ConvertFromRoundToNearest) {
  using FloatMX = TypeParam;

  // Try all pairs of values and check the middle point (which should be exactly
  // representable as a float), as well as adjacent values.
  for (int i = 1; i < NumValues<FloatMX>(); ++i) {
    FloatMX left = FloatMX::FromRep(i - 1);
    FloatMX right = FloatMX::FromRep(i);
    if (!right) continue;  // Skip jump to negative zero.

    float l = static_cast<float>(left);
    float r = static_cast<float>(right);
    float m = (l + r) / 2;
    float m_minus_eps = std::nexttoward(m, l);
    float m_plus_eps = std::nexttoward(m, r);

    EXPECT_EQ(static_cast<FloatMX>(m).rep(), i & 1 ? left.rep() : right.rep());
    EXPECT_EQ(static_cast<FloatMX>(m_minus_eps).rep(), left.rep());
    EXPECT_EQ(static_cast<FloatMX>(m_plus_eps).rep(), right.rep());
  }
}

TYPED_TEST(FloatMXTest, CompareOperator) {
  using FloatMX = TypeParam;

  for (int i = 0; i < NumValues<FloatMX>(); ++i) {
    FloatMX a = FloatMX::FromRep(i);
    for (int j = 0; j < NumValues<FloatMX>(); ++j) {
      FloatMX b = FloatMX::FromRep(j);

      EXPECT_EQ(a == b, float{a} == float{b});
      EXPECT_EQ(a != b, float{a} != float{b});
      EXPECT_EQ(a < b, float{a} < float{b});
      EXPECT_EQ(a <= b, float{a} <= float{b});
      EXPECT_EQ(a > b, float{a} > float{b});
      EXPECT_EQ(a >= b, float{a} >= float{b});
    }
  }
}

#define GEN_FLOAT_TYPE_PAIRS(Type)                                           \
  std::pair<Type, float>, std::pair<Type, Eigen::bfloat16>,                  \
      std::pair<Type, Eigen::half>, std::pair<Type, float8_e3m4>,            \
      std::pair<Type, float8_e4m3>, std::pair<Type, float8_e4m3fn>,          \
      std::pair<Type, float8_e4m3fnuz>, std::pair<Type, float8_e4m3b11fnuz>, \
      std::pair<Type, float8_e5m2>, std::pair<Type, float8_e5m2fnuz>,        \
      std::pair<Type, float8_e5m3fnu>, std::pair<Type, float8_e8m0fnu>

#define GEN_TEST_TYPE_PAIRS()                                               \
  GEN_FLOAT_TYPE_PAIRS(float6_e2m3fn), GEN_FLOAT_TYPE_PAIRS(float6_e3m2fn), \
      GEN_FLOAT_TYPE_PAIRS(float4_e2m1fn),                                  \
      std::pair<float6_e2m3fn, float6_e3m2fn>,                              \
      std::pair<float4_e2m1fn, float6_e2m3fn>,                              \
      std::pair<float4_e2m1fn, float6_e3m2fn>

template <typename T>
class FloatMXCastTest : public ::testing::Test {};

struct FloatMXCastTestNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    std::string first_name =
        ::testing::internal::GetTypeName<typename T::first_type>();
    std::string second_name =
        ::testing::internal::GetTypeName<typename T::second_type>();
    return first_name + "_" + second_name;
  }
};

using FloatMXCastTypePairs = ::testing::Types<GEN_TEST_TYPE_PAIRS()>;
TYPED_TEST_SUITE(FloatMXCastTest, FloatMXCastTypePairs,
                 FloatMXCastTestNameGenerator);

TYPED_TEST(FloatMXCastTest, FromFloatMX) {
  using FloatMX = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0; i < NumValues<FloatMX>(); ++i) {
    FloatMX mx = FloatMX::FromRep(i);
    DestType converted = static_cast<DestType>(mx);
    DestType expected = static_cast<DestType>(static_cast<double>(mx));
    if (Eigen::numext::isnan(expected)) {
      EXPECT_TRUE(Eigen::numext::isnan(converted));
    } else {
      EXPECT_EQ(converted, expected);
    }
  }
}

TYPED_TEST(FloatMXCastTest, ToFloatMX) {
  using FloatMX = typename TypeParam::first_type;
  using SrcType = typename TypeParam::second_type;
  using SrcTraits = typename float8_internal::Traits<SrcType>;

  // For float8, iterate over all possible values.
  // For other floating point types, discard lower mantissa bits that do not
  // participate in rounding calculation to keep the test size reasonable.
  constexpr bool is_fp8 = sizeof(SrcType) == 1;

  int test_bits = SrcTraits::kBits, shift = 0;
  if (!is_fp8) {
    int e_bits = test_bits - std::numeric_limits<SrcType>::digits;
    int m_bits = std::numeric_limits<FloatMX>::digits + 1;
    test_bits = 1 + e_bits + m_bits;
    shift = sizeof(SrcType) * CHAR_BIT - test_bits;
  }

  using BitsType = typename SrcTraits::BitsType;
  for (int i = 0; i < (1 << test_bits); ++i) {
    BitsType value = static_cast<BitsType>(i) << shift;
    SrcType fp = Eigen::numext::bit_cast<SrcType>(value);
    FloatMX converted = static_cast<FloatMX>(fp);
    FloatMX expected = static_cast<FloatMX>(static_cast<double>(fp));
    EXPECT_EQ(converted, expected);
  }
}

template <typename To, typename From>
void ExpectCastBitsEqual(From raw, From canonical) {
  using Bits = float8_internal::GetUnsignedInteger<sizeof(To)>;
  const To from_raw = static_cast<To>(raw);
  const To from_canonical = static_cast<To>(canonical);
  if constexpr (std::is_base_of_v<float8_internal::float8_base<To>, To>) {
    constexpr uint8_t kMask = (1U << To::kBits) - 1;
    EXPECT_EQ(from_raw.rep() & kMask, from_canonical.rep() & kMask);
  } else {
    EXPECT_EQ(Eigen::numext::bit_cast<Bits>(from_raw),
              Eigen::numext::bit_cast<Bits>(from_canonical));
  }
}

TYPED_TEST(FloatMXTest, UpperPaddingBits) {
  using FloatMX = TypeParam;
  constexpr uint8_t kValueMask = (uint8_t{1} << FloatMX::kBits) - 1;
  constexpr uint8_t kSignMask = uint8_t{1} << (FloatMX::kBits - 1);

  for (int i = 0; i < 256; ++i) {
    const uint8_t b = static_cast<uint8_t>(i);
    const FloatMX raw = FloatMX::FromRep(b);
    const FloatMX c = FloatMX::FromRep(b & kValueMask);

    // Negation, abs, classification, signbit, and formatting ignore upper bits.
    ExpectCastBitsEqual<float>(-raw, -c);
    EXPECT_EQ((-raw).rep() & kValueMask, (-c).rep() & kValueMask);
    ExpectCastBitsEqual<float>(Eigen::numext::abs(raw), Eigen::numext::abs(c));
    EXPECT_EQ(Eigen::numext::abs(raw).rep() & kValueMask,
              Eigen::numext::abs(c).rep() & kValueMask);
    EXPECT_FALSE(Eigen::numext::isnan(raw));
    EXPECT_FALSE(Eigen::numext::isinf(raw));
    EXPECT_TRUE(Eigen::numext::isfinite(raw));
    EXPECT_EQ(Eigen::numext::signbit(raw).rep(),
              Eigen::numext::signbit(c).rep());
    EXPECT_EQ(Eigen::numext::signbit(raw).rep(),
              (b & kSignMask) != 0 ? 0xFF : 0x00);

    std::ostringstream raw_os, c_os;
    raw_os << raw;
    c_os << c;
    EXPECT_EQ(raw_os.str(), c_os.str());

    // Conversions to floating point types (bit exact, including -0.0 sign).
    ExpectCastBitsEqual<float>(raw, c);
    ExpectCastBitsEqual<double>(raw, c);
    ExpectCastBitsEqual<Eigen::half>(raw, c);
    ExpectCastBitsEqual<Eigen::bfloat16>(raw, c);
    ExpectCastBitsEqual<float8_e3m4>(raw, c);
    ExpectCastBitsEqual<float8_e4m3>(raw, c);
    ExpectCastBitsEqual<float8_e4m3fn>(raw, c);
    ExpectCastBitsEqual<float8_e4m3fnuz>(raw, c);
    ExpectCastBitsEqual<float8_e4m3b11fnuz>(raw, c);
    ExpectCastBitsEqual<float8_e5m2>(raw, c);
    ExpectCastBitsEqual<float8_e5m2fnuz>(raw, c);
    ExpectCastBitsEqual<float8_e5m3fnu>(raw, c);
    ExpectCastBitsEqual<float8_e8m0fnu>(raw, c);
    ExpectCastBitsEqual<float6_e2m3fn>(raw, c);
    ExpectCastBitsEqual<float6_e3m2fn>(raw, c);
    ExpectCastBitsEqual<float4_e2m1fn>(raw, c);
    EXPECT_EQ(
        (FloatMX::template ConvertTo<float8_e8m0fnu, /*kSaturate=*/true>(raw)
             .rep()),
        (FloatMX::template ConvertTo<float8_e8m0fnu, /*kSaturate=*/true>(c)
             .rep()));
    if constexpr (!std::is_same_v<FloatMX, float4_e2m1fn>) {
      EXPECT_EQ((FloatMX::template ConvertTo<float4_e2m1fn, /*kSaturate=*/false,
                                             /*kTruncate=*/true>(raw)
                     .rep()),
                (FloatMX::template ConvertTo<float4_e2m1fn, /*kSaturate=*/false,
                                             /*kTruncate=*/true>(c)
                     .rep()));
    }

    // Conversions to bool and integer types.
    EXPECT_EQ(static_cast<bool>(raw), static_cast<bool>(c));
    EXPECT_EQ(static_cast<int8_t>(raw), static_cast<int8_t>(c));
    EXPECT_EQ(static_cast<int16_t>(raw), static_cast<int16_t>(c));
    EXPECT_EQ(static_cast<int32_t>(raw), static_cast<int32_t>(c));
    EXPECT_EQ(static_cast<int64_t>(raw), static_cast<int64_t>(c));
    if (static_cast<float>(c) > -1.0f) {
      EXPECT_EQ(static_cast<uint8_t>(raw), static_cast<uint8_t>(c));
      EXPECT_EQ(static_cast<uint16_t>(raw), static_cast<uint16_t>(c));
      EXPECT_EQ(static_cast<uint32_t>(raw), static_cast<uint32_t>(c));
      EXPECT_EQ(static_cast<uint64_t>(raw), static_cast<uint64_t>(c));
    }

    // Exhaustive 256x256 comparisons.
    for (int j = 0; j < 256; ++j) {
      const FloatMX raw_rhs = FloatMX::FromRep(static_cast<uint8_t>(j));
      const FloatMX c_rhs =
          FloatMX::FromRep(static_cast<uint8_t>(j & kValueMask));
      EXPECT_EQ(raw == raw_rhs, c == c_rhs);
      EXPECT_EQ(raw != raw_rhs, c != c_rhs);
      EXPECT_EQ(raw < raw_rhs, c < c_rhs);
      EXPECT_EQ(raw <= raw_rhs, c <= c_rhs);
      EXPECT_EQ(raw > raw_rhs, c > c_rhs);
      EXPECT_EQ(raw >= raw_rhs, c >= c_rhs);
    }

    // Arithmetic with dirty and clean operands matches c.
    for (uint8_t rhs_byte :
         {uint8_t{0x00}, kSignMask, uint8_t{0x01}, kValueMask, uint8_t{0x80},
          static_cast<uint8_t>(0xC0 | kSignMask), uint8_t{0xC1},
          uint8_t{0xFF}}) {
      const FloatMX raw_rhs = FloatMX::FromRep(rhs_byte);
      const FloatMX c_rhs = FloatMX::FromRep(rhs_byte & kValueMask);
      ExpectCastBitsEqual<float>(raw + raw_rhs, c + c_rhs);
      ExpectCastBitsEqual<float>(raw - raw_rhs, c - c_rhs);
      ExpectCastBitsEqual<float>(raw * raw_rhs, c * c_rhs);
      ExpectCastBitsEqual<float>(raw / raw_rhs, c / c_rhs);
    }
  }
}

}  // namespace
}  // namespace ml_dtypes
