/* Copyright 2023 The ml_dtypes Authors. All Rights Reserved.

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
#include "ml_dtypes/include/intn.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace ml_dtypes {
namespace {

template <class T>
struct is_intN : std::false_type {};
template <int kN, typename UnderlyingType>
struct is_intN<intN<kN, UnderlyingType>> : std::true_type {};

template <typename T>
inline constexpr bool is_intN_v = is_intN<T>::value;

template <typename IntN_>
class IntNTest : public ::testing::Test {};

// Helper utility for prettier test names.
struct IntNTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    if constexpr (is_intN_v<TypeParam>) {
      std::string name;
      name.reserve(5);
      if constexpr (std::is_unsigned_v<typename TypeParam::underlying_type>) {
        name.append("u");
      }
      name.append("int");
      name.append(std::to_string(TypeParam::bits));
      return name;
    }
    return std::to_string(idx);
  }
};

using IntNTypes = ::testing::Types<int1, uint1, int2, uint2, int4, uint4>;
TYPED_TEST_SUITE(IntNTest, IntNTypes, IntNTestParamNames);

TEST(IntNTest, NumericLimits) {
  EXPECT_EQ(std::numeric_limits<int4>::is_signed, true);
  EXPECT_EQ(std::numeric_limits<int4>::is_modulo, false);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::min()), -8);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::max()), 7);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::lowest()), -8);
  EXPECT_EQ(std::numeric_limits<int4>::digits, 3);
  EXPECT_EQ(std::numeric_limits<int4>::digits10, 0);
  EXPECT_EQ(std::numeric_limits<int1>::is_signed, true);
  EXPECT_EQ(std::numeric_limits<int1>::is_modulo, false);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int1>::min()), -1);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int1>::max()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int1>::lowest()), -1);
  EXPECT_EQ(std::numeric_limits<int1>::digits, 0);
  EXPECT_EQ(std::numeric_limits<int1>::digits10, 0);
}

TEST(UIntNTest, NumericLimits) {
  EXPECT_EQ(std::numeric_limits<uint4>::is_signed, false);
  EXPECT_EQ(std::numeric_limits<uint4>::is_modulo, true);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::min()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::max()), 15);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::lowest()), 0);
  EXPECT_EQ(std::numeric_limits<uint4>::digits, 4);
  EXPECT_EQ(std::numeric_limits<uint4>::digits10, 1);
  EXPECT_EQ(std::numeric_limits<uint1>::is_signed, false);
  EXPECT_EQ(std::numeric_limits<uint1>::is_modulo, true);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint1>::min()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint1>::max()), 1);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::lowest()), 0);
  EXPECT_EQ(std::numeric_limits<uint1>::digits, 1);
  EXPECT_EQ(std::numeric_limits<uint1>::digits10, 0);
}

TYPED_TEST(IntNTest, NumericLimitsBase) {
  using IntN = TypeParam;
  EXPECT_EQ(std::numeric_limits<IntN>::is_specialized, true);
  EXPECT_EQ(std::numeric_limits<IntN>::is_integer, true);
  EXPECT_EQ(std::numeric_limits<IntN>::is_exact, true);
  EXPECT_EQ(std::numeric_limits<IntN>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<IntN>::has_quiet_NaN, false);
  EXPECT_EQ(std::numeric_limits<IntN>::has_signaling_NaN, false);
#if !defined(__cplusplus) || __cplusplus < 202302L
  EXPECT_EQ(std::numeric_limits<IntN>::has_denorm, std::denorm_absent);
  EXPECT_EQ(std::numeric_limits<IntN>::has_denorm_loss, false);
#endif
  EXPECT_EQ(std::numeric_limits<IntN>::round_style, std::round_toward_zero);
  EXPECT_EQ(std::numeric_limits<IntN>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<IntN>::is_bounded, true);
  EXPECT_EQ(std::numeric_limits<IntN>::radix, 2);
  EXPECT_EQ(std::numeric_limits<IntN>::min_exponent, 0);
  EXPECT_EQ(std::numeric_limits<IntN>::min_exponent10, 0);
  EXPECT_EQ(std::numeric_limits<IntN>::max_exponent, 0);
  EXPECT_EQ(std::numeric_limits<IntN>::max_exponent10, 0);
  EXPECT_EQ(std::numeric_limits<IntN>::traps, true);
  EXPECT_EQ(std::numeric_limits<IntN>::tinyness_before, false);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::epsilon()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::round_error()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::infinity()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::quiet_NaN()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::signaling_NaN()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<IntN>::denorm_min()), 0);
}

TYPED_TEST(IntNTest, TypeTraits) {
  using IntN = TypeParam;
  EXPECT_TRUE(std::is_trivially_copyable_v<IntN>);
  EXPECT_TRUE(std::is_default_constructible_v<IntN>);
  EXPECT_TRUE(std::is_nothrow_constructible_v<IntN>);
}

TYPED_TEST(IntNTest, CreateAndAssign) {
  using IntN = TypeParam;

  // Constructors.
  EXPECT_EQ(IntN(), IntN(0));
  IntN a(1);
  EXPECT_EQ(a, IntN(1));
  IntN b(std::move(a));
  EXPECT_EQ(b, IntN(1));

  // Assignments.
  EXPECT_EQ(a = IntN(2), IntN(2));
  EXPECT_EQ(b = a, IntN(2));
  EXPECT_EQ((a = IntN(3), b = std::move(a)), IntN(3));
}

// To ensure an expression is evaluated in a constexpr context,
// we use the trick of inserting the expression in a template
// parameter.
template <int ignored>
struct ConstexprEvaluator {
  static constexpr bool val = true;
};

// To avoid warnings about unused left-side of comma expressions,
// we additionally pass the expression through a constexpr function.
template <typename T>
constexpr void ConstexprEvaluatorFunc(T&&) {}

#define TEST_CONSTEXPR(expr)                                                   \
  do {                                                                         \
    EXPECT_TRUE((ConstexprEvaluator<(ConstexprEvaluatorFunc(expr), 1)>::val)); \
  } while (false)

TYPED_TEST(IntNTest, Constexpr) {
  TEST_CONSTEXPR(int4(0));
  TEST_CONSTEXPR(static_cast<int64_t>(int4(0)));

  TEST_CONSTEXPR(-int4(1));
  TEST_CONSTEXPR(int4(0) + int4(1));
  TEST_CONSTEXPR(int4(1) - int4(0));
  TEST_CONSTEXPR(int4(0) * int4(1));
  TEST_CONSTEXPR(int4(0) / int4(1));
  TEST_CONSTEXPR(int4(0) % int4(1));

  TEST_CONSTEXPR(int4(1) & int4(0xF));
  TEST_CONSTEXPR(int4(1) | int4(0xF));
  TEST_CONSTEXPR(int4(1) ^ int4(0xF));
  TEST_CONSTEXPR(~int4(1));
  TEST_CONSTEXPR(int4(1) >> 1);
  TEST_CONSTEXPR(int4(1) << 1);

  TEST_CONSTEXPR(int4(1) == int4(1));
  TEST_CONSTEXPR(int4(1) != int4(1));
  TEST_CONSTEXPR(int4(1) < int4(1));
  TEST_CONSTEXPR(int4(1) > int4(1));
  TEST_CONSTEXPR(int4(1) <= int4(1));
  TEST_CONSTEXPR(int4(1) >= int4(1));

  TEST_CONSTEXPR(++int4(1));
  TEST_CONSTEXPR(int4(1)++);
  TEST_CONSTEXPR(--int4(1));
  TEST_CONSTEXPR(int4(1)--);

  TEST_CONSTEXPR(int4(1) += int4(2));
  TEST_CONSTEXPR(int4(1) -= int4(2));
  TEST_CONSTEXPR(int4(1) *= int4(2));
  TEST_CONSTEXPR(int4(1) /= int4(2));
  TEST_CONSTEXPR(int4(1) %= int4(2));
  TEST_CONSTEXPR(int4(1) &= int4(2));
  TEST_CONSTEXPR(int4(1) |= int4(2));
  TEST_CONSTEXPR(int4(1) ^= int4(2));
  TEST_CONSTEXPR(int4(1) >>= 1);
  TEST_CONSTEXPR(int4(1) <<= 1);
}

template <typename IntN>
IntN CreateIntNWithRandomHighBits(int val) {
  return Eigen::numext::bit_cast<IntN>(static_cast<uint8_t>(
      val | (Eigen::internal::random<uint8_t>() << IntN::bits)));
}

TYPED_TEST(IntNTest, Casts) {
  using IntN = TypeParam;

  // Explicit integer types.
  if constexpr (IntN::bits == 4) {
    EXPECT_EQ(static_cast<int>(IntN(4)), 4);
    EXPECT_EQ(static_cast<int8_t>(IntN(5)), 5);
    EXPECT_EQ(static_cast<int16_t>(IntN(6)), 6);
    EXPECT_EQ(static_cast<int32_t>(IntN(7)), 7);
    EXPECT_EQ(static_cast<int64_t>(IntN(1)), 1);
  }

  // Implicit conversion to optional.
  std::optional<int64_t> c = IntN(0);
  EXPECT_EQ(c, 0);

  // Loop through all valid values.
  for (int i = static_cast<int>(std::numeric_limits<IntN>::min());
       i <= static_cast<int>(std::numeric_limits<IntN>::max()); ++i) {
    // Round-trip.
    EXPECT_EQ(static_cast<int>(CreateIntNWithRandomHighBits<IntN>(i)), i);

    // Float truncation.
    for (int j = 1; j < 10; ++j) {
      float offset = -1.f + j * 1.f / 5;
      float f = i + offset;
      EXPECT_EQ(IntN(f), IntN(static_cast<int>(f)));
    }
  }
}

TYPED_TEST(IntNTest, Operators) {
  using IntN = TypeParam;
  for (int i = static_cast<int>(std::numeric_limits<IntN>::min());
       i <= static_cast<int>(std::numeric_limits<IntN>::max()); ++i) {
    IntN x = CreateIntNWithRandomHighBits<IntN>(i);

    EXPECT_EQ(-x, IntN(-i));
    EXPECT_EQ(~x, IntN(~i));
    IntN a;
    EXPECT_EQ((a = x, ++a), IntN(i + 1));
    EXPECT_EQ(a, IntN(i + 1));
    EXPECT_EQ((a = x, a++), IntN(i));
    EXPECT_EQ(a, IntN(i + 1));
    EXPECT_EQ((a = x, --a), IntN(i - 1));
    EXPECT_EQ(a, IntN(i - 1));
    EXPECT_EQ((a = x, a--), IntN(i));
    EXPECT_EQ(a, IntN(i - 1));

    for (int j = static_cast<int>(std::numeric_limits<IntN>::min());
         j <= static_cast<int>(std::numeric_limits<IntN>::max()); ++j) {
      IntN y = CreateIntNWithRandomHighBits<IntN>(j);

      EXPECT_EQ(x + y, IntN(i + j));
      EXPECT_EQ(x - y, IntN(i - j));
      EXPECT_EQ(x * y, IntN(i * j));
      if (j != 0) {
        EXPECT_EQ(x / y, IntN(i / j));
        EXPECT_EQ(x % y, IntN(i % j));
      }
      EXPECT_EQ(x & y, IntN(i & j));
      EXPECT_EQ(x | y, IntN(i | j));
      EXPECT_EQ(x ^ y, IntN(i ^ j));

      EXPECT_EQ(x == y, i == j);
      EXPECT_EQ(x != y, i != j);
      EXPECT_EQ(x < y, i < j);
      EXPECT_EQ(x > y, i > j);
      EXPECT_EQ(x <= y, i <= j);
      EXPECT_EQ(x >= y, i >= j);

      EXPECT_EQ(x == static_cast<int64_t>(j), i == j);
      EXPECT_EQ(x != static_cast<int64_t>(j), i != j);
      EXPECT_EQ(x < static_cast<int64_t>(j), i < j);
      EXPECT_EQ(x > static_cast<int64_t>(j), i > j);
      EXPECT_EQ(x <= static_cast<int64_t>(j), i <= j);
      EXPECT_EQ(x >= static_cast<int64_t>(j), i >= j);

      EXPECT_EQ(static_cast<int64_t>(j) == x, j == i);
      EXPECT_EQ(static_cast<int64_t>(j) != x, j != i);
      EXPECT_EQ(static_cast<int64_t>(j) < x, j < i);
      EXPECT_EQ(static_cast<int64_t>(j) > x, j > i);
      EXPECT_EQ(static_cast<int64_t>(j) <= x, j <= i);
      EXPECT_EQ(static_cast<int64_t>(j) >= x, j >= i);

      EXPECT_EQ((a = x, a += y), IntN(i + j));
      EXPECT_EQ((a = x, a -= y), IntN(i - j));
      EXPECT_EQ((a = x, a *= y), IntN(i * j));
      if (j != 0) {
        EXPECT_EQ((a = x, a /= y), IntN(i / j));
        EXPECT_EQ((a = x, a %= y), IntN(i % j));
      }
      EXPECT_EQ((a = x, a &= y), IntN(i & j));
      EXPECT_EQ((a = x, a |= y), IntN(i | j));
      EXPECT_EQ((a = x, a ^= y), IntN(i ^ j));
    }

    for (int amount = 0; amount < IntN::bits; ++amount) {
      EXPECT_EQ(x >> amount, IntN(i >> amount));
      EXPECT_EQ(x << amount, IntN(i << amount));
      EXPECT_EQ((a = x, a >>= amount), IntN(i >> amount));
      EXPECT_EQ((a = x, a <<= amount), IntN(i << amount));
    }
  }
}

TYPED_TEST(IntNTest, ToString) {
  using IntN = TypeParam;
  for (int i = static_cast<int>(std::numeric_limits<IntN>::min());
       i <= static_cast<int>(std::numeric_limits<IntN>::max()); ++i) {
    IntN x = CreateIntNWithRandomHighBits<IntN>(i);
    std::stringstream ss;
    ss << x;
    EXPECT_EQ(ss.str(), std::to_string(i));
    EXPECT_EQ(x.ToString(), std::to_string(i));
  }
}

struct CustomInt {
  constexpr CustomInt() : x(0) {}
  constexpr CustomInt(int x) : x(x) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator int() const { return x; }
  constexpr bool operator==(const CustomInt& other) const {
    return x == other.x;
  }

 private:
  int x;
};

#define GEN_DEST_TYPES(Type)                                                   \
  std::pair<Type, bool>, std::pair<Type, uint1>, std::pair<Type, uint2>,       \
      std::pair<Type, uint4>, std::pair<Type, uint8_t>,                        \
      std::pair<Type, uint16_t>, std::pair<Type, uint32_t>,                    \
      std::pair<Type, uint64_t>, std::pair<Type, int1>, std::pair<Type, int2>, \
      std::pair<Type, int4>, std::pair<Type, int8_t>,                          \
      std::pair<Type, int16_t>, std::pair<Type, int32_t>,                      \
      std::pair<Type, int64_t>, std::pair<Type, CustomInt>

#define GEN_TYPE_PAIRS()                                             \
  GEN_DEST_TYPES(int1), GEN_DEST_TYPES(uint1), GEN_DEST_TYPES(int2), \
      GEN_DEST_TYPES(uint2), GEN_DEST_TYPES(int4), GEN_DEST_TYPES(uint4)

using IntNCastTypePairs = ::testing::Types<GEN_TYPE_PAIRS()>;
template <typename CastPair>
class IntNCastTest : public ::testing::Test {};

// Helper utility for prettier test names.
struct IntNCastTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    using first_type = typename TypeParam::first_type;
    using second_type = typename TypeParam::second_type;
    return ::testing::internal::GetTypeName<first_type>() + "_" +
           ::testing::internal::GetTypeName<second_type>();
  }
};

TYPED_TEST_SUITE(IntNCastTest, IntNCastTypePairs, IntNCastTestParamNames);

TYPED_TEST(IntNCastTest, CastThroughInt) {
  using IntN = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0; i < (1 << IntN::bits); ++i) {
    IntN x = CreateIntNWithRandomHighBits<IntN>(i);
    DestType dest = static_cast<DestType>(x);
    DestType expected = static_cast<DestType>(static_cast<int>(x));
    EXPECT_EQ(dest, expected);
  }
}

TYPED_TEST(IntNCastTest, DeviceCast) {
  using IntN = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

#if defined(EIGEN_USE_GPU)
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
#elif defined(EIGEN_USE_THREADS)
  constexpr int kThreads = 4;
  Eigen::ThreadPool tp(kThreads);
  Eigen::ThreadPoolDevice device(&tp, kThreads);
#else
  Eigen::DefaultDevice device;
#endif

  const int kNumElems = 256;
  // Allocate device buffers and create device tensors.
  IntN* src_device_buffer = (IntN*)device.allocate(kNumElems * sizeof(IntN));
  DestType* dst_device_buffer =
      (DestType*)device.allocate(kNumElems * sizeof(DestType));

  Eigen::TensorMap<Eigen::Tensor<IntN, 1>, Eigen::Aligned> src_device(
      src_device_buffer, kNumElems);
  Eigen::TensorMap<Eigen::Tensor<DestType, 1>, Eigen::Aligned> dst_device(
      dst_device_buffer, kNumElems);

  // Allocate host buffers and initialize src memory.
  Eigen::Tensor<IntN, 1> src_cpu(kNumElems);
  Eigen::Tensor<DestType, 1> dst_cpu(kNumElems);
  for (int i = 0; i < kNumElems; ++i) {
    src_cpu(i) = Eigen::numext::bit_cast<IntN>(static_cast<uint8_t>(i));
  }

  // Transfer data to device, perform a cast to DestType, then transfer result
  // back to host.
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(IntN));
  dst_device.device(device) = src_device.template cast<DestType>();
  device.memcpyDeviceToHost(dst_cpu.data(), dst_device_buffer,
                            kNumElems * sizeof(DestType));
  device.synchronize();

  for (int i = 0; i < kNumElems; ++i) {
    DestType expected = static_cast<DestType>(src_cpu(i));
    EXPECT_EQ(dst_cpu(i), expected);
  }

  // Cast back from DestType to IntN.
  // First clear out the device src buffer, since that will be the destination.
  src_cpu.setZero();
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(IntN));
  src_device.device(device) = dst_device.template cast<IntN>();
  device.memcpyDeviceToHost(src_cpu.data(), src_device_buffer,
                            kNumElems * sizeof(IntN));
  device.synchronize();

  for (int i = 0; i < kNumElems; ++i) {
    IntN expected = static_cast<IntN>(dst_cpu(i));
    EXPECT_EQ(src_cpu(i), expected);
  }

  // Clean up.
  device.deallocate(src_device_buffer);
  device.deallocate(dst_device_buffer);
  device.synchronize();
}

}  // namespace
}  // namespace ml_dtypes
