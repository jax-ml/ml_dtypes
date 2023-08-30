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
#include "include/int4.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace ml_dtypes {
namespace {

template <typename Int4_>
class Int4Test : public ::testing::Test {};

// Helper utility for prettier test names.
struct Int4TestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    if constexpr (std::is_same_v<TypeParam, int4>) {
      return "int4";
    } else if constexpr (std::is_same_v<TypeParam, uint4>) {
      return "uint4";
    }
    return std::to_string(idx);
  }
};

using Int4Types = ::testing::Types<int4, uint4>;
TYPED_TEST_SUITE(Int4Test, Int4Types, Int4TestParamNames);

TEST(Int4Test, NumericLimits) {
  EXPECT_EQ(std::numeric_limits<int4>::is_signed, true);
  EXPECT_EQ(std::numeric_limits<int4>::is_modulo, false);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::min()), -8);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::max()), 7);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<int4>::lowest()), -8);
  EXPECT_EQ(std::numeric_limits<int4>::digits, 3);
  EXPECT_EQ(std::numeric_limits<int4>::digits10, 0);
}

TEST(UInt4Test, NumericLimits) {
  EXPECT_EQ(std::numeric_limits<uint4>::is_signed, false);
  EXPECT_EQ(std::numeric_limits<uint4>::is_modulo, true);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::min()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::max()), 15);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<uint4>::lowest()), 0);
  EXPECT_EQ(std::numeric_limits<uint4>::digits, 4);
  EXPECT_EQ(std::numeric_limits<uint4>::digits10, 1);
}

TYPED_TEST(Int4Test, NumericLimitsBase) {
  using Int4 = TypeParam;
  EXPECT_EQ(std::numeric_limits<Int4>::is_specialized, true);
  EXPECT_EQ(std::numeric_limits<Int4>::is_integer, true);
  EXPECT_EQ(std::numeric_limits<Int4>::is_exact, true);
  EXPECT_EQ(std::numeric_limits<Int4>::has_infinity, false);
  EXPECT_EQ(std::numeric_limits<Int4>::has_quiet_NaN, false);
  EXPECT_EQ(std::numeric_limits<Int4>::has_signaling_NaN, false);
  EXPECT_EQ(std::numeric_limits<Int4>::has_denorm, std::denorm_absent);
  EXPECT_EQ(std::numeric_limits<Int4>::has_denorm_loss, false);
  EXPECT_EQ(std::numeric_limits<Int4>::round_style, std::round_toward_zero);
  EXPECT_EQ(std::numeric_limits<Int4>::is_iec559, false);
  EXPECT_EQ(std::numeric_limits<Int4>::is_bounded, true);
  EXPECT_EQ(std::numeric_limits<Int4>::radix, 2);
  EXPECT_EQ(std::numeric_limits<Int4>::min_exponent, 0);
  EXPECT_EQ(std::numeric_limits<Int4>::min_exponent10, 0);
  EXPECT_EQ(std::numeric_limits<Int4>::max_exponent, 0);
  EXPECT_EQ(std::numeric_limits<Int4>::max_exponent10, 0);
  EXPECT_EQ(std::numeric_limits<Int4>::traps, true);
  EXPECT_EQ(std::numeric_limits<Int4>::tinyness_before, false);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::epsilon()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::round_error()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::infinity()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::quiet_NaN()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::signaling_NaN()), 0);
  EXPECT_EQ(static_cast<int>(std::numeric_limits<Int4>::denorm_min()), 0);
}

TYPED_TEST(Int4Test, CreateAndAssign) {
  using Int4 = TypeParam;

  // Constructors.
  EXPECT_EQ(Int4(), Int4(0));
  Int4 a(1);
  EXPECT_EQ(a, Int4(1));
  Int4 b(std::move(a));
  EXPECT_EQ(b, Int4(1));

  // Assignments.
  EXPECT_EQ(a = Int4(2), Int4(2));
  EXPECT_EQ(b = a, Int4(2));
  EXPECT_EQ((a = Int4(3), b = std::move(a)), Int4(3));
}

// To ensure an expression is evaluated in a constexpr context,
// we use the trick of inserting the expression in a template
// parameter.
template <int ignored>
struct ConstexprEvaluator {
  static constexpr bool val = true;
};

// To avoid warnings about unused left-side of comma expressions,
// we additionally pass the expression through a contexpr function.
template <typename T>
constexpr void ConstexprEvaluatorFunc(T&&){};

#define TEST_CONSTEXPR(expr)                                                   \
  do {                                                                         \
    EXPECT_TRUE((ConstexprEvaluator<(ConstexprEvaluatorFunc(expr), 1)>::val)); \
  } while (false)

TYPED_TEST(Int4Test, Constexpr) {
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

TYPED_TEST(Int4Test, Casts) {
  using Int4 = TypeParam;

  // Explicit integer types.
  EXPECT_EQ(static_cast<int>(Int4(4)), 4);
  EXPECT_EQ(static_cast<int8_t>(Int4(5)), 5);
  EXPECT_EQ(static_cast<int16_t>(Int4(6)), 6);
  EXPECT_EQ(static_cast<int32_t>(Int4(7)), 7);
  EXPECT_EQ(static_cast<int64_t>(Int4(1)), 1);

  // Implicit conversion to optional.
  std::optional<int64_t> c = Int4(2);
  EXPECT_EQ(c, 2);

  // Loop through all valid values.
  for (int i = static_cast<int>(std::numeric_limits<Int4>::min());
       i <= static_cast<int>(std::numeric_limits<Int4>::max()); ++i) {
    // Round-trip.
    EXPECT_EQ(static_cast<int>(Int4(i)), i);

    // Float truncation.
    for (int j = 1; j < 10; ++j) {
      float offset = -1.f + j * 1.f / 5;
      float f = i + offset;
      EXPECT_EQ(Int4(f), Int4(static_cast<int>(f)));
    }
  }
}

TYPED_TEST(Int4Test, Operators) {
  using Int4 = TypeParam;
  for (int i = static_cast<int>(std::numeric_limits<Int4>::min());
       i <= static_cast<int>(std::numeric_limits<Int4>::max()); ++i) {
    Int4 x = Int4(i);

    EXPECT_EQ(-x, Int4(-i));
    EXPECT_EQ(~x, Int4(~i));
    Int4 a;
    EXPECT_EQ((a = x, ++a), Int4(i + 1));
    EXPECT_EQ(a, Int4(i + 1));
    EXPECT_EQ((a = x, a++), Int4(i));
    EXPECT_EQ(a, Int4(i + 1));
    EXPECT_EQ((a = x, --a), Int4(i - 1));
    EXPECT_EQ(a, Int4(i - 1));
    EXPECT_EQ((a = x, a--), Int4(i));
    EXPECT_EQ(a, Int4(i - 1));

    for (int j = static_cast<int>(std::numeric_limits<Int4>::min());
         j <= static_cast<int>(std::numeric_limits<Int4>::max()); ++j) {
      Int4 y = Int4(j);

      EXPECT_EQ(x + y, Int4(i + j));
      EXPECT_EQ(x - y, Int4(i - j));
      EXPECT_EQ(x * y, Int4(i * j));
      if (j != 0) {
        EXPECT_EQ(x / y, Int4(i / j));
        EXPECT_EQ(x % y, Int4(i % j));
      }
      EXPECT_EQ(x & y, Int4(i & j));
      EXPECT_EQ(x | y, Int4(i | j));
      EXPECT_EQ(x ^ y, Int4(i ^ j));

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

      EXPECT_EQ((a = x, a += y), Int4(i + j));
      EXPECT_EQ((a = x, a -= y), Int4(i - j));
      EXPECT_EQ((a = x, a *= y), Int4(i * j));
      if (j != 0) {
        EXPECT_EQ((a = x, a /= y), Int4(i / j));
        EXPECT_EQ((a = x, a %= y), Int4(i % j));
      }
      EXPECT_EQ((a = x, a &= y), Int4(i & j));
      EXPECT_EQ((a = x, a |= y), Int4(i | j));
      EXPECT_EQ((a = x, a ^= y), Int4(i ^ j));
    }

    for (int amount = 0; amount < 4; ++amount) {
      EXPECT_EQ(x >> amount, Int4(i >> amount));
      EXPECT_EQ(x << amount, Int4(i << amount));
      EXPECT_EQ((a = x, a >>= amount), Int4(i >> amount));
      EXPECT_EQ((a = x, a <<= amount), Int4(i << amount));
    }
  }
}

TYPED_TEST(Int4Test, ToString) {
  using Int4 = TypeParam;
  for (int i = static_cast<int>(std::numeric_limits<Int4>::min());
       i <= static_cast<int>(std::numeric_limits<Int4>::max()); ++i) {
    Int4 x = Int4(i);
    std::stringstream ss;
    ss << x;
    EXPECT_EQ(ss.str(), std::to_string(i));
    EXPECT_EQ(x.ToString(), std::to_string(i));
  }
}

#define GEN_DEST_TYPES(Type)                                               \
  std::pair<Type, bool>, std::pair<Type, uint4>, std::pair<Type, uint8_t>, \
      std::pair<Type, uint16_t>, std::pair<Type, uint32_t>,                \
      std::pair<Type, uint64_t>, std::pair<Type, int4>,                    \
      std::pair<Type, int8_t>, std::pair<Type, int16_t>,                   \
      std::pair<Type, int32_t>, std::pair<Type, int64_t>

#define GEN_TYPE_PAIRS() GEN_DEST_TYPES(int4), GEN_DEST_TYPES(uint4)

using Int4CastTypePairs = ::testing::Types<GEN_TYPE_PAIRS()>;
template <typename CastPair>
class Int4CastTest : public ::testing::Test {};

// Helper utility for prettier test names.
struct Int4CastTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    using first_type = typename TypeParam::first_type;
    using second_type = typename TypeParam::second_type;
    return ::testing::internal::GetTypeName<first_type>() + "_" +
           ::testing::internal::GetTypeName<second_type>();
  }
};

TYPED_TEST_SUITE(Int4CastTest, Int4CastTypePairs, Int4CastTestParamNames);

TYPED_TEST(Int4CastTest, CastThroughInt) {
  using Int4 = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0x00; i <= 0x0F; ++i) {
    Int4 x = Int4(i);
    DestType dest = static_cast<DestType>(x);
    DestType expected = static_cast<DestType>(static_cast<int>(x));
    EXPECT_EQ(dest, expected);
  }
}

TYPED_TEST(Int4CastTest, DeviceCast) {
  using Int4 = typename TypeParam::first_type;
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
  Int4* src_device_buffer = (Int4*)device.allocate(kNumElems * sizeof(Int4));
  DestType* dst_device_buffer =
      (DestType*)device.allocate(kNumElems * sizeof(DestType));

  Eigen::TensorMap<Eigen::Tensor<Int4, 1>, Eigen::Aligned> src_device(
      src_device_buffer, kNumElems);
  Eigen::TensorMap<Eigen::Tensor<DestType, 1>, Eigen::Aligned> dst_device(
      dst_device_buffer, kNumElems);

  // Allocate host buffers and initially src memory.
  Eigen::Tensor<Int4, 1> src_cpu(kNumElems);
  Eigen::Tensor<DestType, 1> dst_cpu(kNumElems);
  for (int i = 0; i < kNumElems; ++i) {
    src_cpu(i) = Eigen::numext::bit_cast<Int4>(static_cast<uint8_t>(i));
  }

  // Transfer data to device, perform a cast to DestType, then transfer result
  // back to host.
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(Int4));
  dst_device.device(device) = src_device.template cast<DestType>();
  device.memcpyDeviceToHost(dst_cpu.data(), dst_device_buffer,
                            kNumElems * sizeof(DestType));
  device.synchronize();

  for (int i = 0; i < kNumElems; ++i) {
    DestType expected = static_cast<DestType>(src_cpu(i));
    EXPECT_EQ(dst_cpu(i), expected);
  }

  // Cast back from DestType to Int4.
  // First clear out the device src buffer, since that will be the destination.
  src_cpu.setZero();
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(Int4));
  src_device.device(device) = dst_device.template cast<Int4>();
  device.memcpyDeviceToHost(src_cpu.data(), src_device_buffer,
                            kNumElems * sizeof(Int4));
  device.synchronize();

  for (int i = 0; i < kNumElems; ++i) {
    Int4 expected = static_cast<Int4>(dst_cpu(i));
    EXPECT_EQ(src_cpu(i), expected);
  }

  // Clean up.
  device.deallocate(src_device_buffer);
  device.deallocate(dst_device_buffer);
  device.synchronize();
}

}  // namespace
}  // namespace ml_dtypes