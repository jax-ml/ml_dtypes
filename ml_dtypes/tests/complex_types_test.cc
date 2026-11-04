/* Copyright 2026 The ml_dtypes Authors.

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

#include "ml_dtypes/include/complex_types.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <type_traits>

#include <gtest/gtest.h>

namespace ml_dtypes {
namespace {

// Representation properties of the current implementation.
static_assert(sizeof(complex32) == 4, "complex32 must be four bytes");
static_assert(sizeof(bcomplex32) == 4, "bcomplex32 must be four bytes");
static_assert(std::is_standard_layout<complex32>::value,
              "complex32 must be standard layout");
static_assert(std::is_standard_layout<bcomplex32>::value,
              "bcomplex32 must be standard layout");
static_assert(std::is_trivially_copyable<complex32>::value,
              "complex32 must be trivially copyable");
static_assert(std::is_trivially_copyable<bcomplex32>::value,
              "bcomplex32 must be trivially copyable");
static_assert(std::is_trivially_copyable<half>::value &&
                  std::is_trivially_copyable<bfloat16>::value,
              "component types must be trivially copyable");

template <typename ComplexT>
class ComplexTypesTest : public ::testing::Test {};

using ComplexTypes = ::testing::Types<complex32, bcomplex32>;
TYPED_TEST_SUITE(ComplexTypesTest, ComplexTypes);

TYPED_TEST(ComplexTypesTest, SizeAndAlignment) {
  EXPECT_EQ(sizeof(TypeParam), 4);
  EXPECT_EQ(alignof(TypeParam), 2);  // Two 16-bit components.
}

TYPED_TEST(ComplexTypesTest, ConstructionAndComponentAccess) {
  const TypeParam z(1.5f, -2.5f);
  EXPECT_EQ(static_cast<float>(z.real()), 1.5f);
  EXPECT_EQ(static_cast<float>(z.imag()), -2.5f);

  const TypeParam real_only(3.0f);
  EXPECT_EQ(static_cast<float>(real_only.real()), 3.0f);
  EXPECT_EQ(static_cast<float>(real_only.imag()), 0.0f);

  const TypeParam from_std(std::complex<float>(4.0f, -1.0f));
  EXPECT_EQ(static_cast<float>(from_std.real()), 4.0f);
  EXPECT_EQ(static_cast<float>(from_std.imag()), -1.0f);
}

TYPED_TEST(ComplexTypesTest, RealComponentStoredFirst) {
  const TypeParam z(1.5f, -2.5f);
  unsigned short raw[2];
  std::memcpy(raw, &z, sizeof(z));
  using Component = typename TypeParam::value_type;
  EXPECT_EQ(static_cast<float>(
                Eigen::numext::bit_cast<Component, unsigned short>(raw[0])),
            1.5f);
  EXPECT_EQ(static_cast<float>(
                Eigen::numext::bit_cast<Component, unsigned short>(raw[1])),
            -2.5f);
}

TYPED_TEST(ComplexTypesTest, ArithmeticPromotesThroughFloat32) {
  const TypeParam a(1.0f, 2.0f);
  const TypeParam b(0.5f, -0.25f);
  const std::complex<float> af = a;
  const std::complex<float> bf = b;

  const TypeParam sum = a + b;
  EXPECT_FLOAT_EQ(static_cast<float>(sum.real()), af.real() + bf.real());
  EXPECT_FLOAT_EQ(static_cast<float>(sum.imag()), af.imag() + bf.imag());

  const TypeParam diff = a - b;
  EXPECT_FLOAT_EQ(static_cast<float>(diff.real()), af.real() - bf.real());
  EXPECT_FLOAT_EQ(static_cast<float>(diff.imag()), af.imag() - bf.imag());

  const TypeParam product = a * b;
  EXPECT_FLOAT_EQ(static_cast<float>(product.real()),
                  af.real() * bf.real() - af.imag() * bf.imag());
  EXPECT_FLOAT_EQ(static_cast<float>(product.imag()),
                  af.real() * bf.imag() + af.imag() * bf.real());

  const TypeParam quotient = a / b;
  const std::complex<float> expected = af / bf;
  EXPECT_FLOAT_EQ(static_cast<float>(quotient.real()), expected.real());
  EXPECT_FLOAT_EQ(static_cast<float>(quotient.imag()), expected.imag());

  const TypeParam negated = -a;
  EXPECT_FLOAT_EQ(static_cast<float>(negated.real()), -1.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(negated.imag()), -2.0f);
}

TYPED_TEST(ComplexTypesTest, BoolConversion) {
  EXPECT_FALSE(static_cast<bool>(TypeParam(0.0f, 0.0f)));
  EXPECT_TRUE(static_cast<bool>(TypeParam(1.0f, 0.0f)));
  EXPECT_TRUE(static_cast<bool>(TypeParam(0.0f, 1.0f)));
}

TYPED_TEST(ComplexTypesTest, ConversionToComplexFloat) {
  const TypeParam z(1.5f, -2.5f);
  const std::complex<float> converted = z;
  EXPECT_EQ(converted.real(), 1.5f);
  EXPECT_EQ(converted.imag(), -2.5f);
}

TYPED_TEST(ComplexTypesTest, LexicographicComparison) {
  const TypeParam a(0.0f, 1.0f);
  const TypeParam b(0.0f, 2.0f);
  const TypeParam c(1.0f, 0.0f);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b <= b);
  EXPECT_TRUE(b < c);
  EXPECT_TRUE(c > b);
  EXPECT_TRUE(c >= c);

  // NaN imaginary parts suppress the ordered comparisons.
  const TypeParam imag_nan(std::complex<float>(0.0f, std::nanf("")));
  const TypeParam d(1.0f, 0.0f);
  EXPECT_FALSE(imag_nan < d);
  EXPECT_FALSE(d > imag_nan);
}

}  // namespace
}  // namespace ml_dtypes
