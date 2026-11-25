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

#include <array>
#include <cfenv>  // NOLINT(build/c++11)
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ML_DTYPES_HAS_X86_RUNTIME_DISPATCH 1
#else
#define ML_DTYPES_HAS_X86_RUNTIME_DISPATCH 0
#endif

namespace ml_dtypes {
namespace {

template <typename T>
inline constexpr bool is_custom_dtype_v =
    is_custom_float_v<T> || is_intn_v<T> || is_custom_complex_v<T>;

template <typename T>
inline constexpr bool is_integer_target_v =
    is_intn_v<T> || (std::is_integral_v<T> && !std::is_same_v<T, bool>);

template <typename T>
inline constexpr bool is_byte_custom_float_v =
    is_custom_float_v<T> && sizeof(T) == 1;

template <typename T>
inline constexpr bool is_simd_fp8_target_v =
    std::is_same_v<T, float8_e4m3fn> || std::is_same_v<T, float8_e5m2>;

// --- Vectorized and lookup-table contiguous cast kernels ---

// Builds a 256-entry L1-resident lookup table mapping every raw byte of a
// 1-byte custom float (including sub-byte types with arbitrary upper padding
// bits) to its exact converted `To` (`float` or `double`) value.
template <typename From, typename To>
const std::array<To, 256>& ByteFloatLut() {
  static const std::array<To, 256> kTable = [] {
    std::array<To, 256> table;
    for (int i = 0; i < 256; ++i) {
      const From v = Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i));
      table[i] = static_cast<To>(static_cast<float>(v));
    }
    return table;
  }();
  return kTable;
}

template <typename From, typename To>
void CastByteFloatViaLutContiguous(const uint8_t* src, To* dst, npy_intp n) {
  const auto& table = ByteFloatLut<From, To>();
  npy_intp i = 0;
  for (; i + 4 <= n; i += 4) {
    dst[i + 0] = table[src[i + 0]];
    dst[i + 1] = table[src[i + 1]];
    dst[i + 2] = table[src[i + 2]];
    dst[i + 3] = table[src[i + 3]];
  }
  for (; i < n; ++i) {
    dst[i] = table[src[i]];
  }
}

inline void Bf16ToF32Scalar(const uint16_t* src, uint32_t* dst, npy_intp n) {
  for (npy_intp i = 0; i < n; ++i) {
    dst[i] = static_cast<uint32_t>(src[i]) << 16;
  }
}

// Branchless integer bit conversion matching Eigen's `float_to_bfloat16_rtne`:
// rounds ties to even, preserves subnormals, and canonicalizes positive and
// negative NaNs to `0x7fc0` and `0xffc0` without floating-point comparisons.
inline uint16_t F32BitsToBf16Bits(uint32_t u) {
  const uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u > 0x7f800000u) {
    return static_cast<uint16_t>(((u & 0x80000000u) >> 16) | 0x7fc0u);
  }
  const uint32_t lsb = (u >> 16) & 1u;
  return static_cast<uint16_t>((u + (0x7fffu + lsb)) >> 16);
}

inline void F32ToBf16Scalar(const uint32_t* src, uint16_t* dst, npy_intp n) {
  for (npy_intp i = 0; i < n; ++i) {
    dst[i] = F32BitsToBf16Bits(src[i]);
  }
}

template <typename To>
struct Float8SimdParams;

template <>
struct Float8SimdParams<float8_e4m3fn> {
  static constexpr uint32_t kBiasDiff = 120u << 23;  // 0x3c000000
  static constexpr int kMantShift = 20;
  static constexpr uint32_t kHalfMinus1 = (1u << 19) - 1;  // 0x7ffff
  static constexpr uint32_t kMinNormal = 0x3c800000u;      // 2^-6
  static constexpr uint32_t kUnderflow = 0x3a800000u;      // 2^-10
  static constexpr uint32_t kOverflowClamp = 0x43f00000u;  // >=464+ -> 0x7f
  static constexpr int kSubnormShiftBase = 141;            // 20 + (121 - exp)
  static constexpr bool kDistinctNan = false;
  static constexpr uint32_t kNanBits = 0x7fu;
};

template <>
struct Float8SimdParams<float8_e5m2> {
  static constexpr uint32_t kBiasDiff = 112u << 23;  // 0x38000000
  static constexpr int kMantShift = 21;
  static constexpr uint32_t kHalfMinus1 = (1u << 20) - 1;  // 0xfffff
  static constexpr uint32_t kMinNormal = 0x38800000u;      // 2^-14
  static constexpr uint32_t kUnderflow = 0x37000000u;      // 2^-17
  static constexpr uint32_t kOverflowClamp = 0x47800000u;  // >=61440 -> 0x7c
  static constexpr int kSubnormShiftBase = 134;            // 21 + (113 - exp)
  static constexpr bool kDistinctNan = true;
  static constexpr uint32_t kNanBits = 0x7eu;
};

template <typename To>
inline uint8_t F32BitsToFloat8Bits(uint32_t u) {
  using P = Float8SimdParams<To>;
  const uint32_t sign = (u >> 24) & 0x80u;
  const uint32_t abs_u = u & 0x7fffffffu;
  if constexpr (P::kDistinctNan) {
    if (abs_u > 0x7f800000u) {
      return static_cast<uint8_t>(sign | P::kNanBits);
    }
  }
  if (abs_u >= P::kMinNormal) {
    const uint32_t clamped =
        abs_u < P::kOverflowClamp ? abs_u : P::kOverflowClamp;
    const uint32_t sub = clamped - P::kBiasDiff;
    const uint32_t lsb = (sub >> P::kMantShift) & 1u;
    const uint32_t res = (sub + P::kHalfMinus1 + lsb) >> P::kMantShift;
    return static_cast<uint8_t>(sign | res);
  }
  if (abs_u <= P::kUnderflow) {
    return static_cast<uint8_t>(sign);
  }
  const int exp = static_cast<int>(abs_u >> 23);
  const int shift = P::kSubnormShiftBase - exp;
  const uint32_t mant = (abs_u & 0x007fffffu) | 0x00800000u;
  const uint32_t lsb = (mant >> shift) & 1u;
  const uint32_t half_minus_1 = (uint32_t{1} << (shift - 1)) - 1u;
  const uint32_t res = (mant + half_minus_1 + lsb) >> shift;
  return static_cast<uint8_t>(sign | res);
}

template <typename To>
inline void F32ToFloat8Scalar(const uint32_t* src, uint8_t* dst, npy_intp n) {
  for (npy_intp i = 0; i < n; ++i) {
    dst[i] = F32BitsToFloat8Bits<To>(src[i]);
  }
}

#if defined(__x86_64__) || defined(_M_X64)

inline void Bf16ToF32Sse2(const uint16_t* src, uint32_t* dst, npy_intp n) {
  const __m128i zero = _mm_setzero_si128();
  npy_intp i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i v =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m128i lo = _mm_unpacklo_epi16(zero, v);
    const __m128i hi = _mm_unpackhi_epi16(zero, v);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 4), hi);
  }
  Bf16ToF32Scalar(src + i, dst + i, n - i);
}

inline __m128i F32ToBf16Sse2Step(__m128i u) {
  const __m128i abs_u = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
  const __m128i is_nan = _mm_cmpgt_epi32(abs_u, _mm_set1_epi32(0x7f800000));
  const __m128i lsb = _mm_and_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(1));
  const __m128i bias = _mm_add_epi32(lsb, _mm_set1_epi32(0x7fff));
  const __m128i rounded = _mm_srli_epi32(_mm_add_epi32(u, bias), 16);
  const __m128i sign16 = _mm_srli_epi32(
      _mm_and_si128(u, _mm_set1_epi32(static_cast<int>(0x80000000u))), 16);
  const __m128i nan_val = _mm_or_si128(sign16, _mm_set1_epi32(0x7fc0));
  return _mm_or_si128(_mm_and_si128(is_nan, nan_val),
                      _mm_andnot_si128(is_nan, rounded));
}

inline void F32ToBf16Sse2(const uint32_t* src, uint16_t* dst, npy_intp n) {
  const __m128i bias32 = _mm_set1_epi32(0x8000);
  const __m128i bias16 = _mm_set1_epi16(static_cast<short>(0x8000));
  npy_intp i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i u0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m128i u1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 4));
    const __m128i r0 = _mm_sub_epi32(F32ToBf16Sse2Step(u0), bias32);
    const __m128i r1 = _mm_sub_epi32(F32ToBf16Sse2Step(u1), bias32);
    const __m128i packed = _mm_xor_si128(_mm_packs_epi32(r0, r1), bias16);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
  }
  F32ToBf16Scalar(src + i, dst + i, n - i);
}

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX2__)

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx2"), noinline))
#endif
static void Bf16ToF32Avx2(const uint16_t* src, uint32_t* dst, npy_intp n) {
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m128i v0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m128i v1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 8));
    const __m256i r0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v0), 16);
    const __m256i r1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v1), 16);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), r0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 8), r1);
  }
  for (; i + 8 <= n; i += 8) {
    const __m128i v =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m256i r = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), r);
  }
  Bf16ToF32Scalar(src + i, dst + i, n - i);
}

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx2"), always_inline))
#endif
static inline __m256i F32ToBf16Avx2Step(__m256i u) {
  const __m256i abs_u = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
  const __m256i is_nan =
      _mm256_cmpgt_epi32(abs_u, _mm256_set1_epi32(0x7f800000));
  const __m256i lsb =
      _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
  const __m256i bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff));
  const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(u, bias), 16);
  const __m256i sign16 = _mm256_srli_epi32(
      _mm256_and_si256(u, _mm256_set1_epi32(static_cast<int>(0x80000000u))),
      16);
  const __m256i nan_val = _mm256_or_si256(sign16, _mm256_set1_epi32(0x7fc0));
  return _mm256_blendv_epi8(rounded, nan_val, is_nan);
}

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx2"), noinline))
#endif
static void F32ToBf16Avx2(const uint32_t* src, uint16_t* dst, npy_intp n) {
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256i u0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i u1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 8));
    const __m256i r0 = F32ToBf16Avx2Step(u0);
    const __m256i r1 = F32ToBf16Avx2Step(u1);
    const __m256i packed =
        _mm256_permute4x64_epi64(_mm256_packus_epi32(r0, r1), 0xd8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), packed);
  }
  for (; i + 8 <= n; i += 8) {
    const __m256i u =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i r = F32ToBf16Avx2Step(u);
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(r),
                                            _mm256_extracti128_si256(r, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
  }
  F32ToBf16Scalar(src + i, dst + i, n - i);
}

template <typename To>
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx2"), always_inline))
#endif
static inline __m256i F32ToFloat8Avx2Step(__m256i u) {
  using P = Float8SimdParams<To>;
  const __m256i abs_u = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
  const __m256i sign =
      _mm256_and_si256(_mm256_srli_epi32(u, 24), _mm256_set1_epi32(0x80));

  // Normal / overflow path.
  const __m256i n_clamped = _mm256_min_epi32(
      abs_u, _mm256_set1_epi32(static_cast<int>(P::kOverflowClamp)));
  const __m256i n_sub = _mm256_sub_epi32(
      n_clamped, _mm256_set1_epi32(static_cast<int>(P::kBiasDiff)));
  const __m256i n_lsb = _mm256_and_si256(
      _mm256_srli_epi32(n_sub, P::kMantShift), _mm256_set1_epi32(1));
  const __m256i n_bias = _mm256_add_epi32(
      n_lsb, _mm256_set1_epi32(static_cast<int>(P::kHalfMinus1)));
  const __m256i n_res =
      _mm256_srli_epi32(_mm256_add_epi32(n_sub, n_bias), P::kMantShift);

  // Subnormal / underflow path.
  const __m256i exp = _mm256_srli_epi32(abs_u, 23);
  const __m256i shift =
      _mm256_sub_epi32(_mm256_set1_epi32(P::kSubnormShiftBase), exp);
  const __m256i mant =
      _mm256_or_si256(_mm256_and_si256(abs_u, _mm256_set1_epi32(0x007fffff)),
                      _mm256_set1_epi32(0x00800000));
  const __m256i s_lsb =
      _mm256_and_si256(_mm256_srlv_epi32(mant, shift), _mm256_set1_epi32(1));
  const __m256i shift_m1 = _mm256_sub_epi32(shift, _mm256_set1_epi32(1));
  const __m256i half_m1 = _mm256_sub_epi32(
      _mm256_sllv_epi32(_mm256_set1_epi32(1), shift_m1), _mm256_set1_epi32(1));
  const __m256i s_rounded = _mm256_srlv_epi32(
      _mm256_add_epi32(_mm256_add_epi32(mant, half_m1), s_lsb), shift);
  const __m256i above_underflow = _mm256_cmpgt_epi32(
      abs_u, _mm256_set1_epi32(static_cast<int>(P::kUnderflow)));
  const __m256i s_res = _mm256_and_si256(s_rounded, above_underflow);

  // Select normal vs subnormal: abs_u >= kMinNormal iff abs_u > kMinNormal - 1.
  const __m256i is_normal = _mm256_cmpgt_epi32(
      abs_u, _mm256_set1_epi32(static_cast<int>(P::kMinNormal - 1)));
  __m256i res = _mm256_blendv_epi8(s_res, n_res, is_normal);

  if constexpr (P::kDistinctNan) {
    const __m256i is_nan =
        _mm256_cmpgt_epi32(abs_u, _mm256_set1_epi32(0x7f800000));
    res = _mm256_blendv_epi8(
        res, _mm256_set1_epi32(static_cast<int>(P::kNanBits)), is_nan);
  }
  return _mm256_or_si256(sign, res);
}

template <typename To>
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx2"), noinline))
#endif
static void F32ToFloat8Avx2(const uint32_t* src, uint8_t* dst, npy_intp n) {
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256i u0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m256i u1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 8));
    const __m256i r0 = F32ToFloat8Avx2Step<To>(u0);
    const __m256i r1 = F32ToFloat8Avx2Step<To>(u1);
    const __m256i p16 =
        _mm256_permute4x64_epi64(_mm256_packus_epi32(r0, r1), 0xd8);
    const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16),
                                        _mm256_extracti128_si256(p16, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), p8);
  }
  F32ToFloat8Scalar<To>(src + i, dst + i, n - i);
}

#endif  // ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX2__)

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX512F__)

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx512f"), noinline))
#endif
static void Bf16ToF32Avx512(const uint16_t* src, uint32_t* dst, npy_intp n) {
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256i v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    const __m512i r = _mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), r);
  }
  Bf16ToF32Avx2(src + i, dst + i, n - i);
}

#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx512f"), noinline))
#endif
static void F32ToBf16Avx512(const uint32_t* src, uint16_t* dst, npy_intp n) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
  const __m512i sign_mask = _mm512_set1_epi32(static_cast<int>(0x80000000u));
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i bias_base = _mm512_set1_epi32(0x7fff);
  const __m512i qnan_base = _mm512_set1_epi32(0x7fc0);
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512i u =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
    const __m512i abs_u = _mm512_and_si512(u, abs_mask);
    const __mmask16 is_nan = _mm512_cmpgt_epu32_mask(abs_u, exp_mask);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), one);
    const __m512i rounded = _mm512_srli_epi32(
        _mm512_add_epi32(u, _mm512_add_epi32(lsb, bias_base)), 16);
    const __m512i nan_val = _mm512_or_si512(
        _mm512_srli_epi32(_mm512_and_si512(u, sign_mask), 16), qnan_base);
    const __m512i res = _mm512_mask_blend_epi32(is_nan, rounded, nan_val);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                        _mm512_cvtepi32_epi16(res));
  }
  F32ToBf16Avx2(src + i, dst + i, n - i);
}

template <typename To>
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
__attribute__((target("avx512f"), noinline))
#endif
static void F32ToFloat8Avx512(const uint32_t* src, uint8_t* dst, npy_intp n) {
  using P = Float8SimdParams<To>;
  const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i overflow_clamp =
      _mm512_set1_epi32(static_cast<int>(P::kOverflowClamp));
  const __m512i bias_diff = _mm512_set1_epi32(static_cast<int>(P::kBiasDiff));
  const __m512i half_m1_norm =
      _mm512_set1_epi32(static_cast<int>(P::kHalfMinus1));
  const __m512i min_normal = _mm512_set1_epi32(static_cast<int>(P::kMinNormal));
  const __m512i underflow = _mm512_set1_epi32(static_cast<int>(P::kUnderflow));
  const __m512i subnorm_base = _mm512_set1_epi32(P::kSubnormShiftBase);
  const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
  const __m512i implicit_bit = _mm512_set1_epi32(0x00800000);
  const __m512i sign_mask8 = _mm512_set1_epi32(0x80);
  npy_intp i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512i u =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
    const __m512i abs_u = _mm512_and_si512(u, abs_mask);
    const __m512i sign = _mm512_and_si512(_mm512_srli_epi32(u, 24), sign_mask8);

    const __m512i n_clamped = _mm512_min_epu32(abs_u, overflow_clamp);
    const __m512i n_sub = _mm512_sub_epi32(n_clamped, bias_diff);
    const __m512i n_lsb =
        _mm512_and_si512(_mm512_srli_epi32(n_sub, P::kMantShift), one);
    const __m512i n_res = _mm512_srli_epi32(
        _mm512_add_epi32(n_sub, _mm512_add_epi32(n_lsb, half_m1_norm)),
        P::kMantShift);

    const __m512i exp = _mm512_srli_epi32(abs_u, 23);
    const __m512i shift = _mm512_sub_epi32(subnorm_base, exp);
    const __m512i mant =
        _mm512_or_si512(_mm512_and_si512(abs_u, mant_mask), implicit_bit);
    const __m512i s_lsb = _mm512_and_si512(_mm512_srlv_epi32(mant, shift), one);
    const __m512i half_m1 = _mm512_sub_epi32(
        _mm512_sllv_epi32(one, _mm512_sub_epi32(shift, one)), one);
    const __m512i s_rounded = _mm512_srlv_epi32(
        _mm512_add_epi32(_mm512_add_epi32(mant, half_m1), s_lsb), shift);
    const __mmask16 above_underflow = _mm512_cmpgt_epu32_mask(abs_u, underflow);
    const __m512i s_res = _mm512_maskz_mov_epi32(above_underflow, s_rounded);

    const __mmask16 is_normal = _mm512_cmpge_epu32_mask(abs_u, min_normal);
    __m512i res = _mm512_mask_blend_epi32(is_normal, s_res, n_res);
    if constexpr (P::kDistinctNan) {
      const __mmask16 is_nan =
          _mm512_cmpgt_epu32_mask(abs_u, _mm512_set1_epi32(0x7f800000));
      res = _mm512_mask_blend_epi32(
          is_nan, res, _mm512_set1_epi32(static_cast<int>(P::kNanBits)));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                     _mm512_cvtepi32_epi8(_mm512_or_si512(sign, res)));
  }
  F32ToFloat8Avx2<To>(src + i, dst + i, n - i);
}

#endif  // ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX512F__)

inline void InitX86CpuDispatch() {
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH && defined(__GNUC__) && \
    !defined(__clang__)
  __builtin_cpu_init();
#endif
}

inline bool HasAvx512F() {
#if defined(__AVX512F__)
  return true;
#elif ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
  InitX86CpuDispatch();
  return __builtin_cpu_supports("avx512f");
#else
  return false;
#endif
}

inline bool HasAvx2() {
#if defined(__AVX2__)
  return true;
#elif ML_DTYPES_HAS_X86_RUNTIME_DISPATCH
  InitX86CpuDispatch();
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

inline void Bf16ToF32Contiguous(const uint16_t* src, uint32_t* dst,
                                npy_intp n) {
  using Fn = void (*)(const uint16_t*, uint32_t*, npy_intp);
  static const Fn kFn = []() -> Fn {
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX512F__)
    if (HasAvx512F()) {
      return &Bf16ToF32Avx512;
    }
#endif
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX2__)
    if (HasAvx2()) {
      return &Bf16ToF32Avx2;
    }
#endif
    return &Bf16ToF32Sse2;
  }();
  kFn(src, dst, n);
}

inline void F32ToBf16Contiguous(const uint32_t* src, uint16_t* dst,
                                npy_intp n) {
  using Fn = void (*)(const uint32_t*, uint16_t*, npy_intp);
  static const Fn kFn = []() -> Fn {
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX512F__)
    if (HasAvx512F()) {
      return &F32ToBf16Avx512;
    }
#endif
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX2__)
    if (HasAvx2()) {
      return &F32ToBf16Avx2;
    }
#endif
    return &F32ToBf16Sse2;
  }();
  kFn(src, dst, n);
}

template <typename To>
inline void F32ToFloat8Contiguous(const uint32_t* src, uint8_t* dst,
                                  npy_intp n) {
  using Fn = void (*)(const uint32_t*, uint8_t*, npy_intp);
  static const Fn kFn = []() -> Fn {
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX512F__)
    if (HasAvx512F()) {
      return &F32ToFloat8Avx512<To>;
    }
#endif
#if ML_DTYPES_HAS_X86_RUNTIME_DISPATCH || defined(__AVX2__)
    if (HasAvx2()) {
      return &F32ToFloat8Avx2<To>;
    }
#endif
    return &F32ToFloat8Scalar<To>;
  }();
  kFn(src, dst, n);
}

#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)

inline void Bf16ToF32Contiguous(const uint16_t* src, uint32_t* dst,
                                npy_intp n) {
  npy_intp i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t v = vld1q_u16(src + i);
    const uint32x4_t lo = vshll_n_u16(vget_low_u16(v), 16);
    const uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
    vst1q_u32(dst + i, lo);
    vst1q_u32(dst + i + 4, hi);
  }
  Bf16ToF32Scalar(src + i, dst + i, n - i);
}

inline uint16x4_t F32ToBf16NeonStep(uint32x4_t u) {
  const uint32x4_t abs_u = vandq_u32(u, vdupq_n_u32(0x7fffffffu));
  const uint32x4_t is_nan = vcgtq_u32(abs_u, vdupq_n_u32(0x7f800000u));
  const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u));
  const uint32x4_t bias = vaddq_u32(lsb, vdupq_n_u32(0x7fffu));
  const uint16x4_t rounded = vshrn_n_u32(vaddq_u32(u, bias), 16);
  const uint16x4_t sign16 =
      vshrn_n_u32(vandq_u32(u, vdupq_n_u32(0x80000000u)), 16);
  const uint16x4_t nan_val = vorr_u16(sign16, vdup_n_u16(0x7fc0u));
  return vbsl_u16(vmovn_u32(is_nan), nan_val, rounded);
}

inline void F32ToBf16Contiguous(const uint32_t* src, uint16_t* dst,
                                npy_intp n) {
  npy_intp i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x4_t lo = F32ToBf16NeonStep(vld1q_u32(src + i));
    const uint16x4_t hi = F32ToBf16NeonStep(vld1q_u32(src + i + 4));
    vst1q_u16(dst + i, vcombine_u16(lo, hi));
  }
  F32ToBf16Scalar(src + i, dst + i, n - i);
}

template <typename To>
inline uint32x4_t F32ToFloat8NeonStep(uint32x4_t u) {
  using P = Float8SimdParams<To>;
  const uint32x4_t abs_u = vandq_u32(u, vdupq_n_u32(0x7fffffffu));
  const uint32x4_t sign = vandq_u32(vshrq_n_u32(u, 24), vdupq_n_u32(0x80u));

  const uint32x4_t n_clamped = vminq_u32(abs_u, vdupq_n_u32(P::kOverflowClamp));
  const uint32x4_t n_sub = vsubq_u32(n_clamped, vdupq_n_u32(P::kBiasDiff));
  const uint32x4_t n_lsb =
      vandq_u32(vshrq_n_u32(n_sub, P::kMantShift), vdupq_n_u32(1u));
  const uint32x4_t n_res = vshrq_n_u32(
      vaddq_u32(n_sub, vaddq_u32(n_lsb, vdupq_n_u32(P::kHalfMinus1))),
      P::kMantShift);

  const uint32x4_t exp = vshrq_n_u32(abs_u, 23);
  const int32x4_t raw_shift =
      vsubq_s32(vdupq_n_s32(P::kSubnormShiftBase), vreinterpretq_s32_u32(exp));
  const int32x4_t shift =
      vmaxq_s32(vminq_s32(raw_shift, vdupq_n_s32(31)), vdupq_n_s32(1));
  const int32x4_t neg_shift = vnegq_s32(shift);
  const uint32x4_t mant = vorrq_u32(vandq_u32(abs_u, vdupq_n_u32(0x007fffffu)),
                                    vdupq_n_u32(0x00800000u));
  const uint32x4_t s_lsb =
      vandq_u32(vshlq_u32(mant, neg_shift), vdupq_n_u32(1u));
  const uint32x4_t half_m1 =
      vsubq_u32(vshlq_u32(vdupq_n_u32(1u), vsubq_s32(shift, vdupq_n_s32(1))),
                vdupq_n_u32(1u));
  const uint32x4_t s_rounded =
      vshlq_u32(vaddq_u32(vaddq_u32(mant, half_m1), s_lsb), neg_shift);
  const uint32x4_t above_underflow =
      vcgtq_u32(abs_u, vdupq_n_u32(P::kUnderflow));
  const uint32x4_t s_res = vandq_u32(s_rounded, above_underflow);

  const uint32x4_t is_normal = vcgeq_u32(abs_u, vdupq_n_u32(P::kMinNormal));
  uint32x4_t res = vbslq_u32(is_normal, n_res, s_res);
  if constexpr (P::kDistinctNan) {
    const uint32x4_t is_nan = vcgtq_u32(abs_u, vdupq_n_u32(0x7f800000u));
    res = vbslq_u32(is_nan, vdupq_n_u32(P::kNanBits), res);
  }
  return vorrq_u32(sign, res);
}

template <typename To>
inline void F32ToFloat8Contiguous(const uint32_t* src, uint8_t* dst,
                                  npy_intp n) {
  npy_intp i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint32x4_t r0 = F32ToFloat8NeonStep<To>(vld1q_u32(src + i));
    const uint32x4_t r1 = F32ToFloat8NeonStep<To>(vld1q_u32(src + i + 4));
    const uint16x8_t p16 = vcombine_u16(vmovn_u32(r0), vmovn_u32(r1));
    vst1_u8(dst + i, vmovn_u16(p16));
  }
  F32ToFloat8Scalar<To>(src + i, dst + i, n - i);
}

#else

inline void Bf16ToF32Contiguous(const uint16_t* src, uint32_t* dst,
                                npy_intp n) {
  Bf16ToF32Scalar(src, dst, n);
}

inline void F32ToBf16Contiguous(const uint32_t* src, uint16_t* dst,
                                npy_intp n) {
  F32ToBf16Scalar(src, dst, n);
}

template <typename To>
inline void F32ToFloat8Contiguous(const uint32_t* src, uint8_t* dst,
                                  npy_intp n) {
  F32ToFloat8Scalar<To>(src, dst, n);
}

#endif

// True for casts that have a dedicated contiguous SIMD or LUT fast path.
template <typename From, typename To>
inline constexpr bool kHasContiguousFastPath =
    (std::is_same_v<From, bfloat16> && std::is_same_v<To, float>) ||
    (std::is_same_v<From, float> && std::is_same_v<To, bfloat16>) ||
    (is_byte_custom_float_v<From> &&
     (std::is_same_v<To, float> || std::is_same_v<To, double>)) ||
    (std::is_same_v<From, float> && is_simd_fp8_target_v<To>);

// True for casts performed purely via integer bit manipulation or lookup table
// that never execute floating-point operations or raise FP exceptions.
template <typename From, typename To>
inline constexpr bool kIsBitwiseNoFpErrorCast =
    kHasContiguousFastPath<From, To>;

template <typename From, typename To>
int ContiguousCastLoop(PyArrayMethod_Context* /*context*/, char* const data[],
                       const npy_intp dimensions[], const npy_intp* /*strides*/,
                       NpyAuxData* /*auxdata*/) {
  const npy_intp n = dimensions[0];
  if constexpr (std::is_same_v<From, bfloat16> && std::is_same_v<To, float>) {
    Bf16ToF32Contiguous(reinterpret_cast<const uint16_t*>(data[0]),
                        reinterpret_cast<uint32_t*>(data[1]), n);
  } else if constexpr (std::is_same_v<From, float> &&
                       std::is_same_v<To, bfloat16>) {
    F32ToBf16Contiguous(reinterpret_cast<const uint32_t*>(data[0]),
                        reinterpret_cast<uint16_t*>(data[1]), n);
  } else if constexpr (is_byte_custom_float_v<From> &&
                       (std::is_same_v<To, float> ||
                        std::is_same_v<To, double>)) {
    CastByteFloatViaLutContiguous<From, To>(
        reinterpret_cast<const uint8_t*>(data[0]),
        reinterpret_cast<To*>(data[1]), n);
  } else if constexpr (std::is_same_v<From, float> &&
                       is_simd_fp8_target_v<To>) {
    F32ToFloat8Contiguous<To>(reinterpret_cast<const uint32_t*>(data[0]),
                              reinterpret_cast<uint8_t*>(data[1]), n);
  }
  return 0;
}

// Converts `value` to `half`, setting `overflow` when a finite `float` rounds
// to `+-inf` so `FE_OVERFLOW` is raised consistently on platforms where
// `Eigen::half(float)` uses software bit manipulation instead of hardware FP16.
half FloatToHalf(float value, bool& overflow) {
  const half result = static_cast<half>(value);
  if ((Eigen::numext::isinf)(result) && !(Eigen::numext::isinf)(value)) {
    overflow = true;
  }
  return result;
}

// Exclusive upper bound `2^Digits` as a compile-time floating-point power of 2
// (`Digits <= 64`, which is exact in `float`, `double`, and `long double`).
template <typename Fp, int Digits>
constexpr Fp Pow2() {
  static_assert(Digits >= 0 && Digits <= 64);
  if constexpr (Digits == 64) {
    return static_cast<Fp>(uint64_t{1} << 63) * Fp{2};
  } else {
    return static_cast<Fp>(uint64_t{1} << Digits);
  }
}

// Converts `value` to integer type `To` (`int1`..`uint4` or built-in
// non-boolean integers) without C++17 undefined or implementation-defined
// behavior:
// - Complex inputs drop their imaginary part.
// - Integer and boolean inputs reduce modulo `2^bits` in unsigned arithmetic
//   when narrowing to `intN` so signed conversions always receive in-range
//   values.
// - Floating-point inputs truncate toward zero when finite and within
//   `[min(To), max(To)]`; non-finite or out-of-range values set `invalid`
//   (raising `FE_INVALID` after the loop) and return 0.
template <typename To, typename From>
To CastToInt(From value, bool& invalid) {
  static_assert(is_integer_target_v<To>);
  if constexpr (is_complex_v<From>) {
    return CastToInt<To>(value.real(), invalid);
  } else if constexpr (is_intn_v<To> &&
                       (is_intn_v<From> || std::is_integral_v<From>)) {
    constexpr uint8_t kMask = (uint8_t{1} << To::bits) - 1;
    uint8_t raw;
    if constexpr (is_intn_v<From>) {
      raw = static_cast<uint8_t>(
          static_cast<typename From::underlying_type>(value));
    } else {
      raw = static_cast<uint8_t>(value);
    }
    // `raw & kMask` is in [0, 15], which is representable in both `int8_t` and
    // `uint8_t` without implementation-defined narrowing.
    return To(static_cast<typename To::underlying_type>(raw & kMask));
  } else if constexpr (is_intn_v<From>) {
    // `To` is a built-in integer (`int8`..`uint64`). `From` values lie in
    // [-8, 15], which fits in every signed target `To`; unsigned `To` reduces
    // modulo `2^bits` by standard C++17 unsigned conversion.
    return static_cast<To>(static_cast<typename From::underlying_type>(value));
  } else {
    // Widen `half` and custom narrow floats to `float` (while keeping `double`
    // and `long double` at native precision) so `Fp` has `max_exponent >= 128`,
    // guaranteeing `2^digits` (for `digits <= 64`) is exact and cannot overflow
    // `Fp`.
    using Fp = std::conditional_t<std::is_same_v<From, double> ||
                                      std::is_same_v<From, long double>,
                                  From, float>;
    static_assert(std::numeric_limits<Fp>::max_exponent >
                  std::numeric_limits<To>::digits);
    const Fp fp = static_cast<Fp>(value);
    if (!std::isfinite(fp)) {
      invalid = true;
      return To(0);
    }
    const Fp truncated = std::trunc(fp);
    constexpr Fp kMaxExcl = Pow2<Fp, std::numeric_limits<To>::digits>();
    if constexpr (std::numeric_limits<To>::is_signed) {
      constexpr Fp kMinIncl = -kMaxExcl;
      if (truncated < kMinIncl || truncated >= kMaxExcl) {
        invalid = true;
        return To(0);
      }
    } else {
      if (truncated < Fp{0} || truncated >= kMaxExcl) {
        invalid = true;
        return To(0);
      }
    }
    if constexpr (is_intn_v<To>) {
      return To(static_cast<typename To::underlying_type>(truncated));
    } else {
      return static_cast<To>(truncated);
    }
  }
}

// True when casting `From` to `To` discards the imaginary part of a complex
// number and should emit `np.exceptions.ComplexWarning`. Casting to `bool` is
// excluded because `bool(z)` checks whether either part is nonzero.
template <typename From, typename To>
inline constexpr bool kWarnComplexToReal =
    is_complex_v<From> && !is_complex_v<To> && !std::is_same_v<To, bool>;

// True when both `From` and `To` are integer or boolean types (`int1`..`uint4`,
// built-in integers, or `bool`), so the cast performs no floating-point
// operations and cannot raise floating-point exceptions.
template <typename From, typename To>
inline constexpr bool kIsIntegerOrBoolCast =
    (is_intn_v<From> || std::is_integral_v<From>) &&
    (is_intn_v<To> || std::is_integral_v<To>);

template <typename From, typename To>
constexpr NPY_ARRAYMETHOD_FLAGS CastMethodFlags() {
  if constexpr (kIsIntegerOrBoolCast<From, To> ||
                kIsBitwiseNoFpErrorCast<From, To>) {
    return NPY_METH_NO_FLOATINGPOINT_ERRORS;
  } else {
    return static_cast<NPY_ARRAYMETHOD_FLAGS>(0);
  }
}

// Elementwise cross-dtype cast loop (custom <-> built-in and custom <->
// custom). Runs without holding the Python GIL.
template <typename From, typename To>
int CastLoop(PyArrayMethod_Context* /*context*/, char* const data[],
             const npy_intp dimensions[], const npy_intp strides[],
             NpyAuxData* /*auxdata*/) {
  const char* src = data[0];
  char* dst = data[1];
  const npy_intp n = dimensions[0];
  const npy_intp src_stride = strides[0];
  const npy_intp dst_stride = strides[1];
  bool invalid = false;
  bool overflow = false;

  if constexpr (is_byte_custom_float_v<From> &&
                (std::is_same_v<To, float> || std::is_same_v<To, double>)) {
    const auto& table = ByteFloatLut<From, To>();
    for (npy_intp i = 0; i < n; ++i) {
      const uint8_t raw =
          *reinterpret_cast<const uint8_t*>(src + i * src_stride);
      *reinterpret_cast<To*>(dst + i * dst_stride) = table[raw];
    }
    return 0;
  }
  if constexpr (std::is_same_v<From, float> && std::is_same_v<To, bfloat16>) {
    for (npy_intp i = 0; i < n; ++i) {
      const uint32_t u =
          *reinterpret_cast<const uint32_t*>(src + i * src_stride);
      *reinterpret_cast<uint16_t*>(dst + i * dst_stride) = F32BitsToBf16Bits(u);
    }
    return 0;
  }
  if constexpr (std::is_same_v<From, float> && is_simd_fp8_target_v<To>) {
    for (npy_intp i = 0; i < n; ++i) {
      const uint32_t u =
          *reinterpret_cast<const uint32_t*>(src + i * src_stride);
      *reinterpret_cast<uint8_t*>(dst + i * dst_stride) =
          F32BitsToFloat8Bits<To>(u);
    }
    return 0;
  }

  for (npy_intp i = 0; i < n; ++i) {
    const From& in = *reinterpret_cast<const From*>(src + i * src_stride);
    To& out = *reinterpret_cast<To*>(dst + i * dst_stride);
    if constexpr (is_integer_target_v<To>) {
      out = CastToInt<To>(in, invalid);
    } else if constexpr (std::is_same_v<To, bool>) {
      if constexpr (is_complex_v<From>) {
        out = static_cast<bool>(in.real() != typename From::value_type(0) ||
                                in.imag() != typename From::value_type(0));
      } else if constexpr (is_intn_v<From>) {
        out = static_cast<bool>(in);
      } else {
        out = static_cast<bool>(static_cast<float>(in));
      }
    } else if constexpr (std::is_same_v<To, half>) {
      out = FloatToHalf(NarrowToFloat(in), overflow);
    } else if constexpr (std::is_same_v<To, complex32>) {
      if constexpr (is_complex_v<From>) {
        out = complex32(FloatToHalf(NarrowToFloat(in.real()), overflow),
                        FloatToHalf(NarrowToFloat(in.imag()), overflow));
      } else {
        out = complex32(FloatToHalf(NarrowToFloat(in), overflow), half(0));
      }
    } else if constexpr (std::is_same_v<To, bcomplex32>) {
      if constexpr (is_complex_v<From>) {
        out = bcomplex32(bfloat16(NarrowToFloat(in.real())),
                         bfloat16(NarrowToFloat(in.imag())));
      } else {
        out = bcomplex32(bfloat16(NarrowToFloat(in)), bfloat16(0));
      }
    } else if constexpr (is_complex_v<From> && is_complex_v<To>) {
      out = static_cast<To>(static_cast<std::complex<float>>(in));
    } else {
      out = static_cast<To>(NarrowToFloat(in));
    }
  }
  int exceptions = 0;
#ifdef FE_INVALID
  if (invalid) {
    exceptions |= FE_INVALID;
  }
#endif
#ifdef FE_OVERFLOW
  if (overflow) {
    exceptions |= FE_OVERFLOW;
  }
#endif
  if (exceptions != 0) {
    std::feraiseexcept(exceptions);
  }
  return 0;
}

// Emits `ComplexWarning` once during cast setup (while the GIL is held) so the
// inner `CastLoop` runs without `NPY_METH_REQUIRES_PYAPI` and does not repeat
// the warning across chunked or multi-dimensional iterations.
template <typename From, typename To>
int ComplexToRealGetLoop(PyArrayMethod_Context* /*context*/, int /*aligned*/,
                         int /*move_references*/, const npy_intp* /*strides*/,
                         PyArrayMethod_StridedLoop** out_loop,
                         NpyAuxData** out_transferdata,
                         NPY_ARRAYMETHOD_FLAGS* flags) {
  if (GiveComplexWarning() < 0) {
    return -1;
  }
  *out_loop = &CastLoop<From, To>;
  *out_transferdata = nullptr;
  *flags = CastMethodFlags<From, To>();
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
    // Built-in -> custom: `bool` is marked safe for all custom types (matching
    // `CommonDType`, even for `int1` and `float8_e8m0fnu`), and `float16` fits
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

// --- Cast spec registration helpers ---

template <typename From, typename To>
void AddCast(CastSpecList& list, const char* name, PyArray_DTypeMeta* from,
             PyArray_DTypeMeta* to) {
  if constexpr (kWarnComplexToReal<From, To>) {
    list.Add(name, from, to, CastingLevel<From, To>(),
             CastMethodFlags<From, To>(), &ComplexToRealGetLoop<From, To>);
  } else if constexpr (kHasContiguousFastPath<From, To>) {
    list.Add(name, from, to, CastingLevel<From, To>(),
             CastMethodFlags<From, To>(), &CastLoop<From, To>,
             &ContiguousCastLoop<From, To>);
  } else {
    list.Add(name, from, to, CastingLevel<From, To>(),
             CastMethodFlags<From, To>(), &CastLoop<From, To>);
  }
}

// Appends cast specs in both directions between `Self` (represented by
// `nullptr` for `PyArrayInitDTypeMeta_FromSpec`) and `BuiltinT`.
template <typename Self, typename BuiltinT>
void AddBuiltinCastsForOne(CastSpecList& list) {
  PyArray_DTypeMeta* builtin_dm = DtypeTraits<BuiltinT>::DTypeMeta();
  AddCast<Self, BuiltinT>(list, "ml_dtypes_to_numpy_cast", /*from=*/nullptr,
                          /*to=*/builtin_dm);
  AddCast<BuiltinT, Self>(list, "numpy_to_ml_dtypes_cast", /*from=*/builtin_dm,
                          /*to=*/nullptr);
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
    if (other_dm == nullptr || other_dm->singleton == nullptr) {
      return;
    }
    AddCast<Self, Other>(list, "ml_dtypes_custom_cast", /*from=*/nullptr,
                         /*to=*/other_dm);
    AddCast<Other, Self>(list, "ml_dtypes_custom_cast", /*from=*/other_dm,
                         /*to=*/nullptr);
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
                       PyArrayMethod_StridedLoop* strided_loop,
                       PyArrayMethod_StridedLoop* contiguous_loop) {
  PyType_Slot extra_slot = {0, nullptr};
  if (contiguous_loop != nullptr) {
    extra_slot = {NPY_METH_contiguous_loop,
                  reinterpret_cast<void*>(contiguous_loop)};
  }
  AddSlot(name, from, to, casting, flags,
          {NPY_METH_strided_loop, reinterpret_cast<void*>(strided_loop)},
          extra_slot);
}

void CastSpecList::Add(const char* name, PyArray_DTypeMeta* from,
                       PyArray_DTypeMeta* to, NPY_CASTING casting,
                       NPY_ARRAYMETHOD_FLAGS flags,
                       PyArrayMethod_GetLoop* get_loop) {
  AddSlot(name, from, to, casting, flags,
          {NPY_METH_get_loop, reinterpret_cast<void*>(get_loop)});
}

void CastSpecList::AddSlot(const char* name, PyArray_DTypeMeta* from,
                           PyArray_DTypeMeta* to, NPY_CASTING casting,
                           NPY_ARRAYMETHOD_FLAGS flags, PyType_Slot loop_slot,
                           PyType_Slot extra_slot) {
  auto entry = std::make_unique<CastSpecEntry>();
  entry->dtypes[0] = from;
  entry->dtypes[1] = to;
  entry->slots[0] = loop_slot;
  entry->slots[1] = extra_slot;
  entry->slots[2] = {0, nullptr};
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
