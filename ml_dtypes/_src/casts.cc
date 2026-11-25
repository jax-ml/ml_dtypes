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
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

#if defined(__GNUC__) || defined(__clang__)
#define ML_DTYPES_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ML_DTYPES_ALWAYS_INLINE inline
#endif

// Clang's `#pragma clang loop` turns failed optimization hints into
// `-Wpass-failed` warnings (errors under `-Werror`). Skip the hints in
// non-optimized, size-optimized (`-Os`), and sanitizer builds.
#if defined(__clang__) && defined(__OPTIMIZE__) &&                \
    !defined(__OPTIMIZE_SIZE__) && !defined(ADDRESS_SANITIZER) && \
    !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) &&   \
    !defined(UNDEFINED_BEHAVIOR_SANITIZER) && !defined(HWADDRESS_SANITIZER)
#define ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS 1
#else
#define ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS 0
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

template <typename From, typename To>
inline constexpr bool kHasByteLutFastPath =
    !std::is_same_v<From, To> &&
    (is_byte_custom_float_v<From> || is_intn_v<From>) &&
    (std::is_same_v<To, float> || std::is_same_v<To, double> ||
     std::is_same_v<To, bfloat16> || is_byte_custom_float_v<To>);

template <typename From, typename To>
inline constexpr bool kHasIntBitwiseFastPath =
    !std::is_same_v<From, To> && (is_intn_v<From> || is_intn_v<To>) &&
    is_integer_target_v<From> && is_integer_target_v<To>;

// --- Contiguous fast-path cast kernels ---

// Lookup table mapping each raw byte of a 1-byte custom dtype to `To`.
template <typename From, typename To>
const std::array<To, 256>& ByteTypeLut() {
  static const std::array<To, 256> kTable = [] {
    std::array<To, 256> table;
    for (int i = 0; i < 256; ++i) {
      const From v = Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i));
      table[i] = static_cast<To>(NarrowToFloat(v));
    }
    return table;
  }();
  return kTable;
}

#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
// Casts `n` contiguous elements of 1-byte custom dtype `From` to `To` via
// `ByteTypeLut`. Unrolled so independent table loads can overlap.
template <typename From, typename To>
void CastByteTypeViaLutContiguous(const uint8_t* src, To* dst, npy_intp n) {
  const auto& table = ByteTypeLut<From, To>();
#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang loop unroll_count(4)
#elif defined(__GNUC__) && !defined(__clang__) && defined(__OPTIMIZE__) && \
    !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 4
#endif
  for (npy_intp i = 0; i < n; ++i) {
    dst[i] = table[src[i]];
  }
}
#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang diagnostic pop
#endif

// Widens raw bfloat16 bits to float32 bits by zero-extending the low 16 bits.
ML_DTYPES_ALWAYS_INLINE uint32_t Bf16BitsToF32Bits(uint16_t u) {
  return static_cast<uint32_t>(u) << 16;
}

// Matches Eigen's `float_to_bfloat16_rtne` (round-to-nearest-even, signed
// qNaN).
ML_DTYPES_ALWAYS_INLINE uint16_t F32BitsToBf16Bits(uint32_t u) {
  const uint32_t abs_u = u & 0x7fffffffu;
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t rounded = (u + (0x7fffu + lsb)) >> 16;
  const uint32_t nan_val = ((u & 0x80000000u) >> 16) | 0x7fc0u;
  return static_cast<uint16_t>(abs_u > 0x7f800000u ? nan_val : rounded);
}

// Branchless float32 -> 1-byte custom float bit conversion (RTNE).
template <typename To>
ML_DTYPES_ALWAYS_INLINE uint8_t F32BitsToByteFloatBits(uint32_t u) {
  using T = float8_internal::Traits<To>;
  const uint32_t sign_bit = u >> 31;
  const uint32_t abs_u = u & 0x7fffffffu;

  constexpr uint32_t kPosInf = std::numeric_limits<To>::infinity().rep();
  constexpr uint32_t kNegInf = (-std::numeric_limits<To>::infinity()).rep();
  constexpr uint32_t kPosNan = std::numeric_limits<To>::quiet_NaN().rep();
  constexpr uint32_t kNegNan = (-std::numeric_limits<To>::quiet_NaN()).rep();
  constexpr uint32_t kMaxFinite = std::numeric_limits<To>::max().rep();

  if constexpr (!T::kHasZero) {
    // Unsigned exponent-only format (float8_e8m0fnu).
    const uint32_t normal_res = (abs_u + 0x400000u) >> 23;
    const uint32_t subnorm_res = abs_u >= 0x00600000u ? 1u : 0u;
    uint32_t res = abs_u >= 0x00800000u ? normal_res : subnorm_res;
    res = res > kMaxFinite ? kPosInf : res;
    res = (sign_bit != 0u || abs_u == 0u) ? kPosNan : res;
    res = abs_u == 0x7f800000u ? (sign_bit ? kNegInf : kPosInf) : res;
    res = abs_u > 0x7f800000u ? (sign_bit ? kNegNan : kPosNan) : res;
    return static_cast<uint8_t>(res);
  } else {
    constexpr int kM = T::kMantissaBits;
    constexpr int kB = T::kExponentBias;
    constexpr int kMantShift = 23 - kM;
    constexpr uint32_t kBiasDiff = static_cast<uint32_t>(127 - kB) << 23;
    constexpr uint32_t kHalfMinus1 = (uint32_t{1} << (kMantShift - 1)) - 1u;
    constexpr uint32_t kMinNormal = static_cast<uint32_t>(128 - kB) << 23;
    constexpr uint32_t kUnderflow = static_cast<uint32_t>(127 - kB - kM) << 23;
    constexpr int kSubnormShiftBase = 151 - kM - kB;
    constexpr bool kHasNegZero = (-To::FromRep(0)).rep() != 0;

    const uint32_t n_clamped = abs_u < 0x7f800000u ? abs_u : 0x7f800000u;
    const uint32_t n_sub = n_clamped - kBiasDiff;
    const uint32_t n_lsb = (n_sub >> kMantShift) & 1u;
    const uint32_t n_raw = (n_sub + kHalfMinus1 + n_lsb) >> kMantShift;
    const uint32_t n_res = n_raw > kMaxFinite ? kPosInf : n_raw;

    // Clamp `shift` to [1, 31] so unconditional shifts are defined on unused
    // lanes.
    const uint32_t exp = abs_u >> 23;
    const uint32_t raw_shift = static_cast<uint32_t>(kSubnormShiftBase) - exp;
    const uint32_t shift =
        raw_shift < 1u ? 1u : (raw_shift > 31u ? 31u : raw_shift);
    const uint32_t mant = (abs_u & 0x007fffffu) | 0x00800000u;
    const uint32_t s_lsb = (mant >> shift) & 1u;
    const uint32_t half_m1 = (uint32_t{1} << (shift - 1u)) - 1u;
    const uint32_t s_rounded = (mant + half_m1 + s_lsb) >> shift;
    const uint32_t s_res = abs_u > kUnderflow ? s_rounded : 0u;

    uint32_t res = abs_u >= kMinNormal ? n_res : s_res;
    if constexpr (T::kIsSigned) {
      const uint32_t sign = sign_bit << (T::kBits - 1);
      if constexpr (kHasNegZero) {
        res = sign | res;
      } else {
        res = res == 0u ? 0u : (sign | res);
      }
    } else {
      res = (sign_bit != 0u && abs_u != 0u) ? kPosNan : res;
    }

    res = abs_u == 0x7f800000u ? (sign_bit ? kNegInf : kPosInf) : res;
    res = abs_u > 0x7f800000u ? (sign_bit ? kNegNan : kPosNan) : res;
    return static_cast<uint8_t>(res);
  }
}

template <typename To>
ML_DTYPES_ALWAYS_INLINE uint8_t Bf16BitsToByteFloatBits(uint16_t u) {
  return F32BitsToByteFloatBits<To>(Bf16BitsToF32Bits(u));
}

template <typename From, typename To>
struct IntBitCastTypes {
  using SrcBits = std::conditional_t<is_intn_v<From>, uint8_t, From>;
  using DstBits = std::conditional_t<is_intn_v<To>, uint8_t, To>;
};

// Converts between raw integer representations where at least one side is a
// sub-byte `intN` type: masks unused upper bits, sign-extends signed `intN`
// inputs, and reduces `intN` outputs modulo 2^To::bits.
template <typename From, typename To>
ML_DTYPES_ALWAYS_INLINE typename IntBitCastTypes<From, To>::DstBits
IntBitsToIntBits(typename IntBitCastTypes<From, To>::SrcBits in) {
  using DstBits = typename IntBitCastTypes<From, To>::DstBits;
  if constexpr (is_intn_v<From>) {
    constexpr int kShift = 8 - From::bits;
    using Underlying = typename From::underlying_type;
    const Underlying extended =
        static_cast<Underlying>(static_cast<uint8_t>(in << kShift)) >> kShift;
    if constexpr (is_intn_v<To>) {
      constexpr uint8_t kMask = (uint8_t{1} << To::bits) - 1;
      return static_cast<uint8_t>(extended) & kMask;
    } else {
      return static_cast<DstBits>(extended);
    }
  } else {
    static_assert(is_intn_v<To>);
    constexpr uint8_t kMask = (uint8_t{1} << To::bits) - 1;
    return static_cast<uint8_t>(in) & kMask;
  }
}

// Runtime-dispatched SIMD targets in priority order:
//   X(Name, TargetAttr, VecBytes, Scalable, ProbeExpr)
inline bool InitCpuDispatch() {
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
  __builtin_cpu_init();
#endif
  return true;
}
#define ML_DTYPES_CPU_SUPPORTS(feature) \
  (InitCpuDispatch() && __builtin_cpu_supports(feature))

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ML_DTYPES_DISPATCH_TARGETS(X)                                \
  X(Avx512, "avx512f", 64, false, ML_DTYPES_CPU_SUPPORTS("avx512f")) \
  X(Avx2, "avx2", 32, false, ML_DTYPES_CPU_SUPPORTS("avx2"))
#elif defined(__aarch64__) && defined(__clang__) && !defined(__APPLE__) && \
    __clang_major__ >= 19
#define ML_DTYPES_DISPATCH_TARGETS(X) \
  X(Sve, "+sve", 16, true, ML_DTYPES_CPU_SUPPORTS("sve"))
#else
#define ML_DTYPES_DISPATCH_TARGETS(X)
#endif

template <typename SrcT, typename DstT>
inline constexpr int kMaxElemBytes = sizeof(SrcT) > sizeof(DstT) ? sizeof(SrcT)
                                                                 : sizeof(DstT);

// Converts element `i` from raw `SrcT` bits to raw `DstT` bits via `memcpy` to
// avoid strict-aliasing and alignment UB.
template <typename SrcT, typename DstT, DstT (*Fn)(SrcT)>
ML_DTYPES_ALWAYS_INLINE void BitwiseConvertElement(const char* src, char* dst,
                                                   npy_intp i) {
  SrcT in;
  std::memcpy(&in, src + i * sizeof(SrcT), sizeof(SrcT));
  const DstT out = Fn(in);
  std::memcpy(dst + i * sizeof(DstT), &out, sizeof(DstT));
}

#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
// Contiguous elementwise loop over `Fn`. Explicit `vectorize_width` hints on
// Clang override default vector-width caps (such as `-mprefer-vector-width`)
// and opt into scalable SVE vectors on AArch64.
template <typename SrcT, typename DstT, DstT (*Fn)(SrcT), int kVecBytes,
          bool kScalable>
struct BitwiseContiguousLoop {
  ML_DTYPES_ALWAYS_INLINE void operator()(const char* src, char* dst,
                                          npy_intp n) const {
#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang loop vectorize_width(kVecBytes / kMaxElemBytes<SrcT, DstT>)
#endif
    for (npy_intp i = 0; i < n; ++i) {
      BitwiseConvertElement<SrcT, DstT, Fn>(src, dst, i);
    }
  }
};

template <typename SrcT, typename DstT, DstT (*Fn)(SrcT), int kVecBytes>
struct BitwiseContiguousLoop<SrcT, DstT, Fn, kVecBytes, /*kScalable=*/true> {
  ML_DTYPES_ALWAYS_INLINE void operator()(const char* src, char* dst,
                                          npy_intp n) const {
#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang loop vectorize_width(kVecBytes / kMaxElemBytes<SrcT, DstT>, \
                                   scalable)
#endif
    for (npy_intp i = 0; i < n; ++i) {
      BitwiseConvertElement<SrcT, DstT, Fn>(src, dst, i);
    }
  }
};

template <typename SrcT, typename DstT, DstT (*Fn)(SrcT)>
struct BitwiseContiguousLoop<SrcT, DstT, Fn, /*kVecBytes=*/0,
                             /*kScalable=*/false> {
  ML_DTYPES_ALWAYS_INLINE void operator()(const char* src, char* dst,
                                          npy_intp n) const {
    for (npy_intp i = 0; i < n; ++i) {
      BitwiseConvertElement<SrcT, DstT, Fn>(src, dst, i);
    }
  }
};

template <typename SrcT, typename DstT, DstT (*Fn)(SrcT)>
inline void RunContiguousDefault(const char* src, char* dst, npy_intp n) {
  BitwiseContiguousLoop<SrcT, DstT, Fn, /*kVecBytes=*/0,
                        /*kScalable=*/false>{}(src, dst, n);
}

#define ML_DTYPES_DEFINE_RUNNER(Name, TargetAttr, VecBytes, Scalable, Probe)  \
  template <typename SrcT, typename DstT, DstT (*Fn)(SrcT)>                   \
  __attribute__((target(TargetAttr), noinline)) static void                   \
  RunContiguous##Name(const char* src, char* dst, npy_intp n) {               \
    BitwiseContiguousLoop<SrcT, DstT, Fn, VecBytes, Scalable>{}(src, dst, n); \
  }
ML_DTYPES_DISPATCH_TARGETS(ML_DTYPES_DEFINE_RUNNER)
#undef ML_DTYPES_DEFINE_RUNNER
#if ML_DTYPES_HAS_CLANG_LOOP_PRAGMAS
#pragma clang diagnostic pop
#endif

// Dispatches a contiguous bitwise loop to the highest-priority SIMD target
// supported by the current CPU, caching the selected function pointer once.
template <typename SrcT, typename DstT, DstT (*Fn)(SrcT)>
inline void DispatchContiguous(const char* src, char* dst, npy_intp n) {
  using LoopFn = void (*)(const char*, char*, npy_intp);
  static const LoopFn kFn = []() -> LoopFn {
#define ML_DTYPES_TRY_TARGET(Name, TargetAttr, VecBytes, Scalable, Probe) \
  if (Probe) {                                                            \
    return &RunContiguous##Name<SrcT, DstT, Fn>;                          \
  }
    ML_DTYPES_DISPATCH_TARGETS(ML_DTYPES_TRY_TARGET)
#undef ML_DTYPES_TRY_TARGET
    return &RunContiguousDefault<SrcT, DstT, Fn>;
  }();
  kFn(src, dst, n);
}

// Maps a `(From, To)` pair with a branchless bitwise fast path to its raw
// integer bit types and element conversion function (used by both contiguous
// and strided cast loops).
template <typename From, typename To, typename Enable = void>
struct BitwiseCastTraits {
  static constexpr bool kSupported = false;
};

template <>
struct BitwiseCastTraits<bfloat16, float> {
  static constexpr bool kSupported = true;
  using SrcBits = uint16_t;
  using DstBits = uint32_t;
  static constexpr DstBits (*kFn)(SrcBits) = &Bf16BitsToF32Bits;
};

template <>
struct BitwiseCastTraits<float, bfloat16> {
  static constexpr bool kSupported = true;
  using SrcBits = uint32_t;
  using DstBits = uint16_t;
  static constexpr DstBits (*kFn)(SrcBits) = &F32BitsToBf16Bits;
};

template <typename To>
struct BitwiseCastTraits<float, To,
                         std::enable_if_t<is_byte_custom_float_v<To>>> {
  static constexpr bool kSupported = true;
  using SrcBits = uint32_t;
  using DstBits = uint8_t;
  static constexpr DstBits (*kFn)(SrcBits) = &F32BitsToByteFloatBits<To>;
};

template <typename To>
struct BitwiseCastTraits<bfloat16, To,
                         std::enable_if_t<is_byte_custom_float_v<To>>> {
  static constexpr bool kSupported = true;
  using SrcBits = uint16_t;
  using DstBits = uint8_t;
  static constexpr DstBits (*kFn)(SrcBits) = &Bf16BitsToByteFloatBits<To>;
};

template <typename From, typename To>
struct BitwiseCastTraits<From, To,
                         std::enable_if_t<kHasIntBitwiseFastPath<From, To>>> {
  static constexpr bool kSupported = true;
  using SrcBits = typename IntBitCastTypes<From, To>::SrcBits;
  using DstBits = typename IntBitCastTypes<From, To>::DstBits;
  static constexpr DstBits (*kFn)(SrcBits) = &IntBitsToIntBits<From, To>;
};

template <typename From, typename To>
inline constexpr bool kHasContiguousFastPath =
    BitwiseCastTraits<From, To>::kSupported || kHasByteLutFastPath<From, To>;

// Bitwise and LUT casts never raise floating-point exceptions.
template <typename From, typename To>
inline constexpr bool kIsBitwiseNoFpErrorCast =
    BitwiseCastTraits<From, To>::kSupported || kHasByteLutFastPath<From, To>;

// NumPy `NPY_METH_contiguous_loop` entry point for casts with a vectorized
// bitwise or lookup-table fast path. Runs without holding the Python GIL.
template <typename From, typename To>
int ContiguousCastLoop(PyArrayMethod_Context* /*context*/, char* const data[],
                       const npy_intp dimensions[], const npy_intp* /*strides*/,
                       NpyAuxData* /*auxdata*/) {
  const npy_intp n = dimensions[0];
  if constexpr (BitwiseCastTraits<From, To>::kSupported) {
    using Traits = BitwiseCastTraits<From, To>;
    DispatchContiguous<typename Traits::SrcBits, typename Traits::DstBits,
                       Traits::kFn>(data[0], data[1], n);
  } else if constexpr (kHasByteLutFastPath<From, To>) {
    CastByteTypeViaLutContiguous<From, To>(
        reinterpret_cast<const uint8_t*>(data[0]),
        reinterpret_cast<To*>(data[1]), n);
  } else {
    static_assert(BitwiseCastTraits<From, To>::kSupported ||
                      kHasByteLutFastPath<From, To>,
                  "Missing contiguous fast path implementation");
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

  if constexpr (kHasByteLutFastPath<From, To>) {
    const auto& table = ByteTypeLut<From, To>();
    for (npy_intp i = 0; i < n; ++i) {
      uint8_t raw;
      std::memcpy(&raw, src + i * src_stride, sizeof(raw));
      const To out = table[raw];
      std::memcpy(dst + i * dst_stride, &out, sizeof(out));
    }
    return 0;
  }
  if constexpr (BitwiseCastTraits<From, To>::kSupported) {
    using Traits = BitwiseCastTraits<From, To>;
    for (npy_intp i = 0; i < n; ++i) {
      typename Traits::SrcBits raw;
      std::memcpy(&raw, src + i * src_stride, sizeof(raw));
      const typename Traits::DstBits out = Traits::kFn(raw);
      std::memcpy(dst + i * dst_stride, &out, sizeof(out));
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
