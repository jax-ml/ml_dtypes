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

#ifndef ML_DTYPES_CPP_COMMON_H_
#define ML_DTYPES_CPP_COMMON_H_

// Pure C++ core; must stay free of Python and NumPy dependencies.

#include <complex>
#include <type_traits>

#include "Eigen/Core"

namespace ml_dtypes {

using half = Eigen::half;
using bfloat16 = Eigen::bfloat16;

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

}  // namespace ml_dtypes

#endif  // ML_DTYPES_CPP_COMMON_H_
