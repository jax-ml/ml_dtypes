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

// Compile-only guard: built without Python or NumPy include paths, so the
// public header fails the build if it regains interpreter dependencies.

#include "ml_dtypes/include/complex_types.h"

#include <type_traits>

static_assert(sizeof(ml_dtypes::complex32) == 4 &&
                  sizeof(ml_dtypes::bcomplex32) == 4,
              "low-precision complex types must be four bytes");
static_assert(std::is_standard_layout<ml_dtypes::complex32>::value &&
                  std::is_standard_layout<ml_dtypes::bcomplex32>::value,
              "low-precision complex types must be standard layout");
static_assert(std::is_trivially_copyable<ml_dtypes::complex32>::value &&
                  std::is_trivially_copyable<ml_dtypes::bcomplex32>::value,
              "low-precision complex types must be trivially copyable");
