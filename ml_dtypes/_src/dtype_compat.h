/* Copyright 2025 The ml_dtypes Authors.

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

/*
 * Compatibility helper for registering DTypes that work with both the new-style
 * NumPy DType API (PyArrayInitDTypeMeta_FromSpec) and the legacy
 * PyArray_RegisterDataType path.
 *
 * The PyArrayInitDTypeMeta_FromSpec_WithLegacy function is a backport hack for
 * NumPy 2.0.  Starting with NumPy 2.5/2.6 (TBD) there will be native NumPy
 * support and this can be replaced.
 *
 * NOTE: This file requires NumPy >= 2.0.
 */

#ifndef ML_DTYPES_DTYPE_COMPAT_H_
#define ML_DTYPES_DTYPE_COMPAT_H_

// clang-format off
#include "ml_dtypes/_src/numpy.h"  // NOLINT (must be first)
// clang-format on

#include <Python.h>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#if NPY_ABI_VERSION < 0x02000000
#error "ml_dtypes dtype_compat.h requires NumPy >= 2.0"
#endif

namespace ml_dtypes {

/*
 * Within-dtype copy strided loop: a plain memcpy per element.
 * Registered as the within_dtype_castingimpl for every fixed-size dtype.
 * MUST carry NPY_METH_SUPPORTS_UNALIGNED (NumPy enforces this for self-casts).
 */
static inline int TrivialStridedCopyLoop(PyArrayMethod_Context *context,
                                         char *const data[],
                                         npy_intp const dimensions[],
                                         npy_intp const strides[],
                                         NpyAuxData * /*auxdata*/) {
  const npy_intp N = dimensions[0];
  const npy_intp elsize = context->descriptors[0]->elsize;
  const char *in = data[0];
  char *out = data[1];
  for (npy_intp i = 0; i < N; ++i) {
    std::memcpy(out, in, elsize);
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

}  // namespace ml_dtypes

#endif  // ML_DTYPES_DTYPE_COMPAT_H_
