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

#if NPY_TARGET_VERSION >= 0x15  // NUMPY_2_4_API_VERSION
#define ARRFUNCS_OFFSET_FIX(v) (v)
#else
#define ARRFUNCS_OFFSET_FIX(v) \
  (v) - (NPY_DT_PyArray_ArrFuncs_getitem) + 1 + (((PyArray_RUNTIME_VERSION >= 0x15) ? (1 << 11) : (1 << 10)))
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

/*
 * PyArrayInitDTypeMeta_FromSpec_WithLegacy
 *
 * Initialises a new-style user DType (via PyArrayInitDTypeMeta_FromSpec) while
 * also plumbing in legacy compatibility so that NumPy < 2.5 assigns a
 * type_num, singleton, and legacy flag.
 *
 * Algorithm (for NumPy 2.0 – 2.4):
 *
 *   Step 1 – Legacy registration (for type_num + singleton allocation):
 *     Temporarily replace proto->typeobj with &PyBaseObject_Type so that
 *     _PyArray_MapPyTypeToDType sees a non-generic type and hits the
 *     NPY_DT_is_legacy bail-out, meaning the auto-DTypeMeta created by
 *     PyArray_RegisterDataType is NOT inserted into the pytype-to-DType dict.
 *
 *   Step 2 – New-style init:
 *     PyArrayInitDTypeMeta_FromSpec sets up slots, casts, and the
 *     pytype-to-DType mapping for the real scalar type with no conflict.
 *
 *   Step 3 – Swap:
 *     Steal type_num + singleton from the old legacy registration and point
 *     the singleton at the user's new DType.  Set the legacy flag so NumPy
 *     uses legacy descriptor code paths where needed.
 *
 * If proto is NULL the function just forwards to PyArrayInitDTypeMeta_FromSpec.
 */
static inline int PyArrayInitDTypeMeta_FromSpec_WithLegacy(
    PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec,
    PyArray_DescrProto *proto) {
  if (proto == nullptr) {
    return PyArrayInitDTypeMeta_FromSpec(DType, spec);
  }

  /*
   * Step 1: Register old-style with a garbage typeobj so that
   * _PyArray_MapPyTypeToDType does NOT add the auto-DTypeMeta to the
   * pytype-to-DType dict (it bails out on NPY_DT_is_legacy for non-generic
   * types), regardless of whether the real scalar subclasses np.generic.
   */
  PyTypeObject *real_typeobj = proto->typeobj;
  proto->typeobj = &PyBaseObject_Type;

  int typenum = PyArray_RegisterDataType(proto);

  proto->typeobj = real_typeobj;
  if (typenum < 0) {
    return -1;
  }

  /*
   * Step 2: Initialise the user's DType with new-style slots and casts.
   * type_num stays at -1 / 0 for now; we fix it in step 3.
   */
  if (PyArrayInitDTypeMeta_FromSpec(DType, spec) < 0) {
    return -1;
  }

  /*
   * Step 3: Steal the singleton descriptor and type_num from the legacy
   * registration.  Point the descriptor's Python type at the user's DType
   * and fix up its typeobj field (which we temporarily set to PyBaseObject_Type
   * in step 1).
   */
  PyArray_Descr *descr = PyArray_DescrFromType(typenum);
  if (descr == nullptr) {
    return -1;
  }

  /* Save the auto-DTypeMeta so we can decref it after the swap. */
  PyObject *old_meta = reinterpret_cast<PyObject *>(Py_TYPE(descr));

  DType->type_num = typenum;
  /* PyArray_DescrFromType returns a new reference; transfer ownership. */
  DType->singleton = descr;
  /* Set the legacy flag (bit 0 == _NPY_DT_LEGACY_FLAG) so NumPy uses legacy
   * code paths (copyswap, ArrFuncs, etc.) where the new-style API doesn't
   * cover them yet. */
  DType->flags |= 1;

  /* Re-type the descriptor so it belongs to the user's DType class. */
  Py_INCREF(reinterpret_cast<PyObject *>(DType));
  Py_SET_TYPE(descr, reinterpret_cast<PyTypeObject *>(DType));
  Py_DECREF(old_meta);

  /* Fix the descriptor's scalar-type field (it was set to PyBaseObject_Type
   * in step 1 by PyArray_RegisterDataType copying proto->typeobj). */
  Py_INCREF(real_typeobj);
  Py_XDECREF(descr->typeobj);
  descr->typeobj = real_typeobj;

  /*
   * Patch copyswap/copyswapn into the new DType's legacy f-slots.
   *
   * copyswap and copyswapn are disabled as public NPY_DT_PyArray_ArrFuncs_*
   * spec slots (commented out in dtype_api.h), so dtypemeta_initialize_struct_
   * from_spec leaves them as stubs from default_funcs.  PyArray_Scalar and
   * other legacy paths call copyswap directly, so we must fill it in.
   */
  if (proto->f != nullptr) {
    PyArray_ArrFuncs *f = _PyDataType_GetArrFuncs(descr);
    f->copyswap = proto->f->copyswap;
    f->copyswapn = proto->f->copyswapn;
  }

  return 0;
}

}  // namespace ml_dtypes

#endif  // ML_DTYPES_DTYPE_COMPAT_H_
