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
 * Shared machinery for defining new style array methods (casts/ufuncs) and
 * creating the new style DTypes related to PyArrayInitDTypeMeta_FromSpec.
 */

#ifndef ML_DTYPES_DTYPE_COMMON_H_
#define ML_DTYPES_DTYPE_COMMON_H_

// clang-format off
#include "ml_dtypes/_src/numpy.h"  // NOLINT (must be first)
// clang-format on

#include <Python.h>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#include "common.h"

namespace ml_dtypes {

// ---------------------------------------------------------------------------
// Metaclass setup
// ---------------------------------------------------------------------------

// NumPy < 2.5 forces every DType metaclass to define tp_repr / tp_str, so we
// provide real functions that forward to the base PyArrayDescr_Type behaviour.
inline PyObject* DTypeRepr(PyObject* self) {
  return PyArrayDescr_Type.tp_repr(self);
}
inline PyObject* DTypeStr(PyObject* self) {
  return PyArrayDescr_Type.tp_str(self);
}

// Initializes the common fields of a new-style DType metaclass object and runs
// PyType_Ready.  All our DTypes derive from PyArrayDescr_Type and are plain
// value types, so the only thing that varies is the name.
inline bool InitDTypeMeta(PyArray_DTypeMeta* dm, const char* name) {
  auto* tp = reinterpret_cast<PyTypeObject*>(dm);
  tp->tp_name = name;
  tp->tp_base = &PyArrayDescr_Type;
  tp->tp_flags = Py_TPFLAGS_DEFAULT;
  tp->tp_basicsize = sizeof(_PyArray_LegacyDescr);
  tp->tp_repr = DTypeRepr;
  tp->tp_str = DTypeStr;
  return PyType_Ready(tp) >= 0;
}

// ---------------------------------------------------------------------------
// Within-dtype cast (copy / byte swap)
// ---------------------------------------------------------------------------

// Identity used for the within-dtype copy.  The cast loops below are templated
// on the operation applied to each element so they can be reused when porting
// the remaining casts off the legacy ArrFuncs path; for the copy it is inlined
// away to a plain load/store.
template <typename T>
struct CopyOp {
  T operator()(const T& x) const { return x; }
};

template <typename Op, typename In, typename Out, bool contiguous>
static int StridedUnaryLoop(PyArrayMethod_Context* /*context*/,
                            char* const data[],
                            const npy_intp dimensions[],
                            const npy_intp strides[],
                            NpyAuxData* /*auxdata*/) {
  const npy_intp n = dimensions[0];
  const char* in = data[0];
  char* out = data[1];
  npy_intp stride_in, stride_out;
  if constexpr (contiguous) {
    stride_in = sizeof(In);
    stride_out = sizeof(Out);
  } else {
    stride_in = strides[0];
    stride_out = strides[1];
  }

  Op op;
  for (npy_intp i = 0; i < n; ++i) {
    *reinterpret_cast<Out*>(out) = op(*reinterpret_cast<const In*>(in));
    in += stride_in;
    out += stride_out;
  }
  return 0;
}

// Unaligned within-dtype copy: a plain memcpy per element handles any
// alignment and stride. This also handles byte-swapping if needed.
template <typename T, bool swap = false>
static int UnalignedStridedCopyLoop(PyArrayMethod_Context* /*context*/,
                                    char* const data[],
                                    const npy_intp dimensions[],
                                    const npy_intp strides[],
                                    NpyAuxData* /*auxdata*/) {
  const npy_intp n = dimensions[0];
  const char* in = data[0];
  char* out = data[1];
  for (npy_intp i = 0; i < n; ++i) {
    std::memcpy(out, in, sizeof(T));
    if constexpr (swap) {
      if constexpr (is_complex_v<T>) {
        static_assert(sizeof(T) == 4);  // currently only have 32bit complex
        ByteSwap16(out);
        ByteSwap16(out + 2);
      }
      else if constexpr (sizeof(T) == 2) {
        ByteSwap16(out);
      } else if constexpr (sizeof(T) == 4) {
        ByteSwap32(out);
      }
      else {
        // static assert needs to depend on T, so check sizeof(T) is single byte.
        static_assert(sizeof(T) == 1);
      }
    }
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

// resolve_descriptors for the within-dtype cast which doesn't do much
// since given dtypes are also the loop ones.  Does indicate view and casting
// safety.  (The default resolve_descriptors may normalize the byte order.)
static NPY_CASTING WithinDTypeCastResolve(
    struct PyArrayMethodObject_tag* /*method*/,
    PyArray_DTypeMeta* const* dtypes, PyArray_Descr* const* given_descrs,
    PyArray_Descr** loop_descrs, npy_intp* view_offset) {
  Py_INCREF(given_descrs[0]);
  loop_descrs[0] = given_descrs[0];
  if (given_descrs[1] != nullptr) {
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
  } else {
    Py_INCREF(dtypes[1]->singleton);
    loop_descrs[1] = dtypes[1]->singleton;
  }
  if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
      PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
    *view_offset = 0;
    return NPY_NO_CASTING;
  }
  return NPY_EQUIV_CASTING;
}

// get_loop for the within-dtype cast.  Selects the appropriate specialization
// based on byte order, alignment and contiguity.
template <typename T>
static int WithinDTypeCastGetLoop(PyArrayMethod_Context* context, int aligned,
                                  int /*move_references*/,
                                  const npy_intp* strides,
                                  PyArrayMethod_StridedLoop** out_loop,
                                  NpyAuxData** out_transferdata,
                                  NPY_ARRAYMETHOD_FLAGS* flags) {
  PyArray_Descr* const* descrs = context->descriptors;
  *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
  *out_transferdata = nullptr;
  const npy_intp elsize = static_cast<npy_intp>(sizeof(T));

  if (PyDataType_ISNOTSWAPPED(descrs[0]) != PyDataType_ISNOTSWAPPED(descrs[1])) {
    *out_loop = UnalignedStridedCopyLoop<T, /* swap */ true>;
  }
  else if (!aligned) {
    *out_loop = UnalignedStridedCopyLoop<T>;
  }
  else if (strides[0] == elsize && strides[1] == elsize) {
    *out_loop = StridedUnaryLoop<CopyOp<T>, T, T, true>;
  } else {
    *out_loop = StridedUnaryLoop<CopyOp<T>, T, T, false>;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Registers a fixed-size DType from its slots, wiring up the within-dtype cast
// shared by all our dtypes.  Returns 0 on success and -1 on failure.
template <typename T>
inline int InitDTypeFromSlots(PyArray_DTypeMeta* dm, PyTypeObject* scalar_type,
                              PyType_Slot* slots) {
  PyArray_DTypeMeta* self_cast_dtypes[2] = {nullptr, nullptr};
  PyType_Slot self_cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(WithinDTypeCastResolve)},
      {NPY_METH_get_loop,
       reinterpret_cast<void*>(WithinDTypeCastGetLoop<T>)},
      {0, nullptr}};
  PyArrayMethod_Spec self_cast_spec;
  self_cast_spec.name = "within_dtype_cast";
  self_cast_spec.nin = 1;
  self_cast_spec.nout = 1;
  self_cast_spec.casting = NPY_NO_CASTING;
  self_cast_spec.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_NO_FLOATINGPOINT_ERRORS);
  self_cast_spec.dtypes = self_cast_dtypes;
  self_cast_spec.slots = self_cast_slots;
  // TODO(seberg): It would be good to define all other casts here as well.
  PyArrayMethod_Spec* casts[] = {&self_cast_spec, nullptr};

  PyArrayDTypeMeta_Spec dtype_spec;
  dtype_spec.typeobj = scalar_type;
  dtype_spec.flags = NPY_DT_NUMERIC;
  dtype_spec.casts = casts;
  dtype_spec.slots = slots;
  dtype_spec.baseclass = nullptr;

  return PyArrayInitDTypeMeta_FromSpec(dm, &dtype_spec);
}

}  // namespace ml_dtypes

#endif  // ML_DTYPES_DTYPE_COMMON_H_
