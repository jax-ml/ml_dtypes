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

// Shared DType registration and within-dtype casting.

#ifndef ML_DTYPES_DTYPE_COMMON_H_
#define ML_DTYPES_DTYPE_COMMON_H_

#include <Python.h>

#include <vector>

// clang-format off
#include "ml_dtypes/_src/numpy.h"  // NOLINT (must precede numpy/*)
// clang-format on

#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

namespace ml_dtypes {

// DType metaclass setup.

// NumPy < 2.5 requires explicit tp_repr and tp_str slots.
inline PyObject* DTypeRepr(PyObject* self) {
  return PyArrayDescr_Type.tp_repr(self);
}
inline PyObject* DTypeStr(PyObject* self) {
  return PyArrayDescr_Type.tp_str(self);
}

// Initializes the DType class and calls PyType_Ready.
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

// Native descriptors retain their metadata; swapped descriptors normalize
// to the singleton, discarding metadata.
inline PyArray_Descr* EnsureCanonical(PyArray_Descr* self) {
  if (PyDataType_ISNOTSWAPPED(self)) {
    Py_INCREF(self);
    return self;
  }
  PyArray_Descr* singleton = NPY_DTYPE(self)->singleton;
  Py_INCREF(singleton);
  return singleton;
}

// Returns the singleton descriptor; the inherited constructor fails on NumPy
// < 2.5.
inline PyArray_Descr* DefaultDescr(PyArray_DTypeMeta* cls) {
  Py_INCREF(cls->singleton);
  return cls->singleton;
}

// Within-dtype casts.

// Copies values using the legacy copy/swap loop.
static int WithinDTypeCastLoop(PyArrayMethod_Context* context,
                               char* const data[], const npy_intp dimensions[],
                               const npy_intp strides[],
                               NpyAuxData* /*auxdata*/) {
  const bool swap = PyDataType_ISNOTSWAPPED(context->descriptors[0]) !=
                    PyDataType_ISNOTSWAPPED(context->descriptors[1]);
  PyArray_ArrFuncs* funcs = PyDataType_GetArrFuncs(context->descriptors[0]);
  funcs->copyswapn(data[1], strides[1], data[0], strides[0], dimensions[0],
                   swap, nullptr);
  return 0;
}

// Resolves descriptors without normalizing byte order; matching orders allow a
// view.
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

// DType registration.

// Registers a fixed-size DType with its self-cast and cross-dtype ArrayMethods.
inline int InitDTypeFromSlots(
    PyArray_DTypeMeta* dm, PyTypeObject* scalar_type, PyType_Slot* slots,
    const std::vector<PyArrayMethod_Spec*>& cross_casts = {}) {
  PyArray_DTypeMeta* self_cast_dtypes[2] = {nullptr, nullptr};
  PyType_Slot self_cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(WithinDTypeCastResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(WithinDTypeCastLoop)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void*>(WithinDTypeCastLoop)},
      {0, nullptr}};
  PyArrayMethod_Spec self_cast_spec;
  self_cast_spec.name = "within_dtype_cast";
  self_cast_spec.nin = 1;
  self_cast_spec.nout = 1;
  // Byte swaps need equiv casting; the resolver permits views when possible.
  self_cast_spec.casting = NPY_EQUIV_CASTING;
  self_cast_spec.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_NO_FLOATINGPOINT_ERRORS);
  self_cast_spec.dtypes = self_cast_dtypes;
  self_cast_spec.slots = self_cast_slots;

  std::vector<PyArrayMethod_Spec*> casts;
  casts.reserve(cross_casts.size() + 2);
  casts.push_back(&self_cast_spec);
  casts.insert(casts.end(), cross_casts.begin(), cross_casts.end());
  casts.push_back(nullptr);

  PyArrayDTypeMeta_Spec dtype_spec;
  dtype_spec.typeobj = scalar_type;
  dtype_spec.flags = NPY_DT_NUMERIC;
  dtype_spec.casts = casts.data();
  dtype_spec.slots = slots;
  dtype_spec.baseclass = nullptr;

  return PyArrayInitDTypeMeta_FromSpec(dm, &dtype_spec);
}

}  // namespace ml_dtypes

#endif  // ML_DTYPES_DTYPE_COMMON_H_
