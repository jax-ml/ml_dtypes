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

// Enable cmath defines on Windows
#define _USE_MATH_DEFINES

// Must be included first
// clang-format off
#include "_src/numpy.h" //NOLINT
// clang-format on

#include <array>   // NOLINT
#include <cmath>   // NOLINT
#include <limits>  // NOLINT
#include <locale>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "Eigen/Core"
#include "_src/custom_float.h"
#include "_src/int4.h"
#include "include/float8.h"

namespace ml_dtypes {

using bfloat16 = Eigen::bfloat16;

template <>
struct TypeDescriptor<bfloat16> : CustomFloatType<bfloat16> {
  typedef bfloat16 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bfloat16";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.bfloat16";
  static constexpr const char* kTpDoc = "bfloat16 floating-point values";
  // We must register bfloat16 with a kind other than "f", because numpy
  // considers two types with the same kind and size to be equal, but
  // float16 != bfloat16.
  // The downside of this is that NumPy scalar promotion does not work with
  // bfloat16 values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'E';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3b11fnuz>
    : CustomFloatType<float8_e4m3b11fnuz> {
  typedef float8_e4m3b11fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3b11fnuz";
  static constexpr const char* kQualifiedTypeName =
      "ml_dtypes.float8_e4m3b11fnuz";
  static constexpr const char* kTpDoc =
      "float8_e4m3b11fnuz floating-point values";
  // We must register float8_e4m3b11fnuz with a kind other than "f", because
  // numpy considers two types with the same kind and size to be equal, and we
  // expect multiple 1 byte floating point types.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8_e4m3b11fnuz values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'L';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fn> : CustomFloatType<float8_e4m3fn> {
  typedef float8_e4m3fn T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  // We must register float8_e4m3fn with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.  Using 'V' to mirror bfloat16 vs float16.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '4';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fnuz> : CustomFloatType<float8_e4m3fnuz> {
  typedef float8_e4m3fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fnuz";
  static constexpr const char* kTpDoc = "float8_e4m3fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'G';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2> : CustomFloatType<float8_e5m2> {
  typedef float8_e5m2 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  // Treating e5m2 as the natural "float" type since it is IEEE-754 compliant.
  static constexpr char kNpyDescrKind = 'f';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '5';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2fnuz> : CustomFloatType<float8_e5m2fnuz> {
  typedef float8_e5m2fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2fnuz";
  static constexpr const char* kTpDoc = "float8_e5m2fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'C';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int4> : Int4TypeDescriptor<int4> {
  typedef int4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int4";
  static constexpr const char* kTpDoc = "int4 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'a';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint4> : Int4TypeDescriptor<uint4> {
  typedef uint4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint4";
  static constexpr const char* kTpDoc = "uint4 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'A';
  static constexpr char kNpyDescrByteorder = '=';
};

namespace {

// Performs a NumPy array cast from type 'From' to 'To' via float.
template <typename From, typename To>
void FloatPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<float>(from[i]));
  }
}

template <typename Type1, typename Type2>
bool RegisterTwoWayCustomCast() {
  int nptype1 = TypeDescriptor<Type1>::npy_type;
  int nptype2 = TypeDescriptor<Type2>::npy_type;
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, FloatPyCast<Type1, Type2>) <
      0) {
    return false;
  }
  PyArray_Descr* descr2 = PyArray_DescrFromType(nptype2);
  if (PyArray_RegisterCastFunc(descr2, nptype1, FloatPyCast<Type2, Type1>) <
      0) {
    return false;
  }
  return true;
}

}  // namespace

// Initializes the module.
bool Initialize() {
  ml_dtypes::ImportNumpy();
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  if (!RegisterFloatDtype<bfloat16>(numpy.get())) {
    return false;
  }
  bool float8_e4m3b11fnuz_already_registered;
  if (!RegisterFloatDtype<float8_e4m3b11fnuz>(
          numpy.get(), &float8_e4m3b11fnuz_already_registered)) {
    return false;
  }
  bool float8_e4m3fn_already_registered;
  if (!ml_dtypes::RegisterFloatDtype<float8_e4m3fn>(
          numpy.get(), &float8_e4m3fn_already_registered)) {
    return false;
  }
  bool float8_e4m3fnuz_already_registered;
  if (!ml_dtypes::RegisterFloatDtype<float8_e4m3fnuz>(
          numpy.get(), &float8_e4m3fnuz_already_registered)) {
    return false;
  }
  bool float8_e5m2_already_registered;
  if (!ml_dtypes::RegisterFloatDtype<float8_e5m2>(
          numpy.get(), &float8_e5m2_already_registered)) {
    return false;
  }
  bool float8_e5m2fnuz_already_registered;
  if (!ml_dtypes::RegisterFloatDtype<float8_e5m2fnuz>(
          numpy.get(), &float8_e5m2fnuz_already_registered)) {
    return false;
  }

  if (!ml_dtypes::RegisterInt4Dtype<int4>(numpy.get())) {
    return false;
  }

  if (!ml_dtypes::RegisterInt4Dtype<uint4>(numpy.get())) {
    return false;
  }

  // Casts between bfloat16 and float8_e4m3b11nuz. Only perform the cast if
  // float8_e4m3b11nuz hasn't been previously registered, presumably by a
  // different library. In this case, we assume the cast has also already been
  // registered, and registering it again can cause segfaults due to accessing
  // an uninitialized type descriptor in this library.
  if (!float8_e4m3b11fnuz_already_registered &&
      !RegisterCustomFloatCast<float8_e4m3b11fnuz, bfloat16>()) {
    return false;
  }
  if (!float8_e4m3fnuz_already_registered &&
      !float8_e5m2fnuz_already_registered &&
      !RegisterTwoWayCustomCast<float8_e4m3fnuz, float8_e5m2fnuz>()) {
    return false;
  }
  if (!float8_e4m3fn_already_registered && !float8_e5m2_already_registered &&
      !RegisterCustomFloatCast<float8_e4m3fn, float8_e5m2>()) {
    return false;
  }
  bool success = true;
  // Continue trying to register casts, just in case some types are not
  // registered (i.e. float8_e4m3b11fnuz)
  success &= RegisterTwoWayCustomCast<float8_e4m3b11fnuz, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<float8_e4m3b11fnuz, float8_e5m2>();
  success &= RegisterTwoWayCustomCast<bfloat16, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<bfloat16, float8_e5m2>();
  success &= RegisterTwoWayCustomCast<float8_e4m3fnuz, bfloat16>();
  success &= RegisterTwoWayCustomCast<float8_e5m2fnuz, bfloat16>();
  success &= RegisterTwoWayCustomCast<float8_e4m3fnuz, float8_e4m3b11fnuz>();
  success &= RegisterTwoWayCustomCast<float8_e5m2fnuz, float8_e4m3b11fnuz>();
  success &= RegisterTwoWayCustomCast<float8_e4m3fnuz, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<float8_e5m2fnuz, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<float8_e4m3fnuz, float8_e5m2>();
  success &= RegisterTwoWayCustomCast<float8_e5m2fnuz, float8_e5m2>();
  return success;
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_custom_floats",
};

// TODO(phawkins): PyMODINIT_FUNC handles visibility correctly in Python 3.9+.
// Just use PyMODINIT_FUNC after dropping Python 3.8 support.
#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C" EXPORT_SYMBOL PyObject* PyInit__custom_floats() {
  Safe_PyObjectPtr m = make_safe(PyModule_Create(&module_def));
  if (!m) {
    return nullptr;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load _custom_floats module.");
    }
    return nullptr;
  }

  if (PyObject_SetAttrString(
          m.get(), "float8_e4m3b11fnuz",
          reinterpret_cast<PyObject*>(
              TypeDescriptor<float8_e4m3b11fnuz>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float8_e4m3fn",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float8_e4m3fn>::type_ptr)) <
      0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float8_e4m3fnuz",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float8_e4m3fnuz>::type_ptr)) <
      0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float8_e5m2",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float8_e5m2>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float8_e5m2fnuz",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float8_e5m2fnuz>::type_ptr)) <
      0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "bfloat16",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<bfloat16>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(
          m.get(), "int4",
          reinterpret_cast<PyObject*>(TypeDescriptor<int4>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(
          m.get(), "uint4",
          reinterpret_cast<PyObject*>(TypeDescriptor<uint4>::type_ptr)) < 0) {
    return nullptr;
  }
  return m.release();
}
}  // namespace ml_dtypes
