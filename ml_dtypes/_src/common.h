/* Copyright 2022 The ml_dtypes Authors.

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

#ifndef ML_DTYPES_COMMON_H_
#define ML_DTYPES_COMMON_H_

#include <Python.h>

#include <complex>
#include <memory>

#include "Eigen/Core"
#include "ml_dtypes/_src/cpp_common.h"
#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {

inline void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

inline void ByteSwap32(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}

inline int CompareFloats(float fx, float fy) {
  if (fx < fy) {
    return -1;
  }
  if (fy < fx) {
    return 1;
  }
  // NaNs sort to the end.
  if (!Eigen::numext::isnan(fx) && Eigen::numext::isnan(fy)) {
    return -1;
  }
  if (Eigen::numext::isnan(fx) && !Eigen::numext::isnan(fy)) {
    return 1;
  }
  return 0;
}

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
inline Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

template <typename T, typename Enable = void>
struct DtypeTraits {
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
  // static PyArray_DTypeMeta* DTypeMeta();  // Numpy DTypeMeta for T.
};

template <>
struct DtypeTraits<unsigned char> {
  static int Dtype() { return NPY_UBYTE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_UByteDType; }
};

template <>
struct DtypeTraits<unsigned short> {  // NOLINT
  static int Dtype() { return NPY_USHORT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_UShortDType; }
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct DtypeTraits<unsigned int> {
  static int Dtype() { return NPY_UINT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_UIntDType; }
};

template <>
struct DtypeTraits<unsigned long> {  // NOLINT
  static int Dtype() { return NPY_ULONG; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_ULongDType; }
};

template <>
struct DtypeTraits<unsigned long long> {  // NOLINT
  static int Dtype() { return NPY_ULONGLONG; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_ULongLongDType; }
};

template <>
struct DtypeTraits<signed char> {
  static int Dtype() { return NPY_BYTE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_ByteDType; }
};

template <>
struct DtypeTraits<short> {  // NOLINT
  static int Dtype() { return NPY_SHORT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_ShortDType; }
};

template <>
struct DtypeTraits<int> {
  static int Dtype() { return NPY_INT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_IntDType; }
};

template <>
struct DtypeTraits<long> {  // NOLINT
  static int Dtype() { return NPY_LONG; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_LongDType; }
};

template <>
struct DtypeTraits<long long> {  // NOLINT
  static int Dtype() { return NPY_LONGLONG; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_LongLongDType; }
};

template <>
struct DtypeTraits<bool> {
  static int Dtype() { return NPY_BOOL; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_BoolDType; }
};

template <>
struct DtypeTraits<ml_dtypes::half> {
  static int Dtype() { return NPY_HALF; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_HalfDType; }
};

template <>
struct DtypeTraits<float> {
  static int Dtype() { return NPY_FLOAT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_FloatDType; }
};

template <>
struct DtypeTraits<double> {
  static int Dtype() { return NPY_DOUBLE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_DoubleDType; }
};

template <>
struct DtypeTraits<long double> {
  static int Dtype() { return NPY_LONGDOUBLE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_LongDoubleDType; }
};

template <>
struct DtypeTraits<std::complex<float>> {
  static int Dtype() { return NPY_CFLOAT; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_CFloatDType; }
};

template <>
struct DtypeTraits<std::complex<double>> {
  static int Dtype() { return NPY_CDOUBLE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_CDoubleDType; }
};

template <>
struct DtypeTraits<std::complex<long double>> {
  static int Dtype() { return NPY_CLONGDOUBLE; }
  static PyArray_DTypeMeta* DTypeMeta() { return &PyArray_CLongDoubleDType; }
};

}  // namespace ml_dtypes

#endif  // ML_DTYPES_COMMON_H_
