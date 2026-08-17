/* Copyright 2024 The ml_dtypes Authors.

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

#ifndef ML_DTYPES_FLOATS_H_
#define ML_DTYPES_FLOATS_H_

#include <cstring>
#include <type_traits>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/include/float8.h"
#include "ml_dtypes/include/mxfloat.h"

namespace ml_dtypes {

template <typename T>
struct CustomFloatTraits {};

template <>
struct CustomFloatTraits<bfloat16> {
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
struct CustomFloatTraits<float8_e3m4> {
  static constexpr const char* kTypeName = "float8_e3m4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e3m4";
  static constexpr const char* kTpDoc = "float8_e3m4 floating-point values";
  // Set e3m4 kind as Void since kind=f (float) with itemsize=1 is used by e5m2
  static constexpr char kNpyDescrKind = 'V';  // Void
  static constexpr char kNpyDescrType = '3';
  static constexpr char kNpyDescrByteorder = '=';  // Native byte order
};

template <>
struct CustomFloatTraits<float8_e4m3> {
  static constexpr const char* kTypeName = "float8_e4m3";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3";
  static constexpr const char* kTpDoc = "float8_e4m3 floating-point values";
  // Set e4m3 kind as Void since kind=f (float) with itemsize=1 is used by e5m2
  static constexpr char kNpyDescrKind = 'V';       // Void
  static constexpr char kNpyDescrType = '7';       // '4' is reserved for e4m3fn
  static constexpr char kNpyDescrByteorder = '=';  // Native byte order
};

template <>
struct CustomFloatTraits<float8_e4m3b11fnuz> {
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
struct CustomFloatTraits<float8_e4m3fn> {
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
struct CustomFloatTraits<float8_e4m3fnuz> {
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
struct CustomFloatTraits<float8_e5m2> {
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
struct CustomFloatTraits<float8_e5m2fnuz> {
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
struct CustomFloatTraits<float6_e2m3fn> {
  static constexpr const char* kTypeName = "float6_e2m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e2m3fn";
  static constexpr const char* kTpDoc = "float6_e2m3fn floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = '8';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct CustomFloatTraits<float6_e3m2fn> {
  static constexpr const char* kTypeName = "float6_e3m2fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e3m2fn";
  static constexpr const char* kTpDoc = "float6_e3m2fn floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = '9';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct CustomFloatTraits<float4_e2m1fn> {
  static constexpr const char* kTypeName = "float4_e2m1fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float4_e2m1fn";
  static constexpr const char* kTpDoc = "float4_e2m1fn floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = '0';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct CustomFloatTraits<float8_e8m0fnu> {
  static constexpr const char* kTypeName = "float8_e8m0fnu";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e8m0fnu";
  static constexpr const char* kTpDoc = "float8_e8m0fnu floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'W';
  static constexpr char kNpyDescrByteorder = '=';
};

template <typename T, typename = void>
struct is_custom_float : std::false_type {};

template <typename T>
struct is_custom_float<T,
                       std::void_t<decltype(CustomFloatTraits<T>::kTypeName)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_custom_float_v = is_custom_float<T>::value;

template <typename T>
struct CustomFloatType {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyObject* type_ptr;

  static PyMethodDef methods[];
  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;
};

template <typename T>
struct DtypeTraits<T, std::enable_if_t<is_custom_float_v<T>>> {
  static int Dtype() { return CustomFloatType<T>::Dtype(); }
};

bool RegisterCustomFloats(PyObject* numpy);

}  // namespace ml_dtypes

#endif  // ML_DTYPES_FLOATS_H_
