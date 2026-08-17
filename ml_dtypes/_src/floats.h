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
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNumPy1DescrType = 'E';
};

template <>
struct CustomFloatTraits<float8_e3m4> {
  static constexpr const char* kTypeName = "float8_e3m4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e3m4";
  static constexpr const char* kTpDoc = "float8_e3m4 floating-point values";
  static constexpr char kNumPy1DescrType = '3';
};

template <>
struct CustomFloatTraits<float8_e4m3> {
  static constexpr const char* kTypeName = "float8_e4m3";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3";
  static constexpr const char* kTpDoc = "float8_e4m3 floating-point values";
  static constexpr char kNumPy1DescrType = '7';
};

template <>
struct CustomFloatTraits<float8_e4m3b11fnuz> {
  static constexpr const char* kTypeName = "float8_e4m3b11fnuz";
  static constexpr const char* kQualifiedTypeName =
      "ml_dtypes.float8_e4m3b11fnuz";
  static constexpr const char* kTpDoc =
      "float8_e4m3b11fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'L';
};

template <>
struct CustomFloatTraits<float8_e4m3fn> {
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  static constexpr char kNumPy1DescrType = '4';
};

template <>
struct CustomFloatTraits<float8_e4m3fnuz> {
  static constexpr const char* kTypeName = "float8_e4m3fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fnuz";
  static constexpr const char* kTpDoc = "float8_e4m3fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'G';
};

template <>
struct CustomFloatTraits<float8_e5m2> {
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  static constexpr char kNumPy1DescrType = '5';
};

template <>
struct CustomFloatTraits<float8_e5m2fnuz> {
  static constexpr const char* kTypeName = "float8_e5m2fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2fnuz";
  static constexpr const char* kTpDoc = "float8_e5m2fnuz floating-point values";
  static constexpr char kNumPy1DescrType = 'C';
};

template <>
struct CustomFloatTraits<float6_e2m3fn> {
  static constexpr const char* kTypeName = "float6_e2m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e2m3fn";
  static constexpr const char* kTpDoc = "float6_e2m3fn floating-point values";
  static constexpr char kNumPy1DescrType = '8';
};

template <>
struct CustomFloatTraits<float6_e3m2fn> {
  static constexpr const char* kTypeName = "float6_e3m2fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e3m2fn";
  static constexpr const char* kTpDoc = "float6_e3m2fn floating-point values";
  static constexpr char kNumPy1DescrType = '9';
};

template <>
struct CustomFloatTraits<float4_e2m1fn> {
  static constexpr const char* kTypeName = "float4_e2m1fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float4_e2m1fn";
  static constexpr const char* kTpDoc = "float4_e2m1fn floating-point values";
  static constexpr char kNumPy1DescrType = '0';
};

template <>
struct CustomFloatTraits<float8_e8m0fnu> {
  static constexpr const char* kTypeName = "float8_e8m0fnu";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e8m0fnu";
  static constexpr const char* kTpDoc = "float8_e8m0fnu floating-point values";
  static constexpr char kNumPy1DescrType = 'W';
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
  static PyArray_Descr* npy_descr;
  static PyArray_DTypeMeta* dtype_meta;
  static PyArray_ArrFuncs numpy_1_arr_funcs;
  static PyArray_DescrProto numpy_1_descr_proto;
};

template <typename T>
struct DtypeTraits<T, std::enable_if_t<is_custom_float_v<T>>> {
  static int Dtype() { return CustomFloatType<T>::Dtype(); }
};

bool RegisterFloatDtypes(PyObject* numpy, bool use_new_dtype_api);

}  // namespace ml_dtypes

#endif  // ML_DTYPES_FLOATS_H_
