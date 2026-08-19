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

#ifndef ML_DTYPES_INTS_H_
#define ML_DTYPES_INTS_H_

#include <type_traits>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/include/intn.h"

namespace ml_dtypes {

template <typename T>
struct CustomIntTraits {};

template <>
struct CustomIntTraits<int1> {
  static constexpr const char* kTypeName = "int1";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int1";
  static constexpr const char* kTpDoc = "int1 integer values";
  static constexpr char kNumPy1DescrType = 'e';
};

template <>
struct CustomIntTraits<uint1> {
  static constexpr const char* kTypeName = "uint1";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint1";
  static constexpr const char* kTpDoc = "uint1 integer values";
  static constexpr char kNumPy1DescrType = 'E';
};

template <>
struct CustomIntTraits<int2> {
  static constexpr const char* kTypeName = "int2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int2";
  static constexpr const char* kTpDoc = "int2 integer values";
  static constexpr char kNumPy1DescrType = 'c';
};

template <>
struct CustomIntTraits<uint2> {
  static constexpr const char* kTypeName = "uint2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint2";
  static constexpr const char* kTpDoc = "uint2 integer values";
  static constexpr char kNumPy1DescrType = 'C';
};

template <>
struct CustomIntTraits<int4> {
  static constexpr const char* kTypeName = "int4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int4";
  static constexpr const char* kTpDoc = "int4 integer values";
  static constexpr char kNumPy1DescrType = 'a';
};

template <>
struct CustomIntTraits<uint4> {
  static constexpr const char* kTypeName = "uint4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint4";
  static constexpr const char* kTpDoc = "uint4 integer values";
  static constexpr char kNumPy1DescrType = 'A';
};

template <typename T, typename = void>
struct is_intn : std::false_type {};

template <typename T>
struct is_intn<T, std::void_t<decltype(CustomIntTraits<T>::kTypeName)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_intn_v = is_intn<T>::value;

template <typename T>
struct CustomIntType {
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
struct DtypeTraits<T, std::enable_if_t<is_intn_v<T>>> {
  static int Dtype() { return CustomIntType<T>::Dtype(); }
};

bool RegisterCustomInts(PyObject* numpy);

}  // namespace ml_dtypes

#endif  // ML_DTYPES_INTS_H_
