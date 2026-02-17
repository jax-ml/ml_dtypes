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

#ifndef ML_DTYPES_COMPLEX_H_
#define ML_DTYPES_COMPLEX_H_

#include <type_traits>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/include/complex_types.h"

namespace ml_dtypes {

template <typename T>
struct CustomComplexTraits {};

template <>
struct CustomComplexTraits<bcomplex32> {
  static constexpr const char* kTypeName = "bcomplex32";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.bcomplex32";
  static constexpr const char* kTpDoc =
      "complex bfloat16 floating-point values";
  static constexpr char kNumPy1DescrType = 'P';
};

template <>
struct CustomComplexTraits<complex32> {
  static constexpr const char* kTypeName = "complex32";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.complex32";
  static constexpr const char* kTpDoc = "complex half floating-point values";
  static constexpr char kNumPy1DescrType = 'O';
};

template <typename T, typename = void>
struct is_custom_complex : std::false_type {};

template <typename T>
struct is_custom_complex<
    T, std::void_t<decltype(CustomComplexTraits<T>::kTypeName)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_custom_complex_v = is_custom_complex<T>::value;

template <typename T>
struct CustomComplexType {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyObject* type_ptr;

  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];
  static PyMethodDef methods[];
  static PyGetSetDef getset[];
  static PyArray_Descr* npy_descr;
  static PyArray_DTypeMeta* dtype_meta;
  static PyArray_ArrFuncs numpy_1_arr_funcs;
  static PyArray_DescrProto numpy_1_descr_proto;
};

template <typename T>
struct DtypeTraits<T, std::enable_if_t<is_custom_complex_v<T>>> {
  static int Dtype() { return CustomComplexType<T>::Dtype(); }
};

bool RegisterComplexDtypes(PyObject* numpy, bool use_new_dtype_api);

}  // namespace ml_dtypes

#endif  // ML_DTYPES_COMPLEX_H_
