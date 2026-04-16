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
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING

// Must be included first
// clang-format off
#include "ml_dtypes/_src/numpy.h" //NOLINT
// clang-format on

#include <array>    // NOLINT
#include <cmath>    // NOLINT
#include <cstdint>  // NOLINT
#include <limits>   // NOLINT
#include <locale>   // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "Eigen/Core"
#include "ml_dtypes/_src/custom_complex.h"
#include "ml_dtypes/_src/custom_float.h"
#include "ml_dtypes/_src/intn_numpy.h"
#include "ml_dtypes/include/float8.h"
#include "ml_dtypes/include/intn.h"
#include "ml_dtypes/include/mxfloat.h"

namespace ml_dtypes {

template <>
struct TypeDescriptor<bfloat16> : CustomFloatType<bfloat16> {
  typedef bfloat16 T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bfloat16";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.bfloat16";
  static constexpr const char* kTpDoc = "bfloat16 floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e3m4> : CustomFloatType<float8_e3m4> {
  typedef float8_e3m4 T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e3m4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e3m4";
  static constexpr const char* kTpDoc = "float8_e3m4 floating-point values";
  static constexpr char kNpyDescrKind = 'f';  // float
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';  // Native byte order
};

template <>
struct TypeDescriptor<float8_e4m3> : CustomFloatType<float8_e4m3> {
  typedef float8_e4m3 T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3";
  static constexpr const char* kTpDoc = "float8_e4m3 floating-point values";
  static constexpr char kNpyDescrKind = 'f';       // float
  static constexpr char kNpyDescrType = '?';       // '4' is reserved for e4m3fn
  static constexpr char kNpyDescrByteorder = '=';  // Native byte order
};

template <>
struct TypeDescriptor<float8_e4m3b11fnuz>
    : CustomFloatType<float8_e4m3b11fnuz> {
  typedef float8_e4m3b11fnuz T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3b11fnuz";
  static constexpr const char* kQualifiedTypeName =
      "ml_dtypes.float8_e4m3b11fnuz";
  static constexpr const char* kTpDoc =
      "float8_e4m3b11fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fn> : CustomFloatType<float8_e4m3fn> {
  typedef float8_e4m3fn T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fnuz> : CustomFloatType<float8_e4m3fnuz> {
  typedef float8_e4m3fnuz T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e4m3fnuz";
  static constexpr const char* kTpDoc = "float8_e4m3fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2> : CustomFloatType<float8_e5m2> {
  typedef float8_e5m2 T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2fnuz> : CustomFloatType<float8_e5m2fnuz> {
  typedef float8_e5m2fnuz T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2fnuz";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e5m2fnuz";
  static constexpr const char* kTpDoc = "float8_e5m2fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float6_e2m3fn> : CustomFloatType<float6_e2m3fn> {
  typedef float6_e2m3fn T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float6_e2m3fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e2m3fn";
  static constexpr const char* kTpDoc = "float6_e2m3fn floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float6_e3m2fn> : CustomFloatType<float6_e3m2fn> {
  typedef float6_e3m2fn T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float6_e3m2fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float6_e3m2fn";
  static constexpr const char* kTpDoc = "float6_e3m2fn floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float4_e2m1fn> : CustomFloatType<float4_e2m1fn> {
  typedef float4_e2m1fn T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float4_e2m1fn";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float4_e2m1fn";
  static constexpr const char* kTpDoc = "float4_e2m1fn floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e8m0fnu> : CustomFloatType<float8_e8m0fnu> {
  typedef float8_e8m0fnu T;
  typedef float builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e8m0fnu";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.float8_e8m0fnu";
  static constexpr const char* kTpDoc = "float8_e8m0fnu floating-point values";
  static constexpr char kNpyDescrKind = 'f';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int1> : IntNTypeDescriptor<int1> {
  typedef int1 T;
  typedef int8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int1";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int1";
  static constexpr const char* kTpDoc = "int1 integer values";
  static constexpr char kNpyDescrKind = 'i';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint1> : IntNTypeDescriptor<uint1> {
  typedef uint1 T;
  typedef uint8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint1";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint1";
  static constexpr const char* kTpDoc = "uint1 integer values";
  static constexpr char kNpyDescrKind = 'u';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int2> : IntNTypeDescriptor<int2> {
  typedef int2 T;
  typedef int8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int2";
  static constexpr const char* kTpDoc = "int2 integer values";
  static constexpr char kNpyDescrKind = 'i';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint2> : IntNTypeDescriptor<uint2> {
  typedef uint2 T;
  typedef uint8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint2";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint2";
  static constexpr const char* kTpDoc = "uint2 integer values";
  static constexpr char kNpyDescrKind = 'u';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int4> : IntNTypeDescriptor<int4> {
  typedef int4 T;
  typedef int8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.int4";
  static constexpr const char* kTpDoc = "int4 integer values";
  static constexpr char kNpyDescrKind = 'i';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint4> : IntNTypeDescriptor<uint4> {
  typedef uint4 T;
  typedef uint8_t builtin_type;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint4";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.uint4";
  static constexpr const char* kTpDoc = "uint4 integer values";
  static constexpr char kNpyDescrKind = 'u';
  static constexpr char kNpyDescrType = '?';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bcomplex32> : CustomComplexType<bcomplex32> {
  typedef bcomplex32 T;
  typedef std::complex<float> builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bcomplex32";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.bcomplex32";
  static constexpr const char* kTpDoc =
      "complex bfloat16 floating-point values";
  // See also bfloat16, the kind argument is tricky to choose well.
  static constexpr char kNpyDescrKind = 'c';  // TODO(seberg): better name?
  static constexpr char kNpyDescrType = '?';  // TODO(seberg): better name?
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<complex32> : CustomComplexType<complex32> {
  typedef complex32 T;
  typedef std::complex<float> builtin_type;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "complex32";
  static constexpr const char* kQualifiedTypeName = "ml_dtypes.complex32";
  static constexpr const char* kTpDoc = "complex half floating-point values";
  // See also bfloat16. `E` type char is used for bfloat16 unfortunately.
  static constexpr char kNpyDescrKind = 'c';  // TODO(seberg): better name?
  static constexpr char kNpyDescrType = '?';  // TODO(seberg): better name?
  static constexpr char kNpyDescrByteorder = '=';
};

namespace {

// Performs a NumPy array cast from type 'From' to 'To' via `Via`.
template <typename From, typename To, typename Via>
void PyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
            void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);

  if constexpr (is_complex_v<From> && !is_complex_v<To>) {
    if (GiveComplexWarningNoGIL() < 0) {
      return;
    }
    for (npy_intp i = 0; i < n; ++i) {
      to[i] = static_cast<To>(static_cast<Via>(from[i].real()));
    }
  } else {
    for (npy_intp i = 0; i < n; ++i) {
      to[i] = static_cast<To>(static_cast<Via>(from[i]));
    }
  }
}

template <typename T>
void PreallocateDTypeMeta() {
  if (!CustomFloatType<T>::dtype_meta) {
    CustomFloatType<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
  }
}

template <typename T>
void PreallocateAll() {
  PreallocateDTypeMeta<T>();
}

template <typename T, typename U, typename... Args>
void PreallocateAll() {
  PreallocateDTypeMeta<T>();
  PreallocateAll<U, Args...>();
}

template <typename T, typename U>
void AddCustomToCustomCastSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  CustomFloatCastSpec<T, U>::dtypes[0] = CustomFloatType<T>::dtype_meta;
  CustomFloatCastSpec<T, U>::dtypes[1] = CustomFloatType<U>::dtype_meta;
  casts.push_back(&CustomFloatCastSpec<T, U>::spec);

  CustomFloatCastSpec<U, T>::dtypes[0] = CustomFloatType<U>::dtype_meta;
  CustomFloatCastSpec<U, T>::dtypes[1] = CustomFloatType<T>::dtype_meta;
  casts.push_back(&CustomFloatCastSpec<U, T>::spec);
}

template <typename T>
void AddTwoWayFloatCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {}

template <typename T, typename U, typename... Args>
void AddTwoWayFloatCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  AddCustomToCustomCastSpec<T, U>(casts);
  if constexpr (sizeof...(Args) > 0) {
    AddTwoWayFloatCastsSpec<T, Args...>(casts);
  }
}

template <typename From, typename To>
int PyCrossTypeCastLoop(PyArrayMethod_Context* context, char* const data[],
                        npy_intp const dimensions[], npy_intp const strides[],
                        NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];

  for (npy_intp i = 0; i < N; i++) {
    From f;
    memcpy(&f, in, sizeof(From));
    To t;
    if constexpr (is_complex_v<From> && !is_complex_v<To>) {
      t = static_cast<To>(static_cast<float>(f.real()));
    } else if constexpr (!is_complex_v<From> && is_complex_v<To>) {
      t = To(static_cast<float>(f));
    } else if constexpr (is_complex_v<From> && is_complex_v<To>) {
      t = To(static_cast<std::complex<float>>(f));
    } else {
      t = static_cast<To>(static_cast<float>(f));
    }
    memcpy(out, &t, sizeof(To));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

template <typename From, typename To>
struct GenericCastSpec {
  static PyArray_DTypeMeta* dtypes[2];
  static PyType_Slot slots[3];
  static PyArrayMethod_Spec spec;
};

template <typename From, typename To>
PyType_Slot GenericCastSpec<From, To>::slots[3] = {
    {NPY_METH_strided_loop,
     reinterpret_cast<void*>(PyCrossTypeCastLoop<From, To>)},
    {NPY_METH_unaligned_strided_loop,
     reinterpret_cast<void*>(PyCrossTypeCastLoop<From, To>)},
    {0, nullptr}};

template <typename From, typename To>
PyArray_DTypeMeta* GenericCastSpec<From, To>::dtypes[2] = {nullptr, nullptr};

template <typename From, typename To>
PyArrayMethod_Spec GenericCastSpec<From, To>::spec = {
    /*name=*/"cross_type_cast",
    /*nin=*/1,
    /*nout=*/1,
    /*casting=*/NPY_UNSAFE_CASTING,
    /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*dtypes=*/GenericCastSpec<From, To>::dtypes,
    /*slots=*/GenericCastSpec<From, To>::slots,
};

template <typename T, typename U>
void AddCrossTypeCastSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  if (!TypeDescriptor<U>::dtype_meta) return;

  GenericCastSpec<T, U>::dtypes[0] = nullptr;
  GenericCastSpec<T, U>::dtypes[1] = TypeDescriptor<U>::dtype_meta;
  casts.push_back(&GenericCastSpec<T, U>::spec);

  GenericCastSpec<U, T>::dtypes[0] = TypeDescriptor<U>::dtype_meta;
  GenericCastSpec<U, T>::dtypes[1] = nullptr;
  casts.push_back(&GenericCastSpec<U, T>::spec);
}

template <typename T>
void AddTwoWayCrossTypeCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {}

template <typename T, typename U, typename... Args>
void AddTwoWayCrossTypeCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  AddCrossTypeCastSpec<T, U>(casts);
  if constexpr (sizeof...(Args) > 0) {
    AddTwoWayCrossTypeCastsSpec<T, Args...>(casts);
  }
}

template <typename T, typename U>
NPY_CASTING GetIntCastingSafety() {
  bool t_signed = std::numeric_limits<T>::is_signed;
  bool u_signed = std::numeric_limits<U>::is_signed;
  int t_bits = T::bits;
  int u_bits = U::bits;

  if (t_signed == u_signed) {
    return t_bits <= u_bits ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
  }
  if (t_signed && !u_signed) {
    return NPY_UNSAFE_CASTING;
  }
  // !t_signed && u_signed
  return t_bits < u_bits ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
}

template <typename T, typename U>
void AddCustomToCustomIntCastSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  // Use CustomIntCastSpec from intn_numpy.h
  // We need to set dtype_meta for both T and U
  CustomIntCastSpec<T, U>::dtypes[0] = IntNTypeDescriptor<T>::dtype_meta;
  CustomIntCastSpec<T, U>::dtypes[1] = IntNTypeDescriptor<U>::dtype_meta;
  CustomIntCastSpec<T, U>::spec.casting = GetIntCastingSafety<T, U>();
  casts.push_back(&CustomIntCastSpec<T, U>::spec);

  CustomIntCastSpec<U, T>::dtypes[0] = IntNTypeDescriptor<U>::dtype_meta;
  CustomIntCastSpec<U, T>::dtypes[1] = IntNTypeDescriptor<T>::dtype_meta;
  CustomIntCastSpec<U, T>::spec.casting = GetIntCastingSafety<U, T>();
  casts.push_back(&CustomIntCastSpec<U, T>::spec);
}

template <typename T>
void AddTwoWayIntCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {}

template <typename T, typename U, typename... Args>
void AddTwoWayIntCastsSpec(std::vector<PyArrayMethod_Spec*>& casts) {
  AddCustomToCustomIntCastSpec<T, U>(casts);
  if constexpr (sizeof...(Args) > 0) {
    AddTwoWayIntCastsSpec<T, Args...>(casts);
  }
}

// Initialize type attribute in the module object.
template <typename T>
bool InitModuleType(PyObject* obj, const char* name) {
  return PyObject_SetAttrString(
             obj, name,
             reinterpret_cast<PyObject*>(TypeDescriptor<T>::type_ptr)) >= 0;
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

  Safe_PyObjectPtr exceptions;
  if (PyObject_HasAttrString(numpy.get(), "exceptions")) {
    exceptions = make_safe(PyObject_GetAttrString(numpy.get(), "exceptions"));
    if (!exceptions) {
      return false;
    }
  } else {
    exceptions = make_safe(numpy.get());  // main module holds the objects.
  }
  ComplexWarning = PyObject_GetAttrString(exceptions.get(), "ComplexWarning");
  if (!ComplexWarning) {
    return false;
  }

  PreallocateAll<bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
                 float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
                 float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>();

  auto cb_bfloat16 = [](std::vector<PyArrayMethod_Spec*>& c) {};
  if (!RegisterFloatDtype<bfloat16>(numpy.get(), cb_bfloat16)) return false;

  auto cb_float8_e3m4 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e3m4, bfloat16>(c);
  };
  if (!RegisterFloatDtype<float8_e3m4>(numpy.get(), cb_float8_e3m4))
    return false;

  auto cb_float8_e4m3 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e4m3, bfloat16, float8_e3m4>(c);
  };
  if (!RegisterFloatDtype<float8_e4m3>(numpy.get(), cb_float8_e4m3))
    return false;

  auto cb_float8_e4m3b11fnuz = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e4m3b11fnuz, bfloat16, float8_e3m4,
                            float8_e4m3>(c);
  };
  if (!RegisterFloatDtype<float8_e4m3b11fnuz>(numpy.get(),
                                              cb_float8_e4m3b11fnuz))
    return false;

  auto cb_float8_e4m3fn = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e4m3fn, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz>(c);
  };
  if (!RegisterFloatDtype<float8_e4m3fn>(numpy.get(), cb_float8_e4m3fn))
    return false;

  auto cb_float8_e4m3fnuz = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e4m3fnuz, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn>(c);
  };
  if (!RegisterFloatDtype<float8_e4m3fnuz>(numpy.get(), cb_float8_e4m3fnuz))
    return false;

  auto cb_float8_e5m2 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e5m2, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz>(
        c);
  };
  if (!RegisterFloatDtype<float8_e5m2>(numpy.get(), cb_float8_e5m2))
    return false;

  auto cb_float8_e5m2fnuz = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e5m2fnuz, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz,
                            float8_e5m2>(c);
  };
  if (!RegisterFloatDtype<float8_e5m2fnuz>(numpy.get(), cb_float8_e5m2fnuz))
    return false;

  auto cb_float6_e2m3fn = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float6_e2m3fn, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz,
                            float8_e5m2, float8_e5m2fnuz>(c);
  };
  if (!RegisterFloatDtype<float6_e2m3fn>(numpy.get(), cb_float6_e2m3fn))
    return false;

  auto cb_float6_e3m2fn = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float6_e3m2fn, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz,
                            float8_e5m2, float8_e5m2fnuz, float6_e2m3fn>(c);
  };
  if (!RegisterFloatDtype<float6_e3m2fn>(numpy.get(), cb_float6_e3m2fn))
    return false;

  auto cb_float4_e2m1fn = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float4_e2m1fn, bfloat16, float8_e3m4, float8_e4m3,
                            float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz,
                            float8_e5m2, float8_e5m2fnuz, float6_e2m3fn,
                            float6_e3m2fn>(c);
  };
  if (!RegisterFloatDtype<float4_e2m1fn>(numpy.get(), cb_float4_e2m1fn))
    return false;

  auto cb_float8_e8m0fnu = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayFloatCastsSpec<float8_e8m0fnu, bfloat16>(c);
  };
  if (!RegisterFloatDtype<float8_e8m0fnu>(numpy.get(), cb_float8_e8m0fnu))
    return false;

  auto cb_int1 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayCrossTypeCastsSpec<
        int1, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<int1>(numpy.get(), cb_int1)) return false;

  auto cb_uint1 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayIntCastsSpec<uint1, int1>(c);
    AddTwoWayCrossTypeCastsSpec<
        uint1, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<uint1>(numpy.get(), cb_uint1)) return false;

  auto cb_int2 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayIntCastsSpec<int2, int1, uint1>(c);
    AddTwoWayCrossTypeCastsSpec<
        int2, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<int2>(numpy.get(), cb_int2)) return false;

  auto cb_uint2 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayIntCastsSpec<uint2, int1, uint1, int2>(c);
    AddTwoWayCrossTypeCastsSpec<
        uint2, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<uint2>(numpy.get(), cb_uint2)) return false;

  auto cb_int4 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayIntCastsSpec<int4, int1, uint1, int2, uint2>(c);
    AddTwoWayCrossTypeCastsSpec<
        int4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<int4>(numpy.get(), cb_int4)) return false;

  auto cb_uint4 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayIntCastsSpec<uint4, int1, uint1, int2, uint2, int4>(c);
    AddTwoWayCrossTypeCastsSpec<
        uint4, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu>(c);
  };
  if (!RegisterIntNDtype<uint4>(numpy.get(), cb_uint4)) return false;

  auto cb_bcomplex32 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayCrossTypeCastsSpec<
        bcomplex32, bfloat16, float8_e3m4, float8_e4m3, float8_e4m3b11fnuz,
        float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
        float6_e2m3fn, float6_e3m2fn, float4_e2m1fn, float8_e8m0fnu, int1,
        uint1, int2, uint2, int4, uint4>(c);
  };
  auto cb_complex32 = [](std::vector<PyArrayMethod_Spec*>& c) {
    AddTwoWayCrossTypeCastsSpec<
        complex32, bcomplex32, bfloat16, float8_e3m4, float8_e4m3,
        float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz, float8_e5m2,
        float8_e5m2fnuz, float6_e2m3fn, float6_e3m2fn, float4_e2m1fn,
        float8_e8m0fnu, int1, uint1, int2, uint2, int4, uint4>(c);
  };
  if (!RegisterComplexDtype<bcomplex32>(numpy.get(), cb_bcomplex32) ||
      !RegisterComplexDtype<complex32>(numpy.get(), cb_complex32)) {
    return false;
  }

  // Casts should be registered in the callbacks above or via DTypeMeta.
  return true;
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_ml_dtypes_ext",
};

// TODO(phawkins): PyMODINIT_FUNC handles visibility correctly in Python 3.9+.
// Just use PyMODINIT_FUNC after dropping Python 3.8 support.
#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C" EXPORT_SYMBOL PyObject* PyInit__ml_dtypes_ext() {
  Safe_PyObjectPtr m = make_safe(PyModule_Create(&module_def));
  if (!m) {
    return nullptr;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load _ml_dtypes_ext module.");
    }
    return nullptr;
  }

  if (!InitModuleType<float4_e2m1fn>(m.get(), "float4_e2m1fn") ||
      !InitModuleType<float6_e2m3fn>(m.get(), "float6_e2m3fn") ||
      !InitModuleType<float6_e3m2fn>(m.get(), "float6_e3m2fn") ||
      !InitModuleType<float8_e3m4>(m.get(), "float8_e3m4") ||
      !InitModuleType<float8_e4m3>(m.get(), "float8_e4m3") ||
      !InitModuleType<float8_e4m3b11fnuz>(m.get(), "float8_e4m3b11fnuz") ||
      !InitModuleType<float8_e4m3fn>(m.get(), "float8_e4m3fn") ||
      !InitModuleType<float8_e4m3fnuz>(m.get(), "float8_e4m3fnuz") ||
      !InitModuleType<float8_e5m2>(m.get(), "float8_e5m2") ||
      !InitModuleType<float8_e5m2fnuz>(m.get(), "float8_e5m2fnuz") ||
      !InitModuleType<float8_e8m0fnu>(m.get(), "float8_e8m0fnu") ||
      !InitModuleType<bfloat16>(m.get(), "bfloat16") ||
      !InitModuleType<bcomplex32>(m.get(), "bcomplex32") ||
      !InitModuleType<complex32>(m.get(), "complex32") ||
      !InitModuleType<int1>(m.get(), "int1") ||
      !InitModuleType<int2>(m.get(), "int2") ||
      !InitModuleType<int4>(m.get(), "int4") ||
      !InitModuleType<uint1>(m.get(), "uint1") ||
      !InitModuleType<uint2>(m.get(), "uint2") ||
      !InitModuleType<uint4>(m.get(), "uint4")) {
    return nullptr;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED);
#endif

  return m.release();
}
}  // namespace ml_dtypes
