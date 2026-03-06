/* Copyright 2022 The ml_dtypes Authors

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

#ifndef ML_DTYPES_CUSTOM_FLOAT_H_
#define ML_DTYPES_CUSTOM_FLOAT_H_

// Must be included first
// clang-format off
#include "ml_dtypes/_src/numpy.h" // NOLINT
// clang-format on

// Support utilities for adding custom floating-point dtypes to TensorFlow,
// such as bfloat16, and float8_*.

#include <array>    // NOLINT
#include <cmath>    // NOLINT
#include <limits>   // NOLINT
#include <locale>   // NOLINT
#include <memory>   // NOLINT
#include <sstream>  // NOLINT
#include <vector>   // NOLINT
// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"  // NOLINT
#include "ml_dtypes/_src/ufuncs.h"  // NOLINT

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

namespace ml_dtypes {

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

  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];
  static PyArray_Descr* npy_descr;
  static PyArray_DTypeMeta* dtype_meta;
  // Temporarily disable array functions, descr, and proto for NumPy 2 testing
#if 0
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
#endif
};

template <typename T>
int CustomFloatType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomFloatType<T>::type_ptr = nullptr;
template <typename T>
PyArray_Descr* CustomFloatType<T>::npy_descr = nullptr;
template <typename T>
PyArray_DTypeMeta* CustomFloatType<T>::dtype_meta = nullptr;
#if 0
template <typename T>
PyArray_DescrProto CustomFloatType<T>::npy_descr_proto;
#endif

// Representation of a Python custom float object.
template <typename T>
struct PyCustomFloat {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyCustomFloat.
template <typename T>
bool PyCustomFloat_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

// Extracts the value of a PyCustomFloat object.
template <typename T>
T PyCustomFloat_CustomFloat(PyObject* object) {
  return reinterpret_cast<PyCustomFloat<T>*>(object)->value;
}

// Constructs a PyCustomFloat object from PyCustomFloat<T>::T.
template <typename T>
Safe_PyObjectPtr PyCustomFloat_FromT(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyCustomFloat<T>* p = reinterpret_cast<PyCustomFloat<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a reduced float value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToCustomFloat(PyObject* arg, T* output) {
  if (PyCustomFloat_Check<T>(arg)) {
    *output = PyCustomFloat_CustomFloat<T>(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = T(d);
    return true;
  }
  if (PyLong_Check(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = T(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Generic)) {
    // Allow conversion from any NumPy scalar if conversion to complex float
    // is defined.
    // NOTE: Should use `PyArray_Pack` with NumPy>=2, which is better and may
    // make even more conversions (ie. casts) work. (May want to use new dtypes
    // then also.) (If a limitation is found, could do this already on NumPy 2
    // at runtime.)
    float c;
    PyArray_Descr* f_descr = PyArray_DescrFromType(NPY_FLOAT32);
    // Similar to our code, NumPy accepts the array to be NULL here.
    // TODO(phawkins): check for overflow
    PyDataType_GetArrFuncs(f_descr)->setitem(arg, &c, NULL);
    Py_DECREF(f_descr);
    *output = T(c);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      Py_INCREF(CustomFloatType<T>::npy_descr);
      ref =
          make_safe(PyArray_CastToType(arr, CustomFloatType<T>::npy_descr, 0));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<T*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

template <typename T>
bool SafeCastToCustomFloat(PyObject* arg, T* output) {
  if (PyCustomFloat_Check<T>(arg)) {
    *output = PyCustomFloat_CustomFloat<T>(arg);
    return true;
  }
  return false;
}

// Converts a PyReduceFloat into a PyFloat.
template <typename T>
PyObject* PyCustomFloat_Float(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  return PyFloat_FromDouble(static_cast<double>(static_cast<float>(x)));
}

// Converts a PyReduceFloat into a PyInt.
template <typename T>
PyObject* PyCustomFloat_Int(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  long y = static_cast<long>(static_cast<float>(x));  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Negative(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  return PyCustomFloat_FromT<T>(-x).release();
}

template <typename T>
PyObject* PyCustomFloat_Add(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyCustomFloat_Subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyCustomFloat_Multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyCustomFloat_TrueDivide(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Constructs a new PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_New(PyTypeObject* type, PyObject* args,
                            PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "expected number as argument to %s constructor",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (PyCustomFloat_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToCustomFloat<T>(arg, &value)) {
    return PyCustomFloat_FromT<T>(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      Py_INCREF(CustomFloatType<T>::npy_descr);
      return PyArray_CastToType(arr, CustomFloatType<T>::npy_descr, 0);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse float from string, then cast to T.
    PyObject* f = PyFloat_FromString(arg);
    if (CastToCustomFloat<T>(f, &value)) {
      return PyCustomFloat_FromT<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

// Comparisons on PyCustomFloats.
template <typename T>
PyObject* PyCustomFloat_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  bool a_is_custom = SafeCastToCustomFloat<T>(a, &x);
  bool b_is_custom = SafeCastToCustomFloat<T>(b, &y);
  if (a_is_custom && b_is_custom) {
    bool result;
    switch (op) {
      case Py_LT:
        result = x < y;
        break;
      case Py_LE:
        result = x <= y;
        break;
      case Py_EQ:
        result = x == y;
        break;
      case Py_NE:
        result = x != y;
        break;
      case Py_GT:
        result = x > y;
        break;
      case Py_GE:
        result = x >= y;
        break;
      default:
        PyErr_SetString(PyExc_ValueError, "Invalid op type");
        return nullptr;
    }
    PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
  }

  // Fallback to double comparison for float/int scalars.
  // This avoids issues where NumPy might cast the operand to the custom type
  // (potentially losing precision or saturating) before comparing.
  // E.g. e8m0 has no zero, so 0.0 -> NaN (or min).
  // e8m0(min) != 0.0 should be True, but if 0.0 -> e8m0(min), it becomes Eq.
  double val_a, val_b;
  bool a_is_double = false;
  bool b_is_double = false;

  if (a_is_custom) {
    val_a = static_cast<double>(static_cast<float>(x));
    a_is_double = true;
  } else if (PyFloat_Check(a)) {
    val_a = PyFloat_AsDouble(a);
    a_is_double = true;
  } else if (PyLong_Check(a)) {
    val_a = PyLong_AsDouble(a);
    if (PyErr_Occurred()) return nullptr;
    a_is_double = true;
  }

  if (b_is_custom) {
    val_b = static_cast<double>(static_cast<float>(y));
    b_is_double = true;
  } else if (PyFloat_Check(b)) {
    val_b = PyFloat_AsDouble(b);
    b_is_double = true;
  } else if (PyLong_Check(b)) {
    val_b = PyLong_AsDouble(b);
    if (PyErr_Occurred()) return nullptr;
    b_is_double = true;
  }

  if (a_is_double && b_is_double) {
    if (std::isnan(val_a) || std::isnan(val_b)) {
      if (op == Py_NE) {
        PyArrayScalar_RETURN_BOOL_FROM_LONG(1);
      }
      PyArrayScalar_RETURN_BOOL_FROM_LONG(0);
    }
    bool result;
    switch (op) {
      case Py_LT:
        result = val_a < val_b;
        break;
      case Py_LE:
        result = val_a <= val_b;
        break;
      case Py_EQ:
        result = val_a == val_b;
        break;
      case Py_NE:
        result = val_a != val_b;
        break;
      case Py_GT:
        result = val_a > val_b;
        break;
      case Py_GE:
        result = val_a >= val_b;
        break;
      default:
        return nullptr;
    }
    PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
  }

  return PyGenericArrType_Type.tp_richcompare(a, b, op);
}

// Implementation of repr() for PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Repr(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  float f = static_cast<float>(x);
  std::ostringstream s;
  s << (std::isnan(f) ? std::abs(f) : f);
  return PyUnicode_FromString(s.str().c_str());
}

// Implementation of str() for PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Str(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  float f = static_cast<float>(x);
  std::ostringstream s;
  s << (std::isnan(f) ? std::abs(f) : f);
  return PyUnicode_FromString(s.str().c_str());
}

// _Py_HashDouble changed its prototype for Python 3.10 so we use an overload to
// handle the two possibilities.
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t HashImpl(Py_hash_t (*hash_double)(PyObject*, double),
                          PyObject* self, double value) {
  return hash_double(self, value);
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t HashImpl(Py_hash_t (*hash_double)(double), PyObject* self,
                          double value) {
  return hash_double(value);
}

// Hash function for PyCustomFloat.
template <typename T>
Py_hash_t PyCustomFloat_Hash(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  return HashImpl(&_Py_HashDouble, self, static_cast<double>(x));
}

template <typename T>
PyType_Slot CustomFloatType<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyCustomFloat_New<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyCustomFloat_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyCustomFloat_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyCustomFloat_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyCustomFloat_RichCompare<T>)},
    {Py_nb_add, reinterpret_cast<void*>(PyCustomFloat_Add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyCustomFloat_Subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyCustomFloat_Multiply<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyCustomFloat_Negative<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyCustomFloat_Int<T>)},
    {Py_nb_float, reinterpret_cast<void*>(PyCustomFloat_Float<T>)},
    {Py_nb_true_divide, reinterpret_cast<void*>(PyCustomFloat_TrueDivide<T>)},
    {0, nullptr},
};

template <typename T>
PyType_Spec CustomFloatType<T>::type_spec = {
    /*.name=*/TypeDescriptor<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyCustomFloat<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/CustomFloatType<T>::type_slots,
};

// Numpy support
#if 0
template <typename T>
PyArray_ArrFuncs CustomFloatType<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetCustomFloatDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr)
      /*typeobj=*/nullptr,  // Filled in later
      /*kind=*/TypeDescriptor<T>::kNpyDescrKind,
      /*type=*/TypeDescriptor<T>::kNpyDescrType,
      /*byteorder=*/TypeDescriptor<T>::kNpyDescrByteorder,
      /*flags=*/NPY_USE_SETITEM,
      /*type_num=*/0,
      /*elsize=*/sizeof(T),
      /*alignment=*/alignof(T),
      /*subarray=*/nullptr,
      /*fields=*/nullptr,
      /*names=*/nullptr,
      /*f=*/&CustomFloatType<T>::arr_funcs,
      /*metadata=*/nullptr,
      /*c_metadata=*/nullptr,
      /*hash=*/-1,  // -1 means "not computed yet".
  };
}
#endif

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyCustomFloat_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyFloat_FromDouble(static_cast<float>(x));
}

template <typename T>
int NPyCustomFloat_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (!CastToCustomFloat<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
int NPyCustomFloat_Compare(const void* a, const void* b, void* arr) {
  T x;
  memcpy(&x, a, sizeof(T));

  T y;
  memcpy(&y, b, sizeof(T));
  float fy(y);
  float fx(x);

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

template <typename T>
void NPyCustomFloat_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                              npy_intp sstride, npy_intp n, int swap,
                              void* arr) {
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t),
                "Not supported");
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);

  if (src) {
    if (swap && sizeof(T) == sizeof(int16_t)) {
      for (npy_intp i = 0; i < n; i++) {
        char* r = dst + dstride * i;
        memcpy(r, src + sstride * i, sizeof(T));
        ml_dtypes::ByteSwap16(r);
      }
    } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (npy_intp i = 0; i < n; i++) {
        memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
      }
    }
  } else {
    // In-place swap when src is NULL
    if (swap && sizeof(T) == sizeof(int16_t)) {
      for (npy_intp i = 0; i < n; i++) {
        char* r = dst + dstride * i;
        ml_dtypes::ByteSwap16(r);
      }
    }
  }
}

template <typename T>
void NPyCustomFloat_CopySwap(void* dst, void* src, int swap, void* arr) {
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t),
                "Not supported");

  if (src) {
    memcpy(dst, src, sizeof(T));
  }

  if (swap && sizeof(T) == sizeof(int16_t)) {
    ml_dtypes::ByteSwap16(dst);
  }
}

template <typename T>
npy_bool NPyCustomFloat_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<T>(0);
}

template <typename T>
int NPyCustomFloat_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  T* const buffer = reinterpret_cast<T*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<T>(start + i * delta);
  }
  return 0;
}

template <typename T>
void NPyCustomFloat_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                            void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
int NPyCustomFloat_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

template <typename T>
int NPyCustomFloat_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                              void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  // Start with a max_val of NaN, this results in the first iteration preferring
  // bdata[0].
  float max_val = std::numeric_limits<float>::quiet_NaN();
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "max".
    if (!(static_cast<float>(bdata[i]) <= max_val)) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
      // NumPy stops at the first NaN.
      if (Eigen::numext::isnan(max_val)) {
        break;
      }
    }
  }
  return 0;
}

template <typename T>
int NPyCustomFloat_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                              void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  float min_val = std::numeric_limits<float>::quiet_NaN();
  // Start with a min_val of NaN, this results in the first iteration preferring
  // bdata[0].
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "min".
    if (!(static_cast<float>(bdata[i]) >= min_val)) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
      // NumPy stops at the first NaN.
      if (Eigen::numext::isnan(min_val)) {
        break;
      }
    }
  }
  return 0;
}

template <typename T>
float CastToFloat(T value) {
  if constexpr (ml_dtypes::is_complex_v<T>) {
    return CastToFloat(value.real());
  } else {
    return static_cast<float>(value);
  }
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
int PyCustomFloatCastLoop(PyArrayMethod_Context* context, char* const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  using FromT = typename ml_dtypes::TypeDescriptor<From>::T;
  using ToT = typename ml_dtypes::TypeDescriptor<To>::T;
  for (npy_intp i = 0; i < N; i++) {
    FromT f;
    memcpy(&f, in, sizeof(FromT));
    ToT t =
        static_cast<ToT>(static_cast<To>(CastToFloat(static_cast<From>(f))));
    memcpy(out, &t, sizeof(ToT));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

template <typename From, typename To>
struct CustomFloatCastSpec {
  static PyType_Slot slots[3];
  static PyArray_DTypeMeta* dtypes[2];
  static PyArrayMethod_Spec spec;
  // Initialize assigns the NumPy types for this Cast.
  // 'from_type' and 'to_type' are the target TypeDescriptors. We use a boolean
  // 'from_is_custom' to determine whether 'from_type' represents the new custom
  // DType being initialized.
  static bool Initialize(int from_type, int to_type, bool from_is_custom,
                         bool to_is_custom) {
    if (from_is_custom) {
      dtypes[0] = nullptr;
    } else {
      PyArray_Descr* d = PyArray_DescrFromType(from_type);
      if (!d) return false;
      dtypes[0] = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(d));
      Py_DECREF(d);
    }
    if (to_is_custom) {
      dtypes[1] = nullptr;
    } else {
      PyArray_Descr* d = PyArray_DescrFromType(to_type);
      if (!d) return false;
      dtypes[1] = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(d));
      Py_DECREF(d);
    }
    return true;
  }
};

template <typename From, typename To>
PyType_Slot CustomFloatCastSpec<From, To>::slots[3] = {
    {NPY_METH_strided_loop,
     reinterpret_cast<void*>(PyCustomFloatCastLoop<From, To>)},
    {NPY_METH_unaligned_strided_loop,
     reinterpret_cast<void*>(PyCustomFloatCastLoop<From, To>)},
    {0, nullptr}};

template <typename From, typename To>
PyArray_DTypeMeta* CustomFloatCastSpec<From, To>::dtypes[2] = {nullptr,
                                                               nullptr};

template <typename From, typename To>
PyArrayMethod_Spec CustomFloatCastSpec<From, To>::spec = {
    /*name=*/"customfloat_cast",
    /*nin=*/1,
    /*nout=*/1,
    /*casting=*/NPY_UNSAFE_CASTING,
    /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*dtypes=*/dtypes,
    /*slots=*/slots,
};

// Registers a cast between T (a reduced float) and type 'OtherT'.
template <typename T, typename OtherT>
bool AddCustomFloatCast(int numpy_type, NPY_CASTING to_safety,
                        NPY_CASTING from_safety,
                        std::vector<PyArrayMethod_Spec*>& casts) {
  if (!CustomFloatCastSpec<T, OtherT>::Initialize(
          ml_dtypes::TypeDescriptor<T>::Dtype(), numpy_type,
          /*from_is_custom=*/true, /*to_is_custom=*/false))
    return false;
  CustomFloatCastSpec<T, OtherT>::spec.casting = to_safety;
  casts.push_back(&CustomFloatCastSpec<T, OtherT>::spec);

  if (!CustomFloatCastSpec<OtherT, T>::Initialize(
          numpy_type, ml_dtypes::TypeDescriptor<T>::Dtype(),
          /*from_is_custom=*/false, /*to_is_custom=*/true))
    return false;
  CustomFloatCastSpec<OtherT, T>::spec.casting = from_safety;
  casts.push_back(&CustomFloatCastSpec<OtherT, T>::spec);
  return true;
}

template <typename T>
bool GetFloatCasts(std::vector<PyArrayMethod_Spec*>& casts) {
  if (!AddCustomFloatCast<T, half>(NPY_HALF, NPY_SAME_KIND_CASTING,
                                   NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, float>(NPY_FLOAT, NPY_SAFE_CASTING,
                                    NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, double>(NPY_DOUBLE, NPY_SAFE_CASTING,
                                     NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, long double>(NPY_LONGDOUBLE, NPY_SAFE_CASTING,
                                          NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, bool>(NPY_BOOL, NPY_UNSAFE_CASTING,
                                   NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned char>(NPY_UBYTE, NPY_UNSAFE_CASTING,
                                            NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned short>(NPY_USHORT, NPY_UNSAFE_CASTING,
                                             NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned int>(NPY_UINT, NPY_UNSAFE_CASTING,
                                           NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned long>(NPY_ULONG, NPY_UNSAFE_CASTING,
                                            NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned long long>(
          NPY_ULONGLONG, NPY_UNSAFE_CASTING, NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, signed char>(NPY_BYTE, NPY_UNSAFE_CASTING,
                                          NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, short>(NPY_SHORT, NPY_UNSAFE_CASTING,
                                    NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, int>(NPY_INT, NPY_UNSAFE_CASTING,
                                  NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, long>(NPY_LONG, NPY_UNSAFE_CASTING,
                                   NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, long long>(NPY_LONGLONG, NPY_UNSAFE_CASTING,
                                        NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<float>>(NPY_CFLOAT, NPY_SAFE_CASTING,
                                                  NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<double>>(
          NPY_CDOUBLE, NPY_SAFE_CASTING, NPY_SAME_KIND_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE, NPY_SAFE_CASTING, NPY_SAME_KIND_CASTING, casts))
    return false;
  return true;
}

template <typename T>
bool RegisterFloatUFuncs(PyObject* numpy) {
  bool ok =
      RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
      RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                            "subtract") &&
      RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                            "multiply") &&
      RegisterUFunc<UFunc<ufuncs::LogAddExp<T>, T, T, T>, T>(numpy,
                                                             "logaddexp") &&
      RegisterUFunc<UFunc<ufuncs::LogAddExp2<T>, T, T, T>, T>(numpy,
                                                              "logaddexp2") &&
      RegisterUFunc<UFunc<ufuncs::Negative<T>, T, T>, T>(numpy, "negative") &&
      RegisterUFunc<UFunc<ufuncs::Positive<T>, T, T>, T>(numpy, "positive") &&
      RegisterUFunc<UFunc<ufuncs::TrueDivide<T>, T, T, T>, T>(numpy,
                                                              "true_divide") &&
      RegisterUFunc<UFunc<ufuncs::FloorDivide<T>, T, T, T>, T>(
          numpy, "floor_divide") &&
      RegisterUFunc<UFunc<ufuncs::Power<T>, T, T, T>, T>(numpy, "power") &&
      RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(numpy,
                                                             "remainder") &&
      RegisterUFunc<UFunc<ufuncs::Fmod<T>, T, T, T>, T>(numpy, "fmod") &&
      RegisterUFunc<UFunc2<ufuncs::Divmod<T>, T, T, T, T>, T>(numpy,
                                                              "divmod") &&
      RegisterUFunc<UFunc<ufuncs::Abs<T>, T, T>, T>(numpy, "absolute") &&
      RegisterUFunc<UFunc<ufuncs::Abs<T>, T, T>, T>(numpy, "fabs") &&
      RegisterUFunc<UFunc<ufuncs::Rint<T>, T, T>, T>(numpy, "rint") &&
      RegisterUFunc<UFunc<ufuncs::Sign<T>, T, T>, T>(numpy, "sign") &&
      RegisterUFunc<UFunc<ufuncs::Heaviside<T>, T, T, T>, T>(numpy,
                                                             "heaviside") &&
      RegisterUFunc<UFunc<ufuncs::Conjugate<T>, T, T>, T>(numpy, "conjugate") &&
      RegisterUFunc<UFunc<ufuncs::Exp<T>, T, T>, T>(numpy, "exp") &&
      RegisterUFunc<UFunc<ufuncs::Exp2<T>, T, T>, T>(numpy, "exp2") &&
      RegisterUFunc<UFunc<ufuncs::Expm1<T>, T, T>, T>(numpy, "expm1") &&
      RegisterUFunc<UFunc<ufuncs::Log<T>, T, T>, T>(numpy, "log") &&
      RegisterUFunc<UFunc<ufuncs::Log2<T>, T, T>, T>(numpy, "log2") &&
      RegisterUFunc<UFunc<ufuncs::Log10<T>, T, T>, T>(numpy, "log10") &&
      RegisterUFunc<UFunc<ufuncs::Log1p<T>, T, T>, T>(numpy, "log1p") &&
      RegisterUFunc<UFunc<ufuncs::Sqrt<T>, T, T>, T>(numpy, "sqrt") &&
      RegisterUFunc<UFunc<ufuncs::Square<T>, T, T>, T>(numpy, "square") &&
      RegisterUFunc<UFunc<ufuncs::Cbrt<T>, T, T>, T>(numpy, "cbrt") &&
      RegisterUFunc<UFunc<ufuncs::Reciprocal<T>, T, T>, T>(numpy,
                                                           "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UFunc<ufuncs::Sin<T>, T, T>, T>(numpy, "sin") &&
      RegisterUFunc<UFunc<ufuncs::Cos<T>, T, T>, T>(numpy, "cos") &&
      RegisterUFunc<UFunc<ufuncs::Tan<T>, T, T>, T>(numpy, "tan") &&
      RegisterUFunc<UFunc<ufuncs::Arcsin<T>, T, T>, T>(numpy, "arcsin") &&
      RegisterUFunc<UFunc<ufuncs::Arccos<T>, T, T>, T>(numpy, "arccos") &&
      RegisterUFunc<UFunc<ufuncs::Arctan<T>, T, T>, T>(numpy, "arctan") &&
      RegisterUFunc<UFunc<ufuncs::Arctan2<T>, T, T, T>, T>(numpy, "arctan2") &&
      RegisterUFunc<UFunc<ufuncs::Hypot<T>, T, T, T>, T>(numpy, "hypot") &&
      RegisterUFunc<UFunc<ufuncs::Sinh<T>, T, T>, T>(numpy, "sinh") &&
      RegisterUFunc<UFunc<ufuncs::Cosh<T>, T, T>, T>(numpy, "cosh") &&
      RegisterUFunc<UFunc<ufuncs::Tanh<T>, T, T>, T>(numpy, "tanh") &&
      RegisterUFunc<UFunc<ufuncs::Arcsinh<T>, T, T>, T>(numpy, "arcsinh") &&
      RegisterUFunc<UFunc<ufuncs::Arccosh<T>, T, T>, T>(numpy, "arccosh") &&
      RegisterUFunc<UFunc<ufuncs::Arctanh<T>, T, T>, T>(numpy, "arctanh") &&
      RegisterUFunc<UFunc<ufuncs::Deg2rad<T>, T, T>, T>(numpy, "deg2rad") &&
      RegisterUFunc<UFunc<ufuncs::Rad2deg<T>, T, T>, T>(numpy, "rad2deg") &&

      // Comparison functions
      RegisterUFunc<UFunc<ufuncs::Eq<T>, bool, T, T>, T>(numpy, "equal") &&
      RegisterUFunc<UFunc<ufuncs::Ne<T>, bool, T, T>, T>(numpy, "not_equal") &&
      RegisterUFunc<UFunc<ufuncs::Lt<T>, bool, T, T>, T>(numpy, "less") &&
      RegisterUFunc<UFunc<ufuncs::Gt<T>, bool, T, T>, T>(numpy, "greater") &&
      RegisterUFunc<UFunc<ufuncs::Le<T>, bool, T, T>, T>(numpy, "less_equal") &&
      RegisterUFunc<UFunc<ufuncs::Ge<T>, bool, T, T>, T>(numpy,
                                                         "greater_equal") &&
      RegisterUFunc<UFunc<ufuncs::Maximum<T>, T, T, T>, T>(numpy, "maximum") &&
      RegisterUFunc<UFunc<ufuncs::Minimum<T>, T, T, T>, T>(numpy, "minimum") &&
      RegisterUFunc<UFunc<ufuncs::Fmax<T>, T, T, T>, T>(numpy, "fmax") &&
      RegisterUFunc<UFunc<ufuncs::Fmin<T>, T, T, T>, T>(numpy, "fmin") &&
      RegisterUFunc<UFunc<ufuncs::LogicalAnd<T>, bool, T, T>, T>(
          numpy, "logical_and") &&
      RegisterUFunc<UFunc<ufuncs::LogicalOr<T>, bool, T, T>, T>(numpy,
                                                                "logical_or") &&
      RegisterUFunc<UFunc<ufuncs::LogicalXor<T>, bool, T, T>, T>(
          numpy, "logical_xor") &&
      RegisterUFunc<UFunc<ufuncs::LogicalNot<T>, bool, T>, T>(numpy,
                                                              "logical_not") &&

      // Floating point functions
      RegisterUFunc<UFunc<ufuncs::IsFinite<T>, bool, T>, T>(numpy,
                                                            "isfinite") &&
      RegisterUFunc<UFunc<ufuncs::IsInf<T>, bool, T>, T>(numpy, "isinf") &&
      RegisterUFunc<UFunc<ufuncs::IsNan<T>, bool, T>, T>(numpy, "isnan") &&
      RegisterUFunc<UFunc<ufuncs::SignBit<T>, bool, T>, T>(numpy, "signbit") &&
      RegisterUFunc<UFunc<ufuncs::CopySign<T>, T, T, T>, T>(numpy,
                                                            "copysign") &&
      RegisterUFunc<UFunc2<ufuncs::Modf<T>, T, T, T>, T>(numpy, "modf") &&
      RegisterUFunc<UFunc<ufuncs::Ldexp<T>, T, T, int>, T>(numpy, "ldexp") &&
      RegisterUFunc<UFunc2<ufuncs::Frexp<T>, T, int, T>, T>(numpy, "frexp") &&
      RegisterUFunc<UFunc<ufuncs::Floor<T>, T, T>, T>(numpy, "floor") &&
      RegisterUFunc<UFunc<ufuncs::Ceil<T>, T, T>, T>(numpy, "ceil") &&
      RegisterUFunc<UFunc<ufuncs::Trunc<T>, T, T>, T>(numpy, "trunc") &&
      RegisterUFunc<UFunc<ufuncs::NextAfter<T>, T, T, T>, T>(numpy,
                                                             "nextafter") &&
      RegisterUFunc<UFunc<ufuncs::Spacing<T>, T, T>, T>(numpy, "spacing");

  return ok;
}
template <typename T>
PyObject* PyCustomFloatDType_Repr(PyObject* self) {
  std::string repr = std::string("dtype(") + TypeDescriptor<T>::kTypeName + ")";
  return PyUnicode_FromString(repr.c_str());
}

template <typename T>
PyObject* PyCustomFloatDType_name_get(PyObject* self, void* closure) {
  return PyUnicode_FromString(TypeDescriptor<T>::kTypeName);
}

template <typename T>
PyArray_DTypeMeta* PyCustomFloatDType_CommonDType(PyArray_DTypeMeta* cls,
                                                  PyArray_DTypeMeta* other) {
  if (cls == other) {
    Py_INCREF(cls);
    return cls;
  }

  int next_largest_typenum = NPY_FLOAT32;
  if constexpr (sizeof(T) == 1) {
    next_largest_typenum = NPY_FLOAT16;
  } else if constexpr (sizeof(T) == 2) {
    next_largest_typenum = NPY_FLOAT32;
  } else if constexpr (sizeof(T) == 4) {
    next_largest_typenum = NPY_FLOAT64;
  } else {
    next_largest_typenum = NPY_LONGDOUBLE;
  }

  PyArray_Descr* descr1 = PyArray_DescrFromType(next_largest_typenum);
  if (!descr1) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    return reinterpret_cast<PyArray_DTypeMeta*>(Py_NotImplemented);
  }

  PyArray_DTypeMeta* dtype1 =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(descr1));
  PyArray_DTypeMeta* dtypes[2] = {dtype1, other};
  PyArray_DTypeMeta* out_meta = PyArray_PromoteDTypeSequence(2, dtypes);
  Py_DECREF(descr1);

  if (!out_meta) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    return reinterpret_cast<PyArray_DTypeMeta*>(Py_NotImplemented);
  }

  return out_meta;
}

template <typename T>
PyObject* PyCustomFloatDType_Str(PyObject* self) {
  return PyUnicode_FromString(TypeDescriptor<T>::kTypeName);
}

template <typename T>
PyObject* PyCustomFloatDType_GetItem(PyArray_Descr* descr, char* data) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyFloat_FromDouble(static_cast<float>(x));
}

template <typename T>
int PyCustomFloatDType_SetItem(PyArray_Descr* descr, PyObject* item,
                               char* data) {
  T x;
  if (!CastToCustomFloat<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

static inline PyArray_Descr* PyCustomFloatDType_EnsureCanonical(
    PyArray_Descr* dtype) {
  Py_INCREF(dtype);
  return dtype;
}

template <typename T>
int PyCustomFloatDType_to_CustomFloatDType_resolve_descriptors(
    struct PyArrayMethodObject_tag* method, PyArray_DTypeMeta* dtypes[2],
    PyArray_Descr* given_descrs[2], PyArray_Descr* loop_descrs[2],
    npy_intp* view_offset) {
  loop_descrs[0] = given_descrs[0];
  Py_INCREF(loop_descrs[0]);
  if (given_descrs[1] == nullptr) {
    loop_descrs[1] = given_descrs[0];
  } else {
    loop_descrs[1] = given_descrs[1];
  }
  Py_INCREF(loop_descrs[1]);
  *view_offset = 0;
  return NPY_NO_CASTING;
}

template <typename T>
int PyCustomFloatDType_to_CustomFloatDType_CastLoop(
    PyArrayMethod_Context* context, char* const data[],
    npy_intp const dimensions[], npy_intp const strides[],
    NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  for (npy_intp i = 0; i < N; i++) {
    memcpy(out, in, sizeof(T));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

template <typename T>
static PyObject* PyCustomFloatDType_Reduce(PyObject* self) {
  PyObject* type_obj = reinterpret_cast<PyObject*>(TypeDescriptor<T>::type_ptr);
  PyObject* tuple = PyTuple_Pack(1, type_obj);
  PyObject* numpy = PyImport_ImportModule("numpy");
  PyObject* dtype_callable = PyObject_GetAttrString(numpy, "dtype");
  PyObject* res = Py_BuildValue("(OO)", dtype_callable, tuple);
  Py_DECREF(dtype_callable);
  Py_DECREF(numpy);
  Py_DECREF(tuple);
  return res;
}

template <typename T>
static PyObject* PyCustomFloatDType_New(PyTypeObject* type, PyObject* args,
                                        PyObject* kwds) {
  PyObject* obj = PyArrayDescr_Type.tp_new(type, args, kwds);
  if (obj != nullptr) {
    PyArray_Descr* descr = reinterpret_cast<PyArray_Descr*>(obj);
    descr->elsize = sizeof(T);
    descr->alignment = alignof(T);
    descr->kind = TypeDescriptor<T>::kNpyDescrKind;
    descr->type = TypeDescriptor<T>::kNpyDescrType;
    descr->byteorder = TypeDescriptor<T>::kNpyDescrByteorder;
    descr->flags = NPY_USE_SETITEM;
  }
  return obj;
}

template <typename T>
bool RegisterFloatDtype(
    PyObject* numpy,
    void (*add_custom_casts)(std::vector<PyArrayMethod_Spec*>&) = nullptr) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
  // TODO(jakevdp): it would be better to inherit from PyNumberArrType or
  // PyFloatingArrType, but this breaks some assumptions made by NumPy, because
  // dtype.kind='V' is then interpreted as a 'void' type in some contexts.
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomFloatType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  TypeDescriptor<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
#if 0
  PyArray_ArrFuncs& arr_funcs = CustomFloatType<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyCustomFloat_GetItem<T>;
  arr_funcs.setitem = NPyCustomFloat_SetItem<T>;
  arr_funcs.compare = NPyCustomFloat_Compare<T>;
  arr_funcs.copyswapn = NPyCustomFloat_CopySwapN<T>;
  arr_funcs.copyswap = NPyCustomFloat_CopySwap<T>;
  arr_funcs.nonzero = NPyCustomFloat_NonZero<T>;
  arr_funcs.fill = NPyCustomFloat_Fill<T>;
  arr_funcs.dotfunc = NPyCustomFloat_DotFunc<T>;
  arr_funcs.compare = NPyCustomFloat_CompareFunc<T>;
  arr_funcs.argmax = NPyCustomFloat_ArgMaxFunc<T>;
  arr_funcs.argmin = NPyCustomFloat_ArgMinFunc<T>;

  // This is messy, but that's because the NumPy 2.0 API transition is messy.
  // Before 2.0, NumPy assumes we'll keep the descriptor passed in to
  // RegisterDataType alive, because it stores its pointer.
  // After 2.0, the proto and descriptor types diverge, and NumPy allocates
  // and manages the lifetime of the descriptor itself.
  PyArray_DescrProto& descr_proto = CustomFloatType<T>::npy_descr_proto;
  descr_proto = GetCustomFloatDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) {
    return false;
  }

  // TODO(phawkins): We intentionally leak the pointer to the descriptor.
  // Implement a better module destructor to handle this.
  CustomFloatType<T>::npy_descr =
      PyArray_DescrFromType(TypeDescriptor<T>::npy_type);
#endif

#ifndef NPY_DT_PyArray_ArrFuncs_copyswapn
#define NPY_DT_PyArray_ArrFuncs_copyswapn (3 + (1 << 11))
#endif

#ifndef NPY_DT_PyArray_ArrFuncs_copyswap
#define NPY_DT_PyArray_ArrFuncs_copyswap (4 + (1 << 11))
#endif

  static PyType_Slot slots[] = {
      {NPY_DT_getitem, reinterpret_cast<void*>(PyCustomFloatDType_GetItem<T>)},
      {NPY_DT_setitem, reinterpret_cast<void*>(PyCustomFloatDType_SetItem<T>)},
      {NPY_DT_ensure_canonical,
       reinterpret_cast<void*>(PyCustomFloatDType_EnsureCanonical)},
      {NPY_DT_PyArray_ArrFuncs_copyswap,
       reinterpret_cast<void*>(NPyCustomFloat_CopySwap<T>)},
      {NPY_DT_PyArray_ArrFuncs_copyswapn,
       reinterpret_cast<void*>(NPyCustomFloat_CopySwapN<T>)},
      {NPY_DT_PyArray_ArrFuncs_compare,
       reinterpret_cast<void*>(NPyCustomFloat_CompareFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_nonzero,
       reinterpret_cast<void*>(NPyCustomFloat_NonZero<T>)},
      {NPY_DT_PyArray_ArrFuncs_fill,
       reinterpret_cast<void*>(NPyCustomFloat_Fill<T>)},
      {NPY_DT_PyArray_ArrFuncs_dotfunc,
       reinterpret_cast<void*>(NPyCustomFloat_DotFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_argmax,
       reinterpret_cast<void*>(NPyCustomFloat_ArgMaxFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_argmin,
       reinterpret_cast<void*>(NPyCustomFloat_ArgMinFunc<T>)},
      {NPY_DT_common_dtype,
       reinterpret_cast<void*>(PyCustomFloatDType_CommonDType<T>)},
      {0, nullptr}};

  static PyType_Slot cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(
           PyCustomFloatDType_to_CustomFloatDType_resolve_descriptors<T>)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void*>(
           PyCustomFloatDType_to_CustomFloatDType_CastLoop<T>)},
      {NPY_METH_strided_loop,
       reinterpret_cast<void*>(
           PyCustomFloatDType_to_CustomFloatDType_CastLoop<T>)},
      {0, nullptr}};

  static PyArray_DTypeMeta* cast_dtypes[2] = {nullptr, nullptr};

  static PyArrayMethod_Spec cast_spec = {
      /*name=*/"customfloat_to_customfloat_cast",
      /*nin=*/1,
      /*nout=*/1,
      /*casting=*/NPY_NO_CASTING,
      /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
      /*dtypes=*/cast_dtypes,
      /*slots=*/cast_slots,
  };

  static std::vector<PyArrayMethod_Spec*> cast_specs;
  static bool casts_initialized = [&]() {
    cast_specs.push_back(&cast_spec);
    bool ok = GetFloatCasts<T>(cast_specs);
    if (ok && add_custom_casts) {
      add_custom_casts(cast_specs);
    }
    cast_specs.push_back(nullptr);
    return ok;
  }();

  if (!casts_initialized) return false;

  static PyArrayDTypeMeta_Spec spec = {
      /*typeobj=*/reinterpret_cast<PyTypeObject*>(type),
      /*flags=*/0,
      /*casts=*/cast_specs.data(),
      /*slots=*/slots,
      /*baseclass=*/nullptr};

  if (!CustomFloatType<T>::dtype_meta) {
    CustomFloatType<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
  }
  PyArray_DTypeMeta* dtype_meta = CustomFloatType<T>::dtype_meta;
  if (!dtype_meta) return false;

  PyTypeObject* tm = reinterpret_cast<PyTypeObject*>(dtype_meta);
  Py_SET_TYPE(tm, &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(tm, 1);
  tm->tp_name = TypeDescriptor<T>::kQualifiedTypeName;
  tm->tp_basicsize = sizeof(PyArray_Descr);
  tm->tp_base = &PyArrayDescr_Type;
  tm->tp_new = PyCustomFloatDType_New<T>;
  tm->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  tm->tp_repr = PyCustomFloatDType_Repr<T>;
  tm->tp_str = PyCustomFloatDType_Str<T>;

  static PyGetSetDef dtype_getset[] = {
      {const_cast<char*>("name"), PyCustomFloatDType_name_get<T>, nullptr,
       nullptr, nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr}};
  tm->tp_getset = dtype_getset;

  static PyMethodDef dtype_methods[] = {
      {const_cast<char*>("__reduce__"),
       reinterpret_cast<PyCFunction>(PyCustomFloatDType_Reduce<T>), METH_NOARGS,
       nullptr},
      {nullptr, nullptr, 0, nullptr}};
  tm->tp_methods = dtype_methods;

  if (PyType_Ready(tm) < 0) {
    return false;
  }

  if (PyArrayInitDTypeMeta_FromSpec(dtype_meta, &spec) < 0) {
    return false;
  }

  TypeDescriptor<T>::npy_type = dtype_meta->type_num;

  Safe_PyObjectPtr dtype_func =
      make_safe(PyObject_GetAttrString(numpy, "dtype"));
  if (!dtype_func) return false;
  Safe_PyObjectPtr descr_obj = make_safe(PyObject_CallFunctionObjArgs(
      dtype_func.get(), TypeDescriptor<T>::type_ptr, nullptr));
  if (!descr_obj) return false;
  CustomFloatType<T>::npy_descr =
      reinterpret_cast<PyArray_Descr*>(descr_obj.release());

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), TypeDescriptor<T>::kTypeName,
                           TypeDescriptor<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          TypeDescriptor<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomFloatType<T>::npy_descr)) < 0) {
    return false;
  }

  // RegisterFloatCasts<T>();
  if (!RegisterFloatUFuncs<T>(numpy)) {
    return false;
  }
  return true;
}

}  // namespace ml_dtypes

// LEGACY
#if 0
#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif
#endif

#endif  // ML_DTYPES_CUSTOM_FLOAT_H_
