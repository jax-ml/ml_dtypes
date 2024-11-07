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
#include "_src/numpy.h" // NOLINT
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
#include "_src/common.h"  // NOLINT
#include "_src/ufuncs.h"  // NOLINT

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

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
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;
};

template <typename T>
int CustomFloatType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomFloatType<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto CustomFloatType<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* CustomFloatType<T>::npy_descr = nullptr;

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
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, LongDouble)) {
    long double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      ref = make_safe(PyArray_Cast(arr, TypeDescriptor<T>::Dtype()));
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
      return PyArray_Cast(arr, TypeDescriptor<T>::Dtype());
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
  if (!SafeCastToCustomFloat<T>(a, &x) || !SafeCastToCustomFloat<T>(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
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
      /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_SETITEM,
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

inline void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
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
  if (!src) {
    return;
  }
  if (swap && sizeof(T) == sizeof(int16_t)) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(T));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
    }
  }
}

template <typename T>
void NPyCustomFloat_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(T));
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t),
                "Not supported");
  if (swap && sizeof(T) == sizeof(int16_t)) {
    ByteSwap16(dst);
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
  if constexpr (is_complex_v<T>) {
    return CastToFloat(value.real());
  } else {
    return static_cast<float>(value);
  }
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<typename TypeDescriptor<To>::T>(
        static_cast<To>(CastToFloat(from[i])));
  }
}

// Registers a cast between T (a reduced float) and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterCustomFloatCast(int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               NPyCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(CustomFloatType<T>::npy_descr, numpy_type,
                               NPyCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterFloatCasts() {
  if (!RegisterCustomFloatCast<T, Eigen::half>(NPY_HALF)) {
    return false;
  }

  if (!RegisterCustomFloatCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCustomFloatCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_FLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_DOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_LONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_CFLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_CDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_CLONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  // Safe casts to T from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_UBYTE),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BYTE),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }

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
      RegisterUFunc<UFunc<ufuncs::TrueDivide<T>, T, T, T>, T>(numpy,
                                                              "divide") &&
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
      RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(numpy, "mod") &&
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
bool RegisterFloatDtype(PyObject* numpy) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
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

  return RegisterFloatCasts<T>() && RegisterFloatUFuncs<T>(numpy);
}

}  // namespace ml_dtypes

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif

#endif  // ML_DTYPES_CUSTOM_FLOAT_H_
