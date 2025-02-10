/* Copyright 2023 The ml_dtypes Authors

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

#ifndef ML_DTYPES_INT4_NUMPY_H_
#define ML_DTYPES_INT4_NUMPY_H_

#include <limits>
#include <type_traits>

// Must be included first
// clang-format off
#include "ml_dtypes/_src/numpy.h"
// clang-format on

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"  // NOLINT
#include "ml_dtypes/_src/ufuncs.h"  // NOLINT
#include "ml_dtypes/include/intn.h"

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace ml_dtypes {

constexpr char kOutOfRange[] = "out of range value cannot be converted to int4";

template <typename T>
struct IntNTypeDescriptor {
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
int IntNTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* IntNTypeDescriptor<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto IntNTypeDescriptor<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* IntNTypeDescriptor<T>::npy_descr = nullptr;

// Representation of a Python custom integer object.
template <typename T>
struct PyIntN {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyIntN.
template <typename T>
bool PyIntN_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

// Extracts the value of a PyIntN object.
template <typename T>
T PyIntN_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyIntN<T>*>(object)->value;
}

template <typename T>
bool PyIntN_Value(PyObject* arg, T* output) {
  if (PyIntN_Check<T>(arg)) {
    *output = PyIntN_Value_Unchecked<T>(arg);
    return true;
  }
  return false;
}

// Constructs a PyIntN object from PyIntN<T>::T.
template <typename T>
Safe_PyObjectPtr PyIntN_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(PyObject_New(PyObject, type));
  PyIntN<T>* p = reinterpret_cast<PyIntN<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a reduced integer value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToIntN(PyObject* arg, T* output) {
  if (PyIntN_Check<T>(arg)) {
    *output = PyIntN_Value_Unchecked<T>(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    if (std::isnan(d)) {
      PyErr_SetString(PyExc_ValueError, "cannot convert float NaN to integer");
    }
    if (std::isinf(d)) {
      PyErr_SetString(PyExc_OverflowError,
                      "cannot convert float infinity to integer");
    }
    if (d < static_cast<double>(T::lowest()) ||
        d > static_cast<double>(T::highest())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
    }
    *output = T(d);
    return true;
  }
  if (PyLong_Check(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(l);
    return true;
  }
  if (PyArray_IsScalar(arg, Integer)) {
    int64_t v;
    PyArray_CastScalarToCtype(arg, &v, PyArray_DescrFromType(NPY_INT64));

    if (!(std::numeric_limits<T>::min() <= v &&
          v <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(v);
    return true;
  }
  auto floating_conversion = [&](auto type) -> bool {
    decltype(type) f;
    PyArray_ScalarAsCtype(arg, &f);
    if (!(std::numeric_limits<T>::min() <= f &&
          f <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(static_cast<::int8_t>(f));
    return true;
  };
  if (PyArray_IsScalar(arg, Half)) {
    return floating_conversion(Eigen::half{});
  }
  if (PyArray_IsScalar(arg, Float)) {
    return floating_conversion(float{});
  }
  if (PyArray_IsScalar(arg, Double)) {
    return floating_conversion(double{});
  }
  if (PyArray_IsScalar(arg, LongDouble)) {
    using ld = long double;
    return floating_conversion(ld{});
  }
  return false;
}

// Constructs a new PyIntN.
template <typename T>
PyObject* PyIntN_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
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
  if (PyIntN_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToIntN<T>(arg, &value)) {
    return PyIntN_FromValue<T>(value).release();
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
    Safe_PyObjectPtr bytes(PyUnicode_AsUTF8String(arg));
    if (!bytes) {
      return nullptr;
    }
    PyObject* f =
        PyLong_FromString(PyBytes_AsString(bytes.get()), /*end=*/nullptr,
                          /*base=*/0);
    if (!f) {
      return nullptr;
    }
    if (CastToIntN<T>(f, &value)) {
      return PyIntN_FromValue<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %R", Py_TYPE(arg));
  return nullptr;
}

template <typename T>
PyObject* PyIntN_nb_float(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

template <typename T>
PyObject* PyIntN_nb_int(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  return PyLong_FromLong(static_cast<long>(x));  // NOLINT
}

template <typename T>
PyObject* PyIntN_nb_negative(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  return PyIntN_FromValue<T>(-x).release();
}

template <typename T>
PyObject* PyIntN_nb_positive(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  return PyIntN_FromValue<T>(x).release();
}

template <typename T>
PyObject* PyIntN_nb_add(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    return PyIntN_FromValue<T>(x + y).release();
  }
  auto array_nb_add =
      reinterpret_cast<binaryfunc>(PyType_GetSlot(&PyArray_Type, Py_nb_add));
  return array_nb_add(a, b);
}

template <typename T>
PyObject* PyIntN_nb_subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    return PyIntN_FromValue<T>(x - y).release();
  }
  auto array_nb_subtract = reinterpret_cast<binaryfunc>(
      PyType_GetSlot(&PyArray_Type, Py_nb_subtract));
  return array_nb_subtract(a, b);
}

template <typename T>
PyObject* PyIntN_nb_multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    return PyIntN_FromValue<T>(x * y).release();
  }
  auto array_nb_multiply = reinterpret_cast<binaryfunc>(
      PyType_GetSlot(&PyArray_Type, Py_nb_multiply));
  return array_nb_multiply(a, b);
}

template <typename T>
PyObject* PyIntN_nb_remainder(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    T v = x % y;
    if (v != 0 && ((v < 0) != (y < 0))) {
      v = v + y;
    }
    return PyIntN_FromValue<T>(v).release();
  }
  auto array_nb_remainder = reinterpret_cast<binaryfunc>(
      PyType_GetSlot(&PyArray_Type, Py_nb_remainder));
  return array_nb_remainder(a, b);
}

template <typename T>
PyObject* PyIntN_nb_floor_divide(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    T v = x / y;
    if (((x > 0) != (y > 0)) && x % y != 0) {
      v = v - T(1);
    }
    return PyIntN_FromValue<T>(v).release();
  }
  auto array_nb_floor_divide = reinterpret_cast<binaryfunc>(
      PyType_GetSlot(&PyArray_Type, Py_nb_floor_divide));
  return array_nb_floor_divide(a, b);
}

// Implementation of repr() for PyIntN.
template <typename T>
PyObject* PyIntN_Repr(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

// Implementation of str() for PyIntN.
template <typename T>
PyObject* PyIntN_Str(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

// Hash function for PyIntN.
template <typename T>
Py_hash_t PyIntN_Hash(PyObject* self) {
  T x = PyIntN_Value_Unchecked<T>(self);
  // Hash functions must not return -1.
  return static_cast<int>(x) == -1 ? static_cast<Py_hash_t>(-2)
                                   : static_cast<Py_hash_t>(x);
}

// Comparisons on PyIntNs.
template <typename T>
PyObject* PyIntN_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!PyIntN_Value<T>(a, &x) || !PyIntN_Value<T>(b, &y)) {
    auto generic_tp_richcompare = reinterpret_cast<richcmpfunc>(
        PyType_GetSlot(&PyGenericArrType_Type, Py_tp_richcompare));
    return generic_tp_richcompare(a, b, op);
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

template <typename T>
PyType_Slot IntNTypeDescriptor<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyIntN_tp_new<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyIntN_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyIntN_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyIntN_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyIntN_RichCompare<T>)},
    {Py_nb_add, reinterpret_cast<void*>(PyIntN_nb_add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyIntN_nb_subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyIntN_nb_multiply<T>)},
    {Py_nb_remainder, reinterpret_cast<void*>(PyIntN_nb_remainder<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyIntN_nb_negative<T>)},
    {Py_nb_positive, reinterpret_cast<void*>(PyIntN_nb_positive<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyIntN_nb_int<T>)},
    {Py_nb_float, reinterpret_cast<void*>(PyIntN_nb_float<T>)},
    {Py_nb_floor_divide, reinterpret_cast<void*>(PyIntN_nb_floor_divide<T>)},
    {0, nullptr},
};

template <typename T>
PyType_Spec IntNTypeDescriptor<T>::type_spec = {
    /*.name=*/TypeDescriptor<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyIntN<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/IntNTypeDescriptor<T>::type_slots,
};

// Numpy support
template <typename T>
PyArray_ArrFuncs IntNTypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetIntNDescrProto() {
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
      /*f=*/&IntNTypeDescriptor<T>::arr_funcs,
      /*metadata=*/nullptr,
      /*c_metadata=*/nullptr,
      /*hash=*/-1,  // -1 means "not computed yet".
  };
}

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyIntN_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyLong_FromLong(static_cast<int>(x));
}

template <typename T>
int NPyIntN_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (!CastToIntN<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %R", Py_TYPE(item));
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
int NPyIntN_Compare(const void* a, const void* b, void* arr) {
  T x;
  memcpy(&x, a, sizeof(T));

  T y;
  memcpy(&y, b, sizeof(T));
  int fy(y);
  int fx(x);
  if (fx < fy) {
    return -1;
  }
  if (fy < fx) {
    return 1;
  }
  return 0;
}

template <typename T>
void NPyIntN_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                       npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (dstride == sizeof(T) && sstride == sizeof(T)) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
    }
  }
}

template <typename T>
void NPyIntN_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(T));
}

template <typename T>
npy_bool NPyIntN_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<T>(0);
}

template <typename T>
int NPyIntN_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  T* const buffer = reinterpret_cast<T*>(buffer_raw);
  const int start(buffer[0]);
  const int delta = static_cast<int>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<T>(start + i * delta);
  }
  return 0;
}

template <typename T>
void NPyIntN_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2, void* op,
                     npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  int acc = 0;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += static_cast<int>(*b1) * static_cast<int>(*b2);
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
int NPyIntN_CompareFunc(const void* v1, const void* v2, void* arr) {
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
int NPyIntN_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  // Start with a max_val of INT_MIN, this results in the first iteration
  // preferring bdata[0].
  int max_val = std::numeric_limits<int>::lowest();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<int>(bdata[i]) > max_val) {
      max_val = static_cast<int>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

template <typename T>
int NPyIntN_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  int min_val = std::numeric_limits<int>::max();
  // Start with a min_val of INT_MAX, this results in the first iteration
  // preferring bdata[0].
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<int>(bdata[i]) < min_val) {
      min_val = static_cast<int>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

template <typename T>
int CastToInt(T value) {
  if constexpr (is_complex_v<T>) {
    return CastToInt(value.real());
  } else {
    static_assert(std::numeric_limits<T>::is_specialized);
    if constexpr (!std::numeric_limits<T>::is_integer) {
      if (std::isnan(value) || std::isinf(value) ||
          value < std::numeric_limits<int>::lowest() ||
          value > std::numeric_limits<int>::max()) {
        return 0;
      }
    }
    return static_cast<int>(value);
  }
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void IntegerCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<typename TypeDescriptor<To>::T>(
        static_cast<To>(CastToInt(from[i])));
  }
}

// Registers a cast between T (a reduced float) and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterCustomIntCast(int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               IntegerCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(IntNTypeDescriptor<T>::npy_descr, numpy_type,
                               IntegerCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterIntNCasts() {
  if (!RegisterCustomIntCast<T, Eigen::half>(NPY_HALF)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCustomIntCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_INT8,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_INT16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_INT32,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_INT64,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  if (!std::numeric_limits<T>::is_signed) {
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_UINT8,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_UINT16,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_UINT32,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_UINT64,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr, NPY_HALF,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
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

  return true;
}

template <typename T>
bool RegisterIntNUFuncs(PyObject* numpy) {
  bool ok = RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
            RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                                  "subtract") &&
            RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                                  "multiply") &&
            RegisterUFunc<UFunc<ufuncs::FloorDivide<T>, T, T, T>, T>(
                numpy, "floor_divide") &&
            RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(numpy,
                                                                   "remainder");

  return ok;
}

template <typename T>
bool RegisterIntNDtype(PyObject* numpy) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
  // TODO(jakevdp): it would be better to inherit from PyNumberArrType or
  // PyIntegerArrType, but this breaks some assumptions made by NumPy, because
  // dtype.kind='V' is then interpreted as a 'void' type in some contexts.
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&IntNTypeDescriptor<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  TypeDescriptor<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "__module__",
                             module.get()) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs& arr_funcs = IntNTypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyIntN_GetItem<T>;
  arr_funcs.setitem = NPyIntN_SetItem<T>;
  arr_funcs.compare = NPyIntN_Compare<T>;
  arr_funcs.copyswapn = NPyIntN_CopySwapN<T>;
  arr_funcs.copyswap = NPyIntN_CopySwap<T>;
  arr_funcs.nonzero = NPyIntN_NonZero<T>;
  arr_funcs.fill = NPyIntN_Fill<T>;
  arr_funcs.dotfunc = NPyIntN_DotFunc<T>;
  arr_funcs.compare = NPyIntN_CompareFunc<T>;
  arr_funcs.argmax = NPyIntN_ArgMaxFunc<T>;
  arr_funcs.argmin = NPyIntN_ArgMinFunc<T>;

  // This is messy, but that's because the NumPy 2.0 API transition is messy.
  // Before 2.0, NumPy assumes we'll keep the descriptor passed in to
  // RegisterDataType alive, because it stores its pointer.
  // After 2.0, the proto and descriptor types diverge, and NumPy allocates
  // and manages the lifetime of the descriptor itself.
  PyArray_DescrProto& descr_proto = IntNTypeDescriptor<T>::npy_descr_proto;
  descr_proto = GetIntNDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) {
    return false;
  }
  // TODO(phawkins): We intentionally leak the pointer to the descriptor.
  // Implement a better module destructor to handle this.
  IntNTypeDescriptor<T>::npy_descr =
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
          reinterpret_cast<PyObject*>(IntNTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterIntNCasts<T>() && RegisterIntNUFuncs<T>(numpy);
}

}  // namespace ml_dtypes

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif

#endif  // ML_DTYPES_INT4_NUMPY_H_
