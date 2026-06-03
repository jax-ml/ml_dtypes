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

  static PyArray_Descr* npy_descr;
  static PyArray_DTypeMeta* dtype_meta;
};

template <typename T>
int IntNTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* IntNTypeDescriptor<T>::type_ptr = nullptr;
template <typename T>
PyArray_Descr* IntNTypeDescriptor<T>::npy_descr = nullptr;
template <typename T>
PyArray_DTypeMeta* IntNTypeDescriptor<T>::dtype_meta = nullptr;

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
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
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
      return false;
    }
    if (std::isinf(d)) {
      PyErr_SetString(PyExc_OverflowError,
                      "cannot convert float infinity to integer");
      return false;
    }
    if (d < static_cast<double>(T::lowest()) ||
        d > static_cast<double>(T::highest())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
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
    if (!(static_cast<double>(static_cast<int64_t>(
              std::numeric_limits<T>::min())) <= static_cast<double>(f) &&
          static_cast<double>(f) <= static_cast<double>(static_cast<int64_t>(
                                        std::numeric_limits<T>::max())))) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(static_cast<::int8_t>(static_cast<int64_t>(f)));
    return true;
  };
  if (PyArray_IsScalar(arg, Half)) {
    return floating_conversion(half{});
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
      Py_INCREF(IntNTypeDescriptor<T>::npy_descr);
      return PyArray_CastToType(arr, IntNTypeDescriptor<T>::npy_descr, 0);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse float from string, then cast to T.
    PyObject* f = PyLong_FromUnicodeObject(arg, /*base=*/0);
    if (PyErr_Occurred()) {
      return nullptr;
    }
    if (CastToIntN<T>(f, &value)) {
      return PyIntN_FromValue<T>(value).release();
    }
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
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
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyIntN_nb_subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    return PyIntN_FromValue<T>(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyIntN_nb_multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (PyIntN_Value<T>(a, &x) && PyIntN_Value<T>(b, &y)) {
    return PyIntN_FromValue<T>(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
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
  return PyArray_Type.tp_as_number->nb_remainder(a, b);
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
  return PyArray_Type.tp_as_number->nb_floor_divide(a, b);
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
  double val_a, val_b;
  bool a_ok = false, b_ok = false;

  if (PyIntN_Check<T>(a)) {
    val_a = static_cast<double>(PyIntN_Value_Unchecked<T>(a));
    a_ok = true;
  } else if (PyFloat_Check(a)) {
    val_a = PyFloat_AsDouble(a);
    a_ok = true;
  } else if (PyLong_Check(a)) {
    val_a = PyLong_AsDouble(a);
    if (!PyErr_Occurred()) a_ok = true;
  }

  if (PyIntN_Check<T>(b)) {
    val_b = static_cast<double>(PyIntN_Value_Unchecked<T>(b));
    b_ok = true;
  } else if (PyFloat_Check(b)) {
    val_b = PyFloat_AsDouble(b);
    b_ok = true;
  } else if (PyLong_Check(b)) {
    val_b = PyLong_AsDouble(b);
    if (!PyErr_Occurred()) b_ok = true;
  }

  if (a_ok && b_ok) {
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

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyIntN_GetItem(PyArray_Descr* descr, char* data) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyLong_FromLong(static_cast<int>(x));
}

template <typename T>
int NPyIntN_SetItem(PyArray_Descr* descr, PyObject* item, char* data) {
  T x;
  if (!CastToIntN<T>(item, &x)) {
    if (PyErr_Occurred()) {
      return -1;
    }
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
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

  if (src) {
    if (dstride == sizeof(T) && sstride == sizeof(T)) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (npy_intp i = 0; i < n; i++) {
        memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
      }
    }
  }
  // Note: No byte swapping needed for 8-bit integer types
}

template <typename T>
void NPyIntN_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (src) {
    memcpy(dst, src, sizeof(T));
  }
  // Note: No byte swapping needed for 8-bit integer types
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

template <typename From, typename To>
struct CustomIntCastSpec {
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
      PyArray_Descr* descr = PyArray_DescrFromType(from_type);
      if (!descr) {
        fprintf(stderr, "Failed to get descr for from_type %d\n", from_type);
        PyErr_Print();
        return false;
      }
      dtypes[0] = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(descr));
      Py_DECREF(descr);
    }
    if (to_is_custom) {
      dtypes[1] = nullptr;
    } else {
      PyArray_Descr* descr = PyArray_DescrFromType(to_type);
      if (!descr) {
        fprintf(stderr, "Failed to get descr for to_type %d\n", to_type);
        PyErr_Print();
        return false;
      }
      dtypes[1] = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(descr));
      Py_DECREF(descr);
    }
    // Debug print
    // fprintf(stderr, "Initialized cast spec %d -> %d\n", from_type, to_type);
    return true;
  }
};

template <typename T>
int PyCustomIntDType_to_CustomIntDType_resolve_descriptors(
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

template <typename From, typename To>
int PyCustomIntCastLoop(PyArrayMethod_Context* context, char* const data[],
                        npy_intp const dimensions[], npy_intp const strides[],
                        NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];

  for (npy_intp i = 0; i < N; i++) {
    From f;
    memcpy(&f, in, sizeof(From));
    To t;
    if constexpr (std::is_same_v<To, std::complex<float>> ||
                  std::is_same_v<To, std::complex<double>> ||
                  std::is_same_v<To, std::complex<long double>>) {
      t = To(static_cast<typename To::value_type>(CastToInt(f)));
    } else {
      t = static_cast<To>(CastToInt(f));
    }
    memcpy(out, &t, sizeof(To));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

template <typename From, typename To>
PyType_Slot CustomIntCastSpec<From, To>::slots[3] = {
    {NPY_METH_strided_loop,
     reinterpret_cast<void*>(PyCustomIntCastLoop<From, To>)},
    {NPY_METH_unaligned_strided_loop,
     reinterpret_cast<void*>(PyCustomIntCastLoop<From, To>)},
    {0, nullptr}};

template <typename From, typename To>
PyArray_DTypeMeta* CustomIntCastSpec<From, To>::dtypes[2] = {nullptr, nullptr};

template <typename From, typename To>
PyArrayMethod_Spec CustomIntCastSpec<From, To>::spec = {
    /*name=*/"customint_cast",
    /*nin=*/1,
    /*nout=*/1,
    /*casting=*/NPY_NO_CASTING,
    /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*dtypes=*/CustomIntCastSpec<From, To>::dtypes,
    /*slots=*/CustomIntCastSpec<From, To>::slots,
};

// Registers a cast between T (a reduced int) and type 'OtherT'.
template <typename T, typename OtherT>
bool AddCustomIntCast(int numpy_type, NPY_CASTING to_safety,
                      NPY_CASTING from_safety,
                      std::vector<PyArrayMethod_Spec*>& casts) {
  if (!CustomIntCastSpec<T, OtherT>::Initialize(
          ml_dtypes::IntNTypeDescriptor<T>::Dtype(), numpy_type,
          /*from_is_custom=*/true, /*to_is_custom=*/false))
    return false;
  CustomIntCastSpec<T, OtherT>::dtypes[0] =
      ml_dtypes::IntNTypeDescriptor<T>::dtype_meta;
  CustomIntCastSpec<T, OtherT>::spec.casting = to_safety;
  casts.push_back(&CustomIntCastSpec<T, OtherT>::spec);

  if (!CustomIntCastSpec<OtherT, T>::Initialize(
          numpy_type, ml_dtypes::IntNTypeDescriptor<T>::Dtype(),
          /*from_is_custom=*/false, /*to_is_custom=*/true))
    return false;
  CustomIntCastSpec<OtherT, T>::dtypes[1] =
      ml_dtypes::IntNTypeDescriptor<T>::dtype_meta;
  CustomIntCastSpec<OtherT, T>::spec.casting = from_safety;
  casts.push_back(&CustomIntCastSpec<OtherT, T>::spec);
  return true;
}

template <typename T>
bool AddCustomIntSelfCast(std::vector<PyArrayMethod_Spec*>& casts) {
  static PyType_Slot cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(
           PyCustomIntDType_to_CustomIntDType_resolve_descriptors<T>)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void*>(PyCustomIntCastLoop<T, T>)},
      {NPY_METH_strided_loop,
       reinterpret_cast<void*>(PyCustomIntCastLoop<T, T>)},
      {0, nullptr}};

  static PyArray_DTypeMeta* cast_dtypes[2] = {nullptr, nullptr};

  static PyArrayMethod_Spec cast_spec = {
      /*name=*/"customint_to_customint_cast",
      /*nin=*/1,
      /*nout=*/1,
      /*casting=*/NPY_NO_CASTING,
      /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
      /*dtypes=*/cast_dtypes,
      /*slots=*/cast_slots,
  };

  cast_dtypes[0] = IntNTypeDescriptor<T>::dtype_meta;
  cast_dtypes[1] = IntNTypeDescriptor<T>::dtype_meta;
  casts.push_back(&cast_spec);
  return true;
}

template <typename T>
bool GetIntCasts(std::vector<PyArrayMethod_Spec*>& casts) {
  if (!AddCustomIntSelfCast<T>(casts)) return false;

  NPY_CASTING signed_from_safety = NPY_UNSAFE_CASTING;
  NPY_CASTING unsigned_from_safety =
      std::numeric_limits<T>::is_signed ? NPY_UNSAFE_CASTING : NPY_SAFE_CASTING;

  if (!AddCustomIntCast<T, bool>(NPY_BOOL, NPY_UNSAFE_CASTING,
                                 signed_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, signed char>(NPY_BYTE, NPY_SAFE_CASTING,
                                        signed_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, short>(NPY_SHORT, NPY_SAFE_CASTING,
                                  signed_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, int>(NPY_INT, NPY_SAFE_CASTING, signed_from_safety,
                                casts))
    return false;
  if (!AddCustomIntCast<T, long>(NPY_LONG, NPY_SAFE_CASTING, signed_from_safety,
                                 casts))
    return false;
  if (!AddCustomIntCast<T, long long>(NPY_LONGLONG, NPY_SAFE_CASTING,
                                      signed_from_safety, casts))
    return false;

  if (!AddCustomIntCast<T, unsigned char>(NPY_UBYTE,
                                          std::numeric_limits<T>::is_signed
                                              ? NPY_UNSAFE_CASTING
                                              : NPY_SAFE_CASTING,
                                          unsigned_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, unsigned short>(NPY_USHORT,
                                           std::numeric_limits<T>::is_signed
                                               ? NPY_UNSAFE_CASTING
                                               : NPY_SAFE_CASTING,
                                           unsigned_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, unsigned int>(NPY_UINT,
                                         std::numeric_limits<T>::is_signed
                                             ? NPY_UNSAFE_CASTING
                                             : NPY_SAFE_CASTING,
                                         unsigned_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, unsigned long>(NPY_ULONG,
                                          std::numeric_limits<T>::is_signed
                                              ? NPY_UNSAFE_CASTING
                                              : NPY_SAFE_CASTING,
                                          unsigned_from_safety, casts))
    return false;
  if (!AddCustomIntCast<T, unsigned long long>(NPY_ULONGLONG,
                                               std::numeric_limits<T>::is_signed
                                                   ? NPY_UNSAFE_CASTING
                                                   : NPY_SAFE_CASTING,
                                               unsigned_from_safety, casts))
    return false;

  if (!AddCustomIntCast<T, half>(NPY_HALF, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING,
                                 casts))
    return false;
  if (!AddCustomIntCast<T, float>(NPY_FLOAT, NPY_SAFE_CASTING,
                                  NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, double>(NPY_DOUBLE, NPY_SAFE_CASTING,
                                   NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, long double>(NPY_LONGDOUBLE, NPY_SAFE_CASTING,
                                        NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<float>>(NPY_CFLOAT, NPY_SAFE_CASTING,
                                                NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<double>>(NPY_CDOUBLE, NPY_SAFE_CASTING,
                                                 NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  return true;
}

template <typename T>
bool RegisterIntNUFuncs(PyObject* numpy) {
  bool ok =
      RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
      RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                            "subtract") &&
      RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                            "multiply") &&
      RegisterUFunc<UFunc<ufuncs::FloorDivide<T>, T, T, T>, T>(
          numpy, "floor_divide") &&
      RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(numpy,
                                                             "remainder") &&
      RegisterUFunc<UFunc<ufuncs::Eq<T>, bool, T, T>, T>(numpy, "equal") &&
      RegisterUFunc<UFunc<ufuncs::Ne<T>, bool, T, T>, T>(numpy, "not_equal") &&
      RegisterUFunc<UFunc<ufuncs::Lt<T>, bool, T, T>, T>(numpy, "less") &&
      RegisterUFunc<UFunc<ufuncs::Le<T>, bool, T, T>, T>(numpy, "less_equal") &&
      RegisterUFunc<UFunc<ufuncs::Gt<T>, bool, T, T>, T>(numpy, "greater") &&
      RegisterUFunc<UFunc<ufuncs::Ge<T>, bool, T, T>, T>(numpy,
                                                         "greater_equal") &&
      RegisterUFunc<UFunc<ufuncs::LogicalAnd<T>, bool, T, T>, T>(
          numpy, "logical_and") &&
      RegisterUFunc<UFunc<ufuncs::LogicalOr<T>, bool, T, T>, T>(numpy,
                                                                "logical_or") &&
      RegisterUFunc<UFunc<ufuncs::LogicalXor<T>, bool, T, T>, T>(
          numpy, "logical_xor") &&
      RegisterUFunc<UFunc<ufuncs::LogicalNot<T>, bool, T>, T>(numpy,
                                                              "logical_not") &&
      RegisterUFunc<UFunc<ufuncs::IsFinite<T>, bool, T>, T>(numpy,
                                                            "isfinite") &&
      RegisterUFunc<UFunc<ufuncs::IsInf<T>, bool, T>, T>(numpy, "isinf") &&
      RegisterUFunc<UFunc<ufuncs::IsNan<T>, bool, T>, T>(numpy, "isnan") &&
      RegisterUFunc<UFunc<ufuncs::SignBit<T>, bool, T>, T>(numpy, "signbit");

  return ok;
}

template <typename T>
static PyObject* PyIntNDType_New(PyTypeObject* type, PyObject* args,
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
static PyObject* PyIntNDType_name_get(PyObject* self, void* context) {
  return PyUnicode_FromString(TypeDescriptor<T>::kTypeName);
}

template <typename T>
static PyObject* PyIntNDType_Str(PyObject* self) {
  return PyUnicode_FromString(TypeDescriptor<T>::kTypeName);
}

template <typename T>
static PyObject* PyIntNDType_Repr(PyObject* self) {
  return PyUnicode_FromFormat("dtype(%s)", TypeDescriptor<T>::kTypeName);
}

static inline PyArray_Descr* PyIntNDType_EnsureCanonical(PyArray_Descr* self) {
  Py_INCREF(self);
  return self;
}

template <typename T>
PyArray_DTypeMeta* PyIntNDType_CommonDType(PyArray_DTypeMeta* cls,
                                           PyArray_DTypeMeta* other) {
  if (cls == other) {
    Py_INCREF(cls);
    return cls;
  }

  // Fallback to a standard integer type of the same size.
  // This allows promotion with other standard types.
  int next_largest_typenum =
      std::numeric_limits<T>::is_signed ? NPY_BYTE : NPY_UBYTE;

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
static PyObject* PyIntNDType_Reduce(PyObject* self) {
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

#include <functional>

template <typename T>
bool RegisterIntNDtype(
    PyObject* numpy,
    std::function<void(std::vector<PyArrayMethod_Spec*>&)> output_casts = {}) {
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

#ifndef NPY_DT_PyArray_ArrFuncs_copyswapn
#define NPY_DT_PyArray_ArrFuncs_copyswapn (3 + (1 << 11))
#endif

#ifndef NPY_DT_PyArray_ArrFuncs_copyswap
#define NPY_DT_PyArray_ArrFuncs_copyswap (4 + (1 << 11))
#endif

  static PyType_Slot slots[] = {
      {NPY_DT_getitem, reinterpret_cast<void*>(NPyIntN_GetItem<T>)},
      {NPY_DT_setitem, reinterpret_cast<void*>(NPyIntN_SetItem<T>)},
      {NPY_DT_ensure_canonical,
       reinterpret_cast<void*>(PyIntNDType_EnsureCanonical)},
      {NPY_DT_PyArray_ArrFuncs_copyswap,
       reinterpret_cast<void*>(NPyIntN_CopySwap<T>)},
      {NPY_DT_PyArray_ArrFuncs_copyswapn,
       reinterpret_cast<void*>(NPyIntN_CopySwapN<T>)},
      {NPY_DT_PyArray_ArrFuncs_compare,
       reinterpret_cast<void*>(NPyIntN_CompareFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_nonzero,
       reinterpret_cast<void*>(NPyIntN_NonZero<T>)},
      {NPY_DT_PyArray_ArrFuncs_fill, reinterpret_cast<void*>(NPyIntN_Fill<T>)},
      {NPY_DT_PyArray_ArrFuncs_dotfunc,
       reinterpret_cast<void*>(NPyIntN_DotFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_argmax,
       reinterpret_cast<void*>(NPyIntN_ArgMaxFunc<T>)},
      {NPY_DT_PyArray_ArrFuncs_argmin,
       reinterpret_cast<void*>(NPyIntN_ArgMinFunc<T>)},
      {NPY_DT_common_dtype,
       reinterpret_cast<void*>(PyIntNDType_CommonDType<T>)},
      {0, nullptr}};

  if (!IntNTypeDescriptor<T>::dtype_meta) {
    IntNTypeDescriptor<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
  }

  static std::vector<PyArrayMethod_Spec*> cast_specs;
  static bool casts_initialized = [&]() {
    bool ok = GetIntCasts<T>(cast_specs);
    if (ok && output_casts) {
      output_casts(cast_specs);
    }
    cast_specs.push_back(nullptr);
    return ok;
  }();

  if (!casts_initialized) return false;

  PyArrayDTypeMeta_Spec spec = {
      /*typeobj=*/reinterpret_cast<PyTypeObject*>(type),
      /*flags=*/0,
      /*casts=*/cast_specs.data(),
      /*slots=*/slots,
      /*baseclass=*/nullptr};

  PyArray_DTypeMeta* dtype_meta = IntNTypeDescriptor<T>::dtype_meta;
  if (!dtype_meta) return false;

  PyTypeObject* tm = reinterpret_cast<PyTypeObject*>(dtype_meta);
  Py_SET_TYPE(tm, &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(tm, 1);
  tm->tp_name = TypeDescriptor<T>::kQualifiedTypeName;
  tm->tp_basicsize = sizeof(PyArray_Descr);
  tm->tp_base = &PyArrayDescr_Type;
  tm->tp_new = PyIntNDType_New<T>;
  tm->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  tm->tp_repr = PyIntNDType_Repr<T>;
  tm->tp_str = PyIntNDType_Str<T>;

  static PyGetSetDef dtype_getset[] = {
      {const_cast<char*>("name"), PyIntNDType_name_get<T>, nullptr, nullptr,
       nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr}};
  tm->tp_getset = dtype_getset;

  static PyMethodDef dtype_methods[] = {
      {const_cast<char*>("__reduce__"),
       reinterpret_cast<PyCFunction>(PyIntNDType_Reduce<T>), METH_NOARGS,
       nullptr},
      {nullptr, nullptr, 0, nullptr}};
  tm->tp_methods = dtype_methods;

  if (PyType_Ready(tm) < 0) {
    return false;
  }

  if (PyArrayInitDTypeMeta_FromSpec(dtype_meta, &spec) < 0) {
    return false;
  }

  IntNTypeDescriptor<T>::npy_type = dtype_meta->type_num;
  Safe_PyObjectPtr dtype_func =
      make_safe(PyObject_GetAttrString(numpy, "dtype"));
  if (!dtype_func) return false;
  Safe_PyObjectPtr descr_obj = make_safe(PyObject_CallFunctionObjArgs(
      dtype_func.get(), TypeDescriptor<T>::type_ptr, nullptr));
  if (!descr_obj) return false;
  IntNTypeDescriptor<T>::npy_descr =
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
          reinterpret_cast<PyObject*>(IntNTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterIntNUFuncs<T>(numpy);
}

}  // namespace ml_dtypes

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif

#endif  // ML_DTYPES_INT4_NUMPY_H_
