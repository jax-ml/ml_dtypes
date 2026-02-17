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

#include "ml_dtypes/_src/ints.h"

#include <Python.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/numpy.h"
#include "ml_dtypes/_src/ufuncs.h"

namespace ml_dtypes {

constexpr char kOutOfRange[] = "out of range value cannot be converted to int4";

template <typename T>
int CustomIntType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomIntType<T>::type_ptr = nullptr;
template <typename T>
PyArray_Descr* CustomIntType<T>::npy_descr = nullptr;
template <typename T>
PyArray_DTypeMeta* CustomIntType<T>::dtype_meta = nullptr;
template <typename T>
PyArray_DescrProto CustomIntType<T>::numpy_1_descr_proto;

namespace {

// Representation of a Python custom integer object.
template <typename T>
struct PyIntN {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyIntN.
template <typename T>
bool PyIntN_Check(PyObject* object) {
  return PyObject_IsInstance(object, CustomIntType<T>::type_ptr);
}

template <typename T>
bool CastToIntN(PyObject* arg, T* output);

// Extracts the value of a PyIntN object.
template <typename T>
bool PyIntN_Value(PyObject* object, T* value) {
  if (PyIntN_Check<T>(object)) {
    *value = reinterpret_cast<PyIntN<T>*>(object)->value;
    return true;
  }
  return false;
}

template <typename T>
T PyIntN_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyIntN<T>*>(object)->value;
}

// Constructs a PyIntN object from PyIntN<T>::T.
template <typename T>
Safe_PyObjectPtr PyIntN_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(CustomIntType<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyIntN<T>* p = reinterpret_cast<PyIntN<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to an intN value. Returns true on success,
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
    *output = T(static_cast<int64_t>(f));
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
  if (PyLong_Check(arg)) {
    int overflow;
    long long v = PyLong_AsLongLongAndOverflow(arg, &overflow);  // NOLINT
    if (overflow) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(v);
    return true;
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
                 CustomIntTraits<T>::kTypeName);
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
    if (PyArray_TYPE(arr) != CustomIntType<T>::Dtype()) {
      Py_INCREF(CustomIntType<T>::npy_descr);
      return PyArray_CastToType(arr, CustomIntType<T>::npy_descr, 0);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse int from string, then cast to T.
    PyObject* i = PyLong_FromUnicodeObject(arg, 0);
    if (!i) {
      return nullptr;
    }
    bool ok = CastToIntN<T>(i, &value);
    Py_DECREF(i);
    if (ok) {
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
  T x, y;
  if (!PyIntN_Value<T>(a, &x) || !PyIntN_Value<T>(b, &y)) {
    if ((op == Py_EQ || op == Py_NE) &&
        (PyUnicode_Check(b) || PyBytes_Check(b) ||
         (!PyNumber_Check(b) && !PyArray_Check(b) && !PySequence_Check(b)))) {
      Py_RETURN_NOTIMPLEMENTED;
    }
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

// Format function for PyIntN.
template <typename T>
PyObject* PyIntN_Format(PyObject* self, PyObject* format_spec) {
  if (!PyUnicode_Check(format_spec)) {
    PyErr_Format(PyExc_TypeError, "__format__() argument 1 must be str, not %s",
                 Py_TYPE(format_spec)->tp_name);
    return nullptr;
  }
  PyObject* i = PyIntN_nb_int<T>(self);
  if (!i) {
    return nullptr;
  }
  PyObject* result = PyObject_Format(i, format_spec);
  Py_DECREF(i);
  return result;
}

}  // namespace

template <typename T>
PyMethodDef CustomIntType<T>::methods[] = {
    {"__format__", reinterpret_cast<PyCFunction>(PyIntN_Format<T>), METH_O,
     "Format a custom integer value."},
    {nullptr, nullptr, 0, nullptr},
};

template <typename T>
PyType_Slot CustomIntType<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyIntN_tp_new<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyIntN_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyIntN_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyIntN_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(CustomIntTraits<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyIntN_RichCompare<T>)},
    {Py_tp_methods, reinterpret_cast<void*>(CustomIntType<T>::methods)},
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
PyType_Spec CustomIntType<T>::type_spec = {
    /*.name=*/CustomIntTraits<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyIntN<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/CustomIntType<T>::type_slots,
};

template <typename T>
PyArray_ArrFuncs CustomIntType<T>::numpy_1_arr_funcs;

namespace {

template <typename T>
PyArray_DescrProto GetNumPy1IntNDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr)
      /*typeobj=*/nullptr,  // Filled in later
      /*kind=*/'V',
      /*type=*/CustomIntTraits<T>::kNumPy1DescrType,
      /*byteorder=*/'=',
      /*flags=*/NPY_USE_SETITEM,
      /*type_num=*/0,
      /*elsize=*/sizeof(T),
      /*alignment=*/alignof(T),
      /*subarray=*/nullptr,
      /*fields=*/nullptr,
      /*names=*/nullptr,
      /*f=*/&CustomIntType<T>::numpy_1_arr_funcs,
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
}

// Note: No byte swapping needed for 8-bit integer types
template <typename T>
void NPyIntN_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (src) {
    memcpy(dst, src, sizeof(T));
  }
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

template <typename T>
PyObject* PyCustomIntDType_GetItem(PyArray_Descr* descr, char* data) {
  return NPyIntN_GetItem<T>(data, nullptr);
}

template <typename T>
int PyCustomIntDType_SetItem(PyArray_Descr* descr, PyObject* item, char* data) {
  return NPyIntN_SetItem<T>(item, data, nullptr);
}

static inline PyArray_Descr* PyCustomIntDType_EnsureCanonical(
    PyArray_Descr* dtype) {
  Py_INCREF(dtype);
  return dtype;
}

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
  return NPY_SUCCEED;
}

template <typename T>
int PyCustomIntDType_to_CustomIntDType_CastLoop(PyArrayMethod_Context* context,
                                                char* const data[],
                                                npy_intp const dimensions[],
                                                npy_intp const strides[],
                                                NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  if (in == out) return 0;
  for (npy_intp i = 0; i < N; i++) {
    memcpy(out, in, sizeof(T));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

template <typename T>
PyObject* PyCustomIntDType_New(PyTypeObject* type, PyObject* args,
                               PyObject* kwds) {
  if ((args == nullptr || PyTuple_Size(args) == 0) &&
      (kwds == nullptr || PyDict_Size(kwds) == 0) &&
      CustomIntType<T>::npy_descr != nullptr) {
    Py_INCREF(CustomIntType<T>::npy_descr);
    return reinterpret_cast<PyObject*>(CustomIntType<T>::npy_descr);
  }
  PyTypeObject* meta_type =
      reinterpret_cast<PyTypeObject*>(CustomIntType<T>::dtype_meta);
  if (!meta_type) meta_type = type;
  PyObject* obj = PyArrayDescr_Type.tp_new(meta_type, args, kwds);
  if (obj != nullptr) {
    PyArray_Descr* descr = reinterpret_cast<PyArray_Descr*>(obj);
    descr->elsize = sizeof(T);
    descr->alignment = alignof(T);
    descr->kind = std::numeric_limits<T>::is_signed ? 'i' : 'u';
    descr->type = '?';
    descr->byteorder = '=';
    descr->type_num = CustomIntType<T>::npy_type;
    descr->flags = NPY_USE_SETITEM;
  }
  return obj;
}

template <typename T>
PyObject* PyCustomIntDType_Str(PyObject* self) {
  return PyUnicode_FromString(CustomIntTraits<T>::kTypeName);
}

template <typename T>
PyObject* PyCustomIntDType_Reduce(PyObject* self) {
  PyObject* name = PyUnicode_FromString(CustomIntTraits<T>::kTypeName);
  PyObject* dtype_fn = reinterpret_cast<PyObject*>(&PyArrayDescr_Type);
  Py_INCREF(dtype_fn);
  PyObject* res = PyTuple_Pack(2, dtype_fn, PyTuple_Pack(1, name));
  Py_DECREF(name);
  Py_DECREF(dtype_fn);
  return res;
}

template <typename T>
PyObject* PyCustomIntDType_Repr(PyObject* self) {
  std::string repr =
      std::string("dtype('") + CustomIntTraits<T>::kTypeName + "')";
  return PyUnicode_FromString(repr.c_str());
}

template <typename T>
PyObject* PyCustomIntDType_name_get(PyObject* self, void* closure) {
  return PyUnicode_FromString(CustomIntTraits<T>::kTypeName);
}

template <typename T>
PyArray_DTypeMeta* PyCustomIntDType_CommonDType(PyArray_DTypeMeta* cls,
                                                PyArray_DTypeMeta* other) {
  if (other == nullptr || cls == other) {
    Py_INCREF(cls);
    return cls;
  }

  int next_largest_typenum =
      std::numeric_limits<T>::is_signed ? NPY_INT8 : NPY_UINT8;

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
T CastToCustomIntT(T value) {
  return value;
}

template <typename To, typename From>
To CastToCustomIntT(From value) {
  return static_cast<To>(CastToInt(value));
}

// Performs a NumPy array cast from type 'From' to 'To'.
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
    To t = CastToCustomIntT<To>(f);
    memcpy(out, &t, sizeof(To));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

}  // namespace

template <typename T>
constexpr int get_int_bits() {
  if constexpr (std::is_same_v<T, int1> || std::is_same_v<T, uint1>) {
    return 1;
  } else if constexpr (std::is_same_v<T, int2> || std::is_same_v<T, uint2>) {
    return 2;
  } else if constexpr (std::is_same_v<T, int4> || std::is_same_v<T, uint4>) {
    return 4;
  } else {
    return sizeof(T) * 8;
  }
}

template <typename T, typename U>
NPY_CASTING GetIntCastingSafety() {
  if constexpr (std::is_same_v<T, bool>) {
    return NPY_SAFE_CASTING;
  }
  if constexpr (std::is_same_v<U, bool>) {
    return NPY_UNSAFE_CASTING;
  }
  if constexpr (std::is_floating_point_v<U> || is_complex_v<U> ||
                std::is_same_v<U, half>) {
    return NPY_SAFE_CASTING;
  }
  if constexpr (std::is_floating_point_v<T> || is_complex_v<T> ||
                std::is_same_v<T, half>) {
    return NPY_UNSAFE_CASTING;
  }
  bool t_is_signed = std::numeric_limits<T>::is_signed;
  bool u_is_signed = std::numeric_limits<U>::is_signed;

  int t_bits = get_int_bits<T>();
  int u_bits = get_int_bits<U>();

  if (t_is_signed == u_is_signed) {
    return u_bits >= t_bits ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
  } else if (!t_is_signed && u_is_signed) {
    return u_bits > t_bits ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
  } else {
    return NPY_UNSAFE_CASTING;
  }
}

template <typename From, typename To>
struct CustomIntCastSpec {
  static PyType_Slot slots[3];
  static PyArray_DTypeMeta* dtypes[2];
  static PyArrayMethod_Spec spec;
  static bool Initialize(PyArray_DTypeMeta* from_meta,
                         PyArray_DTypeMeta* to_meta) {
    dtypes[0] = from_meta;
    dtypes[1] = to_meta;
    return true;
  }
};

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
    /*casting=*/NPY_UNSAFE_CASTING,
    /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*dtypes=*/dtypes,
    /*slots=*/slots,
};

namespace {

// Registers a cast between T (a reduced float) and type 'OtherT'.
template <typename T, typename OtherT>
bool AddCustomIntCast(int numpy_type, NPY_CASTING to_safety,
                      NPY_CASTING from_safety,
                      std::vector<PyArrayMethod_Spec*>& casts) {
  PyArray_Descr* d =
      numpy_type >= 0 ? PyArray_DescrFromType(numpy_type) : nullptr;
  PyArray_DTypeMeta* other_meta = nullptr;
  if (d) {
    other_meta = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(d));
  } else {
    other_meta = GetCustomDTypeMeta<OtherT>();
  }
  if (!other_meta) return true;
  if (!CustomIntCastSpec<T, OtherT>::Initialize(nullptr, other_meta)) {
    Py_XDECREF(d);
    return false;
  }
  CustomIntCastSpec<T, OtherT>::spec.casting = to_safety;
  casts.push_back(&CustomIntCastSpec<T, OtherT>::spec);

  if (!CustomIntCastSpec<OtherT, T>::Initialize(other_meta, nullptr)) {
    Py_XDECREF(d);
    return false;
  }
  CustomIntCastSpec<OtherT, T>::spec.casting = from_safety;
  casts.push_back(&CustomIntCastSpec<OtherT, T>::spec);
  Py_XDECREF(d);
  return true;
}

template <typename T>
bool GetIntCasts(std::vector<PyArrayMethod_Spec*>& casts) {
  if (!AddCustomIntCast<T, half>(NPY_HALF, NPY_SAFE_CASTING,
                                 GetIntCastingSafety<half, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, float>(NPY_FLOAT, NPY_SAFE_CASTING,
                                  GetIntCastingSafety<float, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, double>(NPY_DOUBLE, NPY_SAFE_CASTING,
                                   GetIntCastingSafety<double, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, long double>(NPY_LONGDOUBLE, NPY_SAFE_CASTING,
                                        GetIntCastingSafety<long double, T>(),
                                        casts))
    return false;
  if (!AddCustomIntCast<T, bool>(NPY_BOOL, NPY_UNSAFE_CASTING,
                                 GetIntCastingSafety<bool, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, unsigned char>(
          NPY_UBYTE, GetIntCastingSafety<T, unsigned char>(),
          GetIntCastingSafety<unsigned char, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, unsigned short>(
          NPY_USHORT, GetIntCastingSafety<T, unsigned short>(),
          GetIntCastingSafety<unsigned short, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, unsigned int>(
          NPY_UINT, GetIntCastingSafety<T, unsigned int>(),
          GetIntCastingSafety<unsigned int, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, unsigned long>(
          NPY_ULONG, GetIntCastingSafety<T, unsigned long>(),
          GetIntCastingSafety<unsigned long, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, unsigned long long>(
          NPY_ULONGLONG, GetIntCastingSafety<T, unsigned long long>(),
          GetIntCastingSafety<unsigned long long, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, signed char>(
          NPY_BYTE, GetIntCastingSafety<T, signed char>(),
          GetIntCastingSafety<signed char, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, short>(NPY_SHORT, GetIntCastingSafety<T, short>(),
                                  GetIntCastingSafety<short, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, int>(NPY_INT, GetIntCastingSafety<T, int>(),
                                GetIntCastingSafety<int, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, long>(NPY_LONG, GetIntCastingSafety<T, long>(),
                                 GetIntCastingSafety<long, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, long long>(
          NPY_LONGLONG, GetIntCastingSafety<T, long long>(),
          GetIntCastingSafety<long long, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<float>>(
          NPY_CFLOAT, NPY_SAFE_CASTING,
          GetIntCastingSafety<std::complex<float>, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<double>>(
          NPY_CDOUBLE, NPY_SAFE_CASTING,
          GetIntCastingSafety<std::complex<double>, T>(), casts))
    return false;
  if (!AddCustomIntCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE, NPY_SAFE_CASTING,
          GetIntCastingSafety<std::complex<long double>, T>(), casts))
    return false;

  if constexpr (!std::is_same_v<T, int1>) {
    if (!AddCustomIntCast<T, int1>(NPY_NOTYPE, GetIntCastingSafety<T, int1>(),
                                   GetIntCastingSafety<int1, T>(), casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, uint1>) {
    if (!AddCustomIntCast<T, uint1>(NPY_NOTYPE, GetIntCastingSafety<T, uint1>(),
                                    GetIntCastingSafety<uint1, T>(), casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, int2>) {
    if (!AddCustomIntCast<T, int2>(NPY_NOTYPE, GetIntCastingSafety<T, int2>(),
                                   GetIntCastingSafety<int2, T>(), casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, uint2>) {
    if (!AddCustomIntCast<T, uint2>(NPY_NOTYPE, GetIntCastingSafety<T, uint2>(),
                                    GetIntCastingSafety<uint2, T>(), casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, int4>) {
    if (!AddCustomIntCast<T, int4>(NPY_NOTYPE, GetIntCastingSafety<T, int4>(),
                                   GetIntCastingSafety<int4, T>(), casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, uint4>) {
    if (!AddCustomIntCast<T, uint4>(NPY_NOTYPE, GetIntCastingSafety<T, uint4>(),
                                    GetIntCastingSafety<uint4, T>(), casts))
      return false;
  }

  if (!AddCustomIntCast<T, bfloat16>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                     NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e3m4>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                        NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e4m3>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                        NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e4m3b11fnuz>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                               NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e4m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                          NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e4m3fnuz>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e5m2>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                        NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e5m2fnuz>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float8_e8m0fnu>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                           NPY_UNSAFE_CASTING, casts))
    return false;
  if constexpr (!std::is_same_v<T, int4> && !std::is_same_v<T, uint4>) {
    if (!AddCustomIntCast<T, float6_e2m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
      return false;
  }
  if (!AddCustomIntCast<T, float6_e3m2fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                          NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomIntCast<T, float4_e2m1fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                          NPY_UNSAFE_CASTING, casts))
    return false;

  return true;
}

template <typename T>
bool RegisterIntNUFuncs(PyObject* numpy, bool use_new_dtype_api) {
  bool ok = RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(
                numpy, "add", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(
                numpy, "subtract", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(
                numpy, "multiply", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::FloorDivide<T>, T, T, T>, T>(
                numpy, "floor_divide", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(
                numpy, "remainder", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Eq<T>, bool, T, T>, T>(
                numpy, "equal", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Ne<T>, bool, T, T>, T>(
                numpy, "not_equal", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Lt<T>, bool, T, T>, T>(
                numpy, "less", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Gt<T>, bool, T, T>, T>(
                numpy, "greater", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Le<T>, bool, T, T>, T>(
                numpy, "less_equal", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Ge<T>, bool, T, T>, T>(
                numpy, "greater_equal", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Maximum<T>, T, T, T>, T>(
                numpy, "maximum", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Minimum<T>, T, T, T>, T>(
                numpy, "minimum", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::Clip<T>, T, T, T, T>, T>(
                numpy, "clip", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::LogicalNot<T>, bool, T>, T>(
                numpy, "logical_not", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::LogicalAnd<T>, bool, T, T>, T>(
                numpy, "logical_and", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::LogicalOr<T>, bool, T, T>, T>(
                numpy, "logical_or", use_new_dtype_api) &&
            RegisterUFunc<UFunc<ufuncs::LogicalXor<T>, bool, T, T>, T>(
                numpy, "logical_xor", use_new_dtype_api);
  return ok;
}

template <typename From, typename To>
void NPyIntNCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = CastToCustomIntT<To>(from[i]);
  }
}

template <typename T, typename U>
bool RegisterNumPy1IntNCast(int numpy_type) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (!descr) {
    return false;
  }
  if (PyArray_RegisterCastFunc(descr, CustomIntType<T>::Dtype(),
                               NPyIntNCast<U, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(CustomIntType<T>::npy_descr, numpy_type,
                               NPyIntNCast<T, U>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy1IntNCasts() {
  if (!RegisterNumPy1IntNCast<T, half>(NPY_HALF)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterNumPy1IntNCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1IntNCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_INT8,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_INT16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_INT32,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_INT64,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (!std::numeric_limits<T>::is_signed) {
    if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_UINT8,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_UINT16,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_UINT32,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_UINT64,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_HALF,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_FLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_DOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_LONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_CFLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_CDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(CustomIntType<T>::npy_descr, NPY_CLONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy1IntNDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomIntType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomIntType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  PyArray_ArrFuncs& arr_funcs = CustomIntType<T>::numpy_1_arr_funcs;
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

  PyArray_DescrProto& descr_proto = CustomIntType<T>::numpy_1_descr_proto;
  descr_proto = GetNumPy1IntNDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  CustomIntType<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (CustomIntType<T>::npy_type < 0) {
    return false;
  }
  CustomIntType<T>::npy_descr =
      PyArray_DescrFromType(CustomIntType<T>::npy_type);

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), CustomIntTraits<T>::kTypeName,
                           CustomIntType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomIntType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomIntType<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterNumPy1IntNCasts<T>() &&
         RegisterIntNUFuncs<T>(numpy, /*use_new_dtype_api=*/false);
}

template <typename T>
bool RegisterNumPy2IntNDtype(PyObject* numpy) {
  PyTypeObject* base_type = std::numeric_limits<T>::is_signed
                                ? &PySignedIntegerArrType_Type
                                : &PyUnsignedIntegerArrType_Type;
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(base_type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomIntType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomIntType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  static PyType_Slot slots[] = {
      {NPY_DT_getitem, reinterpret_cast<void*>(PyCustomIntDType_GetItem<T>)},
      {NPY_DT_setitem, reinterpret_cast<void*>(PyCustomIntDType_SetItem<T>)},
      {NPY_DT_ensure_canonical,
       reinterpret_cast<void*>(PyCustomIntDType_EnsureCanonical)},
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
       reinterpret_cast<void*>(PyCustomIntDType_CommonDType<T>)},
      {0, nullptr}};

  static PyType_Slot cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(
           PyCustomIntDType_to_CustomIntDType_resolve_descriptors<T>)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void*>(PyCustomIntDType_to_CustomIntDType_CastLoop<T>)},
      {NPY_METH_strided_loop,
       reinterpret_cast<void*>(PyCustomIntDType_to_CustomIntDType_CastLoop<T>)},
      {0, nullptr}};

  static PyArray_DTypeMeta* cast_dtypes[2] = {nullptr, nullptr};

  static PyArrayMethod_Spec cast_spec = {
      /*name=*/"customint_to_customint_cast",
      /*nin=*/1,
      /*nout=*/1,
      /*casting=*/NPY_EQUIV_CASTING,
      /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
      /*dtypes=*/cast_dtypes,
      /*slots=*/cast_slots,
  };

  static std::vector<PyArrayMethod_Spec*> cast_specs;
  static bool casts_initialized = [&]() {
    cast_specs.push_back(&cast_spec);
    bool ok = GetIntCasts<T>(cast_specs);
    cast_specs.push_back(nullptr);
    return ok;
  }();

  if (!casts_initialized) {
    PyErr_SetString(PyExc_RuntimeError, "casts_initialized failed");
    return false;
  }

  static PyArrayDTypeMeta_Spec spec = {
      /*typeobj=*/reinterpret_cast<PyTypeObject*>(type),
      /*flags=*/0,
      /*casts=*/cast_specs.data(),
      /*slots=*/slots,
      /*baseclass=*/nullptr};

  if (!CustomIntType<T>::dtype_meta) {
    CustomIntType<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
    if (!CustomIntType<T>::dtype_meta) return false;
  }
  PyArray_DTypeMeta* dtype_meta = CustomIntType<T>::dtype_meta;

  PyTypeObject* tm = reinterpret_cast<PyTypeObject*>(dtype_meta);

  static PyGetSetDef dtype_getset[] = {
      {const_cast<char*>("name"),
       reinterpret_cast<getter>(PyCustomIntDType_name_get<T>), nullptr, nullptr,
       nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr}};
  Py_SET_TYPE(tm, &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(tm, 1);
  tm->tp_name = CustomIntTraits<T>::kQualifiedTypeName;
  tm->tp_basicsize = sizeof(PyArray_Descr);
  tm->tp_base = &PyArrayDescr_Type;
  tm->tp_new = PyCustomIntDType_New<T>;
  tm->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  tm->tp_repr = PyCustomIntDType_Repr<T>;
  tm->tp_str = PyCustomIntDType_Str<T>;
  tm->tp_getset = dtype_getset;

  static PyMethodDef dtype_methods[] = {
      {const_cast<char*>("__reduce__"),
       reinterpret_cast<PyCFunction>(PyCustomIntDType_Reduce<T>), METH_NOARGS,
       nullptr},
      {nullptr, nullptr, 0, nullptr}};
  tm->tp_methods = dtype_methods;

  if (PyType_Ready(tm) < 0) {
    return false;
  }

  if (PyArrayInitDTypeMeta_FromSpec(dtype_meta, &spec) < 0) {
    return false;
  }

  CustomIntType<T>::npy_type = dtype_meta->type_num;

  CustomIntType<T>::npy_descr = PyArray_GetDefaultDescr(dtype_meta);
  if (!CustomIntType<T>::npy_descr) return false;
  PyDataType_GetArrFuncs(CustomIntType<T>::npy_descr)->copyswap =
      NPyIntN_CopySwap<T>;
  PyDataType_GetArrFuncs(CustomIntType<T>::npy_descr)->copyswapn =
      NPyIntN_CopySwapN<T>;

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), CustomIntTraits<T>::kTypeName,
                           CustomIntType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomIntType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomIntType<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterIntNUFuncs<T>(numpy, /*use_new_dtype_api=*/true);
}

template <typename T>
bool RegisterIntNDtype(PyObject* numpy, bool use_new_dtype_api) {
  if (use_new_dtype_api) {
    return RegisterNumPy2IntNDtype<T>(numpy);
  } else {
    return RegisterNumPy1IntNDtype<T>(numpy);
  }
}

}  // namespace

bool RegisterIntDtypes(PyObject* numpy, bool use_new_dtype_api) {
  return RegisterIntNDtype<int1>(numpy, use_new_dtype_api) &&
         RegisterIntNDtype<uint1>(numpy, use_new_dtype_api) &&
         RegisterIntNDtype<int2>(numpy, use_new_dtype_api) &&
         RegisterIntNDtype<uint2>(numpy, use_new_dtype_api) &&
         RegisterIntNDtype<int4>(numpy, use_new_dtype_api) &&
         RegisterIntNDtype<uint4>(numpy, use_new_dtype_api);
}

}  // namespace ml_dtypes
