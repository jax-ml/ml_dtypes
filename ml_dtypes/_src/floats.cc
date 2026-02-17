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

// Enable cmath defines on Windows
#define _USE_MATH_DEFINES

#include "ml_dtypes/_src/floats.h"

#include <Python.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/numpy.h"
#include "ml_dtypes/_src/ufuncs.h"

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

namespace ml_dtypes {

template <typename T>
int CustomFloatType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomFloatType<T>::type_ptr = nullptr;
template <typename T>
PyArray_Descr* CustomFloatType<T>::npy_descr = nullptr;
template <typename T>
PyArray_DTypeMeta* CustomFloatType<T>::dtype_meta = nullptr;
template <typename T>
PyArray_DescrProto CustomFloatType<T>::numpy_1_descr_proto;

namespace {

// Representation of a Python custom float object.
template <typename T>
struct PyCustomFloat {
  PyObject_HEAD;  // Python object header
  T value;
};

// Constructs a PyCustomFloat object from PyCustomFloat<T>::T.
template <typename T>
Safe_PyObjectPtr PyCustomFloat_FromT(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(CustomFloatType<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyCustomFloat<T>* p = reinterpret_cast<PyCustomFloat<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Returns true if 'object' is a PyCustomFloat.
template <typename T>
bool PyCustomFloat_Check(PyObject* object) {
  return PyObject_IsInstance(object, CustomFloatType<T>::type_ptr);
}

// Extracts the value of a PyCustomFloat object.
template <typename T>
T PyCustomFloat_CustomFloat(PyObject* object) {
  return reinterpret_cast<PyCustomFloat<T>*>(object)->value;
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
    // Allow conversion from any NumPy scalar if conversion to float32
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
    if (PyArray_TYPE(arr) != CustomFloatType<T>::Dtype()) {
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
                 CustomFloatTraits<T>::kTypeName);
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
    if (PyArray_TYPE(arr) != CustomFloatType<T>::Dtype()) {
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
template <typename U>
inline PyObject* CompareValues(const U& val_a, const U& val_b, int op) {
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
      PyErr_SetString(PyExc_ValueError, "Invalid op type");
      return nullptr;
  }
  PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
}

template <typename T>
inline bool GetFloatDoubleValue(PyObject* obj, const T& val, bool is_custom,
                                double* out) {
  if (is_custom) {
    *out = static_cast<double>(static_cast<float>(val));
    return true;
  }
  if (PyFloat_Check(obj)) {
    *out = PyFloat_AsDouble(obj);
    return true;
  }
  if (PyLong_Check(obj)) {
    *out = PyLong_AsDouble(obj);
    return !PyErr_Occurred();
  }
  return false;
}

template <typename T>
PyObject* PyCustomFloat_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  bool a_is_custom = SafeCastToCustomFloat<T>(a, &x);
  bool b_is_custom = SafeCastToCustomFloat<T>(b, &y);
  if (a_is_custom && b_is_custom) {
    return CompareValues(x, y, op);
  }

  // Fallback to double comparison for float/int scalars.
  double val_a, val_b;
  if (GetFloatDoubleValue(a, x, a_is_custom, &val_a) &&
      GetFloatDoubleValue(b, y, b_is_custom, &val_b)) {
    return CompareValues(val_a, val_b, op);
  }

  if ((op == Py_EQ || op == Py_NE) &&
      (PyUnicode_Check(b) || PyBytes_Check(b) ||
       (!PyNumber_Check(b) && !PyArray_Check(b) && !PySequence_Check(b)))) {
    Py_RETURN_NOTIMPLEMENTED;
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

// Format function for PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Format(PyObject* self, PyObject* format_spec) {
  if (!PyUnicode_Check(format_spec)) {
    PyErr_Format(PyExc_TypeError, "__format__() argument 1 must be str, not %s",
                 Py_TYPE(format_spec)->tp_name);
    return nullptr;
  }
  PyObject* f = PyCustomFloat_Float<T>(self);
  if (!f) {
    return nullptr;
  }
  PyObject* result = PyObject_Format(f, format_spec);
  Py_DECREF(f);
  return result;
}

}  // namespace

template <typename T>
PyMethodDef CustomFloatType<T>::methods[] = {
    {"__format__", reinterpret_cast<PyCFunction>(PyCustomFloat_Format<T>),
     METH_O, "Format a custom float value."},
    {nullptr, nullptr, 0, nullptr},
};

template <typename T>
PyType_Slot CustomFloatType<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyCustomFloat_New<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyCustomFloat_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyCustomFloat_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyCustomFloat_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(CustomFloatTraits<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyCustomFloat_RichCompare<T>)},
    {Py_nb_add, reinterpret_cast<void*>(PyCustomFloat_Add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyCustomFloat_Subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyCustomFloat_Multiply<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyCustomFloat_Negative<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyCustomFloat_Int<T>)},
    {Py_nb_float, reinterpret_cast<void*>(PyCustomFloat_Float<T>)},
    {Py_tp_methods, reinterpret_cast<void*>(CustomFloatType<T>::methods)},
    {0, nullptr},
};

template <typename T>
PyType_Spec CustomFloatType<T>::type_spec = {
    /*.name=*/CustomFloatTraits<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyCustomFloat<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/CustomFloatType<T>::type_slots,
};

template <typename T>
PyArray_ArrFuncs CustomFloatType<T>::numpy_1_arr_funcs;

namespace {

template <typename T>
PyArray_DescrProto GetNumPy1FloatDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr)
      /*typeobj=*/nullptr,  // Filled in later
      /*kind=*/'V',
      /*type=*/CustomFloatTraits<T>::kNumPy1DescrType,
      /*byteorder=*/'=',
      /*flags=*/NPY_USE_SETITEM,
      /*type_num=*/0,
      /*elsize=*/sizeof(T),
      /*alignment=*/alignof(T),
      /*subarray=*/nullptr,
      /*fields=*/nullptr,
      /*names=*/nullptr,
      /*f=*/&CustomFloatType<T>::numpy_1_arr_funcs,
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
  return PyFloat_FromDouble(static_cast<double>(static_cast<float>(x)));
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
  return CompareFloats(static_cast<float>(x), static_cast<float>(y));
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
        ByteSwap16(r);
      }
    } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (npy_intp i = 0; i < n; i++) {
        memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
      }
    }
  } else if (swap && sizeof(T) == sizeof(int16_t)) {
    // In-place swap when src is NULL
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      ByteSwap16(r);
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
  if (!swap) {
    return;
  }

  if (sizeof(T) == sizeof(int16_t)) {
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
  return CompareFloats(static_cast<float>(b1), static_cast<float>(b2));
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
PyObject* PyCustomFloatDType_GetItem(PyArray_Descr* descr, char* data) {
  return NPyCustomFloat_GetItem<T>(data, nullptr);
}

template <typename T>
int PyCustomFloatDType_SetItem(PyArray_Descr* descr, PyObject* item,
                               char* data) {
  return NPyCustomFloat_SetItem<T>(item, data, nullptr);
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
  return NPY_SUCCEED;
}

template <typename T>
int PyCustomFloatDType_to_CustomFloatDType_CastLoop(
    PyArrayMethod_Context* context, char* const data[],
    npy_intp const dimensions[], npy_intp const strides[],
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
PyObject* PyCustomFloatDType_New(PyTypeObject* type, PyObject* args,
                                 PyObject* kwds) {
  if ((args == nullptr || PyTuple_Size(args) == 0) &&
      (kwds == nullptr || PyDict_Size(kwds) == 0) &&
      CustomFloatType<T>::npy_descr != nullptr) {
    Py_INCREF(CustomFloatType<T>::npy_descr);
    return reinterpret_cast<PyObject*>(CustomFloatType<T>::npy_descr);
  }
  PyTypeObject* meta_type =
      reinterpret_cast<PyTypeObject*>(CustomFloatType<T>::dtype_meta);
  if (!meta_type) meta_type = type;
  PyObject* obj = PyArrayDescr_Type.tp_new(meta_type, args, kwds);
  if (obj != nullptr) {
    PyArray_Descr* descr = reinterpret_cast<PyArray_Descr*>(obj);
    descr->elsize = sizeof(T);
    descr->alignment = alignof(T);
    descr->kind = 'f';
    descr->type = '?';
    descr->byteorder = '=';
    descr->type_num = CustomFloatType<T>::npy_type;
    descr->flags = NPY_USE_SETITEM;
  }
  return obj;
}

template <typename T>
PyObject* PyCustomFloatDType_Str(PyObject* self) {
  return PyUnicode_FromString(CustomFloatTraits<T>::kTypeName);
}

template <typename T>
PyObject* PyCustomFloatDType_Reduce(PyObject* self) {
  PyObject* name = PyUnicode_FromString(CustomFloatTraits<T>::kTypeName);
  PyObject* dtype_fn = reinterpret_cast<PyObject*>(&PyArrayDescr_Type);
  Py_INCREF(dtype_fn);
  PyObject* res = PyTuple_Pack(2, dtype_fn, PyTuple_Pack(1, name));
  Py_DECREF(name);
  Py_DECREF(dtype_fn);
  return res;
}

template <typename T>
PyObject* PyCustomFloatDType_Repr(PyObject* self) {
  std::string repr =
      std::string("dtype('") + CustomFloatTraits<T>::kTypeName + "')";
  return PyUnicode_FromString(repr.c_str());
}

template <typename T>
PyObject* PyCustomFloatDType_name_get(PyObject* self, void* closure) {
  return PyUnicode_FromString(CustomFloatTraits<T>::kTypeName);
}

template <typename T>
PyArray_DTypeMeta* PyCustomFloatDType_CommonDType(PyArray_DTypeMeta* cls,
                                                  PyArray_DTypeMeta* other) {
  if (other == nullptr || cls == other) {
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
  PyArray_Descr* descr2 = PyArray_GetDefaultDescr(other);
  if (!descr2) {
    Py_DECREF(descr1);
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    return reinterpret_cast<PyArray_DTypeMeta*>(Py_NotImplemented);
  }
  PyArray_Descr* common_descr = PyArray_PromoteTypes(descr1, descr2);
  Py_DECREF(descr1);
  Py_DECREF(descr2);
  if (!common_descr) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    return reinterpret_cast<PyArray_DTypeMeta*>(Py_NotImplemented);
  }
  PyArray_DTypeMeta* common_meta =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(common_descr));
  Py_INCREF(common_meta);
  Py_DECREF(common_descr);
  return common_meta;
}

template <typename T>
float CastToFloat(T value) {
  if constexpr (is_complex_v<T>) {
    return CastToFloat(value.real());
  } else {
    return static_cast<float>(value);
  }
}

template <typename T>
T CastToCustomFloatT(T value) {
  return value;
}

template <typename To, typename From>
To CastToCustomFloatT(From value) {
  return static_cast<To>(CastToFloat(value));
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
int PyCustomFloatCastLoop(PyArrayMethod_Context* context, char* const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  for (npy_intp i = 0; i < N; i++) {
    From f;
    memcpy(&f, in, sizeof(From));
    To t = CastToCustomFloatT<To>(f);
    memcpy(out, &t, sizeof(To));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

}  // namespace

template <typename From, typename To>
struct CustomFloatCastSpec {
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

namespace {

template <typename T, typename OtherT>
bool AddCustomFloatCast(int numpy_type, NPY_CASTING to_safety,
                        NPY_CASTING from_safety,
                        std::vector<PyArrayMethod_Spec*>& casts) {
  PyArray_Descr* d =
      numpy_type >= 0 ? PyArray_DescrFromType(numpy_type) : nullptr;
  PyArray_DTypeMeta* other_meta = nullptr;
  if (d) {
    other_meta = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(d));
  } else {
    other_meta = CustomFloatType<OtherT>::dtype_meta;
  }
  if (!other_meta) return true;
  if (!CustomFloatCastSpec<T, OtherT>::Initialize(nullptr, other_meta)) {
    Py_XDECREF(d);
    return false;
  }
  CustomFloatCastSpec<T, OtherT>::spec.casting = to_safety;
  casts.push_back(&CustomFloatCastSpec<T, OtherT>::spec);

  if (!CustomFloatCastSpec<OtherT, T>::Initialize(other_meta, nullptr)) {
    Py_XDECREF(d);
    return false;
  }
  CustomFloatCastSpec<OtherT, T>::spec.casting = from_safety;
  casts.push_back(&CustomFloatCastSpec<OtherT, T>::spec);
  Py_XDECREF(d);
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
                                             NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned int>(NPY_UINT, NPY_UNSAFE_CASTING,
                                           NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned long>(NPY_ULONG, NPY_UNSAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, unsigned long long>(
          NPY_ULONGLONG, NPY_UNSAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, signed char>(NPY_BYTE, NPY_UNSAFE_CASTING,
                                          NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, short>(NPY_SHORT, NPY_UNSAFE_CASTING,
                                    NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, int>(NPY_INT, NPY_UNSAFE_CASTING,
                                  NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, long>(NPY_LONG, NPY_UNSAFE_CASTING,
                                   NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, long long>(NPY_LONGLONG, NPY_UNSAFE_CASTING,
                                        NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<float>>(NPY_CFLOAT, NPY_SAFE_CASTING,
                                                  NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<double>>(
          NPY_CDOUBLE, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomFloatCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;

  if constexpr (!std::is_same_v<T, bfloat16>) {
    if (!AddCustomFloatCast<T, bfloat16>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                         NPY_UNSAFE_CASTING, casts))
      return false;
  }
  if constexpr (!std::is_same_v<T, float8_e8m0fnu>) {
    if constexpr (!std::is_same_v<T, float8_e3m4>) {
      if (!AddCustomFloatCast<T, float8_e3m4>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e4m3>) {
      if (!AddCustomFloatCast<T, float8_e4m3>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e4m3b11fnuz>) {
      if (!AddCustomFloatCast<T, float8_e4m3b11fnuz>(
              NPY_NOTYPE, NPY_UNSAFE_CASTING, NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e4m3fn>) {
      if (!AddCustomFloatCast<T, float8_e4m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e4m3fnuz>) {
      if (!AddCustomFloatCast<T, float8_e4m3fnuz>(
              NPY_NOTYPE, NPY_UNSAFE_CASTING, NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e5m2>) {
      if (!AddCustomFloatCast<T, float8_e5m2>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float8_e5m2fnuz>) {
      if (!AddCustomFloatCast<T, float8_e5m2fnuz>(
              NPY_NOTYPE, NPY_UNSAFE_CASTING, NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float6_e2m3fn>) {
      if (!AddCustomFloatCast<T, float6_e2m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float6_e3m2fn>) {
      if (!AddCustomFloatCast<T, float6_e3m2fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_UNSAFE_CASTING, casts))
        return false;
    }
    if constexpr (!std::is_same_v<T, float4_e2m1fn>) {
      if (!AddCustomFloatCast<T, float4_e2m1fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_UNSAFE_CASTING, casts))
        return false;
    }
  }

  return true;
}

template <typename T>
bool RegisterFloatUFuncs(PyObject* numpy, bool use_new_dtype_api) {
#define REG_UFUNC(name, ...) \
  RegisterUFunc<__VA_ARGS__, T>(numpy, name, use_new_dtype_api)
  bool ok =
      REG_UFUNC("add", UFunc<ufuncs::Add<T>, T, T, T>) &&
      REG_UFUNC("subtract", UFunc<ufuncs::Subtract<T>, T, T, T>) &&
      REG_UFUNC("multiply", UFunc<ufuncs::Multiply<T>, T, T, T>) &&
      REG_UFUNC("divide", UFunc<ufuncs::TrueDivide<T>, T, T, T>) &&
      REG_UFUNC("logaddexp", UFunc<ufuncs::LogAddExp<T>, T, T, T>) &&
      REG_UFUNC("logaddexp2", UFunc<ufuncs::LogAddExp2<T>, T, T, T>) &&
      REG_UFUNC("negative", UFunc<ufuncs::Negative<T>, T, T>) &&
      REG_UFUNC("positive", UFunc<ufuncs::Positive<T>, T, T>) &&
      REG_UFUNC("true_divide", UFunc<ufuncs::TrueDivide<T>, T, T, T>) &&
      REG_UFUNC("floor_divide", UFunc<ufuncs::FloorDivide<T>, T, T, T>) &&
      REG_UFUNC("power", UFunc<ufuncs::Power<T>, T, T, T>) &&
      REG_UFUNC("float_power", UFunc<ufuncs::Power<T>, T, T, T>) &&
      REG_UFUNC("remainder", UFunc<ufuncs::Remainder<T>, T, T, T>) &&
      REG_UFUNC("mod", UFunc<ufuncs::Remainder<T>, T, T, T>) &&
      REG_UFUNC("fmod", UFunc<ufuncs::Fmod<T>, T, T, T>) &&
      REG_UFUNC("divmod", UFunc2<ufuncs::Divmod<T>, T, T, T, T>) &&
      REG_UFUNC("absolute", UFunc<ufuncs::Abs<T>, T, T>) &&
      REG_UFUNC("fabs", UFunc<ufuncs::Abs<T>, T, T>) &&
      REG_UFUNC("rint", UFunc<ufuncs::Rint<T>, T, T>) &&
      REG_UFUNC("sign", UFunc<ufuncs::Sign<T>, T, T>) &&
      REG_UFUNC("heaviside", UFunc<ufuncs::Heaviside<T>, T, T, T>) &&
      REG_UFUNC("conjugate", UFunc<ufuncs::Conjugate<T>, T, T>) &&
      REG_UFUNC("exp", UFunc<ufuncs::Exp<T>, T, T>) &&
      REG_UFUNC("exp2", UFunc<ufuncs::Exp2<T>, T, T>) &&
      REG_UFUNC("expm1", UFunc<ufuncs::Expm1<T>, T, T>) &&
      REG_UFUNC("log", UFunc<ufuncs::Log<T>, T, T>) &&
      REG_UFUNC("log2", UFunc<ufuncs::Log2<T>, T, T>) &&
      REG_UFUNC("log10", UFunc<ufuncs::Log10<T>, T, T>) &&
      REG_UFUNC("log1p", UFunc<ufuncs::Log1p<T>, T, T>) &&
      REG_UFUNC("sqrt", UFunc<ufuncs::Sqrt<T>, T, T>) &&
      REG_UFUNC("square", UFunc<ufuncs::Square<T>, T, T>) &&
      REG_UFUNC("cbrt", UFunc<ufuncs::Cbrt<T>, T, T>) &&
      REG_UFUNC("reciprocal", UFunc<ufuncs::Reciprocal<T>, T, T>) &&

      // Trigonometric functions
      REG_UFUNC("sin", UFunc<ufuncs::Sin<T>, T, T>) &&
      REG_UFUNC("cos", UFunc<ufuncs::Cos<T>, T, T>) &&
      REG_UFUNC("tan", UFunc<ufuncs::Tan<T>, T, T>) &&
      REG_UFUNC("arcsin", UFunc<ufuncs::Arcsin<T>, T, T>) &&
      REG_UFUNC("arccos", UFunc<ufuncs::Arccos<T>, T, T>) &&
      REG_UFUNC("arctan", UFunc<ufuncs::Arctan<T>, T, T>) &&
      REG_UFUNC("arctan2", UFunc<ufuncs::Arctan2<T>, T, T, T>) &&
      REG_UFUNC("hypot", UFunc<ufuncs::Hypot<T>, T, T, T>) &&
      REG_UFUNC("sinh", UFunc<ufuncs::Sinh<T>, T, T>) &&
      REG_UFUNC("cosh", UFunc<ufuncs::Cosh<T>, T, T>) &&
      REG_UFUNC("tanh", UFunc<ufuncs::Tanh<T>, T, T>) &&
      REG_UFUNC("arcsinh", UFunc<ufuncs::Arcsinh<T>, T, T>) &&
      REG_UFUNC("arccosh", UFunc<ufuncs::Arccosh<T>, T, T>) &&
      REG_UFUNC("arctanh", UFunc<ufuncs::Arctanh<T>, T, T>) &&
      REG_UFUNC("deg2rad", UFunc<ufuncs::Deg2rad<T>, T, T>) &&
      REG_UFUNC("rad2deg", UFunc<ufuncs::Rad2deg<T>, T, T>) &&

      // Comparison functions
      REG_UFUNC("equal", UFunc<ufuncs::Eq<T>, bool, T, T>) &&
      REG_UFUNC("not_equal", UFunc<ufuncs::Ne<T>, bool, T, T>) &&
      REG_UFUNC("less", UFunc<ufuncs::Lt<T>, bool, T, T>) &&
      REG_UFUNC("greater", UFunc<ufuncs::Gt<T>, bool, T, T>) &&
      REG_UFUNC("less_equal", UFunc<ufuncs::Le<T>, bool, T, T>) &&
      REG_UFUNC("greater_equal", UFunc<ufuncs::Ge<T>, bool, T, T>) &&
      REG_UFUNC("maximum", UFunc<ufuncs::Maximum<T>, T, T, T>) &&
      REG_UFUNC("minimum", UFunc<ufuncs::Minimum<T>, T, T, T>) &&
      REG_UFUNC("fmax", UFunc<ufuncs::Fmax<T>, T, T, T>) &&
      REG_UFUNC("fmin", UFunc<ufuncs::Fmin<T>, T, T, T>) &&
      REG_UFUNC("clip", UFunc<ufuncs::Clip<T>, T, T, T, T>) &&
      REG_UFUNC("logical_and", UFunc<ufuncs::LogicalAnd<T>, bool, T, T>) &&
      REG_UFUNC("logical_or", UFunc<ufuncs::LogicalOr<T>, bool, T, T>) &&
      REG_UFUNC("logical_xor", UFunc<ufuncs::LogicalXor<T>, bool, T, T>) &&
      REG_UFUNC("logical_not", UFunc<ufuncs::LogicalNot<T>, bool, T>) &&

      // Floating point functions
      REG_UFUNC("isfinite", UFunc<ufuncs::IsFinite<T>, bool, T>) &&
      REG_UFUNC("isinf", UFunc<ufuncs::IsInf<T>, bool, T>) &&
      REG_UFUNC("isnan", UFunc<ufuncs::IsNan<T>, bool, T>) &&
      REG_UFUNC("signbit", UFunc<ufuncs::SignBit<T>, bool, T>) &&
      REG_UFUNC("copysign", UFunc<ufuncs::CopySign<T>, T, T, T>) &&
      REG_UFUNC("modf", UFunc2<ufuncs::Modf<T>, T, T, T>) &&
      REG_UFUNC("ldexp", UFunc<ufuncs::Ldexp<T>, T, T, int32_t>) &&
      REG_UFUNC("ldexp", UFunc<ufuncs::Ldexp<T>, T, T, int64_t>) &&
      REG_UFUNC("frexp", UFunc2<ufuncs::Frexp<T>, T, int, T>) &&
      REG_UFUNC("floor", UFunc<ufuncs::Floor<T>, T, T>) &&
      REG_UFUNC("ceil", UFunc<ufuncs::Ceil<T>, T, T>) &&
      REG_UFUNC("trunc", UFunc<ufuncs::Trunc<T>, T, T>) &&
      REG_UFUNC("nextafter", UFunc<ufuncs::NextAfter<T>, T, T, T>) &&
      REG_UFUNC("spacing", UFunc<ufuncs::Spacing<T>, T, T>);
#undef REG_UFUNC
  return ok;
}

template <typename From, typename To>
void NPyFloatCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                  void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = CastToCustomFloatT<To>(from[i]);
  }
}

template <typename T, typename U>
bool RegisterNumPy1FloatCast(int type_num) {
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);
  if (!descr) {
    return false;
  }
  if (PyArray_RegisterCastFunc(CustomFloatType<T>::npy_descr, type_num,
                               NPyFloatCast<T, U>) < 0) {
    return false;
  }
  if (type_num == NPY_FLOAT || type_num == NPY_DOUBLE ||
      type_num == NPY_LONGDOUBLE || type_num == NPY_CFLOAT ||
      type_num == NPY_CDOUBLE || type_num == NPY_CLONGDOUBLE) {
    if (PyArray_RegisterCanCast(CustomFloatType<T>::npy_descr, type_num,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  if (PyArray_RegisterCastFunc(descr, CustomFloatType<T>::npy_type,
                               NPyFloatCast<U, T>) < 0) {
    return false;
  }
  if (type_num == NPY_BOOL || type_num == NPY_UBYTE || type_num == NPY_BYTE) {
    if (PyArray_RegisterCanCast(descr, CustomFloatType<T>::npy_type,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool RegisterNumPy1FloatCasts() {
  if (!RegisterNumPy1FloatCast<T, half>(NPY_HALF)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, unsigned long long>(
          NPY_ULONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1FloatCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy1FloatDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomFloatType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomFloatType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  PyArray_ArrFuncs& arr_funcs = CustomFloatType<T>::numpy_1_arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyCustomFloat_GetItem<T>;
  arr_funcs.setitem = NPyCustomFloat_SetItem<T>;
  arr_funcs.compare = NPyCustomFloat_Compare<T>;
  arr_funcs.copyswapn = NPyCustomFloat_CopySwapN<T>;
  arr_funcs.copyswap = NPyCustomFloat_CopySwap<T>;
  arr_funcs.nonzero = NPyCustomFloat_NonZero<T>;
  arr_funcs.fill = NPyCustomFloat_Fill<T>;
  arr_funcs.dotfunc = NPyCustomFloat_DotFunc<T>;
  arr_funcs.argmax = NPyCustomFloat_ArgMaxFunc<T>;
  arr_funcs.argmin = NPyCustomFloat_ArgMinFunc<T>;

  PyArray_DescrProto& descr_proto = CustomFloatType<T>::numpy_1_descr_proto;
  descr_proto = GetNumPy1FloatDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  CustomFloatType<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (CustomFloatType<T>::npy_type < 0) {
    return false;
  }
  CustomFloatType<T>::npy_descr =
      PyArray_DescrFromType(CustomFloatType<T>::npy_type);

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), CustomFloatTraits<T>::kTypeName,
                           CustomFloatType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomFloatType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomFloatType<T>::npy_descr)) < 0) {
    return false;
  }

  if (!RegisterNumPy1FloatCasts<T>() ||
      !RegisterFloatUFuncs<T>(numpy, /*use_new_dtype_api=*/false)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "RegisterFloatUFuncs failed");
    }
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy2FloatDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyFloatingArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomFloatType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomFloatType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  static PyType_Slot slots[] = {
      {NPY_DT_getitem, reinterpret_cast<void*>(PyCustomFloatDType_GetItem<T>)},
      {NPY_DT_setitem, reinterpret_cast<void*>(PyCustomFloatDType_SetItem<T>)},
      {NPY_DT_ensure_canonical,
       reinterpret_cast<void*>(PyCustomFloatDType_EnsureCanonical)},
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
      /*casting=*/NPY_EQUIV_CASTING,
      /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
      /*dtypes=*/cast_dtypes,
      /*slots=*/cast_slots,
  };

  static std::vector<PyArrayMethod_Spec*> cast_specs;
  static bool casts_initialized = [&]() {
    cast_specs.push_back(&cast_spec);
    bool ok = GetFloatCasts<T>(cast_specs);
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

  if (!CustomFloatType<T>::dtype_meta) {
    CustomFloatType<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
    if (!CustomFloatType<T>::dtype_meta) return false;
  }
  PyArray_DTypeMeta* dtype_meta = CustomFloatType<T>::dtype_meta;

  PyTypeObject* tm = reinterpret_cast<PyTypeObject*>(dtype_meta);
  Py_SET_TYPE(tm, &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(tm, 1);
  tm->tp_name = CustomFloatTraits<T>::kQualifiedTypeName;
  tm->tp_basicsize = sizeof(PyArray_Descr);
  tm->tp_base = &PyArrayDescr_Type;
  tm->tp_new = PyCustomFloatDType_New<T>;
  tm->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

  static PyGetSetDef dtype_getset[] = {
      {const_cast<char*>("name"),
       reinterpret_cast<getter>(PyCustomFloatDType_name_get<T>), nullptr,
       nullptr, nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr}};
  tm->tp_repr = PyCustomFloatDType_Repr<T>;
  tm->tp_str = PyCustomFloatDType_Str<T>;
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

  CustomFloatType<T>::npy_type = dtype_meta->type_num;

  CustomFloatType<T>::npy_descr = PyArray_GetDefaultDescr(dtype_meta);
  if (!CustomFloatType<T>::npy_descr) return false;
  PyDataType_GetArrFuncs(CustomFloatType<T>::npy_descr)->copyswap =
      NPyCustomFloat_CopySwap<T>;
  PyDataType_GetArrFuncs(CustomFloatType<T>::npy_descr)->copyswapn =
      NPyCustomFloat_CopySwapN<T>;

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) {
    return false;
  }
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), CustomFloatTraits<T>::kTypeName,
                           CustomFloatType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomFloatType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomFloatType<T>::npy_descr)) < 0) {
    return false;
  }

  if (!RegisterFloatUFuncs<T>(numpy, /*use_new_dtype_api=*/true)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "RegisterFloatUFuncs failed");
    }
    return false;
  }
  return true;
}

template <typename T>
bool RegisterFloatDtype(PyObject* numpy, bool use_new_dtype_api) {
  if (use_new_dtype_api) {
    return RegisterNumPy2FloatDtype<T>(numpy);
  } else {
    return RegisterNumPy1FloatDtype<T>(numpy);
  }
}

}  // namespace

bool RegisterFloatDtypes(PyObject* numpy, bool use_new_dtype_api) {
  return RegisterFloatDtype<bfloat16>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e4m3>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e4m3b11fnuz>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e4m3fn>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e4m3fnuz>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e5m2>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e5m2fnuz>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e3m4>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float8_e8m0fnu>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float6_e2m3fn>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float6_e3m2fn>(numpy, use_new_dtype_api) &&
         RegisterFloatDtype<float4_e2m1fn>(numpy, use_new_dtype_api);
}

}  // namespace ml_dtypes
