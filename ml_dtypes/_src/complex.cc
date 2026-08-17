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

#include "ml_dtypes/_src/complex.h"

#include <Python.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/numpy.h"
#include "ml_dtypes/_src/ufuncs.h"
#include "ml_dtypes/include/complex_types.h"

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

namespace ml_dtypes {

template <typename T>
int CustomComplexType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomComplexType<T>::type_ptr = nullptr;
template <typename T>
PyArray_Descr* CustomComplexType<T>::npy_descr = nullptr;
template <typename T>
PyArray_DTypeMeta* CustomComplexType<T>::dtype_meta = nullptr;
template <typename T>
PyArray_DescrProto CustomComplexType<T>::numpy_1_descr_proto;

namespace {

// Representation of a Python custom float object.
template <typename T>
struct PyCustomComplex {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyCustomComplex.
template <typename T>
bool PyCustomComplex_Check(PyObject* object) {
  return PyObject_IsInstance(object, CustomComplexType<T>::type_ptr);
}

// Extracts the value of a PyCustomComplex object.
template <typename T>
T PyCustomComplex_CustomComplex(PyObject* object) {
  return reinterpret_cast<PyCustomComplex<T>*>(object)->value;
}

// Constructs a PyCustomComplex object from PyCustomComplex<T>::T.
template <typename T>
Safe_PyObjectPtr PyCustomComplex_FromT(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(CustomComplexType<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyCustomComplex<T>* p = reinterpret_cast<PyCustomComplex<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

inline const std::complex<double> to_cpp(const Py_complex& p) {
  return *reinterpret_cast<const std::complex<double>*>(&p);
}

inline const Py_complex to_python(const std::complex<double> p) {
  return Py_complex{p.real(), p.imag()};
}

// Converts a Python object to a reduced float value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToCustomComplex(PyObject* arg, T* output) {
  using real_type = typename T::value_type;
  // Complex part is often zero, so initialize it here.
  output->imag(static_cast<real_type>(0));

  if (PyCustomComplex_Check<T>(arg)) {
    *output = PyCustomComplex_CustomComplex<T>(arg);
    return true;
  }
  if (PyComplex_Check(arg)) {
    std::complex<double> c = to_cpp(PyComplex_AsCComplex(arg));
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(c);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(std::complex<float>(d, 0));
    return true;
  }
  if (PyLong_Check(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(std::complex<float>(static_cast<float>(l), 0));
    return true;
  }
  if (PyArray_IsScalar(arg, Generic)) {
    // Allow conversion from any NumPy scalar if conversion to complex float
    // is defined.
    std::complex<float> c;
    PyArray_Descr* cf_descr = PyArray_DescrFromType(NPY_COMPLEX64);
    if (PyArray_Pack(cf_descr, &c, arg) < 0) {
      Py_DECREF(cf_descr);
      return false;
    }
    Py_DECREF(cf_descr);
    *output = T(c);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != CustomComplexType<T>::Dtype()) {
      Py_INCREF(CustomComplexType<T>::npy_descr);
      ref = make_safe(
          PyArray_CastToType(arr, CustomComplexType<T>::npy_descr, 0));
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
bool SafeCastToCustomComplex(PyObject* arg, T* output) {
  if (PyCustomComplex_Check<T>(arg)) {
    *output = PyCustomComplex_CustomComplex<T>(arg);
    return true;
  }
  return false;
}

// Converts a PyReduceFloat into a PyInt.
template <typename T>
PyObject* PyCustomComplex_Int(PyObject* self) {
  if (GiveComplexWarning() < 0) {
    return nullptr;
  }
  T x = PyCustomComplex_CustomComplex<T>(self);
  long y = static_cast<long>(static_cast<float>(x.real()));  // NOLINT
  return PyLong_FromLong(y);
}

// Converts a PyReduceFloat into a PyInt.
template <typename T>
PyObject* PyCustomComplex_Float(PyObject* self) {
  if (GiveComplexWarning() < 0) {
    return nullptr;
  }
  T x = PyCustomComplex_CustomComplex<T>(self);
  return PyFloat_FromDouble(static_cast<float>(x.real()));
}

// Converts to Python complex.
template <typename T>
PyObject* PyCustomComplex_Complex(PyObject* self, PyObject*) {
  T x = PyCustomComplex_CustomComplex<T>(self);
  std::complex<float> c = static_cast<std::complex<float>>(x);
  return PyComplex_FromDoubles(c.real(), c.imag());
}

// Negates a PyCustomComplex.
template <typename T>
PyObject* PyCustomComplex_Negative(PyObject* self) {
  T x = PyCustomComplex_CustomComplex<T>(self);
  return PyCustomComplex_FromT<T>(-x).release();
}

template <typename T>
PyObject* PyCustomComplex_Add(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomComplex<T>(a, &x) && SafeCastToCustomComplex<T>(b, &y)) {
    return PyCustomComplex_FromT<T>(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyCustomComplex_Subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomComplex<T>(a, &x) && SafeCastToCustomComplex<T>(b, &y)) {
    return PyCustomComplex_FromT<T>(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyCustomComplex_Multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomComplex<T>(a, &x) && SafeCastToCustomComplex<T>(b, &y)) {
    auto res = static_cast<std::complex<float>>(x) *
               static_cast<std::complex<float>>(y);
    return PyCustomComplex_FromT<T>(static_cast<T>(res)).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyCustomComplex_TrueDivide(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomComplex<T>(a, &x) && SafeCastToCustomComplex<T>(b, &y)) {
    auto res = static_cast<std::complex<float>>(x) /
               static_cast<std::complex<float>>(y);
    return PyCustomComplex_FromT<T>(static_cast<T>(res)).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Constructs a new PyCustomComplex.
template <typename T>
PyObject* PyCustomComplex_New(PyTypeObject* type, PyObject* args,
                              PyObject* kwds) {
  T value;

  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size == 2) {
    // The user passed two arguments, just forward them to the complex
    // constructor.
    Safe_PyObjectPtr c =
        make_safe(PyComplex_Type.tp_new(&PyComplex_Type, args, kwds));
    if (!c) {
      return nullptr;
    }
    if (CastToCustomComplex<T>(c.get(), &value)) {
      return PyCustomComplex_FromT<T>(value).release();
    }
  } else if (size == 1) {
    PyObject* arg = PyTuple_GetItem(args, 0);

    if (PyCustomComplex_Check<T>(arg)) {
      Py_INCREF(arg);
      return arg;
    } else if (CastToCustomComplex<T>(arg, &value)) {
      return PyCustomComplex_FromT<T>(value).release();
    } else if (PyArray_Check(arg)) {
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
      if (PyArray_TYPE(arr) != CustomComplexType<T>::Dtype()) {
        Py_INCREF(CustomComplexType<T>::npy_descr);
        return PyArray_CastToType(arr, CustomComplexType<T>::npy_descr, 0);
      } else {
        Py_INCREF(arg);
        return arg;
      }
    } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
      // Parse float from string, then cast to T.
      Safe_PyObjectPtr f =
          make_safe(PyComplex_Type.tp_new(&PyComplex_Type, args, kwds));
      if (!f) {
        return nullptr;
      }
      if (CastToCustomComplex<T>(f.get(), &value)) {
        return PyCustomComplex_FromT<T>(value).release();
      }
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number as argument to %s constructor",
               CustomComplexTraits<T>::kTypeName);
  return nullptr;
}

// Comparisons on PyCustomComplexes.
template <typename T>
PyObject* PyCustomComplex_RichCompare(PyObject* a, PyObject* b, int op) {
  T x;
  if (SafeCastToCustomComplex<T>(a, &x)) {
    T y;
    if (CastToCustomComplex<T>(b, &y)) {
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
  }
  if ((op == Py_EQ || op == Py_NE) &&
      (PyUnicode_Check(b) || PyBytes_Check(b) ||
       (!PyNumber_Check(b) && !PyArray_Check(b) && !PySequence_Check(b)))) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  return PyGenericArrType_Type.tp_richcompare(a, b, op);
}

// Implementation of repr() for PyCustomComplex.
template <typename T>
PyObject* PyCustomComplex_Repr(PyObject* self) {
  T x = reinterpret_cast<PyCustomComplex<T>*>(self)->value;
  float real = static_cast<float>(x.real());
  float imag = static_cast<float>(x.imag());
  std::ostringstream s;
  bool print_real = real != 0 || std::signbit(real);
  if (print_real) {
    // Print real part (but not if it's positive zero)
    s << "(" << (std::isnan(real) ? std::abs(real) : real);
    if (!std::signbit(imag) || std::isnan(imag)) {
      s << "+";
    }
  }
  s << (std::isnan(imag) ? std::abs(imag) : imag) << "j";
  if (print_real) {
    s << ")";
  }
  return PyUnicode_FromString(s.str().c_str());
}

// Implementation of str() for PyCustomComplex.
template <typename T>
PyObject* PyCustomComplex_Str(PyObject* self) {
  return PyCustomComplex_Repr<T>(self);
}

#ifndef PyHASH_IMAG  // Made public without _ Python 3.13
#define PyHASH_IMAG 1000003UL
#endif

// _Py_HashDouble changed its prototype for Python 3.10 so we use an overload to
// handle the two possibilities.
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t ComplexHashImpl(Py_hash_t (*hash_double)(PyObject*, double),
                                 PyObject* self, std::complex<double> value) {
  Py_hash_t hashreal = hash_double(self, value.real());
  if (hashreal == -1) {
    return -1;
  }
  Py_hash_t hashimag = hash_double(self, value.imag());
  if (hashimag == -1) {
    return -1;
  }
  Py_hash_t combined =
      static_cast<Py_hash_t>(static_cast<Py_uhash_t>(hashreal) +
                             PyHASH_IMAG * static_cast<Py_uhash_t>(hashimag));
  if (combined == -1) {
    return -2;
  }
  return combined;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t ComplexHashImpl(Py_hash_t (*hash_double)(double),
                                 PyObject* self, std::complex<double> value) {
  Py_hash_t hashreal = hash_double(value.real());
  if (hashreal == -1) {
    return -1;
  }
  Py_hash_t hashimag = hash_double(value.imag());
  if (hashimag == -1) {
    return -1;
  }
  Py_hash_t combined =
      static_cast<Py_hash_t>(static_cast<Py_uhash_t>(hashreal) +
                             PyHASH_IMAG * static_cast<Py_uhash_t>(hashimag));
  if (combined == -1) {
    return -2;
  }
  return combined;
}

// Hash function for PyCustomComplex.
template <typename T>
Py_hash_t PyCustomComplex_Hash(PyObject* self) {
  T x = reinterpret_cast<PyCustomComplex<T>*>(self)->value;
  return ComplexHashImpl(&_Py_HashDouble, self, to_system(x));
}

template <typename T>
PyObject* PyCustomComplex_Real(PyObject* self, PyObject*) {
  typename T::value_type val =
      reinterpret_cast<PyCustomComplex<T>*>(self)->value.real();
  return make_safe(PyArray_Scalar(
                       &val, CustomFloatType<typename T::value_type>::npy_descr,
                       nullptr))
      .release();
}
template <typename T>
PyObject* PyCustomComplex_Imag(PyObject* self, PyObject*) {
  typename T::value_type val =
      reinterpret_cast<PyCustomComplex<T>*>(self)->value.imag();
  return make_safe(PyArray_Scalar(
                       &val, CustomFloatType<typename T::value_type>::npy_descr,
                       nullptr))
      .release();
}

// We need explicit specializations for complex32 to create the NumPy
// owned scalars. (At least unless we define `PyCustomFloat_FromT` for it.)
template <>
PyObject* PyCustomComplex_Real<complex32>(PyObject* self, PyObject*) {
  half val = reinterpret_cast<PyCustomComplex<complex32>*>(self)->value.real();

  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT16);
  auto scalar = make_safe(PyArray_Scalar(&val, descr, NULL));
  Py_DECREF(descr);
  return scalar.release();
}
template <>
PyObject* PyCustomComplex_Imag<complex32>(PyObject* self, PyObject*) {
  half val = reinterpret_cast<PyCustomComplex<complex32>*>(self)->value.imag();

  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT16);
  auto scalar = make_safe(PyArray_Scalar(&val, descr, NULL));
  Py_DECREF(descr);
  return scalar.release();
}

// Format function for PyCustomComplex.
template <typename T>
PyObject* PyCustomComplex_Format(PyObject* self, PyObject* format_spec) {
  if (!PyUnicode_Check(format_spec)) {
    PyErr_Format(PyExc_TypeError, "__format__() argument 1 must be str, not %s",
                 Py_TYPE(format_spec)->tp_name);
    return nullptr;
  }
  PyObject* c = PyCustomComplex_Complex<T>(self, nullptr);
  if (!c) {
    return nullptr;
  }
  PyObject* result = PyObject_Format(c, format_spec);
  Py_DECREF(c);
  return result;
}

}  // namespace

template <typename T>
PyMethodDef CustomComplexType<T>::methods[] = {
    {"__complex__", reinterpret_cast<PyCFunction>(PyCustomComplex_Complex<T>),
     METH_NOARGS, "Convert to Python complex"},
    {"__format__", reinterpret_cast<PyCFunction>(PyCustomComplex_Format<T>),
     METH_O, "Format a custom complex value."},
    {NULL, NULL, 0, NULL}};

template <typename T>
PyGetSetDef CustomComplexType<T>::getset[] = {
    {"real", reinterpret_cast<getter>(PyCustomComplex_Real<T>), NULL, NULL,
     NULL},
    {"imag", reinterpret_cast<getter>(PyCustomComplex_Imag<T>), NULL, NULL,
     NULL},
    {NULL, NULL, NULL, NULL, NULL}};

template <typename T>
PyType_Slot CustomComplexType<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyCustomComplex_New<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyCustomComplex_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyCustomComplex_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyCustomComplex_Str<T>)},
    {Py_tp_doc, reinterpret_cast<void*>(
                    const_cast<char*>(CustomComplexTraits<T>::kTpDoc))},
    {Py_tp_richcompare,
     reinterpret_cast<void*>(PyCustomComplex_RichCompare<T>)},
    {Py_nb_add, reinterpret_cast<void*>(PyCustomComplex_Add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyCustomComplex_Subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyCustomComplex_Multiply<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyCustomComplex_Negative<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyCustomComplex_Int<T>)},
    {Py_nb_float, reinterpret_cast<void*>(PyCustomComplex_Float<T>)},
    {Py_tp_methods, reinterpret_cast<void*>(CustomComplexType<T>::methods)},
    {Py_tp_getset, reinterpret_cast<void*>(CustomComplexType<T>::getset)},
    {0, nullptr},
};

template <typename T>
PyType_Spec CustomComplexType<T>::type_spec = {
    /*.name=*/CustomComplexTraits<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyCustomComplex<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/CustomComplexType<T>::type_slots,
};

template <typename T>
PyArray_ArrFuncs CustomComplexType<T>::numpy_1_arr_funcs;

template <typename T>
PyArray_DescrProto GetNumPy1ComplexDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr)
      /*typeobj=*/nullptr,  // Filled in later
      /*kind=*/'W',
      /*type=*/CustomComplexTraits<T>::kNumPy1DescrType,
      /*byteorder=*/'=',
      /*flags=*/NPY_USE_SETITEM,
      /*type_num=*/0,
      /*elsize=*/sizeof(T),
      /*alignment=*/alignof(T),
      /*subarray=*/nullptr,
      /*fields=*/nullptr,
      /*names=*/nullptr,
      /*f=*/&CustomComplexType<T>::numpy_1_arr_funcs,
      /*metadata=*/nullptr,
      /*c_metadata=*/nullptr,
      /*hash=*/-1,  // -1 means "not computed yet".
  };
}

namespace {

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyCustomComplex_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyComplex_FromCComplex(to_python(static_cast<std::complex<float>>(x)));
}

template <typename T>
int NPyCustomComplex_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (!CastToCustomComplex<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

// TODO: If float ocmpare supports byte-swapping this'll be wrong.
template <typename T>
int NPyCustomComplex_Compare(const void* a, const void* b, void* arr) {
  T x;
  memcpy(&x, a, sizeof(T));

  T y;
  memcpy(&y, b, sizeof(T));

  int res =
      CompareFloats(static_cast<float>(x.real()), static_cast<float>(y.real()));
  if (res != 0) {
    return res;
  }
  return CompareFloats(static_cast<float>(x.imag()),
                       static_cast<float>(y.imag()));
}

template <typename T>
void NPyCustomComplex_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                                npy_intp sstride, npy_intp n, int swap,
                                void* arr) {
  static_assert(sizeof(T) == sizeof(int32_t) || sizeof(T) == sizeof(int16_t),
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
    }
    if (swap && sizeof(T) == sizeof(int32_t)) {
      for (npy_intp i = 0; i < n; i++) {
        char* r = dst + dstride * i;
        memcpy(r, src + sstride * i, sizeof(T));
        ByteSwap32(r);
      }
    } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (npy_intp i = 0; i < n; i++) {
        memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
      }
    }
  } else if (swap) {
    // In-place swap when src is NULL
    if (sizeof(T) == sizeof(int16_t)) {
      for (npy_intp i = 0; i < n; i++) {
        char* r = dst + dstride * i;
        ByteSwap16(r);
      }
    } else if (sizeof(T) == sizeof(int32_t)) {
      for (npy_intp i = 0; i < n; i++) {
        char* r = dst + dstride * i;
        ByteSwap32(r);
      }
    }
  }
}

template <typename T>
void NPyCustomComplex_CopySwap(void* dst, void* src, int swap, void* arr) {
  static_assert(sizeof(T) == sizeof(int32_t) || sizeof(T) == sizeof(int16_t),
                "Not supported");

  if (src) {
    memcpy(dst, src, sizeof(T));
  }
  if (!swap) {
    return;
  }

  if (sizeof(T) == sizeof(int16_t)) {
    ByteSwap16(dst);
  } else if (sizeof(T) == sizeof(int32_t)) {
    ByteSwap32(dst);
  }
}
template <typename T>
npy_bool NPyCustomComplex_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return x.real() != static_cast<decltype(x.real())>(0) ||
         x.imag() != static_cast<decltype(x.imag())>(0);
}

template <typename T>
void NPyCustomComplex_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                              void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  std::complex<float> acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += static_cast<std::complex<float>>(*b1) *
           static_cast<std::complex<float>>(*b2);
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
PyObject* PyCustomComplexDType_GetItem(PyArray_Descr* descr, char* data) {
  return NPyCustomComplex_GetItem<T>(data, nullptr);
}

template <typename T>
int PyCustomComplexDType_SetItem(PyArray_Descr* descr, PyObject* item,
                                 char* data) {
  return NPyCustomComplex_SetItem<T>(item, data, nullptr);
}

static inline PyArray_Descr* PyCustomComplexDType_EnsureCanonical(
    PyArray_Descr* dtype) {
  Py_INCREF(dtype);
  return dtype;
}

template <typename T>
int PyCustomComplexDType_to_CustomComplexDType_resolve_descriptors(
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
int PyCustomComplexDType_to_CustomComplexDType_CastLoop(
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
PyObject* PyCustomComplexDType_New(PyTypeObject* type, PyObject* args,
                                   PyObject* kwds) {
  if ((args == nullptr || PyTuple_Size(args) == 0) &&
      (kwds == nullptr || PyDict_Size(kwds) == 0) &&
      CustomComplexType<T>::npy_descr != nullptr) {
    Py_INCREF(CustomComplexType<T>::npy_descr);
    return reinterpret_cast<PyObject*>(CustomComplexType<T>::npy_descr);
  }
  PyTypeObject* meta_type =
      reinterpret_cast<PyTypeObject*>(CustomComplexType<T>::dtype_meta);
  if (!meta_type) meta_type = type;
  PyObject* obj = PyArrayDescr_Type.tp_new(meta_type, args, kwds);
  if (obj != nullptr) {
    PyArray_Descr* descr = reinterpret_cast<PyArray_Descr*>(obj);
    descr->elsize = sizeof(T);
    descr->alignment = alignof(T);
    descr->kind = 'c';
    descr->type = '?';
    descr->byteorder = '=';
    descr->type_num = CustomComplexType<T>::npy_type;
    descr->flags = NPY_USE_SETITEM;
  }
  return obj;
}

template <typename T>
PyObject* PyCustomComplexDType_Str(PyObject* self) {
  return PyUnicode_FromString(CustomComplexTraits<T>::kTypeName);
}

template <typename T>
PyObject* PyCustomComplexDType_Reduce(PyObject* self) {
  PyObject* name = PyUnicode_FromString(CustomComplexTraits<T>::kTypeName);
  PyObject* dtype_fn = reinterpret_cast<PyObject*>(&PyArrayDescr_Type);
  Py_INCREF(dtype_fn);
  PyObject* res = PyTuple_Pack(2, dtype_fn, PyTuple_Pack(1, name));
  Py_DECREF(name);
  Py_DECREF(dtype_fn);
  return res;
}

template <typename T>
PyObject* PyCustomComplexDType_Repr(PyObject* self) {
  std::string repr =
      std::string("dtype('") + CustomComplexTraits<T>::kTypeName + "')";
  return PyUnicode_FromString(repr.c_str());
}

template <typename T>
PyObject* PyCustomComplexDType_name_get(PyObject* self, void* closure) {
  return PyUnicode_FromString(CustomComplexTraits<T>::kTypeName);
}

template <typename T>
PyArray_DTypeMeta* PyCustomComplexDType_CommonDType(PyArray_DTypeMeta* cls,
                                                    PyArray_DTypeMeta* other) {
  if (other == nullptr || cls == other) {
    Py_INCREF(cls);
    return cls;
  }

  int next_largest_typenum = NPY_COMPLEX64;
  if constexpr (sizeof(T) == 8) {
    next_largest_typenum = NPY_COMPLEX128;
  } else if constexpr (sizeof(T) >= 16) {
    next_largest_typenum = NPY_CLONGDOUBLE;
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
T CastToCustomComplexT(T value) {
  return value;
}

template <typename To, typename From>
To CastToCustomComplexT(From value) {
  if constexpr (is_complex_v<From> && is_complex_v<To>) {
    auto via = static_cast<std::complex<float>>(value);
    return static_cast<To>(via);
  } else if constexpr (is_complex_v<From> && !is_complex_v<To>) {
    if constexpr (std::is_same_v<To, bool>) {
      return static_cast<bool>(value.real()) || static_cast<bool>(value.imag());
    } else {
      if (GiveComplexWarningNoGIL() < 0) {
        return To{};
      }
      auto via = static_cast<float>(value.real());
      return static_cast<To>(via);
    }
  } else if constexpr (!is_complex_v<From> && is_complex_v<To>) {
    auto via = static_cast<float>(value);
    return static_cast<To>(via);
  } else {
    static_assert(is_complex_v<From>);
  }
}

template <typename From, typename To>
int PyCustomComplexCastLoop(PyArrayMethod_Context* context, char* const data[],
                            npy_intp const dimensions[],
                            npy_intp const strides[], NpyAuxData* auxdata) {
  npy_intp N = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  for (npy_intp i = 0; i < N; i++) {
    From f;
    memcpy(&f, in, sizeof(From));
    To t = CastToCustomComplexT<To>(f);
    memcpy(out, &t, sizeof(To));
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

}  // namespace

template <typename From, typename To>
struct CustomComplexCastSpec {
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
PyType_Slot CustomComplexCastSpec<From, To>::slots[3] = {
    {NPY_METH_strided_loop,
     reinterpret_cast<void*>(PyCustomComplexCastLoop<From, To>)},
    {NPY_METH_unaligned_strided_loop,
     reinterpret_cast<void*>(PyCustomComplexCastLoop<From, To>)},
    {0, nullptr}};

template <typename From, typename To>
PyArray_DTypeMeta* CustomComplexCastSpec<From, To>::dtypes[2] = {nullptr,
                                                                 nullptr};

template <typename From, typename To>
PyArrayMethod_Spec CustomComplexCastSpec<From, To>::spec = {
    /*name=*/"customcomplex_cast",
    /*nin=*/1,
    /*nout=*/1,
    /*casting=*/NPY_UNSAFE_CASTING,
    /*flags=*/NPY_METH_SUPPORTS_UNALIGNED,
    /*dtypes=*/dtypes,
    /*slots=*/slots,
};

namespace {

template <typename T, typename OtherT>
bool AddCustomComplexCast(int numpy_type, NPY_CASTING to_safety,
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
  if (!CustomComplexCastSpec<T, OtherT>::Initialize(nullptr, other_meta)) {
    Py_XDECREF(d);
    return false;
  }
  CustomComplexCastSpec<T, OtherT>::spec.casting = to_safety;
  casts.push_back(&CustomComplexCastSpec<T, OtherT>::spec);

  if (!CustomComplexCastSpec<OtherT, T>::Initialize(other_meta, nullptr)) {
    Py_XDECREF(d);
    return false;
  }
  CustomComplexCastSpec<OtherT, T>::spec.casting = from_safety;
  casts.push_back(&CustomComplexCastSpec<OtherT, T>::spec);
  Py_XDECREF(d);
  return true;
}

template <typename T>
bool GetComplexCasts(std::vector<PyArrayMethod_Spec*>& casts) {
  if (!AddCustomComplexCast<T, half>(NPY_HALF, NPY_SAFE_CASTING,
                                     NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float>(NPY_FLOAT, NPY_SAFE_CASTING,
                                      NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, double>(NPY_DOUBLE, NPY_SAFE_CASTING,
                                       NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, long double>(NPY_LONGDOUBLE, NPY_SAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, bool>(NPY_BOOL, NPY_UNSAFE_CASTING,
                                     NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, unsigned char>(NPY_UBYTE, NPY_UNSAFE_CASTING,
                                              NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, unsigned short>(NPY_USHORT, NPY_UNSAFE_CASTING,
                                               NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, unsigned int>(NPY_UINT, NPY_UNSAFE_CASTING,
                                             NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, unsigned long>(NPY_ULONG, NPY_UNSAFE_CASTING,
                                              NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, unsigned long long>(
          NPY_ULONGLONG, NPY_UNSAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, signed char>(NPY_BYTE, NPY_UNSAFE_CASTING,
                                            NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, short>(NPY_SHORT, NPY_UNSAFE_CASTING,
                                      NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, int>(NPY_INT, NPY_UNSAFE_CASTING,
                                    NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, long>(NPY_LONG, NPY_UNSAFE_CASTING,
                                     NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, long long>(NPY_LONGLONG, NPY_UNSAFE_CASTING,
                                          NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, std::complex<float>>(
          NPY_CFLOAT, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, std::complex<double>>(
          NPY_CDOUBLE, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE, NPY_SAFE_CASTING, NPY_UNSAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, bfloat16>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                         NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e3m4>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e4m3>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e4m3b11fnuz>(
          NPY_NOTYPE, NPY_UNSAFE_CASTING, NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e4m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e4m3fnuz>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e5m2>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                            NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e5m2fnuz>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                                NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float8_e8m0fnu>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                               NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float6_e2m3fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float6_e3m2fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_SAFE_CASTING, casts))
    return false;
  if (!AddCustomComplexCast<T, float4_e2m1fn>(NPY_NOTYPE, NPY_UNSAFE_CASTING,
                                              NPY_SAFE_CASTING, casts))
    return false;
  if constexpr (std::is_same_v<T, complex32>) {
    if (!AddCustomComplexCast<T, bcomplex32>(NPY_NOTYPE, NPY_SAME_KIND_CASTING,
                                             NPY_SAME_KIND_CASTING, casts))
      return false;
  }

  return true;
}

template <typename T>
bool RegisterComplexUFuncs(PyObject* numpy, bool use_new_dtype_api) {
#define REG_UFUNC(name, ...) \
  RegisterUFunc<__VA_ARGS__, T>(numpy, name, use_new_dtype_api)
  bool ok =
      REG_UFUNC("add", UFunc<ufuncs::Add<T>, T, T, T>) &&
      REG_UFUNC("subtract", UFunc<ufuncs::Subtract<T>, T, T, T>) &&
      REG_UFUNC("multiply", UFunc<ufuncs::Multiply<T>, T, T, T>) &&
      REG_UFUNC("divide", UFunc<ufuncs::TrueDivide<T>, T, T, T>) &&
      REG_UFUNC("negative", UFunc<ufuncs::Negative<T>, T, T>) &&
      REG_UFUNC("positive", UFunc<ufuncs::Positive<T>, T, T>) &&
      REG_UFUNC("true_divide", UFunc<ufuncs::TrueDivide<T>, T, T, T>) &&
      REG_UFUNC("power", UFunc<ufuncs::Power<T>, T, T, T>) &&
      REG_UFUNC("float_power", UFunc<ufuncs::Power<T>, T, T, T>) &&
      REG_UFUNC("absolute", UFunc<ufuncs::Abs<T>, typename T::value_type, T>) &&
      REG_UFUNC("rint", UFunc<ufuncs::Rint<T>, T, T>) &&
      // NumPy defines the complex signum as z/|z|.
      REG_UFUNC("sign", UFunc<ufuncs::Sign<T>, T, T>) &&
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
      REG_UFUNC("reciprocal", UFunc<ufuncs::Reciprocal<T>, T, T>) &&

      // Trigonometric functions
      REG_UFUNC("sin", UFunc<ufuncs::Sin<T>, T, T>) &&
      REG_UFUNC("cos", UFunc<ufuncs::Cos<T>, T, T>) &&
      REG_UFUNC("tan", UFunc<ufuncs::Tan<T>, T, T>) &&
      REG_UFUNC("arcsin", UFunc<ufuncs::Arcsin<T>, T, T>) &&
      REG_UFUNC("arccos", UFunc<ufuncs::Arccos<T>, T, T>) &&
      REG_UFUNC("arctan", UFunc<ufuncs::Arctan<T>, T, T>) &&
      REG_UFUNC("sinh", UFunc<ufuncs::Sinh<T>, T, T>) &&
      REG_UFUNC("cosh", UFunc<ufuncs::Cosh<T>, T, T>) &&
      REG_UFUNC("tanh", UFunc<ufuncs::Tanh<T>, T, T>) &&
      REG_UFUNC("arcsinh", UFunc<ufuncs::Arcsinh<T>, T, T>) &&
      REG_UFUNC("arccosh", UFunc<ufuncs::Arccosh<T>, T, T>) &&
      REG_UFUNC("arctanh", UFunc<ufuncs::Arctanh<T>, T, T>) &&

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

      // Floating point / logical functions
      REG_UFUNC("logical_not", UFunc<ufuncs::LogicalNot<T>, bool, T>) &&
      REG_UFUNC("logical_and", UFunc<ufuncs::LogicalAnd<T>, bool, T, T>) &&
      REG_UFUNC("logical_or", UFunc<ufuncs::LogicalOr<T>, bool, T, T>) &&
      REG_UFUNC("logical_xor", UFunc<ufuncs::LogicalXor<T>, bool, T, T>) &&
      REG_UFUNC("isfinite", UFunc<ufuncs::IsFinite<T>, bool, T>) &&
      REG_UFUNC("isinf", UFunc<ufuncs::IsInf<T>, bool, T>) &&
      REG_UFUNC("isnan", UFunc<ufuncs::IsNan<T>, bool, T>);
#undef REG_UFUNC
  return ok;
}

template <typename From, typename To>
void NPyComplexCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                    void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = CastToCustomComplexT<To>(from[i]);
  }
}

template <typename T, typename U>
bool RegisterNumPy1ComplexCast(int type_num) {
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);
  if (!descr) {
    return false;
  }
  if (PyArray_RegisterCastFunc(CustomComplexType<T>::npy_descr, type_num,
                               NPyComplexCast<T, U>) < 0) {
    return false;
  }
  if (type_num == NPY_CFLOAT || type_num == NPY_CDOUBLE ||
      type_num == NPY_CLONGDOUBLE) {
    if (PyArray_RegisterCanCast(CustomComplexType<T>::npy_descr, type_num,
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  if (PyArray_RegisterCastFunc(descr, CustomComplexType<T>::npy_type,
                               NPyComplexCast<U, T>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy1ComplexCasts() {
  if (!RegisterNumPy1ComplexCast<T, half>(NPY_HALF)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, unsigned long long>(
          NPY_ULONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterNumPy1ComplexCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE)) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterNumPy1ComplexDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomComplexType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomComplexType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  PyArray_ArrFuncs& arr_funcs = CustomComplexType<T>::numpy_1_arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyCustomComplex_GetItem<T>;
  arr_funcs.setitem = NPyCustomComplex_SetItem<T>;
  arr_funcs.compare = NPyCustomComplex_Compare<T>;
  arr_funcs.copyswapn = NPyCustomComplex_CopySwapN<T>;
  arr_funcs.copyswap = NPyCustomComplex_CopySwap<T>;
  arr_funcs.nonzero = NPyCustomComplex_NonZero<T>;
  arr_funcs.fill = nullptr;
  arr_funcs.dotfunc = NPyCustomComplex_DotFunc<T>;
  arr_funcs.argmax = nullptr;
  arr_funcs.argmin = nullptr;

  PyArray_DescrProto& descr_proto = CustomComplexType<T>::numpy_1_descr_proto;
  descr_proto = GetNumPy1ComplexDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  CustomComplexType<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (CustomComplexType<T>::npy_type < 0) {
    return false;
  }
  CustomComplexType<T>::npy_descr =
      PyArray_DescrFromType(CustomComplexType<T>::npy_type);

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(),
                           CustomComplexTraits<T>::kTypeName,
                           CustomComplexType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomComplexType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomComplexType<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterNumPy1ComplexCasts<T>() &&
         RegisterComplexUFuncs<T>(numpy, /*use_new_dtype_api=*/false);
}

template <typename T>
bool RegisterNumPy2ComplexDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(PyTuple_Pack(
      1, reinterpret_cast<PyObject*>(&PyComplexFloatingArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomComplexType<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  CustomComplexType<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("ml_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(type, "__module__", module.get()) < 0) {
    return false;
  }

  static PyType_Slot slots[] = {
      {NPY_DT_getitem,
       reinterpret_cast<void*>(PyCustomComplexDType_GetItem<T>)},
      {NPY_DT_setitem,
       reinterpret_cast<void*>(PyCustomComplexDType_SetItem<T>)},
      {NPY_DT_ensure_canonical,
       reinterpret_cast<void*>(PyCustomComplexDType_EnsureCanonical)},
      {NPY_DT_PyArray_ArrFuncs_compare,
       reinterpret_cast<void*>(NPyCustomComplex_Compare<T>)},
      {NPY_DT_PyArray_ArrFuncs_nonzero,
       reinterpret_cast<void*>(NPyCustomComplex_NonZero<T>)},
      {NPY_DT_PyArray_ArrFuncs_dotfunc,
       reinterpret_cast<void*>(NPyCustomComplex_DotFunc<T>)},
      {NPY_DT_common_dtype,
       reinterpret_cast<void*>(PyCustomComplexDType_CommonDType<T>)},
      {0, nullptr}};

  static PyType_Slot cast_slots[] = {
      {NPY_METH_resolve_descriptors,
       reinterpret_cast<void*>(
           PyCustomComplexDType_to_CustomComplexDType_resolve_descriptors<T>)},
      {NPY_METH_unaligned_strided_loop,
       reinterpret_cast<void*>(
           PyCustomComplexDType_to_CustomComplexDType_CastLoop<T>)},
      {NPY_METH_strided_loop,
       reinterpret_cast<void*>(
           PyCustomComplexDType_to_CustomComplexDType_CastLoop<T>)},
      {0, nullptr}};

  static PyArray_DTypeMeta* cast_dtypes[2] = {nullptr, nullptr};

  static PyArrayMethod_Spec cast_spec = {
      /*name=*/"customcomplex_to_customcomplex_cast",
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
    bool ok = GetComplexCasts<T>(cast_specs);
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

  if (!CustomComplexType<T>::dtype_meta) {
    CustomComplexType<T>::dtype_meta = reinterpret_cast<PyArray_DTypeMeta*>(
        PyMem_Calloc(1, sizeof(PyArray_DTypeMeta)));
    if (!CustomComplexType<T>::dtype_meta) return false;
  }
  PyArray_DTypeMeta* dtype_meta = CustomComplexType<T>::dtype_meta;

  PyTypeObject* tm = reinterpret_cast<PyTypeObject*>(dtype_meta);
  Py_SET_TYPE(tm, &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(tm, 1);
  tm->tp_name = CustomComplexTraits<T>::kQualifiedTypeName;
  tm->tp_basicsize = sizeof(PyArray_Descr);
  tm->tp_base = &PyArrayDescr_Type;
  tm->tp_new = PyCustomComplexDType_New<T>;
  tm->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

  static PyGetSetDef dtype_getset[] = {
      {const_cast<char*>("name"),
       reinterpret_cast<getter>(PyCustomComplexDType_name_get<T>), nullptr,
       nullptr, nullptr},
      {nullptr, nullptr, nullptr, nullptr, nullptr}};
  tm->tp_repr = PyCustomComplexDType_Repr<T>;
  tm->tp_str = PyCustomComplexDType_Str<T>;
  tm->tp_getset = dtype_getset;

  static PyMethodDef dtype_methods[] = {
      {const_cast<char*>("__reduce__"),
       reinterpret_cast<PyCFunction>(PyCustomComplexDType_Reduce<T>),
       METH_NOARGS, nullptr},
      {nullptr, nullptr, 0, nullptr}};
  tm->tp_methods = dtype_methods;

  if (PyType_Ready(tm) < 0) {
    return false;
  }

  if (PyArrayInitDTypeMeta_FromSpec(dtype_meta, &spec) < 0) {
    return false;
  }

  CustomComplexType<T>::npy_type = dtype_meta->type_num;

  CustomComplexType<T>::npy_descr = PyArray_GetDefaultDescr(dtype_meta);
  if (!CustomComplexType<T>::npy_descr) return false;
  PyDataType_GetArrFuncs(CustomComplexType<T>::npy_descr)->copyswap =
      NPyCustomComplex_CopySwap<T>;
  PyDataType_GetArrFuncs(CustomComplexType<T>::npy_descr)->copyswapn =
      NPyCustomComplex_CopySwapN<T>;

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(),
                           CustomComplexTraits<T>::kTypeName,
                           CustomComplexType<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          CustomComplexType<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(CustomComplexType<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterComplexUFuncs<T>(numpy, /*use_new_dtype_api=*/true);
}

template <typename T>
bool RegisterComplexDtype(PyObject* numpy, bool use_new_dtype_api) {
  if (use_new_dtype_api) {
    return RegisterNumPy2ComplexDtype<T>(numpy);
  } else {
    return RegisterNumPy1ComplexDtype<T>(numpy);
  }
}

}  // namespace

bool RegisterComplexDtypes(PyObject* numpy, bool use_new_dtype_api) {
  return RegisterComplexDtype<bcomplex32>(numpy, use_new_dtype_api) &&
         RegisterComplexDtype<complex32>(numpy, use_new_dtype_api);
}

}  // namespace ml_dtypes
