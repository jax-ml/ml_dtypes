/* Copyright 2026 The ml_dtypes Authors

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

#ifndef ML_DTYPES_CUSTOM_COMPLEX_H_
#define ML_DTYPES_CUSTOM_COMPLEX_H_

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
#include "ml_dtypes/include/complex_types.h"

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace ml_dtypes {

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
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;
};

template <typename T>
int CustomComplexType<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* CustomComplexType<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto CustomComplexType<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* CustomComplexType<T>::npy_descr = nullptr;

// Representation of a Python custom float object.
template <typename T>
struct PyCustomComplex {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyCustomComplex.
template <typename T>
bool PyCustomComplex_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
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
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
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
    output->real(static_cast<real_type>(d));
    return true;
  }
  if (PyLong_Check(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    output->real(static_cast<real_type>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Generic)) {
    // Allow conversion from any NumPy scalar if conversion to complex float
    // is defined.
    // NOTE: Should use `PyArray_Pack` with NumPy>=2, which is better and may
    // make even more conversions (ie. casts) work. (May want to use new dtypes
    // then also.) (If a limitation is found, could do this already on NumPy 2
    // at runtime.)
    std::complex<float> c;
    PyArray_Descr* cf_descr = PyArray_DescrFromType(NPY_COMPLEX64);
    // Similar to our code, NumPy accepts the array to be NULL here.
    // TODO(phawkins): check for overflow
    PyDataType_GetArrFuncs(cf_descr)->setitem(arg, &c, NULL);
    Py_DECREF(cf_descr);
    *output = T(c);
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
    return PyCustomComplex_FromT<T>(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyCustomComplex_TrueDivide(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomComplex<T>(a, &x) && SafeCastToCustomComplex<T>(b, &y)) {
    return PyCustomComplex_FromT<T>(x / y).release();
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
  } else if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "expected number as argument to %s constructor",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  if (PyCustomComplex_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToCustomComplex<T>(arg, &value)) {
    return PyCustomComplex_FromT<T>(value).release();
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
    Safe_PyObjectPtr f =
        make_safe(PyComplex_Type.tp_new(&PyComplex_Type, args, kwds));
    if (!f) {
      return nullptr;
    }
    if (CastToCustomComplex<T>(f.get(), &value)) {
      return PyCustomComplex_FromT<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

// Comparisons on PyCustomComplexes.
template <typename T>
PyObject* PyCustomComplex_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!SafeCastToCustomComplex<T>(a, &x) ||
      !SafeCastToCustomComplex<T>(b, &y)) {
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
  T x = reinterpret_cast<PyCustomComplex<T>*>(self)->value;
  return PyCustomFloat_FromT(x.real()).release();
}
template <typename T>
PyObject* PyCustomComplex_Imag(PyObject* self, PyObject*) {
  T x = reinterpret_cast<PyCustomComplex<T>*>(self)->value;
  return PyCustomFloat_FromT(x.imag()).release();
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

template <typename T>
PyMethodDef CustomComplexType<T>::methods[] = {
    {"__complex__", reinterpret_cast<PyCFunction>(PyCustomComplex_Complex<T>),
     METH_NOARGS, "Convert to Python complex"},
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
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
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
    /*.name=*/TypeDescriptor<T>::kQualifiedTypeName,
    /*.basicsize=*/static_cast<int>(sizeof(PyCustomComplex<T>)),
    /*.itemsize=*/0,
    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /*.slots=*/CustomComplexType<T>::type_slots,
};

// Numpy support
template <typename T>
PyArray_ArrFuncs CustomComplexType<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetCustomComplexDescrProto() {
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
      /*f=*/&CustomComplexType<T>::arr_funcs,
      /*metadata=*/nullptr,
      /*c_metadata=*/nullptr,
      /*hash=*/-1,  // -1 means "not computed yet".
  };
}

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

template <typename T>
int NPyCustomComplex_Compare(const void* a, const void* b, void* arr) {
  using real_type = typename T::value_type;
  // TODO: If float ocmpare supports byte-swapping this'll be wrong.
  int res = NPyCustomFloat_Compare<real_type>(a, b, arr);
  if (res != 0) {
    return res;
  }
  a = reinterpret_cast<const char*>(a) + sizeof(real_type);
  b = reinterpret_cast<const char*>(b) + sizeof(real_type);
  return NPyCustomFloat_Compare<real_type>(a, b, arr);
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
        ByteSwap16(r);
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
  return x != static_cast<T>(0);
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
int NPyCustomComplex_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1.real() < b2.real()) {
    return -1;
  }
  if (b1.real() > b2.real()) {
    return 1;
  }
  if (b1.imag() < b2.imag()) {
    return -1;
  }
  if (b1.imag() > b2.imag()) {
    return 1;
  }
  return 0;
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
              void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    // TODO(seberg): Casts from complex to float are dubious anyway, maybe error
    // if imaginary (rather than warn?)
    if constexpr (is_complex_v<From> && is_complex_v<To>) {
      auto via = static_cast<std::complex<float>>(from[i]);
      to[i] = static_cast<typename TypeDescriptor<To>::T>(via);
    } else if constexpr (is_complex_v<From> && !is_complex_v<To>) {
      if (GiveComplexWarningNoGIL() < 0) {
        return;
      }
      auto via = static_cast<float>(from[i].real());
      to[i] = static_cast<typename TypeDescriptor<To>::T>(via);
    } else if constexpr (!is_complex_v<From> && is_complex_v<To>) {
      auto via = static_cast<float>(from[i]);
      to[i] = static_cast<typename TypeDescriptor<To>::T>(via);
    } else {
      static_assert(0);
    }
  }
}

// Registers a cast between T (a reduced complex) and type 'OtherT'.
// 'numpy_type' is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterCustomComplexCast(
    int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               NPyCCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(CustomComplexType<T>::npy_descr, numpy_type,
                               NPyCCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterComplexCasts() {
  if (!RegisterCustomComplexCast<T, half>(NPY_HALF)) {
    return false;
  }

  if (!RegisterCustomComplexCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomComplexCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomComplexCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomComplexCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomComplexCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomComplexCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterCustomComplexCast<T, std::complex<long double>>(
          NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
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
bool RegisterComplexUFuncs(PyObject* numpy) {
  bool ok =
      RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
      RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                            "subtract") &&
      RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                            "multiply") &&
      RegisterUFunc<UFunc<ufuncs::TrueDivide<T>, T, T, T>, T>(numpy,
                                                              "divide") &&
      RegisterUFunc<UFunc<ufuncs::Negative<T>, T, T>, T>(numpy, "negative") &&
      RegisterUFunc<UFunc<ufuncs::Positive<T>, T, T>, T>(numpy, "positive") &&
      RegisterUFunc<UFunc<ufuncs::TrueDivide<T>, T, T, T>, T>(numpy,
                                                              "true_divide") &&
      RegisterUFunc<UFunc<ufuncs::Power<T>, T, T, T>, T>(numpy, "power") &&
      RegisterUFunc<UFunc<ufuncs::Abs<T>, typename T::value_type, T>, T>(
          numpy, "absolute") &&
      RegisterUFunc<UFunc<ufuncs::Rint<T>, T, T>, T>(numpy, "rint") &&
      // NumPy defines the complex signum as z/|z|.
      RegisterUFunc<UFunc<ufuncs::Sign<T>, T, T>, T>(numpy, "sign") &&
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
      RegisterUFunc<UFunc<ufuncs::Reciprocal<T>, T, T>, T>(numpy,
                                                           "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UFunc<ufuncs::Sin<T>, T, T>, T>(numpy, "sin") &&
      RegisterUFunc<UFunc<ufuncs::Cos<T>, T, T>, T>(numpy, "cos") &&
      RegisterUFunc<UFunc<ufuncs::Tan<T>, T, T>, T>(numpy, "tan") &&
      RegisterUFunc<UFunc<ufuncs::Arcsin<T>, T, T>, T>(numpy, "arcsin") &&
      RegisterUFunc<UFunc<ufuncs::Arccos<T>, T, T>, T>(numpy, "arccos") &&
      RegisterUFunc<UFunc<ufuncs::Arctan<T>, T, T>, T>(numpy, "arctan") &&
      RegisterUFunc<UFunc<ufuncs::Sinh<T>, T, T>, T>(numpy, "sinh") &&
      RegisterUFunc<UFunc<ufuncs::Cosh<T>, T, T>, T>(numpy, "cosh") &&
      RegisterUFunc<UFunc<ufuncs::Tanh<T>, T, T>, T>(numpy, "tanh") &&
      RegisterUFunc<UFunc<ufuncs::Arcsinh<T>, T, T>, T>(numpy, "arcsinh") &&
      RegisterUFunc<UFunc<ufuncs::Arccosh<T>, T, T>, T>(numpy, "arccosh") &&
      RegisterUFunc<UFunc<ufuncs::Arctanh<T>, T, T>, T>(numpy, "arctanh") &&

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
      RegisterUFunc<UFunc<ufuncs::IsNan<T>, bool, T>, T>(numpy, "isnan");

  return ok;
}

template <typename T>
bool RegisterComplexDtype(PyObject* numpy) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
  // TODO(jakevdp): it would be better to inherit from PyNumberArrType or
  // PyFloatingArrType, but this breaks some assumptions made by NumPy, because
  // dtype.kind='V' is then interpreted as a 'void' type in some contexts.
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&CustomComplexType<T>::type_spec, bases.get());
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
  PyArray_ArrFuncs& arr_funcs = CustomComplexType<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyCustomComplex_GetItem<T>;
  arr_funcs.setitem = NPyCustomComplex_SetItem<T>;
  arr_funcs.compare = NPyCustomComplex_Compare<T>;
  arr_funcs.copyswapn = NPyCustomComplex_CopySwapN<T>;
  arr_funcs.copyswap = NPyCustomComplex_CopySwap<T>;
  arr_funcs.nonzero = NPyCustomComplex_NonZero<T>;
  arr_funcs.fill = nullptr;  // NPyCustomComplex_Fill<T>;
  arr_funcs.dotfunc = NPyCustomComplex_DotFunc<T>;
  arr_funcs.compare = NPyCustomComplex_CompareFunc<T>;
  arr_funcs.argmax = nullptr;  // NumPy defines them, but it's shaky
  arr_funcs.argmin = nullptr;

  // This is messy, but that's because the NumPy 2.0 API transition is messy.
  // Before 2.0, NumPy assumes we'll keep the descriptor passed in to
  // RegisterDataType alive, because it stores its pointer.
  // After 2.0, the proto and descriptor types diverge, and NumPy allocates
  // and manages the lifetime of the descriptor itself.
  PyArray_DescrProto& descr_proto = CustomComplexType<T>::npy_descr_proto;
  descr_proto = GetCustomComplexDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) {
    return false;
  }

  // TODO(phawkins): We intentionally leak the pointer to the descriptor.
  // Implement a better module destructor to handle this.
  CustomComplexType<T>::npy_descr =
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
          reinterpret_cast<PyObject*>(CustomComplexType<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterComplexCasts<T>() && RegisterComplexUFuncs<T>(numpy);
}

}  // namespace ml_dtypes

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif

#endif  // ML_DTYPES_CUSTOM_COMPLEX_H_
