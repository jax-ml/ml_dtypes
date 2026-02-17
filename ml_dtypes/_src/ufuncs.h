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

#ifndef ML_DTYPES_UFUNCS_H_
#define ML_DTYPES_UFUNCS_H_

#include <array>
#include <bitset>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

// Some versions of MSVC define a "copysign" macro which wreaks havoc.
#if defined(_MSC_VER) && defined(copysign)
#undef copysign
#endif

namespace ml_dtypes {

template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline float to_system(const T& value) {
  return static_cast<float>(value);
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline std::complex<float> to_system(const T& value) {
  return static_cast<std::complex<float>>(value);
}

// isnan definition that works for all of our float and complex types.
template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline bool my_isnan(const T& value) {
  if constexpr (std::is_integral_v<T> || is_intn_v<T>) {
    return false;
  } else {
    return Eigen::numext::isnan(value);
  }
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline bool my_isnan(const T& value) {
  return my_isnan(value.real()) || my_isnan(value.imag());
}

// isfinite definition that works for all of our float and complex types.
template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline bool my_isfinite(const T& value) {
  if constexpr (std::is_integral_v<T> || is_intn_v<T>) {
    return true;
  } else {
    return Eigen::numext::isfinite(value);
  }
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline bool my_isfinite(const T& value) {
  return my_isfinite(value.real()) && my_isfinite(value.imag());
}

// isinf definition that works for all of our float and complex types.
template <typename T, std::enable_if_t<!is_complex_v<T>, bool> = false>
inline bool my_isinf(const T& value) {
  if constexpr (std::is_integral_v<T> || is_intn_v<T>) {
    return false;
  } else {
    return Eigen::numext::isinf(value);
  }
}
template <typename T, std::enable_if_t<is_complex_v<T>, bool> = false>
inline bool my_isinf(const T& value) {
  return my_isinf(value.real()) || my_isinf(value.imag());
}

struct UFuncLoop {
  virtual ~UFuncLoop() = default;
  virtual void Execute(char** args, npy_intp const* dimensions,
                       npy_intp const* steps, void* data) = 0;
};

// Helper to get DTypeMeta for a type T
template <typename T>
PyArray_DTypeMeta* GetDTypeMeta(std::vector<PyObject*>& dtypes_to_decref) {
  if constexpr (is_custom_float_v<T>) {
    return CustomFloatType<T>::dtype_meta;
  } else if constexpr (is_intn_v<T>) {
    return CustomIntType<T>::dtype_meta;
  } else if constexpr (is_custom_complex_v<T>) {
    return CustomComplexType<T>::dtype_meta;
  } else {
    int type_num = DtypeTraits<T>::Dtype();
    if (type_num != NPY_NOTYPE && type_num != -1) {
      PyArray_Descr* descr = PyArray_DescrFromType(type_num);
      if (descr) {
        PyObject* dtype = reinterpret_cast<PyObject*>(Py_TYPE(descr));
        Py_INCREF(dtype);
        dtypes_to_decref.push_back(dtype);
        Py_DECREF(descr);
        return reinterpret_cast<PyArray_DTypeMeta*>(dtype);
      }
    }
    return nullptr;
  }
}

template <typename T, typename = void>
struct is_logical_ufunc : std::false_type {};

template <typename T>
struct is_logical_ufunc<T, std::void_t<decltype(T::is_logical)>>
    : std::integral_constant<bool, T::is_logical> {};

template <typename Functor>
inline constexpr bool is_logical_ufunc_v = is_logical_ufunc<Functor>::value;

template <typename T, typename = void>
struct is_reorderable_ufunc : std::false_type {};

template <typename T>
struct is_reorderable_ufunc<T, std::void_t<decltype(T::is_reorderable)>>
    : std::integral_constant<bool, T::is_reorderable> {};

template <typename Functor>
inline constexpr bool is_reorderable_ufunc_v =
    is_reorderable_ufunc<Functor>::value;

template <typename T, typename = void>
struct has_custom_promoter : std::false_type {};

template <typename T>
struct has_custom_promoter<T, std::void_t<decltype(&T::Promoter)>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_custom_promoter_v = has_custom_promoter<T>::value;

template <typename Functor, typename OutType, typename... InTypes>
struct UFunc {
  using ReturnType = OutType;
  using FunctorType = Functor;
  static constexpr bool kIsLogical = is_logical_ufunc_v<Functor>;
  static constexpr bool kIsReorderable = is_reorderable_ufunc_v<Functor>;
  static constexpr bool kHasCustomPromoter = has_custom_promoter_v<Functor>;
  static constexpr bool kIsHomogeneous =
      (sizeof...(InTypes) <= 1) ||
      (std::is_same_v<std::tuple_element_t<0, std::tuple<InTypes...>>,
                      InTypes> &&
       ...);
  using InTypesTuple = std::tuple<InTypes...>;
  using ResultTypesTuple = std::tuple<OutType>;
  static std::vector<int> Types() {
    return {DtypeTraits<InTypes>::Dtype()..., DtypeTraits<OutType>::Dtype()};
  }
  static constexpr int kInputArity = sizeof...(InTypes);

  template <std::size_t... Is>
  static void CallImpl(std::index_sequence<Is...>, char** args,
                       const npy_intp* dimensions, const npy_intp* steps,
                       void* data) {
    std::array<const char*, kInputArity> inputs = {args[Is]...};
    char* o = args[kInputArity];
    for (npy_intp k = 0; k < *dimensions; k++) {
      *reinterpret_cast<OutType*>(o) =
          Functor()(*reinterpret_cast<const InTypes*>(inputs[Is])...);
      ([&]() { inputs[Is] += steps[Is]; }(), ...);
      o += steps[kInputArity];
    }
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    return CallImpl(std::index_sequence_for<InTypes...>(), args, dimensions,
                    steps, data);
  }
  static int Call_Numpy2(PyArrayMethod_Context* context, char* const* args,
                         const npy_intp* dimensions, const npy_intp* steps,
                         NpyAuxData* data) {
    CallImpl(std::index_sequence_for<InTypes...>(), const_cast<char**>(args),
             dimensions, steps, nullptr);
    return 0;
  }
};

template <typename Functor, typename OutType, typename OutType2,
          typename... InTypes>
struct UFunc2 {
  using ReturnType = OutType;
  using ReturnType2 = OutType2;
  using FunctorType = Functor;
  static constexpr bool kIsLogical = is_logical_ufunc_v<Functor>;
  static constexpr bool kIsReorderable = is_reorderable_ufunc_v<Functor>;
  static constexpr bool kHasCustomPromoter = has_custom_promoter_v<Functor>;
  static constexpr bool kIsHomogeneous =
      (sizeof...(InTypes) <= 1) ||
      (std::is_same_v<std::tuple_element_t<0, std::tuple<InTypes...>>,
                      InTypes> &&
       ...);
  using InTypesTuple = std::tuple<InTypes...>;
  using ResultTypesTuple = std::tuple<OutType, OutType2>;
  static std::vector<int> Types() {
    return {
        DtypeTraits<InTypes>::Dtype()...,
        DtypeTraits<OutType>::Dtype(),
        DtypeTraits<OutType2>::Dtype(),
    };
  }
  static constexpr int kInputArity = sizeof...(InTypes);

  template <std::size_t... Is>
  static void CallImpl(std::index_sequence<Is...>, char** args,
                       const npy_intp* dimensions, const npy_intp* steps,
                       void* data) {
    std::array<const char*, kInputArity> inputs = {args[Is]...};
    char* o0 = args[kInputArity];
    char* o1 = args[kInputArity + 1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      std::tie(*reinterpret_cast<OutType*>(o0),
               *reinterpret_cast<OutType2*>(o1)) =
          Functor()(*reinterpret_cast<const InTypes*>(inputs[Is])...);
      ([&]() { inputs[Is] += steps[Is]; }(), ...);
      o0 += steps[kInputArity];
      o1 += steps[kInputArity + 1];
    }
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    return CallImpl(std::index_sequence_for<InTypes...>(), args, dimensions,
                    steps, data);
  }
  static int Call_Numpy2(PyArrayMethod_Context* context, char* const* args,
                         const npy_intp* dimensions, const npy_intp* steps,
                         NpyAuxData* data) {
    CallImpl(std::index_sequence_for<InTypes...>(), const_cast<char**>(args),
             dimensions, steps, nullptr);
    return 0;
  }
};

inline int GetReductionInitial(PyArrayMethod_Context* context,
                               npy_bool reduction_is_empty, void* initial) {
  if (context->caller == nullptr ||
      !PyObject_TypeCheck(context->caller, &PyUFunc_Type)) {
    return 0;
  }
  auto* ufunc = reinterpret_cast<PyUFuncObject*>(context->caller);
  PyObject* identity_obj = nullptr;
  switch (ufunc->identity) {
    case PyUFunc_Zero:
      identity_obj = PyLong_FromLong(0);
      break;
    case PyUFunc_One:
      identity_obj = PyLong_FromLong(1);
      break;
    case PyUFunc_MinusOne:
      identity_obj = PyLong_FromLong(-1);
      break;
    case PyUFunc_IdentityValue:
      identity_obj = ufunc->identity_value;
      Py_XINCREF(identity_obj);
      break;
    default:
      return 0;
  }
  if (identity_obj == nullptr || identity_obj == Py_None) {
    Py_XDECREF(identity_obj);
    return 0;
  }
  int res = PyArray_Pack(context->descriptors[0], initial, identity_obj);
  Py_DECREF(identity_obj);
  if (res < 0) {
    return -1;
  }
  return 1;
}

template <typename T>
PyArray_DTypeMeta* GetCustomDTypeMeta() {
  if constexpr (is_custom_float_v<T>) {
    return CustomFloatType<T>::dtype_meta;
  } else if constexpr (is_intn_v<T>) {
    return CustomIntType<T>::dtype_meta;
  } else if constexpr (is_custom_complex_v<T>) {
    return CustomComplexType<T>::dtype_meta;
  } else {
    return nullptr;
  }
}

inline int DefaultHomogeneousPromoter(PyObject* ufunc,
                                      PyArray_DTypeMeta* const op_dtypes[],
                                      PyArray_DTypeMeta* const signature[],
                                      PyArray_DTypeMeta* new_op_dtypes[]) {
  PyUFuncObject* ufunc_obj = reinterpret_cast<PyUFuncObject*>(ufunc);
  std::vector<PyArray_DTypeMeta*> non_null_dtypes;
  non_null_dtypes.reserve(ufunc_obj->nin);
  for (int i = 0; i < ufunc_obj->nin; ++i) {
    if (op_dtypes[i] != nullptr) {
      non_null_dtypes.push_back(op_dtypes[i]);
    }
  }
  if (non_null_dtypes.empty()) {
    return -1;
  }
  PyArray_DTypeMeta* common = nullptr;
  if (signature != nullptr) {
    for (int i = ufunc_obj->nin; i < ufunc_obj->nargs; ++i) {
      if (signature[i] != nullptr) {
        if (common == nullptr) {
          Py_INCREF(signature[i]);
          common = signature[i];
        } else if (common != signature[i]) {
          Py_CLEAR(common);
          break;
        }
      }
    }
  }

  if (common == nullptr) {
    common = PyArray_PromoteDTypeSequence(non_null_dtypes.size(),
                                          non_null_dtypes.data());
    if (common == nullptr) {
      if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
      }
      return -1;
    }
  }

  for (int i = 0; i < ufunc_obj->nargs; ++i) {
    PyArray_DTypeMeta* tmp = common;
    if (signature != nullptr && signature[i] != nullptr) {
      tmp = signature[i];
    }
    Py_INCREF(tmp);
    new_op_dtypes[i] = tmp;
  }
  Py_DECREF(common);
  return 0;
}

inline int DefaultComparisonPromoter(PyObject* ufunc,
                                     PyArray_DTypeMeta* const op_dtypes[],
                                     PyArray_DTypeMeta* const signature[],
                                     PyArray_DTypeMeta* new_op_dtypes[]) {
  PyUFuncObject* ufunc_obj = reinterpret_cast<PyUFuncObject*>(ufunc);
  std::vector<PyArray_DTypeMeta*> non_null_dtypes;
  non_null_dtypes.reserve(ufunc_obj->nin);
  for (int i = 0; i < ufunc_obj->nin; ++i) {
    if (op_dtypes[i] != nullptr) {
      non_null_dtypes.push_back(op_dtypes[i]);
    }
  }
  if (non_null_dtypes.empty()) {
    return -1;
  }
  PyArray_DTypeMeta* common = nullptr;
  common = PyArray_PromoteDTypeSequence(non_null_dtypes.size(),
                                        non_null_dtypes.data());
  if (common == nullptr) {
    if (PyErr_ExceptionMatches(PyExc_TypeError)) {
      PyErr_Clear();
    }
    return -1;
  }

  for (int i = 0; i < ufunc_obj->nin; ++i) {
    PyArray_DTypeMeta* tmp = common;
    if (signature != nullptr && signature[i] != nullptr) {
      tmp = signature[i];
    }
    Py_INCREF(tmp);
    new_op_dtypes[i] = tmp;
  }
  for (int i = ufunc_obj->nin; i < ufunc_obj->nargs; ++i) {
    PyArray_DTypeMeta* tmp = &PyArray_BoolDType;
    if (signature != nullptr && signature[i] != nullptr) {
      tmp = signature[i];
    }
    Py_INCREF(tmp);
    new_op_dtypes[i] = tmp;
  }
  Py_DECREF(common);
  return 0;
}

template <typename T>
struct type_identity {
  using type = T;
};

inline Safe_PyObjectPtr GetUFunc(PyObject* numpy, const char* name) {
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (ufunc_obj && PyObject_TypeCheck(ufunc_obj.get(), &PyUFunc_Type)) {
    return ufunc_obj;
  }
  PyErr_Clear();
  Safe_PyObjectPtr core = make_safe(PyObject_GetAttrString(numpy, "_core"));
  if (core) {
    Safe_PyObjectPtr umath =
        make_safe(PyObject_GetAttrString(core.get(), "umath"));
    if (umath) {
      ufunc_obj = make_safe(PyObject_GetAttrString(umath.get(), name));
      if (ufunc_obj && PyObject_TypeCheck(ufunc_obj.get(), &PyUFunc_Type)) {
        return ufunc_obj;
      }
    }
  }
  if (!ufunc_obj) {
    PyErr_Format(PyExc_AttributeError, "Could not find ufunc '%s'", name);
  } else if (!PyObject_TypeCheck(ufunc_obj.get(), &PyUFunc_Type)) {
    PyErr_Format(PyExc_TypeError, "'%s' is not a ufunc", name);
    return nullptr;
  }
  return ufunc_obj;
}

using PromoterFunc = int (*)(PyObject*, PyArray_DTypeMeta* const[],
                             PyArray_DTypeMeta* const[], PyArray_DTypeMeta*[]);

inline bool RegisterUFuncImpl(
    PyObject* numpy, const char* name, bool use_new_dtype_api, int nin,
    int nout, bool is_reorderable, bool is_logical, bool is_bool_output,
    bool is_homogeneous, const std::vector<int>& types, int custom_type_num,
    const std::vector<PyArray_DTypeMeta*>& spec_dtypes,
    PyUFuncGenericFunction fn, PyArrayMethod_StridedLoop* loop_numpy2,
    PromoterFunc custom_promoter = nullptr) {
  Safe_PyObjectPtr ufunc_obj = GetUFunc(numpy, name);
  if (!ufunc_obj) {
    if (!use_new_dtype_api) {
      PyErr_Clear();
      return true;
    }
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());

  if (!use_new_dtype_api) {
    if (static_cast<int>(types.size()) != ufunc->nargs) {
      PyErr_Format(PyExc_AssertionError,
                   "ufunc %s takes %d arguments, loop takes %lu", name,
                   ufunc->nargs, types.size());
      return false;
    }
    if (PyUFunc_RegisterLoopForType(ufunc, custom_type_num, fn,
                                    const_cast<int*>(types.data()),
                                    nullptr) < 0) {
      return false;
    }
    return true;
  }

  const int expected_arity = ufunc->nin + ufunc->nout;
  if (static_cast<int>(spec_dtypes.size()) != expected_arity) {
    PyErr_Format(PyExc_RuntimeError,
                 "RegisterUFunc(%s) failed: arity mismatch %d != %d", name,
                 static_cast<int>(spec_dtypes.size()), expected_arity);
    return false;
  }

  PyType_Slot slots[] = {
      {NPY_METH_strided_loop, reinterpret_cast<void*>(loop_numpy2)},
      {NPY_METH_get_reduction_initial,
       reinterpret_cast<void*>(GetReductionInitial)},
      {0, nullptr}};

  PyArrayMethod_Spec spec;
  memset(&spec, 0, sizeof(spec));
  spec.name = name;
  spec.nin = ufunc->nin;
  spec.nout = ufunc->nout;
  spec.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      is_reorderable ? NPY_METH_IS_REORDERABLE : 0);
  spec.dtypes = const_cast<PyArray_DTypeMeta**>(spec_dtypes.data());
  spec.slots = slots;
  static std::set<std::pair<PyUFuncObject*, std::vector<PyArray_DTypeMeta*>>>
      registered_loops;
  auto loop_key = std::make_pair(ufunc, spec_dtypes);
  if (!registered_loops.count(loop_key)) {
    registered_loops.insert(loop_key);
    if (PyUFunc_AddLoopFromSpec(ufunc_obj.get(), &spec) < 0) {
      return false;
    }
  }

  PyObject* dtype_meta_obj = nullptr;
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == custom_type_num) {
      dtype_meta_obj = reinterpret_cast<PyObject*>(spec_dtypes[i]);
      break;
    }
  }

  static std::set<std::pair<PyUFuncObject*, PyObject*>> registered_promoters;
  auto promoter_key = std::make_pair(ufunc, dtype_meta_obj);

  if (!is_logical && is_homogeneous) {
    if (dtype_meta_obj && ufunc->nin >= 2) {
      if (!registered_promoters.count(promoter_key)) {
        registered_promoters.insert(promoter_key);
        int num_combinations = (1 << ufunc->nin) - 1;
        std::vector<int> masks;
        for (int mask = 1; mask <= num_combinations; ++mask) {
          masks.push_back(mask);
        }
        std::sort(masks.begin(), masks.end(), [](int a, int b) {
          auto count_a = std::bitset<32>(a).count();
          auto count_b = std::bitset<32>(b).count();
          return count_a != count_b ? count_a > count_b : a > b;
        });
        for (int mask : masks) {
          PyObject* DType_tuple = PyTuple_New(ufunc->nargs);
          for (int i = 0; i < ufunc->nin; ++i) {
            if ((mask >> i) & 1) {
              PyTuple_SET_ITEM(DType_tuple, i, dtype_meta_obj);
              Py_INCREF(dtype_meta_obj);
            } else {
              PyTuple_SET_ITEM(DType_tuple, i, Py_None);
              Py_INCREF(Py_None);
            }
          }
          for (int i = ufunc->nin; i < ufunc->nargs; ++i) {
            PyTuple_SET_ITEM(DType_tuple, i, Py_None);
            Py_INCREF(Py_None);
          }
          void* promoter_fn =
              is_bool_output
                  ? reinterpret_cast<void*>(&DefaultComparisonPromoter)
                  : reinterpret_cast<void*>(&DefaultHomogeneousPromoter);
          PyObject* promoter =
              PyCapsule_New(promoter_fn, "numpy._ufunc_promoter", nullptr);
          if (PyUFunc_AddPromoter(ufunc_obj.get(), DType_tuple, promoter) < 0) {
            Py_DECREF(DType_tuple);
            Py_DECREF(promoter);
            return false;
          }
          Py_DECREF(DType_tuple);
          Py_DECREF(promoter);
        }
      }
    }
  }

  if (custom_promoter != nullptr) {
    if (dtype_meta_obj) {
      if (!registered_promoters.count(promoter_key)) {
        registered_promoters.insert(promoter_key);
        PyObject* DType_tuple = PyTuple_New(ufunc->nargs);
        PyTuple_SET_ITEM(DType_tuple, 0, dtype_meta_obj);
        Py_INCREF(dtype_meta_obj);
        for (int i = 1; i < ufunc->nargs; ++i) {
          PyTuple_SET_ITEM(DType_tuple, i, Py_None);
          Py_INCREF(Py_None);
        }
        PyObject* promoter =
            PyCapsule_New(reinterpret_cast<void*>(custom_promoter),
                          "numpy._ufunc_promoter", nullptr);
        if (PyUFunc_AddPromoter(ufunc_obj.get(), DType_tuple, promoter) < 0) {
          Py_DECREF(DType_tuple);
          Py_DECREF(promoter);
          return false;
        }
        Py_DECREF(DType_tuple);
        Py_DECREF(promoter);
      }
    }
  }

  return true;
}

template <typename UFuncT, typename CustomT>
bool RegisterUFunc(PyObject* numpy, const char* name, bool use_new_dtype_api) {
  constexpr bool is_bool_output =
      std::is_same_v<typename UFuncT::ReturnType, bool>;
  if (!use_new_dtype_api) {
    return RegisterUFuncImpl(
        numpy, name, /*use_new_dtype_api=*/false,
        /*nin=*/UFuncT::kInputArity,
        /*nout=*/std::tuple_size_v<typename UFuncT::ResultTypesTuple>,
        /*is_reorderable=*/UFuncT::kIsReorderable,
        /*is_logical=*/UFuncT::kIsLogical,
        /*is_bool_output=*/is_bool_output,
        /*is_homogeneous=*/UFuncT::kIsHomogeneous,
        /*types=*/UFuncT::Types(),
        /*custom_type_num=*/DtypeTraits<CustomT>::Dtype(),
        /*spec_dtypes=*/{},
        /*fn=*/reinterpret_cast<PyUFuncGenericFunction>(UFuncT::Call),
        /*loop_numpy2=*/nullptr,
        /*custom_promoter=*/nullptr);
  }

  std::vector<PyObject*> dtypes_to_decref;
  std::vector<PyArray_DTypeMeta*> spec_dtypes;
  spec_dtypes.reserve(UFuncT::kInputArity +
                      std::tuple_size_v<typename UFuncT::ResultTypesTuple>);
  bool ok = true;
  auto add_meta = [&](auto tag) {
    using T = typename decltype(tag)::type;
    if (!ok) return;
    auto* meta = GetDTypeMeta<T>(dtypes_to_decref);
    if (!meta) {
      ok = false;
    } else {
      spec_dtypes.push_back(meta);
    }
  };
  std::apply(
      [&](auto... args) { (add_meta(type_identity<decltype(args)>{}), ...); },
      typename UFuncT::InTypesTuple{});
  std::apply(
      [&](auto... args) { (add_meta(type_identity<decltype(args)>{}), ...); },
      typename UFuncT::ResultTypesTuple{});

  if (!ok) {
    for (auto* d : dtypes_to_decref) Py_XDECREF(d);
    PyErr_Format(PyExc_RuntimeError,
                 "RegisterUFunc(%s) failed: GetDTypeMeta returned nullptr",
                 name);
    return false;
  }

  PromoterFunc custom_promoter = nullptr;
  if constexpr (UFuncT::kHasCustomPromoter) {
    custom_promoter =
        reinterpret_cast<PromoterFunc>(&UFuncT::FunctorType::Promoter);
  }

  bool result = RegisterUFuncImpl(
      numpy, name, /*use_new_dtype_api=*/true,
      /*nin=*/UFuncT::kInputArity,
      /*nout=*/std::tuple_size_v<typename UFuncT::ResultTypesTuple>,
      /*is_reorderable=*/UFuncT::kIsReorderable,
      /*is_logical=*/UFuncT::kIsLogical,
      /*is_bool_output=*/is_bool_output,
      /*is_homogeneous=*/UFuncT::kIsHomogeneous,
      /*types=*/UFuncT::Types(),
      /*custom_type_num=*/DtypeTraits<CustomT>::Dtype(),
      /*spec_dtypes=*/spec_dtypes,
      /*fn=*/reinterpret_cast<PyUFuncGenericFunction>(UFuncT::Call),
      /*loop_numpy2=*/
      reinterpret_cast<PyArrayMethod_StridedLoop*>(UFuncT::Call_Numpy2),
      /*custom_promoter=*/custom_promoter);

  for (auto* d : dtypes_to_decref) Py_XDECREF(d);
  return result;
}

namespace ufuncs {

template <typename T>
struct Add {
  static constexpr bool is_reorderable = true;
  T operator()(T a, T b) { return a + b; }
};
template <typename T>
struct Subtract {
  T operator()(T a, T b) { return a - b; }
};
template <typename T>
struct Multiply {
  static constexpr bool is_reorderable = true;
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a, T b) {
    return a * b;
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a, T b) {
    auto result = to_system(a) * to_system(b);
    using ValueType = typename T::value_type;
    return T(static_cast<ValueType>(result.real()),
             static_cast<ValueType>(result.imag()));
  }
};
template <typename T>
struct TrueDivide {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a, T b) {
    return a / b;
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a, T b) {
    auto result = to_system(a) / to_system(b);
    using ValueType = typename T::value_type;
    return T(static_cast<ValueType>(result.real()),
             static_cast<ValueType>(result.imag()));
  }
};

static std::pair<float, float> divmod_impl(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();

    if (std::isnan(a) || (a == 0.0f)) {
      return {nan, nan};
    } else {
      return {std::signbit(a) == std::signbit(b) ? inf : -inf, nan};
    }
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

template <typename T>
struct Divmod {
  std::pair<T, T> operator()(T a, T b) {
    float c, d;
    std::tie(c, d) = divmod_impl(to_system(a), to_system(b));
    return {T(c), T(d)};
  }
};
template <typename T>
struct FloorDivide {
  template <typename U = T,
            std::enable_if_t<std::numeric_limits<U>::is_integer, bool> = true>
  T operator()(T x, T y) {
    if (y == T(0)) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in floor_divide", 1);
      return T(0);
    }
    T v = x / y;
    if (((x > 0) != (y > 0)) && x % y != 0) {
      v = v - T(1);
    }
    return v;
  }
  template <typename U = T,
            std::enable_if_t<!std::numeric_limits<U>::is_integer, bool> = true>
  T operator()(T a, T b) {
    return T(divmod_impl(to_system(a), to_system(b)).first);
  }
};
template <typename T>
struct Remainder {
  template <typename U = T,
            std::enable_if_t<std::numeric_limits<U>::is_integer, bool> = true>
  T operator()(T x, T y) {
    if (y == 0) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in remainder", 1);
      return T(0);
    }
    T v = x % y;
    if (v != 0 && ((v < 0) != (y < 0))) {
      v = v + y;
    }
    return v;
  }
  template <typename U = T,
            std::enable_if_t<!std::numeric_limits<U>::is_integer, bool> = true>
  T operator()(T a, T b) {
    return T(divmod_impl(to_system(a), to_system(b)).second);
  }
};

template <typename T>
struct Fmod {
  T operator()(T a, T b) { return T(std::fmod(to_system(a), to_system(b))); }
};
template <typename T>
struct Negative {
  T operator()(T a) { return -a; }
};
template <typename T>
struct Positive {
  T operator()(T a) { return a; }
};
template <typename T>
struct Power {
  T operator()(T a, T b) { return T(std::pow(to_system(a), to_system(b))); }
};
template <typename T>
struct Abs {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return Eigen::numext::abs(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  typename U::value_type operator()(T a) {
    using real_type = typename U::value_type;
    return real_type(std::abs(to_system(a)));
  }
};
template <typename T>
struct Cbrt {
  T operator()(T a) { return T(std::cbrt(to_system(a))); }
};
template <typename T>
struct Ceil {
  T operator()(T a) { return T(std::ceil(to_system(a))); }
};

// Helper struct for getting a bit representation provided a byte size.
template <int kNumBytes>
struct GetUnsignedInteger;

template <>
struct GetUnsignedInteger<1> {
  using type = uint8_t;
};

template <>
struct GetUnsignedInteger<2> {
  using type = uint16_t;
};

template <typename T>
using BitsType = typename GetUnsignedInteger<sizeof(T)>::type;

template <typename T>
std::pair<BitsType<T>, BitsType<T>> SignAndMagnitude(T x) {
  const BitsType<T> x_bits = Eigen::numext::bit_cast<BitsType<T>>(x);
  // Unsigned floating point format (e.g. E8M0) => no sign bit (zero by
  // default).
  if constexpr (!std::numeric_limits<T>::is_signed) {
    return {BitsType<T>(0), x_bits};
  }
  // For types that represent NaN by -0, (i.e. *fnuz), abs(x) remains -0 without
  // flipping the sign. Therefore, we need to explicitly check the
  // most-significant bit.
  // For types without NaNs (i.e. mxfloat), use xor to keep the sign bit, which
  // may be not the most-significant bit.
  constexpr BitsType<T> kSignMask = BitsType<T>(1)
                                    << (sizeof(BitsType<T>) * CHAR_BIT - 1);
  constexpr bool has_nan = std::numeric_limits<T>::has_quiet_NaN;
  const BitsType<T> x_abs_bits =
      Eigen::numext::bit_cast<BitsType<T>>(Eigen::numext::abs(x));
  return {has_nan ? x_bits & kSignMask : x_bits ^ x_abs_bits, x_abs_bits};
}

template <typename T>
struct CopySign {
  T operator()(T a, T b) {
    // Unsigned floating point format => no change.
    if constexpr (!std::numeric_limits<T>::is_signed) {
      return a;
    }
    auto [a_sign, a_abs_bits] = SignAndMagnitude(a);
    auto [b_sign, b_abs_bits] = SignAndMagnitude(b);
    BitsType<T> rep = a_abs_bits | b_sign;
    return Eigen::numext::bit_cast<T>(rep);
  }
};

template <typename T>
struct Exp {
  T operator()(T a) { return T(std::exp(to_system(a))); }
};
template <typename T>
struct Exp2 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::exp2(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    constexpr float LOGE2 = 0.6931471805599453f;
    auto x = to_system(a) * LOGE2;
    auto res = std::exp(x);
    return T(res);
  }
};
template <typename T>
struct Expm1 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::expm1(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T x_) {
    auto x = to_system(x_);
    auto a = std::sin(x.imag() / 2);
    auto res_real = std::expm1(x.real()) * std::cos(x.imag()) - 2 * a * a;
    auto res_imag = std::exp(x.real()) * std::sin(x.imag());
    return T(res_real, res_imag);
  }
};
template <typename T>
struct Floor {
  T operator()(T a) { return T(std::floor(to_system(a))); }
};
template <typename T>
struct Frexp {
  std::pair<T, int> operator()(T a) {
    int exp;
    float f = std::frexp(to_system(a), &exp);
    return {T(f), exp};
  }
};
template <typename T>
struct Heaviside {
  T operator()(T x, T h0) {
    if (Eigen::numext::isnan(x)) {
      return x;
    }
    auto [sign_x, abs_x] = SignAndMagnitude(x);
    // x == 0
    if (abs_x == 0) {
      return h0;
    }
    return sign_x ? T(0.0f) : T(1.0f);
  }
};

template <typename T>
struct Conjugate {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return a;
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  U operator()(U a) {
    return U{a.real(), -a.imag()};
  }
};

template <typename T>
struct IsFinite {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return my_isfinite(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return my_isfinite(a.real()) && my_isfinite(a.imag());
  }
};
template <typename T>
struct IsInf {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  bool operator()(U a) {
    return my_isinf(a);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  bool operator()(T a) {
    return my_isinf(a.real()) || my_isinf(a.imag());
  }
};
template <typename T>
struct IsNan {
  bool operator()(T a) { return my_isnan(a); }
};

template <typename T>
struct Ldexp {
  static int Promoter(PyObject* ufunc, PyArray_DTypeMeta* const op_dtypes[],
                      PyArray_DTypeMeta* const signature[],
                      PyArray_DTypeMeta* new_op_dtypes[]) {
    PyArray_DTypeMeta* custom_dtype = GetCustomDTypeMeta<T>();
    PyArray_DTypeMeta* int64_dtype = &PyArray_Int64DType;
    PyArray_DTypeMeta* int32_dtype = &PyArray_Int32DType;

    new_op_dtypes[0] = custom_dtype;
    Py_INCREF(custom_dtype);

    if (op_dtypes[1] == int32_dtype) {
      new_op_dtypes[1] = int32_dtype;
      Py_INCREF(int32_dtype);
    } else {
      new_op_dtypes[1] = int64_dtype;
      Py_INCREF(int64_dtype);
    }

    if (signature != nullptr && signature[2] != nullptr) {
      new_op_dtypes[2] = signature[2];
      Py_INCREF(signature[2]);
    } else {
      new_op_dtypes[2] = custom_dtype;
      Py_INCREF(custom_dtype);
    }
    return 0;
  }

  template <typename IntT,
            std::enable_if_t<std::is_integral_v<IntT>, bool> = true>
  T operator()(T a, IntT exp) {
    if (exp > std::numeric_limits<int>::max()) {
      exp = std::numeric_limits<int>::max();
    } else if (exp < std::numeric_limits<int>::min()) {
      exp = std::numeric_limits<int>::min();
    }
    return T(std::ldexp(to_system(a), static_cast<int>(exp)));
  }
};
template <typename T>
struct Log {
  T operator()(T a) { return T(std::log(to_system(a))); }
};
template <typename T>
struct Log2 {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::log2(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    auto x = to_system(a);
    constexpr float LOG2E = 1.442695040888963407359924681001892137f;
    return T(std::log(x) * LOG2E);
  }
};
template <typename T>
struct Log10 {
  T operator()(T a) { return T(std::log10(to_system(a))); }
};
template <typename T>
struct Log1p {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::log1p(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    auto x = to_system(a);
    auto l = std::abs(x + 1.0f);
    auto res_imag = std::atan2(x.imag(), x.real() + 1);
    auto res_real = std::log(l);
    return T(res_real, res_imag);
  }
};
template <typename T>
struct LogAddExp {
  T operator()(T bx, T by) {
    auto x = to_system(bx);
    auto y = to_system(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return T(out);
  }
};
template <typename T>
struct LogAddExp2 {
  T operator()(T bx, T by) {
    float x = to_system(bx);
    float y = to_system(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return T(out);
  }
};
template <typename T>
struct Modf {
  std::pair<T, T> operator()(T a) {
    float integral;
    float f = std::modf(to_system(a), &integral);
    return {T(f), T(integral)};
  }
};

template <typename T>
struct Reciprocal {
  T operator()(T a) { return T(1.f / to_system(a)); }
};
template <typename T>
struct Rint {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::rint(to_system(a)));
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    return T(std::rint(to_system(a.real())), std::rint(to_system(a.imag())));
  }
};

template <typename T>
struct Sign {
  template <typename U = T, std::enable_if_t<!is_complex_v<U>, bool> = false>
  T operator()(T a) {
    if (Eigen::numext::isnan(a)) {
      return a;
    }
    auto [sign_a, abs_a] = SignAndMagnitude(a);
    if (abs_a == 0) {
      return a;
    }
    return sign_a ? T(-1) : T(1);
  }
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = false>
  T operator()(T a) {
    // The complex signum is defined via z/|z|, the implementation below
    // is adopted from NumPy.
    auto c = to_system(a);
    auto abs =
        std::hypot(c.real(), c.imag());  // NumPy uses hypot which is the same.
    constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
    if (std::isnan(abs)) {
      return T(nan, nan);
    }
    if (std::isinf(abs)) {
      if (std::isinf(c.real())) {
        if (std::isinf(c.imag())) {
          return T(nan, nan);
        } else {
          return T(c.real() > 0. ? 1. : -1., 0.);
        }
      } else {
        return T{0., c.imag() > 0 ? 1. : -1.};
      }
    } else if (abs == 0) {
      return T{0., 0.};
    }
    return T{c.real() / abs, c.imag() / abs};
  }
};
template <typename T>
struct SignBit {
  bool operator()(T a) {
    if constexpr (std::is_integral_v<T> || is_intn_v<T>) {
      return a < 0;
    } else {
      auto [sign_a, abs_a] = SignAndMagnitude(a);
      return sign_a;
    }
  }
};
template <typename T>
struct Sqrt {
  T operator()(T a) { return T(std::sqrt(to_system(a))); }
};
template <typename T>
struct Square {
  T operator()(T a) {
    auto f = to_system(a);
    return T(f * f);
  }
};
template <typename T>
struct Trunc {
  T operator()(T a) { return T(std::trunc(to_system(a))); }
};

// Trigonometric functions
template <typename T>
struct Sin {
  T operator()(T a) { return T(std::sin(to_system(a))); }
};
template <typename T>
struct Cos {
  T operator()(T a) { return T(std::cos(to_system(a))); }
};
template <typename T>
struct Tan {
  T operator()(T a) { return T(std::tan(to_system(a))); }
};
template <typename T>
struct Arcsin {
  T operator()(T a) { return T(std::asin(to_system(a))); }
};
template <typename T>
struct Arccos {
  T operator()(T a) { return T(std::acos(to_system(a))); }
};
template <typename T>
struct Arctan {
  T operator()(T a) { return T(std::atan(to_system(a))); }
};
template <typename T>
struct Arctan2 {
  T operator()(T a, T b) { return T(std::atan2(to_system(a), to_system(b))); }
};
template <typename T>
struct Hypot {
  T operator()(T a, T b) { return T(std::hypot(to_system(a), to_system(b))); }
};
template <typename T>
struct Sinh {
  T operator()(T a) { return T(std::sinh(to_system(a))); }
};
template <typename T>
struct Cosh {
  T operator()(T a) { return T(std::cosh(to_system(a))); }
};
template <typename T>
struct Tanh {
  T operator()(T a) { return T(std::tanh(to_system(a))); }
};
template <typename T>
struct Arcsinh {
  T operator()(T a) { return T(std::asinh(to_system(a))); }
};
template <typename T>
struct Arccosh {
  T operator()(T a) { return T(std::acosh(to_system(a))); }
};
template <typename T>
struct Arctanh {
  T operator()(T a) { return T(std::atanh(to_system(a))); }
};
template <typename T>
struct Deg2rad {
  T operator()(T a) {
    static constexpr float radians_per_degree =
        static_cast<float>(M_PI) / 180.0f;
    return T(to_system(a) * radians_per_degree);
  }
};
template <typename T>
struct Rad2deg {
  T operator()(T a) {
    static constexpr float degrees_per_radian =
        180.0f / static_cast<float>(M_PI);
    return T(to_system(a) * degrees_per_radian);
  }
};

template <typename T>
struct Eq {
  static constexpr bool is_reorderable = true;
  npy_bool operator()(T a, T b) { return a == b; }
};
template <typename T>
struct Ne {
  static constexpr bool is_reorderable = true;
  npy_bool operator()(T a, T b) { return a != b; }
};
template <typename T>
struct Lt {
  npy_bool operator()(T a, T b) { return a < b; }
};
template <typename T>
struct Gt {
  npy_bool operator()(T a, T b) { return a > b; }
};
template <typename T>
struct Le {
  npy_bool operator()(T a, T b) { return a <= b; }
};
template <typename T>
struct Ge {
  npy_bool operator()(T a, T b) { return a >= b; }
};
template <typename T>
struct Maximum {
  static constexpr bool is_reorderable = true;
  T operator()(T a, T b) { return my_isnan(a) || a > b ? a : b; }
};
template <typename T>
struct Minimum {
  static constexpr bool is_reorderable = true;
  T operator()(T a, T b) { return my_isnan(a) || a < b ? a : b; }
};
template <typename T>
struct Clip {
  T operator()(T a, T min_val, T max_val) {
    return Minimum<T>()(Maximum<T>()(a, min_val), max_val);
  }
};
template <typename T>
struct Fmax {
  static constexpr bool is_reorderable = true;
  T operator()(T a, T b) { return my_isnan(b) || a > b ? a : b; }
};
template <typename T>
struct Fmin {
  static constexpr bool is_reorderable = true;
  T operator()(T a, T b) { return my_isnan(b) || a < b ? a : b; }
};

template <typename T>
struct LogicalNot {
  npy_bool operator()(T a) { return !static_cast<bool>(a); }
};
template <typename T>
struct LogicalAnd {
  static constexpr bool is_logical = true;
  static constexpr bool is_reorderable = true;
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalOr {
  static constexpr bool is_logical = true;
  static constexpr bool is_reorderable = true;
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalXor {
  static constexpr bool is_logical = true;
  static constexpr bool is_reorderable = true;
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

template <typename T>
struct NextAfter {
  T operator()(T from, T to) {
    BitsType<T> from_rep = Eigen::numext::bit_cast<BitsType<T>>(from);
    BitsType<T> to_rep = Eigen::numext::bit_cast<BitsType<T>>(to);
    if (Eigen::numext::isnan(from) || Eigen::numext::isnan(to)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (from_rep == to_rep) {
      return to;
    }
    auto [from_sign, from_abs] = SignAndMagnitude(from);
    auto [to_sign, to_abs] = SignAndMagnitude(to);
    if (from_abs == 0) {
      if (to_abs == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        return Eigen::numext::bit_cast<T>(
            static_cast<BitsType<T>>(0x01 | to_sign));
      }
    }
    BitsType<T> magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign)
            ? static_cast<BitsType<T>>(-1)
            : static_cast<BitsType<T>>(1);
    BitsType<T> out_int = from_rep + magnitude_adjustment;
    T out = Eigen::numext::bit_cast<T>(out_int);
    // Some non-IEEE compatible formats may have a representation for NaN
    // instead of -0, ensure we return a zero in such cases.
    if constexpr (!std::numeric_limits<T>::is_iec559) {
      if (Eigen::numext::isnan(out)) {
        return Eigen::numext::bit_cast<T>(BitsType<T>{0});
      }
    }
    return out;
  }
};

template <typename T>
struct Spacing {
  T operator()(T x) {
    CopySign<T> copysign;
    if constexpr (!std::numeric_limits<T>::has_infinity) {
      if (Eigen::numext::abs(x) == std::numeric_limits<T>::max()) {
        if constexpr (!std::numeric_limits<T>::has_quiet_NaN) return T();
        return copysign(std::numeric_limits<T>::quiet_NaN(), x);
      }
    }
    // Compute the distance between the input and the next number with greater
    // magnitude. The result should have the sign of the input.
    T away = std::numeric_limits<T>::has_infinity
                 ? std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::max();
    away = copysign(away, x);
    return NextAfter<T>()(x, away) - x;
  }
};

}  // namespace ufuncs
}  // namespace ml_dtypes

#endif  // ML_DTYPES_UFUNCS_H_
