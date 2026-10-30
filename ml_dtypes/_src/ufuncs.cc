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

#include "ml_dtypes/_src/ufuncs.h"

#include <Python.h>

#include <array>
#include <utility>

#include "ml_dtypes/_src/common.h"
#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {
namespace {

bool IsSignedIntNDType(PyArray_DTypeMeta* dm) {
  return dm == DtypeTraits<int1>::DTypeMeta() ||
         dm == DtypeTraits<int2>::DTypeMeta() ||
         dm == DtypeTraits<int4>::DTypeMeta();
}

bool IsUnsignedIntNDType(PyArray_DTypeMeta* dm) {
  return dm == DtypeTraits<uint1>::DTypeMeta() ||
         dm == DtypeTraits<uint2>::DTypeMeta() ||
         dm == DtypeTraits<uint4>::DTypeMeta();
}

bool IsIntNDType(PyArray_DTypeMeta* dm) {
  return IsSignedIntNDType(dm) || IsUnsignedIntNDType(dm);
}

PyArray_DTypeMeta* WidenIntNDType(PyArray_DTypeMeta* dm) {
  if (IsSignedIntNDType(dm)) {
    return &PyArray_ByteDType;
  }
  if (IsUnsignedIntNDType(dm)) {
    return &PyArray_UByteDType;
  }
  return dm;
}

bool IsBuiltinIntegerDType(PyArray_DTypeMeta* dm) {
  return dm->type_num >= 0 && dm->type_num < NPY_NTYPES_LEGACY &&
         PyTypeNum_ISINTEGER(dm->type_num);
}

bool IsIntegerOrPyLongDType(PyArray_DTypeMeta* dm) {
  return IsIntNDType(dm) || IsBuiltinIntegerDType(dm) ||
         dm == &PyArray_PyLongDType;
}

bool IsPythonScalarDType(PyArray_DTypeMeta* dm) {
  return dm == &PyArray_PyLongDType || dm == &PyArray_PyFloatDType ||
         dm == &PyArray_PyComplexDType;
}

bool IsCustomFloatDType(PyArray_DTypeMeta* dm) {
  return dm == DtypeTraits<bfloat16>::DTypeMeta() ||
         dm == DtypeTraits<float4_e2m1fn>::DTypeMeta() ||
         dm == DtypeTraits<float6_e2m3fn>::DTypeMeta() ||
         dm == DtypeTraits<float6_e3m2fn>::DTypeMeta() ||
         dm == DtypeTraits<float8_e3m4>::DTypeMeta() ||
         dm == DtypeTraits<float8_e4m3>::DTypeMeta() ||
         dm == DtypeTraits<float8_e4m3fn>::DTypeMeta() ||
         dm == DtypeTraits<float8_e4m3fnuz>::DTypeMeta() ||
         dm == DtypeTraits<float8_e4m3b11fnuz>::DTypeMeta() ||
         dm == DtypeTraits<float8_e5m2>::DTypeMeta() ||
         dm == DtypeTraits<float8_e5m2fnuz>::DTypeMeta() ||
         dm == DtypeTraits<float8_e5m3fnu>::DTypeMeta() ||
         dm == DtypeTraits<float8_e8m0fnu>::DTypeMeta();
}

enum class PromoterKind {
  // Loops registered for custom floats, intN, and (where valid) custom complex.
  kStandard,
  // Loops registered for custom floats (and custom complex where valid), but
  // not intN; sub-byte integer results widen to int8/uint8.
  kWidenIntN,
  // True division: integer inputs promote to float64.
  kTrueDivide,
  // float_power: integers, float16, and float32 promote to float64; complex64
  // promotes to complex128; custom floats/complex use their registered loops.
  kFloatPower,
  // Real floating-point binary ufuncs (hypot, arctan2, copysign, etc.):
  // integer inputs promote to the appropriate floating-point DType.
  kFloatOnly,
  // Binary comparisons returning bool: sub-byte integers widen to int8/uint8
  // (preserving exact PyLong comparisons when paired with Python ints).
  kComparison,
};

template <PromoterKind Kind>
int HomogeneousBinaryPromoter(PyObject* ufunc,
                              PyArray_DTypeMeta* const op_dtypes[],
                              PyArray_DTypeMeta* const signature[],
                              PyArray_DTypeMeta* new_op_dtypes[]) {
  PyUFuncObject* ufunc_obj = reinterpret_cast<PyUFuncObject*>(ufunc);
  const int nin = ufunc_obj->nin;
  const int nargs = ufunc_obj->nargs;

  // Preserve exact range-checked comparisons against Python integers by
  // widening intN to int8/uint8 and delegating to NumPy's PyLong loops.
  if constexpr (Kind == PromoterKind::kComparison) {
    if (op_dtypes[0] != nullptr && signature[0] == nullptr &&
        signature[1] == nullptr && signature[2] == nullptr) {
      if (IsIntNDType(op_dtypes[0]) && op_dtypes[1] == &PyArray_PyLongDType) {
        PyArray_DTypeMeta* widened = WidenIntNDType(op_dtypes[0]);
        Py_INCREF(widened);
        new_op_dtypes[0] = widened;
        Py_INCREF(&PyArray_PyLongDType);
        new_op_dtypes[1] = &PyArray_PyLongDType;
        Py_XINCREF(op_dtypes[2]);
        new_op_dtypes[2] = op_dtypes[2];
        return 0;
      }
      if (op_dtypes[0] == &PyArray_PyLongDType && IsIntNDType(op_dtypes[1])) {
        Py_INCREF(&PyArray_PyLongDType);
        new_op_dtypes[0] = &PyArray_PyLongDType;
        PyArray_DTypeMeta* widened = WidenIntNDType(op_dtypes[1]);
        Py_INCREF(widened);
        new_op_dtypes[1] = widened;
        Py_XINCREF(op_dtypes[2]);
        new_op_dtypes[2] = op_dtypes[2];
        return 0;
      }
    }
  }

  PyArray_DTypeMeta* effective_inputs[2];
  if (op_dtypes[0] == nullptr) {
    // Reduction with unspecified accumulator dtype (`op_dtypes = {NULL, D,
    // ...}`).
    effective_inputs[0] = op_dtypes[1];
    effective_inputs[1] = op_dtypes[1];
  } else {
    effective_inputs[0] = op_dtypes[0];
    effective_inputs[1] = op_dtypes[1];
  }

  PyArray_DTypeMeta* common = nullptr;
  bool from_output_signature = false;
  if constexpr (Kind == PromoterKind::kComparison) {
    // Output of comparisons is bool unless object is explicitly requested.
    if (signature[nin] == &PyArray_ObjectDType) {
      Py_INCREF(&PyArray_ObjectDType);
      common = &PyArray_ObjectDType;
      from_output_signature = true;
    }
  } else {
    // Match NumPy's default_ufunc_promoter: if output signature is homogeneous,
    // use it as the target computation DType (e.g. accumulate(..., dtype=...)).
    for (int i = nin; i < nargs; ++i) {
      if (signature[i] != nullptr) {
        if (common == nullptr) {
          Py_INCREF(signature[i]);
          common = signature[i];
          from_output_signature = true;
        } else if (common != signature[i]) {
          Py_CLEAR(common);
          from_output_signature = false;
          break;
        }
      }
    }
  }

  if (common == nullptr) {
    if (effective_inputs[0] == effective_inputs[1] &&
        !IsPythonScalarDType(effective_inputs[0])) {
      common = effective_inputs[0];
      Py_INCREF(common);
    } else {
      // On NumPy < 2.6, Python abstract scalar DTypes (such as PyFloatDType)
      // treat custom DTypes with NPY_DT_is_legacy set as legacy float types
      // when placed first, probing CommonDType(other, HalfDType) instead of
      // deferring to other->common_dtype. Placing the concrete DType first
      // ensures commutative promotion across all NumPy 2.x versions.
      if (IsPythonScalarDType(effective_inputs[0]) &&
          !IsPythonScalarDType(effective_inputs[1])) {
        std::swap(effective_inputs[0], effective_inputs[1]);
      }
      common = PyArray_PromoteDTypeSequence(nin, effective_inputs);
      if (common == nullptr) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
          PyErr_Clear();
        }
        return -1;
      }
    }
  }

  if (!from_output_signature) {
    if constexpr (Kind == PromoterKind::kWidenIntN ||
                  Kind == PromoterKind::kComparison) {
      if (IsIntNDType(common)) {
        PyArray_DTypeMeta* widened = WidenIntNDType(common);
        Py_INCREF(widened);
        Py_SETREF(common, widened);
      }
    } else if constexpr (Kind == PromoterKind::kTrueDivide) {
      if (IsIntegerOrPyLongDType(common) || common == &PyArray_BoolDType) {
        Py_INCREF(&PyArray_DoubleDType);
        Py_SETREF(common, &PyArray_DoubleDType);
      }
    } else if constexpr (Kind == PromoterKind::kFloatPower) {
      if (IsIntegerOrPyLongDType(common) || common == &PyArray_BoolDType ||
          common == &PyArray_HalfDType || common == &PyArray_FloatDType) {
        Py_INCREF(&PyArray_DoubleDType);
        Py_SETREF(common, &PyArray_DoubleDType);
      } else if (common == &PyArray_CFloatDType) {
        Py_INCREF(&PyArray_CDoubleDType);
        Py_SETREF(common, &PyArray_CDoubleDType);
      }
    } else if constexpr (Kind == PromoterKind::kFloatOnly) {
      if (IsIntNDType(common) || common == &PyArray_BoolDType) {
        Py_INCREF(&PyArray_HalfDType);
        Py_SETREF(common, &PyArray_HalfDType);
      } else if (common == &PyArray_PyLongDType ||
                 common == &PyArray_PyFloatDType) {
        Py_INCREF(&PyArray_DoubleDType);
        Py_SETREF(common, &PyArray_DoubleDType);
      } else if (IsBuiltinIntegerDType(common)) {
        PyArray_DTypeMeta* flt =
            PyArray_CommonDType(common, &PyArray_HalfDType);
        if (flt == nullptr) {
          Py_DECREF(common);
          if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
          }
          return -1;
        }
        Py_SETREF(common, flt);
      }
    }
  }

  for (int i = 0; i < nin; ++i) {
    PyArray_DTypeMeta* tmp = signature[i] != nullptr ? signature[i] : common;
    Py_INCREF(tmp);
    new_op_dtypes[i] = tmp;
  }
  for (int i = nin; i < nargs; ++i) {
    Py_XINCREF(op_dtypes[i]);
    new_op_dtypes[i] = op_dtypes[i];
  }

  Py_DECREF(common);
  return 0;
}

int LdexpPromoter(PyObject* /*ufunc*/, PyArray_DTypeMeta* const op_dtypes[],
                  PyArray_DTypeMeta* const signature[],
                  PyArray_DTypeMeta* new_op_dtypes[]) {
  if (op_dtypes[0] == nullptr) {
    // ldexp is inhomogeneous (`(float, int) -> float`) and has no valid
    // unforced reduction loop.
    return -1;
  }

  PyArray_DTypeMeta* exp_in = op_dtypes[1];
  if (!IsIntegerOrPyLongDType(exp_in) && exp_in != &PyArray_BoolDType) {
    return -1;
  }

  PyArray_DTypeMeta* mant_dt = nullptr;
  if (signature[0] != nullptr) {
    mant_dt = signature[0];
    Py_INCREF(mant_dt);
  } else if (signature[2] != nullptr) {
    mant_dt = signature[2];
    Py_INCREF(mant_dt);
  } else if (IsCustomFloatDType(op_dtypes[0]) ||
             (op_dtypes[0]->type_num >= 0 &&
              op_dtypes[0]->type_num < NPY_NTYPES_LEGACY &&
              PyTypeNum_ISFLOAT(op_dtypes[0]->type_num))) {
    mant_dt = op_dtypes[0];
    Py_INCREF(mant_dt);
  } else if (op_dtypes[0] == &PyArray_PyFloatDType ||
             op_dtypes[0] == &PyArray_PyLongDType) {
    mant_dt = &PyArray_DoubleDType;
    Py_INCREF(mant_dt);
  } else if (IsIntNDType(op_dtypes[0]) || op_dtypes[0] == &PyArray_BoolDType) {
    mant_dt = &PyArray_HalfDType;
    Py_INCREF(mant_dt);
  } else if (IsBuiltinIntegerDType(op_dtypes[0])) {
    mant_dt = PyArray_CommonDType(op_dtypes[0], &PyArray_HalfDType);
    if (mant_dt == nullptr) {
      if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
      }
      return -1;
    }
  } else {
    return -1;
  }

  PyArray_DTypeMeta* exp_dt = nullptr;
  if (signature[1] != nullptr) {
    exp_dt = signature[1];
  } else if (exp_in == &PyArray_Int64DType || exp_in == &PyArray_UInt32DType ||
             exp_in == &PyArray_UInt64DType ||
             exp_in == &PyArray_LongLongDType ||
             exp_in == &PyArray_ULongLongDType) {
    exp_dt = DtypeTraits<npy_int64>::DTypeMeta();
  } else {
    exp_dt = DtypeTraits<int>::DTypeMeta();
  }

  new_op_dtypes[0] = mant_dt;  // steals reference to mant_dt
  Py_INCREF(exp_dt);
  new_op_dtypes[1] = exp_dt;
  Py_XINCREF(op_dtypes[2]);
  new_op_dtypes[2] = op_dtypes[2];
  return 0;
}

bool AddPromoterPair(PyObject* ufunc_obj, int nargs, PyArray_DTypeMeta* dt0,
                     PyArray_DTypeMeta* dt1, PyObject* capsule) {
  Safe_PyObjectPtr dtype_tuple = make_safe(PyTuple_New(nargs));
  if (!dtype_tuple) {
    return false;
  }
  PyObject* o0 = reinterpret_cast<PyObject*>(dt0);
  PyObject* o1 = reinterpret_cast<PyObject*>(dt1);
  Py_INCREF(o0);
  PyTuple_SET_ITEM(dtype_tuple.get(), 0, o0);
  Py_INCREF(o1);
  PyTuple_SET_ITEM(dtype_tuple.get(), 1, o1);
  for (int i = 2; i < nargs; ++i) {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(dtype_tuple.get(), i, Py_None);
  }
  return PyUFunc_AddPromoter(ufunc_obj, dtype_tuple.get(), capsule) >= 0;
}

// Registers scoped numeric promoters for a binary ufunc using concrete
// `DTypeMeta` pairs (without `None` wildcards or abstract DTypes in input
// positions). Every input pair matches at most one promoter by pointer
// equality regardless of dict iteration order, and non-numeric DTypes such as
// `StringDType` never collide with our promoters.
bool AddBinaryPromoters(PyObject* numpy, const char* name,
                        PyArrayMethod_PromoterFunction* promoter_fn,
                        bool include_intn_self, bool is_ldexp = false) {
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (ufunc->nin != 2) {
    PyErr_Format(PyExc_AssertionError, "expected binary ufunc for %s, got %d",
                 name, ufunc->nin);
    return false;
  }
  const int nargs = ufunc->nargs;

  Safe_PyObjectPtr capsule = make_safe(PyCapsule_New(
      reinterpret_cast<void*>(promoter_fn), "numpy._ufunc_promoter", nullptr));
  if (!capsule) {
    return false;
  }

  const std::array<PyArray_DTypeMeta*, 21> all_custom = {
      DtypeTraits<bfloat16>::DTypeMeta(),
      DtypeTraits<float4_e2m1fn>::DTypeMeta(),
      DtypeTraits<float6_e2m3fn>::DTypeMeta(),
      DtypeTraits<float6_e3m2fn>::DTypeMeta(),
      DtypeTraits<float8_e3m4>::DTypeMeta(),
      DtypeTraits<float8_e4m3>::DTypeMeta(),
      DtypeTraits<float8_e4m3fn>::DTypeMeta(),
      DtypeTraits<float8_e4m3fnuz>::DTypeMeta(),
      DtypeTraits<float8_e4m3b11fnuz>::DTypeMeta(),
      DtypeTraits<float8_e5m2>::DTypeMeta(),
      DtypeTraits<float8_e5m2fnuz>::DTypeMeta(),
      DtypeTraits<float8_e5m3fnu>::DTypeMeta(),
      DtypeTraits<float8_e8m0fnu>::DTypeMeta(),
      DtypeTraits<int1>::DTypeMeta(),
      DtypeTraits<int2>::DTypeMeta(),
      DtypeTraits<int4>::DTypeMeta(),
      DtypeTraits<uint1>::DTypeMeta(),
      DtypeTraits<uint2>::DTypeMeta(),
      DtypeTraits<uint4>::DTypeMeta(),
      DtypeTraits<bcomplex32>::DTypeMeta(),
      DtypeTraits<complex32>::DTypeMeta(),
  };

  const std::array<PyArray_DTypeMeta*, 22> builtin_targets = {
      &PyArray_BoolDType,       &PyArray_ByteDType,
      &PyArray_UByteDType,      &PyArray_ShortDType,
      &PyArray_UShortDType,     &PyArray_IntDType,
      &PyArray_UIntDType,       &PyArray_LongDType,
      &PyArray_ULongDType,      &PyArray_LongLongDType,
      &PyArray_ULongLongDType,  &PyArray_HalfDType,
      &PyArray_FloatDType,      &PyArray_DoubleDType,
      &PyArray_LongDoubleDType, &PyArray_CFloatDType,
      &PyArray_CDoubleDType,    &PyArray_CLongDoubleDType,
      &PyArray_ObjectDType,     &PyArray_PyLongDType,
      &PyArray_PyFloatDType,    &PyArray_PyComplexDType,
  };

  for (PyArray_DTypeMeta* dt : all_custom) {
    for (PyArray_DTypeMeta* target : builtin_targets) {
      // `ldexp` already registers concrete `(CustomFloat, int)` and
      // `(CustomFloat, int64)` loops; skip `(dt, target)` for those exact
      // signatures so the promoter does not tie with the loop.
      const bool skip_forward = is_ldexp && IsCustomFloatDType(dt) &&
                                (target == DtypeTraits<int>::DTypeMeta() ||
                                 target == DtypeTraits<npy_int64>::DTypeMeta());
      if (!skip_forward &&
          !AddPromoterPair(ufunc_obj.get(), nargs, dt, target, capsule.get())) {
        return false;
      }
      if (!AddPromoterPair(ufunc_obj.get(), nargs, target, dt, capsule.get())) {
        return false;
      }
    }
    if (include_intn_self && IsIntNDType(dt) &&
        !AddPromoterPair(ufunc_obj.get(), nargs, dt, dt, capsule.get())) {
      return false;
    }
  }

  for (PyArray_DTypeMeta* d0 : all_custom) {
    for (PyArray_DTypeMeta* d1 : all_custom) {
      if (d0 != d1 &&
          !AddPromoterPair(ufunc_obj.get(), nargs, d0, d1, capsule.get())) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool RegisterUFuncPromoters(PyObject* numpy) {
  constexpr const char* kStandardUFuncs[] = {
      "add", "subtract", "multiply", "floor_divide", "remainder",
  };
  for (const char* name : kStandardUFuncs) {
    if (!AddBinaryPromoters(numpy, name,
                            &HomogeneousBinaryPromoter<PromoterKind::kStandard>,
                            /*include_intn_self=*/false)) {
      return false;
    }
  }

  constexpr const char* kWidenIntNUFuncs[] = {
      "fmod", "divmod", "power", "maximum", "minimum", "fmax", "fmin",
  };
  for (const char* name : kWidenIntNUFuncs) {
    if (!AddBinaryPromoters(
            numpy, name, &HomogeneousBinaryPromoter<PromoterKind::kWidenIntN>,
            /*include_intn_self=*/true)) {
      return false;
    }
  }

  if (!AddBinaryPromoters(numpy, "divide",
                          &HomogeneousBinaryPromoter<PromoterKind::kTrueDivide>,
                          /*include_intn_self=*/true)) {
    return false;
  }

  if (!AddBinaryPromoters(numpy, "float_power",
                          &HomogeneousBinaryPromoter<PromoterKind::kFloatPower>,
                          /*include_intn_self=*/true)) {
    return false;
  }

  constexpr const char* kFloatOnlyUFuncs[] = {
      "logaddexp", "logaddexp2", "heaviside", "arctan2",
      "hypot",     "copysign",   "nextafter",
  };
  for (const char* name : kFloatOnlyUFuncs) {
    if (!AddBinaryPromoters(
            numpy, name, &HomogeneousBinaryPromoter<PromoterKind::kFloatOnly>,
            /*include_intn_self=*/true)) {
      return false;
    }
  }

  constexpr const char* kComparisonUFuncs[] = {
      "equal", "not_equal", "less", "greater", "less_equal", "greater_equal",
  };
  for (const char* name : kComparisonUFuncs) {
    if (!AddBinaryPromoters(
            numpy, name, &HomogeneousBinaryPromoter<PromoterKind::kComparison>,
            /*include_intn_self=*/true)) {
      return false;
    }
  }

  // Note: logical_and, logical_or, and logical_xor already install NumPy's
  // built-in logical_ufunc_promoter at (dtype, dtype, dtype), which promotes
  // any mixed or reduction call to (bool, bool, bool).

  if (!AddBinaryPromoters(numpy, "ldexp", &LdexpPromoter,
                          /*include_intn_self=*/true, /*is_ldexp=*/true)) {
    return false;
  }

  return true;
}

}  // namespace ml_dtypes
