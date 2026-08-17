/* Copyright 2017 The ml_dtypes Authors.

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

#include <Python.h>

#include "ml_dtypes/_src/complex.h"
#include "ml_dtypes/_src/floats.h"
#include "ml_dtypes/_src/ints.h"
#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {
bool RegisterCustomCasts();

// Initializes the module.
bool Initialize() {
  ml_dtypes::ImportNumpy();

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  Safe_PyObjectPtr exceptions =
      make_safe(PyObject_GetAttrString(numpy.get(), "exceptions"));
  if (!exceptions) {
    return false;
  }
  ComplexWarning = PyObject_GetAttrString(exceptions.get(), "ComplexWarning");
  if (!ComplexWarning) {
    return false;
  }

  return RegisterCustomFloats(numpy.get()) && RegisterCustomInts(numpy.get()) &&
         RegisterCustomComplex(numpy.get()) && RegisterCustomCasts();
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_ml_dtypes_ext",
};

PyMODINIT_FUNC PyInit__ml_dtypes_ext() {
  Safe_PyObjectPtr m = make_safe(PyModule_Create(&module_def));
  if (!m) {
    return nullptr;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load _ml_dtypes_ext module.");
    }
    return nullptr;
  }

  if (PyObject_SetAttrString(m.get(), "float4_e2m1fn",
                             CustomFloatType<float4_e2m1fn>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float6_e2m3fn",
                             CustomFloatType<float6_e2m3fn>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float6_e3m2fn",
                             CustomFloatType<float6_e3m2fn>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e3m4",
                             CustomFloatType<float8_e3m4>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e4m3",
                             CustomFloatType<float8_e4m3>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e4m3b11fnuz",
                             CustomFloatType<float8_e4m3b11fnuz>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "float8_e4m3fn",
                             CustomFloatType<float8_e4m3fn>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e4m3fnuz",
                             CustomFloatType<float8_e4m3fnuz>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e5m2",
                             CustomFloatType<float8_e5m2>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e5m2fnuz",
                             CustomFloatType<float8_e5m2fnuz>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "float8_e8m0fnu",
                             CustomFloatType<float8_e8m0fnu>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "bfloat16",
                             CustomFloatType<bfloat16>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "bcomplex32",
                             CustomComplexType<bcomplex32>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "complex32",
                             CustomComplexType<complex32>::type_ptr) < 0 ||
      PyObject_SetAttrString(m.get(), "int1", CustomIntType<int1>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "int2", CustomIntType<int2>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "int4", CustomIntType<int4>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "uint1", CustomIntType<uint1>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "uint2", CustomIntType<uint2>::type_ptr) <
          0 ||
      PyObject_SetAttrString(m.get(), "uint4", CustomIntType<uint4>::type_ptr) <
          0) {
    return nullptr;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED);
#endif

  return m.release();
}

}  // namespace ml_dtypes
