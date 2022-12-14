/* Copyright 2022 The ml_dtypes Authors.

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

#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/py/ml_dtypes/_src/custom_floats.h"


PYBIND11_MODULE(_custom_floats, m) {
  ml_dtypes::RegisterNumpyBfloat16();

  m.def("bfloat16_type",
        [] { return pybind11::handle(ml_dtypes::Bfloat16Dtype()); });
}