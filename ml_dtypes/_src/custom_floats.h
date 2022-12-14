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

#ifndef ML_DTYPES_BFLOAT16_H_
#define ML_DTYPES_BFLOAT16_H_

#include <Python.h>

namespace ml_dtypes {

// Register the bfloat16 numpy type. Returns true on success.
bool RegisterNumpyBfloat16();

// Returns a pointer to the bfloat16 dtype object.
PyObject* Bfloat16Dtype();

// Returns the id number of the bfloat16 numpy type.
int Bfloat16NumpyType();

}  // namespace ml_dtypes

#endif  // ML_DTYPES_BFLOAT16_H_