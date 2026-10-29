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

#ifndef ML_DTYPES_CASTS_H_
#define ML_DTYPES_CASTS_H_

#include <Python.h>

#include <memory>
#include <vector>

#include "ml_dtypes/_src/numpy.h"

namespace ml_dtypes {

// Holds a single NumPy cast specification and the arrays its pointers
// reference (`dtypes` and `slots`), so those pointers remain valid until the
// dtype finishes initializing.
struct CastSpecEntry {
  PyArray_DTypeMeta* dtypes[2];
  PyType_Slot slots[2];
  PyArrayMethod_Spec spec;
};

// Collects cast specifications for a custom dtype and keeps their backing
// storage alive while NumPy registers the dtype.
class CastSpecList {
 public:
  CastSpecList() = default;
  CastSpecList(const CastSpecList&) = delete;
  CastSpecList& operator=(const CastSpecList&) = delete;
  CastSpecList(CastSpecList&&) = default;
  CastSpecList& operator=(CastSpecList&&) = default;

  void Add(const char* name, PyArray_DTypeMeta* from, PyArray_DTypeMeta* to,
           NPY_CASTING casting, NPY_ARRAYMETHOD_FLAGS flags,
           PyArrayMethod_StridedLoop* loop);

  const std::vector<PyArrayMethod_Spec*>& specs() const { return specs_; }

 private:
  std::vector<std::unique_ptr<CastSpecEntry>> entries_;
  std::vector<PyArrayMethod_Spec*> specs_;
};

// Builds the list of casts between `Self` and all built-in numeric dtypes, as
// well as any custom dtypes that have already been registered.
template <typename Self>
CastSpecList BuildCastSpecs();

}  // namespace ml_dtypes

#endif  // ML_DTYPES_CASTS_H_
