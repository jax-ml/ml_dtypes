# Copyright 2022 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setuptool-based build for ml_dtypes."""

import fnmatch
import platform
import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_py import build_py as build_py_orig

if platform.system() == "Windows":
  COMPILE_ARGS = [
      "/std:c++17",
      "/DEIGEN_MPL2_ONLY",
  ]
else:
  COMPILE_ARGS = [
      "-std=c++17",
      "-DEIGEN_MPL2_ONLY",
  ]

exclude = ["third_party*"]


class build_py(build_py_orig):  # pylint: disable=invalid-name

  def find_package_modules(self, package, package_dir):
    modules = super().find_package_modules(package, package_dir)
    return [  # pylint: disable=g-complex-comprehension
        (pkg, mod, file)
        for (pkg, mod, file) in modules
        if not any(
            fnmatch.fnmatchcase(pkg + "." + mod, pat=pattern)
            for pattern in exclude
        )
    ]


setup(
    ext_modules=[
        Pybind11Extension(
            "ml_dtypes._ml_dtypes_ext",
            [
                "ml_dtypes/_src/dtypes.cc",
                "ml_dtypes/_src/numpy.cc",
            ],
            include_dirs=[
                "third_party/eigen",
                "ml_dtypes",
                np.get_include(),
            ],
            extra_compile_args=COMPILE_ARGS,
        )
    ],
    cmdclass={"build_py": build_py},
)
