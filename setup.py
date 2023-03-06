import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
  ext_modules = [
    Pybind11Extension(
      "ml_dtypes._custom_floats",
      [
        "ml_dtypes/_src/custom_floats.cc",
        "ml_dtypes/_src/custom_float_wrapper.cc",
        "ml_dtypes/_src/numpy.cc",
      ],
      include_dirs = [
        ".",
        np.get_include(),
      ],
      extra_compile_args = [
        "-std=c++17",
        "-DEIGEN_MPL2_ONLY",
      ],
    )
  ]
)
