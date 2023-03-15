# ml_dtypes

[![Unittests](https://github.com/jax-ml/ml_dtypes/actions/workflows/test.yml/badge.svg)](https://github.com/jax-ml/ml_dtypes/actions/workflows/test.yml)
[![Wheel Build](https://github.com/jax-ml/ml_dtypes/actions/workflows/wheels.yml/badge.svg)](https://github.com/jax-ml/ml_dtypes/actions/workflows/wheels.yml)
[![PyPI version](https://badge.fury.io/py/ml_dtypes.svg)](https://badge.fury.io/py/ml_dtypes)

*This is not an officially supported Google product.*

`ml_dtypes` is a stand-alone implementation of several NumPy dtype extensions used in machine learning libraries, including:

- [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format):
  an alternative to the standard [`float16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) format
- `float8_*`: several experimental 8-bit floating point representations
  including:
  * `float8_e4m3b11`
  * `float8_e4m3fn`
  * `float8_e5m2`

## Installation

We will soon release pre-built wheels for this package at http://pypi.org/project/ml-dtypes, so this package can be easily pip installed.

To build locally, clone the repository and run:
```
$ git submodule init
$ git submodule update
$ pip install .
```

## Example Usage

```python
>>> from ml_dtypes import bfloat16
>>> import numpy as np
>>> np.zeros(4, dtype=bfloat16)
array([0, 0, 0, 0], dtype=bfloat16)
```
Importing `ml_dtypes` also registers the data types with numpy, so that they may
be referred to by their string name:
```python
>>> np.dtype('bfloat16')
dtype(bfloat16)
>>> np.dtype('float8_e5m2')
dtype(float8_e5m2)
```
