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
  * `float8_e4m3fnuz`
  * `float8_e5m2`
  * `float8_e5m2fnuz`

## Installation

The `ml_dtypes` package is tested with Python versions 3.8-3.11, and can be installed
with the following command:
```
pip install ml_dtypes
```
To test your installation, you can run the following:
```
pip install absl-py pytest
pytest --pyargs ml_dtypes
```
To build from source, clone the repository and run:
```
git submodule init
git submodule update
pip install .
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

## License

The `ml_dtypes` source code is licensed under the Apache 2.0 license
(see [LICENSE](LICENSE)). Pre-compiled wheels are built with the
[EIGEN](https://eigen.tuxfamily.org/) project, which is released under the
MPL 2.0 license (see [LICENSE.eigen](LICENSE.eigen)).
