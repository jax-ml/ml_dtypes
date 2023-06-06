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

__version__ = '0.2.0'  # Keep in sync with pyproject.toml:version
__all__ = [
    '__version__',
    'bfloat16',
    'finfo',
    'float8_e4m3b11fnuz',
    'float8_e4m3fn',
    'float8_e4m3fnuz',
    'float8_e5m2',
    'float8_e5m2fnuz',
    'iinfo',
    'int4',
    'uint4',
]

from typing import Type

from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11fnuz
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e4m3fnuz
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._custom_floats import float8_e5m2fnuz
from ml_dtypes._custom_floats import int4
from ml_dtypes._custom_floats import uint4
from ml_dtypes._finfo import finfo
from ml_dtypes._iinfo import iinfo

import numpy as np

bfloat16: Type[np.generic]
float8_e4m3b11fnuz: Type[np.generic]
float8_e4m3fn: Type[np.generic]
float8_e4m3fnuz: Type[np.generic]
float8_e5m2: Type[np.generic]
float8_e5m2fnuz: Type[np.generic]
int4: Type[np.generic]
uint4: Type[np.generic]

del np, Type


# TODO(jakevdp) remove this deprecated name.
def __getattr__(name):  # pylint: disable=invalid-name
  if name == 'float8_e4m3b11':
    import warnings  # pylint: disable=g-import-not-at-top

    warnings.warn(
        (
            'ml_dtypes.float8_e4m3b11 is deprecated. Use'
            ' ml_dtypes.float8_e4m3b11fnuz'
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return float8_e4m3b11fnuz
  raise AttributeError(f'cannot import name {name!r} from {__name__!r}')
