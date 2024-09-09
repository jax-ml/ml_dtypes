# Copyright 2024 The ml_dtypes Authors.
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

"""Utilities for multi-threaded tests."""

import concurrent.futures
import functools
from typing import Optional


def multi_threaded(*, num_workers: int, skip_tests: Optional[list[str]] = None):
  """Decorator that runs a test in a multi-threaded environment."""

  def decorator(test_cls):
    for name, test_fn in test_cls.__dict__.copy().items():
      if not (name.startswith("test") and callable(test_fn)):
        continue

      if skip_tests is not None:
        if any(test_name in name for test_name in skip_tests):
          continue

      @functools.wraps(test_fn)  # pylint: disable=cell-var-from-loop
      def multi_threaded_test_fn(*args, __test_fn__=test_fn, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
          futures = []
          for _ in range(num_workers):
            futures.append(executor.submit(__test_fn__, *args, **kwargs))
          # We should call future.result() to re-raise an exception if test has
          # failed
          list(f.result() for f in futures)

      setattr(test_cls, f"{name}_multi_threaded", multi_threaded_test_fn)

    return test_cls

  return decorator
