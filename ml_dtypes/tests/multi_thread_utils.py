from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


def multi_threaded(*, num_workers: int, skip_tests: Optional[list[str]] = None):
  def decorator(test_cls):
    for name, test_fn in test_cls.__dict__.copy().items():
      if not (name.startswith("test") and callable(test_fn)):
        continue

      if skip_tests is not None:
        if any(test_name in name for test_name in skip_tests):
          continue

      @wraps(test_fn)
      def multi_threaded_test_fn(*args, __test_fn__=test_fn, **kwargs):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
          futures = []
          for _ in range(num_workers):
            futures.append(executor.submit(__test_fn__, *args, **kwargs))
          # We should call future.result() to re-raise an exception if test has failed
          list(f.result() for f in futures)

      setattr(test_cls, f"{name}_multi_threaded", multi_threaded_test_fn)

    return test_cls

  return decorator
