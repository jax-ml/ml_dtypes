# Copyright 2026 The ml_dtypes Authors.
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

"""Benchmarks for ml_dtypes cross-dtype casts."""

import google_benchmark
import ml_dtypes
import numpy as np

_SIZES = (1 << 14, 1 << 24)

_CAST_PAIRS = (
    ("bf16_to_f32", ml_dtypes.bfloat16, np.float32),
    ("f32_to_bf16", np.float32, ml_dtypes.bfloat16),
    ("e4m3fn_to_f32", ml_dtypes.float8_e4m3fn, np.float32),
    ("f32_to_e4m3fn", np.float32, ml_dtypes.float8_e4m3fn),
    ("e5m2_to_f32", ml_dtypes.float8_e5m2, np.float32),
    ("f32_to_e5m2", np.float32, ml_dtypes.float8_e5m2),
    ("e8m0fnu_to_f32", ml_dtypes.float8_e8m0fnu, np.float32),
    ("f32_to_e8m0fnu", np.float32, ml_dtypes.float8_e8m0fnu),
    ("f4_e2m1fn_to_f32", ml_dtypes.float4_e2m1fn, np.float32),
    ("f32_to_f4_e2m1fn", np.float32, ml_dtypes.float4_e2m1fn),
    ("bf16_to_e4m3fn", ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn),
    ("e4m3fn_to_bf16", ml_dtypes.float8_e4m3fn, ml_dtypes.bfloat16),
    ("int4_to_int32", ml_dtypes.int4, np.int32),
    ("int32_to_int4", np.int32, ml_dtypes.int4),
)


def _make_cast_benchmark(src_dtype, dst_dtype, *, strided: bool):
  def _bench(state):
    n = state.range(0)
    if strided:
      src = np.ones(2 * n, dtype=src_dtype)[::2]
    else:
      src = np.ones(n, dtype=src_dtype)
    dst = np.empty(n, dtype=dst_dtype)
    # Warm up lazy LUT initialization and caches before timing.
    np.copyto(dst, src)
    while state:
      np.copyto(dst, src)
    state.items_processed = state.iterations * n
    state.bytes_processed = (
        state.iterations * n * (src.dtype.itemsize + dst.dtype.itemsize)
    )

  return _bench


def _register_benchmarks():
  for name, src_dtype, dst_dtype in _CAST_PAIRS:
    for strided, suffix in ((False, ""), (True, "_strided")):
      fn = _make_cast_benchmark(src_dtype, dst_dtype, strided=strided)
      fn = google_benchmark.option.unit(google_benchmark.kMillisecond)(fn)
      fn = google_benchmark.option.arg_names(["n"])(fn)
      for size in _SIZES:
        fn = google_benchmark.option.arg(size)(fn)
      google_benchmark.register(fn, name=f"cast_{name}{suffix}")


_register_benchmarks()


if __name__ == "__main__":
  google_benchmark.main()
