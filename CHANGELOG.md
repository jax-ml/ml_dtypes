# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/jax-ml/ml_dtypes/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

* Fixed `np.power` and `np.float_power` for `complex32` and `bcomplex32` to
  match NumPy's complex power: `x**0` is now `1` (it was `nan` for `x` equal to
  `0`, `inf` or `nan`), and integer exponents are now exact (`1j**2` was
  `-1-5.96e-08j`).

## [0.7.0] - 2026-11-16

* Added new 8-bit unsigned float type: `ml_dtypes.float8_e5m3fnu`
  ([#407](https://github.com/jax-ml/ml_dtypes/pull/407)).
* `ml_dtypes` types now use the NumPy 2 DType API
  ([#374](https://github.com/jax-ml/ml_dtypes/pull/374),
  [#418](https://github.com/jax-ml/ml_dtypes/pull/418),
  [#424](https://github.com/jax-ml/ml_dtypes/pull/424)):
  * Fixed `np.result_type` and `np.promote_types` for `ml_dtypes` types and
    Python scalars. Mixed operations with Python `int`, `float`, and `complex`
    values now follow NumPy 2 (NEP 50) promotion rules instead of upcasting to
    `float64` or `complex128` (for example, `bfloat16(2.0) + 1e-3` stays in
    `bfloat16`). Comparing a finite-only or unsigned float against an
    out-of-range Python float converts the Python float to the `ml_dtypes` type
    first (e.g., `float8_e4m3fn(448.0) < 1000.0` is `False` because
    `float8_e4m3fn(1000.0)` overflows to `NaN`; use `float(x) < 1000.0` for
    wider bounds).
  * Fixed `np.can_cast` so lossy casts between 1-byte types (such as
    `float8_e4m3fn` to `float4_e2m1fn`) are no longer reported as `"safe"`, and
    widening `intN` casts to `longlong` and `ulonglong` are now `"safe"`.
  * Added casts between all pairs of `ml_dtypes` types, including narrowing and
    mixed-sign `intN` casts, `int4`/`uint4` <-> `float6_e2m3fn`, and casts
    involving `float8_e8m0fnu`.
  * Multi-axis and empty-array ufunc reductions now work across all `ml_dtypes`
    types.
  * Casting non-finite or out-of-range floats to integers now consistently
    returns `0` and emits an `invalid` warning; casting complex types to real or
    integer `ml_dtypes` types emits `ComplexWarning`; integer division by zero
    respects `np.errstate`; and `np.ldexp` supports 64-bit integer exponents on
    all platforms.
* Registered byte-aligned `ml_dtypes` types with NumPy's DLPack support on
  NumPy 2.5+ ([#402](https://github.com/jax-ml/ml_dtypes/pull/402)).
* `np.finfo` now supports `ml_dtypes` float and complex types, `ml_dtypes.finfo`
  now accepts `"complex32"`, and `np.float_power` is now supported for float and
  complex types ([#395](https://github.com/jax-ml/ml_dtypes/pull/395)).
* Changed `np.dtype(complex32).kind` from `'V'` to `'c'`, and changed
  `dtype.char` for `complex32` and `bcomplex32` to `'J'` and `'K'`
  ([#357](https://github.com/jax-ml/ml_dtypes/pull/357)).
* `arr.real` and `arr.imag` now return correct results for `bcomplex32` and
  `complex32` arrays on NumPy 2.5+ ([#383](https://github.com/jax-ml/ml_dtypes/pull/383)).
* Fixed byte-swapping of `bcomplex32` and `complex32` arrays
  ([#383](https://github.com/jax-ml/ml_dtypes/pull/383)).
* Fixed `NaN` ordering for `complex32` and `bcomplex32` in `np.sort` and
  `np.searchsorted` ([#429](https://github.com/jax-ml/ml_dtypes/pull/429)),
  casting purely imaginary values to `bool`
  ([#395](https://github.com/jax-ml/ml_dtypes/pull/395)), and `TypeError`
  handling in two-argument `complex32(real, imag)` / `bcomplex32(real, imag)`
  ([#439](https://github.com/jax-ml/ml_dtypes/pull/439)).
* Fixed double rounding when converting to `ml_dtypes` float and complex types
  from `float64`, `longdouble`, `complex128`, `clongdouble`, 32- and 64-bit
  integers, and Python/NumPy scalars, and fixed a crash when constructing floats
  from invalid strings ([#401](https://github.com/jax-ml/ml_dtypes/pull/401)).
* Constructing a float or complex scalar from a Python `int` outside the `int64`
  range no longer raises `OverflowError` when the value fits in `float64`
  ([#447](https://github.com/jax-ml/ml_dtypes/pull/447)).
* Fixed `int(x)` on `ml_dtypes` float and complex scalars for large values,
  `inf`, and `NaN` ([#444](https://github.com/jax-ml/ml_dtypes/pull/444)).
* Constructing `intN` values from Python floats now truncates toward zero before
  range checking (e.g. `int4(7.5)` returns `7` instead of raising
  `OverflowError`) ([#443](https://github.com/jax-ml/ml_dtypes/pull/443)), and
  out-of-range errors now name the actual `intN` type instead of `"int4"`
  ([#403](https://github.com/jax-ml/ml_dtypes/pull/403)).
* Casts to `float8_e8m0fnu` now round values between 2<sup>-127</sup> and
  1.5 × 2<sup>-127</sup> down to 2<sup>-127</sup> instead of up to
  2<sup>-126</sup>
  ([#434](https://github.com/jax-ml/ml_dtypes/pull/434)).
* `np.copysign` of zero no longer returns `NaN` for `*fnuz` types, and
  `np.sign` and `np.heaviside` of the smallest `float8_e8m0fnu` value now
  return 1 ([#408](https://github.com/jax-ml/ml_dtypes/pull/408),
  [#416](https://github.com/jax-ml/ml_dtypes/pull/416)).
* Fixed spurious `invalid value` warnings when casting quiet `NaN`s on ARM64
  ([#426](https://github.com/jax-ml/ml_dtypes/pull/426),
  [#428](https://github.com/jax-ml/ml_dtypes/pull/428)).
* `complex_types.h` can now be included in C++ without Python or NumPy headers
  ([#422](https://github.com/jax-ml/ml_dtypes/pull/422)).
* `float8.h` no longer depends on `Eigen::numext::get_integer_by_size`, which is
  unavailable in Eigen 3.4.x
  ([#414](https://github.com/jax-ml/ml_dtypes/pull/414)).

## [0.6.0] - 2026-08-13

* Added new 16-bit complex types: `ml_dtypes.complex32` (based on float16) and
  `ml_dtypes.bcomplex32` (based on bfloat16) ([#351](https://github.com/jax-ml/ml_dtypes/pull/351)).
* Added new 1-bit integer types: `ml_dtypes.int1` and `ml_dtypes.uint1`.
* Added `__format__` method to custom float, complex, and integer types
  ([#341](https://github.com/jax-ml/ml_dtypes/issues/341)). Previously,
  formatting custom scalars (e.g. in f-strings) fell back to string formatting,
  which could truncate exponents or fail on numeric format specifiers.
* `ml_dtypes.finfo` and `ml_dtypes.iinfo` now allow passing an array object to
  their constructor ([#350](https://github.com/jax-ml/ml_dtypes/pull/350)).
* Fixed equality comparison (`==`, `!=`) when comparing custom dtypes against
  incompatible types like strings or `None`.
* Dropped support for Python 3.9, which reached end-of-life in October 2025.
* Dropped support for Python 3.13 free-threading, because cibuildwheel dropped support.
* Dropped support for NumPy < 2.0.
* Added support for Python 3.15.
* Switched build system to `scikit-build-core` and CMake ([#361](https://github.com/jax-ml/ml_dtypes/pull/361)).
* Updated Eigen dependency to v5.0.1.

## [0.5.4] - 2025-11-17

* We now register casts from int2 and int4 to all of the custom float types,
  except `float6_e2m3fn` and `float8_e8m0fnu`.
* Custom floats may now be constructed from Python integers ([#317](https://github.com/jax-ml/ml_dtypes/issues/317))
* Fixed bug in byte-swap operation for custom floats ([#311](https://github.com/jax-ml/ml_dtypes/pull/311))
* Wheels now support Python 3.14 free threading on Windows.

## [0.5.3] - 2025-07-29

* NPY_NEEDS_PYAPI was removed from the dtype flags. This should improve the
  speed of array operations, but it does mean that values pickled using
  previous versions of ml_dtypes are incompatible with the current release
  and should be regenerated with the current release.
* Wheels now support Python 3.14.
* Wheels now support Windows 11 ARM.

## [0.5.2] - 2025-01-31

* Dropped support for Power wheels again. These turned out to cause problems in
  our release process. We will consider readding these if NumPy ships Power
  wheels.
* Fixed GCC compilation issues related to ambiguous casts.

## [0.5.1] - 2025-01-06

* Fixed sign bit handling for float4 and float6 types.
* Wheels now support Python 3.13 free-threading.
* Wheels now support the Power architecture.

## [0.5.0] - 2024-09-13

* Added new 8-bit float types following IEEE 754 convention:
  `ml_dtypes.float8_e4m3`, `ml_dtypes.float8_e3m4`
* Added the 8-bit floating point type `ml_dtypes.float8_e8m0fnu`, which is the
  OpenCompute MX scale format.
* Added new 4-bit and 6-bit float types:
  `ml_dtypes.float4_e2m1fn`, `ml_dtypes.float6_e2m3fn` and `ml_dtypes.float6_e3m2fn`.
* Fix outputs of float `divmod` and `floor_divide` when denominator is zero.

## [0.4.1] - 2024-09-13

* Updates build requirements to use NumPy 2.0 release

## [0.4.0] - 2024-04-1

* Updates `ml_dtypes` for compatibility with future NumPy 2.0 release.
* Wheels are built against NumPy 2.0.0rc1.

## [0.4.0b1] - 2024-03-12

* Updates `ml_dtypes` for compatibility with future NumPy 2.0 release.
* Wheels for the release candidate are built against NumPy 2.0.0b1.

## [0.3.2] - 2024-01-03

* Fixed spurious invalid value warnings when casting between floating point
  types on Mac ARM.
* Remove `pybind11` build requirement
* Update C++ sources for compatibility with NumPy 2.0

## [0.3.1] - 2023-09-22

* Added support for int4 casting to wider integers such as int8
* Addes support to cast np.float32 and np.float64 into int4

## [0.3.0] - 2023-09-19

* Dropped support for Python 3.8, following [NEP 29].
* Added support for Python 3.12.
* Removed deprecated name `ml_dtypes.float8_e4m3b11`;
  use `ml_dtypes.float8_e4m3b11fnuz` instead.

## [0.2.0] - 2023-06-06

* New features:
  * added new 4-bit integer types: `ml_dtypes.int4` and `ml_dtypes.uint4`

* Deprecations:
  * `ml_dtypes.float8_e4m3b11` has been renamed to `ml_dtypes.float8_e4m3b11fnuz` for more
    consistency with other dtype names. The former name will still be available until
    version 0.3.0, but will raise a deprecation warning.

## [0.1.0] - 2023-04-11

* Initial release

[Unreleased]: https://github.com/jax-ml/ml_dtypes/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/jax-ml/ml_dtypes/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/jax-ml/ml_dtypes/compare/v0.5.2....v0.5.3
[0.5.2]: https://github.com/jax-ml/ml_dtypes/compare/v0.5.1....v0.5.2
[0.5.1]: https://github.com/jax-ml/ml_dtypes/compare/v0.5.0....v0.5.1
[0.5.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.4.0....v0.5.0
[0.4.1]: https://github.com/jax-ml/ml_dtypes/compare/v0.4.0....v0.4.1
[0.4.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.4.0b1....v0.4.0
[0.4.0b1]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.2...v0.4.0b1
[0.3.2]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jax-ml/ml_dtypes/releases/tag/v0.1.0
[NEP 29]: https://numpy.org/neps/nep-0029-deprecation_policy.html
