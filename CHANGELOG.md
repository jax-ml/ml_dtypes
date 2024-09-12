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

* Added new 8-bit float types following IEEE 754 convention:
  `ml_dtypes.float8_e4m3` and `ml_dtypes.float8_e3m4`.
* Added new 4-bit and 6-bit float types:
  `ml_dtypes.float4_e2m1fn`, `ml_dtypes.float6_e2m3fn` and `ml_dtypes.float6_e3m2fn`.
* Fix outputs of float `divmod` and `floor_divide` when denominator is zero.

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

[Unreleased]: https://github.com/jax-ml/ml_dtypes/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.4.0b1....v0.4.0
[0.4.0b1]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.2...v0.4.0b1
[0.3.2]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/jax-ml/ml_dtypes/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jax-ml/ml_dtypes/releases/tag/v0.1.0
[NEP 29]: https://numpy.org/neps/nep-0029-deprecation_policy.html
