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

[Unreleased]: https://github.com/jax-ml/ml_dtypes/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jax-ml/ml_dtypes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jax-ml/ml_dtypes/releases/tag/v0.1.0
[NEP 29]: https://numpy.org/neps/nep-0029-deprecation_policy.html
