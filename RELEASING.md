# Releasing ml_dtypes

To create a new `ml_dtypes` release, take the following steps:

1. Send a pull request updating the version in `ml_dtypes/__init__.py` to
   the new version number, as well as updating `CHANGELOG.md` with the
   changes since the previous release (an example for the 0.2.0 release
   is [PR #78]).
2. Once this is merged, create the release tag and push it to github. An
   example from the 0.2.0 release:
   ```
   $ git checkout main
   $ git pull upstream main  # upstream is github.com:jax-ml/ml_dtypes.git
   $ git log  # view commit log & ensure the most recent commit
              # is your version update PR
   $ git tag -a v0.2.0 -m "v0.2.0 Release"
   $ git push upstream v0.2.0
   ```
3. Navigate to https://github.com/jax-ml/ml_dtypes/releases/new, and select
   this new tag. Copy the change description from `CHANGELOG.md` into the
   release notes, and click *Publish release*.
4. Publishing the release will trigger the CI jobs configured in
   `.github/workflows/wheels.yml`, which will build the wheels and source
   distributions and publish them to PyPI. Navigate to
   https://github.com/jax-ml/ml_dtypes/actions/workflows/wheels.yml and
   look for the job associated with this release; monitor it to ensure it
   finishes green (this will take approximately 30 minutes).
5. Once the build is complete, check https://pypi.org/project/ml-dtypes/
   to ensure that the new release is present.

[PR #78]: https://github.com/jax-ml/ml_dtypes/pull/78
