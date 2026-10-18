# Contributing

Thanks for helping. Bug reports and feature ideas are welcome as [issues](https://github.com/tobiaswuerth/sift/issues);
for security problems, see [SECURITY.md](SECURITY.md).

## Setup

```bash
git clone https://github.com/tobiaswuerth/sift
cd sift
uv venv && uv pip install --torch-backend auto -e ".[dev]"   # or with pip: pip install -r requirements.txt -e ".[dev]"
pre-commit install
```

## How changes are made

- **Tests first.** Write a test that fails, then make it pass. Every bug fix comes with a regression test that
  reproduces the bug.
- **Full coverage.** `pytest --cov` requires 100% line and branch coverage. The tests fake the models and GPU memory,
  so they run anywhere, without a GPU or downloads.
- **Quality gates.** pre-commit runs on every commit and again in CI: ruff (lint and format), whitespace and line
  endings, YAML/TOML checks, `validate-pyproject`, workflow and Dependabot schema checks, actionlint and codespell.
- **Pull requests.** `main` is protected: changes arrive through pull requests, which are squash-merged once every CI
  check passes. CI runs the tests on Linux (x86 and ARM), macOS and Windows for Python 3.11 to 3.15, and builds and
  checks the package.
- **Changelog.** Note user-visible changes under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## Releasing

1. Set the version in `sift/__init__.py`, and move the "Unreleased" changes in `CHANGELOG.md` under that version.
   Merge this through a pull request.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. The release workflow checks that the tag matches the version, runs the tests, builds the package, publishes it to
   PyPI through trusted publishing (no tokens), and creates a GitHub release.
