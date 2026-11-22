# Contributing

Bug reports and feature ideas are welcome as [issues](https://github.com/tobiaswuerth/sift/issues); security problems
go to [SECURITY.md](SECURITY.md).

## Setup

```bash
git clone https://github.com/tobiaswuerth/sift
cd sift
uv sync
uv run pre-commit install
```

## The gate

```bash
uv run --only-group gate nox
```

It runs every check a change must pass, stops at the first that fails, and CI runs the same sessions: `lint`
(pre-commit hooks and import contracts; it refuses files git neither tracks nor ignores), `types` (basedpyright for
Linux, macOS and Windows), `tests-3.12` and `tests-3.15` (100% line and branch coverage of `sift` and `tools`),
`lowest` (the lowest versions sift allows), `package` (the wheel and sdist hold the tracked sources and nothing
else, and the installed commands run). One session: `uv run --only-group gate nox -s types`.

## Rules the checks hold

- No comments and no docstrings: names, types, tests and structure carry the meaning (`python -m tools prose`).
  An exception needs the owner's yes and an entry in `tools/prose_allowed.toml`.
- `sift.api` and `sift.domain` hold pure functions and immutable values: no I/O, no result of a call left unused.
  Module-level values are immutable everywhere; no module has more than 400 lines, no class more than 10 methods
  (`python -m tools shape`, the import contracts, basedpyright).
- Packages import only those below them: interfaces, app, retrieval and ingest, storage and inference, domain, api.
  No module imports one that imports it (`[tool.importlinter]` in `pyproject.toml`).
- Tests pass fakes and values in; `monkeypatch` is not used, and no test reads a clock: what costs time is counted
  (`python -m tools idioms`, which also holds lessons of earlier bugs: each of its findings names the fix).
- The tokenizer's table is generated from SQLite's FTS5 (`python -m tools fts5 --write`), and a test holds it equal.
- Published text names no machine's model, and prose is wrapped as the code is (`python -m tools text`).
- No check is silenced. A suppression needs the owner's yes, scoped to one rule in one place.

## How changes are made

- A failing test of the behaviour first, through public functions; then the change.
- User-visible changes go under "Unreleased" in [CHANGELOG.md](../CHANGELOG.md).
- Measurements in docs state the hardware by kind (a GPU, 8 CPU threads), not by model.

## Releasing

Set the version in `pyproject.toml` and move the "Unreleased" changes under it in a pull request. Once CI passes on
its merge commit, the release workflow builds the package, publishes it to PyPI by trusted publishing, tags the
commit and creates the GitHub release. If a job fails after PyPI took the package, re-run only the failed jobs.
