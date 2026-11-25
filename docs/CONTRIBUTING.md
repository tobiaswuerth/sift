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

It runs every check a change must pass and stops at the first that fails: `lint`
(pre-commit hooks and import contracts; it refuses files git neither tracks nor ignores), `types` (basedpyright for
Linux, macOS and Windows, each as Python 3.12 and 3.15), `tests-3.12` and `tests-3.15` (100% line and branch
coverage of `sift` and `tools`), `lowest` (the lowest versions sift allows, installed afresh on each run, and
`python -m tools floors` holds each to the version `pyproject.toml` names), `package` (the wheel and sdist hold the
tracked sources and nothing else, and the installed commands run). One session:
`uv run --only-group gate nox -s types`. CI runs these sessions too, the tests on every Python from 3.12 to 3.15
on Linux (x86 and ARM), macOS and Windows, which no local run covers, and on Python 3.12.0 itself, uv's own build
of it (`nox -s oldest`). `lowest` and `package` install packages, so they need the network. No session runs the
lowest SQLite sift allows, 3.38: every Python build brings a newer one.

Before a release, `nox -s explore` runs the property tests with fresh random examples, 2,000 each, and 200 at least
for those that build stores (`few` in `tests/examples.py`); the gate runs them derandomized. And `nox -s real` runs
the default models, as the Hugging Face cache holds them or
downloaded: it indexes a few notes and a scan with OCR, embeds them and searches them three ways, and on a CUDA GPU
checks that the models rest and come back. Its session has CPU PyTorch; for the GPU, run
`python -m tests.real_models` where PyTorch has CUDA. Its output goes into the release notes.
Tests run on at most 8 workers: each worker imports every test module, the model libraries among them, and on 32
threads more workers made the suite slower (35 s on 8, 101 s on 32).

## Rules the checks hold

- No comments and no docstrings: names, types, tests and structure carry the meaning (`python -m tools prose`).
  An exception needs the owner's yes and an entry in `tools/prose_allowed.toml`.
- `sift.api` and `sift.domain` hold pure functions and immutable values: no I/O, no result of a call left unused.
  Module-level values are immutable everywhere; no module has more than 400 lines, no class more than 10 methods
  (`python -m tools shape`, the import contracts, basedpyright).
- Every name of sift is used in sift, and no function is written twice, in sift, its tools or its tests: what two
  places need is kept where both can import it (`python -m tools dead`, `python -m tools twins`).
- Packages import only those below them: interfaces, app, retrieval and ingest, storage and inference, domain, api.
  No module imports one that imports it (`[tool.importlinter]` in `pyproject.toml`).
- SQLite, the model libraries, the document converter and worker processes are each imported in their place alone
  (the import contracts).
- sift imports only packages `pyproject.toml` depends on, and the stubs in `typings/` name only what their packages
  have (`python -m tools imports`, `python -m tools stubs`, which the lint session runs where sift is installed).
- Tests pass fakes and values in; `monkeypatch` is not used, and no test reads a clock: what costs time is counted
  (`python -m tools idioms`, which also holds lessons of earlier bugs: each of its findings names the fix).
- sift reads its environment in `sift.settings` alone, and only `sift.interfaces` imports that module: the use cases
  and the core take values (`python -m tools idioms`, the import contracts).
- A character that shows as nothing, or as another, is written as an escape (`"\N{ZERO WIDTH SPACE}"`), never as
  itself (`python -m tools idioms`).
- The tokenizer's table is generated from SQLite's FTS5 (`python -m tools fts5 --write`), and a test holds it equal.
- Stores need SQLite 3.38 or newer (STRICT tables, the JSON functions built in); an older one is refused with what
  to do. A store's layout changes by a numbered migration appended to `LAYOUT`, never by editing one a release has
  shipped.
- Published text names no machine's model, and prose is wrapped as the code is (`python -m tools text`).
- A bound is a module constant named `MOST_*`, `FEWEST_*`, `LEAST_*`, `*_SECONDS` or `*_BYTES`, beside what it bounds;
  [limits.md](limits.md) lists them all, written anew by `python -m tools limits --write`, and a test holds it equal
  to the code.
- No check is silenced. A suppression needs the owner's yes, scoped to one rule in one place.

## How changes are made

- A failing test of the behaviour first, through public functions; then the change.
- User-visible changes go under "Unreleased" in [CHANGELOG.md](../CHANGELOG.md).
- Measurements in docs state the hardware by kind (a GPU, 8 CPU threads), not by model.

## Releasing

Set the version in `pyproject.toml` and move the "Unreleased" changes under `## [VERSION] - YYYY-MM-DD`, dated the
day of the release, in a pull request: the release workflow refuses a version without that section, or with an empty
one, before it builds anything (`python -m tools release notes VERSION` prints what it would publish). Once CI passes
on its merge commit, the release workflow builds the package, publishes it to PyPI by trusted publishing, tags the
commit and creates the GitHub release, marked a pre-release for a, b, rc and dev versions. If a job fails after PyPI
took the package, re-run only the failed jobs.
