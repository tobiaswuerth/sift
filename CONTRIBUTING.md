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

## Code layout

| Module | What it does |
|---|---|
| `cli.py` | The `sift` command |
| `store.py` | The SQLite index: layout, its version and upgrades, connection, collections, deleting documents |
| `readers.py` | Files into documents: one reader per kind of file (Markdown, JSON Lines, documents via Xberg, emails), chunking, and the links of Markdown notes |
| `indexing.py` | `sift index`: scanning folders, reading changed files in worker processes, incremental updates; `reindex` of given files |
| `ignore.py` | `.siftignore`: the gitignore patterns of files and folders a collection leaves out, matched in linear time |
| `links.py` | Links between notes in the index: the names files answer to, resolving links when asked, backlinks |
| `search.py` | Keyword, semantic and hybrid search over the index, documents like a document, `get`, and links, mentions and `reindex` |
| `bench.py` | `sift bench`: nDCG@10, MRR@10 and Recall@10 of each search mode on questions with known answers |
| `embed.py` | `sift embed`: embedding every chunk without a vector, with progress |
| `models.py` | The `Models` interface, and `default()`: the models this process uses |
| `local.py` | `Local`: the embedding model and reranker of this machine |
| `remote.py` | `Remote`: a `sift worker` on another machine, used like `Local` |
| `accelerator.py` | `Accelerator`: the device, its memory budget, batch sizes, and what is on the GPU when |
| `vectors.py` | The index's vectors as one matrix to score queries against: `InRam` (NumPy) or `OnGpu` (PyTorch) |
| `server.py` | `sift serve`: the MCP server, and the UI under /ui |
| `ui.py` | `sift ui`: the views of the browser UI; `web/` holds its templates, styles and script, and htmx 2.0.11 (0BSD) as its release ships it |
| `worker.py` | `sift worker`: lends this machine's GPU to others, over HTTPS |
| `settings.py` | The four optional environment variables, read once (`settings.current()`) |
| `console.py` | What sift tells the user, on stderr: messages, plurals, commands to paste, and one progress bar for every long step |

Search and embedding never ask where the models run: they take a `Models` object, `Local` or `Remote`. Only
`accelerator.py` imports PyTorch at the top, so a machine that searches through a worker never loads it. No module
reads the environment except `settings.py`. Results go to stdout from `cli.py` alone, and everything else sift says
goes to stderr through `console.py`, so output can be piped. Tests build these objects around fakes (Sentence
Transformers, GPU memory, the clock) instead of patching module state, and change settings with the `configure`
fixture.

## How changes are made

- **Tests first.** Write a test that fails, then make it pass. Every bug fix comes with a regression test that
  reproduces the bug.
- **Full coverage.** `pytest --cov` requires 100% line and branch coverage. The tests fake the models and GPU memory,
  so they run anywhere, without a GPU or downloads.
- **Quality gates.** pre-commit runs on every commit and again in CI: ruff (lint and format), whitespace and line
  endings, YAML/TOML checks, `validate-pyproject`, workflow and Dependabot schema checks, actionlint and codespell.
- **Pull requests.** `main` is protected: changes arrive through pull requests, which are squash-merged once every CI
  check passes. CI runs the tests on Linux (x86 and ARM), macOS and Windows for Python 3.11 to 3.15, and builds and
  checks the package. It runs on pull requests, and again on the commit each merge makes on `main`, which may combine
  changes no pull request tested together.
- **Changelog.** Note user-visible changes under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## Releasing

1. In a pull request, set the version in `sift/__init__.py` and move the "Unreleased" changes in `CHANGELOG.md` under
   that version. CI checks that the changelog has a section for it.
2. Merge it. That's all: once CI has passed on the merge commit, the release workflow sees a version without a tag and
   publishes that commit.
   - It builds the package and publishes it to PyPI through trusted publishing (no tokens). The `pypi` environment
     deploys from `main` only.
   - It tags the commit CI tested and creates a GitHub release with that version's changelog.
   - It doesn't run the tests itself: it starts only after CI passed on that exact commit, and never for a failed run.
