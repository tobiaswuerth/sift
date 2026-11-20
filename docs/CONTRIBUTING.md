# Contributing

Bug reports and feature ideas are welcome as [issues](https://github.com/tobiaswuerth/sift/issues); security problems
go to [SECURITY.md](SECURITY.md).

## Setup

```bash
git clone https://github.com/tobiaswuerth/sift
cd sift
uv venv && uv pip install --torch-backend auto -e ".[dev]"   # or: pip install -r requirements.txt -e ".[dev]"
pre-commit install      # lint, format and config checks on every commit
python scripts/gate.py  # the gate: every hook, the tests with the coverage gate, and the smoke run
```

The tests replace the models, GPU memory and the clock with small deterministic fakes, so they need no GPU and no
downloads; everything else, the MCP server included, runs for real.

## Code layout

| Module | What it does |
|---|---|
| `cli.py` | The `sift` command |
| `catalog.py` | The catalog: collections, their folders, where their stores lie, which are active; adding, removing, moving |
| `store.py` | A collection's store, its SQLite file: layout, ids, codes, the last parts of paths, deleting documents, holding it alone |
| `upgrade.py` | Splitting an index of sift 1.7 or older into a store per collection |
| `readers.py` | Files into documents: a reader per kind of file, chunking, and the links of Markdown notes |
| `indexing.py` | `sift index`: scanning folders, reading changed files in worker processes; `reindex` of given files |
| `ocr.py` | Reading scans: a PDF's pages as pictures, read by an OCR model, once per scan |
| `ignore.py` | `.siftignore`: gitignore patterns, matched in linear time |
| `links.py` | Links between notes: the names files answer to, resolving links, backlinks |
| `search.py` | Keyword, semantic and hybrid search, documents like a document, `get`, links, mentions and `reindex` |
| `expression.py` | Keyword queries (words, phrases, AND, OR, NOT, parentheses) as FTS5 reads them, for the postings |
| `postings.py` | Keyword search's own postings in a topical order; deltas, merges and rebuilds as FTS5 changes |
| `order.py` | The topical order of a collection's chunks, by binary k-means of their codes |
| `topk.py` | The best documents for a keyword query from the postings, by Block-Max Pruning, until a deadline if given |
| `bench.py` | `sift bench`: nDCG@10, MRR@10 and Recall@10 of each search mode |
| `embed.py` | `sift embed`: embedding every chunk without a vector |
| `models.py` | The `Models` interface, and `default()`: the models this process uses |
| `local.py` | `Local`: this machine's embedding model and reranker |
| `remote.py` | `Remote`: a `sift worker` on another machine, used like `Local` |
| `accelerator.py` | `Accelerator`: the device, its memory budget, batch sizes, what is on the GPU when |
| `vectors.py` | The codes in RAM: candidates by Hamming distance, ranked by the query's floats, scored exactly; filters |
| `scan.py` | The Hamming scan compiled by Numba, for a server's large collections |
| `server.py` | `sift serve`: the MCP server, and the UI under /ui |
| `jobs.py` | A server's calls off its event loop, one at a time, stopped when cancelled; the models on the main thread |
| `ui.py` | `sift ui`: the browser UI's views; `web/` holds its templates, styles, script and htmx 2.0.11 (0BSD) |
| `worker.py` | `sift worker`: lends this machine's GPU to others, over HTTPS |
| `update.py` | A newer sift on PyPI: asking once a day, offering it in a terminal, updating once no sift runs |
| `settings.py` | The environment variables, read once (`settings.current()`) |
| `console.py` | What sift tells the user on stderr: messages, plurals, commands to paste, one progress bar |
| `logs.py` | A log for each command that changes the index or serves it |

Rules the code keeps, several checked by `tests/test_package.py`:

- Search and embedding take a `Models` object, `Local` or `Remote`, and never ask where the models run. Only
  `accelerator.py` imports PyTorch at the top, so a machine searching through a worker never loads it.
- Only `settings.py` reads or sets the environment.
- Results go to stdout from `cli.py` alone; everything else goes to stderr through `console.py`, so output can be piped.
- Tests build objects around fakes instead of patching module state, and change settings with the `configure` fixture.

## How changes are made

- **Tests first:** a test that fails, then the change that makes it pass; every bug fix comes with a regression test.
- **Full coverage:** `pytest --cov` requires 100% line and branch coverage.
- **The gate:** `python scripts/gate.py` runs every check a change must pass, and is green when all did: the
  pre-commit hooks, the tests with the coverage gate, and the smoke run.
- **Quality gates:** pre-commit runs on every commit and in CI: ruff (lint and format), whitespace, line endings, byte
  order marks, YAML/TOML checks, `validate-pyproject`, workflow and Dependabot schemas, actionlint and codespell.
  Workflows name their actions by commit, which a test holds them to.
- **The smoke run:** `python scripts/smoke.py` runs every command as a user runs it, each in a process of its own, on
  a small folder of notes; with `--models` also what loads the models: `embed`, the semantic searches, `bench` and
  `sift serve` with its page and MCP endpoint. A new command or option gets a line there.
- **Pull requests:** `main` is protected; pull requests are squash-merged once every check passes. CI runs the tests on
  Linux (x86 and ARM), macOS and Windows for Python 3.11 to 3.15 and builds the package, on pull requests and again on
  each merge commit.
- **Changelog:** user-visible changes go under "Unreleased" in [CHANGELOG.md](../CHANGELOG.md).
- **Measurements** in docs state the hardware by kind (a GPU, 8 CPU threads), not by model, and leave out what else
  ran on the machine.

## Releasing

1. In a pull request, set the version in `sift/__init__.py` and move the "Unreleased" changes in `CHANGELOG.md` under
   it, with its compare link at the bottom; CI checks the changelog has that section.
2. Merge it. Once CI passes on the merge commit, the release workflow publishes that commit: it builds the package,
   publishes it to PyPI by trusted publishing (no tokens; the `pypi` environment deploys from `main` only), tags the
   commit and creates a GitHub release with that version's changelog. It never runs for a failed CI run.
