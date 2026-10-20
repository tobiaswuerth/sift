# sift

[![CI](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sift-rag)](https://pypi.org/project/sift-rag/)
[![Python](https://img.shields.io/pypi/pyversions/sift-rag?logo=python&logoColor=white)](https://pypi.org/project/sift-rag/)
[![Downloads](https://img.shields.io/pypi/dm/sift-rag)](https://pypistats.org/packages/sift-rag)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/tobiaswuerth/sift)](https://github.com/tobiaswuerth/sift/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CodeQL](https://github.com/tobiaswuerth/sift/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/tobiaswuerth/sift/actions/workflows/github-code-scanning/codeql)

Fast local search over your documents and notes, built for AI agents.

sift indexes folders of Markdown, text, PDF, office and JSON Lines files, embeds them on your GPU, and runs a server
that agents query over MCP.
It offers three kinds of search: exact keyword matching, semantic search, and hybrid retrieval with reranking for RAG.
Everything runs locally. No cloud services, no API keys.

## Install

Requires Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --torch-backend auto sift-rag   # installs the `sift` command
```

Or run it without installing: `uvx --torch-backend auto sift-rag serve`. With pip, in a virtual environment:
`pip install sift-rag`.

On first use, sift downloads its two models from Hugging Face (about 1.5 GB).

### GPU support

sift uses an NVIDIA GPU (CUDA) if there is one, else an Apple GPU, else the CPU, which works but embeds slowly.

- **Linux, macOS:** nothing to do. PyTorch from PyPI includes CUDA on Linux and uses the Apple GPU on macOS.
- **Windows:** PyTorch from PyPI has no CUDA support. uv's `--torch-backend auto`, used above, picks the CUDA build
  that matches your driver. With pip, install it afterwards:
  `pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

If an NVIDIA GPU sits unused, sift says so on start-up and prints the command that fixes it.

## Usage

```bash
sift add ~/papers --name papers        # register a folder as a collection
sift index                             # scan files, chunk them, build the keyword index
sift embed                             # embed every chunk that has no vector yet
sift search '"reward shaping"'         # exact keyword search
sift vsearch "how do robots plan brush strokes"
sift query "mixing pigments to match a target colour"
sift get papers/2401.02225.md          # full document
sift get papers/2401.02225.md --lines 14-23   # only some lines, e.g. those of a search hit
sift status                            # collections and counts
sift rm papers                         # remove a collection from the index; its files stay on disk
sift serve                             # MCP server on http://127.0.0.1:8765/mcp
```

`sift index` and `sift embed` are incremental: run them again after files change, and only new or changed files are
processed. Deleted files are removed. An interrupted `sift embed` continues where it stopped. A running server picks up
new vectors on its own.

Search commands accept `-k` (number of results), `-c` (collection), `--path` (glob within the collection, e.g.
`"2024/*"`), `-f KEY=VALUE` (front-matter filter, repeatable), `--fields` (front-matter fields to show with each hit,
e.g. `url,created`, or `*` for all), `--compact` (no text) and `--json`.

`search` requires every term. If no chunk has them all and the query has no operators (`AND`, `OR`, `NOT`, `NEAR`),
it returns the chunks that have any of them, those with more terms first.

### Front matter

YAML front matter with flat `key: value` lines becomes filterable metadata; lists can be inline or one `- item` per
line. A list field matches when it contains the value:

```markdown
---
title: "Brush stroke planning"
categories: ["cs.RO", "cs.GR"]
year: 2024
tags:
  - painting
  - robots
---
```

```bash
sift query "stroke order" -f categories=cs.RO -f year=2024
```

Filters can also be ranges, with `>`, `>=`, `<` and `<=`. Dates compare at the precision given, so `created<=2024`
includes all of 2024 and `created>2024` starts in 2025; a number compares with the number a date starts with:

```bash
sift search pigment -f "created>=2024-06" -f "created<2025"
```

The `title` field, or else the first heading outside code blocks, or else the file name, becomes the document title.

### JSON Lines

A `.jsonl` (or `.ndjson`) file holds one document per line, as a JSON object. sift finds the fields by name:

| Part | Field |
|---|---|
| id | `id` or `_id`, else the line number |
| title | `title`, `name`, `headline` or `subject` |
| text | the first of `text`, `content`, `body`, `abstract`, `description` and `summary`, else the longest string; a list of strings becomes paragraphs |
| metadata | every other field, for filters and `fields`, as front matter is for Markdown |

A record's id is its file's path plus its id: line `{"id": "2101.00001", ...}` of `2021/2021-01.jsonl` becomes
`papers/2021/2021-01/2101.00001`, which `get` also finds as `papers/2101.00001`. Other field names can be chosen
when adding the collection:

```bash
sift add ~/qa --id key --title question --text question,answer
```

Lines that aren't JSON objects, have no text or repeat an id are skipped and reported.

### Documents

PDF, Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`), OpenDocument (`.odt`, `.odp`, `.ods`), RTF, EPUB and HTML
files are converted to Markdown by [Xberg](https://github.com/xberg-io/xberg). Headings stay headings, slides
become sections and sheets become tables, and `get` returns that Markdown. The document's properties become metadata
as far as the file has them: `title`, `authors`, `subject`, `keywords`, `language`, `created` and `modified`. The
title is the document's own, else the first heading, else the file name.

sift runs no OCR, so a scanned PDF without a text layer has no text. Such files are skipped and reported, as are
damaged ones, until they change. Older binary Office files (`.doc`, `.ppt`, `.xls`) aren't read.

## MCP

| Tool | Purpose |
|---|---|
| `search` | Exact keyword search, ranked by BM25. Supports `"phrases"`, `prefix*`, `AND`, `OR`, `NOT`. |
| `vsearch` | Semantic search. |
| `query` | Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder. |
| `get` | Text of an indexed document by its id: all of it, or only `lines` such as a hit's `"17-26"`. An id may leave out folders if the file name is unique. |
| `status` | Collections, counts and front-matter fields. |

The search tools take `query`, `k`, `collection`, `path`, `filters`, `fields` and `compact`. Every result carries `id`
(`collection/path`), `title`, `lines`, `score` and the chunk `text`, so agents can cite their sources. `fields` adds
front-matter fields to each result, e.g. `["url", "created"]`, and `compact: true` leaves out the text, so an agent
can survey many results without filling its context and then read the ones that matter. Filters take values or
ranges, e.g. `{"created": {">=": "2024-06", "<": "2025"}}`. The server tells agents when they connect which
collections there are and which front-matter fields each has.

### Claude Code

With the server running (`sift serve`), register it once for all your projects:

```bash
claude mcp add --scope user --transport http sift http://127.0.0.1:8765/mcp
```

Keep `--scope user`. Without it, `claude mcp add` registers sift for the current folder only, and sessions started
anywhere else don't see it, including those in the Claude desktop app's Code tab. Start a new session afterwards:
running sessions don't pick up new servers. One server serves any number of agents, with the models loaded once.

After upgrading sift, restart `sift serve` and start a new Claude Code session: a running session can keep an
outdated list of sift's tools and their parameters, even after reconnecting.

Or let every Claude Code session start a server of its own over stdio. Then each session loads the models into
memory itself:

```bash
claude mcp add --scope user sift -- sift serve --stdio
```

A registration made without `--scope user` can be removed in the folder where it was made, with
`claude mcp remove sift --scope local`.

Connectors in claude.ai and in the Chat tab of the Claude apps can't use sift: claude.ai reaches connectors from
Anthropic's servers, which can't reach a server on your computer.

## Configuration

None needed: sift picks the device, the GPU memory budget, the batch size and the number of worker processes itself.
Three optional settings exist for special cases:

| Variable | Default | Purpose |
|---|---|---|
| `SIFT_HOME` | `~/.sift` | Where the index lives |
| `SIFT_EMBED_MODEL` | `google/embeddinggemma-2` | Any Sentence Transformers embedding model |
| `SIFT_RERANK_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any Sentence Transformers cross-encoder |

After changing the embedding model, run `sift embed --reset`. Models are downloaded from Hugging Face on first use.

## How it works

- **Indexing:** worker processes read and parse files in parallel while the previous batch is written; small
  updates skip the process start-up. Files are compared by modification time and size, so unchanged files cost
  nothing, and documents by content hash, so a changed JSON Lines file re-indexes only its changed records. Text
  files are read as UTF-8, or as UTF-16 when they start with its byte order mark (Windows PowerShell 5.1 writes
  those). Each kind of file has a reader, chosen by extension, that turns it into documents; adding a format means
  adding a reader. Markdown, including documents converted to it, is split into chunks of about 3,000 characters
  (about 750 tokens) on line boundaries, preferring headings, with a small overlap.
- **Storage:** one SQLite database in WAL mode holds every document's text, metadata, the FTS5 keyword index and
  the vectors (fp16). The text is stored once: chunks are spans of it, and the keyword index reads it through a view.
  Searching and `get` never touch the files, so results and documents always agree, until the next `sift index`.
- **Embedding:** PyTorch in fp16 on the GPU (NVIDIA, else Apple), in batches sorted by length to keep padding low.
  Each slab of 20,000 chunks is committed, so progress is never lost.
- **GPU memory:** sift shares the GPU with other programs. On NVIDIA GPUs it caps itself at the memory that is free
  when it starts, minus headroom (10% of the GPU, at least 1 GB), asking NVML as well as CUDA, since on Windows CUDA
  counts memory held by other programs as free. It then measures how much memory its longest text needs at growing
  batch sizes and uses the largest batch that fills 90% of that budget; if a batch still doesn't fit, it is halved.
  With less than 2 GB to spare, sift uses the CPU instead.
- **Idle GPU:** models and vectors wait in RAM and move to the GPU when a search needs them. After 5 minutes
  without GPU use, `sift serve` moves them back to RAM and hands the memory back; only CUDA's own context
  (about 0.3 GB) stays. The first search afterwards takes about a second longer; if other programs have taken the
  memory meanwhile, sift stays on the CPU rather than squeeze them. Keyword search never uses the GPU.
- **`search`:** SQLite FTS5 with BM25, titles weighted double. Input that isn't valid FTS5 syntax is treated as exact
  phrases.
- **`vsearch`:** exact nearest-neighbour search as one matrix multiply over all vectors on the GPU. No approximate
  index, so nothing to tune and no recall loss; a million 768-dimensional vectors take about 1.5 GB of VRAM.
- **`query`:** the top 40 (or `k`, if larger) keyword and semantic candidates are merged by reciprocal rank fusion and reranked in
  one GPU batch.
- **Server:** `sift serve` loads models and vectors once and keeps them in memory.

## Scope

In scope: text, Markdown and JSON Lines files, PDF, office documents, e-books and web pages; Linux, macOS and Windows;
NVIDIA GPUs (CUDA) and Apple GPUs (MPS), otherwise CPU; one user, any number of local agents.

Out of scope for now: OCR of scans and images; answer generation (sift retrieves, the agent writes); approximate
nearest-neighbour indexes; live file watching; remote access and authentication.

## Development

```bash
git clone https://github.com/tobiaswuerth/sift
cd sift
uv venv && uv pip install --torch-backend auto -e ".[dev]"   # or with pip: pip install -r requirements.txt -e ".[dev]"
pre-commit install      # lint, format and config checks on every commit
pytest --cov            # 100% line and branch coverage is required
```

The tests replace the embedding and reranking models with small deterministic fakes, so they need no GPU and no
model downloads; everything else, including the MCP server, runs for real. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how changes and releases work.

## License

[MIT](LICENSE)
