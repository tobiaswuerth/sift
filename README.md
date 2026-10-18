# sift

[![PyPI](https://img.shields.io/pypi/v/sift-rag)](https://pypi.org/project/sift-rag/)
[![Python](https://img.shields.io/pypi/pyversions/sift-rag)](https://pypi.org/project/sift-rag/)
[![CI](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast local search over your text and Markdown files, built for AI agents.

sift indexes folders of `.md` and `.txt` files, embeds them on your GPU, and runs a server that agents query over MCP.
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
  `pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128`.

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
sift status                            # collections and counts
sift rm papers                         # remove a collection from the index; its files stay on disk
sift serve                             # MCP server on http://127.0.0.1:8765/mcp
```

`sift index` and `sift embed` are incremental: run them again after files change, and only new or changed files are
processed. Deleted files are removed. An interrupted `sift embed` continues where it stopped. A running server picks up
new vectors on its own.

Search commands accept `-k` (number of results), `-c` (collection), `--path` (glob within the collection, e.g.
`"2024/*"`), `-f KEY=VALUE` (front-matter filter, repeatable) and `--json`.

### Front matter

YAML front matter with flat `key: value` lines becomes filterable metadata. A list field matches when it contains the
value:

```markdown
---
title: "Brush stroke planning"
categories: ["cs.RO", "cs.GR"]
year: 2024
---
```

```bash
sift query "stroke order" -f categories=cs.RO -f year=2024
```

The `title` field, or else the first heading, or else the file name, becomes the document title.

## MCP

| Tool | Purpose |
|---|---|
| `search` | Exact keyword search, ranked by BM25. Supports `"phrases"`, `prefix*`, `AND`, `OR`, `NOT`. |
| `vsearch` | Semantic search. |
| `query` | Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder. |
| `get` | Full text of a document by its id. |
| `status` | Collections and counts. |

The search tools take `query`, `k`, `collection`, `path` and `filters`. Every result carries `id`
(`collection/path`), `title`, `lines`, `score` and the chunk `text`, so agents can cite their sources.

Claude Code, with the server running (`sift serve`), shared by any number of agents:

```bash
claude mcp add --transport http sift http://127.0.0.1:8765/mcp
```

Or let Claude Code start its own server over stdio:

```bash
claude mcp add sift -- sift serve --stdio
```

## Configuration

None needed: sift picks the device, the GPU memory budget, the batch size and the number of worker processes itself.
Three optional settings exist for special cases:

| Variable | Default | Purpose |
|---|---|---|
| `SIFT_HOME` | `~/.sift` | Where the index lives |
| `SIFT_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Any Sentence Transformers embedding model |
| `SIFT_RERANK_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any Sentence Transformers cross-encoder |

After changing the embedding model, run `sift embed --reset`. Models are downloaded from Hugging Face on first use.

## How it works

- **Indexing:** worker processes read and parse files in parallel while the previous batch is written; small
  updates skip the process start-up. Files are compared by modification time and size, then by content hash, so
  unchanged files cost nothing. Markdown is split into chunks of about 3,000 characters (about 750 tokens) on line
  boundaries, preferring headings, with a small overlap.
- **Storage:** one SQLite database in WAL mode holds documents, chunks, metadata, an FTS5 keyword index and the
  vectors (fp16).
- **Embedding:** PyTorch in fp16 on the GPU (NVIDIA, else Apple), in batches sorted by length to keep padding low.
  Each slab of 20,000 chunks is committed, so progress is never lost.
- **GPU memory:** sift shares the GPU with other programs. On NVIDIA GPUs it caps itself at the memory that is free
  when it starts, minus headroom (10% of the GPU, at least 1 GB), asking NVML as well as CUDA, since on Windows CUDA
  counts memory held by other programs as free. It then measures how much memory its longest text needs at growing
  batch sizes and uses the largest batch that fills 90% of that budget; if a batch still doesn't fit, it is halved.
  With less than 2 GB to spare, sift uses the CPU instead.
- **`search`:** SQLite FTS5 with BM25, titles weighted double. Input that isn't valid FTS5 syntax is treated as exact
  phrases.
- **`vsearch`:** exact nearest-neighbour search as one matrix multiply over all vectors on the GPU. No approximate
  index, so nothing to tune and no recall loss; a million 1024-dimensional vectors take about 2 GB of VRAM.
- **`query`:** the top 40 keyword and top 40 semantic candidates are merged by reciprocal rank fusion and reranked in
  one GPU batch.
- **Server:** `sift serve` loads models and vectors once and keeps them in memory.

## Scope

In scope: plain text and Markdown files; Linux, macOS and Windows; NVIDIA GPUs (CUDA) and Apple GPUs (MPS), otherwise
CPU; one user, any number of local agents.

Out of scope for now: PDF, Word and HTML parsing (convert to Markdown first); answer generation (sift retrieves, the
agent writes); approximate nearest-neighbour indexes; live file watching; remote access and authentication.

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
