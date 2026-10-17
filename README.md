# sift

[![CI](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast local search over your text and Markdown files, built for AI agents.

sift indexes folders of `.md` and `.txt` files, embeds them on your GPU, and runs a server that agents query over MCP.
It offers three kinds of search: exact keyword matching, semantic search, and hybrid retrieval with reranking for RAG.
Everything runs locally. No cloud services, no API keys.

## Install

Requires Python 3.11 or newer. An NVIDIA GPU or an Apple silicon Mac is strongly recommended for embedding; everything
also runs on CPU.

```bash
git clone https://github.com/tobiaswuerth/sift
cd sift
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # CUDA build of PyTorch; edit the first line for CPU or another CUDA version
pip install -e .
```

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

| Variable | Default | Purpose |
|---|---|---|
| `SIFT_HOME` | `~/.sift` | Where the index lives |
| `SIFT_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Any Sentence Transformers embedding model |
| `SIFT_RERANK_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any Sentence Transformers cross-encoder |
| `SIFT_BATCH_SIZE` | `128` | Embedding batch size; halved automatically on out-of-memory |
| `SIFT_DEVICE` | `cuda` if available, else `mps`, else `cpu` | Force a device, e.g. `cpu` while the GPU is busy |
| `SIFT_WORKERS` | CPUs available minus one | Maximum number of processes parsing files during `sift index` |

After changing the embedding model, run `sift embed --reset`. Models are downloaded from Hugging Face on first use.

## How it works

- **Indexing:** worker processes read and parse files in parallel while the previous batch is written; small
  updates skip the process start-up. Files are compared by modification time and size, then by content hash, so
  unchanged files cost nothing. Markdown is split into chunks of about 3,000 characters (about 750 tokens) on line
  boundaries, preferring headings, with a small overlap.
- **Storage:** one SQLite database in WAL mode holds documents, chunks, metadata, an FTS5 keyword index and the
  vectors (fp16).
- **Embedding:** PyTorch in fp16 on the GPU, in large batches sorted by length to keep padding low. Each slab of
  20,000 chunks is committed, so progress is never lost.
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
pip install -e ".[dev]"
pre-commit install      # ruff lint and format, whitespace and YAML/TOML checks on every commit
pytest --cov            # 100% line and branch coverage is required
```

The tests replace the embedding and reranking models with small deterministic fakes, so they need no GPU and no
model downloads; everything else, including the MCP server, runs for real. Every bug fix comes with a regression test.

GitHub Actions runs the pre-commit checks and the tests on Linux (x86 and ARM), macOS and Windows for Python 3.11 to
3.15 on every push and pull request. Dependabot keeps dependencies and actions up to date.

## License

[MIT](LICENSE)
