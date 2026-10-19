# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- `sift serve` gives its GPU memory back when idle. Models and vectors load into RAM, move to the GPU when a
  search needs them, and return to RAM after 5 minutes without GPU use. Once released, the memory is free
  for other programs; the first search afterwards takes about a second longer.

## [0.1.0] - 2026-10-18

First release.

### Added

- `sift add`, `rm`, `index`, `embed`, `search`, `vsearch`, `query`, `get`, `status` and `serve`; also available as
  `sift-rag`, so `uvx sift-rag serve` works.
- Incremental indexing of `.md`, `.markdown` and `.txt` files in parallel worker processes, with filterable YAML
  front matter.
- Embedding on NVIDIA GPUs (CUDA), Apple GPUs (MPS) or the CPU. sift caps its GPU memory below what other programs
  leave free, calibrates the batch size by measurement, and reports progress.
- Keyword search (SQLite FTS5, BM25), exact GPU vector search, and hybrid search with cross-encoder reranking.
- An MCP server (streamable HTTP and stdio) whose instructions tell agents what is indexed, and whose errors tell
  them what went wrong.
- A start-up hint when an NVIDIA GPU can't be used, with the command that fixes it.

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tobiaswuerth/sift/releases/tag/v0.1.0
