# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-20

### Added

- JSON Lines files (`.jsonl`, `.ndjson`): one document per line, with the id, title and text found by name (`id`,
  `title`, `text`, `abstract`, ...) and every other field as metadata. `sift add --id/--title/--text` chooses other
  fields. A changed file re-indexes only its changed records.

### Changed

- **The index holds the documents' text** (once: chunks are spans of it), so `get` reads the index instead of the
  file, and results and documents always agree. **Indexes made by sift 0.2 or older must be rebuilt**: delete
  `sift.db` and run `sift add`, `sift index` and `sift embed` again. The layout now has a version, so later changes
  can migrate.
- The default embedding model is `google/embeddinggemma-2` (768 dimensions, Apache-2.0), replacing
  `Qwen/Qwen3-Embedding-0.6B`. Run `sift embed --reset` to switch an existing index. It needs Pillow and
  torchvision, now dependencies; with pip, install torchvision from the same index as torch.

## [0.2.0] - 2026-10-19

### Added

- Search results can carry front-matter fields (`fields`, e.g. `["url", "created"]`, or `["*"]` for all) and leave
  out the text (`compact`), so agents can survey many results cheaply. `--fields` and `--compact` on the CLI.
- Range filters on front-matter fields, e.g. `{"created": {">=": "2024-06", "<": "2025"}}` or
  `-f "created>=2024-06"`. Dates compare at the precision given.
- `get` returns only some lines if asked, e.g. a hit's `"17-26"` (`--lines` on the CLI).
- `get` finds a document by its file name if no other document in the collection has that name, e.g.
  `arxiv/2603.29315.md` for `arxiv/2026/03/2603.29315.md`; otherwise it lists the matches.
- The server tells agents which front-matter fields each collection has; `status` lists them too.

### Changed

- `sift serve` gives its GPU memory back when idle. Models and vectors load into RAM, move to the GPU when a
  search needs them, and return to RAM after 5 minutes without GPU use. Once released, the memory is free
  for other programs; the first search afterwards takes about a second longer. If other programs have taken the
  memory meanwhile, sift stays on the CPU instead.
- `search` returns chunks with any of the query's terms when none has them all, unless the query uses operators.
- `get` serves indexed documents only, never other files below a collection's folder, and returns `\n` line
  endings, like search results.
- Front matter lists can be written one `- item` per line, as Obsidian, Jekyll and Hugo do; comment lines are
  skipped, and a key without a value means null.
- Blank queries return nothing in every search mode, and `query` gathers at least `k` candidates per mode.
- Index worker processes are spawned on every platform; forking a process that runs threads can deadlock.

### Fixed

- `sift serve` could hang on Ctrl+C, however often it was pressed: on shutdown, uvicorn waits for every
  connection to close (Python 3.12+), and a client that left a request unfinished never closed its connection.
  The server now waits at most 3 seconds for open requests, then stops.
- `sift index` stalled for minutes on a heading followed by a long run of spaces.
- Titles: "# Learning C#" lost its "#", a bare "#" line took the next paragraph as title, and comments in fenced
  code blocks counted as headings.
- Line numbers in search hits were off after form feeds and other characters that `str.splitlines` treats as line
  ends.
- UTF-16 files (Windows PowerShell 5.1 writes them with `>`) were indexed as text full of NUL characters.
- Empty front matter (`---` twice) was indexed as text.
- A file name with bytes that aren't UTF-8 (Linux) aborted `sift index`; such files are now skipped with a message.
- A huge `k` or filter number crashed the search tools; `k` is capped, a filter number is a clear error.
- Calibrating the batch size divided by zero when memory didn't grow with the batch.
- `sift embed` crashed on chunks that had no text in the keyword index.
- The embedding progress divided by zero when the clock hadn't advanced.
- Ctrl+C on `sift serve` printed a traceback instead of stopping quietly; errors such as a port in use now print
  one line.
- Database errors reach MCP clients as messages, and the CLI reports an unusable `SIFT_HOME` instead of crashing.
- `sift add ~/notes` works where the shell doesn't expand `~`.

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

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/tobiaswuerth/sift/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tobiaswuerth/sift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tobiaswuerth/sift/releases/tag/v0.1.0
