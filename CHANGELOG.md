# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `similar`: documents like a given one, by its id, e.g. more papers like one found (MCP tool, `sift similar ID`).
  It runs no model: half a second over 3 million arXiv abstracts, with or without a GPU.
- Filters take a list for any of its values, `{"categories": ["cs.RO", "cs.CV"]}`, and `{"not": ...}` for none of
  them. On the command line: `-f KEY=VALUE` repeated, and `-f KEY!=VALUE`.
- `sort` orders the results by a front-matter field instead of score, e.g. `"-created"` for the newest first
  (`--sort=-created`).
- `get` takes `fields`, and then returns the document with them as `{id, title, fields, text}`
  (`sift get --fields url,created`). It takes a line number as well as a range.
- `all_passages` (`--all-passages`) returns every matching passage, as searches did before (see Changed).
- Every MCP tool parameter is described in the tool's schema, and the tools are marked read-only.
- The README explains setting up MCP and updating, step by step.
- `sift index` shows a progress bar, like `sift embed`.

### Changed

- Searches return one result per document, with its best passage: the passages of one long document no longer fill
  the results.
- `query` scores are the reranker's raw scores whatever activation its configuration names, also from a worker.
  With the default reranker, close matches score about 8 to 12, and unrelated results below 5.
- Keyword search weighs a title like the text. Titles were weighted double on top of being indexed twice, so a paper
  that only named a term in its title outranked an abstract about it.
- An empty environment variable counts as unset; `SIFT_HOME=` meant the current folder.

### Fixed

- A tool that failed for an unexpected reason told the agent only "Error executing tool"; the agent now gets the
  error's type and message, and the server's log the traceback.
- `get` ignored `fields` without a word: the MCP library drops arguments a tool doesn't take.

## [0.6.0] - 2026-10-20

### Added

- Through a worker, searches rerank there too, and the machine searching never loads PyTorch: on a notebook,
  `vsearch` and `query` take about 0.3 s instead of 10 to 25 s. Both machines need 0.6.
- `sift embed`, `vsearch`, `query`, `serve` and `worker` say at once that they started; `sift index` names the folder
  it scans and counts files one by one. The README shows how to upgrade.

### Changed

- The models compute in bfloat16 on NVIDIA GPUs that have it, and in float32 elsewhere, Apple GPUs included, instead
  of fp16.

### Fixed

- `vsearch` and `query` found fewer results, or none, when some vectors were broken. In fp16, EmbeddingGemma
  overflows now and then into NaN vectors, and the search ranked NaN first, so a few broken vectors pushed every real
  result out. NaN now ranks last, broken vectors are never stored, and the next `sift embed` drops those of older
  indexes once and embeds their chunks again.

## [0.5.1] - 2026-10-20

### Added

- `sift worker` lends a machine's GPU to sift on other machines, and `SIFT_EMBED_REMOTE` points sift there:
  `sift embed` sends its chunks to the worker without loading the model itself, and searches embed their queries
  there too. The worker only computes, sees no index or files, and gives the GPU back when idle.
- The worker speaks HTTPS with a certificate of its own and answers only requests with its token. It prints the
  address to set, `https://<key>@<machine>:8766`, whose key holds the token and the certificate's fingerprint:
  sift checks the certificate before it sends anything, and never shows the key. The README shows how to let the
  worker through the firewall. `cryptography` is a new dependency.

### Changed

- The instructions agents get from `sift serve`, and `sift --help`, speak of documents and notes instead of text
  and Markdown files.

## [0.5.0] - 2026-10-20

Tagged but never published: its worker spoke plain HTTP, without a key. 0.5.1 replaces it.

## [0.4.1] - 2026-10-20

### Fixed

- Office files whose XML compresses better than 100:1, common in presentations and spreadsheets, were skipped as
  a "potential ZIP bomb". sift now accepts up to 10,000:1, beyond anything deflate produces; Xberg's caps on
  unpacked size (100 MB a file, 500 MB in all) still stop zip bombs.
- `sift embed` showed nothing for a long time without a GPU: its first progress line came after eight batches of
  the longest chunks. It now says what it embeds before loading the model, then shows a progress bar with the
  chunks done, the speed and the time taken and left, updated after every batch and redrawn every second.

## [0.4.0] - 2026-10-20

### Added

- PDF, Word, PowerPoint and Excel files, OpenDocument text, presentations and spreadsheets, RTF, EPUB and HTML,
  converted to Markdown by [Xberg](https://github.com/xberg-io/xberg), a new dependency. Their properties (title,
  authors, dates, ...) become metadata. Damaged files and scans without text are skipped and reported.
- Each kind of file has a reader, chosen by extension, so a new format needs only a new reader.

### Changed

- The default reranker is `cross-encoder/ettin-reranker-150m-v1` (Apache-2.0), replacing
  `Alibaba-NLP/gte-reranker-modernbert-base`: the same size and speed, better in Hugging Face's published comparison
  (MTEB English retrieval and NanoBEIR, reranking six embedders' results, EmbeddingGemma's among them), and it
  improves on the first-stage ranking of the scientific sets SciFact, NFCorpus and SCIDOCS. It downloads on first
  use; nothing needs re-indexing or re-embedding. `SIFT_RERANK_MODEL=Alibaba-NLP/gte-reranker-modernbert-base` keeps the old one.

### Fixed

- A chunk's line range no longer includes the blank lines around its text, such as the one after front matter.
  Chunks already indexed keep their line numbers until their file changes.

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

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/tobiaswuerth/sift/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/tobiaswuerth/sift/compare/v0.4.1...v0.5.1
[0.5.0]: https://github.com/tobiaswuerth/sift/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/tobiaswuerth/sift/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/tobiaswuerth/sift/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/tobiaswuerth/sift/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tobiaswuerth/sift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tobiaswuerth/sift/releases/tag/v0.1.0
