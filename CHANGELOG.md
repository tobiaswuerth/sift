# Changelog

All notable changes to sift. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/). Releases before 2.0 are listed in
[the changelog of 1.15.2](https://github.com/tobiaswuerth/sift/blob/v1.15.2/CHANGELOG.md).

## [Unreleased]

### Fixed

- In a terminal that shows colours, `sift worker` printed the colour codes of its hint on where to set
  `SIFT_EMBED_REMOTE` as text (`\x1b[2m…\x1b[0m`) instead of dimming it.
- Ctrl+C, or a signal to stop, that reached `sift serve` after it wrote the file saying where it listens but before
  it began to answer could end it with exit code 130 or 1, at times leaving that file behind, or leave it hanging. It
  now stops as at any other time.

### Changed

- Semantic and hybrid search scan the vectors' codes in the compiled extension, on all cores and without copying
  them: over tens of millions of passages, that part of a search takes about half as long.

## [2.0.0] - 2026-11-28

A rebuild of sift: a new store layout, a keyword index of its own built by a compiled Rust core, and a home process
that serves agents, the browser and the command line. sift 2 does not open the stores of sift 1.x: bring them in
once with `sift import`, as [Updating](https://github.com/tobiaswuerth/sift/blob/main/README.md#updating) describes.

### Added

- `sift import` brings in the collections of a 1.x home with their settings, keeping the vectors of unchanged
  passages.
- `sift serve` is the home process: one per home, on 127.0.0.1, serving MCP, the UI and the CLI with the models
  loaded once. `sift serve --stdio` connects to it, or starts one that exits after 15 idle minutes.
- `sift jobs` lists the home process's background jobs; `--stop` stops one.
- `--offset` and `--budget` on every search, `--candidates` on `query`, and all search options on `mentions`.
- `sift add --agents` and `--no-agents`, `sift status --unreadable`, `sift serve --idle-exit`, `sift ui --no-browser`.
- `--json` and `--color` on every command; `NO_COLOR` and `TERM=dumb` turn colours off.
- [docs/limits.md](https://github.com/tobiaswuerth/sift/blob/main/docs/limits.md) lists every bound sift sets.

### Changed

- Requires Python 3.12 or newer with SQLite 3.38 or newer, as the Python builds of uv, python.org and Homebrew have.
  A Python linked against an older system SQLite, such as Ubuntu 22.04's 3.37, is refused with an explanation.
- Ships a compiled extension, written in Rust, as wheels for Linux on x86-64 and ARM (glibc 2.28 or newer), macOS on
  Apple silicon and Windows on x86-64, for every CPython from 3.12 on. Elsewhere, and on free-threaded builds, pip
  builds it from source, which needs [rustup](https://rustup.rs) (it installs the pinned Rust 1.100.0) and a C
  compiler. Each package includes the licences of the Rust crates built into it.
- Requires PyTorch 2.14 and torchvision 0.29 or newer.
- Each collection is one store, listed in `catalog.sqlite`. Store layouts change through numbered migrations that a
  newer sift applies once; each store records whether older versions may read or write it, and refuses them
  otherwise with a request to upgrade.
- Keyword search uses sift's own index, written in the same transaction as the documents, so it is current as soon as
  `sift index` ends. Terms follow SQLite's `unicode61` tokenizer without diacritics, ranked by BM25. Malformed
  queries, such as one with a word starting with `-` (write `NOT word`), are refused with the position of the error.
- `sift index` builds a new topical order beside the live index and switches to it in one transaction, so keyword
  search keeps answering; an interrupted build resumes on the next run.
- `sift embed` sorts passages by length, writes each batch as soon as it is embedded, and on an NVIDIA GPU compiles
  the model before embedding 20,000 passages or more: 20 to 60 seconds once per process, then about twice the
  throughput. On Windows this uses `triton-windows`. If compiling fails, the model runs uncompiled, and where the
  installed Triton is another release than PyTorch was built for, sift names the one to install.
- The MCP tools are now `search` (hybrid, keyword or semantic; up to 20 queries per call), `get`, `related`,
  `collections` and `reindex`. Agents need a new session to see them.
- MCP `get` takes `documents`, each an id or `{"id": ..., "first_line": ..., "last_line": ...}`, in place of
  `id` and `lines`.
- JSON Lines records and mail attachments are identified as `FILE#KEY`, such as `papers/2024.jsonl#2401.00001`.
- `--json` output is always an object, and errors are too:
  `{"error": {"kind": ..., "code": ..., "message": ..., "remedy": ..., "field": ...}}`. Search results are under
  `hits`, with `collection`, `path`, `first_line` and `last_line` per hit; `sift status` reports `home` and per
  collection `folder`, `agents` and `contents`; `sift links` reports `out` and `into`, and `sift unresolved` reports
  `targets`.
- Exit code 2 means a request was refused as asked, 130 that it was stopped (Ctrl+C), and 1 any other failure.
- Unknown `SIFT_` environment variables are refused by name.
- `sift status` prints a table.
- Renamed options: `sift index --file` (repeatable) replaces `--files`; `sift add --id-field`, `--title-field` and
  `--text-fields` replace `--id`, `--title` and `--text`.
- Folders containing `.obsidian/` are titled by file name unless `--title-from` says otherwise.
- Collection names are limited to 100 characters and may not be a Windows device name (`con`, `nul`, `com1`, …).
- `sift move` works while `sift serve` runs.
- Models other than the defaults must be named at a full commit (`owner/model@<40 hex digits>`) or as a local folder;
  branches, tags and short commits are refused.
- `SIFT_UPDATE_CHECK` is off by default. When on, a command run in a terminal reports a newer version and the
  command that installs it.
- `sift worker` accepts at most 256 texts per request and serves only clients of its own major version.

### Removed

- `sift serve --host` and `--allow-host`, and `sift ui --host`, `--port` and `--allow-host`: the home process serves
  this machine only. `sift worker` still lends a GPU to other machines.
- The keyword syntax `NEAR(...)`, column filters (`meta:`, `title:`) and `^`.
- Self-updating, with `update.lock` and the update logs.
- The Numba dependency.

### Security

- CLI commands authenticate to the home process with a token stored, with its port, in a per-user folder outside the
  home (`%LOCALAPPDATA%\sift` on Windows, `$XDG_STATE_HOME/sift` or `~/.local/state/sift` elsewhere) and readable
  only by the user. MCP and the UI answer only requests addressed to `127.0.0.1` or `localhost`; MCP rejects requests
  from web pages of other sites, and other sites cannot frame, load or trigger the UI
  ([security policy](https://github.com/tobiaswuerth/sift/blob/main/docs/SECURITY.md)).

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/tobiaswuerth/sift/compare/v1.15.2...v2.0.0
