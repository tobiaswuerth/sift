# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/). Releases before 2.0 are listed in
[the changelog of 1.15.2](https://github.com/tobiaswuerth/sift/blob/v1.15.2/CHANGELOG.md).

## [Unreleased]

sift 2: stores of a new layout with a keyword index of its own, built by a compiled core, and a home process that
answers agents, the browser and your own commands. A home of an older sift is brought in by `sift import`, once, as
the [README](README.md#updating) says; until a collection is added there, any command in such a home says so too.

### Changed

- sift needs Python 3.12 or newer, built with SQLite 3.38 or newer, as uv's, python.org's and Homebrew's Pythons are.
  A Python built against an older system SQLite (Ubuntu 22.04 has 3.37) is refused, saying so.
- sift ships a compiled extension written in Rust, which builds, merges and searches the keyword index and puts the
  passages in their topical order, on all cores. PyPI holds it built for Linux on x86-64 and ARM (glibc 2.28 or
  newer), macOS on Apple silicon and Windows on x86-64, one wheel each for every CPython from 3.12 on. Anywhere else,
  and on the free-threaded builds, pip builds it from the sources, which needs Rust 1.100 or newer and a C compiler.
  The licences of the Rust crates built into it are in each package, beside sift's own.
- Each collection is one store, listed in `catalog.sqlite`. A store's layout changes by numbered migrations, which the
  first command of a newer sift runs once. Each store records which older sifts may still read it and which may write
  it: an older sift opens a store a newer one changed in a way it can live with, reads one it could not keep right,
  and refuses the rest, saying to upgrade. A keyword index of a newer format is refused likewise, never made anew.
- Keyword search reads sift's own index, written in the documents' transaction, so nothing is behind after
  `sift index`. Its terms are those of SQLite's `unicode61` tokenizer, without diacritics, ranked by BM25. A query sift
  can't read is refused, saying where: a quote or a parenthesis not closed, a `)` closing nothing, a word starting
  with `-` (`NOT word` leaves it out).
- When `sift index` puts the passages in their topical order, it builds the index of that order beside the one
  searches read, its passages read by eight readers at once, and switches to it in one transaction: keyword search
  answers throughout, and an order stopped midway is taken up by the next run.
- `sift serve` is the home process: one per home, on 127.0.0.1, for agents (MCP), the browser (the UI) and your own
  `sift` commands, which it answers with its models and codes loaded. `sift serve --stdio` talks to it, starting one
  if none runs (on Windows in a minimized window), which leaves after 15 idle minutes; `sift ui` opens its pages, or
  serves the home itself on a free port.
- The MCP tools are five: `search` (hybrid, keyword or semantic, up to 20 queries a call, paged by `offset`), `get`,
  `related` (documents alike, a note's links, its backlinks, its mentions), `collections` (with the jobs that run),
  and `reindex`, which runs as a job of the home process. Agents need a new session to see them.
- A JSON Lines record and a mail attachment are named after their file and a `#`: `papers/2024.jsonl#2401.00001` and
  `mail/offer.eml#contract.pdf`.
- `--json` answers are objects, and so are refusals and failures:
  `{"error": {"kind": ..., "code": ..., "message": ..., "remedy": ..., "field": ...}}`. A search gives `hits`, with
  what was searched, left out, loosened or cut short beside them, and each hit its `collection`, `path`, `first_line`
  and `last_line`. `sift status` gives `home`, and for each collection `folder`, `agents` and its counts under
  `contents`: `documents`, `chunks`, `embedded`, `model`, `stored_bytes`, `unreadable`. `sift links --json` gives `out`
  and `into`, each `null` where it was not asked for, a linking document its `lines` and `through`;
  `sift unresolved --json` gives `targets`.
- A request sift refuses as written (an unknown flag, a value out of range, a query it can't read) ends with exit
  code 2, any other failure with 1.
- A `SIFT_` variable sift doesn't know, a misspelt one or one of a newer sift, is refused by every command, naming it.
- `sift status` prints a table; `--json` gives JSON.
- `sift index --file A --file B` names files to read, in place of `--files`; `sift add` names the fields of JSON
  Lines with `--id-field`, `--title-field` and `--text-fields`, in place of `--id`, `--title` and `--text`.
- A folder holding `.obsidian/` is titled by file name unless `--title-from` says otherwise, as Obsidian names its
  notes.
- A collection's name has at most 100 characters and is none Windows keeps for a device (`con`, `nul`, `com1`, …),
  on every system.
- `sift move` runs beside `sift serve`: it waits for the searches reading the store, and the next searches wait for
  it.
- A model other than the defaults is named at a whole commit (`owner/model@<40 hex digits>`) or as a folder: a
  branch, a tag or a short commit is refused, as each may name other weights tomorrow.
- `SIFT_UPDATE_CHECK` is off by default. Turned on, a command run in a terminal says when a newer sift is out and which
  command updates it; sift never updates itself.
- `sift worker` takes at most 256 texts a request, and serves sifts of its own major version.
- `sift embed` reads 16,384 passages at a time and embeds them longest first, writing each batch's vectors as soon
  as it is done, the next two batches' words cut while the model runs on the one before. On an NVIDIA GPU, with
  20,000 passages or more waiting, it compiles the model first: 20 to 60 s, once a process and again after the home
  process let the idle model go, shown as its own step, and then about twice as many passages a second. Fewer
  passages, queries and other devices use the model as it is. On Windows sift depends on `triton-windows` for it;
  PyTorch brings Triton on Linux.

### Removed

- `sift serve --host` and `--allow-host`, and `sift ui --host`, `--port` and `--allow-host`: the home process
  answers this machine alone. To lend a GPU to other machines, `sift worker` remains.
- The keyword syntax `NEAR(...)`, column filters (`meta:`, `title:`) and `^`.
- Updating itself, with `update.lock` and the update logs.
- Numba: the codes are scanned with NumPy.

### Added

- `sift import`: the collections of an older home, with their choices and vectors, into this one.
- `sift jobs`: what the home process does in the background, and `--stop` to stop a job.
- `--offset` and `--budget` on every search, `--candidates` on `query`, and every search option on `mentions`.
- `sift add --agents` and `--no-agents`: whether agents see a collection.
- `sift status --unreadable`, `sift serve --idle-exit`, `sift ui --no-browser`.
- `--json` and `--color` on every command, before or after its name; `NO_COLOR` and `TERM=dumb` turn colours off.
- [docs/limits.md](docs/limits.md): every bound sift sets.

### Security

- Your commands send the home process a token, kept with its port in your own folder (`%LOCALAPPDATA%\sift` on
  Windows, `~/.local/state/sift` elsewhere), readable by you alone, never in the home: it grants the whole command
  line. MCP and the UI take none, answer only requests addressed to `127.0.0.1` or `localhost`, and MCP refuses what
  web pages of other sites send; no other site can show the UI's pages or files in a frame, load them, or make the
  UI search in the background ([security](docs/SECURITY.md)).

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v1.15.2...HEAD
