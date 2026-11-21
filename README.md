# sift: local RAG for AI agents

[![CI](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sift-rag)](https://pypi.org/project/sift-rag/)
[![Python](https://img.shields.io/pypi/pyversions/sift-rag?logo=python&logoColor=white)](https://pypi.org/project/sift-rag/)
[![Downloads](https://img.shields.io/pypi/dm/sift-rag)](https://pypistats.org/packages/sift-rag)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/tobiaswuerth/sift/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/tobiaswuerth/sift)](https://github.com/tobiaswuerth/sift/blob/main/LICENSE)

**Local search over your documents, for AI agents.** sift indexes folders of documents and notes, embeds them on your
GPU, and serves keyword, semantic and hybrid search to Claude Code, Cursor and other agents as an MCP (Model Context
Protocol) server. Everything runs on your own machines: no cloud services, no API keys.

![sift ui: a keyword search over 3.2 million arXiv abstracts, filtered by category, with one paper open](https://raw.githubusercontent.com/tobiaswuerth/sift/main/docs/images/ui-search.png)

- **Four kinds of search:** keywords (BM25), meaning (vectors), hybrid with cross-encoder reranking for RAG, and
  documents like a given one.
- **Your files as they are:** Markdown with front matter, Obsidian vaults, text, PDF (scans too, by OCR), Word,
  PowerPoint, Excel, OpenDocument, RTF, EPUB, HTML, emails and JSON Lines, indexed incrementally. sift never writes to
  them.
- **Built for agents:** front-matter filters, results citing their source lines, compact results, several queries or
  documents per call; a note an agent just wrote is found by its next search.
- **Linked notes:** the links and backlinks of `[[wikilinks]]`, notes not written yet, and unlinked mentions.
- **A browser UI** to search, browse and read the index yourself.
- **Runs on what you have:** NVIDIA or Apple GPUs, the CPU, or another machine's GPU, on Linux, macOS and Windows; tens
  of millions of passages on one machine.

sift retrieves; it doesn't generate answers, watch files (an agent calls `reindex` after writing), OCR images other
than PDFs, or serve remote users with authentication.

## Install

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --torch-backend auto sift-rag   # installs the `sift` command
```

With pip, in a virtual environment: `pip install sift-rag`. The first use downloads two models from Hugging Face
(about 2 GB), each at the commit this version of sift pins it to: what their owners push later changes no vectors.

sift uses an NVIDIA GPU (CUDA) if there is one, else an Apple GPU, else the CPU, where `query` takes about 25 seconds a
question instead of 0.3. PyTorch from PyPI uses the GPU on Linux and macOS. On Windows, `--torch-backend auto` installs
the CUDA build for your driver; with pip, run `pip install --force-reinstall --no-deps torch torchvision --index-url
https://download.pytorch.org/whl/cu126` afterwards. If an NVIDIA GPU sits unused, sift says so and prints the fix. A
machine without a GPU can [use another machine's](https://github.com/tobiaswuerth/sift/blob/main/docs/worker.md).

## Quick start

```bash
sift add ~/handbook --name handbook    # register a folder as a collection
sift index                             # read its files and build the keyword index
sift embed                             # embed every passage that has no vector yet
sift query "who approves travel expenses abroad"
sift serve                             # MCP server on http://127.0.0.1:8765/mcp, and the UI under /ui
```

`sift index` and `sift embed` process only what changed since their last run; an interrupted `sift embed` continues
where it stopped. Or let your agent do it all: tell it to read and follow
[Setting sift up with an agent](https://github.com/tobiaswuerth/sift/blob/main/docs/agents.md).

## Connect an agent

Leave `sift serve` running and register it with Claude Code once, for all projects:

```bash
claude mcp add --scope user --transport http sift http://127.0.0.1:8765/mcp
```

New sessions then show sift in `claude mcp list`. Without `--scope user` sift is registered for the current folder
only (undo there with `claude mcp remove sift --scope local`). Other MCP clients, such as Cursor, VS Code or
Windsurf, connect to the same streamable HTTP endpoint; one server serves any number of agents with the models loaded
once. For a server per session over stdio: `claude mcp add --scope user sift -- sift serve --stdio`. Connectors in
claude.ai and the Claude apps' Chat tab can't use sift: they connect from Anthropic's servers, which can't reach your
computer.

| Tool | Purpose |
|---|---|
| `search` | Exact keyword search, ranked by BM25: `"phrases"`, `prefix*`, `AND`, `OR`, `NOT`, parentheses |
| `vsearch` | Semantic search |
| `query` | Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder |
| `similar` | Documents like a given one |
| `get` | Documents by id, whole or some `lines`, with front-matter `fields`; up to 100 at once |
| `status` | The collections: folders, counts, front-matter fields, links |
| `links` | A note's links and where they lead, and the notes linking to it |
| `unresolved` | Link targets no note answers to yet |
| `mentions` | Documents naming a note without linking to it |
| `reindex` | Read files again right after they were written or deleted; changes only sift's index, never a file |

Every result carries `id` (`collection/path`), `title`, `lines`, `score` and the passage's `text`. The search tools
take up to 20 queries per call, filters on front matter, paths and links, `compact` (no text), `adaptive` (only the
results before the largest drop in score) and more; the tools' schemas describe every parameter. When an agent
connects, the server tells it which tool fits which kind of question, how to research in rounds (several phrasings
at once, a compact survey, then the best few read), and which collections and front-matter fields exist.

## Commands

| Command | Does |
|---|---|
| `sift add FOLDER [--name NAME]` | Register a folder as a collection (`--store PATH` puts its index elsewhere, `--follow-links` reads files linked from outside it) |
| `sift index [NAME ...] [--files PATH ...]` | Read new and changed files, of every active collection or those named; `--files`: only these |
| `sift embed [NAME ...]` | Embed every passage without a vector |
| `sift search`, `vsearch`, `query` `TEXT` | Keyword, semantic and hybrid search |
| `sift similar ID` | Documents like this one |
| `sift get ID [ID ...] [--lines 14-23] [--fields a,b]` | Documents, or some of their lines |
| `sift links`, `unresolved`, `mentions` | Linked notes |
| `sift status` | Collections, their index files, sizes and counts, as JSON |
| `sift activate`, `deactivate NAME ...` | Take collections into searches, or leave them out |
| `sift move [NAME ...] TO` | Move collections' indexes to another folder or drive |
| `sift rm NAME` | Remove a collection and its index; its files stay |
| `sift ui [--host H] [--port N] [--allow-host NAME]` | Search, browse and read the index in your browser |
| `sift serve [--stdio] [--host H] [--port N] [--allow-host NAME]` | The MCP server, and the UI under `/ui` (port 8765) |
| `sift worker [--port N]` | Embed and rerank for other machines on this one's GPU (port 8766) |
| `sift bench QUESTIONS.json` | How well each search mode finds the answers you expect |

Search options, keyword syntax, filters, the formats sift reads, `.siftignore` and where indexes lie:
[usage](https://github.com/tobiaswuerth/sift/blob/main/docs/usage.md).

```bash
sift search '"notice period" AND (contract OR agreement)'
sift query "per diem abroad" -f departments=finance -f "created>=2024-06"
```

## Configuration

None needed: sift picks the device, its GPU memory budget and batch sizes itself. Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SIFT_HOME` | `~/.sift` | The catalog, logs, the worker's key, and the indexes unless moved |
| `SIFT_EMBED_MODEL` | `google/embeddinggemma-2` | Any Sentence Transformers embedding model, at a revision after `@`; run `sift embed --reset` after a change |
| `SIFT_RERANK_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any Sentence Transformers cross-encoder, at a revision after `@` |
| `SIFT_EMBED_REMOTE` | none | A `sift worker` to embed and rerank on: the `https://<key>@host:8766` it prints |
| `SIFT_OCR` | `auto` | Where scans are read: `auto` on a GPU, `on` also on the CPU (slow), `off` never |
| `SIFT_UPDATE_CHECK` | `on` | `off`: never ask PyPI for a newer sift (nor with `HF_HUB_OFFLINE=1`) |

For business documents, `Snowflake/snowflake-arctic-embed-l-v2.0` ties the default embedding model on average: better
on financial questions and long German documents, worse on scientific claims, 1.6 times as fast.

## Performance

Over 45.7 million passages of paper abstracts, warm, through `sift serve` on a desktop workstation with a GPU:

| Use case | Mode | Time |
|---|---|---|
| A name, an id, a term | `search` | 10 to 50 ms |
| A question, by its keywords | `search` | 0.2 s |
| By meaning | `vsearch` | 0.6 s (0.1 s over 3.2 million) |
| A question for RAG, reranked on the GPU | `query` | 0.6 s (0.3 s over 3.2 million) |
| Documents like one | `similar` | 0.7 s |

`query` scores nDCG@10 0.81 on scientific claims (SciFact), 0.69 on long German documents and 0.99 on German
questions, and ranks a known paper among 3.2 million at MRR@10 0.93. `sift embed` embeds 370 to 520 passages a second.
Every figure and how it was measured: [benchmarks](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md).

## Updating

In a terminal, sift says when a newer version is out and updates itself if you answer `y`, once no sift of the
installation runs any more; while the installer runs, a command started says so and stops. It asks PyPI at most once
a day, in the background; `SIFT_UPDATE_CHECK=off` stops that. Where it can't update itself (an environment without
pip, as uv's are, or an installation you may not write), it says the command instead. By hand, stop `sift serve` and
`sift worker` first (on Windows a running sift holds the files an update replaces):

```bash
uv tool upgrade sift-rag     # with pip: pip install -U sift-rag
```

Then start them again, and new agent sessions, which read sift's tools as they start.

sift follows [semantic versioning](https://semver.org/): within a major version, the commands and their JSON output,
the MCP tools, the settings, the index and the worker protocol stay compatible (sift's Python modules may change in any
release). When a release changes the index's layout, its first command updates it once, without reading files or
embedding again; older sifts then can't open it, so upgrade every sift sharing it. The
[changelog](https://github.com/tobiaswuerth/sift/blob/main/CHANGELOG.md) says what each update takes; from 1.7 or
older, read 1.8.0's entry first.

## Documentation

- [Usage](https://github.com/tobiaswuerth/sift/blob/main/docs/usage.md): collections, search options and syntax,
  filters, the formats sift reads, linked notes, the server, the UI, `sift bench`, where indexes and logs lie
- [Setting sift up with an agent](https://github.com/tobiaswuerth/sift/blob/main/docs/agents.md): the steps for an
  agent that installs sift and connects it for you
- [Another machine's GPU](https://github.com/tobiaswuerth/sift/blob/main/docs/worker.md): `sift worker`
- [How sift works](https://github.com/tobiaswuerth/sift/blob/main/docs/architecture.md): indexing, storage,
  embedding and search, and why
- [Benchmarks](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md): reference timings and the
  measurements behind the defaults
- [Contributing](https://github.com/tobiaswuerth/sift/blob/main/docs/CONTRIBUTING.md),
  [security](https://github.com/tobiaswuerth/sift/blob/main/docs/SECURITY.md),
  [changelog](https://github.com/tobiaswuerth/sift/blob/main/CHANGELOG.md),
  [license (MIT)](https://github.com/tobiaswuerth/sift/blob/main/LICENSE)
