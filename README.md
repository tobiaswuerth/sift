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
It offers four kinds of search: exact keyword matching, semantic search, hybrid retrieval with reranking for RAG,
and documents like a given one.
Everything runs locally. No cloud services, no API keys.

## Install

Requires Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --torch-backend auto sift-rag   # installs the `sift` command
```

Or run it without installing: `uvx --torch-backend auto sift-rag serve`. With pip, in a virtual environment:
`pip install sift-rag`.

To upgrade later, see [Updating](#updating).

On first use, sift downloads its two models from Hugging Face (about 2 GB).

### GPU support

sift uses an NVIDIA GPU (CUDA) if there is one, else an Apple GPU, else the CPU, which works but embeds slowly.
`query`'s reranker is slow there too: about 25 seconds per query on 8 CPU threads, against 0.3 seconds on an RTX
4090. A machine without a GPU can borrow one through a [`sift worker`](#embedding-on-another-machine), which reranks as
well as embeds, or use `search` and `vsearch`, which need no reranker.

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
sift similar papers/2401.02225.md      # documents like this one
sift get papers/2401.02225.md          # full document
sift get papers/2401.02225.md --lines 14-23   # only some lines, e.g. those of a search hit
sift get papers/2401.02225.md --fields url,created   # front-matter fields first, then the text
sift status                            # collections and counts
sift bench questions.json              # how well each mode finds what you expect (see below)
sift rm papers                         # remove a collection from the index; its files stay on disk
sift serve                             # MCP server on http://127.0.0.1:8765/mcp
sift worker                            # embed for other machines on this one's GPU (see below)
```

`sift index` and `sift embed` are incremental: run them again after files change, and only new or changed files are
processed. Deleted files are removed. An interrupted `sift embed` continues where it stopped. A running server picks up
new vectors on its own.

Search commands, `similar` included, return one result per document, with its best passage. They accept `-k` (number
of results), `-c` (collection), `--path` (glob within the collection, e.g. `"2024/*"`), `-f` (front-matter filter,
repeatable, see below), `--fields` (front-matter fields to show with each hit, e.g. `url,created`, or `*` for all),
`--compact` (no text), `--all-passages` (every matching passage, several per document), `--sort=-created` (order the
results by a front-matter field instead of score; `-` for descending) and `--json`.

`search` requires every term. If no chunk has them all and the query has no operators (`AND`, `OR`, `NOT`, `NEAR`),
it returns the chunks that have any of them, those with more terms first. Operators are written in capitals, and
parentheses group: `'"stroke" AND (painting OR rendering)'`. A query with operators is taken as it is; when it finds
nothing, drop terms.

Scores rank the results of one search, on a scale of its own: BM25 for `search` (0 and up), cosine similarity for
`vsearch` and `similar` (-1 to 1), and for `query` the reranker's score, nudged by the passage's cosine, on the
reranker's scale. It compares across queries: with the default reranker, close matches score about 2.5 to 4, and
unrelated results below 2.

### Measuring search on your own documents

`sift bench` runs each search mode over questions whose answers you know and reports nDCG@10, MRR@10 and Recall@10,
the measures of the BEIR and TREC benchmarks, and the time per query. Write the questions as an agent would ask
them, with the ids of the documents that answer them, as search results show them:

```json
[
  {"query": "Which supplier terms changed in 2025?", "relevant": ["work/contracts/supplier-2025.docx"]},
  {"query": "onboarding checklist for new hires", "relevant": {"work/hr/onboarding.pdf": 2, "work/hr/it-setup.md": 1}}
]
```

A few dozen questions are enough to compare modes and to see whether a change helps your documents. Grades (2 for
the answer, 1 for a useful one) are optional.

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
sift search robot -f categories=cs.RO -f categories=cs.CV   # in either category
sift search robot -f categories!=cs.CV                      # not in cs.CV
```

Filters on different fields must all hold. Repeated for one field, `=` matches any of the values, and `!=` leaves out
documents with any of them; documents without the field stay in.

Filters can also be ranges, with `>`, `>=`, `<` and `<=`. Dates compare at the precision given, so `created<=2024`
includes all of 2024 and `created>2024` starts in 2025; a number compares with the number a date starts with:

```bash
sift search pigment -f "created>=2024-06" -f "created<2025"
```

The `title` field, or else the first heading outside code blocks, or else the file name, becomes the document title.

Keyword search covers the front matter too, so an author's name or an id finds its documents, in `search` and in
`query`. `meta:` limits a term to the front matter, e.g. `sift search "meta: Vaswani"`.

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

`sift serve` runs an MCP server that agents search through:

| Tool | Purpose |
|---|---|
| `search` | Exact keyword search, ranked by BM25. Supports `"phrases"`, `prefix*`, `AND`, `OR`, `NOT` and parentheses. |
| `vsearch` | Semantic search. |
| `query` | Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder. |
| `similar` | Documents like a given one, by its id: more papers like one found. |
| `get` | Text of an indexed document by its id: all of it, or only `lines` such as a hit's `"17-26"`. With `fields`, it returns the document with those front-matter fields. An id may leave out folders if the file name is unique. |
| `status` | Collections, counts and front-matter fields. |

The search tools take `query` (`similar`: `id`), `k`, `collection`, `path`, `filters`, `fields`, `compact`,
`all_passages` and `sort`, as the commands above do. Every result carries `id` (`collection/path`), `title`, `lines`,
`score` and the passage's `text`, so agents can cite their sources. `fields` adds front-matter fields to each result,
e.g. `["url", "created"]`, and `compact: true` leaves out the text, so an agent can survey many results without
filling its context and then read the ones that matter. Filters take a value, a list for any of its values,
`{"not": ...}`, or ranges, e.g. `{"created": {">=": "2024-06", "<": "2025"}}`. Every parameter is described in the
tools' schemas, and the server tells agents when they connect which collections there are and which front-matter
fields each has.

### Setting up

1. Start the server, and leave it running:

   ```bash
   sift serve        # prints: sift MCP server on http://127.0.0.1:8765/mcp
   ```

2. Register it with Claude Code, once, for all your projects:

   ```bash
   claude mcp add --scope user --transport http sift http://127.0.0.1:8765/mcp
   ```

   Keep `--scope user`. Without it, `claude mcp add` registers sift for the current folder only, and sessions started
   anywhere else don't see it, including those in the Claude desktop app's Code tab.

3. Start a new Claude Code session: running sessions don't pick up new servers. `claude mcp list` should now show
   sift as connected, and an agent asked for sift's status gets your collections.

One server serves any number of agents, with the models loaded once. Other MCP clients connect the same way, to the
streamable HTTP endpoint `http://127.0.0.1:8765/mcp`.

Or let every session start a server of its own over stdio; then each session loads the models into memory itself:

```bash
claude mcp add --scope user sift -- sift serve --stdio
```

A registration made without `--scope user` can be removed in the folder where it was made, with
`claude mcp remove sift --scope local`.

Connectors in claude.ai and in the Chat tab of the Claude apps can't use sift: claude.ai reaches connectors from
Anthropic's servers, which can't reach a server on your computer.

## Updating

A running sift keeps the version it started with, so an update takes an upgrade and a restart of each part:

1. **Upgrade** on every machine that runs sift; a machine that embeds through a `sift worker` needs the same version
   as the worker:

   ```bash
   uv tool upgrade sift-rag     # with pip: pip install -U sift-rag
   sift --version
   ```

2. **Restart** what runs: stop `sift serve` and `sift worker` with Ctrl+C and start them again. A server registered
   with `--stdio` starts anew with each session.

3. **Start new agent sessions.** An MCP client reads sift's tools and their parameters when a session starts, and
   may keep that list even after reconnecting: a running session then misses new tools and parameters, and its calls
   with them fail or lose arguments. In Claude Code, start a new session; the registration stays as it is.

The index and the vectors carry over. When a new version changes how the keyword index is laid out, the first
command rebuilds it, once, and says so: a few minutes for millions of documents; nothing gets embedded again. The
[changelog](CHANGELOG.md) lists what changed.
## Embedding on another machine

A machine without a GPU, such as a notebook, can borrow another machine's. Run `sift worker` on the one with the GPU:

```bash
sift worker            # listens on port 8766 and prints the address to use, with its key
```

It prints a line such as `SIFT_EMBED_REMOTE=https://<key>@workstation:8766`. Set that on the other machine, e.g. in
PowerShell (`setx` keeps it for new terminals; bash: `export SIFT_EMBED_REMOTE=...`):

```powershell
setx SIFT_EMBED_REMOTE https://<key>@workstation:8766
```

`sift embed` then sends the chunks to the worker, 256 at a time, and stores the vectors it gets back, without loading
the model itself. Searches embed their queries and rerank there too, in the CLI and in `sift serve`, so the machine
searching never loads PyTorch: `vsearch` and `query` answer in a fraction of a second, even on a light notebook.
Both machines need the same sift version. The worker only computes: it sees the texts it embeds and reranks, but no
index and no files. It sizes its batches to its GPU
and gives the GPU back after 5 idle minutes, like `sift serve`. The worker's `SIFT_EMBED_MODEL` decides the model,
and sift refuses a worker whose model differs from the one the index was embedded with.

The texts travel encrypted, over HTTPS. On its first start the worker makes a certificate of its own and an access
token, kept in `~/.sift/worker` (readable by you only), and the key it prints holds both: the token, and the
certificate's fingerprint. sift checks the worker's certificate against that fingerprint before it sends anything,
so no other machine can pose as the worker, and the worker answers only requests that carry the token. Keep the key
private, like a password; sift never shows it in its output. To make a new key, delete `~/.sift/worker` and restart
the worker; machines with the old key are then refused.

To let the worker through the Windows firewall from your own subnet only, in PowerShell as administrator:

```powershell
New-NetFirewallRule -DisplayName "sift worker" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow -RemoteAddress LocalSubnet
```

If Windows asks whether to let Python through the firewall, allow it: cancelling adds a rule that blocks Python, and
blocking rules win over allowing ones. On Linux with ufw: `sudo ufw allow from 192.168.1.0/24 to any port 8766
proto tcp`, with your network's range.

## Configuration

None needed: sift picks the device, the GPU memory budget, the batch size and the number of worker processes itself.
Four optional settings exist for special cases:

| Variable | Default | Purpose |
|---|---|---|
| `SIFT_HOME` | `~/.sift` | Where the index lives |
| `SIFT_EMBED_MODEL` | `google/embeddinggemma-2` | Any Sentence Transformers embedding model |
| `SIFT_RERANK_MODEL` | `Alibaba-NLP/gte-reranker-modernbert-base` | Any Sentence Transformers cross-encoder |
| `SIFT_EMBED_REMOTE` | none | A `sift worker` to embed on: the `https://<key>@workstation:8766` it prints |

After changing the embedding model, run `sift embed --reset`. Models are downloaded from Hugging Face on first use.

For a new index of business documents, `Snowflake/snowflake-arctic-embed-l-v2.0` is a good alternative embedding
model. On our [benchmarks](docs/benchmarks.md) it ties EmbeddingGemma-2 on average. It's better on financial questions
and long German documents, worse on scientific claims, and it embeds 1.6 times as fast.

## How it works

- **Indexing:** worker processes read and parse files in parallel while the previous batch is written; small
  updates skip the process start-up. Files are compared by modification time and size, so unchanged files cost
  nothing, and documents by content hash, so a changed JSON Lines file re-indexes only its changed records. Text
  files are read as UTF-8, or as UTF-16 when they start with its byte order mark (Windows PowerShell 5.1 writes
  those). Each kind of file has a reader, chosen by extension, that turns it into documents; adding a format means
  adding a reader. Markdown, including documents converted to it, is split into chunks of about 3,000 characters
  (about 750 tokens) on line boundaries, preferring headings, with a small overlap.
- **Remote embedding:** HTTPS with the worker's self-made certificate (EC P-256), pinned by its SHA-256 in the key,
  and a bearer token of 32 random bytes. `GET /` names the worker's models and vector length; `POST /embed` takes
  `{"texts": [...], "query": false}` as JSON and returns the vectors as raw fp16, one row per text; `POST /rerank`
  takes `{"query": "...", "texts": [...]}` and returns the raw scores, `{"scores": [...]}`. The worker runs one
  request at a time on the GPU. Through a worker, the vectors are searched with NumPy, in RAM.
- **Storage:** one SQLite database in WAL mode holds every document's text, metadata, the FTS5 keyword index and
  the vectors (fp16). The text is stored once: chunks are spans of it, and the keyword index reads it through a view.
  Searching and `get` never touch the files, so results and documents always agree, until the next `sift index`.
- **Embedding:** PyTorch on the GPU (NVIDIA, else Apple), in batches sorted by length to keep padding low. The
  models compute in bfloat16 on NVIDIA GPUs that have it and in float32 elsewhere, never in fp16, where EmbeddingGemma
  overflows now and then into broken (NaN) vectors. Vectors are stored in fp16, and broken ones are never stored.
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
- **`search`:** SQLite FTS5 with BM25, over three columns weighted alike: every passage is indexed with its
  document's title (the first passage holds it as its heading too) and its front-matter values. Matches are grouped
  by document in SQL, each represented by its best passage. Input that isn't valid FTS5 syntax is treated as exact
  phrases.
- **`vsearch`:** exact nearest-neighbour search as one matrix multiply over all vectors on the GPU. No approximate
  index, so nothing to tune and no recall loss; a million 768-dimensional vectors take about 1.5 GB of VRAM. When the
  best passages come from a few documents, it looks further down until it has `k` documents.
- **`query`:** the top 40 (or `k`, if larger) keyword and semantic candidates are reranked together, in one GPU
  batch. The reranker reads each candidate's title, the front-matter values that share a word with the query (an
  author's name, an id; all of them would dilute the passage), and the passage.
  - **Scoring.** A candidate's score is the reranker's, plus half its cosine with the query, measured in units of the
    reranker's spread over the candidates. One weight thus fits any reranker, and the scores keep the reranker's
    scale. A reranker alone ranked scientific claims worse than the embedding did. Candidates whose front matter the
    query names keep the best cosine, since the embedding never sees front matter.
  - **The evidence** is in [docs/benchmarks.md](docs/benchmarks.md): rerankers, embedding models and blends on
    seven datasets, from scientific claims to long German documents. Scores are raw:
    a sigmoid saturates, at 0.9999 for every good match.
- **`similar`:** the mean of a document's passage vectors, scored against every passage as `vsearch` scores a query.
  No model runs, so it takes half a second over 3 million passages, with or without a GPU.
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
