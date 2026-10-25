# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.6.0] - 2026-10-25

### Fixed

- `sift index` keeps the documents of a folder it can't list for the moment (access denied, a network share that
  doesn't answer) as indexed, and says so for each such folder; it took their files for deleted and dropped them with
  their vectors, which then had to be read and embedded again. A folder that is gone is still dropped. So too a
  file it can't stat, and a folder whose listing breaks off midway; `reindex` calls such a file `unreadable`, not
  `removed`.
- Two commands started in the same second, in two processes, each keep a log of their own: both could take the same
  free name and write into one file.

## [1.5.0] - 2026-10-24

### Added

- A document's absolute path is an id too: `get` (with `fields` too), `similar`, `links` and `mentions` take the path
  of a file in a collection's folder, in any case where it names one file, as an agent that wrote the file or was
  given its path has it.

### Fixed

- `sift index` and `sift embed` crashed now and then on Python 3.15 (seen on macOS) while a progress line was drawn:
  the garbage collector ran on the thread that redraws it and freed PyTorch's tensors there, which PyTorch doesn't
  survive. While a progress line is drawn, the collector now runs on the working thread, at each update.

## [1.4.1] - 2026-10-24

### Fixed

- A long step's progress goes into the command's log once a minute: a first `sift index` of a large folder, scans
  read with OCR, said nothing in its log between loading the model and its end, half an hour or more later.

## [1.4.0] - 2026-10-23

### Added

- A log per run: `sift add`, `rm`, `index`, `embed`, `serve` and `worker` each write what they say, with the time of
  each line, to `SIFT_HOME/logs/<command>-<time>.log`: a progress line once, where it ended; the server's requests
  and the records of the libraries it runs; and how the command ended if it failed. Every log is kept.

## [1.3.2] - 2026-10-23

### Fixed

- On Windows, `sift serve` (and `sift worker`) could stop answering for good while it kept running: when one
  client went away in the middle of being accepted (`WinError 64`), Windows' default event loop closed the
  listening socket. It happened with 20 agents connecting at once. Both now serve HTTP on Python's selector loop,
  which only logs such a failed accept; 6,400 connections reset mid-accept leave it answering.

## [1.3.1] - 2026-10-23

### Fixed

- The server's `reindex` skipped every PDF, office file, e-book and web page it was given, and every one attached to
  a mail, with "asyncio.run() cannot be called from a running event loop": the converter runs in a coroutine of its
  own, and the server's tools run on its event loop. They are read now; the files it skipped, and mails with one
  attached, are read again by the next `sift index`, once.
- Without a GPU, the models computed in the type their checkpoint is saved in, bfloat16 for embeddinggemma-2, which
  is slow on CPUs, and Sentence Transformers warned about it; they load in float32 there, as intended.
- With huggingface_hub 2, which uses httpx2, `sift serve` logged every request to Hugging Face again.

## [1.3.0] - 2026-10-23

Scans are read, and two things real collections met are mended: mails whose HTML the converter failed on, and a
collection added inside another.

### Added

- Scans: PDFs without text of their own are read with OCR, by LightOnOCR-2 (one billion parameters, Apache-2.0), on
  the GPU the other models use, in about 3 GB: each page as Markdown, up to 50 pages a scan, once per scan new or
  changed. A mail's attached scan too. Their metadata says `ocr`. On a machine without a GPU they are skipped and
  reported, as before; `SIFT_OCR=on` reads them on the CPU, `SIFT_OCR=off` never. The first scan downloads the model
  (2 GB). Scans an older sift skipped, and mails with a PDF attached, are read again by the next `sift index`, once.
  The server's `reindex` leaves scans to `sift index`, so that no search waits for one. A page whose reading falls
  into a loop, as greedy decoding does now and then on forms, is stopped there and read once more with a repetition
  penalty, which got 5 to 10 times as far on the pages that looped.

### Changed

- `sift add` refuses a folder inside a collection, or one that holds a collection, since the files of both were
  indexed twice and found twice; `sift add --nested` adds it anyway.

### Fixed

- A mail whose HTML Xberg's converter fails on ("Internal panic: begin > end") keeps its words: its HTML is read as
  plain text, and that is reported. Its text was lost before.
- A message said while a progress line is drawn, such as the GPU memory sift takes, gets a line of its own instead of
  running on after the progress line.

## [1.2.1] - 2026-10-22

### Changed

- The package's description and keywords on PyPI name what 1.1 and 1.2 added: Obsidian notes, backlinks and saved
  emails. Nothing else changed.

## [1.2.0] - 2026-10-22

For collections that other programs write to: a `.siftignore`, saved emails, the folder of each collection, and
errors that say what to do ([Leaving files out](README.md#leaving-files-out), [Emails](README.md#emails)).

### Added

- `.siftignore`: a collection's folder may name, in gitignore's syntax, the files and folders sift leaves out, as
  apps keep their own files out and users their drafts, exports or binaries. `sift index` follows its changes, and
  `reindex` refuses what it leaves out. Hidden folders, `node_modules` and `__pycache__` stay out as before. A
  `.siftignore` sift can't read, as an offline OneDrive file, skips its collection, which keeps its documents.
- Saved emails (`.eml`): one document per mail, titled by its subject, with its sender, recipients and date above
  its text and as metadata for filters: the addresses bare, the date as ISO 8601 in UTC, so mails sort by time.
  Each attachment sift reads is a document of its own, a text file in the charset its mail declares; so is a mail
  attached to a mail, as one forwarded "as attachment", with its own attachments, three levels deep.
- `sift status --json`, as the other commands take it; `status` prints JSON anyway.

### Changed

- The instructions agents read when they connect name each collection's folder, and the docs say that `status`
  holds it as `path`: agents can tell which collection a file belongs to.
- `reindex` of a file in no collection lists the collections and their folders, nearest first, and the `sift add`
  that would cover it; an agent told only "is in no collection" tried other spellings of the path. A collection
  name the index doesn't know, in `reindex` or a search, lists the names it does.
- The README states as part of the format that a front-matter value is read as JSON first, so programs that write
  front matter can write any value as JSON and rely on reading it back as written; tests hold it.

### Fixed

- The `sift add` that `sift add` suggests for an Obsidian vault quotes a path with spaces, so it can be pasted.
- A file named like an extension, such as `.md`, stopped `sift index` with `error: ''`. As `reindex` did already,
  sift takes it for a name that starts with a dot and has no extension, and leaves it out.

## [1.1.0] - 2026-10-22

Links between Markdown notes, for Obsidian vaults and any notes that link to each other
([Linked notes](README.md#linked-notes)).

### Added

- Links: sift reads the wikilinks, embeds and Markdown links of every Markdown note, in its text and its front matter,
  and resolves them when asked, within the note's collection, as Obsidian does: by path, by name, preferring the
  note's own folder, and by alias. So they follow notes as they come and go, without the linking notes being read
  again.
  - `links` (MCP tool and `sift links`): the links of a note, each with where it leads, `ambiguous` or `unresolved`;
    the notes linking to it, with the lines that do; two steps away with `depth` 2; through one front-matter field.
  - `unresolved` (MCP tool and `sift unresolved`): the link targets no note answers to yet, most linked first.
  - `mentions` (MCP tool and `sift mentions`): the documents that name a note without linking to it.
  - `links_to` and `linked_from` on `search`, `vsearch`, `query` and `similar` (`--links-to`, `--linked-from`): only
    documents linking to these notes, or linked from them. A note is named by its id or as links name it.
  - `status` counts each collection's links and how many are unresolved, and agents learn of both when they connect.
  - `sift ui`: the reader leads to a note's backlinks and links.
- `reindex` (MCP tool) and `sift index --files`: read the files given at once, or drop the documents of files that
  are gone, so that the next search finds a note an agent just wrote. Over MCP it embeds the new passages too, and the
  server takes up their vectors without loading the others again. It reads only files `sift index` would read.
- `sift add --title-from name`: Markdown and text files take their file name as title, unless their front matter
  sets one, as notes in an Obsidian vault are named. In a vault whose notes start with `## Summary`, every note was
  titled "Summary". The same command changes a collection added before: it retitles the notes from the text the
  index holds, without reading a file, and says which passages `sift embed` must embed again. `sift add` of an
  Obsidian vault suggests it.

### Changed

- The index's layout is version 4. An index of sift 1.0 updates itself on first use, once, from what it holds: no
  file is read and nothing is embedded again. 20,000 notes take about 3 seconds, 3.2 million JSON Lines records a
  moment. sift 1.0 doesn't open an index so updated: upgrade every sift that uses it.
- A filter value that is a wikilink, `{"org": "[[Acme AG]]"}`, also matches links to the same note written otherwise:
  `[[acme ag|Acme]]`, `[[02_Organizations/Acme AG]]`, or by an alias. Other values compare exactly, as before.
- A note of front matter alone, such as a contact's, has its front matter as its passage: no search found it before,
  not even by its name. An index of sift 1.0 gets these passages when it updates; `sift embed` embeds them.
- `sift ui`: a document's passage while browsing is the start of its first passage, which kept the closing `---` of
  a note of front matter alone out.

### Fixed

- Front matter: a YAML list with unquoted items, `aliases: [Hans Mueller, H. Müller]`, was one string. It is a list
  now; a single wikilink, `org: [[Acme AG]]`, stays the link it means.
- `search`: when no passage held every term, the search for any of them dropped column filters such as `title:`,
  and found the words anywhere.

## [1.0.0] - 2026-10-21

sift is stable: from now on, its commands, MCP tools, settings, index and worker protocol change incompatibly only
with a new major version. [Versioning](README.md#versioning) lists what that covers.

### Changed

- `get` of lines past the end of a document, in the CLI and over MCP, is an error that says how many lines the
  document has. It returned an empty text, which agents took for an empty passage.
- A `sift worker` and the machines that use it must run the same major version (before 1.0: the same minor one).
  sift refuses another worker and says which version runs where.
- `sift add` refuses a folder that is a collection already: registered twice, its documents were found twice.
- `sift serve` on an address other than `127.0.0.1` or `localhost` warns that anyone who reaches it can read every
  indexed document.
- `sift bench` takes grade 0 for a document read and found irrelevant, as TREC's judgements do.
- The package declares every package sift imports itself: uvicorn, Starlette, AnyIO, Pydantic and MarkupSafe came
  only through mcp and Jinja2.
- `sift ui`: the facets that most results share come first; folders take a few lines instead of one each; a long
  document shows 2,000 lines at a time; and on a narrow window the search box gets its own row, with the filters
  above the results instead of hidden.

### Fixed

- Filters, sorting and `-f` took field names of letters, digits and `_` only, so fields such as `journal-ref` or
  `date created` couldn't be used. Any name without a double quote works now.
- On an IPv6 address, `sift ui` failed with "nonnumeric port", and `sift serve` printed addresses without the
  brackets IPv6 needs in them.
- A front-matter title that reads as a number, such as `title: 1984`, was passed over: the document took its first
  heading or its file name as title instead.
- `sift bench` split a single id given as text, not in a list, into letters, and failed with a division by zero on a
  case whose grades were all 0. Both are now errors that say what a case needs.
- `sift ui`:
  - after *Similar* or another link that changes the search, the search box still showed the old query, and the
    next change of mode brought it back; sorting a list of similar documents lost the list;
  - filtering by a value such as `007` or `1.10` found nothing: the value was read as the number 7 or 1.1;
  - yes/no fields, such as `draft: true`, weren't facets;
  - a folder whose name holds `[` showed other folders' documents, and `?path=2024` without its slash showed a folder
    named "" and the documents of `2024x/` too;
  - the window title didn't follow the search;
  - the mode buttons couldn't be reached by keyboard; ArrowDown on the sort menu jumped to the results, and Ctrl+J
    and Ctrl+K were taken for `j` and `k`;
  - browsing showed a Markdown document's front matter as its passage;
  - a page that held all the results still offered *More*, and counted "30+" for 30;
  - `?k=²` failed with a server error.

### Security

- `sift ui` opened a collection's files in the UI's own origin, so a saved web page with scripts could read the
  whole index through it. Files now open sandboxed, PDFs excepted (see [SECURITY.md](SECURITY.md)).
- `sift ui` kept copies of the last pages it showed in the browser's local storage, the text of documents included;
  going back now asks the server again. htmx no longer runs scripts or evaluates code in what it swaps in.

## [0.10.0] - 2026-10-21

### Added

- `sift ui`: the index in your browser, to search, browse and read the sources yourself. It opens the UI of the
  `sift serve` on port 8765, or starts one, which then serves agents too. The page has three panes:
  - collections and folders, and the front-matter values of the results with their counts, each a filter;
  - the results of a keyword search as you type, or of a search by meaning or hybrid search, with the words found
    marked;
  - the document a result leads to, its passage marked, with its front matter, similar documents and its original
    file.

  Every view has its own address. On 3.2 million arXiv abstracts, a keyword search answers in 30 to 250 ms.
- `sift serve` serves the UI under `/ui`, and its address alone leads there. Bound to a local address, it answers
  only requests that name a local host, as MCP does.

## [0.9.1] - 2026-10-21

### Fixed

- Embedding and searching through a `sift worker` waited 21 seconds before each request of 256 texts when the
  worker's name resolved to an IPv6 address first, as Windows machine names often do. The worker listens on IPv4,
  and Windows Firewall drops the IPv6 attempt instead of refusing it. sift now connects over IPv4 when the name has
  an IPv4 address, and keeps one connection open for all requests instead of opening one per request. On one
  notebook, embedding had slowed to about 9 chunks per second.

## [0.9.0] - 2026-10-21

### Added

- `sift bench FILE`: how well each search mode finds the documents a set of questions should find, on your own
  index (nDCG@10, MRR@10, Recall@10 and time per query), from a JSON list of questions and the ids of their answers.

### Changed

- `query` ranks better, across subjects and languages. The new default reranker is gte-reranker-modernbert-base, the
  same size and speed as Ettin-150M, and each candidate's score now counts its closeness in meaning to the query as
  well. A reranker alone ranked scientific claims worse than the embedding did. Over seven benchmarks (scientific,
  medical, financial, German, literature search, arXiv known items), the mean went from 0.646 to 0.696; on long
  German documents from 0.48 to 0.69; on SciFact from 0.76 to 0.81. The arXiv known items cost a little (MRR 0.936 to
  0.929). Details in [docs/benchmarks.md](docs/benchmarks.md).
- The first `query` after upgrading downloads the new reranker (about 600 MB). Indexes and vectors stay as they are;
  nothing is re-embedded. `SIFT_RERANK_MODEL=cross-encoder/ettin-reranker-150m-v1` brings the old reranker back.
- Scores keep the reranker's scale: close matches score about 2.5 to 4, and unrelated results below 2.

### Fixed

- A reranker that reads fewer than 1024 tokens, such as MiniLM (512), failed on every long passage ("The size of
  tensor a must match the size of tensor b"). sift now keeps the model's own limit and caps longer ones at 1024.

## [0.8.0] - 2026-10-20

### Added

- Keyword search covers front-matter values, such as authors and ids, in `search` and in `query`: "Vaswani
  transformer paper" finds "Attention Is All You Need", and an arXiv id finds its paper. `meta:` limits a term to
  them.
- `query`'s reranker reads the front-matter values a query names, such as an author's name or an id, with each
  candidate's title and text. On 29 known-item queries over 3.2 million arXiv abstracts, `query` went from MRR@10
  0.69 to 0.92 and Recall@10 0.72 to 0.97, with topical queries as good as before.

### Changed

- The first command after upgrading rebuilds the keyword index of an existing index, once: 3½ minutes and 0.7 GB
  more for 3.2 million arXiv abstracts. Documents and vectors stay. Older versions of sift can't open it afterwards.

### Fixed

- No "Loading weights" progress bars from transformers in the logs of `sift serve`, `sift worker` and `sift embed`;
  on Windows consoles they came out garbled.

## [0.7.0] - 2026-10-20

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
  use; nothing needs re-indexing or re-embedding. `SIFT_RERANK_MODEL=Alibaba-NLP/gte-reranker-modernbert-base` keeps
  the old one.

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

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/tobiaswuerth/sift/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/tobiaswuerth/sift/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/tobiaswuerth/sift/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/tobiaswuerth/sift/compare/v1.3.2...v1.4.0
[1.3.2]: https://github.com/tobiaswuerth/sift/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/tobiaswuerth/sift/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/tobiaswuerth/sift/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/tobiaswuerth/sift/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/tobiaswuerth/sift/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/tobiaswuerth/sift/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/tobiaswuerth/sift/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/tobiaswuerth/sift/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/tobiaswuerth/sift/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/tobiaswuerth/sift/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/tobiaswuerth/sift/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/tobiaswuerth/sift/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/tobiaswuerth/sift/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/tobiaswuerth/sift/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/tobiaswuerth/sift/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/tobiaswuerth/sift/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/tobiaswuerth/sift/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/tobiaswuerth/sift/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tobiaswuerth/sift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tobiaswuerth/sift/releases/tag/v0.1.0
