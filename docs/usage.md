# Using sift

The details behind the [README](https://github.com/tobiaswuerth/sift#readme).

- [Collections](#collections): adding, indexing, leaving files out, where sift keeps its files
- [Searching](#searching): modes and scores, options, keyword syntax, reading documents
- [Front matter and filters](#front-matter-and-filters)
- [What sift reads](#what-sift-reads): formats, JSON Lines, documents and scans, emails
- [Linked notes](#linked-notes)
- [Output and errors](#output-and-errors): UTF-8, JSON, problem codes, exit codes
- [The home process](#the-home-process): `sift serve`, agents, jobs
- [The browser UI](#the-browser-ui)
- [Measuring search on your own documents](#measuring-search-on-your-own-documents)
- [Coming from sift 1.x](#coming-from-sift-1x)

## Collections

A collection is a folder sift indexes: `sift add FOLDER [--name NAME]` adds it, `sift index` reads its files into the
collection's store and `sift embed` embeds their passages. Both are incremental: they process only new and changed
files, remove deleted ones, and an interrupted `sift embed` continues where it stopped. Files of a folder that can't be
read for the moment (access denied, a share that doesn't answer), or of a collection whose folder is missing, stay as
they were stored.

A folder inside a collection, or holding one, isn't added, as its files would be indexed twice; `--nested` adds it
anyway. `sift add` given a folder that is a collection already changes that collection's choices: `--title-from`,
`--follow-links` and `--agents`.

**Active and inactive.** `sift deactivate NAME ...` leaves collections out of searches, and of `sift index` and
`sift embed` unless named. Their stores stay, `reindex` still updates them, and `sift activate NAME ...` takes them
back. `sift add FOLDER --no-agents` keeps a collection from agents alone: the MCP tools neither list nor search it,
while your own commands and the UI do.

**Status.** `sift status` shows a table of the collections: folder, documents, passages, how many are embedded and by
which model, the size of each store, and the links of collections holding any. A store that can't be read (a drive not
connected, damaged, or written by a newer sift) is said beside it, `sift status` then ends with exit code 1, and
searches pass it over; sift never makes such a store anew. `--unreadable` lists the files that could not be read, and
why. With `--json` each collection gives `name`, `folder`, `store`, `active`, `agents` and its `contents`:
`documents`, `chunks`, `embedded`, `model`, `stored_bytes`, `links`, `unresolved`, `fields`, `unreadable` and
`unreadable_files`, or the reason it is `unavailable`.

**Files gone.** `sift index` drops the documents of files that are gone. But a folder that is missing, or holds none
of its files any more, keeps what is stored, and `sift index` says so: a drive not mounted, or a folder being filled
anew, is no folder emptied. If its files are gone for good, `sift rm NAME` forgets the collection.

### Leaving files out

sift reads the files it has a reader for, except in folders whose name starts with a dot (`.obsidian/`, `.git/`), in
`node_modules` and `__pycache__`, in folders Windows marks both hidden and system, and in linked folders (symbolic
links, junctions). A linked file is read as the file it points to if that lies in the collection's folder too; one that
links to outside it is left out, and said, as a link may lead to any file of the machine: `sift add FOLDER
--follow-links` reads those too, for a collection added before as well, and `--no-follow-links` leaves them out again;
the next `sift index` follows either. A `.siftignore` in the collection's folder leaves out more, in gitignore's
syntax:

```gitignore
drafts/
/exports
*.tmp.md
**/private/**
!private/README.md
```

- `*`, `?` and `[abc]` match within a name, `**` across folders. A pattern with a slash at its start or middle is
  taken from the collection's folder; one without matches at any level; a trailing slash matches folders only.
- `!` takes back what an earlier pattern left out, but never a file in a folder left out.
- Folders are separated by `/`, on Windows too; `\` escapes the next character (`\#notes.md`).
- Case counts as on the file system: on Windows and macOS not at all.

Only the `.siftignore` in the collection's folder counts. If it can't be read, `sift index` skips the collection and
says so. The next `sift index` follows a changed `.siftignore` both ways; `reindex` refuses files it leaves out.

### Where sift keeps its files

Everything sift keeps lies under `SIFT_HOME` (`~/.sift`), but stores placed elsewhere:

| Path | What | Lifecycle |
|---|---|---|
| `catalog.sqlite`, `catalog.sqlite-wal`, `catalog.sqlite-shm` | The catalog: the collections, their folders, choices and stores | Made by the first `sift add`; the `-wal` and `-shm` files are SQLite's, beside it while a sift has it open |
| `collections/NAME.sqlite`, `NAME.sqlite-wal`, `NAME.sqlite-shm` | A collection's store: documents, passages, the keyword index, links, vectors | Made by `sift add`, moved by `sift move`, deleted by `sift rm` (one open elsewhere then, by the next `sift index`) |
| `collections/NAME.sqlite.writer`, `NAME.sqlite.writer-journal` | Locked while a command writes the store, so one writes at a time | Made by the first write; moved and deleted with the store |
| `NAME.sqlite.part`, `NAME.sqlite.part-journal` | A store being copied to another drive by `sift move`, at its new place | Renamed to the store once copied; deleted if the copy stops |
| `server.writer` | Locked while `sift serve` serves this home, so one home process serves it | Made by the first `sift serve`; delete freely while none runs |
| `worker/token`, `worker/cert.pem`, `worker/key.pem` | `sift worker`'s access token, certificate and private key, readable by you only | Made on its first start, each through `FILE.PID.part` for a moment; delete the folder to make a new key |
| `worker.writer` | Locked while `sift worker` runs | Made by the first `sift worker`; delete freely while none runs |
| `logs/COMMAND-DATE.log` | What `add`, `rm`, `move`, `activate`, `deactivate`, `index`, `embed`, `serve`, `worker` and `import` said, timed, with errors; the home process's requests by name and time, without their queries; `sift serve --stdio` writes its own, `stdio-DATE.log` | The latest 20 of each command kept; past 16 MB a log goes on in a new part, its last 3 kept as `.log.1` (the latest) to `.log.3`; delete freely |
| `update.json` | With `SIFT_UPDATE_CHECK=on`: when PyPI was asked, the latest version it named, or why asking failed | Rewritten at most daily, through `update.json.RANDOM` for a moment; delete freely |

The home process keeps one file outside the home, in your own folder (`%LOCALAPPDATA%\sift` on Windows,
`$XDG_STATE_HOME/sift` or `~/.local/state/sift` elsewhere), so that only you may read it wherever the home lies:

| Path | What | Lifecycle |
|---|---|---|
| `servers/KEY.json` | The port, process, token and version of the home process that serves a home (KEY from the home's path), and the models, OCR and GPUs it uses, readable by you only | Written by `sift serve` through `KEY.json.PID.part` for a moment, deleted when it stops; one a killed `sift serve` left is replaced by the next |

`sift import` writes beside each store of sift 1.x it brings in, and writes nothing else of sift 1.x:

| Path | What | Lifecycle |
|---|---|---|
| `NAME-v2.sqlite` and its files as above | The collection's sift 2 store, beside sift 1.x's `NAME.sqlite` | Made by `sift import`; then a store like any other |

A store can lie anywhere else:

```bash
sift add ~/archive --store /mnt/big/sift            # a new collection's store in another folder
sift add ~/archive --store /mnt/big/archive.sqlite  # or under a name of your choice
sift move /mnt/big/sift                             # every collection's store into another folder
sift move archive /mnt/big/archive.sqlite           # one collection's
```

A move on the same drive is instant; to another drive the store is copied, then deleted. A move waits for the searches
that read the store, and the next searches wait for it.

## Searching

| Mode | Finds | Score |
|---|---|---|
| `search` | Exact keyword matches, ranked by BM25 | 0 and up |
| `vsearch` | Passages close in meaning | cosine, -1 to 1 |
| `query` | Keyword and semantic candidates reranked by a cross-encoder: the best for RAG | the reranker's, moved by the cosine |
| `similar` | Documents like a given one, by its id | cosine, -1 to 1 |

Scores rank the results of one search alone. Each result is a document with its best passage; a paper that several
collections hold under one DOI (front-matter `doi`) is one result, the best ranked copy, listing the others under
`also`.

### Options

The commands' options; the MCP tool `search` takes the same as parameters (`k`, `collections`, `path`, `filters`, ...).

| Option | Effect |
|---|---|
| `-k N` | Number of results, at most 10,000 |
| `-c NAME`, `--collection NAME` | Only this collection; repeated, each of them |
| `--path GLOB` | Only paths within the collection that match, e.g. `"2024/*"` |
| `-f KEY=VALUE`, `--filter KEY=VALUE` | Front-matter filter, repeatable ([below](#front-matter-and-filters)) |
| `--fields url,created` | Front-matter fields to show with each hit, or `*` for all |
| `--compact` | Results without their text |
| `--all-passages` | Every passage found, several per document, and copies of one work apart |
| `--sort=-created` | Order the `k` results by a front-matter field instead of score; `-` for descending. It orders the best `k`, not every match: for the newest of all, filter on a range too (`-f created>=2026`) |
| `--links-to`, `--linked-from` | Only documents linking to a note, or linked from it ([Linked notes](#linked-notes)) |
| `--adaptive` | Only the results before the largest drop in score, and five more, up to `k` |
| `--offset N` | Leave out the first N results, to page through them: `--offset 10 -k 10` gives results 11 to 20 |
| `--budget SECONDS` | Stop after about this long, once anything is found, with the results marked `partial` |
| `--candidates N` | `query` only: how many passages each of keyword and semantic search hands the reranker (default 40) |

Over MCP, `search` takes up to 20 queries in one call, returning the results of each; the queries share the budget.

### Keyword syntax

`search` requires every word; when no passage holds them all and the query has no operators, it returns passages with
any of them, ranked by BM25, and says it loosened the query. Write words, `"phrases"`, `prefix*` (one word of letters
and digits), `AND`, `OR`, `NOT` and parentheses: `'"notice period" AND (contract OR agreement)'`. Operators are
capitals; `NOT` binds closer than `AND`, and `AND` closer than `OR`. A quote inside a phrase is written `""`. A word
under `NOT` only leaves passages out and adds nothing to a score (sift 1.x, through FTS5, added it where a passage held
it under an `OR`).

A query sift can't read is refused, saying where: a quote or a parenthesis not closed, a `)` that closes nothing,
`NEAR(...)`, which sift doesn't support, or a word that starts with `-`: write `NOT java` to leave out what holds it,
or `"-5"` in quotes to search for it. Any space parts words, also a no-break space or the wide space of East Asian
text.

The front matter is searched too, so an author's name or an id finds its documents.

### Reading a document

```bash
sift get handbook/hr/leave.md                           # the whole document
sift get handbook/hr/leave.md --lines 14-23             # only some lines, e.g. a hit's
sift get handbook/hr/leave.md --fields owner,modified   # front-matter fields, then the text
sift get handbook/hr/leave.md handbook/hr/travel.md     # several, each headed by its id
```

An id is `collection/path`, or the file's absolute path. After the collection, the end of a path is enough when only
one document's path ends with it (`handbook/leave.md`); where several do, sift names them. Over MCP, `get` takes
documents as ids or as `{"id": ..., "first_line": 14, "last_line": 23}`. One answer holds 100,000 characters of text:
a text that passes them is cut at a line and says which lines are left out and how to get them.

## Front matter and filters

YAML front matter of flat `key: value` lines becomes filterable metadata. Lists can be in brackets
(`aliases: [Hans Mueller, H. Müller]`, over several lines too) or one `- item` per line. A value is read as JSON
first, so a program can write each value as JSON and get exactly it back; anything else is read as YAML-like text.
Text over several lines goes on the indented lines below `key: |`, which keeps its lines, or `key: >`, which joins
each paragraph's.

```markdown
---
title: "Travel expenses"
departments: ["finance", "hr"]
year: 2025
---
```

```bash
sift query "per diem abroad" -f departments=finance -f year=2025
sift search policy -f departments=finance -f departments=legal   # either department
sift search policy -f departments!=legal                         # not legal
sift search invoice -f "created>=2024-06" -f "created<2025"      # a range
```

- Filters on different fields must all hold. Repeated on one field, `=` matches any of the values and `!=` excludes
  all of them; documents without the field stay in.
- A list field matches when it contains the value.
- A value matches whether the document writes it as a number or as text: `-f year=2024` finds `year: 2024` and
  `"year": "2024"`, and `true` and `false` likewise.
- Ranges take `>`, `>=`, `<`, `<=`. Dates compare at the precision given: `created<=2024` includes all of 2024; a number
  compares with the number a date starts with, and passes over text that doesn't start with one (`year: unknown`).
- Field names are as written in the front matter: `-f journal-ref=Nature`, `-f "date created>=2024"`.
- Over MCP, `filters` takes the same, as a list of `{"field": ..., "operator": ..., "values": [...]}`.

A document's title is its `title` field, else its first heading outside code, else its file name (with
`--title-from name`, the file name first; the default in an Obsidian vault, a folder holding `.obsidian/`). A note of
front matter alone has its front matter as its passage. Lines between two `---` lines are front matter only if each
could be part of it (`key: value`, `- item`, indented, a closing bracket, a comment or blank); a paragraph between two
thematic breaks stays text, and is searched.

## What sift reads

| Format | Read as |
|---|---|
| Markdown, text (`.md`, `.markdown`, `.txt`) | Passages of at most 3,000 characters on line boundaries, preferring headings, each overlapping the one before by up to 300; YAML front matter as metadata |
| JSON Lines (`.jsonl`, `.ndjson`) | One document per line ([below](#json-lines)) |
| PDF, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, HTML | Markdown, by [Xberg](https://github.com/xberg-io/xberg) |
| Scanned PDFs | Markdown, by OCR ([below](#documents-and-scans)) |
| Saved emails (`.eml`) | The message, and each attachment as a document of its own ([below](#emails)) |

Text files are read as UTF-8, or UTF-16/UTF-32 after a byte order mark; bytes that aren't UTF-8 as Windows' ANSI
(cp1252), as Notepad and Excel saved text, so that "Müller" stays "Müller". Damaged files are skipped and reported until
they change. A file whose reader crashes or comes to no end in its time (ten minutes, and more for a large file) is
skipped, said once, and read again only once it changes, or by the next version of sift. Old binary Office files
(`.doc`, `.ppt`, `.xls`) aren't read.

A JSON Lines record or an email attachment is a document named after its file and a `#`: record `T-1042` of
`2025/tickets.jsonl` is `2025/tickets.jsonl#T-1042`, attachment `contract.pdf` of `offer.eml` is
`offer.eml#contract.pdf`.

### JSON Lines

Each line is a JSON object; sift finds its parts by field name:

| Part | Field |
|---|---|
| id | `id` or `_id`, a text or a whole number, else the line number |
| title | `title`, `name`, `headline` or `subject` |
| text | the first of `text`, `content`, `body`, `abstract`, `description`, `summary`, else the longest string; a list of strings becomes paragraphs |
| metadata | every other field, as front matter |

Line `{"id": "T-1042", ...}` of `2025/tickets.jsonl` in collection `support` is `support/2025/tickets.jsonl#T-1042`.
Records without an id are keyed by line number, so a line inserted above them embeds every record after it again; give
every record an id or none. Other field names, when the collection is added:
`sift add ~/qa --id-field key --title-field question --text-fields question,answer`. Once an id field is named, a
record without it is skipped. Lines that aren't objects, have no text or repeat an id are skipped and reported.

### Documents and scans

Office documents, PDFs, e-books and web pages become Markdown: headings stay headings, slides become sections, sheets
tables, and `get` returns that Markdown. Their properties become metadata where present: `title`, `authors`,
`subject`, `keywords`, `language`, `created`, `modified`. A web page's title is its first heading, else its file name.

**Scans** (PDFs without text, as scanners and phones make them) are read by
[LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B) (1B parameters, Apache-2.0), page by page as Markdown,
tables and columns in reading order; their metadata says `ocr`.

- On a GPU, at most the first 50 pages of a scan. The first scan downloads the model.
- Without a GPU, scans are skipped and reported; `SIFT_OCR=on` reads them on the CPU (minutes a page), `off` never.
  A scan skipped because the GPU's memory was taken, or OCR ran out of it, is read again by the next `sift index`;
  so are those skipped before `SIFT_OCR` was turned on, or from `off` to `auto`.
- `reindex` reads a scan as `sift index` does, in a job of the home process, so searches go on meanwhile.

### Emails

A `.eml` is a document titled by its subject, with `From`, `To`, `Cc` and `Date` above its plain text (else its HTML as
Markdown). Its metadata: `from`, `to`, `cc` (bare addresses), `date` (ISO 8601, in UTC where the mail gives its time
zone), `message_id`, `in_reply_to`, `attachments`.

Each attachment sift can read is a document of the mail: `mail/2026/offer.eml` holds
`mail/2026/offer.eml#contract.pdf`. A forwarded mail is one too, named by its subject if it has no file name, with its
attachments below it.

```bash
sift search "price list" -f from=anna@acme.example -f "date>=2026-06"
```

## Linked notes

In an Obsidian vault, or any folder of Markdown notes that link to each other, sift keeps each note's links as written
and resolves them when asked, as Obsidian does. Obsidian names notes by file name, so a vault is titled by file name
(`--title-from name`, the default for a folder holding `.obsidian/`; for a vault added before, the same command
retitles).

**Links** are `[[Note]]`, `[[Folder/Note]]`, `[[Note|text]]`, `[[Note#Heading]]`, `[[Note#^block]]`, embeds
(`![[contract.pdf]]`) and Markdown links (`[text](../People/Hans.md)`), in the text and, as wikilinks, in front-matter
values (`org: "[[Acme AG]]"`). Links in code, to web addresses and to files sift doesn't read are left out.

**A link resolves** within its note's collection, in this order: a relative path, from the note's folder; a path, to
the file with that path, else the one file whose path ends with it; a name, to the one file of that name, or of several
the one in the note's folder; else the one note whose `aliases` hold it. Names compare in any case, without `.md`. A
link several files match is ambiguous; one none matches is unresolved, a note not written yet. Resolved when asked,
links follow notes as they come and go.

```bash
sift links vault/People/Hans.md                  # the note's links, and where each leads
sift links "Hans Müller" --direction in          # the notes linking to it, with the lines
sift links "Acme AG" --direction in --field org --depth 2   # through org only, two steps
sift unresolved -c vault                         # notes the vault lacks, most linked first
sift mentions "Hans Müller" --also emails        # documents naming the note without a link
sift search contract --links-to "Acme AG"        # only documents linking to Acme AG
sift search contract -f "org=[[Acme AG]]"        # org links to Acme AG, however written
```

A note is named by its id, or as links name it if one note answers to that. `mentions` looks for a note's file name,
title and aliases, and the values of the fields `--also` names, as phrases (ä, ö, ü, ß also as ae, oe, ue, ss; names
under three characters left out). `links` and `unresolved` return at most 50 (`-k`).

**A note just written** is found by the next search once the agent calls `reindex`: it reads those files at once, or
drops the documents of files gone, and embeds their new passages where the collection is embedded. From the command
line, `sift index --file` does the same, and `sift embed` embeds.

## Output and errors

Results go to standard output, and what sift says while it works (progress, notes, errors and what to do) to standard
error. sift writes UTF-8, whatever the console's code page; in Windows PowerShell 5.1,
`[Console]::OutputEncoding = [Text.UTF8Encoding]::new()` lets a script read it as written. Colours follow `--color`
(`auto`: in a terminal), and `NO_COLOR` or `TERM=dumb` turn them off.

With `--json` the answer is one JSON object on standard output. A refusal or a failure is one too,
`{"error": {"kind": ..., "code": ..., "message": ..., "remedy": ..., "field": ...}}`, and standard error says it as
without `--json`. Documents or collections a command could not read are listed in its answer with the same `code`,
`message` and `remedy`. A `code` is one of `invalid_request`, `ambiguous`, `taken`, `too_many_words`,
`widened_enough`, `field_not_held`, `not_found`, `folder_missing`, `unknown_collection`, `unknown_document`,
`unknown_note`, `inactive_collection`, `store_unavailable`, `store_missing`, `store_damaged`, `catalog_damaged`,
`store_newer`, `sqlite_too_old`, `keywords_remade`, `not_embedded`, `other_model`, `out_of_time`, `busy`,
`model_unavailable`, `unauthorized`, `os`, `stopped` and `internal`. A failure sift did not foresee is `internal`:
its traceback goes to standard error, and with `--json` the error is on standard output all the same.

| Exit code | Means |
|---|---|
| 0 | Done |
| 1 | Failed: a document not found, a store that can't be read, a collection skipped |
| 2 | Refused as asked: a flag or value sift doesn't take, a query it can't read |
| 130 | Stopped, as by Ctrl+C |

## The home process

`sift serve` is the home process: one process per home for you, your agents and your browser. It answers MCP
(streamable HTTP) at `http://127.0.0.1:8765/mcp`, the UI under `/ui`, and your own `sift` commands, which find it by
`servers/KEY.json` ([above](#where-sift-keeps-its-files)) and send it what they are asked. It keeps the models and the
vectors' codes loaded, runs searches side by side, and frees the GPU after 5 minutes without use. Without it, each
command loads what it needs itself.

| Option | Effect |
|---|---|
| `--port N` | Another port than 8765 on 127.0.0.1; `0` for any free one |
| `--stdio` | MCP over standard input and output, for an agent that starts its servers itself: it talks to the home process, starting one if none runs (on Windows in a minimized window), which leaves after 15 minutes without a request |
| `--idle-exit SECONDS` | Stop after this long with no request and no job |

- It listens on 127.0.0.1 alone and answers only requests addressed to `127.0.0.1` or `localhost`. Your commands send
  it the token it keeps in your own folder; MCP and the UI refuse pages of other sites. What it protects, and what
  not: [security](SECURITY.md).
- It serves the user who started it. It says so when its settings (the models, OCR, GPUs, a worker) differ from the
  command's; stop it to use yours.
- Writes wait for the searches reading a store, and searches wait for a removal or a move; each store takes one write
  at a time.
- An agent's `reindex` runs as a job: `sift jobs` lists the jobs that run or wait, and `sift jobs --stop 3` stops one at
  its next step, keeping what it wrote.
- What it says is kept in `SIFT_HOME/logs`, its requests by name and time, without their queries.
- To lend a GPU to other machines, without the stores: [worker](worker.md).

## The browser UI

`sift ui` opens the UI of the home process in your browser, at `http://127.0.0.1:8765/ui` for the default port; if no
`sift serve` runs, it serves the home itself on a free port until you stop it. `--no-browser` prints the address alone.

![sift ui in dark mode: a keyword search over the ACL Anthology, filtered by venue, with the publishers, months and places of the papers found to drill into](https://raw.githubusercontent.com/tobiaswuerth/sift/main/docs/images/ui-browse-dark.png)

- **Search** by keywords, meaning or both; found words are marked.
- **Drill down** through collections, folders and the results' front-matter values with their counts; each a click to
  filter by. Sort by any front-matter field.
- **Read** a result beside the list, its passage marked, with its front matter as filters, its similar documents,
  backlinks and links, and the original file (PDFs in the browser's viewer; other files open without running their
  scripts).
- Every view has its own address, and the pages run no scripts.

## Measuring search on your own documents

`sift bench` runs each search mode over questions whose answers you know and reports nDCG@10, MRR@10, Recall@10 and the
time per query. Write the questions as an agent would ask them, with the ids of the documents that answer them:

```json
[
  {"query": "Which supplier terms changed in 2025?", "relevant": ["work/contracts/supplier-2025.docx"]},
  {"query": "onboarding checklist for new hires", "relevant": {"work/hr/onboarding.pdf": 2, "work/hr/it-setup.md": 1}}
]
```

```bash
sift bench questions.json            # -c NAME: one collection only
```

A few dozen questions are enough to compare modes, or to see whether a change helps your documents. Grades are
optional, from 0 to 10: 2 for the answer, 1 for a useful one, 0 for one read and found irrelevant. A copy of the same
paper listed under `also` counts as found.

## Coming from sift 1.x

sift 2 keeps its collections in stores of a new layout and opens none of sift 1.x. A command in a home that holds sift
1.x's catalog and none of sift 2 says so and names the fix:

```bash
sift import                          # every collection of this home's sift 1.x
sift import --home ~/old-sift -c notes   # one collection of another home
```

`sift import` adds each collection with its choices (where titles come from, following links, the fields of JSON
Lines, whether it is active), indexes its folder afresh, and gives each passage whose text is unchanged the vector sift
1.x made, so nothing is embedded twice; passages changed since are left for `sift embed`. Vectors of a model sift 1.x
named without a whole commit stay behind, said so. The table it prints sets the counts of documents, passages and
vectors before and after side by side.

Nothing of sift 1.x is written: each new store goes beside the one it comes from, as `NAME-v2.sqlite`; a file already
there skips that collection. Run again, an import goes on with what it added. Once the collections are as you want
them, remove sift 1.x's stores and its `sift.db`. A store older than sift 1.8 is refused: open it once with sift 1.15,
then import it.
