# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

sift 2 is rebuilt from the ground: stores of a new layout with a keyword index of its own, and a home process that
answers agents, the browser and your own commands. It opens no store of sift 1.x. After updating, run `sift import`
once in each home (any command there says so): it brings the collections of sift 1.x in, with their vectors, and
leaves sift 1.x's files as they were. Agents need a new session, as the MCP tools changed; scripts that read `--json`
need its new keys; ids of JSON Lines records and mail attachments are written anew (below).

### Changed

- sift needs Python 3.12 or newer, and SQLite 3.38 or newer (every Python build brings one).
- Each collection's store has a new layout; the catalog is `catalog.sqlite` in place of `sift.db`. A store's layout
  changes from now on by numbered migrations, which the first command of a newer sift runs once; an older sift
  refuses a store a newer one migrated, and says to upgrade.
- A JSON Lines record and a mail attachment are named after their file and a `#`: `papers/2024.jsonl#2401.00001` and
  `mail/offer.eml#contract.pdf`, where sift 1.x wrote `papers/2024/2401.00001` and `mail/offer/contract.pdf`.
- Keyword search reads an index of sift's own, written in the documents' transaction, in place of SQLite's FTS5 and
  the postings read out of it: nothing is behind after `sift index`, and no search falls back to FTS5. Its terms are
  FTS5's, and it ranks as sift 1.15 ranked, by the same BM25. A query sift can't read is refused, saying where (a
  quote or parenthesis not closed, a `)` closing nothing), where sift 1.x searched it as phrases.
- `sift serve` is the home process: one per home, on 127.0.0.1, for agents (MCP), the browser (the UI) and your own
  `sift` commands, which it answers with its models and codes loaded. `sift serve --stdio` talks to it, starting one
  if none runs (on Windows in a minimized window), which leaves after 15 idle minutes; `sift ui` opens its pages, or
  serves the home itself on a free port.
- The MCP tools are five: `search` (hybrid, keyword or semantic, up to 20 queries a call, paged by `offset`), `get`,
  `related` (documents alike, a note's links, its backlinks, its mentions), `collections` (with the jobs that run),
  and `reindex`, which runs as a job of the home process.
- `--json` answers are objects: a search gives `hits` with what was searched, left out, loosened or cut short beside
  them, and each hit its `collection`, `path`, `first_line` and `last_line` (in place of `lines`). `sift status`
  gives `home` (in place of `index`), and for each collection `folder` (in place of `path`), `agents`, and its counts
  under `contents`: `documents` (in place of `docs`), `chunks`, `embedded`, `model`, `stored_bytes` (in place of
  `size`), `unreadable`.
- `sift status` prints a table; `--json` gives JSON.
- `sift index --files A B` is `sift index --file A --file B`; `sift add --id`, `--title` and `--text` are `--id-field`,
  `--title-field` and `--text-fields`.
- A model other than the defaults is named at a whole commit (`owner/model@<40 hex digits>`) or as a folder: a
  branch, a tag or a short commit is refused, as each may name other weights tomorrow.
- `SIFT_UPDATE_CHECK` is off by default. Turned on, a command run in a terminal says when a newer sift is out and which
  command updates it; sift never updates itself.
- `sift worker` takes at most 256 texts a request; both machines need sift 2.
- `sift embed` reads 1,024 passages, embeds them and writes them in turn, where sift 1.15 read and wrote on threads
  of their own: on a GPU shared with another job it embedded 15% fewer passages a second.

### Removed

- `sift serve --host` and `--allow-host`, and `sift ui --host`, `--port` and `--allow-host`: the home process
  answers this machine alone. To lend a GPU to other machines, `sift worker` remains.
- The keyword syntax `NEAR(...)`, column filters (`meta:`, `title:`) and `^`.
- Updating itself, with `update.lock` and the update logs.
- Numba: the codes are scanned with NumPy.

### Added

- `sift import`: the collections of a sift 1.x home, with their choices and vectors, into this one.
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

## [1.15.2] - 2026-11-22

The lowest cryptography sift installs with is 50. Nothing else changes, and nothing needs doing after the update.

### Security

- The lowest version of cryptography sift named, 49, has an advisory known against it now (PYSEC-2026-3552, in how it
  decrypts PKCS#7 messages). sift makes the certificate of `sift worker` with cryptography and decrypts no such
  message, but the lowest is now 50, where it is fixed. cryptography 50 is older than sift's first release: an
  installation holds it or a newer one already, unless another package kept it back.

## [1.15.1] - 2026-11-22

The fixes of a second review of all of sift. No index changes its layout, and nothing needs doing after the update.

### Security

- The UI's "Open file" opened whatever lay at an indexed file's path by now: a link put there after indexing, as by a
  folder synced or checked out again, led it to any file of the machine, though `sift index` reads no such link. A
  file is now opened only if `sift index` would still read it: no link to outside its collection's folder (unless
  the collection follows links), and through no folder linked from elsewhere.
- The log of `sift serve` kept why each MCP tool failed, and with it the document a `get` or `mentions` was asked for
  ("unknown document '...'"). It now keeps that the tool failed; the caller still gets the reason.

### Fixed

- `sift index` ended in an OverflowError, on every run, over a file dated before 1678 or after 2262, as Windows
  dates a file without a time (1601): the files changed with it stayed unread. Such a time is now kept as the nearest
  the index can hold.
- On Windows with 63 or more CPU threads, `sift index`, a large keyword build and the upgrade of an old index ended
  in "max_workers must be <= 61". A pool now has as many processes as Windows allows.
- Two `sift embed` at once, one of them with `--reset` and another model, or a server's `reindex` beside a reset,
  could leave one collection with vectors of both models under the name of one. Each write now checks the
  collection's model as it is then, and the later one stops, or leaves the note unembedded and says so.
- A file whose reader crashed or hung was read again by every `sift index`: ten minutes and more each time for one
  that hangs. It is now skipped until it changes, said once, and the next version of sift reads it once more.
- A keyword build in worker processes that failed to write a part ran every other part before it said so, for
  minutes to hours on a large collection; so did the upgrade of an old index. What hasn't started is no longer run.
- Front matter over several lines, YAML's `key: |` and `key: >`, came out as "|" or ">", and a list in brackets
  written over several lines as "[": no search found their text. Each is now the field's value.
- A title with a line break in it, as front matter, a JSON Lines record, a mail's subject or a document's
  properties may hold one, broke the line each result is printed on. A title is one line now; the field stays as
  written.
- Code in a block quote or a list item counted as prose: Bash's `[[ -f x ]]` in it became a link to a note "-f x",
  which `unresolved` listed, and a `#` line in it could become the note's title.
- `sift serve` and `sift worker` on a port already taken loaded the models first, then ended with the web server's
  own error. They now say "port N is taken" before anything loads, and refuse a port past 65535, which Windows took
  for another.
- After a restart of the machine during an update, another process could have the update's process id: every sift
  command, a server started with the machine too, stopped for up to three hours as if sift were being upgraded, and
  an update that had only waited was neither made nor offered again for eight days. The update's process now holds
  a lock (`SIFT_HOME/update.lock`) that tells whether it runs.
- A search with a NUL character in it found its passages or failed with "unterminated string", by whether the
  keyword postings were ready. It now parts words either way.
- A mail whose text names a charset that is none ("undefined", "idna", "punycode", "unicode_escape") was skipped
  whole, or read as other characters than it holds.
- A worker's answer that fits no request, scores for another number of passages or vectors of another length, as a
  proxy or another sift may send, ranked results by other passages' scores or ended in numpy's error. It is now an
  error that says so.
- `.siftignore`: a range written backwards, `[z-a]`, matched nothing, where git matches its first character.

### Changed

- The guide for an agent that sets sift up is now `docs/agent-setup.md`: under its old name, `agents.md`, coding
  agents took it for the instructions of the repository they worked in. The old address says where it went. The guide
  now also says how to get uv, what to do when the shell finds no `sift`, and how to start `sift serve` so that it
  outlives the agent's shell.

## [1.15.0] - 2026-11-21

A review of all of sift for what could go wrong, and its fixes: files, models and requests from elsewhere are trusted
less, text that isn't UTF-8 is read as it was written, and an agent can set sift up by a guide of its own. No index
changes its layout: an older sift still uses a collection this one indexed and embedded.

What updating takes: the first `sift index` looks once through each collection for text an older sift stored with
replacement characters, minutes for tens of millions of documents, and reads the files concerned again. A file that
links to outside its collection's folder is no longer read unless told (`--follow-links`), and a server on an address
other machines reach answers only to this machine's addresses and names (`--allow-host`). The oldest PyTorch sift
installs with is 2.13.

### Added

- [Setting sift up with an agent](https://github.com/tobiaswuerth/sift/blob/main/docs/agents.md): the steps for an
  agent that installs sift, indexes the folders you name and connects itself, with what to ask first and what to
  leave alone.
- `sift add FOLDER --follow-links` and `--no-follow-links`, `--allow-host NAME` for `sift serve` and `sift ui`, and
  a revision after an `@` in `SIFT_EMBED_MODEL` and `SIFT_RERANK_MODEL`; each is described with the change it came
  with, below.
- usage.md describes the server and its options.

### Security

- The log of `sift serve` kept every request with its query, and with it every search made in the UI and every
  document read there ("?q=...", "?doc=..."), on disk in SIFT_HOME/logs, up to 64 MB a log. Requests are logged
  without their queries now. Logs written before keep theirs: delete them if they matter.
- sift's lowest versions of its dependencies had advisories known against them, 142 in nine packages: they are now
  PyTorch 2.13 (torchvision 0.28), Pillow 13, Starlette 1.3.1, cryptography 49, mcp 2.2, AnyIO 4.14.2 and Jinja2
  3.1.6. The lowest versions sift named didn't even work together: transformers 5.18 disables a PyTorch older than
  2.5, and takes no tokenizers older than 0.23.1, now the lowest named. The CUDA index that the fix sift prints, the
  README and requirements.txt name is now cu126, as cu128 carries no PyTorch past 2.11.
- An update agreed to with pip ran `python -m pip` at the root of the drive, which Python searches for modules first:
  on Windows, where any user may make a folder there, another user's `pip` folder would have run as you. It runs with
  `-P` now, which leaves that folder out.
- A file that is a link was read as the file it points to, wherever that lay: a folder from elsewhere, such as a
  cloned repository, could hand any file of the machine, a private key named notes.txt, to the index, and to the
  agents that read it. A file linking to outside its collection's folder is now left out, and said, by `sift index`
  and the server's `reindex`; one indexed before is dropped by the next `sift index`. `sift add FOLDER
  --follow-links` reads them, as before, for a new collection or one added already; `--no-follow-links` leaves them
  out again.
- Any web page could have the browser ask the UI of `sift serve` for searches, each holding the server's one call
  at a time and the GPU, or show the UI in a frame of its own. The UI refuses what another site's page asks for in
  the background, as browsers say by Fetch Metadata, and forbids frames; a link to it from elsewhere still opens it.
- `sift serve --host 0.0.0.0` (or another address) answered requests naming any host: a page of another site whose
  name led to this machine's address (DNS rebinding) read every document, from any browser on the network. It now
  answers only requests naming this machine by an address, by its own name, or by a name given with the new
  `--allow-host NAME` of `sift serve` and `sift ui`.
- The package published to PyPI was built with whatever setuptools, twine and uv were newest that day. The release
  now builds with setuptools pinned by hash (`.github/requirements-build.txt`, kept current by Dependabot), twine and
  uv pinned by version; CI's package job builds the same way.
- The models loaded from the latest commit of their repositories on Hugging Face: a push of their owners could change
  the vectors of new passages beside the old ones, or the code that computes them, unseen. The embedding model, the
  reranker and the OCR model now load at a commit this version of sift pins, and a collection records its model at
  that commit; a release that moves a pin says to run `sift embed --reset`. A collection embedded with the default
  model by an older sift needs no embedding again: the pinned commit holds the same weights. `SIFT_EMBED_MODEL` and
  `SIFT_RERANK_MODEL` take a revision after an `@` (`org/model@v2`). A worker of an older sift still works, and an
  older sift still embeds into a collection this one embedded.

### Fixed

- A note that opens with `---` and has another `---` line further down kept the lines between them out of every
  search when they were no front matter, such as a paragraph between two thematic breaks: `get` showed them, no search
  found them. Such lines are now the note's text; front matter is lines each of which could be part of it.
- Text that isn't UTF-8, such as a note Notepad or Excel saved as "ANSI", had its accented letters indexed as "�", so
  a search for "Müller" never found it. Bytes that aren't UTF-8 are now read as Windows' ANSI code page (cp1252), in
  text files, JSON Lines, and mail text that its mail calls us-ascii or UTF-8 and is neither, or that is in a charset
  Python doesn't know; what is UTF-8 stays as it is. The next `sift index` looks once through each collection
  for documents an older sift stored with a replacement character, and reads their files again where the bytes
  aren't UTF-8, and such mails; `sift embed` then embeds the passages that changed. A file of UTF-8 that holds the
  character itself is left as it is.
- A mail with an empty plain part beside its HTML, as some mail programs send, was indexed without its words: the
  empty part was taken. Its HTML is read instead.
- A line such as "```code```", code on its line alone, was taken for the start of a code block that never ended: the
  rest of the note lost its title, its headings and its links. A backtick fence is one only without another backtick
  after it on its line, as CommonMark and Obsidian read it.
- A front-matter list of many `- item` lines took time growing with the square of its length: 100,000 items took 14
  seconds, and a note with several hundred thousand was left out by `sift index` as hung, and held up the server's
  `reindex`. It now reads in a fraction of a second.
- `sift embed --reset` dropped every vector before it knew the model could load or the worker answered: offline, or
  with a model that couldn't be downloaded, the vectors were gone and nothing embedded. The model now loads, or the
  worker answers, first.
- `query` passed over every word of one character: "Python 3" lost its version, and a query of one Chinese or Japanese
  word, such as "猫", got no keyword candidates at all. Digits and single characters of other scripts are words now;
  single Latin letters ("a", the "s" of "it's", "à") still aren't. The UI marks them in its results alike.
- Documents and mails, whose readers run native code that may crash or hang, were read in the indexing process
  itself in an update of fewer than a thousand files, and always by the server's `reindex`: one such file ended `sift
  index` on every run, without naming it, or ended `sift serve`. They are now always read in a worker process, each in
  a task of its own with its own ten minutes; up to 64 of them shared one task's time, and forty slow PDFs were taken
  for hung. When a file does end or hang its process, what the other workers had read by then is kept, not read
  again one file at a time, and a document that hangs is left out after its time, not given it a second time.
- The source package (sdist) on PyPI held tests that can't run from it, as they read the repository's docs, scripts
  and workflows it doesn't hold. It holds sift and its metadata alone now.
- The MCP tool `unresolved` failed with a bare `KeyError` when another sift deleted a linking note while it ran. It
  reads the targets and the notes linking them as the index was at one moment, as `links` does.
- The catalog was made a table at a time, its layout version last: a sift stopped in between, or a second one
  starting at that moment on a new SIFT_HOME (a server and a command, say), found tables without a version and asked
  for the catalog to be deleted as an older sift's. It is made in one transaction now.
- When the GPU ran out of memory, `sift embed` embedded the batch again in smaller parts, and emptied PyTorch's cache,
  while the error's traceback still held the failed batch's tensors: the parts could run short of memory as well, and
  the batch size stayed halved for the rest of the run. Both now happen once the error is let go.
- On Windows, Ctrl+C didn't stop `sift embed` while it waited for its first slab of a large collection, read for
  minutes: a wait without a timeout there takes no Ctrl+C until it ends. It waits in steps of a tenth of a second.
- `adaptive` cut hits whose scores were all alike to the first six, as if the first one stood out. With no drop in
  score, every hit is returned.
- A filter or sort on a front-matter field whose name ends in a backslash failed with SQLite's "bad JSON path", one
  with a backslash inside matched nothing, and one with a double quote was refused: any name now filters and sorts
  (one with a double quote where SQLite reads it in a name, 3.47 and newer; elsewhere it is refused, with the reason).
  A filter of a list of some 16,000 values failed with "too many SQL variables"; its values now go to SQLite as one.
  A filter value that is no finite number (NaN, infinity) is a clear error.
- `k: true` was taken for one result, as Python counts `True` as 1; it is refused as `k: "3"` is.
- A latest version with a digit such as "²" in sift's update check, as PyPI could answer, failed every command until
  `update.json` was deleted. Such a version is no release's, and is passed over.
- A file waiting for a path that another sift freed just as `sift index` took the freed ones was marked taken without
  being read; the next `sift index` read it. It is now taken by the running one.

## [1.14.0] - 2026-11-21

Agents learn on connecting how to search well, `sift index` survives readers that crash or hang and keeps a
collection whose folder came back empty, and keyword search no longer falls back to FTS5 for a collection without
documents.

### Changed

- The instructions `sift serve` gives agents when they connect start with how to search well: which tool fits what
  the agent has (a question, an exact string such as a version or error code, an idea, a document), and how to research
  in rounds: several phrasings in one call, a compact survey, the best few read in one `get`, then `similar`, `links`
  and new terms, until a round finds nothing new. The collections come last: a client may keep only the start of a
  server's instructions (Claude Code is reported to cut them at 2,048 characters), and a long list of collections
  then loses its own end, not the tools. Each search tool's description says what its results carry and how to read
  on, for agents that find a tool by tool search and read its description alone.
- The MCP tools `status` and `links` return their result as data too (structured content), as the other tools do,
  beside the same text as before: a client that hands its model the data hands it compact, less than half the size
  for `status`.
- Where stderr is no terminal, in a file, a pipe, a service's log or an agent's shell, the progress line is a line
  a minute and one where it ended, as in the logs. It was drawn again there too, one more line an update and a
  second: seven for five files, thousands for a night of embedding. In a terminal nothing changes; `sift worker`'s
  count of what it served shows in a terminal only.

### Fixed

- A worker that shows no certificate is refused as one showing another than its key names: "has another
  certificate than its key", before anything is sent. It ended in a TypeError.
- A worker whose name resolves only to addresses this Python can't connect to, IPv6 ones where Python was built
  without IPv6, says so: it ended in a TypeError.
- A collection's index that doesn't say which collection it is, as only damage leaves one, says so, and what to do,
  when its keyword index is built or brought up to date: it ended in a TypeError.
- `sift index` keeps what is indexed of a collection whose folder holds none of its files any more, and says so,
  with how to let it go (`sift rm NAME`). A drive not mounted leaves its folder there, empty, as does a folder being
  filled anew: every document went, and the vectors embedded for them.
- A file whose reader ends its worker process, as a converter's native code can, or comes to no end, is left out and
  said, and the other files are read: `sift index` ended in a traceback, or never, and every later run met the same
  file again. A file gets ten minutes, and a second more for every 100 KB. Ctrl+C ends the worker processes at once,
  not once they have read every file started.
- A sift whose stderr no one reads any more, a pipe to a program that ended, goes on with its work. `sift serve`
  stopped at its next message, five minutes later that the models had left the GPU, and a batch too large for the
  GPU ended in the error of saying so.
- A keyword search of every collection works between `sift add` and the new collection's first `sift index`: it
  ended in "The weights and list don't have the same length", as did `query`, on the command line and over MCP.
- A text file of more than a billion bytes, the most a document's text may be, is left out unread, and said, until
  it changes: it was read whole, cut into passages, then refused by the index, and so again on every run.
- A server keeps what it has read of the keyword postings' long lists within 256 MB, as it keeps their blocks and
  the phrases: it kept those of 1,024 words per collection however long, 12 MB for a word in every passage of 50
  million.
- `links` reads a note's links and the notes they name as they are at one moment: a `sift index` beside the server
  that removed a linking note in between ended the call in "KeyError".
- `sift index` passes over what is no file under a name it reads, a named pipe or a folder linked as `x.md`: a pipe
  held it for good, a folder was unreadable on every run.
- The page looks for a document's file off the server's event loop: on a share that doesn't answer, the server heard
  no request for as long as the system waited.
- `reindex` says a file whose reader fails `unreadable`, with why, and reads the call's other files, as `sift index`
  does: the call failed whole.
- A server keeps its last searches' results through a search of a phrase it hasn't counted yet: each one let every
  result kept go, and wasn't kept itself, so an agent asking phrases got little from memory.
- A collection without documents sends the searches of the others to their keyword postings, not to FTS5, seconds
  over millions of passages; and `reindex` builds the postings of a small collection that has none (under 2,000
  passages), as one an agent fills through `reindex` alone, which had none until a `sift index`.

## [1.13.0] - 2026-11-21

sift updates itself on a yes, looks up a shortened id instead of reading every path, and indexes long documents and
large JSON Lines files in a fraction of the time and memory.

### Added

- In a terminal, sift says when a newer version is out and, on `y`, updates itself with the installer it came by (uv,
  pipx or pip) once no sift of the installation runs any more; the next command says how it went. It asks PyPI once a
  day, in the background; `SIFT_UPDATE_CHECK=off` or `HF_HUB_OFFLINE=1` stops it. Never for `serve`, `worker`, or
  output to a pipe. A new major version is only told of, with its changelog. While the installer runs, a command
  started stops and says so; an update that a restart of the machine ended is said and offered again. Where sift
  can't update itself, it says the newer version and the command: in an environment without pip, as uv's are (`uv pip
  install --upgrade sift-rag`, or `uvx sift-rag@latest`), and in an installation you may not write.

### Changed

- Logs no longer grow without end: the latest 20 of each command are kept, and a log past 16 MB, as a server's that
  logs every request, goes on in a new part, its last three kept (`.log.1` to `.log.3`).
- Indexing a JSON Lines file whose records changed takes half the time: the old versions go together, not one by
  one. 50,000 changed records took 11.4 s to store, and take 5.7 s.
- A document named otherwise than by its whole id is looked up, where every path of its collection was read: a
  shortened id (`papers/2603.29315.md` for `papers/2026/03/2603.29315.md`), a record's id alone, a mistyped id, or an
  absolute path in another case, for `get`, `similar`, `links`, `mentions`, and the `links_to` and `linked_from` of
  searches. Over 300,000 documents that took 16 to 32 ms, and 322 ms for an absolute path not indexed, which read every
  document; it grew with the collection. Now 0.03 to 0.05 ms, whatever the collection holds.
- The index layout is version 10: a store lists the last part of every document's path, a file's name or a record's
  id, in 23 bytes a document; indexing 300,000 records took 38.5 s instead of 36.9. An index of 1.12 gets the list on
  first use, empty, and the next `sift index` fills it, 3 million documents in 5 seconds; stopped, it goes on where
  it stopped, and other sifts read and change the collection meanwhile. Until then such ids are searched for as
  before. An older sift doesn't open an index so updated.
- Updating by hand: stop `sift serve` and `sift worker` first. On Windows a running sift holds the files an update
  replaces: `uv tool upgrade` under a running server upgraded the packages but failed on `sift.exe`.
- A filter's value matches whether a document writes it as a number or as text: `-f year=2024`, and `{"year": 2024}`
  over MCP, find `year: 2024` and `"year": "2024"` alike, and `true` and `false` likewise. Before, only the page in
  the browser asked both ways.
- A search without an active collection is an error that says what to do (`sift activate NAME`, or `sift add FOLDER`
  where none is registered), on the command line, over MCP and on the page. A keyword search answered "no matches".
- Passages end before a Markdown heading only, not before any line starting with `#`: a comment in a code block, a C
  include or a tag cut a passage in two. Files already indexed keep their passages until they change.

### Fixed

- A large document took time by the square of its size: each of its passages was cut out of its whole text anew. An
  80 MB text file, 32,795 passages, took 11 minutes to index and takes 16 s; an 8 MB file's passages took 20 s to
  read for `sift embed` and take 0.01 s; 50 passages of it among a search's results took 0.3 s and take 7 ms; a
  phrase is counted in 200 of them in 0.04 s, not 1.7 s. Documents of a few passages are stored as before, and
  nothing in an index changes.
- A large JSON Lines file was read whole: one of 300,000 records, 69 MB, took ten times its size in memory. Files over
  32 MB are now read as they are stored, a batch of records at a time: storing that file peaks at 297 MB, against
  681 MB, most of it SQLite's page cache, and no longer grows with the file. They are parsed in the indexing process
  rather than in worker processes. Building the keyword postings afterwards still peaks near 700 MB for it.
- `sift embed` beside `sift index` embedded the chunks indexed meanwhile, but its total stayed as it began: the bar
  went past 100% and the time left below zero. The total now follows the chunks indexed or deleted while it runs.
- `sift ui` asked for the server that runs through the proxy set for the web, which knows no server of this machine,
  and took a server busy for five seconds for none: it started a second one on the port taken.
- A browser kept the page's parts, which share the page's address, and showed one alone on the way back to it. Pages
  now forbid keeping them, which also keeps documents' texts out of the browser's cache.
- `sift search ... | head -1` ended with "error: [Errno 22] Invalid argument" on Windows. A reader that leaves now
  ends the command quietly.
- What sift says on stderr is UTF-8 now where it wasn't, as its results are: in a Windows pipe, file names and titles
  came out unreadable.
- Every command ended in an `AttributeError` where stdout or stderr is no file, as within a program that set its own
  (`contextlib.redirect_stdout(io.StringIO())`), or where there is none, as under `pythonw`. Such a stream is left as
  it is now, and a command without streams runs and says nothing.
- Without a stderr but with a stdout, as `pythonw -m sift search ... > results.txt` from a console has them, what sift
  says went to stdout, among the results: "no matches", a progress line, "starting ..." before the JSON of `--json`.
  It is said nowhere now; the command's log holds it as before.
- A store that was locked or damaged, or a call cancelled, was said as "invalid search query".
- A bug that is a `KeyError` or an `IndexError` came out as a bare message (`error: 'fields'`); it is reported as a
  bug now, with its type, and its traceback on the command line and in the server's log.
- On Windows, an index on a network share, or on a drive mapped to one, was made and then never opened again ("unable
  to open database file"): `SIFT_HOME` there, `sift add --store`, `sift move`. SQLite refused the address of a file on
  another machine as sift wrote it.
- A file named by its absolute path through the folder as it was added, a link or junction to it or a drive mapped to
  a network share, was "in no collection" for `reindex`, `sift index --files` and `get`, and the `sift add` suggested
  was refused: a collection keeps its folder as it resolves. Such a path is resolved now where it isn't found as given.
- A range on a number took text without a number for 0: `-f "year<2000"` found every note whose year was `unknown`,
  empty, or a list inside a list. Text that starts with a number, as a date, still compares as that number. A range
  given as text, `-f "created>2024"`, compared a list inside a list by its JSON, and found it.
- A scan skipped as `SIFT_OCR` was off, or as there was no GPU, was read only once its PDF changed: setting
  `SIFT_OCR=on`, as the skip says, read none of them. The next `sift index` after `SIFT_OCR` is turned on, or from
  `off` to `auto`, reads them.
- `reindex` and `sift index --files` failed the whole call, every file of it left as it was, when the index refused
  one file's text, as text over SQLite's limit of 1 GB. That file is "unreadable" now, with why, and stays as it was
  indexed; the others are read, as `sift index` reads them.
- `sift worker` failed a request whose sender's address the server didn't give, and took any number of texts in one,
  holding the GPU for as long as a request of millions took. It takes at most 20,000, the most a `query` reranks.
- One collection's index damaged, every page of the UI failed: its collection list now says that index is damaged
  and what to do, and the others' pages work.
- `sift embed` and `sift serve` said "loading PyTorch" through a worker, where they never load it, and so did `sift
  similar` anywhere, which runs no model. `sift serve` said it loaded the models into RAM: it now names where they
  are, the worker's address or the device.
- `sift worker` wrote its token and certificate in place: one cut short by a crash or a full disk changed the key other
  machines were given, unsaid. Each is written whole or not at all now.
- `SIFT_HOME=~/sift`, as an MCP client's config or Windows' `setx` passes it, unexpanded, was a folder named `~`
  wherever a command ran. `~` is the user's folder now, and a relative home is taken from where the command runs, so
  the update's process, which runs at the root of the drive, keeps the state the commands read.
- A full disk left `update.json.PID.part` behind in `SIFT_HOME`, one more for each command that tried to write the
  update check's state. It goes now, as when the state can't be replaced.
- The docs said `sort` orders results by a field "instead of score", which read as the newest of every match first.
  It orders the best `k`; a range filter on the field finds the newest of all. They also said a mail's `date` is in
  UTC, which it is only where the mail gives its time zone.

## [1.12.0] - 2026-11-17

Several documents or queries per call, the vectors scanned three times as fast by `sift serve`, and no file left out
for good behind a record's path.

### Added

- `get` takes up to 100 documents at once, each an id or `{"id", "lines"}`; an unreadable one comes back with its
  error, and texts past 100,000 characters in all are left out, each saying why. `sift get` takes several ids too.
- `search`, `vsearch` and `query` take a list of up to 20 queries and return a list of results; the queries of
  `vsearch` and `query` are embedded in one pass, those of `search` share its `budget`.

### Changed

- `sift serve` scans the vectors' codes three times as fast. sift now depends on [Numba](https://numba.pydata.org);
  the server and `sift bench` load the compiled scan at start (a second or two, cached in `SIFT_HOME/numba`). A single
  command-line search, or a server that can't load Numba (it says so), uses NumPy.
- `query` reranks in batches of 16 sorted by length: 180 ms a question instead of 331.
- A server keeps the embeddings of the last 256 questions and the results of the last 32 searches, so asking again
  with another `k` or filter skips the embedding. Results are dropped when anything they read changes; searches with a
  time budget, over 1,000 hits, or over an unreadable collection are never kept.
- `query` over collections not embedded yet no longer loads the model.
- The documentation: the README is the entry point, and the details are in `docs/` (usage, worker, architecture,
  benchmarks, contributing, security), each fact in one place.

### Fixed

- A semantic search of one collection (`collection`, `-c`) read the codes of every active collection first (half a
  minute and 4 GB beside a collection of 30 million passages); it reads only those it searches.
- A server embedding through a worker (`SIFT_EMBED_REMOTE`) answered nothing while a request waited, up to ten minutes
  if the worker's machine slept; cancelled calls now stop waiting.
- A file whose path a JSON Lines record or mail attachment had taken (`data.jsonl` with a record `a.md`, beside
  `data/a.md`) was skipped for good; it is now read as soon as the path is free. On upgrade, stores re-read once the
  files without documents and the JSON Lines files and mails with a file below their names; other left-out records
  come back when their file changes. A store sift may only read is upgraded once it may write.
- `sift index` messages under its progress line are now in its log.
- Front-matter JSON strings no longer end at U+2028, U+2029 or NEL; a NUL in a file no longer hides the text after it
  (stored as U+FFFD; documents indexed before keep the cut until their text changes); `.siftignore` no longer ends
  patterns at U+2028, form feeds or NEL; `get` works for ids holding a NUL.
- A mail dated in year 9999 with a time zone, or in UTF-7 holding half a character, is no longer skipped; a UTF-32
  file with a byte order mark is no longer read as UTF-16.
- `get` of several documents no longer fails for all when one's store is out of reach.
- With the GPU hidden on purpose (`CUDA_VISIBLE_DEVICES=-1`), sift no longer says to update the NVIDIA driver.
- One damaged store (a table broken though the store opens) stopped `sift index`, `status` and the server's start for
  every collection; it is passed over and said, with how to forget it. A search that reads its damaged tables fails.
- Ctrl+C or an error while the keyword postings were built in-process left an undeletable temporary file on Windows.

## [1.11.2] - 2026-11-08

No search held up while `sift embed` runs beside the server, and repeated phrases two to twenty times as fast.

### Changed

- Phrases are counted once per passage and kept: a common phrase took 131 ms instead of 2.6 s, same results.

### Fixed

- `sift serve` no longer stalls while `sift embed` writes beside it; only new codes are read, other vector changes
  reload in the background.
- One bad file (lone emoji half in JSON, JSON nested past 1,000, a number like `1e999`, a mail with raw 8-bit
  addresses, a converter panic) no longer stops `sift index` for every collection; that file is skipped or read
  leniently, said, and retried by the next run.
- `sift index` beside a server's `reindex` of the same folder no longer fails (`UNIQUE constraint failed`) or leaves
  deleted documents in the index; it holds no write lock while reading files, so `sift embed` or `reindex` beside it
  no longer gives up after 30 seconds.
- A collection whose index can't be opened (damaged, newer sift, locked, out of reach) is passed over and said by
  searches, `sift index`, `sift embed` and `sift serve`; `status` gives the reason as `error`. It stopped all of them.
- A scan OCR couldn't read for the moment (GPU memory taken by another program) is read again by the next `sift
  index`.
- Smaller fixes: UI folders cost one query each and a U+10FFFF folder no longer hangs the server; deeply nested
  keyword queries are searched as words; very long lines are cut in linear time; `sift embed` beside `sift index`
  no longer skips embedding a replacement passage; searches return at most 10,000 hits; a running server releases a
  removed or deactivated collection's store.

### Security

- `reindex` (and `sift index --files`) of an id such as `notes/D:/file.md` read that file on another drive into the
  collection on Windows. Windows names that reach another file (`C:x.md`, `x.md:stream`, short names such as
  `PRIVAT~1`, which also bypassed `.siftignore`) are refused, and junctions in a collection's folder are passed over.
- `sift worker` no longer writes its key into its log.
- `sift serve --host ::1` answers only requests that name a local host: a web page reaching it under its own domain
  could read documents through the UI.
- The UI serves no file outside its collection's folder, though an older sift's `reindex` may have stored one.

## [1.11.1] - 2026-11-08

Keyword questions and phrases in milliseconds, a quicker vsearch over many collections, and `sift embed` starting at
once.

### Changed

- Keyword search reads a query's common words only where they can matter, with the same results: questions of 10 to
  14 words took 34 to 81 ms instead of 0.44 to 2.5 s.
- Phrases (`"retrieval augmented"`, "first-stage", "COVID-19") are answered from the postings too: 0.43 s instead of
  8.7 over arXiv, and no longer over ten minutes over PubMed. In a nested query a phrase now counts in every passage
  that holds it. A word of no token, such as a dash, is left out as FTS5 does.
- `vsearch` over many collections takes half the time; `sift embed` starts at once.

### Fixed

- A keyword query of several collections miscounted a self-overlapping phrase and phrases in passages holding U+0001.

## [1.11.0] - 2026-11-08

Keyword postings in topical order, quicker filtered and exact semantic search, one hit per paper, and fewer results
for agents when few stand out.

### Added

- A paper several collections hold (arXiv and PubMed, say) is one hit: the best ranked copy, listing the others under
  `also`. Copies share a DOI (front-matter field `doi`). `all_passages` keeps every copy.
- `adaptive` (`--adaptive`) for every search: up to `k` hits, those before the largest drop in score and five more.
- `budget` (`--budget SECONDS`) for keyword search: once spent, it returns the best found so far, each hit marked
  `"partial": true`. Without it, results stay exact.

### Changed

- The index layout goes to 9 by itself. The next `sift index` builds a collection's postings anew once it has
  vectors, and again once a fifth of its passages got vectors since. A sift 1.10 still running searches such a
  collection through FTS5.
- `vsearch` finds 99.75% of exact search's top 10 instead of 99.6%, reading a quarter of the vectors; a filtered
  `vsearch`, `query` or `similar` takes 0.6 seconds instead of 6 for a filter letting 9% through.

### Fixed

- A keyword search reads the postings in one transaction; another sift rebuilding them meanwhile made it fail.

## [1.10.1] - 2026-11-08

### Changed

- `sift embed` keeps the GPU busy: 472 passages a second instead of 382, with identical vectors.

### Fixed

- `sift embed` sized batches past PyTorch's cap, so the first batch ran out of memory and later ones were halved.
- A collection's write-ahead log shrinks back to 64 MB once written into the index.
- A keyword search no longer fails when another sift deletes a passage it had found.

## [1.10.0] - 2026-11-07

Keyword search in milliseconds over tens of millions of passages, and a server that stops what agents give up on.

### Changed

- Keyword search (`search`, and the keyword candidates of `query`) ranks from postings of its own, with FTS5's
  results, score for score: over 29.9 million PubMed passages a common word took 7 ms instead of 1.3 s, a question
  26 ms instead of 5.3 s. Phrases, prefixes, NEAR and column filters are FTS5's to answer, as before.
- The index layout is version 8. An index of 1.9 gets the postings' tables on first use, and `sift index` builds them
  (18 minutes for PubMed); until then keyword search reads FTS5. They add a third of FTS5's index in size. An older
  sift doesn't open an index so updated.
- The postings follow every change; a collection whose postings are behind is searched through FTS5 meanwhile.
  `sift status` and the MCP `status` tool show `"keywords"`: `"ready"`, `"not built"`, or how many chunks behind.

### Fixed

- A call an agent gave up on (timeout, cancel) no longer runs on in `sift serve`, and `sift serve` answers
  connections, pings and cancellations while a call runs.

## [1.9.0] - 2026-11-07

Status in a moment on any index, embedding a third faster with the CPU left free, and a README to the point.

### Changed

- `sift status`, the MCP `status` tool, the UI and a starting server read counts as the index keeps them instead of
  reading every vector and document (`sift status` took over ten minutes over tens of millions of chunks).
- The index layout is version 7. An index of 1.8 counts its rows on first use, once. An older sift doesn't open an
  index so updated.
- `sift embed` batches by length in tokens (171 to 230 chunks a second on PubMed abstracts, same vectors) and no
  longer reads every pending chunk at the start. Beside an NVIDIA GPU, PyTorch keeps a single CPU thread.
- The README is a third as long; details moved to
  [docs/usage.md](https://github.com/tobiaswuerth/sift/blob/main/docs/usage.md),
  [docs/architecture.md](https://github.com/tobiaswuerth/sift/blob/main/docs/architecture.md) and
  [docs/worker.md](https://github.com/tobiaswuerth/sift/blob/main/docs/worker.md).

## [1.8.0] - 2026-11-06

Each collection's index in a file of its own, wherever it fits, and collections left out of searches until wanted.

### Added

- `sift move [NAME ...] TO` moves the index of the named collections, or every one, to another folder or drive, or
  one collection's to a `.sqlite` file name. Instant on the same drive; refuses while another sift has it open.
- `sift add --store PATH` puts a new collection's index in another folder or file.
- `sift deactivate NAME ...` and `sift activate NAME ...`: an inactive collection is passed over by searches, leaves a
  running server's memory, isn't named to agents, and is skipped by `sift index` and `sift embed` unless named.
  `reindex` still updates it.
- `sift status` and the MCP `status` tool show each collection's `active`, `store` file and `size`, or `missing`.

### Changed

- Each collection's index is a SQLite file of its own, `collections/NAME.sqlite` under `SIFT_HOME`;
  `SIFT_HOME/sift.db` is a small catalog. `sift rm` deletes the file at once; commands on different collections no
  longer wait for each other.
- The index layout is version 6. An index of an earlier sift (0.3 on) splits itself on first use, once: it needs as
  much free space as the index takes, re-reads and re-embeds nothing, and resumes if stopped. `sift move FOLDER` as
  the first command splits straight into that folder. An older sift doesn't open an index so updated.
- Searches merge the collections' results as one index ranked them. A query with NEAR or a column filter keeps each
  collection's own keyword scores.
- A collection whose index file is missing (drive not connected) is passed over with a warning, never made anew.

## [1.7.0] - 2026-11-05

Semantic search over tens of millions of passages, in RAM and without a GPU.

### Added

- `sift embed COLLECTION ...` embeds only the named collections. Without names, and with `--reset`, it embeds every
  collection, as all share one model.

### Changed

- `vsearch`, `query` and `similar` keep one bit per vector dimension in memory, a sixteenth of the vectors: 3.2
  million passages take 0.3 GB of RAM, and 40 million fit a desktop's RAM. They find 99.6% of exact search's top 10
  ([benchmarks](docs/benchmarks.md#vectors-in-memory)). Only the models use the GPU.
- The index layout is version 5. An index of an earlier sift (0.3 on) updates itself on first use, once, without
  embedding again (about a minute for 3.3 million vectors). An older sift doesn't open an index so updated.
- NumPy 2.0 or later is required.

## [1.6.0] - 2026-10-25

### Fixed

- `sift index` keeps the documents of a folder it can't list for the moment (access denied, a silent network share)
  instead of dropping them with their vectors; `reindex` calls such a file `unreadable`, not `removed`.

## [1.5.0] - 2026-10-24

### Added

- A document's absolute path is an id too: `get`, `similar`, `links` and `mentions` take it.

### Fixed

- A crash of `sift index` and `sift embed` on Python 3.15 while a progress line was drawn.

## [1.4.1] - 2026-10-24

### Fixed

- A long step's progress goes into the command's log once a minute.

## [1.4.0] - 2026-10-23

### Added

- A log per run: `sift add`, `rm`, `index`, `embed`, `serve` and `worker` write what they say to
  `SIFT_HOME/logs/<command>-<time>.log`. Every log is kept.

## [1.3.2] - 2026-10-23

### Fixed

- On Windows, `sift serve` and `sift worker` could stop answering for good when a client went away mid-accept.

## [1.3.1] - 2026-10-23

### Fixed

- The server's `reindex` skipped every PDF, office file, e-book, web page and mail attachment; the next `sift index`
  reads once what it had skipped. Without a GPU, models load in float32, not bfloat16.

## [1.3.0] - 2026-10-23

Scans are read, and two things real collections met are mended: mails whose HTML the converter failed on, and a
collection added inside another.

### Added

- Scans: PDFs without text of their own, and a mail's attached scans, are read with OCR by LightOnOCR-2 (Apache-2.0)
  on the GPU the other models use, in about 3 GB; metadata says `ocr`. The first scan downloads the model (2 GB).
  Without a GPU scans are skipped and reported; `SIFT_OCR=on` reads them on the CPU, `SIFT_OCR=off` never. The next
  `sift index` reads once the scans and mails with a PDF an older sift skipped; the server's `reindex` leaves scans to
  `sift index`.

### Changed

- `sift add` refuses a folder inside a collection or holding one; `sift add --nested` adds it anyway.

### Fixed

- A mail whose HTML the converter fails on keeps its words: it is read as plain text, and that is reported.

## [1.2.1] - 2026-10-22

### Changed

- The package's description and keywords on PyPI name what 1.1 and 1.2 added. Nothing else changed.

## [1.2.0] - 2026-10-22

For collections that other programs write to: a `.siftignore`, saved emails, the folder of each collection, and
errors that say what to do ([Leaving files out](docs/usage.md#leaving-files-out), [Emails](docs/usage.md#emails)).

### Added

- `.siftignore`: a collection's folder may name, in gitignore's syntax, the files and folders sift leaves out.
  Hidden folders, `node_modules` and `__pycache__` stay out as before.
- Saved emails (`.eml`): one document per mail, titled by subject, with sender, recipients and date (ISO 8601, UTC)
  above its text and as filterable metadata. Attachments sift reads, and mails attached to mails, are documents too.
- `sift status --json`.

### Changed

- The instructions agents read name each collection's folder; `status` holds it as `path`.
- `reindex` of a file in no collection lists the collections and the `sift add` that would cover it; an unknown
  collection name lists the names there are.
- A front-matter value is read as JSON first, so programs can write any value as JSON and read it back as written.

### Fixed

- A file named like an extension, such as `.md`, no longer stops `sift index`.

## [1.1.0] - 2026-10-22

Links between Markdown notes, for Obsidian vaults and any notes that link to each other
([Linked notes](docs/usage.md#linked-notes)).

### Added

- Links: sift reads the wikilinks, embeds and Markdown links of every Markdown note, and resolves them within the
  collection as Obsidian does (path, name, alias).
  - `links` (MCP tool and `sift links`): a note's links and backlinks; `depth` 2 for two steps.
  - `unresolved` (MCP tool and `sift unresolved`): link targets no note answers to yet.
  - `mentions` (MCP tool and `sift mentions`): documents that name a note without linking to it.
  - `links_to` and `linked_from` on `search`, `vsearch`, `query` and `similar` (`--links-to`, `--linked-from`).
  - `status` counts each collection's links and unresolved ones; `sift ui` leads to backlinks and links.
- `reindex` (MCP tool) and `sift index --files`: read the given files at once, or drop the documents of files that
  are gone, so the next search finds a note an agent just wrote. Over MCP it embeds the new passages too.
- `sift add --title-from name`: Markdown and text files take their file name as title unless front matter sets one.
  On a collection added before, it retitles from the index and says which passages `sift embed` must embed again.

### Changed

- The index layout is version 4. An index of 1.0 updates itself on first use, once (20,000 notes take about 3
  seconds), without reading files or embedding again. sift 1.0 doesn't open an index so updated.
- A filter value that is a wikilink, `{"org": "[[Acme AG]]"}`, also matches links to the same note written otherwise.
- A note of front matter alone has its front matter as its passage, so searches find it; an index of 1.0 gets these
  passages when it updates, and `sift embed` embeds them.

### Fixed

- A YAML list with unquoted items, `aliases: [Hans Mueller, H. Müller]`, was one string; it is a list now.

## [1.0.0] - 2026-10-21

sift is stable: from now on, its commands, MCP tools, settings, index and worker protocol change incompatibly only
with a new major version. [Updating](https://github.com/tobiaswuerth/sift#updating) lists what that covers.

### Changed

- `get` of lines past the end of a document is an error that says how many lines it has (it returned empty text).
- A `sift worker` and the machines that use it must run the same major version (before 1.0: the same minor one).
- `sift add` refuses a folder that is a collection already.
- `sift serve` on an address other than `127.0.0.1` or `localhost` warns that anyone who reaches it can read every
  indexed document.
- `sift bench` takes grade 0 for a document read and found irrelevant, as TREC does.
- The package declares every package sift imports itself: uvicorn, Starlette, AnyIO, Pydantic and MarkupSafe.

### Fixed

- Filters, sorting and `-f` take any field name without a double quote; IPv6 addresses work in `sift ui` and `sift
  serve`; a front-matter title such as `title: 1984` is no longer passed over; `sift bench` reports unusable cases as
  errors; many `sift ui` view and keyboard bugs.

### Security

- `sift ui` opened a collection's files in its own origin, so a saved web page with scripts could read the whole
  index. Files now open sandboxed, PDFs excepted (see
  [SECURITY.md](https://github.com/tobiaswuerth/sift/blob/main/docs/SECURITY.md)).
- `sift ui` kept the last pages shown, document text included, in the browser's local storage; it no longer does.
  htmx no longer runs scripts in what it swaps in.

## [0.10.0] - 2026-10-21

### Added

- `sift ui`: the index in your browser, to search, browse and read the sources yourself. It opens the UI of the
  `sift serve` on port 8765, or starts one, which then serves agents too.
- `sift serve` serves the UI under `/ui`. Bound to a local address, it answers only requests that name a local host.

## [0.9.1] - 2026-10-21

### Fixed

- Embedding and searching through a `sift worker` waited 21 seconds per request when the worker's name resolved to an
  IPv6 address first; sift now connects over IPv4 when it can and keeps one connection open.

## [0.9.0] - 2026-10-21

### Added

- `sift bench FILE`: how well each search mode finds the documents a set of questions should find, on your own index
  (nDCG@10, MRR@10, Recall@10, time per query), from a JSON list of questions and the ids of their answers.

### Changed

- `query` ranks better across subjects and languages: the default reranker is gte-reranker-modernbert-base and each
  candidate's score also counts its closeness in meaning to the query
  ([benchmarks](docs/benchmarks.md)). The first `query` after upgrading downloads it (about 600 MB); indexes and
  vectors stay. `SIFT_RERANK_MODEL=cross-encoder/ettin-reranker-150m-v1` brings the old reranker back.
- Scores keep the reranker's scale: close matches score about 2.5 to 4, unrelated results below 2.

### Fixed

- A reranker that reads fewer than 1024 tokens, such as MiniLM, no longer fails on long passages.

## [0.8.0] - 2026-10-20

### Added

- Keyword search covers front-matter values, such as authors and ids, in `search` and `query`: "Vaswani transformer
  paper" finds "Attention Is All You Need". `meta:` limits a term to them. `query`'s reranker reads them too.

### Changed

- The first command after upgrading rebuilds the keyword index, once (3½ minutes and 0.7 GB more for 3.2 million
  arXiv abstracts). Documents and vectors stay. Older versions of sift can't open it afterwards.

## [0.7.0] - 2026-10-20

### Added

- `similar`: documents like a given one, by id (MCP tool, `sift similar ID`). It runs no model.
- Filters take a list for any of its values, `{"categories": ["cs.RO", "cs.CV"]}`, and `{"not": ...}` for none of
  them. On the command line: `-f KEY=VALUE` repeated, and `-f KEY!=VALUE`.
- `sort` orders results by a front-matter field instead of score (`--sort=-created`).
- `get` takes `fields` and returns `{id, title, fields, text}` (`sift get --fields url,created`), and a line number
  as well as a range.
- `all_passages` (`--all-passages`) returns every matching passage, as searches did before.
- Every MCP tool parameter is described in the tool's schema, and the tools are marked read-only.

### Changed

- Searches return one result per document, with its best passage.
- `query` scores are the reranker's raw scores: with the default reranker, close matches score about 8 to 12,
  unrelated results below 5.
- Keyword search weighs a title like the text; an empty environment variable counts as unset.

### Fixed

- A tool failing unexpectedly now tells the agent the error's type and message; `get` honours `fields` over MCP.

## [0.6.0] - 2026-10-20

### Added

- Through a worker, searches rerank there too and the searching machine never loads PyTorch: `vsearch` and `query`
  take about 0.3 s instead of 10 to 25 s on a notebook. Both machines need 0.6.

### Changed

- The models compute in bfloat16 on NVIDIA GPUs that have it, and in float32 elsewhere, instead of fp16.

### Fixed

- `vsearch` and `query` found fewer results, or none, when some vectors were NaN (fp16 overflow). Broken vectors are
  never stored, and the next `sift embed` drops those of older indexes and embeds their chunks again.

## [0.5.1] - 2026-10-20

### Added

- `sift worker` lends a machine's GPU to sift on other machines, and `SIFT_EMBED_REMOTE` points sift there: `sift
  embed` and searches send their texts to the worker, which sees no index or files and gives the GPU back when idle.
- The worker speaks HTTPS with a certificate of its own and answers only requests with its token. It prints the
  address to set, `https://<key>@<machine>:8766`, whose key holds the token and the certificate's fingerprint.
  `cryptography` is a new dependency.

## [0.5.0] - 2026-10-20

Tagged but never published: its worker spoke plain HTTP, without a key. 0.5.1 replaces it.

## [0.4.1] - 2026-10-20

### Fixed

- Office files whose XML compresses well, common in presentations and spreadsheets, were skipped as a "potential ZIP
  bomb"; Xberg's caps on unpacked size still stop zip bombs. `sift embed` shows progress without a GPU.

## [0.4.0] - 2026-10-20

### Added

- PDF, Word, PowerPoint and Excel files, OpenDocument text, presentations and spreadsheets, RTF, EPUB and HTML,
  converted to Markdown by [Xberg](https://github.com/xberg-io/xberg), a new dependency. Their properties (title,
  authors, dates, ...) become metadata. Damaged files and scans without text are skipped and reported.

### Changed

- The default reranker is `cross-encoder/ettin-reranker-150m-v1` (Apache-2.0), replacing
  `Alibaba-NLP/gte-reranker-modernbert-base`. It downloads on first use; nothing needs re-indexing or re-embedding.
  `SIFT_RERANK_MODEL=Alibaba-NLP/gte-reranker-modernbert-base` keeps the old one.

## [0.3.0] - 2026-10-20

### Added

- JSON Lines files (`.jsonl`, `.ndjson`): one document per line, with id, title and text found by name (`id`,
  `title`, `text`, `abstract`, ...) and every other field as metadata. `sift add --id/--title/--text` chooses other
  fields. A changed file re-indexes only its changed records.

### Changed

- **The index holds the documents' text**, so `get` reads the index instead of the file. **Indexes made by sift 0.2
  or older must be rebuilt**: delete `sift.db` and run `sift add`, `sift index` and `sift embed` again.
- The default embedding model is `google/embeddinggemma-2` (768 dimensions, Apache-2.0), replacing
  `Qwen/Qwen3-Embedding-0.6B`. Run `sift embed --reset` to switch an existing index. It needs Pillow and torchvision,
  now dependencies; with pip, install torchvision from the same index as torch.

## [0.2.0] - 2026-10-19

### Added

- Search results can carry front-matter fields (`fields`, e.g. `["url", "created"]`, or `["*"]`) and leave out the
  text (`compact`). `--fields` and `--compact` on the CLI.
- Range filters on front-matter fields, e.g. `{"created": {">=": "2024-06", "<": "2025"}}` or `-f "created>=2024-06"`.
- `get` returns only some lines if asked (`--lines` on the CLI), and finds a document by its file name if no other
  document in the collection has that name.
- The server tells agents which front-matter fields each collection has; `status` lists them too.

### Changed

- `sift serve` gives its GPU memory back when idle: after 5 minutes without GPU use models and vectors return to RAM,
  leaving the memory free for other programs; the first search afterwards takes about a second longer.
- `search` returns chunks with any of the query's terms when none has them all, unless the query uses operators.
- `get` serves indexed documents only and returns `\n` line endings.
- Front matter lists can be written one `- item` per line; blank queries return nothing in every search mode.

### Fixed

- Many crashes and indexing errors (Ctrl+C hangs of `sift serve`, titles with `#`, UTF-16 files, file names that
  aren't UTF-8, a huge `k`, `sift add ~/notes`).

## [0.1.0] - 2026-10-18

First release.

### Added

- `sift add`, `rm`, `index`, `embed`, `search`, `vsearch`, `query`, `get`, `status` and `serve`; also available as
  `sift-rag`, so `uvx sift-rag serve` works.
- Incremental indexing of `.md`, `.markdown` and `.txt` files with filterable YAML front matter.
- Embedding on NVIDIA GPUs (CUDA), Apple GPUs (MPS) or the CPU; sift caps its GPU memory below what other programs
  leave free.
- Keyword search (SQLite FTS5, BM25), exact GPU vector search, and hybrid search with cross-encoder reranking.
- An MCP server (streamable HTTP and stdio) whose instructions tell agents what is indexed.

[Unreleased]: https://github.com/tobiaswuerth/sift/compare/v1.15.2...HEAD
[1.15.2]: https://github.com/tobiaswuerth/sift/compare/v1.15.1...v1.15.2
[1.15.1]: https://github.com/tobiaswuerth/sift/compare/v1.15.0...v1.15.1
[1.15.0]: https://github.com/tobiaswuerth/sift/compare/v1.14.0...v1.15.0
[1.14.0]: https://github.com/tobiaswuerth/sift/compare/v1.13.0...v1.14.0
[1.13.0]: https://github.com/tobiaswuerth/sift/compare/v1.12.0...v1.13.0
[1.12.0]: https://github.com/tobiaswuerth/sift/compare/v1.11.2...v1.12.0
[1.11.2]: https://github.com/tobiaswuerth/sift/compare/v1.11.1...v1.11.2
[1.11.1]: https://github.com/tobiaswuerth/sift/compare/v1.11.0...v1.11.1
[1.11.0]: https://github.com/tobiaswuerth/sift/compare/v1.10.1...v1.11.0
[1.10.1]: https://github.com/tobiaswuerth/sift/compare/v1.10.0...v1.10.1
[1.10.0]: https://github.com/tobiaswuerth/sift/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/tobiaswuerth/sift/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/tobiaswuerth/sift/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/tobiaswuerth/sift/compare/v1.6.0...v1.7.0
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
