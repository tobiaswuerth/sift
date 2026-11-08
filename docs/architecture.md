# How sift works

How sift indexes, stores, embeds and searches, and why. The measurements behind its defaults are in
[benchmarks.md](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md); the code's layout is in
[CONTRIBUTING.md](https://github.com/tobiaswuerth/sift/blob/main/CONTRIBUTING.md).

## Indexing

- Worker processes read and parse files in parallel while the previous batch is written; small updates skip the
  process start-up.
- Files are compared by modification time and size, so unchanged files cost nothing, and documents by content hash,
  so a changed JSON Lines file re-indexes only its changed records.
- Text files are read as UTF-8, or as UTF-16 when they start with its byte order mark (Windows PowerShell 5.1 writes
  those).
- Each kind of file has a reader, chosen by extension, that turns it into documents; adding a format means adding a
  reader.
- Markdown, including documents converted to it, is split into chunks of about 3,000 characters (about 750 tokens) on
  line boundaries, preferring headings, with a small overlap.

## Storage

- A SQLite database in WAL mode per collection holds its documents' text, metadata, the FTS5 keyword index and the
  keyword postings, the vectors (fp16) and their codes (one bit per dimension); a small one, the catalog, lists the
  collections. Its write-ahead log shrinks back to 64 MB once written into the database, and a postings build, which
  writes through it while its own workers read, empties it at the end: SQLite keeps the log as large as it grew for
  as long as any connection is open.
- The text is stored once: chunks are spans of it, and the keyword index reads it through a view.
- The row ids of a collection start at its id times 2^40, so the results of several collections merge as they are.
- Each index keeps the number of its documents, chunks and vectors, counted by triggers in the transaction of each
  change, so `sift status` and a starting server read no table whole: SQLite keeps no counts, and counting a large
  index's vectors reads all of them.
- Keyword search ranks from postings of its own, read out of FTS5's index, which stays the reference (see below): per
  term, its chunks and how often it occurs in each, zlib-compressed. A long list comes in blocks of 4,096 postings,
  after its ranges: for every 128 consecutive places holding the term, how many postings it has there and its best
  score there, rounded up to a byte. Chunks' lengths come in blocks of 1,024 places. They take about a third of FTS5's
  index, which holds every word's positions.
- The postings number chunks by their place in a topical order, once the chunks have vectors: clusters of clusters of
  their codes (binary k-means by Hamming distance, 64 by 64), each next to the one closest to it, and a cluster's
  chunks in the order of their ids. The chunks of a range then share their words, so its bounds are tight: over 3.2
  million arXiv abstracts, questions scored a half to a twentieth as many ranges, the slowest in half the time, and
  the postings took 8% less space. A table holds the chunk at each place; a chunk changed since takes the next free
  place. `sift index` orders them anew once a fifth of the chunks got vectors since.
- The postings follow FTS5 change by change: every chunk whose keyword entries change is marked in the same
  transaction, and `sift index` (or `reindex`) then adds a delta of those chunks, tokenized by FTS5 in a table of its
  own, and masks their old postings, as Lucene's segments do. Many deltas are merged into one; once 5% of the chunks
  have changed, the postings are built anew. Until a collection's postings are up to date, its keyword searches read
  FTS5, as before. Building them anew for a collection of a million passages or more reads FTS5's vocabulary in
  worker processes, one per CPU but one.
- Searching and `get` never touch the files, so results and documents always agree, until the next `sift index`.

## Embedding

- PyTorch on the GPU (NVIDIA, else Apple), in batches of texts of similar length in tokens: a batch is padded to its
  longest text, and characters tell tokens poorly.
- The models compute in bfloat16 on NVIDIA GPUs that have it and in float32 elsewhere, never in fp16, where
  EmbeddingGemma overflows now and then into broken (NaN) vectors. Vectors are stored in fp16, and broken ones are
  never stored.
- Each slab of 20,000 chunks is committed, so progress is never lost.
- `sift embed` keeps the GPU busy (99% of the time on PubMed). It reads the next slabs and sorts them by length in a
  thread of its own, and writes the vectors in another, the two taking the collection's connection in turn; it counts
  the tokens with a copy of the model's tokenizer, outside PyTorch. The main thread runs the model, through the public
  steps of Sentence Transformers' `encode_document`: each batch is tokenized while the GPU embeds the one before.
  PyTorch stays on the main thread, as it must: it crashes in others on Python 3.15.
- Beside an NVIDIA GPU, PyTorch keeps one CPU thread: the GPU does the work, and a thread per core only waited busily
  after each batch. On the CPU it keeps a thread per core.

## GPU memory

- sift shares the GPU with other programs. On NVIDIA GPUs it caps itself at the memory that is free when it starts,
  minus headroom (10% of the GPU, at least 1 GB), asking NVML as well as CUDA, since on Windows CUDA counts memory
  held by other programs as free.
- It then measures how much memory texts of the longest input the model takes need at growing batch sizes, and uses
  the largest batch that fills 90% of that budget: no text needs more, whatever the model. It counts what PyTorch
  reserves, as its cap does: the tensors and the allocator's rounding, about a fifth more for EmbeddingGemma. If a
  batch still doesn't fit, it is halved. With less than 2 GB to spare, sift uses the CPU instead.
- The models wait in RAM and move to the GPU when a search needs them. After 5 minutes without GPU use, `sift serve`
  moves them back to RAM and hands the memory back; only CUDA's own context (about 0.3 GB) stays. The first search
  afterwards takes about a second longer; if other programs have taken the memory meanwhile, sift stays on the CPU
  rather than squeeze them. Keyword search never uses the GPU.

## Searching

- **`search`:** BM25 as FTS5 computes it, over three columns weighted alike: every passage is indexed with its
  document's title (the first passage holds it as its heading too) and its front-matter values. Each document is
  represented by its best passage. Input that isn't valid FTS5 syntax is treated as exact phrases.
  - A query of words joined by AND, OR, NOT and parentheses (all of `query`'s, most of `search`'s) is answered from
    the postings by Block-Max Pruning ([Mallia et al. 2024](https://arxiv.org/abs/2405.01117)): each term's best
    score in a range of places, summed over the query's terms, bounds every chunk there, so ranges are scored best
    bound first until none left can beat the k-th best. FTS5 scores every match, about a microsecond each: "patients"
    matches 8 million PubMed passages and took 8 seconds, a question of a dozen words 15. Now most of a common word's
    postings are never read. The results are FTS5's, score for score, but where FTS5 counts a word or phrase by where
    its readers happen to stand, in nested queries alone (see `topk`). A search reads the postings in one transaction,
    as they were when it began, whatever another sift builds meanwhile.
  - A query's common words are read only where they can matter. With all words required, the rarest is read first,
    and each next one only where chunks holding all before it are left; with any of them, the words that together
    can't bring a chunk to the k-th best score ("the", "of") only for the chunks the others bring within reach
    (MaxScore). A question of 14 words over 3.2 million arXiv abstracts: 81 ms instead of 2.5 s.
  - A phrase, quoted or a word FTS5 can't read as written ("first-stage"), is bounded by its words, at most as often
    in a chunk as the rarest of them, and FTS5, which has the positions, finds the chunks that hold it, once for the
    postings as they stand, and counts it in those within reach, each once, kept with them (up to 512 MB in all): a
    question asked again, or another reaching the same chunks, costs FTS5 nothing more, until the postings change. A
    word of no token, such as a dash between words, matches nothing, as in FTS5: words side by side and OR leave it
    out.
  - Prefixes, NEAR and column filters are FTS5's to answer, as before.
  - With a time budget, a search from the postings stops once it is spent and has found any, and returns the best found
    so far, marked partial: ranges best bound first make that the anytime ranking of
    [Mackenzie et al. 2021](https://arxiv.org/abs/2104.08976), every collection's best ranges scored. Out of time, it
    looks no further down when copies left fewer than `k` works, and marks those partial too. Without, the results are
    exact.
- **Several collections:** searches ask each collection's index and merge the results, scored with the word counts of
  all of them, as one index would score them: a collection's own counts would take a word common in a small
  collection for a common word, and sink its pages. The postings add up the counts of all before scoring; through
  FTS5, each collection's best hits are scored again so; a query with NEAR or a column filter keeps each collection's
  scores.
- **`vsearch`:** every vector's code, its signs as one bit per dimension, waits in RAM: 96 bytes for 768 dimensions, a
  sixteenth of the vector, so 3.2 million passages take 0.3 GB and 40 million 4 GB. A query's code is compared with
  all of them by Hamming distance, on the CPU's cores, each counting its codes at each distance; the counts summed
  tell how near the nearest lie, and the cores pick the codes that near (46 million codes: 0.3 s). The 10,000
  nearest of all collections together, ties by id, are ranked again by the query's own floats against their bits (a
  table lookup per byte), and the best 1,000 are scored exactly with their vectors from the index. Over 3.2 million
  arXiv abstracts this finds 99.75% of exact search's top 10, and its known-item MRR, 0.9530; reading 1,000 vectors
  instead of 4,000 saves 18 ms warm
  ([benchmarks](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md#vectors-in-memory)). No index to
  build or tune, and no GPU. When the best passages come from a few documents, it looks further down until it has `k`
  documents.
  - A server follows each collection's changes without stalling a search: the codes written past the last it holds,
    as `sift embed` writes them slab by slab, are read alone, and join those held as parts merged as they grow, so
    that a code is copied a few times over a day of slabs; its own `reindex` patches them. Any other change to the
    vectors (deleted, embedded again, or written below the last code) gives the store's codes a new epoch, and the
    server reads them all again in a thread of its own, while searches go on with those held, the codes past the last
    among them, and the vectors as the store holds them. Every deletion of vectors goes through one function of the
    store, which keeps the epoch.
  - A filter on front matter is checked for the nearest candidates first, a batch at a time, until enough pass: listing
    every chunk it lets through reads every document, 6 seconds over arXiv warm and 56 cold, while a check costs 4 to
    90 microseconds. A filter that lets too few through for that, so that checking would cost more than a fifth of the
    passages, is listed instead. Either way, the candidates are the nearest the filter lets through.
- **Copies of a work:** a paper in several collections (arXiv and PubMed, say) is one hit, the best ranked, which
  names the others under `also`: copies share a DOI, compared in any case and without a link's prefix. `search`,
  `vsearch` and `similar` look further down until they have `k` works, and `similar` leaves out the document's own
  copies; `query` keeps one hit per work of the candidates it reranks.
- **Adaptive k:** asked for it, a search returns those of its `k` hits before the largest drop in score, among the
  first 90% of the gaps between neighbours, and five more: Adaptive-k
  ([Taguchi et al. 2025](https://arxiv.org/abs/2506.08479)). Over 2,000 arXiv titles, vsearch then returned 6 hits of
  10 or 50 on average, with the paper sought in 99.2 to 99.6% of the cases it was among them.
- **`query`:** the top 40 (or `k`, if larger) keyword and semantic candidates are reranked together, in one GPU
  batch. The reranker reads each candidate's title, the front-matter values that share a word with the query (an
  author's name, an id; all of them would dilute the passage), and the passage.
  - A candidate's score is the reranker's, plus half its cosine with the query, measured in units of the reranker's
    spread over the candidates. One weight thus fits any reranker, and the scores keep the reranker's scale. A
    reranker alone ranked scientific claims worse than the embedding did. Candidates whose front matter the query
    names keep the best cosine, since the embedding never sees front matter.
  - Scores are raw: a sigmoid saturates, at 0.9999 for every good match.
  - The evidence, rerankers, embedding models and blends on seven datasets, from scientific claims to long German
    documents, is in [benchmarks.md](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md).
- **`similar`:** the mean of a document's passage vectors, scored against every passage as `vsearch` scores a query.
  No model runs, so it needs no GPU.

## Links

Links are read with each Markdown note, as written, with the front-matter field and line of each, and stored with the
names each file answers to: its file name and aliases. They resolve when asked, by those names, through an index on
each, so the backlinks of a note linked by 6,000 others take 16 ms
([timings](https://github.com/tobiaswuerth/sift/blob/main/docs/benchmarks.md#links-between-notes)).

## Server and UI

- `sift serve` loads the models and the codes once and keeps them in memory, for any number of agents.
- Calls run one at a time on a worker thread, so the event loop stays free to answer, to accept connections and to
  hear that a client cancelled a call or went away. A cancelled call stops: its SQLite statements end within about a
  millisecond, through a progress handler on every connection, and it runs the models no more. `reindex` runs to its
  end. The models run on the main thread, which holds the loop: PyTorch crashes in worker threads on Python 3.15.
- The UI's pages are rendered on the server with Jinja2, whose panes [htmx](https://htmx.org) swaps in place: no build
  step and no Node, and the UI is tested with the rest. Its pages are made as the MCP tools' calls are run.
  Browsing a folder jumps through the index folder by folder, so it stays quick in collections of millions of
  documents, and the collections' sizes are the counts their indexes keep.

## Worker protocol

HTTPS with the worker's self-made certificate (EC P-256), pinned by its SHA-256 in the key, and a bearer token of 32
random bytes:

| Request | Answer |
|---|---|
| `GET /` | The worker's models, vector length and sift version |
| `POST /embed` with `{"texts": [...], "query": false}` | The vectors as raw fp16, one row per text |
| `POST /rerank` with `{"query": "...", "texts": [...]}` | The raw scores, `{"scores": [...]}` |

The worker runs one request at a time on the GPU. See [worker.md](https://github.com/tobiaswuerth/sift/blob/main/docs/worker.md)
for setting one up.
