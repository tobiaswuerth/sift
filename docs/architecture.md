# How sift works

How sift indexes, stores, embeds and searches, and why. The measurements behind these choices are in
[benchmarks.md](benchmarks.md); the code's layout is in [CONTRIBUTING.md](CONTRIBUTING.md#code-layout).

## Indexing

- Worker processes read and parse files in parallel while the previous batch is written; small updates skip the
  process start-up. Each kind of file has a reader, chosen by extension; adding a format means adding a reader.
- Files are compared by modification time and size, so unchanged files cost nothing, and documents by content hash, so
  a changed JSON Lines file re-indexes only its changed records.
- Markdown, and everything converted to it, is split into chunks of about 3,000 characters (about 750 tokens) on line
  boundaries, preferring headings, with a small overlap.
- A JSON Lines record or an attachment is a document below its file's name, a path a real file may have too. The first
  keeps it, and the store records that the other waits for it (table `waiting`, made by the first such document). Every
  deletion of documents goes through one function of the store, which marks the files waiting for the paths it frees,
  so the running `sift index` reads them.

## Storage

- One SQLite database in WAL mode per collection holds its documents' text and metadata, the FTS5 keyword index, the
  keyword postings, the vectors (fp16) and their codes (one bit per dimension); a small catalog lists the collections.
  The write-ahead log is truncated to 64 MB after each checkpoint, since SQLite otherwise keeps it as large as it grew
  while any connection is open.
- The text is stored once: chunks are spans of it, and a view cuts them out for FTS5. SQLite's `substr` loads the
  whole text and walks to the position for every chunk it cuts, so sift reads the view only for documents of up to
  eight chunks, all in one statement, and cuts the chunks of longer ones itself, from a text read once per document:
  the keyword entries of an 80 MB file's 32,795 chunks take 3 s to write, against 11 minutes through the view.
  Embedding, search results and phrase counts cut their chunks the same way; a test holds the two cuts equal.
- A collection's row ids start at its id times 2^40, so results of several collections merge as they are.
- Counts of documents, chunks and vectors are kept by triggers in each change's transaction, so `sift status` and a
  starting server read no table whole.
- A collection whose store is damaged is passed over by `sift index`, `sift status` and the server's start, with how to
  forget it; a search reading its damaged tables fails.
- Searches and `get` never read the files, so results and documents agree until the next `sift index`.

### Keyword postings

FTS5 scores every match, about a microsecond each, so a common word over tens of millions of passages takes seconds.
sift therefore ranks from postings of its own, read out of FTS5's index, which stays the reference:

- Per term: its chunks and how often it occurs in each, zlib-compressed in blocks of 4,096 postings. Before them come
  its ranges: for every 128 consecutive places holding the term, its count there and its best score there, rounded up
  to a byte. About a third of FTS5's size, which holds every word's positions.
- Chunks are numbered in a topical order once they have vectors: clusters of clusters of their codes (binary k-means
  by Hamming distance), each placed next to its nearest. A range's chunks then share their words, so its score bounds
  are tight. A changed chunk takes the next free place; `sift index` orders anew once a fifth of the chunks got
  vectors since.
- The postings follow FTS5: each chunk whose keyword entries change is marked in the same transaction, and `sift
  index` (or `reindex`) adds a delta of those chunks and masks their old postings, as Lucene's segments do. Deltas are
  merged; past 5% of the chunks changed, the postings are built anew, in worker processes for large collections. Until
  they are up to date, the collection's keyword searches read FTS5.

## Embedding

- PyTorch on the GPU (NVIDIA, else Apple), in batches of texts of similar length in tokens: a batch is padded to its
  longest text, and characters predict tokens poorly.
- bfloat16 on NVIDIA GPUs that have it, float32 elsewhere, never fp16, where EmbeddingGemma overflows into NaN vectors
  now and then. Vectors are stored in fp16; broken ones never are.
- `sift embed` keeps the GPU busy: one thread reads the next slabs of 20,000 chunks and sorts them by length, another
  writes the vectors and commits each slab, and the main thread runs the model, tokenizing each batch while the GPU
  embeds the one before. PyTorch stays on the main thread, as it crashes in others on Python 3.15.
- Beside an NVIDIA GPU, PyTorch keeps one CPU thread: more only waited busily after each batch.

### GPU memory

- sift shares the GPU with other programs. On NVIDIA GPUs it caps itself at the memory free when it starts, minus
  headroom (10% of the GPU, at least 1 GB), asking NVML as well as CUDA, since on Windows CUDA counts memory other
  programs hold as free.
- It measures the memory batches of the model's longest input take at growing sizes, as PyTorch's allocator reserves
  it, and uses the largest that fills 90% of the budget; a batch that still doesn't fit is halved. With less than 2 GB
  to spare, sift uses the CPU.
- The models wait in RAM and move to the GPU when needed. After 5 minutes without GPU use, `sift serve` moves them back
  and frees the memory; the next search takes about a second longer, and stays on the CPU if other programs took the
  memory meanwhile. Keyword search never uses the GPU.

## Searching

### `search`

- BM25 as FTS5 computes it, over each passage, its document's title and its front-matter values, weighted alike. A
  document is represented by its best passage.
- Queries of words, AND, OR, NOT and parentheses are answered from the postings by Block-Max Pruning
  ([Mallia et al. 2024](https://arxiv.org/abs/2405.01117)): each term's best score in a range, summed over the query's
  terms, bounds every chunk there, so ranges are scored best bound first until none left can beat the k-th best. The
  results are FTS5's, score for score (nested queries aside).
- Common words are read only where they can matter: with all words required, the rarest first and each next only
  where chunks holding all before it remain; with any of them, the words that together can't reach the k-th best score
  only for chunks the others bring within reach (MaxScore).
- Phrases (quoted, or a word FTS5 splits, such as "first-stage") are bounded by their words; FTS5, which has the
  positions, counts them in the chunks within reach, and the counts are kept (up to 512 MB) until the postings change.
  Prefixes, NEAR and column filters go to FTS5.
- With a time budget, a search stops once it is spent and has found anything, returning the best so far marked
  `partial`: scoring ranges best bound first makes this an anytime ranking
  ([Mackenzie et al. 2021](https://arxiv.org/abs/2104.08976)).
- A search reads the postings in one transaction, as they were when it began.
- Over several collections, terms are weighted by the counts of all of them, as one index would; a collection's own
  counts would sink a small collection's matches of a word common there.

### `vsearch`

- Each vector's code, its signs as one bit per dimension, waits in RAM: 96 bytes for 768 dimensions, a sixteenth of the
  vector, 4 GB for 40 million passages. No index to build or tune, and no GPU.
- A query's code is compared with all codes by Hamming distance on every core. `sift serve` and `sift bench` scan with
  code compiled by [Numba](https://numba.pydata.org) as they start (kept in `SIFT_HOME/numba`), reading each code once;
  a single command-line search, a collection under 65,536 passages, or a server without Numba uses NumPy.
- The 10,000 nearest of all collections (ties by id) are ranked again by the query's own floats against their bits, a
  table lookup per byte, and the best 1,000 are scored exactly with their vectors.
- A filter on front matter is checked for the nearest candidates first, a batch at a time, until enough pass; a filter
  that lets so few through that checking would cost more than a fifth of the passages is listed instead.
- When the best passages come from few documents, it looks further down until it has `k` documents.
- A server follows changes without stalling searches: codes `sift embed` appends are read alone and merged into those
  held; any other change to the vectors gives the store's codes a new epoch, and the server reads them all again in a
  thread while searches go on with those held.

### `query`, `similar` and results

- **`query`:** the top 40 (or `k`) keyword and semantic candidates are reranked together, sorted by length, in batches
  of 16. The reranker reads each candidate's title, the front-matter values sharing a word with the query (an author,
  an id), and the passage.
- **The blend:** a candidate's score is the reranker's plus half its cosine with the query, scaled to the reranker's
  spread over the candidates, so one weight fits any reranker and scores keep its scale. Candidates whose front matter
  the query names keep the best cosine, since the embedding never sees front matter. A reranker alone ranks scientific
  text worse than the embedding ([why](benchmarks.md#why-query-blends-in-the-cosine)). Scores stay raw: a sigmoid
  saturates near 1 for every good match.
- **`similar`:** the mean of a document's passage vectors, scored as `vsearch` scores a query. No model runs.
- **Copies of a work:** documents sharing a DOI (compared in any case, without a link's prefix) are one hit, the best
  ranked, listing the others under `also`; searches look further down until they have `k` works.
- **Adaptive k:** the hits before the largest drop in score, among the first 90% of the gaps, and five more
  ([Taguchi et al. 2025](https://arxiv.org/abs/2506.08479)).

## Links

Links are read with each Markdown note, as written, with their front-matter field and line, and stored with the names
each file answers to (file name, aliases). They resolve when asked, through an index on those names, so notes can come
and go without the notes linking to them being read again.

## Server and UI

- `sift serve` loads the models and codes once, for any number of agents. It keeps the embeddings of the last 256
  questions and the results of the last 32 searches, valid while nothing they read changed: SQLite's `data_version` and
  `total_changes` of the catalog and the stores, and the codes held. Searches with a time budget, of more than 1,000
  hits, or that passed over an unreadable collection are not kept.
- Calls run one at a time on a worker thread, so the event loop stays free to accept connections and hear
  cancellations. A cancelled call stops within about a millisecond (a progress handler on every SQLite connection) and
  runs the models no more; `reindex` runs to its end. Local models run on the main thread (PyTorch); a worker's
  (`SIFT_EMBED_REMOTE`) are asked in a thread of their own, one request at a time, so a cancelled call stops waiting.
- The UI is rendered on the server with Jinja2, its panes swapped in place by [htmx](https://htmx.org): no build step,
  tested with the rest. Browsing jumps through the index folder by folder, so it stays quick over millions of
  documents.

## Worker protocol

HTTPS with the worker's self-made certificate (EC P-256), pinned by its SHA-256 in the key, and a bearer token of 32
random bytes. The worker runs one request at a time on its GPU.

| Request | Answer |
|---|---|
| `GET /` | The worker's models, vector length and sift version |
| `POST /embed` with `{"texts": [...], "query": false}` | The vectors as raw fp16, one row per text |
| `POST /rerank` with `{"query": "...", "texts": [...]}` | The raw scores, `{"scores": [...]}` |
