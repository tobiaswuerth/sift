# How sift works

How sift stores what it reads and how each search ranks it. The measurements behind these choices are in
[benchmarks.md](benchmarks.md); the code's layout and rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Reading and storing

- `sift index` lists a collection's folder and compares each file by modification time and size, so unchanged files
  cost nothing. Documents, PDFs and mails are read in reader processes with a time limit (ten minutes, and a second
  more for every 100 KB), since a converter's native code can crash or hang; a file that does is left out and said,
  and read again once it changes. JSON Lines are read in the indexing process, a batch at a time, whatever the file's
  size, and only the records that changed are stored again.
- Text becomes passages of at most 3,000 characters on line boundaries, preferring headings, each overlapping the one
  before by up to 300.
- Each collection is one SQLite database (a store) in WAL mode: documents and their text, stored once (passages are
  spans of it), front matter, links, the keyword index and the vectors with their codes. A catalog lists the
  collections. Ids carry their collection in the high bits (a collection's rows start at its id times 2^40), so results
  of several collections merge as they are.
- One command writes a store at a time, held by a lock file beside it; the documents, their keywords and their links
  change in one transaction, so a search never sees one without the others.

## Keyword search

sift's own index, written in the documents' transaction. Its terms are those of SQLite's `unicode61` tokenizer with
diacritics removed, reproduced exactly as a table of every code point's separators and folds.

- A term's postings hold the passages holding it, how often, and where, compressed. The short lists of a segment lie
  many to a row, a pack of about a page with its terms and their counts at its head; a long list is kept in blocks of
  up to 4,096 passages, and for every 128 consecutive passages it keeps the term's best score there. A changed
  collection gets a new segment, and segments are merged as they accumulate.
- Nothing of the index spans its segments. In how many passages a term occurs is the sum of its segments' counts,
  less what was deleted from them since, which each segment counts until a merge drops it: writing a segment costs
  the same in a large store as in a small one.
- Building a segment and merging segments is the work of a compiled extension written in Rust (`native/`). It splits
  passages into words, sorts them into lists and packs those on all cores; Python hands it the rows of the store and
  stores what it hands back.
- Passages are numbered in a topical order once they have vectors (clusters of clusters of their codes), so a range's
  passages share their words and its best scores are tight bounds. The clustering runs in the compiled extension, on
  all cores. `sift index` builds the index of the new order beside the one searches read, a batch at a time, and puts
  it in its place in one transaction: keyword search answers throughout, and an order stopped midway is taken up by
  the next `sift index`. Eight readers, each on a connection of its own, read the passages of a batch ahead while the
  writer tokenizes and writes, as passages in a new order lie scattered over the store.
- BM25 (k1 1.2, b 0.75) over each passage with its document's title and front-matter values. A query is answered by
  Block-Max Pruning ([Mallia et al. 2024](https://arxiv.org/abs/2405.01117)): the summed best scores of a range bound
  every passage in it, so ranges are scored best bound first until none left can beat the k-th best. Words that
  together cannot reach it are read only where the others bring a passage within reach.
- Phrases are bounded by their words, then matched by their positions; a prefix is expanded from the terms of the
  packs between its bounds, counted no further than the most words a search expands. With a time budget, a search
  stops once it is spent and has found anything, returning the best so far marked `partial`.
- Over several collections, terms are weighted by the counts of all of them, as one index would.

## Vectors and `vsearch`

- The embedding model runs on the GPU (NVIDIA, else Apple) or the CPU, in batches of texts of similar length, in
  bfloat16 where the GPU has it and float32 elsewhere; vectors are stored as fp16. The default models load at a commit
  sift pins, so that what their owners push later changes no vector; a store records the model at its commit, and
  `sift embed` refuses to mix models.
- `sift embed` reads 16,384 passages at a time and embeds them longest first, so that a batch's texts are of similar
  length, writing each batch's vectors as soon as it is embedded; a thread cuts the next two batches' words while
  the model runs. On an NVIDIA GPU, with 20,000 passages or more to embed, it first compiles the model with PyTorch
  (Triton's kernels, the whole model as one graph): 20 to 60 seconds, once a process and again after the home
  process let the idle model go, then about twice the passages a second. Queries, single texts and other devices use
  the model as it is.
- Each vector's signs, one bit per dimension, are held in memory: 96 bytes for 768 dimensions. A query is compared with
  every code by Hamming distance, without an index to build or tune. The nearest 10,000 are ranked again by the
  query's own floats against their bits, and the best 1,000 are scored exactly with their vectors.
- A filter on front matter is checked for the nearest candidates first, a batch at a time, until enough pass; one that
  lets few through is listed instead.

## `query`, `similar` and results

- **`query`:** the best 40 keyword and 40 semantic candidates (or `k`, if more) are reranked together by a
  cross-encoder, which reads each candidate's title, the front-matter values sharing a word with the query (an author,
  an id), and the passage.
- **The blend:** a candidate's score is the reranker's plus half its cosine with the query, scaled to the reranker's
  spread over the candidates, so one weight fits any reranker. Candidates whose front matter the query names get the
  best cosine, since the embedding never sees front matter. A reranker alone ranks scientific text worse than the
  embedding ([why](benchmarks.md#why-query-blends-in-the-cosine)).
- **`similar`:** the mean of a document's passage vectors, scored as `vsearch` scores a query. No model runs.
- **Results:** each document once, by its best passage. Documents sharing a DOI are one hit, the best ranked, naming
  the others under `also`. Adaptive k keeps the hits before the largest drop in score among the first 90% of the
  gaps, and five more ([Taguchi et al. 2025](https://arxiv.org/abs/2506.08479)).

## Links

Links are read with each Markdown note as written, with their front-matter field and line, and stored with the names
each file answers to (file name, aliases). They resolve when asked, as Obsidian resolves them, so notes can come
and go without the notes linking to them being read again.

## The home process

- `sift serve` holds the models (loaded at the first search that needs them, moved off the GPU after 5 idle minutes)
  and the vectors' codes, and answers MCP, the UI and the CLI's commands. Searches run side by side, and beside an
  index or an embed, whose transactions they read past; only removing or moving a collection waits for the searches
  reading its store.
- It keeps the embeddings of the last 256 questions and the results of the last 32 searches until anything is written.
- The UI is rendered on the server, page by page, and runs no scripts.
