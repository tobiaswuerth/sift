# Benchmarks

This page explains how sift's search defaults were chosen. Every change to how `query` ranks is measured on seven
datasets, with several rerankers and embedding models. The default is the setting that does best overall, not the
winner on a single dataset. To measure sift on your own documents, use
[`sift bench`](usage.md#measuring-search-on-your-own-documents).

## Results in brief

- **Reranker:** `query` uses
  [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) (150M parameters,
  Apache-2.0) and blends each candidate's cosine similarity into its score. Over the seven datasets
  the mean score is 0.696, against 0.646 for Ettin-150M alone, the default up to 0.8.0. Size and speed are the same.
- **Embedding model:** [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) remains the default.
  Snowflake Arctic-Embed-L v2.0 ties it on average: it does better on financial questions and long German documents,
  worse on scientific text, and indexes 1.6 times as fast.
- **Speed:** reranking takes about 0.3 seconds per query on an RTX 4090 and about 25 seconds on a CPU. Machines without
  a GPU should rerank through a [`sift worker`](worker.md).
- **Vectors in memory:** `vsearch` keeps one bit per dimension in RAM, ranks the 10,000 nearest by Hamming distance
  again by the query's own floats, and scores the best 1,000 exactly. Over 3.2 million arXiv abstracts it finds
  99.75% of exact search's top 10, and its MRR of 0.9530, in a sixteenth of the memory
  ([details](#vectors-in-memory)).
- **Keyword search:** `search` ranks from postings of its own, scoring ranges of passages best bound first. Over 29.9
  million PubMed passages, a common word takes 7 ms instead of 1.3 seconds, a question 26 ms instead of 5.3 seconds
  (medians, the postings read before; 67 and 200 ms read anew), with FTS5's results, score for score, in about a
  third of FTS5's storage ([details](#keyword-search)).
- **Embedding:** `sift embed` keeps the GPU 99% busy: 472 PubMed passages per second on an RTX 4090, against 382 for
  sift 1.10.0, the vectors the same ([details](#keeping-the-gpu-busy)).

## Datasets

| Dataset | Kind | Documents | Queries | Metric |
|---|---|---|---|---|
| [SciFact](https://github.com/beir-cellar/beir) (BEIR) | Scientific claims against abstracts | 5,183 | 300 | nDCG@10 |
| [NFCorpus](https://github.com/beir-cellar/beir) (BEIR) | Medical questions, graded relevance | 3,633 | 323 | nDCG@10 |
| [FiQA](https://github.com/beir-cellar/beir) (BEIR) | Financial questions against forum answers | 57,638 | 648 | nDCG@10 |
| [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad) | German questions against passages | 474 | 500 of 2,204, sampled with seed 0 | nDCG@10 |
| [MLDR](https://huggingface.co/datasets/Shitao/MLDR), German | Long German documents | 10,000 | 200 | nDCG@10 |
| [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) | Literature-search questions over ML and NLP papers | 64,183 | 597 | nDCG@10 |
| arXiv known items | One paper to find, named by author, id, paraphrase or acronym | 3,215,242 | 49 | MRR@10 |

- **arXiv known items** is a set written for sift. It searches the titles and abstracts of all of arXiv (the Kaggle
  snapshot). 12 queries name front matter (an author's name, an arXiv id), and 37 describe a topic (a paraphrase, an
  acronym). It isn't distributed with sift.
- **LitSearch** is reported here by nDCG@10, like the others. Its paper reports Recall@5 and Recall@20.

## Method

- **Indexing.** Each dataset becomes a sift collection, as JSON Lines with `id`, `title` and `text`. `sift index` and
  `sift embed` index it as they would any other.
- **Candidate pools.** For every query, the candidates `query` reranks are collected once per embedding model: the best
  passage of each of the 40 best documents by keyword (BM25), and of the 40 best by cosine. Every reranker scores the
  same pools, so differences between rerankers come from the rerankers alone.
- **Summary measures.** Each setting gets the mean over the seven datasets, plus its *worst gap*: its largest
  shortfall on any dataset against the best setting measured there. A small worst gap means no kind of document is
  left behind.
- **Hardware.** An NVIDIA RTX 4090, with the models in bfloat16, as sift runs them.
- **Sanity check.** sift's keyword search scores 0.666 on SciFact, matching BEIR's published BM25 baseline (0.665).

## Why `query` blends in the cosine

A reranker alone ranks scientific text worse than the embedding does. With Ettin-150M, `query` scored 0.760 on
SciFact, against 0.840 for the cosine alone (`vsearch`). Rerankers trained on web search lose precision on scientific
text ([SciRet](https://arxiv.org/abs/2608.03860)). Yet on the arXiv known items the reranker is essential: MRR 0.936,
against 0.627 for the cosine alone. Author and id queries depend on it, because only the reranker reads the
front-matter values a query names.

Combining the two keeps both strengths. The forms tried, with Ettin-150M:

| Order | SciFact | NFCorpus | arXiv | arXiv, front matter | arXiv, topical |
|---|---|---|---|---|---|
| Reranker alone | 0.760 | 0.374 | 0.936 | 0.889 | 0.951 |
| Reranker + 20 × cosine, front-matter matches exempt | 0.811 | **0.394** | **0.956** | **0.903** | **0.973** |
| Reranker + 40 × cosine, front-matter matches exempt | 0.840 | 0.393 | 0.940 | **0.903** | 0.953 |
| Reranker + 80 × cosine, front-matter matches exempt | **0.856** | 0.389 | 0.914 | **0.903** | 0.918 |
| Reranker + 20 × cosine, no exemption | 0.811 | **0.394** | 0.920 | 0.757 | **0.973** |
| Mean of the two z-scores | 0.834 | – | – | 0.675 | 0.953 |

- **The exemption.** A candidate whose front matter matches a word of the query keeps the pool's best cosine, since the
  embedding never sees front matter. Without the exemption, author and id queries fall from 0.889 to 0.757.
- **Scale-free weight.** A fixed weight such as 20 only suits rerankers whose scores span the same range, and they
  don't. sift therefore scales the cosine to the reranker's spread over the query's candidates:

  score = rerank + w · (σ<sub>rerank</sub> / σ<sub>cosine</sub>) · (cosine − mean cosine)

  Here σ is the standard deviation over the candidates. With w = 0.5, the best or tied-best weight under every
  reranker and embedding model below, the scores keep the reranker's scale. With gte, close matches score about 2.5
  to 4, and unrelated results below 2.

## Rerankers

All rerankers score the same EmbeddingGemma-2 pools; w is the blend weight.

| Reranker | Size | w | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv | Mean | Worst gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **gte-reranker-modernbert-base** | 150M | 0.5 | 0.813 | 0.361 | 0.464 | 0.987 | 0.685 | 0.631 | 0.929 | **0.696** | 0.046 |
| gte-reranker-modernbert-base | 150M | 0.75 | 0.834 | 0.368 | 0.459 | 0.989 | 0.687 | 0.625 | 0.915 | 0.697 | 0.039 |
| mxbai-rerank-base-v2 | 494M | 0.75 | 0.859 | 0.404 | 0.441 | 0.985 | 0.624 | 0.602 | 0.913 | 0.690 | 0.091 |
| Ettin-400M | 400M | 0.75 | 0.838 | 0.399 | 0.458 | 0.979 | 0.572 | 0.627 | 0.942 | 0.688 | 0.143 |
| bge-reranker-v2-m3 | 568M | 0.75 | 0.829 | 0.377 | 0.432 | 0.992 | 0.711 | 0.592 | 0.849 | 0.683 | 0.102 |
| Ettin-150M | 149M | 0.75 | 0.822 | 0.396 | 0.452 | 0.976 | 0.551 | 0.611 | 0.946 | 0.679 | 0.164 |
| Qwen3-Reranker-0.6B | 596M | 0.75 | 0.835 | 0.403 | 0.419 | 0.983 | 0.602 | 0.613 | 0.859 | 0.674 | 0.113 |
| mxbai-rerank-large-v1 | 435M | 0.75 | 0.830 | 0.406 | 0.437 | 0.985 | 0.447 | 0.604 | 0.870 | 0.654 | 0.268 |
| Ettin-150M, no blend (up to 0.8.0) | 149M | 0 | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.936 | 0.646 | 0.235 |

- **gte-reranker-modernbert-base, blended,** is the smallest of the leaders and the most balanced. It's never more
  than 0.05 below the best setting on any dataset. Against Ettin-150M alone it gains 0.05 on the mean and 0.20 on long
  German documents, and loses 0.007 on the arXiv known items. Weights 0.5 and 0.75 tie on the mean. sift uses 0.5,
  which keeps more of the reranker's lead on the known items.
- **mxbai-rerank-base-v2** leads on scientific and medical text, at three times the size and twice the time per query.
- **The multilingual rerankers** (bge-reranker-v2-m3, Qwen3-Reranker-0.6B) lead on German but fall behind on arXiv and
  FiQA.
- **The blend helps every reranker:** their means rise by 0.03 to 0.05.

## Embedding models

Re-embedding all of arXiv once per model would take hours, so here the arXiv known items search a 100,000-paper subset
that includes every target.

Each embedding model alone, ranking the candidates by cosine as `vsearch` does:

| Embedding model | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
|---|---|---|---|---|---|---|---|---|
| EmbeddingGemma-2 | **0.840** | **0.358** | 0.314 | 0.955 | 0.534 | 0.486 | 0.806 | 0.613 |
| Snowflake Arctic-Embed-L v2.0 | 0.704 | 0.355 | 0.451 | 0.964 | 0.539 | **0.499** | **0.843** | **0.622** |
| bge-m3 | 0.644 | 0.313 | 0.412 | **0.969** | **0.552** | 0.436 | 0.818 | 0.592 |
| Qwen3-Embedding-0.6B | 0.700 | 0.355 | **0.469** | 0.950 | 0.531 | 0.174 | 0.819 | 0.571 |

With `query`, that is gte-reranker-modernbert-base and the blend at w = 0.5 over each model's candidates:

| Embedding model | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
|---|---|---|---|---|---|---|---|---|
| EmbeddingGemma-2 | **0.813** | **0.361** | 0.464 | 0.987 | 0.685 | **0.631** | **0.960** | **0.700** |
| Snowflake Arctic-Embed-L v2.0 | 0.756 | 0.358 | 0.505 | 0.986 | **0.700** | 0.624 | 0.959 | 0.698 |
| bge-m3 | 0.753 | 0.340 | 0.486 | 0.986 | 0.695 | 0.601 | 0.948 | 0.687 |
| Qwen3-Embedding-0.6B | 0.762 | 0.360 | **0.518** | **0.988** | 0.695 | 0.417 | 0.959 | 0.671 |
| *Ettin-150M alone over EmbeddingGemma-2 (up to 0.8.0)* | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.976 | 0.651 |

- **No embedding model wins everywhere.** EmbeddingGemma-2 is far ahead on scientific claims and weak on financial
  questions.
- **The reranker narrows the differences.** Alone, the models' means span 0.571 to 0.622; under `query`, 0.671 to
  0.700.
- **The blend carries each model's strengths through.** Under Ettin-150M alone, SciFact scores 0.76 whatever the
  embedding model. With the blend, EmbeddingGemma-2 keeps its lead on scientific claims (+0.06). Arctic keeps its lead
  on financial questions (+0.04) and long German documents (+0.015).
- **gte-reranker-modernbert-base with the blend is the best reranker under every embedding model.** Its worst gap is
  0.02 to 0.04.
- **EmbeddingGemma-2 stays the default.** Arctic ties it on average, and changing the default would force every
  existing index to be embedded again. For a new index of business documents, Arctic is a good choice
  (`SIFT_EMBED_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0`).
- **Qwen3-Embedding's LitSearch score is out of line** with its other results, and is unexplained. It isn't held
  against the model.

## Speed

Each model ran alone on an otherwise idle machine, after a warm-up. The rerankers scored 50 queries × 80 candidates,
half from SciFact (short abstracts) and half from MLDR (long German documents). On the GPU the models ran in bfloat16,
5 repetitions, with a spread of at most 1%. On the CPU they ran in float32 on 8 threads of an Intel Core i9-14900K,
over a tenth of the inputs, 3 repetitions, with a spread of at most 10%.

| Reranker | Size | RTX 4090, ms per query | CPU, ms per query |
|---|---|---|---|
| **gte-reranker-modernbert-base** | 150M | **284** | 26,596 |
| Ettin-150M | 149M | 284 | 25,217 |
| bge-reranker-v2-m3 | 568M | 305 | 41,511 |
| mxbai-rerank-base-v2 | 494M | 559 | 56,657 |
| Ettin-400M | 400M | 608 | not measured |
| Qwen3-Reranker-0.6B | 596M | 1,025 | 90,160 |

The embedding models embedded 2,000 passages (half SciFact, half MLDR) and 100 queries, on the RTX 4090:

| Embedding model | Passages per second | ms per query |
|---|---|---|
| EmbeddingGemma-2 | 120 | 51 |
| Snowflake Arctic-Embed-L v2.0 | 190 | 13 |
| bge-m3 | 190 | 16 |
| Qwen3-Embedding-0.6B | 68 | 36 |

- **gte-reranker-modernbert-base costs nothing in speed:** it's as fast as Ettin-150M on the GPU, and within 6% on the
  CPU.
- **On a CPU, reranking takes about 25 seconds per query.** A machine without a GPU should use a `sift worker`, or
  `search` and `vsearch`, which need no reranker.
- **Query embedding is minor next to reranking:** 51 ms against 284 ms.

### Batching

`sift embed` on the RTX 4090 with EmbeddingGemma-2, over 6,000 PubMed abstract passages (339 tokens on average) and
their 6,000 titles (30 tokens); the mean of 2 runs, and for the titles the best of 2:

| Batches | Passages per second | CPU cores busy |
|---|---|---|
| sift 1.8: grouped by characters, a CPU thread per core | 171 | 4.7 |
| sift 1.9: grouped by tokens, one CPU thread | **230** | 0.9 |

| Titles, batch of | Grouped by characters | Grouped by tokens |
|---|---|---|
| 140 | 2,295 per second, padded to 1.50 times their tokens | **2,504**, 1.04 times |
| 1,024 | 1,742, 1.78 times | 2,424, 1.30 times |

- **Grouping by tokens is what pays:** a batch is padded to its longest text, and characters tell tokens poorly.
  Grouped by characters, the passages were padded to 1.51 times their tokens; by tokens, to 1.04 times.
- **Larger batches of short texts don't:** a batch of 140 already gives the GPU all it can compute at once, so 1,024
  only adds padding.
  sift therefore sizes one batch for the longest input the model takes, and uses it for every length.
- **One CPU thread beside the GPU** made embedding 5% faster and freed most of the cores: PyTorch's other threads only
  waited busily after each batch.

### Keeping the GPU busy

`sift embed` on the RTX 4090 with EmbeddingGemma-2, over 40,114 PubMed passages (the first 40,000 records of a source
file), from no vectors; the GPU sampled every 100 ms, past loading and calibration:

| | Passages per second | GPU busy |
|---|---|---|
| sift 1.10.0 | 382 | 83% |
| sift 1.10.1 | **472** | 99% |

- **Where the time went,** profiling a running 1.10.0 over PubMed: the model 72%; tokenizing each slab's 20,000
  texts to sort them by length 12%, the GPU idle (12 seconds every 75); tokenizing each batch 11%, idle again;
  reading and writing the index 2%. Now a thread reads and sorts the next slabs, another writes, and each batch is
  tokenized while the GPU embeds the one before. The vectors are the same, bit for bit.
- **Batch size:** calibration measured the memory the tensors take, but PyTorch's cap counts what its allocator
  reserves, about a fifth more. 140 texts of 1,024 tokens take 18.0 of a 20.1 GB cap, and ran out of memory at the
  first batch of full length, so every run on PubMed went on at 70. Counted as reserved, calibration picks 112 to 116,
  which fit; 112 embeds the passages above as fast as 140 did (they are at most 930 tokens, where 140 fits).

### Keyword search

63 queries: 18 single words (from "crispr" to "the"), 20 of all their words ("graph neural network", ids), and 25
questions as `query` asks them, any of their words ("effect of approximate or unsafe dynamic pruning in first-stage
retrieval on effectiveness after neural reranking"). The best 10 documents, each by its best passage, on the RTX 4090
machine (i9-14900K, NVMe); FTS5 read warm, as sift 1.9 did without its join to the documents, which only made it
slower (196 seconds cold for "patients"). "Cold": the postings read anew, as by a `sift search`; "warm": read before,
as by `sift serve`. Every query found the same documents as FTS5, score for score, on both.

| Median | FTS5 | Postings, cold | Postings, warm |
|---|---|---|---|
| arXiv (3.2 million passages), single words | 120 ms | 10 ms | 0.7 ms |
| arXiv, all words | 20 ms | 21 ms | 2 ms |
| arXiv, questions | 510 ms | 115 ms | 13 ms |
| PubMed (29.9 million passages), single words | 1.33 s | 67 ms | 7 ms |
| PubMed, all words | 30 ms | 86 ms | 7 ms |
| PubMed, questions | 5.3 s | 200 ms | 26 ms |

- **FTS5 scores every passage a word occurs in:** about a microsecond each. "the" occurs in 29.6 million PubMed
  passages: 25 seconds. The postings bound each range of 128 passages and score the best ranges first: for "the", a
  few hundred postings.
- **The slowest question** took 16 seconds in FTS5, 2.7 seconds cold and 1.1 warm from the postings: a dozen common
  words bound most ranges high, so pruning skips less. Smaller ranges would bound them tighter, at more storage.
- **All words of a few rare ones** are the one kind FTS5 answered faster, against the postings read cold (30 ms
  against 86 on PubMed): it intersects the lists as it reads them. Read before, the postings take 7 ms. sift answers
  them from the postings too.
- **Storage:** 0.94 GB for arXiv's 431 million postings against 2.71 GB of FTS5's index, which holds every word's
  positions: 2.2 bytes per posting. 12.2 GB for PubMed.
- **Ranges of 128 or 64 passages:** on PubMed, with the postings in memory, 128 answered single words in 2.2 ms against
  5.8 and questions in 16 against 23 (median), with the longest question slower (0.53 s against 0.24); 128 also takes
  8% less storage.
- **Building** reads FTS5's vocabulary in worker processes, a range of terms each: 69 seconds for arXiv on 24, 13
  minutes for PubMed on 31 (beside a running `sift embed`), where "the" alone, read by one worker, takes minutes.
- **Passages in topical order** (sift 1.11): numbered by clusters of their codes, a range's passages share their
  words. In a prototype of the postings over arXiv, read warm, questions took 3.7 ms instead of 5.2 (median) and the
  slowest 45 instead of 90, scoring a half to a twentieth as many ranges ("graph neural networks for molecular
  property prediction": 197 instead of 3,821), with the same results. Single words took 0.7 ms instead of 0.3: a
  word's best passages lie in more ranges. The blocks took 0.61 GB instead of 0.66, the order 13 MB more. Ordering
  3.2 million codes takes about a minute.
- **Common words read only where they matter** (sift 1.11.1): with all words required, the rarest read first and the
  next only where chunks holding all before it are left; with any of them, the words that together can't reach the
  k-th best score ("the", "of") read only for the chunks within reach (MaxScore). Over arXiv, warm, with the same
  results:

  | Query | 1.11.0 | 1.11.1 |
  |---|---|---|
  | seven words, all required | 65 ms | 7 ms |
  | questions of 10 to 14 words | 0.44 to 2.5 s | 34 to 81 ms |
  | `query`'s keyword candidates (any of the words) | 22 to 230 ms | 9 to 62 ms |
  | a question with a hyphenated word ("first-stage"), through FTS5 before | 8.7 s | 0.43 s |
  | quoted phrases, through FTS5 before | 10 to 97 ms | 3 to 16 ms |

  On the live index of six collections (46 million passages), the 14-word question had taken 13.5 s and the
  hyphenated one over ten minutes; with 1.11.1, 0.97 s and 8.7 s at first, 0.32 s and 2.2 s asked again.
- **A phrase counted once per passage** (sift 1.11.2), kept with the passages that hold it: of a hyphenated question's
  2.2 s, 62% had been FTS5 counting the phrase in the same passages again. Over arXiv, warm, six queries with phrases
  measured before and after, with the same results: the hyphenated question 93 ms instead of 429, "in this paper we
  propose" 131 ms instead of 2.6 s, the four others 1 to 7 ms instead of 3 to 14. A new question's first call costs
  what it did.
- **A time budget** ends a search from the postings early, with the best found so far: ranges are scored best bound
  first, so the best results tend to come first ([Mackenzie et al. 2021](https://arxiv.org/abs/2104.08976)). Not
  measured on these queries yet.

### Links between notes

A synthetic vault of 20,000 notes and 220,000 links: one note linked by 6,000 others, a tenth of the links to notes
not written yet. On the same machine, best of 3:

| Operation | Time |
|---|---|
| `sift index` (the notes read in 20 processes) | 6.9 s |
| `links` out of a note; both ways, two steps | 0.1 ms; 1.5 ms |
| `links` into the most linked note, 50; 10,000 | 16 ms; 42 ms |
| `unresolved` over the vault | 210 ms |
| `search` with `links_to` the most linked note (without it) | 25 ms (19 ms) |
| `mentions` of the most linked note | 20 ms |
| Upgrading an index of sift 1.0 that holds these notes | 3.0 s |

- **Notes without links read as fast as before:** 20,000 of them in 0.617 s before reading links, 0.626 s after.
- **An index without notes upgrades in a moment:** 3.2 million arXiv abstracts in JSON Lines took under a second,
  and searched as before, result for result.

## Vectors in memory

Exact search over fp16 vectors holds 1.5 KB per 768-dimensional passage: 4.9 GB for 3.2 million arXiv abstracts, 60 GB
for 40 million passages. Each scheme below was compared with exact search over the 3,215,416 arXiv abstract passages
(EmbeddingGemma-2), for 2,000 known-item title queries: recall@k is the share of exact search's top k it finds, MRR@10
and hit@10 are those of each paper's own first passage. "+ exact top N" scores the N nearest by the scheme exactly.

| Scheme | Bytes per vector | Recall@10 | Recall@40 | MRR@10 | Hit@10 |
|---|---|---|---|---|---|
| Exact (fp16) | 1,536 | 1.000 | 1.000 | 0.9530 | 0.9905 |
| int8, a scale per vector | 772 | 0.965 | 0.968 | 0.9527 | 0.9905 |
| First 512 dimensions | 1,024 | 0.852 | 0.848 | 0.9483 | 0.9885 |
| First 256 dimensions | 512 | 0.686 | 0.671 | 0.9280 | 0.9790 |
| Binary, Hamming distance | 96 | 0.454 | 0.420 | 0.8936 | 0.9680 |
| Binary, Hamming + exact top 100 | 96 | 0.839 | 0.614 | 0.9513 | 0.9865 |
| Binary, Hamming + exact top 400 | 96 | 0.949 | 0.847 | 0.9525 | 0.9900 |
| Binary, Hamming + exact top 1,000 | 96 | 0.980 | 0.933 | 0.9528 | 0.9905 |
| **Binary, Hamming + exact top 4,000** | **96** | **0.996** | **0.986** | **0.9528** | **0.9905** |
| Binary, float query + exact top 400 | 96 | 0.992 | 0.955 | 0.9530 | 0.9905 |
| Binary, float query + exact top 1,000 | 96 | 0.998 | 0.988 | 0.9528 | 0.9905 |

- **One bit per dimension, then exact scores, keeps the quality** at a sixteenth of the memory. Truncating to fewer
  dimensions (EmbeddingGemma is trained for it) loses more than binary codes rescored.
- **Hamming distance with 4,000 rescored** comes close to the float query's ranking (the query's values against the
  codes' signs) with 1,000. Hamming distance is 12 bit counts per code instead of 768 multiply-adds, and the 4,000
  vectors read from SQLite cost little next to a scan of millions.
- **Both, one after the other** (sift 1.11): Hamming distance over all codes, the float query over the nearest N
  (a table lookup per byte of a code: 3 ms for 10,000), exact scores for its best M. Reading the vectors is most of
  a search's time: 25 ms for 4,000 from the arXiv index, warm, 7 ms for 1,000.

| Hamming nearest N | Float query's best M, scored exactly | Recall@10 | MRR@10 | Hit@10 |
|---|---|---|---|---|
| 4,000 | (all 4,000, sift 1.10) | 0.9961 | 0.9530 | 0.9905 |
| 1,000 | (all 1,000) | 0.9794 | 0.9530 | 0.9905 |
| 10,000 | 400 | 0.9921 | 0.9530 | 0.9905 |
| **10,000** | **1,000** | **0.9975** | **0.9530** | **0.9905** |
| 10,000 | 2,000 | 0.9986 | 0.9530 | 0.9905 |
| 20,000 | 1,000 | 0.9978 | 0.9530 | 0.9905 |
| 40,000 | 1,000 | 0.9979 | 0.9530 | 0.9905 |

- **Filters** on front matter: listing every passage a filter lets through reads every document. Over arXiv, a
  category took 6.2 seconds warm (56 cold), whether it let 9% through (cs.LG) or 0.9% (cs.IR). Checking candidates
  one by one costs 4 to 5 microseconds each warm, 90 cold, so sift checks the nearest first, until enough pass, and
  lists only when checking would cost more: when it would check more than a fifth of the passages. Until 10,000
  candidates passed, cs.LG checked 130,000 in 0.6 seconds, cs.CV (6%) 193,000 in 0.85; cs.IR gave up after 10,000
  checks (0.02 to 0.08 seconds) and was listed.
- **Adaptive k:** a search asked to adapt returns its hits up to the largest drop in score, among the first 90% of
  the gaps, and five more ([Taguchi et al. 2025](https://arxiv.org/abs/2506.08479)). For the 2,000 titles, of exact
  search's top 10, 20 or 50:

| Hits kept after the largest drop | Hits returned, on average | The paper sought kept, when among them |
|---|---|---|
| 0 | 1.3 | 96.9 to 97.2% |
| 2 | 3.3 | 98.5 to 98.9% |
| **5** | **6.2 to 6.3** | **99.2 to 99.6%** |

  Known items stand out: most queries had their largest drop after the first hit. Questions with many answers need
  more, so sift keeps the paper's five.
- **Centering before the signs** (not adopted): codes of v − mean against q − mean find more of exact search's top
  10 by Hamming distance alone, 0.9917 instead of 0.9794 among the 1,000 nearest. But ranking the nearest by the
  query's floats against such bits loses it: the centered vectors are no longer of one length. The whole cascade
  found 0.9788 instead of 0.9975; a correction per vector (as RaBitQ stores) might win back about 0.0015, against a
  new format and new codes for every vector.
- **Scanning the nearest clusters only** (not adopted): the codes grouped by sift's own clustering of them (the
  postings' topical order, 4,096 leaves), and only the leaves nearest the query scanned, as an inverted file does.
  Over the 2,000 titles, of exact search's top 10, with the cascade among the codes scanned:

  | Codes scanned | Leaves by Hamming distance | Leaves by the query's floats against their bits |
  |---|---|---|
  | 10% | 0.9822 | 0.9882 |
  | 20% | 0.9928 | 0.9952 |
  | 30% | 0.9957 | 0.9966 |
  | all (sift) | 0.9975 | 0.9975 |

  Scanning a third of the codes would save at most two thirds of the scan, 0.3 of the 0.5 s it took live over 46
  million passages, at a loss of recall, and need clusters kept in the index, the codes embedded since apart, and
  filters checked otherwise. An exact variant, a cluster skipped where the triangle inequality bounds its codes
  beyond the 10,000th distance, is unlikely to skip any (reasoned, not measured): in 768 bits a cluster's radius is
  of the order of the distances it would have to bound.
- **On a copy of a real index**, 145 GB with 3,261,593 vectors (arXiv abstracts and notes), on the i9-14900K while it
  was busy with other work: the upgrade from sift 1.6 added the codes in 65 seconds, `sift serve` loads them in 2.2
  seconds, and a search (Hamming scan and exact scores) takes 95 ms (median; 108 ms at the 90th percentile). Of 200
  of the title queries, it found 99.4% of exact search's top 10.

## Query expansion by the agent

QMD ([github.com/tobi/qmd](https://github.com/tobi/qmd)) expands each query into keywords, a rewrite and a
hypothetical answer passage. sift tried the same, with the agent writing the expansions, over Ettin-150M on the arXiv
known items. MRR@10 rose only from 0.936 to 0.938: one query was rescued and none were hurt. The set is close to its
ceiling, so sift doesn't expand queries itself, though agents are free to search several phrasings.

## Not measured

- **Rerankers above 1B parameters**, such as Ettin-1B, mxbai-rerank-large-v2, Qwen3-Reranker-4B and zerank-2. sift aims
  for a small footprint, and its 150M default already ties models three to four times its size.
- **Rerankers that need `trust_remote_code`, or have non-commercial licences.** sift's defaults must run anywhere,
  under licences that allow commercial use.
- **Your documents.** The datasets above stand in for scientific, medical, financial, German and literature search.
  Your own documents may behave differently. [`sift bench`](usage.md#measuring-search-on-your-own-documents)
  measures each search mode on questions you write, and `SIFT_RERANK_MODEL` and `SIFT_EMBED_MODEL` switch models.
