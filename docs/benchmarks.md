# Benchmarks

What to expect of sift, and the measurements behind its defaults. Every change to how `query` ranks is measured on
seven datasets with several rerankers and embedding models; the default is the setting best overall, not the winner of
one dataset. To measure sift on your own documents, use
[`sift bench`](usage.md#measuring-search-on-your-own-documents).

Timings come from one desktop workstation with a 24 GB NVIDIA GPU, 32 CPU threads and NVMe disks; the models run in
bfloat16 on the GPU, as sift runs them.

- [sift 2](#sift-2): reference values, and sift 2 against sift 1.15
- [Search quality](#search-quality): datasets, method, the blend, rerankers, embedding models
- [Reference values of sift 1.11](#reference-values-of-sift-111), and [speed](#speed) as sift 1.x measured it
- [Tried and not adopted](#tried-and-not-adopted), [not measured](#not-measured)

## sift 2

sift 2 replaced sift 1.x's keyword index (SQLite's FTS5, with postings of sift's own read out of it) by an index of its
own, and its stores' layout; how it ranks (BM25 over the same terms, the scan of the vectors' codes, the blend of
`query`) stayed, so the [search quality](#search-quality) measured on sift 1.x holds for it, as the comparison below
shows. Measured on sift 2.0, with another job sharing the GPU at about 45%.

**Search**, over 3.4 million passages (arXiv abstracts 3,252,534, the ACL Anthology 90,546, Swiss federal law 75,851),
through `sift serve` with its models loaded, 10 hits each; each row is 8 questions, each asked once, none answered from
the results the home process keeps:

| Use case | Mode | Median | Range |
|---|---|---|---|
| A name, an id, a term | `search`, one word | 27 ms | 17–48 ms |
| A question of 6 to 10 words, by its keywords | `search` | 128 ms | 44–209 ms |
| By meaning | `vsearch` | 215 ms | 211–227 ms |
| A question for RAG, reranked on the GPU | `query` | 454 ms | 387–524 ms |
| More papers like this one | `similar` | 610 ms | 435–740 ms |

- The first semantic search loaded the embedding model, in 11.6 s; the first `query` the reranker, in 0.9 s more.
- On the CPU alone, over Swiss federal law and a collection of notes (76,738 passages), `query` took 14 to 23 s a
  question (median 19 s); `search`, `vsearch` and `similar` stayed under 0.2 s.

**Against sift 1.15.2**, by `sift bench`, 12 questions on each of two collections, the documents each should find
judged in pools of both versions' results (nDCG@10 / MRR@10 / Recall@10):

| Collection | Mode | sift 1.15.2 | sift 2.0 | sift 2.0, imported from 1.15.2 |
|---|---|---|---|---|
| Swiss federal law (75,851 passages) | `search` | 0.612 / 1.000 / 0.463 | 0.612 / 1.000 / 0.463 | 0.612 / 1.000 / 0.463 |
| | `vsearch` | 0.752 / 1.000 / 0.681 | 0.753 / 1.000 / 0.681 | 0.752 / 1.000 / 0.681 |
| | `query` | 0.872 / 0.958 / 0.928 | 0.870 / 0.958 / 0.907 | 0.868 / 0.958 / 0.928 |
| Technical notes (887 passages) | `search` | 0.410 / 0.667 / 0.306 | 0.410 / 0.667 / 0.306 | 0.410 / 0.667 / 0.306 |
| | `vsearch` | 0.794 / 1.000 / 0.731 | 0.802 / 1.000 / 0.734 | 0.794 / 1.000 / 0.731 |
| | `query` | 0.877 / 1.000 / 0.924 | 0.877 / 1.000 / 0.924 | 0.886 / 1.000 / 0.935 |

- Keyword search ranks alike to the last digit. Collections `sift import` brought in keep sift 1.x's vectors, and
  `vsearch` finds exactly what sift 1.15.2 found; embedded anew by sift 2, in batches of other lengths, the vectors
  differ in their last digits, and the scores move by less than 0.01. `query` moves within 0.01 both ways.

**Embedding**, `sift embed` of all passages, the GPU shared alike:

| Collection | sift 1.15.2 | sift 2.0 |
|---|---|---|
| Swiss federal law (long passages) | 129 a second (76,738 with the notes, in 9 min 55 s) | 110 a second (75,851 in 11 min 29 s) |
| The ACL Anthology (abstracts) | | 391 a second (90,546 in 3 min 52 s) |

These were measured before sift 2 compiled the model for the GPU. On one GPU and the same 40,579 PubMed abstracts,
sift 2 now embeds 1,180 passages a second where sift 1.15.2 embeds 590, after 20 s of compiling (41 s the first
time). Both sort the passages they read by length; sift 2 also cuts the words of the next batch while the GPU runs.

**Indexing** the ACL Anthology anew (1,134 files of JSON Lines, 90,504 records), `sift index` with the keyword index:
sift 2.0 in 24 s, sift 1.15.2 in 29 s (12 s reading, then its postings).

## Reference values of sift 1.11

sift 1.11's figures, kept for the scale sift 2 has not been measured at yet. The large index holds 45.7 million
passages of paper abstracts (arXiv 3.2 million, PubMed 29.9 million, OpenAlex 12.0 million, bioRxiv 0.47 million, the
ACL Anthology 0.09 million), every passage embedded, searched through `sift serve` (sift 1.11.1), 10 hits each. Times
are a query's first call and the median of two more ("warm": what it reads is in memory by then), including the MCP
round trip.

| Use case | Mode | Large index, first / warm | arXiv alone, warm |
|---|---|---|---|
| A name, an id, a rare term | `search`, one word | 93 / 10 ms | 0.7 ms |
| A common word ("patients") | `search`, one word | 332 / 17 ms | |
| A few words, all required | `search` | 66–122 / 31–46 ms | 2–7 ms |
| A question of 15 words, by keywords | `search` | 594 / 223 ms | 34–81 ms |
| A quoted phrase | `search` | 1.9 / 0.6–1.0 s | 1–131 ms (1.11.2) |
| A question with a hyphenated word ("first-stage") | `search` | 2.8 / 1.6 s | 93 ms (1.11.2) |
| By meaning | `vsearch` | 635 / 613 ms | 94 ms |
| By meaning, filtered (a small collection, by year) | `vsearch` | 343 / 189 ms | |
| A question for RAG, reranked on the GPU | `query` | 1.1 / 0.57 s | 0.28 s |
| More papers like this one | `similar` | 708 / 702 ms | |

- **Phrases** got faster in 1.11.2 (counted once per passage and kept): a phrase asked again costs a half to a
  twentieth of what it did over arXiv. The large-index figures are 1.11.1's.
- **On the CPU**, `query`'s reranker takes about 25 seconds a question; the other modes need no GPU, and a
  [`sift worker`](worker.md) lends one.
- **Memory:** the vectors' codes take 96 bytes a passage in RAM, 4.4 GB for 45.7 million; the keyword postings a third
  of FTS5's index on disk.
- **Building:** `sift embed` embeds 370 to 520 PubMed passages a second (EmbeddingGemma-2). `sift index` builds the
  keyword postings of arXiv in 69 seconds (24 processes) and of PubMed in 13 to 15 minutes (31), orders 3.2 million
  passages by topic in about a minute, and passes over unchanged folders in a moment (0.1 s for 2,364 files).

How well each mode finds what is sought (nDCG@10; MRR@10 for the known items):

| Kind of question | `search` (BM25) | `vsearch` | `query` |
|---|---|---|---|
| Scientific claims against abstracts (SciFact) | 0.666 | 0.840 | 0.813 |
| Medical questions (NFCorpus) | | 0.358 | 0.361 |
| Financial questions against forum answers (FiQA) | | 0.314 | 0.464 |
| German questions against passages (GermanQuAD) | | 0.955 | 0.987 |
| Long German documents (MLDR, German) | | 0.534 | 0.685 |
| Literature search over ML and NLP papers (LitSearch) | | 0.486 | 0.631 |
| One known paper among 3.2 million (arXiv; MRR@10) | | | 0.929 |

## Search quality

### Datasets

| Dataset | Kind | Documents | Queries | Metric |
|---|---|---|---|---|
| [SciFact](https://github.com/beir-cellar/beir) (BEIR) | Scientific claims against abstracts | 5,183 | 300 | nDCG@10 |
| [NFCorpus](https://github.com/beir-cellar/beir) (BEIR) | Medical questions, graded relevance | 3,633 | 323 | nDCG@10 |
| [FiQA](https://github.com/beir-cellar/beir) (BEIR) | Financial questions against forum answers | 57,638 | 648 | nDCG@10 |
| [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad) | German questions against passages | 474 | 500 of 2,204 (seed 0) | nDCG@10 |
| [MLDR](https://huggingface.co/datasets/Shitao/MLDR), German | Long German documents | 10,000 | 200 | nDCG@10 |
| [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) | Literature-search questions over ML and NLP papers | 64,183 | 597 | nDCG@10 |
| arXiv known items | One paper, named by author, id, paraphrase or acronym | 3,215,242 | 49 | MRR@10 |

- **arXiv known items** was written for sift and isn't distributed: titles and abstracts of all of arXiv (the Kaggle
  snapshot); 12 queries name front matter (an author, an arXiv id), 37 describe a topic.
- **LitSearch** is reported by nDCG@10 like the others; its paper reports Recall@5 and Recall@20.

### Method

- Each dataset is a sift collection of JSON Lines (`id`, `title`, `text`), indexed and embedded as any other.
- For every query, the candidates `query` reranks are collected once per embedding model: the best passage of each of
  the 40 best documents by BM25 and of the 40 best by cosine. Every reranker scores the same pools, so differences
  between rerankers are theirs alone.
- Each setting gets its mean over the seven datasets and its *worst gap*: its largest shortfall on any dataset against
  the best setting measured there.
- Sanity check: sift's BM25 scores 0.666 on SciFact, BEIR's published baseline 0.665.

### Why `query` blends in the cosine

A reranker alone ranks scientific text worse than the embedding does: with Ettin-150M, `query` scored 0.760 on SciFact
against 0.840 for the cosine alone; rerankers trained on web search lose precision there
([SciRet](https://arxiv.org/abs/2608.03860)). On the arXiv known items the reranker is essential: MRR 0.936 against
0.627 for the cosine, since only the reranker reads the front-matter values a query names. The forms tried, with
Ettin-150M:

| Order | SciFact | NFCorpus | arXiv | arXiv, front matter | arXiv, topical |
|---|---|---|---|---|---|
| Reranker alone | 0.760 | 0.374 | 0.936 | 0.889 | 0.951 |
| Reranker + 20 × cosine, front-matter matches exempt | 0.811 | **0.394** | **0.956** | **0.903** | **0.973** |
| Reranker + 40 × cosine, front-matter matches exempt | 0.840 | 0.393 | 0.940 | **0.903** | 0.953 |
| Reranker + 80 × cosine, front-matter matches exempt | **0.856** | 0.389 | 0.914 | **0.903** | 0.918 |
| Reranker + 20 × cosine, no exemption | 0.811 | **0.394** | 0.920 | 0.757 | **0.973** |
| Mean of the two z-scores | 0.834 | – | – | 0.675 | 0.953 |

- **The exemption:** a candidate whose front matter matches a word of the query keeps the pool's best cosine. Without
  it, author and id queries fall from 0.889 to 0.757.
- **A scale-free weight:** rerankers' scores span different ranges, so the cosine is scaled to the reranker's spread
  over the query's candidates:

  score = rerank + w · (σ<sub>rerank</sub> / σ<sub>cosine</sub>) · (cosine − mean cosine)

  σ is the standard deviation over the candidates. w = 0.5 is the best or tied-best weight under every reranker and
  embedding model below.

### Rerankers

All score the same EmbeddingGemma-2 pools; w is the blend weight.

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

- **gte-reranker-modernbert-base, blended** (the default; Apache-2.0) is the smallest of the leaders and never more
  than 0.05 below the best on any dataset. Against Ettin-150M alone it gains 0.05 on the mean and 0.20 on long German
  documents. Weights 0.5 and 0.75 tie on the mean; 0.5 keeps more of the reranker's lead on the known items.
- **mxbai-rerank-base-v2** leads on scientific and medical text, at three times the size and twice the time.
- **The multilingual rerankers** (bge-reranker-v2-m3, Qwen3-Reranker-0.6B) lead on German, trail on arXiv and FiQA.
- **The blend helps every reranker:** their means rise by 0.03 to 0.05.

### Embedding models

The arXiv known items search a 100,000-paper subset holding every target, as embedding all of arXiv once per model
takes hours. Each model alone, ranking the candidates by cosine as `vsearch` does:

| Embedding model | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
|---|---|---|---|---|---|---|---|---|
| EmbeddingGemma-2 | **0.840** | **0.358** | 0.314 | 0.955 | 0.534 | 0.486 | 0.806 | 0.613 |
| Snowflake Arctic-Embed-L v2.0 | 0.704 | 0.355 | 0.451 | 0.964 | 0.539 | **0.499** | **0.843** | **0.622** |
| bge-m3 | 0.644 | 0.313 | 0.412 | **0.969** | **0.552** | 0.436 | 0.818 | 0.592 |
| Qwen3-Embedding-0.6B | 0.700 | 0.355 | **0.469** | 0.950 | 0.531 | 0.174 | 0.819 | 0.571 |

With `query` (gte-reranker-modernbert-base, w = 0.5) over each model's candidates:

| Embedding model | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
|---|---|---|---|---|---|---|---|---|
| EmbeddingGemma-2 | **0.813** | **0.361** | 0.464 | 0.987 | 0.685 | **0.631** | **0.960** | **0.700** |
| Snowflake Arctic-Embed-L v2.0 | 0.756 | 0.358 | 0.505 | 0.986 | **0.700** | 0.624 | 0.959 | 0.698 |
| bge-m3 | 0.753 | 0.340 | 0.486 | 0.986 | 0.695 | 0.601 | 0.948 | 0.687 |
| Qwen3-Embedding-0.6B | 0.762 | 0.360 | **0.518** | **0.988** | 0.695 | 0.417 | 0.959 | 0.671 |
| *Ettin-150M alone over EmbeddingGemma-2 (up to 0.8.0)* | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.976 | 0.651 |

- **No model wins everywhere:** EmbeddingGemma-2 leads far on scientific claims and trails on financial questions.
- **The reranker narrows the gaps** (means 0.571–0.622 alone, 0.671–0.700 under `query`), and the blend carries each
  model's strengths through: EmbeddingGemma-2 keeps +0.06 on scientific claims, Arctic +0.04 on financial questions
  and +0.015 on long German documents.
- **gte-reranker-modernbert-base, blended, is the best reranker under every embedding model** (worst gap 0.02–0.04).
- **EmbeddingGemma-2 stays the default:** Arctic ties it on average, and a new default would force every index to be
  embedded again. For a new index of business documents, Arctic is a good choice.
- **Qwen3-Embedding's LitSearch score** is out of line with its others and unexplained; it isn't held against it.

## Speed

Measured on sift 1.8 to 1.11.2, as sift 1.x ran: keyword search read SQLite's FTS5 and postings of its own read out of
it, and `sift serve` scanned the vectors' codes with code compiled by Numba. The models, the reranking batches and the
scheme of the vectors in memory are sift 2's too; its own timings are [above](#sift-2).

### Models

Each model ran alone on an otherwise idle machine, after a warm-up. The rerankers scored 50 queries × 80 candidates,
half from SciFact (short abstracts), half from MLDR (long German documents): on the GPU 5 repetitions (spread ≤ 1%), on
8 CPU threads in float32 over a tenth of the inputs, 3 repetitions (spread ≤ 10%).

| Reranker | Size | GPU, ms per query | CPU, ms per query |
|---|---|---|---|
| **gte-reranker-modernbert-base** | 150M | **284** | 26,596 |
| Ettin-150M | 149M | 284 | 25,217 |
| bge-reranker-v2-m3 | 568M | 305 | 41,511 |
| mxbai-rerank-base-v2 | 494M | 559 | 56,657 |
| Ettin-400M | 400M | 608 | not measured |
| Qwen3-Reranker-0.6B | 596M | 1,025 | 90,160 |

The embedding models embedded 2,000 passages (half SciFact, half MLDR) and 100 queries on the GPU:

| Embedding model | Passages per second | ms per query |
|---|---|---|
| EmbeddingGemma-2 | 120 | 51 |
| Snowflake Arctic-Embed-L v2.0 | 190 | 13 |
| bge-m3 | 190 | 16 |
| Qwen3-Embedding-0.6B | 68 | 36 |

### Reranking in batches of similar length

A question's 80 candidates were reranked in one batch, padded to the longest. Over the benchmark pools, the work that
is against none padded (tokens / attention's square):

| Pool | One batch of 80 | Batches of 16, sorted by length | Batches of 10 |
|---|---|---|---|
| arXiv known items | 1.71 / 2.74 | 1.14 / 1.32 | 1.08 / 1.18 |
| FiQA | 3.53 / 8.40 | 1.42 / 2.25 | 1.24 / 1.69 |
| GermanQuAD | 1.89 / 3.00 | 1.18 / 1.36 | 1.11 / 1.21 |
| LitSearch | 2.43 / 5.47 | 1.29 / 1.89 | 1.17 / 1.51 |

Over 10 questions each from the arXiv and FiQA pools, ms a question: 331 in one batch (sift 1.11), 213 in batches of
32, **180 in 16** (sift 1.12), 259 in 8; the scores moved by at most 0.08, bfloat16's rounding.

### Embedding

`sift embed` with EmbeddingGemma-2 over 6,000 PubMed abstract passages (339 tokens on average) and their 6,000 titles
(30 tokens); the mean of 2 runs:

| Batches | Passages per second | CPU cores busy |
|---|---|---|
| sift 1.8: grouped by characters, a CPU thread per core | 171 | 4.7 |
| sift 1.9: grouped by tokens, one CPU thread | **230** | 0.9 |

| Titles, batch of | Grouped by characters | Grouped by tokens |
|---|---|---|
| 140 | 2,295 per second, padded to 1.50 times their tokens | **2,504**, 1.04 times |
| 1,024 | 1,742, 1.78 times | 2,424, 1.30 times |

- Grouped by characters, passages were padded to 1.51 times their tokens; by tokens, 1.04 times.
- A batch of 140 short texts already fills the GPU; 1,024 only adds padding. So one batch size, sized for the longest
  input, serves every length.
- One CPU thread beside the GPU embedded 5% faster and freed most cores.

Over 40,114 PubMed passages from no vectors, the GPU sampled every 100 ms: sift 1.10.0 embedded 382 passages a second
with the GPU busy 83% of the time, sift 1.10.1 **472** at 99%, the vectors the same bit for bit. Profiled, 1.10.0 spent
72% in the model, 12% tokenizing each slab to sort it (the GPU idle) and 11% tokenizing each batch (idle again); now a
thread sorts the next slabs, another writes, and each batch is tokenized while the GPU embeds the one before.

### Keyword search

63 queries: 18 single words ("crispr" to "the"), 20 requiring all their words ("graph neural network", ids), and 25
questions as `query` asks them, any of their words. The best 10 documents; FTS5 read warm. "Cold": the postings read
anew, as by `sift search`; "warm": read before, as by `sift serve`. Every query found FTS5's documents, score for score.

| Median | FTS5 | Postings, cold | Postings, warm |
|---|---|---|---|
| arXiv (3.2 million passages), single words | 120 ms | 10 ms | 0.7 ms |
| arXiv, all words | 20 ms | 21 ms | 2 ms |
| arXiv, questions | 510 ms | 115 ms | 13 ms |
| PubMed (29.9 million passages), single words | 1.33 s | 67 ms | 7 ms |
| PubMed, all words | 30 ms | 86 ms | 7 ms |
| PubMed, questions | 5.3 s | 200 ms | 26 ms |

- **FTS5 scores every passage a word occurs in**, about a microsecond each: "the" in 29.6 million PubMed passages took
  25 seconds. The postings score the best ranges first: for "the", a few hundred postings.
- **The slowest question** took 16 s in FTS5, 2.7 s cold and 1.1 warm from the postings: a dozen common words bound
  most ranges high. Smaller ranges would bound them tighter, at more storage.
- **All of a few rare words** is the one kind FTS5 answers faster cold (30 ms against 86 on PubMed), as it intersects
  lists while reading; warm, the postings take 7 ms.
- **Storage:** 0.94 GB for arXiv's 431 million postings against 2.71 GB for FTS5's index (2.2 bytes a posting); 12.2 GB
  for PubMed.
- **Ranges of 128 rather than 64 passages:** on PubMed, warm, single words in 2.2 ms against 5.8 and questions in 16
  against 23 (median), the longest question slower (0.53 s against 0.24), and 8% less storage.
- **Topical order** (sift 1.11): over arXiv, warm, questions took 3.7 ms instead of 5.2 (median) and the slowest 45
  instead of 90, scoring a half to a twentieth as many ranges; single words 0.7 ms instead of 0.3, as a word's best
  passages spread over more ranges. The blocks took 0.61 GB instead of 0.66, the order 13 MB.
- **Common words read only where they matter** (sift 1.11.1), over arXiv, warm, the same results:

  | Query | 1.11.0 | 1.11.1 |
  |---|---|---|
  | seven words, all required | 65 ms | 7 ms |
  | questions of 10 to 14 words | 0.44 to 2.5 s | 34 to 81 ms |
  | `query`'s keyword candidates (any of the words) | 22 to 230 ms | 9 to 62 ms |
  | a question with a hyphenated word, through FTS5 before | 8.7 s | 0.43 s |
  | quoted phrases, through FTS5 before | 10 to 97 ms | 3 to 16 ms |

- **A phrase counted once per passage** (sift 1.11.2): 62% of a hyphenated question's time had been FTS5 counting the
  phrase in the same passages again. Over arXiv, warm: that question 93 ms instead of 429, "in this paper we propose"
  131 ms instead of 2.6 s, four others 1 to 7 ms instead of 3 to 14. A new question's first call costs what it did.

### Vectors in memory

Exact search over fp16 vectors holds 1.5 KB per 768-dimensional passage: 4.9 GB for 3.2 million, 60 GB for 40
million. Each scheme against exact search over the 3,215,416 arXiv abstract passages (EmbeddingGemma-2), for 2,000
known-item title queries: recall@k is the share of exact search's top k found; MRR@10 and hit@10 are each paper's own.
"+ exact top N" scores the N nearest exactly.

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
| Binary, Hamming + exact top 4,000 | 96 | 0.996 | 0.986 | 0.9528 | 0.9905 |
| Binary, float query + exact top 400 | 96 | 0.992 | 0.955 | 0.9530 | 0.9905 |
| Binary, float query + exact top 1,000 | 96 | 0.998 | 0.988 | 0.9528 | 0.9905 |

One bit per dimension, then exact scores, keeps the quality at a sixteenth of the memory; truncating dimensions loses
more. Ranking by the float query (its values against the codes' signs) needs a quarter of the exact scores Hamming
distance needs. sift does both, one after the other (1.11): Hamming over all codes, the float query over the nearest N
(3 ms for 10,000), exact scores for its best M. Reading vectors is most of a search's time: 25 ms for 4,000, warm,
7 ms for 1,000.

| Hamming nearest N | Float query's best M, scored exactly | Recall@10 | MRR@10 | Hit@10 |
|---|---|---|---|---|
| 4,000 | (all 4,000, sift 1.10) | 0.9961 | 0.9530 | 0.9905 |
| 1,000 | (all 1,000) | 0.9794 | 0.9530 | 0.9905 |
| 10,000 | 400 | 0.9921 | 0.9530 | 0.9905 |
| **10,000** | **1,000** | **0.9975** | **0.9530** | **0.9905** |
| 10,000 | 2,000 | 0.9986 | 0.9530 | 0.9905 |
| 20,000 | 1,000 | 0.9978 | 0.9530 | 0.9905 |
| 40,000 | 1,000 | 0.9979 | 0.9530 | 0.9905 |

- **The compiled scan** (sift 1.12): NumPy scanned the codes in three passes, writing and reading two arrays as large
  as the codes, at 10 GB/s; compiled by Numba, each code is read once. Over 12.9 million codes (arXiv's four times
  over), alternating runs: 133 ms (median 164) with NumPy, 43 (56) compiled, the same distances. Four queries in one
  pass took 35 ms. Loading Numba takes about 1.6 s once per process, hence only `sift serve` and `sift bench` load it,
  and collections under 65,536 passages, where it gains under a millisecond, keep NumPy.
- **Filters:** listing every passage a filter lets through reads every document: over arXiv, a category took 6.2 s warm
  (56 cold), whether it let 9% through or 0.9%. Checking candidates costs 4 to 5 µs each warm, 90 cold, so sift checks
  the nearest first and lists only when it would check more than a fifth of the passages. Until 10,000 candidates
  passed, cs.LG (9%) checked 130,000 in 0.6 s; cs.IR (0.9%) gave up after 10,000 checks (0.02 to 0.08 s) and was
  listed.
- **Adaptive k** ([Taguchi et al. 2025](https://arxiv.org/abs/2506.08479)), for the 2,000 titles, of exact search's top
  10, 20 or 50:

  | Hits kept after the largest drop | Hits returned, on average | The paper sought kept, when among them |
  |---|---|---|
  | 0 | 1.3 | 96.9 to 97.2% |
  | 2 | 3.3 | 98.5 to 98.9% |
  | **5** | **6.2 to 6.3** | **99.2 to 99.6%** |

  Known items mostly drop after the first hit; questions with many answers need more, hence five.

### Links between notes

A synthetic vault of 20,000 notes and 220,000 links, one note linked by 6,000 others, a tenth of the links to notes not
written yet; best of 3:

| Operation | Time |
|---|---|
| `sift index` (20 processes) | 6.9 s |
| `links` out of a note; both ways, two steps | 0.1 ms; 1.5 ms |
| `links` into the most linked note, 50; 10,000 | 16 ms; 42 ms |
| `unresolved` over the vault | 210 ms |
| `search` with `links_to` the most linked note (without) | 25 ms (19 ms) |
| `mentions` of the most linked note | 20 ms |

Reading links costs nothing measurable: 20,000 notes without links read in 0.617 s before, 0.626 s after.

## Tried and not adopted

- **Reranking fewer candidates when the lists agree:** simulated on the stored scores of the pools. 20 keyword and 20
  semantic candidates instead of 40 each cost the arXiv known items 5 points of nDCG@10 and FiQA 2; doing so only when 5
  of the top 20 agree still reranked 65 to 85% and cost 0.5 to 0.7 points. Batching by length took most of the time.
- **Query expansion by the agent** (as [QMD](https://github.com/tobi/qmd) does: keywords, a rewrite, a hypothetical
  answer): MRR@10 on the arXiv known items rose from 0.936 to 0.938, one query rescued. Agents may still search several
  phrasings.
- **zstd instead of zlib for the postings' blocks:** decoding 43 to 54 µs a block against 40 to 76, the same size; only
  writing was faster. Not worth a store older sifts can't read.
- **int8 vectors for the exact scores:** over 500 title queries, recall@10 0.962 (a scale per vector) or 0.974 (per
  dimension) against 0.996, MRR@10 unchanged. Half the vectors' storage for three points of recall.
- **A first scan of the first 256 dimensions' bits,** keeping the nearest P: recall@10 0.9938 keeping 100,000, 0.9954
  keeping 300,000, against 0.9958. It would read a third of the codes but need them apart in memory (1.5 GB more at 46
  million), and the compiled scan took most of its gain.
- **Centering before the signs:** codes of v − mean found more of the top 10 by Hamming distance alone (0.9917 against
  0.9794 among the 1,000 nearest), but the float query's ranking loses it, as centered vectors differ in length: the
  whole cascade found 0.9788 instead of 0.9975.
- **Scanning the nearest clusters only** (the postings' topical order, 4,096 leaves), as an inverted file does: of the
  top 10, 0.9882 scanning 10% of the codes, 0.9952 at 20%, 0.9966 at 30%, against 0.9975 for all. A third would save
  at most two thirds of the scan, at a loss of recall, with clusters to keep up and filters to check otherwise.

## Not measured

- **Rerankers above 1B parameters** (Ettin-1B, mxbai-rerank-large-v2, Qwen3-Reranker-4B, zerank-2): the 150M default
  already ties models three to four times its size.
- **Rerankers needing `trust_remote_code` or with non-commercial licences:** the defaults must run anywhere, under
  licences allowing commercial use.
- **Your documents.** The datasets stand in for scientific, medical, financial, German and literature search; yours may
  differ. [`sift bench`](usage.md#measuring-search-on-your-own-documents) measures them, and `SIFT_RERANK_MODEL` and
  `SIFT_EMBED_MODEL` switch models.
