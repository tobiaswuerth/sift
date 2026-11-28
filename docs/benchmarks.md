# Benchmarks

What to expect of sift, and the measurements behind its defaults. Every change to how `query` ranks is measured on
seven datasets with several rerankers and embedding models; the default is the setting best overall, not the winner of
one dataset. To measure sift on your own documents, use
[`sift bench`](usage.md#measuring-search-on-your-own-documents).

Timings come from one desktop workstation with a 24 GB NVIDIA GPU, 32 CPU threads, 192 GB of RAM and NVMe disks; the
models run in bfloat16 on the GPU, as sift runs them.

- [Reference values](#reference-values): search, indexing, topical order, embedding
- [Search quality](#search-quality): datasets, method, the blend, rerankers, embedding models
- [Design measurements](#design-measurements): models, reranking batches, vectors in memory, the keyword index
- [Tried and not adopted](#tried-and-not-adopted), [not measured](#not-measured)

## Reference values

Measured on copies of seven collections, 45.7 million passages of abstracts, papers, laws and notes: PubMed 29.9
million, OpenAlex 12.0 million, arXiv 3.3 million, bioRxiv 0.47 million, the ACL Anthology 0.09 million, Swiss
federal law 0.08 million and a collection of notes. Another workload shared the CPU (at a lower priority) and the
GPU throughout.

**Search** over all seven, every passage embedded, through `sift serve` with its models loaded, 10 hits each. Each row
is the median over 8 questions, of each question's first call and of two more calls asking for 11 and 12 hits, which the
home process's cache of results cannot answer ("warm": what they read is in memory by then):

| Use case | Mode | First call | Warm |
|---|---|---|---|
| A name, an id, a rare term | `search`, one word | 157 ms | 31 ms |
| A common word ("patients") | `search`, one word | 466 ms | 184 ms |
| A few words, all required | `search` | 261 ms | 76 ms |
| A question of 6 to 15 words, by its keywords | `search` | 600 ms | 274 ms |
| A quoted phrase | `search` | 732 ms | 326 ms |
| By meaning | `vsearch` | 794 ms | 416 ms |
| A question for RAG, reranked on the GPU | `query` | 1.08 s | 0.53 s |
| More papers like this one | `similar` | 604 ms | 349 ms |

- **The first semantic search** of a home process reads the codes of every passage's vector, 4.4 GB for 45.7 million,
  and loads the embedding model: 104 s from a cold disk. Later searches find both in memory.
- **A phrase of the commonest words**, such as "in this paper we propose", would read 76 million passages here, more
  than the 10 million a search reads ([limits](limits.md)); it is refused, saying to add a rarer word.
- **On the CPU alone**, over Swiss federal law and the notes (76,738 passages), `query` took 14 to 23 s a question
  (median 19 s); `search`, `vsearch` and `similar` stayed under 0.2 s.

**Indexing**: reading every file into a new store, then `sift index`, which puts each collection in its topical
order:

| Collection | Files | Passages | Reading the files | Merging | Topical order: clusters / index built beside / merged |
|---|---|---|---|---|---|
| arXiv (JSON Lines) | 425 | 3,252,534 | 8 min 22 s | 30 s | 9.4 s / 1 min 29 s / 26 s |
| bioRxiv (JSON Lines) | 247 | 472,612 | 42 s | 6.7 s | 3.1 s / 13 s / 5.2 s |
| ACL Anthology (JSON Lines) | 1,134 | 90,546 | 7.3 s | | 0.7 s / 1.8 s |
| Swiss federal law (Markdown) | 5,339 | 75,851 | 16 s | | 0.6 s / 3.8 s |

- **Reading the files** includes splitting them into passages and writing their keywords, in the documents'
  transaction. Run alone on the arXiv papers of 2024 (12 files, 244,047 passages), it took 18 s, and the merge 2 s.
- **Clustering PubMed's** 29.9 million codes for its topical order took 2 min 4 s.
- **Unchanged folders** cost a look: arXiv's 425 files were passed over in 56 ms.

**Embedding**, `sift embed` of 40,579 PubMed abstracts on the GPU: 1,180 passages a second, after compiling the model
for 20 s (41 s the first time, when PyTorch's cache of compiled kernels is empty).

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

How well each mode finds what is sought, with the default models (nDCG@10; MRR@10 for the known items):

| Kind of question | `search` (BM25) | `vsearch` | `query` |
|---|---|---|---|
| Scientific claims against abstracts (SciFact) | 0.666 | 0.840 | 0.813 |
| Medical questions (NFCorpus) | | 0.358 | 0.361 |
| Financial questions against forum answers (FiQA) | | 0.314 | 0.464 |
| German questions against passages (GermanQuAD) | | 0.955 | 0.987 |
| Long German documents (MLDR, German) | | 0.534 | 0.685 |
| Literature search over ML and NLP papers (LitSearch) | | 0.486 | 0.631 |
| One known paper among 3.2 million (arXiv; MRR@10) | | | 0.929 |

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

  σ is the standard deviation over the candidates. sift uses w = 0.5, which ties the best weight under its reranker
  (below).

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
| Ettin-150M, no blend | 149M | 0 | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.936 | 0.646 | 0.235 |

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
| *Ettin-150M alone over EmbeddingGemma-2* | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.976 | 0.651 |

- **No model wins everywhere:** EmbeddingGemma-2 leads far on scientific claims and trails on financial questions.
- **The reranker narrows the gaps** (means 0.571–0.622 alone, 0.671–0.700 under `query`), and the blend carries each
  model's strengths through: EmbeddingGemma-2 keeps +0.06 on scientific claims, Arctic +0.04 on financial questions
  and +0.015 on long German documents.
- **gte-reranker-modernbert-base, blended, is the best reranker under every embedding model** (worst gap 0.02–0.04).
- **EmbeddingGemma-2 is the default:** Arctic ties it on average, and a change of default would force every index to
  be embedded again. For a new index of business documents, Arctic is a good choice.
- **Qwen3-Embedding's LitSearch score** is out of line with its other scores and unexplained, so it isn't held
  against the model.

## Design measurements

### Models

Each model ran alone on an otherwise idle machine, after a warm-up, as PyTorch runs it without compiling. The
rerankers scored 50 queries × 80 candidates, half from SciFact (short abstracts), half from MLDR (long German
documents): on the GPU 5 repetitions (spread ≤ 1%), on 8 CPU threads in float32 over a tenth of the inputs, 3
repetitions (spread ≤ 10%).

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

A question's 80 candidates, reranked in one batch, are padded to the longest. Over the benchmark pools, the work
against none padded (tokens / attention's square):

| Pool | One batch of 80 | Batches of 16, sorted by length | Batches of 10 |
|---|---|---|---|
| arXiv known items | 1.71 / 2.74 | 1.14 / 1.32 | 1.08 / 1.18 |
| FiQA | 3.53 / 8.40 | 1.42 / 2.25 | 1.24 / 1.69 |
| GermanQuAD | 1.89 / 3.00 | 1.18 / 1.36 | 1.11 / 1.21 |
| LitSearch | 2.43 / 5.47 | 1.29 / 1.89 | 1.17 / 1.51 |

Over 10 questions each from the arXiv and FiQA pools, ms a question: 331 in one batch, 213 in batches of 32, **180 in
16**, sift's batches, and 259 in 8; the scores moved by at most 0.08, bfloat16's rounding.

### Embedding many passages

`sift embed` over the 40,579 PubMed abstracts above, the GPU sampled every 100 ms:

- **Longest first:** it reads 16,384 passages at a time and embeds them longest first, so that a batch's texts are
  of similar length and little of the GPU's work is padding.
- **The next batch's words cut ahead:** a thread tokenizes the next two batches, their tensors pinned in memory and
  copied without waiting, while the model runs on the one before.
- **The model compiled:** PyTorch compiles the whole model into one graph, with no break in it, for any batch size
  and length (`dynamic=True`). That doubles the passages a second; the GPU is then busy 82 to 84% of the time, the
  rest mostly the store's reads and writes. A batch of one text would be compiled again, so queries and single texts
  run the model as it is; and the model is compiled under the threads the process runs with, as changing them later
  compiles it again.

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
distance needs. sift does both, one after the other: Hamming over all codes, the float query over the nearest N
(3 ms for 10,000), exact scores for its best M. Reading vectors is most of a search's time: 25 ms for 4,000, warm,
7 ms for 1,000.

| Hamming nearest N | Float query's best M, scored exactly | Recall@10 | MRR@10 | Hit@10 |
|---|---|---|---|---|
| 4,000 | (all 4,000) | 0.9961 | 0.9530 | 0.9905 |
| 1,000 | (all 1,000) | 0.9794 | 0.9530 | 0.9905 |
| 10,000 | 400 | 0.9921 | 0.9530 | 0.9905 |
| **10,000** | **1,000** | **0.9975** | **0.9530** | **0.9905** |
| 10,000 | 2,000 | 0.9986 | 0.9530 | 0.9905 |
| 20,000 | 1,000 | 0.9978 | 0.9530 | 0.9905 |
| 40,000 | 1,000 | 0.9979 | 0.9530 | 0.9905 |

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

### The keyword index

- **Searches in the compiled extension:** the lengths of passages, the positions a phrase needs, the blocks a search
  reads and the union a prefix makes are unpacked in Rust, a call for a whole read rather than one a block. Over a
  collection of 2.2 million records, cold, the same hits at every search: a rare word in 15 ms, a question in 43, a
  phrase in 52, a rare phrase in 126, a phrase of common words in 369 and a prefix of many words in 358.
- **Clustering in the compiled extension:** the topical order's k-means measures every code against every centroid
  in one pass on all cores. bioRxiv's 472,612 codes were clustered in 2.1 s where NumPy, a centroid at a time, took
  14.7 s, and into the same order.
- **Readers beside the writer:** a new order reads every passage again, scattered over the store. Eight readers on
  their own connections read the parts ahead while the writer tokenizes and writes: bioRxiv's index was built beside
  in 19 s instead of 44 from a cold disk, and in 12.5 s instead of 18 from memory. A reader takes a part's
  documents and text blocks as one row of JSON each: read row by row, each thread let go of Python's lock at every
  row and waited to take it back, and eight readers read slower than one.

## Tried and not adopted

- **Reranking fewer candidates when the lists agree:** simulated on the stored scores of the pools. 20 keyword and 20
  semantic candidates instead of 40 each cost the arXiv known items 5 points of nDCG@10 and FiQA 2; doing so only when 5
  of the top 20 agree still reranked 65 to 85% and cost 0.5 to 0.7 points. Batching by length already saves most of that
  time.
- **Query expansion by the agent** (as [QMD](https://github.com/tobi/qmd) does: keywords, a rewrite, a hypothetical
  answer): MRR@10 on the arXiv known items rose from 0.936 to 0.938, one query rescued. Agents may still search several
  phrasings.
- **int8 vectors for the exact scores:** over 500 title queries, recall@10 0.962 (a scale per vector) or 0.974 (per
  dimension) against 0.996, MRR@10 unchanged. Half the vectors' storage for three points of recall.
- **A first scan of the first 256 dimensions' bits,** keeping the nearest P: recall@10 0.9938 keeping 100,000, 0.9954
  keeping 300,000, against 0.9958. It would read a third of the codes but need a separate copy of them in memory (1.5 GB
  more at 46 million).
- **Centering before the signs:** codes of v − mean found more of the top 10 by Hamming distance alone (0.9917 against
  0.9794 among the 1,000 nearest), but the float query's ranking loses it, as centered vectors differ in length: the
  whole cascade found 0.9788 instead of 0.9975.
- **Scanning the nearest clusters only** (the postings' topical order, 4,096 leaves), as an inverted file does: of the
  top 10, 0.9882 scanning 10% of the codes, 0.9952 at 20%, 0.9966 at 30%, against 0.9975 for all. A third would save
  at most two thirds of the scan, at a loss of recall, with clusters to keep up and filters to check otherwise.
- **Compiling the model region by region,** each repeated layer once: the first batch came after 5.6 s instead of
  20, but the model embedded 14% slower than compiled whole.
- **Compiling the model ahead of time** (AOTInductor) or saving the exported graph: on Windows the first needs
  Microsoft's C++ compiler on the user's machine; the second took 6.6 s to load from a file of 1.5 GB and then as
  long to compile as the model itself.
- **A wider window of passages sorted by length** for `sift embed`: 3% faster end to end, not worth the memory.
- **Memory-mapping the store** while an order is built beside: the cold reads took a third of their time, but SQLite
  maps at most 2 GB of a store, so a large one gains little.
- **Reading a new order's passages in the store's order** rather than the new one: no faster from a cold disk, as
  each passage was still a read of its own.
- **Sixteen readers** instead of eight: slower, as they wait for Python's lock more than for the disk.

## Not measured

- **Rerankers above 1B parameters** (Ettin-1B, mxbai-rerank-large-v2, Qwen3-Reranker-4B, zerank-2): the 150M default
  already ties models three to four times its size.
- **Rerankers needing `trust_remote_code` or with non-commercial licences:** the defaults must run anywhere, under
  licences allowing commercial use.
- **Your documents.** The datasets stand in for scientific, medical, financial, German and literature search; yours may
  differ. [`sift bench`](usage.md#measuring-search-on-your-own-documents) measures them, and `SIFT_RERANK_MODEL` and
  `SIFT_EMBED_MODEL` switch models.
