# Benchmarks

How sift's retrieval is measured, and what the measurements decided. Numbers come from runs on a desktop with an
RTX 4090. The harness lives outside the repository, in `D:\sift-bench`: datasets, converters, one index per
dataset and embedding model, candidate pools, and results.

## Method

- **Datasets.** All are converted to sift collections, JSON Lines with `id`, `title` and `text`, and indexed as
  `sift index` and `sift embed` would.

  | Set | Kind | Documents | Queries | Metric |
  |---|---|---|---|---|
  | SciFact (BEIR) | Scientific claims, abstracts | 5,183 | 300 | nDCG@10 |
  | NFCorpus (BEIR) | Medical, graded relevance | 3,633 | 323 | nDCG@10 |
  | FiQA (BEIR) | Financial questions, forum answers | 57,638 | 648 | nDCG@10 |
  | GermanQuAD | German questions, passages | 474 | 500 of 2,204 (sampled, seed 0) | nDCG@10 |
  | MLDR (German) | Long German documents | 10,000 | 200 | nDCG@10 |
  | LitSearch | Literature search questions, ML/NLP papers (titles and abstracts) | 64,183 | 597 | Recall@5, @20 |
  | arXiv known items (ours) | Authors, ids, paraphrases and acronyms over 3.2M arXiv abstracts | 3,215,242 | 49 | MRR@10 |

- **Candidate pools.** Every reranker scores the same candidates per query, as `query` builds them: the best passage
  of each of the 40 best documents by keyword and by meaning.
- **Orders compared.** The reranker alone, the cosine alone, and z-score blends of the two.

## Findings so far

1. **The test setup is sound.** sift's keyword search scores nDCG@10 0.666 on SciFact, matching BEIR's published BM25
   (0.665).
2. **The embedding model is strong.** EmbeddingGemma-2 reaches 0.840 on SciFact; a 2026 study reports 0.742 for BGE
   (arXiv 2609.22770).
3. **The default reranker hurts on scientific claims.** With Ettin-150M, `query` scores 0.759 on SciFact, against
   0.840 for `vsearch`.
   - The cause is domain mismatch: rerankers trained on web search lose precision on scientific text (SciRet, arXiv
     2608.03860).
   - On the arXiv known items the same reranker is essential: MRR 0.936, against 0.627 for cosine alone. Author and id
     queries depend on it, because only the reranker reads the front-matter values a query names.
   - Blending reranker and cosine z-scores equally: SciFact 0.834, arXiv topical queries 0.953 (reranker alone 0.951),
     arXiv author and id queries 0.675 (reranker alone 0.889).
4. **Agent-written query expansion** (keywords, a rewrite, a HyDE passage; QMD's approach) moved the arXiv set only
   from MRR 0.936 to 0.938. It rescued one query and hurt none. That set is close to its ceiling.

5. **A fixed linear blend fixes most of it:** score = reranker logit + λ × cosine, where a candidate whose front
   matter the query names keeps the pool's best cosine. The embedding can't see front matter, so it shouldn't count
   against such a candidate. The scale stays absolute, unlike a per-query z-score.

   | Order (Ettin-150M) | SciFact nDCG@10 | NFCorpus nDCG@10 | arXiv MRR@10 | arXiv, front matter | arXiv, topical |
   |---|---|---|---|---|---|
   | reranker alone (0.8.0) | 0.760 | 0.374 | 0.936 | 0.889 | 0.951 |
   | + 20 × cosine, named exempt | 0.811 | 0.394 | **0.956** | 0.903 | 0.973 |
   | + 40 × cosine, named exempt | 0.840 | 0.393 | 0.940 | 0.903 | 0.953 |
   | + 80 × cosine, named exempt | **0.856** | 0.389 | 0.914 | 0.903 | 0.918 |
   | + 20 × cosine, none exempt | 0.811 | 0.394 | 0.920 | 0.757 | 0.973 |

   Without the exemption, author and id queries fall (0.889 → 0.757 at λ 20). With it, every set gains.
6. **A larger reranker helps a little.** Ettin-400M against 150M, reranker alone: SciFact 0.778 vs 0.760, NFCorpus
   0.387 vs 0.374. Blended with λ 40: 0.848 and 0.399.
7. **Rerankers compared (7 so far).** The decisive measure is the mean over all seven sets, plus the worst set
   relative to the best setting. "Blend" is sift's scale-free blend: the reranker's score plus `w` times the cosine,
   in units of the reranker's spread within the query's candidates, with front-matter matches exempt.

   | Reranker | Size | w | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv | Mean | Worst gap |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | **gte-reranker-modernbert-base** | 150M | 0.5 | 0.813 | 0.361 | 0.464 | 0.987 | 0.685 | 0.631 | 0.929 | **0.696** | **0.046** |
   | gte-reranker-modernbert-base | 150M | 0.75 | 0.834 | 0.368 | 0.459 | 0.989 | 0.687 | 0.625 | 0.915 | 0.697 | 0.039 |
   | mxbai-rerank-base-v2 | 494M | 0.75 | 0.859 | 0.404 | 0.441 | 0.985 | 0.624 | 0.602 | 0.913 | 0.690 | 0.091 |
   | Ettin-400M | 400M | 0.75 | 0.838 | 0.399 | 0.458 | 0.979 | 0.572 | 0.627 | 0.942 | 0.688 | 0.143 |
   | bge-reranker-v2-m3 | 568M | 0.75 | 0.829 | 0.377 | 0.432 | 0.992 | 0.711 | 0.592 | 0.849 | 0.683 | 0.102 |
   | Ettin-150M | 149M | 0.75 | 0.822 | 0.396 | 0.452 | 0.976 | 0.551 | 0.611 | 0.946 | 0.679 | 0.164 |
   | Qwen3-Reranker-0.6B | 596M | 0.75 | 0.835 | 0.403 | 0.419 | 0.983 | 0.602 | 0.613 | 0.859 | 0.674 | 0.113 |
   | mxbai-rerank-large-v1 | 435M | 0.75 | 0.830 | 0.406 | 0.437 | 0.985 | 0.447 | 0.604 | 0.870 | 0.654 | 0.268 |
   | Ettin-150M, no blend (0.8.0) | 149M | 0 | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.936 | 0.646 | 0.235 |

   - **The leader:** gte-reranker-modernbert-base, blended, is the same size as 0.8.0's reranker and the most balanced
     (never more than 0.05 below the best setting on any set). It gains 0.05 on the mean, and 0.20 on long German
     documents, for 0.007 on arXiv.
   - **The runner-up,** mxbai-rerank-base-v2, leads on scientific and medical text, at three times the size and twice
     the time per query.
   - **The multilingual rerankers** (bge-reranker-v2-m3, Qwen3-0.6B) lead on German, but lose on arXiv and FiQA.
   - **The blend helps every reranker:** the means rise by 0.03 to 0.05.

8. **Embedding models alone** (each pool's candidates by cosine, as `vsearch` ranks them). The arXiv known items run
   over a 100,000-paper subset (`arxivsub`), which holds the targets, since re-embedding all 3.2 million per model
   would take hours.

   | Embedder | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
   |---|---|---|---|---|---|---|---|---|
   | EmbeddingGemma-2 (default) | **0.840** | **0.358** | 0.314 | 0.955 | 0.534 | 0.486 | 0.806 | 0.613 |
   | Snowflake Arctic-Embed-L v2.0 | 0.704 | 0.355 | 0.451 | 0.964 | 0.539 | **0.499** | **0.843** | **0.622** |
   | bge-m3 | 0.644 | 0.313 | 0.412 | **0.969** | **0.552** | 0.436 | 0.818 | 0.592 |
   | Qwen3-Embedding-0.6B | 0.700 | 0.355 | **0.469** | 0.950 | 0.531 | 0.174 | 0.819 | 0.571 |

   - **No embedder wins broadly.** EmbeddingGemma-2 is far ahead on scientific claims and weak on financial
     questions.
   - **Qwen3-Embedding's LitSearch score is out of line** with its other sets. It's left unexplained, and isn't counted
     against the model.

9. **Embedders with rerankers.** The leading rerankers were run on every embedder's candidates. All four embedders are
   measured on the same seven sets, the arXiv subset included.

   | Embedder | gte + blend 0.5 | gte + blend 0.75 | gte alone | Ettin-150M alone (0.8.0) |
   |---|---|---|---|---|
   | EmbeddingGemma-2 | **0.700** | **0.703** | 0.669 | 0.651 |
   | Arctic-Embed-L v2.0 | 0.698 | 0.697 | 0.676 | 0.649 |
   | bge-m3 | 0.687 | 0.685 | 0.672 | 0.649 |
   | Qwen3-Embedding-0.6B | 0.671 | 0.669 | 0.647 | 0.638 |

   - **The reranker:** gte-reranker-modernbert-base with the blend is the best under every embedder. Its worst gap
     to the best setting on any set is 0.02 to 0.04.
   - **The weight:** 0.5 is the best weight, or tied for best, under every embedder.

   Per set, gte with the blend at 0.5:

   | Embedder | SciFact | NFCorpus | FiQA | GermanQuAD | MLDR-de | LitSearch | arXiv subset | Mean |
   |---|---|---|---|---|---|---|---|---|
   | EmbeddingGemma-2 | **0.813** | **0.361** | 0.464 | 0.987 | 0.685 | **0.631** | **0.960** | **0.700** |
   | Arctic-Embed-L v2.0 | 0.756 | 0.358 | 0.505 | 0.986 | **0.700** | 0.624 | 0.959 | 0.698 |
   | bge-m3 | 0.753 | 0.340 | 0.486 | 0.986 | 0.695 | 0.601 | 0.948 | 0.687 |
   | Qwen3-Embedding-0.6B | 0.762 | 0.360 | **0.518** | **0.988** | 0.695 | 0.417 | 0.959 | 0.671 |
   | *Ettin-150M alone, under EmbeddingGemma-2 (0.8.0)* | 0.760 | 0.374 | 0.463 | 0.937 | 0.481 | 0.568 | 0.976 | 0.651 |

   - **The reranker narrows the embedders' differences:** alone they span 0.571 to 0.622, with gte 0.671 to 0.700.
   - **The blend passes each embedder's strengths on.** With Ettin alone, SciFact scores 0.76 under every embedder;
     with the blend, EmbeddingGemma-2 keeps its lead on scientific claims (+0.06), and Arctic its lead on financial
     questions and long German documents (+0.04, +0.015).
   - **The embedder:** EmbeddingGemma-2 stays. Arctic is level on average (0.698 against 0.700), which wouldn't
     repay re-embedding an index. For a new index of business documents, Arctic is the alternative.

10. **Speed,** each model alone on an otherwise idle machine, with a warm-up first. Rerankers score 50 queries × 80
    candidates, half from SciFact (short abstracts) and half from MLDR (long German text). The GPU runs in bfloat16
    (5 repetitions, spread at most 1%); the CPU runs in float32 on 8 threads, over a tenth of the inputs
    (3 repetitions, spread at most 10%).

    | Reranker | Size | RTX 4090, ms per query | CPU, ms per query |
    |---|---|---|---|
    | **gte-reranker-modernbert-base** | 150M | **284** | 26,596 |
    | Ettin-150M (0.8.0) | 149M | 284 | 25,217 |
    | bge-reranker-v2-m3 | 568M | 305 | 41,511 |
    | mxbai-rerank-base-v2 | 494M | 559 | 56,657 |
    | Ettin-400M | 400M | 608 | not run |
    | Qwen3-Reranker-0.6B | 596M | 1,025 | 90,160 |

    | Embedder | RTX 4090, passages per second | RTX 4090, ms per query |
    |---|---|---|
    | EmbeddingGemma-2 | 120 | 51 |
    | Arctic-Embed-L v2.0 | 190 | 13 |
    | bge-m3 | 190 | 16 |
    | Qwen3-Embedding-0.6B | 68 | 36 |

    - **gte costs nothing in speed:** it's as fast as 0.8.0's reranker on the GPU and within 6% on the CPU.
    - **On a CPU, reranking takes about 25 seconds a query** for either. A machine without a GPU needs a
      `sift worker`, or uses `search` and `vsearch`.
    - **Query embedding is small next to reranking:** 51 ms against 284 ms. Arctic would save 38 ms a query and index
      1.6 times as fast; neither repays re-embedding an existing index.

## Decision

`query` moves to gte-reranker-modernbert-base (Apache-2.0, 150M parameters, the size of 0.8.0's reranker) and the
blend at weight 0.5, while the embedder stays EmbeddingGemma-2. On the mean over the seven sets this gains about 0.05
over 0.8.0, without re-embedding and without a bigger model.

## Running the benchmarks safely

Two crashes interrupted the runs on 2026-10-20:

- **20:10, bugcheck 0x119 (VIDEO_SCHEDULER_INTERNAL_ERROR):** four CUDA processes held about 22 of 24 GB at 100%. Each
  sift process sizes its GPU budget from the memory free when it starts, so several of them together fill the GPU.
- **22:07, bugcheck 0xF7 (DRIVER_OVERRAN_STACK_BUFFER):** this time a single GPU job was running. The machine's CPU is
  known to be unstable under long load.

The harness therefore runs:

- one GPU job at a time;
- with CPU threads capped at 8;
- with the most decisive work first;
- resuming every step after a crash (`D:\sift-bench\scripts\queue_rest.ps1`).

Not run: rerankers above 1B parameters (Ettin-1B, mxbai-rerank-large-v2, Qwen3-Reranker-4B, zerank-2). sift aims
for a small footprint, and the 150M leader already ties the next-best models at three to four times its size.
