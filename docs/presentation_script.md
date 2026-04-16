# Presentation Script — Citation Recommendation with Hybrid Retrieval

Total target: ~11–12 minutes. Timings are approximate.

---

# SPEAKER 1 — Intro, data, baselines (Slides 1–6, ~4 min)

---

## Slide 1 — Title (15s)

Hello everyone. Today we'll present our work on the CodaBench citation
recommendation competition: predicting which papers a scientific article should
cite, using a hybrid retrieval pipeline with rank fusion.

---

## Slide 2 — The Challenge (40s)

Citation recommendation is not topical similarity. Given a query paper, we need
to retrieve the papers it would actually cite from a corpus of 20,000 articles.
The task is **citation proximity** — would paper A cite paper B? Two papers can
cover the same topic without citing each other, and a paper can cite a
foundational methods paper from a completely different field.

One constraint: 8 GB of VRAM on WSL rules out large cross-encoders.

---

## Slide 3 — Dataset & Evaluation (35s)

The corpus is 20,000 papers in Parquet format. 100 public queries with
ground-truth citations for local evaluation, held-out queries for CodaBench.

The primary metric is NDCG@10 — the leaderboard ranking metric. With only 100
queries, the noise floor is about ±0.005, so small differences need careful
interpretation.

---

## Slide 4 — TF-IDF Baseline (40s)

We start with TF-IDF cosine similarity on title plus abstract. NDCG@10 is 0.484.

The important number is Recall@100 at 0.75 — three quarters of relevant papers
are already findable by lexical overlap alone. The weak link is ranking, not
retrieval. This sets the bar every neural method has to beat.

---

## Slide 5 — Dense Baseline (45s)

Replacing TF-IDF with BGE-large-en-v1.5 brings NDCG@10 to 0.50. MRR@10 is
already 0.72 — rank-1 is usually correct. The problem is the 2nd through 5th
relevant papers, which land scattered between ranks 6 and 30. The bottleneck is
ordering quality, not candidate generation. This shapes every decision that
follows.

---

# SPEAKER 2 — Improvements 2–5 + final system (Slides 7–11, ~4 min)

---

## Slide 6 — SPECTER2 (55s)

The first improvement: replace the generic encoder with a **citation-trained**
model. SPECTER2 is trained on citation triplets from Semantic Scholar — its
objective is "papers that cite each other should be close in embedding space,"
exactly our task. We used the base model with the proximity adapter.

This single swap brought NDCG@10 from 0.50 to 0.54 — a gain of 0.04, the
largest single improvement in the progression. Matching the training objective
to the task beats picking whatever tops a generic benchmark.



## Slide 7 — BM25 + RRF (1 min)

Dense encoders miss exact lexical matches: author surnames, dataset names like
ImageNet, method acronyms like BERT. So we added BM25Okapi — Porter-stemmed
and stopword-filtered — as a sparse retriever.

To combine the two ranking lists we used Reciprocal Rank Fusion: for each
document, sum the inverse of its rank plus a constant across all retrievers,
weighted per retriever. RRF uses only ranks, so there's no need to normalise
scores across heterogeneous retrievers.

Grid search found BM25 weight 0.5 to be optimal. NDCG@10: 0.568.

---

## Slide 8 — Body Chunks (45s)

So far every model only saw title and abstract. Papers have a body full of
method names and dataset references that never appear in abstracts. We appended
the first body chunks — at least 50 characters each — to the document
representation for both SPECTER2 and BM25. SPECTER2 only sees the first 512
tokens, but BM25 indexes everything. NDCG@10: 0.568 → 0.577.

---

## Slide 9 — BGE as Second Dense Retriever (50s)

We added BGE-large-en-v1.5 as a second dense retriever. The insight is that
**fusion gains come from disagreement**. SPECTER2 captures citation proximity.
BGE captures general semantic similarity. BM25 captures lexical overlap. Three
genuinely different signals.

With 3-way weighted RRF — all weights 1.0 except BM25 at 0.5 — NDCG@10
reached 0.585 locally and 0.60 on CodaBench.

---

## Slide 10 — Wider Pool + More Chunks (35s)

Two hyperparameter changes: retrieval pool from 200 to 300, body chunks from 3
to 6. With more body text, the optimal BM25 weight shifted to 1.0 — equal with
the dense retrievers. Final score: 0.587. Diminishing returns, but consistent.

---


# SPEAKER 3 — Failures, results, takeaways (Slides 12–15, ~3m30s)

## Slide 11 — System Architecture (35s)

Here's what the full system looks like with all components in place. Three
independent retrievers — SPECTER2, BGE-large, and BM25Okapi — each contribute
their top-300 candidates, fused with equal weights via Weighted RRF into a
final top-100.

---


## Slide 12 — What Didn't Work (1 min 20s)

Two failures worth explaining.

**Jina-embeddings-v3** — 2048-token context, near the top of MTEB. The
hypothesis: longer context lets it read the full body. Result: NDCG@10 dropped
to 0.573. Jina is optimised for web and multilingual retrieval; citation
proximity is a different task. Longer context doesn't help when the
representation is aimed at the wrong objective.

**SciNCL as a 4th retriever** — also citation-trained, like SPECTER2. Delta:
−0.004, inside the noise floor. SciNCL and SPECTER2 agree on which papers are
relevant — RRF smooths out their micro-differences rather than exploiting them.
A second citation-trained encoder adds nothing.

The lesson: fusion requires genuine diversity in signal, not just more
retrievers.

---

## Slide 13 — NDCG@10 Progression (20s)

Here's the full progression. TF-IDF sets the bar. Dense beats it narrowly. The
steepest jump comes from the citation-trained model. Each subsequent step added
a different signal, and the curve flattens as easy gains run out.

---

## Slide 14 — Key Takeaways (45s)

Four takeaways. **Task-aligned models matter most** — SPECTER2 gave the biggest
single jump; generic MTEB winners underperformed. **Fusion gains come from
disagreement** — adding a redundant model adds nothing. **More text surface
helps up to a point** — body chunks helped BM25, but returns diminished beyond
6. **Weighted RRF is simple and effective** — one hyperparameter per retriever,
no learned weights to overfit.

---

## Slide 15 — Thank You (15s)

Best local NDCG@10: 0.587. Best CodaBench: 0.60. Three retrievers, one fusion
step, no reranker. Thank you — happy to take questions.
