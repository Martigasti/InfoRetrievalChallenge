# Presentation Script — Citation Recommendation with Hybrid Retrieval

Total target: ~15 minutes. Timings are approximate.

---

## Slide 1 — Title (30s)

Hello everyone. Today I'll present my work on the CodaBench Scientific Article
citation recommendation competition. The goal is to predict which papers a
given scientific article should cite, using a hybrid retrieval pipeline that
combines dense and sparse retrieval with rank fusion.

---

## Slide 2 — The Challenge (1 min)

So what is citation recommendation? Given a scientific paper — the query — we
need to retrieve the papers it should cite from a corpus of 20,000 articles.

This is NOT the same as topical similarity. Two papers can be about the same
topic without citing each other, and a paper can cite a foundational methods
paper from a completely different field. The task is specifically about
**citation proximity** — would paper A cite paper B?

The corpus spans multiple domains: biology, medicine, computer science,
physics, chemistry, and more. Each paper comes with a title, abstract, body
text, and metadata.

One important constraint: I'm working with 8 GB of VRAM on a WSL machine,
which rules out large cross-encoders and billion-parameter models.

---

## Slide 3 — Dataset & Evaluation (1 min)

The data consists of 20,000 papers in Parquet format. We have 100 public
queries with ground-truth citation lists for local evaluation, and a set of
held-out queries without labels — those are what we submit to CodaBench.

The primary metric is NDCG@10 — Normalized Discounted Cumulative Gain at rank
10. This is what the leaderboard uses to rank submissions. We also track
Recall, Precision, MRR, and MAP for diagnostic purposes.

With only 100 queries, the noise floor is about plus or minus 0.005 NDCG@10,
which means small improvements need to be interpreted carefully.

---

## Slide 4 — System Architecture (1 min)

Before diving into the progression, let me show you the final architecture.
The system uses three independent retrievers: SPECTER2 with the proximity
adapter for citation-aware dense retrieval, BGE-large for general semantic
similarity, and BM25Okapi for exact lexical matching. Each retriever returns
its top-300 candidates, and they are fused using Weighted Reciprocal Rank
Fusion with equal weights. The fused list is truncated to 100 results for
submission.

Now let me walk through how we got here, step by step.

---

## Slide 5 — Dense Baseline (1 min)

The starting point is a single dense encoder — BGE-large-en-v1.5 — encoding
title plus abstract, with cosine similarity retrieval and no fusion.

This gives an NDCG@10 of about 0.50.

Looking at the diagnostics, MRR@10 was already around 0.72, meaning the
rank-1 result was usually correct. The problem was not finding the most
relevant paper — it was the ordering of the 2nd through 5th relevant papers,
which were scattered between ranks 6 and 30. So the bottleneck was ranking
quality, not candidate generation.

---

## Slide 6 — SPECTER2 (1.5 min)

The first improvement was replacing the generic encoder with a
**citation-trained** model. SPECTER2, from the Allen Institute, is trained
on citation triplets from Semantic Scholar. Its training objective is literally
"papers that cite each other should be close in embedding space" — exactly
our task.

I used the base model with the proximity adapter, which is specifically tuned
for citation proximity. The input format is "title [SEP] abstract" with CLS
pooling and L2 normalisation.

This single change brought NDCG@10 from 0.50 to 0.54 — a gain of 0.04,
which turned out to be the single largest improvement in the entire
progression. The lesson: choosing a model whose training objective matches
your task matters more than choosing the highest-ranked model on a generic
benchmark.

---

## Slide 7 — BM25 + RRF (2 min)

Dense encoders are great at capturing semantic similarity, but they miss
exact lexical matches — author surnames, dataset names like "ImageNet",
method acronyms like "BERT" or "ResNet". If the query paper mentions a
specific method by name, and a corpus paper describes that method, a
lexical match is very strong evidence of citation relevance.

So I added BM25Okapi as a sparse retriever. The corpus text is
Porter-stemmed and stopword-filtered. To combine the two ranking lists,
I used Reciprocal Rank Fusion, which is a simple but effective fusion
method. For each document, you sum the inverse of its rank (plus a
constant k) across all retrievers, weighted by a per-retriever weight.

The key advantage of RRF over score-level fusion is that it only uses
ranks, not raw scores, so you don't need to worry about score
normalisation across heterogeneous retrievers.

I grid-searched the BM25 weight and found 0.5 to be optimal — BM25
corrects the dense ordering without overwhelming it. This brought
NDCG@10 to 0.568.

---

## Slide 8 — Body Chunks (1.5 min)

Up to this point, all models only saw the title and abstract. But papers
have a full body that contains method names, dataset names, and equation
references that never appear in abstracts.

The corpus includes body text split into chunks with metadata. I appended
the first body chunks — at least 50 characters each — to the document
representation for both SPECTER2 and BM25.

SPECTER2 truncates at 512 tokens, so it only sees the beginning of the
appended text. But BM25 indexes everything, so it now picks up terms
that were invisible before. This gave us NDCG@10 of 0.577.

---

## Slide 9 — BGE as Second Dense Retriever (1.5 min)

At this point I had SPECTER2 and BM25. Both are good, but they capture
overlapping information — a paper about a topic will both be semantically
close and share terminology.

I added BGE-large-en-v1.5 as a second dense retriever. The key insight
is that **fusion gains come from disagreement**. SPECTER2 captures
citation proximity — "these papers cite each other." BGE captures
general semantic similarity — "these papers are about the same topic."
BM25 captures lexical overlap. These are three genuinely different
signals.

With 3-way weighted RRF (SPECTER2 = 1.0, BGE = 1.0, BM25 = 0.5),
NDCG@10 reached 0.585 locally and 0.60 on CodaBench.

---

## Slide 10 — Wider Pool + More Chunks (1 min)

The last successful improvement was a hyperparameter adjustment. I
increased the retrieval pool from 200 to 300 candidates per retriever,
and the number of body chunks from 3 to 6.

With more body text available, the grid search shifted the optimal BM25
weight from 0.5 to 1.0 — meaning BM25 is now weighted equally with the
dense retrievers. This makes sense: with 6 chunks instead of 3, BM25
has enough textual surface to be as discriminative as the dense models.

The improvement is small — 0.585 to 0.587 — but consistent. At this
point we're deep in diminishing returns territory.

---

## Slide 11 — What Didn't Work (2 min)

Two experiments that failed, and the lessons they taught.

**Jina-embeddings-v3.** This model supports 2048 tokens — four times
BGE's context — and ranks near the top of MTEB. The hypothesis was that
it could read the full paper body and outperform BGE. The result: NDCG@10
dropped to 0.573. The grid search pushed BM25 weight to 1.0, meaning the
Jina signal was so weak that BM25 had to compensate. The problem is that
Jina is optimised for web and multilingual retrieval — the tasks on the
MTEB leaderboard. Citation proximity is a different task. A longer context
window doesn't help when the representation is pointed at the wrong
objective.

**SciNCL as a 4th retriever.** SciNCL is also trained on citation data,
like SPECTER2, but with a different negative sampling strategy. The result
was a delta of minus 0.004 — within the noise floor. The reason: SciNCL
and SPECTER2 are trained on the same signal. They agree on which papers
are relevant and only disagree on micro-ordering. RRF smooths out those
micro-differences rather than exploiting them, so a second citation-trained
encoder adds essentially no new information.

The lesson from both failures: effective fusion requires genuine diversity
in the signal each retriever provides.

---

## Slide 12 — NDCG@10 Progression (30s)

Here's the full progression chart. You can see the steepest gain came from
switching to SPECTER2 — a task-aligned model. Each subsequent improvement
added a new type of signal: lexical matching, more text surface, a second
dense perspective, and wider candidate pools. The curve flattens as we
exhaust the easy gains.

---

## Slide 13 — Key Takeaways (1 min)

Four takeaways from this project.

First, **task-aligned models matter most**. SPECTER2, trained specifically
on citation triplets, gave the biggest single improvement. Generic
leaderboard winners like Jina underperformed.

Second, **fusion gains come from disagreement**. SPECTER2, BGE, and BM25
each capture a different signal. Adding a model that agrees with an
existing one adds nothing.

Third, **more text surface helps, up to a point**. Body chunks improved
BM25 substantially, but beyond 6 chunks the returns diminished.

Fourth, **Weighted RRF is simple and effective**. One hyperparameter per
retriever, robust to tuning, and no learned weights that could overfit
on just 100 queries.

---

## Slide 14 — Thank You (30s)

To summarise: our best local NDCG@10 is 0.587, and our best CodaBench
score is 0.60, achieved with a 3-way Weighted Reciprocal Rank Fusion of
SPECTER2, BGE-large, and BM25Okapi. Thank you — I'm happy to take
questions.
