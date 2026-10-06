# Presentation Script — Citation Recommendation with Hybrid Retrieval and Rank Fusion

**Total time: ~10–11 minutes**

---

## Speaker 1 — Martin: Intro, pipeline overview (~2 min)

### Slide 1 — Title

"We're going to present our system for the CodaBench Scientific Article citation recommendation challenge. The task: given a query paper, retrieve its 100 most likely citations from a corpus of 20 000 papers. The metric is NDCG@10."

---

### Slide 2 — Pipeline Overview (diagram)

"Our final pipeline has three parallel retrievers feeding into a single fusion step.
- SPECTER2 with the proximity adapter — a citation-trained encoder.
- BGE large v1.5 — a general-purpose semantic encoder.
- BM25Okapi — a classical lexical retriever.
All three run independently on the query, then their ranked lists are combined with Weighted Reciprocal Rank Fusion."

---

### Slide 3 — Pipeline Overview: What Each Component Does

"Quickly on the technical side:
- SPECTER2 uses CLS pooling, 768 dimensions, trained specifically on citation triplets.
- BGE uses mean pooling, 1024 dimensions, general semantic similarity.
- BM25 has no token cap — it indexes the full enriched text including body chunks.
The RRF formula is shown at the bottom: for each document we sum the weighted inverse of its rank from each retriever, with k=10."

---

### Slide 4 — Pipeline Overview: Why Three Different Retrievers?

"The key motivation is that these three retrievers answer different questions and therefore disagree on which papers are relevant.
- BM25 catches exact string matches — tool names, dataset names, author surnames.
- SPECTER2 captures what a scientist would cite based on citation graph proximity.
- BGE captures semantic similarity even across different vocabulary.

The biology/PyTorch example makes it concrete: a biology paper citing PyTorch won't be in the same embedding neighbourhood as a PyTorch paper, but BM25 matches on the string. Fusion is only valuable when they disagree."

---

## Speaker 2 — Javier: Ablation study and results (~3 min 30s)

### Slide 5 — Ablation Study: Contribution of Each Component

"Here is the full progression of our system. We started from the TF-IDF baseline at 0.484. Adding a dense encoder brought us to 0.50. The biggest single jump was swapping to SPECTER2: +0.040 to 0.540. Adding BM25 fusion gave another +0.028. Body chunk enrichment and adding BGE gave smaller incremental gains. Final local score: 0.587."

---

### Slide 6 — Ablation Study: Tuning Decisions

"We had two main tuning decisions.
For BM25 weight, we grid-searched over {0.3, 0.5, 0.7, 1.0}. With 3 body chunks the optimal was 0.5 — BM25 is weaker than the dense retrievers. With 6 chunks it shifted to 1.0 — more indexed text makes BM25 as informative as the dense ones.
For RRF k, we tried {10, 60, 100}. k=10 preserves top-rank differences best for our candidate pool size.
We didn't use learned weights: 100 queries is not enough to fit weights without overfitting."

---

### Slide 7 — Results: NDCG@10 Progression

"The chart shows the same progression visually. Each step is a deliberate, tested change — no step was added without a measurable improvement on the public leaderboard. Our best local score is 0.587 and our best CodaBench submission reached 0.61."

---

## Speaker 3 — Raphael: Error analysis, surprises, learnings (~4 min 30s)

### Slide 8 — Error Analysis: What Did Not Work

"We tried five other directions — none of them helped.

**Cross-encoder reranking** (`scripts/bge_rerank.py`): A cross-encoder sits after RRF and re-scores the top 20 candidates by reading the query and each document together as one input — much more expressive than the bi-encoders. We tried bge-reranker-v2-m3, SciBERT, and MS-MARCO rerankers. No lift. These rerankers are trained on web search pairs (short query → web page), not paper-to-paper citation matching, so their extra expressiveness was pointed at the wrong task.


**BM25 pseudo-relevance feedback** (`scripts/bge_bm25_prf.py`): Take the top fused results, extract their most distinctive terms, and expand the BM25 query with them (RM3-style). Classic IR trick for vocabulary mismatch. Failed because the top results contain false positives that share vocabulary with the query — expanding toward them caused query drift, pulling retrieval away from true citations.

**Jina embeddings v3** (`scripts/jina_v3_rrf.py`): Replaced BGE with Jina v3 — top of MTEB, 2048-token context. Result: −0.012 NDCG@10. Jina is optimised for web/multilingual retrieval. Longer context can't compensate for a wrong training objective.

**SciNCL as 4th retriever** (`scripts/scincl_4way_rrf.py`): SciNCL is also citation-trained like SPECTER2 but with different negative sampling. Adding it gave −0.004 (within noise). It agrees with SPECTER2 on which papers are relevant and only disagrees on micro-ordering — RRF smooths that out rather than exploiting it. Fusion gains require genuine disagreement."

---

### Slide 9 — Surprises We Encountered

"A few things genuinely surprised us during the project.
- MTEB ranking didn't predict performance: Jina v3 is higher on MTEB than BGE but worse on our task.
- BM25 weight depended on how many chunks we indexed — we didn't anticipate that interaction before running the grid search.
- Two citation-trained models were near-redundant: we expected more citation signal = better; we got a null result.
- Simple RRF beat a learned ranker: LightGBM had the right objective and richer features, but 100 queries wasn't enough data."

---

### Slide 10 — What We Learned

"To summarise the four main takeaways:
1. Task alignment matters more than model size or leaderboard ranking. SPECTER2 at 110M parameters beat every general encoder we tried.
2. Fusion gains come from disagreement. Three orthogonal signals beat four overlapping ones.
3. More text helps, up to a point. Body chunks improved BM25 recall; returns flattened beyond 6 chunks.
4. Weighted RRF is simple and effective. One weight per retriever, no score normalisation, no overfitting risk."

---

### Slide 11 — Thank you

"Thank you. Our final result: 0.587 locally, 0.61 on CodaBench, using SPECTER2, BGE large, and BM25 with Weighted Reciprocal Rank Fusion."
