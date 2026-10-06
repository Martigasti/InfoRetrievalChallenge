# Citation Recommendation — Progression Report

Competition: CodaBench Scientific Article 2025–2026. Primary metric: **NDCG@10**.
Corpus: 20k papers. Public eval set: 100 queries. Noise floor ≈ ±0.005 NDCG@10.
Hardware: 8 GB VRAM (WSL).

---

## 1. Starting point — dense baseline

A single dense encoder (bge-large-en-v1.5 or similar) with `title + abstract` as the
document representation, cosine similarity retrieval, no fusion.

**NDCG@10 ≈ 0.50.**

The gap to the top of the leaderboard was entirely in *ordering*: MRR@10 was already
high (≈0.72), meaning a relevant paper was usually at rank 1, but the 2nd–5th relevant
papers were scattered between ranks 6 and 30.

---

## 2. Improvement 1 — SPECTER2 with the proximity adapter

Swapped the generic dense model for [`allenai/specter2_base`](https://huggingface.co/allenai/specter2_base)
loaded with the `proximity` adapter. SPECTER2 is trained on citation triplets, so
"papers that cite each other" is exactly its objective.

Document format: `"title [SEP] abstract"`, CLS pooling, L2-normalised.

**NDCG@10 ≈ 0.54** (+0.04 over the dense baseline).

---

## 3. Improvement 2 — Hybrid with BM25 (weighted RRF)

Dense models miss exact lexical matches (author surnames, dataset names, method
acronyms). Added BM25Okapi over a Porter-stemmed, stop-word-filtered corpus and
fused with SPECTER2 via weighted Reciprocal Rank Fusion:

```
score(d) = Σ_r  w_r / (k + rank_r(d))          k = 10
```

Grid-searching `w_bm25 ∈ {0.3, 0.5, 0.7, 1.0}` with dense fixed at 1.0 settled on
`w_bm25 = 0.5` — BM25 corrects the dense ordering without overwhelming it.

**NDCG@10 ≈ 0.568.**

---

## 4. Improvement 3 — Body-chunk enrichment

`title + abstract` throws away the entire paper body. Added the first 6 body chunks
(min 50 chars each) to both the SPECTER2 and BM25 document text. SPECTER2 still only
sees the first 512 tokens, but BM25 now indexes method names, dataset names, and
equation references that never appear in abstracts.

**NDCG@10 ≈ 0.577.**

---

## 5. Improvement 4 — Adding BGE-large as a second dense retriever

SPECTER2 and BGE disagree enough to be complementary: SPECTER2 captures citation
proximity, BGE captures general semantic similarity. 3-way weighted RRF
(`SPECTER2 = 1.0`, `BGE = 1.0`, `BM25 = 0.5`) — this is
[scripts/bge_weighted_rrf.py](scripts/bge_weighted_rrf.py).

**NDCG@10 = 0.5854 locally, 0.60 on CodaBench.**

---

## 6. Improvement 5 — Wider retrieval pool + more body chunks

The previous configuration retrieved the top 200 candidates per retriever and
used 3 body chunks. Two targeted changes
([scripts/wider_rrf.py](scripts/wider_rrf.py)):

- **RETRIEVAL_TOP_K 200 → 300**: more candidates enter RRF, raising the recall
  ceiling.
- **BODY_CHUNKS 3 → 6**: more document surface for both BGE and BM25 to match
  on — method names, dataset names, and equation references that only appear
  deep in the body.

Grid-searching `w_bm25 ∈ {0.3, 0.5, 0.7, 1.0}` shifted the optimal BM25
weight from 0.5 to 1.0 — with more body text BM25 becomes as informative as
the dense retrievers, so equal weighting is optimal.

**NDCG@10 = 0.5873 locally. Current best.**

---

## 7. Two things that did NOT work (and why)

### Jina-embeddings-v3 (replacing BGE)

Rationale: Jina v3 supports a 2048-token context (4× BGE's 512) and ranks near the
top of MTEB, so it should read the whole paper body and beat BGE as the generic
dense retriever. Ran fp16 on GPU, batch size 4, task=`retrieval.passage`.

Result: **local NDCG@10 = 0.5731, CodaBench = 0.58** — *worse* than BGE.

Why it failed: the grid search picked `w_bm25 = 1.0`, meaning the Jina signal was
so weak that BM25 had to compensate. Jina v3 is optimised for web/multilingual
retrieval — the domains on the MTEB leaderboard. Paper-to-paper citation
similarity is a *different* task: what matters is whether two papers would be
cited together, not whether they are topically similar. A longer context window
doesn't help if the representation is pointed at the wrong objective.

### SciNCL as a 4th retriever

Rationale: [`malteos/scincl`](https://huggingface.co/malteos/scincl) is also
trained on scientific citation data, just with a different negative-sampling
strategy than SPECTER2. Added it as a 4th source in the RRF fusion
(`SPECTER2 + BGE + SciNCL + BM25`), grid-searched `w_scincl ∈ {0.5, 1.0, 1.5}`.

Result: best was `w_scincl = 0.5`, **delta = −0.0038** vs the 3-way baseline —
inside the ±0.005 noise floor, i.e. a null result.

Why it failed: SciNCL and SPECTER2 are trained on the *same* task with the
*same* signal (citation graphs over S2ORC/similar). They agree on which papers
are relevant and disagree mostly on micro-ordering. RRF smooths out micro-ordering
differences rather than exploiting them, so a second citation-trained encoder
adds almost no new information on top of SPECTER2. Fusion gains come from
*disagreement* (SPECTER2 ≠ BGE ≠ BM25), not from stacking near-duplicates.
