# Citation Recommendation with Hybrid Retrieval and Rank Fusion

Given a scientific paper, retrieve the papers it is most likely to cite from a corpus of 20,000 articles.
Built for the **CodaBench Scientific Article Retrieval challenge (2025–2026)**, M1 Data Science, Université Paris-Saclay.

**Final system:** SPECTER2 (citation-trained) + BGE-large (general semantic) + BM25 (lexical), combined with Weighted Reciprocal Rank Fusion.
**Result:** NDCG@10 = **0.587** on the public queries, **0.61** on the CodaBench leaderboard (dense baseline: 0.50).

Team: Martin Leiva, Javier Peña Castaño, Raphael Leonardi.

---

## Task

| | |
|---|---|
| Input | A query paper (title, abstract, body sections, domain) |
| Output | Top-100 ranked candidates from a 20k-paper corpus |
| Primary metric | NDCG@10 (also MAP, Recall@100, MRR@10) |
| Public queries | 100, with gold citations (noise floor ≈ ±0.005 NDCG@10) |
| Constraint | 8 GB of VRAM, so large cross-encoders and LLM rerankers are impractical |

The key point is that **citation proximity is not the same as topical similarity**: two papers on the same topic may never cite each other, while a paper often cites a methods paper from a different field.

## Pipeline

```text
                ┌─ SPECTER2 + proximity adapter (CLS, 768d) ─┐
query paper ────┼─ BGE-large-en-v1.5 (mean pool, 1024d) ─────┼─> Weighted RRF ─> top-100
                └─ BM25Okapi (Porter stemming, stopwords) ───┘
```

- **Enriched text**: `title + abstract + first N body chunks` (≥50 chars each). Dense models see the first 512 tokens; BM25 indexes the whole text, so it catches method names, datasets and equation references that only appear in the body.
- **Weighted RRF**: `score(d) = Σ_r w_r / (k + rank_r(d))` with `k = 10`, dense weights fixed at 1.0 and the BM25 weight grid-searched.
- **Wider pool**: each retriever returns its top 300 before fusion.

## Results

Each step was kept only if it improved NDCG@10 on the public queries. [REPORT.md](REPORT.md) explains the reasoning behind every step.

| Step | Script | NDCG@10 |
|---|---|---|
| TF-IDF baseline | `tfidf_baseline.py` | ~0.45 |
| Dense baseline (MiniLM / BGE, title + abstract) | `dense_baseline.py` | ~0.50 |
| SPECTER2 with proximity adapter | `specter2_retrieval.py` | ~0.54 |
| + BM25, weighted RRF | `three_way_rrf.py` | 0.568 |
| + body-chunk enrichment | | 0.577 |
| + BGE-large as second dense retriever | `bge_weighted_rrf.py` | 0.585 (CodaBench 0.60) |
| + wider pool (top-300) and 6 body chunks | `wider_rrf.py` | **0.587 (CodaBench 0.61)** |

### What did not work

| Experiment | Script | Takeaway |
|---|---|---|
| MS-MARCO cross-encoder reranking | `dense_rerank.py`, `wider_rrf_rerank.py` | Rerankers trained on web queries hurt paper-to-paper ranking (0.43 on the dense baseline) |
| Jina-embeddings-v3 instead of BGE | `jina_v3_rrf.py` | 0.573: a longer context does not fix a training objective aimed at the wrong task |
| SciNCL as a 4th retriever | `scincl_4way_rrf.py` | −0.004 (noise): it agrees with SPECTER2, and fusion gains come from *disagreement* |
| SciBERT embeddings | `scibert_rerank.py` | 0.27: a masked LM, not a sentence encoder |

Also explored: HyDE query expansion with a small LLM (`hyde_weighted_rrf.py`), doc2query-T5 document expansion (`doc2query_*.py`), LightGBM LambdaRank learned fusion (`lightgbm_fusion.py`), GPL unsupervised domain adaptation (`gpl_finetune_rrf.py`), BM25 pseudo-relevance feedback (`bge_bm25_prf.py`), alternative encoders (E5, Snowflake Arctic, Yuan), and a light post-hoc rerank on title overlap and domain match (`light_rerank.py`).

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

The challenge data is distributed through CodaBench and is **not included** in this repository. Put it under `data/`:

```text
data/
├── corpus.parquet       # 20k papers
├── queries.parquet      # 100 public queries
└── qrels.json           # gold citations for the public queries
```

`held_out_queries.parquet` (the leaderboard queries) is included.

## Usage

```bash
python scripts/wider_rrf.py          # best configuration
python scripts/bge_weighted_rrf.py   # 3-way weighted RRF
```

Each script evaluates on the public queries (NDCG@10, MAP, Recall@100), then writes leaderboard predictions to `submissions/`.

## Repository structure

```text
├── scripts/                 # one self-contained script per experiment
├── utils.py                 # data loading, text formatting, chunking, metrics
├── notebooks/challenge.ipynb  # data exploration and baselines (course starter notebook)
├── docs/                    # presentation slides (Beamer) and speaker notes
├── REPORT.md                # step-by-step progression report
└── held_out_queries.parquet
```
