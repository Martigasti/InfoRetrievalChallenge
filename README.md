# Scientific Paper Citation Recommendation — IR Starter Kit

A retrieval system for recommending scientific paper citations. Given a query paper, the system returns a ranked list of 100 candidate documents from a corpus of 20,000 papers. The primary evaluation metric is **NDCG@10**, with MAP and Recall@100 as complementary metrics.

---

## Task Overview

- **Input**: A query paper (title, abstract, body sections)
- **Output**: A ranked list of 100 documents from the corpus
- **Corpus size**: 20,000 scientific papers
- **Evaluation**: NDCG@10 (primary), MAP, Recall@100
- **Platform**: CodaBench

---

## Best Model: `bge_weighted_rrf.py`

The top-performing approach combines three complementary retrievers using **Weighted Reciprocal Rank Fusion (RRF)**. Each retriever captures a different aspect of relevance, and their ranked lists are merged into a single ranking.

### Retrievers

#### 1. SPECTER2 (Domain-Specific Dense Retrieval)
- **Model**: `allenai/specter2_base` with the `proximity` adapter
- **Purpose**: Semantic similarity specialized for scientific papers. SPECTER2 is pre-trained on citation graphs — papers that cite each other are pulled closer in embedding space.
- **Pooling**: CLS token (first token of the output)
- **Input format**: `"title [SEP] abstract + body_chunks"`
- **Why it helps**: Captures the high-level scientific topic and domain of the paper better than general-purpose models.

#### 2. BGE-large-en-v1.5 (Strong General Dense Retrieval)
- **Model**: `BAAI/bge-large-en-v1.5`
- **Purpose**: General-purpose semantic similarity, top-ranked on the MTEB benchmark. Much stronger than smaller models like MiniLM.
- **Pooling**: Mean pooling with L2 normalization
- **Input format**: Enriched text — `title + abstract + first 3 body chunks`
- **Why it helps**: Captures fine-grained semantic similarity across the full text. Complements SPECTER2 by using different training data and architecture.

#### 3. BM25 (Lexical Retrieval)
- **Algorithm**: BM25Okapi
- **Purpose**: Exact keyword matching. Retrieves papers that share specific terminology with the query.
- **Text preprocessing**: NLTK Porter stemming + English stopword removal
- **Input format**: Same enriched text as BGE
- **Why it helps**: Dense models can miss papers that share very specific technical terms or named entities. BM25 excels at exact-match retrieval.

### Enriched Text Representation

All retrievers use an enriched representation of each paper that goes beyond just the title and abstract:

```
title + abstract + first 3 body section chunks (≥50 chars each)
```

This gives retrievers access to the paper's methodology and key content, not just its summary.

### Weighted Reciprocal Rank Fusion

RRF merges multiple ranked lists without requiring score normalization. For each document `d`:

```
score(d) = w_specter / (k + rank_specter(d))
         + w_bge    / (k + rank_bge(d))
         + w_bm25   / (k + rank_bm25(d))
```

- **k = 10**: Controls rank sensitivity. Lower k means top-ranked documents are weighted much more heavily.
- **Dense weights (SPECTER2, BGE) = 1.0**: Fixed.
- **BM25 weight**: Grid-searched over [0.3, 0.5, 0.7, 1.0] on the public query set to find the optimal balance.
- Each retriever fetches **top-200 candidates** before fusion, giving RRF a wide pool to rerank from.
- Final output is the **top-100** documents by fused score.

---

## Other Scripts

| Script | Description | Local NDCG@10 |
|---|---|---|
| `tfidf_baseline.py` | TF-IDF sparse retrieval | ~0.45 |
| `dense_baseline.py` | MiniLM dense retrieval | ~0.50 |
| `specter2_retrieval.py` | SPECTER2 only | ~0.51 |
| `specter2_minilm_rrf.py` | 2-way RRF: SPECTER2 + MiniLM + enriched text | 0.5332 |
| `three_way_rrf.py` | 3-way RRF: SPECTER2 + MiniLM + BM25, k=10 | 0.5786 |
| `bge_weighted_rrf.py` | 3-way Weighted RRF: SPECTER2 + BGE-large + BM25 | TBD |

Scripts that did not improve over the dense baseline:
- `dense_rerank.py` — MiniLM + MS-MARCO cross-encoder reranking (0.43): cross-encoders trained on web queries hurt paper-to-paper similarity.
- `dense_interpolated_rerank.py` — Score interpolation with cross-encoder (0.50): no gain over pure dense.
- `scibert_rerank.py` — SciBERT embeddings (0.27): SciBERT is a masked LM, not a sentence encoder.

---

## Setup

```bash
pip install sentence-transformers rank-bm25 nltk adapters
python -m nltk.downloader stopwords punkt
```

## Running the Best Model

```bash
python scripts/bge_weighted_rrf.py
```

Output is saved to `submissions/bge_weighted_rrf.json` for CodaBench submission.

## Running the 3-way RRF (previous best)

```bash
python scripts/three_way_rrf.py
```

Output saved to `submissions/three_way_rrf.json`.

---

## Project Structure

```
starter_kit/
├── data/
│   ├── corpus.parquet        # 20k scientific papers
│   ├── queries.parquet       # Public evaluation queries
│   └── qrels.json            # Relevance judgements
├── held_out_queries.parquet  # CodaBench submission queries
├── scripts/
│   ├── bge_weighted_rrf.py   # Best model
│   ├── three_way_rrf.py      # Previous best
│   ├── specter2_minilm_rrf.py
│   ├── specter2_retrieval.py
│   ├── dense_baseline.py
│   └── tfidf_baseline.py
├── submissions/              # JSON outputs for CodaBench
├── utils.py                  # Shared evaluation + data loading utilities
└── README.md
```
