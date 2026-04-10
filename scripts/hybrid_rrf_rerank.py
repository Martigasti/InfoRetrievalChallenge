"""
Hybrid retrieval pipeline: BM25 + Dense → RRF → Cross-Encoder re-ranking.

Architecture (from Class 4 slides):
  Stage 1: Parallel retrieval — BM25 top-100 + Dense top-100
  Stage 2: RRF fusion — combine both ranked lists
  Stage 3: Cross-Encoder re-rank — accurate re-ranking of top candidates

Outputs predictions for the held-out queries to submissions/.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_queries, load_corpus, load_qrels, format_text,
    load_embeddings, evaluate,
)

TOP_K = 100
RRF_K = 60  # standard RRF constant (Cormack et al., 2009)
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


# ── Helpers ──────────────────────────────────────────────────

def rrf_fuse(ranked_lists: list[dict[str, list[str]]], k: int = 60, top_n: int = 100) -> dict[str, list[str]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    RRF(d) = sum_r 1 / (k + rank_r(d))
    """
    all_qids = set()
    for rl in ranked_lists:
        all_qids.update(rl.keys())

    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for rl in ranked_lists:
            for rank, doc_id in enumerate(rl.get(qid, []), start=1):
                scores[doc_id] += 1.0 / (k + rank)
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        fused[qid] = sorted_docs[:top_n]
    return fused


def bm25_retrieve(query_texts, query_ids, corpus_texts, corpus_ids, top_k=100):
    """BM25 retrieval using rank_bm25."""
    from rank_bm25 import BM25Okapi

    # Tokenize
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    submission = {}
    for qtext, qid in tqdm(zip(query_texts, query_ids), total=len(query_ids), desc="BM25"):
        tokenized_query = qtext.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-scores)[:top_k]
        submission[qid] = [corpus_ids[j] for j in top_indices]
    return submission


def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k=100):
    """Dense retrieval via dot product (L2-normalised embeddings)."""
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def cross_encoder_rerank(submission, query_texts_map, corpus_texts_map, top_n=100):
    """Re-rank top candidates with a cross-encoder."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CE_MODEL_NAME)

    reranked = {}
    for qid in tqdm(submission, desc="Cross-Encoder re-ranking"):
        q_text = query_texts_map[qid]
        doc_ids = submission[qid][:top_n]
        pairs = [(q_text, corpus_texts_map[did]) for did in doc_ids]

        scores = model.predict(pairs, show_progress_bar=False)
        ranked_indices = np.argsort(-scores)
        reranked[qid] = [doc_ids[i] for i in ranked_indices]
    return reranked


# ── Main ─────────────────────────────────────────────────────

def run_pipeline(queries_df, query_ids, query_texts, corpus, corpus_ids, corpus_texts,
                 corpus_embs, c_ids, dense_model_name, label=""):
    """Run the full hybrid pipeline on a set of queries."""

    # Build text lookup maps
    query_texts_map = dict(zip(query_ids, query_texts))
    corpus_texts_map = dict(zip(corpus_ids, corpus_texts))

    # Stage 1a: BM25 retrieval
    print(f"\n{'='*60}")
    print(f"Stage 1a: BM25 retrieval {label}")
    bm25_sub = bm25_retrieve(query_texts, query_ids, corpus_texts, corpus_ids, top_k=TOP_K)

    # Stage 1b: Dense retrieval
    print(f"\nStage 1b: Dense retrieval {label}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(dense_model_name)
    q_embs = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dense_sub = dense_retrieve(q_embs, query_ids, corpus_embs, c_ids, top_k=TOP_K)

    # Stage 2: RRF fusion
    print(f"\nStage 2: RRF fusion {label}")
    rrf_sub = rrf_fuse([bm25_sub, dense_sub], k=RRF_K, top_n=TOP_K)

    # Stage 3: Cross-encoder re-ranking
    print(f"\nStage 3: Cross-Encoder re-ranking {label}")
    final_sub = cross_encoder_rerank(rrf_sub, query_texts_map, corpus_texts_map, top_n=TOP_K)

    return bm25_sub, dense_sub, rrf_sub, final_sub


def main():
    # Load data
    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]

    # Load pre-computed corpus embeddings
    corpus_embs, c_ids = load_embeddings(EMB_DIR / "corpus_embeddings.npy",
                                          EMB_DIR / "corpus_ids.json")

    # ── Evaluate on public queries ──
    pub_ids = queries["doc_id"].tolist()
    pub_texts = [format_text(row) for _, row in queries.iterrows()]
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    bm25_sub, dense_sub, rrf_sub, final_sub = run_pipeline(
        queries, pub_ids, pub_texts, corpus, corpus_ids, corpus_texts,
        corpus_embs, c_ids, DENSE_MODEL_NAME, label="(public)"
    )

    print("\n--- BM25 only ---")
    evaluate(bm25_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)
    print("\n--- Dense only ---")
    evaluate(dense_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)
    print("\n--- RRF (BM25 + Dense) ---")
    evaluate(rrf_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)
    print("\n--- RRF + Cross-Encoder ---")
    evaluate(final_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Predict on held-out queries ──
    ho_ids = held_out["doc_id"].tolist()
    ho_texts = [format_text(row) for _, row in held_out.iterrows()]

    _, _, _, ho_final = run_pipeline(
        held_out, ho_ids, ho_texts, corpus, corpus_ids, corpus_texts,
        corpus_embs, c_ids, DENSE_MODEL_NAME, label="(held-out)"
    )

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "hybrid_rrf_rerank.json"
    with open(out_path, "w") as f:
        json.dump(ho_final, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
