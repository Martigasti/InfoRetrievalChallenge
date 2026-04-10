"""
Dense baseline (MiniLM, 0.50 NDCG@10) + Cross-Encoder re-ranking with score
interpolation.

Instead of fully replacing the dense ranking with cross-encoder scores
(which hurt performance before), we interpolate:
    final_score = alpha * dense_score_norm + (1 - alpha) * ce_score_norm

This preserves the strong dense signal while letting the cross-encoder
refine the top positions.

Also retrieves top-200 to give the reranker more candidates.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_queries, load_corpus, load_qrels, load_embeddings, format_text, evaluate

RETRIEVAL_TOP_K = 200
FINAL_TOP_K = 100
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]  # grid search for best interpolation weight
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 12-layer, stronger
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def dense_retrieve_with_scores(query_embs, q_ids, corpus_embs, c_ids, top_k):
    """Return top-k doc ids AND their dense similarity scores per query."""
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    results = {}
    for i, qid in enumerate(q_ids):
        doc_ids = [c_ids[j] for j in top_indices[i]]
        scores = sim_matrix[i, top_indices[i]]
        results[qid] = (doc_ids, scores)
    return results


def compute_ce_scores(retrieval_results, query_texts_map, corpus_texts_map):
    """Run cross-encoder once and cache scores for all queries."""
    model = CrossEncoder(CE_MODEL_NAME)
    ce_cache = {}
    for qid in tqdm(retrieval_results, desc="Cross-Encoder scoring"):
        doc_ids, _ = retrieval_results[qid]
        q_text = query_texts_map[qid]
        pairs = [(q_text, corpus_texts_map[did]) for did in doc_ids]
        ce_cache[qid] = model.predict(pairs, show_progress_bar=False)
    return ce_cache


def rerank_with_alpha(retrieval_results, ce_cache, alpha, top_k_out):
    """Rerank using cached CE scores and a given alpha (no recomputation)."""
    reranked = {}
    for qid in retrieval_results:
        doc_ids, dense_scores = retrieval_results[qid]
        dense_norm = normalize_scores(np.array(dense_scores))
        ce_norm = normalize_scores(np.array(ce_cache[qid]))
        final_scores = alpha * dense_norm + (1 - alpha) * ce_norm
        ranked_indices = np.argsort(-final_scores)[:top_k_out]
        reranked[qid] = [doc_ids[i] for i in ranked_indices]
    return reranked


def main():
    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Pre-computed MiniLM embeddings
    corpus_embs, c_ids = load_embeddings(EMB_DIR / "corpus_embeddings.npy",
                                          EMB_DIR / "corpus_ids.json")
    query_embs, q_ids = load_embeddings(EMB_DIR / "query_embeddings.npy",
                                         EMB_DIR / "query_ids.json")

    # Corpus texts for cross-encoder
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    corpus_texts_map = dict(zip(corpus_ids, corpus_texts))

    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # ── Public queries: evaluate ──
    pub_texts_map = {row["doc_id"]: format_text(row) for _, row in queries.iterrows()}

    # Dense-only baseline for comparison
    print(f"\nDense retrieval (top-{RETRIEVAL_TOP_K})...")
    retrieval = dense_retrieve_with_scores(query_embs, q_ids, corpus_embs, c_ids,
                                            top_k=RETRIEVAL_TOP_K)
    dense_only = {qid: ids[:FINAL_TOP_K] for qid, (ids, _) in retrieval.items()}
    print("\n--- Dense only ---")
    evaluate(dense_only, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # Cross-encoder scoring (done once)
    print(f"\nCross-Encoder scoring (12-layer)...")
    ce_cache = compute_ce_scores(retrieval, pub_texts_map, corpus_texts_map)

    # Grid search over alpha
    best_alpha, best_ndcg = 0, 0
    for alpha in ALPHAS:
        reranked = rerank_with_alpha(retrieval, ce_cache, alpha=alpha, top_k_out=FINAL_TOP_K)
        res = evaluate(reranked, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        mapv = res["overall"]["MAP"]
        print(f"  alpha={alpha:.1f}  NDCG@10={ndcg:.4f}  MAP={mapv:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = alpha

    print(f"\n*** Best alpha={best_alpha} (NDCG@10={best_ndcg:.4f}) ***")

    # Show full results for best alpha
    reranked = rerank_with_alpha(retrieval, ce_cache, alpha=best_alpha, top_k_out=FINAL_TOP_K)
    print(f"\n--- Dense + CE (alpha={best_alpha}) ---")
    evaluate(reranked, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Held-out queries: predict ──
    print(f"\nEncoding held-out queries...")
    model = SentenceTransformer(DENSE_MODEL_NAME)
    ho_texts = [format_text(row) for _, row in held_out.iterrows()]
    ho_embs = model.encode(ho_texts, normalize_embeddings=True,
                           show_progress_bar=True).astype(np.float32)
    ho_ids = held_out["doc_id"].tolist()
    ho_texts_map = dict(zip(ho_ids, ho_texts))

    ho_retrieval = dense_retrieve_with_scores(ho_embs, ho_ids, corpus_embs, c_ids,
                                               top_k=RETRIEVAL_TOP_K)
    print("Cross-Encoder scoring (held-out)...")
    ho_ce = compute_ce_scores(ho_retrieval, ho_texts_map, corpus_texts_map)
    ho_final = rerank_with_alpha(ho_retrieval, ho_ce, alpha=best_alpha, top_k_out=FINAL_TOP_K)

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "dense_interpolated_rerank.json"
    with open(out_path, "w") as f:
        json.dump(ho_final, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
