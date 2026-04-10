"""
Dense retrieval + Cross-Encoder re-ranking.

Stage 1: Dense retrieval (all-MiniLM-L6-v2) — top 100
Stage 2: Cross-Encoder re-rank — reorder top 100 for better NDCG@10

Outputs predictions for held-out queries to submissions/.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_queries, load_corpus, load_qrels, format_text,
    load_embeddings, evaluate,
)

TOP_K = 100
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k=100):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def cross_encoder_rerank(submission, query_texts_map, corpus_texts_map, top_n=100):
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


def main():
    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]
    corpus_texts_map = dict(zip(corpus_ids, corpus_texts))

    # Load pre-computed corpus embeddings
    corpus_embs, c_ids = load_embeddings(EMB_DIR / "corpus_embeddings.npy",
                                          EMB_DIR / "corpus_ids.json")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DENSE_MODEL_NAME)

    # ── Evaluate on public queries ──
    pub_ids = queries["doc_id"].tolist()
    pub_texts = [format_text(row) for _, row in queries.iterrows()]
    pub_texts_map = dict(zip(pub_ids, pub_texts))
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print("\nStage 1: Dense retrieval (public)")
    pub_embs = model.encode(pub_texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dense_sub = dense_retrieve(pub_embs, pub_ids, corpus_embs, c_ids, top_k=TOP_K)

    print("\n--- Dense only ---")
    evaluate(dense_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    print("\nStage 2: Cross-Encoder re-ranking (public)")
    reranked_sub = cross_encoder_rerank(dense_sub, pub_texts_map, corpus_texts_map, top_n=TOP_K)

    print("\n--- Dense + Cross-Encoder ---")
    evaluate(reranked_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Predict on held-out queries ──
    ho_ids = held_out["doc_id"].tolist()
    ho_texts = [format_text(row) for _, row in held_out.iterrows()]
    ho_texts_map = dict(zip(ho_ids, ho_texts))

    print("\nStage 1: Dense retrieval (held-out)")
    ho_embs = model.encode(ho_texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    ho_dense = dense_retrieve(ho_embs, ho_ids, corpus_embs, c_ids, top_k=TOP_K)

    print("\nStage 2: Cross-Encoder re-ranking (held-out)")
    ho_final = cross_encoder_rerank(ho_dense, ho_texts_map, corpus_texts_map, top_n=TOP_K)

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "dense_rerank.json"
    with open(out_path, "w") as f:
        json.dump(ho_final, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
