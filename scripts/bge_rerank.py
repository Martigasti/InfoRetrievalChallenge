"""
Phase 2: Reranker on top of bge_weighted_rrf.

Pipeline:
  1. Run the full SPECTER2 + BGE-large + BM25 weighted RRF (same as bge_weighted_rrf.py)
  2. Take top-20 from the fused ranking
  3. Rerank those 20 with BAAI/bge-reranker-v2-m3 (a cross-encoder trained
     on diverse multilingual + academic data — unlike MS-MARCO rerankers
     which failed on paper-to-paper matching)
  4. Put the reranked top-20 at positions 1..20, keep positions 21..100 unchanged

NDCG@10 only looks at the top 10 — reranking 20 gives a small buffer but
keeps the cross-encoder cost low (20 pairs per query, not 100).
"""

import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_queries, load_corpus, load_qrels,
    format_text, get_body_chunks, evaluate,
)

RETRIEVAL_TOP_K = 200
FINAL_TOP_K = 100
RERANK_TOP_K = 20       # rerank the top-20 from RRF
RRF_K = 10
BODY_CHUNKS = 6
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LEN = 512
RERANKER_BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


# ── Text formatting ──────────────────────────────────────────

def format_specter(row):
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body_extra = ""
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            body_extra = " ".join(chunks[:BODY_CHUNKS])
    except Exception:
        pass
    parts = [p for p in [title, abstract, body_extra] if p]
    if len(parts) >= 2:
        return parts[0] + " [SEP] " + " ".join(parts[1:])
    return parts[0] if parts else ""


def format_enriched(row):
    base = format_text(row)
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            base += " " + " ".join(chunks[:BODY_CHUNKS])
    except Exception:
        pass
    return base


def format_rerank(row):
    """Reranker input: title + abstract (short, since cross-encoder truncates at 512)."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + ". " + abstract
    return title or abstract


# ── Encoding ─────────────────────────────────────────────────

def encode_specter(texts, tokenizer, model, batch_size=32, device="cpu"):
    import torch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SPECTER2 encoding"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        emb = output.last_hidden_state[:, 0, :]
        emb = emb / emb.norm(dim=1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


# ── Retrieval ────────────────────────────────────────────────

def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def bm25_tokenize(text):
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import re
    stemmer = PorterStemmer()
    stops = set(stopwords.words("english"))
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]


def bm25_retrieve(query_texts, q_ids, corpus_tokenized, c_ids, bm25_model, top_k):
    results = {}
    for qid, q_text in tqdm(zip(q_ids, query_texts), total=len(q_ids), desc="BM25 retrieval"):
        q_tokens = bm25_tokenize(q_text)
        scores = bm25_model.get_scores(q_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        results[qid] = [c_ids[i] for i in top_indices]
    return results


# ── Weighted RRF ─────────────────────────────────────────────

def weighted_rrf_fuse(rankings_with_weights, k=10, top_k=100):
    """score(d) = sum_r weight_r / (k + rank_r(d))"""
    all_qids = set()
    for ranking, _ in rankings_with_weights:
        all_qids.update(ranking.keys())

    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for ranking, weight in rankings_with_weights:
            if qid not in ranking:
                continue
            for rank, doc_id in enumerate(ranking[qid], start=1):
                scores[doc_id] += weight / (k + rank)
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
        fused[qid] = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    return fused


# ── Reranking ────────────────────────────────────────────────

def rerank_topk(fused_ranking, query_texts_by_id, doc_text_by_id,
                reranker, top_rerank=20, device="cpu"):
    """
    For each query:
      - take top-`top_rerank` doc IDs from fused_ranking
      - score (query_text, doc_text) pairs with the cross-encoder
      - reorder those `top_rerank` by cross-encoder score
      - tail (positions top_rerank..end) stays unchanged
    """
    reranked = {}
    qids = list(fused_ranking.keys())
    for qid in tqdm(qids, desc="Reranking"):
        ranked = fused_ranking[qid]
        head = ranked[:top_rerank]
        tail = ranked[top_rerank:]
        q_text = query_texts_by_id[qid]
        pairs = [(q_text, doc_text_by_id[d]) for d in head]
        scores = reranker.predict(pairs,
                                  batch_size=RERANKER_BATCH_SIZE,
                                  show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        new_head = [head[i] for i in order]
        reranked[qid] = new_head + tail
    return reranked


# ── Main ─────────────────────────────────────────────────────

def main():
    import torch
    import nltk
    from transformers import AutoTokenizer, AutoModel
    from adapters import init as adapters_init
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids = corpus["doc_id"].tolist()
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print(f"Building enriched text (body_chunks={BODY_CHUNKS})...")
    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter = [format_specter(row) for _, row in corpus.iterrows()]

    # Short text for reranker — indexed by doc_id
    corpus_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in corpus.iterrows()
    }

    # ── BM25 index ──
    print("Tokenizing corpus for BM25 (stemmed)...")
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_enriched, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── SPECTER2 ──
    print(f"\nLoading {SPECTER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    specter_model = AutoModel.from_pretrained(SPECTER_MODEL)
    adapters_init(specter_model)
    specter_model.load_adapter(PROXIMITY_ADAPTER, source="hf", set_active=True)
    specter_model.to(device)
    specter_model.eval()

    print("Encoding corpus with SPECTER2...")
    specter_corpus_embs = encode_specter(corpus_specter, tokenizer,
                                          specter_model, batch_size=32, device=device)

    # ── BGE-large ──
    print(f"\nLoading {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched,
                                        normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — retrieve
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]
    pub_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in queries.iterrows()
    }

    print("\nBGE retrieval...")
    bge_q_embs = bge_model.encode(pub_enriched, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
    bge_ranking = dense_retrieve(bge_q_embs, pub_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval...")
    bm25_ranking = bm25_retrieve(pub_enriched, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # Free dense models from GPU before loading the reranker
    print("\nFreeing dense models from GPU for reranker...")
    specter_model.to("cpu")
    bge_model.to("cpu")
    torch.cuda.empty_cache()

    # ── Fuse with best weights from bge_weighted_rrf (bm25_w=0.5) ──
    BM25_WEIGHT = 0.5
    fused = weighted_rrf_fuse(
        [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )

    print("\n--- Weighted RRF (no reranker) baseline ---")
    evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Reranker ──
    print(f"\nLoading {RERANKER_NAME}...")
    reranker = CrossEncoder(RERANKER_NAME, max_length=RERANKER_MAX_LEN, device=device)

    print(f"Reranking top-{RERANK_TOP_K} of each query...")
    reranked = rerank_topk(
        fused, pub_rerank_by_id, corpus_rerank_by_id,
        reranker, top_rerank=RERANK_TOP_K, device=device,
    )

    print(f"\n--- Weighted RRF + BGE reranker (top-{RERANK_TOP_K}) ---")
    evaluate(reranked, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]
    ho_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in held_out.iterrows()
    }

    # Put dense models back on GPU, move reranker off
    print("\nSwapping reranker off GPU for held-out retrieval...")
    reranker.model.to("cpu")
    torch.cuda.empty_cache()
    specter_model.to(device)
    bge_model.to(device)

    print("BGE (held-out)...")
    ho_bge_embs = bge_model.encode(ho_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    ho_bge = dense_retrieve(ho_bge_embs, ho_ids,
                             bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = weighted_rrf_fuse(
        [(ho_specter, 1.0), (ho_bge, 1.0), (ho_bm25, BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )

    # Free dense models, put reranker back
    print("\nSwapping dense models off GPU for reranker (held-out)...")
    specter_model.to("cpu")
    bge_model.to("cpu")
    torch.cuda.empty_cache()
    reranker.model.to(device)

    print(f"Reranking top-{RERANK_TOP_K} on held-out...")
    ho_reranked = rerank_topk(
        ho_fused, ho_rerank_by_id, corpus_rerank_by_id,
        reranker, top_rerank=RERANK_TOP_K, device=device,
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_reranked))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()
