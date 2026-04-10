"""
3-way RRF: SPECTER2 + MiniLM + BM25 with RRF k tuning.

Improvements over specter2_minilm_rrf.py:
- BM25 as third retriever (lexical matching for exact terms)
- Top-200 per retriever before fusion (more candidates)
- Grid search over RRF k parameter to optimize NDCG@10
- BM25 uses NLTK stemming + stopword removal for proper tokenization
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
    load_queries, load_corpus, load_qrels, load_embeddings,
    format_text, get_body_chunks, evaluate,
)

RETRIEVAL_TOP_K = 200  # per retriever, before fusion
FINAL_TOP_K = 100
RRF_K = 10  # best from grid search
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


# ── Text formatting ──────────────────────────────────────────

def format_specter(row):
    """SPECTER2: 'title [SEP] abstract + first body chunks'."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body_extra = ""
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            body_extra = " ".join(chunks[:3])
    except Exception:
        pass
    parts = [p for p in [title, abstract, body_extra] if p]
    if len(parts) >= 2:
        return parts[0] + " [SEP] " + " ".join(parts[1:])
    return parts[0] if parts else ""


def format_enriched(row):
    """MiniLM/BM25: title + abstract + first body chunks."""
    base = format_text(row)
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            base += " " + " ".join(chunks[:3])
    except Exception:
        pass
    return base


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
    """Tokenize with NLTK stemming and stopword removal."""
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


# ── Fusion ───────────────────────────────────────────────────

def rrf_fuse(rankings_list, k=60, top_k=100):
    all_qids = set()
    for r in rankings_list:
        all_qids.update(r.keys())

    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for ranking in rankings_list:
            if qid not in ranking:
                continue
            for rank, doc_id in enumerate(ranking[qid], start=1):
                scores[doc_id] += 1.0 / (k + rank)
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
        fused[qid] = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    return fused


# ── Main ─────────────────────────────────────────────────────

def main():
    import torch
    import nltk
    from transformers import AutoTokenizer, AutoModel
    from adapters import init as adapters_init
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    # Ensure NLTK data is available
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

    # ── Build enriched texts ──
    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter = [format_specter(row) for _, row in corpus.iterrows()]

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

    # ── MiniLM ──
    minilm_model = SentenceTransformer(DENSE_MODEL_NAME)
    print("Encoding corpus with MiniLM...")
    minilm_corpus_embs = minilm_model.encode(corpus_enriched,
                                              normalize_embeddings=True,
                                              show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]

    # MiniLM retrieval
    print("\nMiniLM retrieval...")
    minilm_q_embs = minilm_model.encode(pub_enriched, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
    minilm_ranking = dense_retrieve(minilm_q_embs, pub_ids,
                                     minilm_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # SPECTER2 retrieval
    print("SPECTER2 retrieval...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # BM25 retrieval
    print("BM25 retrieval...")
    bm25_ranking = bm25_retrieve(pub_enriched, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # 3-way RRF with best k
    fused_best = rrf_fuse([minilm_ranking, specter_ranking, bm25_ranking],
                           k=RRF_K, top_k=FINAL_TOP_K)
    print(f"\n--- 3-way RRF (k={RRF_K}) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict (using best k)
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]

    # MiniLM
    print("\nMiniLM (held-out)...")
    ho_minilm_embs = minilm_model.encode(ho_enriched, normalize_embeddings=True,
                                          show_progress_bar=True).astype(np.float32)
    ho_minilm = dense_retrieve(ho_minilm_embs, ho_ids,
                                minilm_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # SPECTER2
    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # BM25
    print("BM25 (held-out)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # Fuse with best k
    ho_fused = rrf_fuse([ho_minilm, ho_specter, ho_bm25], k=RRF_K, top_k=FINAL_TOP_K)

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "three_way_rrf.json"
    with open(out_path, "w") as f:
        json.dump(ho_fused, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
