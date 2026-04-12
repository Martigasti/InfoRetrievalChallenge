"""
3-way RRF: SPECTER2 + E5-large-v2 + BM25.

Improvements over three_way_rrf.py:
1. E5-large-v2 replaces MiniLM: instruction-tuned for relevance matching,
   better at abstract/conceptual domains (Philosophy, Geography, Engineering).
   Uses 'query: ' / 'passage: ' prefixes to guide the model.
3. More body chunks (3 → 6): abstract papers bury key concepts in later
   sections; more context improves recall on those domains.
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
RRF_K = 10
BODY_CHUNKS = 6
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
E5_MODEL_NAME = "intfloat/e5-large-v2"

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


# ── RRF ──────────────────────────────────────────────────────

def rrf_fuse(rankings_list, k=10, top_k=100):
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
    corpus_e5 = ["passage: " + t for t in corpus_enriched]

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

    # ── E5-large ──
    print(f"\nLoading {E5_MODEL_NAME}...")
    e5_model = SentenceTransformer(E5_MODEL_NAME)
    print("Encoding corpus with E5-large...")
    e5_corpus_embs = e5_model.encode(corpus_e5,
                                      normalize_embeddings=True,
                                      show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]
    pub_e5 = ["query: " + t for t in pub_enriched]

    print("\nE5 retrieval...")
    e5_q_embs = e5_model.encode(pub_e5, normalize_embeddings=True,
                                 show_progress_bar=True).astype(np.float32)
    e5_ranking = dense_retrieve(e5_q_embs, pub_ids,
                                 e5_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval...")
    bm25_ranking = bm25_retrieve(pub_enriched, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    fused = rrf_fuse([specter_ranking, e5_ranking, bm25_ranking], k=RRF_K, top_k=FINAL_TOP_K)
    print("\n--- 3-way RRF (SPECTER2 + E5-large + BM25) ---")
    evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]
    ho_e5_texts = ["query: " + t for t in ho_enriched]

    print("\nE5 (held-out)...")
    ho_e5_embs = e5_model.encode(ho_e5_texts, normalize_embeddings=True,
                                  show_progress_bar=True).astype(np.float32)
    ho_e5 = dense_retrieve(ho_e5_embs, ho_ids,
                            e5_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = rrf_fuse([ho_specter, ho_e5, ho_bm25], k=RRF_K, top_k=FINAL_TOP_K)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()
