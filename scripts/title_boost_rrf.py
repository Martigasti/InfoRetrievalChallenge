"""
Title-boosted BM25 + more query body chunks on top of wider_rrf.py.

Two changes vs wider_rrf.py:
  1. BM25F approximation: title tokens repeated TITLE_BOOST times in BM25
     corpus text, giving title matches higher weight without changing the
     BM25Okapi implementation.
  2. More query body chunks (QUERY_CHUNKS=12 vs BODY_CHUNKS=6 for corpus):
     query papers' body text contains their reference sections — more chunks
     pulls cited-paper titles into dense query embeddings.

Known best from grid search (wider_rrf.py): pool=300, chunks=6, bm25_w=1.0.
Grid search only over BM25 weight since title boost changes the BM25 signal.
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

RETRIEVAL_TOP_K = 300
FINAL_TOP_K     = 100
RRF_K           = 10
BODY_CHUNKS     = 6    # corpus document chunks (unchanged)
QUERY_CHUNKS    = 12   # query paper chunks — captures reference section
TITLE_BOOST     = 3    # repeat title tokens this many extra times in BM25 index

SPECTER_MODEL     = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME    = "BAAI/bge-large-en-v1.5"

ROOT            = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH   = ROOT / "held_out_queries.parquet"


# ── Text formatting ──────────────────────────────────────────

def format_specter(row, n_chunks=BODY_CHUNKS):
    title      = str(row.get("title", "") or "").strip()
    abstract   = str(row.get("abstract", "") or "").strip()
    body_extra = ""
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            body_extra = " ".join(chunks[:n_chunks])
    except Exception:
        pass
    parts = [p for p in [title, abstract, body_extra] if p]
    if len(parts) >= 2:
        return parts[0] + " [SEP] " + " ".join(parts[1:])
    return parts[0] if parts else ""


def format_enriched(row, n_chunks=BODY_CHUNKS):
    base = format_text(row)
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            base += " " + " ".join(chunks[:n_chunks])
    except Exception:
        pass
    return base


def format_bm25_corpus(row):
    """BM25 corpus text: title repeated TITLE_BOOST+1 times, then body chunks."""
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body_extra = ""
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            body_extra = " ".join(chunks[:BODY_CHUNKS])
    except Exception:
        pass
    # Repeat title to simulate field-level boost
    boosted_title = " ".join([title] * (TITLE_BOOST + 1))
    parts = [p for p in [boosted_title, abstract, body_extra] if p]
    return " ".join(parts)


# ── Encoding ─────────────────────────────────────────────────

def encode_specter(texts, tokenizer, model, batch_size=32, device="cpu"):
    import torch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SPECTER2 encoding"):
        batch   = texts[i:i + batch_size]
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
    sim_matrix  = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def bm25_tokenize(text):
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import re
    stemmer = PorterStemmer()
    stops   = set(stopwords.words("english"))
    tokens  = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]


def bm25_retrieve(query_texts, q_ids, corpus_tokenized, c_ids, bm25_model, top_k):
    results = {}
    for qid, q_text in tqdm(zip(q_ids, query_texts), total=len(q_ids), desc="BM25 retrieval"):
        q_tokens   = bm25_tokenize(q_text)
        scores     = bm25_model.get_scores(q_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        results[qid] = [c_ids[i] for i in top_indices]
    return results


# ── Weighted RRF ─────────────────────────────────────────────

def weighted_rrf_fuse(rankings_with_weights, k=10, top_k=100):
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
        fused[qid] = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
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
    nltk.download("punkt",     quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: pool={RETRIEVAL_TOP_K}, corpus_chunks={BODY_CHUNKS}, "
          f"query_chunks={QUERY_CHUNKS}, title_boost={TITLE_BOOST}x")

    print("\nLoading data...")
    queries  = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus   = load_corpus(DATA_DIR / "corpus.parquet")
    qrels    = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids    = corpus["doc_id"].tolist()
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # Corpus: standard enriched for dense, title-boosted for BM25
    print(f"\nBuilding corpus text (body_chunks={BODY_CHUNKS}, title_boost={TITLE_BOOST}x)...")
    corpus_enriched  = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter   = [format_specter(row)  for _, row in corpus.iterrows()]
    corpus_bm25      = [format_bm25_corpus(row) for _, row in corpus.iterrows()]

    # ── BM25 index with title boost ──
    print("Tokenizing corpus for BM25 (title-boosted)...")
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_bm25, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── SPECTER2 ──
    print(f"\nLoading {SPECTER_MODEL}...")
    tokenizer     = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    specter_model = AutoModel.from_pretrained(SPECTER_MODEL)
    adapters_init(specter_model)
    specter_model.load_adapter(PROXIMITY_ADAPTER, source="hf", set_active=True)
    specter_model.to(device).eval()

    print("Encoding corpus with SPECTER2...")
    specter_corpus_embs = encode_specter(corpus_specter, tokenizer,
                                          specter_model, batch_size=32, device=device)

    # ── BGE-large ──
    print(f"\nLoading {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched, normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — use QUERY_CHUNKS (more than corpus)
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()

    # Query encoding uses more chunks to capture reference section
    pub_enriched = [format_enriched(row, n_chunks=QUERY_CHUNKS) for _, row in queries.iterrows()]
    pub_specter  = [format_specter(row,  n_chunks=QUERY_CHUNKS) for _, row in queries.iterrows()]
    # BM25 query text: plain enriched (no title boost — queries are not indexed)
    pub_bm25     = [format_enriched(row, n_chunks=BODY_CHUNKS)  for _, row in queries.iterrows()]

    print("\nBGE retrieval...")
    bge_q_embs  = bge_model.encode(pub_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    bge_ranking = dense_retrieve(bge_q_embs, pub_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval...")
    specter_q_embs  = encode_specter(pub_specter, tokenizer,
                                      specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval...")
    bm25_ranking = bm25_retrieve(pub_bm25, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # Grid search BM25 weight (title boost may shift optimal weight)
    print("\n" + "=" * 55)
    print(f"Grid search: BM25 weight (title_boost={TITLE_BOOST}x, query_chunks={QUERY_CHUNKS})")
    print("=" * 55)
    best_cfg, best_ndcg = None, 0
    for bm25_w in [0.3, 0.5, 0.7, 1.0, 1.5]:
        fused = weighted_rrf_fuse(
            [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, bm25_w)],
            k=RRF_K, top_k=FINAL_TOP_K,
        )
        res  = evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        rec  = res["overall"]["Recall@100"]
        print(f"  bm25_w={bm25_w:.1f}  NDCG@10={ndcg:.4f}  Recall@100={rec:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_cfg  = bm25_w

    print(f"\n*** Best bm25_weight={best_cfg} (NDCG@10={best_ndcg:.4f}) ***")
    fused_best = weighted_rrf_fuse(
        [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )
    print(f"\n--- Title-boost BM25 + more query chunks (bm25_w={best_cfg}) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()

    ho_enriched = [format_enriched(row, n_chunks=QUERY_CHUNKS) for _, row in held_out.iterrows()]
    ho_specter  = [format_specter(row,  n_chunks=QUERY_CHUNKS) for _, row in held_out.iterrows()]
    ho_bm25     = [format_enriched(row, n_chunks=BODY_CHUNKS)  for _, row in held_out.iterrows()]

    print("\nBGE (held-out)...")
    ho_bge_embs = bge_model.encode(ho_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    ho_bge      = dense_retrieve(ho_bge_embs, ho_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter_rank = dense_retrieve(ho_specter_embs, ho_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    ho_bm25_rank = bm25_retrieve(ho_bm25, ho_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = weighted_rrf_fuse(
        [(ho_specter_rank, 1.0), (ho_bge, 1.0), (ho_bm25_rank, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path}  [title_boost={TITLE_BOOST}x, query_chunks={QUERY_CHUNKS}, bm25_w={best_cfg}]")


if __name__ == "__main__":
    main()
