"""
4-way Weighted RRF: SPECTER2 + BGE-large + SciNCL + BM25.

Phase 3 of the overnight plan. SciNCL (malteos/scincl) is specifically trained
on scientific citation graphs — unlike Jina/Arctic, it targets exactly the task
at hand (paper-to-paper citation similarity).

Strategy: add SciNCL as a *fourth* retriever (not replacing BGE), because each
dense model captures different patterns and BGE is our proven backbone.

Grid search: w_scincl ∈ {0.5, 1.0, 1.5}, with specter=bge=1.0 and bm25=0.5
(the BM25 weight locked from bge_weighted_rrf.py).

SciNCL is SciBERT-based (~110M params), so no GPU swap needed — it coexists
with SPECTER2 and BGE on 8GB VRAM.
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
BM25_WEIGHT = 0.5  # locked from bge_weighted_rrf.py

SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SCINCL_MODEL = "malteos/scincl"
SCINCL_MAX_LEN = 512
SCINCL_WEIGHT_GRID = [0.5, 1.0, 1.5]

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


def format_scincl(row):
    """SciNCL canonical input: title [SEP] abstract (no body chunks)."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " [SEP] " + abstract
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


def encode_scincl(texts, tokenizer, model, batch_size=32, device="cpu"):
    """SciNCL: CLS pooling + L2 normalize (same pattern as SPECTER2)."""
    import torch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SciNCL encoding"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=SCINCL_MAX_LEN, return_tensors="pt",
        ).to(device)
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
    corpus_scincl = [format_scincl(row) for _, row in corpus.iterrows()]

    # ── BM25 index ──
    print("Tokenizing corpus for BM25 (stemmed)...")
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_enriched, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── SPECTER2 ──
    print(f"\nLoading {SPECTER_MODEL}...")
    specter_tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    specter_model = AutoModel.from_pretrained(SPECTER_MODEL)
    adapters_init(specter_model)
    specter_model.load_adapter(PROXIMITY_ADAPTER, source="hf", set_active=True)
    specter_model.to(device)
    specter_model.eval()

    print("Encoding corpus with SPECTER2...")
    specter_corpus_embs = encode_specter(corpus_specter, specter_tokenizer,
                                          specter_model, batch_size=32, device=device)

    # ── BGE-large ──
    print(f"\nLoading {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched,
                                        normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)

    # ── SciNCL ──
    print(f"\nLoading {SCINCL_MODEL}...")
    scincl_tokenizer = AutoTokenizer.from_pretrained(SCINCL_MODEL)
    scincl_model = AutoModel.from_pretrained(SCINCL_MODEL).to(device).eval()

    print("Encoding corpus with SciNCL...")
    scincl_corpus_embs = encode_scincl(corpus_scincl, scincl_tokenizer,
                                        scincl_model, batch_size=32, device=device)

    # ══════════════════════════════════════════════════════════════
    # Public queries — evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]
    pub_scincl = [format_scincl(row) for _, row in queries.iterrows()]

    print("\nBGE retrieval (public)...")
    bge_q_embs = bge_model.encode(pub_enriched, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
    bge_ranking = dense_retrieve(bge_q_embs, pub_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval (public)...")
    specter_q_embs = encode_specter(pub_specter, specter_tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SciNCL retrieval (public)...")
    scincl_q_embs = encode_scincl(pub_scincl, scincl_tokenizer,
                                   scincl_model, batch_size=32, device=device)
    scincl_ranking = dense_retrieve(scincl_q_embs, pub_ids,
                                     scincl_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval (public)...")
    bm25_ranking = bm25_retrieve(pub_enriched, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # ── Baseline: 3-way (BGE + SPECTER2 + BM25) for direct delta ──
    print("\n--- 3-way baseline (SPECTER2 + BGE + BM25), no SciNCL ---")
    fused_baseline = weighted_rrf_fuse(
        [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )
    res = evaluate(fused_baseline, qrels, ks=[10, 100],
                   query_domains=query_domains, verbose=False)
    baseline_ndcg = res["overall"]["NDCG@10"]
    print(f"  3-way NDCG@10={baseline_ndcg:.4f}  "
          f"MAP={res['overall']['MAP']:.4f}  "
          f"Recall@100={res['overall']['Recall@100']:.4f}")

    # ── Grid search over w_scincl ──
    print("\n" + "=" * 50)
    print("Grid search: SciNCL weight (specter=bge=1.0, bm25=0.5)")
    print("=" * 50)
    best_w, best_ndcg = None, 0.0
    for w in SCINCL_WEIGHT_GRID:
        fused = weighted_rrf_fuse(
            [
                (specter_ranking, 1.0),
                (bge_ranking, 1.0),
                (scincl_ranking, w),
                (bm25_ranking, BM25_WEIGHT),
            ],
            k=RRF_K, top_k=FINAL_TOP_K,
        )
        res = evaluate(fused, qrels, ks=[10, 100],
                       query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        mapv = res["overall"]["MAP"]
        recall = res["overall"]["Recall@100"]
        delta = ndcg - baseline_ndcg
        print(f"  w_scincl={w:.1f}  NDCG@10={ndcg:.4f} ({delta:+.4f})  "
              f"MAP={mapv:.4f}  Recall@100={recall:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_w = w

    print(f"\n*** Best w_scincl={best_w} (NDCG@10={best_ndcg:.4f}, "
          f"Δ={best_ndcg - baseline_ndcg:+.4f}) ***")

    fused_best = weighted_rrf_fuse(
        [
            (specter_ranking, 1.0),
            (bge_ranking, 1.0),
            (scincl_ranking, best_w),
            (bm25_ranking, BM25_WEIGHT),
        ],
        k=RRF_K, top_k=FINAL_TOP_K,
    )
    print(f"\n--- 4-way Weighted RRF "
          f"(specter=1.0, bge=1.0, scincl={best_w}, bm25={BM25_WEIGHT}) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]
    ho_scincl_texts = [format_scincl(row) for _, row in held_out.iterrows()]

    print("\nBGE (held-out)...")
    ho_bge_embs = bge_model.encode(ho_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    ho_bge = dense_retrieve(ho_bge_embs, ho_ids,
                             bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, specter_tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SciNCL (held-out)...")
    ho_scincl_embs = encode_scincl(ho_scincl_texts, scincl_tokenizer,
                                    scincl_model, batch_size=32, device=device)
    ho_scincl = dense_retrieve(ho_scincl_embs, ho_ids,
                                scincl_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = weighted_rrf_fuse(
        [
            (ho_specter, 1.0),
            (ho_bge, 1.0),
            (ho_scincl, best_w),
            (ho_bm25, BM25_WEIGHT),
        ],
        k=RRF_K, top_k=FINAL_TOP_K,
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()
