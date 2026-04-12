"""
3-way Weighted RRF: SPECTER2 + Jina-embeddings-v3 + BM25.

Based on bge_weighted_rrf.py, replacing BGE-large with jinaai/jina-embeddings-v3:
- 570M params, fp16, max_length=2048 (4× the context of BGE's 512)
- Uses task='retrieval.passage' for both corpus and queries (symmetric task)
- trust_remote_code=True (custom Jina encode method)
- batch_size=4 to stay within 8GB VRAM at 2048 token context
- GPU-swapped with SPECTER2 to share 8GB VRAM
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
JINA_MODEL = "jinaai/jina-embeddings-v3"
JINA_MAX_LEN = 2048
JINA_TASK = "retrieval.passage"   # symmetric — same task for corpus and queries
JINA_BATCH_SIZE = 4               # 2048 tokens × 570M fp16 params on 8GB VRAM

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


def encode_jina(texts, model, batch_size=JINA_BATCH_SIZE):
    """Jina v3 custom encode with retrieval.passage task and fp16."""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Jina v3 encoding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(
            batch,
            task=JINA_TASK,
            max_length=JINA_MAX_LEN,
            truncate_dim=None,
            convert_to_numpy=True,
        )
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        emb = (emb / norm).astype(np.float32)
        all_embs.append(emb)
    return np.vstack(all_embs)


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

    # ── BM25 index ──
    print("Tokenizing corpus for BM25 (stemmed)...")
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_enriched, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── SPECTER2 — encode corpus, then move to CPU ──
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

    # Free GPU for Jina v3 (570M fp16 needs room alongside activations)
    specter_model.to("cpu")
    torch.cuda.empty_cache()

    # ── Jina v3 — encode corpus ──
    print(f"\nLoading {JINA_MODEL} (fp16, trust_remote_code)...")
    jina_model = AutoModel.from_pretrained(
        JINA_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
    )
    jina_model.to(device).eval()

    print(f"Encoding corpus with Jina v3 (max_length={JINA_MAX_LEN}, batch_size={JINA_BATCH_SIZE})...")
    jina_corpus_embs = encode_jina(corpus_enriched, jina_model)

    # ══════════════════════════════════════════════════════════════
    # Public queries — evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]

    print("\nJina v3 retrieval (public)...")
    jina_q_embs = encode_jina(pub_enriched, jina_model)
    jina_ranking = dense_retrieve(jina_q_embs, pub_ids,
                                   jina_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # Swap Jina off GPU, restore SPECTER2 for query encoding
    jina_model.to("cpu")
    torch.cuda.empty_cache()
    specter_model.to(device)

    print("SPECTER2 retrieval (public)...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval (public)...")
    bm25_ranking = bm25_retrieve(pub_enriched, pub_ids, corpus_tokenized,
                                  corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # Grid search over BM25 weight
    print("\n" + "=" * 50)
    print("Grid search: BM25 weight (dense=1.0 fixed)")
    print("=" * 50)
    bm25_weights = [0.3, 0.5, 0.7, 1.0]
    best_cfg, best_ndcg = None, 0
    for bm25_w in bm25_weights:
        fused = weighted_rrf_fuse(
            [(specter_ranking, 1.0), (jina_ranking, 1.0), (bm25_ranking, bm25_w)],
            k=RRF_K, top_k=FINAL_TOP_K
        )
        res = evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        mapv = res["overall"]["MAP"]
        recall = res["overall"]["Recall@100"]
        print(f"  bm25_w={bm25_w:.1f}  NDCG@10={ndcg:.4f}  MAP={mapv:.4f}  Recall@100={recall:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_cfg = bm25_w

    print(f"\n*** Best bm25_weight={best_cfg} (NDCG@10={best_ndcg:.4f}) ***")

    fused_best = weighted_rrf_fuse(
        [(specter_ranking, 1.0), (jina_ranking, 1.0), (bm25_ranking, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K
    )
    print(f"\n--- Weighted RRF (specter=1.0, jina=1.0, bm25={best_cfg}) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]

    print("\nJina v3 (held-out)...")
    jina_model.to(device)
    specter_model.to("cpu")
    torch.cuda.empty_cache()
    ho_jina_embs = encode_jina(ho_enriched, jina_model)
    ho_jina = dense_retrieve(ho_jina_embs, ho_ids,
                              jina_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    jina_model.to("cpu")
    specter_model.to(device)
    torch.cuda.empty_cache()
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = weighted_rrf_fuse(
        [(ho_specter, 1.0), (ho_jina, 1.0), (ho_bm25, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()