"""
RRF fusion: SPECTER2 + MiniLM dense retrieval.

Two retrievers with different strengths:
- SPECTER2: domain-specific scientific paper similarity
- MiniLM: general-purpose semantic similarity (proven 0.50 baseline)

Fused via Reciprocal Rank Fusion (RRF):
    RRF(d) = sum_r 1 / (k + rank_r(d))
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

TOP_K = 100
RRF_K = 60
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


def format_specter(row):
    """SPECTER2: 'title [SEP] abstract + first body chunks' to fill 512 tokens."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    body_extra = ""
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            body_extra = " ".join(chunks[:3])  # first 3 body chunks
    except Exception:
        pass
    parts = [p for p in [title, abstract, body_extra] if p]
    if len(parts) >= 2:
        return parts[0] + " [SEP] " + " ".join(parts[1:])
    return parts[0] if parts else ""


def format_enriched(row):
    """MiniLM: title + abstract + first body chunks."""
    base = format_text(row)
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            base += " " + " ".join(chunks[:3])
    except Exception:
        pass
    return base


def encode_specter(texts, tokenizer, model, batch_size=32, device="cpu"):
    """Encode texts with SPECTER2 using CLS pooling."""
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


def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def rrf_fuse(rankings_list, k=60, top_k=100):
    """Reciprocal Rank Fusion over multiple ranking dicts."""
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


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from adapters import init as adapters_init
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids = corpus["doc_id"].tolist()

    # Load SPECTER2
    print(f"Loading {SPECTER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    specter_model = AutoModel.from_pretrained(SPECTER_MODEL)
    adapters_init(specter_model)
    specter_model.load_adapter(PROXIMITY_ADAPTER, source="hf", set_active=True)
    specter_model.to(device)
    specter_model.eval()

    # Encode corpus with SPECTER2 (enriched text: title + abstract + body chunks)
    corpus_specter_texts = [format_specter(row) for _, row in corpus.iterrows()]
    print("Encoding corpus with SPECTER2 (enriched)...")
    specter_corpus_embs = encode_specter(corpus_specter_texts, tokenizer,
                                          specter_model, batch_size=32, device=device)

    # MiniLM: re-encode corpus with enriched text (body chunks added)
    minilm_model = SentenceTransformer(DENSE_MODEL_NAME)
    corpus_enriched_texts = [format_enriched(row) for _, row in corpus.iterrows()]
    print("Encoding corpus with MiniLM (enriched)...")
    minilm_corpus_embs = minilm_model.encode(corpus_enriched_texts,
                                              normalize_embeddings=True,
                                              show_progress_bar=True).astype(np.float32)

    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # ══════════════════════════════════════════════════════════════
    # Public queries — evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()

    # MiniLM retrieval (enriched queries too)
    pub_enriched_texts = [format_enriched(row) for _, row in queries.iterrows()]
    print("\nEncoding public queries with MiniLM (enriched)...")
    minilm_q_embs = minilm_model.encode(pub_enriched_texts,
                                         normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
    print("MiniLM retrieval...")
    minilm_ranking = dense_retrieve(minilm_q_embs, pub_ids,
                                     minilm_corpus_embs, corpus_ids, top_k=TOP_K)

    # SPECTER2 retrieval
    pub_specter_texts = [format_specter(row) for _, row in queries.iterrows()]
    print("SPECTER2 retrieval...")
    specter_q_embs = encode_specter(pub_specter_texts, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=TOP_K)

    # Individual results
    print("\n--- MiniLM only ---")
    evaluate(minilm_ranking, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)
    print("\n--- SPECTER2 only ---")
    evaluate(specter_ranking, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # RRF fusion
    fused = rrf_fuse([minilm_ranking, specter_ranking], k=RRF_K, top_k=TOP_K)
    print("\n--- RRF (MiniLM + SPECTER2) ---")
    evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()

    # MiniLM (enriched)
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_minilm_embs = minilm_model.encode(ho_enriched, normalize_embeddings=True,
                                          show_progress_bar=True).astype(np.float32)
    ho_minilm = dense_retrieve(ho_minilm_embs, ho_ids,
                                minilm_corpus_embs, corpus_ids, top_k=TOP_K)

    # SPECTER2
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=TOP_K)

    # RRF
    ho_fused = rrf_fuse([ho_minilm, ho_specter], k=RRF_K, top_k=TOP_K)

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "specter2_minilm_rrf.json"
    with open(out_path, "w") as f:
        json.dump(ho_fused, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
