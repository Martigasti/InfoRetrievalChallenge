"""
Comparison: SPECTER2 + BGE-large vs SPECTER2 + Snowflake-arctic-embed-m-v1.5
2-way RRF (no BM25) on public queries. Saves Snowflake predictions to submissions.
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
from utils import load_queries, load_corpus, load_qrels, format_text, get_body_chunks, evaluate

RETRIEVAL_TOP_K = 200
FINAL_TOP_K = 100
RRF_K = 10
BODY_CHUNKS = 6
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
ARCTIC_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


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


def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


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
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter = [format_specter(row) for _, row in corpus.iterrows()]

    # ── SPECTER2 (shared by both runs) ──
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

    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]

    print("SPECTER2 query encoding...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    # ── BGE-large ──
    print(f"\nLoading {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched, normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)
    print("BGE query encoding...")
    bge_q_embs = bge_model.encode(pub_enriched, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
    bge_ranking = dense_retrieve(bge_q_embs, pub_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    bge_fused = rrf_fuse([specter_ranking, bge_ranking], k=RRF_K, top_k=FINAL_TOP_K)
    print("\n--- SPECTER2 + BGE-large (2-way RRF) ---")
    evaluate(bge_fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Snowflake Arctic ──
    print(f"\nLoading {ARCTIC_MODEL_NAME}...")
    arctic_model = SentenceTransformer(ARCTIC_MODEL_NAME)
    print("Encoding corpus with Snowflake Arctic...")
    arctic_corpus_embs = arctic_model.encode(corpus_enriched, normalize_embeddings=True,
                                              show_progress_bar=True).astype(np.float32)
    print("Arctic query encoding...")
    arctic_q_embs = arctic_model.encode(pub_enriched, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
    arctic_ranking = dense_retrieve(arctic_q_embs, pub_ids,
                                     arctic_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    arctic_fused = rrf_fuse([specter_ranking, arctic_ranking], k=RRF_K, top_k=FINAL_TOP_K)
    print("\n--- SPECTER2 + Snowflake Arctic (2-way RRF) ---")
    evaluate(arctic_fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Held-out predictions with Snowflake ──
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]

    print("\nSPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("Arctic (held-out)...")
    ho_arctic_embs = arctic_model.encode(ho_enriched, normalize_embeddings=True,
                                          show_progress_bar=True).astype(np.float32)
    ho_arctic = dense_retrieve(ho_arctic_embs, ho_ids,
                                arctic_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    ho_fused = rrf_fuse([ho_specter, ho_arctic], k=RRF_K, top_k=FINAL_TOP_K)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "arctic_specter2_rrf.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()
