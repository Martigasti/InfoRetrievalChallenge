"""
HyDE (Hypothetical Document Embeddings) + 3-way Weighted RRF.

Builds on bge_weighted_rrf.py. For each query paper, a small LLM
(flan-t5-base) generates a hypothetical abstract of a paper that the
query would likely cite. SPECTER2 and BGE embed that hypothetical
instead of the raw query, putting queries in the same representation
space as corpus documents.

BM25 still uses the original query text — no representational gap there.

Hypotheticals are cached to cache/hyde_hypotheticals.json so generation
only runs once. T5 is freed before loading the dense models.
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
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"

HYDE_MODEL = "google/flan-t5-base"
HYDE_BATCH_SIZE = 16
HYDE_MAX_INPUT = 512
HYDE_MAX_NEW_TOKENS = 150

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"
CACHE_DIR = ROOT / "cache"
HYDE_CACHE = CACHE_DIR / "hyde_hypotheticals.json"

HYDE_PROMPT = (
    "A researcher wrote a paper titled \"{title}\". "
    "The abstract is: {abstract} "
    "Write a short abstract of a different paper that this paper would likely cite."
)


# ── HyDE generation ──────────────────────────────────────────

def build_prompt(row):
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    # Truncate abstract to keep prompt manageable
    if len(abstract) > 800:
        abstract = abstract[:800] + "..."
    return HYDE_PROMPT.format(title=title, abstract=abstract)


def generate_hypotheticals(query_df, device):
    """
    For each query row, generate a hypothetical cited-paper abstract.
    Returns dict: doc_id -> hypothetical string.
    """
    q_ids = query_df["doc_id"].tolist()

    if HYDE_CACHE.exists():
        with open(HYDE_CACHE) as f:
            cached = json.load(f)
        if all(qid in cached for qid in q_ids):
            print(f"Loaded {len(cached)} hypotheticals from cache.")
            return cached
        print(f"Cache incomplete ({len(cached)}/{len(q_ids)}), generating missing...")
    else:
        cached = {}

    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    print(f"\nLoading {HYDE_MODEL} for HyDE generation...")
    tok   = T5Tokenizer.from_pretrained(HYDE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(HYDE_MODEL)
    model.to(device).eval()

    hypotheticals = dict(cached)
    todo_rows = [(qid, row) for qid, row in zip(q_ids, query_df.itertuples())
                 if qid not in hypotheticals]
    todo_rows_data = [(qid, query_df[query_df["doc_id"] == qid].iloc[0])
                      for qid in q_ids if qid not in hypotheticals]

    print(f"Generating hypotheticals for {len(todo_rows_data)} queries...")
    for i in tqdm(range(0, len(todo_rows_data), HYDE_BATCH_SIZE), desc="HyDE generation"):
        batch = todo_rows_data[i:i + HYDE_BATCH_SIZE]
        batch_ids    = [qid for qid, _ in batch]
        batch_prompts = [build_prompt(row) for _, row in batch]

        inputs = tok(
            batch_prompts,
            max_length=HYDE_MAX_INPUT,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=HYDE_MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        for qid, text in zip(batch_ids, decoded):
            hypotheticals[qid] = text

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(HYDE_CACHE, "w") as f:
        json.dump(hypotheticals, f)
    print(f"Saved -> {HYDE_CACHE}")

    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()

    return hypotheticals


# ── Text formatting ──────────────────────────────────────────

def format_specter(row):
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
    base = format_text(row)
    try:
        chunks = get_body_chunks(row, min_chars=50)
        if chunks:
            base += " " + " ".join(chunks[:3])
    except Exception:
        pass
    return base


def format_hyde_specter(hyp_text):
    """Wrap hypothetical abstract in SPECTER2's expected format (no title available)."""
    return "[SEP] " + hyp_text if hyp_text else ""


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
    nltk.download("punkt", quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    queries  = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus   = load_corpus(DATA_DIR / "corpus.parquet")
    qrels    = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids    = corpus["doc_id"].tolist()
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    # ── HyDE: generate hypotheticals for public + held-out queries ──
    # Run T5 first, then free GPU before loading dense models.
    import pandas as pd
    all_queries = pd.concat([queries, held_out], ignore_index=True)
    hypotheticals = generate_hypotheticals(all_queries, device)

    # ── Corpus text (unchanged from bge_weighted_rrf) ──
    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter  = [format_specter(row)  for _, row in corpus.iterrows()]

    # ── BM25 index (unchanged — original text, no HyDE) ──
    print("\nTokenizing corpus for BM25...")
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_enriched, desc="BM25 tokenizing")]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── SPECTER2 ──
    print(f"\nLoading {SPECTER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
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
    # Public queries — encode hypotheticals, retrieve, evaluate
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]

    # Dense query embeddings use the hypothetical, not the original query
    pub_hyde_specter = [format_hyde_specter(hypotheticals.get(qid, "")) for qid in pub_ids]
    pub_hyde_plain   = [hypotheticals.get(qid, format_enriched(row))
                        for qid, (_, row) in zip(pub_ids, queries.iterrows())]

    print("\nBGE retrieval (HyDE queries)...")
    bge_q_embs = bge_model.encode(pub_hyde_plain, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
    bge_ranking = dense_retrieve(bge_q_embs, pub_ids,
                                  bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval (HyDE queries)...")
    specter_q_embs = encode_specter(pub_hyde_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking = dense_retrieve(specter_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval (original query text)...")
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
            [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, bm25_w)],
            k=RRF_K, top_k=FINAL_TOP_K,
        )
        res = evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
        ndcg  = res["overall"]["NDCG@10"]
        mapv  = res["overall"]["MAP"]
        rec   = res["overall"]["Recall@100"]
        print(f"  bm25_w={bm25_w:.1f}  NDCG@10={ndcg:.4f}  MAP={mapv:.4f}  Recall@100={rec:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_cfg  = bm25_w

    print(f"\n*** Best bm25_weight={best_cfg} (NDCG@10={best_ndcg:.4f}) ***")

    fused_best = weighted_rrf_fuse(
        [(specter_ranking, 1.0), (bge_ranking, 1.0), (bm25_ranking, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )
    print(f"\n--- HyDE + Weighted RRF (specter=1.0, bge=1.0, bm25={best_cfg}) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — retrieve + predict
    # ══════════════════════════════════════════════════════════════
    ho_ids      = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]

    ho_hyde_specter = [format_hyde_specter(hypotheticals.get(qid, "")) for qid in ho_ids]
    ho_hyde_plain   = [hypotheticals.get(qid, format_enriched(row))
                       for qid, (_, row) in zip(ho_ids, held_out.iterrows())]

    print("\nBGE (held-out, HyDE)...")
    ho_bge_embs = bge_model.encode(ho_hyde_plain, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    ho_bge = dense_retrieve(ho_bge_embs, ho_ids,
                             bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out, HyDE)...")
    ho_specter_embs = encode_specter(ho_hyde_specter, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter_rank = dense_retrieve(ho_specter_embs, ho_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out, original text)...")
    ho_bm25 = bm25_retrieve(ho_enriched, ho_ids, corpus_tokenized,
                              corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    ho_fused = weighted_rrf_fuse(
        [(ho_specter_rank, 1.0), (ho_bge, 1.0), (ho_bm25, best_cfg)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path}  [HyDE + weighted RRF, bm25_w={best_cfg}]")


if __name__ == "__main__":
    main()
