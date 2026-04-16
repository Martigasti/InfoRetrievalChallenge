"""
Alternative pipeline: SPECTER2 + BM25-PRF + cross-encoder reranking.

Fundamentally different from the main pipeline (SPECTER2 + BGE + BM25 + RRF):
  - Drops BGE entirely (less VRAM, faster, avoids retriever overlap)
  - Adds Pseudo-Relevance Feedback (PRF) to BM25: run initial BM25, take
    top-5 docs, extract discriminative terms, expand query, re-run BM25.
    This pulls in terminology the original query lacks (method names, dataset
    names, acronyms from related papers).
  - Uses score-level interpolation instead of RRF: normalise SPECTER2 cosine
    and BM25-PRF scores to [0,1], weighted sum. Preserves retriever confidence
    (RRF only uses ranks, throwing away whether rank-5 had 0.95 or 0.45 sim).
  - Cross-encoder reranks top-50 of the fused list.

Why this might beat the main pipeline:
  - PRF compensates for the missing BGE signal by enriching BM25 queries
  - Score fusion is more informative than rank fusion for 2-way combinations
  - Different failure modes → useful as a second submission even if slightly
    worse overall
"""

import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_queries, load_corpus, load_qrels,
    format_text, get_body_chunks, evaluate,
)

RETRIEVAL_TOP_K = 300
FINAL_TOP_K = 100
RERANK_TOP_K = 50
BODY_CHUNKS = 6

# PRF parameters
PRF_TOP_DOCS = 5          # number of top BM25 docs to extract expansion terms from
PRF_EXPANSION_TERMS = 20  # number of terms to add to the query

# Score interpolation: final = alpha * specter_norm + (1-alpha) * bm25_prf_norm
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]

SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
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


# ── BM25 tokenization ───────────────────────────────────────

def bm25_tokenize(text):
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import re
    stemmer = PorterStemmer()
    stops = set(stopwords.words("english"))
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]


# ── BM25 with Pseudo-Relevance Feedback ─────────────────────

def bm25_prf_retrieve(query_texts, q_ids, corpus_tokenized, c_ids, bm25_model,
                      top_k, prf_top=5, prf_terms=20):
    """
    Two-pass BM25 with pseudo-relevance feedback:
      Pass 1: standard BM25 → top-prf_top docs
      Pass 2: expand query with top-prf_terms discriminative terms from
              those docs, re-run BM25 → final ranking

    Returns both rankings and raw BM25 scores for score-level fusion.
    """
    # Precompute document frequencies for IDF-like term weighting
    doc_freq = Counter()
    for tokens in corpus_tokenized:
        for t in set(tokens):
            doc_freq[t] += 1
    n_docs = len(corpus_tokenized)

    results = {}
    scores_dict = {}
    for qid, q_text in tqdm(zip(q_ids, query_texts), total=len(q_ids),
                             desc="BM25-PRF retrieval"):
        q_tokens = bm25_tokenize(q_text)
        q_token_set = set(q_tokens)

        # Pass 1: initial retrieval
        initial_scores = bm25_model.get_scores(q_tokens)
        top_indices = np.argsort(-initial_scores)[:prf_top]

        # Extract expansion terms from top docs
        term_scores = Counter()
        for idx in top_indices:
            doc_tokens = corpus_tokenized[idx]
            tf = Counter(doc_tokens)
            for term, count in tf.items():
                if term in q_token_set:
                    continue  # skip terms already in query
                # TF-IDF-like score: tf * log(N/df)
                idf = np.log(n_docs / (1 + doc_freq.get(term, 0)))
                term_scores[term] += count * idf

        expansion = [t for t, _ in term_scores.most_common(prf_terms)]

        # Pass 2: expanded query
        expanded_tokens = q_tokens + expansion
        prf_scores = bm25_model.get_scores(expanded_tokens)
        top_indices = np.argsort(-prf_scores)[:top_k]

        results[qid] = [c_ids[i] for i in top_indices]
        scores_dict[qid] = {c_ids[i]: float(prf_scores[i]) for i in top_indices}

    return results, scores_dict


# ── Dense retrieval with scores ──────────────────────────────

def dense_retrieve_with_scores(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    rankings = {}
    scores = {}
    for i, qid in enumerate(q_ids):
        doc_ids = [c_ids[j] for j in top_indices[i]]
        rankings[qid] = doc_ids
        scores[qid] = {c_ids[j]: float(sim_matrix[i, j]) for j in top_indices[i]}
    return rankings, scores


# ── Score-level fusion ───────────────────────────────────────

def score_fuse(specter_scores, bm25_scores, alpha, top_k=100):
    """
    Fuse by normalised score interpolation:
      final(d) = alpha * specter_norm(d) + (1 - alpha) * bm25_norm(d)

    Documents not retrieved by one system get score 0 after normalisation.
    """
    all_qids = set(specter_scores) | set(bm25_scores)
    fused = {}
    for qid in all_qids:
        sp = specter_scores.get(qid, {})
        bm = bm25_scores.get(qid, {})
        all_docs = set(sp) | set(bm)

        # Min-max normalise each retriever's scores for this query
        if sp:
            sp_vals = np.array(list(sp.values()))
            sp_min, sp_max = sp_vals.min(), sp_vals.max()
            sp_range = sp_max - sp_min if sp_max - sp_min > 1e-9 else 1.0
        else:
            sp_min, sp_range = 0, 1.0

        if bm:
            bm_vals = np.array(list(bm.values()))
            bm_min, bm_max = bm_vals.min(), bm_vals.max()
            bm_range = bm_max - bm_min if bm_max - bm_min > 1e-9 else 1.0
        else:
            bm_min, bm_range = 0, 1.0

        combined = {}
        for doc in all_docs:
            sp_norm = (sp.get(doc, sp_min) - sp_min) / sp_range if doc in sp else 0.0
            bm_norm = (bm.get(doc, bm_min) - bm_min) / bm_range if doc in bm else 0.0
            combined[doc] = alpha * sp_norm + (1 - alpha) * bm_norm

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
        fused[qid] = [doc for doc, _ in ranked]
    return fused


# ── Reranking ────────────────────────────────────────────────

def rerank_topk(fused_ranking, query_texts_by_id, doc_text_by_id,
                reranker, top_rerank=50):
    reranked = {}
    for qid in tqdm(fused_ranking, desc="Reranking"):
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
    from sentence_transformers import CrossEncoder
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

    print(f"Building enriched text (body_chunks={BODY_CHUNKS})...")
    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter  = [format_specter(row)  for _, row in corpus.iterrows()]

    corpus_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in corpus.iterrows()
    }

    # ── BM25 index ──
    print("Tokenizing corpus for BM25...")
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

    # ══════════════════════════════════════════════════════════════
    # Public queries — retrieve + grid search + rerank
    # ══════════════════════════════════════════════════════════════
    pub_ids      = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter  = [format_specter(row)  for _, row in queries.iterrows()]
    pub_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in queries.iterrows()
    }

    print("\nSPECTER2 retrieval...")
    specter_q_embs = encode_specter(pub_specter, tokenizer,
                                     specter_model, batch_size=32, device=device)
    specter_ranking, specter_scores = dense_retrieve_with_scores(
        specter_q_embs, pub_ids, specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25-PRF retrieval...")
    bm25_ranking, bm25_scores = bm25_prf_retrieve(
        pub_enriched, pub_ids, corpus_tokenized, corpus_ids, bm25,
        top_k=RETRIEVAL_TOP_K, prf_top=PRF_TOP_DOCS, prf_terms=PRF_EXPANSION_TERMS)

    # Grid search over alpha
    print("\n" + "=" * 50)
    print(f"Grid search: alpha (pool={RETRIEVAL_TOP_K}, PRF={PRF_TOP_DOCS}docs/{PRF_EXPANSION_TERMS}terms)")
    print("=" * 50)
    best_alpha, best_ndcg = None, 0
    for alpha in ALPHAS:
        fused = score_fuse(specter_scores, bm25_scores, alpha, top_k=FINAL_TOP_K)
        res = evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
        ndcg  = res["overall"]["NDCG@10"]
        mapv  = res["overall"]["MAP"]
        rec   = res["overall"]["Recall@100"]
        print(f"  alpha={alpha:.1f}  NDCG@10={ndcg:.4f}  MAP={mapv:.4f}  Recall@100={rec:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = alpha

    print(f"\n*** Best alpha={best_alpha} (NDCG@10={best_ndcg:.4f}) ***")

    fused_best = score_fuse(specter_scores, bm25_scores, best_alpha, top_k=FINAL_TOP_K)
    print(f"\n--- Score fusion baseline (alpha={best_alpha}, no reranker) ---")
    evaluate(fused_best, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Free SPECTER2, load reranker ──
    print("Freeing SPECTER2 from GPU for reranker...")
    specter_model.to("cpu")
    torch.cuda.empty_cache()

    print(f"\nLoading {RERANKER_NAME}...")
    reranker = CrossEncoder(RERANKER_NAME, max_length=RERANKER_MAX_LEN, device=device)

    print(f"Reranking top-{RERANK_TOP_K}...")
    reranked = rerank_topk(
        fused_best, pub_rerank_by_id, corpus_rerank_by_id,
        reranker, top_rerank=RERANK_TOP_K,
    )

    print(f"\n--- Score fusion + reranker (alpha={best_alpha}, top-{RERANK_TOP_K}) ---")
    evaluate(reranked, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict
    # ══════════════════════════════════════════════════════════════
    ho_ids      = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter  = [format_specter(row)  for _, row in held_out.iterrows()]
    ho_rerank_by_id = {
        row["doc_id"]: format_rerank(row) for _, row in held_out.iterrows()
    }

    # Swap reranker off, SPECTER2 back on
    print("\nSwapping reranker off GPU for held-out retrieval...")
    reranker.model.to("cpu")
    torch.cuda.empty_cache()
    specter_model.to(device)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter_ranking, ho_specter_scores = dense_retrieve_with_scores(
        ho_specter_embs, ho_ids, specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25-PRF (held-out)...")
    ho_bm25_ranking, ho_bm25_scores = bm25_prf_retrieve(
        ho_enriched, ho_ids, corpus_tokenized, corpus_ids, bm25,
        top_k=RETRIEVAL_TOP_K, prf_top=PRF_TOP_DOCS, prf_terms=PRF_EXPANSION_TERMS)

    ho_fused = score_fuse(ho_specter_scores, ho_bm25_scores, best_alpha, top_k=FINAL_TOP_K)

    # Swap back to reranker
    print("\nSwapping SPECTER2 off GPU for reranker (held-out)...")
    specter_model.to("cpu")
    torch.cuda.empty_cache()
    reranker.model.to(device)

    print(f"Reranking top-{RERANK_TOP_K} on held-out...")
    ho_reranked = rerank_topk(
        ho_fused, ho_rerank_by_id, corpus_rerank_by_id,
        reranker, top_rerank=RERANK_TOP_K,
    )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_reranked))
    print(f"\nSaved -> {zip_path}  [SPECTER2+BM25-PRF, alpha={best_alpha}, "
          f"rerank_top={RERANK_TOP_K}]")


if __name__ == "__main__":
    main()
