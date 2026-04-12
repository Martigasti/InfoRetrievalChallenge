"""
3-way Weighted RRF + BM25 Pseudo-Relevance Feedback (RM3-style query expansion).

Same pipeline as bge_weighted_rrf.py, but the BM25 leg is run twice:

  1. Initial BM25 with the raw query  ->  fuse with SPECTER2 + BGE  ->  initial_fusion.
  2. Take top-N docs from initial_fusion as the pseudo-relevant set.
  3. Score every term in those docs by (tf_in_pseudo_rel * idf_corpus),
     pick the top-M terms not already in the original query.
  4. Build the expanded query: original tokens repeated ORIG_WEIGHT times + new terms.
  5. Re-run BM25 with the expanded queries -> bm25_prf_ranking.
  6. Final fusion: SPECTER2 + BGE + bm25_prf_ranking.

The intent is to fix lexical-overlap weakness in abstract domains (Geography,
Engineering, Philosophy) where the query paper's vocabulary often does not
overlap with its cited papers'. Dense models stay untouched, so the cost is
exactly one extra BM25 retrieval pass per grid point.
"""

import json
import os
import sys
from collections import Counter, defaultdict
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

# Weights from bge_weighted_rrf.py grid (kept fixed so the only varying factor
# is BM25-with-PRF vs BM25-without-PRF).
BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0

# PRF parameters
PRF_NUM_DOCS = 5                       # how many top fusion docs to treat as pseudo-relevant
PRF_ORIG_WEIGHT = 2                    # times to repeat original tokens (higher = trust the query more)
PRF_EXPANSION_TERMS_GRID = [5, 10, 20] # number of new terms to inject

SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"

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


def bm25_retrieve_tokens(q_tokens_list, q_ids, bm25_model, c_ids, top_k, desc):
    """Run BM25 retrieval given pre-tokenized query token lists."""
    results = {}
    for qid, q_tokens in tqdm(zip(q_ids, q_tokens_list), total=len(q_ids), desc=desc):
        scores = bm25_model.get_scores(q_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        results[qid] = [c_ids[i] for i in top_indices]
    return results


# ── Pseudo-relevance feedback ────────────────────────────────

def build_expanded_tokens(orig_tokens, pseudo_rel_doc_indices, corpus_tokenized,
                          idf_dict, num_expansion_terms, orig_weight):
    """
    RM3 / Rocchio-flavoured query expansion.

    For each candidate term t appearing in the pseudo-relevant docs, score it as
        score(t) = tf_in_pseudo_rel(t) * idf_corpus(t)
    Pick the top-`num_expansion_terms` terms that are not already in the query.
    Returned token list = orig_tokens * orig_weight + new_terms, so the original
    query still dominates the BM25 score.
    """
    orig_set = set(orig_tokens)
    term_counts = Counter()
    for idx in pseudo_rel_doc_indices:
        term_counts.update(corpus_tokenized[idx])

    candidates = []
    for term, tf in term_counts.items():
        if term in orig_set:
            continue
        idf = idf_dict.get(term, 0.0)
        if idf <= 0:  # BM25Okapi can return negative idf for very common terms
            continue
        candidates.append((term, tf * idf))

    candidates.sort(key=lambda x: -x[1])
    expansion_terms = [t for t, _ in candidates[:num_expansion_terms]]
    return list(orig_tokens) * orig_weight + expansion_terms


def build_prf_token_lists(query_orig_tokens, q_ids, initial_fusion,
                          corpus_id_to_idx, corpus_tokenized, bm25_model,
                          num_prf_docs, num_expansion_terms, orig_weight):
    """For each query, build the expanded BM25 token list using PRF."""
    expanded = []
    for qid, orig_tokens in zip(q_ids, query_orig_tokens):
        top_doc_ids = initial_fusion[qid][:num_prf_docs]
        pseudo_rel_idxs = [corpus_id_to_idx[d] for d in top_doc_ids if d in corpus_id_to_idx]
        new_tokens = build_expanded_tokens(
            orig_tokens, pseudo_rel_idxs, corpus_tokenized,
            bm25_model.idf, num_expansion_terms, orig_weight,
        )
        expanded.append(new_tokens)
    return expanded


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
    corpus_id_to_idx = {d: i for i, d in enumerate(corpus_ids)}
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

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

    # ── BGE-large ──
    print(f"\nLoading {BGE_MODEL_NAME}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched,
                                        normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — base retrievals
    # ══════════════════════════════════════════════════════════════
    pub_ids = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter = [format_specter(row) for _, row in queries.iterrows()]

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

    print("BM25 (initial) retrieval...")
    pub_orig_tokens = [bm25_tokenize(t) for t in pub_enriched]
    bm25_ranking = bm25_retrieve_tokens(pub_orig_tokens, pub_ids, bm25,
                                         corpus_ids, top_k=RETRIEVAL_TOP_K,
                                         desc="BM25 (initial)")

    # ── Initial fusion (also serves as the pseudo-relevant source) ──
    initial_fusion = weighted_rrf_fuse(
        [(specter_ranking, DENSE_WEIGHT), (bge_ranking, DENSE_WEIGHT),
         (bm25_ranking, BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K
    )

    print("\n--- Baseline: Weighted RRF (no PRF) ---")
    base_res = evaluate(initial_fusion, qrels, ks=[10, 100],
                        query_domains=query_domains, verbose=True)
    base_ndcg = base_res["overall"]["NDCG@10"]

    # ══════════════════════════════════════════════════════════════
    # PRF grid search (only num_expansion_terms varies)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PRF grid: num_prf_docs={PRF_NUM_DOCS}, orig_weight={PRF_ORIG_WEIGHT}")
    print("=" * 60)

    best_terms, best_ndcg, best_fused = None, base_ndcg, initial_fusion
    for num_terms in PRF_EXPANSION_TERMS_GRID:
        print(f"\n[m={num_terms}] building expanded queries...")
        expanded_tokens = build_prf_token_lists(
            pub_orig_tokens, pub_ids, initial_fusion, corpus_id_to_idx,
            corpus_tokenized, bm25, PRF_NUM_DOCS, num_terms, PRF_ORIG_WEIGHT,
        )
        bm25_prf = bm25_retrieve_tokens(expanded_tokens, pub_ids, bm25,
                                         corpus_ids, top_k=RETRIEVAL_TOP_K,
                                         desc=f"BM25-PRF (m={num_terms})")
        fused = weighted_rrf_fuse(
            [(specter_ranking, DENSE_WEIGHT), (bge_ranking, DENSE_WEIGHT),
             (bm25_prf, BM25_WEIGHT)],
            k=RRF_K, top_k=FINAL_TOP_K
        )
        res = evaluate(fused, qrels, ks=[10, 100],
                       query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        mapv = res["overall"]["MAP"]
        recall = res["overall"]["Recall@100"]
        delta = ndcg - base_ndcg
        print(f"  m={num_terms:>2}  NDCG@10={ndcg:.4f}  MAP={mapv:.4f}  "
              f"Recall@100={recall:.4f}  (delta={delta:+.4f})")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_terms = num_terms
            best_fused = fused

    if best_terms is None:
        print("\n*** PRF did not improve over baseline. Submitting baseline. ***")
    else:
        print(f"\n*** Best PRF config: m={best_terms} (NDCG@10={best_ndcg:.4f}) ***")
        print(f"\n--- Final: Weighted RRF + BM25-PRF (m={best_terms}) ---")
        evaluate(best_fused, qrels, ks=[10, 100],
                 query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — predict using the chosen config
    # ══════════════════════════════════════════════════════════════
    ho_ids = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter_texts = [format_specter(row) for _, row in held_out.iterrows()]

    print("\nBGE (held-out)...")
    ho_bge_embs = bge_model.encode(ho_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    ho_bge = dense_retrieve(ho_bge_embs, ho_ids,
                             bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_specter_embs = encode_specter(ho_specter_texts, tokenizer,
                                      specter_model, batch_size=32, device=device)
    ho_specter = dense_retrieve(ho_specter_embs, ho_ids,
                                 specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out, initial)...")
    ho_orig_tokens = [bm25_tokenize(t) for t in ho_enriched]
    ho_bm25 = bm25_retrieve_tokens(ho_orig_tokens, ho_ids, bm25, corpus_ids,
                                    top_k=RETRIEVAL_TOP_K,
                                    desc="BM25 (held-out)")

    ho_initial = weighted_rrf_fuse(
        [(ho_specter, DENSE_WEIGHT), (ho_bge, DENSE_WEIGHT),
         (ho_bm25, BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K
    )

    if best_terms is not None:
        print(f"\nBM25-PRF (held-out, m={best_terms})...")
        ho_expanded_tokens = build_prf_token_lists(
            ho_orig_tokens, ho_ids, ho_initial, corpus_id_to_idx,
            corpus_tokenized, bm25, PRF_NUM_DOCS, best_terms, PRF_ORIG_WEIGHT,
        )
        ho_bm25_prf = bm25_retrieve_tokens(ho_expanded_tokens, ho_ids, bm25,
                                            corpus_ids, top_k=RETRIEVAL_TOP_K,
                                            desc="BM25-PRF (held-out)")
        ho_fused = weighted_rrf_fuse(
            [(ho_specter, DENSE_WEIGHT), (ho_bge, DENSE_WEIGHT),
             (ho_bm25_prf, BM25_WEIGHT)],
            k=RRF_K, top_k=FINAL_TOP_K
        )
    else:
        ho_fused = ho_initial

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    import zipfile
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_fused))
    print(f"\nSaved -> {zip_path} (contains submission_data.json)")


if __name__ == "__main__":
    main()
