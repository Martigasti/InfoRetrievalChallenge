"""
Phase 6: doc2query-T5 document expansion + LightGBM LambdaRank fusion.

Builds on lightgbm_fusion.py by adding a doc2query expansion step:
  - For each corpus document, generate synthetic queries with doc2query-T5
  - Append these to the BM25 document text (not the dense encoders)
  - This expands the lexical surface so BM25 retrieves papers that share
    concepts but not exact terms with the query

The dense retrievers (SPECTER2, BGE) are unchanged — doc2query only
improves first-stage BM25 recall, which feeds better candidates into
the LightGBM reranker.

Expansions are cached to disk so the T5 generation only runs once.

Pipeline:
  1. Generate & cache doc2query expansions for the corpus
  2. Build BM25 index over expanded text, encode corpus (SPECTER2 + BGE)
  3. Retrieve top-300 from all 3 retrievers, keeping scores + ranks
  4. Build feature matrix, 5-fold GroupKFold CV
  5. If CV NDCG@10 > baseline: use LightGBM, else fall back to weighted RRF
"""

import json
import math
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
FINAL_TOP_K = 100
RRF_K = 10
BM25_WEIGHT = 0.5
BODY_CHUNKS = 6
SENTINEL_RANK = RETRIEVAL_TOP_K + 1

SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"

DOC2QUERY_MODEL = "doc2query/msmarco-t5-base-v1"
DOC2QUERY_NUM_QUERIES = 10   # synthetic queries per document
DOC2QUERY_BATCH_SIZE = 8     # T5 batch size (limited by VRAM)

RRF_BASELINE_NDCG10 = 0.583

N_CV_FOLDS = 5
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 1,
    "verbose": -1,
    "n_jobs": -1,
}
LGB_ROUNDS = 300
LGB_EARLY_STOP = 40

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"
CACHE_DIR = ROOT / "cache"
DOC2QUERY_CACHE = CACHE_DIR / "doc2query_expansions.json"


# ── doc2query expansion ─────────────────────────────────────

def generate_doc2query_expansions(corpus_texts, corpus_ids, num_queries=10,
                                   batch_size=8, device="cpu"):
    """
    Generate synthetic queries for each document using doc2query-T5.
    Returns dict: doc_id -> list of generated query strings.
    """
    # Check cache first
    if DOC2QUERY_CACHE.exists():
        print(f"Loading cached doc2query expansions from {DOC2QUERY_CACHE}")
        with open(DOC2QUERY_CACHE) as f:
            cached = json.load(f)
        # Verify cache covers all docs
        if all(did in cached for did in corpus_ids):
            print(f"  Cache hit: {len(cached)} documents")
            return cached
        print(f"  Cache incomplete ({len(cached)}/{len(corpus_ids)}), regenerating missing...")

    from transformers import T5ForConditionalGeneration, T5Tokenizer

    print(f"\nLoading {DOC2QUERY_MODEL}...")
    t5_tokenizer = T5Tokenizer.from_pretrained(DOC2QUERY_MODEL)
    t5_model = T5ForConditionalGeneration.from_pretrained(DOC2QUERY_MODEL)
    t5_model.to(device).eval()

    import torch

    # Load partial cache if it exists
    expansions = {}
    if DOC2QUERY_CACHE.exists():
        with open(DOC2QUERY_CACHE) as f:
            expansions = json.load(f)

    # Find docs that need generation
    todo_indices = [i for i, did in enumerate(corpus_ids) if did not in expansions]
    print(f"Generating {num_queries} synthetic queries for {len(todo_indices)} documents...")

    for batch_start in tqdm(range(0, len(todo_indices), batch_size),
                            desc="doc2query generation"):
        batch_indices = todo_indices[batch_start:batch_start + batch_size]
        batch_texts = [corpus_texts[i] for i in batch_indices]
        batch_ids = [corpus_ids[i] for i in batch_indices]

        inputs = t5_tokenizer(
            batch_texts,
            max_length=384,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = t5_model.generate(
                **inputs,
                max_length=64,
                num_return_sequences=num_queries,
                num_beams=num_queries,
                early_stopping=True,
            )

        decoded = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # outputs has (batch_size * num_queries) sequences
        for j, did in enumerate(batch_ids):
            start = j * num_queries
            end = start + num_queries
            expansions[did] = decoded[start:end]

        # Checkpoint every 500 batches
        if (batch_start // batch_size) % 500 == 499:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(DOC2QUERY_CACHE, "w") as f:
                json.dump(expansions, f)
            print(f"  Checkpoint saved ({len(expansions)}/{len(corpus_ids)} docs)")

    # Save final cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(DOC2QUERY_CACHE, "w") as f:
        json.dump(expansions, f)
    print(f"Saved doc2query cache -> {DOC2QUERY_CACHE}")

    # Free T5 memory
    del t5_model, t5_tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return expansions


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


def format_bm25_expanded(row, expansions):
    """BM25 text with doc2query expansions appended."""
    base = format_enriched(row)
    doc_id = row.get("doc_id", "")
    if doc_id in expansions:
        base += " " + " ".join(expansions[doc_id])
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


# ── Retrieval with scores ────────────────────────────────────

def dense_retrieve_scored(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    result = {}
    for i, qid in enumerate(q_ids):
        top_idx = np.argsort(-sim_matrix[i])[:top_k]
        result[qid] = [
            (c_ids[j], float(sim_matrix[i, j]), rank + 1)
            for rank, j in enumerate(top_idx)
        ]
    return result


def bm25_tokenize(text):
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import re
    stemmer = PorterStemmer()
    stops = set(stopwords.words("english"))
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]


def bm25_retrieve_scored(query_texts, q_ids, corpus_tokenized, c_ids, bm25_model, top_k):
    result = {}
    for qid, q_text in tqdm(zip(q_ids, query_texts), total=len(q_ids), desc="BM25 retrieval"):
        q_tokens = bm25_tokenize(q_text)
        scores = bm25_model.get_scores(q_tokens)
        top_idx = np.argsort(-scores)[:top_k]
        result[qid] = [
            (c_ids[j], float(scores[j]), rank + 1)
            for rank, j in enumerate(top_idx)
        ]
    return result


# ── Feature construction ─────────────────────────────────────

def build_query_features(qid, specter_res, bge_res, bm25_res, relevant_set=None):
    spec_dict = {d: (s, r) for d, s, r in specter_res.get(qid, [])}
    bge_dict  = {d: (s, r) for d, s, r in bge_res.get(qid, [])}
    bm25_dict = {d: (s, r) for d, s, r in bm25_res.get(qid, [])}

    candidates = list(set(spec_dict) | set(bge_dict) | set(bm25_dict))

    rows, labels, doc_ids = [], [], []
    for doc_id in candidates:
        s_score, s_rank = spec_dict.get(doc_id, (0.0, SENTINEL_RANK))
        b_score, b_rank = bge_dict.get(doc_id,  (0.0, SENTINEL_RANK))
        m_score, m_rank = bm25_dict.get(doc_id, (0.0, SENTINEL_RANK))

        n_ret = (1 if doc_id in spec_dict else 0) + \
                (1 if doc_id in bge_dict  else 0) + \
                (1 if doc_id in bm25_dict else 0)

        rrf = (
            (1.0 / (RRF_K + s_rank)          if s_rank < SENTINEL_RANK else 0.0) +
            (1.0 / (RRF_K + b_rank)          if b_rank < SENTINEL_RANK else 0.0) +
            (BM25_WEIGHT / (RRF_K + m_rank)  if m_rank < SENTINEL_RANK else 0.0)
        )

        rows.append([
            s_rank, b_rank, m_rank,
            s_score, b_score, m_score,
            n_ret,
            min(s_rank, b_rank, m_rank),
            rrf,
        ])
        labels.append(1 if (relevant_set and doc_id in relevant_set) else 0)
        doc_ids.append(doc_id)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32), doc_ids


# ── NDCG@10 helper ───────────────────────────────────────────

def ndcg10(pred_scores, true_labels):
    order = np.argsort(-pred_scores)
    ranked = true_labels[order]
    dcg  = sum(ranked[i] / math.log2(i + 2) for i in range(min(10, len(ranked))))
    ideal = sorted(true_labels, reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(min(10, len(ideal))))
    return dcg / idcg if idcg > 0 else 0.0


# ── Weighted RRF fallback ────────────────────────────────────

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


def scored_to_ranking(scored_res):
    return {qid: [d for d, _, _ in entries] for qid, entries in scored_res.items()}


# ── Main ─────────────────────────────────────────────────────

def main():
    import torch
    import nltk
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
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

    # ── doc2query expansion (runs T5 once, then cached) ──
    print(f"\nBuilding enriched text (body_chunks={BODY_CHUNKS})...")
    corpus_enriched = [format_enriched(row) for _, row in corpus.iterrows()]
    corpus_specter  = [format_specter(row)  for _, row in corpus.iterrows()]

    # Use title + abstract as input to doc2query (cleaner signal than full body)
    corpus_ta = [format_text(row) for _, row in corpus.iterrows()]
    expansions = generate_doc2query_expansions(
        corpus_ta, corpus_ids,
        num_queries=DOC2QUERY_NUM_QUERIES,
        batch_size=DOC2QUERY_BATCH_SIZE,
        device=device,
    )

    # ── BM25 index with expanded text ──
    print("\nBuilding BM25 index with doc2query expansions...")
    corpus_bm25_expanded = [
        format_bm25_expanded(row, expansions) for _, row in corpus.iterrows()
    ]
    corpus_tokenized = [bm25_tokenize(t) for t in tqdm(corpus_bm25_expanded, desc="BM25 tokenizing")]
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
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    print("Encoding corpus with BGE-large...")
    bge_corpus_embs = bge_model.encode(corpus_enriched, normalize_embeddings=True,
                                        show_progress_bar=True).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # Public queries — retrieve with scores
    # ══════════════════════════════════════════════════════════════
    pub_ids      = queries["doc_id"].tolist()
    pub_enriched = [format_enriched(row) for _, row in queries.iterrows()]
    pub_specter  = [format_specter(row)  for _, row in queries.iterrows()]

    print("\nBGE retrieval (public, top-300)...")
    bge_q_embs = bge_model.encode(pub_enriched, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
    bge_pub = dense_retrieve_scored(bge_q_embs, pub_ids,
                                     bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 retrieval (public, top-300)...")
    spec_q_embs = encode_specter(pub_specter, tokenizer,
                                  specter_model, batch_size=32, device=device)
    spec_pub = dense_retrieve_scored(spec_q_embs, pub_ids,
                                      specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 retrieval (public, top-300)...")
    bm25_pub = bm25_retrieve_scored(pub_enriched, pub_ids, corpus_tokenized,
                                     corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    # ── Build per-query feature data ──
    print("\nBuilding feature matrix...")
    query_data = {}
    for qid in pub_ids:
        rel = set(qrels.get(qid, []))
        X_q, y_q, dids = build_query_features(qid, spec_pub, bge_pub, bm25_pub, rel)
        query_data[qid] = (X_q, y_q, dids)
        if y_q.sum() == 0:
            print(f"  WARNING: query {qid} has 0 relevant docs in candidates")

    total_cands = sum(len(v[2]) for v in query_data.values())
    total_rel   = sum(int(v[1].sum()) for v in query_data.values())
    print(f"  {total_cands} total candidates across {len(pub_ids)} queries "
          f"({total_rel} relevant, {total_cands - total_rel} non-relevant)")

    # ── 5-fold GroupKFold CV ──
    print(f"\n{'='*55}")
    print(f"{N_CV_FOLDS}-fold GroupKFold CV (LambdaRank, ndcg@10)")
    print(f"{'='*55}")
    print(f"RRF baseline to beat: {RRF_BASELINE_NDCG10:.4f}")

    X_all  = np.vstack([query_data[q][0] for q in pub_ids])
    y_all  = np.concatenate([query_data[q][1] for q in pub_ids])
    gid_all = np.concatenate([[q] * len(query_data[q][2]) for q in pub_ids])

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    cv_ndcg = []

    for fold, (train_rows, val_rows) in enumerate(gkf.split(X_all, y_all, groups=gid_all)):
        train_qids = list(dict.fromkeys(gid_all[train_rows]))
        val_qids   = list(dict.fromkeys(gid_all[val_rows]))

        X_tr = np.vstack([query_data[q][0] for q in train_qids])
        y_tr = np.concatenate([query_data[q][1] for q in train_qids])
        g_tr = [len(query_data[q][2]) for q in train_qids]

        X_va = np.vstack([query_data[q][0] for q in val_qids])
        y_va = np.concatenate([query_data[q][1] for q in val_qids])
        g_va = [len(query_data[q][2]) for q in val_qids]

        tr_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
        va_ds = lgb.Dataset(X_va, label=y_va, group=g_va, reference=tr_ds)

        model_cv = lgb.train(
            LGB_PARAMS, tr_ds,
            num_boost_round=LGB_ROUNDS,
            valid_sets=[va_ds],
            callbacks=[
                lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model_cv.predict(X_va)
        fold_scores = []
        start = 0
        for q in val_qids:
            n = len(query_data[q][2])
            score = ndcg10(preds[start:start + n], query_data[q][1])
            fold_scores.append(score)
            start += n

        fold_mean = float(np.mean(fold_scores))
        cv_ndcg.append(fold_mean)
        print(f"  Fold {fold+1}/{N_CV_FOLDS}: NDCG@10 = {fold_mean:.4f}"
              f"  ({len(val_qids)} val queries)")

    cv_mean = float(np.mean(cv_ndcg))
    cv_std  = float(np.std(cv_ndcg))
    print(f"\nCV mean NDCG@10 = {cv_mean:.4f} ± {cv_std:.4f}")
    use_lgbm = cv_mean > RRF_BASELINE_NDCG10
    print(f"Use LightGBM? {'YES' if use_lgbm else 'NO — falling back to weighted RRF'}")

    # ── Full-data evaluation (public) ──
    print("\nTraining on all 100 public queries for evaluation + held-out inference...")
    g_all = [len(query_data[q][2]) for q in pub_ids]
    full_ds = lgb.Dataset(X_all, label=y_all, group=g_all)
    model_full = lgb.train(
        LGB_PARAMS, full_ds,
        num_boost_round=LGB_ROUNDS,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    preds_pub = model_full.predict(X_all)
    pub_scores = []
    start = 0
    lgbm_ranking_pub = {}
    for q in pub_ids:
        n   = len(query_data[q][2])
        p   = preds_pub[start:start + n]
        dids = query_data[q][2]
        order = np.argsort(-p)
        lgbm_ranking_pub[q] = [dids[i] for i in order[:FINAL_TOP_K]]
        pub_scores.append(ndcg10(p, query_data[q][1]))
        start += n
    print(f"\n--- LightGBM public NDCG@10 (in-sample, informational) = "
          f"{np.mean(pub_scores):.4f} ---")
    print("(In-sample is always optimistic; CV mean above is the honest estimate)")

    rrf_ranking_pub = weighted_rrf_fuse(
        [(scored_to_ranking(spec_pub), 1.0),
         (scored_to_ranking(bge_pub),  1.0),
         (scored_to_ranking(bm25_pub), BM25_WEIGHT)],
        k=RRF_K, top_k=FINAL_TOP_K,
    )
    rrf_res = evaluate(rrf_ranking_pub, qrels, ks=[10, 100],
                       query_domains=query_domains, verbose=False)
    print(f"--- Weighted RRF public NDCG@10 (same run) = "
          f"{rrf_res['overall']['NDCG@10']:.4f} ---")

    if use_lgbm:
        print("\n--- Full evaluation: LightGBM fusion (doc2query-expanded BM25) ---")
        evaluate(lgbm_ranking_pub, qrels, ks=[10, 100],
                 query_domains=query_domains, verbose=True)
    else:
        print("\n--- Full evaluation: Weighted RRF (fallback, doc2query-expanded BM25) ---")
        evaluate(rrf_ranking_pub, qrels, ks=[10, 100],
                 query_domains=query_domains, verbose=True)

    # ══════════════════════════════════════════════════════════════
    # Held-out queries — retrieve + predict
    # ══════════════════════════════════════════════════════════════
    ho_ids      = held_out["doc_id"].tolist()
    ho_enriched = [format_enriched(row) for _, row in held_out.iterrows()]
    ho_specter  = [format_specter(row)  for _, row in held_out.iterrows()]

    print("\nBGE (held-out)...")
    ho_bge_embs = bge_model.encode(ho_enriched, normalize_embeddings=True,
                                    show_progress_bar=True).astype(np.float32)
    bge_ho = dense_retrieve_scored(ho_bge_embs, ho_ids,
                                    bge_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("SPECTER2 (held-out)...")
    ho_spec_embs = encode_specter(ho_specter, tokenizer,
                                   specter_model, batch_size=32, device=device)
    spec_ho = dense_retrieve_scored(ho_spec_embs, ho_ids,
                                     specter_corpus_embs, corpus_ids, top_k=RETRIEVAL_TOP_K)

    print("BM25 (held-out)...")
    bm25_ho = bm25_retrieve_scored(ho_enriched, ho_ids, corpus_tokenized,
                                    corpus_ids, bm25, top_k=RETRIEVAL_TOP_K)

    if use_lgbm:
        print("\nApplying LightGBM to held-out...")
        ho_ranking = {}
        for qid in ho_ids:
            X_q, _, dids = build_query_features(qid, spec_ho, bge_ho, bm25_ho)
            preds = model_full.predict(X_q)
            order = np.argsort(-preds)
            ho_ranking[qid] = [dids[i] for i in order[:FINAL_TOP_K]]
    else:
        ho_ranking = weighted_rrf_fuse(
            [(scored_to_ranking(spec_ho), 1.0),
             (scored_to_ranking(bge_ho),  1.0),
             (scored_to_ranking(bm25_ho), BM25_WEIGHT)],
            k=RRF_K, top_k=FINAL_TOP_K,
        )

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    zip_path = SUBMISSIONS_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission_data.json", json.dumps(ho_ranking))
    method = "LightGBM + doc2query" if use_lgbm else "Weighted RRF + doc2query (fallback)"
    print(f"\nSaved -> {zip_path}  [{method}]")


if __name__ == "__main__":
    main()
