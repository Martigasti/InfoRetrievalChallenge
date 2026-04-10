"""TF-IDF baseline: sparse retrieval over title + abstract."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_queries, load_corpus, load_qrels, format_text, evaluate

TOP_K = 100
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"

# Load data
queries = load_queries(DATA_DIR / "queries.parquet")
held_out = load_queries(HELD_OUT_PATH)
corpus = load_corpus(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")

corpus_ids = corpus["doc_id"].tolist()
corpus_texts = [format_text(row) for _, row in corpus.iterrows()]

# Build TF-IDF index on corpus
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 1),
    stop_words="english",
)
corpus_matrix = vectorizer.fit_transform(corpus_texts)
print(f"Corpus matrix: {corpus_matrix.shape}  vocab={corpus_matrix.shape[1]:,}")

# --- Evaluate on public queries ---
query_ids = queries["doc_id"].tolist()
query_texts = [format_text(row) for _, row in queries.iterrows()]
query_matrix = vectorizer.transform(query_texts)
sim_matrix = cosine_similarity(query_matrix, corpus_matrix)

eval_submission = {}
for i, qid in enumerate(query_ids):
    top_indices = np.argsort(-sim_matrix[i])[:TOP_K]
    eval_submission[qid] = [corpus_ids[j] for j in top_indices]

query_domains = dict(zip(queries["doc_id"], queries["domain"]))
evaluate(eval_submission, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

# --- Predict on held-out queries ---
ho_ids = held_out["doc_id"].tolist()
ho_texts = [format_text(row) for _, row in held_out.iterrows()]
ho_matrix = vectorizer.transform(ho_texts)
ho_sim = cosine_similarity(ho_matrix, corpus_matrix)

submission = {}
for i, qid in enumerate(ho_ids):
    top_indices = np.argsort(-ho_sim[i])[:TOP_K]
    submission[qid] = [corpus_ids[j] for j in top_indices]

print(f"Held-out submission: {len(submission)} queries x {TOP_K} results")

# Save
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
out_path = SUBMISSIONS_DIR / "submission_data.json"
with open(out_path, "w") as f:
    json.dump(submission, f)
print(f"Saved -> {out_path}")
