"""Dense retrieval baseline using pre-computed all-MiniLM-L6-v2 embeddings."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_queries, load_qrels, load_embeddings, format_text, evaluate

TOP_K = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings" / "sentence-transformers_all-MiniLM-L6-v2"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"

# Load data
queries = load_queries(DATA_DIR / "queries.parquet")
held_out = load_queries(HELD_OUT_PATH)
qrels = load_qrels(DATA_DIR / "qrels.json")

# Load pre-computed embeddings
query_embs, q_ids = load_embeddings(EMB_DIR / "query_embeddings.npy",
                                     EMB_DIR / "query_ids.json")
corpus_embs, c_ids = load_embeddings(EMB_DIR / "corpus_embeddings.npy",
                                      EMB_DIR / "corpus_ids.json")

print(f"Query embeddings:  {query_embs.shape}")
print(f"Corpus embeddings: {corpus_embs.shape}")

# --- Evaluate on public queries ---
sim_matrix = query_embs @ corpus_embs.T
top_indices = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]

eval_submission = {qid: [c_ids[j] for j in top_indices[i]]
                   for i, qid in enumerate(q_ids)}

query_domains = dict(zip(queries["doc_id"], queries["domain"]))
evaluate(eval_submission, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

# --- Predict on held-out queries ---
print(f"\nEncoding {len(held_out)} held-out queries with {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
ho_texts = [format_text(row) for _, row in held_out.iterrows()]
ho_embs = model.encode(ho_texts, normalize_embeddings=True, show_progress_bar=True)
ho_embs = ho_embs.astype(np.float32)

ho_sim = ho_embs @ corpus_embs.T
ho_top = np.argsort(-ho_sim, axis=1)[:, :TOP_K]
ho_ids = held_out["doc_id"].tolist()

submission = {qid: [c_ids[j] for j in ho_top[i]]
              for i, qid in enumerate(ho_ids)}

print(f"Held-out submission: {len(submission)} queries x {TOP_K} results")

# Save
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
out_path = SUBMISSIONS_DIR / "submission_data.json"
with open(out_path, "w") as f:
    json.dump(submission, f)
print(f"Saved -> {out_path}")
