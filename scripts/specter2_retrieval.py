"""
SPECTER2 dense retrieval for scientific paper citation recommendation.

SPECTER2 (allenai/specter2) is trained specifically for scientific paper
similarity, unlike general-purpose sentence transformers.

Input format: "title [SEP] abstract" as recommended by the model authors.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_queries, load_corpus, load_qrels, evaluate

TOP_K = 100
SPECTER_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2_proximity"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
HELD_OUT_PATH = ROOT / "held_out_queries.parquet"


def format_specter(row):
    """SPECTER2 expects 'title [SEP] abstract'."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " [SEP] " + abstract
    return title or abstract


def encode_specter(texts, tokenizer, model, batch_size=32, device="cpu"):
    """Encode texts with SPECTER2 using CLS pooling."""
    import torch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        # SPECTER2 uses CLS token embedding
        emb = output.last_hidden_state[:, 0, :]
        # L2 normalize for cosine similarity via dot product
        emb = emb / emb.norm(dim=1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def dense_retrieve(query_embs, q_ids, corpus_embs, c_ids, top_k):
    sim_matrix = query_embs @ corpus_embs.T
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
    return {qid: [c_ids[j] for j in top_indices[i]] for i, qid in enumerate(q_ids)}


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from adapters import init as adapters_init

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    queries = load_queries(DATA_DIR / "queries.parquet")
    held_out = load_queries(HELD_OUT_PATH)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = [format_specter(row) for _, row in corpus.iterrows()]

    # Load SPECTER2 with proximity adapter
    print(f"Loading {SPECTER_MODEL} + proximity adapter...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    model = AutoModel.from_pretrained(SPECTER_MODEL)
    adapters_init(model)
    model.load_adapter(PROXIMITY_ADAPTER, source="hf",
                       set_active=True)
    model.to(device)
    model.eval()

    # Encode corpus
    print("Encoding corpus...")
    corpus_embs = encode_specter(corpus_texts, tokenizer, model,
                                  batch_size=32, device=device)

    # ── Evaluate on public queries ──
    pub_ids = queries["doc_id"].tolist()
    pub_texts = [format_specter(row) for _, row in queries.iterrows()]
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print("Encoding public queries...")
    pub_embs = encode_specter(pub_texts, tokenizer, model,
                               batch_size=32, device=device)

    submission = dense_retrieve(pub_embs, pub_ids, corpus_embs, corpus_ids, top_k=TOP_K)
    print("\n--- SPECTER2 ---")
    evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains, verbose=True)

    # ── Predict on held-out queries ──
    ho_ids = held_out["doc_id"].tolist()
    ho_texts = [format_specter(row) for _, row in held_out.iterrows()]

    print("\nEncoding held-out queries...")
    ho_embs = encode_specter(ho_texts, tokenizer, model,
                              batch_size=32, device=device)

    ho_submission = dense_retrieve(ho_embs, ho_ids, corpus_embs, corpus_ids, top_k=TOP_K)

    # Save
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "specter2_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(ho_submission, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
