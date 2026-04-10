"""
embed.py — Encode queries and corpus with a sentence-transformers model.

Produces:
  <output>/query_embeddings.npy    float32 array (n_queries, dim)
  <output>/corpus_embeddings.npy   float32 array (n_corpus, dim)
  <output>/query_ids.json          ordered list of query doc_ids
  <output>/corpus_ids.json         ordered list of corpus doc_ids

Usage:
  python embed.py
  python embed.py --model BAAI/bge-small-en-v1.5 --batch-size 512
  python embed.py --queries ../data/queries.parquet \
                  --corpus ../data/corpus.parquet \
                  --output ../data/embeddings/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(CHALLENGE_DIR, "data")

DEFAULT_QUERIES = os.path.join(DATA_DIR, "queries.parquet")
DEFAULT_CORPUS = os.path.join(DATA_DIR, "corpus.parquet")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "embeddings")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def format_text(row: pd.Series) -> str:
    """Concatenate title and abstract for encoding."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


def encode(model: SentenceTransformer, texts: list, batch_size: int,
           desc: str = "Encoding") -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Embed queries and corpus")
    parser.add_argument("--queries", default=DEFAULT_QUERIES,
                        help="Path to queries.parquet")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="Path to corpus.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Sentence-transformers model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for embedding files")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256)")
    args = parser.parse_args()

    # When using the default output directory, append a sanitised model name
    # so different models never overwrite each other.
    # e.g. data/embeddings/sentence-transformers_all-MiniLM-L6-v2/
    # Passing --output explicitly bypasses this and writes directly to that path.
    if args.output == DEFAULT_OUTPUT:
        model_slug = args.model.replace("/", "_").replace("\\", "_")
        args.output = os.path.join(DEFAULT_OUTPUT, model_slug)

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"Loading queries from {args.queries} …")
    queries_df = pd.read_parquet(args.queries)
    print(f"  {len(queries_df)} queries")

    print(f"Loading corpus from {args.corpus} …")
    corpus_df = pd.read_parquet(args.corpus)
    print(f"  {len(corpus_df)} documents")

    query_ids = queries_df["doc_id"].tolist()
    corpus_ids = corpus_df["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries_df.iterrows()]
    corpus_texts = [format_text(row) for _, row in corpus_df.iterrows()]

    # Load model
    print(f"\nLoading model: {args.model} …")
    model = SentenceTransformer(args.model)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Encode
    print("\nEncoding queries …")
    query_embs = encode(model, query_texts, args.batch_size, "Queries")
    print(f"  query_embeddings shape: {query_embs.shape}")

    print("\nEncoding corpus …")
    corpus_embs = encode(model, corpus_texts, args.batch_size, "Corpus")
    print(f"  corpus_embeddings shape: {corpus_embs.shape}")

    # Save
    out = args.output
    np.save(os.path.join(out, "query_embeddings.npy"), query_embs)
    np.save(os.path.join(out, "corpus_embeddings.npy"), corpus_embs)
    with open(os.path.join(out, "query_ids.json"), "w") as f:
        json.dump(query_ids, f)
    with open(os.path.join(out, "corpus_ids.json"), "w") as f:
        json.dump(corpus_ids, f)

    # Save model info
    with open(os.path.join(out, "model_info.txt"), "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"embedding_dim: {query_embs.shape[1]}\n")
        f.write(f"normalize_embeddings: True\n")
        f.write(f"text_format: title + ' ' + abstract\n")

    print(f"\nSaved to {out}/")
    print("  query_embeddings.npy")
    print("  corpus_embeddings.npy")
    print("  query_ids.json")
    print("  corpus_ids.json")
    print("  model_info.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
