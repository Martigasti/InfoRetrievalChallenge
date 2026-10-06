from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

W_TITLE = 0.15
BOOST_DEFAULT = 0.05
BOOST_BIOLOGY = 0.25
BIOLOGY_LABEL = "Biology"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def overlap_ratio(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / (len(query_tokens) + 1e-9)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def evaluate_ndcg10(submission: dict[str, list[str]], qrels: dict[str, list[str]]) -> float:
    return float(np.mean([ndcg_at_k(submission.get(qid, []), set(rel), 10) for qid, rel in qrels.items()]))


def build_query_map(df: pd.DataFrame) -> dict[str, dict]:
    qmap = {}
    for _, row in df.iterrows():
        qid = row["doc_id"]
        qmap[qid] = {
            "title_tokens": tokenize(str(row.get("title", "") or "")),
            "domain": row.get("domain", ""),
        }
    return qmap


def build_corpus_map(df: pd.DataFrame) -> dict[str, dict]:
    cmap = {}
    for _, row in df.iterrows():
        cid = row["doc_id"]
        cmap[cid] = {
            "title_tokens": tokenize(str(row.get("title", "") or "")),
            "domain": row.get("domain", ""),
        }
    return cmap


def rerank(
    base_submission: dict[str, list[str]],
    qmap: dict,
    cmap: dict,
    w_title: float,
    boost_default: float,
    boost_biology: float,
) -> dict[str, list[str]]:
    out = {}
    for qid, docs in base_submission.items():
        q = qmap.get(qid)
        if q is None:
            out[qid] = docs
            continue

        scored = []
        q_dom = q["domain"]
        for rank, cid in enumerate(docs):
            d = cmap.get(cid)
            if d is None:
                scored.append((1.0 / (rank + 1), cid))
                continue

            base_rank = 1.0 / (rank + 1)
            title_ov = overlap_ratio(q["title_tokens"], d["title_tokens"])

            same_domain = q_dom and (q_dom == d["domain"])
            if same_domain and q_dom == BIOLOGY_LABEL:
                domain_boost = boost_biology
            elif same_domain:
                domain_boost = boost_default
            else:
                domain_boost = 0.0

            score = base_rank + w_title * title_ov + domain_boost
            scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        out[qid] = [cid for _, cid in scored[:100]]

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queries", type=Path, default=Path("data/queries.parquet"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.parquet"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path("submissions/wider_submission/submission_data.json"),
    )
    parser.add_argument("--w-title", type=float, default=W_TITLE)
    parser.add_argument("--boost-default", type=float, default=BOOST_DEFAULT)
    parser.add_argument("--boost-biology", type=float, default=BOOST_BIOLOGY)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/wider_submission/submission_data_light_rerank.json"),
    )
    args = parser.parse_args()

    root = args.root

    queries = pd.read_parquet(root / args.queries)
    corpus = pd.read_parquet(root / args.corpus)

    with open(root / args.base_submission, encoding="utf-8") as f:
        base_submission = json.load(f)

    qmap = build_query_map(queries)
    cmap = build_corpus_map(corpus)
    reranked = rerank(
        base_submission,
        qmap,
        cmap,
        w_title=args.w_title,
        boost_default=args.boost_default,
        boost_biology=args.boost_biology,
    )

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reranked, f)
    print(f"Saved reranked output: {out_path}")

    qrels_path = root / args.qrels
    if qrels_path.exists():
        with open(qrels_path, encoding="utf-8") as f:
            qrels = json.load(f)
        if set(qrels.keys()).issubset(set(reranked.keys())):
            base_score = evaluate_ndcg10(base_submission, qrels)
            new_score = evaluate_ndcg10(reranked, qrels)
            print(f"Base NDCG@10: {base_score:.4f}")
            print(f"New  NDCG@10: {new_score:.4f}")
        else:
            print("Skipped local eval: query IDs differ from qrels.")


if __name__ == "__main__":
    main()
