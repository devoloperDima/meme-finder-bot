"""Метрики ранжирования. Свои, а не ranx: его зависимость numba блокирует Smart App Control."""

import math

import numpy as np

METRICS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def _per_query(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    relevant = {d for d, g in rel.items() if g > 0}
    out = {}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = len(relevant & set(ranked[:k])) / len(relevant)
    out["mrr@10"] = next((1 / r for r, d in enumerate(ranked[:10], 1) if d in relevant), 0.0)
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(r + 1) for r, d in enumerate(ranked[:10], 1))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def score(results: list[list[tuple[str, float]]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """results[i] — ранжирование для запроса str(i); qrels: {query_id: {doc_id: градация}}."""
    per_query = [_per_query([d for d, _ in results[int(q)]], rel) for q, rel in qrels.items()]
    return {m: round(float(np.mean([p[m] for p in per_query])), 4) for m in METRICS}
