from collections import defaultdict


def rrf(runs: list[list[list[tuple[str, float]]]], k: int = 60, top: int = 100) -> list[list[tuple[str, float]]]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank). Не требует калибровки скоров между ретриверами."""
    fused = []
    for per_query in zip(*runs):
        scores: dict[str, float] = defaultdict(float)
        for ranking in per_query:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        fused.append(sorted(scores.items(), key=lambda x: -x[1])[:top])
    return fused
