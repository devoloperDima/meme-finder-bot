"""Онлайн-метрики бота по логу в SQLite.

Определения:
  like rate      — доля показов с 👍 среди показов, получивших хоть какую-то реакцию;
  hit@1          — доля поисков, где первый же мем получил 👍;
  success rate   — доля поисков, где 👍 получил хоть один из показанных мемов;
  depth          — сколько мемов в среднем пролистали в одном поиске (по 🔄);
  latency        — p50 / p95 полного поиска (BM25 + dense + слияние), мс.

  python -m memebot.evaluation.online
"""

import json
import sqlite3
import sys

import numpy as np

from memebot.config import settings


def compute(db_path) -> dict:
    db = sqlite3.connect(db_path)
    searches = db.execute("SELECT id, user_hash, bm25_ms + dense_ms + fuse_ms FROM searches").fetchall()
    impressions = db.execute(
        "SELECT i.id, i.search_id, i.rank, f.action FROM impressions i LEFT JOIN feedback f ON f.impression_id = i.id"
    ).fetchall()
    if not searches:
        return {"searches": 0}

    by_search: dict[int, list[tuple[int, str | None]]] = {}
    for _, search_id, rank, action in impressions:
        by_search.setdefault(search_id, []).append((rank, action))
    reacted = [a for *_, a in impressions if a in ("like", "dislike")]
    latency = np.array([s[2] for s in searches if s[2] is not None])

    return {
        "searches": len(searches),
        "users": len({s[1] for s in searches}),
        "impressions": len(impressions),
        "like_rate": round(reacted.count("like") / len(reacted), 3) if reacted else None,
        "hit@1": round(np.mean([any(r == 0 and a == "like" for r, a in v) for v in by_search.values()]), 3),
        "success_rate": round(np.mean([any(a == "like" for _, a in v) for v in by_search.values()]), 3),
        "depth": round(np.mean([len(v) for v in by_search.values()]), 2),
        "latency_p50_ms": round(float(np.percentile(latency, 50)), 1) if len(latency) else None,
        "latency_p95_ms": round(float(np.percentile(latency, 95)), 1) if len(latency) else None,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else settings.bot_db_path
    print(json.dumps(compute(path), ensure_ascii=False, indent=1))
