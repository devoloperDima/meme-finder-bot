"""Поисковый сервис для бота и API: гибрид BM25 + плотная модель через RRF и бонус за свежесть."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from memebot.search.base import build_docs, load_memes
from memebot.search.bm25 import BM25Retriever
from memebot.search.dense import DenseRetriever
from memebot.search.hybrid import rrf

FRESHNESS_HALF_LIFE_DAYS = 365


@dataclass
class Hit:
    meme_id: str
    image_path: str
    score: float


def freshness(memes: pd.DataFrame) -> dict[str, float]:
    """1.0 — мем вышел сегодня, 0.5 — год назад; мемы без даты считаем старыми (0)."""
    if "date" not in memes:
        return {}
    dates = pd.to_datetime(memes["date"], utc=True, errors="coerce", format="ISO8601")
    age_days = (pd.Timestamp.now(tz="UTC") - dates).dt.total_seconds() / 86400
    fresh = np.power(0.5, age_days.clip(lower=0) / FRESHNESS_HALF_LIFE_DAYS).fillna(0.0)
    return dict(zip(memes["meme_id"], fresh))


class MemeSearcher:
    def __init__(self, dense_model: str = "deepvk/USER-bge-m3", field: str = "human",
                 device: str | None = None, recency_weight: float = 0.15):
        memes = load_memes()
        if "vlm_is_meme" in memes:  # VLM пометил рекламу / анонсы / новости без шутки — в выдачу не берём
            memes = memes[memes["vlm_is_meme"].map(lambda v: str(v).strip().lower() != "false")]
        self.image_path = dict(zip(memes["meme_id"], memes["image_path"]))
        self.freshness = freshness(memes)
        self.recency_weight = recency_weight
        doc_ids, docs = build_docs(memes, field)
        self.bm25 = BM25Retriever()
        self.bm25.fit(doc_ids, docs)
        self.dense = DenseRetriever(dense_model, device=device)
        self.dense.fit(doc_ids, docs)  # эмбеддинги документов берутся из кэша, если уже считались

    def rerank_fresh(self, fused: list[tuple[str, float]], weight: float) -> list[tuple[str, float]]:
        """Скор RRF нормируем на лучший результат и добавляем weight × свежесть."""
        if not fused or weight == 0:
            return fused
        top = fused[0][1]
        scored = [(m, s / top + weight * self.freshness.get(m, 0.0)) for m, s in fused]
        return sorted(scored, key=lambda x: -x[1])

    def search(self, query: str, k: int = 10, recency_weight: float | None = None) -> tuple[list[Hit], dict[str, float]]:
        """Возвращает выдачу и задержку по этапам в мс."""
        weight = self.recency_weight if recency_weight is None else recency_weight
        t0 = time.perf_counter()
        bm25 = self.bm25.search([query], k=100)
        t1 = time.perf_counter()
        dense = self.dense.search([query], k=100)
        t2 = time.perf_counter()
        fused = self.rerank_fresh(rrf([bm25, dense], top=50)[0], weight)[:k]
        t3 = time.perf_counter()
        timings = {"bm25_ms": (t1 - t0) * 1000, "dense_ms": (t2 - t1) * 1000, "fuse_ms": (t3 - t2) * 1000}
        return [Hit(str(m), self.image_path[m], float(s)) for m, s in fused], timings
