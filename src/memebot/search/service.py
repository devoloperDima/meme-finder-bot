"""Поисковый сервис для бота и API: гибрид BM25 + плотная модель через RRF."""

import time
from dataclasses import dataclass

from memebot.search.base import build_docs, load_memes
from memebot.search.bm25 import BM25Retriever
from memebot.search.dense import DenseRetriever
from memebot.search.hybrid import rrf


@dataclass
class Hit:
    meme_id: str
    image_path: str
    score: float


class MemeSearcher:
    def __init__(self, dense_model: str = "deepvk/USER-bge-m3", field: str = "human", device: str | None = None):
        memes = load_memes()
        self.image_path = dict(zip(memes["meme_id"], memes["image_path"]))
        doc_ids, docs = build_docs(memes, field)
        self.bm25 = BM25Retriever()
        self.bm25.fit(doc_ids, docs)
        self.dense = DenseRetriever(dense_model, device=device)
        self.dense.fit(doc_ids, docs)  # эмбеддинги документов берутся из кэша, если уже считались

    def search(self, query: str, k: int = 10) -> tuple[list[Hit], dict[str, float]]:
        """Возвращает выдачу и задержку по этапам в мс."""
        t0 = time.perf_counter()
        bm25 = self.bm25.search([query], k=100)
        t1 = time.perf_counter()
        dense = self.dense.search([query], k=100)
        t2 = time.perf_counter()
        fused = rrf([bm25, dense], top=k)[0]
        t3 = time.perf_counter()
        timings = {"bm25_ms": (t1 - t0) * 1000, "dense_ms": (t2 - t1) * 1000, "fuse_ms": (t3 - t2) * 1000}
        return [Hit(str(m), self.image_path[m], float(s)) for m, s in fused], timings
