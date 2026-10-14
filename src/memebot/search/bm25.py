import re

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")
_stemmer = snowballstemmer.stemmer("russian")


def tokenize(text: str) -> list[str]:
    return _stemmer.stemWords(TOKEN_RE.findall(text.lower().replace("ё", "е")))


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, doc_ids: list[str], docs: list[str]) -> None:
        self.doc_ids = np.array(doc_ids)
        self.bm25 = BM25Okapi([tokenize(d) for d in docs], k1=self.k1, b=self.b)

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        results = []
        for q in queries:
            scores = self.bm25.get_scores(tokenize(q))
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            top = top[np.argsort(-scores[top])]
            results.append([(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0])
        return results
