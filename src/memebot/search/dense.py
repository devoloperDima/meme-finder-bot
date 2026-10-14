import hashlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from memebot.config import settings

EMB_CACHE = settings.data_dir / "runs" / "emb"

# Префиксы, с которыми модели обучались. Без них качество e5/FRIDA заметно падает.
PREFIXES = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "ai-forever/FRIDA": ("search_query: ", "search_document: "),
}


class DenseRetriever:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32, max_seq_length: int = 512):
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.q_prefix, self.d_prefix = PREFIXES.get(model_name, ("", ""))

    def encode(self, texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def fit(self, doc_ids: list[str], docs: list[str]) -> None:
        self.doc_ids = np.array(doc_ids)
        key = hashlib.sha1("\x00".join([self.model_name, *docs]).encode()).hexdigest()[:12]
        cache = EMB_CACHE / f"{self.name}-{key}.npy"
        if cache.exists():
            self.doc_emb = np.load(cache)
            return
        self.doc_emb = self.encode(docs, self.d_prefix, show_progress=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, self.doc_emb)

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        q = self.encode(queries, self.q_prefix)
        scores = q @ self.doc_emb.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[(self.doc_ids[j], float(scores[i, j])) for j in row] for i, row in enumerate(top)]
