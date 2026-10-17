import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from memebot.config import settings

EMB_CACHE = settings.data_dir / "runs" / "emb"

# Префиксы, с которыми модели обучались. Без них качество e5/FRIDA заметно падает.
PREFIX_RULES = [
    ("multilingual-e5", ("query: ", "passage: ")),
    ("FRIDA", ("search_query: ", "search_document: ")),
]


def prefixes_for(model_name: str) -> tuple[str, str]:
    return next((p for key, p in PREFIX_RULES if key in model_name), ("", ""))


class DenseRetriever:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32, max_seq_length: int = 512):
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.q_prefix, self.d_prefix = prefixes_for(model_name)
        local = Path(model_name) / "prefixes.json"  # дообученная модель хранит префиксы базовой
        if local.exists():
            saved = json.loads(local.read_text(encoding="utf-8"))
            self.q_prefix, self.d_prefix = saved["query"], saved["doc"]

    def encode(self, texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def _cache_path(self) -> Path:
        """Кэш по документам; у локальной (дообученной) модели в имя входит время изменения весов,
        чтобы переобученная модель с тем же именем не взяла старые векторы."""
        tag = self.name
        weights = Path(self.model_name) / "model.safetensors"
        if weights.exists():
            tag += f"-{int(weights.stat().st_mtime)}"
        return EMB_CACHE / f"{tag}.npz"

    def fit(self, doc_ids: list[str], docs: list[str]) -> None:
        """Векторы документов кэшируются по хешу текста: при росте корпуса считаются только новые."""
        self.doc_ids = np.array(doc_ids)
        keys = [hashlib.sha1((self.d_prefix + d).encode()).hexdigest() for d in docs]
        path = self._cache_path()
        cache: dict[str, np.ndarray] = {}
        if path.exists():
            try:
                data = np.load(path)
                cache = dict(zip(data["keys"], data["vectors"]))
            except (OSError, ValueError, KeyError):  # недописанный файл — пересчитаем
                cache = {}
        text_by_key = dict(zip(keys, docs))
        todo = [k for k in text_by_key if k not in cache]
        if todo:
            vectors = self.encode([text_by_key[k] for k in todo], self.d_prefix, show_progress=True)
            cache.update(zip(todo, vectors))
            path.parent.mkdir(parents=True, exist_ok=True)
            # Кэш общий для бота и оценки: пишем во временный файл и атомарно подменяем.
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
            os.replace(tmp, path)
        self.doc_emb = np.stack([cache[k] for k in keys])

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        q = self.encode(queries, self.q_prefix)
        scores = q @ self.doc_emb.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[(self.doc_ids[j], float(scores[i, j])) for j in row] for i, row in enumerate(top)]
