"""Поиск по самой картинке: SigLIP2 кладёт текст запроса и изображение в одно пространство.

Не нужен ни ручной текст, ни VLM — поэтому это и абляция («сколько даёт текст мема»),
и запасной канал для мемов, у которых описания ещё нет.
"""

import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from memebot.config import settings
from memebot.search.dense import EMB_CACHE

DEFAULT_IMAGE_MODEL = "google/siglip2-base-patch16-256"


def _features(output) -> torch.Tensor:
    # transformers 5 возвращает объект с pooler_output, более старые версии — тензор
    return output if torch.is_tensor(output) else output.pooler_output


class ImageRetriever:
    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL, device: str | None = None, batch_size: int = 32):
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.image_path: dict[str, str] = {}

    def set_images(self, image_path: dict[str, str]) -> None:
        """Ретриверу нужны пути к картинкам, а не тексты документов."""
        self.image_path = image_path

    @torch.no_grad()
    def _encode_images(self, paths: list[str]) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc=f"{self.name}: картинки"):
            imgs = [Image.open(settings.data_dir / p).convert("RGB") for p in paths[i : i + self.batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats = _features(self.model.get_image_features(**inputs))
            out.append(torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, padding="max_length", max_length=64, truncation=True,
                                return_tensors="pt").to(self.device)
        feats = _features(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy()

    def fit(self, doc_ids: list[str], docs: list[str] | None = None) -> None:
        """Кэш по meme_id: картинка мема не меняется, в отличие от текстов."""
        self.doc_ids = np.array(doc_ids)
        path = EMB_CACHE / f"img-{self.name}.npz"
        cache: dict[str, np.ndarray] = {}
        if path.exists():
            try:
                data = np.load(path)
                cache = dict(zip(data["keys"], data["vectors"]))
            except (OSError, ValueError, KeyError):
                cache = {}
        todo = [m for m in doc_ids if m not in cache]
        if todo:
            cache.update(zip(todo, self._encode_images([self.image_path[m] for m in todo])))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
            os.replace(tmp, path)
        self.doc_emb = np.stack([cache[m] for m in doc_ids])

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        q = np.concatenate([self._encode_texts(queries[i : i + 256]) for i in range(0, len(queries), 256)])
        scores = q @ self.doc_emb.T
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[(self.doc_ids[j], float(scores[i, j])) for j in row] for i, row in enumerate(top)]
