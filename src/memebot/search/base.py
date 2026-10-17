"""Общий интерфейс ретриверов и тексты документов."""

import json
from typing import Protocol

import pandas as pd

from memebot.config import settings


class Retriever(Protocol):
    name: str

    def fit(self, doc_ids: list[str], docs: list[str]) -> None: ...

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        """Для каждого запроса — список (doc_id, score) по убыванию score."""
        ...


# Корпус для оценки зафиксирован: исходные 6 222 мема из HF-датасетов. Свежие мемы (скрейпинг,
# новые датасеты) ищет бот, но в метрики они не входят — иначе цифры разных дней несравнимы.
EVAL_SOURCES = {"justatom", "yaneivan", "foldl"}


def load_memes(scope: str = "all") -> pd.DataFrame:
    """memes.parquet + поля VLM (если прогон уже есть). scope="eval" — только зафиксированный корпус оценки."""
    memes = pd.read_parquet(settings.processed_dir / "memes.parquet")
    if scope == "eval":
        memes = memes[memes["source"].isin(EVAL_SOURCES)].reset_index(drop=True)
    path = settings.processed_dir / "vlm_captions.jsonl"
    if path.exists():
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # строка, оборванная при остановке прогона
                    continue
        if rows:
            vlm = pd.DataFrame(rows).drop_duplicates("meme_id", keep="last")
            vlm = vlm[["meme_id", "ocr", "description", "situations", "emotions"]].add_prefix("vlm_")
            memes = memes.merge(vlm, left_on="meme_id", right_on="vlm_meme_id", how="left").drop(columns="vlm_meme_id")
    return memes


def _join(*parts) -> str:
    out = []
    for p in parts:
        if isinstance(p, list) or hasattr(p, "tolist"):
            p = "; ".join(map(str, list(p)))
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
    return "\n".join(out)


# Варианты текстового представления мема — для абляции «что индексировать».
DOC_FIELDS = {
    "human": lambda r: _join(r["description"]),
    "vlm": lambda r: _join(r.get("vlm_ocr"), r.get("vlm_description"), r.get("vlm_situations"), r.get("vlm_emotions")),
    "human+vlm": lambda r: _join(r["description"], r.get("vlm_situations"), r.get("vlm_emotions")),
}


def build_docs(memes: pd.DataFrame, field: str) -> tuple[list[str], list[str]]:
    fn = DOC_FIELDS[field]
    records = memes.to_dict("records")
    return [r["meme_id"] for r in records], [fn(r) for r in records]
