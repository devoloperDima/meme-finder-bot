"""Разбиение мемов и наборы запросов для оценки.

Разбиение по мемам, а не по запросам: запросы к тестовым мемам никогда не попадают
в обучение, иначе дообученная модель просто запомнит пары.

Silver-набор: синтетические запросы GPT-4.1 к тестовым мемам (known-item search,
у каждого запроса ровно один релевантный мем). Они сгенерированы по ручному описанию,
поэтому лексически к нему близки и переоценивают BM25 — для итоговых цифр нужен
ручной gold-набор с запросами-ситуациями.
"""

import hashlib

import pandas as pd


def split_of(meme_id: str) -> str:
    bucket = int(hashlib.md5(meme_id.encode()).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def add_split(memes: pd.DataFrame) -> pd.DataFrame:
    return memes.assign(split=memes["meme_id"].map(split_of))


def silver_queries(memes: pd.DataFrame, split: str = "test") -> tuple[list[str], dict[str, dict[str, int]]]:
    """Возвращает (тексты запросов, qrels {query_id: {meme_id: 1}}); query_id = индекс в списке."""
    memes = add_split(memes)
    texts, qrels = [], {}
    for row in memes[memes["split"] == split].itertuples():
        for q in row.synthetic_queries:
            qrels[str(len(texts))] = {row.meme_id: 1}
            texts.append(q)
    return texts, qrels
