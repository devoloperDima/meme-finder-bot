"""День 1: собрать единый датасет мемов из скачанных HF-датасетов.

Источники (все из Telegram-канала @text_meme, где мемы вручную описывают для незрячих):
  - justatom/meme-russian-ir  — описания + 3 синтетических запроса от GPT-4.1 на мем
  - yaneivan/memes_caption    — описания
  - foldl/rumeme-desc         — описания

Шаги: декодируем картинки → отбрасываем битые/мелкие/многокартиночные →
дедуплицируем по перцептивному хешу → сохраняем JPEG и memes.parquet.

Запуск:  python -m memebot.data.build_dataset
"""

import difflib
import io
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from memebot.config import settings

MIN_SIDE = 128
MAX_SIDE = 1280
PHASH_MAX_DIST = 6  # 64-битный pHash: ≤6 отличающихся бит — похожие картинки
DESC_MIN_SIM = 0.8  # ...и описания должны совпадать, иначе это тот же шаблон с другим текстом


@dataclass
class Raw:
    source: str
    source_id: int
    image: bytes
    description: str
    queries: list[str] = field(default_factory=list)


# Порядок = приоритет при дедупликации: первый источник побеждает (у justatom есть запросы).
SOURCES = {
    "justatom": "justatom__meme-russian-ir",
    "yaneivan": "yaneivan__memes_caption",
    "foldl": "foldl__rumeme-desc",
}


def load_raw(raw_dir: Path) -> tuple[list[Raw], int]:
    items: list[Raw] = []
    skipped_multi = 0

    for row in pq.read_table(next((raw_dir / SOURCES["justatom"]).rglob("*.parquet"))).to_pylist():
        if len(row["photos"]) != 1:
            skipped_multi += 1
            continue
        items.append(Raw("justatom", row["id"], row["photos"][0]["bytes"], row["description"], row["generated"]))

    for row in pq.read_table(next((raw_dir / SOURCES["yaneivan"]).rglob("*.parquet"))).to_pylist():
        items.append(Raw("yaneivan", row["id"], row["image"]["bytes"], row["label"]))

    for i, row in enumerate(pq.read_table(next((raw_dir / SOURCES["foldl"]).rglob("*.parquet"))).to_pylist()):
        items.append(Raw("foldl", i, row["image"]["bytes"], row["text"]))

    return items, skipped_multi


def decode(raw: Raw) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw.image))
        img.load()
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    return img.convert("RGB")


def same_text(a: str, b: str) -> bool:
    a, b = a.lower()[:300], b.lower()[:300]
    return difflib.SequenceMatcher(None, a, b).ratio() >= DESC_MIN_SIM


def find_duplicates(hashes: np.ndarray, descriptions: list[str]) -> np.ndarray:
    """Для каждого элемента — индекс первого (по порядку) дубликата, иначе он сам.

    Похожего pHash мало: один шаблон с разными подписями даёт близкие хеши,
    но это разные мемы. Поэтому дубликат = близкий хеш И совпадающее описание.
    """
    canonical = np.arange(len(hashes))
    for start in range(0, len(hashes), 1024):
        block = hashes[start : start + 1024]
        dist = np.bitwise_count(block[:, None] ^ hashes[None, :])  # расстояние Хэмминга, (block, n)
        for i, row in enumerate(dist):
            j = start + i
            for k in np.flatnonzero(row[:j] <= PHASH_MAX_DIST):
                if same_text(descriptions[k], descriptions[j]):
                    canonical[j] = canonical[k]
                    break
    return canonical


def main() -> None:
    out_dir = settings.processed_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    raws, skipped_multi = load_raw(settings.raw_dir)
    print(f"Загружено записей: {len(raws)} (пропущено постов с несколькими картинками: {skipped_multi})")

    kept: list[tuple[Raw, Image.Image]] = []
    bad = 0
    for raw in tqdm(raws, desc="decode"):
        if not raw.description or not raw.description.strip():
            bad += 1
            continue
        img = decode(raw)
        if img is None:
            bad += 1
            continue
        kept.append((raw, img))
    print(f"Битых/мелких/пустых: {bad}")

    hashes = np.array([int(str(imagehash.phash(img)), 16) for _, img in tqdm(kept, desc="phash")], dtype=np.uint64)
    canonical = find_duplicates(hashes, [raw.description for raw, _ in kept])

    groups: dict[int, list[int]] = {}
    for i, c in enumerate(canonical):
        groups.setdefault(int(c), []).append(i)

    records = []
    for n, (head, members) in enumerate(tqdm(sorted(groups.items()), desc="save")):
        raw, img = kept[head]
        meme_id = f"m{n:05d}"
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        path = img_dir / f"{meme_id}.jpg"
        img.save(path, "JPEG", quality=90)

        alt = [kept[m][0].description for m in members[1:] if kept[m][0].description != raw.description]
        queries = [q for m in members for q in kept[m][0].queries]
        records.append(
            {
                "meme_id": meme_id,
                "image_path": path.relative_to(settings.data_dir).as_posix(),
                "width": img.width,
                "height": img.height,
                "phash": f"{hashes[head]:016x}",
                "description": raw.description.strip(),
                "alt_descriptions": alt,
                "synthetic_queries": list(dict.fromkeys(queries)),
                "source": raw.source,
                "source_id": raw.source_id,
                "duplicates": [f"{kept[m][0].source}:{kept[m][0].source_id}" for m in members[1:]],
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(out_dir / "memes.parquet", index=False)

    print(f"\nУникальных мемов: {len(df)} (из {len(kept)}, дубликатов {len(kept) - len(df)})")
    print("По источнику-победителю:", df["source"].value_counts().to_dict())
    print(f"С синтетическими запросами: {(df['synthetic_queries'].str.len() > 0).sum()}")
    print(f"Медиана длины описания: {int(df['description'].str.len().median())} симв.")
    print(f"Сохранено: {out_dir / 'memes.parquet'}, картинки в {img_dir}")


if __name__ == "__main__":
    main()
