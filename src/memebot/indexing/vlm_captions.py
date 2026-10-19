"""День 2: описания мемов от локальной VLM через Ollama.

Ручные описания из @text_meme уже есть, поэтому VLM здесь нужна для двух вещей:
  1. абляция «ручное описание vs описание VLM» — насколько поиск зависит от качества текста;
  2. индексация новых мемов, у которых ручного описания нет.

Результат дописывается в data/processed/vlm_captions.jsonl, поэтому прогон можно
прервать и продолжить с того же места.

Запуск:  python -m memebot.indexing.vlm_captions [--limit 50] [--model ...]
"""

import argparse
import io
import json
import time

import httpx
import ollama
import pandas as pd
from PIL import Image
from tqdm import tqdm

from memebot.config import settings
from memebot.search.base import EVAL_SOURCES
from memebot.utils import keep_awake, local_ollama_client, retry_ollama, too_many_errors

DEFAULT_MODEL = "hf.co/Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M"

PROMPT = """Ты описываешь мем для поисковой системы. Ответь строго JSON без пояснений:
{
  "ocr": "весь текст на картинке дословно, на языке оригинала",
  "description": "2-4 предложения на русском: что изображено и в чём шутка",
  "situations": ["3-5 жизненных ситуаций, в которых этот мем уместно отправить, коротко"],
  "emotions": ["1-3 эмоции: например, усталость, сарказм, радость"],
  "is_meme": true или false — false, если это реклама, анонс канала, новость, розыгрыш или картинка без шутки
}"""


def load_image(path: str, max_side: int) -> bytes:
    """Уменьшаем картинку: число визуальных токенов, а с ним и время, растёт с площадью."""
    img = Image.open(path)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def caption(client: ollama.Client, model: str, image: bytes) -> dict:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image]}],
        format="json",
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    return json.loads(resp["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="обработать не больше N новых мемов")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--out", default="vlm_captions.jsonl")
    parser.add_argument("--scope", default="all", choices=["all", "eval"],
                        help="eval — только зафиксированный корпус оценки (нужен для обучения раньше остального)")
    args = parser.parse_args()
    keep_awake()

    memes = pd.read_parquet(settings.processed_dir / "memes.parquet")
    if args.scope == "eval":
        memes = memes[memes["source"].isin(EVAL_SOURCES)]
    out_path = settings.processed_dir / args.out

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["meme_id"])
                except json.JSONDecodeError:  # строка, оборванная выключением машины
                    continue
    todo = memes[~memes["meme_id"].isin(done)]
    # Сначала мемы без ручного описания (без VLM их не найти), среди них — самые свежие.
    no_desc = todo["description"].fillna("").str.strip().eq("")
    date = todo["date"] if "date" in todo else pd.Series("", index=todo.index)
    todo = todo.assign(_no_desc=no_desc, _date=date.fillna("")).sort_values(
        ["_no_desc", "_date"], ascending=[False, False], kind="stable")
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Готово ранее: {len(done)}, в очереди: {len(todo)}")

    client = local_ollama_client(settings.ollama_host)
    errors = 0
    with out_path.open("a", encoding="utf-8") as f:
        for row in tqdm(todo.itertuples(), total=len(todo)):
            t0 = time.perf_counter()
            try:
                image = load_image(settings.data_dir / row.image_path, args.max_side)
                result = retry_ollama(caption, client, args.model, image)
            except (json.JSONDecodeError, ollama.ResponseError, httpx.TransportError) as e:
                errors += 1
                tqdm.write(f"{row.meme_id}: {type(e).__name__}: {e}")
                continue
            record = {"meme_id": row.meme_id, "model": args.model, "seconds": round(time.perf_counter() - t0, 2), **result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Ошибок: {errors}")
    if too_many_errors(errors, len(todo)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
