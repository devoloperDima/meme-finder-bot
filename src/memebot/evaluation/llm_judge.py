"""LLM-судья для gold-набора: оценивает пары (запрос-ситуация, мем) по той же шкале, что и человек.

Судья видит картинку и её описание. Оценки пишутся в data/eval/gold_judge.jsonl;
прогон можно прерывать. Когда есть ручная разметка, `agreement` считает согласие
судьи с человеком (точность и κ Коэна) на общих парах — чтобы знать, насколько ему верить.

  python -m memebot.evaluation.llm_judge run
  python -m memebot.evaluation.llm_judge agreement
"""

import argparse
import json
import time

import ollama
from tqdm import tqdm

from memebot.config import settings
from memebot.evaluation.gold import JUDGE, LABELS, POOL, load_labels, load_queries, read_jsonl
from memebot.indexing.vlm_captions import DEFAULT_MODEL, load_image
from memebot.search.base import load_memes
from memebot.utils import keep_awake

PROMPT = """Пользователь Telegram-бота описал ситуацию и хочет отправить в ответ мем.
Ситуация: «{query}»

Ниже картинка мема и её описание:
{description}

Оцени, насколько этот мем подходит как ответ на ситуацию:
0 — не подходит: про другое, или связь только по отдельному слову;
1 — подходит: по смыслу уместен, его можно отправить;
2 — идеально: мем ровно про эту ситуацию.
Ответь строго JSON: {{"grade": 0|1|2, "reason": "до 12 слов"}}"""


def judge(client: ollama.Client, model: str, query: str, description: str, image: bytes) -> dict:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(query=query, description=description[:1500]),
                   "images": [image]}],
        format="json",
        options={"temperature": 0, "num_ctx": 4096},
    )
    out = json.loads(resp["message"]["content"])
    grade = int(out["grade"])
    if grade not in (0, 1, 2):
        raise ValueError(f"оценка вне шкалы: {grade}")
    return {"grade": grade, "reason": str(out.get("reason", ""))[:200]}


def run(model: str, limit: int | None) -> None:
    keep_awake()
    memes = load_memes("eval").set_index("meme_id")
    queries = {q["qid"]: q["text"] for q in load_queries()}
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    done = {(r["qid"], r["meme_id"]) for r in read_jsonl(JUDGE)}
    todo = [(q, m) for q, ms in pool.items() for m in ms if (q, m) not in done][:limit]
    print(f"Оценено ранее: {len(done)}, в очереди: {len(todo)}")

    client = ollama.Client(host=settings.ollama_host)
    errors = 0
    with JUDGE.open("a", encoding="utf-8") as f:
        for qid, meme_id in tqdm(todo):
            t0 = time.perf_counter()
            try:
                image = load_image(settings.data_dir / memes.loc[meme_id, "image_path"], 768)
                out = judge(client, model, queries[qid], memes.loc[meme_id, "description"], image)
            except (json.JSONDecodeError, KeyError, ValueError, ollama.ResponseError) as e:
                errors += 1
                tqdm.write(f"{qid}/{meme_id}: {type(e).__name__}: {e}")
                continue
            record = {"qid": qid, "meme_id": meme_id, **out, "model": model, "seconds": round(time.perf_counter() - t0, 2)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Ошибок: {errors}")


def agreement() -> None:
    human = load_labels(LABELS)
    judged = load_labels(JUDGE)
    pairs = [(g, judged[q][m]) for q, ms in human.items() for m, g in ms.items() if m in judged.get(q, {})]
    if not pairs:
        print("Нет пар, размеченных и человеком, и судьёй")
        return
    n = len(pairs)
    for name, f in [("0/1/2", lambda g: g), ("релевантен да/нет", lambda g: int(g > 0))]:
        h = [f(a) for a, _ in pairs]
        j = [f(b) for _, b in pairs]
        p_o = sum(a == b for a, b in zip(h, j)) / n
        cats = set(h) | set(j)
        p_e = sum((h.count(c) / n) * (j.count(c) / n) for c in cats)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
        print(f"{name:18} пар {n}, совпадение {p_o:.3f}, κ Коэна {kappa:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("agreement")
    args = p.parse_args()
    if args.cmd == "run":
        run(args.model, args.limit)
    else:
        agreement()


if __name__ == "__main__":
    main()
