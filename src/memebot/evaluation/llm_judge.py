"""LLM-судья для gold-набора: оценивает пары (запрос-ситуация, мем) по той же шкале, что и человек.

Судья видит картинку и её описание. Оценки пишутся в data/eval/gold_judge.jsonl;
прогон можно прерывать. Когда есть ручная разметка, `agreement` считает согласие
судьи с человеком (точность и κ Коэна) на общих парах — чтобы знать, насколько ему верить.

  python -m memebot.evaluation.llm_judge run --judge qwen     # или gemma — второй, независимый судья
  python -m memebot.evaluation.llm_judge agreement           # согласие судей между собой (и с человеком)
  python -m memebot.evaluation.llm_judge consensus           # пара релевантна, только если согласны оба
"""

import argparse
import json
import time

import ollama
from tqdm import tqdm

from memebot.config import settings
from memebot.evaluation.gold import CONSENSUS, JUDGE, JUDGE_GEMMA, LABELS, POOL, load_labels, load_queries, read_jsonl
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


JUDGES = {
    "qwen": (DEFAULT_MODEL, JUDGE),
    "gemma": ("hf.co/ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", JUDGE_GEMMA),
}


def run(judge_name: str, limit: int | None) -> None:
    keep_awake()
    model, out_path = JUDGES[judge_name]
    memes = load_memes("eval").set_index("meme_id")
    queries = {q["qid"]: q["text"] for q in load_queries()}
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    done = {(r["qid"], r["meme_id"]) for r in read_jsonl(out_path)}
    todo = [(q, m) for q, ms in pool.items() for m in ms if (q, m) not in done][:limit]
    print(f"{judge_name}: оценено ранее {len(done)}, в очереди {len(todo)}")

    client = ollama.Client(host=settings.ollama_host)
    errors = 0
    with out_path.open("a", encoding="utf-8") as f:
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


def kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """(доля совпадений, κ Коэна)."""
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return p_o, (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def agreement() -> None:
    sources = {"человек": load_labels(LABELS), **{name: load_labels(path) for name, (_, path) in JUDGES.items()}}
    names = [n for n, labels in sources.items() if labels]
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            pairs = [(g, sources[y][q][m]) for q, ms in sources[x].items() for m, g in ms.items() if m in sources[y].get(q, {})]
            if not pairs:
                continue
            for scale, f in [("0/1/2", lambda g: g), ("подходит да/нет", lambda g: int(g > 0))]:
                p_o, k = kappa([f(a) for a, _ in pairs], [f(b) for _, b in pairs])
                print(f"{x} vs {y:8} {scale:16} пар {len(pairs):5}  совпадение {p_o:.3f}  κ {k:.3f}")


def consensus() -> None:
    """Консервативный консенсус судей: оценка пары = минимум по судьям, только для пар, оценённых всеми."""
    labels = [load_labels(path) for _, path in JUDGES.values()]
    rows = []
    for qid, memes in labels[0].items():
        for meme_id, grade in memes.items():
            grades = [grade] + [lab.get(qid, {}).get(meme_id) for lab in labels[1:]]
            if None not in grades:
                rows.append({"qid": qid, "meme_id": meme_id, "grade": min(grades)})
    with CONSENSUS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rel = sum(r["grade"] > 0 for r in rows)
    print(f"Консенсус: {len(rows)} пар, подходящих {rel} ({rel / max(len(rows), 1):.1%}) → {CONSENSUS}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--judge", default="qwen", choices=list(JUDGES))
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("agreement")
    sub.add_parser("consensus")
    args = p.parse_args()
    if args.cmd == "run":
        run(args.judge, args.limit)
    elif args.cmd == "agreement":
        agreement()
    else:
        consensus()


if __name__ == "__main__":
    main()
