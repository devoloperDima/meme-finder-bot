"""Сводка результатов: таблицы по всем наборам оценки → results/RESULTS.md.

Для каждой пары (система, текст мема) берётся последний прогон. Сравнивать системы
имеет смысл только внутри одного набора: silver — known-item по синтетическим запросам,
gold-* — запросы-ситуации с оценками LLM-судей (gold-consensus — оба судьи согласны).

  python -m memebot.evaluation.report            # все наборы → results/RESULTS.md
  python -m memebot.evaluation.report silver     # одна таблица в консоль
"""

import json
import sys
from pathlib import Path

from memebot.evaluation.metrics import METRICS

RESULTS = Path("results")
SETS = {
    "silver": "Silver: синтетические запросы GPT-4.1 к тестовым мемам (один правильный мем на запрос)",
    "gold-consensus": "Gold: 150 запросов-ситуаций, мем подходит, если так решили оба LLM-судьи",
    "gold-judge": "Gold, судья Qwen3-VL-8B",
    "gold-gemma": "Gold, судья Gemma 3 4B",
}


def load(eval_set: str) -> list[dict]:
    path = RESULTS / f"{eval_set}.jsonl"
    if not path.exists():
        return []
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["system"], r["field"])] = r
    return sorted(rows.values(), key=lambda r: r["ndcg@10"])


def table(eval_set: str) -> str:
    rows = load(eval_set)
    if not rows:
        return "_нет прогонов_"
    best = max(r["ndcg@10"] for r in rows)
    head = "| Система | Текст мема | " + " | ".join(METRICS) + " |"
    sep = "|---|---|" + "---|" * len(METRICS)
    body = []
    for r in rows:
        cells = [f"{r[m]:.3f}" for m in METRICS]
        if r["ndcg@10"] == best:
            cells = [f"**{c}**" for c in cells]
        body.append(f"| {r['system']} | {r['field']} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def train_table() -> str:
    path = RESULTS / "train_runs.jsonl"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return "_нет прогонов_"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    head = "| Модель | Запросы | Hard negatives | Пар | Минут | VRAM, ГБ | val nDCG@10 до → после |"
    sep = "|---|---|---|---|---|---|---|"
    body = [
        f"| {r['name']} | {' + '.join(r['queries'])} | {r['hard_negatives']} | {r['pairs']} | {r['train_minutes']} "
        f"| {r['peak_vram_gb']} | {r['val_ndcg@10_before']:.3f} → {r['val_ndcg@10_after']:.3f} |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def main() -> None:
    if len(sys.argv) > 1:
        print(table(sys.argv[1]))
        return
    parts = ["# Результаты\n", "_Файл собирается командой `python -m memebot.evaluation.report`._\n"]
    for eval_set, title in SETS.items():
        parts += [f"## {title}\n", table(eval_set), ""]
    parts += ["## Дообучение эмбеддера\n", train_table(), ""]
    agreement = RESULTS / "judge_agreement.txt"
    if agreement.exists():
        parts += ["## Согласие LLM-судей\n", "```", agreement.read_text(encoding="utf-8").strip(), "```", ""]
    out = RESULTS / "RESULTS.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
