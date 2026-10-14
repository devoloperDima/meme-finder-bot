"""Сводная таблица из results/<set>.jsonl (последний прогон каждой системы).

  python -m memebot.evaluation.report silver
"""

import json
import sys
from pathlib import Path

from memebot.evaluation.metrics import METRICS


def table(eval_set: str) -> str:
    rows = {}
    with Path(f"results/{eval_set}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["system"], r["field"])] = r
    head = "| Система | Текст мема | " + " | ".join(METRICS) + " |"
    sep = "|---|---|" + "---|" * len(METRICS)
    body = [
        f"| {r['system']} | {r['field']} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS) + " |"
        for r in sorted(rows.values(), key=lambda r: r["ndcg@10"])
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    print(table(sys.argv[1] if len(sys.argv) > 1 else "silver"))
