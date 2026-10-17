"""День 3: бейзлайны поиска.

Индекс — все мемы. Запросы: silver — синтетические запросы к тестовым мемам,
gold — запросы-ситуации с ручной разметкой (см. memebot.evaluation.gold).
Ранжирования кэшируются в data/runs/, чтобы гибриды собирать без пересчёта.
Результаты дописываются в results/<set>.jsonl.

Примеры:
  python -m memebot.evaluation.run_baselines --retrievers bm25
  python -m memebot.evaluation.run_baselines --retrievers intfloat/multilingual-e5-base --device cpu
  python -m memebot.evaluation.run_baselines --fuse bm25 multilingual-e5-base
"""

import argparse
import json
import time
from pathlib import Path

from memebot.config import settings
from memebot.evaluation.datasets import silver_queries
from memebot.evaluation.gold import gold_qrels
from memebot.evaluation.metrics import score
from memebot.search.base import build_docs, load_memes
from memebot.search.hybrid import rrf
from memebot.utils import keep_awake

RUNS_DIR = settings.data_dir / "runs"
RESULTS_DIR = Path("results")


def make_retriever(name: str, device: str | None):
    if name == "bm25":
        from memebot.search.bm25 import BM25Retriever

        return BM25Retriever()
    from memebot.search.dense import DenseRetriever

    return DenseRetriever(name, device=device)


def report(eval_set: str, system: str, field: str, metrics: dict, extra: dict | None = None) -> None:
    row = {"system": system, "field": field, **metrics, **(extra or {})}
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / f"{eval_set}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrievers", nargs="*", default=[])
    parser.add_argument("--fuse", nargs="*", default=[], help="имена уже посчитанных ранжирований для RRF")
    parser.add_argument("--field", default="human", choices=["human", "vlm", "human+vlm"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", dest="eval_set", default="silver", choices=["silver", "gold"])
    args = parser.parse_args()
    keep_awake()

    memes = load_memes()
    if args.field != "human":
        memes = memes[memes["vlm_description"].notna()]
        print(f"Мемов с VLM-полями: {len(memes)}")
    doc_ids, docs = build_docs(memes, args.field)
    if args.eval_set == "gold":
        queries, qrels, _ = gold_qrels()
    else:
        queries, qrels = silver_queries(memes)
    print(f"Документов: {len(docs)}, запросов: {len(queries)}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.retrievers:
        retriever = make_retriever(name, args.device)
        t0 = time.perf_counter()
        retriever.fit(doc_ids, docs)
        t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        results = retriever.search(queries, k=100)
        ms_per_query = (time.perf_counter() - t0) / len(queries) * 1000
        (RUNS_DIR / f"{retriever.name}__{args.field}__{args.eval_set}.json").write_text(json.dumps(results), encoding="utf-8")
        report(args.eval_set, retriever.name, args.field, score(results, qrels),
               {"index_s": round(t_index, 1), "ms_per_query": round(ms_per_query, 2), "device": args.device or "auto"})

    if args.fuse:
        runs = []
        for name in args.fuse:
            runs.append(json.loads((RUNS_DIR / f"{name}__{args.field}__{args.eval_set}.json").read_text(encoding="utf-8")))
        report(args.eval_set, "rrf(" + " + ".join(args.fuse) + ")", args.field, score(rrf(runs), qrels))


if __name__ == "__main__":
    main()
