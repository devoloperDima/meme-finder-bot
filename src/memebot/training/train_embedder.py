"""День 4: дообучение эмбеддера на парах «запрос → мем».

Данные — только мемы из train-сплита (val используется для валидации, test и gold не видны).
Запросы-якоря:
  synthetic — запросы GPT-4.1 из justatom/meme-russian-ir (про конкретный мем);
  vlm       — «ситуации, когда уместно отправить мем» от Qwen3-VL (ближе к реальным запросам).
Hard negatives майнятся базовой моделью по train-корпусу: пропускаем самые близкие
(среди мемов много подходящих к одному запросу — это не негативы) и берём из диапазона ниже.

Лосс — CachedMultipleNegativesRankingLoss: in-batch негативы с большим батчем при малой памяти.

  python -m memebot.training.train_embedder --queries synthetic vlm --hard-negatives 1
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives

from memebot.evaluation.datasets import add_split
from memebot.search.base import build_docs, load_memes
from memebot.search.dense import prefixes_for
from memebot.utils import keep_awake

MODELS_DIR = Path("models")


def as_list(value) -> list[str]:
    """Поля-списки из parquet приходят как numpy-массивы, а у мемов без VLM — как NaN."""
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return [str(v) for v in list(value)]
    return []


def make_pairs(memes, doc_text: dict[str, str], sources: list[str], q_prefix: str) -> list[dict]:
    pairs = []
    for r in memes.to_dict("records"):
        queries = []
        if "synthetic" in sources:
            queries += as_list(r["synthetic_queries"])
        if "vlm" in sources:
            queries += as_list(r.get("vlm_situations"))
        for q in dict.fromkeys(q.strip() for q in queries if q and q.strip()):
            pairs.append({"anchor": q_prefix + q, "positive": doc_text[r["meme_id"]]})
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="deepvk/USER-bge-m3")
    p.add_argument("--field", default="human")
    p.add_argument("--queries", nargs="+", default=["synthetic"], choices=["synthetic", "vlm"])
    p.add_argument("--hard-negatives", type=int, default=0, help="сколько hard negatives на пару (0 = только in-batch)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--mini-batch", type=int, default=16, help="кусок батча для CachedMNRL, влияет только на память")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="взять только N пар — для быстрой проверки")
    args = p.parse_args()
    keep_awake()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = args.name or "-".join([args.base.split("/")[-1], *args.queries, f"hn{args.hard_negatives}"])
    out_dir = MODELS_DIR / name
    q_prefix, d_prefix = prefixes_for(args.base)

    memes = add_split(load_memes("eval"))
    if "vlm" in args.queries or args.field != "human":
        memes = memes[memes["vlm_description"].notna()]
    doc_ids, docs = build_docs(memes, args.field)
    doc_text = {m: d_prefix + d for m, d in zip(doc_ids, docs)}
    train = memes[memes["split"] == "train"]
    val = memes[memes["split"] == "val"]

    pairs = make_pairs(train, doc_text, args.queries, q_prefix)
    random.shuffle(pairs)
    pairs = pairs[: args.limit]
    print(f"Мемов train/val: {len(train)}/{len(val)}, обучающих пар: {len(pairs)}")

    model = SentenceTransformer(args.base, device=args.device)
    model.max_seq_length = args.max_seq
    if args.freeze_embeddings:
        # Словарные эмбеддинги — бо́льшая часть параметров: заморозка экономит память под Adam и снижает переобучение.
        for n, param in model.named_parameters():
            if "embeddings" in n:
                param.requires_grad = False

    train_ds = Dataset.from_list(pairs)
    if args.hard_negatives:
        train_corpus = [doc_text[m] for m in train["meme_id"]]
        train_ds = mine_hard_negatives(
            train_ds, model, corpus=train_corpus,
            range_min=10, range_max=100, relative_margin=0.05,
            num_negatives=args.hard_negatives, sampling_strategy="random",
            output_format="n-tuple", batch_size=64,
        )
        print(f"После майнинга hard negatives: {len(train_ds)} примеров, колонки {train_ds.column_names}")

    val_queries, val_rel = {}, {}
    for r in val.to_dict("records"):
        for q in r["synthetic_queries"]:
            qid = f"q{len(val_queries)}"
            val_queries[qid] = q_prefix + q
            val_rel[qid] = {r["meme_id"]}
    evaluator = InformationRetrievalEvaluator(
        val_queries, doc_text, val_rel, name="val", ndcg_at_k=[10], mrr_at_k=[10],
        accuracy_at_k=[1, 5, 10], precision_recall_at_k=[10], map_at_k=[10], batch_size=64,
    )

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints" / name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=args.device == "cuda",
        use_cpu=args.device == "cpu",
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # один мем дважды в батче = ложный негатив
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )

    before = evaluator(model)
    t0 = time.perf_counter()
    SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator
    ).train()
    train_min = (time.perf_counter() - t0) / 60
    after = evaluator(model)

    model.save(str(out_dir))
    (out_dir / "prefixes.json").write_text(json.dumps({"query": q_prefix, "doc": d_prefix}), encoding="utf-8")
    key = "val_cosine_ndcg@10"
    row = {
        "name": name, **{k: v for k, v in vars(args).items() if k != "name"},
        "pairs": len(pairs), "train_minutes": round(train_min, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
        "val_ndcg@10_before": round(before[key], 4), "val_ndcg@10_after": round(after[key], 4),
    }
    Path("results").mkdir(exist_ok=True)
    with Path("results/train_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
