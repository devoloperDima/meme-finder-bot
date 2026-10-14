"""Gold-набор: запросы-ситуации с ручной разметкой.

1. pool  — для каждого запроса собираем кандидатов: топ-N от нескольких систем (pooling, как в TREC).
           Повторный запуск с новыми системами только добавляет кандидатов, старая разметка сохраняется.
2. serve — локальная страница разметки: клик по мему = «подходит» (1), второй клик = «идеально» (2).

  python -m memebot.evaluation.gold pool --systems bm25 intfloat/multilingual-e5-base --depth 10
  python -m memebot.evaluation.gold serve        # → http://127.0.0.1:8765
"""

import argparse
import html
import json
import random
import time

from memebot.config import settings
from memebot.search.base import build_docs, load_memes
from memebot.search.hybrid import rrf

EVAL_DIR = settings.eval_dir
QUERIES = EVAL_DIR / "gold_queries.jsonl"
POOL = EVAL_DIR / "gold_pool.json"
LABELS = EVAL_DIR / "gold_labels.jsonl"


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_queries() -> list[dict]:
    return read_jsonl(QUERIES)


def load_labels() -> dict[str, dict[str, int]]:
    """{qid: {meme_id: grade}}; при повторной разметке побеждает последняя запись."""
    labels: dict[str, dict[str, int]] = {}
    for r in read_jsonl(LABELS):
        labels.setdefault(r["qid"], {})[r["meme_id"]] = r["grade"]
    return labels


def gold_qrels() -> tuple[list[str], dict[str, dict[str, int]], list[str]]:
    """(тексты, qrels по индексу запроса, qid) — только размеченные запросы с ≥1 релевантным мемом."""
    labels = load_labels()
    texts, qrels, qids = [], {}, []
    for q in load_queries():
        rel = {m: g for m, g in labels.get(q["qid"], {}).items() if g > 0}
        if rel:
            qrels[str(len(texts))] = rel
            texts.append(q["text"])
            qids.append(q["qid"])
    return texts, qrels, qids


def build_pool(systems: list[str], depth: int, field: str, device: str | None) -> None:
    from memebot.evaluation.run_baselines import make_retriever

    memes = load_memes()
    if field != "human":
        memes = memes[memes["vlm_description"].notna()]
    doc_ids, docs = build_docs(memes, field)
    queries = load_queries()
    texts = [q["text"] for q in queries]

    runs = []
    for name in systems:
        retriever = make_retriever(name, device)
        retriever.fit(doc_ids, docs)
        runs.append(retriever.search(texts, k=depth))
        print(f"{name}: готово")

    pool = json.loads(POOL.read_text(encoding="utf-8")) if POOL.exists() else {}
    added = 0
    for q, fused in zip(queries, rrf(runs, top=depth * len(runs))):
        current = pool.setdefault(q["qid"], [])
        for meme_id, _ in fused:
            if meme_id not in current:
                current.append(str(meme_id))
                added += 1
    POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=0), encoding="utf-8")
    sizes = [len(v) for v in pool.values()]
    print(f"Добавлено кандидатов: {added}; в среднем {sum(sizes) / len(sizes):.1f} на запрос")


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Разметка мемов</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:12px 20px;border-bottom:1px solid #333;z-index:1}
h1{font-size:22px;margin:4px 0}.muted{color:#999;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;padding:16px 20px}
.card{background:#222;border:4px solid transparent;border-radius:10px;cursor:pointer;overflow:hidden;user-select:none}
.card img{width:100%;display:block}
.card[data-g="1"]{border-color:#3fb950}.card[data-g="2"]{border-color:#e3b341}
.card .badge{padding:4px 8px;font-size:13px;color:#999}
button{font-size:16px;padding:8px 18px;margin-right:8px;border-radius:8px;border:0;cursor:pointer}
.save{background:#3fb950;color:#000}.nav{background:#333;color:#eee}
</style></head><body>
<header><div class="muted">__PROGRESS__ · клик — подходит (зелёный), ещё клик — идеально (жёлтый), третий — сброс</div>
<h1>«__QUERY__»</h1>
<form method="post" action="/save" id="f"><input type="hidden" name="qid" value="__QID__">
<input type="hidden" name="grades" id="grades">
<button class="save" type="submit">Сохранить и дальше (Enter)</button>
<a href="/?qid=__PREV__"><button class="nav" type="button">← Назад</button></a>
<a href="/?qid=__NEXT__"><button class="nav" type="button">Пропустить →</button></a></form></header>
<div class="grid">__CARDS__</div>
<script>
document.querySelectorAll('.card').forEach(c=>c.onclick=()=>{c.dataset.g=(+c.dataset.g+1)%3;
c.querySelector('.badge').textContent=['—','подходит','идеально'][c.dataset.g]});
document.getElementById('f').onsubmit=e=>{if(Date.now()<ready){e.preventDefault();return}const g={};document.querySelectorAll('.card').forEach(c=>g[c.dataset.id]=+c.dataset.g);
document.getElementById('grades').value=JSON.stringify(g)};
// Зажатый Enter даёт автоповтор и пролистывает запросы без разметки: игнорируем повторы и первую секунду.
const ready=Date.now()+1000;
document.onkeydown=e=>{if(e.key!=='Enter')return;e.preventDefault();if(e.repeat||Date.now()<ready)return;document.getElementById('f').requestSubmit()};
</script></body></html>"""


def serve(host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, Form
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    memes = load_memes().set_index("meme_id")
    queries = load_queries()
    order = [q["qid"] for q in queries]
    by_qid = {q["qid"]: q for q in queries}
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    app = FastAPI()

    def first_unlabeled(labels) -> str:
        for qid in order:
            if any(m not in labels.get(qid, {}) for m in pool.get(qid, [])):
                return qid
        return order[-1]

    @app.get("/", response_class=HTMLResponse)
    def page(qid: str | None = None):
        labels = load_labels()
        qid = qid if qid in by_qid else first_unlabeled(labels)
        i = order.index(qid)
        done = sum(1 for q in order if pool.get(q) and all(m in labels.get(q, {}) for m in pool[q]))
        cands = list(pool.get(qid, []))
        random.Random(qid).shuffle(cands)  # фиксированный случайный порядок — без позиционного смещения
        cards = []
        for m in cands:
            g = labels.get(qid, {}).get(m, 0)
            title = html.escape(memes.loc[m, "description"][:400], quote=True)
            cards.append(f'<div class="card" data-id="{m}" data-g="{g}" title="{title}"><img loading="lazy" '
                         f'src="/img/{m}"><div class="badge">{["—", "подходит", "идеально"][g]}</div></div>')
        return (PAGE.replace("__PROGRESS__", f"Запрос {i + 1}/{len(order)} · размечено {done}")
                .replace("__QUERY__", html.escape(by_qid[qid]["text"])).replace("__QID__", qid)
                .replace("__PREV__", order[max(i - 1, 0)]).replace("__NEXT__", order[min(i + 1, len(order) - 1)])
                .replace("__CARDS__", "\n".join(cards)))

    @app.post("/save")
    def save(qid: str = Form(...), grades: str = Form(...)):
        ts = time.time()
        with LABELS.open("a", encoding="utf-8") as f:
            for meme_id, g in json.loads(grades).items():
                f.write(json.dumps({"qid": qid, "meme_id": meme_id, "grade": int(g), "ts": ts}) + "\n")
        i = order.index(qid)
        return RedirectResponse(f"/?qid={order[min(i + 1, len(order) - 1)]}", status_code=303)

    @app.get("/img/{meme_id}")
    def img(meme_id: str):
        return FileResponse(settings.data_dir / memes.loc[meme_id, "image_path"])

    print(f"Разметка: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--field", default="human")
    p.add_argument("--device", default=None)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.cmd == "pool":
        build_pool(args.systems, args.depth, args.field, args.device)
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
