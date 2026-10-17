"""Свежие мемы из публичного Telegram-канала через его веб-версию t.me/s/<канал>.

Для @text_meme текст поста — ручное описание мема, как и в HF-датасетах, только новее.
Для любого другого канала текст может быть пустым — тогда описание даст VLM.

Берём посты ровно с одной картинкой. Между запросами пауза, прогон можно прервать
и продолжить: уже скачанные посты пропускаются.

  python -m memebot.data.scrape_telegram --channel text_meme --min-id 6446
"""

import argparse
import html
import json
import re
import time
import urllib.request

from tqdm import tqdm

from memebot.config import settings

UA = {"User-Agent": "Mozilla/5.0 (meme-finder-bot research project)"}
POST_RE = re.compile(r'<div class="tgme_widget_message_wrap.*?(?=<div class="tgme_widget_message_wrap|\Z)', re.DOTALL)
ID_RE = re.compile(r'data-post="[^/"]+/(\d+)"')
DATE_RE = re.compile(r'<time datetime="([^"]+)"')
PHOTO_RE = re.compile(r"tgme_widget_message_photo_wrap[^>]*background-image:url\('([^']+)'\)")
TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.DOTALL)


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
        except OSError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"не удалось скачать {url}")


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parse_page(page: str) -> list[dict]:
    posts = []
    for block in POST_RE.findall(page):
        m = ID_RE.search(block)
        if not m:
            continue
        photos = PHOTO_RE.findall(block)
        text = TEXT_RE.search(block)
        date = DATE_RE.search(block)
        posts.append({
            "id": int(m.group(1)),
            "date": date.group(1) if date else None,
            "photos": photos,
            "text": html_to_text(text.group(1)) if text else "",
            "has_video": "tgme_widget_message_video" in block,
        })
    return posts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="text_meme")
    p.add_argument("--min-id", type=int, default=0, help="остановиться на постах старше этого id")
    p.add_argument("--delay", type=float, default=1.5)
    args = p.parse_args()

    out_dir = settings.raw_dir / f"tg_{args.channel}"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    posts_path = out_dir / "posts.jsonl"
    seen = set()
    if posts_path.exists():
        seen = {json.loads(line)["id"] for line in posts_path.open(encoding="utf-8")}

    before, pages, kept, skipped = None, 0, 0, 0
    bar = tqdm(desc="страниц")
    with posts_path.open("a", encoding="utf-8") as out:
        while True:
            url = f"https://t.me/s/{args.channel}" + (f"?before={before}" if before else "")
            posts = parse_page(fetch(url).decode("utf-8"))
            pages += 1
            bar.update()
            if not posts:
                break
            for post in posts:
                if post["id"] < args.min_id or post["id"] in seen:
                    continue
                if len(post["photos"]) != 1 or post["has_video"]:
                    skipped += 1
                    continue
                (img_dir / f"{post['id']}.jpg").write_bytes(fetch(post["photos"][0]))
                record = {"id": post["id"], "date": post["date"], "text": post["text"], "channel": args.channel}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                seen.add(post["id"])
                kept += 1
                time.sleep(args.delay / 3)
            oldest = min(post["id"] for post in posts)
            if oldest <= args.min_id or oldest == before:
                break
            before = oldest
            time.sleep(args.delay)
    bar.close()
    print(f"Страниц: {pages}, сохранено постов: {kept}, пропущено (не одна картинка / видео): {skipped}")


if __name__ == "__main__":
    main()
