"""Свежие мемы из публичного Telegram-канала через его веб-версию t.me/s/<канал>.

Для @text_meme текст поста — ручное описание мема, как и в HF-датасетах, только новее.
Для любого другого канала текст может быть пустым — тогда описание даст VLM.

Берём посты ровно с одной картинкой. Между запросами пауза, прогон можно прервать
и продолжить: уже скачанные посты пропускаются.

  python -m memebot.data.scrape_telegram --channels text_meme --min-id 6446
  python -m memebot.data.scrape_telegram --channels memachh memdoze --since 2025-10-17 --max-posts 1500
"""

import argparse
import html
import json
import re
import time
import urllib.request

from tqdm import tqdm

from memebot.config import settings
from memebot.utils import keep_awake

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


AD_MARKERS = ("erid", "#реклама", "реклама.", "на правах рекламы")


def is_ad(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in AD_MARKERS)


def scrape_channel(channel: str, min_id: int = 0, since: str | None = None, max_posts: int | None = None,
                   delay: float = 1.5) -> None:
    out_dir = settings.raw_dir / f"tg_{channel}"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    posts_path = out_dir / "posts.jsonl"
    seen = set()
    if posts_path.exists():
        seen = {json.loads(line)["id"] for line in posts_path.open(encoding="utf-8")}

    before, pages, kept, skipped, done = None, 0, 0, 0, False
    bar = tqdm(desc=channel)
    with posts_path.open("a", encoding="utf-8") as out:
        while not done:
            url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
            posts = parse_page(fetch(url).decode("utf-8"))
            pages += 1
            bar.update()
            if not posts:
                break
            for post in sorted(posts, key=lambda x: -x["id"]):  # от новых к старым
                if post["id"] < min_id or (since and post["date"] and post["date"] < since):
                    done = True
                    break
                if post["id"] in seen:
                    continue
                if len(post["photos"]) != 1 or post["has_video"] or is_ad(post["text"]):
                    skipped += 1
                    continue
                (img_dir / f"{post['id']}.jpg").write_bytes(fetch(post["photos"][0]))
                record = {"id": post["id"], "date": post["date"], "text": post["text"], "channel": channel}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                seen.add(post["id"])
                kept += 1
                if max_posts and len(seen) >= max_posts:
                    done = True
                    break
                time.sleep(delay / 3)
            oldest = min(post["id"] for post in posts)
            if oldest == before:
                break
            before = oldest
            time.sleep(delay)
    bar.close()
    print(f"{channel}: страниц {pages}, сохранено {kept}, пропущено (не одна картинка / видео / реклама) {skipped}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channels", nargs="+", default=["text_meme"])
    p.add_argument("--min-id", type=int, default=0, help="остановиться на постах старше этого id")
    p.add_argument("--since", default=None, help="остановиться на постах старше даты, например 2025-10-17")
    p.add_argument("--max-posts", type=int, default=None, help="не больше N постов с канала (самые свежие)")
    p.add_argument("--delay", type=float, default=1.5)
    args = p.parse_args()
    keep_awake()
    for channel in args.channels:
        scrape_channel(channel, args.min_id, args.since, args.max_posts, args.delay)


if __name__ == "__main__":
    main()
