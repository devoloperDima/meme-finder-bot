"""SQLite-лог показов и реакций: из него считаются онлайн-метрики и собираются пары для дообучения.

user_id храним в виде соли+хеша: для метрик нужен только факт «тот же пользователь».
"""

import hashlib
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY, user_hash TEXT, query TEXT, results TEXT, ts REAL,
    bm25_ms REAL, dense_ms REAL, fuse_ms REAL
);
CREATE TABLE IF NOT EXISTS impressions (
    id INTEGER PRIMARY KEY, search_id INTEGER, meme_id TEXT, rank INTEGER, ts REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, impression_id INTEGER, action TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS photo_cache (meme_id TEXT PRIMARY KEY, file_id TEXT);
"""


class Storage:
    def __init__(self, path: Path, salt: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.salt = salt

    def user_hash(self, user_id: int) -> str:
        return hashlib.sha256(f"{self.salt}:{user_id}".encode()).hexdigest()[:16]

    def log_search(self, user_id: int, query: str, meme_ids: list[str], timings: dict[str, float]) -> int:
        cur = self.db.execute(
            "INSERT INTO searches (user_hash, query, results, ts, bm25_ms, dense_ms, fuse_ms) VALUES (?,?,?,?,?,?,?)",
            (self.user_hash(user_id), query, ",".join(meme_ids), time.time(),
             timings["bm25_ms"], timings["dense_ms"], timings["fuse_ms"]),
        )
        self.db.commit()
        return cur.lastrowid

    def results(self, search_id: int) -> list[str]:
        row = self.db.execute("SELECT results FROM searches WHERE id = ?", (search_id,)).fetchone()
        return row[0].split(",") if row and row[0] else []

    def log_impression(self, search_id: int, meme_id: str, rank: int) -> int:
        cur = self.db.execute(
            "INSERT INTO impressions (search_id, meme_id, rank, ts) VALUES (?,?,?,?)",
            (search_id, meme_id, rank, time.time()),
        )
        self.db.commit()
        return cur.lastrowid

    def impression(self, impression_id: int) -> tuple[int, str, int] | None:
        """(search_id, meme_id, rank)"""
        return self.db.execute(
            "SELECT search_id, meme_id, rank FROM impressions WHERE id = ?", (impression_id,)
        ).fetchone()

    def log_feedback(self, impression_id: int, action: str) -> None:
        self.db.execute("INSERT INTO feedback (impression_id, action, ts) VALUES (?,?,?)", (impression_id, action, time.time()))
        self.db.commit()

    def cached_file_id(self, meme_id: str) -> str | None:
        row = self.db.execute("SELECT file_id FROM photo_cache WHERE meme_id = ?", (meme_id,)).fetchone()
        return row[0] if row else None

    def cache_file_id(self, meme_id: str, file_id: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO photo_cache VALUES (?,?)", (meme_id, file_id))
        self.db.commit()
