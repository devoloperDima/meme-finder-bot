"""HTTP API поиска мемов: тот же MemeSearcher, что и в боте.

  uvicorn memebot.api.main:app --port 8000
  GET /search?q=понедельник снова на работу&k=5&fresh=0.15
  GET /memes/{meme_id}.jpg
  GET /health
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from memebot.config import settings
from memebot.search.service import MemeSearcher

searcher: MemeSearcher | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global searcher
    searcher = MemeSearcher(settings.search_model, device=settings.search_device)
    yield


app = FastAPI(title="meme-finder", lifespan=lifespan)


class SearchHit(BaseModel):
    meme_id: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    timings_ms: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if searcher else "loading", "memes": len(searcher.image_path) if searcher else 0}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=300),
    k: int = Query(5, ge=1, le=50),
    fresh: float | None = Query(None, ge=0, le=5, description="вес свежести; по умолчанию как в боте"),
) -> SearchResponse:
    if searcher is None:
        raise HTTPException(503, "индекс ещё загружается")
    hits, timings = searcher.search(q, k, fresh)
    return SearchResponse(
        query=q,
        hits=[SearchHit(meme_id=h.meme_id, score=round(h.score, 4), image_url=f"/memes/{h.meme_id}.jpg") for h in hits],
        timings_ms={name: round(v, 2) for name, v in timings.items()},
    )


@app.get("/memes/{meme_id}.jpg")
def meme_image(meme_id: str) -> FileResponse:
    path = searcher.image_path.get(meme_id) if searcher else None  # только id из индекса — без путей от клиента
    if path is None:
        raise HTTPException(404, "нет такого мема")
    return FileResponse(settings.data_dir / path, media_type="image/jpeg")
