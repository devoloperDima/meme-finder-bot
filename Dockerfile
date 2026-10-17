FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache \
    SEARCH_DEVICE=cpu

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

# Данные (memes.parquet, картинки, кэш эмбеддингов) монтируются томом в /app/data.
RUN useradd --create-home app && mkdir -p /hf-cache && chown app /hf-cache
USER app

CMD ["python", "-m", "memebot.bot.main"]
