from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = ""
    hf_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""

    ollama_host: str = "http://localhost:11434"
    qdrant_path: Path = Path("data/qdrant")
    qdrant_url: str = ""
    qdrant_collection: str = "memes"

    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


settings = Settings()
