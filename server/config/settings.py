from pydantic_settings import BaseSettings
from typing import List
import os
from functools import lru_cache


class Settings(BaseSettings, extra="allow"):
    ENV: str = os.getenv("ENV", "prod")

    GROUPME_BOT_ID: str = os.getenv("GROUPME_BOT_ID", "")
    GROUPME_BOT_NAME: str = os.getenv("GROUPME_BOT_NAME", "eme")
    GROUPME_API_URL: str = os.getenv("GROUPME_API_URL", "https://api.groupme.com/v3")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Pinecone settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "groupme-messages")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "")

    EMERGING_CODERS_URL: str = os.getenv("EMERGING_CODERS_URL", "")

    ALLOWED_ORIGINS: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base_origins = [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
        ]

        if self.ENV == "dev":
            self.ALLOWED_ORIGINS = base_origins
        else:
            self.ALLOWED_ORIGINS = base_origins.copy()
            if self.FRONTEND_URL:
                self.ALLOWED_ORIGINS.append(self.FRONTEND_URL)
            if self.EMERGING_CODERS_URL:
                self.ALLOWED_ORIGINS.append(self.EMERGING_CODERS_URL)

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
