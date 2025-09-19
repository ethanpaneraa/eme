from pydantic_settings import BaseSettings
from typing import List
import os
from functools import lru_cache


class Settings(BaseSettings, extra="allow"):
    ENV: str = "dev"

    # GroupMe Bot Configuration
    GROUPME_BOT_ID: str = os.getenv("GROUPME_BOT_ID", "")
    GROUPME_BOT_NAME: str = os.getenv("GROUPME_BOT_NAME", "eme")
    GROUPME_API_URL: str = os.getenv("GROUPME_API_URL", "https://api.groupme.com/v3")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Frontend URL
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "")

    ALLOWED_ORIGINS: List[str] = []  # empty default

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Base origins for local development
        base_origins = [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
        ]

        if self.ENV == "dev":
            self.ALLOWED_ORIGINS = base_origins
        else:
            # Production: start with base origins and add frontend URL if provided
            self.ALLOWED_ORIGINS = base_origins.copy()
            if self.FRONTEND_URL:
                self.ALLOWED_ORIGINS.append(self.FRONTEND_URL)

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
