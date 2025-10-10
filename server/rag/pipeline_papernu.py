import os, json, re
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from datetime import datetime, timezone
import time
from typing import Any

from ingestion.models import MessageChunk
from utils.cleaning import sanitize_metadata
from logging_config import get_logger, log_rag_operation
from pydantic import BaseModel, field_validator, ConfigDict
import pinecone as _pinecone_mod
from pinecone import Pinecone

load_dotenv()

CHROMADB_SERVER = os.getenv("CHROMADB_SERVER") or "http://localhost:8000"
CHROMADB_TOKEN  = os.getenv("CHROMADB_TOKEN") or "dummy"
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "groupme"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"
CHAT_MODEL  = os.getenv("CHAT_MODEL") or "gpt-4o-mini"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "groupme-messages"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND")

class RAGPipelinePaperNU(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: str = str(VECTOR_BACKEND) or "chroma"
    _pinecone_mod = _pinecone_mod
    pc: Pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index: Any = pc.Index(PINECONE_INDEX_NAME)
    chroma_client: chromadb.Client = chromadb.PersistentClient(path="index/chroma")
    collection: chromadb.Collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, )
    llm: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
    batch_size: int = 64

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]

            try:
                resp = self.llm.embeddings.create(model=EMBED_MODEL, input=chunk)
                out.extend([e.embedding for e in resp.data])
            except Exception as e:              
                raise
        return out

    # def _add_embedding_chunk(self, chunks: list[list[float]]):
    #     if self.backend == "pinecone":
    #         try:
    #             if not PINECONE_API_KEY:
    #                 raise ValueError("PINECONE_API_KEY environment variable is required for Pinecone backend")
    #             from pinecone import Pinecone
    #             import pinecone as _pinecone_mod

    #             self._pinecone_mod = _pinecone_mod
    #             self.pc = Pinecone(api_key=PINECONE_API_KEY)

    #             try:
    #                 indexes = self.pc.list_indexes()
    #                 index_names = [idx.name for idx in indexes]
    #                 if PINECONE_INDEX_NAME not in index_names:
    #                     logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim=1536, metric=cosine)")
    #                     self.pc.create_index(
    #                         name=PINECONE_INDEX_NAME,
    #                         dimension=1536,
    #                         metric="cosine",
    #                         spec=self._pinecone_mod.ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
    #                     )
    #                     logger.info(f"Created Pinecone index '{PINECONE_INDEX_NAME}'")
    #                 self.index = self.pc.Index(PINECONE_INDEX_NAME)
    #                 logger.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
    #             except Exception as e:
    #                 logger.error(f"Failed to get/create Pinecone index: {str(e)}")
    #                 raise
    #         except Exception as e:
    #             logger.error(f"Failed to initialize Pinecone backend: {str(e)}")
    #             raise
