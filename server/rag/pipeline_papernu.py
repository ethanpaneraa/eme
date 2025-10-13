import os, json, re
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import time
from typing import Any
from ingestion.models import FullCourseRecord

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

class RAGPipelinePaperNU():
    def __init__(self):
        self.backend = str(VECTOR_BACKEND) or "chroma"
        self._pinecone_mod = _pinecone_mod
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.chroma_client = chromadb.PersistentClient(path="index/chroma")
        self.collection: chromadb.Collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, )
        self.llm: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
        self.batch_size: int = 64
        self.records: list[FullCourseRecord] = []

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

    def add_all_records(self, records: list[FullCourseRecord]):
        self.records = records
        try:
            ids = [r.subject + " " + r.catalog_number for r in records]
            texts = [r.get_message() for r in records]
            embeddings = self._batch_embed(texts)

            if self.backend == "pinecone":
                if not PINECONE_API_KEY:
                    raise ValueError("PINECONE_API_KEY environment variable is required for Pinecone backend")
                try:
                    indexes = self.pc.list_indexes()
                    index_names = [idx.name for idx in indexes]
                    if PINECONE_INDEX_NAME not in index_names:
                        self.pc.create_index(
                            name=PINECONE_INDEX_NAME,
                            dimension=1536,
                            metric="cosine",
                            spec=self._pinecone_mod.ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
                        )
                    self.index = self.pc.Index(PINECONE_INDEX_NAME)
                except Exception as e:
                    raise
            else:
                # Insert into ChromaDB
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                )
        except Exception as e:
            raise
    
    def retrieve(self, query: str, k: int = 6) -> list[str]:
        print(f"Retrieving {k} documents for query: {query[:50]}...")

        hits: list[str] = []
        try:
            query_embedding = self._batch_embed([query])

            if self.backend == "pinecone":
                result = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
                for match in result.matches:
                    meta = dict(match.metadata or {})
                    text = meta.pop("text", "")
                    hits.append(text)
            else:
                res = self.collection.query(
                    query_embeddings=[query_embedding][0],
                    n_results=k,
                )
                docs = res.get("documents", [[]])[0]
                hits = docs
        except Exception as e:
            raise
        return hits
    
    def _match_catalog_number_to_name(self, catalog_number: str):
        """Given a catalog number, match it to the course name.
        For example, given "336", we match and return with "Design & Analysis of Algorithms". Returns all names that match. 
        
        This is to improve RAG matching for the actual courses since a query with the number by itself (such as "what are the prereqs for CS330?") doesn't perform well RAG-wise matching with the actula course.

        However, providing RAG with the context of CS330 Human Computer Interaction greatly improves matching, hence this function's usefulness.
        """

        def isSimilar()

        for record in self.records:
            if catalog_number == record.catalog_number
        
    def build_context(self, query: str) -> str:
        header: str = "Below is official course information from paper.nu for relevant Computer Science and Computer Engineering coures:\n"

        docs = self.retrieve(query)
        full_context = header + "\n".join(docs)
        return full_context
    
