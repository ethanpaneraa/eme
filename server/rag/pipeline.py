import os, json, logging
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from ingestion.models import MessageChunk
from utils.cleaning import sanitize_metadata

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHROMADB_SERVER = os.getenv("CHROMADB_SERVER") or "http://localhost:8000"
CHROMADB_TOKEN  = os.getenv("CHROMADB_TOKEN") or "dummy"
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "groupme"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"
CHAT_MODEL  = os.getenv("CHAT_MODEL") or "gpt-4o-mini"

@dataclass
class RetrievedHit:
    text: str
    meta: Dict[str, Any]
    score: float  # similarity (larger is better when we invert distances)

class RAGPipelineGM:
    def __init__(self, batch_size: int = 64):
        # self.client = chromadb.HttpClient(
        #     host=CHROMADB_SERVER,
        #     settings=Settings(
        #         chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        #         chroma_client_auth_credentials=CHROMADB_TOKEN,
        #     ),
        # )
        self.client = chromadb.PersistentClient(path="index/chroma")
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.batch_size = batch_size

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            resp = self.llm.embeddings.create(model=EMBED_MODEL, input=chunk)
            out.extend([e.embedding for e in resp.data])
        return out

    def add_messages(self, chunks: List[MessageChunk]):
        texts = [c.text for c in chunks]
        ids   = [c.to_id() for c in chunks]
        metas = [sanitize_metadata(c.metadata) for c in chunks]
        embs  = self._batch_embed(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs,
        )
        logger.info(f"Indexed {len(chunks)} message-chunks into collection '{COLLECTION_NAME}'")

    def retrieve(self, query: str, k: int = 6) -> List[RetrievedHit]:
        q_emb = self._batch_embed([query])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # chroma returns distance; for cosine, similarity ~ 1 - distance
            sim = 1.0 - float(dist)
            hits.append(RetrievedHit(text=doc, meta=meta, score=sim))
        return hits

    def _build_prompt(self, query: str, hits: List[RetrievedHit]) -> List[Dict[str, str]]:
        sys = (
            "You answer newcomer questions using ONLY the provided GroupMe excerpts. "
            "Be concise and actionable. If the context doesn’t answer, say so. "
            "Add short citations like (sender • YYYY-MM-DD). Prefer announcements for logistics."
        )
        ctx = "\n\n--- EXCERPTS ---\n"
        for i, h in enumerate(hits, 1):
            sender = h.meta.get("sender_name","Unknown")
            date = (h.meta.get("created_at_iso") or "")[:10]
            mtype = h.meta.get("msg_type","")
            ctx += f"[{i}] ({mtype}) ({sender} • {date})\n{h.text}\n\n"
        user = f"{ctx}--- QUESTION ---\n{query}\n\nAnswer with brief citations."
        return [{"role":"system","content":sys}, {"role":"user","content":user}]


    def generate(self, query: str, k: int = 6) -> str:
        hits = self.retrieve(query, k=k)
        if not hits:
            return "I couldn’t find anything relevant in the chat."
        msgs = self._build_prompt(query, hits)
        resp = self.llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
            temperature=0.2,
        )
        return resp.choices[0].message.content
