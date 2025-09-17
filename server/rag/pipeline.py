import os, json, logging
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from datetime import datetime, timezone

from ingestion.models import MessageChunk
from utils.cleaning import sanitize_metadata

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Chroma / OpenAI config ---
CHROMADB_SERVER   = os.getenv("CHROMADB_SERVER", "http://localhost:8001")
CHROMADB_TOKEN    = os.getenv("CHROMADB_TOKEN")  # leave unset for local unless container enforces auth
CHROMADB_TENANT   = os.getenv("CHROMADB_TENANT", "default_tenant")
CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "default_database")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "groupme")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def _host_only_client():
    """Create a client WITHOUT tenant/database so we can bootstrap them first."""
    kwargs: Dict[str, Any] = dict(host=CHROMADB_SERVER)
    if CHROMADB_TOKEN:
        kwargs["settings"] = Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=CHROMADB_TOKEN,
        )
    return chromadb.HttpClient(**kwargs)


def _scoped_client():
    """Create the normal, scoped client AFTER tenant/database exist."""
    kwargs: Dict[str, Any] = dict(
        host=CHROMADB_SERVER,
        tenant=CHROMADB_TENANT,
        database=CHROMADB_DATABASE,
    )
    if CHROMADB_TOKEN:
        kwargs["settings"] = Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=CHROMADB_TOKEN,
        )
    return chromadb.HttpClient(**kwargs)


def _ensure_tenant_db():
    """
    Idempotently create tenant+database via the admin client.
    We must use a host-only client to avoid validation failures.
    """
    client = _host_only_client()
    admin = client._admin_client
    try:
        admin.create_tenant(name=CHROMADB_TENANT)
    except Exception:
        pass
    try:
        admin.create_database(name=CHROMADB_DATABASE, tenant=CHROMADB_TENANT)
    except Exception:
        pass


@dataclass
class RetrievedHit:
    text: str
    meta: Dict[str, Any]
    score: float


class RAGPipelineGM:
    def __init__(self, batch_size: int = 64):
        # 1) Ensure tenant/db exist (safe to run every boot)
        _ensure_tenant_db()
        # 2) Now connect scoped to tenant/db
        self.client = _scoped_client()
        # 3) Collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # 4) LLM + batching
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.batch_size = batch_size

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
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
        hits: List[RetrievedHit] = []
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - float(dist)
            hits.append(RetrievedHit(text=doc, meta=meta, score=sim))
        return hits

    def _build_prompt(self, query: str, hits: List[RetrievedHit]) -> List[Dict[str, str]]:
        sys = (
            "You answer newcomer questions using ONLY the provided GroupMe excerpts.\n"
            "Write a helpful, clear answer of roughly 120–200 words. Be specific and actionable.\n"
            "If the context doesn’t contain the answer, say so briefly and suggest what to ask next.\n"
            "Do NOT include inline citations in the body. You will not reference numbers in the body.\n"
            "Prefer announcements for logistics and time-sensitive info."
        )
        ctx = "\n\n--- EXCERPTS ---\n"
        for i, h in enumerate(hits, 1):
            sender = h.meta.get("sender_name", "Unknown")
            date = (h.meta.get("created_at_iso") or "")[:10]
            mtype = h.meta.get("msg_type", "")
            ctx += f"[{i}] ({mtype}) ({sender} • {date})\n{h.text}\n\n"
        user = (
            f"{ctx}--- QUESTION ---\n{query}\n\n"
            "Write only the answer body (no headings, no inline citations)."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def _format_citations(self, hits: List[RetrievedHit]) -> str:
        """
        Return a 'citations:' block where each line is:
          [n] Sender Name — MM/DD/YY
        Falls back gracefully if a date is missing or non-ISO.
        """
        lines = ["", "citations:", ""]

        def _fmt_mmddyy(meta: Dict[str, Any]) -> str:
            iso = meta.get("created_at_iso")
            if iso:
                try:
                    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    return dt.strftime("%m/%d/%y")
                except Exception:
                    pass
            created_at = meta.get("created_at")
            if isinstance(created_at, (int, float)):
                try:
                    dt = datetime.fromtimestamp(int(created_at), tz=timezone.utc)
                    return dt.strftime("%m/%d/%y")
                except Exception:
                    pass
            return ""

        for i, h in enumerate(hits, 1):
            sender = h.meta.get("sender_name", "Unknown")
            date_s = _fmt_mmddyy(h.meta)
            lines.append(f"[{i}] {sender} — {date_s}" if date_s else f"[{i}] {sender}")

        return "\n".join(lines)

    def generate(self, query: str, k: int = 6) -> str:
        hits = self.retrieve(query, k=k)
        if not hits:
            return "I couldn’t find anything relevant in the chat."

        msgs = self._build_prompt(query, hits)
        resp = self.llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
            temperature=0.3,
            max_tokens=350,
        )
        body = resp.choices[0].message.content.strip()

        citations = self._format_citations(hits)
        return f"{body}\n\n{citations}"
